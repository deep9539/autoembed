#!/usr/bin/env bash
# Run an agent on the task, then score final_model. An NVIDIA GPU is required.
#   scripts/run_task.sh <agent> [agent-model] [hours]    agents: claude | codex | gemini
# MODE=native uses the repo venv; MODE=enroot/docker uses an immutable image.
# GPU_ID=<physical index> selects a GPU outside Slurm. REQUIRE_GPU_ENFORCEMENT=1
# refuses native runs unless they are inside Slurm; container device isolation
# and Slurm cgroups are the supported enforcement boundaries.
# Per-run artifacts land in results/<id>/: prompt, trace, final_model, scores, meta.
set -euo pipefail

# A requeued job would start the agent over from zero and bill a second time, so
# a restarted allocation stops here instead.
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
  echo "!! Slurm restarted this allocation (SLURM_RESTART_COUNT=$SLURM_RESTART_COUNT);" >&2
  echo "!! refusing to relaunch the agent. Recover from results/<run>/recovery." >&2
  exit 75
fi

# ---- Configuration and provenance ----
AGENT="${1:?usage: scripts/run_task.sh <agent> [agent-model] [hours]}"
case "$AGENT" in
  claude|codex|gemini) ;;
  *) echo "unknown agent: $AGENT (expected claude, codex, or gemini)" >&2; exit 2 ;;
esac
AGENT_CONFIG="${2:-}"
if [ -z "$AGENT_CONFIG" ]; then
  case "$AGENT" in
    claude) AGENT_CONFIG="claude-opus-5" ;;
    codex) AGENT_CONFIG="gpt-5.6-sol" ;;
    gemini) AGENT_CONFIG="gemini-3.6-flash" ;;
  esac
fi
HOURS="${3:-${HOURS:-3}}"
AUTOEMBED_AGENT_REASONING="${AUTOEMBED_AGENT_REASONING:-high}"
MODE="${MODE:-native}"
GPU_ID="${GPU_ID:-}"
REQUIRE_GPU_ENFORCEMENT="${REQUIRE_GPU_ENFORCEMENT:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENROOT_IMAGE="${ENROOT_IMAGE:-$ROOT/artifacts/autoembed-$AGENT.sqsh}"
ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-$ROOT/artifacts/enroot-data}"
ENROOT_CONTAINER="${ENROOT_CONTAINER:-autoembed-$AGENT}"
AUTH_ROOT="${AUTOEMBED_AUTH_DIR:-$ROOT/.agent-auth}"
AGENT_AUTH_DIR="$AUTH_ROOT/$AGENT"
AGENT_AUTH_MODE="none"
AGENT_PROTOCOL="official-cli-reprompt-v1"
REPROMPT_CUTOFF_MINUTES=30
AGENT_WRAPPER_SHA256="$(sha256sum "$ROOT/agents/$AGENT/solve.sh" | cut -d" " -f1)"
AUTH_STAGE=""
CONTAINER_IMAGE_SHA256=""
CONFIG="${AUTOEMBED_CONFIG:-$ROOT/configs/specialization/legal.json}"   # experiment spec: base model and evaluation protocol
CONFIG_BASE_MODEL="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["base_model"])' "$CONFIG")"
CONFIG_BASE_REVISION="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("base_revision",""))' "$CONFIG")"
if [ -n "${AUTOEMBED_BASE_MODEL:-}" ]; then
  BASE_MODEL="$AUTOEMBED_BASE_MODEL"
  BASE_REVISION="${AUTOEMBED_BASE_REVISION:-}"
else
  BASE_MODEL="$CONFIG_BASE_MODEL"
  BASE_REVISION="${AUTOEMBED_BASE_REVISION:-$CONFIG_BASE_REVISION}"
fi
[[ "$BASE_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
  echo "!! base checkpoint must be pinned: set AUTOEMBED_BASE_REVISION to a 40-character lowercase commit hash" >&2
  exit 1
}
PER_TASK_TIMEOUT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("per_task_timeout", 3600))' "$CONFIG")"
REQUIRE_FILESYSTEM_ISOLATION="$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("require_filesystem_isolation", False)))' "$CONFIG")"
export AUTOEMBED_PER_TASK_TIMEOUT="$PER_TASK_TIMEOUT"
export TORCHDYNAMO_DISABLE=1   # mteb torch.compiles on H100; the runtime image has no compiler
PROTOCOL_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("protocol_version", "unspecified"))' "$CONFIG")"
PROTOCOL_TYPE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("protocol_type", "transfer"))' "$CONFIG")"
DEV_BENCHMARK="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("dev_benchmark", ""))' "$CONFIG")"
BENCHMARK_REVISION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("benchmark_revision", ""))' "$CONFIG")"
NANO_SOURCE="${AUTOEMBED_NANO_DIR:-$ROOT/runs/nano}"
CONFIG_SHA256="$(sha256sum "$CONFIG" | cut -d" " -f1)"
SCORER_SHA256="$(sha256sum "$ROOT/autoembed/scoring.py" | cut -d" " -f1)"
CONFIG_ID="${CONFIG#"$ROOT/"}"
HARNESS_GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
if git -C "$ROOT" diff --quiet \
  && git -C "$ROOT" diff --cached --quiet \
  && [ -z "$(git -C "$ROOT" ls-files --others --exclude-standard)" ]; then
  HARNESS_GIT_DIRTY=false
else
  HARNESS_GIT_DIRTY=true
fi
export AUTOEMBED_BASE_MODEL="$BASE_MODEL"
export AUTOEMBED_BASE_REVISION="$BASE_REVISION"
# The complete config stays harness-only; the agent receives only its development protocol.
DEV_TASKS="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); print(",".join(c.get("dev_tasks") or c.get("tasks") or []))' "$CONFIG" 2>/dev/null || true)"
[ -n "$DEV_TASKS" ] && export AUTOEMBED_DEV_TASKS="$DEV_TASKS"   # agent sees dev only
AUTOEMBED_QUERY_SPLIT="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); q=c.get("query_split"); print(json.dumps(q,separators=(",", ":")) if q else "")' "$CONFIG")"
AUTOEMBED_EXAMPLE_SPLIT="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); q=c.get("example_split"); print(json.dumps(q,separators=(",", ":")) if q else "")' "$CONFIG")"
ALLOW_TARGET_CORPUS_TRAINING="$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("allow_target_corpus_training", False)))' "$CONFIG")"
export AUTOEMBED_QUERY_SPLIT
export AUTOEMBED_EXAMPLE_SPLIT

# Validate local benchmark data before allocating a GPU or starting an agent.
if [ "$DEV_BENCHMARK" = mteb-nano ]; then
  ( cd "$ROOT" && AUTOEMBED_NANO_DIR="$NANO_SOURCE" PYTHONPATH="$ROOT/agent_task" \
    UV_PROJECT_ENVIRONMENT="$ROOT/.venv" uv run --no-sync python -c \
    'import nano_dev,sys; print("MTEB-nano assets:", nano_dev.validate_assets(sys.argv[1]))' \
    "$BENCHMARK_REVISION" )
fi

if [ "$REQUIRE_FILESYSTEM_ISOLATION" = 1 ]; then
  case "$MODE" in
    docker|enroot) ;;
    *) echo "!! this reportable config requires MODE=enroot or MODE=docker" >&2; exit 1 ;;
  esac
fi
# ---- Authentication and isolation ----
if [ "$MODE" = enroot ]; then
  command -v enroot >/dev/null 2>&1 || { echo "!! enroot is unavailable" >&2; exit 1; }
  [ -f "$ENROOT_IMAGE" ] || {
    echo "!! missing Enroot image: $ENROOT_IMAGE" >&2
    exit 1
  }
  if [ -f "$ENROOT_IMAGE.sha256" ]; then
    CONTAINER_IMAGE_SHA256="$(cut -d" " -f1 "$ENROOT_IMAGE.sha256")"
  else
    CONTAINER_IMAGE_SHA256="$(sha256sum "$ENROOT_IMAGE" | cut -d" " -f1)"
  fi
  [ -d "$ENROOT_DATA_PATH/$ENROOT_CONTAINER" ] || {
    echo "!! missing extracted Enroot runtime; run scripts/build_enroot.sh $AGENT on a compute node" >&2
    exit 1
  }
  export ENROOT_DATA_PATH
fi

if [ "$MODE" = enroot ] || [ "$MODE" = docker ]; then
  case "$AGENT" in
    claude)
      if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then AGENT_AUTH_MODE=subscription
      elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then AGENT_AUTH_MODE=api-key
      else
        echo "!! isolated Claude runs require CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY" >&2
        exit 1
      fi ;;
    codex)
      if [ -f "$AGENT_AUTH_DIR/auth.json" ]; then
        AGENT_AUTH_MODE=subscription
      elif [ -n "${OPENAI_API_KEY:-}" ]; then AGENT_AUTH_MODE=api-key
      else
        echo "!! Codex is not authenticated; run scripts/agent_auth.sh codex" >&2
        exit 1
      fi ;;
    gemini)
      if [ -n "${GEMINI_API_KEY:-}" ] || [ -n "${GOOGLE_API_KEY:-}" ]; then
        AGENT_AUTH_MODE=api-key
      else
        echo "!! isolated Gemini runs require GEMINI_API_KEY or GOOGLE_API_KEY" >&2
        exit 1
      fi ;;
  esac
fi
export AGENT_AUTH_MODE AUTOEMBED_AGENT_REASONING

DOCKER_AUTH_MOUNT=()
ENROOT_AUTH_MOUNT=()
DOCKER_AUTH_ENV=()
ENROOT_AUTH_ENV=()

cleanup_auth_stage() {
  if [ -n "$AUTH_STAGE" ] && [ -d "$AUTH_STAGE" ]; then
    rm -rf -- "$AUTH_STAGE"
    AUTH_STAGE=""
  fi
}

if [ "$AGENT_AUTH_MODE" = subscription ] && [ "$AGENT" = codex ]; then
  chmod 700 "$AUTH_ROOT" "$AGENT_AUTH_DIR"
  AUTH_STAGE="$(mktemp -d "$AUTH_ROOT/.run-${AGENT}.XXXXXX")"
  chmod 700 "$AUTH_STAGE"
  trap cleanup_auth_stage EXIT
  install -m 600 "$AGENT_AUTH_DIR/auth.json" "$AUTH_STAGE/auth.json"
  DOCKER_AUTH_MOUNT=(-v "$AUTH_STAGE:/agent-auth")
  ENROOT_AUTH_MOUNT=(-m "$AUTH_STAGE:/agent-auth")
  DOCKER_AUTH_ENV=(-e CODEX_HOME=/agent-auth)
  ENROOT_AUTH_ENV=(-e CODEX_HOME=/agent-auth)
fi

# ---- Work directory and hidden-data preflight ----
RUN_ID="$(date +%Y%m%d-%H%M%S)_${SLURM_JOB_ID:-$$}_${AGENT}"
RESULTS="$ROOT/results/$RUN_ID"
mkdir -p "$RESULTS"

WORK="$(mktemp -d)"
cp "$ROOT"/agent_task/{task.py,instructions.md,timer.sh,check_cuda.py} "$WORK"/
cp "$ROOT"/{pyproject.toml,uv.lock} "$WORK"/
# Ship only the manifest-pinned nano files used by the general benchmark.
if [ "$DEV_BENCHMARK" = mteb-nano ]; then
  cp "$ROOT"/agent_task/{nano_dev.py,nano_assets.json} "$WORK"/
  mkdir -p "$WORK/nano"
  while IFS= read -r asset; do
    cp "$NANO_SOURCE/$asset" "$WORK/nano/$asset"
  done < <(python3 -c 'import json,sys; print("\n".join(item["name"] for item in json.load(open(sys.argv[1]))["files"]))' "$ROOT/agent_task/nano_assets.json")
fi
# Validate the frozen split and build/configure the exhaustive hidden-text cache.
( cd "$ROOT" && AUTOEMBED_CONFIG="$CONFIG" AUTOEMBED_ROOT="$ROOT" AUTOEMBED_NANO_DIR="$NANO_SOURCE" UV_PROJECT_ENVIRONMENT="$ROOT/.venv" \
  uv run --no-sync python -c "from autoembed import scoring; scoring.validate_config(); print('eval-cache hashes:', scoring.ensure_eval_cache())" )
if [ "$REQUIRE_FILESYSTEM_ISOLATION" = 1 ]; then
  # public corpus hashes let the agent self-filter training data; query hashes stay with the harness
  ( cd "$ROOT" && AUTOEMBED_CONFIG="$CONFIG" AUTOEMBED_ROOT="$ROOT" AUTOEMBED_NANO_DIR="$NANO_SOURCE" UV_PROJECT_ENVIRONMENT="$ROOT/.venv" \
    uv run --no-sync python -c "from autoembed import scoring; print('agent corpus hashes:', scoring.write_agent_eval_cache('$WORK/_eval_texts.json'))" )
elif [ -f "$ROOT/_eval_texts.json" ]; then
  cp "$ROOT/_eval_texts.json" "$WORK"/
else
  echo "!! no eval cache — contamination check will be manifest-only"
fi
cp "$ROOT/agents/$AGENT/solve.sh" "$WORK/solve.sh"
mkdir -p "$WORK/final_model" "$WORK/.home" "$WORK/.cache/huggingface" "$WORK/.cache/torch" "$WORK/.cache/uv"

PROMPT="$(cat "$ROOT/agent_task/instructions.md")"
PROMPT="$PROMPT

## Fixed starting checkpoint

The fixed starting checkpoint for this run is \`$BASE_MODEL\` at immutable revision
\`$BASE_REVISION\`. Initialize the submitted model from this checkpoint. Do not substitute a different checkpoint; other models may only assist the work."
if [ "$MODE" != docker ]; then
  PROMPT="$PROMPT

## Submission format for this runtime

Save a standard offline SentenceTransformer model directory. Do not submit mteb_model.py or other
custom executable model code: hidden scoring accepts executable submissions only in MODE=docker."
fi
if [ "$ALLOW_TARGET_CORPUS_TRAINING" = 1 ]; then
  PROMPT="$PROMPT

## Run-specific data policy

This is a target-specialization run. You may inspect and train on documents from the target
corpora, including documents that may be relevant to hidden queries. Development and hidden
query text and all relevance judgments remain evaluation data: do not train on them."
elif [ "$PROTOCOL_TYPE" = benchmark-target ]; then
  PROMPT="$PROMPT

## Run-specific data policy

This is a general benchmark-target run. Optimize the visible development score using external
public or synthetic training data. Do not train on any text or label from the development or
hidden evaluation suites, including retrieval corpora. The hidden scorer uses complementary
examples from the same tasks."
else
  PROMPT="$PROMPT

## Run-specific data policy

This is a transfer run. Do not train on queries, corpus documents, or labels from either the
development or hidden evaluation tasks."
fi
export PROMPT
printf '%s\n' "$PROMPT" > "$WORK/instructions.md"
export AGENT_CONFIG
export DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
LIMIT="$(( HOURS * 3600 + 300 ))s"   # hard cap: budget + 5min grace
printf '%s' "$PROMPT" > "$RESULTS/prompt.txt"
cp "$CONFIG" "$RESULTS/config.json"   # per-run task provenance

# ---- Recovery and run metadata ----
RECOVERY_PID=""
RECOVERY_INTERVAL="${RECOVERY_INTERVAL:-300}"

final_model_signature() {
  find "$WORK/final_model" -type f -printf '%P:%s:%T@\n' | sort | sha256sum | cut -d" " -f1
}

snapshot_final_model() {
  local settle="${1:-settle}"
  [ -n "$(ls -A "$WORK/final_model" 2>/dev/null)" ] || return 0
  local signature previous stamp staging destination
  signature="$(final_model_signature)"
  previous="$(cat "$RESULTS/recovery.last-signature" 2>/dev/null || true)"
  [ "$signature" != "$previous" ] || return 0
  if [ "$settle" = settle ]; then  # skip mid-write states; retry next interval
    sleep 5
    [ "$signature" = "$(final_model_signature)" ] || return 0
  fi
  stamp="$(date +%s)"
  mkdir -p "$RESULTS/recovery"
  staging="$RESULTS/recovery/.${stamp}.partial"
  destination="$RESULTS/recovery/${stamp}"
  mkdir -p "$staging"
  cp -a "$WORK/final_model/." "$staging/" || return 0
  mv "$staging" "$destination"
  printf '%s\n' "$signature" > "$RESULTS/recovery.last-signature"
  printf '%s\n' "$destination" > "$RESULTS/recovery.latest"
  echo ">> recovery snapshot: $destination"
}

write_run_meta() {
  local complete="${1:-false}" now run_rc score_rc budget_hit node gpu_name final_files
  now="$(date +%s)"
  run_rc="${rc:--1}"
  score_rc="${SCORE_RC:-3}"
  budget_hit=$([ "$run_rc" -eq 124 ] && echo true || echo false)
  node="$(hostname 2>/dev/null)"
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$SELECTED_GPU" 2>/dev/null || true)"
  final_files="$(ls -1A "$WORK/final_model" 2>/dev/null | wc -l | tr -d ' ')"
  RUN_ID="$RUN_ID" RUN_KIND=experiment AGENT="$AGENT" AGENT_CONFIG="$AGENT_CONFIG" BASE_MODEL="$BASE_MODEL" BASE_REVISION="$BASE_REVISION" PROTOCOL_VERSION="$PROTOCOL_VERSION" CONFIG_ID="$CONFIG_ID" CONFIG_SHA256="$CONFIG_SHA256" \
  AGENT_REASONING="$AUTOEMBED_AGENT_REASONING" AGENT_AUTH_MODE="$AGENT_AUTH_MODE" AGENT_VERSION="$AGENT_VERSION" \
  AGENT_PROTOCOL="$AGENT_PROTOCOL" REPROMPT_CUTOFF_MINUTES="$REPROMPT_CUTOFF_MINUTES" AGENT_WRAPPER_SHA256="$AGENT_WRAPPER_SHA256" \
  HOURS="$HOURS" MODE="$MODE" DURATION="$((now - START))" RC="$run_rc" BUDGET_HIT="$budget_hit" \
  NODE="$node" GPU_NAME="$gpu_name" GPU_BOUNDARY="$GPU_BOUNDARY" GPU_SELECTOR="$SELECTED_GPU" FINAL_FILES="$final_files" SCORE_RC="$score_rc" \
  CONTAINER_IMAGE_SHA256="$CONTAINER_IMAGE_SHA256" HARNESS_COMPLETE="$complete" \
  HARNESS_GIT_COMMIT="$HARNESS_GIT_COMMIT" HARNESS_GIT_DIRTY="$HARNESS_GIT_DIRTY" \
  SCORER_SHA256="$SCORER_SHA256" \
  python3 "$ROOT/scripts/run_meta.py" "$RESULTS/trace.log" "$RESULTS/meta.json" "$RESULTS/scores.json"
}

recovery_loop() {
  while :; do
    sleep "$RECOVERY_INTERVAL"
    snapshot_final_model || true
    write_run_meta false || true
  done
}

stop_recovery() {
  if [ -n "$RECOVERY_PID" ]; then
    kill "$RECOVERY_PID" 2>/dev/null || true
    wait "$RECOVERY_PID" 2>/dev/null || true
    RECOVERY_PID=""
  fi
}

VLLM_PID=""
start_vllm() {
  if [ -z "${VLLM_GPUS:-}" ]; then
    return 0
  fi
  local port="${QWEN_PORT:-8000}"
  echo ">> Multi-GPU partition detected. Starting local Qwen-27B serving on GPUs $VLLM_GPUS..."
  
  local model_arg="Qwen/Qwen3.8-27B"
  local tp_size
  tp_size="$(echo "$VLLM_GPUS" | tr ',' '\n' | wc -l)"
  
  # Dynamically configure context len and memory utilization based on TP size
  local max_model_len=262144
  local gpu_utilization=0.92
  if [ "$tp_size" -eq 1 ]; then
    max_model_len=32768
    gpu_utilization=0.80
  fi
  
  echo ">> Serve configuration: tp_size=$tp_size, max_model_len=$max_model_len, gpu_utilization=$gpu_utilization"
  
  # Launch inside an isolated subshell so that compiler library exports 
  # do NOT bleed into and contaminate the parent autoembed script's active python environment.
  (
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TRITON_CACHE_DIR="/tmp/triton-${SLURM_JOB_ID:-local}"
    mkdir -p "$TRITON_CACHE_DIR" && chmod 1777 "$TRITON_CACHE_DIR"
    # Since Qwen3.8-27B is not on disk yet, do not set HF_HUB_OFFLINE=1 so it can download online
    export HF_HUB_OFFLINE=0
    export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
    export CPATH="/data/home/lakshyaaagrawal/.python-include:${CPATH:-}"
    export C_INCLUDE_PATH="/data/home/lakshyaaagrawal/.python-include:${C_INCLUDE_PATH:-}"
    export CUDA_VISIBLE_DEVICES="$VLLM_GPUS"

    vllm serve "$model_arg" \
      --tensor-parallel-size "$tp_size" \
      --host 0.0.0.0 \
      --port "$port" \
      --dtype bfloat16 \
      --max-model-len "$max_model_len" \
      --gpu-memory-utilization "$gpu_utilization" \
      --trust-remote-code \
      --disable-custom-all-reduce \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --reasoning-parser qwen3 \
      --served-model-name qwen3.8-27b
  ) > "$RESULTS/vllm.log" 2>&1 &
  VLLM_PID=$!
  
  echo ">> Waiting for Qwen (vLLM) to start on port $port..."
  local retries=120
  while ! curl -s "http://127.0.0.1:$port/v1/models" >/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "!! vLLM failed to start. Last lines of $RESULTS/vllm.log:" >&2
      tail -n 100 "$RESULTS/vllm.log" >&2
      exit 1
    fi
    retries=$((retries - 1))
    if [ "$retries" -le 0 ]; then
      echo "!! Timed out waiting for vLLM to start." >&2
      exit 1
    fi
    sleep 5
  done
  echo ">> Qwen (vLLM) is ready on http://127.0.0.1:$port/v1"
  if [ "$MODE" = docker ]; then
    export QWEN_API_BASE="http://host.docker.internal:$port/v1"
  else
    export QWEN_API_BASE="http://127.0.0.1:$port/v1"
  fi
}

stop_vllm() {
  if [ -n "$VLLM_PID" ]; then
    echo ">> Terminating background vLLM server (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
  fi
}

on_signal() {
  stop_recovery
  stop_vllm
  snapshot_final_model best-effort || true
  [ -z "${START:-}" ] || write_run_meta false || true
  exit 143
}

on_exit() {
  snapshot_final_model best-effort || true
  stop_vllm
  cleanup_auth_stage
}

trap on_signal TERM INT
trap on_exit EXIT

# ---- Agent runtime ----
sandbox() {   # run "$1" in the sandbox; the venv comes from the environment
  if [ "$MODE" = native ]; then
    ( cd "$WORK" && export UV_PROJECT_ENVIRONMENT="$ROOT/.venv" && bash -c "$1" )
  elif [ "$MODE" = docker ]; then
    # NVIDIA's device cgroup exposes only SELECTED_GPU to the container. It is
    # renumbered to logical CUDA device 0 inside, so do not pass the host index
    # as CUDA_VISIBLE_DEVICES.
    docker run --rm --gpus "device=$SELECTED_GPU" --add-host=host.docker.internal:host-gateway -v "$WORK":/work "${DOCKER_AUTH_MOUNT[@]}" -w /work \
      -e PROMPT -e AGENT_CONFIG -e DEADLINE -e AGENT_AUTH_MODE -e AUTOEMBED_AGENT_REASONING \
      -e AUTOEMBED_BASE_MODEL -e AUTOEMBED_BASE_REVISION -e AUTOEMBED_DEV_TASKS -e AUTOEMBED_QUERY_SPLIT -e AUTOEMBED_EXAMPLE_SPLIT -e AUTOEMBED_PER_TASK_TIMEOUT \
      -e TORCHDYNAMO_DISABLE \
      -e CUDA_VISIBLE_DEVICES=0 \
      -e HOME=/work/.home -e HF_HOME=/work/.cache/huggingface \
      -e TORCH_HOME=/work/.cache/torch -e UV_CACHE_DIR=/work/.cache/uv \
      -e UV_PROJECT_ENVIRONMENT=/opt/autoembed/.venv \
      -e ANTHROPIC_API_KEY -e CLAUDE_CODE_OAUTH_TOKEN \
      -e OPENAI_API_KEY -e GEMINI_API_KEY -e GOOGLE_API_KEY \
      -e QWEN_API_BASE \
      "${DOCKER_AUTH_ENV[@]}" \
      autoembed bash -c "$1"
  elif [ "$MODE" = enroot ]; then
    # Enroot mounts no host home. Restrict /dev, then inject only the selected
    # NVIDIA device; the physical device is logical CUDA device 0 inside.
    env -u ENROOT_MOUNT_HOME -u ENROOT_ROOTFS_WRITABLE \
      ENROOT_SYSCONF_PATH="$ROOT/enroot" ENROOT_RESTRICT_DEV=yes \
      NVIDIA_VISIBLE_DEVICES="$SELECTED_GPU" NVIDIA_DRIVER_CAPABILITIES=compute,utility \
      enroot start -m "$WORK:/work" "${ENROOT_AUTH_MOUNT[@]}" \
        -e PROMPT -e AGENT_CONFIG -e DEADLINE -e AGENT_AUTH_MODE -e AUTOEMBED_AGENT_REASONING \
        -e AUTOEMBED_BASE_MODEL -e AUTOEMBED_BASE_REVISION -e AUTOEMBED_DEV_TASKS -e AUTOEMBED_QUERY_SPLIT -e AUTOEMBED_EXAMPLE_SPLIT -e AUTOEMBED_PER_TASK_TIMEOUT \
        -e TORCHDYNAMO_DISABLE \
        -e CUDA_VISIBLE_DEVICES=0 -e HOME=/work/.home \
        -e HF_HOME=/work/.cache/huggingface -e TORCH_HOME=/work/.cache/torch -e UV_CACHE_DIR=/work/.cache/uv \
        -e UV_PROJECT_ENVIRONMENT=/opt/autoembed/.venv \
        -e ANTHROPIC_API_KEY -e CLAUDE_CODE_OAUTH_TOKEN -e OPENAI_API_KEY -e GEMINI_API_KEY -e GOOGLE_API_KEY \
        -e QWEN_API_BASE \
        "${ENROOT_AUTH_ENV[@]}" \
        "$ENROOT_CONTAINER" bash -c "cd /work && $1"
  else
    echo "!! unknown MODE=$MODE (expected native, enroot, or docker)" >&2
    return 2
  fi
}

echo ">> agent=$AGENT config=${AGENT_CONFIG:-default} base=$BASE_MODEL@$BASE_REVISION budget=${HOURS}h mode=$MODE"
echo ">> results=$RESULTS"

# ---- GPU selection and preflight ----
# Choose one physical device. Never replace a scheduler-assigned GPU with a
# globally visible one: doing so can escape the allocation on permissive nodes.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ -n "$GPU_ID" ]; then
    echo "!! GPU_ID must not be set inside Slurm; the scheduler owns GPU selection" >&2
    exit 1
  fi
  VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-}"
  if [ -z "$VISIBLE_GPUS" ] || [ "$VISIBLE_GPUS" = "NoDevFiles" ]; then
    echo "!! Slurm job has no visible GPU allocation" >&2
    exit 1
  fi
  if [[ "$VISIBLE_GPUS" == *,* ]]; then
    SELECTED_GPU="${VISIBLE_GPUS%%,*}"
    VLLM_GPUS="${VISIBLE_GPUS#*,}"
    echo ">> Multi-GPU Slurm allocation detected: $VISIBLE_GPUS"
    echo ">> SELECTED_GPU (agent) = $SELECTED_GPU"
    echo ">> VLLM_GPUS (Qwen server) = $VLLM_GPUS"
  else
    SELECTED_GPU="$VISIBLE_GPUS"
    VLLM_GPUS=""
    echo ">> Single-GPU Slurm allocation detected: SELECTED_GPU=$SELECTED_GPU"
  fi
  # fail fast if the assigned device is already occupied by an untracked process (e.g. a vLLM server)
  FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$SELECTED_GPU" 2>/dev/null || echo 0)"
  if [ "${FREE_MIB:-0}" -lt "${AUTOEMBED_MIN_FREE_MIB:-20000}" ]; then
    echo "!! assigned GPU $SELECTED_GPU has only ${FREE_MIB} MiB free (Slurm cannot see non-Slurm jobs); relaunch" >&2
    exit 75   # EX_TEMPFAIL: batch submissions requeue on this
  fi
  GPU_BOUNDARY="slurm"
elif [ -n "$GPU_ID" ]; then
  [[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "!! GPU_ID must be a numeric physical GPU index" >&2; exit 1; }
  SELECTED_GPU="$GPU_ID"
  if [ "$MODE" = docker ] || [ "$MODE" = enroot ]; then
    GPU_BOUNDARY="$MODE"
  else
    GPU_BOUNDARY="environment"
  fi
else
  command -v nvidia-smi >/dev/null 2>&1 \
    || { echo "!! nvidia-smi unavailable and no GPU_ID was provided" >&2; exit 1; }
  read -r SELECTED_GPU _ <<<"$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | tr ',' ' ')"
  if [ "$MODE" = docker ] || [ "$MODE" = enroot ]; then
    GPU_BOUNDARY="$MODE"
  else
    GPU_BOUNDARY="environment"
  fi
fi

if { [ "$MODE" = docker ] || [ "$MODE" = enroot ]; } && [ "$GPU_BOUNDARY" = slurm ]; then
  GPU_BOUNDARY="slurm+$MODE"
fi

if [ "$MODE" = native ]; then
  export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
fi

if [ "$REQUIRE_GPU_ENFORCEMENT" = 1 ] && [ "$GPU_BOUNDARY" = environment ]; then
  echo "!! native CUDA_VISIBLE_DEVICES is not an enforcement boundary: the agent can unset it" >&2
  echo "!! use MODE=enroot/docker, or launch through a one-GPU Slurm allocation" >&2
  exit 1
fi

echo ">> selected physical GPU $SELECTED_GPU; boundary=$GPU_BOUNDARY"
[ "$GPU_BOUNDARY" = environment ] \
  && echo ">> WARNING: native isolation is cooperative only; set REQUIRE_GPU_ENFORCEMENT=1 for reportable runs"

sandbox 'uv run --no-sync python check_cuda.py' \
  || { echo "!! exactly one CUDA GPU must be visible — aborting"; exit 1; }

case "$AGENT" in
  claude) VERSION_COMMAND='claude --version' ;;
  codex) VERSION_COMMAND='codex --version' ;;
  gemini) VERSION_COMMAND='gemini --version' ;;
esac
AGENT_VERSION="$(sandbox "$VERSION_COMMAND" 2>/dev/null | tail -n1 || true)"
echo ">> agent version=${AGENT_VERSION:-unknown}"

if [ "${AUTOEMBED_PREFLIGHT_ONLY:-0}" = 1 ]; then
  echo ">> preflight passed; agent and hidden scoring were not started"
  cleanup_auth_stage
  trap - EXIT TERM INT
  rm -rf -- "$WORK" "$RESULTS"
  exit 0
fi

# ---- Agent execution, hidden scoring, and archival ----
# START must precede the fork: the background loop inherits the environment as it
# stands at fork time, and write_run_meta reads START under `set -u`.
START=$(date +%s)
start_vllm
recovery_loop &
RECOVERY_PID=$!

write_run_meta false
set +e
# Append, never truncate: the sandbox can restart within a run, and a fresh `tee`
# would discard every session before it. usage_tap records token and cost events as
# they stream, so accounting survives a truncated or lost trace.
: > "$RESULTS/trace.log"
: > "$RESULTS/usage.jsonl"
sandbox "timeout --signal=TERM --kill-after=30s $LIMIT bash solve.sh" 2>&1 \
  | python3 "$ROOT/scripts/timestamp_lines.py" \
  | tee -a "$RESULTS/trace.log" \
  | python3 "$ROOT/scripts/usage_tap.py" "$RESULTS/usage.jsonl" \
  | { python3 "$ROOT/scripts/pretty_stream.py" 2>/dev/null || true; cat >/dev/null; }
rc=${PIPESTATUS[0]}
set -e
END=$(date +%s)
stop_recovery
snapshot_final_model || true
cleanup_auth_stage
[ "$rc" -eq 124 ] && echo ">> agent hit the ${HOURS}h budget (killed)"

python3 "$ROOT/scripts/parse_trace.py" "$RESULTS/trace.log" "$RESULTS/trace.md" || true

SCORE_RC=3
if [ -n "$(ls -A "$WORK/final_model" 2>/dev/null)" ]; then
  echo ">> scoring final_model"
  set +e
  ( cd "$ROOT" && AUTOEMBED_CONFIG="$CONFIG" AUTOEMBED_ROOT="$ROOT" \
      AUTOEMBED_NANO_DIR="$NANO_SOURCE" \
      AUTOEMBED_MTEB_OUTPUT="$RESULTS/mteb" \
      AUTOEMBED_ENCODER_WORKER_COMMAND="$ROOT/scripts/encoder_worker.sh" \
      AUTOEMBED_REQUIRE_ISOLATED_CUSTOM=1 AUTOEMBED_WORKER_MODE="$MODE" \
      AUTOEMBED_WORKER_MODEL="$WORK/final_model" AUTOEMBED_WORKER_ROOT="$ROOT" \
      AUTOEMBED_WORKER_GPU="$SELECTED_GPU" AUTOEMBED_WORKER_CONTAINER="$ENROOT_CONTAINER" \
      uv run python -m autoembed.scoring "$WORK/final_model" "$RESULTS/scores.json" ) 2>&1 \
    | tee "$RESULTS/score.log"
  SCORE_RC=${PIPESTATUS[0]}
  set -e
  cp -r "$WORK/final_model" "$RESULTS/final_model"
else
  echo ">> no final_model produced — skipping scoring"
fi

# workspace snapshot: agent's code + logs, minus weights, venv, caches
( cd "$WORK" && tar cf - --exclude=final_model --exclude=.venv --exclude=.claude-agent \
    --exclude=__pycache__ --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' \
    --exclude='*.ckpt' . 2>/dev/null ) \
  | ( mkdir -p "$RESULTS/workspace" && tar xf - -C "$RESULTS/workspace" 2>/dev/null ) || true

write_run_meta true

echo ">> done: $RESULTS"
trap - EXIT TERM INT
if [ "$SCORE_RC" -ne 0 ]; then
  echo "!! run is not reportable; scorer exit=$SCORE_RC" >&2
  exit "$SCORE_RC"
fi
