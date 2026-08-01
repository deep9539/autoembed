#!/usr/bin/env bash
# Start the hidden scorer's custom encoder with only final_model and worker code visible.
set -euo pipefail
MODE="${AUTOEMBED_WORKER_MODE:?missing AUTOEMBED_WORKER_MODE}"
MODEL="${AUTOEMBED_WORKER_MODEL:?missing AUTOEMBED_WORKER_MODEL}"
ROOT="${AUTOEMBED_WORKER_ROOT:?missing AUTOEMBED_WORKER_ROOT}"
GPU="${AUTOEMBED_WORKER_GPU:?missing AUTOEMBED_WORKER_GPU}"
case "$MODE" in
  enroot)
    echo "custom Python submissions require MODE=docker for hidden scoring; Enroot is not an untrusted-code security boundary" >&2
    exit 2
    ;;
  docker)
    docker run --rm --read-only --network none --gpus "device=$GPU" \
      -v "$MODEL:/model:ro" -v "$ROOT/autoembed:/worker/autoembed:ro" \
      -v "$ROOT/agent_task:/worker/agent_task:ro" --tmpfs /tmp:rw,noexec,nosuid,size=2g \
      -w /tmp -e CUDA_VISIBLE_DEVICES=0 -e HOME=/tmp -e PYTHONPATH=/worker \
      -e AUTOEMBED_BASE_MODEL -e AUTOEMBED_BASE_REVISION -e AUTOEMBED_PER_TASK_TIMEOUT \
      -e UV_PROJECT_ENVIRONMENT=/opt/autoembed/.venv \
      autoembed bash -c 'uv run --no-sync python -m autoembed.encoder_worker /model'
    ;;
  native)
    [ "${AUTOEMBED_TRUST_CUSTOM_CODE:-0}" = 1 ] || {
      echo "native custom scoring is disabled; use enroot/docker or explicitly set AUTOEMBED_TRUST_CUSTOM_CODE=1" >&2
      exit 2
    }
    cd /tmp
    PYTHONPATH="$ROOT" UV_PROJECT_ENVIRONMENT="$ROOT/.venv" \
      uv run --no-sync python -m autoembed.encoder_worker "$MODEL"
    ;;
  *) echo "unknown worker mode: $MODE" >&2; exit 2 ;;
esac
