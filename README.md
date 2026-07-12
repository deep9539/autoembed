# autoembed

Can an agent train a good embedding model on its own? `autoembed` is a benchmark
for *agentic* embedding-model training: a CLI agent (Claude Code, Codex, Gemini…)
is given a fixed base model, a wall-clock budget on one GPU, and a dev eval to
query — and must produce the best embedding model it can. The harness then scores
the agent's `final_model/` on a frozen eval, including a held-out set the agent
never sees.

The measurement is frozen; the method is free.

## How it works

1. `run_task.sh <agent>` seeds a fresh workdir with the **agent-facing files only**
   (`task.py`, `instructions.md`, `timer.sh`), sets a deadline, and launches the
   agent via `agents/<agent>/solve.sh`.
2. The agent reads `instructions.md`, writes its own training code (using the
   helpers in `task.py`), self-evaluates on the **dev suite** (`evaluate`),
   paces itself with `bash timer.sh`, and saves its best model to `final_model/`.
3. When the agent stops, `score.py` runs **outside** the workdir and scores
   `final_model/` on the **hidden held-out** set, plus a contamination audit.

Integrity is structural, not honor-system: the scorer (`score.py`), which defines
and runs the held-out set, never enters the agent's workdir, and the harness scores
from its own copy — so the agent can't see or tamper with the official metric.

## Files

| | |
|---|---|
| `config.json` | the task: base model, dev tasks, held-out tasks |
| `task.py` | fixed base model, dev eval, contamination check (the agent gets a copy) |
| `instructions.md` | the prompt given to the agent |
| `agents/<agent>/solve.sh` | per-agent launchers (claude · codex · gemini) |
| `run_task.sh` | orchestrator: seed workdir → run agent → score |
| `timer.sh` | remaining-budget query the agent calls |
| `score.py` | harness scoring: hidden held-out + contamination (harness-only, never copied to the agent) |
| `reference.py` | reference ladder on the held-out: base floor + model ids passed as args |
| `Dockerfile` | container env; run with `docker run --gpus all` |

## Define a task

`config.json` is the task definition — everything that varies between experiments:

```jsonc
{
  "base_model":    "answerdotai/ModernBERT-base",   // the frozen starting point
  "dev_tasks":     ["NanoMSMARCORetrieval", "…"],   // given to the agent (MTEB task names)
  "heldout_tasks": ["NFCorpus", "…"]                // hidden test (harness-only)
}
```

Point `AUTOEMBED_CONFIG` at another file to switch tasks (e.g. a domain variant).
Keep `dev_tasks` and `heldout_tasks` on disjoint datasets — `score.py` warns on
overlap (a Nano dev task and its full version count as the same dataset). The
contamination cache (`_eval_texts.json`) is rebuilt automatically whenever the
config is newer than the cache.

To place a result, score reference models on the same held-out:

```bash
uv run python reference.py nomic-ai/modernbert-embed-base intfloat/e5-base-v2
```

## Run

Needs a local NVIDIA GPU, the agent CLI installed + authenticated (API key or
subscription), and `uv`.

```bash
uv sync
./run_task.sh claude              # native: run Claude Code for the default budget
HOURS=10 ./run_task.sh codex      # 10-hour budget with Codex
MODE=docker ./run_task.sh gemini  # isolated container run (needs the agent CLI in the image)
```

GPU note: `pyproject.toml` pins torch to a CUDA 12.8 wheel index; adjust it to
your driver (see the comment there).
