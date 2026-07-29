# autoembed

`autoembed` measures how well a coding agent can train an embedding model within a
fixed time and GPU budget. The agent receives a pinned base checkpoint, a visible
development evaluation, and an isolated work directory. The harness then scores its
`final_model/` on the complementary hidden split and audits the declared training data.

## Requirements

- Linux and one NVIDIA GPU
- [`uv`](https://docs.astral.sh/uv/)
- one supported agent CLI: Claude Code, Codex, or Antigravity
- Slurm + Enroot for cluster runs, or Docker for local isolated runs

Run commands from a source checkout:

```bash
uv sync --group dev
uv run autoembed list
uv run pytest tests/
```

`pyproject.toml` currently selects CUDA 12.8 PyTorch wheels. Adjust that index if the
host driver requires another build.

## Configs and reference scores

The config is the complete experiment protocol: pinned base and reference revisions,
development/hidden split, contamination policy, timeout, and isolation requirements.
Canonical configs are under `configs/general/` and `configs/specialization/`.

```bash
uv run autoembed list
uv run autoembed reference --config finance
uv run autoembed reference --config finance Alibaba-NLP/gte-modernbert-base
```

`reference` and `score` use Slurm by default. Add `--local --gpu 0` to run on a GPU
attached to the current machine. Model IDs passed to `reference` must already have a
pinned revision and loader in the selected config.

The two general MTEB-Nano configs also need the frozen JSON bundle in `runs/nano/`, or
`AUTOEMBED_NANO_DIR` pointing to it. The large data files are not stored in Git.
`agent_task/nano_assets.json` pins every filename and checksum; the launcher stops
before starting an agent if the bundle is missing or changed.

## Authentication and images

Build one immutable Enroot image per agent on the cluster:

```bash
srun --partition=guest --time=00:30:00 scripts/build_enroot.sh claude
srun --partition=guest --time=00:30:00 scripts/build_enroot.sh codex
srun --partition=guest --time=00:30:00 scripts/build_enroot.sh antigravity
```

Claude accepts `CLAUDE_CODE_OAUTH_TOKEN` for a subscription or `ANTHROPIC_API_KEY`.
Codex accepts `OPENAI_API_KEY`, or an imported subscription credential. Antigravity
uses an imported portable subscription token:

```bash
scripts/agent_auth.sh codex /secure/path/auth.json
scripts/agent_auth.sh antigravity /secure/path/antigravity-oauth-token
```

Credentials are copied into the ignored `.agent-auth/` directory, mounted through a
private per-run staging directory, and never copied into results.

## Run and score

```bash
# End-to-end preflight. This does not start the agent or hidden scorer.
MODE=enroot AUTOEMBED_PREFLIGHT_ONLY=1 \
  uv run autoembed run --config finance --agent claude --hours 10

# Cluster run: 10 agent hours plus 2 hours for setup and hidden scoring.
uv run autoembed run --config finance --agent claude --hours 10 --time 12:00:00
uv run autoembed run --config mteb-nano-create --agent codex --hours 10 --time 12:00:00

# Local Docker run on physical GPU 0.
uv run autoembed run --config medical --agent antigravity \
  --local --gpu 0 --isolation docker --hours 10

# Preview a pinned base override without launching.
uv run autoembed run --config finance --agent codex \
  --base <hf-id> --base-revision <40-character-commit> --dry-run

# Rescore a saved model or recovery snapshot.
uv run autoembed score results/<run>/recovery/<timestamp> --config finance
```

The shell entry points remain usable directly. For example:

```bash
MODE=enroot AUTOEMBED_CONFIG=configs/specialization/finance.json HOURS=10 \
  scripts/gpu.sh scripts/run_task.sh claude claude-opus-5 10
```

Canonical configs require filesystem isolation. Native mode is available for trusted
local debugging, but it is cooperative and not a security boundary. Custom
`mteb_model.py` submissions are executable code and therefore score only through the
isolated Docker encoder worker; standard offline SentenceTransformer directories work
with Enroot or Docker.

## Evaluation and run records

Canonical protocols use a deterministic 50/50 example split: agents see one half and
the harness scores the exact complement. `allow_target_corpus_training` is `false`, so
evaluation queries, labels, and corpora are not training data. This is the appropriate
inductive setup for the current claims; a separate transductive protocol would set it
to `true` explicitly.

Every submitted model must include an exhaustive `training_manifest.json` declaring its
sources. Three things flag a run: a missing, non-exhaustive, or unsourced manifest; a
hidden query and one of its relevant documents in the same training row; and wholesale
ingestion of evaluation text. A flagged run is scored as its own base model rather than
dropped, so it keeps its row in the results table and the flag costs the claimed gain.
Text collisions short of those thresholds are reported on the unchanged test set instead
of silently removing examples — the evaluation tasks are built from the same public
datasets embedders train on, so overlap is expected. This audit detects exact reuse and
records provenance; it does not prove that an adversarial participant could not
reconstruct the split.

Each `results/<run>/` directory contains the prompt, timestamped trace, workspace
snapshot, recovery checkpoints, final model, score log, `scores.json`, and `meta.json`.
Metadata records the config/scorer hashes, Git state, agent version, authentication
mode, GPU/container provenance, completion state, and available usage/cost fields.
Unavailable cost is `null`, not zero; subscription runs are not presented as per-run
billed API cost.

## Project layout

- `autoembed/`: CLI, hidden scoring, reference scoring, encoder worker
- `configs/`: canonical versioned experiment protocols
- `agent_task/`: files copied into the agent work directory
- `agents/`: Claude, Codex, and Antigravity launchers
- `scripts/`: run, GPU, image, authentication, trace, and metadata tools
- `tests/`: scoring, contamination, submission, and accounting tests

See `configs/README.md` for the rationale behind the canonical task sets and data
policies.
