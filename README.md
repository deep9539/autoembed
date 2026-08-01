# autoembed

`autoembed` measures how well a coding agent can train an embedding model within a
fixed time and GPU budget. The agent receives a pinned base checkpoint, a visible
development evaluation, and an isolated work directory. The harness then scores its
`final_model/` on the complementary hidden split and audits the declared training data.

A run costs one GPU for the agent budget you set, plus roughly two hours for setup and
hidden scoring. Ten hours is the canonical budget.

## Requirements

- Linux and one NVIDIA GPU
- [`uv`](https://docs.astral.sh/uv/)
- one supported agent CLI installed and licensed: Claude Code, Codex, or Gemini CLI
- a container runtime, see *Choosing how to run* below

```bash
uv sync --group dev
uv run autoembed list
uv run pytest tests/
```

`pyproject.toml` selects CUDA 12.8 PyTorch wheels. Adjust that index if the host
driver needs another build.

## Choosing how to run

The agent runs inside a container so that it cannot reach the hidden evaluation data,
the repository, or your credentials. Pick whichever fits your machine.

| Your setup | Flags | Runtime |
|---|---|---|
| Slurm cluster | default | Enroot |
| Workstation with a GPU | `--local --gpu 0` | Docker |
| Debugging only | `--local --gpu 0 --isolation native` | none |

[Enroot](https://github.com/NVIDIA/enroot) is NVIDIA's rootless container runtime for
HPC. It is the default for cluster runs because it needs no daemon and no root, which
is usually what shared clusters allow. If you are not on Slurm, use Docker instead;
nothing in the protocol depends on Enroot specifically.

Native mode skips containers entirely. It is cooperative rather than enforced, so use
it only for trusted local debugging. Canonical configs refuse to run natively.

`scripts/gpu.sh` is our Slurm submitter and is **environment-specific**: it hardcodes
partition defaults for our cluster. Adapt or replace it for yours.

## Authenticating the agent

The agent CLI needs your own subscription or API credentials. Nothing is bundled.

**Claude Code** reads an environment variable, which the harness forwards into the
container:

```bash
read -rsp 'token: ' CLAUDE_CODE_OAUTH_TOKEN && export CLAUDE_CODE_OAUTH_TOKEN
# or, for API billing instead of a subscription:
export ANTHROPIC_API_KEY=...
```

**Codex** accepts `OPENAI_API_KEY` the same way, or an imported subscription
credential. **Gemini CLI** uses `GEMINI_API_KEY` (or `GOOGLE_API_KEY`):

```bash
scripts/agent_auth.sh codex /secure/path/auth.json
export GEMINI_API_KEY=...
```

`agent_auth.sh` copies the credential into the gitignored `.agent-auth/` directory at
mode 600. At run time it is mounted through a private per-run staging directory and is
never written into `results/`. Export variables in the shell that launches the run; if
you use tmux, export them in each session separately.

## Building the container images

Once per agent, on a machine with the runtime available:

```bash
# cluster
srun --partition=<your-partition> --time=00:30:00 scripts/build_enroot.sh claude

# workstation
docker build -t autoembed-claude --build-arg AGENT_CLI=claude .
```

## A first run

```bash
# 1. check the config list
uv run autoembed list

# 2. preflight: validates the config, data, image, and credentials without
#    starting the agent or the hidden scorer
MODE=enroot AUTOEMBED_PREFLIGHT_ONLY=1 \
  uv run autoembed run --config specialization/finance --agent claude --hours 10

# 3. launch
uv run autoembed run --config specialization/finance --agent claude --hours 10 --time 12:00:00

# 4. results land in results/<run>/ ; rescore a checkpoint at any time
uv run autoembed score results/<run>/recovery/<timestamp> --config specialization/finance
```

On a workstation, replace step 3 with:

```bash
uv run autoembed run --config specialization/medical --agent claude \
  --local --gpu 0 --isolation docker --hours 10
```

## Choosing the agent and its model

`--agent` selects the CLI; `--model` selects the model that CLI drives. Both are
recorded in `meta.json`, so a result is always attributable to one exact pairing.

| `--agent` | CLI | default `--model` |
|---|---|---|
| `claude` | Claude Code | `claude-opus-5` |
| `codex` | Codex | `gpt-5.6-sol` |
| `gemini` | Gemini CLI | `gemini-3.6-flash` |

```bash
# the same setting, run by two models from one family
uv run autoembed run --config specialization/legal --agent claude --model claude-opus-5   --hours 10
uv run autoembed run --config specialization/legal --agent claude --model claude-sonnet-5 --hours 10

# and by another family
uv run autoembed run --config specialization/legal --agent gemini --model gemini-3.6-flash --hours 10
```

Pass any model id the CLI accepts. `meta.json` records both the requested id and the
one the CLI reports serving, because a CLI may substitute a model it does not
recognise; compare `model_identity.requested` with `model_identity.served` before
reporting a result.

Other useful forms:

```bash
# score the pinned base and reference encoders on the hidden split
uv run autoembed reference --config specialization/finance

# preview a base-model override without launching
uv run autoembed run --config specialization/finance --agent codex \
  --base <hf-id> --base-revision <40-character-commit> --dry-run
```

## Data

The four `configs/specialization/` configs pull everything they need from the Hugging Face
Hub on first use.

The two `configs/general/` configs evaluate a 40-task MTEB subset. Nine heavy retrieval
tasks are replaced by frozen subsamples of 200 queries against roughly 10,000 documents
each, about 120 MB that is not stored in Git.

**Pending release.** MTEB-nano is being prepared for release as an MTEB benchmark, in
line with the other benchmarks there. Until then the two `configs/general/` protocols
cannot be run elsewhere; the four `configs/specialization/` protocols need nothing
beyond the Hub and are unaffected. Subsampling is frozen rather than regenerated, so a
bundle rebuilt from the same tasks would not reproduce these protocols — the released
files are what makes them reproducible.

Once the bundle is available, place it in `runs/nano/` or point `AUTOEMBED_NANO_DIR` at
it. `agent_task/nano_assets.json` pins every filename and checksum, and the launcher
stops before starting an agent if anything is missing or altered.

## Configs

A config is the complete experiment protocol: pinned base and reference revisions,
the development and hidden split, contamination policy, per-task timeout, and isolation
requirements. Changing any field changes the config fingerprint recorded with every
score, so results are always attributable to an exact specification.

See `configs/README.md` for the rationale behind the task sets and data policies.

## Evaluation and run records

Canonical protocols use a deterministic 50/50 example split: agents see one half and the
harness scores the exact complement. `allow_target_corpus_training` is `false`, so
evaluation queries, labels, and corpora are not training data.

Every submitted model must include an exhaustive `training_manifest.json` declaring its
sources. Three things flag a run: a missing, non-exhaustive, or unsourced manifest; a
hidden query and one of its relevant documents in the same training row; and wholesale
ingestion of evaluation text. A flagged run is scored as its own base model rather than
dropped, so it keeps its row in the results table and the flag costs the claimed gain.
Smaller text collisions are reported on the unchanged test set instead of silently
removing examples, because the evaluation tasks are built from the same public datasets
embedders train on, so some overlap is expected. The audit detects exact reuse and
records provenance; it does not prove that an adversarial participant could not
reconstruct the split.

Each `results/<run>/` directory contains the prompt, timestamped trace, workspace
snapshot, recovery checkpoints, final model, score log, `scores.json`, and `meta.json`.
Metadata records the config and scorer hashes, Git state, agent version, authentication
mode, GPU and container provenance, completion state, and usage fields. Unavailable cost
is `null` rather than zero, and subscription runs are not presented as per-run billed
API cost.

Submissions containing a custom `mteb_model.py` are executable code and score only
through the isolated Docker encoder worker. Standard offline SentenceTransformer
directories work under Enroot or Docker.

## Project layout

- `autoembed/`: CLI, hidden scoring, reference scoring, encoder worker
- `configs/`: canonical experiment protocols
- `agent_task/`: files copied into the agent work directory
- `agents/`: Claude, Codex, and Gemini launchers
- `scripts/`: run, GPU, image, authentication, trace, and metadata tools
- `tests/`: scoring, contamination, submission, and accounting tests
