# Task

Train a text-embedding model that scores as high as possible on a hidden held-out suite of
tasks, starting from the fixed base model in `task.py` (`BASE_MODEL`). Your submitted model
must be a fine-tune of `BASE_MODEL`; save it to `final_model/` so it loads with
`SentenceTransformer('final_model')`.

You have complete freedom over method and data: research relevant approaches, experiment, and
iterate on both the model and your approach. A development suite is provided:
`evaluate('final_model')` in `task.py` scores your model on the dev tasks (`DEV_TASKS`). The
hidden test is disjoint from the dev suite but the same kind of evaluation. Use the full
budget: keep improving `final_model/` until your time is nearly up.

## Evaluation

We score `final_model/` on the hidden suite with plain `SentenceTransformer` encoding
(batch size 64, sequence length capped at 512), aggregated as the equal-weight mean over
task types. Scoring uses only what is saved in `final_model/` — if your model needs prompts
or other encoding settings, save them in its config.

## Environment

One NVIDIA GPU; run Python with `uv run`. `sentence-transformers`, `mteb`, `datasets`,
`accelerate`, and `torch` are installed, and you have internet access to fetch public data
and models. Deprecation warnings are harmless.

Tools you can use:
- `evaluate('final_model', task_names=[...])` in `task.py` — score any model path on MTEB
  tasks; defaults to the dev suite.
- `check_contamination(ds)` in `task.py` — overlap of a dataset's text with the hidden test.
- `bash timer.sh` — remaining budget.

## Rules

- Your submitted model must be a fine-tune of `BASE_MODEL`. Using other models, data, or
  tools to help is allowed.
- Don't modify `task.py`.
- Don't train on the hidden test data — verify any dataset with `check_contamination`.
- Never ask for feedback — operate autonomously and decide for yourself.
- Non-interactive session: run long jobs in the foreground and let them finish before you end
  your turn. We score `final_model/` as it is whenever you stop.
