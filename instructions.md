# Task

Train a text-embedding model that scores as high as possible on a held-out benchmark,
starting from the fixed base model in `task.py` (`BASE_MODEL`). Save your best model to
`final_model/` as a sentence-transformers model; we score it once you stop.

You have complete freedom over the method — architecture, loss, optimization, and
training data (your own datasets, synthetic data, hard negatives, methods from recent
papers). How you iterate is up to you — but use the full budget: keep improving
`final_model/` until your time is nearly up rather than stopping early.

## Evaluation
We score `final_model/` on a hidden, comprehensive English embedding benchmark, aggregated as
the **equal-weight mean over its task types**. `evaluate_dev` mirrors that aggregation on
disjoint tasks — run it to see where you stand. Scoring is plain `SentenceTransformer.encode`
on raw text (no added prompts/prefixes).

Tools you can use:
- `evaluate_dev('final_model')` in `task.py` — a fast, multi-task validation proxy (disjoint
  from the hidden benchmark); prints per-type scores and the equal-weight mean.
- `check_contamination(ds)` in `task.py` — overlap of a dataset's text with the benchmark eval.
- `bash timer.sh` — remaining budget.

## Rules
- Only fine-tune `BASE_MODEL`; don't start from another already-trained embedding model.
- Don't modify `task.py`.
- Don't train on the benchmark's eval data — verify any dataset with `check_contamination`.
  Other public data is fine.
- Never ask for feedback — decide and act. We score `final_model/` when you stop.
- Single non-interactive session, no re-invocation: run long jobs like training in the
  **foreground** and let the command block until it finishes. Do not run them in the
  background (`&`, `nohup`, run-in-background) and do not wait via monitors or scheduled
  wakeups — you will not be resumed. We score `final_model/` as it is when you stop.
