# Task

Train a text-embedding model that scores as high as possible on a held-out benchmark,
starting from the fixed base model in `task.py` (`BASE_MODEL`). Save your best model to
`final_model/` as a sentence-transformers model; we score it once you stop.

You have complete freedom over the method — architecture, loss, optimization, and
training data (your own datasets, synthetic data, hard negatives, methods from recent
papers). How you iterate is up to you.

## Evaluation
We score `final_model/` on a comprehensive English embedding benchmark spanning retrieval,
STS, classification, clustering, reranking, and pair classification. You do **not** get to
run it — build your own validation to guide training, as you would when training a real
embedding model. Scoring is plain `SentenceTransformer.encode` on raw text (no added
prompts/prefixes).

Tools you can use:
- `check_contamination(ds)` in `task.py` — reports how much a dataset's text overlaps the
  eval. Use it to keep both training and validation data off the eval; you don't need to
  guess what's safe, just check it.
- `bash timer.sh` — remaining budget.

## Rules
- Only fine-tune `BASE_MODEL`; don't start from another already-trained embedding model.
- Don't modify `task.py`.
- Don't train or validate on the benchmark's eval data — verify any dataset with
  `check_contamination`. Other public data is fine.
- Never ask for feedback — decide and act. We score `final_model/` when you stop.
- Single non-interactive session, no re-invocation: run long jobs like training in the
  **foreground** and let the command block until it finishes. Do not run them in the
  background (`&`, `nohup`, run-in-background) and do not wait via monitors or scheduled
  wakeups — you will not be resumed. We score `final_model/` as it is when you stop.
