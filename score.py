# Harness scoring of final_model on the hidden held-out (MTEB(eng, v2) minus
# MindSmallReranking) + a contamination audit. Runs outside the agent workdir; writes
# scores.json when an output path is given. Uses the same scorer as the agent's evaluate_dev.
import json
import sys
from pathlib import Path

import mteb

from task import _score, _eval_texts, MODEL_DIR, EVAL_BENCHMARK

_EXCLUDE = {"MindSmallReranking"}   # >1h/model to score (> all 40 other tasks combined)
HELDOUT_TASKS = [t for t in mteb.get_benchmark(EVAL_BENCHMARK).tasks
                 if t.metadata.name not in _EXCLUDE]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else str(MODEL_DIR)
    out = sys.argv[2] if len(sys.argv) > 2 else None

    r = _score(model, HELDOUT_TASKS, tag="heldout")

    contam = None
    sample = Path(model).resolve().parent / "runs" / "_train_texts.json"
    if sample.exists():
        train = set(json.loads(sample.read_text()))
        evalset = _eval_texts()
        hits = sum(1 for t in train if t in evalset)
        contam = {"hits": hits, "frac": round(hits / max(len(train), 1), 6)}

    result = {"mean_type": r["mean_type"], "mean_task": r["mean_task"],
              "per_type": r["per_type"], "heldout_per_task": r["per_task"],
              "skipped": r["skipped"], "contamination": contam}
    print(f"MEAN_TYPE={r['mean_type']:.4f}  MEAN_TASK={r['mean_task']:.4f}  "
          f"scored={len(r['per_task'])}  skipped={len(r['skipped'])}  contam={contam}")
    if out:
        Path(out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
