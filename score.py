# Harness-only scoring on the hidden held-out + contamination audit. The held-out
# task identities live in config.json (never copied to the agent workdir).
import hashlib
import json
import os
import sys
from pathlib import Path

import mteb

from task import _score, _eval_texts, _collect, _EVAL_CACHE, MODEL_DIR

CONFIG_PATH = Path(os.environ.get("AUTOEMBED_CONFIG", Path(__file__).parent / "config.json"))
CONFIG = json.loads(CONFIG_PATH.read_text())
HELDOUT_TASK_NAMES = CONFIG["heldout_tasks"]
HELDOUT_TASKS = mteb.get_tasks(tasks=HELDOUT_TASK_NAMES)


def _dataset_root(name):
    return name.lower().removeprefix("nano").removesuffix("retrieval")


_overlap = ({_dataset_root(n) for n in CONFIG["dev_tasks"]}
            & {_dataset_root(n) for n in HELDOUT_TASK_NAMES})
if _overlap:
    print(f"!! dev/heldout dataset overlap: {sorted(_overlap)}")


def build_eval_cache(cap=300_000):
    # Hash the held-out text set for the agent-side contamination check.
    # Queries first (all must be present), then corpus text up to cap.
    splits = []
    for t in HELDOUT_TASKS:
        t.load_data()
        for subset in t.dataset.values():
            splits.extend(subset.values())
    raw = set()
    for key in ("queries", "corpus"):
        for s in splits:
            _collect(s.get(key), raw, cap)
    hashes = sorted(hashlib.md5(s.encode("utf-8")).hexdigest()[:16] for s in raw)
    _EVAL_CACHE.write_text(json.dumps(hashes))
    return len(hashes)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else str(MODEL_DIR)
    out = sys.argv[2] if len(sys.argv) > 2 else None

    r = _score(model, HELDOUT_TASKS, tag="heldout")

    contam = None
    sample = Path(model).resolve().parent / "runs" / "_train_texts.json"
    if sample.exists():
        train = set(json.loads(sample.read_text()))  # agent-side hashes
        evalset = _eval_texts()
        hits = sum(1 for t in train if t in evalset)
        contam = {"hits": hits, "frac": round(hits / max(len(train), 1), 6)}

    per_type = {k: round(v, 4) for k, v in r["per_type"].items()}
    result = {"mean_type": r["mean_type"], "mean_task": r["mean_task"],
              "per_type": r["per_type"], "heldout_per_task": r["per_task"],
              "skipped": r["skipped"], "contamination": contam}
    print(f"MEAN_TYPE={r['mean_type']:.4f}  MEAN_TASK={r['mean_task']:.4f}  "
          f"per_type={per_type}  skipped={len(r['skipped'])}  contam={contam}")
    if out:
        Path(out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
