# The task: fixed base model + a training-vs-eval contamination check.
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"
MODEL_DIR = ROOT / "final_model"  # the agent's submitted model
BASE_MODEL = os.environ.get("AUTOEMBED_BASE_MODEL", "intfloat/e5-base-unsupervised")

EVAL_BENCHMARK = "MTEB(eng, v2)"   # the hidden held-out; the agent never scores on it


def _eval_tasks():
    import mteb
    return mteb.get_benchmark(EVAL_BENCHMARK).tasks


def _norm(s):
    return " ".join(s.lower().split())


def _collect(obj, out, cap):
    if len(out) >= cap:
        return
    if isinstance(obj, str):
        if obj.strip():
            out.add(_norm(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, out, cap)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, out, cap)
    elif hasattr(obj, "column_names"):  # datasets.Dataset
        for col in obj.column_names:
            _collect(obj[col], out, cap)


def _eval_texts(tasks=None, cap=200_000):
    out = set()
    for t in (tasks or _eval_tasks()):
        t.load_data()
        for attr in ("corpus", "queries", "dataset"):
            _collect(getattr(t, attr, None), out, cap)
    return out


def check_contamination(train_dataset, sample=100_000):
    # Exact-match overlap of your training text with the eval; writes a training-text sample.
    evalset = _eval_texts()
    n = min(sample, len(train_dataset))
    ds = train_dataset.select(range(n))
    cols = [c for c in ("anchor", "positive", "negative", "query", "text") if c in ds.column_names]
    train_texts, hits, examples = set(), 0, []
    for col in cols:
        for s in ds[col]:
            if not s:
                continue
            t = _norm(s)
            train_texts.add(t)
            if t in evalset:
                hits += 1
                if len(examples) < 5:
                    examples.append(s[:80])
    total = max(n * len(cols), 1)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "_train_texts.json").write_text(json.dumps(sorted(train_texts)))
    return {"checked": total, "eval_texts": len(evalset), "hits": hits,
            "frac": round(hits / total, 6), "examples": examples}
