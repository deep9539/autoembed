# The task: fixed base model, a runnable multi-task dev proxy, and a contamination check.
import json
import os
import signal
from pathlib import Path

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"
MODEL_DIR = ROOT / "final_model"  # the agent's submitted model
_EVAL_CACHE = ROOT / "_eval_texts.json"   # cached eval-text set (fast contamination check)
BASE_MODEL = os.environ.get("AUTOEMBED_BASE_MODEL", "intfloat/e5-base-unsupervised")

EVAL_BENCHMARK = "MTEB(eng, v2)"   # the hidden held-out; the harness scores it

# Dev proxy: one task per type, disjoint from the held-out; tracks it (Spearman 0.959).
DEV_TASKS = ["NFCorpus", "STS16", "EmotionClassification",
             "WikiCitiesClustering", "SciDocsRR", "OpusparcusPC"]
PER_TASK_TIMEOUT = 1200   # seconds; a task exceeding this is skipped, not fatal


class _Timeout(Exception):
    pass


def _on_timeout(signum, frame):
    raise _Timeout()


def _score(model_path, tasks, tag):
    # Score a model on the given MTEB task objects: Mean over types + per-type + per-task.
    import mteb
    from sentence_transformers import SentenceTransformer
    type_of = {t.metadata.name: t.metadata.type for t in tasks}
    model = SentenceTransformer(str(model_path))
    per_task, skipped = {}, []
    signal.signal(signal.SIGALRM, _on_timeout)
    for t in tasks:
        name = t.metadata.name
        try:
            signal.alarm(PER_TASK_TIMEOUT)
            res = mteb.MTEB(tasks=[t]).run(
                model, output_folder=str(RUNS_DIR / "mteb" / tag),
                verbosity=0, overwrite_results=True, encode_kwargs={"batch_size": 256})
            per_task[name] = float(res[0].get_score())
        except Exception as e:
            skipped.append(name)
            print(f"  !! skipped {name}: {repr(e)[:100]}")
        finally:
            signal.alarm(0)
    per_type = {}
    for name, sc in per_task.items():
        per_type.setdefault(type_of[name], []).append(sc)
    type_means = {ty: sum(v) / len(v) for ty, v in per_type.items()}
    mean_type = sum(type_means.values()) / len(type_means) if type_means else 0.0
    mean_task = sum(per_task.values()) / len(per_task) if per_task else 0.0
    return {"mean_type": mean_type, "mean_task": mean_task,
            "per_type": type_means, "per_task": per_task, "skipped": skipped}


def evaluate_dev(model_path=MODEL_DIR):
    # Fast multi-task validation proxy (disjoint from the hidden held-out, tracks it well).
    # Optimize mean_type; watch the per-type breakdown so you don't over-specialize.
    import mteb
    tasks = mteb.get_tasks(tasks=DEV_TASKS, languages=["eng"])
    r = _score(model_path, tasks, tag="dev")
    print(f"DEV  mean_type={r['mean_type']:.4f}  mean_task={r['mean_task']:.4f}")
    for ty, s in sorted(r["per_type"].items()):
        print(f"     {ty:18s} {s:.4f}")
    if r["skipped"]:
        print(f"     (skipped: {r['skipped']})")
    return r


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


def _eval_texts(cap=200_000):
    # Cache the eval-text set; recompute only when the cache is missing.
    if _EVAL_CACHE.exists():
        return set(json.loads(_EVAL_CACHE.read_text()))
    import mteb
    out = set()
    for t in mteb.get_benchmark(EVAL_BENCHMARK).tasks:
        t.load_data()
        for attr in ("corpus", "queries", "dataset"):
            _collect(getattr(t, attr, None), out, cap)
    _EVAL_CACHE.write_text(json.dumps(sorted(out)))
    return out


def check_contamination(train_dataset, sample=100_000):
    # Exact-match overlap of your training text with the benchmark eval; writes a sample.
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
