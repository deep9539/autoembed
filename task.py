# Agent-facing API: fixed base model, dev evaluation, contamination check.
import hashlib
import json
import os
import signal
from pathlib import Path

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"
MODEL_DIR = ROOT / "final_model"  # the agent's submitted model
_EVAL_CACHE = ROOT / "_eval_texts.json"  # hashed held-out text set

BASE_MODEL = os.environ.get("AUTOEMBED_BASE_MODEL", "answerdotai/ModernBERT-base")
DEV_TASKS = os.environ.get(
    "AUTOEMBED_DEV_TASKS",
    "NanoMSMARCORetrieval,NanoNQRetrieval,NanoFiQA2018Retrieval,"
    "NanoArguAnaRetrieval,NanoSCIDOCSRetrieval").split(",")
MAX_SEQ = int(os.environ.get("AUTOEMBED_MAX_SEQ", "512"))  # scoring seq-length cap
PER_TASK_TIMEOUT = 1200  # seconds; a task exceeding this is skipped, not fatal


class _Timeout(Exception):
    pass


def _on_timeout(signum, frame):
    raise _Timeout()


def _score(model_path, tasks, tag, trust_remote_code=False):
    # Mean over tasks + per-type + per-task on the given MTEB task objects.
    import mteb
    from sentence_transformers import SentenceTransformer
    type_of = {t.metadata.name: t.metadata.type for t in tasks}
    model = SentenceTransformer(str(model_path), trust_remote_code=trust_remote_code)
    model.max_seq_length = min(model.max_seq_length or MAX_SEQ, MAX_SEQ)
    per_task, skipped = {}, []
    signal.signal(signal.SIGALRM, _on_timeout)
    for t in tasks:
        name = t.metadata.name
        try:
            signal.alarm(PER_TASK_TIMEOUT)
            res = mteb.MTEB(tasks=[t]).run(
                model, output_folder=str(RUNS_DIR / "mteb" / tag),
                verbosity=0, overwrite_results=True, encode_kwargs={"batch_size": 64})
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


def evaluate(model_path=MODEL_DIR, task_names=None):
    # Score a model on MTEB tasks; defaults to the dev suite (DEV_TASKS).
    import mteb
    names = task_names or DEV_TASKS
    r = _score(model_path, mteb.get_tasks(tasks=names), tag="dev")
    print(f"mean_type={r['mean_type']:.4f}  mean_task={r['mean_task']:.4f}")
    for ty, s in sorted(r["per_type"].items()):
        print(f"   {ty:14s} {s:.4f}")
    if r["skipped"]:
        print(f"   (skipped: {r['skipped']})")
    return r


def _norm(s):
    return " ".join(s.lower().split())


def _h(s):
    return hashlib.md5(_norm(s).encode("utf-8")).hexdigest()[:16]


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
            for v in obj[col]:
                if len(out) >= cap:
                    return
                _collect(v, out, cap)


def _eval_texts():
    # Hashed held-out text set, precomputed by the harness (score.build_eval_cache).
    if _EVAL_CACHE.exists():
        return set(json.loads(_EVAL_CACHE.read_text()))
    return set()


def check_contamination(train_dataset, sample=100_000):
    # Hash-overlap of training text with the held-out set; writes a hashed sample
    # the harness re-checks at scoring time.
    evalset = _eval_texts()
    n = min(sample, len(train_dataset))
    ds = train_dataset.select(range(n))
    cols = [c for c in ("anchor", "positive", "negative", "query", "text") if c in ds.column_names]
    train_hashes, hits, examples = set(), 0, []
    for col in cols:
        for s in ds[col]:
            if not s:
                continue
            h = _h(s)
            train_hashes.add(h)
            if h in evalset:
                hits += 1
                if len(examples) < 5:
                    examples.append(s[:80])
    total = max(n * len(cols), 1)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "_train_texts.json").write_text(json.dumps(sorted(train_hashes)))
    return {"checked": total, "eval_texts": len(evalset), "hits": hits,
            "frac": round(hits / total, 6), "examples": examples}
