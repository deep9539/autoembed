# Harness scoring of final_model on the hidden held-out (full MTEB(eng, v2), Mean over
# task types) + a training-data contamination audit. Runs outside the agent workdir;
# writes scores.json when an output path is given. Scoring is plain
# SentenceTransformer.encode on raw text (no added prompts/prefixes). Each task has a hard
# timeout: a task that exceeds it is skipped and recorded, never hangs the whole run.
import json
import signal
import sys
from pathlib import Path

import mteb
from sentence_transformers import SentenceTransformer

from task import _eval_texts, MODEL_DIR, RUNS_DIR, EVAL_BENCHMARK

_EXCLUDE = {"MindSmallReranking"}   # >1h/model to score (> all 40 other tasks combined)
HELDOUT_TASKS = [t for t in mteb.get_benchmark(EVAL_BENCHMARK).tasks
                 if t.metadata.name not in _EXCLUDE]
PER_TASK_TIMEOUT = 1200   # seconds; a task exceeding this is skipped, not fatal


class _Timeout(Exception):
    pass


def _on_timeout(signum, frame):
    raise _Timeout()


def score_model(model_path, tasks=HELDOUT_TASKS, tag="heldout"):
    model = SentenceTransformer(str(model_path))
    type_of = {t.metadata.name: t.metadata.type for t in tasks}
    per_task, skipped = {}, []
    signal.signal(signal.SIGALRM, _on_timeout)
    for t in tasks:
        name = t.metadata.name
        try:
            signal.alarm(PER_TASK_TIMEOUT)
            res = mteb.MTEB(tasks=[t]).run(
                model, output_folder=str(RUNS_DIR / "mteb" / tag),
                verbosity=0, overwrite_results=True,
                encode_kwargs={"batch_size": 256})
            per_task[name] = float(res[0].get_score())
        except Exception as e:
            skipped.append(name)
            print(f"  !! skipped {name}: {repr(e)[:100]}")
        finally:
            signal.alarm(0)
    per_type = {}
    for name, sc in per_task.items():
        per_type.setdefault(type_of[name], []).append(sc)
    type_means = [sum(v) / len(v) for v in per_type.values()]
    mean_type = sum(type_means) / len(type_means) if type_means else 0.0   # capability-balanced
    mean_task = sum(per_task.values()) / len(per_task) if per_task else 0.0  # leaderboard headline
    return mean_type, mean_task, per_task, skipped


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else str(MODEL_DIR)
    out = sys.argv[2] if len(sys.argv) > 2 else None

    mean_type, mean_task, ho_per, skipped = score_model(model)

    contam = None
    sample = Path(model).resolve().parent / "runs" / "_train_texts.json"
    if sample.exists():
        train = set(json.loads(sample.read_text()))
        evalset = _eval_texts()
        hits = sum(1 for t in train if t in evalset)
        contam = {"hits": hits, "frac": round(hits / max(len(train), 1), 6)}

    result = {"mean_type": mean_type, "mean_task": mean_task,
              "heldout_per_task": ho_per, "skipped": skipped, "contamination": contam}
    print(f"MEAN_TYPE={mean_type:.4f}  MEAN_TASK={mean_task:.4f}  "
          f"scored={len(ho_per)}  skipped={len(skipped)}  contam={contam}")
    if out:
        Path(out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
