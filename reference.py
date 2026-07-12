# Reference ladder on the held-out: the base floor + any model ids passed as args.
#   uv run python reference.py nomic-ai/modernbert-embed-base intfloat/e5-base-v2
# Prints per-task; writes runs/reference.md.
import sys

from score import CONFIG, HELDOUT_TASKS
from task import _score, RUNS_DIR


def main():
    models = [CONFIG["base_model"]] + sys.argv[1:]
    lines = []
    for model in models:
        try:
            r = _score(model, HELDOUT_TASKS, tag="ref", trust_remote_code=True)
            tasks = "  ".join(f"{n}={s:.3f}" for n, s in sorted(r["per_task"].items()))
            line = f"- mean_type={r['mean_type']:.4f}  {model}\n    {tasks}"
            if r["skipped"]:
                line += f"\n    skipped: {r['skipped']}"
        except Exception as e:
            line = f"- FAIL  {model}: {repr(e)[:120]}"
        print(line)
        lines.append(line)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "reference.md").write_text(
        "# Reference ladder — held-out (mean over task types)\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
