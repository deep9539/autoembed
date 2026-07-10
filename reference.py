# Reference ladder on the held-out: raw base (floor), same-base anchor, strong small
# encoder. Written to runs/reference.md.
from score import HELDOUT_TASKS
from task import _score, BASE_MODEL, RUNS_DIR

LADDER = [
    (BASE_MODEL, "floor: raw base"),
    ("intfloat/e5-base-v2", "anchor: same base, fully fine-tuned"),
    ("BAAI/bge-base-en-v1.5", "strong small encoder"),
]


def main():
    lines = []
    for model, label in LADDER:
        try:
            r = _score(model, HELDOUT_TASKS, tag="ref")
            line = f"- type={r['mean_type']:.4f} task={r['mean_task']:.4f}  {model}  ({label})"
        except Exception as e:
            line = f"- FAIL  {model}: {repr(e)[:120]}"
        print(line)
        lines.append(line)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "reference.md").write_text(
        "# Reference ladder — held-out: MTEB(eng, v2) − MindSmall, Mean over task types\n\n"
        + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
