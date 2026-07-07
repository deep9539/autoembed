# Reference ladder on the held-out: raw base (floor), same-base anchor, strong small
# encoder. Written to runs/reference.md. SOTA (e.g. Qwen3-Embedding) is a cited line.
from score import score_model
from task import BASE_MODEL, RUNS_DIR

LADDER = [
    (BASE_MODEL, "floor: raw base"),
    ("intfloat/e5-base-v2", "anchor: same base, fully fine-tuned"),
    ("BAAI/bge-base-en-v1.5", "strong small encoder"),
]


def main():
    lines = []
    for model, label in LADDER:
        try:
            mt, mtask, _, _ = score_model(model, tag="ref")
            line = f"- type={mt:.4f} task={mtask:.4f}  {model}  ({label})"
        except Exception as e:
            line = f"- FAIL  {model}: {repr(e)[:120]}"
        print(line)
        lines.append(line)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "reference.md").write_text(
        "# Reference ladder — held-out: full MTEB(eng, v2), Mean over task types\n\n"
        + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
