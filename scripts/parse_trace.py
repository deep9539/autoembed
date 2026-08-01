# Best-effort human-readable trace from the agent's JSON-stream output.
# Generic across scaffolds; falls back to a note if nothing parses.
import json
import sys


def walk(obj, out):
    if isinstance(obj, dict):
        if isinstance(obj.get("text"), str):
            out.append(obj["text"])
        if obj.get("type") == "tool_use" or obj.get("tool"):
            name = obj.get("name") or obj.get("tool")
            args = obj.get("input") or obj.get("arguments") or {}
            out.append(f"[tool: {name}] {json.dumps(args)[:500]}")
        for v in obj.values():
            walk(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, out)


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    chunks = []
    for raw in open(inp, encoding="utf-8", errors="replace"):
        line = raw.split("] ", 1)[-1].strip()  # drop the timestamp prefix
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        buf = []
        walk(ev, buf)
        if buf:
            chunks.append("\n".join(buf))
    text = "\n\n".join(chunks).strip()
    open(outp, "w").write(text or "(no parseable events — see trace.log)\n")


if __name__ == "__main__":
    main()
