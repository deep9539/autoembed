#!/usr/bin/env python3
# Render claude stream-json (stdin) as a clean timeline; raw stays in trace.log.
import sys, json, re

PREFIX = re.compile(r'^\[\s*[\d.]+s\]\s*')


def fmt(line):
    m = PREFIX.match(line)
    ts = line[:m.end()].strip() if m else ""
    body = (line[m.end():] if m else line).strip()
    if not body.startswith("{"):
        return line.rstrip("\n") if body else None      # pass through non-JSON (training logs, errors)
    try:
        d = json.loads(body)
    except Exception:
        return line.rstrip("\n")

    t = d.get("type")
    if t == "system" and d.get("subtype") == "init":
        return f"{ts} · session start · model={d.get('model')}"
    if t == "assistant":
        out = []
        for c in d.get("message", {}).get("content", []):
            if c.get("type") == "text" and c.get("text", "").strip():
                out.append(f"{ts} 💬 {c['text'].strip()[:400]}")
            elif c.get("type") == "tool_use":
                n, inp = c.get("name"), c.get("input", {})
                if n == "Bash":
                    out.append(f"{ts} ⚙  Bash: {inp.get('description', '')[:70]}")
                    out.append(f"      $ {inp.get('command', '')[:220]}")
                elif n in ("Read", "Edit", "Write"):
                    out.append(f"{ts} ⚙  {n}: {inp.get('file_path', '')}")
                else:
                    out.append(f"{ts} ⚙  {n}: {str(inp)[:140]}")
        return "\n".join(out) if out else None
    if t == "user":
        out = []
        for c in d.get("message", {}).get("content", []):
            if c.get("type") == "tool_result":
                v = c.get("content", "")
                if isinstance(v, list):
                    v = " ".join(x.get("text", "") for x in v if isinstance(x, dict))
                first = next((l for l in str(v).splitlines() if l.strip()), "")
                out.append(f"      {'✗' if c.get('is_error') else '✓'} {first[:160]}")
        return "\n".join(out) if out else None
    if t == "result":
        return f"{ts} == result (error={d.get('is_error')}): {str(d.get('result', ''))[:200]}"
    return None      # skip thinking_tokens, thinking blocks, usage noise


try:
    for line in sys.stdin:
        try:
            s = fmt(line)
        except Exception:
            s = line.rstrip("\n")
        if s:
            print(s, flush=True)
except BrokenPipeError:      # terminal went away
    pass
