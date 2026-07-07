# Per-run metadata: run config + agent usage scraped from the trace. Writes meta.json.
import json
import os
import sys

trace, out = sys.argv[1], sys.argv[2]

msg_out, tool_calls, turns, cost, result_usage = {}, 0, 0, None, {}
try:
    for raw in open(trace, encoding="utf-8", errors="replace"):
        line = raw.split("] ", 1)[-1].strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        t = ev.get("type")
        if t == "assistant":
            m = ev.get("message", {})
            mid = m.get("id")
            if mid is not None:   # streaming repeats a message id; keep its max
                msg_out[mid] = max(msg_out.get(mid, 0), (m.get("usage") or {}).get("output_tokens", 0) or 0)
            for c in m.get("content", []):
                if c.get("type") == "tool_use":
                    tool_calls += 1
        elif t == "result":
            turns = ev.get("num_turns", turns)
            cost = ev.get("total_cost_usd", cost)
            result_usage = ev.get("usage", {}) or result_usage
except FileNotFoundError:
    pass


def env(k, d=""):
    return os.environ.get(k, d)


meta = {
    "run_id": env("RUN_ID"), "agent": env("AGENT"),
    "agent_config": env("AGENT_CONFIG"), "base_model": env("BASE_MODEL"),
    "budget_hours": int(env("HOURS", "0")), "mode": env("MODE"),
    "duration_s": int(env("DURATION", "0")), "agent_exit": int(env("RC", "0")),
    "budget_hit": env("BUDGET_HIT") == "true",
    "node": env("NODE"), "gpu": env("GPU_NAME"),
    "final_model_files": int(env("FINAL_FILES", "0")),
    "usage": {
        "turns": turns, "tool_calls": tool_calls,
        "output_tokens": sum(msg_out.values()),
        "input_tokens": result_usage.get("input_tokens"),
        "cache_read_tokens": result_usage.get("cache_read_input_tokens"),
        "cache_creation_tokens": result_usage.get("cache_creation_input_tokens"),
        "total_cost_usd": cost,
    },
}
json.dump(meta, open(out, "w"), indent=2)
