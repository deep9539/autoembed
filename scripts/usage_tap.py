"""Pass through a CLI JSON stream while recording usage-bearing events.

The sidecar is append-only and flushed after every record so usage already observed
survives timeout or preemption. Raw stdout continues unchanged to trace.log.
"""
import json
import sys
import time

COST_KEYS = ("total_cost_usd", "cost_usd", "cost")


def _usage_record(event, session_id=None):
    if not isinstance(event, dict):
        return None
    usage = event.get("usage")
    if not isinstance(usage, dict) and event.get("type") == "result":
        # Gemini CLI places aggregate usage under `stats`.
        usage = event.get("stats")
    cost = next(
        (event[key] for key in COST_KEYS if isinstance(event.get(key), (int, float))),
        None,
    )
    if not isinstance(usage, dict) and cost is None:
        return None
    return {
        "t": round(time.time(), 3),
        "type": event.get("type"),
        "session_id": event.get("session_id") or event.get("thread_id") or session_id,
        "num_turns": event.get("num_turns"),
        "is_error": event.get("is_error"),
        "cost_usd": cost,
        "usage": usage if isinstance(usage, dict) else None,
    }


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: usage_tap.py <usage.jsonl>")
    current_session = None
    sink = open(sys.argv[1], "a", encoding="utf-8")
    try:
        for line in sys.stdin:
            sys.stdout.write(line)
            sys.stdout.flush()
            payload = line.split("] ", 1)[-1].strip()
            if not payload.startswith("{"):
                continue
            try:
                event = json.loads(payload)
            except (TypeError, ValueError):
                continue
            if event.get("type") in {"init", "thread.started"}:
                current_session = event.get("session_id") or event.get("thread_id") or current_session
            record = _usage_record(event, current_session)
            if record is None:
                continue
            sink.write(json.dumps(record) + "\n")
            sink.flush()
    except BrokenPipeError:
        pass
    finally:
        sink.close()


if __name__ == "__main__":
    main()
