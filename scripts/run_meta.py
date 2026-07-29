"""Build per-run provenance, usage, cost, quota, and score metadata."""
import json
import os
import re
import sys
from pathlib import Path


DEFAULT_AGENT_MODELS = {
    "claude": "claude-opus-5",
    "codex": "gpt-5.6-sol",
    "antigravity": "gemini-3.6-flash",
}
CODEX_API_RATES = {
    # USD per million tokens; update when model pricing changes.
    "gpt-5.6-sol": {
        "input": 5.0, "cached_input": 0.5,
        "cache_write_input": 6.25, "output": 30.0,
    },
    "gpt-5.6": {
        "input": 5.0, "cached_input": 0.5,
        "cache_write_input": 6.25, "output": 30.0,
    },
}
TOOL_ITEM_TYPES = {
    "command_execution", "mcp_tool_call", "web_search", "file_change",
    "computer_use", "dynamic_tool_call",
}


def _event(raw):
    line = raw.split("] ", 1)[-1].strip()
    if not line.startswith("{"):
        return None
    try:
        return json.loads(line)
    except (TypeError, json.JSONDecodeError):
        return None


def parse_trace(path):
    message_output = {}
    claude_usage = {}
    claude_turns = claude_tools = 0
    provider_cost = 0.0
    codex_thread = "unknown"
    codex_usage = {}
    codex_turns = codex_tools = 0
    quota_messages = []
    try:
        lines = Path(path).open(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        lines = None
    for raw in lines or ():
        event = _event(raw)
        if not event:
            continue
        event_type = event.get("type")
        if event_type == "thread.started":
            codex_thread = event.get("thread_id") or codex_thread
        elif event_type == "turn.completed":
            codex_turns += 1
            # Codex reports cumulative thread usage; repeated resume calls use
            # the same thread id, so retain the latest counters, never sum them.
            codex_usage[codex_thread] = event.get("usage") or {}
        elif event_type == "item.completed":
            if (event.get("item") or {}).get("type") in TOOL_ITEM_TYPES:
                codex_tools += 1
        elif event_type == "error":
            message = str(event.get("message") or "")
            if re.search(r"usage limit|rate limit|quota", message, re.I):
                quota_messages.append(message)
        elif event_type == "assistant":
            message = event.get("message", {})
            message_id = message.get("id")
            if message_id is not None:
                output = (message.get("usage") or {}).get("output_tokens", 0) or 0
                message_output[message_id] = max(message_output.get(message_id, 0), output)
            claude_tools += sum(
                item.get("type") == "tool_use"
                for item in message.get("content", [])
            )
        elif event_type == "result":
            claude_turns += event.get("num_turns", 0) or 0
            provider_cost += event.get("total_cost_usd", 0) or 0
            for key, value in (event.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    claude_usage[key] = claude_usage.get(key, 0) + value
    if lines is not None:
        lines.close()

    if codex_usage:
        usage = {}
        for thread in codex_usage.values():
            for key, value in thread.items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
        return {
            "source": "codex-turn-cumulative",
            "turns": codex_turns,
            "tool_calls": codex_tools,
            "input_tokens": usage.get("input_tokens", 0),
            "cache_read_tokens": usage.get("cached_input_tokens", 0),
            "cache_creation_tokens": usage.get("cache_write_input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "reasoning_output_tokens": usage.get("reasoning_output_tokens", 0),
            "provider_reported_cost_usd": None,
            "quota_messages": quota_messages,
        }
    if claude_usage:
        return {
            "source": "claude-result",
            "turns": claude_turns,
            "tool_calls": claude_tools,
            "input_tokens": claude_usage.get("input_tokens", 0),
            "cache_read_tokens": claude_usage.get("cache_read_input_tokens", 0),
            "cache_creation_tokens": claude_usage.get("cache_creation_input_tokens", 0),
            "output_tokens": claude_usage.get("output_tokens") or sum(message_output.values()),
            "reasoning_output_tokens": None,
            "provider_reported_cost_usd": round(provider_cost, 6),
            "quota_messages": quota_messages,
        }
    partial_output = sum(message_output.values())
    if partial_output or claude_tools:
        return {
            "source": "claude-stream-partial",
            "turns": None,
            "tool_calls": claude_tools,
            "input_tokens": None,
            "cache_read_tokens": None,
            "cache_creation_tokens": None,
            "output_tokens": partial_output or None,
            "reasoning_output_tokens": None,
            "provider_reported_cost_usd": None,
            "quota_messages": quota_messages,
        }
    return {
        "source": "unknown",
        "turns": None,
        "tool_calls": None,
        "input_tokens": None,
        "cache_read_tokens": None,
        "cache_creation_tokens": None,
        "output_tokens": None,
        "reasoning_output_tokens": None,
        "provider_reported_cost_usd": None,
        "quota_messages": quota_messages,
    }


def estimate_codex_cost(model, usage):
    rates = CODEX_API_RATES.get(model)
    total_input = usage.get("input_tokens")
    output = usage.get("output_tokens")
    if not rates or total_input is None or output is None:
        return None
    cached = usage.get("cache_read_tokens") or 0
    cache_write = usage.get("cache_creation_tokens") or 0
    uncached = max(total_input - cached - cache_write, 0)
    return round((
        uncached * rates["input"] + cached * rates["cached_input"]
        + cache_write * rates["cache_write_input"]
        + output * rates["output"]
    ) / 1_000_000, 6)


def score_summary(path):
    if not path:
        return None
    try:
        score = json.loads(Path(path).read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    protocol_valid = score.get("protocol_valid")
    skipped = score.get("skipped") or []
    reasons = score.get("invalid_reasons") or []
    reportability = score.get("reportability")
    warnings = score.get("warnings") or []
    if protocol_valid is None:
        reasons = [*reasons, "legacy score lacks protocol_valid; reportability is unknown"]
    return {
        "score_schema_version": score.get("score_schema_version"),
        "contamination_policy": score.get("contamination_policy"),
        "reportability": reportability,
        "warnings": warnings,
        "protocol_valid": protocol_valid,
        "reportable": protocol_valid is True and not skipped,
        "invalid_reasons": reasons,
        "substituted_base_model": score.get("substituted_base_model"),
        "mean_type": score.get("mean_type"),
        "mean_task": score.get("mean_task"),
        "tasks_scored": len(score.get("heldout_per_task") or score.get("per_task") or {}),
        "skipped": skipped,
    }


def env(key, default=""):
    return os.environ.get(key, default)


def build_meta(trace, score_path=None):
    usage = parse_trace(trace)
    agent = env("AGENT")
    model = env("AGENT_CONFIG") or DEFAULT_AGENT_MODELS.get(agent)
    estimate = (
        estimate_codex_cost(model, usage)
        if usage.get("source") == "codex-turn-cumulative" else None
    )
    auth_mode = env("AGENT_AUTH_MODE")
    provider_cost = usage.pop("provider_reported_cost_usd")
    quota_messages = usage.pop("quota_messages")
    reset_times = []
    for message in quota_messages:
        match = re.search(r"try again at ([^.]+(?:\.)?)", message, re.I)
        reset = match.group(1).rstrip(".") if match else None
        if reset and reset not in reset_times:
            reset_times.append(reset)
    duration = int(env("DURATION", "0"))
    budget_hours = int(env("HOURS", "0"))
    budget_fraction = round(duration / (budget_hours * 3600), 6) if budget_hours else None
    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        uncached = None
    elif usage.get("source") == "codex-turn-cumulative":
        uncached = max(
            input_tokens - (usage.get("cache_read_tokens") or 0)
            - (usage.get("cache_creation_tokens") or 0),
            0,
        )
    else:
        uncached = input_tokens
    measurement_status = (
        "complete" if usage.get("source") in ("codex-turn-cumulative", "claude-result")
        else "partial" if usage.get("source") == "claude-stream-partial" else "unavailable"
    )
    usage.update({
        "measurement_status": measurement_status,
        "uncached_input_tokens": uncached,
        "provider_reported_cost_usd": provider_cost,
        "api_equivalent_basic_rate_usd": estimate,
        "actual_billed_usd": (
            provider_cost if auth_mode == "api-key" and provider_cost is not None
            else None
        ),
        "api_key_estimated_cost_usd": estimate if auth_mode == "api-key" else None,
        "billing_basis": (
            "subscription-included; fixed plan price and optional usage credits are not attributable from the trace"
            if auth_mode == "subscription" else "provider API billing" if auth_mode == "api-key" else "unknown"
        ),
        "cost_is_estimate": estimate is not None,
        "pricing_rates_usd_per_million": CODEX_API_RATES.get(model),
        "pricing_checked_at": "2026-07-28" if estimate is not None else None,
        "cost_note": (
            "API-equivalent estimate from trace token counters; request-level >272K context multipliers are not recoverable from cumulative usage."
            if estimate is not None else
            "Provider-reported API-equivalent session cost; subscription runs are not billed this amount."
            if provider_cost is not None else
            "Usage and cost unavailable from this trace; null values are not zero usage."
        ),
    })
    # Codex and Antigravity take an effort argument; Claude Code exposes no
    # equivalent, so a Claude run is the CLI default whatever was requested.
    reasoning = "default" if env("AGENT") == "claude" else (env("AGENT_REASONING") or None)

    return {
        "run_id": env("RUN_ID"), "run_kind": env("RUN_KIND") or "experiment", "agent": env("AGENT"),
        "harness_complete": env("HARNESS_COMPLETE", "true") == "true",
        "agent_config": env("AGENT_CONFIG"), "agent_model": model,
        "base_model": env("BASE_MODEL"),
        "base_revision": env("BASE_REVISION") or None,
        "agent_version": env("AGENT_VERSION"),
        "agent_reasoning": reasoning,
        "agent_auth_mode": auth_mode,
        "protocol_version": env("PROTOCOL_VERSION"),
        "config": env("CONFIG_ID"), "config_sha256": env("CONFIG_SHA256"),
        "harness_git_commit": env("HARNESS_GIT_COMMIT") or None,
        "harness_git_dirty": (
            env("HARNESS_GIT_DIRTY").lower() == "true" if env("HARNESS_GIT_DIRTY") else None
        ),
        "scorer_sha256": env("SCORER_SHA256") or None,
        "container_image_sha256": env("CONTAINER_IMAGE_SHA256"),
        "score_exit": int(env("SCORE_RC", "0")),
        "score": score_summary(score_path),
        "budget_hours": budget_hours, "mode": env("MODE"),
        "duration_s": duration, "budget_fraction": budget_fraction,
        "near_budget": budget_fraction is not None and budget_fraction >= 0.95,
        "agent_exit": int(env("RC", "0")),
        "budget_hit": env("BUDGET_HIT") == "true",
        "quota": {
            "hit": bool(quota_messages), "rejected_attempts": len(quota_messages),
            "reset_at": reset_times,
        },
        "node": env("NODE"), "gpu": env("GPU_NAME"),
        "gpu_boundary": env("GPU_BOUNDARY"), "gpu_selector": env("GPU_SELECTOR"),
        "final_model_files": int(env("FINAL_FILES", "0")),
        "usage": usage,
    }


def main():
    trace, output = sys.argv[1], sys.argv[2]
    score_path = sys.argv[3] if len(sys.argv) > 3 else None
    Path(output).write_text(json.dumps(build_meta(trace, score_path), indent=2))


if __name__ == "__main__":
    main()
