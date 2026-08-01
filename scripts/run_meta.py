"""Build per-run provenance, usage, cost, quota, and score metadata."""
import json
import os
import re
import sys
from pathlib import Path


DEFAULT_AGENT_MODELS = {
    "claude": "claude-opus-5",
    "codex": "gpt-5.6-sol",
    "gemini": "gemini-3.6-flash",
}
API_RATES = {
    # USD per million tokens; update when model pricing changes.
    "gpt-5.6-sol": {
        "input": 5.0, "cached_input": 0.5,
        "cache_write_input": 6.25, "output": 30.0,
    },
    "gpt-5.6": {
        "input": 5.0, "cached_input": 0.5,
        "cache_write_input": 6.25, "output": 30.0,
    },
    # Gemini bills cache creation at the standard input rate and folds thinking
    # tokens into the output rate.
    "gemini-3.6-flash": {
        "input": 1.5, "cached_input": 0.15,
        "cache_write_input": 1.5, "output": 7.5,
    },
}
PRICING_CHECKED_AT = {
    "gpt-5.6-sol": "2026-07-28",
    "gpt-5.6": "2026-07-28",
    "gemini-3.6-flash": "2026-07-31",
}
MESSAGE_INPUT_KEYS = (
    "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
)
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
    message_input = {}
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
                usage = message.get("usage") or {}
                output = usage.get("output_tokens", 0) or 0
                message_output[message_id] = max(message_output.get(message_id, 0), output)
                # Streaming repeats a message id with growing counters; keep the
                # high-water mark per id so a session killed before its result
                # event still yields input and cache totals.
                seen = message_input.setdefault(message_id, {})
                for key in MESSAGE_INPUT_KEYS:
                    value = usage.get(key) or 0
                    if isinstance(value, (int, float)):
                        seen[key] = max(seen.get(key, 0), value)
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
        partial = {
            key: sum(seen.get(key, 0) for seen in message_input.values())
            for key in MESSAGE_INPUT_KEYS
        }
        return {
            "source": "claude-stream-partial",
            "turns": None,
            "tool_calls": claude_tools,
            "input_tokens": partial["input_tokens"] or None,
            "cache_read_tokens": partial["cache_read_input_tokens"] or None,
            "cache_creation_tokens": partial["cache_creation_input_tokens"] or None,
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


def trace_summary(path):
    """Return provider-neutral trace identity and terminal-event facts."""
    facts = {
        "event_count": 0,
        "init_events": 0,
        "terminal_events": 0,
        "session_ids": set(),
        "reported_models": set(),
    }
    try:
        handle = Path(path).open(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return {**facts, "session_ids": [], "reported_models": []}
    with handle:
        for raw in handle:
            event = _event(raw)
            if not event:
                continue
            facts["event_count"] += 1
            event_type = event.get("type")
            subtype = event.get("subtype")
            if event_type in {"init", "thread.started"} or (
                event_type == "system" and subtype == "init"
            ):
                facts["init_events"] += 1
            if event_type in {"result", "turn.completed"}:
                facts["terminal_events"] += 1
            session = event.get("session_id") or event.get("thread_id")
            if session:
                facts["session_ids"].add(session)
            model = event.get("model")
            message = event.get("message")
            if not model and isinstance(message, dict):
                model = message.get("model")
            if model:
                facts["reported_models"].add(model)
    facts["session_ids"] = sorted(facts["session_ids"])
    facts["reported_models"] = sorted(facts["reported_models"])
    return facts


def estimate_api_cost(model, usage):
    rates = API_RATES.get(model)
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


def _normalized_usage(usage):
    """Normalize one provider aggregate into the shared token schema."""
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens", "promptTokenCount", "prompt"),
        "output_tokens": ("output_tokens", "completion_tokens", "candidatesTokenCount", "response", "candidates"),
        "cache_read_tokens": ("cache_read_input_tokens", "cached_input_tokens", "cachedContentTokenCount", "cached"),
        "cache_creation_tokens": ("cache_creation_input_tokens", "cache_write_input_tokens"),
        "reasoning_output_tokens": ("reasoning_output_tokens", "thoughtsTokenCount", "thoughts"),
    }
    models = usage.get("models") if isinstance(usage, dict) else None
    if isinstance(models, dict):
        # Counters sit under a `tokens` map in some CLI versions and directly on
        # the model entry in others.
        candidates = [
            details.get("tokens") or details
            for details in models.values()
            if isinstance(details, dict)
        ]
    elif isinstance(usage, dict):
        candidates = [usage]
    else:
        candidates = []

    totals = dict.fromkeys(aliases, 0)
    tool_tokens = 0
    for candidate in candidates:
        for field, names in aliases.items():
            for name in names:
                value = candidate.get(name)
                if isinstance(value, (int, float)):
                    totals[field] += value
                    break
        tool = candidate.get("tool") or candidate.get("tool_tokens")
        if isinstance(tool, (int, float)):
            tool_tokens += tool
    if isinstance(models, dict):
        totals["output_tokens"] += totals["reasoning_output_tokens"] + tool_tokens
    return totals


def usage_from_sidecar(path, agent):
    """Aggregate terminal events from the append-only usage sidecar.

    Claude and Gemini result records are per invocation and are summed. Codex
    turn usage is cumulative, so only the latest record per thread is retained.
    Assistant-message records are excluded because result events include them.
    """
    records = []
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except (TypeError, ValueError):
                    continue
    except (FileNotFoundError, OSError):
        return None

    terminal = [
        record for record in records
        if record.get("type") in {"result", "turn.completed"}
    ]
    if not terminal:
        return None

    source = f"{agent}-result-sidecar"
    if agent == "codex":
        latest = {}
        for record in terminal:
            if record.get("type") == "turn.completed":
                latest[record.get("session_id") or "default"] = record
        terminal = list(latest.values())
        source = "codex-turn-cumulative-sidecar"
    else:
        terminal = [record for record in terminal if record.get("type") == "result"]
    if not terminal:
        return None

    fields = (
        "input_tokens", "output_tokens", "cache_read_tokens",
        "cache_creation_tokens", "reasoning_output_tokens",
    )
    totals = dict.fromkeys(fields, 0)
    cost = 0.0
    turns = 0
    tool_calls = 0
    seen_cost = seen_tools = False
    for record in terminal:
        usage = record.get("usage") or {}
        normalized = _normalized_usage(usage)
        for field, value in normalized.items():
            totals[field] += value
        if isinstance(record.get("cost_usd"), (int, float)):
            cost += record["cost_usd"]
            seen_cost = True
        if isinstance(record.get("num_turns"), int):
            turns += record["num_turns"]
        if isinstance(usage.get("tool_calls"), int):
            tool_calls += usage["tool_calls"]
            seen_tools = True
    uncached = max(
        totals["input_tokens"]
        - totals["cache_read_tokens"]
        - totals["cache_creation_tokens"],
        0,
    )
    return {
        "source": source,
        "records": len(terminal),
        "turns": turns or None,
        "tool_calls": tool_calls if seen_tools else None,
        "input_tokens": totals["input_tokens"],
        "uncached_input_tokens": uncached,
        "cache_read_tokens": totals["cache_read_tokens"],
        "cache_creation_tokens": totals["cache_creation_tokens"],
        "output_tokens": totals["output_tokens"],
        "reasoning_output_tokens": totals["reasoning_output_tokens"] or None,
        "provider_reported_cost_usd": round(cost, 6) if seen_cost else None,
    }


def served_models_from_sessions(directory):
    """Return the models the CLI recorded actually serving each message.

    A CLI may substitute a model it does not recognise, so the requested id is
    not evidence. Claude carries it on message.model, Gemini on the record.
    """
    served = set()
    for path in sorted(Path(directory).glob("*.jsonl")) if Path(directory).is_dir() else ():
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                try:
                    record = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if not isinstance(record, dict):
                    continue
                name = record.get("model")
                message = record.get("message")
                if not isinstance(name, str) and isinstance(message, dict):
                    name = message.get("model")
                if isinstance(name, str) and name:
                    served.add(name)
    return sorted(served)


def effort_from_sessions(directory):
    """Return the reasoning effort the CLI recorded, or None if unavailable."""
    efforts = set()
    for path in sorted(Path(directory).glob("*.jsonl")) if Path(directory).is_dir() else ():
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                try:
                    value = json.loads(raw).get("effort")
                except (TypeError, ValueError):
                    continue
                if isinstance(value, str) and value:
                    efforts.add(value)
    return ",".join(sorted(efforts)) if efforts else None


def env(key, default=""):
    return os.environ.get(key, default)


def build_meta(trace, score_path=None):
    agent = env("AGENT")
    # Prefer the streamed sidecar; fall back to the trace for older runs.
    sidecar = usage_from_sidecar(Path(trace).with_name("usage.jsonl"), agent)
    parsed = parse_trace(trace)
    trace_facts = trace_summary(trace)
    usage = dict(parsed)
    if sidecar:
        usage.update({k: v for k, v in sidecar.items() if v not in (None, 0) or k == "source"})
        # The trace counts tool calls per event; fall back to the CLI's own total.
        usage["tool_calls"] = parsed.get("tool_calls") or sidecar.get("tool_calls")
        usage["trace_source"] = parsed.get("source")
    model = env("AGENT_CONFIG") or DEFAULT_AGENT_MODELS.get(agent)
    # Claude reports its own cost; the others are priced from token counters.
    estimate = estimate_api_cost(model, usage) if agent != "claude" else None
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
    elif usage.get("uncached_input_tokens") is not None:
        uncached = usage["uncached_input_tokens"]
    elif agent == "codex":
        uncached = max(
            input_tokens - (usage.get("cache_read_tokens") or 0)
            - (usage.get("cache_creation_tokens") or 0),
            0,
        )
    else:
        uncached = input_tokens
    source = usage.get("source") or ""
    terminal_complete = (
        trace_facts["init_events"] > 0
        and trace_facts["terminal_events"] >= trace_facts["init_events"]
    )
    if source.endswith("sidecar") or source in {"codex-turn-cumulative", "claude-result"}:
        measurement_status = "complete" if terminal_complete else "partial"
    elif source == "claude-stream-partial":
        measurement_status = "partial"
    else:
        measurement_status = "unavailable"
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
        "pricing_rates_usd_per_million": API_RATES.get(model),
        "pricing_checked_at": PRICING_CHECKED_AT.get(model) if estimate is not None else None,
        "cost_note": (
            "API-equivalent estimate from trace token counters; request-level context-length multipliers and cache storage time are not recoverable from cumulative usage."
            if estimate is not None else
            "Provider-reported API-equivalent session cost; subscription runs are not billed this amount."
            if provider_cost is not None else
            "Token usage recorded, but provider USD cost is unavailable from this CLI stream."
            if input_tokens is not None else
            "Usage and cost unavailable from this trace; null values are not zero usage."
        ),
    })
    # Codex takes an effort argument; Claude Code exposes no
    # equivalent, so a Claude run is the CLI default whatever was requested.
    # The CLI records the effort it actually ran at; prefer that over the request.
    observed = effort_from_sessions(Path(trace).with_name("agent_sessions"))
    reasoning = observed or (
        "default" if env("AGENT") == "claude" else (env("AGENT_REASONING") or None)
    )
    reported_models = trace_facts["reported_models"]
    served_models = served_models_from_sessions(Path(trace).with_name("agent_sessions"))
    # The trace echoes the requested id (or an alias); only the CLI's own session
    # record shows what served the request.
    model_verified = (model in served_models) if served_models else None
    trace_facts["terminal_event_complete"] = terminal_complete

    return {
        "run_id": env("RUN_ID"), "run_kind": env("RUN_KIND") or "experiment", "agent": env("AGENT"),
        "harness_complete": env("HARNESS_COMPLETE", "true") == "true",
        "agent_config": env("AGENT_CONFIG"), "agent_model": model,
        "model_identity": {
            "requested": model,
            "reported": reported_models,
            "served": served_models,
            "verified": model_verified,
        },
        "trace": trace_facts,
        "base_model": env("BASE_MODEL"),
        "base_revision": env("BASE_REVISION") or None,
        "agent_version": env("AGENT_VERSION"),
        "agent_protocol": env("AGENT_PROTOCOL") or None,
        "reprompt_cutoff_minutes": (
            int(env("REPROMPT_CUTOFF_MINUTES"))
            if env("REPROMPT_CUTOFF_MINUTES") else None
        ),
        "agent_wrapper_sha256": env("AGENT_WRAPPER_SHA256") or None,
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
