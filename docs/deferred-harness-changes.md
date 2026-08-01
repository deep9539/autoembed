# Deferred edits to scripts/run_task.sh

Cannot be applied while any job is executing the script: bash keeps a byte offset
into the file and re-reads from it after each top-level command, so an insertion
makes a running job resume mid-token and die with a syntax error. Verified by
reproducer. Apply only when `squeue -u $USER | grep autoembed` is empty.

## 1. Make the script immune to this class of failure (do this first)

Wrap the entire body in a brace group so bash must parse to the closing brace
before executing anything:

    #!/usr/bin/env bash
    { # ---- whole body, unchanged ----
    ...
    }

## 2. Detach stdin from the terminal

    : > "$RESULTS/trace.log"
    : > "$RESULTS/trace.raw.log"
    : > "$RESULTS/usage.jsonl"
    # Detach stdin from the terminal: the sandbox runs in a background process group,
    # so a CLI that reads stdin takes SIGTTIN and stops for the rest of the run.
    # trace.raw.log is the CLI stream byte-for-byte; trace.log adds elapsed prefixes.
    sandbox "timeout --signal=TERM --kill-after=30s $LIMIT bash solve.sh" </dev/null 2>&1 \
      | tee -a "$RESULTS/trace.raw.log" \
      | python3 "$ROOT/scripts/timestamp_lines.py" \
      | tee -a "$RESULTS/trace.log" \
      | python3 "$ROOT/scripts/usage_tap.py" "$RESULTS/usage.jsonl" \
      | { python3 "$ROOT/scripts/pretty_stream.py" 2>/dev/null || true; cat >/dev/null; }

Note: agents/gemini/solve.sh already carries its own `</dev/null` and is copied to
$WORK at launch, so Gemini is already protected without this change.

## 3. Retain the CLI's own session transcript

Insert immediately before the `# workspace snapshot:` line:

    # The CLI's own session transcript is the richest record of the run: full message
    # history and per-message usage, beyond what stdout carries. Copy only the
    # transcripts; the surrounding config directories hold credentials.
    mkdir -p "$RESULTS/agent_sessions"
    ( cd "$WORK" && find .claude-agent/projects .home/.gemini .codex/sessions \
        -name '*.jsonl' -size -200M 2>/dev/null -exec cp --parents -t "$RESULTS/agent_sessions" {} + ) || true
    rmdir "$RESULTS/agent_sessions" 2>/dev/null || true

## 4. Bound the workspace snapshot

Current exclusions miss datasets: 20 archived snapshots total 589 GB, one is 142 GB
(134 GB of it a single `data/` directory). Add size/extension bounds before the tar,
or drop `data/` and `.cache/` outright.

## 5. Bump the agent-protocol identifier

`AGENT_PROTOCOL="official-cli-reprompt-v1"` -> `"official-cli-reprompt-v2"`.
The Gemini wrapper gained a poisoned-session fallback (fresh session after two
consecutive resume failures, fast failures counted double). The wrapper sha256
already distinguishes runs, but the protocol string is what the paper cites.

## Test coverage already written (tests/test_agent_wrappers.py)

- test_agent_stdin_is_never_the_terminal   -> asserts the `</dev/null` on the sandbox line
- test_the_unmodified_cli_stream_is_retained -> asserts raw tee precedes timestamp_lines

Both currently FAIL against the reverted run_task.sh and must be re-enabled with the
edits above. They are the reason to apply 2 and 3 together.
