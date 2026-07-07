#!/bin/bash
# Launch Claude Code on the task. Reads $PROMPT, runs in the current workdir.
unset OPENAI_API_KEY GEMINI_API_KEY
export CLAUDE_CONFIG_DIR="$PWD/.claude-agent"
export BASH_MAX_TIMEOUT_MS=36000000   # Bash-tool timeout ceiling = 10h
printf '%s' "$PROMPT" | claude --print --verbose \
    --model "${AGENT_CONFIG:-claude-opus-4-8}" \
    --disallowedTools "Monitor ScheduleWakeup" \
    --output-format stream-json --dangerously-skip-permissions
