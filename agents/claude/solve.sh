#!/bin/bash
# Run the official Claude Code CLI and resume only after an early exit.
set -uo pipefail

unset OPENAI_API_KEY GEMINI_API_KEY GOOGLE_API_KEY
if [ "${AGENT_AUTH_MODE:-}" = subscription ]; then
    unset ANTHROPIC_API_KEY
else
    unset CLAUDE_CODE_OAUTH_TOKEN
fi

export CLAUDE_CONFIG_DIR="$PWD/.claude-agent"
export BASH_MAX_TIMEOUT_MS=36000000
MODEL="${AGENT_CONFIG:-claude-opus-5}"
MIN_REMAINING_MINUTES=30
INITIAL_PROMPT="Read $PWD/instructions.md completely, then carry out the task autonomously. Run required training and evaluation processes to completion before exiting, and ensure the best valid submission is saved in final_model/."

agent() {
    claude --print --verbose --model "$MODEL" \
        --disallowedTools "Monitor ScheduleWakeup" \
        --output-format stream-json --dangerously-skip-permissions "$@"
}

started="$(date +%s)"
if ! printf '%s' "$INITIAL_PROMPT" | agent; then
    if [ "$(( $(date +%s) - started ))" -lt 300 ]; then
        echo "!! Claude failed within 5 minutes; treating as setup/auth failure" >&2
        exit 1
    fi
    echo "!! Claude exited with an error after doing work; attempting a resume" >&2
fi

while :; do
    left="$(bash timer.sh 2>/dev/null)"
    mins="$(printf '%s\n' "$left" | grep -oE '^[0-9]+' || true)"
    { [ -z "$mins" ] || [ "$mins" -lt "$MIN_REMAINING_MINUTES" ]; } && break

    continuation="You have $left. Continue improving the result and maximize development-benchmark performance. Run required training and evaluation processes to completion before exiting, and ensure the best valid submission is saved in final_model/."
    if ! printf '%s' "$continuation" | agent --continue; then
        echo "!! Claude resume failed; retrying after a short backoff" >&2
        sleep 120
    fi
done
