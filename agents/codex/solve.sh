#!/bin/bash
# Run the official Codex CLI and resume only after an early exit.
set -uo pipefail

unset ANTHROPIC_API_KEY CLAUDE_CODE_OAUTH_TOKEN GEMINI_API_KEY GOOGLE_API_KEY
MODEL="${AGENT_CONFIG:-gpt-5.6-sol}"
REASONING="${AUTOEMBED_AGENT_REASONING:-high}"
[ "${AGENT_AUTH_MODE:-}" = subscription ] && unset OPENAI_API_KEY
MIN_REMAINING_MINUTES=30
INITIAL_PROMPT="Read $PWD/instructions.md completely, then carry out the task autonomously. Run required training and evaluation processes to completion before exiting, and ensure the best valid submission is saved in final_model/."

run_codex() {
    codex --search exec --json --skip-git-repo-check --yolo \
        --model "$MODEL" -c "model_reasoning_effort=\"$REASONING\"" "$@"
}

started="$(date +%s)"
if ! run_codex "$INITIAL_PROMPT"; then
    if [ "$(( $(date +%s) - started ))" -lt 300 ]; then
        echo "!! Codex failed within 5 minutes; treating as setup/auth failure" >&2
        exit 1
    fi
    echo "!! Codex exited with an error after doing work; attempting a resume" >&2
fi

while :; do
    left="$(bash timer.sh 2>/dev/null)"
    mins="$(printf '%s\n' "$left" | grep -oE '^[0-9]+' || true)"
    { [ -z "$mins" ] || [ "$mins" -lt "$MIN_REMAINING_MINUTES" ]; } && break

    continuation="You have $left. Continue improving the result and maximize development-benchmark performance. Run required training and evaluation processes to completion before exiting, and ensure the best valid submission is saved in final_model/."
    if ! codex --search exec resume --last --json --skip-git-repo-check --yolo \
        --model "$MODEL" -c "model_reasoning_effort=\"$REASONING\"" \
        "$continuation"; then
        echo "!! Codex resume failed; retrying after a short backoff" >&2
        sleep 120
    fi
done
