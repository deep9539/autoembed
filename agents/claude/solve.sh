#!/bin/bash
# Launch Claude Code; re-prompt on early exit until the budget is spent.
unset OPENAI_API_KEY GEMINI_API_KEY
export CLAUDE_CONFIG_DIR="$PWD/.claude-agent"
export BASH_MAX_TIMEOUT_MS=36000000   # Bash-tool timeout ceiling = 10h
MODEL="${AGENT_CONFIG:-claude-opus-4-8}"

agent() {   # one headless pass; extra args (e.g. --continue) appended
    claude --print --verbose --model "$MODEL" \
        --disallowedTools "Monitor ScheduleWakeup" \
        --output-format stream-json --dangerously-skip-permissions "$@"
}

printf '%s' "$PROMPT" | agent
while :; do
    left="$(bash timer.sh 2>/dev/null)"
    printf '%s' "$left" | grep -qiE 'exhaust|expired' && break
    mins="$(printf '%s' "$left" | grep -oE '[0-9]+' | head -1)"
    { [ -z "$mins" ] || [ "$mins" -lt 5 ]; } && break
    sleep 60   # let background work progress
    printf 'You still have %s. Keep improving final_model/: check any background jobs, run evaluate_dev to see per-type scores, and continue training/iterating until the model is as good as you can make it.' "$left" | agent --continue
done
