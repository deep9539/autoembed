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

# fail fast if the CLI isn't authenticated (headless auth = CLAUDE_CODE_OAUTH_TOKEN)
probe="$(printf 'Reply with exactly: ok' | agent 2>&1 | tail -c 2000)"
case "$probe" in *"Not logged in"*|*authentication_failed*)
    echo "!! claude auth failed — export CLAUDE_CODE_OAUTH_TOKEN before launching"; exit 1;;
esac

printf '%s' "$PROMPT" | agent
while :; do
    left="$(bash timer.sh 2>/dev/null)"
    printf '%s' "$left" | grep -qiE 'exhaust|expired' && break
    mins="$(printf '%s' "$left" | grep -oE '[0-9]+' | head -1)"
    { [ -z "$mins" ] || [ "$mins" -lt 5 ]; } && break
    sleep 60   # let background work progress
    printf 'You still have %s. Keep improving final_model/: check any background jobs, run evaluate() on the dev suite, and continue training/iterating until the model is as good as you can make it.' "$left" | agent --continue
done
