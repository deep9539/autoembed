#!/bin/bash
# Launch Claude Code; re-prompt on early exit until the budget is spent.
unset OPENAI_API_KEY
export CLAUDE_CONFIG_DIR="$PWD/.claude-agent"
export BASH_MAX_TIMEOUT_MS=36000000   # Bash-tool timeout ceiling = 10h
MODEL="${AGENT_CONFIG:-claude-opus-5}"

agent() {   # one headless pass; extra args (e.g. --continue) appended
    claude --print --verbose --model "$MODEL" \
        --disallowedTools "Monitor ScheduleWakeup" \
        --output-format stream-json --dangerously-skip-permissions "$@"
}

# The first real task request is also the authentication check, so every paid
# request remains in the captured stream and cost accounting.
if ! printf 'Read %s/instructions.md completely, then carry out the task it specifies. Begin now and continue autonomously until the time budget expires.' "$PWD" | agent; then
    echo "!! Claude exited with an error; stopping instead of retrying" >&2
    exit 1
fi
while :; do
    left="$(bash timer.sh 2>/dev/null)"
    printf '%s' "$left" | grep -qiE 'exhaust|expired' && break
    mins="$(printf '%s' "$left" | grep -oE '[0-9]+' | head -1)"
    { [ -z "$mins" ] || [ "$mins" -lt 5 ]; } && break
    sleep 60   # let background work progress
    # varied wording + timestamp: an unvarying reprompt invites the model to pattern-complete a fake user turn
    case $((RANDOM % 3)) in
        0) msg="You still have $left. Keep improving final_model/: check any background jobs, run evaluate() on the dev suite, and continue training/iterating until the model is as good as you can make it." ;;
        1) msg="Time remaining: $left. Continue improving final_model/ — check background jobs, run evaluate() on the dev suite, and keep iterating." ;;
        2) msg="Harness check-in, $left left. Resume work on final_model/: verify background jobs and the dev-suite score, then continue training/iterating." ;;
    esac
    if ! printf '[harness %s] %s' "$(date +%H:%M:%S)" "$msg" | agent --continue; then
        echo "!! Claude continuation failed; stopping retries" >&2
        exit 1
    fi
done
