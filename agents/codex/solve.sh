#!/bin/bash
# Launch Codex CLI; re-prompt on early exit until the budget is spent.
unset ANTHROPIC_API_KEY
MODEL="${AGENT_CONFIG:-gpt-5.6-sol}"
REASONING="${AUTOEMBED_AGENT_REASONING:-high}"
[ "${AGENT_AUTH_MODE:-}" = subscription ] && unset OPENAI_API_KEY

# Point the agent at the staged file: a multi-line $PROMPT is truncated at its first
# newline crossing the container boundary, so the env var cannot carry the task.
TASK="Read $PWD/instructions.md completely, then carry out the task it specifies. Begin now and continue autonomously until the time budget expires."
STATUS_FILE="/tmp/autoembed-codex-status.$$"
trap 'rm -f -- "$STATUS_FILE"' EXIT

run_codex() {
    codex "$@" | tee "$STATUS_FILE"
    local command_rc=${PIPESTATUS[0]}
    if grep -qiE '"type"[[:space:]]*:[[:space:]]*"error".*(usage limit|quota)' "$STATUS_FILE"; then
        echo "!! Codex usage quota exhausted; stopping retries" >&2
        return 75
    fi
    return "$command_rc"
}

run_codex --search exec --json --skip-git-repo-check --yolo \
    --model "$MODEL" -c "model_reasoning_effort=\"$REASONING\"" "$TASK"
agent_rc=$?
[ "$agent_rc" -eq 75 ] && exit 75
[ "$agent_rc" -ne 0 ] && exit "$agent_rc"
while :; do
    left="$(bash timer.sh 2>/dev/null)"
    printf '%s' "$left" | grep -qiE 'exhaust|expired' && break
    mins="$(printf '%s' "$left" | grep -oE '[0-9]+' | head -1)"
    { [ -z "$mins" ] || [ "$mins" -lt 5 ]; } && break
    sleep 60   # let background work progress
    # varied wording + timestamp: an unvarying reprompt invites the model to pattern-complete a fake user turn
    case $((RANDOM % 3)) in
        0) msg="You still have $left. Keep improving final_model/ (your work is in this directory): check any background jobs, run evaluate() on the dev suite, and continue training until the model is as good as you can make it." ;;
        1) msg="Time remaining: $left. Continue improving final_model/ in this directory — check background jobs, run evaluate() on the dev suite, and keep iterating." ;;
        2) msg="Harness check-in, $left left. Resume work on final_model/ in this directory: verify background jobs and the dev-suite score, then continue training." ;;
    esac
    msg="$(printf '[harness %s] %s' "$(date +%H:%M:%S)" "$msg")"
    run_codex --search exec resume --last --json --skip-git-repo-check --yolo \
        --model "$MODEL" -c "model_reasoning_effort=\"$REASONING\"" "$msg"
    agent_rc=$?
    [ "$agent_rc" -eq 75 ] && exit 75
    [ "$agent_rc" -ne 0 ] && exit "$agent_rc"
done
