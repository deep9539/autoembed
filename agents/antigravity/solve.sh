#!/bin/bash
# Launch Antigravity CLI headlessly; continue the same session until time expires.
unset ANTHROPIC_API_KEY OPENAI_API_KEY
MODEL="${AGENT_CONFIG:-gemini-3.6-flash}"
REASONING="${AUTOEMBED_AGENT_REASONING:-high}"
case "$REASONING" in
    low|medium|high) ;;
    *) echo "Antigravity reasoning must be low, medium, or high" >&2; exit 2 ;;
esac

run_agy() {
    remaining=$((DEADLINE - $(date +%s)))
    [ "$remaining" -gt 0 ] || return 0
    agy --dangerously-skip-permissions --add-dir "$PWD" --model "$MODEL" --effort "$REASONING" \
        --print-timeout "${remaining}s" "$@" </dev/null
}

save_trajectory() {
    src="$(find "$HOME/.agy/antigravity-cli/tmp" -type f \
        \( -name 'session-*.jsonl' -o -name 'session-*.json' \) \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
    [ -z "$src" ] || cp "$src" "antigravity-session.${src##*.}"
}

if ! run_agy --print "Read $PWD/instructions.md completely, then carry out the task it specifies. Begin now and continue autonomously until the time budget expires."; then
    echo "!! Antigravity exited with an error; stopping instead of retrying" >&2
    exit 1
fi
save_trajectory
while :; do
    left="$(bash timer.sh 2>/dev/null)"
    printf '%s' "$left" | grep -qiE 'exhaust|expired' && break
    mins="$(printf '%s' "$left" | grep -oE '[0-9]+' | head -1)"
    { [ -z "$mins" ] || [ "$mins" -lt 5 ]; } && break
    sleep 60
    # varied wording + timestamp: an unvarying reprompt invites the model to pattern-complete a fake user turn
    case $((RANDOM % 3)) in
        0) msg="You still have $left. Keep improving final_model/ (your work is in this directory): run evaluate() on the dev suite, and continue training until the model is as good as you can make it." ;;
        1) msg="Time remaining: $left. Continue improving final_model/ in this directory — run evaluate() on the dev suite and keep iterating." ;;
        2) msg="Harness check-in, $left left. Resume work on final_model/ in this directory: verify the dev-suite score, then continue training." ;;
    esac
    msg="$(printf '[harness %s] %s' "$(date +%H:%M:%S)" "$msg")"
    if ! run_agy --continue --print "$msg"; then
        echo "!! Antigravity continuation failed; stopping retries" >&2
        exit 1
    fi
    save_trajectory
done
