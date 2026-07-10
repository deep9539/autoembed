#!/bin/bash
# Launch Codex CLI; re-prompt on early exit until the budget is spent.
unset ANTHROPIC_API_KEY GEMINI_API_KEY
MODEL="${AGENT_CONFIG:-gpt-5.3-codex}"

printf '%s' "$PROMPT" | codex --search exec --json --skip-git-repo-check --yolo --model "$MODEL"
while :; do
    left="$(bash timer.sh 2>/dev/null)"
    printf '%s' "$left" | grep -qiE 'exhaust|expired' && break
    mins="$(printf '%s' "$left" | grep -oE '[0-9]+' | head -1)"
    { [ -z "$mins" ] || [ "$mins" -lt 5 ]; } && break
    sleep 60   # let background work progress
    msg="$(printf 'You still have %s. Keep improving final_model/ (your work is in this directory): check any background jobs, run evaluate_dev to see per-type scores, and continue training until the model is as good as you can make it.' "$left")"
    codex --search exec resume --last --json --skip-git-repo-check --yolo --model "$MODEL" "$msg"
done
