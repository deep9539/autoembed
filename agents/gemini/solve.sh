#!/bin/bash
# Launch Gemini CLI; re-prompt on early exit until the budget is spent.
# No non-interactive resume in Gemini, so each re-prompt is a fresh pass over the workdir.
unset ANTHROPIC_API_KEY OPENAI_API_KEY
export GEMINI_SANDBOX="false"
MODEL="${AGENT_CONFIG:-gemini-3-pro}"

gemini --yolo --model "$MODEL" --output-format stream-json -p "$PROMPT"
while :; do
    left="$(bash timer.sh 2>/dev/null)"
    printf '%s' "$left" | grep -qiE 'exhaust|expired' && break
    mins="$(printf '%s' "$left" | grep -oE '[0-9]+' | head -1)"
    { [ -z "$mins" ] || [ "$mins" -lt 5 ]; } && break
    sleep 60   # let background work progress
    msg="$(printf 'You still have %s. Keep improving final_model/ (your work is in this directory): run evaluate_dev to see per-type scores, and continue training until the model is as good as you can make it.' "$left")"
    gemini --yolo --model "$MODEL" --output-format stream-json -p "$msg"
done
