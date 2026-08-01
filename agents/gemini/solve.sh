#!/bin/bash
# Run the official Gemini CLI and resume only after an early exit.
set -uo pipefail

unset ANTHROPIC_API_KEY CLAUDE_CODE_OAUTH_TOKEN OPENAI_API_KEY
MODEL="${AGENT_CONFIG:-gemini-3.6-flash}"
export GEMINI_CLI_TRUST_WORKSPACE=true
MIN_REMAINING_MINUTES=30
INITIAL_PROMPT="Read $PWD/instructions.md completely, then carry out the task autonomously. Run required training and evaluation processes to completion before exiting, and ensure the best valid submission is saved in final_model/."

# Prefer one unambiguous API-key variable when both aliases are present.
[ -n "${GEMINI_API_KEY:-}" ] && unset GOOGLE_API_KEY

# Headless mode cannot display the authentication chooser, so API-key
# authentication is pre-selected, as Harbor's Gemini adapter does.
#
# The model is requested through a custom alias rather than --model directly.
# The CLI's own resolver rewrites any id containing "flash" that it does not
# recognise to its default flash model, so `--model gemini-3.6-flash` silently
# runs gemini-3.5-flash. An alias resolves through modelConfigService instead,
# which returns before that rewrite; it requires the experimental flag below.
MODEL_ALIAS="autoembed-$MODEL"
mkdir -p "$HOME/.gemini"
printf '{"security":{"auth":{"selectedType":"gemini-api-key"}},%s%s%s\n' \
    '"experimental":{"dynamicModelConfiguration":true},' \
    "\"modelConfigs\":{\"customAliases\":{\"$MODEL_ALIAS\":{\"modelConfig\":{\"model\":\"$MODEL\"," \
    '"generateContentConfig":{"thinkingConfig":{"includeThoughts":true,"thinkingLevel":"HIGH"}}}}}}}' \
    > "$HOME/.gemini/settings.json"

# stdin must not be the terminal: the sandbox runs in a background process group,
# so a terminal read raises SIGTTIN and stops the CLI.
run_gemini() {
    gemini --yolo --skip-trust --model "$MODEL_ALIAS" --output-format stream-json "$@" </dev/null
}

started="$(date +%s)"
if ! run_gemini -p "$INITIAL_PROMPT"; then
    if [ "$(( $(date +%s) - started ))" -lt 300 ]; then
        echo "!! Gemini failed within 5 minutes; treating as setup/auth failure" >&2
        exit 1
    fi
    echo "!! Gemini exited with an error after doing work; attempting a resume" >&2
fi

resume_failures=0
while :; do
    left="$(bash timer.sh 2>/dev/null)"
    mins="$(printf '%s\n' "$left" | grep -oE '^[0-9]+' || true)"
    { [ -z "$mins" ] || [ "$mins" -lt "$MIN_REMAINING_MINUTES" ]; } && break

    continuation="You have $left. Continue improving the result and maximize development-benchmark performance. Run required training and evaluation processes to completion before exiting, and ensure the best valid submission is saved in final_model/."
    attempt="$(date +%s)"
    if [ "$resume_failures" -ge 2 ]; then
        # The saved session is rejected on replay (an interruption mid tool call
        # leaves history the API refuses, e.g. a missing thought_signature).
        # Start a clean session; the workspace and final_model/ carry over.
        echo "!! Resume is stuck; starting a fresh session" >&2
        run_gemini -p "$continuation" && resume_failures=0 || sleep 120
    elif run_gemini --resume latest -p "$continuation"; then
        resume_failures=0
    else
        resume_failures=$((resume_failures + 1))
        # Failing before any real work means the history itself was rejected,
        # not an outage; count it double so the fallback engages next pass.
        [ "$(( $(date +%s) - attempt ))" -lt 30 ] && resume_failures=$((resume_failures + 1))
        echo "!! Gemini resume failed; retrying after a short backoff" >&2
        sleep 120
    fi
done
