#!/usr/bin/env bash
# Launch a wave of runs, one tmux session each.
#
# Every Gemini launch sources ~/.config/autoembed/gemini.env, so a session that
# still holds the rotated-out key cannot be used by accident. The launcher
# refuses to start if the key is missing or no longer authenticates.
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD"
KEYFILE="$HOME/.config/autoembed/gemini.env"

start() {   # start <session> <agent> <config> <extra-args...>
  local sess="$1" agent="$2" config="$3"; shift 3
  if tmux has-session -t "$sess" 2>/dev/null; then
    local pane; pane="$(tmux list-panes -t "$sess" -F '#{pane_current_command}' | head -1)"
    if [ "$pane" != "bash" ] && [ "$pane" != "zsh" ]; then
      echo "!! $sess is busy ($pane); skipping"; return
    fi
  else
    tmux new-session -d -s "$sess"
  fi
  local pre=""
  [ "$agent" = gemini ] && pre="set -a; . $KEYFILE; set +a; "
  tmux send-keys -t "$sess" \
    "cd $ROOT && ${pre}PART=guest uv run --no-sync autoembed run --config $config --agent $agent $* --hours 10 --time 11:00:00 2>&1 | tee runs/logs/$sess.log" Enter
  echo ">> $sess: $agent $config"
}

if [ ! -s "$KEYFILE" ]; then
  echo "!! $KEYFILE is missing; Gemini runs would fail on auth" >&2; exit 1
fi
key="$(sed -n 's/^GEMINI_API_KEY=//p' "$KEYFILE")"
code="$(curl -s -o /dev/null -w '%{http_code}' -H "x-goog-api-key: $key" \
  https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash)"
[ "$code" = 200 ] || { echo "!! key does not authenticate (HTTP $code)" >&2; exit 1; }
echo ">> key ok (HTTP 200)"

start gemini-nano-improve-s1 gemini general/mteb-nano-improve
start gemini-legal-s1        gemini specialization/legal
start gemini-finance-s1      gemini specialization/finance
start gemini-medical-s1      gemini specialization/medical
start gemini-code-s1         gemini specialization/code
start sonnet-code-s2         claude specialization/code --model claude-sonnet-5
