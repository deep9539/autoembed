#!/usr/bin/env bash
# Import subscription credentials created on a trusted machine.
set -euo pipefail

AGENT="${1:?usage: agent_auth.sh <codex|antigravity> <credential-file-or-directory>}"
SOURCE="${2:?usage: agent_auth.sh <codex|antigravity> <credential-file-or-directory>}"
case "$AGENT" in codex|antigravity) ;; *) echo "expected codex or antigravity" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUTH_DIR="${AUTOEMBED_AUTH_DIR:-$ROOT/.agent-auth}/$AGENT"
mkdir -p "$(dirname "$AUTH_DIR")" "$AUTH_DIR/home"
chmod 700 "$(dirname "$AUTH_DIR")" "$AUTH_DIR" "$AUTH_DIR/home"

if [ "$AGENT" = codex ]; then
  [ -f "$SOURCE" ] || { echo "Codex source must be your local ~/.codex/auth.json" >&2; exit 1; }
  install -m 600 "$SOURCE" "$AUTH_DIR/auth.json"
else
  if [ -d "$SOURCE" ]; then
    if [ -f "$SOURCE/antigravity-oauth-token" ]; then
      SOURCE="$SOURCE/antigravity-oauth-token"
    else
      SOURCE="$SOURCE/.gemini/antigravity-cli/antigravity-oauth-token"
    fi
  fi
  [ -f "$SOURCE" ] || {
    echo "Antigravity source must be an antigravity-oauth-token file or a directory containing it" >&2
    exit 1
  }
  python3 -c 'import json,sys; value=json.load(open(sys.argv[1])); assert isinstance(value,dict)' "$SOURCE" 2>/dev/null || {
    echo "Antigravity credential must be a valid JSON object" >&2
    exit 1
  }
  DEST="$AUTH_DIR/home/.gemini/antigravity-cli"
  mkdir -p "$DEST"
  chmod 700 "$AUTH_DIR/home/.gemini" "$DEST"
  install -m 600 "$SOURCE" "$DEST/antigravity-oauth-token"
fi

echo ">> imported $AGENT subscription credentials into private store $AUTH_DIR"
