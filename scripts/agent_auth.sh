#!/usr/bin/env bash
# Import Codex subscription credentials created on a trusted machine.
set -euo pipefail

AGENT="${1:?usage: agent_auth.sh codex <auth.json>}"
SOURCE="${2:?usage: agent_auth.sh codex <auth.json>}"
[ "$AGENT" = codex ] || { echo "expected codex" >&2; exit 2; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUTH_DIR="${AUTOEMBED_AUTH_DIR:-$ROOT/.agent-auth}/$AGENT"
mkdir -p "$(dirname "$AUTH_DIR")" "$AUTH_DIR"
chmod 700 "$(dirname "$AUTH_DIR")" "$AUTH_DIR"

[ -f "$SOURCE" ] || { echo "Codex source must be your local ~/.codex/auth.json" >&2; exit 1; }
install -m 600 "$SOURCE" "$AUTH_DIR/auth.json"

echo ">> imported Codex subscription credentials into private store $AUTH_DIR"
