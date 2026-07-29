#!/usr/bin/env bash
# Build an immutable Enroot image for cluster runs without a Docker daemon.
# Usage: scripts/build_enroot.sh [claude|codex|antigravity|none]
set -euo pipefail

AGENT_CLI="${1:-claude}"
case "$AGENT_CLI" in
  claude) NPM_PACKAGE="@anthropic-ai/claude-code" ;;
  codex) NPM_PACKAGE="@openai/codex" ;;
  antigravity) NPM_PACKAGE="" ;;
  none) NPM_PACKAGE="" ;;
  *) echo "unknown agent CLI: $AGENT_CLI" >&2; exit 2 ;;
esac

command -v enroot >/dev/null 2>&1 || {
  echo "enroot is required; use the Dockerfile on Docker hosts" >&2
  exit 1
}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARTIFACTS="$ROOT/artifacts"
OUT="${ENROOT_IMAGE:-$ARTIFACTS/autoembed-$AGENT_CLI.sqsh}"
BASE_URI="${ENROOT_BASE_URI:-docker://nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04}"
RUNTIME_NAME="${ENROOT_CONTAINER:-autoembed-$AGENT_CLI}"
NAME="autoembed-build-${USER//[^a-zA-Z0-9]/}-$$"
BASE_IMAGE="$ARTIFACTS/.base-$$.sqsh"
STAGING="$OUT.partial"
mkdir -p "$ARTIFACTS"

# The cluster-wide default points at a root-owned path on compute nodes.
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-$ARTIFACTS/enroot-data}"
export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-${TMPDIR:-/tmp}/enroot-$UID-${SLURM_JOB_ID:-local}}"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH"

cleanup() {
  enroot remove -f "$NAME" >/dev/null 2>&1 || true
  rm -f "$BASE_IMAGE" "$STAGING"
}
trap cleanup EXIT

echo ">> importing $BASE_URI"
enroot import -o "$BASE_IMAGE" "$BASE_URI"
echo ">> creating temporary rootfs $NAME"
enroot create -n "$NAME" "$BASE_IMAGE"

echo ">> installing locked Python environment and $AGENT_CLI CLI"
enroot start --root --rw \
  -e NPM_PACKAGE="$NPM_PACKAGE" \
  -e AGENT_CLI="$AGENT_CLI" \
  -m "$ROOT/pyproject.toml:/build/pyproject.toml" \
  -m "$ROOT/uv.lock:/build/uv.lock" \
  "$NAME" bash -lc '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends python3 python3-pip curl git ca-certificates
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y --no-install-recommends nodejs
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
    mkdir -p /opt/autoembed
    cp /build/pyproject.toml /build/uv.lock /opt/autoembed/
    cd /opt/autoembed
    uv sync --no-dev --frozen
    if [ -n "$NPM_PACKAGE" ]; then
      npm install -g "$NPM_PACKAGE"
    fi
    if [ "$AGENT_CLI" = antigravity ]; then
      mkdir -p /tmp/agy-install
      curl -fsSL https://antigravity.google/cli/install.sh \
        | HOME=/tmp/agy-install bash -s -- --dir /usr/local/bin
    fi
    printf "%s\n" "$AGENT_CLI" > /opt/autoembed/agent-cli
    apt-get clean
    rm -rf /var/lib/apt/lists/* /root/.cache /root/.npm
  '

mkdir -p "$(dirname "$OUT")"
enroot export -o "$STAGING" "$NAME"
mv "$STAGING" "$OUT"
enroot remove -f "$RUNTIME_NAME" >/dev/null 2>&1 || true
enroot create -n "$RUNTIME_NAME" "$OUT"
sha256sum "$OUT" > "$OUT.sha256"
echo ">> built $OUT"
echo ">> extracted read-only runtime: $ENROOT_DATA_PATH/$RUNTIME_NAME"
echo ">> checksum: $(cat "$OUT.sha256")"
