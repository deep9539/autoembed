#!/usr/bin/env bash
# Cluster hosts omit /dev/log; use Enroot's restriction hook without that mount.
set -euo pipefail
patched="$(mktemp)"
trap 'rm -f "$patched"' EXIT
sed '\|/dev/log|d' /etc/enroot/hooks.d/10-devices.sh > "$patched"
source "$patched"
