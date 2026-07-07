#!/usr/bin/env bash
# Environment-specific launcher (NOT part of the framework) — submits a command
# to a single-GPU node on our Slurm cluster. Adapt or replace for your setup.
#
#   ./gpu.sh uv run python train.py        # one experiment on a GPU
#   ./gpu.sh nvidia-smi                     # sanity check
#
# Tunables via env: PART (partition), TIME (wall limit).
set -euo pipefail
PART="${PART:-guest}"
TIME="${TIME:-01:00:00}"  # default wall limit
PTY=""
[ -t 1 ] && PTY="--pty"  # interactive only; --pty breaks GPU init when backgrounded
exec srun --partition="$PART" --gres=gpu:1 --time="$TIME" \
     --job-name=autoembed $PTY "$@"
