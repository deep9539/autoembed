#!/usr/bin/env bash
# Environment-specific launcher (NOT part of the framework) — submits a command
# to a single-GPU node on our Slurm cluster. Adapt or replace for your setup.
#
#   scripts/gpu.sh uv run python train.py        # one experiment on a GPU
#   scripts/gpu.sh nvidia-smi                    # sanity check
#   TIME=12:00:00 AUTOEMBED_CONFIG=configs/specialization/legal.json HOURS=10 scripts/gpu.sh scripts/run_task.sh claude
#
# Tunables via env: PART (partition), TIME (wall limit), NODE (nodelist), EXCLUDE (nodes to avoid).
set -euo pipefail
PART="${PART:-shared}"   # guest is preemptible; a preempted long run loses its budget
TIME="${TIME:-01:00:00}"  # default wall limit
srun_args=(--partition="$PART" --gres=gpu:1 --time="$TIME"
           ${NODE:+--nodelist="$NODE"} ${EXCLUDE:+--exclude="$EXCLUDE"}
           --job-name=autoembed --export=ALL,REQUIRE_GPU_ENFORCEMENT=1)

if [ -t 1 ]; then
  # --pty is interactive only; it breaks GPU init when backgrounded. It also puts
  # this terminal in raw mode, and srun restores it only on a clean exit — a
  # scancel or preemption otherwise leaves the shell with no echo. Run srun as a
  # child rather than exec so the saved termios can always be put back.
  tty_state="$(stty -g 2>/dev/null || true)"
  restore_tty() { [ -n "$tty_state" ] && stty "$tty_state" 2>/dev/null || true; }
  trap 'restore_tty' EXIT INT TERM HUP
  srun "${srun_args[@]}" --pty "$@"
  exit $?
fi
exec srun "${srun_args[@]}" "$@"
