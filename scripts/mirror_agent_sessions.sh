#!/usr/bin/env bash
# Keep each active run's CLI session transcript mirrored into its results dir.
#
# The transcripts live on node-local /tmp inside $WORK and are excluded from the
# end-of-run archival, so they are destroyed when the job exits. This refreshes a
# copy on shared storage until the last run finishes. Stopgap: once no job is
# executing run_task.sh, the copy step moves into the harness itself.
set -uo pipefail
ROOT=/data/home/niklas/adnan/autoembed
STAGE="$ROOT/artifacts/session-rescue"
INTERVAL="${INTERVAL:-900}"

cd "$ROOT" || exit 1

while :; do
  jobs_running=$(squeue -u "$USER" -h -t RUNNING -o "%j" | grep -c autoembed)
  [ "$jobs_running" -eq 0 ] && { echo "$(date +%H:%M:%S) no autoembed runs left; stopping"; break; }

  rm -rf "$STAGE"; mkdir -p "$STAGE"
  for j in $(squeue -u "$USER" -h -t RUNNING -o "%N %i %j" | awk '$3=="autoembed"' \
             | sort -u -k1,1 | awk '{print $2}'); do
    timeout 150 srun --overlap --jobid="$j" bash -c "
      for w in /tmp/tmp.*; do
        # Claude writes its transcript under CLAUDE_CONFIG_DIR; Gemini keeps an
        # equivalent per-message chat log, both carrying per-message token counts.
        for p in \"\$w/.claude-agent/projects/-work\" \"\$w/.home/.gemini/tmp/work/chats\"; do
          [ -d \"\$p\" ] || continue
          d=\"$STAGE/\$(basename \$w)\"; mkdir -p \"\$d\"
          cp -f \"\$p\"/*.jsonl \"\$d/\" 2>/dev/null
        done
      done
    " >/dev/null 2>&1
  done

  python3 - <<'PY'
import glob, json, os, shutil
owner = {}
for d in sorted(glob.glob("results/*_claude/")) + sorted(glob.glob("results/*_gemini/")):
    t = os.path.join(d, "trace.log")
    if not os.path.exists(t): continue
    for line in open(t, errors="replace"):
        if '"session_id"' not in line: continue
        try: o = json.loads(line.split("] ", 1)[-1].strip())
        except Exception: continue
        s = o.get("session_id")
        if s: owner[s] = d
placed = 0
for wd in glob.glob("artifacts/session-rescue/*/"):
    # claude files are <session>.jsonl; gemini are session-<date>-<prefix>.jsonl
    ids = []
    for f in glob.glob(os.path.join(wd, "*.jsonl")):
        b = os.path.basename(f)[:-6]
        ids.append(b)
        if b.startswith("session-"):
            ids.append(b.rsplit("-", 1)[-1])

    hit = next((owner[i] for i in ids if i in owner), None)
    if hit is None:
        hit = next((d for i in ids for sid, d in owner.items() if sid.startswith(i)), None)
    if not hit: continue
    dest = os.path.join(hit, "agent_sessions"); os.makedirs(dest, exist_ok=True)
    for f in glob.glob(os.path.join(wd, "*.jsonl")):
        shutil.copy2(f, dest)
    placed += 1
print(f"  mirrored {placed} run(s)")
PY

  echo "$(date +%H:%M:%S) refreshed ($jobs_running run(s) active)"
  sleep "$INTERVAL"
done
