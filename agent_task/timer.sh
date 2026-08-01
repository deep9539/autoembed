#!/bin/bash
# Print how much wall-clock budget is left. The agent calls this to pace itself.
# DEADLINE (epoch seconds) is set by run_task.sh.
now=$(date +%s)
left=$(( ${DEADLINE:-now} - now ))
if [ "$left" -le 0 ]; then echo "0m left — budget exhausted, finalize now."; else
  echo "$(( left / 60 ))m left"; fi
