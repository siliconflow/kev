#!/bin/bash
# Launch the snapshot candidates' reads in order, each only once its snapshot is complete (snapshot.json) AND the registered
# spend gate passes for one more whole batch (runs/r26-gate.py 400.97, which reads the metered cost first).
cd /tmp/kev-r26run
export KEV_APP_NAME=kev-r26
for arm in "$@"; do
  ck=$(python3 -c "import json;print(json.load(open('experiments/rounds/r26.json'))['arms']['$arm']['checkpoint'])")
  snap=${ck#/runs}
  held=""
  while true; do
    if uv run modal volume ls kev-runs "$snap" 2>/dev/null | grep -q snapshot.json; then
      out=$(uv run python runs/r26-gate.py 400.97 2>&1 | grep -v VIRTUAL); rc=$?
      if echo "$out" | grep -q -- "-> PASS"; then
        echo "$(date -u +%H:%MZ) auto: snapshot $snap complete; $out" >> /tmp/r26-run.log
        uv run python -m kev.rounds launch-reads experiments/rounds/r26.json --arms $arm >> /tmp/r26-run.log 2>&1
        echo "$(date -u +%H:%MZ) auto: launched reads $arm" >> /tmp/r26-run.log
        sleep 120; break
      elif [ -z "$held" ]; then
        echo "$(date -u +%H:%MZ) auto: $arm snapshot complete but gate HOLDS: $out" >> /tmp/r26-run.log; held=1
      fi
    fi
    sleep 180
  done
done
echo "$(date -u +%H:%MZ) auto: queue done ($*)" >> /tmp/r26-run.log
