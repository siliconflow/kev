#!/bin/bash
# Round 26: the watcher is stopped so the final's read batch goes through the registered spend gate (the watcher launches a
# finished trial's reads without one). This polls the trial's call; on a timeout it restarts the watcher at once (it
# continues the trial within the ledger). On done it waits for the snapshot queue (s75 launched first, as PLAN's wall-clock
# plan orders it), then for the gate to pass one more whole batch, then restarts the watcher, which pulls the study, launches
# the final's reads once and writes the read-out when every read has landed.
cd /tmp/kev-r26run
export KEV_APP_NAME=kev-r26
restart_watch() { (nohup uv run python -m kev.rounds watch experiments/rounds/r26.json >> /tmp/r26-watch.log 2>&1 < /dev/null &); }
while true; do
  st=$(uv run python -c "
import json; from kev.rounds import poll_modal
c = json.load(open('runs/r26-27b-lr1e6.spawn.json'))['calls']['trial-0']
try: print(poll_modal(c))
except Exception as e: print('error', type(e).__name__, str(e)[:200])
" 2>/dev/null | tail -1)
  case "$st" in
    running) sleep 180 ;;
    done) echo "$(date -u +%H:%MZ) finish: trial call done" >> /tmp/r26-run.log; break ;;
    timeout|refused) echo "$(date -u +%H:%MZ) finish: trial call $st -> restarting the watcher now (continuation within the ledger)" >> /tmp/r26-run.log; restart_watch; exit 0 ;;
    *) echo "$(date -u +%H:%MZ) finish: poll said '$st'" >> /tmp/r26-run.log; sleep 180 ;;
  esac
done
until grep -q "auto: queue done" /tmp/r26-run.log; do sleep 120; done
held=""
while true; do
  out=$(uv run python runs/r26-gate.py 400.97 2>&1 | grep -v VIRTUAL)
  if echo "$out" | grep -q -- "-> PASS"; then
    echo "$(date -u +%H:%MZ) finish: gate for the final's batch: $out" >> /tmp/r26-run.log
    restart_watch
    echo "$(date -u +%H:%MZ) finish: watcher restarted (pull + final's reads + read-out)" >> /tmp/r26-run.log
    exit 0
  elif [ -z "$held" ]; then echo "$(date -u +%H:%MZ) finish: final's batch HOLDS: $out" >> /tmp/r26-run.log; held=1; fi
  sleep 180
done
