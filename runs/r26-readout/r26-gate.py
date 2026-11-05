"""Spend gate for round 26 (PLAN "Round 26 (registered)" > Budget, docs/autoresearch.md s2), as registered:
(metered_now - 4,453.81 baseline) + admission bounds of everything still running + the new launch <= $1,373.
A study counts $494 while its trial runs. A read BATCH counts its whole bound ($400.97 for a 16-read arm batch; a retry
batch n x $25.06) until EVERY job of the batch has its report.json (locally in runs/<name>/ or on the volume at
/bench/<name>); a job that failed keeps its batch counted until a retry of it has landed (conservative). Hard stop $5,700 metered.
usage: python runs/r26-gate.py [new_bound]   -> prints; exit 0 = the launch passes, 1 = it does not."""
import datetime, glob, json, os, subprocess, sys
import modal

BASE, CAP, HARD = 4453.81, 1373.0, 5700.0
new = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
m = float(json.loads(subprocess.run(["modal", "billing", "summary", "--json"], capture_output=True, text=True).stdout)["metered_cost"])
vol = modal.Volume.from_name("kev-runs")
done_path = "runs/r26-gate-done.json"
done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()

def landed(n):
    if n in done: return True
    if os.path.exists(f"runs/{n}/report.json"): done.add(n); return True
    for _ in range(3):
        try:
            if "report.json" in {e.path.rsplit("/", 1)[-1] for e in vol.listdir(f"/bench/{n}")}: done.add(n); return True
            return False
        except FileNotFoundError: return False
        except Exception: pass
    return False

batches = []
for f in sorted(glob.glob("runs/r26-reads-*.json")):
    names = []
    for cmd in json.load(open(f))["commands"]:
        if "--jobs" in cmd: names += [j.split("@")[2] for j in cmd[cmd.index("--jobs") + 1].split(",")]
    bound = 400.97 if len(names) == 16 else round(25.06 * len(names), 2)
    pending = [n for n in names if not landed(n)]
    batches.append((os.path.basename(f), bound, pending))
json.dump(sorted(done), open(done_path, "w"))
from kev.rounds import poll_modal
study = False
for call in json.load(open("runs/r26-27b-lr1e6.spawn.json"))["calls"].values():   # the trial's current call, polled directly
    try: study = study or poll_modal(call) in ("running", "timeout")   # a timeout gets a continuation: keep its bound counted
    except Exception: study = True   # a failure or the network: count it (conservative)
running = [(b, bound, p) for b, bound, p in batches if p]
bounds = 494.0 * study + sum(bound for _, bound, _ in running)
g = m - BASE + bounds + new
ok = g <= CAP and m < HARD
print(f"{datetime.datetime.now(datetime.UTC):%H:%MZ} metered ${m:.2f}; study running {study}; batches running {len(running)}: "
      + "; ".join(f"{b} ${bound} pending {len(p)} {p if len(p) <= 4 else ''}" for b, bound, p in running))
print(f"  gate {m - BASE:.2f} + {bounds:.2f} + {new:.2f} = {g:.2f} {'<=' if g <= CAP else '>'} {CAP:.0f}; metered {'<' if m < HARD else '>='} hard stop {HARD:.0f} -> {'PASS' if ok else 'HOLD'}")
sys.exit(0 if ok else 1)
