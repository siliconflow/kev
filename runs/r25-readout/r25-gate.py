"""Spend gate for round 25 (docs/autoresearch.md s2 + PLAN Round 25 Budget): spend since the 23:45Z baseline + the bounds of
every call still running (a study: $494 while its trial runs; a read job: $25.06 until its report.json is on the volume or it failed)."""
import json, glob, subprocess, sys, datetime, re
import modal
new = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
m = float(json.loads(subprocess.run(["modal", "billing", "summary", "--json"], capture_output=True, text=True).stdout)["metered_cost"])
vol = modal.Volume.from_name("kev-runs")
names = set()
for f in glob.glob("runs/r25-reads-*.log"):
    txt = open(f).read()
    head = txt  # jobs listed in the intent json instead
for f in glob.glob("runs/r25-reads-*.json"):
    for cmd in json.load(open(f))["commands"]:
        if "--jobs" in cmd: names |= {j.split("@")[2] for j in cmd[cmd.index("--jobs") + 1].split(",")}
extra = [l.strip() for l in open("runs/r25-extra-jobs.txt")] if __import__("os").path.exists("runs/r25-extra-jobs.txt") else []
names |= set(n for n in extra if n)
failed = set()
for f in glob.glob("runs/r25-reads-*.log"):
    found = set(re.findall(r"^(\S+): FAILED", open(f).read(), re.M))
    if "retry" not in f: found -= set(extra)   # a retried name is judged by its retry's log
    failed |= found
pending, import_os = [], __import__("os")
done_path = "runs/r25-gate-done.json"
done = set(json.load(open(done_path))) if import_os.path.exists(done_path) else set()
for n in sorted(names):
    if n in done or n in failed: continue
    files = set()
    for _ in range(3):
        try: files = {e.path.rsplit("/", 1)[-1] for e in vol.listdir(f"/bench/{n}")}; break
        except Exception: pass
    if "report.json" in files: done.add(n)
    else: pending.append(n)
json.dump(sorted(done), open(done_path, "w"))
calls = sum(1 for s in ("lr1e6", "lr2e6") if any(c.get("status") == "running" for c in json.load(open(f"runs/r25-27b-{s}.watch.json"))["calls"].values()))
bounds = 494 * calls + 25.06 * len(pending)
g = m - 3966.94 + bounds + new
print(f"{datetime.datetime.now(datetime.UTC):%H:%MZ} metered ${m:.2f}; studies running {calls}; read jobs pending {len(pending)}: {pending}")
print(f"  gate {m-3966.94:.2f} + {bounds:.2f} + {new:.2f} = {g:.2f} {'<=' if g <= 1813 else '>'} 1813; metered+bounds+new {m+bounds+new:.2f} (hard stop 5600 on metered)")
