"""Report only: gated breadth-panel ECE and tasksource-heldout ECE of each round-26 arm at its pooled T, at the ends of the pooled
T's 90 % interval, and the T that would minimise breadth ECE (grid 0.80-2.00), plus the same for round 23's 27b-k-w85 at its own
pooled T (rows in /tmp/kev-r23run). Round 25's arms: runs/r25-readout/breadth-ece-by-t.json. Not a criterion."""
import importlib.util, json
from kev import rounds
from kev.metrics import metrics
from kev.suite import write_json

spec = rounds.load("experiments/rounds/r26.json")
r = json.load(open("runs/r26-readout/round26.json"))
vs = importlib.util.spec_from_file_location("vs", "runs/r26-readout/r26-vs-ref.py"); vs_mod = importlib.util.module_from_spec(vs); vs.loader.exec_module(vs_mod)
grid = [round(0.8 + 0.02 * i, 2) for i in range(61)]


def one(side, t, lo, hi):
    res = {}
    for pname in ("breadth", "tasksource_heldout"):
        panel = spec["rule"]["panels"][pname]; keep = rounds.panel_filter(panel)
        raw = [row for tag in panel["reads"] for row in rounds.read_json(side.rows_path(tag))]
        def ece_at(x):
            return metrics([q for q in rounds.served_at(raw, x) if q["id"] not in side.drop and keep(q)])["ece"]
        at = {f"{x:.4f}": round(ece_at(x), 4) for x in (lo, t, hi)}
        best = min(grid, key=ece_at)
        res[pname] = {"at_lower_T_served_upper": at, "argmin_T": best, "min_ece": round(ece_at(best), 4)}
    return res


out = {}
for arm in spec["arms"]:
    x = r["arms"][arm]
    if not x.get("complete", True) or "temperature_ci" not in x: continue
    out[arm] = one(rounds.arm_side(spec, arm), x["temperature"], x["temperature_ci"]["lower"], x["temperature_ci"]["upper"])
    print(arm, json.dumps(out[arm]))
w = vs_mod.w85_side(spec)
ci = rounds.pooled_temperature_ci(w.pool, w.root)
out["27b-k-w85 (round 23)"] = one(w, w.t, ci["lower"], ci["upper"])
print("w85", w.t, json.dumps(out["27b-k-w85 (round 23)"]))
write_json("runs/r26-readout/breadth-ece-by-t.json", {"report_only": __doc__, "arms": out})
