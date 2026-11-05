"""Report only: gated breadth-panel ECE (and tasksource-heldout ECE) of each round-25 arm at its pooled T, at the ends of the
pooled T's 90 % interval, and the T that would minimise breadth ECE (grid), against Kev-27B at its served T. Not a criterion."""
import json
from kev import rounds
from kev.metrics import metrics
from kev.suite import write_json
spec = rounds.load("experiments/rounds/r25.json")
r = json.load(open("runs/r25-readout/round25.json"))
out = {}
grid = [round(0.8 + 0.02 * i, 2) for i in range(61)]
for arm in spec["arms"]:
    side = rounds.arm_side(spec, arm)
    x = r["arms"][arm]; lo, hi = x["temperature_ci"]["lower"], x["temperature_ci"]["upper"]
    res = {}
    for pname in ("breadth", "tasksource_heldout"):
        panel = spec["rule"]["panels"][pname]; keep = rounds.panel_filter(panel)
        raw = [row for tag in panel["reads"] for row in rounds.read_json(side.rows_path(tag))]
        def ece_at(t):
            rows = [q for q in rounds.served_at(raw, t) if q["id"] not in side.drop and keep(q)]
            return metrics(rows)["ece"]
        at = {f"{t:.4f}": round(ece_at(t), 4) for t in (lo, x["temperature"], hi)}
        best = min(grid, key=ece_at) if pname == "breadth" else None
        res[pname] = {"at_lower_T_served_upper": at, **({"argmin_T": best, "min_ece": round(ece_at(best), 4)} if best else {})}
    out[arm] = res
    print(arm, json.dumps(res))
write_json("runs/r25-readout/breadth-ece-by-t.json", {"report_only": __doc__, "arms": out})
