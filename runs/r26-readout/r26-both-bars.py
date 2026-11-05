"""Report only: for each round-26 arm, the temperatures on a 41-point grid over its pooled 90 % interval at which gated breadth ECE
<= 0.0176 and tasksource-heldout ECE <= 0.0523 (round 24's bars) hold together. Not a criterion; the served T is the pool fit."""
import json
import numpy as np
from kev import rounds
from kev.metrics import metrics
from kev.suite import write_json
spec = rounds.load("experiments/rounds/r26.json"); r = json.load(open("runs/r26-readout/round26.json"))
bars = {"breadth": 0.0176, "tasksource_heldout": 0.0523}
out = {}
for arm in spec["arms"]:
    x = r["arms"][arm]; lo, hi = x["temperature_ci"]["lower"], x["temperature_ci"]["upper"]; side = rounds.arm_side(spec, arm)
    raws = {}
    for p in bars:
        panel = spec["rule"]["panels"][p]; keep = rounds.panel_filter(panel)
        raws[p] = ([row for tag in panel["reads"] for row in rounds.read_json(side.rows_path(tag))], keep)
    ece = lambda p, t: metrics([q for q in rounds.served_at(raws[p][0], t) if q["id"] not in side.drop and raws[p][1](q)])["ece"]
    grid = [float(t) for t in np.linspace(lo, hi, 41)]
    rows = [(round(t, 4), round(ece("breadth", t), 4), round(ece("tasksource_heldout", t), 4)) for t in grid]
    both = [t for t, b, s in rows if b <= bars["breadth"] and s <= bars["tasksource_heldout"]]
    out[arm] = {"interval": [lo, hi], "both_pass_T": both, "closest": min(rows, key=lambda z: max(z[1] - bars["breadth"], z[2] - bars["tasksource_heldout"]))}
    print(arm, len(both), out[arm]["closest"])
write_json("runs/r26-readout/both-bars-by-t.json", {"report_only": __doc__, "bars": bars, "arms": out})
