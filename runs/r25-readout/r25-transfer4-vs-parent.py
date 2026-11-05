"""Report only: each arm vs Kev-27B on transfer-v4 development alone (the locked read's suite; emotion excluded as in the short panel)."""
import json
from kev import rounds
from kev.suite import write_json
spec = rounds.load("experiments/rounds/r25.json")
rule = {"panels": {"transfer4": {"reads": ["transfer"], "exclude_sources": ["emotion"], "metrics": ["acc", "brier", "confident_error_rate", "ece"]}}, "criteria": {}}
out = {}
for arm in spec["arms"]:
    r = rounds.compare(rounds.arm_side(spec, arm), rounds.parent_side(spec, arm), rule)
    out[arm] = r["panels"]["transfer4"]
    p = out[arm]
    print(arm, p["n"], " ".join(f"{m} {p[m]['candidate']:.4f}/{p[m]['parent']:.4f} {p[m]['delta']*(100 if m in ('acc','confident_error_rate') else 1):+.3f} [{p[m]['ci95'][0]*(100 if m in ('acc','confident_error_rate') else 1):+.3f}, {p[m]['ci95'][1]*(100 if m in ('acc','confident_error_rate') else 1):+.3f}]" for m in ("acc", "brier", "confident_error_rate")))
write_json("runs/r25-readout/transfer4-vs-parent.json", {"report_only": __doc__, "arms": out})
