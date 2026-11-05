"""Markdown table of runs/r26-readout/vs-<ref>-<arm>.json (report only). usage: r26-vs-table.py w85|r25"""
import json, os, sys
ref = sys.argv[1]
arms = [a for a in ["27b-lr1e6-s25", "27b-lr1e6-s50", "27b-lr1e6-s75", "27b-lr1e6"] if os.path.exists(f"runs/r26-readout/vs-{ref}-{a}.json")]
R = {a: json.load(open(f"runs/r26-readout/vs-{ref}-{a}.json")) for a in arms}
W = "27b-k-w85" if ref == "w85" else "r25 same-fraction arm"
def pp(e, s=100, d=1): return f"{e['delta']*s:+.{d}f} [{e['ci95'][0]*s:+.{d}f}, {e['ci95'][1]*s:+.{d}f}]"
rows = [("breadth gated acc", "breadth", "acc", 100, 1), ("breadth gated ECE Δ", "breadth", "ece", 1, 3), ("breadth all 14 acc", "breadth_index", "acc", 100, 1),
        ("tasksource-heldout gated acc", "tasksource_heldout", "acc", 100, 1), ("tsheld ECE Δ", "tasksource_heldout", "ece", 1, 3), ("tsheld all 24 acc", "tasksource_heldout_all_families", "acc", 100, 1),
        ("Kev panel acc", "kev", "acc", 100, 1), ("Kev ECE Δ", "kev", "ece", 1, 3), ("hard-v1 acc", "hard", "acc", 100, 1), ("Kev without hard acc", "kev_without_hard", "acc", 100, 1),
        ("short (transfer-v4 dev + r3 test) acc", "short", "acc", 100, 1), ("short Brier Δ", "short", "brier", 1, 3), ("short confident errors", "short", "confident_error_rate", 100, 1),
        ("transfer-v4 dev acc (locked-style)", "short_transfer4", "acc", 100, 1), ("transfer-v4 dev Brier Δ", "short_transfer4", "brier", 1, 3), ("transfer-v4 dev confident errors", "short_transfer4", "confident_error_rate", 100, 1),
        ("transfer-r3 test acc", "short_r3test", "acc", 100, 1), ("CUAD acc", "long", "acc", 100, 1), ("CUAD ECE Δ", "long", "ece", 1, 3),
        ("SemIf acc", "semif", "acc", 100, 1), ("WANLI-v2 acc", "wanli2", "acc", 100, 1), ("TypeSafe acc", "typesafe", "acc", 100, 1),
        ("ood-v2 acc", "ood", "acc", 100, 1), ("ood-v2 ECE Δ", "ood", "ece", 1, 3), ("agents-ood acc", "agents_ood", "acc", 100, 1), ("agents-ood ECE Δ", "agents_ood", "ece", 1, 3),
        ("guardrails-ood acc", "guardrails_ood", "acc", 100, 1), ("guardrails-ood ECE Δ", "guardrails_ood", "ece", 1, 3)]
print(f"| vs {W} (panel, n) | " + " | ".join(a.replace("27b-", "") for a in arms) + " |"); print("|---|" + "---|" * len(arms))
print("| T: arm / ref | " + " | ".join(f"{R[a]['temperature']:.3f} / {R[a]['reference_temperature']:.3f}" for a in arms) + " |")
for label, p, m, s, d in rows:
    n = R[arms[0]]["panels"][p]["n"]
    print(f"| {label} ({n}) | " + " | ".join(pp(R[a]["panels"][p][m], s, d) for a in arms) + " |")
for b in ("under_8k", "8k_16k", "16k_32k", "32k_64k", "16k_plus"):
    e = R[arms[0]]["panels"]["long"][f"acc_{b}"]
    print(f"| CUAD {b} acc / ECE, ref {e['parent']:.3f} / {R[arms[0]]['panels']['long'][f'ece_{b}']['parent']:.3f} ({e['n']}) | " + " | ".join(f"{R[a]['panels']['long'][f'acc_{b}']['candidate']:.3f} / {R[a]['panels']['long'][f'ece_{b}']['candidate']:.3f}" for a in arms) + " |")
