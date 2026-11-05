"""Markdown tables of runs/r26-readout/round26.json (criteria table, report-only panels, CUAD by length) in PLAN's format."""
import json, sys
from pathlib import Path

r = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/r26-readout/round26.json").read_text())
arms = list(r["arms"])
short = {a: a.replace("27b-", "") for a in arms}
A = r["arms"]


def pp(e, s=100, d=1):
    return f"{e['delta']*s:+.{d}f} [{e['ci95'][0]*s:+.{d}f}, {e['ci95'][1]*s:+.{d}f}]"


def get(a, panel, metric):
    return A[a].get("panels", {}).get(panel, {}).get(metric)


def row(label, fn):
    cells = []
    for a in arms:
        try: cells.append(fn(a))
        except Exception: cells.append("-")
    return f"| {label} | " + " | ".join(cells) + " |"


def mark(a, crit, text):
    v = A[a]["criteria"].get(crit)
    return text + (" **fail**" if v is False else "")


out = ["| criterion | " + " | ".join(short[a] for a in arms) + " |", "|---|" + "---|" * len(arms)]
out.append(row(f"T (pool, {A[arms[0]]['temperature_ci']['questions']}) [90 % CI]", lambda a: f"{A[a]['temperature']:.3f} [{A[a]['temperature_ci']['lower']:.3f}, {A[a]['temperature_ci']['upper']:.3f}]"))
n = lambda p: A[arms[0]]["panels"][p]["n"]
out.append(row(f"1 breadth acc, lower > 0 ({n('breadth')})", lambda a: mark(a, "1_breadth_lower_above_0", pp(get(a, "breadth", "acc")))))
out.append(row(f"1 tasksource-heldout acc, lower > 0 ({n('tasksource_heldout')})", lambda a: mark(a, "1_tasksource_heldout_lower_above_0", pp(get(a, "tasksource_heldout", "acc")))))
out.append(row(f"1 Kev panel acc, lower ≥ −1 ({n('kev')})", lambda a: mark(a, "1_kev_lower_at_least_minus_1pp", pp(get(a, "kev", "acc")))))
out.append(row(f"2 short acc, lower ≥ −2 ({n('short')})", lambda a: mark(a, "2_short_acc_lower_at_least_minus_2pp", pp(get(a, "short", "acc"), d=2))))
out.append(row("2 short Brier, upper ≤ +0.02", lambda a: mark(a, "2_short_brier_upper_at_most_0.02", pp(get(a, "short", "brier"), 1, 3))))
out.append(row("2 short confident errors, upper ≤ +1", lambda a: mark(a, "2_short_confident_errors_upper_at_most_1pp", pp(get(a, "short", "confident_error_rate")))))
out.append(row(f"2 CUAD acc, lower ≥ −2 ({n('long')})", lambda a: mark(a, "2_long_cuad_lower_at_least_minus_2pp", pp(get(a, "long", "acc")))))
out.append(row("2 CUAD 16k+ acc, cand − parent ≥ −2", lambda a: mark(a, "2_long_cuad_16k_plus_at_least_minus_2pp", f"{get(a, 'long', 'acc_16k_plus')['candidate']:.3f} vs {get(a, 'long', 'acc_16k_plus')['parent']:.3f}")))
out.append(row("2 unknowable share ≤ 0.05", lambda a: mark(a, "2_unknowable_at_most_0.05", f"{A[a]['unknowable']['candidate']:.3f}")))
pe = lambda p: A[arms[0]]["panels"][p]["ece"]["parent"]
out.append(row(f"3 breadth ECE ≤ {pe('breadth')+0.01:.4f} (Kev-27B {pe('breadth'):.4f})", lambda a: mark(a, "3_breadth_ece_at_most_parent_plus_0.01", f"{get(a, 'breadth', 'ece')['candidate']:.4f}")))
out.append(row(f"3 Kev-panel ECE ≤ {pe('kev')+0.01:.4f} (Kev-27B {pe('kev'):.4f})", lambda a: mark(a, "3_kev_ece_at_most_parent_plus_0.01", f"{get(a, 'kev', 'ece')['candidate']:.4f}")))
out.append(row(f"3 tasksource-heldout ECE ≤ {pe('tasksource_heldout')+0.01:.4f} (Kev-27B {pe('tasksource_heldout'):.4f})", lambda a: mark(a, "3_tasksource_heldout_ece_at_most_parent_plus_0.01", f"{get(a, 'tasksource_heldout', 'ece')['candidate']:.4f}")))
out.append(row("rank score (Δ breadth + Δ tsheld + Δ Kev, pp)", lambda a: f"{sum(get(a, p, 'acc')['delta'] for p in ('breadth', 'tasksource_heldout', 'kev'))*100:+.1f}"))
out.append(row("criteria passed", lambda a: f"{sum(1 for v in A[a]['criteria'].values() if v)}/{len(A[a]['criteria'])}"))
out.append(row("verdict", lambda a: "**PASS**" if A[a].get("passed") else "fail"))
print("\n".join(out))
print()
rep = ["| panel | " + " | ".join(short[a] for a in arms) + " |", "|---|" + "---|" * len(arms)]
for label, p, m, s, d in [("breadth, all 14 sources, acc", "breadth_index", "acc", 100, 1), ("breadth cfcolor+humicroedit+chessbench, acc", "breadth_moved_to_report", "acc", 100, 1),
                          ("tasksource-heldout all 24 families, acc", "tasksource_heldout_all_families", "acc", 100, 1), ("hard-v1 acc", "hard", "acc", 100, 1),
                          ("Kev panel without hard-v1, acc", "kev_without_hard", "acc", 100, 1), ("SemIf acc", "semif", "acc", 100, 1), ("WANLI-v2 acc", "wanli2", "acc", 100, 1),
                          ("TypeSafe acc", "typesafe", "acc", 100, 1), ("ood-v2 acc", "ood", "acc", 100, 1), ("agents-ood-v1 acc", "agents_ood", "acc", 100, 1),
                          ("guardrails-ood-v1 acc", "guardrails_ood", "acc", 100, 1), ("ood-v2 ECE Δ", "ood", "ece", 1, 3), ("agents-ood-v1 ECE Δ", "agents_ood", "ece", 1, 3),
                          ("guardrails-ood-v1 ECE Δ", "guardrails_ood", "ece", 1, 3), ("breadth ECE Δ (gated panel)", "breadth", "ece", 1, 3), ("Kev ECE Δ", "kev", "ece", 1, 3),
                          ("tsheld ECE Δ", "tasksource_heldout", "ece", 1, 3), ("CUAD ECE Δ", "long", "ece", 1, 3)]:
    nn = A[arms[0]].get("panels", {}).get(p, {}).get("n", "?")
    rep.append(row(f"{label} ({nn})", lambda a, p=p, m=m, s=s, d=d: pp(get(a, p, m), s, d)))
rep.append(row(f"CUAD ECE (Kev-27B {get(arms[0], 'long', 'ece')['parent']:.3f})", lambda a: f"{get(a, 'long', 'ece')['candidate']:.3f}"))
rep.append(row(f"CUAD ECE 16k+ (Kev-27B {get(arms[0], 'long', 'ece_16k_plus')['parent']:.3f})", lambda a: f"{get(a, 'long', 'ece_16k_plus')['candidate']:.3f}"))
rep.append(row("longdoc generated acc / ECE", lambda a: f"{get(a, 'long_generated', 'acc')['candidate']:.3f} / {get(a, 'long_generated', 'ece')['candidate']:.3f}"))
rep.append(row(f"short-state acc, candidate (Kev-27B {get(arms[0], 'short', 'acc')['parent']:.4f})", lambda a: f"{get(a, 'short', 'acc')['candidate']:.4f}"))
print("\n".join(rep))
print()
buckets = ["under_8k", "8k_16k", "16k_32k", "32k_64k", "16k_plus"]
cu = ["| CUAD by length, acc / ECE (n) | Kev-27B | " + " | ".join(short[a] for a in arms) + " |", "|---|---|" + "---|" * len(arms)]
for b in buckets:
    e0 = get(arms[0], "long", f"acc_{b}"); c0 = get(arms[0], "long", f"ece_{b}")
    cu.append(f"| {b} ({e0['n']}) | {e0['parent']:.3f} / {c0['parent']:.3f} | " + " | ".join(f"{get(a, 'long', f'acc_{b}')['candidate']:.3f} / {get(a, 'long', f'ece_{b}')['candidate']:.3f}" for a in arms) + " |")
print("\n".join(cu))
