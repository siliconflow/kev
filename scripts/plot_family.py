"""README figure: the Kev family against Jev, per out-of-domain source, plus the capacity/recipe curve.

    uv run python scripts/plot_family.py            # -> docs/kev-family.png, docs/kev-family-summary.json

Reads saved result.json files only (development partitions; the locked test is not plotted). Style: scripts/chartstyle.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
from chartstyle import GRID, HOLLOW, JEV, KEV, RULE, TEXT, TEXT2, body, display, heading, rule, strip, use_style

ROOT = Path(__file__).resolve().parents[1]
TASKS = [("sciq", "SciQ"), ("qnli", "QNLI"), ("contrastive_authorization", "Policy: authorization"), ("composition_held_and_or", "Rule: (A or B) and C"),
         ("composition_held_or_not", "Rule: (A and B) or not C"), ("tweet_offensive", "TweetEval offensive"), ("paws", "PAWS"),
         ("composition_held_conditional", "Rule: if A then not B else C"), ("mmlu", "MMLU, 4-way"), ("contrastive_deadline", "Policy: deadline (3-level Score)"), ("emotion", "Emotion, 6-way")]
MODELS = [("kev-0.5b", "runs/kev-05b-transfer-v4/report.json"), ("kev-8b-qwen3", "runs/v7-final/00-trial-0/result.json"), ("kev-0.8b", "runs/q35-08b/02-trial-2/result.json"),
          ("kev-4b", "runs/q35-4b-s23/00-trial-0/result.json"), ("kev-9b", "runs/q35-9b/01-trial-1/result.json")]
JEV_PATH = "runs/jev-transfer-v4/report.json"
# capacity x recipe curve: (params in B, recipe, [transfer acc per seed], source trials)
CURVE = [(0.6, "default recipe", [0.598, 0.595, 0.605], "v4-06b-hardened"), (0.6, "low lr + random rule structures", [0.613, 0.605, 0.620], "v7-06b"),
         (4.0, "default recipe", [0.704, 0.735], "v4-4b-baseline"), (4.0, "low lr", [0.759, 0.758, 0.759], "lowdrift-4b-v4/01, recipe-4b-v4-s1, recipe-4b-v4-s2"),
         (4.0, "low lr + random rule structures", [0.773, 0.790, 0.770], "v7-rc3, v7-final/02"),
         (8.2, "default recipe", [0.765, 0.741], "v3-8b-s0 (transfer-v3 = same bytes)"), (8.2, "low lr", [0.774, 0.779, 0.774], "recipe-8b-r1/00, recipe-8b-r2/01, recipe-8b-s2"),
         (8.2, "low lr + random rule structures", [0.796, 0.774], "v7-final/00, v7-final/01"),
         (0.8, "Qwen3.5 base, same recipe", [0.622, 0.634, 0.643], "q35-08b"), (4.0, "Qwen3.5 base, same recipe", [0.788, 0.800, 0.794, 0.770], "q35-4b, q35-4b-s23"), (9.7, "Qwen3.5 base, same recipe", [0.802, 0.812], "q35-9b")]
RECIPE_COLOR = {"default recipe": GRID, "low lr": KEV["kev-0.6b"], "low lr + random rule structures": KEV["kev-4b"], "Qwen3.5 base, same recipe": KEV["kev-9b"]}
RECIPE_SHORT = {"default recipe": "default recipe", "low lr": "low lr", "low lr + random rule structures": "Qwen3, low lr +\nrandom rules", "Qwen3.5 base, same recipe": "Qwen3.5,\nsame recipe"}


def transfer(path):
    r = json.loads((ROOT / path).read_text())
    t = r["transfer"] if "transfer" in r else r
    return {k: v["acc"] for k, v in t["tasks"].items()}, t["clean"]["acc"], t["clean"]["brier"], t["paired_flip"]["both_correct_rate"]


def main():
    use_style()
    data = {n: transfer(p) for n, p in MODELS}
    jev_tasks, jev_acc, jev_brier, jev_pairs = transfer(JEV_PATH)
    fig = plt.figure(figsize=(16, 11))
    heading(fig, .05, .93, "Where Kev matches Jev and where it does not, on data Kev never trained on", size=24)
    body(fig, .05, .89, "Accuracy per source on the frozen out-of-domain suite (transfer-v4, 764 records, development partition). Same items for every model. Jev via Vercel AI Gateway.", size=12.5)
    rule(fig, .05, .96, .865)

    # ---- left: dot plot per source, one row per source, one shared plot lane
    ax = fig.add_axes([.27, .175, .40, .645])
    heading(fig, .05, .835, "Accuracy by source", size=15)
    y = np.arange(len(TASKS))
    for yi, (key, _) in zip(y, TASKS):
        vals = {n: 100 * data[n][0][key] for n, _ in MODELS}; vals["Jev"] = 100 * jev_tasks[key]
        lo, hi = min(vals.values()), max(vals.values())
        ax.plot([lo, hi], [yi, yi], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
        for n, v in vals.items():
            if n in HOLLOW: ax.scatter([v], [yi], s=95, facecolor="white", edgecolor=KEV[n], linewidth=1.8, zorder=3)
            else: ax.scatter([v], [yi], s=95, color=JEV if n == "Jev" else KEV[n], zorder=3, edgecolor="white", linewidth=1.2)
        # direct labels: the best Kev and Jev, offset vertically so they never overlap
        best = max((n for n, _ in MODELS if n not in HOLLOW), key=lambda n: vals[n])
        ax.annotate(f"{vals[best]:.0f}", (vals[best], yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9.5, color=KEV[best], weight="medium")
        ax.annotate(f"{vals['Jev']:.0f}", (vals["Jev"], yi), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=9.5, color=JEV, weight="medium")
    ax.set_yticks(y); ax.set_yticklabels([label for _, label in TASKS], fontsize=12); ax.set_ylim(len(TASKS) - .5, -.5)
    ax.set_xlim(20, 104); ax.set_xticks(range(20, 101, 20)); ax.set_xticklabels([f"{v}%" for v in range(20, 101, 20)], fontsize=10.5, color=TEXT2)
    ax.grid(axis="x", color=GRID, lw=.8, zorder=0); ax.tick_params(length=0); strip(ax)
    # key as direct text, once
    kx = .27
    kx = .22
    for i, (n, c, hollow) in enumerate([("Jev", JEV, False), ("kev-9b", KEV["kev-9b"], False), ("kev-4b", KEV["kev-4b"], False), ("kev-0.8b", KEV["kev-0.8b"], False), ("kev-8b-qwen3", KEV["kev-8b"], True), ("kev-0.5b prototype", KEV["kev-0.5b"], True)]):
        fig.text(kx + .085 * i, .835, "○" if hollow else "●", fontsize=13, color=c, ha="left", va="baseline", weight="bold" if hollow else "regular")
        fig.text(kx + .085 * i + .014, .835, display(n), fontsize=10.5, color=TEXT, ha="left", va="baseline")
    body(fig, .05, .125, "Notice: on classification-shaped sources and trained rule families the 4B and 9B are within a few points of Jev or above it.\n"
         "The gap is concentrated in knowledge (MMLU), day-precision date arithmetic (deadline) and noisy-label emotion.\n"
         "Sources: QNLI, SciQ, TweetEval, PAWS, MMLU, Emotion; the policy families and rule structures shown were never trained.", size=11, va="top")

    # ---- right: capacity x recipe
    from matplotlib.lines import Line2D
    fig.add_artist(Line2D([.72, .72], [.06, .84], transform=fig.transFigure, color=RULE, lw=0.8))
    x0 = .755
    heading(fig, x0, .835, "What moved the overall number", size=15)
    cx = fig.add_axes([x0, .52, .17, .28])
    for recipe, color in RECIPE_COLOR.items():
        pts = [(p, [100 * a for a in accs]) for p, r, accs, _ in CURVE if r == recipe]
        for p, vals in pts: cx.scatter([p] * len(vals), vals, color=color, s=40, zorder=3, edgecolor="white", linewidth=1)
        cx.plot([p for p, _ in pts], [np.mean(v) for _, v in pts], color=color, lw=1.6, zorder=2)
    cx.axhline(100 * jev_acc, color=JEV, lw=1.6, zorder=1); cx.annotate(f"Jev {100 * jev_acc:.1f}%", (9, 100 * jev_acc), xytext=(9, 0), textcoords="offset points", fontsize=9.5, color=JEV, va="center", weight="medium")
    for j, (recipe, color) in enumerate(RECIPE_COLOR.items()):     # key as text lines under the axis, one per series (end labels collide at 8-9B)
        fig.text(x0, .445 - .02 * j, "●", fontsize=9, color=color, va="baseline"); fig.text(x0 + .012, .445 - .02 * j, RECIPE_SHORT[recipe].replace("\n", " "), fontsize=9.5, color=TEXT2 if color == GRID else TEXT, va="baseline")
    cx.set_xscale("log"); cx.set_xticks([0.7, 4, 9]); cx.set_xticklabels(["0.6–0.8B", "4B", "8–9B"], fontsize=10.5, color=TEXT2); cx.set_xlim(0.45, 13)
    cx.set_ylim(55, 90); cx.set_yticks(range(60, 91, 10)); cx.set_yticklabels([f"{v}%" for v in range(60, 91, 10)], fontsize=10.5, color=TEXT2)
    cx.grid(axis="y", color=GRID, lw=.8, zorder=0); cx.tick_params(length=0); strip(cx); cx.minorticks_off()
    body(fig, x0, .475, "Backbone size; one point per seed", size=10.5)

    rows = [("Capacity, 0.8B to 4B", "+15 to +17 pp, matched data"), ("Capacity, 4B to 9B", "+0.5 to +2 pp"),
            ("Learning rate 2e-4 to 5e-5, at 4B", "+4.7 pp, 95% CI [+0.4, +9.6]"), ("60 random rule structures instead of 8 shapes", "+1.5 to +3 pp; held-out rules 0.62 to 0.73"),
            ("Qwen3.5 base, same data and recipe", "+1–2 pp dev; locked test +7.3 pp at 9B, +2.9 at 4B, +4.8 at 0.8B"),
            ("Brier out of domain", f"Kev-9B {data['kev-9b'][2]:.3f}, Jev {jev_brier:.3f}"), ("Held-out rule pairs, both siblings correct", f"Kev-9B {data['kev-9b'][3]:.2f}, Jev {jev_pairs:.2f}")]
    for i, (k, v) in enumerate(rows):
        fig.text(x0, .345 - .039 * i, k, fontsize=10.5, color=TEXT, va="baseline"); fig.text(x0, .345 - .039 * i - .016, v, fontsize=10, color=TEXT2, va="baseline")
    body(fig, x0, .075, "Development numbers; provenance in runs/leaderboard.md.\nThe locked test was read once per checkpoint; see the cards.", size=9.5, va="top")
    body(fig, .05, .018, "LoRA r=16 + pointer head, all on decision-v7, Qwen3.5 bases. Kev-8B (Qwen3) is the previous generation, Kev-0.5B the prototype; both scored on the same items.  Regenerate: uv run python scripts/plot_family.py", size=10)

    out = ROOT / "docs/kev-family.png"
    fig.savefig(out, dpi=170, metadata={"Title": "Kev family vs Jev, out of domain"}); plt.close(fig)
    (ROOT / "docs/kev-family-summary.json").write_text(json.dumps({"models": {n: {"path": p, "transfer_acc": data[n][1], "transfer_brier": data[n][2], "tasks": data[n][0]} for n, p in MODELS},
                                                                     "jev": {"path": JEV_PATH, "transfer_acc": jev_acc, "transfer_brier": jev_brier, "tasks": jev_tasks}, "curve": CURVE}, indent=1))
    print(out, {n: round(data[n][1], 3) for n, _ in MODELS}, "jev", round(jev_acc, 3))


if __name__ == "__main__":
    main()
