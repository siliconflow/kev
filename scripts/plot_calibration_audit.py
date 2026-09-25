import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chartstyle as cs
from kev.suite import read_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = read_json(args.report)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    cs.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(left=0.07, right=0.86, top=0.8, bottom=0.30, wspace=0.55)
    cs.heading(fig, 0.07, 0.92, "Calibration audit: the full risk–coverage curve", size=22)
    counts = {(info.get("temperature_replay") or info.get("returned") or info)["micro"]["n"] for info in report["models"].values()}
    if len(counts) != 1:
        raise ValueError("curves must describe equal-size populations")
    cs.body(fig, 0.07, 0.87, f"Same {counts.pop()} clean development questions. Equal-confidence answers are accepted together.", size=12)
    palette = {"Kev-0.8B": cs.KEV["kev-0.8b"], "Kev-4B": cs.KEV["kev-4b"], "Kev-9B": cs.KEV["kev-9b"], "Jev": cs.JEV,
               "parent": cs.KEV["kev-9b"], "ce-control": cs.TEXT2, "smoothing-005": cs.KEV["kev-4b"],
               "ce-brier-05": cs.GREEN[700], "focal-1": cs.RED[700]}
    endpoints = []
    labels = {"parent": "Kev-4B, unchanged", "ce-control": "CE control", "smoothing-005": "Smoothing",
              "ce-brier-05": "CE + Brier", "focal-1": "Focal"}
    for i, (name, info) in enumerate(report["models"].items()):
        entry = info.get("temperature_replay") or info.get("returned") or info
        curve = entry["risk_coverage"]
        x, y = [p["coverage"] * 100 for p in curve], [p["risk"] * 100 for p in curve]
        color = palette[name]
        for ax in axes:
            ax.step(x, y, where="pre", color=color, lw=1.7)
        label = labels.get(name, name)
        endpoints.append((y[-1], label, color))
        lane = 0.07 + 0.88 * i / len(report["models"])
        fig.text(lane, 0.15, label, color=color, fontsize=10, weight="medium")
        fig.text(lane, 0.115, f"AURC {entry['micro']['aurc']:.4f}", fontsize=10, color=cs.TEXT2)
    last_y = -10.0
    for y, label, color in sorted(endpoints):
        text_y = max(y, last_y + 1.8)
        axes[1].annotate(label, xy=(100, y), xytext=(103, text_y), color=color, fontsize=9, va="center",
                         arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6})
        last_y = text_y
    for ax, ymax in zip(axes, (12, 40)):
        ax.axhline(5, color=cs.RULE, lw=1, ls="--")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Accepted questions (%)")
        ax.set_ylabel("Errors among accepted questions (%)")
        ax.set_xticks(np.arange(0, 101, 20))
        ax.grid(axis="y", color=cs.GRID, lw=0.6)
        cs.strip(ax)
    axes[0].set_title("Low-error region", loc="left", fontsize=14)
    axes[1].set_title("Full range", loc="left", fontsize=14)
    detail = "All arms recalibrated on the same independent calibration partition; new raw logits retained." if "selected_for_replication" in report else "Historical temperature replay is approximate; new trials save logits."
    cs.body(fig, 0.07, 0.05, "Descriptive development curves, not an unseen-data error guarantee. " + detail, size=9)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
