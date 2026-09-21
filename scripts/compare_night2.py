"""Night-2 delta read-out: each delta trial vs the released checkpoint it started from, on transfer-v4 dev (same items)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.benchmark import paired_bootstrap
ROOT = Path(__file__).resolve().parents[1]
REF = {"9b": ("Kev-9B", "q35-9b/01-trial-1"), "4b": ("Kev-4B", "q35-4b-s23/00-trial-0")}
NAMES = ["dates", "unknowable", "assertion", "all"]

def read(path):
    p = ROOT / "runs" / path
    if not (p / "result.json").exists(): return None
    r = json.loads((p / "result.json").read_text()); t = r["transfer"]; c = t["clean"]
    return {"dev": r["clean"]["acc"], "acc": c["acc"], "brier": c["brier"], "cerr": c["confident_error_rate"], "cov5": c["coverage_at_5pct_error"], "pairs": t["paired_flip"]["both_correct_rate"],
            "deadline": t["tasks"]["contrastive_deadline"]["acc"], "mmlu": t["tasks"]["mmlu"]["acc"], "paws": t["tasks"]["paws"]["acc"], "emotion": t["tasks"]["emotion"]["acc"], "rows": json.loads((p / "transfer/rows.json").read_text())}

for size, (name, ref_path) in REF.items():
    ref = read(ref_path)
    print(f"\n== deltas from {name} (transfer-v4 dev; paired vs the released checkpoint)")
    print(f"   {'trial':12} {'dev':>6} {'acc':>6} {'paired Δ [95% CI]':>24} {'brier':>6} {'cerr':>6} {'cov@5%':>6} {'pairs':>5} {'deadl':>5} {'mmlu':>5} {'paws':>5} {'emo':>5}")
    print(f"   {name:12} {ref['dev']:6.3f} {ref['acc']:6.3f} {'reference':>24} {ref['brier']:6.3f} {ref['cerr']:6.3f} {ref['cov5']:6.2f} {ref['pairs']:5.2f} {ref['deadline']:5.2f} {ref['mmlu']:5.2f} {ref['paws']:5.2f} {ref['emotion']:5.2f}")
    for i, n in enumerate(NAMES):
        r = read(f"night2-{size}/{i:02d}-trial-{i}")
        if r is None: print(f"   {n:12} (not finished)"); continue
        b = paired_bootstrap(r["rows"], ref["rows"], metric="acc"); ci = f"{b['macro_acc_delta']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]"
        print(f"   {n:12} {r['dev']:6.3f} {r['acc']:6.3f} {ci:>24} {r['brier']:6.3f} {r['cerr']:6.3f} {r['cov5']:6.2f} {r['pairs']:5.2f} {r['deadline']:5.2f} {r['mmlu']:5.2f} {r['paws']:5.2f} {r['emotion']:5.2f}")
