"""Freeze scienthoon/jev-ood-calibration's 900 synthetic tickets as an eval-only Kev suite (three questions per ticket:
queue Choice, priority Score whose label follows an org rule absent from the text, angry Noul).

    uv run python scripts/freeze_scienthoon.py --repo /tmp/jevood --out evals/external/scienthoon-v1

Their live Jev results (results/jev_synth.jsonl, same 900 tickets, AI Gateway) are converted to benchmark rows alongside,
so Kev and Jev are compared on identical items without another Jev call. Source: MIT; the set regenerates from a seed.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from kev.suite import digest, record_digest, write_json

QUESTIONS = {"choice": ("queue", "unknowable_none"), "score": ("priority", "org_rule"), "noul": ("angry", "text")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--out", default="evals/external/scienthoon-v1")
    a = ap.parse_args()
    repo, out = Path(a.repo), Path(a.out)
    if out.exists(): raise FileExistsError(out)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    rows = [json.loads(l) for l in (repo / "data/val.jsonl").read_text().splitlines()]
    jev = [json.loads(l) for l in (repo / "results/jev_synth.jsonl").read_text().splitlines()]
    assert len(rows) == len(jev) == 900
    # one record per ticket with its three questions; tickets are identified by their state bytes
    tickets = {}
    for i, (r, j) in enumerate(zip(rows, jev)):
        key = json.dumps(r["state"], sort_keys=True); t = tickets.setdefault(key, {"state": r["state"], "questions": {}, "jev": {}})
        qid, note = QUESTIONS[r["type"]]
        if r["type"] == "choice": q = {"type": "choice", "instructions": r["question"], "criteria": r["options"], "label": r["label"], "src": "scienthoon_queue"}
        elif r["type"] == "score": q = {"type": "score", "instructions": r["question"], "criteria": r["levels"], "label": r["label"], "src": "scienthoon_priority"}
        else: q = {"type": "noul", "instructions": r["question"], "label": bool(r["label"]), "src": "scienthoon_angry"}
        t["questions"][qid] = q; t["jev"][qid] = j
    records, jev_rows = [], []
    for n, (key, t) in enumerate(tickets.items()):
        rid = f"scienthoon/{n:04d}"
        rec = {"state": t["state"], "questions": t["questions"],
               "_meta": {"row": n, "source": "scienthoon", "repo": "scienthoon/jev-ood-calibration", "split": "val", "id": rid, "group_id": rid, "variant": "clean",
                         "text_sha256": hashlib.sha256(key.casefold().encode()).hexdigest()}}
        rec["_meta"]["row_sha256"] = record_digest({k: v for k, v in rec.items() if k != "_meta"}); records.append(rec)
        for qid, q in t["questions"].items():
            j = t["jev"][qid]
            if q["type"] == "choice": keys = j["option_keys"]; label = keys.index(q["label"])
            elif q["type"] == "score": keys = [str(i) for i in range(len(q["criteria"]))]; label = q["label"]
            else: keys = ["false", "true"]; label = int(q["label"])
            p = np.array(j["probs"], dtype=float)
            if q["type"] == "noul": p = np.array([p[j["option_keys"].index("no")], p[j["option_keys"].index("yes")]])   # theirs are [yes, no]; ours [false, true]
            total = float(p.sum()); p = p / total if total > 0 else np.ones(len(keys)) / len(keys)
            jev_rows.append({"id": rid, "group": rid, "question": qid, "source": "scienthoon", "task": q["src"], "type": q["type"], "variant": "clean", "keys": keys, "label": label,
                             "pair_id": None, "sibling": None, "parent": rid, "p": p.tolist(), "raw_probability_sum": total, "zero_count": int((p == 0).sum()), "control_id": None})
    out.mkdir(parents=True); files = {}
    for name, rs in (("development.jsonl", records), ("train.jsonl", []), ("calibration.jsonl", []), ("test.jsonl", [])):
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs)); files[name] = {"sha256": digest(out / name), "records": len(rs)}
    write_json(out / "manifest.json", {"version": 1, "external": {"repo": "https://github.com/scienthoon/jev-ood-calibration", "commit": commit, "license": "MIT",
                                                                   "files": {"data/val.jsonl": digest(repo / "data/val.jsonl"), "results/jev_synth.jsonl": digest(repo / "results/jev_synth.jsonl")}},
                                       "base_revisions": {}, "dataset_revisions": {}, "holdout_sources": [], "trainable_sources": [], "eval_only_sources": ["scienthoon"],
                                       "context": {"max_state": 384, "max_branch": 1024, "max_packed": 2048, "truncate": False}, "files": files, "eval_only": True,
                                       "protocol": {"note": "priority is an org rule absent from the text (template urgency + angry + gold/enterprise tier): unknowable from the state; "
                                                            "their live Jev read (2026-09-19): queue 0.890, angry 0.917, priority 0.447, overall ECE 0.107"}})
    from kev.benchmark import summarize
    jd = Path("runs/jev-scienthoon-v1"); jd.mkdir(parents=True, exist_ok=True)
    write_json(jd / "rows.json", jev_rows); rep = summarize(jev_rows, heldout_sources=()); rep.update(source="converted from scienthoon results/jev_synth.jsonl", suite=str(out)); write_json(jd / "report.json", rep)
    print(len(records), "tickets;", "Jev converted:", {k: round(v["acc"], 3) for k, v in rep["tasks"].items()}, "| ECE", round(rep["clean"]["ece"], 3))


if __name__ == "__main__":
    main()
