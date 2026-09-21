"""Freeze SemIf's committed fixtures as an eval-only Kev suite, so Kev and live Jev can be scored on them with our harness.

    uv run python scripts/freeze_semif.py --repo /tmp/semif --out evals/external/semif-v1

Source: github.com/TheoLeeCJ/SemIf benchmarks/data/{authored144,perturbations108}.jsonl (MIT). Each row is one 3-way
choice; option ids and descriptions are carried over verbatim, the label index becomes the option id. Perturbation rows
keep their variant (option_reversal / criterion_wrapper / irrelevant_context) and base_id so paired flips can be read.
SemIf reports balanced accuracy per family; we report accuracy per task plus everything kev.benchmark reports.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from kev.suite import digest, record_digest, write_json


def convert(row, source, variant="clean", parent=None):
    keys = [o["id"] for o in row["options"]]
    rec = {"state": row["state"], "questions": {"decision": {"type": "choice", "instructions": row["question"],
                                                              "criteria": {o["id"]: o["description"] for o in row["options"]}, "label": keys[row["label"]], "src": f"semif_{row['family']}"}},
           "_meta": {"row": row["id"], "source": source, "repo": "TheoLeeCJ/SemIf", "split": row.get("split", "frozen"), "id": f"{source}/{row['id']}", "group_id": f"{source}/{row.get('group_id', row['id'])}",
                     "variant": variant, "family": row["family"], "text_sha256": hashlib.sha256(json.dumps(row["state"], sort_keys=True, ensure_ascii=False).casefold().encode()).hexdigest()}}
    if parent: rec["_meta"]["parent_id"] = parent
    rec["_meta"]["row_sha256"] = record_digest({k: v for k, v in rec.items() if k != "_meta"})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--out", default="evals/external/semif-v1")
    a = ap.parse_args()
    repo, out = Path(a.repo), Path(a.out)
    if out.exists(): raise FileExistsError(out)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    authored = [json.loads(l) for l in (repo / "benchmarks/data/authored144.jsonl").read_text().splitlines()]
    perturb = [json.loads(l) for l in (repo / "benchmarks/data/perturbations108.jsonl").read_text().splitlines()]
    records = [convert(r, "semif_authored") for r in authored]
    records += [convert(r, "semif_perturbation", variant=r["provenance"]["variant"], parent=f"semif_authored/{r['provenance']['base_id']}") for r in perturb]
    out.mkdir(parents=True)
    files = {}
    for name, rows in (("development.jsonl", records), ("train.jsonl", []), ("calibration.jsonl", []), ("test.jsonl", [])):
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)); files[name] = {"sha256": digest(out / name), "records": len(rows)}
    write_json(out / "manifest.json", {"version": 1, "external": {"repo": "https://github.com/TheoLeeCJ/SemIf", "commit": commit, "license": "MIT",
                                                                   "files": {p.name: digest(p) for p in [repo / "benchmarks/data/authored144.jsonl", repo / "benchmarks/data/perturbations108.jsonl"]}},
                                       "base_revisions": {}, "dataset_revisions": {}, "holdout_sources": [], "trainable_sources": [], "eval_only_sources": ["semif_authored", "semif_perturbation"],
                                       "context": {"max_state": 384, "max_branch": 1024, "max_packed": 2048, "truncate": False}, "files": files, "eval_only": True,
                                       "protocol": {"note": "SemIf reports mean family balanced accuracy on the 144 (direct Qwen3.5-4B logits 0.813) and on the 36 perturbation bases (0.723); "
                                                            "perturbation rows carry their variant in _meta.variant and the base row in parent_id"}})
    print({s: sum(r["_meta"]["source"] == s for r in records) for s in ("semif_authored", "semif_perturbation")})


if __name__ == "__main__":
    main()
