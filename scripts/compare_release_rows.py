#!/usr/bin/env python3
"""Compare a served read of a published checkpoint with the committed raw read of the same checkpoint, row for row.

    uv run python scripts/compare_release_rows.py --rows runs/rel27-public/0-jaredpalmer_kev-27b/rows.json \
        --reference runs/r23-27b-k-w85-semif/rows.json

A Hub load serves at head.pt's temperature T; the reference was read at T 1.0. PointerHead divides the logits by T, which
torch computes as a multiply by the fp32 reciprocal, so an identical backbone and head give served logits equal to
fp32(z_reference * fp32(1 / T)) bit for bit. Prints (and with --out writes) the counts: rows matched, logits exact under
that rule, argmax equal, the largest logit and probability differences.
"""
import argparse
import json

import numpy as np

from kev.suite import read_json, write_json


def key(row):
    return row["id"], row["question"], row["variant"]


def compare(rows, reference):
    ref = {key(r): r for r in reference}
    if len(ref) != len(reference): raise ValueError("reference rows are not unique by (id, question, variant)")
    temps = sorted({r.get("inference_temperature", 1.0) for r in rows})
    if len(temps) != 1: raise ValueError(f"served rows mix temperatures {temps}")
    inv = np.float32(1.0 / temps[0])
    missing = [key(r) for r in rows if key(r) not in ref]
    if missing or len(rows) != len(reference): raise ValueError(f"row sets differ: {len(rows)} served, {len(reference)} reference, {len(missing)} unmatched")
    exact = argmax = 0
    dz = dp = 0.0
    for r in rows:
        z_ref = np.asarray(ref[key(r)]["logits"], dtype=np.float32)
        expect, got = (z_ref * inv).astype(np.float32), np.asarray(r["logits"], dtype=np.float32)
        exact += bool(np.array_equal(expect, got))
        argmax += int(np.argmax(got)) == int(np.argmax(z_ref))
        dz = max(dz, float(np.max(np.abs(expect.astype(np.float64) - got))))
        p_ref = np.exp(expect.astype(np.float64) - expect.max()); p_ref /= p_ref.sum()
        dp = max(dp, float(np.max(np.abs(p_ref - np.asarray(r["p"], dtype=np.float64)))))
    return {"rows": len(rows), "inference_temperature": temps[0], "logits_exact": exact, "argmax_equal": argmax,
            "max_abs_dlogit_at_T": dz, "max_abs_dp_at_T": dp,
            "logits_rule": "served logits == fp32(z_reference * fp32(1 / T)), z_reference read at T 1.0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="served rows.json (the Hub load)")
    ap.add_argument("--reference", required=True, help="committed rows.json of the same checkpoint read at T 1.0")
    ap.add_argument("--out", help="write the comparison here")
    a = ap.parse_args()
    result = {"rows_path": a.rows, "reference": a.reference, **compare(read_json(a.rows), read_json(a.reference))}
    if a.out: write_json(a.out, result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
