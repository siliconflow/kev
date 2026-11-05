"""Long-state memory, time and parity of LocalPredictor's long-row attention on CUDA, for one checkpoint on evals/longdoc-v1
development records (the fp32 exact path every small-family read uses).

    KEV_APP_NAME=kev-longmem uv run modal run modal_app.py::script --script long_state_memory.py --name long-state-4b-h100 \
        --gpu H100 --args "--run jaredpalmer/kev-4b"

Every record runs its state once (kev.shared_prefix, what LocalPredictor does for a long record on a hybrid backbone) under
one of these attention settings:
  math     LocalPredictor's global policy alone (flash and memory-efficient SDPA off): the exact math kernel everywhere
  legacy   #149's long-row context as it was: flash / memory-efficient / math allowed, transformers' grouped-query call
  long     kev.predictors.long_row_kernels, what a long row runs under now
  rows     the row form without any context, what a record under kev.model.ROW_PASS_TOKENS gets (8k parity records only)
Memory and time: the longest record of each --buckets bucket, every mode but rows; a mode that runs out of memory is
recorded as such (the rest continue). Parity: --parity records per part (cuad, synthetic) of each --parity-buckets bucket,
math vs long (and legacy, rows): max |dp| over options and argmax flips, on raw logits and at the checkpoint's temperature.
A profile of one short pass per mode names the attention kernels that ran. Writes <out>/report.json.
"""
import argparse, contextlib, json, time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from kev.data import materialize
from kev.device import allocated_bytes, empty_cache, out_of_memory, sync
from kev.model import rows_of
from kev.predictors import LocalPredictor, long_row_kernels
from kev.suite import CONTEXT, load_split, read_manifest, write_json

GB = 2 ** 30


def legacy():
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])


MODES = {"math": contextlib.nullcontext, "legacy": legacy, "long": long_row_kernels, "rows": contextlib.nullcontext}


def run(p, enc, mode):
    """-> (raw logits per question or None on OOM, seconds, peak GiB over resident)."""
    empty_cache("cuda"); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    sync("cuda"); t = time.perf_counter()
    try:
        with torch.no_grad(), MODES[mode]():
            zs = p.model.forward(enc) if mode == "rows" else p.model.forward_batch([enc], shared_prefix=True)[0]
            zs = [z.float().cpu() * p.temperature for z in zs]   # the head divides by its temperature in eval mode: raw logits
    except Exception as e:
        if not out_of_memory(e): raise
        zs = None
    sync("cuda")
    return zs, round(time.perf_counter() - t, 2), round((allocated_bytes("cuda") - base) / GB, 2)   # allocated_bytes: the CUDA peak


def kernels(p, enc, mode):
    """CUDA kernels of one pass whose names look like attention or softmax, by total time."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run(p, enc, mode)
    names = {}
    for e in prof.key_averages():
        n = e.key.lower()
        if any(s in n for s in ("fmha", "flash", "attention", "softmax", "efficient", "cutlass")):
            names[e.key[:120]] = round(getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) / 1000, 2)
    return dict(sorted(names.items(), key=lambda kv: -kv[1])[:8])


def compare(ref, other, temperature):
    dps, flips, n = [], 0, 0
    for a, b in zip(ref, other):
        for t in (1.0, temperature):
            pa, pb = torch.softmax(a / t, -1), torch.softmax(b / t, -1)
            dps.append((t, float((pa - pb).abs().max())))
        flips += int(a.argmax() != b.argmax()); n += 1
    return {"questions": n, "max_dp_raw": max(d for t, d in dps if t == 1.0), "max_dp_shipped": max(d for t, d in dps if t == temperature),
            "max_dlogit": max(float((a - b).abs().max()) for a, b in zip(ref, other)), "flips": flips}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--suite", default="evals/longdoc-v1")
    ap.add_argument("--buckets", default="16384,32768,65536")
    ap.add_argument("--parity-buckets", default="8192,16384")
    ap.add_argument("--parity", type=int, default=5, help="parity records per part and bucket")
    ap.add_argument("--memory-modes", default="math,legacy,long")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = LocalPredictor(a.run, "cuda", context=read_manifest(a.suite).get("context", CONTEXT))   # as kev.benchmark reads the suite
    records = load_split(a.suite, "development")
    enc = lambda r: p.model.encode(p.tok, materialize(r), max_state=p.context["max_state"], max_branch=p.context["max_branch"], strict=True)
    report = {"run": a.run, "gpu": torch.cuda.get_device_name(0), "total_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / GB, 1),
              "resident_gib": round(torch.cuda.memory_allocated() / GB, 2), "temperature": p.temperature, "environment": p.environment,
              "config": {k: getattr(p.model.lm.config.get_text_config(), k, None) for k in ("num_attention_heads", "num_key_value_heads", "head_dim", "hidden_size", "num_hidden_layers")}}

    def save(): write_json(out / "report.json", report)
    warm = enc(next(r for r in records if r["_meta"]["bucket"] == 4096))
    for mode in MODES: run(p, warm, mode)   # Triton autotuning and cuBLAS warm-up
    report["kernels_4k"] = {mode: kernels(p, warm, mode) for mode in ("math", "legacy", "long")}
    print(json.dumps(report["kernels_4k"], indent=1), flush=True); save()

    report["memory"] = {}
    for bucket in map(int, a.buckets.split(",")):
        e = max((enc(r) for r in records if r["_meta"]["bucket"] == bucket), key=lambda e: len(rows_of(e)[0]))
        state, _, rows = rows_of(e)
        entry = report["memory"][f"{bucket // 1024}k"] = {"state_tokens": len(state), "questions": len(rows), "longest_row": len(state) + max(len(r["ids"]) for r in rows)}
        results = {}
        for mode in a.memory_modes.split(","):
            zs, s, gb = run(p, e, mode)
            results[mode] = zs
            entry[mode] = {"seconds": s, "peak_over_resident_gib": gb, "oom": zs is None}
            print(bucket, mode, entry[mode], flush=True); save()
        if results.get("math") is not None and results.get("long") is not None: entry["long_vs_math"] = compare(results["math"], results["long"], p.temperature)

    report["parity"] = {}
    for bucket in map(int, a.parity_buckets.split(",")):
        picked = [r for part in ("cuad", "synthetic") for r in [r for r in records if r["_meta"]["bucket"] == bucket and r["_meta"]["part"] == part][:a.parity]]
        z = {m: [] for m in MODES}
        for r in picked:
            e = enc(r)
            for mode in (("math", "legacy", "long", "rows") if bucket <= 8192 else ("math", "legacy", "long")):
                zs, _, _ = run(p, e, mode)
                if zs is None: raise RuntimeError(f"{mode} ran out of memory on parity record {r['_meta']['id']}")
                z[mode] += zs
        report["parity"][f"{bucket // 1024}k"] = {"records": len(picked), **{f"{m}_vs_math": compare(z["math"], z[m], p.temperature) for m in ("legacy", "long", "rows") if z[m]}}
        print(bucket, json.dumps(report["parity"][f"{bucket // 1024}k"]), flush=True); save()
    save()


if __name__ == "__main__":
    main()
