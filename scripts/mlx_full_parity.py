"""The MLX full-weight path (kev.mlx_model.load_full) against the MLX LoRA path and the fp32 torch path, with the memory
each load takes, on this Mac.

    uv run python scripts/merge_lora_checkpoint.py --lora jaredpalmer/kev-4b@<sha> --out /tmp/kev-4b-full --weights_dtype bf16
    uv run --extra mlx python scripts/mlx_full_parity.py --full /tmp/kev-4b-full/checkpoint --lora jaredpalmer/kev-4b@<sha> \\
        --n 40 --out runs/mlx-full-4b/report.json

For n clean development records of decision-v7, each in its own process (so every peak is that load's alone):
  mlx-full   the full-weight checkpoint on MLX (no merge)
  mlx-lora   its LoRA source on MLX (base + merge_lora). The merged weights are the same bf16 bits, but the Qwen3.5 bases
             store two small tensors per DeltaNet layer (A_log and the gated-norm weight) in fp32, which the LoRA path
             keeps and a full-weight export holds in bf16 (as every kev.train --full_ft run trains and saves them)
  mlx-lora-bf16  the LoRA path with exactly those tensors rounded to bf16: the full checkpoint's values, so the same answers
  torch-full the full-weight checkpoint on the torch path in fp32 on MPS (the saved bf16 values computed in fp32)
and reports max / mean |dp| and argmax flips for each pair, plus per load: seconds, MLX
peak memory (mx.get_peak_memory) and active memory after load, and the process's peak RSS. The checkpoint's bf16 weight
size (shard bytes) is the floor a full-weight load can reach.
"""
import argparse, json, resource, statistics, subprocess, sys, tempfile, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.checkpoint import Checkpoint, LoadOptions  # noqa: E402
from kev.data import materialize  # noqa: E402
from kev.suite import load_split, read_json, write_json  # noqa: E402

GB = 1e9


def records(suite, n):
    return [materialize(r) for r in load_split(suite, "development") if r["_meta"]["variant"] == "clean"][:n]


def rss_peak():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GB   # bytes on macOS


def round_fp32_tensors(m):
    """The tensors the Qwen3.5 bases store in fp32 (every DeltaNet layer's A_log and gated-norm weight) rounded to bf16 in
    an MLX model, as every full-weight export holds them. -> how many."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    fp32 = [(k, v.astype(mx.bfloat16)) for k, v in tree_flatten(m.lm.parameters()) if v.dtype == mx.float32]
    m.lm.load_weights(fp32, strict=False); mx.eval(m.lm.parameters())
    return len(fp32)


def phase(name, run, suite, n, out):
    """One load + scoring pass in this process; writes {probs, stats} to `out`."""
    recs = records(suite, n)
    ck = Checkpoint(run)
    started = time.time()
    if name == "torch-full":
        tok, m = ck.load("mps", LoadOptions(backend="torch", dtype=torch.float32))
        stats = {}
    else:
        import mlx.core as mx
        mx.reset_peak_memory()
        tok, m = ck.load("mps", LoadOptions(backend="mlx"))
        if name == "mlx-lora-bf16": round_fp32_tensors(m)
        stats = {"mlx_peak_load_gb": mx.get_peak_memory() / GB, "mlx_active_after_load_gb": mx.get_active_memory() / GB,
                 "mlx_cache_after_load_gb": mx.get_cache_memory() / GB}
    stats.update(load_seconds=round(time.time() - started, 1), rss_peak_after_load_gb=rss_peak(), backend=m.backend, dtype=m.dtype)
    started = time.time()
    probs = [[p.tolist() for p in m.probs(m.encode(tok, rec))] for rec in recs]
    stats["score_seconds"] = round(time.time() - started, 1)
    stats["rss_peak_gb"] = rss_peak()
    if name != "torch-full":
        import mlx.core as mx
        stats["mlx_peak_gb"] = mx.get_peak_memory() / GB
    write_json(Path(out), {"probs": probs, "stats": stats})


def compare(got, ref):
    dp, flips = [], 0
    for rec_g, rec_r in zip(got, ref):
        for p, r in zip(rec_g, rec_r):
            p, r = torch.tensor(p), torch.tensor(r)
            dp.append(float((p - r).abs().max())); flips += int(p.argmax() != r.argmax())
    return {"questions": len(dp), "max_dp": max(dp), "mean_dp": statistics.mean(dp), "argmax_flips": flips, "over_0.02": sum(d > 0.02 for d in dp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="full-weight checkpoint (scripts/merge_lora_checkpoint.py --weights_dtype bf16)")
    ap.add_argument("--lora", help="its LoRA source (default: head.pt merged_lora.source.path)")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--suite", default="evals/v7/decision-v7")
    ap.add_argument("--phases", default="mlx-full,mlx-lora,mlx-lora-bf16,torch-full"); ap.add_argument("--out", default="")
    ap.add_argument("--phase", help=argparse.SUPPRESS); ap.add_argument("--run", help=argparse.SUPPRESS); ap.add_argument("--to", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.phase: return phase(a.phase, a.run, a.suite, a.n, a.to)
    full = Checkpoint(a.full)
    if not full.full: raise SystemExit(f"{a.full} is not a full-weight checkpoint")
    lora = a.lora or full.meta.extra.get("merged_lora", {}).get("source", {}).get("path")
    report = {"full": a.full, "weights_sha256": full.weights_sha256(), "lora": lora, "suite": a.suite, "records": a.n, "weights_gb": sum(p.stat().st_size for p in full.shards()) / GB,
              "weights_dtype": full.meta.weights_dtype, "machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
              "memory_gb": int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2 ** 30}
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in a.phases.split(","):
            run = lora if name.startswith("mlx-lora") else a.full
            to = Path(tmp) / f"{name}.json"
            print(f"{name}: {run}", flush=True)
            subprocess.run([sys.executable, __file__, "--phase", name, "--run", str(run), "--suite", a.suite, "--n", str(a.n), "--to", str(to), "--full", a.full], check=True)
            results[name] = read_json(to)
            print(json.dumps(results[name]["stats"]), flush=True)
    report["loads"] = {k: v["stats"] for k, v in results.items()}
    pairs = [("mlx-full", "mlx-lora"), ("mlx-full", "mlx-lora-bf16"), ("mlx-full", "torch-full"), ("mlx-lora", "torch-full")]
    report["parity"] = {f"{x} vs {y}": compare(results[x]["probs"], results[y]["probs"]) for x, y in pairs if x in results and y in results}
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); write_json(Path(a.out), report)


if __name__ == "__main__":
    main()
