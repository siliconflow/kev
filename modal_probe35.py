"""Zero-shot base-model probe for Qwen3.5 bases on the frozen transfer suite, in its own image (transformers >= 5 is
required for the qwen3_5 architecture; the main Modal app pins the repo's uv.lock, which is on transformers 4.x).

    uv run modal run modal_probe35.py --bases Qwen/Qwen3.5-4B-Base,Qwen/Qwen3.5-9B-Base

Writes benchmark-compatible rows/report to the kev-runs volume under /probes/<name> and pulls them to runs/probes/.
Same readout as scripts/base_mmlu_probe.py: next-token letter logits over the rendered options, no training.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("kev-probe35")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "transformers>=5.17,<6", "peft>=0.18", "accelerate", "datasets", "numpy", "scikit-learn",
                    "huggingface_hub", "pydantic", "flash-linear-attention")
    .uv_pip_install("triton>=3.7.1")     # second step: torch 2.8 pins triton 3.4 in one resolve; fla refuses its gated chunk backward on Hopper below 3.7.1 (fla#640)
    .add_local_dir(ROOT / "kev", "/root/kev")
    .add_local_dir(ROOT / "evals", "/root/evals")
    .add_local_dir(ROOT / "scripts", "/root/scripts")
)
hf_cache = modal.Volume.from_name("kev-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("kev-runs", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface-secret")]


@app.function(image=image, gpu=os.environ.get("KEV_PROBE_GPU", "H100"), cpu=2, memory=(32768, 131072), retries=0, timeout=3600,
              volumes={"/runs": runs_volume, "/root/.cache/huggingface": hf_cache}, secrets=secrets)
def probe(base, suite, name, tasks="all", prompt="plain", split="development", revision=None, adapter=None):
    import os
    out = Path("/runs/probes") / name
    if out.exists():
        raise FileExistsError(f"probe {name} exists")
    try:
        subprocess.run([sys.executable, "/root/scripts/base_mmlu_probe.py", "--base", base, "--suite", f"/root/{suite}", "--tasks", tasks, "--device", "cuda", "--out", str(out), "--prompt", prompt, "--split", split] + (["--revision", revision] if revision else []) + (["--adapter", adapter] if adapter else []),
                       check=True, cwd="/root", env={**os.environ, "PYTHONPATH": "/root"})
    finally:
        runs_volume.commit(); hf_cache.commit()
    return json.loads((out / "report.json").read_text())["clean"]


@app.local_entrypoint()
def main(bases: str, suite: str = "evals/v4/transfer-v4", tasks: str = "all", prompt: str = "plain", split: str = "development", revision: str = "", adapter: str = "", tag: str = ""):
    jobs = []
    for base in bases.split(","):
        name = base.split("/")[-1].lower().replace(".", "") + ("-semif" if prompt == "semif" else "-base") + (f"-{tag}" if tag else "") + "-" + suite.split("/")[-1] + ("" if split == "development" else f"-{split}")
        if (ROOT / "runs/probes" / name).exists():
            print(f"skip {name}: exists locally"); continue
        jobs.append((base, suite, name, tasks, prompt, split, revision or None, adapter or None))
    for (base, _, name, *_), result in zip(jobs, probe.starmap(jobs, return_exceptions=True)):
        if isinstance(result, Exception):
            print(f"{name}: FAILED {type(result).__name__}: {str(result)[:200]}"); continue
        target = ROOT / "runs/probes" / name; target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "kev-runs", f"/probes/{name}", str(target.parent)], check=True)
        print(f"{name}: acc {result['acc']:.3f} brier {result['brier']:.3f} conf-err {result['confident_error_rate']:.3f}")


@app.function(image=image, gpu=os.environ.get("KEV_PROBE_GPU", "H100"), cpu=2, memory=(32768, 131072), retries=0, timeout=3600,
              volumes={"/runs": runs_volume, "/root/.cache/huggingface": hf_cache}, secrets=secrets)
def bench(run, suite, name, flags=""):
    """kev.benchmark for a Hub checkpoint on any local suite directory (mounted at run time), written to /runs/bench/<name>.
    flags: extra benchmark switches separated by spaces (e.g. "--date_facts")."""
    import os
    out = Path("/runs/bench") / name
    if out.exists(): raise FileExistsError(f"bench {name} exists")
    try:
        source = ["--data", f"/root/{suite}"] if suite.endswith(".jsonl") else ["--suite", f"/root/{suite}"]     # a .jsonl is a --data file (kev.data.load_records)
        subprocess.run([sys.executable, "-m", "kev.benchmark", "--run", run, *source, "--out", str(out), "--device", "cuda", *flags.split()], check=True, cwd="/root", env={**os.environ, "PYTHONPATH": "/root"})
    finally:
        runs_volume.commit(); hf_cache.commit()
    return json.loads((out / "report.json").read_text())["clean"]


@app.local_entrypoint()
def benchmarks(jobs: str):
    """jobs: comma-separated run@suite@name[@flags] entries; flags are extra kev.benchmark switches (e.g. --date_facts)."""
    triples = [(j.split("@") + [""])[:4] for j in jobs.split(",")]
    for (run, suite, name, _), result in zip(triples, bench.starmap(triples, return_exceptions=True)):
        if isinstance(result, Exception): print(f"{name}: FAILED {type(result).__name__}: {str(result)[:300]}"); continue
        target = ROOT / "runs" / name; subprocess.run([sys.executable, "-m", "modal", "volume", "get", "kev-runs", f"/bench/{name}", str(target.parent)], check=True)
        print(f"{name}: acc {result['acc']:.3f} brier {result['brier']:.3f}")


@app.function(image=image, gpu=os.environ.get("KEV_PROBE_GPU", "H200"), cpu=2, memory=(32768, 131072), retries=0, timeout=2400,
              volumes={"/runs": runs_volume, "/root/.cache/huggingface": hf_cache}, secrets=secrets)
def smoke_train(base, revision):
    """Load a base through DecisionModel with the Kev LoRA config, report the adapter size and which modules it hit, run one
    training step on a real record with gradient checkpointing, and report peak memory. For deciding whether a base fits."""
    import os, time, torch
    os.environ["PYTHONPATH"] = "/root"
    from kev.model import DecisionModel, load_tokenizer
    from kev.data import materialize
    from kev.suite import load_split
    t0 = time.time(); tok = load_tokenizer(base, revision=revision)
    m = DecisionModel(base, tok, "cuda", dtype=torch.bfloat16, lora=16, revision=revision)
    m.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.lm.config.use_cache = False; m.train()
    trainable = [(n, p.numel()) for n, p in m.lm.named_parameters() if p.requires_grad]
    hit = sorted({n.split(".lora_")[0].split(".")[-1] for n, _ in trainable})
    expert_hits = sum(k for n, k in trainable if ".experts." in n)
    recs = [materialize(r) for r in load_split("/root/evals/v7/decision-v7", "development")[:2]]
    encs = [m.encode(tok, r, strict=True) for r in recs]
    torch.cuda.reset_peak_memory_stats(); t1 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m.forward_batch(encs)
    loss = sum(torch.nn.functional.cross_entropy(z.float()[None], torch.tensor([q["label"]], device="cuda")) for zs, r in zip(logits, recs) for z, q in zip(zs, r["questions"]))
    loss.backward(); torch.cuda.synchronize()
    steady = []
    for k in range(3):                       # steady-state step time (the first step pays Triton compilation)
        m.lm.zero_grad(set_to_none=True); m.head.zero_grad(set_to_none=True); ts = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = m.forward_batch(encs)
        l2 = sum(torch.nn.functional.cross_entropy(z.float()[None], torch.tensor([q["label"]], device="cuda")) for zs, r in zip(lg, recs) for z, q in zip(zs, r["questions"]))
        l2.backward(); torch.cuda.synchronize(); steady.append(round(time.time() - ts, 2))
    return {"steady_step_seconds_2_records": steady, "questions_per_record": [len(r["questions"]) for r in recs],"base": base, "hybrid": m.hybrid, "load_seconds": round(t1 - t0), "step_seconds": round(time.time() - t1, 1), "trainable_params_M": round(sum(k for _, k in trainable) / 1e6, 1),
            "lora_module_names": hit, "routed_expert_lora_params": expert_hits, "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "weights_gb": round(sum(p.numel() * p.element_size() for p in m.lm.parameters()) / 1e9, 1), "loss": round(loss.item(), 3)}


@app.local_entrypoint()
def smoke(base: str, revision: str):
    print(json.dumps(smoke_train.remote(base, revision), indent=1))
