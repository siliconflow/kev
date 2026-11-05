"""Kernel-drift probe (runs/drift-v1/REPORT.md). Default: Kev-27B v1 (jaredpalmer/kev-27b@v1-lora) on
evals/external/semif-v1 development (252 rows).

    git archive c154ed1 kev | tar -x -C /tmp/kev-drift-old      # the 2026-09-23 code, for the old-code subprocesses
    KEV_APP_NAME=kev-drift uv run modal run runs/drift-v1/probe_app.py::main --tag h200-a
    KEV_APP_NAME=kev-drift uv run modal run runs/drift-v1/probe_app.py::main --tag h100-kev4b-fp32-ref --repeats 1 --no-old \
        --reference --run jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101 --gpu H100   # --ieee: IEEE fp32 Triton dots
    KEV_APP_NAME=kev-drift uv run modal run runs/drift-v1/probe_app.py::main --tag h200-breadth --repeats 1 --no-old \
        --run jaredpalmer/kev-27b@01b81998019be550f0ae858727df49bac9511195 --suite evals/breadth-v1

One container loads the checkpoint once (today's kev, today's image, raw logits: temperature 1.0) and scores the suite
`repeats` times under each kernel configuration, swapped in place in transformers' Qwen3.5 module: new_conv_cuda (the
image as built: causal-conv1d's CUDA kernel), new_conv_torch (transformers' PyTorch conv, what the image ran before
#125), ref_torch_deltanet (--reference: PyTorch conv and PyTorch chunked delta rule, no Triton). Then, unless --no-old, it
runs kev.benchmark from the 2026-09-23 research commit (c154ed1, the code the r6-27bv2-s2-semif read ran) in subprocesses,
with causal-conv1d blocked (old_conv_torch) and not (old_conv_cuda). Writes runs/drift-v1/<tag>/{env.json,<config>.json}
(per row: id, question, raw logits). compare.py / pairs.py / metrics_offset.py read them.
"""
import json
import os
import sys
from pathlib import Path

import modal

RUN = "jaredpalmer/kev-27b@v1-lora"
SUITE = "evals/external/semif-v1"
OLD = "/tmp/kev-drift-old"   # git archive c154ed1 kev | tar -x -C /tmp/kev-drift-old

app = modal.App(os.environ.get("KEV_APP_NAME", "kev-drift"))
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import modal_app  # noqa: E402  (today's image definition, unchanged)
    image = modal_app.image.add_local_dir(f"{OLD}/kev", "/old/kev", copy=False) if os.path.isdir(OLD) else modal_app.image
else:
    image = modal.Image.debian_slim()   # in the container the function is already hydrated with the image above
hf_cache = modal.Volume.from_name("kev-hf-cache")

BLOCK_CONV = "import sys; sys.modules['causal_conv1d'] = None; "


def versions():
    import importlib.metadata as md, subprocess, torch
    out = {p: (md.version(p) if _installed(p) else None) for p in ("torch", "transformers", "peft", "flash-linear-attention", "fla-core", "triton", "causal-conv1d", "accelerate", "safetensors")}
    out.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               driver=subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    return out


def _installed(p):
    import importlib.metadata as md
    try: md.version(p); return True
    except md.PackageNotFoundError: return False


def rows_of_dir(d):
    rows = json.loads((Path(d) / "rows.json").read_text())
    return [{"id": r["id"], "question": r["question"], "logits": r["logits"]} for r in rows]


@app.function(image=image, gpu="H200", cpu=4, memory=(32768, 131072), timeout=3600, volumes={"/hf": hf_cache},
              secrets=[modal.Secret.from_name(os.environ.get("KEV_HF_SECRET", "huggingface-secret"))])   # private suites (breadth-v1)
def probe(repeats: int = 2, old: bool = True, run: str = RUN, reference: bool = False, ieee: bool = False, suite: str = SUITE):
    if ieee:   # IEEE fp32 dot products in every Triton kernel (fla's DeltaNet kernels default to TF32 on Ampere+)
        os.environ["TRITON_F32_DEFAULT"] = "ieee"; os.environ["TRITON_CACHE_DIR"] = "/tmp/triton-ieee"
    import subprocess, time, torch
    os.chdir("/root")
    sys.path.insert(0, "/root")
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq
    if ieee:
        import triton.language as tl
        import fla.ops.gated_delta_rule.chunk_fwd as chunk_fwd
        chunk_fwd.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")
    from kev.benchmark import evaluate_records
    from kev.checkpoint import LoadOptions
    from kev.predictors import LocalPredictor
    from kev.suite import load_split, read_manifest
    env = {"versions": versions(), "conv_impl": getattr(__import__("inspect").getclosurevars(mq.causal_conv1d_fn).nonlocals.get("implementation"), "__module__", None),
           "chunk_impl": getattr(__import__("inspect").getclosurevars(mq.torch_chunk_gated_delta_rule).nonlocals.get("implementation"), "__module__", None)}
    env["triton_f32_default"] = os.environ.get("TRITON_F32_DEFAULT")
    print(json.dumps(env), flush=True)
    records = load_split(suite, "development")
    context = read_manifest(suite).get("context")
    t = time.time()
    pred = LocalPredictor(run, "cuda", LoadOptions(temperature=1.0), context=context)
    env["load_s"] = time.time() - t
    env["deltanet_layers"] = sum(isinstance(m, mq.Qwen3_5GatedDeltaNet) for m in pred.model.lm.modules())
    env["lm_dtype"] = str(next(pred.model.lm.parameters()).dtype)
    env["checkpoint_path"] = pred.run
    results = {}
    fused_fn, fused_up, fla_chunk = mq.causal_conv1d_fn, mq.causal_conv1d_update, mq.torch_chunk_gated_delta_rule
    configs = [("new_conv_cuda", fused_fn, fused_up, fla_chunk), ("new_conv_torch", fused_fn.__wrapped__, fused_up.__wrapped__, fla_chunk)]
    if reference:   # transformers' PyTorch reference for the whole DeltaNet (conv and chunked delta rule): no Triton, no TF32 dots
        configs += [("ref_torch_deltanet", fused_fn.__wrapped__, fused_up.__wrapped__, fla_chunk.__wrapped__)]
    for rep in range(repeats):
        for name, fn, up, chunk in configs:
            mq.causal_conv1d_fn, mq.causal_conv1d_update, mq.torch_chunk_gated_delta_rule = fn, up, chunk
            out = Path(f"/tmp/out/{name}-{rep}")
            t = time.time()
            evaluate_records(records, pred, out)
            results[f"{name}-{rep}"] = rows_of_dir(out)
            print(f"{name}-{rep}: {time.time() - t:.0f}s", flush=True)
    mq.causal_conv1d_fn, mq.causal_conv1d_update, mq.torch_chunk_gated_delta_rule = fused_fn, fused_up, fla_chunk
    del pred; torch.cuda.empty_cache()
    if old:
        for name, prefix in (("old_conv_torch", BLOCK_CONV), ("old_conv_cuda", "")):
            out = f"/tmp/out/{name}"
            code = (prefix + "import runpy, sys; sys.argv = ['kev.benchmark', '--run', %r, '--suite', %r, '--out', %r, '--device', 'cuda']; "
                    "runpy.run_module('kev.benchmark', run_name='__main__')") % (run, f"/root/{suite}", out)
            t = time.time()
            subprocess.run([sys.executable, "-c", code], check=True, cwd="/root", env={**os.environ, "PYTHONPATH": "/old", "KEV_TEMPERATURE": "1.0"})
            results[name] = rows_of_dir(out)
            print(f"{name}: {time.time() - t:.0f}s", flush=True)
    return env, results


@app.local_entrypoint()
def main(tag: str, repeats: int = 2, old: bool = True, run: str = RUN, gpu: str = "H200", reference: bool = False, ieee: bool = False, suite: str = SUITE):
    env, results = probe.with_options(gpu=gpu).remote(repeats, old, run, reference, ieee, suite)
    d = Path(__file__).parent / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "env.json").write_text(json.dumps(env, indent=1) + "\n")
    for name, rows in results.items():
        (d / f"{name}.json").write_text(json.dumps(rows) + "\n")
    print(json.dumps(env, indent=1))
