"""Merge a LoRA checkpoint into its base and write it as a full-weight checkpoint, so full-weight training can start from it
(`kev.train --full_ft 1 --init_from <out>`; round 25 continues from Kev-27B instead of from the base).

    uv run python scripts/merge_lora_checkpoint.py --lora jaredpalmer/kev-27b@01b81998019be550f0ae858727df49bac9511195 \\
        --out runs/r25-init/kev-27b-merged [--like <a full-weight checkpoint of the same base>]    # -> <out>/checkpoint
    KEV_APP_NAME=kev-r25 uv run modal run modal_app.py::merge_adapter --lora jaredpalmer/kev-27b@<sha> --out /runs/r25-init/kev-27b-merged

The backbone is the LoRA checkpoint's base built as its loader builds it (kev.model.DecisionModel in the checkpoint's
weights dtype, bf16 for Kev-27B), and every adapted weight becomes round(fp32(W) + fp32(delta)) in that dtype, where delta
is peft's own get_delta_weight of the layer (the fp32 value peft's merge adds): exactly the backbone
scripts/interpolate_checkpoint.py writes at alpha 0 toward the same checkpoint (tests/test_unit.py), without streaming a
second full-weight checkpoint (round 23 read and hashed a 51 GB SFT checkpoint to use it only for names and shapes). The
backbone is written by kev.full_ft.write_backbone, the function a full-weight run saves with (save_pretrained, 5 GB shards),
so names, shard layout and config.json are what kev.train --full_ft writes. Refused like interpolate_checkpoint's --toward:
DoRA and other LoRA variants, LoRA biases, modules_to_save, trained token embeddings.

`--weights_dtype bf16` writes an fp32-trained LoRA checkpoint (every Kev below 27B) as a bf16 full-weight checkpoint: the
merge stays in fp32 and each tensor is rounded to bf16 once, round(fp32(W) + fp32(delta)), the arithmetic of the served
merge on both backends (LoadOptions.merge, kev.mlx_model.merge_lora). Tensors the base stores in bf16 come back exactly;
the few it stores in fp32 (the Qwen3.5 bases: A_log and the gated-norm weight of each DeltaNet layer) are rounded, as in
every bf16 full-weight export. That is what the MLX full-weight path is checked against on the small Kevs
(tests/test_mlx.py, runs/mlx-full-*). head.pt then says bf16.

head.pt: the LoRA checkpoint's meta and pointer head unchanged (same tensors, head_dim, temperature and its fit, training
args), with lora = 0 and weights = "full" (the loader rule and --init_from's COMPAT_FIELDS read those) and `merged_lora`
added: {source: {path, resolved, weights_sha256, head_sha256}, base, adapted_tensors, formula}. Tokenizer files are copied;
the source run's training records (training_config.json, provenance.json, ...) are not. `--like` checks, from safetensors
headers only, that the result has the same tensor names and shapes as another full-weight checkpoint of the same base
(what kev.train --full_ft --init_from will load it over). Written to <out>/checkpoint.partial and renamed when complete;
<out>/merge.json reports the hashes and the seconds of each phase.
"""
import argparse
import hashlib
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.checkpoint import Checkpoint, write_meta  # noqa: E402
from kev.full_ft import write_backbone  # noqa: E402
from kev.suite import digest, write_json  # noqa: E402
from scripts.interpolate_checkpoint import load_lora, shard_layout  # noqa: E402

NOT_COPIED = {"head.pt", "adapter_config.json", "adapter_model.safetensors", "training_config.json", "training_metrics.json",
              "provenance.json", "result.json", "train.log", "README.md", ".gitattributes"}   # rewritten, or records of the LoRA run
FORMULA = "round_to_weights_dtype(fp32(W) + fp32(peft get_delta_weight)) for adapted weights, W unchanged elsewhere"


def weights_sha256(ck, workers=8):
    """Checkpoint.weights_sha256 of a full-weight checkpoint (sha256 over 'name:sha256' of each shard), shards hashed in
    parallel threads (hashlib releases the GIL): a 51 GB checkpoint in a fraction of the serial time."""
    shards = ck.shards()
    with ThreadPoolExecutor(workers) as pool:
        digests = list(pool.map(digest, shards))
    return hashlib.sha256("".join(f"{p.name}:{d}\n" for p, d in zip(shards, digests)).encode()).hexdigest()


WRITE_DTYPES = {"bf16": torch.bfloat16}   # --weights_dtype: what an fp32-trained checkpoint may be written as (module docstring)


def merge(lora, out, like=None, log=print, weights_dtype=None):
    """Write <out>/checkpoint (a full-weight checkpoint) from the LoRA checkpoint `lora` (directory or Hub id[@rev]). -> the report.
    weights_dtype: None = the checkpoint's own weights dtype; "bf16" = merged in the checkpoint's dtype, then rounded once to bf16."""
    if weights_dtype not in (None, *WRITE_DTYPES): raise ValueError(f"--weights_dtype must be one of {sorted(WRITE_DTYPES)}; got {weights_dtype!r}")
    out = Path(out)
    target, partial = out / "checkpoint", out / "checkpoint.partial"
    if target.exists(): raise FileExistsError(f"refusing to overwrite {target}")
    phases, t0 = {}, time.time()

    def phase(name, started):
        phases[name] = round(time.time() - started, 1); log(f"{name}: {phases[name]} s")

    ck = Checkpoint(lora)
    if ck.full: raise ValueError(f"{lora} is a full-weight checkpoint already")
    ref = None
    if like:
        ref = Checkpoint(like)
        if not ref.full: raise ValueError(f"--like {like} is not a full-weight checkpoint")
        if (ref.meta.base, ref.meta.base_revision) != (ck.meta.base, ck.meta.base_revision):
            raise ValueError(f"--like {like} is a checkpoint of {ref.meta.base}@{ref.meta.base_revision}, not {ck.meta.base}@{ck.meta.base_revision}")
    source = {"path": str(lora), "resolved": ck.path, "weights_sha256": ck.weights_sha256(), "head_sha256": digest(ck.file("head.pt"))}
    log(f"loading {ck.meta.base}@{ck.meta.base_revision} ({ck.meta.weights_dtype}) with the adapter of {lora}")
    started = time.time(); model, plain, layers = load_lora(ck, "--lora"); phase("load_base_and_adapter", started)
    started = time.time()
    with torch.no_grad():
        for name, module in layers.items():
            w = plain[name]
            w.copy_((w.float() + module.get_delta_weight("default").float()).to(w.dtype))   # the backbone's own parameter (shared storage)
    lm = model.unload()   # peft: the LoRA layers replaced by their base layers, nothing merged again (they hold W + delta already)
    adapted = len(layers)
    del plain, layers   # they share the backbone's storage: dropped, the cast below frees each fp32 tensor as it goes
    trained = ck.meta.weights_dtype
    if weights_dtype and weights_dtype != trained:
        lm.to(WRITE_DTYPES[weights_dtype])   # the one rounding of the merged fp32 values (in place, tensor by tensor)
    lm.config.use_cache = False   # as kev.train sets it before a full-weight save: config.json is a full-weight run's (scoring passes use_cache itself)
    phase("merge", started)
    if partial.exists(): shutil.rmtree(partial)   # an earlier attempt that died before its rename: never a checkpoint
    partial.mkdir(parents=True)
    started = time.time(); write_backbone(lm, partial, None); phase("write_backbone", started)
    del model, lm
    for p in Path(ck.path).iterdir():
        if p.is_file() and p.name not in NOT_COPIED: shutil.copy2(p, partial / p.name)
    meta = ck.meta
    meta.lora, meta.weights, meta.weights_dtype = 0, "full", weights_dtype or trained
    meta.extra["merged_lora"] = {"source": source, "base": f"{meta.base}@{meta.base_revision}", "adapted_tensors": adapted, "formula": FORMULA,
                                 **({"weights_dtype": {"trained": trained, "written": weights_dtype}} if weights_dtype and weights_dtype != trained else {})}
    write_meta(partial, meta)
    merged = Checkpoint(partial)
    if not merged.full: raise RuntimeError(f"{partial} does not load as a full-weight checkpoint")
    if ref is not None:
        started = time.time()
        (_, mine), (_, theirs) = shard_layout(merged), shard_layout(ref)
        if mine != theirs:
            only_mine, only_theirs = sorted(set(mine) - set(theirs)), sorted(set(theirs) - set(mine))
            wrong = sorted(n for n in set(mine) & set(theirs) if mine[n] != theirs[n])
            raise ValueError(f"the merged backbone does not match --like {like}: {len(only_mine)} tensors only here (e.g. {only_mine[:2]}), "
                             f"{len(only_theirs)} only there (e.g. {only_theirs[:2]}), {len(wrong)} with another shape (e.g. {wrong[:2]})")
        phase("check_like", started)
    started = time.time(); sha = weights_sha256(merged); phase("hash_output", started)
    partial.rename(target)
    report = {"checkpoint": str(target), "weights_sha256": sha, "head_sha256": digest(target / "head.pt"), "tensors": len(shard_layout(Checkpoint(target))[0]),
              "merged_lora": meta.extra["merged_lora"], "temperature": meta.temperature, "head_dim": meta.head_dim,
              "like": str(like) if like else None, "phases": phases, "seconds": round(time.time() - t0, 1)}
    write_json(out / "merge.json", report)
    log(f"-> {target}: weights {sha[:12]}, {report['tensors']} tensors, {report['seconds']} s")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lora", required=True, help="LoRA checkpoint directory or Hub id[@rev] (pin the revision)")
    ap.add_argument("--out", required=True, help="output directory; the checkpoint goes to <out>/checkpoint, the report to <out>/merge.json")
    ap.add_argument("--like", help="a full-weight checkpoint of the same base whose tensor names and shapes the result must have")
    ap.add_argument("--weights_dtype", choices=sorted(WRITE_DTYPES), help="write an fp32-trained checkpoint in this dtype (one rounding after the fp32 merge); default: as trained")
    a = ap.parse_args()
    merge(a.lora, a.out, a.like, weights_dtype=a.weights_dtype)


if __name__ == "__main__":
    main()
