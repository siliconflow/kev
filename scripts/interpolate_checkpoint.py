"""WiSE-FT for full-weight checkpoints: a text backbone alpha * sft + (1 - alpha) * other, where other is the base the SFT
was trained from (round 20) or another checkpoint of the same base (`--toward`, round 23), with the SFT pointer head or
the two heads blended the same way (`--blend_head`).

A full-weight SFT checkpoint (kev.train --full_ft 1) is a save_pretrained backbone plus head.pt. Interpolating its
weights with the base it was trained from trades what fine-tuning learned against what the base knew, without training
(Wortsman et al., "Robust fine-tuning of zero-shot models", 2022). Round 20 reads six such checkpoints (PLAN.md). Round
23 blends toward Kev-27B instead, a LoRA checkpoint of the same base that is strong where the SFT drifted.

    uv run python scripts/interpolate_checkpoint.py --sft runs/r19-27b-lr2e6/00-trial-0/checkpoint \\
        --alphas 0.85,0.70,0.50 --out runs/r20-wise/27b-a-w{w}      # -> runs/r20-wise/27b-a-w85/checkpoint, ...
    uv run python scripts/interpolate_checkpoint.py --sft <sft checkpoint> --toward jaredpalmer/kev-27b@<sha> \\
        --blend_head --alphas 0.85,0.70,0.50 --out runs/r23-wise/27b-kh-w{w}
    uv run modal run modal_app.py::interpolate --sft /runs/r19-27b-lr2e6/00-trial-0/checkpoint --prefix 27b-a   # on Modal

The other endpoint:
- the base (default): built exactly as training builds it (kev.model.DecisionModel on meta.base @ meta.base_revision in the
  run's weights dtype), so its state_dict carries the names save_backbone wrote;
- `--toward` a full-weight checkpoint: its saved backbone, streamed tensor by tensor from its shards;
- `--toward` a LoRA checkpoint: its merged backbone in fp32, W + delta, where W is the base built as its loader builds it
  (the checkpoint's weights dtype, bf16 values upcast exactly) and delta is peft's own get_delta_weight of each adapted
  layer, the fp32 value peft's merge adds (merge_and_unload on an fp32 base holds exactly W + delta; the bf16 serving
  path holds round(W + delta)). Nothing is rounded before the blend. Refused: DoRA and other LoRA variants, LoRA biases,
  modules_to_save, trained token embeddings (anything a merge would change outside W + delta).
Either checkpoint must name the SFT's base and base revision exactly; every SFT tensor must match one tensor of the other
endpoint in name and shape, and every tensor of the other endpoint must be covered, or nothing is written.

The SFT side streams tensor by tensor from its safetensors shards, and each output shard is written as soon as it is
complete, with the SFT shard's file name, tensor names, dtype and metadata, and the SFT index copied, so the result loads
through kev.checkpoint's full-weight rule unchanged. Arithmetic is fp32 (bf16 sides upcast exactly), rounded once to the
SFT tensor's dtype. head.pt is the SFT's (same meta and temperature) with `interpolation` added to its meta: {alpha, sft:
{path, weights_sha256}, base: "<repo>@<revision>"}, plus with `--toward`: toward {path, resolved, kind, weights_sha256,
head_sha256, merge?} and head {kind: "sft" | "blend", sft: {head_sha256, temperature}, toward: {head_sha256, temperature}}.
`--blend_head` (needs `--toward`) blends the pointer heads the same way (fp32; alpha 1 = the SFT's head, 0 = the other's);
both heads must have the same tensors, shapes and dtypes and the same head_dim, option_isolation and special_embeddings.
Config and tokenizer files are copied; the SFT run's training_config.json / training_metrics.json are not (they describe
a training this checkpoint did not have). Each checkpoint is written to <out>/checkpoint.partial and renamed to
<out>/checkpoint when complete, with a report in <out>/interpolation.json.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.checkpoint import Checkpoint, read_meta, write_meta  # noqa: E402
from kev.model import DecisionModel, load_tokenizer  # noqa: E402
from kev.suite import digest, write_json  # noqa: E402

CHUNK = 1 << 24                                                      # elements per fp32 step: bounds the temporaries (~200 MB)
NOT_COPIED = ("head.pt", "training_config.json", "training_metrics.json")   # head.pt is rewritten; the rest describe the SFT run
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
HEAD_FIELDS = ("head_dim", "option_isolation", "special_embeddings")        # what a blended pair of heads must share


def mix(sft, other, alpha):
    """alpha * sft + (1 - alpha) * other computed in fp32 and rounded once to sft's dtype."""
    if sft.shape != other.shape: raise ValueError(f"shape {tuple(sft.shape)} vs {tuple(other.shape)}")
    out = torch.empty_like(sft)
    s, b, o = sft.reshape(-1), other.reshape(-1), out.view(-1)
    for i in range(0, s.numel(), CHUNK):
        o[i:i + CHUNK] = (alpha * s[i:i + CHUNK].float() + (1 - alpha) * b[i:i + CHUNK].float()).to(sft.dtype)
    return out


class Endpoint(NamedTuple):
    """The other side of the blend: {name: shape}, name -> tensor (any dtype; fp32 for a merged LoRA), and provenance."""
    shapes: dict
    tensor: Callable
    provenance: dict


def build_backbone(meta):
    """The base backbone as kev.train (and kev.checkpoint's LoRA loader) builds it: DecisionModel in the run's weights dtype."""
    tok = load_tokenizer(meta.base, revision=meta.base_revision)
    return DecisionModel(meta.base, tok, "cpu", revision=meta.base_revision, head_dim=meta.head_dim, dtype=DTYPES[meta.weights_dtype])


def base_backbone(meta):
    """{name: tensor} of the base backbone as kev.train builds it before training."""
    return build_backbone(meta).lm.state_dict()


def base_endpoint(meta):
    backbone = base_backbone(meta)
    return Endpoint({n: tuple(t.shape) for n, t in backbone.items()}, backbone.__getitem__, {})


def shard_layout(ck):
    """({name: shard path}, {name: shape}) of a full-weight checkpoint, from the safetensors headers (nothing loaded)."""
    layout, shapes = {}, {}
    for shard in ck.shards():
        with safe_open(shard, "pt") as f:
            for name in f.keys():
                if name in layout: raise ValueError(f"{ck.path}: {name} is in two shards ({layout[name].name}, {shard.name})")
                layout[name], shapes[name] = shard, tuple(f.get_slice(name).get_shape())
    return layout, shapes


def full_endpoint(ck):
    layout, shapes = shard_layout(ck)
    handles = {}

    def tensor(name):
        shard = layout[name]
        if shard not in handles: handles[shard] = safe_open(shard, "pt")
        return handles[shard].get_tensor(name)
    return Endpoint(shapes, tensor, {"kind": "full"})


def load_lora(ck, what="--toward"):
    """(peft model, {name: base tensor}, {weight name: LoraLayer}) of a LoRA checkpoint on its base as its loader builds it
    (the checkpoint's weights dtype). The base tensors share storage with the backbone the peft model wraps. Refuses
    anything a merge would change outside W + delta."""
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    config = ck.adapter_config()
    unsupported = [k for k in ("use_dora", "lora_bias", "modules_to_save", "trainable_token_indices") if config.get(k)]
    if unsupported or ck.meta.special_embeddings:
        raise ValueError(f"{what} {ck.path}: {unsupported or ['special_embeddings']} change weights outside W + delta; not supported")
    lm = build_backbone(ck.meta).lm
    plain = lm.state_dict()                          # the base's tensors under the backbone's own names (shared storage)
    model = PeftModel.from_pretrained(lm, ck.path, torch_device="cpu")
    layers = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        weight = name.removeprefix("base_model.model.") + ".weight"
        if weight not in plain: raise ValueError(f"{what} {ck.path}: adapted layer {name} has no base tensor {weight}")
        if module.merged or list(module.lora_variant): raise ValueError(f"{what} {ck.path}: {name} is merged or a LoRA variant")
        if list(module.active_adapters) != ["default"]: raise ValueError(f"{what} {ck.path}: {name} has adapters {module.active_adapters}")
        layers[weight] = module
    if not layers: raise ValueError(f"{what} {ck.path}: no LoRA layers were loaded")
    return model, plain, layers


def lora_endpoint(ck):
    """The LoRA checkpoint's merged backbone in fp32: W (the base as its loader builds it) + peft's get_delta_weight."""
    _, plain, layers = load_lora(ck)

    def tensor(name):
        if name not in layers: return plain[name]
        return plain[name].float() + layers[name].get_delta_weight("default").float()
    return Endpoint({n: tuple(t.shape) for n, t in plain.items()}, tensor,
                    {"kind": "lora", "merge": "fp32: base (as its loader builds it) + peft get_delta_weight, not rounded before the blend",
                     "adapted_tensors": len(layers)})


def check_layout(ck, shapes, what="its base"):
    """{name: shard file} of the SFT checkpoint after checking that its tensors are the other endpoint's ({name: shape}),
    name for name and shape for shape (read from the safetensors headers; nothing is loaded)."""
    layout, own = shard_layout(ck)
    only_sft, only_other = sorted(set(layout) - set(shapes)), sorted(set(shapes) - set(layout))
    wrong = sorted(n for n in set(layout) & set(shapes) if own[n] != tuple(shapes[n]))
    if only_sft or only_other or wrong:
        raise ValueError(f"{ck.path} does not match {what}: {len(only_sft)} tensors only in the SFT save (e.g. {only_sft[:2]}), "
                         f"{len(only_other)} only in {what} (e.g. {only_other[:2]}), {len(wrong)} with another shape (e.g. {wrong[:2]})")
    return {n: p.name for n, p in layout.items()}


def check_heads(sft, other, where):
    """Refuse to blend two pointer heads that are not the same architecture."""
    for field in HEAD_FIELDS:
        if getattr(sft, field) != getattr(other, field): raise ValueError(f"--blend_head: {field} is {getattr(sft, field)!r} in the SFT and {getattr(other, field)!r} in {where}")
    if set(sft.head) != set(other.head): raise ValueError(f"--blend_head: head tensors {sorted(sft.head)} vs {sorted(other.head)} in {where}")
    for name, t in sft.head.items():
        if tuple(t.shape) != tuple(other.head[name].shape) or t.dtype != other.head[name].dtype:
            raise ValueError(f"--blend_head: head.{name} is {tuple(t.shape)} {t.dtype} in the SFT and {tuple(other.head[name].shape)} {other.head[name].dtype} in {where}")


def write_checkpoint(ck, other, alpha, out, provenance, head=None):
    """One interpolated checkpoint at out (via out.partial); head: the pointer head to write (default the SFT's). -> tensor count."""
    partial = out.with_name(out.name + ".partial")
    if partial.exists(): shutil.rmtree(partial)   # an earlier attempt that died before its rename: never a checkpoint
    partial.mkdir(parents=True)
    count = 0
    for shard in ck.shards():
        with safe_open(shard, "pt") as f:
            tensors = {name: mix(f.get_tensor(name), other(name), alpha) for name in f.keys()}
            save_file(tensors, partial / shard.name, metadata=f.metadata())
        count += len(tensors)
        del tensors
    for p in Path(ck.path).iterdir():
        if p.is_file() and p.name not in NOT_COPIED and not (p.name.startswith("model") and p.name.endswith(".safetensors")):
            shutil.copy2(p, partial / p.name)
    meta = read_meta(ck.path)
    if head is not None: meta.head = head
    meta.extra["interpolation"] = {"alpha": alpha, **provenance}
    write_meta(partial, meta)
    partial.rename(out)
    return count


def interpolate(sft, alphas, outs, base=None, revision=None, on_done=None, log=print, toward=None, blend_head=False):
    """Write one checkpoint per alpha (weight on the SFT backbone) to outs[i] (a .../checkpoint directory; its parent gets
    interpolation.json). The other endpoint is the SFT's base, or `toward` (a checkpoint directory or Hub id[@rev] of the
    same base and revision: full weights, or a LoRA adapter merged in fp32). base / revision, when given, must be the SFT
    run's own. blend_head (needs toward) blends the pointer heads too. on_done(report) runs after each checkpoint
    (modal_app.py commits the runs volume there). -> the reports."""
    if len(alphas) != len(outs): raise ValueError("one output per alpha")
    if any(not 0 <= a <= 1 for a in alphas): raise ValueError(f"alphas must be in [0, 1]: {alphas}")
    if blend_head and not toward: raise ValueError("--blend_head blends the SFT head with the --toward checkpoint's; the base has no head")
    outs = [Path(o) for o in outs]
    if taken := [str(o) for o in outs if o.exists()]: raise FileExistsError(f"refusing to overwrite {taken}")
    ck = Checkpoint(sft)
    if not ck.full: raise ValueError(f"{sft} is a LoRA adapter; interpolate a full-weight checkpoint (kev.train --full_ft 1)")
    meta = ck.meta
    if (base or meta.base) != meta.base or (revision or meta.base_revision) != meta.base_revision:
        raise ValueError(f"{sft} was trained from {meta.base}@{meta.base_revision}, not {base}@{revision}")
    other_ck = Checkpoint(toward) if toward else None
    if other_ck and (other_ck.meta.base, other_ck.meta.base_revision) != (meta.base, meta.base_revision):
        raise ValueError(f"--toward {toward} is a checkpoint of {other_ck.meta.base}@{other_ck.meta.base_revision}, the SFT of "
                         f"{meta.base}@{meta.base_revision}; both must share one base and revision")
    if blend_head: check_heads(meta, other_ck.meta, toward)
    t0 = time.time()
    log(f"hashing {sft}"); provenance = {"sft": {"path": str(sft), "weights_sha256": ck.weights_sha256()}, "base": f"{meta.base}@{meta.base_revision}"}
    head = None
    if other_ck:
        kind = "full" if other_ck.full else "lora"
        log(f"loading {toward} ({kind}{', merged in fp32' if kind == 'lora' else ''})")
        endpoint = full_endpoint(other_ck) if other_ck.full else lora_endpoint(other_ck)
        provenance["toward"] = {"path": str(toward), "resolved": other_ck.path, "weights_sha256": other_ck.weights_sha256(),
                                "head_sha256": digest(other_ck.file("head.pt")), **endpoint.provenance}
        heads = {"sft": {"head_sha256": digest(ck.file("head.pt")), "temperature": meta.temperature},
                 "toward": {"head_sha256": provenance["toward"]["head_sha256"], "temperature": other_ck.meta.temperature}}
        what = f"--toward {toward}"
    else:
        log(f"loading {provenance['base']} as training builds it"); endpoint = base_endpoint(meta)
        what = "its base"
    check_layout(ck, endpoint.shapes, what)
    reports = []
    for alpha, out in zip(alphas, outs):
        started = time.time()
        extra = {}
        if other_ck:
            head = {k: mix(t, other_ck.meta.head[k], alpha) for k, t in meta.head.items()} if blend_head else None
            extra = {"head": {"kind": "blend" if blend_head else "sft", **heads}}
        tensors = write_checkpoint(ck, endpoint.tensor, alpha, out, {**provenance, **extra}, head)
        report = {"alpha": alpha, **provenance, **extra, "checkpoint": str(out), "tensors": tensors, "weights_sha256": Checkpoint(out).weights_sha256(),
                  "seconds": round(time.time() - started, 1),
                  "formula": f"fp32(alpha) * fp32(sft) + fp32(1 - alpha) * fp32({'toward' if other_ck else 'base'}), rounded once to the SFT dtype"}
        write_json(out.parent / "interpolation.json", report)
        log(f"alpha {alpha}: {tensors} tensors -> {out} ({report['seconds']} s)")
        reports.append(report)
        if on_done: on_done(report)
    log(f"done in {time.time() - t0:.0f} s")
    return reports


def weight_label(alpha):
    """0.85 -> "85": the {w} of an output template (round 20 names checkpoints 27b-a-w85)."""
    return f"{round(alpha * 100):02d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sft", required=True, help="full-weight checkpoint directory (config.json + model*.safetensors + head.pt)")
    ap.add_argument("--alphas", required=True, help="comma-separated weights on the SFT backbone, e.g. 0.85,0.70,0.50")
    ap.add_argument("--out", required=True, help="output directory template with {w} (alpha x 100); the checkpoint goes to <out>/checkpoint")
    ap.add_argument("--base", help="refuse unless the SFT run's base is this (default: head.pt's)")
    ap.add_argument("--revision", help="refuse unless the SFT run's base revision is this (default: head.pt's)")
    ap.add_argument("--toward", help="blend toward this checkpoint (directory or Hub id[@rev]; full weights, or LoRA merged in fp32) instead of the base")
    ap.add_argument("--blend_head", action="store_true", help="blend the pointer heads too (needs --toward); default: keep the SFT head")
    a = ap.parse_args()
    if "{w}" not in a.out: ap.error("--out needs {w}")
    alphas = [float(x) for x in a.alphas.split(",")]
    interpolate(a.sft, alphas, [Path(a.out.format(w=weight_label(x))) / "checkpoint" for x in alphas], a.base, a.revision,
                toward=a.toward, blend_head=a.blend_head)


if __name__ == "__main__":
    main()
