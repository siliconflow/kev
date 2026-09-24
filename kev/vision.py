"""Opt-in image channel for the decision model (KEV_VISION=1). Channel open, untrained readout.

Qwen3.5-Base checkpoints ship the full vision tower in the same safetensors (4B-Base: 297
`model.visual.*` tensors; 0.8B-Base: 153), but kev.evaluate.load's path (AutoModelForCausalLM
-> Qwen3_5ForCausalLM -> .model = Qwen3_5TextModel) drops them: the tower is never
instantiated. That is exactly the state the LoRA and pointer head were trained in (training
data is text-only, every generation), so re-attaching the untrained tower opens the image
channel without touching the text path. An open channel says nothing about readout quality -
keep "channel open, untrained readout" framing in reports.

Gating: nothing here runs unless serve.py calls attach() under KEV_VISION=1. With the gate off
the module is never imported by the serving path; encode/forward/train text paths are
unchanged (tests/test_vision.py pins text-path equality with and without the gate).

Packing for a record with images (per question branch, via rows_of; image tokens are part of
the state segment, so every question row repeats them and every question attends to them):

    <|fim_prefix|> state text [ <|vision_start|> <image_pad> x N <|vision_end|> ] per image
    ... question branch rows unchanged

<image_pad> is Qwen's image token (id from config.image_token_id). Preprocessing is the
OFFICIAL AutoImageProcessor (Qwen2VL-style patchify + merge) - hand-rolled patch grids are how
this file went wrong twice; the tower's merged-row count is asserted against its own output,
never trusted from the grid alone. The tower runs once per request; its merged rows (patch-merger
output) replace the placeholder token embeddings in the inputs_embeds stream. On both tested
Qwen3.5 bases the merger's out_hidden equals the text hidden (0.8B: 1024==1024, 4B: 2560==2560),
so the splice needs no projection layer; embed() asserts this so a foreign base fails loudly
instead of silently corrupting the stream. Positions: image rows continue the state's
position sequence, one position per merged row (M-RoPE's 3D positions are not reproduced -
this is a splice channel, not a faithfulness claim).

The forward is the row form (state + branch as one causal row per question): equivalent to
the packed block-causal form on ANY backbone (see model.rows_of), so one implementation
serves hybrid and attention-only bases alike. The serving prefix cache does NOT apply: a
cached text-state KV was built without the spliced embeds, so image requests always pay the
full row pass.

Image refs follow the workspace convention: data URLs (preferred) or https URLs fetched
server-side. JPEG/PNG/WebP only, <= 4 images, 5 MiB decoded each, 8 MiB fetch cap.
"""
import base64
import io
import os
import re
import urllib.request

import torch
import torch.nn.functional as F

IMAGE_MIMES = ("image/jpeg", "image/png", "image/webp")
IMAGE_MAX_BYTES = 5 * 1024 * 1024    # decoded, per image
IMAGE_FETCH_LIMIT = 8 * 1024 * 1024  # https fetch cap
MAX_IMAGES = 4

_DATA_URL = re.compile(r"^data:([\w./+-]+);base64,(.*)$", re.S)


def decode_image(ref):
    """data URL | https URL -> PIL.Image (in-memory). Raises ValueError on any refusal."""
    from PIL import Image

    m = _DATA_URL.match(ref.strip())
    if m:
        mime, b64 = m.group(1), m.group(2)
        if mime not in IMAGE_MIMES:
            raise ValueError(f"unsupported image mime {mime} (jpeg/png/webp)")
        raw = base64.b64decode(b64, validate=True)
        if len(raw) > IMAGE_MAX_BYTES:
            raise ValueError("image exceeds 5 MiB decoded")
        im = Image.open(io.BytesIO(raw))
        im.load()
        return im
    if ref.startswith("https://"):
        req = urllib.request.Request(ref, headers={"user-agent": "kev-vision/1"})
        with urllib.request.urlopen(req, timeout=10) as r:  # type: ignore[attr-defined]
            raw = r.read(IMAGE_FETCH_LIMIT + 1)
        if len(raw) > IMAGE_FETCH_LIMIT:
            raise ValueError("image fetch exceeded 8 MiB cap")
        if len(raw) > IMAGE_MAX_BYTES:
            raise ValueError("image exceeds 5 MiB decoded")
        im = Image.open(io.BytesIO(raw))
        im.load()
        return im
    raise ValueError("images must be data URLs or https URLs")


# ---------------------------------------------------------------------------
# attach: build the tower once at serve startup
# ---------------------------------------------------------------------------

def attach(model, tok, base, revision=None):
    """Build the tower for a loaded DecisionModel from the base checkpoint's local snapshot.

    `base` is the base id exactly as head.pt's meta records it (the resolved snapshot may be
    a ModelScope path); the same safetensors the main load read are therefore already on
    disk. Returns a VisionHook, or None when the base has no vision tower (older text-only
    generations: the channel stays closed - serve then answers image requests with 422).
    """
    from transformers import AutoConfig

    snapshot = _resolve_snapshot(base, revision)
    cfg = AutoConfig.from_pretrained(snapshot)
    vision_cfg = getattr(cfg, "vision_config", None)
    if vision_cfg is None:
        print(f"[vision] {base}: no vision tower in this base; channel stays closed")
        return None
    tower = _load_tower(vision_cfg, snapshot)
    ref = next(model.lm.parameters())
    tower.to(ref.device).to(ref.dtype)
    tower.eval()
    for p in tower.parameters():
        p.requires_grad_(False)
    try:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(snapshot)   # image-only: skips the
        # Qwen3VL video sub-processor (extra deps, ffmpeg for videos) - this route takes images
    except Exception as e:  # processor stack unavailable: refuse rather than hand-roll patches
        raise RuntimeError(f"vision needs AutoImageProcessor for {snapshot}: {e!r}")
    hook = VisionHook(model, tok, tower, processor,
                      cfg.image_token_id, cfg.vision_start_token_id, cfg.vision_end_token_id)
    print(f"[vision] tower attached ({type(tower).__name__}), "
          f"{sum(p.numel() for p in tower.parameters()) / 1e6:.0f}M params, untrained - "
          f"channel open, untrained readout")
    return hook


def _resolve_snapshot(base, revision):
    """Local snapshot dir for the base (or the path itself when already local)."""
    import glob

    if os.path.isdir(base):
        return base
    # evaluate.load swaps the HF id for a local ModelScope snapshot under
    # KEV_BASE_HUB=modelscope; the MS mirror carries the same weights - reuse that copy.
    if os.environ.get("KEV_BASE_HUB") == "modelscope":
        from .evaluate import _ms_snapshot

        return _ms_snapshot(base)
    root = os.path.expanduser("~/.cache/huggingface/hub")
    name = f"models--{base.replace('/', '--')}"
    pats = [f"{root}/{name}/snapshots/*"]
    if revision:
        pats.insert(0, f"{root}/{name}/snapshots/{revision}")
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"no local snapshot for {base}; run the main model load first")


def _load_tower(vision_cfg, snapshot):
    """Instantiate the standalone Qwen3.5 tower and load model.visual.* from the snapshot."""
    import glob as _glob

    from safetensors.torch import load_file as _load_file
    from transformers.models.qwen3_5 import modeling_qwen3_5

    tower = modeling_qwen3_5.Qwen3_5VisionModel._from_config(vision_cfg)
    state = {}
    for path in sorted(_glob.glob(os.path.join(_glob.escape(snapshot), "*.safetensors"))):
        for k, v in _load_file(path).items():
            m = re.match(r"^model\.visual\.([0-9A-Za-z_.]+)$", k)
            if m:
                state[m.group(1)] = v
    if not state:
        raise RuntimeError(f"no model.visual.* tensors under {snapshot}")
    missing, unexpected = tower.load_state_dict(state, strict=False)
    missing = [k for k in missing if "inv_freq" not in k]   # lazily-built non-persistent buffer
    if missing:
        raise RuntimeError(f"vision tower incomplete: missing {missing[:5]} ({len(missing)})")
    return tower


# ---------------------------------------------------------------------------
# the hook: official preprocess, one tower run, splice into the row form
# ---------------------------------------------------------------------------

class VisionHook:
    """All state needed to answer image-bearing requests. No per-request state is kept."""

    def __init__(self, model, tok, tower, processor, image_token, vision_start, vision_end):
        self.model, self.tok, self.tower, self.processor = model, tok, tower, processor
        self.image_token = int(image_token)
        self.vision_start, self.vision_end = int(vision_start), int(vision_end)

    @torch.no_grad()
    def embed(self, images):
        """PIL images -> (merged rows [N, out_hidden] on the tower's device/dtype,
        rows-per-image list). The official processor supplies pixel_values and
        image_grid_thw; N comes from grid_thw // merge**2 and is asserted against the
        tower's own output - never trusted alone."""
        kept = images[:MAX_IMAGES]
        out = self.processor(images=kept, return_tensors="pt")
        ref = next(self.tower.parameters())
        pixel_values = out["pixel_values"].to(ref.device, ref.dtype)
        grid_thw = out["image_grid_thw"].to(ref.device)   # the tower derives position/interp
        # indices from it - they must share the tower's device (MPS embedding fails otherwise)
        m = getattr(self.tower.config, "spatial_merge_size", 2) or 2
        n_per = [t * h * w // (m * m) for t, h, w in grid_thw.tolist()]
        # transformers names the first arg `hidden_states` (Qwen2-VL tradition) but it takes
        # pixel_values - patch_embed() is applied inside the tower. Positional, to stay
        # robust against the name.
        res = self.tower(pixel_values, grid_thw=grid_thw, return_dict=True)
        rows = res.pooler_output
        if rows.shape[0] != sum(n_per):
            raise RuntimeError(f"tower produced {rows.shape[0]} rows, "
                                f"grid_thw says {sum(n_per)} - refusing to guess")
        text_hidden = getattr(self.model.lm.config, "hidden_size", None)
        if text_hidden is not None and rows.shape[1] != text_hidden:
            raise RuntimeError(f"vision out_hidden {rows.shape[1]} != text hidden "
                                f"{text_hidden}: this base cannot be spliced without a "
                                f"projection layer (not implemented)")
        return rows.float(), n_per

    def _image_segment(self, n_per_image):
        """Token ids appended to the state text: per image, vision_start + image_pad x N + vision_end."""
        seg = []
        for n in n_per_image:
            seg += [self.vision_start] + [self.image_token] * n + [self.vision_end]
        return seg

    def _inject(self, enc, n_per_image):
        """Insert the image segment at the END of the state and shift every branch index.

        Works on encode()'s parallel arrays: image tokens join seg 0 (state), positions
        continue the state sequence, decide/opt (absolute indices) and every branch
        position shift by the injection length. rows_of() then splits the padded state
        like any other record.
        """
        from .model import OPT_NONE

        Ls = enc["seg"].count(0)
        seg_img = self._image_segment(n_per_image)
        Ni = len(seg_img)
        out = dict(enc)
        out["ids"] = enc["ids"][:Ls] + seg_img + enc["ids"][Ls:]
        out["seg"] = enc["seg"][:Ls] + [0] * Ni + enc["seg"][Ls:]
        out["opt"] = enc["opt"][:Ls] + [OPT_NONE] * Ni + enc["opt"][Ls:]
        out["pos"] = enc["pos"][:Ls] + list(range(Ls, Ls + Ni)) + [p + Ni for p in enc["pos"][Ls:]]
        out["decide_idx"] = [d + Ni for d in enc["decide_idx"]]
        out["opt_idx"] = [[o + Ni for o in q] for q in enc["opt_idx"]]
        return out

    @torch.no_grad()
    def probs_with_images(self, rec, images, max_state=384, max_branch=1024):
        """probs() for a text record + PIL images: row form with the tower rows spliced
        into the placeholder embeddings. One implementation for every backbone (rows form
        is exact on any architecture, model.rows_of); the serving prefix cache does not
        apply (image requests always pay the full row pass)."""
        from .model import rows_of

        if self.model.option_isolation:
            raise ValueError("image path uses the row form; unavailable with option_isolation")
        rows, n_per = self.embed(images)
        enc = self.model.encode(self.tok, rec, max_state=max_state, max_branch=max_branch)
        enc = self._inject(enc, n_per)
        S, Sp, qrows = rows_of(enc)
        B = len(qrows)
        dev = self.model.device
        lm_dtype = next(self.model.lm.parameters()).dtype
        tower = rows.to(dev, lm_dtype)
        r_ids = [S + r["ids"] for r in qrows]
        r_pos = [Sp + r["pos"] for r in qrows]
        L = max(len(x) for x in r_ids)
        if str(dev) == "mps" and not self.model.training:
            L = -(-L // self.model.SHAPE_BUCKET) * self.model.SHAPE_BUCKET
        ids = torch.full((B, L), self.model.pad_id, device=dev)
        pos = torch.zeros((B, L), dtype=torch.long, device=dev)
        att = torch.zeros((B, L), dtype=torch.long, device=dev)
        for i in range(B):
            ids[i, : len(r_ids[i])] = torch.tensor(r_ids[i], device=dev)
            pos[i, : len(r_pos[i])] = torch.tensor(r_pos[i], device=dev)
            att[i, : len(r_ids[i])] = 1
        emb = self.model.lm.get_input_embeddings()(ids)
        img = ids == self.image_token
        if int(img.sum()) != B * rows.shape[0]:
            raise RuntimeError("image placeholder count mismatch - refusing to splice")
        emb[img] = tower.repeat(B, 1)   # row-major: each row's placeholders, in order
        h = self.model.lm(inputs_embeds=emb, position_ids=pos,
                          attention_mask=att).last_hidden_state.float()
        out = []
        for i, r in enumerate(qrows):
            d = len(S) + r["decide"]
            oi = torch.tensor([len(S) + o for o in r["opts"]], device=dev)
            out.append(F.softmax(self.model.head(h[i, d], h[i, oi]), -1).cpu())
        return out, {"tokens": len(enc["ids"]), "state_tokens": enc["seg"].count(0)}
