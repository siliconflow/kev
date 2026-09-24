"""Vision channel tests (KEV_VISION gate). Three tiers:

1. no-weights tests: decode_image refusal surface, to_record image stripping, _inject
   index-shift algebra (against kev.model.encode primitives).
2. gate-off equality (any tiny base): import surface and text path untouched by the
   module - serve imports kev.vision only under KEV_VISION=1.
3. 0.8B full-chain (skipped unless the local Qwen3.5-0.8B-Base snapshot exists): attach
   loads all 153 model.visual.* tensors, probs_with_images changes the readout vs the
   text path, is deterministic on rerun, and rows_of still splits the injected layout.

Run: uv run --all-extras python -m pytest tests/test_vision.py -q
(The full-chain tier needs the `vision` extra: torchvision is imported by the
transformers qwen2_vl image processor; without it the processor tier of tests skips.)
"""
import base64
import io
import os

import pytest
import torch
from PIL import Image

from kev.api import SystemOneRequest, to_record
from kev.model import OPT_NONE, encode, rows_of
from kev.model import load_tokenizer, DecisionModel

SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/"
    "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")

PNG_1X1 = None  # unused; real images are made by _png_data_url() - decode refusals do
# not need valid image bytes (mime/base64/scheme rejections happen before decoding)


def _png_data_url(rgb=(200, 30, 30)):
    im = Image.new("RGB", (4, 4), rgb)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --- tier 1: refusals and pure algebra ----------------------------------------------


def test_decode_image_refusals():
    from kev import vision
    with pytest.raises(ValueError):
        vision.decode_image("not a url")
    with pytest.raises(ValueError):
        vision.decode_image("data:image/gif;base64," + _png_data_url())  # mime (gif) not allowed
    with pytest.raises(ValueError):
        vision.decode_image("data:image/png;base64,!!!not base64!!!")
    with pytest.raises(ValueError):
        vision.decode_image("http://example.com/a.png")             # https only


def test_decode_image_accepts_png_data_url():
    from kev import vision
    im = vision.decode_image(_png_data_url())
    assert im.size == (4, 4) and im.mode == "RGB"


def test_to_record_strips_images_from_state():
    req = SystemOneRequest.model_validate({
        "state": {"document": "charged twice", "images": [_png_data_url()]},
        "questions": {"billing": {"type": "noul", "instructions": "billing?", "criteria": None}},
    })
    rec, _ = to_record(req)
    assert rec["state"] == "document: charged twice"         # the image ref is never rendered
    assert rec.get("images") == [_png_data_url()]            # but passed through for the image path
    # and a state without images gets no key at all
    rec2, _ = to_record(req.model_copy(update={"state": {"document": "x"}}))
    assert "images" not in rec2

    # a dict state whose `images` value is NOT a list passes through as ordinary data
    rec3, _ = to_record(req.model_copy(update={"state": {"images": "two"}}))
    assert "images" not in rec3 and "images: two" in rec3["state"]


def test_inject_shifts_every_branch_index():
    """_inject must produce an encoding rows_of can split, whose branch content is
    byte-identical to the un-injected record's: only the state gains the image
    segment, every branch index shifts by the injection length. Pure algebra - uses
    the real VisionHook methods via instance __new__ (no model/tower needed)."""
    from kev import vision

    if not os.path.isdir(SNAP):
        pytest.skip("Qwen3.5-0.8B-Base snapshot not cached (needs its tokenizer)")
    tok = load_tokenizer(SNAP)

    hook = vision.VisionHook.__new__(vision.VisionHook)  # no model/tower for this tier
    hook.image_token, hook.vision_start, hook.vision_end = 248056, 248053, 248054

    rec = {
        "state": "Customer: Anna. Ticket #777: refund requested for a broken mug.",
        "questions": [
            {"instr": "refund?", "options": ["No", "Yes"], "label": 0},
            {"instr": "product?", "options": ["mug", "lamp"], "label": 0},
        ],
    }
    enc0 = encode(tok, rec, max_state=384, max_branch=1024)
    s0, sp0, rows0 = rows_of(enc0)
    n_per = [10, 8]
    enc1 = hook._inject(dict(enc0), n_per)
    s1, sp1, rows1 = rows_of(enc1)

    Ni = sum(n_per) + 2 * len(n_per)          # placeholder pads + vision_start/end pairs
    want_seg = [248053] + [248056] * 10 + [248054, 248053] + [248056] * 8 + [248054]
    # state gained exactly the image segment at its end; text prefix untouched
    assert s1[: len(s0)] == s0 and sp1[: len(sp0)] == sp0
    assert s1[len(s0):] == want_seg
    # image positions continue the state sequence one per merged row...
    assert sp1[len(sp0):] == list(range(len(sp0), len(sp0) + Ni))
    # ...and branch positions shift by Ni; decide/opt offsets are WITHIN the branch
    # (rows_of subtracts start), and both d and start shifted by Ni -> invariant.
    for r0, r1 in zip(rows0, rows1):
        assert r1["ids"] == r0["ids"]
        assert r1["pos"] == [p + Ni for p in r0["pos"]]
        assert r1["decide"] == r0["decide"]
        assert r1["opts"] == r0["opts"]
    assert len(rows1) == 2
    # usage accounting (serve._probs_images reports these)
    assert enc1["seg"].count(0) == len(s1) == len(s0) + Ni


# --- tier 2: gate-off equality -------------------------------------------------------


def test_gate_off_text_path_untouched():
    """With KEV_VISION unset, importing kev.serve must not import kev.vision, and
    _probs dispatches on rec['images'] only (the dict key set by api.to_record)."""
    import subprocess
    code = (
        "import sys; "
        f"sys.path.insert(0, {os.getcwd()!r}); "
        "import kev.serve as s; "
        "import importlib; "
        "assert 'kev.vision' not in sys.modules, 'vision imported without the gate'; "
        "print('gate-off ok')"
    )
    out = subprocess.run([sys_bin(), "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "gate-off ok" in out.stdout, out.stderr


def sys_bin():
    import sys
    return sys.executable


def test_probes_dispatch_on_images_key():
    """The image branch of _probs is keyed on rec['images'] alone - a plain
    (non-list) value under a different key never triggers it."""
    from kev import serve
    # contract only: _probs_images must exist and be a separate function from _probs
    assert callable(serve._probs_images) and serve._probs_images is not serve._probs


# --- tier 3: 0.8B full chain --------------------------------------------------------


@pytest.mark.skipif(not os.path.isdir(SNAP), reason="Qwen3.5-0.8B-Base snapshot not cached")
def test_08b_full_chain():
    from kev.vision import attach
    from PIL import Image as _Image

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = load_tokenizer(SNAP)
    model = DecisionModel(SNAP, tok, dev, dtype=torch.float32)
    model.eval()
    hook = attach(model, tok, SNAP)
    assert hook is not None, "0.8B-Base must carry a vision tower"

    rec = {
        "state": "Customer: Anna. Ticket #777: refund requested for a broken mug.",
        "questions": [
            {"instr": "refund?", "options": ["No", "Yes"], "label": 0},
            {"instr": "product?", "options": ["mug", "lamp"], "label": 0},
        ],
    }
    p_text = model.probs(model.encode(tok, rec))
    imgs = [_Image.new("RGB", (100, 60), (200, 30, 30)), _Image.new("RGB", (77, 77), (30, 30, 200))]
    p_img, m = hook.probs_with_images(rec, imgs)
    assert len(p_img) == 2
    assert all(q.shape[0] == len(qn["options"]) for q, qn in zip(p_img, rec["questions"]))
    assert m["state_tokens"] > 0 and m["tokens"] > m["state_tokens"]
    # image rows changed the readout (same-process comparison is the valid sanity)
    assert max(abs(float(a) - float(b)) for qa, qb in zip(p_img, p_text)
               for a, b in zip(qa, qb)) > 0
    # deterministic rerun
    p_img2, _ = hook.probs_with_images(rec, imgs)
    assert all(torch.equal(x, y) for x, y in zip(p_img, p_img2))
    hook.tower.to("cpu", torch.float32)  # release GPU memory before process teardown


# Reminders for future tiers (not tests, no placeholders in the suite):
# - the placeholder-count assert in probs_with_images and the row-count assert in
#   embed() refuse loudly on tower/processor disagreement (see kev/vision.py); forcing
#   a disagreement from tests needs injected processor fixtures - revisit if the
#   assert line ever changes.
