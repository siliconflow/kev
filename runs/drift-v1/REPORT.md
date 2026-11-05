# drift-v1: why Kev-27B v1's semif-v1 logits moved between 2026-09-23 and 2026-09-30

## Question

`jaredpalmer/kev-27b@v1-lora` (Hub `01b81998`, a LoRA r=16 adapter on `Qwen/Qwen3.8-27B@1d4bf0f2`, T 1.3819) was read on
`evals/external/semif-v1` development (252 rows) on 2026-09-23 (`runs/r6-27bv2-s2-semif`, raw logits, T 1.0) and again on
2026-09-30 by `release_verify` (`runs/rel27-public/3-jaredpalmer_kev-27b_v1-lora`, logits divided by T 1.3819). Argmax
was equal on 252 of 252 rows, but the logits were not: max |Δp| 0.032 at T 1.38, max |Δz| 0.18 raw (0.13 after the
temperature), median |Δz| 0.024, and no row identical. The evaluation path is described as fp32-exact on CUDA, so this
needed a cause.

## Answer

**The 2026-09-24 image change, not kev code and not nondeterminism.** PR #125 (`eb45fd2`, 2026-09-24 21:43 -04:00) added
the causal-conv1d 1.7.0 CUDA wheel to the Modal image. Since then transformers runs the short convolution of each of the
48 Gated DeltaNet layers (`causal_conv1d_fn`) through that CUDA kernel. Before, it ran its PyTorch reference
(`F.conv1d`, then SiLU). The 09-23 read's log says so: "`causal_conv1d_fn` is falling back to its reference PyTorch
implementation because `causal_conv1d` is not installed". Kev-27B's frozen backbone is bf16 (`weights_dtype: bf16`), so
the two convolutions round differently (the PyTorch path rounds the convolution to bf16 before SiLU; the kernel computes
conv + bias + SiLU in fp32 and rounds once). 64 layers carry that difference to the logits. The change was intended; #125
added the kernel for training speed. Neither version is wrong, but reads made before and after it cannot be compared bit
for bit.

Nothing else differed. `uv.lock` is identical between the read's code (research commit `c154ed1`, 2026-09-23 16:58) and
`main` (`0a58b699`): torch 2.8.0, transformers 5.17.0, peft 0.21.0. flash-linear-attention was unpinned then and is
pinned at 0.5.2 now, but 0.5.2 was already the newest release (2026-07-27). Triton `>=3.7.1` resolves to 3.8.0 (the newest
release since 2026-08-28). The semif records are short, so both kev versions score them through `forward()`'s row form.
#135's state-once path (`probs()`) and #149's long-row rule are not on this path.

## Evidence (one H200, today's image; `probe_app.py`, `compare.py`; `h200-a/comparison.json`)

One container loaded the v1 adapter once and scored the 252 rows twice under each convolution. The convolution was
swapped in place in `transformers.models.qwen3_5.modeling_qwen3_5`. The container then ran kev.benchmark from `c154ed1`
(the 09-23 code) in subprocesses, once with causal-conv1d hidden and once with it visible. Raw logits, T 1.0:

| read | vs the 09-23 read | vs the 09-30 public read |
|---|---|---|
| today's kev, causal-conv1d kernel (the image as built), twice | 0/252 identical, max \|Δz\| 0.18, max \|Δp\| 0.032, 0 flips | **equal** (see below) |
| today's kev, PyTorch conv, twice | **252/252 bit-identical** | max \|Δz\| 0.18 |
| 09-23 kev (`c154ed1`), PyTorch conv | **252/252 bit-identical** | max \|Δz\| 0.18 |
| 09-23 kev (`c154ed1`), causal-conv1d kernel | 0/252, max \|Δz\| 0.18 | **equal** |

"Equal" against the public read: that read stores z / T, computed on the GPU. z / T recomputed on the CPU from our raw
logits matches it on 98 of 252 rows and is 1 fp32 ulp away on the rest (≤ 4.8e-7). The same 1-ulp pattern shows against
the 09-24 breadth read below, where the PyTorch-conv read otherwise reproduces exactly. Repeats within a container were
bit-identical under both convolutions. The 09-30 read ran in another container with a fresh `HF_HOME`, so reads with fixed
kernels are deterministic across containers too.
The base weights came from `kev-hf-cache`. Downloads and compiled Triton kernels may have been added to that cache,
nothing was deleted from it, and `kev-runs` was not mounted.

- **Code drift: none.** The 09-23 code with today's kernels gives today's numbers, and today's code with the 09-23 kernels
  gives the 09-23 numbers, bit for bit.
- **Dependency drift: one package**, causal-conv1d, on Modal from #125 on.
- **Nondeterminism: none measured.** Every pair of same-configuration reads is bit-identical.

## Magnitude on a longer panel (`h200-breadth/`, `metrics_offset.py`)

breadth-v1 development (1,990 records, 3,075 questions, private mirror), Kev-27B v1 on an H200, against the committed
09-24 read `runs/breadth-v1-kev-27b` (T 1.3819). With the PyTorch conv we reproduce that read: 1,354 rows bit-identical and
the rest within 1 ulp of the temperature division, 0 flips. With the kernel:

| | 09-24 read (PyTorch conv) | today (causal-conv1d) |
|---|---|---|
| max \|Δp\| / argmax flips | — | 0.059 / 13 of 3,075 (0.4 %) |
| accuracy | 0.7451 | 0.7441 |
| NLL | 0.58996 | 0.59006 |
| Brier | 0.32133 | 0.32142 |
| ECE | 0.0118 | 0.0126 |
| coverage at ≤ 5 % error | 0.4985 | 0.5011 |

On semif-v1 (clean, n 144, T 1.38): accuracy 0.9722 in both, NLL 0.1371 → 0.1374, Brier 0.0608 → 0.0610.

## What "fp32-exact" does and does not cover (Kev-4B, fp32 backbone, H100; `h100-kev4b-fp32-*`, `kev4b-tf32-vs-ieee.json`)

The same probe on the released `jaredpalmer/kev-4b@139fdd94` (Qwen3.5-4B, fp32 evaluation path) on semif-v1, T 2.96:

| comparison | max \|Δz\| | median \|Δz\| | max \|Δp\| | flips |
|---|---|---|---|---|
| causal-conv1d vs PyTorch conv (fla at its default precision) | 0.0071 | 0.0011 | 6.0e-4 | 0 |
| fla at its default precision vs transformers' PyTorch DeltaNet (no Triton) | 0.030 | 0.0054 | 2.7e-3 | 1 / 252 |
| same two, with `TRITON_F32_DEFAULT=ieee` (and fla's `SOLVE_TRIL_DOT_PRECISION` set to `ieee`) | 3.3e-5 | 5.7e-6 | 3.7e-6 | 0 |
| causal-conv1d vs PyTorch conv, both with IEEE Triton dots | 3.1e-5 | 5.7e-6 | 2.6e-6 | 0 |

`LocalPredictor` turns TF32 off in torch (matmul and cuDNN) and disables the fused SDPA kernels. That makes attention and
every PyTorch matmul fp32. It does not reach flash-linear-attention's Triton kernels, which run the Gated DeltaNet rule on
CUDA. Their `tl.dot` calls default to TF32 on Ampere and newer, and `chunk_fwd.py` asks for TF32 explicitly. So on CUDA a
Qwen3.5 "fp32" read carries TF32 rounding inside every DeltaNet layer: ≈ 0.003 in p from true fp32 on this panel, with 1
flip in 252. It also amplifies any upstream bit change by about 200× (the 3e-5 conv difference becomes 0.007).
`tests/test_model.py::_exact_kernels` already swaps these kernels for exactly this reason ("fla rounds its fp32 dots like
TF32 on CUDA"). The eval contract did not say so. On CPU and MPS no Triton kernels run, so those reads are fp32 (and
differ from CUDA reads by the TF32 term). A bf16 backbone (Kev-27B) is bf16 in every kernel; there, the conv change alone
moves p by up to 0.03-0.06.

## Consequences

- **Rounds 19-26** pair Kev-27B parent reads made before #125 (`runs/r6-27bv2-s2-{docs,v9,semif,wanli2,typesafe}`,
  `runs/hv1-27b`, `runs/dt1-27b`, `runs/breadth-v1-kev-27b`, `runs/locked/kev-27b-v2-ungated`; first committed
  2026-09-23 22:19 to 2026-09-24 17:46) with arm reads made after it. The parent and arm sides of those panels therefore
  ran on different convolution kernels. Measured here, that offset is up to 0.1 pp of accuracy, +0.0001 NLL and Brier,
  +0.0008 ECE on breadth-v1, and 0 flips on semif-v1. That is an order of magnitude below the margins by which rounds 25
  and 26 failed (for example breadth-v1 ECE 0.020-0.029 against 0.0176). No verdict in those rounds is near enough to
  move. Nothing was re-read or re-decided here.
- **Reads that ran on an image with causal-conv1d reproduce each other bit for bit.** Round 23's reads and the v2
  release verification (`rel27-public`, 252/252 and 764/764) are an example, so that check is sound. A deployed app keeps
  the image of its last deploy, so a read's date alone does not say which kernel set it ran on.
- **A future change to the image** (causal-conv1d, fla, Triton, torch, the CUDA driver's kernels) will move every
  hybrid-backbone read in the same way: to bf16 rounding for Kev-27B and to TF32 rounding for the fp32 Qwen3.5 family.
  A paired comparison must use reads from the same kernel set. Re-read the parent when in doubt. The cost is minutes.

## Not a bug; not changed

The drift comes from an intended environment change, so there is nothing to fix in kev. The TF32 gap in the eval
contract is real, but closing it (IEEE Triton dots in `LocalPredictor`) would move every future Qwen3.5 read by up to
0.003 in p against every committed read, which is a policy decision (and a speed cost), not a fix. AGENTS.md now says what
the contract covers. #191 makes report.json record the kernel set a read ran on.

## Spend

Modal, app `kev-drift` (every run ephemeral, all stopped): 1 failed start, 2 H200 probes (semif-v1, breadth-v1) and 3
H100 probes (Kev-4B). The workspace's metered cost went from $4,581.40 to $4,600.21 while they ran ($18.81, an upper bound; other apps
share the workspace).
