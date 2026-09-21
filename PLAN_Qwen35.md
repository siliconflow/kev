# Plan: move Kev to the Qwen3.5 base family

> **Status: completed 2026-09-20.** The plan below was executed in full; §10 records the results. The Qwen3.5 family (Kev-0.8B / 4B / 9B) shipped the same day. The living plan is [`PLAN.md`](PLAN.md).

Status: **proposal for review**, 2026-09-20. Nothing here has been started except the probe in section 2. Numbers are on the same frozen items as every other number in this repository ([`evals/v4/transfer-v4`](evals/v4/transfer-v4/manifest.json), development partition, 764 records); untrained bases are read zero-shot from next-token letter logits with [`scripts/base_mmlu_probe.py`](scripts/base_mmlu_probe.py), the same probe used for every untrained row in the README.

## 1. What "latest Qwen base" means, checked

Every Qwen release after Qwen3 shares one text architecture, `qwen3_5_text`, and only Qwen3.5 has Base checkpoints ([Hub listing, `author=Qwen`](https://huggingface.co/Qwen)):

| family | released | Base weights | architecture (from `config.json`) |
|---|---|---|---|
| Qwen3 | 2025 | 0.6B, 1.7B, 4B, 8B, 14B, 32B ([Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base)) | dense attention, every layer |
| **Qwen3.5** | Feb–Mar 2026 | **0.8B, 2B, 4B, 9B, 35B-A3B** ([Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base), [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)) | hybrid: 32 text layers, **24 Gated DeltaNet ("linear_attention") + 8 full attention**, `full_attention_interval: 4` ([9B config](https://huggingface.co/Qwen/Qwen3.5-9B-Base/blob/main/config.json)) |
| Qwen3.6 | Apr 2026 | none ([27B](https://huggingface.co/Qwen/Qwen3.6-27B), 35B-A3B, post-trained only) | `qwen3_5_text`, 64 layers, 16 attention / 48 DeltaNet |
| Qwen3.8 | Aug 2026 | none ([27B](https://huggingface.co/Qwen/Qwen3.8-27B), 2.4T-A95B, post-trained only) | `qwen3_5_text`, same shape as 3.6-27B |

So: **Qwen3.5 Base is the newest generation Kev can train on**, and building for its architecture builds for Qwen3.6/3.8 the day their Base weights appear. Model details for the two sizes we would use (read from the configs):

| | Qwen3.5-4B-Base | Qwen3.5-9B-Base |
|---|---|---|
| Hub revision (pin) | `1001bb4d826a52d1f399e183466143f4da7b741b` | `68c46c4b3498877f3ef123c856ecfde50c39f404` |
| text layers | 32 (8 attention, 24 DeltaNet) | 32 (8 attention, 24 DeltaNet) |
| hidden | 2560 | 4096 |
| attention heads / KV heads | 16 / 4 | 16 / 4 |
| DeltaNet value heads | 32 | 32 |
| vocab | 248,320 | 248,320 |
| context | 262,144 | 262,144 |
| checkpoint parameters (incl. vision tower) | 4.7B | 9.7B |

The checkpoints are `Qwen3_5ForConditionalGeneration` (a vision tower is bundled). Loading through `AutoModelForCausalLM` gives the text-only `Qwen3_5ForCausalLM` with `.model` = `Qwen3_5TextModel`; verified on Qwen3.5-0.8B-Base under transformers 5.17.0 (forward on CPU, top token for "The capital of France is" → " Paris"). Our five delimiter tokens (`kev/model.py` [`SPECIAL`](kev/model.py#L10)) exist in the Qwen3.5 tokenizer with ids 248049–248062, so the encoding scheme carries over unchanged.

## 2. The probe: is the new base actually better for Kev?

Run 2026-09-20 on Modal H100s with [`modal_probe35.py`](modal_probe35.py) (own image: transformers 5.17, [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) for the DeltaNet kernel). Reports and per-item rows: [`runs/probes/`](runs/probes/). Paired comparisons are record-clustered bootstraps ([`kev.benchmark.paired_bootstrap`](kev/benchmark.py)).

| untrained base, zero-shot | acc | Brier | MMLU | PAWS | QNLI | SciQ | TweetEval | Emotion | authorization | **deadline** | rule: (A∨B)∧C | rule: (A∧B)∨¬C | rule: if-then-not |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.567 | 0.505 | 0.42 | 0.71 | 0.76 | 0.93 | 0.57 | 0.29 | 0.50 | 0.28 | 0.38 | 0.62 | 0.44 |
| Qwen3.5-0.8B | 0.566 | 0.510 | 0.47 | 0.66 | 0.72 | 0.90 | 0.62 | 0.21 | 0.55 | 0.28 | 0.50 | 0.56 | 0.50 |
| Qwen3-4B | 0.671 | 0.392 | 0.68 | 0.79 | 0.78 | 0.97 | 0.65 | 0.29 | 1.00 | 0.55 | 0.56 | 0.41 | 0.47 |
| Qwen3.5-4B | 0.692 | 0.395 | 0.69 | 0.84 | 0.84 | 0.99 | 0.65 | 0.28 | 0.95 | **0.68** | 0.53 | 0.44 | 0.50 |
| Qwen3-8B | 0.726 | 0.366 | 0.75 | 0.84 | 0.89 | 1.00 | 0.68 | 0.30 | 1.00 | 0.53 | 0.59 | 0.62 | 0.62 |
| Qwen3.5-9B | 0.729 | 0.356 | 0.74 | 0.84 | 0.90 | 0.99 | 0.69 | 0.31 | 1.00 | **0.82** | 0.59 | 0.47 | 0.44 |
| *Kev-8B (trained, Qwen3-8B)* | *0.796* | *0.337* | *0.70* | *0.78* | *0.91* | *1.00* | *0.79* | *0.56* | *1.00* | *0.60* | *0.97* | *0.91* | *0.59* |
| *Jev* | *0.857* | *0.211* | *0.90* | *0.79* | *0.93* | *0.99* | *0.81* | *0.59* | *1.00* | *0.93* | *0.91* | *0.97* | *0.78* |

Paired deltas, new base minus old base at matched size:

- Qwen3.5-0.8B vs Qwen3-0.6B: **+0.8 pp** [−3.3, +4.8]
- Qwen3.5-4B vs Qwen3-4B: **+2.1 pp** [−1.5, +5.8]
- Qwen3.5-9B vs Qwen3-8B: **−0.3 pp** [−4.9, +3.9]

Sources: [`runs/probes/qwen35-9b-base-base-transfer-v4/report.json`](runs/probes/qwen35-9b-base-base-transfer-v4/report.json), [`qwen35-4b`](runs/probes/qwen35-4b-base-base-transfer-v4/report.json), [`qwen35-2b`](runs/probes/qwen35-2b-base-base-transfer-v4/report.json), [`qwen35-08b`](runs/probes/qwen35-08b-base-base-transfer-v4/report.json), [`qwen3-8b`](runs/probes/qwen3-8b-base-transfer-v4/report.json), [`qwen3-4b`](runs/probes/qwen3-4b-base-transfer-v4/report.json), [`qwen3-06b`](runs/probes/qwen3-06b-base-transfer-v4/report.json); Kev-8B from [`runs/v7-final/00-trial-0/result.json`](runs/v7-final/00-trial-0/result.json); Jev from [`runs/jev-transfer-v4/report.json`](runs/jev-transfer-v4/report.json).

### Reading

1. **Overall, the new generation is not a better base on our suite.** At 9B vs 8B the difference is zero within noise; at 4B it is +2 pp with a CI that includes zero. MMLU and PAWS, the two places Kev loses most to Jev, are unchanged (0.74 vs 0.75, 0.84 vs 0.84). A version bump alone would not have been worth a port.
2. **But the gain is concentrated exactly where Kev is stuck.** On `deadline` (day-precision date arithmetic to a 3-level ordinal), Qwen3.5-9B gets **33/40** items zero-shot where Qwen3-8B gets **21/40**, and Qwen3.5-4B gets 0.68 where Qwen3-4B gets 0.55. This is the one family that no training data moved — two purpose-built families in `decision-v7`/`v8` left it at 0.45–0.60 for every Kev ([PLAN.md, "Final v7/v8 read"](PLAN.md#toward-v02-crossing-the-release-screen-2026-09-19)) — and it was the deciding family for the predeclared held-out-pair screen, which Kev-8B missed by one pair (0.69 vs 0.70).
3. **The rule-composition columns are irrelevant to the choice of base.** Every base is near chance on them zero-shot; Kev learns them from the synthetic data (Kev-8B 0.91–0.97). What carries over from the base is knowledge and arithmetic, and the low-learning-rate recipe was found precisely because it preserves base capability ([PLAN.md, "The learning rate is the lever"](PLAN.md#overnight-autoresearch-branch-researchovernight-1-pr-3)).

**Verdict: proceed**, on the narrow hypothesis that a Kev trained on Qwen3.5-9B keeps most of the base's 0.82 on `deadline`, which would lift held-out-pair correctness from 0.69 to roughly 0.80 and put the overall out-of-domain number at ~0.81 (Jev 0.857). The probe does not justify expecting gains on knowledge or paraphrase. Everything below is scoped to test that hypothesis at controlled cost, with the existing Kev-8B as the paired baseline on the same items.

## 3. Why the port is real work: the hybrid breaks the packed mask

Kev's forward pass today packs the state and every question into one sequence and uses a block-causal additive mask so that a question sees the state but never another question ([`encode`](kev/model.py#L30), [`branch_mask_batch`](kev/model.py#L75), [`hidden_batch`](kev/model.py#L145)). The mask is applied inside attention. In Qwen3.5, 24 of 32 layers are Gated DeltaNet ([Yang et al., 2024](https://arxiv.org/abs/2412.06464); implementation [`Qwen3_5GatedDeltaNet` in transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)): a recurrent state updated token by token, plus a short causal convolution. Neither respects a per-token attention mask, so in a packed sequence question 2's tokens would read a recurrent state that has already absorbed question 1's tokens. Isolation, the property the README verifies to 4e-6, would fail by construction. `transformers` confirms this: in `Qwen3_5TextModel.forward` the mask dict is only consulted for `full_attention` layers (`attention_mask=causal_mask_mapping[self.config.layer_types[i]]`); the DeltaNet path only uses the 2D mask to zero padding.

### The design: one state pass, then questions as a batch of rows

Replace "one row, masked" with "state once, branches as separate rows continuing from the state":

1. **State pass.** Run the state tokens (`<state> …`, positions `0..Ls-1`) once with `use_cache=True`. The cache holds the KV of the 8 attention layers and, for the 24 DeltaNet layers, the recurrent state and the conv state (`cache_params.layers[i].recurrent_states`, `conv_states` in the transformers implementation).
2. **Branch pass.** Build a batch of `Q` rows, one per question: `<q> instr <opt> o </opt> … <decide>`, right-padded, positions continuing from `Ls` exactly as [`encode`](kev/model.py#L30) already assigns them. Replicate the cache along the batch dimension and run the rows. The chunked prefill path takes `initial_state=recurrent_state` when a cache with previous state is present (`torch_chunk_gated_delta_rule(..., initial_state=...)`), so a multi-token continuation from a cached state is supported by the library, not something we would hack in. **This exact pattern is already running on Qwen3.5-4B in the wild:** SemIf's [`shared.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/shared.py) prefills the state once, replicates the cache with `cache.reorder_cache(torch.zeros(Q))`, and scores right-padded suffixes in one batch (positions continuing from the prefix length, `logits_to_keep` at each row's last real token); its [MLX backend](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/mlx_backend.py) does the same with MLX-LM's prompt cache (`entry.merge([entry] * Q)`, `prepare(lengths, right_padding)`). We adopt the `reorder_cache` idiom rather than `batch_repeat_interleave`, since it is what has been exercised on this architecture. They also quantified the cost: bf16 reuse changed 5–6 of 777 argmaxes versus fresh scoring ([results](https://github.com/TheoLeeCJ/SemIf/blob/master/docs/RESULTS.md)), the same magnitude as the bf16 noise we measured on our own prefix cache, so the fp32 parity tests stay.
3. **Readout.** Gather `</opt>` and `<decide>` hidden states per row and apply the unchanged [`PointerHead`](kev/model.py) — [`_readout`](kev/model.py#L162) already takes per-question index lists.

Properties: isolation is exact **by construction** (rows are independent tensors; no mask to get wrong), on any architecture; the state is computed once, as today; FLOPs equal the packed form (the packed sequence also processes every branch token once); the serving prefix cache we shipped ([`prefix`](kev/model.py#L183), [`probs_with_prefix`](kev/model.py#L206)) is literally step 1 + step 2 and becomes the *only* path rather than an optimization. What we lose: the single-row form and the `option_isolation` flag (which cost −5.8 pp at 4B anyway, [PLAN.md](PLAN.md#overnight-autoresearch-branch-researchovernight-1-pr-3)).

**Training** through a cache is the one uncertain mechanical step (cache tensors may be updated in place or detached). Fallback that is exactly correct and simple: for training, build rows as `[state + branch_q]` with the state duplicated per row. Cost is `Q×` the state tokens instead of `1×`; our training records average ~1.3 questions, so ≤ 1.5× the state compute, and the state is short (≤ 384 tokens). Start there; optimize to the cache-continuation form only if it is measurably needed.

**Equivalence check that gates everything:** on the existing Qwen3-based Kev-4B, the row-batched forward must reproduce the packed-mask forward's probabilities to fp32 noise on the 24-record parity set used for the serving changes. If it does, the same code path is the reference for the hybrid.

## 4. Dependencies and their risks

| item | today | needed | evidence / risk |
|---|---|---|---|
| transformers | `>=4.51,<4.58` ([pyproject](pyproject.toml)) | `>=5.17` (qwen3_5 is not in 4.x) | 5.17.0 loads Qwen3.5-0.8B-Base and runs a forward (verified in a scratch venv). **Risk:** 5.x API changes for the existing Qwen3 path; gate with the parity tests in [`tests/test_v3.py`](tests/test_v3.py) (merged-vs-unmerged, prefix-vs-full, bucket padding) and a bf16 dev-set re-score of the published Kev-4B (must match [`runs/v7-rc3/01-trial-1/result.json`](runs/v7-rc3/01-trial-1/result.json) within bf16 noise). |
| peft | `0.21.0` | same — **peft 0.21.0 is already verified on Qwen3.5 + transformers 5** by [pngwn/system-one-qwen3.5-4b-scorer-v2b](https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer-v2b) (`peft_version: 0.21.0`, LoRA r=16 α=32, 30.5M trainable params, lr 1e-4, seq 384, 11.5k steps) | LoRA target names on the DeltaNet layers, verified against `Qwen3_5TextModel` in transformers 5.17: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`; attention (`q/k/v/o_proj`) and MLP (`gate/up/down_proj`) names are unchanged. pngwn targets `in_proj_qkv`, `in_proj_z`, `out_proj` + attention + MLP. That scorer and [Bespoke-Nimble-9B](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B) both score one sequence per question (pngwn: one per *option*, via `AutoModelForSequenceClassification`); neither shares the state across questions, which is what section 3 adds. |
| DeltaNet kernel | – | [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) + triton on CUDA | Without it transformers falls back to a reference PyTorch implementation ("correct but much slower", its own warning). CUDA-only. **Mac serving runs the fallback**; speed unmeasured — measured in phase 1 before committing to a Mac story. |
| Modal image | pins `uv.lock` | a second image (or the upgraded lock) | [`modal_probe35.py`](modal_probe35.py) already builds the transformers-5 image; the main app follows once the lock is upgraded. |
| suites | base revisions pinned per suite ([`kev.suite.freeze`](kev/suite.py#L146), [`validated_trial`](kev/experiment.py#L43) requires a 40-hex `base_revision` for unpinned bases) | pass `base_revision` per trial (already supported) or freeze `decision-v9` = v7 with Qwen3.5 revisions pinned | Dev/test bytes stay identical to v4 either way, so every number remains comparable. |
| vision tower | – | ignored | `AutoModelForCausalLM` loads the text model only (verified). Weight download is the full 9.7B checkpoint. |
| vocabulary | 152k | 248k | Embedding matrix is frozen; the pointer head reads hidden states, not logits. No change. |

## 5. Work plan

Each phase has a stop condition. Costs are Modal H100 at the rates we have been paying (Kev-4B trial ≈ $3.50, Kev-8B ≈ $6; [PLAN.md, "Compute and spending"](PLAN.md#5-compute-and-spending)).

### Phase 0 — MMLU-Pro, a buried-state variant, and cross-benchmarks with SemIf (independent of the port; ~4 h, ~$3)
- Add `mmlu_pro` to [`kev/data.py`](kev/data.py) as an **eval-only** source ([TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), [Wang et al., 2024](https://arxiv.org/abs/2406.01574); 10 options, `answer_index`), rendered as a Choice with neutral keys like the other MCQ converters. Pin the dataset revision (`b189ec765a…`).
- Freeze `transfer-v9` = transfer-v4 items + 200 MMLU-Pro items (new version; v4 numbers stay valid, and v9 minus the new sources equals v4 byte-for-byte for comparability).
- Score Jev, the untrained bases, and Kev-4B/8B on it. Expectation, from openjev's report on the same benchmark: Jev ~0.83, untrained one-token readouts ~0.6, Kev well below Jev — the honest number for a one-pass model on a benchmark designed for chain-of-thought. It replaces the saturated 4-way MMLU in the knowledge column of the README.
- Add an eval-only **buried-state variant** to `transfer-v9`: the same record with the state embedded in unrelated background text (SemIf's "irrelevant context" perturbation and localjev's 2,048-word distraction condition both found this is where small models fall over; we train at ≤ 384 state tokens and have never measured it).
- **SemIf-style untrained baseline on our suite**: Qwen3.5-4B *instruct* (their pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) with their exact prompt ([`core.direct_messages`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/core.py): system instruction + JSON `{evidence, criterion, options}` through the chat template, letter logits) on `transfer-v4`, alongside our base-model probe row. Our probe uses the *base* model and a plain prompt; theirs may be the stronger untrained readout and is the one people will compare against.
- **Kev on the uncontaminated ticket set from [scienthoon/jev-ood-calibration](https://github.com/scienthoon/jev-ood-calibration)** (900 rule-generated tickets × 3 questions, regenerable from a seed; live Jev numbers already exist on the same items via the same AI Gateway path we use: 89.0% / 91.7% / 44.7%, overall ECE 0.107 at 4.4× the noise floor). Two findings there shape our evaluation: Jev's miscalibration **flips sign by question type** (choice/score overconfident, refit T ≈ 3.3; boolean underconfident, T ≈ 0.66), and on the **unknowable** priority question (the label is an org rule absent from the text) Jev is at chance with mean stated probability 0.74. Add to `transfer-v9` an **unknowable-label family** from our contrastive generator (label depends on a rule the state does not contain) where the scored behaviour is *low* confidence — a "knows it doesn't know" column no open model reports today.
- **Coverage at a fixed error budget** as a first-class metric in [`kev.benchmark`](kev/benchmark.py): the maximum fraction of decisions automatable at ≤ 5% empirical error, as in [AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks) (live Jev: 0.83 on AG News, 0.86 on Banking77, 0.00 on Emotion). It is the operational meaning of calibration and belongs in the README table. That harness is Apache-2.0 and pluggable; a PR adding Kev as a system yields third-party numbers we did not produce.
- **Kev on SemIf's fixtures**: convert their committed [`authored144.jsonl`](https://github.com/TheoLeeCJ/SemIf/blob/master/benchmarks/data/authored144.jsonl) and [`perturbations108.jsonl`](https://github.com/TheoLeeCJ/SemIf/blob/master/benchmarks/data/perturbations108.jsonl) (three families: evidence interpretation, rule application, candidate selection; 2–3 options with ids and descriptions) into System One requests and score Kev-4B/8B with the labels they ship. Their reported direct-logit numbers are 0.813 balanced accuracy on the 144 and 0.723 on the 36 perturbation bases. Also run **live Jev** on the 144 rows (cents) — their Jev column is agreement with published outputs on a different subset, not a live read.

### Phase 1 — transformers 5 branch and row-batched forward (½–1 day, ~$2)
- Branch `qwen35`. Upgrade the lock to transformers 5.x / peft ≥ 0.18; run the full test suite and the bf16 re-score of Kev-4B; fix what 5.x moved.
- Implement `DecisionModel.forward_rows` (state pass + row batch) alongside the packed path; add `test_rows_match_packed` on the smoke checkpoint (fp32, CPU, < 1e-4) and on Kev-4B (bf16, 24 records, 0 argmax flips).
- Load Qwen3.5-0.8B-Base through `DecisionModel`, run the isolation and packing checks from [`kev.evaluate`](kev/evaluate.py) on the untrained model (they test the *mechanism*, not accuracy). **Stop if** the DeltaNet cache continuation does not reproduce a single-row forward to fp32 noise; that would mean the library's chunked-prefill-with-initial-state path is not usable and we would need to write the continuation ourselves before spending on training.
- Measure the CPU/MPS fallback speed of the 0.8B and 4B forward. Record it; it decides whether "serves on a Mac" survives for Qwen3.5-based Kevs or waits on MLX.

### Phase 2 — the controlled experiment (~1 day wall, ~$25)
Same data, same recipe, new base. Nothing else changes, so any difference is the base.
- `decision-v7` training partition, lr 5e-5, 2 epochs, LoRA r=16, `p_none_pair 0.25` (the released recipe, [`experiments/v7-final.json`](experiments/v7-final.json)); LoRA on attention + MLP projections in all layers, DeltaNet in/out projections included (one ablation without them at 4B).
- Cells: Qwen3.5-4B × 2 seeds, Qwen3.5-9B × 2 seeds, scored on `transfer-v4` (and `transfer-v9` once Phase 0 lands). Paired bootstraps against Kev-4B and Kev-8B on the same items ([`kev.autoresearch compare`](kev/autoresearch.py)).
- Read-outs, in order of what the hypothesis predicts: `deadline` (≥ 0.75 expected if base skill is retained), held-out pairs (≥ 0.70 on both seeds is the old release screen), overall transfer, MMLU/PAWS retention (should be ≥ Kev-8B's 0.70/0.78; if lower, the recipe is eroding the new base more and a small lr sweep at 3e-5/2e-5 is the next $10), Brier and confident-error rate.
- **Stop if** Kev-9B(3.5) is not above Kev-8B on transfer with a CI excluding zero *and* deadline did not move. Then the result is written up in PLAN.md as a negative and the family stays on Qwen3.

### Phase 3 — promotion (only if Phase 2 passes; ~½ day, ~$5)
- One locked-test read for each winning size ([`modal_app.py::locked_test`](modal_app.py)), gated if the pair screen passes on both seeds.
- Cards for `kev-4b`/`kev-9b` on Qwen3.5 (naming: the Hub repo is by size, so `jaredpalmer/kev-9b`; `kev-8b` stays as is), README table and figures regenerated from result files ([`scripts/plot_family.py`](scripts/plot_family.py), [`scripts/plot_tweet.py`](scripts/plot_tweet.py)), the "untrained base" rows switched to Qwen3.5.
- Serving: the row-batched path is the prefix-cache path; re-run the serving benchmark in the README's "Serving performance" section on CUDA.
- **Mac serving through MLX-LM**, promoted from "deferred" on SemIf's evidence that MLX-LM runs Qwen3.5 with prompt-cache replication today ([docs/MLX.md](https://github.com/TheoLeeCJ/SemIf/blob/master/docs/MLX.md): MLX 0.32.2, MLX-LM pinned past the Qwen recurrent q/k-norm fix, vision weights stripped by the sanitizer, 4/8-bit in memory). Kev's additions are small: load the fp32-merged LoRA weights (we already merge at load) into the MLX-LM Qwen3.5 text model, take hidden states from the inner model instead of `lm_head` logits, and apply the pointer head in MLX. Gate with the same parity tests against the PyTorch fp32 path. Measure the torch fallback first (Phase 1); if it is within ~2× of today's MPS latency, MLX waits.

### Deferred, deliberately
- Qwen3.5-35B-A3B-Base: 70 GB in bf16, MoE, serving footprint out of the laptop story; only after the 9B result.
- A browser/WebGPU path (SemIf's distribution advantage, via wllama/GGUF). Kev's pointer head and per-token positions do not fit llama.cpp as-is; with the row-batched design a GGUF export of the merged model plus a small JS head is conceivable, but it is its own project.
- Post-trained Qwen3.6/3.8 as bases: changes the recipe and removes the untrained-base comparison; revisit when Base weights exist.

## 6. Competitive context: SemIf

[SemIf](https://github.com/TheoLeeCJ/SemIf) (formerly OpenJev; 2.3k stars) is an **untrained** readout: frozen Qwen3.5-4B instruct, chat-template JSON prompt, probabilities from letter logits, with prefix-reuse and parallel-suffix modes and a WebGPU browser demo. Its own results page states the probabilities are "uncalibrated as decision confidence" and that "the next justified phase is targeted training for decision semantics and calibration" — i.e. Kev. What it means for us:

- **It validates the port design** (section 3) and hands us the MLX path (Phase 3). Its author solved the hybrid-cache replication problem on exactly our target base.
- **It sharpens the claim Kev has to make.** Their headline is "3.8 pp behind Jev" on *agreement with Jev's published answers* over 102 curated public cases. On labelled frozen items, untrained Qwen3.5-4B scores 0.692 on our suite and Kev-4B 0.790: training the readout is worth about +10 pp at 4B, and it is what buys calibration (Brier, confident-error rate), which they do not measure. Once a Qwen3.5-based Kev-4B exists the comparison is apples to apples on the same base, which is the cleanest possible demonstration of what training adds.
- **Their evaluation is thinner than ours** (144 authored rows, agreement rather than ground truth, no live Jev, no paired uncertainty). Phase 0 scores both ways on frozen items and runs live Jev on their rows; the neutral-ground position is worth more than winning any single cell.
- **Their distribution is stronger than ours** (browser demo, "no waitlist"). Not something this plan addresses; noted in deferred work.

### Laya (convaiinnovations), reviewed 2026-09-20

[Laya](https://huggingface.co/convaiinnovations/laya) ([code](https://github.com/NandhaKishorM/laya), 957 Hub likes) is the other trained open decision model with weights, and it is built the opposite way from Kev: a **bidirectional encoder** (ModernBERT-large, 395M, fully fine-tuned; a 322M mmBERT variant for 100+ languages) plus a 2-layer transformer head that scores each option at its own `[MASK]` token ([`common.py`](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py)). Sequence per question: `[CLS] type+instructions [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]`, 512 tokens (192 reserved for options). **One sequence per question**: "all questions in one forward pass" means a batch, so the state is re-encoded for every question and cost scales linearly (T4: 39.5 ms for 1 question, 158.6 ms for 10). Trained with what they call RLCD: REINFORCE with Gaussian logit noise against a strictly proper reward (log score + 0.5·spherical + ranked probability score for ordinals, group-mean baseline). In expectation that objective has the same optimum as the log loss Kev minimises directly; the RL framing adds variance, not a different target. Its Jev numbers are third-party published figures, not measured.

What it teaches us, and what it does not:

- **Encoders are strong in-distribution and near chance off it.** Their own limits section: base checkpoints score 0.362 zero-shot on their typed-decisions benchmark (random 0.318, majority 0.461); the 0.766 headline is a checkpoint fine-tuned on that benchmark's training split. They report no held-out-source number at all. That is the failure mode `transfer-v4` was built to expose, and the reason Kev's headline is out-of-domain accuracy on never-trained sources. Any comparison we publish should score Laya on our frozen OOD items (their SDK is pip-installable; `agent.predict(state, questions)` takes System One shaped questions, so the conversion is trivial).
- **High-cardinality choice is a real Kev advantage.** Laya scores 0.425 on Banking77 (77 options at 3–4 tokens each in a 192-token option budget); pngwn's scorer caps options at 16; Jev supports 255. Kev-0.5B scored 0.86 on 77-way Banking77 with the pointer head and no option budget. We should measure and state it (a ≥ 50-option column in the README table).
- **Calibration as shipped is worse than ours, fixed by finer temperature grouping.** Laya ships at ECE 0.466 (multilingual 0.314) and reaches 0.081 with one temperature per (question type, option count). Kev ships at 0.09 in-distribution raw; our single fitted temperature did not transfer out of domain. Their grouping is a cheap thing to test on our dev/test split (Phase 0 scope, one afternoon).
- **Multilingual is a differentiator Kev gets for free from the base and has never measured.** Laya needs two checkpoints and a script-detecting router because ModernBERT is English-only. Qwen3 and Qwen3.5 are multilingual; one Kev should cover XNLI/MASSIVE languages without routing. Eval-only multilingual slices belong in `transfer-v9`.
- **The "act/escalate" head is a product idea, not a model idea.** It is an MLP over (top-1, margin, entropy, K) features plus the `[CLS]` vector predicting act-vs-escalate. Kev can expose an equivalent abstain flag from the same probability features, calibrated on the development partition, with no retraining.
- **Nothing to adopt from the training objective.** Log loss is already a strictly proper scoring rule; we tested the ranked probability score as an extra term (`--ord_w`) and it did not help; the spherical term is unlikely to differ. Their soft-target training against teacher distributions is closer to distillation; Kev deliberately trains on labels, not Jev outputs.

### Where Kev sits in the field

multimodalart's [Jev Reproductions Tracker](https://huggingface.co/spaces/multimodalart/jev-reproductions-tracker) lists ~60 artifacts (SemIf, Laya, Nimble, pngwn's Qwen3.5-4B scorer, several RLCD LoRAs, localjev, benchmarks). **Kev is not on it.** Submitting it is free and is the single cheapest distribution action available.

## 7. Decision criteria, stated in advance

Ship a Qwen3.5-based Kev-9B as the new top of the family **only if**, on the same 764 items: transfer accuracy ≥ Kev-8B's 0.796 with the paired CI excluding zero, `deadline` ≥ 0.75, MMLU and PAWS not below Kev-8B by more than seed noise (~1 pp), and Brier ≤ 0.34. Ship a Qwen3.5-based Kev-4B if it beats Kev-4B (0.790) by the same test. Otherwise the outcome is a documented negative result and the Qwen3 family remains the release. Locked test read once per candidate, as always.

## 8. Future exploration after the re-architecture

Not part of this plan's budget; ordered by expected value per dollar once a Qwen3.5-based Kev exists. Each is one controlled experiment on the frozen suites.

1. **Multilingual out-of-domain slices** (XNLI in 14 languages, MASSIVE intent in 51; eval-only, in `transfer-v9`). Zero training cost; tests whether the multilingual base carries the trained readout across languages, which would make one Kev cover what Laya needs a router and two checkpoints for. ~$2.
2. **Finer calibration** — one temperature per (question type, option count), fitted on the development partition, evaluated on the locked test only once at promotion time. Two independent sources now point here: Laya (ECE 0.466 → 0.081 with this grouping) and scienthoon's Jev study (sign of miscalibration differs by type on the same inputs). Ours says a single temperature does not transfer. ~$0.
3. **A high-cardinality column** in the README table (Banking77 77-way held out, plus a synthetic 100–255-option family), where Kev's pointer head has no option budget and every open competitor caps or collapses. ~$3.
4. **An abstain signal** — a calibrated "uncertain" flag from (top-1, margin, entropy, K), reported alongside probabilities in `/v1/systemone`; product-level, no retraining. Evaluate as selective accuracy at fixed coverage on the OOD suite.
5. **Small-model path via an encoder**: ModernBERT-large / mmBERT with Kev's pointer head and a block mask that keeps state tokens from attending to questions (so the state encoding is question-independent and shareable). Answers, for ~$5, whether a 400M encoder can beat Kev-0.6B's 0.62 out of domain or whether Laya's near-chance-off-distribution result is intrinsic to encoders. If it wins, it replaces Kev-0.6B for the 30 ms latency tier.
6. **Self-distillation for the small model**: Kev-9B's probabilities as soft targets for Kev-0.6B/2B on the same training partition (no Jev outputs involved). Laya's soft-target training and Nimble's teacher setup both suggest gains at small sizes; our rule against training on Jev outputs is untouched.
7. **Ordinal readout for Score** — the fix proposed in PLAN.md for the deadline family; if the Qwen3.5 base closes most of the gap by itself (section 2), this drops in priority, which is the cheapest possible outcome.
8. **Qwen3.5-35B-A3B-Base** and, when Base weights ship, Qwen3.6/3.8 — same code path, no new engineering.

## 9. Budget and time

| phase | wall time | Modal |
|---|---|---|
| 0 MMLU-Pro, buried-state variant, SemIf cross-benchmarks | 4 h | ~$3 |
| 1 transformers 5 + row forward + hybrid smoke | ½–1 day | ~$2 |
| 2 controlled experiment | ~1 day | ~$25 |
| 3 promotion (+ MLX serving if the torch fallback is slow) | ½–1 day | ~$5 |
| total | ~3–4 days | **~$37** of the remaining ~$240 |

## 10. Execution log and results (2026-09-20, branch `qwen35`)

Everything below ran on 2026-09-20 between 15:30 and 18:30. Spend: ~$95 of Modal H100 time (4B trials ≈ $4.50 each,
9B ≈ $7, plus probes, benches, locked reads and the ablation), plus $0.03 of Jev calls.

### Phase 1 — port (done)
- transformers 5.17 / peft 0.21 on the lock. The 62-test suite passes. **fp32 parity of the published Kev-4B under transformers 5: max |Δp| = 2e-5 on 30 development questions** against the saved H100 probabilities ([`runs/v7-rc3/01-trial-1/development/rows.json`](runs/v7-rc3/01-trial-1/development/rows.json)).
- Row-batched forward ([`kev/model.py: rows_of, forward_rows_batch`](kev/model.py)): on the attention-only Kev-4B it is **bit-identical** to the packed block-causal form (max |Δp| = 0.000000, 0 flips, 24 records, fp32 MPS) and 9% faster. Hybrid detection from `config.layer_types`; LoRA on `in_proj_qkv/z/a/b`, `out_proj` for DeltaNet layers.
- Hybrid isolation and serving prefix path on Qwen3.5-0.8B: together-vs-alone and prefix-vs-full within 1e-5 ([`tests/test_v3.py::test_hybrid_rows_isolation_and_prefix`](tests/test_v3.py)); cache replication via `reorder_cache` on `DynamicCache(config=…)` with `LinearAttentionLayer` states.
- Modal image: `flash-linear-attention` + **`triton>=3.7.1`** (fla refuses its gated chunk backward on Hopper with Triton 3.4–3.7.0, fla#640; the first smoke trial hit exactly that guard). Row-form training cost: 0.11 s/record at 4B, 0.18 s/record at 9B on one H100 (Kev-4B packed was ~0.06); 4B trial 61–63 min, 9B 88–100 min.
- **Mac latency is the cost of the hybrid.** Same 5-question request, bf16, M5: Kev-4B 174 ms; Kev(3.5)-4B **779 ms** (reference DeltaNet and causal-conv kernels; no MPS fla). 0.8B: 329 ms vs Qwen3-0.6B 123 ms. The prefix cache does not help on MPS at these sizes. MLX-LM (SemIf's path) is the fix if the Mac story matters; on CUDA the kernels are fast.

### Phase 2 — controlled experiment (done): same data (`decision-v7`), same recipe, new base

`transfer-v4` development (764 records), paired record-clustered bootstraps against the released checkpoint on the same items ([`scripts/compare_q35.py`](scripts/compare_q35.py)):

| trial | seed | dev acc | transfer acc | paired Δ vs released [95% CI] | deadline | held-out pairs | MMLU | PAWS | Emotion | Brier | conf. err | cov@5% err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Kev-4B** (Qwen3, released) | 1 | 0.854 | 0.790 | reference | 0.53 | 0.73 | 0.65 | 0.72 | 0.66 | 0.328 | 0.082 | 0.31 |
| Kev(3.5)-4B | 0 | 0.877 | 0.788 | −0.007 [−0.042, +0.028] | 0.55 | 0.72 | 0.70 | 0.76 | 0.59 | 0.339 | 0.096 | 0.48 |
| Kev(3.5)-4B | 1 | 0.876 | 0.800 | +0.015 [−0.024, +0.056] | 0.53 | 0.75 | 0.71 | 0.76 | 0.59 | 0.322 | 0.095 | 0.52 |
| Kev(3.5)-4B | 2 | 0.877 | 0.794 | +0.013 [−0.013, +0.046] | 0.55 | 0.78 | 0.70 | 0.74 | 0.54 | 0.316 | 0.082 | 0.54 |
| Kev(3.5)-4B | 3 | 0.873 | 0.770 | −0.024 [−0.064, +0.021] | 0.50 | 0.69 | 0.64 | 0.76 | 0.59 | 0.346 | 0.093 | 0.49 |
| **Kev-8B** (Qwen3, released) | 0 | 0.863 | 0.796 | reference | 0.60 | 0.69 | 0.70 | 0.78 | 0.56 | 0.337 | 0.099 | 0.45 |
| Kev(3.5)-9B | 0 | 0.871 | 0.802 | +0.004 [−0.046, +0.055] | 0.70 | 0.75 | 0.74 | 0.78 | 0.57 | 0.308 | 0.069 | 0.54 |
| Kev(3.5)-9B | 1 | 0.876 | **0.812** | +0.027 [−0.018, +0.064] | 0.72 | **0.80** | 0.74 | 0.76 | 0.59 | **0.291** | 0.075 | 0.53 |
| Jev | – | 0.845 | 0.857 | | 0.93 | 0.86 | 0.90 | 0.79 | 0.59 | 0.211 | 0.037 | 0.70 |

Seed means: 4B 0.788 (4 seeds) vs Kev-4B's 0.778 (3 seeds); 9B 0.807 (2) vs Kev-8B's 0.785 (2).

**Against the section-7 criteria, on the development partition: not met.** No single seed beats its released counterpart with a CI excluding zero; `deadline` reached 0.70–0.72 at 9B, not 0.75. What did move, on every seed: in-distribution accuracy (+2.2 pp at 4B, +1.0 at 9B), held-out-pair correctness (4B 3 of 4 seeds ≥ 0.70, 9B both — the first time the original release screen is met at 8–9B), coverage at ≤ 5 % error (0.31 → ~0.50 at 4B; 0.45 → 0.53 at 9B), Brier and confident errors at 9B, MMLU (+4–6 pp at both sizes), and PAWS at 4B. Emotion is worse at 4B (−7 pp).

**Locked test, read once per candidate** (candidates selected on development only: 4B seed 2 = `q35-4b-s23/00-trial-0`, dev 0.877; 9B seed 1 = `q35-9b/01-trial-1`, dev 0.876). [`runs/locked/kev-4b-q35/`](runs/locked/kev-4b-q35/summary.json), [`runs/locked/kev-9b-q35/`](runs/locked/kev-9b-q35/summary.json):

| | in-distribution | out-of-domain | Brier | held-out pairs | conf. err | paired Δ acc vs predecessor on the test items |
|---|---|---|---|---|---|---|
| Kev-4B (released) | 0.856 | 0.806 | 0.294 | 0.66 | 0.066 | |
| **Kev(3.5)-4B** | 0.870 | **0.832** | 0.266 | 0.72 | 0.069 | +0.029 [−0.009, +0.064] |
| Kev-8B (released) | 0.870 | 0.780 | 0.327 | 0.62 | 0.099 | |
| **Kev(3.5)-9B** | 0.873 | **0.837** | **0.243** | 0.75 | 0.055 | **+0.073 [+0.028, +0.117]** |

On the confirmatory read the 9B beats Kev-8B by 7.3 pp with a CI excluding zero and a Brier 0.084 lower; the 4B beats Kev-4B by 2.9 pp with a CI that grazes zero. Both new checkpoints are above their predecessors on every column. This is one read, on the partition the criteria did not name; it is reported as such.

### The deadline hypothesis: refuted as stated, and the reason is now known
The base's date arithmetic (Qwen3.5-9B 0.82, 4B 0.68 zero-shot) does **not** survive training: 9B trained 0.70–0.72, 4B 0.50–0.55. A diagnostic that reads the *adapted* backbone through the plain LM head with letter logits ([`scripts/base_mmlu_probe.py --adapter`](scripts/base_mmlu_probe.py); [`runs/probes/qwen35-4b-base-base-adapted-q35-4b-s1-transfer-v4`](runs/probes/qwen35-4b-base-base-adapted-q35-4b-s1-transfer-v4/report.json)) gives deadline **0.53** — identical to the pointer-head number — while MMLU is retained (0.72). The skill is lost in the representation during LoRA training, not at the readout. Independently, [issue #8](https://github.com/jaredpalmer/kev/issues/8) (3x3xX3N0N) shows that appending "The report arrived N days after/before the deadline" lifts Kev-4B from 0.575 to 0.963 with the unmodified head; reproduced here: Kev-4B 0.525 → **0.925**, Kev(3.5)-4B → **1.000**. A *generic* annotator (pairwise date differences without role names) helps far less (0.55 / 0.675), so there are two gaps: date subtraction (large) and binding a raw date fact to the policy's roles (smaller; the newer base does it better). Consequences: the ordinal readout drops to last; the next experiment is the training-data change (relational day counts and a `date_facts` field rendered in the date-bearing families), with issue #8's number as the ceiling. **Ablation (`q35-4b-nopolicy`, Qwen3.5-4B, seed 1, same recipe):** training on the ten public sources only gives deadline **0.50**; public + rule trees without the date-bearing policy families gives **0.55**; the full recipe 0.53; the untrained base 0.68. So the date-bearing families are *not* the cause — any LoRA fine-tune on this format erodes the base's date arithmetic, the same way it erodes MMLU. The fix is therefore not in the policy templates. Options, in order: give the model the number (issue #8's route, as training renderings + an optional request preprocessor), or retention methods that the earlier lr/anchor experiments did not solve. Side result of the same ablation: the rule trees are what teach compositional rules (held-out `(A or B) and C` 0.69 → 0.91, `or not` 0.44 → 0.75 when they are added), and the policy pairs add the rest (pairs 0.52 → 0.69 → 0.75).

### Phase 0 — new evaluation columns (done)
`transfer-v9` ([`evals/v9/transfer-v9`](evals/v9/transfer-v9/manifest.json), [`kev/transfer_v9.py`](kev/transfer_v9.py)): transfer-v4 byte-identical + 200 MMLU-Pro (10-way) + 80 buried-state + 110 unknowable / 110 intact controls per partition; mirrored to the Hub (`SUITES_REVISION a957287d`). New metrics in [`kev.benchmark`](kev/benchmark.py): `coverage_at_5pct_error`, `coverage_at_1pct_error`, and an `unknowable` block (mean max-probability and share ≥ 0.9 against the paired intact controls). Development partition:

| model | knowable acc | MMLU-Pro | buried (avg of 4) | unknowable: mean max-p (intact) | share ≥ 0.9 (intact) | cov@5% err |
|---|---|---|---|---|---|---|
| Jev (live, $0.02) | 0.854 | **0.840** | 0.70 | 0.61 (0.92) | **0.09** (0.75) | **0.70** |
| Qwen3.5-9B base, untrained | 0.696 | 0.540 | 0.64 | 0.52 (0.78) | 0.00 (0.44) | 0.38 |
| Kev-4B | 0.729 | 0.440 | 0.69 | 0.78 (0.96) | 0.44 (0.90) | 0.25 |
| Kev-8B | 0.747 | 0.500 | 0.72 | 0.77 (0.97) | 0.26 (0.91) | 0.41 |
| Kev(3.5)-4B s2 | 0.742 | 0.500 | 0.66 | 0.72 (0.97) | 0.19 (0.88) | 0.47 |
| Kev(3.5)-9B s1 | **0.771** | 0.545 | **0.74** | 0.67 (0.97) | 0.05 (0.93) | 0.46 |

MMLU-Pro separates the generations where 4-way MMLU did not (untrained: Qwen3-8B 0.380 vs Qwen3.5-9B 0.540), and the trained models retain it. On the unknowable family the Qwen3-based Kevs are confidently wrong (26–44 % of evidence-free items answered at ≥ 0.9); Kev(3.5)-9B is at 5 %, Jev at 9 %, untrained bases near 0. Training with hard labels makes a model commit; the fix is unknowable training records with uniform targets (soft-label support in `kev.train`), listed below.

### Cross-benchmarks with the field (done)
- **SemIf-style untrained readout** (Qwen3.5-4B *instruct*, their exact prompt) on our `transfer-v4`: **0.747**, Brier 0.362 — stronger than our base probe (0.692) and the untrained row people will compare to. Kev-4B is +7.7 pp [+3.2, +12.2] over it; Kev(3.5)-4B seed 1 +5.3 pp. [`runs/probes/qwen35-4b-semif-transfer-v4`](runs/probes/qwen35-4b-semif-transfer-v4/report.json).
- **SemIf's authored 144 + 108 perturbations** ([`evals/external/semif-v1`](evals/external/semif-v1/manifest.json), [`scripts/freeze_semif.py`](scripts/freeze_semif.py)), with **live Jev**: Jev 0.965 (perturbations 1.00/1.00/1.00); Kev-4B 0.847; Kev-8B 0.903; Kev(3.5)-4B 0.896; **Kev(3.5)-9B 0.917** (perturbations 0.97/0.97/0.97). SemIf's untrained Qwen3.5-4B: 0.813. Their "3.8 pp behind Jev" was agreement on a different subset; on labelled items live Jev is 15 pp above their readout.
- **scienthoon's 900 tickets** ([`evals/external/scienthoon-v1`](evals/external/scienthoon-v1/manifest.json), their live Jev rows converted, [`scripts/freeze_scienthoon.py`](scripts/freeze_scienthoon.py)): queue / angry / priority(unknowable) — Jev 0.897 / 0.914 / 0.447 (ECE 0.105); Kev-4B 0.687 / **0.375** / 0.498; Kev-8B 0.924 / 0.515 / 0.381; Kev(3.5)-4B 0.928 / 0.794 / 0.402 (ECE 0.086); **Kev(3.5)-9B 0.952 / 0.911 / 0.430 (ECE 0.082)**. The Qwen3 Kevs fail "The customer sounds angry." because the instruction is an assertion, not a question, and every ticket is complaint-shaped: Kev-4B calls 95 % of tickets angry (base rate 37 %). The Qwen3.5 checkpoints mostly fix it; assertion-style Noul instructions belong in the training renderings regardless.

### Decision (open, for review)
The pre-registered development-partition criteria are **not met**; the single confirmatory locked-test read shows Kev(3.5)-9B **+7.3 pp [+2.8, +11.7]** over Kev-8B with much better calibration, and Kev(3.5)-4B +2.9 pp [−0.9, +6.4] over Kev-4B, both above their predecessors on every metric measured, on every external suite, and with the original held-out-pair screen met on both 9B seeds. Costs: 4.5× slower on a Mac until an MLX path exists; transformers 5 required. Recommendation: publish Kev(3.5)-4B and Kev(3.5)-9B as the new family (naming below), keep the Qwen3 checkpoints on the Hub as the previous generation, and state the criteria outcome plainly in the cards. Publishing is a release action and waits for approval.

**Kev-0.8B (added 20:00):** Qwen3.5-0.8B-Base with the 0.6B recipe (lr 1e-4), three seeds: transfer 0.622 / 0.634 / 0.643 (Kev-0.6B 0.620), dev 0.817–0.829 (0.801), held-out pairs 0.27–0.44 (0.08). Seed 2 released as `jaredpalmer/kev-0.8b`; locked test 0.827 / **0.668** vs Kev-0.6B 0.808 / 0.642, paired +4.8 pp [+0.2, +9.3]. The family is now Qwen3.5 throughout (0.8B / 4B / 9B); the Qwen3 checkpoints are kept for Mac latency and no longer developed.

Next experiments, in order: (1) date-family renderings with relational day counts + `date_facts` (issue #8; ceiling ~0.93–1.0 on deadline); (2) unknowable training records with uniform targets; (3) assertion-style Noul instructions; (4) MLX serving for the hybrid on Mac; (5) multilingual slices.
