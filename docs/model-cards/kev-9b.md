---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.5-9B-Base
base_model_relation: adapter
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - lora
  - multiple-choice
  - typesafe
  - qwen3.5
datasets:
  - legacy-datasets/banking77
  - google/boolq
  - fancyzhx/ag_news
  - nyu-mll/multi_nli
  - SetFit/sst5
  - Yelp/yelp_review_full
  - CogComp/trec
  - fancyzhx/dbpedia_14
  - SetFit/amazon_reviews_multi_en
  - stanfordnlp/imdb
metrics:
  - accuracy
  - brier_score
  - expected_calibration_error
model-index:
  - name: Kev-9B
    results:
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v7 development (1,204 records; ten trained public sources + programmatic policy data)" }
        metrics:
          - { type: accuracy, value: 0.876 }
          - { type: expected_calibration_error, value: 0.060, name: "ECE, raw probabilities" }
      - task: { type: text-classification, name: typed decision, out-of-domain }
        dataset: { type: mixed, name: "transfer-v4 development (764 records; six never-trained sources + held-out policy structures)" }
        metrics:
          - { type: accuracy, value: 0.812 }
          - { type: brier_score, value: 0.291 }
      - task: { type: text-classification, name: typed decision, out-of-domain, locked test }
        dataset: { type: mixed, name: "transfer-v4 test (read once)" }
        metrics:
          - { type: accuracy, value: 0.837 }
          - { type: brier_score, value: 0.243 }
---

# Kev-9B

Kev-9B is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution per question out, in one forward pass. No text generation. It is a LoRA adapter (r=16, 45.4M trainable parameters) plus a pointer head on `Qwen/Qwen3.5-9B-Base` (revision `68c46c4b`), serving TypeSafe's public `/v1/systemone` contract.

**The most accurate Kev.** Out of domain it scores 0.812 on the development partition and **0.837 on the locked test**, against Kev-8B's 0.796 / 0.780 on the same items, with the lowest Brier of any Kev (0.243 on the test). It is the first Kev at 8–9B to pass the 70% held-out-pair screen on every seed (0.75 / 0.80). Same recipe at two seeds: transfer 0.802 / **0.812**; this checkpoint is seed 1, selected on the development partition.

- Hub: `jaredpalmer/kev-9b` (this repo; trial `q35-9b/01-trial-1`)
- Code, suites, every trial with hashes and paired bootstraps: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) — `PLAN_Qwen35.md` (the port and this experiment), `PLAN.md`, `runs/leaderboard.md`

## Results (same frozen items for every row)

| | Kev-4B (Qwen3) | Kev-8B (Qwen3) | Kev-4B | **Kev-9B** | Jev |
|---|---|---|---|---|---|
| in-distribution accuracy (decision-v7 dev, 1,204 records) | 0.854 | 0.863 | 0.877 | 0.876 | 0.845 |
| out-of-domain accuracy (transfer-v4 dev, 764 records) | 0.790 | 0.796 | 0.794 | **0.812** | 0.857 |
| out-of-domain Brier | 0.328 | 0.337 | 0.316 | **0.291** | 0.211 |
| confident errors out of domain (p ≥ 0.9 and wrong) | 8.2% | 9.9% | 8.2% | 7.5% | 3.7% |
| coverage at ≤ 5% error (share of decisions automatable) | 0.31 | 0.45 | 0.54 | 0.53 | 0.70 |
| held-out policy structures, both siblings correct | 0.73 | 0.69 | 0.78 | **0.80** | 0.86 |
| option-order flip rate | 0.06 | 0.00 | 0.08 | 0.03 | 0.00 |
| **locked test**, out-of-domain accuracy / Brier | 0.806 / 0.294 | 0.780 / 0.327 | 0.832 / 0.266 | **0.837 / 0.243** | – |
| **locked test**, in-distribution accuracy | 0.856 | 0.870 | 0.870 | 0.873 | – |

Per-source out-of-domain accuracy (Kev-9B / Jev): QNLI 0.93 / 0.93, SciQ 0.96 / 0.99, TweetEval-offensive 0.78 / 0.81, PAWS 0.76 / 0.79, MMLU 0.74 / 0.90, Emotion 0.59 / 0.59, deadline (3-level date arithmetic) 0.72 / 0.93, (A or B) and C 0.84 / 0.91, (A and B) or not C 0.88 / 0.97, if A then not B else C 0.91 / 0.78.

**Paired against Kev-8B on the same items** (record-clustered bootstrap): development +2.7 pp [−1.8, +6.4]; **locked test +7.3 pp [+2.8, +11.7]**, Brier −0.084. The pre-registered criteria for this experiment asked for a development-partition CI excluding zero and `deadline` ≥ 0.75; neither was met (deadline 0.72). The locked read, taken once after selection, is the confirmatory number. Both facts are in `PLAN_Qwen35.md` §10.

**Newer evaluation columns** (`transfer-v9` development, Kev-9B / Kev-8B / Jev): MMLU-Pro (10-way) 0.545 / 0.500 / 0.840; state buried among unrelated records 0.74 / 0.72 / 0.70; share of *unknowable* items (deciding evidence removed) answered at ≥ 0.9 confidence **0.05** / 0.26 / 0.09 — lower is better, and this is the first Kev that mostly declines to commit when the evidence is missing.

**External suites** (same items as their published Jev numbers): SemIf's authored 144 — 0.917 (Kev-8B 0.903, SemIf's untrained Qwen3.5-4B 0.813, live Jev 0.965); scienthoon's 900 tickets — queue 0.952, angry 0.911, ECE 0.082 (Jev 0.897, 0.914, 0.105).

## What changed from Kev-8B

- **Base model**: Qwen3.5-9B-Base, a hybrid of 24 Gated DeltaNet (linear attention) layers and 8 full-attention layers. Because the recurrent layers cannot honour a block-causal mask, questions run as separate causal rows that continue from the shared state (`kev/model.py: forward_rows_batch`); isolation is exact by construction (together vs alone within 1e-5) and on attention-only models this form is bit-identical to the packed one.
- **Same data and recipe** as Kev-4B/8B: `decision-v7`, two epochs, LoRA r=16 (attention, MLP and DeltaNet projections), lr 5e-5. Nothing else changed, so every difference above is the base.
- What the new base bought: knowledge retention (MMLU +4 pp, MMLU-Pro +4.5 pp), calibration (Brier 0.337 → 0.291, confident errors 9.9% → 7.5%), held-out rules (0.69 → 0.80), and much better behaviour on evidence-free questions. What it did not buy: out-of-domain accuracy beyond noise on the development partition, or date arithmetic — the base does it at 0.82 zero-shot and training erodes it to 0.72 (see `PLAN_Qwen35.md`, "The deadline hypothesis").

## Known limits

- **Slow on a Mac.** The DeltaNet kernels have no MPS implementation; PyTorch falls back to reference code. A five-question request that takes 0.3 s on Kev-8B takes about 2 s here in bf16 on an M5. On CUDA with `flash-linear-attention` installed it is fast. Use Kev-8B (Qwen3, `jaredpalmer/kev-8b`) for low latency on Apple Silicon until an MLX path exists.
- Requires `transformers >= 5.17` (the `qwen3_5` architecture) and `peft >= 0.21`.
- Date arithmetic (`deadline` 0.72 vs Jev 0.93) and knowledge (MMLU 0.74 vs 0.90) remain the gap to Jev. Training data that renders day counts is the next experiment ([issue #8](https://github.com/jaredpalmer/kev/issues/8)).
- Out-of-domain probabilities are usable but not calibrated (raw ECE 0.105 dev, 0.093 test); temperature fitted in-domain does not transfer.
- 9B bf16 needs ~19 GB of GPU memory for serving; training took 91 min on one H100 (peak 39.5 GB).

## Training

Frozen suite `evals/v7/decision-v7`: 10,000 public records (1,000 per source), 896 policy minimal-pair records over nine template families, 1,680 records from 60 randomly generated rule structures in four rendering styles. Two epochs, LoRA r=16 α=32 on `q/k/v/o_proj`, `gate/up/down_proj`, `in_proj_qkv/z/a/b`, `out_proj`; pointer head from scratch; cross-entropy on the option distribution; lr 5e-5 (OneCycle), effective batch 8, bf16 autocast with fp32 master weights, gradient checkpointing; option permutation, none-of-the-above insertion, distractors, none minimal pairs on 25% of Choice records. No Jev outputs were used for training.

## Evaluation protocol

Development partitions select models; the locked test partition is read at most once per candidate (`runs/locked/kev-9b-q35/`). Every number carries suite hash, code hashes and git commit in `result.json`. Untrained-base baselines use zero-shot letter logits on the same items (`scripts/base_mmlu_probe.py`).

## Use

```bash
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-9b --port 8008      # KEV_DTYPE=bf16 on a 32 GB Mac; slow on MPS, see limits
```

Any TypeSafe-compatible client works: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`.

## License

Apache-2.0 for the adapter and head; the Qwen3.5 base is Apache-2.0; datasets carry their own licenses.
