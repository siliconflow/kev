---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.5-4B-Base
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
  - name: Kev-4B
    results:
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v7 development (1,204 records; ten trained public sources + programmatic policy data)" }
        metrics:
          - { type: accuracy, value: 0.877 }
          - { type: expected_calibration_error, value: 0.059, name: "ECE, raw probabilities" }
      - task: { type: text-classification, name: typed decision, out-of-domain }
        dataset: { type: mixed, name: "transfer-v4 development (764 records; six never-trained sources + held-out policy structures)" }
        metrics:
          - { type: accuracy, value: 0.794 }
          - { type: brier_score, value: 0.316 }
      - task: { type: text-classification, name: typed decision, out-of-domain, locked test }
        dataset: { type: mixed, name: "transfer-v4 test (read once)" }
        metrics:
          - { type: accuracy, value: 0.832 }
          - { type: brier_score, value: 0.266 }
---

# Kev-4B

Kev-4B is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution per question out, in one forward pass. No text generation. It is a LoRA adapter (r=16, 33.8M trainable parameters) plus a pointer head on `Qwen/Qwen3.5-4B-Base` (revision `1001bb4d`), serving TypeSafe's public `/v1/systemone` contract.

**The recommended Kev.** The best accuracy per byte: out of domain 0.794 on the development partition and **0.832 on the locked test**, against the Qwen3 Kev-4B's 0.790 / 0.806 on the same items, with better calibration (Brier 0.266 vs 0.294 on the test) and the highest held-out-rule score of any 4B (0.78). Same recipe at four seeds: transfer 0.788 / 0.800 / **0.794** / 0.770, held-out pairs 0.72 / 0.75 / 0.78 / 0.69; this checkpoint is seed 2, selected on the development partition (highest development accuracy).

- Hub: `jaredpalmer/kev-4b` (this repo, main revision; trial `q35-4b-s23/00-trial-0`). The previous Qwen3 checkpoint is at revision `qwen3` and has [its own card](kev-4b-qwen3.md).
- Code, suites, every trial with hashes and paired bootstraps: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) — `PLAN_Qwen35.md` (the port and this experiment), `PLAN.md`, `runs/leaderboard.md`

## Results (same frozen items for every row)

| | Kev-4B (Qwen3) | Kev-8B (Qwen3) | **Kev-4B** | Kev-9B | Jev |
|---|---|---|---|---|---|
| in-distribution accuracy (decision-v7 dev, 1,204 records) | 0.854 | 0.863 | **0.877** | 0.876 | 0.845 |
| out-of-domain accuracy (transfer-v4 dev, 764 records) | 0.790 | 0.796 | **0.794** | 0.812 | 0.857 |
| out-of-domain Brier | 0.328 | 0.337 | **0.316** | 0.291 | 0.211 |
| confident errors out of domain (p ≥ 0.9 and wrong) | 8.2% | 9.9% | 8.2% | 7.5% | 3.7% |
| coverage at ≤ 5% error (share of decisions automatable) | 0.31 | 0.45 | 0.54 | 0.53 | 0.70 |
| held-out policy structures, both siblings correct | 0.73 | 0.69 | **0.78** | 0.80 | 0.86 |
| option-order flip rate | 0.06 | 0.00 | 0.08 | 0.03 | 0.00 |
| **locked test**, out-of-domain accuracy / Brier | 0.806 / 0.294 | 0.780 / 0.327 | **0.832 / 0.266** | 0.837 / 0.243 | – |
| **locked test**, in-distribution accuracy | 0.856 | 0.870 | 0.870 | 0.873 | – |

Per-source out-of-domain accuracy (Kev-4B / Jev): QNLI 0.93 / 0.93, SciQ 0.99 / 0.99, TweetEval-offensive 0.74 / 0.81, PAWS 0.74 / 0.79, MMLU 0.70 / 0.90, Emotion 0.54 / 0.59, deadline (3-level date arithmetic) 0.55 / 0.93, (A or B) and C 0.91 / 0.91, (A and B) or not C 0.88 / 0.97, if A then not B else C 1.00 / 0.78.

**Paired against the Qwen3 Kev-4B on the same items** (record-clustered bootstrap): development +1.3 pp [−1.3, +4.6]; **locked test +2.9 pp [−0.9, +6.4]**, Brier −0.028. The pre-registered criteria for this experiment asked for a development-partition CI excluding zero and `deadline` ≥ 0.75; neither was met (deadline 0.55). The locked read, taken once after selection, is the confirmatory number. Both facts are in `PLAN_Qwen35.md` §10.

**Newer evaluation columns** (`transfer-v9` development, Kev-4B / Qwen3 Kev-4B / Jev): MMLU-Pro (10-way) 0.500 / 0.440 / 0.840; state buried among unrelated records 0.66 / 0.69 / 0.70; share of *unknowable* items (deciding evidence removed) answered at ≥ 0.9 confidence 0.19 / 0.44 / 0.09 — lower is better.

**External suites** (same items as their published Jev numbers; measured on seed 1 of this recipe): SemIf's authored 144 — 0.896 (Qwen3 Kev-4B 0.847, SemIf's untrained Qwen3.5-4B 0.813, live Jev 0.965); scienthoon's 900 tickets — queue 0.928, angry 0.794, ECE 0.086 (Jev 0.897, 0.914, 0.105; the Qwen3 Kev-4B scored 0.687 / 0.375).

## What changed from the Qwen3 Kev-4B

- **Base model**: Qwen3.5-4B-Base, a hybrid of 24 Gated DeltaNet (linear attention) layers and 8 full-attention layers. Because the recurrent layers cannot honour a block-causal mask, questions run as separate causal rows that continue from the shared state (`kev/model.py: forward_rows_batch`); isolation is exact by construction (together vs alone within 1e-5) and on attention-only models this form is bit-identical to the packed one.
- **Same data and recipe** as the Qwen3 checkpoints: `decision-v7`, two epochs, LoRA r=16 (attention, MLP and DeltaNet projections), lr 5e-5. Nothing else changed, so every difference above is the base.
- What the new base bought: in-distribution accuracy (+2.3 pp), knowledge retention (MMLU +5 pp, MMLU-Pro +6 pp), held-out rules (0.73 → 0.78, and 3 of 4 seeds ≥ 0.70), coverage at ≤ 5% error (0.31 → 0.54), and behaviour on evidence-free questions (0.44 → 0.19 confidently answered). What it did not buy: out-of-domain accuracy beyond noise on the development partition, Emotion (0.66 → 0.54), or date arithmetic — the base does it at 0.68 zero-shot and training erodes it to 0.55 (see `PLAN_Qwen35.md`, "The deadline hypothesis").

## Known limits

- **Slow on a Mac.** The DeltaNet kernels have no MPS implementation; PyTorch falls back to reference code. A five-question request that takes 0.17 s on the Qwen3 Kev-4B takes 0.78 s here in bf16 on an M5. On CUDA with `flash-linear-attention` installed it is fast. Use `jaredpalmer/kev-4b@qwen3` for low latency on Apple Silicon until an MLX path exists.
- Requires `transformers >= 5.17` (the `qwen3_5` architecture) and `peft >= 0.21`.
- Date arithmetic (`deadline` 0.55 vs Jev 0.93), knowledge (MMLU 0.70 vs 0.90) and noisy-label emotion (0.54 vs 0.59) remain the gap to Jev. Training data that renders day counts is the next experiment ([issue #8](https://github.com/jaredpalmer/kev/issues/8)).
- Out-of-domain probabilities are usable but not calibrated (raw ECE 0.130 dev, 0.102 test); temperature fitted in-domain does not transfer.
- 4B bf16 needs ~9 GB of GPU memory for serving; training took 56 min on one H100 (peak 24.6 GB).

## Training

Frozen suite `evals/v7/decision-v7`: 10,000 public records (1,000 per source), 896 policy minimal-pair records over nine template families, 1,680 records from 60 randomly generated rule structures in four rendering styles. Two epochs, LoRA r=16 α=32 on `q/k/v/o_proj`, `gate/up/down_proj`, `in_proj_qkv/z/a/b`, `out_proj`; pointer head from scratch; cross-entropy on the option distribution; lr 5e-5 (OneCycle), effective batch 8, bf16 autocast with fp32 master weights, gradient checkpointing; option permutation, none-of-the-above insertion, distractors, none minimal pairs on 25% of Choice records. No Jev outputs were used for training.

## Evaluation protocol

Development partitions select models; the locked test partition is read at most once per candidate (`runs/locked/kev-4b-q35/`). Every number carries suite hash, code hashes and git commit in `result.json`. Untrained-base baselines use zero-shot letter logits on the same items (`scripts/base_mmlu_probe.py`).

## Use

```bash
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8008      # KEV_DTYPE=bf16 on a Mac; slow on MPS, see limits
```

Any TypeSafe-compatible client works: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`.

## License

Apache-2.0 for the adapter and head; the Qwen3.5 base is Apache-2.0; datasets carry their own licenses.
