---
language: en
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.5-0.8B-Base
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
  - name: Kev-0.8B
    results:
      - task: { type: text-classification, name: typed decision (choice / noul / score) }
        dataset: { type: mixed, name: "decision-v7 development (1,204 records; ten trained public sources + programmatic policy data)" }
        metrics:
          - { type: accuracy, value: 0.829 }
          - { type: expected_calibration_error, value: 0.095, name: "ECE, raw probabilities" }
      - task: { type: text-classification, name: typed decision, out-of-domain }
        dataset: { type: mixed, name: "transfer-v4 development (764 records; six never-trained sources + held-out policy structures)" }
        metrics:
          - { type: accuracy, value: 0.643 }
          - { type: brier_score, value: 0.513 }
---

# Kev-0.8B

Kev-0.8B is a **decision model**: one document (the *state*) and a set of typed questions in, a probability distribution per question out, in one forward pass. No text generation. It is a LoRA adapter (r=16, 11.3M trainable parameters) plus a pointer head on `Qwen/Qwen3.5-0.8B-Base` (revision `dc7cdfe2`), serving TypeSafe's public `/v1/systemone` contract.

**The small member of the Kev family.** Same data and recipe as the 0.6B it replaces, on the Qwen3.5 base: in-distribution 0.829 (Kev-0.6B 0.801), out of domain 0.643 (0.620), and it is the first small Kev that learns any rule composition (held-out pairs 0.38 vs 0.08). Three seeds: transfer 0.622 / 0.634 / **0.643**; this checkpoint is seed 2, selected on the development partition (highest development accuracy). Out of domain it is still a sub-1B model: use Kev-4B for accuracy; use this one where memory rules the 4B out, and measure on your own data.

- Hub: `jaredpalmer/kev-0.8b` (this repo; trial `q35-08b/02-trial-2`)
- Code, suites, results, and the full research log: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev) — `PLAN_Qwen35.md`, `PLAN.md`, `runs/leaderboard.md`

## Results (same frozen items for every row)

| | Kev-0.6B (Qwen3) | **Kev-0.8B** | Kev-4B | Kev-9B | Jev |
|---|---|---|---|---|---|
| in-distribution accuracy (decision-v7 dev, 1,204 records) | 0.801 | **0.829** | 0.877 | 0.876 | 0.845 |
| out-of-domain accuracy (transfer-v4 dev, 764 records) | 0.620 | **0.643** | 0.794 | 0.812 | 0.857 |
| out-of-domain Brier | 0.536 | **0.513** | 0.316 | 0.291 | 0.211 |
| confident errors out of domain (p ≥ 0.9 and wrong) | 10.8% | 9.6% | 8.2% | 7.5% | 3.7% |
| coverage at ≤ 5% error (share of decisions automatable) | – | 0.17 | 0.54 | 0.53 | 0.70 |
| held-out policy structures, both siblings correct | 0.08 | **0.38** | 0.78 | 0.80 | 0.86 |
| option-order flip rate | 0.07 | 0.06 | 0.08 | 0.03 | 0.00 |
| none-option present, accuracy | 0.80 | 0.83 | 0.93 | 0.90 | – |

Per-source out-of-domain accuracy (Kev-0.8B / Jev): QNLI 0.82 / 0.93, SciQ 0.90 / 0.99, TweetEval-offensive 0.62 / 0.81, PAWS 0.59 / 0.79, MMLU 0.41 / 0.90, Emotion 0.54 / 0.59, authorization 0.90 / 1.00, deadline (3-level date arithmetic) 0.28 / 0.93, (A or B) and C 0.66 / 0.91, (A and B) or not C 0.62 / 0.97, if A then not B else C 0.72 / 0.78.

Paired against Kev-0.6B on the same items (record-clustered bootstrap): +5.7 pp [+1.2, +10.0] out of domain.

**Locked test, read once** (`runs/locked/kev-08b-q35-ungated/`): in-distribution **0.827** (Brier 0.258, ECE 0.096), out-of-domain **0.668** (Brier 0.473, ECE 0.164, confident errors 9.6%, held-out pairs 0.36). Kev-0.6B on the same test items: 0.808 / 0.642. This partition will not be read again for this checkpoint.

## Known limits

- **Out of domain it is a sub-1B model.** Knowledge (MMLU 0.41) and paraphrase (PAWS 0.59) are near the untrained base; the same recipe reaches 0.79 at 4B and 0.81 at 9B on these items.
- **Slow on a Mac for its size.** The DeltaNet kernels have no MPS implementation; a five-question request takes ~0.33 s in bf16 on an M5 (Kev-0.6B: 0.12 s). On CUDA with `flash-linear-attention` it is fast.
- Requires `transformers >= 5.17` and `peft >= 0.21`.
- Ordinal hedging on date arithmetic (`deadline` 0.28): collapses to the middle level.
- Confident-error rate out of domain is 9.6%; raw ECE 0.095 in-domain, 0.184 out of domain. Probabilities are usable in-domain; treat them as advisory elsewhere.

## Training

Frozen suite `evals/v7/decision-v7`: 10,000 public records (1,000 per source), 896 policy minimal-pair records over nine template families, 1,680 records from 60 randomly generated rule structures in four rendering styles. Two epochs, LoRA r=16 α=32 on attention, MLP and DeltaNet projections; pointer head from scratch; cross-entropy on the option distribution; lr 1e-4 (OneCycle), batch 8, bf16 autocast with fp32 master weights; option permutation, none-of-the-above insertion, distractors, none minimal pairs on 25% of Choice records; ~20 min on one H100. No Jev outputs were used for training.

## Evaluation protocol

Development partitions select models; the locked test partition is read at most once per candidate. Every number carries suite hash, code hashes and git commit in `result.json`.

## Use

```bash
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8008
```

Any TypeSafe-compatible client works: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`.

## License

Apache-2.0 for the adapter and head; the Qwen3.5 base is Apache-2.0; datasets carry their own licenses.
