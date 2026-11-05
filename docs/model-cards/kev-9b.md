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
  - bigcode/commitpackft
  - nvidia/Aegis-AI-Content-Safety-Dataset-2.0
  - davidheineman/consumer-finance-complaints-large
metrics:
  - accuracy
  - brier_score
  - expected_calibration_error
model-index:
  - name: Kev-9B
    results:
      - task: { type: text-classification, name: typed decisions, out-of-domain (locked test, read once) }
        dataset: { type: mixed, name: "transfer-v4 test: six never-trained public sources and held-out policy structures (656 questions)" }
        metrics:
          - { type: accuracy, value: 0.852 }
          - { type: brier_score, value: 0.199 }
      - task: { type: text-classification, name: typed decisions, held-out public datasets (test) }
        dataset: { type: mixed, name: "breadth-v1 test: 14 held-out public datasets (3,089 questions)" }
        metrics:
          - { type: accuracy, value: 0.698 }
      - task: { type: text-classification, name: typed decisions, skill records (test) }
        dataset: { type: mixed, name: "hard-v1 test (1,088 questions; programmatic labels, held-out templates)" }
        metrics:
          - { type: accuracy, value: 0.834 }
      - task: { type: text-classification, name: typed decisions, developer tooling (test) }
        dataset: { type: mixed, name: "devtools-v1 test (1,071 questions; six public developer-tooling sources)" }
        metrics:
          - { type: accuracy, value: 0.791 }
      - task: { type: text-classification, name: typed decisions, real documents (test) }
        dataset: { type: mixed, name: "documents-v1 test (936 questions on CFPB complaint narratives)" }
        metrics:
          - { type: accuracy, value: 0.900 }
---

# Kev-9B

## Model summary

Kev-9B is a decision model. It reads one document (the *state*) and a set of typed questions about it, and returns a calibrated probability distribution over the options supplied with each question, in a single forward pass and without generating text. It is intended for developers who classify, route, triage or check documents and who need probabilities that can be thresholded, for example to send uncertain cases to human review. It implements TypeSafe's public System One API (`POST /v1/systemone`), so the TypeSafe SDK works against it unchanged. It is a LoRA adapter and a pointer head on Qwen3.5-9B-Base and fits one 24 GB-class GPU. This card describes version 2, released on 2026-09-30 and included in Kev 1.0.

## Model details

| | |
|---|---|
| Developer | Jared Palmer ([github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)) |
| Model type | Decision model: a causal language-model backbone run prefill-only, with a pointer head over the options |
| Backbone | `Qwen/Qwen3.5-9B-Base` (revision `68c46c4b`): 32 layers, 24 Gated DeltaNet (linear attention) and 8 full attention, hidden size 4,096; frozen |
| Adapter | LoRA, rank 16, α 32, on the attention, MLP and DeltaNet projections (45.4M parameters) |
| Head | Pointer head: two projections score each option's closing token against the question's final token; a softmax gives the probabilities |
| Precision | Trained with bf16 autocast over fp32 weights; served in bf16 (the adapter is merged into the base at load time); evaluated in fp32 |
| Context | States of up to 65,536 tokens are served, plus at least 8,192 tokens per question. Training states were at most 7,552 tokens. |
| Validated context length | 8,192 tokens (see Long documents) |
| Calibration | One temperature, T = 2.19, stored in `head.pt` and applied at load time |
| Languages | English |
| License | Apache-2.0 (adapter and head); the base model is Apache-2.0 |
| Version | v2 (Kev 1.0): `main` of [`jaredpalmer/kev-9b`](https://huggingface.co/jaredpalmer/kev-9b), revision `b5d8c18e` (released 2026-09-30) |
| Previous version | v1, the same recipe without the documents and skills stage (T = 2.30), at tag [`v1`](https://huggingface.co/jaredpalmer/kev-9b/tree/v1); its card is the README at that tag |

**Input.** A state (text, or a JSON object or array rendered as labelled text) and any number of named questions, each of one of three types:

| Type | Options | Output |
|---|---|---|
| `choice` | 1–255 named options, each with an optional description | a probability per option, the most likely option and a confidence |
| `score` | 1–255 ordered levels | a probability per level and the expected level index |
| `noul` | yes / no, with optional descriptions | the probability of yes |

Each question is answered as its own row that continues from the shared state, so questions cannot influence one another; the state is computed once and cached.

## Intended uses

- Typed decisions over documents of a few thousand tokens: classification, routing, triage, extraction choices, policy and eligibility checks, and judging a proposed answer against stated criteria.
- Workflows that act on confidence: automate the confident cases and queue the rest, with thresholds frozen on a labelled sample of the user's own workload.
- A self-hosted, drop-in replacement for a System One endpoint on a single 24 GB-class GPU, and a starting point for fine-tuning on the user's own labels (`kev.train --init_from jaredpalmer/kev-9b`).

## Out-of-scope uses

- Text generation, chat, summarisation or open-ended question answering. The model only scores the options it is given.
- Fully automated decisions with legal, medical, financial, employment or similar consequences for people, without human review.
- Questions whose answer depends on facts not in the state and not general knowledge, and knowledge-heavy exams (see Limitations).
- Day-precision date arithmetic without the `KEV_DATE_FACTS=1` preprocessor, states longer than 65,536 tokens, and languages other than English.

## How to use

Serve it with the Kev repository. On CUDA it runs in bf16 with fused DeltaNet kernels and CUDA graphs (one L40S or H100; about 22 GB resident); on Apple Silicon the same command serves it through MLX, chosen automatically.

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-9b --port 8008          # v2, Kev 1.0 (this card)
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-9b@v1 --port 8008       # v1
```

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")
response = client.system_one(
    state="I was charged twice for order 1182. Please refund one of the charges.",
    questions={
        "team": Choice(instructions="Which team should handle this?",
                       criteria={"billing": "Charges and refunds", "shipping": "Deliveries", "returns": "Exchanges"}),
        "urgent": Noul(instructions="Does this need a reply today?"),
    },
)
print(response.choices["team"].choice, response.nouls["urgent"].noul)
```

The calibrated temperature is applied by default; `KEV_TEMPERATURE=1.0` returns the raw probabilities. `KEV_DTYPE=fp32` selects the exact path used for evaluation. `KEV_DATE_FACTS=1` appends the number of days between each pair of dates found in the state, which the model was trained to use. A state over 65,536 tokens is refused with a 422 that gives its token count.

## Training data

| Stage | Records | Content and labels |
|---|---|---|
| Base recipe (`decision-v7`) | 12,576 | 10,000 records from ten public classification datasets (1,000 each, listed in this card's metadata) with their native labels; 896 generated policy minimal pairs over nine template families; 1,680 records from 60 randomly generated rule structures in four renderings; labels computed by code |
| Dates and missing evidence | 1,425 | Generated: 900 date-bearing policy cases (plain, with a day-count sentence, or with a `date_facts` field); 255 cases with the deciding sentence removed and a uniform target, plus 270 intact controls |
| Documents and skills, one stage | 16,539 | `documents-v1` train: 5,219 US consumer-finance complaint narratives (CFPB, up to about 7k tokens) with 7,488 questions, labels kept where two open-weight teachers agreed with the consumer's own filing. `hard-v1` train: 6,000 programmatically labelled records in seven skill families (long policies, trade-offs, probability, multi-hop, dates and arithmetic, judging a proposed answer, missing-fact abstention), templates 0–3. `devtools-v1` train: 5,320 records from CodeReviewer, CommitPackFT, FlakeFlagger and Aegis with each dataset's own labels |

The two fine-tuning stages replay 2,000 and 10,000 records from `decision-v7`. No output of Jev (TypeSafe's hosted decision model) was used. CodeReviewer and FlakeFlagger come from Zenodo; the CFPB narratives are US government works; per-source licences and revisions are recorded in the suite manifests. The evaluation-only suites below (breadth-v1, tasksource-heldout-v1, transfer-v4, longdoc-v1, and the When2Call and prompt-injection sources of devtools-v1) never enter training.

## Training procedure

1. **Base recipe.** Two epochs on `decision-v7` from the base: LoRA rank 16, α 32; learning rate 5e-5, one-cycle schedule; effective batch 8 (4 × 2 accumulation); bf16 autocast, gradient checkpointing. The loss is cross-entropy over each question's options. Option order is shuffled, "none of the above" options and distractors are inserted at random, and a quarter of choice records also yield a minimal pair (the question with a "none of the above" option, once with the correct option present and once with it removed).
2. **Dates and missing evidence.** One epoch from stage 1 at learning rate 2e-5 with 2,000 replayed records. This is v1.
3. **Documents and skills.** One epoch from v1 on all three training sets together at learning rate 2e-5 with 10,000 replayed records; batch 2 × 4 accumulation; states of at most 7,552 tokens; seed 1; 3,318 optimizer steps.
4. **Calibration.** A single temperature, T = 2.19, minimising negative log-likelihood on 648 questions from held-out datasets: 448 from the calibration split of transfer-r3 (six public datasets, QNLI, SciQ, TweetEval-offensive, PAWS, MMLU and Emotion, plus two held-out families of generated policy records) and 200 MMLU-Pro questions (transfer-v9 development). The pool was checked against the checkpoint's training suites before the fit.

## Evaluation

**Methodology.** Every comparison is against Kev-9B v1 on identical items, each model at its own served temperature (v1: 2.30). The comparisons and their bars were registered before the confirmation reads; test partitions were read once for this checkpoint; the transfer-v4 test is locked (read once per candidate). Paired intervals are 95 % bootstraps that resample whole records (2,000 resamples), so questions sharing a state move together. Differences are in percentage points (pp). Jev (TypeSafe's hosted model, queried through Vercel AI Gateway) is shown where it was read on the same items. The suites:

- **breadth-v1**: 14 held-out public datasets in five areas (knowledge, language, retrieval, tools, arts), never trained on.
- **transfer-v4**: out-of-domain decisions from six never-trained public sources plus held-out policy and rule structures.
- **transfer-r3**: a short-state panel from the same eight held-out sources as the calibration pool.
- **hard-v1**: the skill families above; the test split holds out templates of trained generators.
- **devtools-v1**: developer-tooling decisions from six licence-checked sources (four trained, two evaluation-only).
- **documents-v1 / documents-v2**: CFPB complaint narratives; v2 is a private held-out test set.
- **longdoc-v1**: CUAD commercial contracts and generated agreement bundles, with states of 4k to 64k tokens.

devtools-v1 headline panels exclude two tasks whose labels the state does not determine (`flakeflagger`, commit change type); the same rows leave both sides.

**Results against v1 (registered confirmation).**

| Panel (questions) | Kev-9B v2 | Kev-9B v1 | Δ [95 % CI] |
|---|---|---|---|
| hard-v1 + devtools-v1 development, audited (1,855) | 0.821 | 0.628 | +19.3 [+17.2, +21.6] |
| documents-v1 development (920) | 0.902 | 0.833 | +7.0 [+4.8, +9.2] |
| Short states: transfer-v4 development + transfer-r3 test, without `emotion` (1,586) | 0.871 | 0.871 | +0.1 [−1.1, +1.2] |
| **hard-v1 + devtools-v1 test, audited (1,859)** | **0.822** | 0.635 | +18.7 [+16.7, +20.8] |
| **documents-v1 test (936)** | **0.900** | 0.829 | +7.1 [+4.7, +9.2] |
| **Out-of-domain, transfer-v4 locked test (656): accuracy** | **0.852** | 0.852 | +0.0 [−1.7, +1.8] |
| transfer-v4 locked test: served Brier | 0.199 | 0.224 | −0.025 [−0.047, −0.007] |

**Held-out data (never trained on).**

| Panel (questions) | Kev-9B v2 | Kev-9B v1 | Jev |
|---|---|---|---|
| breadth-v1 development, all 14 datasets (3,075) | 0.700 | 0.697 | 0.757 |
| **breadth-v1 test, all 14 datasets (3,089)** | **0.698** | 0.692 | 0.757 |
| breadth-v1 test, chance-corrected index¹ [95 % CI] | 41.0 [38.8, 43.9] | 40.0 [38.1, 43.0] | 54.0 [51.2, 57.0] |
| Out-of-domain, transfer-v4 development (656): accuracy / Brier | 0.820 / 0.262 | 0.822 / 0.264 | 0.857 / 0.211 |
| transfer-v4 locked test: ECE / confident errors (p ≥ 0.9 and wrong) / coverage at ≤ 5 % error | 0.034 / 1.4% / 0.742 | 0.042 / 3.2% / 0.645 | – |
| Short states, transfer-r3 test (1,150) | 0.847 | 0.847 | – |
| MMLU-Pro, 10 options (transfer-v9 development) | 0.590 | 0.515 | 0.840 |
| Unanswerable items answered with p ≥ 0.9 (lower is better) | 0.00 | 0.00 | 0.09 |

Against v1 the breadth-v1 accuracy difference is +0.3 [−0.7, +1.3] on development and +0.6 [−0.4, +1.6] on test. tasksource-heldout-v1 development has been read for this checkpoint but its read-out is not complete; it is not reported here.

**Trained families (held-out items and templates).**

| Panel (questions) | Kev-9B v2 | Kev-9B v1 | Jev |
|---|---|---|---|
| hard-v1 development (1,083) / test (1,088) | 0.813 / **0.834** | 0.574 / 0.584 | 0.777 / – |
| devtools-v1 development (1,072) / test (1,071), all sources | 0.772 / **0.791** | 0.631 / 0.637 | 0.713 / – |
| documents-v1 development (920) / test (936) | 0.902 / 0.900 | 0.833 / 0.829 | 0.868 / – |
| documents-v2, private held-out test (953) | 0.900 | 0.821 | – |
| decision-v7 development (1,264) / locked test (1,200) | 0.874 / 0.873 | 0.872 / – | 0.845 / – |
| Held-out domains of generated decisions, ood-v2 (4,988) | 0.889 | – | – |

Test differences against v1: hard-v1 +25.0 [+22.0, +28.1], devtools-v1 (all sources) +15.4 [+11.0, +19.4], documents-v2 +8.0 [+5.9, +10.2].

**Long documents.**

- Validated context length: 8,192 tokens, the trained length. The 16k bucket is outside the tolerance: its lower bound is −3.7 pp, below −3 pp, so no longer length is validated.
- Rule, fixed before the read: the validated length is the nominal size of the largest bucket from 16,384 tokens up such that it, and every bucket between it and 8,192, is within tolerance. Within tolerance means the CUAD accuracy difference from the 8k bucket (states of 6,553–7,618 tokens, the trained length), paired on the same contract, repeat and question, has a 95 % lower bound of at least −3 pp, and every record was answered. If the 16k bucket fails, the validated length is 8,192 tokens.

CUAD accuracy, ECE and the paired difference from the 8k bucket by nominal state length (longdoc-v1 development):

| Nominal state length | CUAD questions | Accuracy | ECE | Δ vs 8k, pp [95 % CI] |
|---|---|---|---|---|
| 4k | 443 | 0.876 | 0.049 | – |
| 8k | 453 | 0.850 | 0.051 | reference |
| 16k | 452 | 0.839 | 0.033 | −1.4 [−3.7, +0.9] |
| 32k | 454 | 0.819 | 0.041 | −3.6 [−6.4, −0.9] |
| 64k | 452 | 0.801 | 0.056 | −5.2 [−7.9, −2.5] |

ECE at the shipped T = 2.19. Δ is paired on the 445–447 questions asked about the same contracts at both lengths. The 4k bucket holds different contracts and is not a reference for the rule. Source: `runs/r28-readout/context.json` (round 28's registered read-out, `runs/r29-9b-r18a-longdoc`).

**Calibration** (expected calibration error, ECE, as served; lower is better):

| Panel | Kev-9B v2 (T = 2.19) | Kev-9B v1 (T = 2.30) |
|---|---|---|
| breadth-v1 test, all 14 datasets | 0.034 | 0.044 |
| transfer-v4 development / locked test | 0.041 / 0.034 | 0.042 / 0.042 |
| hard-v1 test | 0.054 | 0.075 |
| devtools-v1 test, all sources | 0.098 | 0.147 |
| documents-v1 test | 0.017 | 0.103 |

The temperature's 90 % bootstrap interval is [2.05, 2.41]. v1's 2.30 was fitted on development rows of its own training distribution; v2 is the first 9B served at a temperature fitted on held-out datasets.

**Other results.**

| Suite | Kev-9B v2 | Jev |
|---|---|---|
| Date arithmetic, `deadline` policy (transfer-v9 development) | 0.725 | 0.95 |
| MMLU, 4 options (transfer-v9 development) | 0.725 | 0.90 |
| SemIf (144 authored decisions; near saturation, reported only) | 0.917 | 0.965 |

**Serving.** CUDA, bf16 with fused kernels and CUDA graphs; model time per request (median of 20) for a new / repeated state, measured on v1 (the same architecture and adapter shape):

| GPU | 6 questions, short state | 5 questions, 2,200-token state | Requests/s, 64 clients |
|---|---|---|---|
| L40S | 66.4 / 42.7 ms | 235.6 / 57.5 ms | 32.7 |
| H100 | 24.0 / 16.6 ms | 88.5 / 26.4 ms | 79.5 |

Resident GPU memory is 21.9 GB. Served probabilities stay within 0.017 of the fp32 evaluation path on 280 questions, with no changed answers. On the fp32 evaluation path (H100, v2), states of 16k / 32k / 64k tokens take 5.1 / 10.9 / 25.4 s and 4.6 / 9.1 / 18.2 GiB above the weights.

Apple Silicon (MLX): the long-state measurements made for Kev-0.8B and Kev-4B have not been made at this size; on the 32 GB M5 used for them, the 19.3 GB bf16 backbone and its working set did not fit in the memory available.

¹ The community Decision Index 0.2's chance-corrected index: per dataset (score − chance) / (1 − chance), averaged within each area, then 100 × the mean of the five areas. Jev's index is from a separate read of the same test items.

## Limitations and trade-offs

- **Selection.** This checkpoint was trained and read on development data in an earlier round, and re-selected under a later rule with those numbers known. The test partitions and the locked read, each read once for it, are the guard; treat the development margins as optimistic.
- **Its large gains are in distribution.** The training splits of hard-v1, devtools-v1 and documents-v1 are in its training data; those gains are held-out items and templates of trained families, not transfer to new tasks.
- **It is not better than v1 on new work.** breadth-v1 test +0.6 pp [−0.4, +1.6], transfer-v4 development −0.2 pp [−2.4, +1.8], transfer-r3 test +0.0 pp [−1.2, +1.2]. Kev-27B leads it by 11 points on the breadth-v1 index, Jev by 13.
- **Knowledge is set by the base.** MMLU-Pro is 0.590, against Kev-27B's 0.675 and Jev's 0.840.
- **Date arithmetic is its weakest family**: 0.725 on the `deadline` policy questions against Jev's 0.95; the `KEV_DATE_FACTS=1` preprocessor helps.
- **Calibration.** devtools-v1 is its least calibrated suite (test ECE 0.098). The temperature's interval is [2.05, 2.41]. The temperature was written into `head.pt` from the registered pool fit with the reason recorded, because the calibration script cannot list the sources of the combined training file to re-check the pool itself; the round's own check had covered it. That check does not cover v1's dates and missing-evidence data, whose manifest lists no sources; those records are four generated families, none of them in the pool.
- **Untrained lengths.** Training states were at most 7,552 tokens. Longer states are served up to 65,536 tokens; how far accuracy holds is the validated context length above.
- **Option order** can change an answer; question isolation does not prevent this.

## Bias, risks and ethical considerations

- Calibrated probabilities can create unwarranted trust. The temperature was fitted on public held-out datasets and does not transfer to every workload; measure accuracy and calibration on a labelled sample of your own data, and refit the temperature there (`python -m kev.calibrate`), before setting thresholds.
- Accuracy and calibration shift under domain change. Monitor production error rates rather than relying on the numbers above.
- Do not use it for consequential automated decisions about people without human review. Biases of the base model and of the training data (including labels produced by other models) are not measured.
- States may contain personal or confidential data. Self-hosting keeps inputs on your own hardware; the server is open unless `KEV_API_KEY` is set, so apply your own access control and data-handling policy.

## Compute

- Base recipe and dates stage (v1): single NVIDIA H100 GPUs.
- Documents and skills stage: 2.6 hours on one NVIDIA H200 (peak 74.1 GB).
- Evaluation and serving checks: single H100 / H200 / L40S GPUs on Modal.

## Provenance and reproducibility

- Code, suites and evaluation reports: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev). Release numbers: `runs/release/kev-9b-r27.json` (`scripts/release_numbers.py --release kev-9b-r27`); registered rule and verdicts `experiments/rounds/r27.json`, `runs/r27-readout/`, `runs/r27-verdict/`; the 2026-09-30 family reads `runs/fam-9bnew-breadth/`, `runs/fam-9b-breadth/` and `runs/fam-breadth-test-report/`; ood-v2 `runs/r29-9b-r18a-ood/`; serving `runs/serve-9b-l40s/`, `runs/serve-9b-h100/`, `runs/long-state-9b-h100/`.
- Stages: base trial `q35-9b/01-trial-1` (tag `v7-base`); dates `night2-9b-du/00-trial-0` (v1, tag `v1`); documents and skills round 18 arm (a) `r18-9b/00-trial-0` (`experiments/round18/joint.json`), selected and confirmed as round 27's `9b-r18a`.
- Released weights: Hub commit `b5d8c18e`; adapter sha256 `2b2a70cf4ef4440b6c22899e1f72c2f8ea5c6f65b19aa344539b4b8971d1f13d`, `head.pt` sha256 `8e1dab2c…` (T = 2.1936).
- Verification: loaded anonymously from the Hub, it reproduced the pre-release evaluation's answers on 252 of 252 SemIf rows and 764 of 764 transfer-v4 development rows at T = 2.19 (`runs/rel9-public/`).

## Citation

```bibtex
@misc{palmer2026kev9b,
  title        = {Kev-9B: a calibrated decision model on Qwen3.5-9B},
  author       = {Palmer, Jared},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/jaredpalmer/kev-9b}},
  note         = {Version 2, released 2026-09-30; Kev 1.0}
}
```

## Contact

Questions and issues: [github.com/jaredpalmer/kev/issues](https://github.com/jaredpalmer/kev/issues).
