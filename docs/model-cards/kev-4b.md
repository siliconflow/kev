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
  - bigcode/commitpackft
  - nvidia/Aegis-AI-Content-Safety-Dataset-2.0
  - davidheineman/consumer-finance-complaints-large
metrics:
  - accuracy
  - brier_score
  - expected_calibration_error
model-index:
  - name: Kev-4B
    results:
      - task: { type: text-classification, name: typed decisions, out-of-domain (locked test, read once) }
        dataset: { type: mixed, name: "transfer-v4 test: six never-trained public sources and held-out policy structures (656 questions)" }
        metrics:
          - { type: accuracy, value: 0.838 }
          - { type: brier_score, value: 0.224 }
      - task: { type: text-classification, name: typed decisions, held-out public datasets (test) }
        dataset: { type: mixed, name: "breadth-v1 test: 14 held-out public datasets (3,089 questions)" }
        metrics:
          - { type: accuracy, value: 0.690 }
      - task: { type: text-classification, name: typed decisions, held-out task families (development) }
        dataset: { type: mixed, name: "tasksource-heldout-v1 development: 17 of 24 held-out task families (1,993 questions)" }
        metrics:
          - { type: accuracy, value: 0.677 }
      - task: { type: text-classification, name: typed decisions, skill records (test) }
        dataset: { type: mixed, name: "hard-v1 test (1,088 questions; programmatic labels, held-out templates)" }
        metrics:
          - { type: accuracy, value: 0.803 }
          - { type: brier_score, value: 0.278 }
      - task: { type: text-classification, name: typed decisions, developer tooling (test) }
        dataset: { type: mixed, name: "devtools-v1 test (1,071 questions; six public developer-tooling sources)" }
        metrics:
          - { type: accuracy, value: 0.756 }
          - { type: brier_score, value: 0.342 }
---

# Kev-4B

## Model summary

Kev-4B is a decision model. It reads one document (the *state*) and a set of typed questions about it, and returns a calibrated probability distribution over the options supplied with each question, in a single forward pass and without generating text. It is intended for developers who classify, route, triage or check documents and who need probabilities that can be thresholded, for example to send uncertain cases to human review. It implements TypeSafe's public System One API (`POST /v1/systemone`), so the TypeSafe SDK works against it unchanged. It is a LoRA adapter and a pointer head on Qwen3.5-4B-Base, small enough for one 24 GB GPU or a 32 GB Apple Silicon Mac. This card describes the checkpoint in Kev 1.0, first published on 2026-09-24.

## Model details

| | |
|---|---|
| Developer | Jared Palmer ([github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)) |
| Model type | Decision model: a causal language-model backbone run prefill-only, with a pointer head over the options |
| Backbone | `Qwen/Qwen3.5-4B-Base` (revision `1001bb4d`): 32 layers, 24 Gated DeltaNet (linear attention) and 8 full attention, hidden size 2,560; frozen |
| Adapter | LoRA, rank 16, α 32, on the attention, MLP and DeltaNet projections (33.8M parameters) |
| Head | Pointer head: two projections score each option's closing token against the question's final token; a softmax gives the probabilities |
| Precision | Trained with bf16 autocast over fp32 weights; served in bf16 (the adapter is merged into the base at load time); evaluated in fp32 |
| Context | States of up to 65,536 tokens are served, plus at least 8,192 tokens per question. Training states were at most 7,552 tokens. |
| Validated context length | 8,192 tokens (see Long documents) |
| Calibration | One temperature, T = 2.41, stored in `head.pt` and applied at load time |
| Languages | English |
| License | Apache-2.0 (adapter and head); the base model is Apache-2.0 |
| Version | Kev 1.0: `main` of [`jaredpalmer/kev-4b`](https://huggingface.co/jaredpalmer/kev-4b), revision `139fdd94` (published 2026-09-24) |
| Previous versions | Hub tags `r8-documents-release` (documents stage only), `night2-du-release`, `v7-base` and `qwen3` (the Qwen3-4B generation) |

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
- A self-hosted, drop-in replacement for a System One endpoint on modest hardware, and a starting point for fine-tuning on the user's own labels (`kev.train --init_from jaredpalmer/kev-4b`).

## Out-of-scope uses

- Text generation, chat, summarisation or open-ended question answering. The model only scores the options it is given.
- Fully automated decisions with legal, medical, financial, employment or similar consequences for people, without human review.
- Questions whose answer depends on facts not in the state and not general knowledge, and knowledge-heavy exams (see Limitations).
- Day-precision date arithmetic without the `KEV_DATE_FACTS=1` preprocessor, states longer than 65,536 tokens, and languages other than English.

## How to use

Serve it with the Kev repository. On CUDA it runs in bf16 with fused DeltaNet kernels and CUDA graphs (one L40S, H100 or any GPU with about 16 GB free); on Apple Silicon the same command serves it through MLX, chosen automatically.

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8008           # Kev 1.0 (this card)
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b@v1.0 --port 8008      # the same weights, pinned
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
| Real documents (`documents-v1` train) | 5,219 | US consumer-finance complaint narratives (CFPB, up to about 7k tokens) with 7,488 questions (product, main issue); labels kept where two open-weight teachers agreed with the consumer's own filing |
| Skills (`hard-v1` train) | 6,000 | Programmatically labelled records in seven families: long policy documents with exceptions and sublimits, trade-offs under stated priorities, probability and expected value, multi-hop reasoning, dates and arithmetic, judging a proposed answer, and missing-fact abstention; generator templates 0–3 |
| Developer tooling (`devtools-v1` train) | 5,320 | CodeReviewer (whether a reviewer commented on a hunk), CommitPackFT (commit type), FlakeFlagger (flaky tests) and Aegis (content safety), each with its dataset's own labels |

Each fine-tuning stage after the first replays records from `decision-v7` (2,000, 2,000 and 4,000). No output of Jev (TypeSafe's hosted decision model) was used. CodeReviewer and FlakeFlagger come from Zenodo; the CFPB narratives are US government works; per-source licences and revisions are recorded in the suite manifests. The evaluation-only suites below (breadth-v1, tasksource-heldout-v1, transfer-v4, longdoc-v1, and the When2Call and prompt-injection sources of devtools-v1) never enter training.

## Training procedure

1. **Base recipe.** Two epochs on `decision-v7` from the base: LoRA rank 16, α 32; learning rate 5e-5, one-cycle schedule; effective batch 8 (4 × 2 accumulation); bf16 autocast, gradient checkpointing; seed 2. The loss is cross-entropy over each question's options. Option order is shuffled, "none of the above" options and distractors are inserted at random, and a quarter of choice records also yield a minimal pair (the question with a "none of the above" option, once with the correct option present and once with it removed).
2. **Dates and missing evidence.** One epoch from stage 1 at learning rate 2e-5 with 2,000 replayed records.
3. **Real documents.** One epoch from stage 2 on `documents-v1` train at learning rate 2e-5 with 2,000 replayed records; batch 2 × 4 accumulation; states of at most 7,552 tokens.
4. **Skills.** One epoch from stage 3 on `hard-v1` and `devtools-v1` train together at learning rate 2e-5 with 4,000 replayed records; batch 2 × 4 accumulation; states of at most 7,552 tokens; seed 1; 1,915 optimizer steps.
5. **Calibration.** A single temperature, T = 2.41, minimising negative log-likelihood on the stage-4 trial's `decision-v7` development rows (1,264 questions). These are held-out items of a training corpus. A refit on held-out datasets was evaluated and not adopted (see Calibration).

## Evaluation

**Methodology.** Every number is the fp32 evaluation path at the shipped temperature unless stated. Development partitions were used for selection; test partitions were read once for this checkpoint; the transfer-v4 test is locked (read once per candidate) and was judged against a bar fixed in advance. Paired intervals are 95 % bootstraps that resample whole records (2,000 resamples), so questions sharing a state move together. Differences are in percentage points (pp). Jev (TypeSafe's hosted model, queried through Vercel AI Gateway) is shown where it was read on the same items. The suites:

- **breadth-v1**: 14 held-out public datasets in five areas (knowledge, language, retrieval, tools, arts), never trained on.
- **tasksource-heldout-v1**: 24 whole task families of a public multi-task collection, never trained on (family names private).
- **transfer-v4**: out-of-domain decisions from six never-trained public sources (QNLI, SciQ, TweetEval-offensive, PAWS, MMLU, Emotion) plus held-out policy and rule structures.
- **hard-v1**: the skill families above; the test split holds out templates of trained generators.
- **devtools-v1**: developer-tooling decisions from six licence-checked sources (four trained, two evaluation-only).
- **documents-v1 / documents-v2**: CFPB complaint narratives; v2 is a private held-out test set.
- **longdoc-v1**: CUAD commercial contracts and generated agreement bundles, with states of 4k to 64k tokens.

Headline panels marked "audited" exclude items that a label audit found unsound¹; every exclusion removes the same rows from both sides of a comparison.

**Held-out data (never trained on).**

| Panel (questions) | Kev-4B | Jev |
|---|---|---|
| Held-out public datasets, breadth-v1 development, audited, 10 datasets (2,475) | 0.768 | – |
| breadth-v1 development, all 14 datasets (3,075) | 0.696 | 0.757 |
| **breadth-v1 test, all 14 datasets (3,089)** | **0.690** | 0.757 |
| breadth-v1 test, chance-corrected index² [95 % CI] | 38.0 [35.5, 41.3] | 54.0 [51.2, 57.0] |
| Held-out task families, tasksource-heldout-v1 development, audited, 17 families (1,993) | 0.677 | – |
| tasksource-heldout-v1 development, all 24 families (2,788) | 0.632 | – |
| Out-of-domain, transfer-v4 development (656): accuracy / Brier | 0.817 / 0.243 | 0.857 / 0.211 |
| **Out-of-domain, transfer-v4 locked test (656): accuracy / Brier** | **0.838 / 0.224** | – |
| transfer-v4 locked test: ECE / confident errors (p ≥ 0.9 and wrong) / coverage at ≤ 5 % error | 0.017 / 1.5% / 0.701 | – |
| MMLU-Pro, 10 options (transfer-v9 development) | 0.565 | 0.840 |
| Unanswerable items answered with p ≥ 0.9 (lower is better) | 0.00 | 0.09 |

**Trained families (held-out items and templates).**

| Panel (questions) | Kev-4B | Jev |
|---|---|---|
| hard-v1 development (1,083) / test (1,088) | 0.786 / **0.803** | 0.777 / – |
| devtools-v1 development, audited sources (772) | 0.780 | – |
| devtools-v1 development (1,072) / test (1,071), all sources | 0.739 / **0.756** | 0.713 / – |
| documents-v1 development (920) / test (936) | 0.891 / 0.903 | 0.868 / – |
| decision-v7 development (1,264) / locked test (1,200) | 0.873 / 0.865 | 0.845 / – |
| Held-out domains of generated decisions, ood-v2 (4,988) | 0.864 | – |

Jev's devtools-v1 figure is over all 1,074 development questions; Kev's rows drop a CodeReviewer id that the suite's builder reused for two records (2 questions).

**Against the previous version** (the documents-stage checkpoint, tag `r8-documents-release`, at its own temperature 2.96; registered criteria, each test read once):

| Panel | Δ [95 % CI] |
|---|---|
| hard-v1 test | +26.3 [+23.3, +29.5] |
| devtools-v1 test | +13.4 [+10.1, +16.1] |
| hard-v1 + devtools-v1 test, pooled | +19.9 [+17.8, +21.8] |
| documents-v1 development | −0.3 [−1.5, +0.9] |
| transfer-v4 locked test | +0.3 [−1.8, +2.3] |

**Long documents.**

- Validated context length: 8,192 tokens, the trained length. The 16k bucket is outside the tolerance: its lower bound is −3.4 pp, below −3 pp, so no longer length is validated.
- Rule, fixed before the read: the validated length is the nominal size of the largest bucket from 16,384 tokens up such that it, and every bucket between it and 8,192, is within tolerance. Within tolerance means the CUAD accuracy difference from the 8k bucket (states of 6,553–7,618 tokens, the trained length), paired on the same contract, repeat and question, has a 95 % lower bound of at least −3 pp, and every record was answered. If the 16k bucket fails, the validated length is 8,192 tokens.

CUAD accuracy, ECE and the paired difference from the 8k bucket by nominal state length (longdoc-v1 development):

| Nominal state length | CUAD questions | Accuracy | ECE | Δ vs 8k, pp [95 % CI] |
|---|---|---|---|---|
| 4k | 443 | 0.847 | 0.047 | – |
| 8k | 453 | 0.837 | 0.048 | reference |
| 16k | 452 | 0.823 | 0.057 | −1.1 [−3.4, +1.2] |
| 32k | 454 | 0.788 | 0.022 | −5.8 [−9.0, −2.8] |
| 64k | 452 | 0.781 | 0.035 | −5.2 [−8.2, −2.0] |

ECE at the shipped T = 2.41. Δ is paired on the 445–447 questions asked about the same contracts at both lengths. The 4k bucket holds different contracts and is not a reference for the rule. Source: `runs/r28-readout/context.json` (round 28's registered read-out, `runs/r28-4b-r10-longdoc`).

**Calibration** (expected calibration error, ECE, at the shipped T = 2.41; lower is better):

| Panel | ECE |
|---|---|
| breadth-v1 development, audited / all 14 datasets | 0.021 / 0.028 |
| breadth-v1 test, all 14 datasets | 0.029 |
| tasksource-heldout-v1 development, audited | 0.042 |
| transfer-v4 development / locked test | 0.042 / 0.017 |
| hard-v1 development / test | 0.095 / 0.084 |
| devtools-v1 development, audited | 0.072 |
| documents-v1 development / test | 0.093 / 0.101 |
| decision-v7 development (the fitting rows) | 0.013 |
| ood-v2 | 0.084 |

The shipped temperature was fitted on held-out items of a training corpus, which the project's rules no longer allow for a new release. A registered refit on 648 questions from held-out datasets (the calibration split of transfer-r3, eight sources, and 200 MMLU-Pro questions) gives T = 2.30 (90 % bootstrap interval [2.05, 2.52]). On the 4,468 audited breadth-v1 and tasksource-heldout-v1 development questions it does not improve on the shipped value: Brier 0.368 at both, a difference of −0.0001 [−0.0005, +0.0003], and ECE 0.025 against 0.024. The rule required a Brier interval below zero and a lower ECE, so T = 2.41 stays. Answers do not depend on T.

**Other results.**

| Suite | Kev-4B | Jev |
|---|---|---|
| Date arithmetic, `deadline` policy (transfer-v9 development) | 0.65 | 0.95 |
| MMLU, 4 options (transfer-v9 development) | 0.725 | 0.90 |
| When2Call / prompt injection (devtools-v1 development, evaluation-only sources) | 0.660 / 0.753 | – / 0.893 |
| SemIf (144 authored decisions; near saturation, reported only) | 0.889 | 0.965 |
| JevBench public items, all 231 / hard tier 111 (ECE) | 0.758 / 0.541 (0.112) | – |

**Serving.** CUDA, bf16 with fused kernels and CUDA graphs; model time per request (median of 20) for a new / repeated state:

| GPU | 6 questions, short state | 5 questions, 2,200-token state | Requests/s, 64 clients |
|---|---|---|---|
| L40S | 41.5 / 27.7 ms | 145.2 / 43.0 ms | 51.4 |
| H100 | 18.1 / 12.9 ms | 89.4 / 22.5 ms | 100.8 |

Resident GPU memory is 14.3 GB. Served probabilities stay within 0.017 of the fp32 evaluation path on 280 questions, with no changed answers. On the fp32 evaluation path (H100), states of 16k / 32k / 64k tokens take 3.0 / 6.8 / 17.2 s and 4.3 / 8.6 / 17.2 GiB above the weights.

Apple Silicon (MLX, bf16, M5 with 32 GB; three questions, one about a fact planted at 60 % depth; the state is prefilled in 1,024-token chunks):

| State tokens | New state | Cached state | MLX peak (8.4 GB of weights) | Process footprint | Planted fact (p) |
|---|---|---|---|---|---|
| 8,192 | 6.6 s | 354 ms | 10.2 GB | 11.9 GB | right (0.97) |
| 16,384 | 14.0 s | 427 ms | 11.0 GB | 12.8 GB | right (0.95) |
| 32,768 | 30.5 s | 533 ms | 11.9 GB | 13.7 GB | right (0.96) |
| 65,000 | 84.5 s | 716 ms | 13.0 GB | 14.1 GB | right (0.94) |

On 60 short-state questions the MLX path is within 0.018 of the fp32 evaluation path, with no changed answers.

¹ Excluded from audited panels: four breadth-v1 datasets (`routerbench`, whose states lack the information asked for; `cfcolor` and `humicroedit`, at chance for every system; `chessbench`, at the floor for every system); seven tasksource-heldout-v1 families with invalid or unrecoverable labels (names private); two devtools-v1 tasks whose labels the state does not determine (`flakeflagger`, commit change type).

² The community Decision Index 0.2's chance-corrected index: per dataset (score − chance) / (1 − chance), averaged within each area, then 100 × the mean of the five areas. Jev's index is from a separate read of the same test items.

## Limitations and trade-offs

- **Its largest gains are in distribution.** The training splits of hard-v1, devtools-v1 and documents-v1 are in its training data, and a hard-v1 test item is a new template of a trained generator. On held-out datasets it trails Jev by 16 points on the breadth-v1 index, and on JevBench's public hard tier, an out-of-distribution check, the skills stage gained about a third as much as on hard-v1 (+9.0 pp [+2.7, +15.3] over 111 items).
- **Some devtools-v1 labels are proxies.** Before training, every model scored, including Jev, was near chance on CodeReviewer and FlakeFlagger; after training on those sources it reaches 0.633 and 0.693 on development, which may be the labelling heuristic being learned rather than the decision.
- **Knowledge is set by the base.** MMLU-Pro is 0.565 against Jev's 0.840.
- **Date arithmetic is its weakest family**: 0.65 on the `deadline` policy questions against Jev's 0.95. The `KEV_DATE_FACTS=1` preprocessor helps (on the earlier checkpoint this one descends from, 0.60 → 0.85); it was not re-measured on this checkpoint.
- **Calibration is one in-distribution temperature.** It is well calibrated on held-out datasets (breadth-v1 test ECE 0.029), less so on the trained skill and document families (ECE 0.084–0.101), and a single temperature cannot reorder confidences: coverage at ≤ 5 % error out of domain is 0.620 on development against Jev's 0.70.
- **Untrained lengths.** Training states were at most 7,552 tokens. Longer states are served up to 65,536 tokens; how far accuracy holds is the validated context length above.
- **Option order** can change an answer; question isolation does not prevent this.

## Bias, risks and ethical considerations

- Calibrated probabilities can create unwarranted trust. The temperature was fitted on development rows of the training distribution and does not transfer to every workload; measure accuracy and calibration on a labelled sample of your own data, and refit the temperature there (`python -m kev.calibrate`), before setting thresholds.
- Accuracy and calibration shift under domain change. Monitor production error rates rather than relying on the numbers above.
- Do not use it for consequential automated decisions about people without human review. Biases of the base model and of the training data (including labels produced by other models) are not measured.
- States may contain personal or confidential data. Self-hosting keeps inputs on your own hardware; the server is open unless `KEV_API_KEY` is set, so apply your own access control and data-handling policy.

## Compute

- Base recipe: about 56 minutes on one NVIDIA H100 (peak 24.6 GB). Dates stage: 9 minutes on one H100.
- Documents stage: 43 minutes on one NVIDIA H200. Skills stage: 1.4 hours on one H200 (peak 47.7 GB).
- Evaluation and serving checks: single H100 / H200 / L40S GPUs on Modal; MLX measurements on an Apple M5.

## Provenance and reproducibility

- Code, suites and evaluation reports: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev). Release numbers: `runs/release/kev-4b-r10.json` (`scripts/release_numbers.py --release kev-4b-r10`), the locked read `runs/locked/kev-4b-r10-ungated/`, the 2026-09-30 family reads `runs/fam-4b-breadth/`, `runs/fam-4b-breadthtest/`, `runs/fam-4b-docs1test/` and `runs/fam-breadth-test-report/`, the calibration refit `runs/r28-readout/round28.json`, serving `runs/serve-4b-l40s/`, `runs/grouping-4b-h100/`, `runs/long-state-4b-h100/`, `runs/mlx-long-states/`, `runs/mlx-full-4b/`.
- Stages: base trial `q35-4b-s23/00-trial-0` (tag `v7-base`); dates `night2-4b-du/00-trial-0` (tag `night2-du-release`); documents round 8 `r8-small/00-trial-0` (tag `r8-documents-release`); skills round 10 `r10-skills/00-trial-0` (`experiments/round10/skills.json`, rule `experiments/rounds/r10.json`). Calibration refit: round 28 arm `4b-r10` (`experiments/rounds/r28.json`).
- Released weights: Hub revision `139fdd94`; adapter sha256 `90e81735…`, `head.pt` sha256 `dd633435…` (T = 2.4061).
- Release history: published 2026-09-24 as round 10's confirmed candidate; included unchanged in Kev 1.0. The record of how it was selected, including suites since retired as unsound (scienthoon, WANLI-v2, TypeSafe), is the README at Hub revision `139fdd94` and `PLAN.md` at git tag `research-archive-2026-09-24`.

## Citation

```bibtex
@misc{palmer2026kev4b,
  title        = {Kev-4B: a calibrated decision model on Qwen3.5-4B},
  author       = {Palmer, Jared},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/jaredpalmer/kev-4b}},
  note         = {Kev 1.0}
}
```

## Contact

Questions and issues: [github.com/jaredpalmer/kev/issues](https://github.com/jaredpalmer/kev/issues).
