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
  - bigcode/commitpackft
  - nvidia/Aegis-AI-Content-Safety-Dataset-2.0
  - davidheineman/consumer-finance-complaints-large
metrics:
  - accuracy
  - brier_score
  - expected_calibration_error
model-index:
  - name: Kev-0.8B
    results:
      - task: { type: text-classification, name: typed decisions, out-of-domain (locked test, read once) }
        dataset: { type: mixed, name: "transfer-v4 test: six never-trained public sources and held-out policy structures (656 questions)" }
        metrics:
          - { type: accuracy, value: 0.697 }
          - { type: brier_score, value: 0.397 }
      - task: { type: text-classification, name: typed decisions, held-out public datasets (test) }
        dataset: { type: mixed, name: "breadth-v1 test: 14 held-out public datasets (3,089 questions)" }
        metrics:
          - { type: accuracy, value: 0.586 }
      - task: { type: text-classification, name: typed decisions, held-out task families (development) }
        dataset: { type: mixed, name: "tasksource-heldout-v1 development: 17 of 24 held-out task families (1,993 questions)" }
        metrics:
          - { type: accuracy, value: 0.515 }
      - task: { type: text-classification, name: typed decisions, real documents (test) }
        dataset: { type: mixed, name: "documents-v1 test (936 questions on CFPB complaint narratives)" }
        metrics:
          - { type: accuracy, value: 0.851 }
          - { type: brier_score, value: 0.244 }
      - task: { type: text-classification, name: typed decisions, skill records (test) }
        dataset: { type: mixed, name: "hard-v1 test (1,088 questions; programmatic labels, held-out templates)" }
        metrics:
          - { type: accuracy, value: 0.665 }
          - { type: brier_score, value: 0.460 }
      - task: { type: text-classification, name: typed decisions, developer tooling (test) }
        dataset: { type: mixed, name: "devtools-v1 test (1,071 questions; six public developer-tooling sources)" }
        metrics:
          - { type: accuracy, value: 0.637 }
          - { type: brier_score, value: 0.442 }
---

# Kev-0.8B

## Model summary

Kev-0.8B is a decision model. It reads one document (the *state*) and a set of typed questions about it, and returns a calibrated probability distribution over the options supplied with each question, in a single forward pass and without generating text. It implements TypeSafe's public System One API (`POST /v1/systemone`), so the TypeSafe SDK works against it unchanged. It is the smallest Kev: a LoRA adapter and a pointer head on Qwen3.5-0.8B-Base that runs on a laptop or a 4 GB GPU. It is intended where memory or cost rules out the larger sizes and the task resembles what it was trained on; out of domain it is markedly less accurate than Kev-4B. This card describes the checkpoint in Kev 1.0, first published on 2026-09-24.

## Model details

| | |
|---|---|
| Developer | Jared Palmer ([github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)) |
| Model type | Decision model: a causal language-model backbone run prefill-only, with a pointer head over the options |
| Backbone | `Qwen/Qwen3.5-0.8B-Base` (revision `dc7cdfe2`): 24 layers, 18 Gated DeltaNet (linear attention) and 6 full attention; frozen |
| Adapter | LoRA, rank 16, α 32, on the attention, MLP and DeltaNet projections (11.3M parameters) |
| Head | Pointer head: two projections score each option's closing token against the question's final token; a softmax gives the probabilities |
| Precision | Trained with bf16 autocast over fp32 weights; served in bf16 (the adapter is merged into the base at load time); evaluated in fp32 |
| Context | States of up to 65,536 tokens are served, plus at least 8,192 tokens per question. Training states were at most 7,552 tokens. |
| Validated context length | 8,192 tokens (see Long documents) |
| Calibration | One temperature, T = 2.35, stored in `head.pt` and applied at load time |
| Languages | English |
| License | Apache-2.0 (adapter and head); the base model is Apache-2.0 |
| Version | Kev 1.0: `main` of [`jaredpalmer/kev-0.8b`](https://huggingface.co/jaredpalmer/kev-0.8b), revision `9a45d25e` (published 2026-09-24) |
| Previous versions | Hub tags `night2-du-release` and `v7-base` |

**Input.** A state (text, or a JSON object or array rendered as labelled text) and any number of named questions, each of one of three types:

| Type | Options | Output |
|---|---|---|
| `choice` | 1–255 named options, each with an optional description | a probability per option, the most likely option and a confidence |
| `score` | 1–255 ordered levels | a probability per level and the expected level index |
| `noul` | yes / no, with optional descriptions | the probability of yes |

Each question is answered as its own row that continues from the shared state, so questions cannot influence one another; the state is computed once and cached.

## Intended uses

- Typed decisions close to its training families (document classification, routing, policy checks over stated rules) on hardware too small for Kev-4B: a laptop, an L4 or a 4 GB GPU.
- A starting point for fine-tuning on the user's own labels where training cost matters (`kev.train --init_from jaredpalmer/kev-0.8b`).
- Prototyping against the System One API before moving to a larger Kev.

## Out-of-scope uses

- Text generation, chat, summarisation or open-ended question answering. The model only scores the options it is given.
- Tool-call routing (whether to call a tool, ask for a missing parameter or decline): its When2Call accuracy is below chance (see Limitations).
- Fully automated decisions with legal, medical, financial, employment or similar consequences for people, without human review.
- Knowledge-heavy questions, date arithmetic, states longer than 65,536 tokens, and languages other than English.

## How to use

Serve it with the Kev repository. On CUDA it runs in bf16 with fused DeltaNet kernels and CUDA graphs (an L4 is enough); on Apple Silicon the same command serves it through MLX, chosen automatically.

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8008        # Kev 1.0 (this card)
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b@v1.0 --port 8008   # the same weights, pinned
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

The calibrated temperature is applied by default; `KEV_TEMPERATURE=1.0` returns the raw probabilities. `KEV_DTYPE=fp32` selects the exact path used for evaluation. A state over 65,536 tokens is refused with a 422 that gives its token count.

## Training data

| Stage | Records | Content and labels |
|---|---|---|
| Base recipe (`decision-v7`) | 12,576 | 10,000 records from ten public classification datasets (1,000 each, listed in this card's metadata) with their native labels; 896 generated policy minimal pairs over nine template families; 1,680 records from 60 randomly generated rule structures in four renderings; labels computed by code |
| Dates and missing evidence | 1,425 | Generated: 900 date-bearing policy cases (plain, with a day-count sentence, or with a `date_facts` field); 255 cases with the deciding sentence removed and a uniform target, plus 270 intact controls |
| Documents and skills, one stage | 16,539 | `documents-v1` train: 5,219 US consumer-finance complaint narratives (CFPB, up to about 7k tokens) with 7,488 questions, labels kept where two open-weight teachers agreed with the consumer's own filing. `hard-v1` train: 6,000 programmatically labelled records in seven skill families (long policies, trade-offs, probability, multi-hop, dates and arithmetic, judging a proposed answer, missing-fact abstention), templates 0–3. `devtools-v1` train: 5,320 records from CodeReviewer, CommitPackFT, FlakeFlagger and Aegis with each dataset's own labels |

The two fine-tuning stages replay 2,000 and 6,000 records from `decision-v7`. No output of Jev (TypeSafe's hosted decision model) was used. CodeReviewer and FlakeFlagger come from Zenodo; the CFPB narratives are US government works; per-source licences and revisions are recorded in the suite manifests. The evaluation-only suites below (breadth-v1, tasksource-heldout-v1, transfer-v4, longdoc-v1, and the When2Call and prompt-injection sources of devtools-v1) never enter training.

## Training procedure

1. **Base recipe.** Two epochs on `decision-v7` from the base: LoRA rank 16, α 32; learning rate 1e-4, one-cycle schedule; batch 8; bf16 autocast; seed 2. The loss is cross-entropy over each question's options. Option order is shuffled, "none of the above" options and distractors are inserted at random, and a quarter of choice records also yield a minimal pair (the question with a "none of the above" option, once with the correct option present and once with it removed).
2. **Dates and missing evidence.** One epoch from stage 1 at learning rate 4e-5 with 2,000 replayed records.
3. **Documents and skills.** One epoch from stage 2 on all three training sets together at learning rate 2e-5 with 6,000 replayed records; batch 4 × 2 accumulation; states of at most 7,552 tokens; gradient checkpointing; seed 1; 2,818 optimizer steps.
4. **Calibration.** A single temperature, T = 2.35, minimising negative log-likelihood on the stage-3 trial's `decision-v7` development rows (1,264 questions). These are held-out items of a training corpus. A refit on held-out datasets was evaluated and not adopted (see Calibration).

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

| Panel (questions) | Kev-0.8B | Jev |
|---|---|---|
| Held-out public datasets, breadth-v1 development, audited, 10 datasets (2,475) | 0.653 | – |
| breadth-v1 development, all 14 datasets (3,075) | 0.597 | 0.757 |
| **breadth-v1 test, all 14 datasets (3,089)** | **0.586** | 0.757 |
| breadth-v1 test, chance-corrected index² [95 % CI] | 23.3 [21.2, 25.9] | 54.0 [51.2, 57.0] |
| Held-out task families, tasksource-heldout-v1 development, audited, 17 families (1,993) | 0.515 | – |
| tasksource-heldout-v1 development, all 24 families (2,788) | 0.513 | – |
| Out-of-domain, transfer-v4 development (656): accuracy / Brier | 0.648 / 0.430 | 0.857 / 0.211 |
| **Out-of-domain, transfer-v4 locked test (656): accuracy / Brier** | **0.697 / 0.397** | – |
| transfer-v4 locked test: ECE / coverage at ≤ 5 % error | 0.045 / 0.274 | – |
| MMLU-Pro, 10 options (transfer-v9 development) | 0.230 | 0.840 |
| Unanswerable items answered with p ≥ 0.9 (lower is better) | 0.00 | 0.09 |

**Trained families (held-out items and templates).**

| Panel (questions) | Kev-0.8B | Jev |
|---|---|---|
| documents-v1 development (920) / test (936) | 0.842 / **0.851** | 0.868 / – |
| documents-v2, private held-out test (953) | 0.848 | – |
| hard-v1 development (1,083) / test (1,088) | 0.594 / **0.665** | 0.777 / – |
| devtools-v1 development, audited sources (772) | 0.633 | – |
| devtools-v1 development (1,072) / test (1,071), all sources | 0.602 / **0.637** | 0.713 / – |
| decision-v7 development (1,264) / locked test (1,200) | 0.827 / 0.838 | 0.845 / – |
| Held-out domains of generated decisions, ood-v2 (4,988) | 0.661 | – |

Jev's devtools-v1 figure is over all 1,074 development questions; Kev's rows drop a CodeReviewer id that the suite's builder reused for two records (2 questions).

**Against the previous version** (tag `night2-du-release`, at its own temperature 2.41; registered criteria, each test read once):

| Panel | Δ [95 % CI] |
|---|---|
| documents-v1 test | +24.4 [+21.3, +27.6] |
| documents-v2 | +23.2 [+19.9, +26.4] |
| hard-v1 test | +26.9 [+23.4, +30.4] |
| devtools-v1 test | +16.4 [+13.1, +19.4] |
| hard-v1 + devtools-v1 test, pooled | +21.7 [+19.5, +24.0] |
| transfer-v4 locked test | +1.2 [−1.1, +3.7] |

**Long documents.**

- Validated context length: 8,192 tokens, the trained length. The 16k bucket is outside the tolerance: its lower bound is −8.5 pp, below −3 pp, so no longer length is validated.
- Rule, fixed before the read: the validated length is the nominal size of the largest bucket from 16,384 tokens up such that it, and every bucket between it and 8,192, is within tolerance. Within tolerance means the CUAD accuracy difference from the 8k bucket (states of 6,553–7,618 tokens, the trained length), paired on the same contract, repeat and question, has a 95 % lower bound of at least −3 pp, and every record was answered. If the 16k bucket fails, the validated length is 8,192 tokens.

CUAD accuracy, ECE and the paired difference from the 8k bucket by nominal state length (longdoc-v1 development):

| Nominal state length | CUAD questions | Accuracy | ECE | Δ vs 8k, pp [95 % CI] |
|---|---|---|---|---|
| 4k | 443 | 0.779 | 0.128 | – |
| 8k | 453 | 0.711 | 0.066 | reference |
| 16k | 452 | 0.659 | 0.047 | −5.2 [−8.5, −2.1] |
| 32k | 454 | 0.663 | 0.055 | −6.0 [−9.5, −2.5] |
| 64k | 452 | 0.637 | 0.038 | −7.9 [−11.8, −4.2] |

ECE at the shipped T = 2.35. Δ is paired on the 445–447 questions asked about the same contracts at both lengths. The 4k bucket holds different contracts and is not a reference for the rule. Source: `runs/r28-readout/context.json` (round 28's registered read-out, `runs/r28-08b-r15-longdoc`).

**Calibration** (expected calibration error, ECE, at the shipped T = 2.35; lower is better):

| Panel | ECE |
|---|---|
| breadth-v1 development, audited / all 14 datasets | 0.022 / 0.032 |
| breadth-v1 test, all 14 datasets | 0.042 |
| tasksource-heldout-v1 development, audited | 0.096 |
| transfer-v4 development / locked test | 0.049 / 0.045 |
| hard-v1 development / test | 0.112 / 0.125 |
| devtools-v1 development, audited | 0.092 |
| documents-v1 development / test | 0.059 / 0.071 |
| decision-v7 development (the fitting rows) | 0.033 |
| ood-v2 | 0.123 |

The shipped temperature was fitted on held-out items of a training corpus, which the project's rules no longer allow for a new release. A registered refit on 648 questions from held-out datasets (the calibration split of transfer-r3, eight sources, and 200 MMLU-Pro questions) gives T = 2.52 (90 % bootstrap interval [2.19, 2.83]). On the 4,468 audited breadth-v1 and tasksource-heldout-v1 development questions it is better calibrated: ECE 0.037 against 0.048, Brier lower by 0.0020 [0.0014, 0.0025]. It is worse on the trained families, each by more than the registered tolerance of 0.005: hard-v1 ECE 0.124 against 0.112, devtools-v1 (audited) 0.099 against 0.092, documents-v1 0.078 against 0.059. The refit therefore did not qualify, and T = 2.35 stays. Users whose workload resembles held-out public datasets more than Kev's training families can serve it at the refit value with `KEV_TEMPERATURE=2.52`; answers do not depend on T.

**Other results.**

| Suite | Kev-0.8B | Jev |
|---|---|---|
| Date arithmetic, `deadline` policy (transfer-v9 development) | 0.35 | 0.95 |
| MMLU, 4 options (transfer-v9 development) | 0.425 | 0.90 |
| Held-out policy structures, both minimal-pair siblings correct (transfer-v4 development) | 0.422 | – |
| When2Call, development / test (evaluation-only source) | 0.167 / 0.133 | – |
| Prompt injection, development (evaluation-only source) | 0.547 | 0.893 |
| SemIf (144 authored decisions; near saturation for larger models, reported only) | 0.722 | 0.965 |
| JevBench public items, all 231 / hard tier 111 (ECE) | 0.636 / 0.360 (0.181) | – |

**Serving.** CUDA, bf16 with fused kernels and CUDA graphs, on an L4; model time per request (median of 20) for a new / repeated state: 22.7 / 16.1 ms for six questions about a short state, 108.6 / 32.3 ms for five questions about a 2,200-token state; 62.8 requests/s at 64 clients. Resident GPU memory is 3.8 GB. Served probabilities stay within 0.017 of the fp32 evaluation path on 280 questions, with 1 changed answer.

Apple Silicon (MLX, bf16, M5 with 32 GB; three questions, one about a fact planted at 60 % depth; the state is prefilled in 1,024-token chunks):

| State tokens | New state | Cached state | MLX peak (1.5 GB of weights) | Process footprint | Planted fact (p) |
|---|---|---|---|---|---|
| 8,192 | 1.4 s | 74 ms | 2.7 GB | 5.3 GB | right (0.68) |
| 16,384 | 3.2 s | 94 ms | 3.0 GB | 6.1 GB | right (0.68) |
| 32,768 | 8.0 s | 134 ms | 3.1 GB | 6.3 GB | right (0.70) |
| 65,000 | 21.2 s | 202 ms | 3.8 GB | 5.4 GB | right (0.62) |

Against fp32 PyTorch on the CPU on the same requests, the MLX answers differ by at most 0.0076 at 8k and 0.0052 at 16k tokens, with no changed answers. On 100 short-state questions the MLX path is within 0.020 of the fp32 evaluation path, with 1 changed answer.

¹ Excluded from audited panels: four breadth-v1 datasets (`routerbench`, whose states lack the information asked for; `cfcolor` and `humicroedit`, at chance for every system; `chessbench`, at the floor for every system); seven tasksource-heldout-v1 families with invalid or unrecoverable labels (names private); two devtools-v1 tasks whose labels the state does not determine (`flakeflagger`, commit change type).

² The community Decision Index 0.2's chance-corrected index: per dataset (score − chance) / (1 − chance), averaged within each area, then 100 × the mean of the five areas. Jev's index is from a separate read of the same test items.

## Limitations and trade-offs

- **It is a sub-1B model out of domain.** It trails Jev by 21 points on transfer-v4 development and by 31 points on the breadth-v1 index; knowledge (MMLU-Pro 0.230) and paraphrase are near the untrained base. Use Kev-4B where accuracy matters.
- **Its gains are in distribution.** The training splits of documents-v1, hard-v1 and devtools-v1 are in its training data; on JevBench's public hard tier, an out-of-distribution check, the documents-and-skills stage moved it +2.7 pp [−1.8, +7.2], not distinguishable from zero.
- **Tool-call routing got worse.** When2Call, an evaluation-only source, fell from 0.260 to 0.167 on development and from 0.233 to 0.133 on test, below the one-in-four rate of guessing among its four options. Do not use it for tool-call routing.
- **One devtools-v1 result is unexplained.** FlakeFlagger moved 0.500 → 0.520 on development but 0.507 → 0.813 on test, on 150 questions per split; treat it as unexplained, not as a skill.
- **Date arithmetic is its weakest family**: 0.35 on the `deadline` policy questions against Jev's 0.95; the `KEV_DATE_FACTS=1` preprocessor helps the larger models more than this one.
- **Rule composition is weak**: both siblings of a held-out policy minimal pair are right 0.422 of the time.
- **Calibration is one in-distribution temperature** (see Calibration). It cannot reorder confidences: coverage at ≤ 5 % error out of domain is 0.145 on development against Jev's 0.70, so few decisions can be automated at a strict error budget.
- **Untrained lengths.** Training states were at most 7,552 tokens. Longer states are served up to 65,536 tokens; how far accuracy holds is the validated context length above.
- **Option order** can change an answer; question isolation does not prevent this.

## Bias, risks and ethical considerations

- Calibrated probabilities can create unwarranted trust. The temperature was fitted on development rows of the training distribution and does not transfer to every workload; measure accuracy and calibration on a labelled sample of your own data, and refit the temperature there (`python -m kev.calibrate`), before setting thresholds.
- Accuracy and calibration shift under domain change, and more at this size than at larger ones. Monitor production error rates rather than relying on the numbers above.
- Do not use it for consequential automated decisions about people without human review. Biases of the base model and of the training data (including labels produced by other models) are not measured.
- States may contain personal or confidential data. Self-hosting keeps inputs on your own hardware; the server is open unless `KEV_API_KEY` is set, so apply your own access control and data-handling policy.

## Compute

- Base recipe: about 20 minutes on one NVIDIA H100. Dates stage: 9 minutes.
- Documents and skills stage: 52 minutes on one NVIDIA H200 (peak 23.9 GB).
- Evaluation and serving checks: single H100 / H200 / L4 GPUs on Modal; MLX measurements on an Apple M5.

## Provenance and reproducibility

- Code, suites and evaluation reports: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev). Release numbers: `runs/release/kev-08b-r15.json` (`scripts/release_numbers.py --release kev-08b-r15`), the locked read `runs/locked/kev-08b-r15-ungated/`, the 2026-09-30 family reads `runs/fam-08b-breadth/`, `runs/fam-08b-breadthtest/` and `runs/fam-breadth-test-report/`, the calibration refit `runs/r28-readout/round28.json`, serving `runs/serve-08b-l4/`, `runs/mlx-long-states/`, `runs/mlx-full-0.8b/`.
- Stages: base trial `q35-08b/02-trial-2` (tag `v7-base`); dates `night2-08b-du2/00-trial-0` (tag `night2-du-release`); documents and skills round 15 `r15-08b/00-trial-0` (`experiments/round15/joint.json`, rule `experiments/rounds/r15.json`). Calibration refit: round 28 arm `08b-r15` (`experiments/rounds/r28.json`).
- Released weights: Hub revision `9a45d25e`; adapter sha256 `9b908623…`, `head.pt` sha256 `f400bd12…` (T = 2.3511).
- Release history: published 2026-09-24 as round 15's confirmed candidate; included unchanged in Kev 1.0. The record of how it was selected, including suites since retired as unsound (scienthoon, WANLI-v2, TypeSafe), is the README at Hub revision `9a45d25e` and `PLAN.md` at git tag `research-archive-2026-09-24`.

## Citation

```bibtex
@misc{palmer2026kev08b,
  title        = {Kev-0.8B: a calibrated decision model on Qwen3.5-0.8B},
  author       = {Palmer, Jared},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/jaredpalmer/kev-0.8b}},
  note         = {Kev 1.0}
}
```

## Contact

Questions and issues: [github.com/jaredpalmer/kev/issues](https://github.com/jaredpalmer/kev/issues).
