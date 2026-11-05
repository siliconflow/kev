---
language: en
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen3.8-27B
base_model_relation: finetune
pipeline_tag: text-classification
tags:
  - decision-model
  - calibration
  - full-weight-sft
  - weight-averaging
  - multiple-choice
  - typesafe
  - qwen3.8
datasets:
  - deepmind/aqua_rat
  - allenai/ai2_arc
  - coastalcph/lex_glue
  - allenai/cosmos_qa
  - tau/commonsense_qa
  - tasksource/esci
  - openai/gsm8k
  - nvidia/HelpSteer2
  - nvidia/HelpSteer3
  - hotpotqa/hotpot_qa
  - AmazonScience/massive
  - allenai/math_qa
  - openlifescienceai/medmcqa
  - pfb30/multi_woz_v22
  - sentence-transformers/natural-questions
  - allenai/openbookqa
  - google-research-datasets/poem_sentiment
  - allenai/qasc
  - allenai/quartz
  - allenai/social_i_qa
  - stanfordnlp/snli
  - ChilleD/StrategyQA
  - allenai/winogrande
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
  - name: Kev-27B
    results:
      - task: { type: text-classification, name: typed decisions, out-of-domain (locked test, read once) }
        dataset: { type: mixed, name: "transfer-v4 test: six never-trained public sources and held-out policy structures (656 questions)" }
        metrics:
          - { type: accuracy, value: 0.8887 }
          - { type: brier_score, value: 0.154 }
      - task: { type: text-classification, name: typed decisions, held-out public datasets (test) }
        dataset: { type: mixed, name: "breadth-v1 test: 10 of 14 held-out public datasets (2,489 questions)" }
        metrics:
          - { type: accuracy, value: 0.832 }
      - task: { type: text-classification, name: typed decisions, held-out task families (test) }
        dataset: { type: mixed, name: "tasksource-heldout-v1 test: 17 of 24 held-out task families (2,024 questions)" }
        metrics:
          - { type: accuracy, value: 0.795 }
      - task: { type: text-classification, name: typed decisions, skills, developer tooling and documents (test) }
        dataset: { type: mixed, name: "hard-v1 + devtools-v1 + documents-v1 test (2,795 questions)" }
        metrics:
          - { type: accuracy, value: 0.889 }
---

# Kev-27B

## Model summary

Kev-27B is a decision model. It reads one document (the *state*) and a set of typed questions about it, and returns a calibrated probability distribution over the options supplied with each question, in a single forward pass and without generating text. It is intended for developers who classify, route, triage or check documents of up to 64k tokens and who need probabilities that can be thresholded, for example to send uncertain cases to human review. It implements TypeSafe's public System One API (`POST /v1/systemone`), so the TypeSafe SDK works against it unchanged. This card describes version 2, released on 2026-09-30: a full-weight fine-tune of Qwen3.8-27B averaged with version 1.

## Model details

| | |
|---|---|
| Developer | Jared Palmer ([github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)) |
| Model type | Decision model: a causal language-model backbone run prefill-only, with a pointer head over the options |
| Backbone | `Qwen/Qwen3.8-27B` (revision `1d4bf0f2`, Qwen's post-trained release): 64 layers, 48 Gated DeltaNet (linear attention) and 16 full attention, hidden size 5,120; every backbone weight fine-tuned |
| Head | Pointer head: two 5,120 → 256 projections score each option's closing token against the question's final token; a softmax gives the probabilities |
| Parameters | 25.6B in the backbone (vision tower, LM head and multi-token-prediction layers are not loaded), 2.6M in the head |
| Precision | bf16 (a 51.3 GB checkpoint); served in bf16 |
| Context | States of up to 65,536 tokens are served, plus at least 8,192 tokens per question. Training states were at most 32,768 tokens. |
| Validated context length | 65,536 tokens (see Long documents) |
| Calibration | One temperature, T = 1.32, stored in `head.pt` and applied at load time |
| Languages | English |
| License | Apache-2.0 (weights and head); the base model is Apache-2.0 |
| Version | v2 (Kev 1.0), released 2026-09-30 on `main` of [`jaredpalmer/kev-27b`](https://huggingface.co/jaredpalmer/kev-27b) |
| Previous version | v1, a rank-16 LoRA adapter on the same base (T = 1.38), at tag [`v1-lora`](https://huggingface.co/jaredpalmer/kev-27b/tree/v1-lora); its card is the README at that tag |

**Input.** A state (text, or a JSON object or array rendered as labelled text) and any number of named questions, each of one of three types:

| Type | Options | Output |
|---|---|---|
| `choice` | 1–255 named options, each with an optional description | a probability per option, the most likely option and a confidence |
| `score` | 1–255 ordered levels | a probability per level and the expected level index |
| `noul` | yes / no, with optional descriptions | the probability of yes |

Each question is answered as its own row that continues from the shared state, so questions cannot influence one another; the state is computed once and cached.

## Intended uses

- Typed decisions over documents: classification, routing, triage, extraction choices, policy and eligibility checks, and judging a proposed answer against stated criteria.
- Workflows that act on confidence: automate the confident cases and queue the rest, with thresholds frozen on a labelled sample of the user's own workload.
- A self-hosted, drop-in replacement for a System One endpoint.

## Out-of-scope uses

- Text generation, chat, summarisation or open-ended question answering. The model only scores the options it is given.
- Fully automated decisions with legal, medical, financial, employment or similar consequences for people, without human review.
- Questions whose answer depends on facts not in the state and not general knowledge (the model cannot look anything up), and knowledge-heavy exams (see Limitations).
- States longer than 65,536 tokens, languages other than English, and hardware smaller than an 80 GB-class GPU or a Mac with about 96 GB of memory (see Limitations).

## How to use

Serve it with the Kev repository on one B200, H200 or H100 80 GB GPU. The weights take 51 GB and the server about 65.5 GB resident; a 64k-token state peaked at 87.1 GB on an H200 and a 32k-token state at 78.7 GB, so the longest states need more than 80 GB.

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-27b --port 8008          # v2 (this card)
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-27b@v1-lora --port 8008  # v1
```

On Apple Silicon the same command serves through MLX (chosen automatically), loading the full bf16 weights as saved, with nothing merged. This path is expected to work on a 96–128 GB Mac but has not yet been run at this size (see Limitations).

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

The calibrated temperature is applied by default; `KEV_TEMPERATURE=1.0` returns the raw probabilities. The server uses bf16, fused DeltaNet kernels and CUDA graphs; `KEV_DTYPE=fp32` selects the exact path used for evaluation.

## Training data

One epoch over a private corpus of 145,840 records (337,130 questions) with states of at most 32,768 tokens. Only its manifest is public (`evals/sft-v2-r22/manifest.json` in the GitHub repository).

| Component | Records | Content and labels |
|---|---|---|
| Kev corpus | 78,786 | Kev-27B v1's training set (ten public classification datasets, generated policy and rule records, date-arithmetic and missing-evidence cases, long states with buried facts); the training splits of Kev's skill (hard-v1), developer-tooling (devtools-v1) and consumer-complaint (documents-v1) suites; 24 public datasets capped at 500 records each; seven generated families (long documents, tool routing, retrieval, intent, rubric judging, abstention, numeric reasoning) |
| Licensed task families | 24,000 | 119 task families from a public multi-task collection, with licences that permit commercial use and native labels; the family list is not published |
| Long states | 3,200 | Kev-corpus states embedded in 8k–32k-token documents; labels inherited exactly |
| Long documents | 7,617 | Documents assembled by code, labels computed by code |
| Out-of-domain decisions | 7,312 | Generated decisions in varied domains |
| Tone | 7,544 | Minimal pairs of the same text written calm, frustrated or angry |
| Prompt injection | 2,699 | Recognising indirect prompt injection (defensive) |
| Agent sessions | 4,875 | Questions about code-generated agent session logs |
| PII | 4,152 | Classifying personal data inserted by code (all of it fictitious) |
| Grounding | 5,655 | Whether a claim is supported by a document |

**Sources and labels.** The public datasets are listed in this card's metadata; two developer-tooling sources, CodeReviewer and FlakeFlagger, come from Zenodo, and CodeReviewer diffs are kept only from permissively licensed projects. Generated text and labels come from code or from open-weight models (GLM-5.3, DeepSeek-V4-Pro, Inkling, Mistral Large 3, gpt-oss-120b, MiMo-V2.6-Pro). The consumer-complaint labels come from open-weight models, filtered by closed-model judges and adjudication. No output of Jev (TypeSafe's hosted decision model) was used.

**Licences.** Four of the capped public datasets are share-alike: ARC (CC-BY-SA-4.0), HotpotQA (CC-BY-SA-4.0), Natural Questions (CC-BY-SA-3.0) and SNLI (CC-BY-SA-4.0); the others are CC-BY-4.0, MIT or Apache-2.0. The CFPB complaint narratives are US government works. GLM-5.3's licence is MIT-style with a condition on very large model-as-a-service operators. Per-source licences, revisions and attributions are recorded in the component manifests.

**Contamination screening.** Before training, every record was screened against 102 evaluation partitions (76,549 reference items): every frozen Kev suite, the private evaluation set, JevBench's public items and the evaluation-only suites below. A record was removed on an exact normalised string match, word 8-gram Jaccard similarity above 0.2, or containment of at least 0.5; 6,388 training records were removed.

## Training procedure

1. **Full-weight fine-tuning.** One epoch from `Qwen/Qwen3.8-27B` on 8 NVIDIA H200 GPUs (FSDP2). AdamW (β 0.9 / 0.999, weight decay 0.01) with fp32 master weights and moments over a bf16 backbone; learning rate 2e-6 for the backbone and 1e-4 for the head, one-cycle schedule with 10 % warm-up; gradient norm clipped at 1.0; 128 records per optimizer step (8 per GPU, 2 accumulation steps), 1,140 steps; seed 0. The loss is cross-entropy over each question's options (soft targets where the data has them). Option order is shuffled and "none of the above" options and distractors are inserted at random. A quarter of choice records with states of at most 8,192 tokens also yield a minimal pair: the question with a "none of the above" option, once with the correct option present and once with it removed. Each state is run once and its questions branch from it.
2. **Weight averaging.** Every backbone tensor is 0.85 × the fine-tuned weight + 0.15 × v1's weight (v1's LoRA adapter merged into the base in fp32), computed in fp32 and rounded once to bf16. The fine-tuned model's pointer head is kept. The ratio was chosen among six blends (0.85 / 0.70 / 0.50, with either head) on development data.
3. **Calibration.** A single temperature, T = 1.32, minimising negative log-likelihood on 648 questions from held-out datasets that neither parent was trained on: 448 from the calibration split of transfer-r3 (six public datasets, QNLI, SciQ, TweetEval-offensive, PAWS, MMLU and Emotion, plus two held-out families of generated policy records) and 200 MMLU-Pro questions (transfer-v9 development). No partition of any training corpus was used, and a check found no overlap with the training data.

## Evaluation

**Methodology.** Every comparison is against Kev-27B v1 on identical items, each model at its own served temperature. Test partitions were read once for this checkpoint; the transfer-v4 test is locked (read once per candidate) and was judged against a bar fixed in advance (accuracy ≥ 0.886, Brier ≤ 0.165). Intervals are 95 % paired bootstraps that resample whole records (2,000 resamples), so questions sharing a state move together. Differences are in percentage points (pp). The suites:

- **breadth-v1**: 14 held-out public datasets in five areas (knowledge, language, retrieval, tools, arts), never trained on.
- **tasksource-heldout-v1**: 24 whole task families of the same multi-task collection as the licensed component, held out of training.
- **hard-v1**: programmatically labelled skill records (long policies, trade-offs, probability, multi-hop, dates and numbers, judging, abstention); the test split holds out templates of trained generators.
- **devtools-v1**: developer-tooling decisions from licence-checked public sources (code review, commit messages, safety, flaky tests).
- **documents-v1 / documents-v2**: US consumer-finance complaint narratives (CFPB); v2 is a private held-out test set.
- **transfer-v4**: out-of-domain decisions from six never-trained public sources plus held-out policy structures.
- **longdoc-v1**: CUAD commercial contracts of up to 64k tokens, and generated agreement bundles.

Headline panels exclude items that a label audit, completed before this checkpoint was built, found unsound¹; every exclusion removes the same rows from both models.

**Results against v1 (test partitions).**

| Panel (questions) | Kev-27B v2 | Kev-27B v1 | Δ [95 % CI] |
|---|---|---|---|
| Held-out public datasets, breadth-v1, 10 datasets (2,489) | 0.832 | 0.820 | +1.2 [+0.3, +2.2] |
| Held-out task families, tasksource-heldout-v1, 17 families (2,024) | 0.795 | 0.743 | +5.3 [+3.7, +6.8] |
| Skills, developer tooling and documents, pooled (2,795) | 0.889 | 0.800 | +8.9 [+7.5, +10.3] |
| &nbsp;&nbsp;hard-v1 (1,088) | 0.918 | 0.749 | +16.9 [+14.2, +19.8] |
| &nbsp;&nbsp;devtools-v1, audited sources (771) | 0.825 | 0.789 | +3.6 [+1.3, +5.9] |
| &nbsp;&nbsp;documents-v1 (936) | 0.908 | 0.869 | +4.0 [+2.1, +5.9] |
| documents-v2, private held-out (953) | 0.921 | 0.881 | +4.0 [+2.0, +6.1] |
| breadth-v1, all 14 datasets (3,089) | 0.757 | 0.748 | +0.8 [−0.1, +1.8] |
| **Out-of-domain, transfer-v4 locked test (656): accuracy** | **0.8887** | 0.8963 | −0.8 [−2.0, +0.5] |
| Out-of-domain, transfer-v4 locked test: Brier / coverage at ≤ 5 % error | 0.154 / 0.875 | 0.160 / 0.835 | – |

**Comparison with other decision models** on breadth-v1 test (all 14 datasets), scored with the chance-corrected index of the community Decision Index 0.2 (per dataset (score − chance) / (1 − chance), averaged within each area, then 100 × the mean of the five areas). Jev was queried through Vercel AI Gateway and AutoJev-27B (`denis-pplx/autojev-27b`, a full-weight fine-tune of the same base) through its own server, once each, on the same items.

| | Kev-27B v2 | Kev-27B v1 | Jev | AutoJev-27B |
|---|---|---|---|---|
| Index [95 % CI] | **52.3** [49.2, 55.4] | 50.2 [47.0, 53.2] | 54.0 [51.2, 57.0] | 50.0 [47.0, 53.3] |

Against v1 the index difference is +2.1 [−0.4, +4.6]. No paired interval against Jev was computed.

**Long documents** (CUAD contracts in longdoc-v1 test, accuracy / ECE by state length in tokens):

| State length (questions) | Kev-27B v2 | Kev-27B v1 |
|---|---|---|
| under 8k (867) | 0.874 / 0.059 | 0.900 / 0.025 |
| 8k–16k (443) | 0.880 / 0.052 | 0.892 / 0.028 |
| 16k–32k (442) | 0.873 / 0.061 | 0.882 / 0.019 |
| 32k–64k (442) | 0.867 / 0.060 | 0.876 / 0.014 |
| all (2,194) | 0.874 / 0.053 | 0.890 / 0.007 |

Accuracy difference over all lengths: −1.6 [−3.0, −0.3]. On the generated agreement bundles (2,400 questions) both models score 1.000.

**Context length.** Validated context length: 65,536 tokens, the serving limit, from longdoc-v1 development (paired CUAD accuracy difference from the 8k bucket, pp [95 % CI]: 16k +0.2 [−0.7, +1.2], 32k −0.2 [−1.2, +0.7], 64k −1.1 [−2.4, +0.0]; 445–447 questions each). The rule is the one applied to every Kev 1.0 size: the validated length is the nominal size of the largest bucket from 16,384 tokens up such that it, and every bucket between it and 8,192, is within tolerance, meaning the CUAD accuracy difference from the 8k bucket, paired on the same contract, repeat and question, has a 95 % lower bound of at least −3 pp, with every record answered.

**Calibration** (expected calibration error, ECE, as served; lower is better):

| Panel | Kev-27B v2 | Kev-27B v1 |
|---|---|---|
| breadth-v1 test, 10 datasets | 0.015 | 0.013 |
| tasksource-heldout-v1 test | 0.049 | 0.051 |
| hard-v1 + devtools-v1 + documents-v1 test | 0.014 | 0.027 |
| transfer-v4 locked test | 0.019 | 0.018 |
| CUAD contracts, longdoc-v1 test | 0.053 | 0.007 |

On the 648 fitting questions, a 5-fold group-disjoint cross-validation lowers ECE from 0.048 (raw) to 0.038 (out of fold); the two intervals overlap. The temperature's 90 % bootstrap interval is [1.20, 1.45]. At its ends the headline panels move in opposite directions: breadth-v1 ECE rises to 0.026 at T = 1.20 and tasksource-heldout-v1 ECE to 0.067 at T = 1.45.

**Other results.**

| Suite | Kev-27B v2 | Kev-27B v1 |
|---|---|---|
| transfer-v4 development, accuracy / Brier | 0.851 / 0.218 | 0.848 / 0.229 |
| Short states, transfer-r3 test (1,150) | 0.858 | 0.879 |
| devtools-v1 test, all sources (1,071) | 0.790 | 0.711 |
| decision-v7 development (v1's training distribution) | 0.865 | 0.866 |
| MMLU-Pro, 10 options (transfer-v9 development) | 0.675 | 0.665 |
| Unanswerable items answered with p ≥ 0.9 (lower is better) | 0.00 | 0.00 |
| SemIf (144) / WANLI-v2 (1,002) / TypeSafe (89 answered rows) | 0.965 / 0.756 / 0.854 | 0.972 / 0.745 / 0.865 |
| Held-out domains of trained generators, accuracy / ECE: ood-v2 (4,988) | 0.956 / 0.020 | 0.944 / 0.044 |
| &nbsp;&nbsp;agents-ood-v1 (2,084) | 0.988 / 0.032 | 0.967 / 0.137 |
| &nbsp;&nbsp;guardrails-ood-v1 (4,949) | 0.984 / 0.010 | 0.944 / 0.079 |

transfer-r3 test is a short-state panel from the same eight held-out sources as the calibration pool; decision-v7 is v1's training distribution; transfer-v9 holds MMLU-Pro and items whose deciding evidence was removed. SemIf (hand-authored decisions from the SemIf project), WANLI-v2 (natural-language inference pairs from the WANLI test split) and TypeSafe (SemIf's selection of TypeSafe workflow cases) are reported only: the label audit found them too small, saturated or noisy to rank models. WANLI-v2 and TypeSafe were retired as evaluations on 2026-09-30 (about a quarter of the WANLI pairs were labelled differently by its two annotators, with the gold set to one of the two labels; TypeSafe's gold is the averaged answer of two closed frontier models, on too few questions to tell checkpoints apart); their figures are kept as the record.

**Serving parity** (H200, bf16 with fused kernels and CUDA graphs, 200 decision-v7 development records / 280 questions):

| Check | Result |
|---|---|
| Served vs the fp32 evaluation path, max \|Δp\| | 0.0223, no changed answers |
| One question alone vs the full request, max \|Δp\| | 0.0039, no changed answers |
| Served vs evaluation at 8k / 32k / 64k-token states, max \|Δp\| | 0.0064 / 0.0095 / 0.0017, no changed answers |
| Model time at a 64k-token state, new / cached state | 9.4 s / 733 ms |
| Resident memory / load time from a warm cache | 65.5 GB / 17.6 s |
| Throughput at 1 / 64 concurrent clients | 21.1 / 36.4 requests/s |

¹ Excluded from the headline panels: four breadth-v1 datasets (`routerbench`, whose states lack the information asked for; `cfcolor` and `humicroedit`, at chance for every system; `chessbench`, at the floor for every system); seven tasksource-heldout-v1 families with invalid or unrecoverable labels (names private); and two devtools-v1 tasks whose labels the state does not determine (`flakeflagger`, commit change type). The transfer-r3 `emotion` source (distant keyword labels) is excluded from the short-state panel in Limitations.

## Limitations and trade-offs

- **Selection.** The released checkpoint was chosen after an earlier candidate's test results were known and was confirmed on the same test sets. Treat the test margins as optimistic.
- **No gain on short states.** It is not better than v1 on short inputs: −0.8 pp [−2.0, +0.5] on the locked transfer-v4 test, −0.9 pp [−2.0, +0.1] on the short-state development panel (transfer-v4 development and transfer-r3 test without `emotion`), and −2.1 pp [−3.5, −0.8] on the whole transfer-r3 test.
- **Worse and overconfident on long contracts.** On CUAD it is 1.6 pp less accurate than v1 and its ECE is 2 to 4 times v1's at every length (0.053 against 0.007 overall). CUAD's questions include some wrong or disputed gold labels, but the gap is consistent across lengths. For contract review, refit the temperature on labelled documents of your own (`python -m kev.calibrate`) or use v1.
- **In-distribution gains.** The training splits of hard-v1, devtools-v1 and documents-v1 are in the training data, and the ood-v2, agents-ood-v1 and guardrails-ood-v1 suites are held-out domains of generators that also produced training data. Gains there measure held-out items of trained families, not transfer to new tasks.
- **Unknown base training.** The base is Qwen's post-trained release; its training data is not known, so overlap between it and any evaluation cannot be ruled out.
- **Knowledge.** Knowledge is set by the base: MMLU-Pro is 0.675, against Jev's 0.840 on the same items.
- **Untrained lengths.** States of 32k–64k tokens were evaluated (longdoc-v1) but not trained on.
- **Retired suite.** A support-ticket suite used to select v1 (scienthoon) was retired as unsound before v2 was built, so v2 has no result on it. v2's unaveraged fine-tuned parent scored 5.5 pp [−7.8, −3.2] below v1 there.
- **Temperature uncertainty.** The temperature's interval ([1.20, 1.45]) moves test ECE by up to about 0.02.
- **Hardware.** It needs an 80 GB-class data-centre GPU (more than 80 GB for the longest states). Apple Silicon via MLX: v2's full bf16 weights need about 51 GB plus working memory, so a 64 GB Mac is borderline and a 96–128 GB Mac should fit. This is expected from measurements on smaller models, not yet measured on a large Mac: loading a full-weight Kev-4B through the same path peaked at the size of its weights (8.4 GB), and its answers were within 0.015 of the fp32 evaluation path (`runs/mlx-full-4b`). v1 (the LoRA adapter, tag `v1-lora`) was served through MLX on a 128 GB M5 Max by an external contributor ([PR #175](https://github.com/jaredpalmer/kev/pull/175)): 0.849 accuracy on transfer-v4 development against the published 0.848, 52 GB steady and 97 GB at peak while the adapter was merged.

## Bias, risks and ethical considerations

- Calibrated probabilities can create unwarranted trust. The temperature was fitted on public held-out datasets and does not transfer to every workload; measure accuracy and calibration on a labelled sample of your own data before setting thresholds.
- Accuracy and calibration shift under domain change (for example on long contracts). Monitor production error rates rather than relying on the numbers above.
- Do not use it for consequential automated decisions about people without human review. Biases of the base model and of the training data (including labels produced by other models) are not measured.
- States may contain personal or confidential data. Self-hosting keeps inputs on your own hardware; the server is open unless `KEV_API_KEY` is set, so apply your own access control and data-handling policy.

## Compute

- Fine-tuning: 8 × NVIDIA H200 for 16.2 h of training time (129 GPU-hours), excluding restarts and evaluation.
- Weight averaging: about 6 minutes on CPU.
- Evaluation and serving checks: single H200 GPUs on Modal.

## Provenance and reproducibility

- Code, suites and evaluation reports: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev). Release numbers: `runs/release/kev-27b-r23.json` (`scripts/release_numbers.py --release kev-27b-r23`).
- Fine-tune: round 22 trial `r22-27b-lr2e6/00-trial-0` (`experiments/round22/lr2e6.json`), weights sha256 `3fa0182a…`. Blend: round 23 arm `27b-k-w85` (`scripts/interpolate_checkpoint.py --toward`; `interpolation.json` in the Hub repository), selection rule and confirmation stages in `experiments/rounds/r23.json`; results in `runs/r23-readout/`, `runs/r23-verdict/`, `runs/r23-breadth-report/`, `runs/serving-27b-r23*/`.
- Blend source: v1 at `jaredpalmer/kev-27b@01b81998` (trial `r6-27b-v2/01-trial-1`), now tag `v1-lora`.
- Released weights sha256 `d27af6ab2be16824166ac639907b4dba40979ff338599c2872721aa6c5072022`; `head.pt` sha256 `7968f17b03479c1ef9d1c0f3ab8a15e31ecb441cf40691b07ee945ab554d45ad` (T = 1.3195). Weights published in Hub commit `28be62e9`.
- Verification: loaded anonymously from the Hub on an H200, it reproduced the pre-release evaluation logits exactly on 252 of 252 SemIf rows and 764 of 764 transfer-v4 development rows (`runs/release/kev-27b-r23-published.json`, `runs/release/kev-27b-r23-staging.json`).

## Citation

```bibtex
@misc{palmer2026kev27b,
  title        = {Kev-27B: a calibrated decision model on Qwen3.8-27B},
  author       = {Palmer, Jared},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/jaredpalmer/kev-27b}},
  note         = {Version 2, released 2026-09-30}
}
```

## Contact

Questions and issues: [github.com/jaredpalmer/kev/issues](https://github.com/jaredpalmer/kev/issues).
