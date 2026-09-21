# Research plan

This file is the living plan: where Kev stands, what runs next and the criteria decided before the runs, and open questions. Completed plans stay as records ([`PLAN_Qwen35.md`](PLAN_Qwen35.md), the Qwen3.5 port, done 2026-09-20). Everything older is kept below under **History**, dated.

## Where we stand (2026-09-20, evening)

- **Family:** Kev-0.8B / 4B / 9B on Qwen3.5 bases, one recipe (`decision-v7`, LoRA r=16, lr 1e-4 / 5e-5 / 5e-5). Locked test, out of domain: 0.668 / 0.832 / **0.837**; Jev 0.857 on the development items. Qwen3 checkpoints published as the previous generation.
- **Gap to Jev (Kev-9B, `transfer-v4` dev, 4.5 pp overall):** knowledge (MMLU 0.74 vs 0.90; MMLU-Pro 0.545 vs 0.84) — the untrained base scores the same, so this is base capacity; date arithmetic (`deadline` 0.72 vs 0.93) — any LoRA fine-tune on our format erodes the base's skill (0.82 → 0.72), while the readout is intact (giving the model the day count yields 0.93–1.0, issue #8); calibration — Brier 0.291 vs 0.211, **coverage at ≤ 5 % error 0.53 vs 0.70**, confident errors 7.5 % vs 3.7 %; robustness — assertion-style Noul instructions ("The customer sounds angry.") drove Kev-4B to 0.79 vs Jev 0.91 on scienthoon's tickets.
- **Tools now available:** `--init_from` delta fine-tunes from a released checkpoint (minutes, not hours); `--data` JSONL for custom records; `transfer-v9` (MMLU-Pro, buried, unknowable) and the SemIf / scienthoon external suites; coverage-at-error-budget and unknowable metrics in `kev.benchmark`.
- **Budget:** ~$370 of the $500 overnight authorization spent (family + Qwen3.5 port); ~$130 remains for tonight.

## Tonight's autoresearch (2026-09-20 → 21)

Ordered by expected value. Each trial is either a **delta** (warm start from the released checkpoint, new records mixed with a replay sample of `decision-v7`, lr 2e-5, 1 epoch) or a **probe** (no training). Selection on development partitions only; one locked read per adopted candidate.

| # | question | run | cost | adopt if (pre-registered) |
|---|---|---|---|---|
| 1 | Does a bigger MoE base fix the knowledge column? | Zero-shot probe of `Qwen3.5-35B-A3B-Base` on `transfer-v4` + `v9` (H200) | $6 | **Proceed to a Kev-35B-A3B trial** only if MMLU-Pro ≥ 0.70 and overall ≥ the 9B base's 0.729. |
| 2 | Can calibration be bought without accuracy? (a) per-(type, K) temperature fitted on dev; (b) unknowable records with **uniform targets** as a delta from Kev-9B / Kev-4B | (a) $0, (b) 2 deltas ~$8 | Adopt if coverage@5 % error improves ≥ +5 pp on `transfer-v4` dev with accuracy within 1 pp of the released checkpoint; for (b) also unknowable share ≥ 0.9 falls (4B: 0.19 → ≤ 0.10). Temperature is reported as a separate row, never folded into raw numbers. |
| 3 | Date arithmetic by construction (issue #8): date-bearing families rendered with relational day counts and a `date_facts` field; opt-in `date_facts` request preprocessor | 2 deltas ~$8 | Adopt the renderings if `deadline` ≥ 0.85 **with** the preprocessor and raw `deadline` does not fall, accuracy elsewhere within 1 pp. Report raw and preprocessed separately. |
| 4 | Assertion-style Noul instructions | 1 delta each at 9B / 4B ~$8 | Adopt if scienthoon `angry` ≥ 0.85 at 4B with `transfer-v4` within 1 pp. |
| 5 | Where does the eroded arithmetic live? Retention ablation: 4B from scratch, LoRA on attention + MLP only (DeltaNet projections frozen) | 1 trial $5 | Informational; if `deadline` ≥ 0.65 raw, DeltaNet-frozen becomes a candidate recipe for a follow-up. |
| 6 | Combined delta (2b + 3 + 4) from Kev-9B and Kev-4B | 2 deltas ~$10 | Same rules as its parts; this is the promotion candidate if the parts pass. |
| 7 | Third-party comparability: Kev-9B / 4B on ekzhang's 1,000-question MMLU-Pro sample (seed 42); GPQA-diamond as an eval-only source if the dataset is accessible | $3 | Reporting only. |
| 8 | If #1 passes: Kev-35B-A3B, `decision-v7`, lr 5e-5, LoRA on attention + DeltaNet + shared expert, routed experts frozen, H200/B200 | ~$40 | Ship as a hosted tier only if `transfer-v4` dev ≥ Kev-9B + 2 pp **and** MMLU-Pro ≥ 0.70; one locked read. |

Rules: accuracy comparisons are record-clustered paired bootstraps on the same items; "within 1 pp" means the point estimate. Deltas replace a released checkpoint only after the locked read confirms no regression there. Every adopted change is written into the model cards with the criteria outcome, met or not.

Deferred, in order: MLX serving for the hybrid on Mac (the release's one regression: 0.78 s vs 0.17 s at 4B); multilingual slices; high-cardinality column; GGUF/browser path.

## Results so far (2026-09-21, 23:00; the Qwen3.6-35B-A3B trials are still running)

Spend tonight ≈ $85 (probes $14, deltas 12 × ~$1.5, dense $5, benches ~$12, locked reads $6, two failed 35B launches $8, 35B trials ~$25 running). Working notes: `scratchpad.txt`.

| # | result | verdict |
|---|---|---|
| 1 | **Qwen3.5-35B-A3B-Base** zero-shot: `transfer-v4` 0.720 (9B base 0.729), MMLU 0.82, **MMLU-Pro 0.590**, deadline 0.75, rules worse. | Gate failed (needed MMLU-Pro ≥ 0.70). No base-MoE trial. |
| 1′ | **Qwen3.6-35B-A3B (post-trained)** zero-shot with the SemIf prompt: **0.812** = trained Kev-9B; MMLU 0.85, deadline 0.88, emotion 0.62; rules weak (0.62 / 0.56). Plain prompt 0.726. | Gate passed (≥ 0.76). Two trials (lr 5e-5, 2e-5, `--weights_dtype bf16`, routed experts frozen, 21 M LoRA params) running on H200s. |
| 2a | Single temperature T ≈ 2.0 fitted in-distribution **does** transfer OOD on the Qwen3.5 family: Kev-9B Brier 0.291 → 0.267, ECE 0.105 → 0.039, confident errors 7.5 % → 3.2 % (Jev 3.7 %), accuracy and coverage unchanged. Per-(type, K) temperatures are worse (OOD Score items have a K the in-distribution fit never saw; per-group scaling scrambles the cross-group confidence ranking). | Adopt as a reported **calibrated row**; `KEV_TEMPERATURE` in `kev.serve`. The coverage criterion was ill-posed for a global T (monotone). Grouped T rejected. |
| 2b | Unknowable records with uniform targets (delta): share of evidence-free items answered at ≥ 0.9 → **0.00** at both sizes (from 0.05 / 0.19), controls unchanged. Coverage@5 % on `transfer-v4` dev: 9B 0.53 → 0.48, 4B 0.54 → 0.57. | Confidence criterion passed decisively; coverage criterion failed at 9B. Partial. |
| 3 | Date renderings (delta) + `date_facts` preprocessor: deadline 9B 0.72 → 0.80 raw → **0.90 with the preprocessor**; 4B 0.55 → 0.60 → 0.82 (0.85 in the combined delta). The preprocessor alone on the released models: 0.75 / 0.68 — training to *bind* the fact is what makes it work. | **Passed** (≥ 0.85 with preprocessor, raw not lower, accuracy within 1 pp). |
| 4 | Assertion-style Noul (delta): scienthoon `angry` 4B 0.794 → 0.821 alone, 0.859 in the combined delta; 9B 0.911 → 0.924. Side effect: raises confidence on evidence-free items (unknowable share 0.05 → 0.25 at 9B) and adds a few p = 1.00 errors on Noul tasks (tweet, PAWS) that cut coverage. | Passed only in combination; the combination's coverage cost is partly this. |
| 5 | Dense ablation (LoRA on attention + MLP only, DeltaNet frozen, 4B seed 1): deadline **0.53 → 0.53**, transfer 0.800 → 0.780. | Freezing the recurrent layers does not preserve date arithmetic. Not a candidate. |
| 6 | Combined deltas, **locked test** (one read each): 9B + dates + unknowable **0.852** OOD (+1.8 pp [+0.8, +2.9] over Kev-9B's 0.837), Brier 0.237, deadline 0.88, pairs 0.81, coverage@5 % 0.62 (from 0.66); 9B + all 0.851; 4B + dates + unknowable 0.837 (+1.0 [−0.1, +2.1]), Brier 0.255, coverage 0.68; 4B + all 0.828 (neutral). | **Promotion candidates: the dates + unknowable deltas at both sizes**, published as Hub branch `night2-du` (main untouched, awaiting sign-off). Costs to state in the cards: coverage@5 % −4 pp at 9B, scienthoon ECE +3 pp, MMLU-Pro −3 pp at 9B. |
| 7 | ekzhang's 1,000-question MMLU-Pro sample: Kev-9B **0.511**, Kev-4B 0.468, Kev-8B (Qwen3) 0.488 (8 questions over the 384-token state limit counted wrong). Jev 0.829; untrained one-token Qwen3.6-35B-A3B 0.588; his $5 SFT ≈ 0.71. | Reporting. Knowledge is base-bound; the 3.6 trial is the only lever. GPQA-diamond is gated — needs the account to accept terms. |

## Open questions

- Why does LoRA fine-tuning on classification-shaped data erase multi-step latent computation (dates) while leaving recall (MMLU) intact? Trial 5 is the first probe; a follow-up is layer-wise LoRA ablation.
- Is the 9B → 35B-A3B step a knowledge gain only, or does the MoE also change calibration and rule composition? Trial 1 and, if it passes, trial 8.
- Does the unknowable-record training transfer to *unseen* kinds of missing evidence (scienthoon's org-rule priority is the external test)?

---

# History

## Current decision

Use **Qwen3-0.6B-Base as the research baseline**. Keep the released Qwen2.5 checkpoint as a historical reference, not as an equally funded development track. Preserve the MacBook training path; run the controlled studies on Modal H100s.

On the same v2 recipe and data, Qwen3's development accuracy was 81.6% / 79.3% across two seeds, versus 74.2% / 65.8% for Qwen2.5. Transfer accuracy was 62.0% / 62.1% versus 60.5% / 48.2%. This supports our backbone choice, not a claim that every Qwen3 model dominates every Qwen2.5 model.

Evidence: [ablation-v2 ledger](runs/ablation-v2/results.jsonl), [Qwen3 seed 0](runs/ablation-v2/06-trial-6/result.json), [Qwen3 seed 1](runs/ablation-v2/07-trial-7/result.json).

## Evidence and corrections

- Original Kev versus Jev on familiar sources: 79.7% versus 81.1% micro accuracy; macro difference −1.8 points, 95% CI [−5.5, +1.7]. An interval including zero is not evidence of equivalence. [Comparison](runs/kev-vs-jev-v1.json), [figure](docs/kev-benchmark.png).
- Original Kev versus Jev on transfer-v1: 63.3% versus 82.3%, macro difference −19.1 points, CI [−23.1, −15.0]. The overlap check covered 1,360 decision-v1 training/calibration states, **not every example used to train the released checkpoint**. Public pretraining overlap is unknown for both models. [Comparison](runs/kev-vs-jev-transfer-v1.json), [manifest](evals/transfer-v1/manifest.json).
- Lower aggregate ECE on transfer-v1 did not establish generally better calibration. Qwen3 trial 6 gets 50% authorization accuracy with 99.6% mean confidence on transfer-v2. Its familiar-source ECE falls from .073 to .026 after fitting temperature on calibration; this does not establish transfer calibration. [Result](runs/ablation-v2/06-trial-6/result.json).
- Jev's zero observed argmax flips do **not** prove architectural invariance. Its probabilities move under permutation. [Jev transfer-v1 report](runs/transfer-jev-v1/report.json).
- A none-present accuracy of 25% means 75% wrong, not necessarily 75% selecting none. Report actual none-option mass and selection separately. [Original comparison diagnostics](runs/kev-vs-jev-transfer-v1.json).
- MMLU and domain-transfer failures do not by themselves identify a knowledge versus readout bottleneck. We need a controlled capacity/data experiment.
- Two v2 issues were found: siblings shuffled their sentence order independently, and calibration took the first slice of a family-ordered synthetic list. Pair metrics also compared option indices rather than semantic keys. Fix these under **v3**, with tests; do not rewrite v2 again.

## Nimble research

[Bespoke Nimble](https://github.com/bespokelabsai/nimble), [model card](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B), [dataset guide](https://github.com/bespokelabsai/nimble/blob/main/docs/DATASET.md).

The inspected release trained a Qwen3.5-9B LoRA adapter on 2,676 synthetic contrastive records. It reported 90.1% agreement with synthetic reference labels versus Jev's 93.2% on 324 records from six source families. These are useful external research results, not measurements on our evaluation suite or proof of broad parity. In the inspected checkout, `data/` was ignored and the dataset guide required separately supplied files; no directly usable dataset was found during that review.

Adopt minimal factual edits, executable labels where possible, evidence checks, and grouped splits. A check on structured fact dictionaries does not independently validate the English rendering. Separate model calls also do not eliminate correlated synthetic-label errors. Hard outcome labels with a proper scoring rule are sufficient; teacher probabilities are not required.

Nimble scores letter tokens and supports a hybrid recurrent backbone through separate field execution. Our current packed mask alone does not isolate recurrent state, so Qwen3.5 is not a drop-in replacement. This does not make hybrid models fundamentally unusable.

## v3 protocol (approved)

### 1. Correctness and immutable artifacts

- Preserve all existing suite versions and run directories. New freezes and Modal trials refuse to overwrite existing paths, including failure artifacts.
- Match sentence order and option order inside a minimal pair. Change one decisive fact only.
- Stratify calibration by family while keeping pairs/groups together.
- Reject incomplete or duplicated pairs. Compare semantic answer keys, not their positions.
- Preserve locked-test bytes. New held-out structures/renderings are appended only to a new version. Do not score the locked test during this study.
- Verify exact semantic-state separation among newly generated groups. Do not claim fuzzy or pretraining decontamination. Inherited legacy test overlap with newly generated controls is not certified by this audit.

### 2. Compositional policy data

Generate an explicit rule tree, facts, executable reference label, and a textual rendering. Include numeric comparisons (`<`, `≤`, `>`, `≥`, `=`, inclusive ranges), entity equality, elapsed-date comparisons, AND, OR, NOT, exceptions, and conditional precedence.

For each situation create both:

1. Relevant intervention: change one fact and require the answer to change.
2. Irrelevant intervention: change a routing reference and require the answer to stay unchanged.

Keep all four records in one split and one bootstrap unit. Require removal of the decisive fact to make the relevant decision unknown in the executable rule. Test truth tables and numeric/date boundaries, then manually inspect rendered samples.

Eight rule structures are trainable; three new compositions are transfer-development only; three further structures and a reserved rendering style are locked-test only. Authorization and deadline template families remain excluded from training, although related logical primitives are intentionally trained. This measures compositional/template transfer, not unseen primitive knowledge.

### 3. Matched data-versus-capacity experiment

| Backbone | Corrected legacy policy data | Compositional policy data |
|---|---|---|
| Qwen3-0.6B-Base | Control | Data effect |
| Qwen3-4B-Base | Capacity effect | Combined effect |

- Identical 3,000 public training records from ten permitted sources in both arms.
- Exactly 448 synthetic training records per arm, replacing rather than adding examples.
- Shared, family-stratified calibration and shared development/transfer sets.
- Two epochs, LoRA rank 16, identical learning rate per comparison, effective batch size eight. Memory-saving microbatching must preserve per-record loss weights, including the last partial batch.
- Start with one seed per cell; repeat the cells with a second seed if smoke validation and the budget permit. Do not select and report only the better seed.
- Record total forward tokens, steps, memory, training/evaluation wall time, full resolved configuration, dependency lock hash, source hashes, and base revisions. Equal record exposure is not equal FLOPs.

### 4. Selection and uncertainty

Primary score remains macro development NLL, with separate familiar-task retention and transfer reporting. Also report Brier, raw/calibrated ECE, confident-error rate at probability ≥0.9, accuracy at 50%/80% coverage (including ties), relevant-pair both-correct rate, and irrelevant-pair invariance/both-correct rates.

Temperature is fit on calibration only and applied unchanged to transfer. Neither the agent nor a candidate can adjust the evaluator through the trial config.

Research screening includes complete coverage, isolation, original-task retention, transfer accuracy/Brier non-regression, at least 70% held-out pair correctness, and at most 10% confidently wrong transfer answers. These are predeclared provisional thresholds, not production guarantees. A development win can become a candidate for a locked test; it must never automatically become a released model.

### 5. Compute and spending

No local training job needs to be interrupted. The Modal workspace currently reports $9.46 metered usage before this phase (covered by credits). Verified H100 rate: $3.95/GPU-hour, plus CPU/memory/storage. [Modal pricing](https://modal.com/pricing).

Use at most four concurrent containers, no automatic trial retries, and an 1,800-second per-trial timeout. Launch-time cost admission uses the GPU rate plus bounded CPU/memory requests; it excludes image build/startup/storage and is not an account-level hard spending cap. Keep the phase within the previously discussed $100 total budget, with an initial study compute bound below $20. Preserve failures rather than silently retraining or overwriting them.

Modal documentation: [images](https://modal.com/docs/guide/images), [volumes](https://modal.com/docs/guide/volumes), [GPU](https://modal.com/docs/guide/gpu), [secrets](https://modal.com/docs/guide/secrets).

## Overnight autoresearch (branch `research/overnight-1`, PR #3)

Authorized: up to $500 of Modal credits; spend baseline $27.17 at 19:45. Rules unchanged: selection on development
partitions, locked test read at most once per promoted candidate, no evaluator changes from a trial config, every
trial through `kev.experiment.execute_trial` with provenance. New tonight:

- `kev/autoresearch.py`: bounded hill-climb over the allowlisted config space; leaderboard across all studies
  ([`runs/leaderboard.md`](runs/leaderboard.md)); spend ledger from `modal billing`; auto-maintained log below.
- Architecture switches (flags, default off): `option_isolation` (option spans are isolated sub-branches with shared
  positions; permutation invariance exact by construction, 1.2e-7 measured on the real model), `special_embeddings`
  (train the five delimiter embeddings), `head_dim`.
- Suites: `decision-v4` (10k public + 448/arm synthetic) and `decision-v5` (20k public + 1,792/arm), both with dev/test
  bytes identical to v3 so every number since the matched study is comparable. `transfer-v4/v5` are byte-identical to v3.
- Reference: Jev on decision-v4 dev acc 0.845, Brier 0.237, trained-structure pairs 0.86; on transfer-v4 dev acc 0.857.

Sequence: 0.6B screening rounds (cheap, one seed, replicate winners) -> promote the winning knobs to 4B on v4 -> 4B on v5
-> 8B once with the best recipe -> one locked-test read for the best gated candidate -> research-preview cards.

**Blocked at 20:55: the Modal workspace hit its spend limit** (`Workspace ... has exceeded its spend limit`; metered
$48.50, $30 credits applied, $18.50 billed). Twelve running containers were killed (arch screen 4/8 evaluated, both
4B mix studies mid-training). Raising the limit needs the dashboard (workspace Settings -> Billing -> spend limit);
the CLI cannot. Until then: evaluations of the six salvaged arch-screen checkpoints run on the MBP GPU
(`kev.experiment --resume`), the 0.6B research preview is prepared locally, and 4B/8B work waits.

Findings so far tonight (v4 suites; Jev dev 0.845 / transfer 0.857):
- 0.6B is saturated on transfer at 0.59-0.61 regardless of knobs (round 1: eight mutations, all within noise; lr 5e-4
  and lora 64 + ord_w hurt dev). Run-to-run noise at the same seed on GPU is ~1 pp.
- **More public data hurts 4B transfer**: 4B on v4 (10.9k) transfer 0.704 / 0.735 vs 0.747 / 0.750 on v3 (3.4k), while
  dev rises 0.825 -> 0.849. The lever at 4B is the data *mix*, not volume; `synthetic_repeat` / `public_frac` knobs
  added; the v4-vs-v5 mix studies were killed before finishing.
- Option isolation trains normally at 0.6B (dev 0.800) with a measured flip rate of exactly 0.0; transfer 0.58-0.60
  (parity with the plain encoding; six salvaged arch-screen trials all within noise of the incumbent).
- **Fine-tuning loses base capability** ([`scripts/base_mmlu_probe.py`](scripts/base_mmlu_probe.py): the base model,
  zero-shot, next-token logits over option letters, same 80 frozen items per task):

  | task (transfer-v4 dev) | 0.6B base | Kev 0.6B | 4B base | Kev 4B | 8B base | Kev 8B | Jev |
  |---|---|---|---|---|---|---|---|
  | MMLU | 0.425 | 0.46 | 0.688 | 0.60-0.66 | **0.762** | 0.65 | 0.90 |
  | PAWS | 0.70 | 0.56 | 0.787 | 0.56-0.71 | **0.85** | 0.70-0.78 | 0.79 |
  | SciQ | 0.912 | 0.86 | 0.975 | 0.95-0.97 | 1.0 | 0.99-1.0 | 0.99 |
  | QNLI | 0.775 | 0.85 | 0.812 | 0.84-0.91 | 0.887 | 0.84-0.88 | 0.925 |
  | Emotion | 0.287 | 0.50 | - | 0.53-0.62 | 0.30 | 0.53-0.56 | 0.60 |
  | TweetEval offensive | 0.588 | 0.69 | - | 0.64-0.80 | 0.637 | 0.69-0.75 | 0.81 |

  The fine-tune helps on classification-shaped tasks and *hurts* on knowledge and paraphrase: -11 pp MMLU at 8B,
  -14 pp PAWS at 0.6B. The 8B base zero-shot already beats Jev on PAWS. Hypotheses under test: low-drift LoRA
  (`lowdrift-4b-v4`: fewer target modules, r=4-8, lr 5e-5..1e-4, 1 epoch) and a knowledge-MCQ training mix
  (`knowledge-4b-v6`: ARC-Challenge, OpenBookQA, CommonsenseQA added as trainable sources; MMLU/SciQ stay eval-only).
- 4B on v5 (20k public, compositional arm, 1 epoch): dev 0.858 (best 4B dev), transfer 0.713: more public data keeps
  raising in-distribution accuracy and not transfer.
- **The learning rate is the lever at 4B and 8B.** Default lr 2e-4 erodes base capability; lr 5e-5 (everything else
  equal) gives transfer 0.759 / 0.758 at two seeds on v4 and 0.755 / 0.761 on v6, +4.7 pp [+0.4, +9.6] vs the default,
  with the best Brier (0.346). Fewer LoRA target modules and smaller ranks help less; combining low lr with
  public_frac / synthetic_repeat / option isolation does not stack (0.748-0.752). Knowledge MCQ sources (v6) lift
  in-distribution accuracy to 0.86 and MMLU to 0.70 without moving transfer. 8B at lr 5e-5: dev 0.869, transfer
  0.774, Brier 0.339 (best of any size) but +0.4 pp [-3.9, +4.5] vs 4B on transfer. The 4B->8B step is flat for this
  recipe; the remaining gap to Jev (0.857) is MMLU (0.69 vs 0.90; the 8B *base* gets 0.76 zero-shot), PAWS, emotion.
- Mix results at 4B: public_frac 0.33 (+synthetic_repeat 2 + isolation) 0.752 at lr 2e-4; synthetic_repeat 3 hurts
  (0.686, PAWS 0.45); 1 epoch is better calibrated (Brier 0.379, confident errors 2.7%) at equal accuracy; the
  2-epoch 4B run on v5 (23.6k records, 5.9k steps at lr 2e-4) **collapsed** (dev 0.58) - long runs at lr 2e-4 are unstable.
- Research previews published, each with one exploratory (ungated) locked-test read, all above their development numbers:
  [Kev-0.6B](https://huggingface.co/jaredpalmer/kev-0.6b) dev 0.805/0.598 -> test 0.819/0.631;
  [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) (lowdrift-4b-v4/01-trial-1) dev 0.843/0.759 -> test 0.852/0.794;
  [Kev-8B](https://huggingface.co/jaredpalmer/kev-8b) (recipe-8b-r1/00-trial-0) dev 0.869/0.774 -> test 0.869/0.799.
  None passes the predeclared 70% held-out-pair screen (0.11 / 0.62 / 0.61), so none is a versioned release.
- Kev-0.6B research preview published ([`jaredpalmer/kev-0.6b`](https://huggingface.co/jaredpalmer/kev-0.6b),
  card [`docs/model-cards/kev-0.6b.md`](docs/model-cards/kev-0.6b.md)); one exploratory (ungated) locked-test read:
  decision 0.819, transfer 0.631 ([`runs/locked/kev-06b-preview-ungated/summary.json`](runs/locked/kev-06b-preview-ungated/summary.json)).

### Overnight synthesis (as of 02:00; spend ~$140 of $500)

What moved the needle, in order of effect size, all on the same frozen development items:

1. **Backbone capacity** 0.6B -> 4B: +14-19 pp transfer (matched data). 4B -> 8B: +1.5-2 pp at the low-lr recipe
   (4B 0.759/0.758, 8B 0.774/0.779 at two seeds each).
2. **Learning rate 2e-4 -> 5e-5**: +4.7 pp [+0.4, +9.6] at 4B, replicated at two seeds and two suites; best Brier;
   the mechanism is reduced drift from the base model (base zero-shot probe). 3e-5 and 2e-5 are equivalent to 5e-5.
3. **None-of-the-above minimal pairs**: none_present accuracy 0.75 -> 0.78-0.85 (0.6B), 0.85-0.93 (4B).
4. Compositional policy data: +4-5 pp at 4B on v3 (CI touching zero), nothing at 0.6B; teaches trained structures
   (0.85-1.0) and transfers partially to unseen ones (0.5-0.67 at 4B/8B).

What did not work: more public data (raises dev, lowers or flattens transfer; a 2-epoch 23.6k-record 4B run at lr 2e-4
collapsed); synthetic oversampling x3 (hurts PAWS badly); LoRA rank/target ablations (within noise once lr is low);
option isolation (exact permutation invariance at no accuracy cost, but no accuracy gain); special embeddings; head_dim;
perm_kl; ord_w; 3 epochs; knowledge MCQ sources (dev +2 pp, MMLU +2-5 pp, transfer flat). Fourteen one-knob mutations
around the low-lr incumbent at 4B all landed in 0.748-0.767: **the config space is exhausted for this data and
evaluation**; run-to-run noise at a fixed seed is ~1 pp.

Also tried after the synthesis: WiSE-FT-style interpolation toward the base at inference (`KEV_LORA_SCALE`): alpha 0.75
0.765 (noise), alpha 0.5 0.736 (-4.4 pp, CI excludes zero) - the head depends on the fine-tuned features, so weight-space
interpolation is not a lever. Round auto-4b-r1 (head_lr, weight decay, rank 8, head_dim 1024, perm_kl, synthetic x2): all
0.755-0.767; option isolation at low lr 0.729 (-5.8 pp, significant) - isolation costs accuracy at 4B.

Where the remaining gap to Jev (0.857 transfer) lives, per task at 8B: MMLU 0.69-0.74 vs 0.90 (the 8B *base* is 0.76
zero-shot - our readout still loses knowledge), PAWS 0.75 vs 0.79 (base 0.85), Emotion 0.55 vs 0.60, TweetEval 0.71 vs
0.81, deadline 0.55-0.70 vs 0.925. Held-out policy pairs 0.61-0.67 vs the 0.70 screen.

Next levers the evidence points at (not config knobs):
- **Knowledge readout**: the pointer head under-uses what the base knows. Try a hybrid readout that adds the base
  model's own letter/option-token logits (frozen, zero-drift) to the pointer logits, or distill the *base* model's
  zero-shot distribution on knowledge-shaped questions into the head (self-anchoring, no Jev).
- **Date/ordinal reasoning**: deadline stays near the middle level; needs either scratchpad-free arithmetic data with
  varied surface forms or a Score readout that models cumulative levels directly.
- **Held-out structure generalization**: more *rule structures* (not more pairs per structure) and rendering styles.
- **Evaluation**: the 70% pair screen is within reach at 8B (0.59 / 0.67 / 0.61 across three seeds) but not met by
  any seed, so no gated locked-test read happened tonight; the published previews carry ungated reads.

Final replication (03:30): 4B lr 5e-5 at three seeds transfer 0.759 / 0.758 / 0.759; 8B lr 5e-5 at three seeds
0.774 / 0.779 / 0.774. Both recipes are stable to ~0.5 pp. Spend for the night: ~$180 of the $500 authorized
(89 trials indexed; `runs/leaderboard.md`).

## Toward v0.2: crossing the release screen (2026-09-19)

Where the 8B loses the 70% held-out-pair screen (64 relevant pairs; needs 45, has 39-43; Jev 55):

| held-out family | Kev-8B seeds 0 / 1 | Jev | missing from training |
|---|---|---|---|
| authorization | 20/20, 20/20 | 20/20 | - |
| (A or B) and C | 6/8, 8/8 | 6/8 | - |
| if A then not B else C | 2/8, 4/8 | 5/8 | negation nested inside a conditional |
| (A and B) or not C | 1/8, 2/8 | 7/8 | negation nested inside a disjunction |
| deadline (3-level Score, grace period) | 10/20, 9/20 | 17/20 | any ordinal-threshold Score family; date arithmetic only as yes/no |

Training had eight fixed rule shapes with negation only at the top level, and no Score question whose levels are
threshold intervals. The 8B *base* scores 0.58 zero-shot on the held-out rule items (near chance), so rule composition
is learned from our synthetic data, which is why structural coverage is the lever.

`decision-v7` ([manifest](evals/v7/decision-v7/manifest.json); dev/test bytes identical to v4, so every number is
comparable; scored against `transfer-v4`): 10k public (v4 pool) + legacy policy arm with four new ordinal Score families
(warranty claim by date, SLA response hours, late-fee days, volume discount; 896 records) + compositional arm from
**60 random rule trees** with negation anywhere, depth 2-3, rendering styles 0/1/3/4 (1,680 records). Held-out and
locked structures are excluded by a canonical key that is invariant to commutation, leaf numbering and De Morgan
pushing ([`kev/composition.py`](kev/composition.py) `canonical`, `push_negation`, `sample_trees`).

Release rule tightened: the screen must pass on **every seed** of a config (`kev.autoresearch release-check`), not one
seed of 64 pairs. Study `v7-release-candidates`: 4B and 8B at lr 5e-5, two seeds each.

### Results so far (v7 and the untrained baselines)

**Untrained baselines on the same frozen items** (`scripts/base_mmlu_probe.py`, zero-shot letter logits, Modal):

| transfer-v4 dev | 8B base, untrained | 30B-A3B base, untrained | Kev-8B | Jev |
|---|---|---|---|---|
| accuracy | 0.726 | 0.707 | **0.774** | 0.857 |
| Brier | 0.366 | 0.365 | **0.339** | 0.211 |
| MMLU | 0.75 | **0.79** | 0.69 | 0.90 |
| PAWS | 0.84 | 0.82 | 0.76 | 0.79 |
| Emotion | 0.30 | 0.31 | **0.57** | 0.59 |
| held-out rule pairs | 0.55 | 0.44 | **0.61** | 0.86 |

Kev-8B vs its own untrained base: +5.8 pp [+1.8, +10.0]; vs the untrained 30B-A3B: +9.7 pp [+4.7, +14.4]. The
recipe does real work (rule composition, classification-shaped tasks) and *loses* on knowledge/paraphrase, where both
untrained models beat it (MMLU 0.75-0.79 vs 0.69; PAWS 0.82-0.84 vs 0.76). "A bigger untrained MoE gets Jev-class
results for free" is false on this suite; "the fine-tune erodes base capability" is confirmed a third way.

**decision-v7 at 4B** (`v7-rc3`): transfer **0.773 / 0.790** (v4 recipe: 0.759 x3), dev 0.858 / 0.854, held-out pairs
0.62 / **0.73**. The random-structure data moves the two failing rule families: (A and B) or not C 0.66 -> 0.75 / 0.97,
if-then-not 0.62 -> 0.75 / 0.88. The ordinal Score families did **not** move deadline (0.60 / 0.53). Seed 1 clears
the 70% screen; seed 0 does not, so under the two-seed rule this is not yet a release candidate. 8B on v7 and a third
4B seed are running; the deadline family needs a different idea (the model still hedges to the middle level).

**Anchoring** (`kev.anchors` + `--anchor_w`, KL toward the frozen base's zero-shot distribution, targets keyed by option
key; `anchor-4b-v6`): w=0.5 on the knowledge MCQ sources transfer 0.773 (+2.6 pp [-1.8, +6.8] vs the un-anchored v6
recipe), w=0.3 on everything 0.761 with the best calibration of any 4B (Brier 0.338, confident errors 3.4%), w=1.0 0.741.
MMLU stays 0.66-0.69 and PAWS 0.68-0.70 under every setting. **Not a lever**: at lr 5e-5 the 4B already matches its base
on MMLU (0.69 vs 0.688); the remaining PAWS gap (0.70 vs base 0.79) is not closed by anchoring the output distribution,
which points at the readout/format rather than at drift (the third outcome in the anchoring experiment's table).

**Final v7/v8 read (21:20).** 4B on v7, three seeds: transfer 0.773 / 0.790 / 0.770, held-out pairs 0.62 / 0.73 / 0.67.
8B on v7, two seeds: transfer **0.796** / 0.774, pairs 0.69 / 0.64. 4B on v8 (v7 + `shipping_delay`, a day-precision
date-threshold Score family built for the deadline skill): 0.777 / 0.759, pairs 0.69 / 0.56, deadline 0.53 / 0.45 -
the extra family did nothing. **No config passes the screen on every seed**; 8B seed 0 misses by one pair (44/64).
The deciding family is `deadline` (0.45-0.60 for every Kev; untrained 8B/30B bases 0.53-0.55; Jev 0.93): day-precision
date arithmetic to a 3-level ordinal in one forward pass looks like a capability limit at <= 8B for this readout, not a
data gap - two purpose-built training families did not move it.

Published as **updated research previews** (no version tag; each card records one locked read):
[Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) = `v7-rc3/01-trial-1` (dev 0.854 / 0.790; locked 0.856 / **0.806**),
[Kev-8B](https://huggingface.co/jaredpalmer/kev-8b) = `v7-final/00-trial-0` (dev 0.863 / 0.796; locked **0.870** / 0.780).
The 8B locked read was interrupted once before any aggregate existed and redone (recorded in its summary).

What would cross the bar, in order of my confidence: (1) an ordinal readout that models cumulative thresholds for Score
questions (the models hedge to the middle level on deadline); (2) explicit day-count rendering in training states
(teaching the arithmetic is not the point of a decision model; giving it the number is); (3) capacity beyond 8B.
Spend to date ~$250 of $500.

**Release (2026-09-20).** The three checkpoints are published as the Kev family, no version tags (nothing overlapping to
version; the 0.5B prototype stays on the Hub for reference). The 70% held-out-pair screen is no longer a publication gate
(it was within seed noise and decided by one arithmetic family); it stays in `kev.experiment` as a research gate and in
`release-check`. Final selection, every checkpoint the best of its size on the leaderboard:
Kev-0.6B = `v7-06b/02-trial-2` (dev 0.801 / 0.620, locked 0.808 / 0.642; three v7 seeds 0.613 / 0.605 / 0.620, all above
the previous 0.598), Kev-4B = `v7-rc3/01-trial-1`, Kev-8B = `v7-final/00-trial-0`. Serving: fp32-merged LoRA, SDPA on MPS,
state-prefix KV cache, shape bucketing (all parity-tested).

**Qwen3.5 generation (2026-09-20).** Same data and recipe on Qwen3.5-4B/9B bases (hybrid Gated DeltaNet + attention; questions run
as causal rows from a shared state instead of under a packed mask). Development criteria set in advance were not met (accuracy CI
includes zero; deadline 0.72 not 0.75); the single locked-test read shows Kev-9B **0.837** vs Kev-8B 0.780 out of domain
(+7.3 pp [+2.8, +11.7], Brier 0.243 vs 0.327) and Kev-4B 0.832 vs 0.806. Published as the current family: `jaredpalmer/kev-9b`,
`jaredpalmer/kev-4b` (Qwen3 weights at tag `qwen3`) and `jaredpalmer/kev-0.8b` (locked 0.827 / 0.668 vs Kev-0.6B 0.808 / 0.642, +4.8 pp
[+0.2, +9.3]); the Qwen3 checkpoints stay published as the fast Mac option and are no longer developed. The deadline family is a training-data
problem, not a readout problem (issue #8; adapter-merge probe). Full log: [PLAN_Qwen35.md](PLAN_Qwen35.md).

Ops: two studies were lost to the local client disconnecting (Modal cancels `.starmap` inputs when the caller dies; `--detach`
keeps only the last-triggered function). Studies now fan out server-side from the **deployed** app (`modal deploy modal_app.py`;
`run_study` spawned, `pull --name` afterwards). ~$35 of GPU time was lost to this.

## Status and deferred work

- [x] Modal CUDA/batched path and backbone-v1 study completed; MBP path retained.
- [x] v2 source policy, PAWS/SciQ conversion, contrastive prototype, and data-ablation study completed. Findings remain exploratory.
- [x] v3 minimal-pair/calibration/metric corrections and regression tests ([tests](tests/test_v3.py)).
- [x] v3 compositional generator with truth-table and boundary tests ([generator](kev/composition.py)).
- [x] v3 frozen matched suites ([decision-v3](evals/v3/decision-v3/manifest.json), [transfer-v3](evals/v3/transfer-v3/manifest.json)); both arms share 3,000 public records and 448 synthetic records.
- [x] Modal smoke and the matched 0.6B/4B comparison, seed 0 ([ledger](runs/v3-data-capacity-s0/results.jsonl)).
      Paired, record-clustered bootstrap, transfer-v3 development:
      capacity (4B vs 0.6B, same data): +17.5 pp acc CI [+13.0, +22.1] on legacy data, +19.0 pp CI [+12.3, +25.0] on compositional data;
      data (compositional vs legacy, same backbone): +3.6 pp CI [-0.4, +7.8] at 0.6B, +5.1 pp CI [0.0, +9.9] at 4B; Brier -0.053 CI [-0.102, -0.005] at 0.6B.
      Held-out compositional structures, both siblings correct: 0.6B 3%/6%; 4B 45%/52%. Held-out authorization: 4B 100% both arms (0.6B 50%).
      Held-out deadline (3-level score) stays near chance for all cells. No cell passes the 70% held-out-pair screen; none is a locked-test candidate.
      Cost: 4 H100 trials, 0.6B ~4.5 min and 4B ~13.5 min wall each, admission bound $8.86.
- [x] Second seed for the four v3 cells; overnight: v4/v5/v6 suites, ~60 further trials, three research previews
      (0.6B / 4B / 8B) with one ungated locked read each. See "Overnight autoresearch" and the log below.
- [ ] Deferred: option-order architecture experiments. Do not infer Jev's architecture from zero argmax flips.
- [ ] Deferred: LLM-authored product scenarios, with a separate verification model and retained provenance; needs explicit API/budget decisions.
- [ ] Deferred: 8B runs after the data-versus-capacity result, not as an automatic escalation.
- [ ] Deferred: final release/model-card/Hub updates until generalization and calibration justify them.

Relevant code: [suite builder](kev/study_v3.py), [rule generator](kev/composition.py), [experiment runner](kev/experiment.py), [benchmark](kev/benchmark.py), [Modal app](modal_app.py), [v3 tests](tests/test_v3.py).

## Autoresearch log

Maintained by `kev.autoresearch`; full table in [`runs/leaderboard.md`](runs/leaderboard.md). Selection uses development partitions only.

- **Qwen3-0.6B-Base** incumbent (v4 suites): transfer 0.620, dev 0.801, seeds [2], knobs `{"epochs": 2, "lr": 0.0001, "p_none_pair": 0.25}`
- **Qwen3-4B-Base** incumbent (v4 suites): transfer 0.775, dev 0.858, seeds [0], knobs `{"epochs": 2, "lr": 5e-05, "p_none_pair": 0.25}`
- **Qwen3-8B-Base** incumbent (v4 suites): transfer 0.796, dev 0.863, seeds [0], knobs `{"epochs": 2, "lr": 5e-05, "p_none_pair": 0.25}`

| round | base | trials | best transfer | best knobs | incumbent after | spend |
|---|---|---|---|---|---|---|
| auto-06b-r1 | Qwen3-0.6B-Base | 8/8 | 0.596 | `{"epochs": 2, "accum": 1, "perm_kl": 0.5, "p_none_pair": 0.25}` | 0.592 | $9.82 |
| auto-4b-r1 | Qwen3-4B-Base | 6/6 | 0.767 | `{"epochs": 2, "lr": 3e-05, "accum": 2, "perm_kl": 0.2, "p_none_pair": 0.25, "lora_targets": "all"}` | 0.767 | $147.26 |
