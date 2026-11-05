# Kev 1.0

Kev 1.0 is the first versioned release of the whole Kev family: four decision models that read a document and a set of typed questions and return calibrated probabilities over the options, in one forward pass, behind TypeSafe's System One API. Nothing in it is newly trained. It pins the checkpoints, model cards, evaluation suites and serving code that the next generation of Kev is measured against, with the same tag (`v1.0`) on every Hub repo.

## What's in 1.0

| Model | Hub repo | Weights revision | Form | Base | Temperature | Validated context |
|---|---|---|---|---|---|---|
| Kev-0.8B | [`jaredpalmer/kev-0.8b`](https://huggingface.co/jaredpalmer/kev-0.8b) | `9a45d25e` | LoRA adapter + head | Qwen3.5-0.8B-Base (Apache-2.0) | 2.35 | 8,192 tokens |
| Kev-4B | [`jaredpalmer/kev-4b`](https://huggingface.co/jaredpalmer/kev-4b) | `139fdd94` | LoRA adapter + head | Qwen3.5-4B-Base (Apache-2.0) | 2.41 | 8,192 tokens |
| Kev-9B (v2) | [`jaredpalmer/kev-9b`](https://huggingface.co/jaredpalmer/kev-9b) | `b5d8c18e` | LoRA adapter + head | Qwen3.5-9B-Base (Apache-2.0) | 2.19 | 8,192 tokens |
| Kev-27B (v2) | [`jaredpalmer/kev-27b`](https://huggingface.co/jaredpalmer/kev-27b) | `28be62e9` | full bf16 weights (51 GB) + head | Qwen3.8-27B, post-trained (Apache-2.0) | 1.32 | 65,536 tokens |

Headline numbers (fp32 evaluation path, each model at its shipped temperature; the transfer-v4 test is locked and was read once per model):

| | Kev-0.8B | Kev-4B | Kev-9B | Kev-27B | Jev |
|---|---|---|---|---|---|
| Held-out datasets: breadth-v1 test, chance-corrected index | 23.3 | 38.0 | 41.0 | 52.3 | 54.0 |
| Out-of-domain: transfer-v4 development accuracy | 0.648 | 0.817 | 0.820 | 0.851 | 0.857 |
| Out-of-domain: transfer-v4 locked test accuracy / Brier | 0.697 / 0.397 | 0.838 / 0.224 | 0.852 / 0.199 | 0.889 / 0.154 | – |
| Skills: hard-v1 test | 0.665 | 0.803 | 0.834 | 0.918 | – |
| Developer tooling: devtools-v1 test, all sources | 0.637 | 0.756 | 0.791 | 0.790 | – |
| Real documents: documents-v1 test | 0.851 | 0.903 | 0.900 | 0.908 | – |
| MMLU-Pro (transfer-v9 development) | 0.230 | 0.565 | 0.590 | 0.675 | 0.840 |

hard-v1, devtools-v1 and documents-v1 have training splits that every Kev trained on: those rows measure held-out items of trained families, not transfer. The breadth-v1 and transfer-v4 rows are datasets no Kev trained on. Jev was read on the development partitions and on breadth-v1 test only. Every number traces to a committed report through `docs/claims.json`; the model cards (`docs/model-cards/`) have the rest, with intervals.

## What changed since the last family release

Measured from the GitHub release `kev-family` as first assembled for the current family on 2026-09-24 (Kev-27B v1, Kev-9B v1, and the same Kev-4B and Kev-0.8B as here). Its 2026-09-30 updates (Kev-27B v2, Kev-9B v2) are listed here too, since 1.0 is where they become part of a versioned release.

- **Kev-27B v2: full weights.** Every weight of Qwen3.8-27B fine-tuned for one epoch on a 145,840-record corpus, then averaged 0.85 / 0.15 with v1. Against v1 on test: held-out datasets +1.2 pp [+0.3, +2.2], held-out task families +5.3 [+3.7, +6.8], skills, tooling and documents +8.9 [+7.5, +10.3]; locked out-of-domain test 0.889 against 0.896, Brier 0.154 against 0.160. Worse and overconfident on long contracts (CUAD ECE 0.053 against 0.007). v1 is at `jaredpalmer/kev-27b@v1-lora`.
- **Kev-9B v2.** v1 plus one epoch on the documents and skills data Kev-4B and Kev-0.8B already had. Against v1 on test: hard-v1 + devtools-v1 +18.7 pp [+16.7, +20.8], documents-v1 +7.1 [+4.7, +9.2]; level on the locked out-of-domain test (0.852 both) with Brier 0.199 against 0.224. v1 is at `jaredpalmer/kev-9b@v1`.
- **No silent truncation.** The server used to cut a state longer than its limit without saying so. It now refuses a state over 65,536 tokens with a 422 that names the token count and the limit; `KEV_TRUNCATE_STATES=1` opts back into truncation, and every response from such a server then says `truncated`. The deploy and fine-tune skills pin `KEV_REF` to a commit that has this fix and the long-document and MLX changes below (`71d4829`), and the Space was republished after the fix.
- **Long documents on every size.** The evaluation path kept fp32 attention for long rows on the math kernel, so Kev-0.8B, 4B and 9B ran out of GPU memory on 32k–64k-token states. Long rows now run the memory-efficient kernel in fp32: Kev-4B reads a 61k-token state in 17.2 s with 17.2 GiB above the weights on an H100, and shorter rows keep their logits bit for bit. This is what makes the validated context lengths above measurable.
- **Apple Silicon.** The MLX backend loads full-weight checkpoints as saved, with no merge, which gives Kev-27B a Mac path (expected to need about 51 GB plus working memory; not yet run at that size). Long states are prefilled 1,024 tokens at a time and the cache evicts before a pass, so Kev-4B serves a 65,000-token state on a 32 GB M5 at a 13.0 GB peak (84.5 s new, 716 ms cached).
- **Kernel provenance.** Every evaluation report and trial now records the kernel set its logits depend on (package versions, GPU, dtype, attention and DeltaNet implementations), after a kernel change in the evaluation image was found to move Kev-27B v1's reads by 0.03–0.06 in probability with no change to Kev's own code.
- **Evaluation audit.** Three suites were removed as unsound for selecting models: scienthoon (templated tickets, one question the text cannot answer), WANLI-v2 / WANLI-v1 (a quarter of gold labels are one of two disagreeing annotators) and TypeSafe's public evals (gold from two closed models, too few questions). Headline panels exclude items the audit found unanswerable or unlabelled. Past releases' figures on those suites are kept in their records, not on the 1.0 cards.
- **Calibration.** Kev-4B and Kev-0.8B ship temperatures fitted on held-out items of their training data. A registered refit on held-out datasets was evaluated for both and adopted for neither: it did not improve Kev-4B (Brier difference −0.0001 [−0.0005, +0.0003]), and it made Kev-0.8B worse calibrated on its document and skill families by more than the registered tolerance. Kev-9B and Kev-27B already ship held-out-dataset temperatures.
- **Training data published.** The documents-v1 and hard-v1 training partitions are in the `jaredpalmer/kev-suites` dataset, so the small models' training data can be fetched and hash-checked.
- **Validated context length.** Each card now states the longest state at which accuracy on CUAD contracts stays within 3 pp (95 % lower bound) of the same model at 8k tokens. Kev-27B holds to 65,536 tokens, the serving limit (its 64k lower bound is −2.4 pp). Kev-0.8B, 4B and 9B validate only their trained 8,192: each already misses the tolerance at 16k (lower bounds −8.5, −3.4 and −3.7 pp), so past 8k tokens their answers on long documents are not covered by the measurement.
- **Formal model cards.** All four cards follow one structure: summary, details, intended and out-of-scope uses, how to use, training data and procedure, evaluation, limitations, risks, compute, provenance.

## Known limitations

- **In-distribution gains.** The large gains of the last year are on suites whose training splits are in the training data. On datasets no Kev trained on, Kev-27B is 1.7 index points below Jev on breadth-v1 test, and the smaller sizes are 13–31 points below.
- **Untrained lengths.** Kev-0.8B, 4B and 9B trained on states of at most 7,552 tokens and Kev-27B on at most 32,768; the server accepts 65,536. Use the validated context length, not the serving limit.
- **Kev-27B on long contracts** is less accurate than v1 and overconfident (CUAD test ECE 0.053 against 0.007); refit the temperature on your own documents or use `@v1-lora` for contract review.
- **Kev-0.8B and tool routing.** Its When2Call accuracy fell below chance (0.133 on test) after its documents-and-skills stage; do not use it for tool-call routing.
- **Date arithmetic** is the weakest family at every size below 27B (`deadline` policy accuracy 0.35 / 0.65 / 0.725 against Jev's 0.95); `KEV_DATE_FACTS=1` helps.
- **Knowledge** is set by the base (MMLU-Pro 0.230–0.675 against Jev's 0.840).
- **Kev-9B on a Mac** has not been measured, and Kev-27B on a Mac is expected to fit 96–128 GB but has not been run.
- **Selection.** Kev-27B v2 and Kev-9B v2 were re-selected under the audited rule with earlier development reads known; their test margins are optimistic.
- **One temperature per model** cannot reorder confidences, so at a 5 % error budget the models automate fewer decisions than Jev out of domain.

## How to run

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b@v1.0 --port 8009    # CUDA, or MLX on Apple Silicon
```

From a release tarball:

```bash
shasum -a 256 -c SHA256SUMS.txt
tar -xzf kev-4b.tar.gz
uv run --extra serve python -m kev.serve --run kev-4b --port 8009
```

The TypeSafe SDK works unchanged: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="kev-latest")`. Kev-27B needs one B200, H200 or H100 80 GB: `--run jaredpalmer/kev-27b@v1.0`. To deploy an HTTPS endpoint on Modal, see `skills/kev-deploy`.

## Assets

Each tarball holds one checkpoint as it is on the Hub at its weights revision (LoRA adapter, `head.pt` with the temperature, tokenizer files, the training trial's `result.json`, `provenance.json`, `training_config.json`, `training_metrics.json` and log), the Kev 1.0 model card as `README.md`, and the locked transfer-v4 read as `locked_test.json`. They are built by `scripts/build_release_assets.py` from `docs/releases/kev-1.0-assets.json`, and rebuilding gives the same bytes. Every file in them is dated 2026-10-01 00:00 UTC, which `kev.serve` reports as the release date of an unpacked checkpoint. The tarballs first attached on 2026-10-01 dated their files 1970-01-01, so `kev.serve` reported 1969-12-31; they were replaced the same day by these. The weights and every other file are byte-identical; only the tarball hashes changed.

| File | SHA-256 | Checkpoint | adapter / head SHA-256 |
|---|---|---|---|
| `kev-0.8b.tar.gz` (46 MB) | `0ae144c7675f0c3f333be0bb878a0f202ab9e6fa84169fb7cb16efe6176c1ef1` | `jaredpalmer/kev-0.8b@9a45d25e` | `9b908623…` / `f400bd12…` |
| `kev-4b.tar.gz` (131 MB) | `2e707e2ebd08980dc7881222b7024cea5606401441c1a086afb170ae7784201c` | `jaredpalmer/kev-4b@139fdd94` | `90e81735…` / `dd633435…` |
| `kev-9b.tar.gz` (172 MB) | `acd13320b7d1b052ce989f19ca9d1d9ba5219b8beced0ee67337908aef1deb3f` | `jaredpalmer/kev-9b@b5d8c18e` | `2b2a70cf…` / `8e1dab2c…` |

Kev-27B is not attached, because its 51 GB of weights are over GitHub's 2 GB limit per asset. Download it from the Hub: [`jaredpalmer/kev-27b@v1.0`](https://huggingface.co/jaredpalmer/kev-27b/tree/v1.0) (weights commit `28be62e9`, `head.pt` `7968f17b…`).

On every Hub repo, the `v1.0` tag points to the commit that uploaded the Kev 1.0 card. That commit changed only `README.md`, so its weights are the same bytes as the weights revision in the first table: kev-0.8b `bf75a6a8`, kev-4b `6cfce5c2`, kev-9b `db029f08`, kev-27b `af0e6d55`.

## Release plan (for the maintainer; not part of the published notes)

**Done 2026-10-01** (record `runs/release/kev-1.0.json`; PLAN.md "Released: Kev 1.0"). Steps 3 and 4: card-only commits, with `v1.0` on each card commit (0.8B `bf75a6a8`, 4B `6cfce5c2`, 9B `db029f08`, 27B `af0e6d55`; every other file unchanged). Steps 5 and 6: assets built twice with identical hashes, the release published and marked Latest. Step 7: `kev-family` kept, its assets removed, its body a pointer to `kev-1.0`, its old notes in `runs/release/kev-family-notes-retired.md`. Step 8: pins unchanged. Step 9: collection and Space checked; the Space was not republished. The plan as written before the release follows. Order:

1. **Placeholders: filled** (2026-10-01) from round 28's registered context read-out, `runs/r28-readout/context.json` (`scripts/longdoc_report.py --context-margin -0.03` over `runs/r28-{4b-r10,08b-r15}-longdoc`, `runs/r29-9b-r18a-longdoc` and `runs/r23-27b-k-w85-longdoc`; raw `runs/r28-context`, ECE at the shipped T `runs/r28-context-served`), with the numbers in `docs/claims.json`.
2. **Merge** this PR.
3. **Hub cards.** Upload each 1.0 card as `README.md` only (no weights): `kev.publish` is not needed for a card-only commit; `hf upload jaredpalmer/kev-<size> docs/model-cards/kev-<size>.md README.md --commit-message "Kev 1.0 model card (weights unchanged)"`. Check with `HfApi().model_info(..., files_metadata=True)` that `adapter_model.safetensors` / `head.pt` (27B: `model.safetensors.index.json` and every shard) hash as below.
4. **Hub tags.** `v1.0` on all four repos. Default (as specified): the exact weight revisions; if step 3 ran first, tag the card commit instead so that `@v1.0` shows the 1.0 card (the weights are byte-identical; record both commits in PLAN.md).

   | Repo | `v1.0` target (weights) | adapter / head sha256 |
   |---|---|---|
   | `jaredpalmer/kev-0.8b` | `9a45d25eb2ab761841196625383fa1dff0e56c1e` | `9b908623…` / `f400bd12…` |
   | `jaredpalmer/kev-4b` | `139fdd94f1b6a6ad80cc15e08fcb99cac885a101` | `90e81735…` / `dd633435…` |
   | `jaredpalmer/kev-9b` | `b5d8c18e44c60888d138b65cb6507ff0a5a448a0` | `2b2a70cf…` / `8e1dab2c…` |
   | `jaredpalmer/kev-27b` | `main` (today `ef78cc8a34d5f426fb229c52089db189218cfe5c`: weights `28be62e9`, then three card-only commits) | weights `d27af6ab…` / head `7968f17b…` |

   ```bash
   hf repos tag create jaredpalmer/kev-0.8b v1.0 --revision 9a45d25eb2ab761841196625383fa1dff0e56c1e -m "Kev 1.0"
   hf repos tag create jaredpalmer/kev-4b   v1.0 --revision 139fdd94f1b6a6ad80cc15e08fcb99cac885a101 -m "Kev 1.0"
   hf repos tag create jaredpalmer/kev-9b   v1.0 --revision b5d8c18e44c60888d138b65cb6507ff0a5a448a0 -m "Kev 1.0"
   hf repos tag create jaredpalmer/kev-27b  v1.0 --revision <main at release> -m "Kev 1.0"
   ```

5. **Assets.** `uv run python scripts/build_release_assets.py --release docs/releases/kev-1.0-assets.json --out /tmp/kev-1.0-assets` builds `kev-0.8b.tar.gz`, `kev-4b.tar.gz`, `kev-9b.tar.gz` (each: the Hub snapshot at the revision above, i.e. adapter, `head.pt` with the temperature, tokenizer files, the trial's `result.json`, `provenance.json`, `training_config.json` and `training_metrics.json`; the 1.0 card as `README.md`; the locked read as `locked_test.json`), `SHA256SUMS.txt` and `manifest.json` (every member's sha256). It refuses a download whose adapter or head hash differs from the spec. Kev-27B is not an asset (51 GB; GitHub caps an asset at 2 GB): the notes point to the Hub.
6. **GitHub release.** Tag `kev-1.0` on the merge commit; create the release as a draft with the published part of these notes as its body (everything above this section), attach the three tarballs and `SHA256SUMS.txt`, download them back, `shasum -a 256 -c SHA256SUMS.txt`, extract one and serve it, then publish and mark it Latest.
7. **One release per size.** The release policy keeps only the best version of each size in a GitHub release. Once `kev-1.0` is published, `kev-family` duplicates it: delete its three tarballs and `SHA256SUMS.txt` and replace its body with a pointer to `kev-1.0` (or delete the release; Jared's call). Earlier versions stay on the Hub tags listed in each card.
8. **Deploy pins.** `skills/kev-deploy` and `skills/kev-finetune` pin `KEV_REF` 71d4829; the 1.0 checkpoints need no newer code. Move the pin only if a later serving fix should ship with 1.0.
9. **Collection and Space.** The Kev collection already lists the four repos. The Space serves Kev-4B and Kev-0.8B from `main`, which is the 1.0 weights; nothing to republish unless `kev/model.py`, `kev/api.py` or `kev/checkpoint.py` changed after its last publish.
