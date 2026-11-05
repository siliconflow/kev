The current Kev family: **Kev-27B**, **Kev-9B**, **Kev-4B**, **Kev-0.8B**. Open decision models: typed questions in, calibrated probabilities out, one forward pass, no text generation. Kev-9B, 4B and 0.8B are LoRA adapters (r=16) plus a pointer head on a frozen Qwen base, which downloads on first load; their tarballs are attached below. **Kev-27B v2 is a full-weight checkpoint (51 GB of bf16 weights), too large for a release asset (2 GB limit), so it is on the Hub only:** [`jaredpalmer/kev-27b`](https://huggingface.co/jaredpalmer/kev-27b) (weights commit `28be62e9`). This release holds only the current best version of each size; every earlier version stays on the Hub (below).

| | base | in-distribution (dev / locked test) | out-of-domain (dev / locked test) | Brier, out-of-domain (dev / test) | runs on |
|---|---|---|---|---|---|
| **Kev-27B** (v2) | Qwen3.8-27B (post-trained), full weights | 0.865 / 0.866 | **0.851 / 0.889** | **0.225 / 0.156** | one B200, H200 or H100 80 GB (bf16, 51 GB) |
| **Kev-9B** (v2) | Qwen3.5-9B-Base | 0.874 / 0.873 | 0.820 / 0.852 | 0.289 / 0.217 | 24 GB GPU |
| **Kev-4B** | Qwen3.5-4B-Base | 0.873 / 0.865 | 0.817 / 0.838 | 0.269 / 0.242 | 12 GB GPU, Apple Silicon (MLX) |
| Kev-0.8B | Qwen3.5-0.8B-Base | 0.827 / 0.838 | 0.648 / 0.697 | 0.481 / 0.416 | 4 GB GPU, Apple Silicon (MLX) |
| Jev (hosted reference) | – | 0.845 / – | 0.857 / – | 0.211 / – | API |

Same frozen items for every row; Brier on raw logits (each checkpoint ships a fitted temperature that lowers it further). **Which to use:** Kev-27B for the best accuracy if you have a data-centre GPU (for long contracts, see its trade-offs below); Kev-4B otherwise.

**2026-09-30 update: Kev-9B v2.**
- **What it is.** Round 27's confirmed `9b-r18a`: Kev-9B v1 plus one epoch on the documents and skills data that Kev-4B and Kev-0.8B already had (documents-v1, hard-v1 and devtools-v1 train, 10,000 records replayed), trained in round 18 and re-selected on the audited rule. Served at temperature 2.19, fitted on held-out datasets it never trained on. Hub: `jaredpalmer/kev-9b` main (`b5d8c18e`), adapter sha256 `2b2a70cf4ef4440b6c22899e1f72c2f8ea5c6f65b19aa344539b4b8971d1f13d`.
- **Confirmed test numbers against v1** (registered; paired 95 % bootstrap): hard-v1 + devtools-v1 test +18.7 pp [+16.7, +20.8] (hard-v1 0.584 → 0.834), documents-v1 test +7.1 [+4.7, +9.2]; locked out-of-domain test level at 0.852 with served Brier 0.199 (v1 0.224).
- **Trade-offs.** The selection was not blind (round 18's development reads were known); its large gains are in distribution; on held-out datasets it is level with v1. [Model card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-9b.md).
- **v1** is at `jaredpalmer/kev-9b@v1`. `kev-9b.tar.gz` below is v2; `SHA256SUMS.txt` is regenerated.

**2026-09-30 update: Kev-27B v2.**
- **What it is.** Round 23's confirmed `27b-k-w85`: every weight of Qwen3.8-27B fine-tuned for one epoch on a 145,840-record corpus, then averaged 0.85 / 0.15 with Kev-27B v1's weights (its adapter merged in fp32). Served at temperature 1.32, fitted on held-out datasets neither parent trained on. Hub: `jaredpalmer/kev-27b` main (`28be62e9`), weights sha256 `d27af6ab2be16824166ac639907b4dba40979ff338599c2872721aa6c5072022`.
- **Confirmed test numbers against v1** (registered before the reads; paired 95 % bootstrap): held-out datasets (breadth-v1 test) +1.2 pp [+0.3, +2.2]; held-out task families (tasksource-heldout-v1 test) +5.3 [+3.7, +6.8]; Kev's skill, developer-tooling and document suites (test) +8.9 [+7.5, +10.3]; locked out-of-domain test 0.889 with served Brier 0.154 (bar 0.886; v1 0.896 / 0.160).
- **Trade-offs.** Not better than v1 on short states (locked −0.8 pp [−2.0, +0.5]). Worse and overconfident on long contracts: CUAD test 0.874 vs 0.890 (−1.6 [−3.0, −0.3]), ECE 0.053 vs 0.007; for contract review refit the temperature or keep v1. The confirmation reused test partitions an earlier round had read, so it is weaker evidence than a fresh read. Its skill and document gains are in distribution. [Model card](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-27b.md) has every number and caveat.
- **v1** (the LoRA adapter this release used to attach as `kev-27b.tar.gz`, now removed) is at `jaredpalmer/kev-27b@v1-lora`. `SHA256SUMS.txt` now lists the three remaining tarballs.

**2026-09-24 update.**
- **Kev-27B (new; v1, superseded 2026-09-30, now `jaredpalmer/kev-27b@v1-lora`).** On `Qwen/Qwen3.8-27B`, Qwen's instruction-tuned release (not a base model; one pre-training gate was overridden, see its card). Locked out-of-domain test 0.896 with served Brier 0.160; buried questions in long states 0.833 vs Kev-9B's 0.556; JevBench public items 0.866.
- **Kev-4B** gained two one-epoch updates: real complaint documents (CFPB narratives; locked documents test 0.804 → 0.904), then skill data (long policies, trade-offs, probability, multi-hop, dates, judging, missing facts) and developer-tooling decisions (hard-v1 test 0.540 → 0.803, devtools-v1 test 0.623 → 0.756, JevBench public hard 0.450 → 0.541).
- **Kev-0.8B** got the documents and skill data in one update (documents test 0.608 → 0.851, hard-v1 test 0.396 → 0.665, devtools-v1 test 0.472 → 0.637). Its When2Call tool-routing score fell; do not use it for tool routing.
- The document and skill gains are measured in distribution (held-out items and templates of the same sources); the cards say so first.

```bash
tar -xzf kev-4b.tar.gz
uv run --extra serve python -m kev.serve --run kev-4b --port 8009
```

Each tarball contains the adapter, `head.pt` (with the fitted temperature), tokenizer files, the model card, the trial's `result.json` and `provenance.json`, and the locked-test read (`locked_test.json`); the weights are byte-identical to the Hub. Verify: `shasum -a 256 -c SHA256SUMS.txt`. Model cards: `docs/model-cards/`. Research log: `PLAN.md`.

**Earlier versions (Hub):** Kev-27B v1 (LoRA) `jaredpalmer/kev-27b@v1-lora`; Kev-4B round 8 `jaredpalmer/kev-4b@r8-documents-release`, night2 `@night2-du-release`, pre-delta `@v7-base`, Qwen3 generation `@qwen3`; Kev-0.8B night2 `jaredpalmer/kev-0.8b@night2-du-release`, `@v7-base`; Kev-9B v1 `jaredpalmer/kev-9b@v1`, pre-delta `@v7-base`; Qwen3 `jaredpalmer/kev-0.6b` and `jaredpalmer/kev-8b`. The 0.5B prototype is at [v0.1.0](https://github.com/jaredpalmer/kev/releases/tag/v0.1.0). [Collection](https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd).



