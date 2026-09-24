# Kev — prototype of a Jev-style decision model

Causal LM (Qwen2.5-0.5B + LoRA) run prefill-only with a block-causal mask (shared state prefix,
isolated question branches) and a pointer readout over option boundary tokens, trained with log loss
on converted public datasets (Banking77, BoolQ, AG News, MNLI, SST-5, Yelp). No text generation.
See README.md (deep dive) and docs/model-cards/ (one card per checkpoint: recipe + metrics; kev-0.5b.md is the superseded prototype). README follows the Vercel Labs house style (tagline, for-the-badge badges, Highlights, Title Case sections, API tables, Authors + License); MODEL_CARD.md is formal.

## Commands
- Env: `uv sync` (torch MPS, transformers, peft, datasets)
- Train: `uv run python -m kev.train --n_per_source 1500 --epochs 2 --out runs/kev` (~1h45m on M5 32GB)
  - `--holdout mnli,sst5` excludes sources (out-of-source eval); `--perm_kl/--perm_frac` permutation-consistency KL;
    `--ord_w` ordinal term for Score. Only one training process at a time: two on MPS slow each other ~10x.
- Eval:  `uv run python -m kev.evaluate --run runs/kev --n_per_source 150 --baseline --baseline_instruct Qwen/Qwen2.5-0.5B-Instruct`
  -> `runs/kev/eval.json` (acc/ECE/NLL per source, temperature scaling, permutation, IIA, isolation, packed-vs-separate, held-out sources)
- Smoke: `--n_per_source 40 --accum 4 --out runs/smoke` (~1 min)
- Research suite: `evals/decision-v1` (frozen, checksummed; train/calibration/development/test; manifest pins dataset + base
  revisions). `kev.suite` freezes; `kev.benchmark --run X --suite evals/decision-v1 --out runs/...` scores development;
  `--allow-test` is the only way to read the locked test. `kev.experiment --plan experiments/*.json` runs config-only trials
  (bounded allowlist, provenance, coverage/isolation gates, results.jsonl ledger, `--wait-pid` to queue behind a training job).
  `kev.jev` scores Jev via Vercel AI Gateway (AI SDK 7 `experimental_evaluate`, node worker in `playground/scripts/`;
  needs `AI_GATEWAY_API_KEY` or `--provision-scope`; budget-capped). `kev.compare` pairs two result dirs (record-clustered
  bootstrap). Historical checkpoints (`runs/kev`, `runs/kev2`) overlap the suite's training data: exploratory only.
  `--ord_w` is now the ranked probability score (proper); the old |E[level]-y| term was removed. `--perm_kl`/`--ord_w` default 0.
- Modal (default for anything beyond smoke): `modal_app.py`; `uv run modal run modal_app.py::{smoke,study,evaluate}`. Image = `uv_sync`
  of pyproject/uv.lock (Linux torch wheel is CUDA) + `kev/` + `evals/`; Volumes `kev-hf-cache` (HF_HOME) and `kev-runs` (trial outputs,
  pulled to runs/<study> then ranked by `kev.experiment --aggregate`). `KEV_GPU` picks the GPU type (H100 default). Legacy checkpoints:
  Hub id, or `modal volume put kev-runs runs/<run> /legacy/<run>` then `--existing /runs/legacy/<run>`. Eval on CUDA is fp32-exact
  (TF32 + fused SDPA off in `LocalPredictor`); training keeps TF32 and may use `--dtype bf16`. `--transfer <suite>` scores OOD per trial.
- Frozen suites: `evals/<version>/<suite>/{manifest.json, *.jsonl}`. Manifests pin dataset/base revisions and the sha256 of every
  partition. Partitions over ~10 MB are not in git; they are mirrored at the Hub dataset `jaredpalmer/kev-suites` (revision pinned in
  `kev/suite.py: SUITES_REVISION`) and `load_split` fetches + verifies them on first use. After freezing a new suite:
  `hf upload jaredpalmer/kev-suites evals . --type dataset --include "*.jsonl" --include "*.json"`, bump `SUITES_REVISION`, gitignore
  the large partitions. Never modify a frozen file; new data = new version.
- Figures: `uv run python scripts/plot_family.py` and `uv run python scripts/plot_tweet.py` regenerate docs/kev-family.png and docs/kev-benchmark.png from
  saved result files. Style lives in `scripts/chartstyle.py` (Geist type, Vercel color tokens, direct labels, no legends, one label/plot/value lane per bar set);
  new figures should import it rather than set their own rcParams. `kev.plot` (loss curves from train logs) is a debugging aid, not a README figure.
- Current family (2026-09-20, all Qwen3.5): `jaredpalmer/kev-9b` (`q35-9b/01-trial-1`), `jaredpalmer/kev-4b` (`q35-4b-s23/00-trial-0`; Qwen3 weights at tag `qwen3`),
  `jaredpalmer/kev-0.8b` (`q35-08b/02-trial-2`). Previous generation, kept for Mac latency: `kev-8b`, `kev-0.6b`, `kev-4b@qwen3` (cards `*-qwen3.md`). Qwen3.5 backbones are hybrid (Gated DeltaNet): `DecisionModel.hybrid`
  routes them through `forward_rows_batch` (one causal row per question, state repeated) and `_branch_rows_from_prefix` for serving; the packed
  block-causal mask is only valid on attention-only bases. Needs transformers>=5.17, peft>=0.21; CUDA wants `flash-linear-attention` + `triton>=3.7.1`
  (in the Modal image). MPS has no fast DeltaNet kernels (Kev-4B 0.78 s vs 0.17 s for the Qwen3 one); MLX is the planned fix. Plan and results: PLAN_Qwen35.md.
- Delta fine-tuning: `kev.train --init_from <run dir | Hub id[@rev]>` warm-starts LoRA + head (compatibility checked before load; source hashes in
  provenance; allowlisted in `kev/experiment.py` so studies can run cheap delta trials from a released checkpoint). Use lr <= 2e-5 for deltas.
- Publish: `uv run python -m kev.publish --run runs/<run> --repo jaredpalmer/kev-<size> --card docs/model-cards/<name>.md` (needs `hf auth login`). Repos are named by
  base model size (Kev-0.5B = Qwen2.5-0.5B); versions within a size are Hub tags (`hf repos tag create jaredpalmer/kev-0.5b vX.Y`).
  Collection: huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd. `--run` in serve/evaluate accepts a Hub id.
- Serve: `uv run --extra serve python -m kev.serve --run runs/kev --port 8008` (falls back to runs/smoke)
  - TypeSafe-compatible: `POST /v1/systemone`, `GET /v1/models` (no auth). Playground routes under `/api/*`.
  - SDK: `TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8008", model="kev-latest")`
- Extra endpoints for the demo: `POST /v1/systemone/permute` (one Choice under n option orders), `POST /v1/systemone/separate`
  (each question alone; packed-vs-separate comparison). `/v1/systemone` also returns `latency_ms`.
- Web demo: `cd playground && npm run dev -- -p 3001` (:3000 is used by another project). Next 16 app router; `/kev/*` is
  rewritten to the FastAPI server (`KEV_API`, default http://127.0.0.1:8009). Presets live in `playground/src/lib/kev.ts`.
  - `/chess` (`src/components/chess-game.tsx`, `src/lib/chess.ts`, chess.js): legal moves -> Choice options, board -> state, Score for eval;
    games in localStorage key `kev.chess.v1`.
  - React Compiler lint forbids sync setState in effects; schedule via setTimeout or move into handlers.
  - Next 16 dev only trusts `localhost`; other hostnames need `allowedDevOrigins` or the page SSRs but never hydrates
    (no console errors). `127.0.0.1` is allowed in `next.config.ts`. Verify hydration with `agent-browser` (CDP), not curl.
- Unit tests (no weights, CI): `uv run --extra serve python -m pytest tests/test_unit.py tests/test_research.py -q`
- API tests (server must be up): `KEV_BASE_URL=http://127.0.0.1:8009 uv run --extra serve python -m pytest tests/test_api.py -q`

## Layout
- `kev/data.py`      dataset -> typed records, permutation / none-of-the-above / distractor augmentation
- `kev/model.py`     encode(), branch_mask(), PointerHead, DecisionModel
- `kev/train.py`     LoRA fine-tune, batch size 1 with grad accumulation (variable-length custom masks)
- `kev/evaluate.py`  acc/ECE, permutation stability, IIA shift, isolation probe, packed-vs-separate
- `kev/plot.py`      loss curve(s) from train logs + accuracy-vs-baselines bars from eval.json
- `kev/api.py`       TypeSafe request/response models; Noul/Choice/Score -> pointer options; confidence formulas
- `kev/serve.py`     FastAPI: /v1/systemone (+ /v1/models) and /api/* playground routes
- `tests/test_api.py` conformance against the docs' example requests + official SDK

## Notes
- Delimiters reuse existing Qwen special tokens (`<|fim_prefix|>` etc.) to avoid resizing embeddings;
  peft `trainable_token_indices` leaked memory on MPS.
- `output_hidden_states=True` on MPS blows memory; use the bare `.model` backbone's `last_hidden_state`.
- `PolyAI/banking77` uses a loading script (unsupported); use `legacy-datasets/banking77`.
- User text is tokenized via `model.user_tokens()`, which rewrites `<|name|>` -> `<¦name¦>` so callers cannot forge
  option/branch delimiter tokens (the fast tokenizer ignores `split_special_tokens`).
- Training data is built as TypeSafe-shaped requests and goes through `api.to_record()` (`data.materialize`),
  so train and serve text are identical.
- Serving path (`kev.evaluate.load` + `kev.serve`): LoRA merged in fp32 then cast (`KEV_MERGE=0` to keep unmerged), `KEV_ATTN=sdpa` default on MPS,
  `KEV_SHAPE_BUCKET=64` on MPS, state-prefix KV LRU (`KEV_PREFIX_CACHE=4`, `KEV_PREFIX_MIN_TOKENS=384`). Any change here must keep the parity
  tests in tests/test_v3.py (merged vs unmerged, prefix vs full pass, bucket padding) passing; report numbers with the fp32 unmerged path.
- Image channel (opt-in, 2026-09-24): `KEV_VISION=1` attaches the base checkpoint's untrained vision tower
  (`model.visual.*` tensors, dropped by the text-only load path) and serves image requests. `state.images`
  (list of data/https URLs) is stripped by `api.to_record` and never rendered into the text;
  `serve._probs_images` forwards through `kev/vision.py` (official AutoImageProcessor patchify + row-form
  splice, no projection: Qwen3.5 merger out_hidden == text hidden on both tested bases). Needs the `vision`
  extra (torchvision, imported by transformers' qwen2_vl processor). Without the gate, or on a text-only
  base, image requests get 422. Text path, training and the prefix cache are untouched
  (tests/test_vision.py + tests/test_v3.py parity all green). "Channel open, untrained readout": an open
  channel is not a capability claim - report TVD/separation, not bacc.

## Writing
- Use simple technical English. For README tone, use Jared's older Formik, TSDX, Razzle, and Backpack READMEs as references: explain the developer's problem, address the reader directly, and show code early. Avoid slogans, canned contrasts, and repeated claims. Keep detailed experiment history in PLAN.md and the model cards rather than repeating it in the README.
