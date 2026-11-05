---
name: kev-deploy
description: Deploy a Kev decision model (the open Jev-style System One model) as the user's own TypeSafe-compatible HTTPS endpoint on Modal with one command, wire it into their code, and take it down. Use when someone wants to host Kev, get a Kev API URL, replace Jev / TypeSafe calls with a self-hosted model, pick a Kev size or GPU, add an API key, keep an endpoint warm, or stop and remove a Kev deployment.
license: Apache-2.0
compatibility: Requires Python 3.10+ and a Modal account (`pip install modal==1.5.5 && modal setup`, free tier works for the small models). No GPU, no clone of the Kev repo, no Hugging Face account for the public checkpoints.
metadata:
  author: jaredpalmer
  version: "1.1"
  repository: https://github.com/jaredpalmer/kev
---

# Deploy Kev on Modal

`scripts/kev_serve.py` is a single self-contained Modal app. `modal deploy` on it builds an image with the kev package at
a pinned commit, loads a Kev checkpoint from the Hugging Face Hub on a GPU sized for it, and serves TypeSafe's System One
protocol (`POST /v1/systemone`, `GET /v1/models`) at `https://<workspace>--kev-api.modal.run`. It scales to zero when idle.
The file is short on purpose: if the user's case does not fit, read it and edit it.

## 1. Check the prerequisites

```bash
pip install modal==1.5.5
modal profile current 2>/dev/null || modal setup      # opens a browser to sign in; the user must do this step
modal skills install -y                               # optional: Modal's own agent skill + docs (into .agents/, or -g for home)
```

If `modal setup` is needed, tell the user and wait; do not try to authenticate for them. `modal skills install` gives you
Modal's official skill and documentation, which helps with anything beyond this file: GPUs, secrets, volumes, logs, billing.

## 2. Pick the model

| `KEV_MODEL` | GPU (automatic; fallbacks in parentheses) | $/h while up | Model time, 6 questions (new / repeated state) | Cold start (cached weights) | When |
| --- | --- | --- | --- | --- | --- |
| `jaredpalmer/kev-0.8b` | L4 (L40S) | 0.80 | 23 / 16 ms | ~40 s | cheapest, prototyping |
| `jaredpalmer/kev-4b` (default) | L40S (H100) | 1.95 | 42 / 28 ms (H100: 18 / 13 ms) | ~35 s | the default: best quality per dollar |
| `jaredpalmer/kev-9b` | H100 (H200, L40S) | 3.95 | 24 / 17 ms | ~55 s | accuracy on smaller GPUs |
| `jaredpalmer/kev-27b` | B200 (H200, H100) | 6.25 | 47 / 32 ms (H200: 67 / 50 ms) | ~50 s | best released accuracy; 51 GB of full weights |

Model time is the `latency_ms` the API returns (median of 20 requests, measured in the Kev repo: `runs/serve-*`,
`runs/grouping-4b-h100`, `runs/fused-27b-*` and `runs/serving-27b-r23`; Kev-27B's B200 and H100 figures were measured on its
previous version, the same architecture in bf16). A new state is the normal call, since every ticket is a
new state; a repeated state is served from a prefix cache. The very first cold start of an account also downloads the
weights and compiles kernels (1-2 minutes); both are cached on the `kev-hf-cache` volume afterwards. Other GPUs work with
`KEV_GPU` but are worse picks: an L4 runs out of compute on Kev-4B, and an A100 is slower than an L40S here and costs more.
Kev-27B is compute-bound under load: a B200 serves ~57 mixed requests/s at 64 concurrent clients (H200 ~40, H100 ~36) for
about the same cost per request, with the lowest latency. Its first cold start downloads 51 GB of weights (several minutes).

A warm container costs the GPU's hourly rate only while it is up; after five idle minutes it scales to zero.
`KEV_MIN_CONTAINERS=1` keeps one warm (no cold starts, pays the hourly rate all the time). `@revision` pins a checkpoint
revision (`jaredpalmer/kev-4b@v7-base`).

## 3. Deploy

Always set an API key unless the user explicitly wants a public URL; without one, anyone with the URL can spend their
GPU time.

```bash
curl -LO https://raw.githubusercontent.com/jaredpalmer/kev/main/skills/kev-deploy/scripts/kev_serve.py   # or use the skill's copy
export KEV_API_KEY=$(openssl rand -hex 24)                # save it: it is the endpoint's bearer token
KEV_MODEL=jaredpalmer/kev-4b modal deploy kev_serve.py    # prints the URL
```

Settings are read at deploy time; redeploying with other values replaces the model behind the same URL.
`KEV_APP_NAME=kev-9b` gives a second, independent endpoint (`https://<workspace>--kev-9b-api.modal.run`).
`KEV_GPU=H100` overrides the GPU list (comma-separated). `KEV_REGION=us` (or `us-east`, `eu`, ...) pins where the container
runs: without it Modal takes the first region with a free GPU, which can be another continent (an unpinned Kev-4B landed in
Frankfurt and added ~150 ms to every round trip from the US). A pinned region costs 1.15-1.75x on Modal; pin it near the
callers for latency-sensitive use. `KEV_TRUNCATE_STATES=1` reads the first 65,536 tokens of a longer document instead
of refusing it (see Long documents under Troubleshooting).

### Throughput

A container answers concurrent requests in batches: its model thread takes everything waiting and runs it through shared
passes. In-process, Kev-4B on an H100 serves about 95 six-question requests/s (120 on mixed short records); an L40S about
45. Over HTTP the front door matters too: `KEV_FLASH=1` uses a Modal Server rather than the Function input plane, with
the same Uvicorn/FastAPI API, GPU worker and bearer check. It now targets 64 concurrent requests per container instead
of 32. The ordinary web endpoint retains its 64-input burst limit and autoscaling target of 32: experiments with
128/32 and 128/64 regressed throughput and median latency, so neither is shipped.

The [paired HTTP benchmark](../../runs/modal-http-4b-h100/report.json) uses Kev-4B on one H100 per variant, four CPUs,
`us-east`, an in-region client, kept-alive connections and six questions per request. At one client, warm p50 fell
from 81 to 34 ms. At 64 clients, p50 fell from 1,236 to 652 ms and throughput rose from 44 to 66 requests/s; p99 was
4.5 versus 4.0 seconds, **not** halved. The rejected web 128/32 variant fell to 37 requests/s at 64 clients. A
[separate 128/64 pilot](../../runs/modal-http-4b-h100/target64/report.json) fell from 56 to 36 requests/s there.
Both runs used the same checkpoint per variant and returned identical six-question answers and no HTTP errors.
An [actual deployment-script smoke](../../runs/modal-http-4b-h100/deploy-smoke/report.json), built independently from the
pinned Git commit, also checked bearer auth, request validation, the SDK, answer parity against the paired run and
20 warm serial requests (model p50 18.4 ms; its client is not in-region, so its RTT is not a paired comparison).
The reports record end-to-end, process and model time separately, status errors and CUDA graph counts. The model path and deployment image are unchanged;
the fused DeltaNet forward uses FLA's own convolution. Shape capture can still happen during a load phase;
these are small pilots, not a sustained traffic or autoscaling SLO. Reproduce from the repository:

```bash
uv run --extra serve modal run scripts/modal_http_bench.py --out runs/modal-http-4b-h100 --levels 1,32,64,128
```

Direct mode remains opt-in: set `KEV_FLASH=1 KEV_REGION=us-east` before the deployment command in Step 3. It requires Modal 1.5.5,
prints `https://<workspace>--<app>-kev.<region>.modal.direct`, and keeps at least one paid GPU container up. It can return
503 while starting; warm it with a redirect-following `/v1/models` call before routing traffic. For steady high traffic,
keep enough containers warm (`KEV_MIN_CONTAINERS=2` or more) so bursts do not wait for a cold start. Both front doors
still queue under overload: a 64-request target is not a promise of low p99 at 128 clients.

## 4. Verify

The first request after a deploy or an idle period waits for the cold start. Modal answers a request that waits longer
than 150 s with an HTTP 303 to a result URL, so warm the endpoint with a redirect-following call first:

```bash
curl -sL --max-time 900 $KEV_URL/v1/models -H "authorization: Bearer $KEV_API_KEY"   # returns once the model is loaded
```

```bash
curl -s $KEV_URL/v1/systemone -H "authorization: Bearer $KEV_API_KEY" -H 'content-type: application/json' -d '{
  "state": "Order 4411 arrived late and the box was crushed. Two charges appear on my card.", "model": "kev-latest",
  "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?",
                         "criteria": {"returns": "Exchanges, refunds", "shipping": "Delivery, delays", "billing": "Charges, payments"}},
                "urgent": {"type": "noul", "instructions": "Does this need urgent human attention?"}}}'
curl -s $KEV_URL/v1/models -H "authorization: Bearer $KEV_API_KEY"       # served checkpoint, base, temperature
```

Expect per-question `probabilities` (calibrated by the checkpoint's own temperature), `choice` / `noul` / `score`, and
`latency_ms`, the model time: tens of milliseconds warm (table above). The round trip adds the network and Modal's proxy,
about 80-100 ms from a client in the US to a us-east container over a kept-alive connection, more with a new TLS
connection per request, so reuse one HTTP client. The first request of a new shape (question set, state length) runs
without a CUDA graph, about 2-4x slower; the server captures one in the background and later requests use it.
`/v1/models` reports the served checkpoint, its temperature and the number of captured graphs. A request without the key
must return 401.

## 5. Wire it in

The protocol is TypeSafe's, so only the base URL and key change:

```python
client = TypeSafeClient(api_key=KEV_API_KEY, base_url=KEV_URL, model="kev-latest")   # was: TypeSafeClient(api_key=TYPESAFE_KEY)
```

```ts
const r = await fetch(`${KEV_URL}/v1/systemone`, { method: "POST",
  headers: { "content-type": "application/json", authorization: `Bearer ${KEV_API_KEY}` },
  body: JSON.stringify({ state, model: "kev-latest", questions }) });   // question type "noul", not "boolean"
```

Kev answers typed questions about a state (Choice over named options, Noul yes/no, Score over ordered levels) without
generating text. It was trained on public classification, policy and rule data; on the user's own domain, measure it on a
few hundred labelled examples before relying on it, and if it falls short, fine-tune it with the `kev-finetune` skill.

## 6. Take it down

```bash
modal app stop kev            # or the KEV_APP_NAME used; the URL stops working immediately
modal volume delete kev-hf-cache   # optional: the cached weights (shared with kev-finetune; next deploy re-downloads)
```

## Troubleshooting

- **First request returns nothing or a 303**: the cold start is still running (weights download on the very first start,
  or Modal is still finding a GPU; `modal app logs kev` says "waiting to be scheduled"). Follow redirects (`curl -L`), use a
  longer client timeout, or deploy with `KEV_MIN_CONTAINERS=1`.
- **CUDA out of memory on start**: the GPU is too small for that checkpoint (Kev-9B needs ~18 GB, Kev-4B ~10 GB); use the
  table above or `KEV_GPU=H100`.
- **Slow round trips with fast `latency_ms`**: the container is far from the caller or every request opens a new
  connection; set `KEV_REGION` and reuse the HTTP client.
- **401 with the right key**: the key is fixed at deploy time; redeploy with the same `KEV_API_KEY` exported.
- **Long documents**: a state may have up to 65,536 tokens and a question with its options 8,192 more (more when the
  state is shorter). A longer state gets a 422 that gives its token count and the limit (the TypeSafe SDK raises
  `TypeSafeUnprocessableEntityError` with that message), as a longer question does: shorten or split the document.
  Deploying with `KEV_TRUNCATE_STATES=1` reads only the first 65,536 tokens instead, and then every response carries
  `truncated` and `usage.state_tokens` / `state_tokens_used`. Kev-27B
  trained on states of up to 32,768 tokens and the smaller models on 384, so all four accept long documents but Kev-27B
  answers them best. A new long state is slow, a repeated one is served from the prefix cache: Kev-27B on an H200 took
  8-10 s for a 60k-token state and 0.28 s for the same request again; a 70k-token state read to 65,536 tokens with
  `KEV_TRUNCATE_STATES=1` took 12 s, then 0.29 s. The cache holds 65,536 state tokens in all, so a new state that long
  replaces the cached one, which is dropped before the new state's pass rather than after it (no second copy resident)
  (`runs/kev-deploy-71d4829` and `runs/kev-deploy-2ea5660` in the repo).
- **Logs**: `modal app logs kev` shows the load line (`serving <model> on <GPU> ... ready in Ns`) and every request.
