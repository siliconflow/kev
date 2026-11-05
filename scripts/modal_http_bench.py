"""Paired Modal front doors: one Kev-4B/H100/us-east container each, ephemeral bearer key, no production deploy.

    uv run --extra serve modal run scripts/modal_http_bench.py --out runs/modal-http-4b-h100

Set KEV_HTTP_TARGET=64 to retry the rejected 128/64 web variant (the default rejected variant is 128/32).

Uses modal_app's locked GPU image. The attached app shuts down on exit. Request/model/process times and graph counts
are recorded separately: a new batch shape can still capture a graph during a measurement after warm-up.
"""
import asyncio
import json
import os
import secrets
import statistics
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modal_app import image, hf_cache

app = modal.App("kev-http-bench")
KEY = os.environ.get("KEV_PILOT_KEY") or secrets.token_hex(24)
TARGET_INPUTS = int(os.environ.get("KEV_HTTP_TARGET", "32"))
if TARGET_INPUTS not in (32, 64): raise ValueError("KEV_HTTP_TARGET must be 32 or 64")
image = image.add_local_file(Path(__file__).resolve().parents[1] / "modal_app.py", "/root/modal_app.py") if modal.is_local() else image
secret = modal.Secret.from_dict({"KEV_API_KEY": KEY})
OPTIONS = dict(image=image, gpu="H100", region="us-east", cpu=4, memory=(16384, 131072),
               volumes={"/hf": hf_cache}, secrets=[secret], max_containers=1, scaledown_window=60,
               timeout=600, startup_timeout=600)


class Loaded:
    @modal.enter()
    def load(self):
        import torch
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.serve import Server, app as api
        from kev.predictors import kernel_environment
        from scripts.serving_bench import request
        ck = Checkpoint("jaredpalmer/kev-4b")
        tok, model = ck.load("cuda", LoadOptions(dtype=torch.bfloat16, cuda_graphs=True, fused=True))
        self.server = Server(ck, tok, model, "cuda")
        for case in ("2 questions, short state", "6 questions, short state", "5 questions, 370-token state"):
            self.server.answer(request(case, 1))
        self.server.wait_idle()
        api.state.server = self.server
        @api.get("/v1/pilot/settle")
        def settle():
            self.server.wait_idle()
            return {"graphs": model.graphs.stats(), "gpu": torch.cuda.get_device_name(0), "threads": torch.get_num_threads(),
                    "checkpoint": Path(ck.path).name, "environment": kernel_environment(model, "cuda")}
        self.api = api

    def start_http(self):
        import threading
        import uvicorn
        threading.Thread(target=uvicorn.run, args=(self.api,),
                         kwargs={"host": "0.0.0.0", "port": 8000, "access_log": False}, daemon=True).start()

    @modal.exit()
    def close(self):
        self.server.close()


@app.cls(**OPTIONS)
@modal.concurrent(max_inputs=64, target_inputs=32)
class Baseline(Loaded):
    @modal.asgi_app()
    def web(self):
        return self.api


@app.cls(**OPTIONS)
@modal.concurrent(max_inputs=128, target_inputs=TARGET_INPUTS)
class Doubled(Loaded):
    @modal.asgi_app()
    def web(self):
        return self.api


@app.server(**{k: v for k, v in OPTIONS.items() if k not in {"region", "timeout"}},
            compute_region="us-east", routing_region="us-east", target_concurrency=64,
            min_containers=1, unauthenticated=True, exit_grace_period=120)
class Direct(Loaded):
    @modal.enter()
    def start(self):
        self.start_http()


@app.function(image=image, secrets=[secret], region="us-east", cpu=4, timeout=600)
async def bench(urls, levels):
    import httpx
    from scripts.serving_bench import request
    out = {}
    for name, url in urls.items():
        async with httpx.AsyncClient(base_url=url, headers={"Authorization": f"Bearer {os.environ['KEV_API_KEY']}"},
                                     timeout=300, follow_redirects=True, limits=httpx.Limits(max_connections=128, max_keepalive_connections=128)) as client:
            for _ in range(600):
                r = await client.get("/v1/models")
                if r.status_code == 200: break
                assert r.status_code in (502, 503), r.status_code
                await asyncio.sleep(.5)
            else: raise RuntimeError(f"{name} did not become ready")
            assert (await client.get("/v1/models", headers={"Authorization": "Bearer invalid"})).status_code == 401
            card = r.json()
            assert {"kev-latest", "jev-latest"} <= {m["name"] for m in card["models"]}
            bad = await client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "score", "criteria": []}}})
            assert bad.status_code == 422
            from typesafe_sdk import AsyncTypeSafeClient, Noul
            async with AsyncTypeSafeClient(api_key=os.environ["KEV_API_KEY"], base_url=url, model="kev-latest") as sdk:
                answer = await sdk.system_one(state="I was charged twice.", questions={"billing": Noul(instructions="Is this about billing?")})
            assert 0 <= answer.nouls["billing"].noul <= 1 and answer.request_id and answer.usage.input_tokens > 0
            async def one(i):
                body = request("6 questions, short state", i).model_dump()
                t = time.perf_counter()
                r = await client.post("/v1/systemone", json=body)
                elapsed = (time.perf_counter() - t) * 1000
                if r.status_code != 200: return elapsed, None, None, r.status_code
                result = r.json()
                return elapsed, float(r.headers["server-timing"].split("dur=")[1]), result["latency_ms"], 200
            for i in range(5): await one(i + 1)
            out[name] = {"url": url, "levels": {}, "model_card": card,
                         "protocol": {"invalid_bearer": 401, "invalid_question": 422, "sdk": True},
                         "settle": (await client.get("/v1/pilot/settle")).json()}
            sample = await client.post("/v1/systemone", json=request("6 questions, short state", 0).model_dump())
            sample.raise_for_status(); out[name]["answers"] = sample.json()["answers"]
            for c in levels:
                sem = asyncio.Semaphore(c)
                async def limited(i):
                    async with sem: return await one(i + 1000)
                await asyncio.gather(*(limited(i) for i in range(max(64, c * 4))))
                settled = await client.get("/v1/pilot/settle"); settled.raise_for_status()
                n = max(128, c * 8)
                start = time.perf_counter()
                rows = await asyncio.gather(*(limited(i + 10000) for i in range(n)))
                wall = time.perf_counter() - start
                good = [row for row in rows if row[3] == 200]
                errors = [row[3] for row in rows if row[3] != 200]
                if not good: raise RuntimeError(f"{name} {c}: no successful requests")
                elapsed, process, model, _ = zip(*good)
                result = {"requests": n, "errors": errors, "rps": round(len(good) / wall, 2), "p50_ms": round(statistics.median(elapsed), 2),
                          "p99_ms": round(sorted(elapsed)[int(.99 * (len(elapsed) - 1))], 2),
                          "process_p50_ms": round(statistics.median(process), 2), "model_p50_ms": round(statistics.median(model), 2),
                          "graphs_before": settled.json(), "graphs_after": (await client.get("/v1/pilot/settle")).json()}
                out[name]["levels"][c] = result
                print(name, c, result, flush=True)
    return out


@app.local_entrypoint()
def main(out: str = "runs/modal-http-4b-h100", levels: str = "1,8,32,64,128"):
    from kev.suite import write_json
    urls = {"asgi": Baseline().web.get_web_url(), "doubled": Doubled().web.get_web_url(), "direct": Direct.get_url()}
    print(urls, flush=True)
    counts = [int(c) for c in levels.split(",")]
    if not counts or any(c < 1 or c > 128 for c in counts): raise ValueError("levels must be between 1 and 128")
    result = bench.remote(urls, counts)
    reference = next(iter(result.values()))["answers"]
    result["answers_equal"] = all(row["answers"] == reference for row in result.values())
    result["configuration"] = {"asgi": {"max_inputs": 64, "target_inputs": 32},
                               "doubled": {"max_inputs": 128, "target_inputs": TARGET_INPUTS}, "direct": {"target_concurrency": 64}}
    dest = Path(out) / "report.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, result)
    if not result["answers_equal"] or any(level["errors"] for row in result.values() if isinstance(row, dict) and "levels" in row
                                           for level in row["levels"].values()):
        raise RuntimeError(f"Parity or HTTP errors in {dest}")
    print(json.dumps(result, indent=2), flush=True)
