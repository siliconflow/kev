"""The long-state checks of deploy_check.json against a deployed endpoint (b: the 70,065-token state, c: two ~60k states, each
    twice, then the first again), with /v1/models prefix-cache stats after each request:
    KEV_URL=https://<app>-api.modal.run KEV_API_KEY=... python runs/kev-deploy-71d4829/long_checks.py <dir> long_checks.json
<dir> holds long70k.json / long60k.json (runs/kev-deploy-2ea5660/make_long_state.py jaredpalmer/kev-27b 70000 | 60000, run in
a checkout of 2ea5660 for its docs/model-cards) and long60k_b.json (long60k.json with "order 8812" -> "order 9317": another state)."""
import json, os, sys, time, urllib.request, urllib.error, datetime
URL = os.environ["KEV_URL"]
KEY = os.environ["KEV_API_KEY"]
H = {"authorization": f"Bearer {KEY}", "content-type": "application/json"}


def call(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body else None, headers=H, method="POST" if body else "GET")
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            code, data, hdr = r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as e:
        code, data, hdr = e.code, json.loads(e.read()), dict(e.headers)
    return {"http": code, "body": data, "round_trip_s": round(time.time() - t, 2), "server-timing": hdr.get("server-timing") or hdr.get("Server-Timing")}


def cache():
    return call("/v1/models")["body"]["models"][0]["prefix_cache"]


out = {"started_utc": datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat(timespec="seconds") + "Z", "steps": []}
out["cache_before"] = cache()
for label, f in [("b_70k", "long70k.json"), ("c_60k_a_first", "long60k.json"), ("c_60k_a_repeat", "long60k.json"),
                 ("c_60k_b_first", "long60k_b.json"), ("c_60k_b_repeat", "long60k_b.json"), ("c_60k_a_again", "long60k.json")]:
    r = call("/v1/systemone", json.load(open(f"{sys.argv[1]}/{f}")))
    r["cache_after"] = cache(); r["label"] = label; r["file"] = f
    print(label, r["http"], r["round_trip_s"], r["body"].get("latency_ms"), r["body"].get("detail"), r["cache_after"], flush=True)
    out["steps"].append(r)
out["ended_utc"] = datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"
json.dump(out, open(sys.argv[2], "w"), indent=1)
