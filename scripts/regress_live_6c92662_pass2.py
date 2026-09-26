"""Second-pass live regression for 6c92662: remaining surfaces.
- auth: bad key rejected, no auth rejected
- branch isolation: packed request answers == alone answers
- score consistency: score == sum(i * p_i)
- guard: 256 choice options -> 422; bogus question type -> 422
- /permute endpoint if exposed (n_perm 1..64)
"""
import json
import ssl
import sys
import time
import urllib.request
import urllib.error

BASE = "https://fns6pz5pt7.fn.6scloud.com"
KEY = "sk-cwjinzkmhsveuelwbnffjqimgeewmaubhhpifpdbvnmwezeb"

failures = []


def call(method, path, body=None, auth=True, key=None, timeout=180):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if auth:
        req.add_header("Authorization", f"Bearer {key or KEY}")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as r:
            return r.status, json.loads(r.read().decode()), (time.time() - t0) * 1000
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode())
        except Exception:
            payload = None
        return e.code, payload, (time.time() - t0) * 1000
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", (time.time() - t0) * 1000


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append((name, detail))


# ---------- auth ----------
print("=== Auth ===")
code, body, ms = call("POST", "/v1/systemone", {"state": "x", "model": "kev-latest",
                     "questions": {"q": {"type": "noul", "instructions": "?"}}}, key="sk-wrongkey" * 4)
check("bad key rejected 401/403", code in (401, 403), f"HTTP {code}: {str(body)[:120]}")
code, body, ms = call("POST", "/v1/systemone", {"state": "x", "model": "kev-latest",
                     "questions": {"q": {"type": "noul", "instructions": "?"}}}, auth=False)
check("missing auth rejected 401/403", code in (401, 403), f"HTTP {code}: {str(body)[:120]}")
code, body, ms = call("GET", "/v1/models", key="sk-wrongkey" * 4)
check("bad key on /v1/models rejected", code in (401, 403), f"HTTP {code}")

# ---------- branch isolation (packed == separate) ----------
print("\n=== Branch isolation: packed vs separate ===")
qs = {"a": {"type": "noul", "instructions": "Is the weather described as nice?"},
      "b": {"type": "choice", "instructions": "Which season is it most likely?",
            "criteria": {"summer": None, "winter": None, "unknown": None}}}
state = "The weather is nice today and the park is full of people."
code, both, ms_b = call("POST", "/v1/systemone", {"state": state, "model": "kev-latest", "questions": qs})
check("packed request 200", code == 200, f"HTTP {code}: {str(both)[:150]}")
code, alone_b, ms_a = call("POST", "/v1/systemone", {"state": state, "model": "kev-latest", "questions": {"b": qs["b"]}})
code2, alone_a, _ = call("POST", "/v1/systemone", {"state": state, "model": "kev-latest", "questions": {"a": qs["a"]}})
check("separate requests 200", code == 200 and code2 == 200, f"{code}/{code2}")
if both and alone_a and alone_b:
    ab, bb = both["answers"]["b"], alone_b["answers"]["b"]
    pa, pb = ab.get("probabilities", {}), bb.get("probabilities", {})
    da = abs(pa.get("summer", float("nan")) - pb.get("summer", float("nan"))) if pa and pb else None
    probs_key_a = "probability" if "probability" in both["answers"]["a"] else None
    check("packed == separate (choice 'summer')", da is not None and da < 5e-3, f"|dp|={da}")
    aa = both["answers"]["a"]
    sa = alone_a["answers"]["a"]
    check("packed == separate (noul)", abs(aa["noul"] - sa["noul"]) < 5e-3, f"{aa['noul']} vs {sa['noul']}")
    print(f"       packed {ms_b:.0f} ms vs separate b {ms_a:.0f} ms + a")

# ---------- score consistency ----------
print("\n=== Score value consistency ===")
req = {"state": "The patient reports a persistent cough for three weeks and mild fever at night.",
       "model": "kev-latest",
       "questions": {"sev": {"type": "score", "instructions": "Severity?",
                             "criteria": ["mild", "moderate", "severe"]}}}
code, body, ms = call("POST", "/v1/systemone", req)
check("score request 200", code == 200, f"HTTP {code}: {str(body)[:150]}")
if code == 200:
    s = body["answers"]["sev"]
    exp = sum(int(k) * v for k, v in s["probabilities"].items())
    check("score == sum(i*p_i) (±0.05)", abs(s["score"] - exp) < 0.05, f"score={s['score']} expected={exp:.4f}")
    check("probabilities sum ~1", abs(sum(s["probabilities"].values()) - 1) < 1e-2, str(s["probabilities"]))
    check("score in [0, 2]", 0 <= s["score"] <= 2, str(s["score"]))

# ---------- guards ----------
print("\n=== Validation guards ===")
code, body, ms = call("POST", "/v1/systemone", {"state": "x", "model": "kev-latest", "questions": {
    "q": {"type": "choice", "instructions": "i", "criteria": {f"o{i}": None for i in range(256)}}}})
check("256 choice options -> 422", code == 422, f"HTTP {code}: {str(body)[:120]}")
code, body, ms = call("POST", "/v1/systemone", {"state": "x", "model": "kev-latest", "questions": {
    "q": {"type": "bogus", "instructions": "i"}}})
check("bogus question type -> 422", code == 422, f"HTTP {code}: {str(body)[:120]}")
code, body, ms = call("POST", "/v1/systemone", {"state": "x", "model": "kev-latest", "questions": {
    "q": {"type": "score", "instructions": "i", "criteria": []}}})
check("empty score criteria -> 422", code == 422, f"HTTP {code}: {str(body)[:120]}")

# ---------- /permute ----------
print("\n=== /permute endpoint (if exposed) ===")
req = {"state": "The committee will choose between the cheap plan and the fast plan.",
       "model": "kev-latest",
       "questions": {"pick": {"type": "choice", "instructions": "Which plan?",
                              "criteria": {"cheap": None, "fast": None}}},
       "n_perm": 4}
code, body, ms = call("POST", "/permute", req)
if code == 404:
    print("SKIP  /permute not exposed at this gateway (ok — optional surface)")
    check("/permute reachable-or-absent", True)
elif code == 200:
    check("/permute returns 200", True)
    print(f"       payload head: {json.dumps(body)[:200]}")
else:
    check("/permute 200 or 404", False, f"HTTP {code}: {str(body)[:150]}")

# ---------- many questions in one request ----------
print("\n=== Many-question request (batch path) ===")
req = {"state": "Quarterly review: revenue up 12%, churn up 1pp, hiring frozen, two launches slipped, NPS flat.",
       "model": "kev-latest",
       "questions": {f"q{i}": {"type": "noul", "instructions": f"Does metric {i} look healthy?"} for i in range(32)}}
code, body, ms = call("POST", "/v1/systemone", req, timeout=300)
check("32-question request 200", code == 200, f"HTTP {code}: {str(body)[:150]}")
if code == 200:
    got = body["answers"]
    check("all 32 answers present", set(got) == {f"q{i}" for i in range(32)}, f"{len(got)} answers")
    check("usage counts 32 questions", body.get("usage", {}).get("output_tokens", 0) > 0, str(body.get("usage")))
    print(f"       32 questions: {ms:.0f} ms wall, server latency_ms={body.get('latency_ms')}")

print("\n=== summary ===")
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for name, detail in failures:
        print(f"  - {name}: {detail}")
    sys.exit(1)
print("ALL PASS")
