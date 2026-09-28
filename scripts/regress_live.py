"""Production regression for the legend fix on the live endpoint.

Covers:
1. OpenRouter's four score cases (string/object/array/mixed levels) — legend must echo
   the caller's criteria levels with original JSON types.
2. Noul / Choice shapes (API contract).
3. /v1/models.
4. 422 on invalid requests.
5. Latency sanity.
"""
import json
import ssl
import sys
import time
import urllib.request
import urllib.error

BASE = "https://fns6pz5pt7.fn.6scloud.com"
KEY = "sk-cwjinzkmhsveuelwbnffjqimgeewmaubhhpifpdbvnmwezeb"

SCORE_CASES = {
    "string_levels": ["low", "medium", "high"],
    "object_levels": [
        {"what": "cosmetic", "examples": ["typo", "spacing"]},
        {"what": "functional", "examples": ["broken export"]},
    ],
    "array_levels": [
        ["cosmetic", "no functional impact"],
        ["blocks the workflow"],
    ],
    "mixed_levels": [
        "no issue",
        {"what": "cosmetic", "examples": ["typo"]},
        ["severe", "data loss"],
    ],
}

failures = []


def call(method, path, body=None, auth=True, timeout=120):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if auth:
        req.add_header("Authorization", f"Bearer {KEY}")
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


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"{status}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        failures.append((name, detail))


print("=== 1. OpenRouter score legend cases (typed echo) ===")
req = {
    "state": "Support ticket: the export button crashes the app on large PDFs; the title bar also shows a typo.",
    "model": "kev-latest",
    "questions": {qid: {"type": "score", "instructions": "Severity of the described issue?", "criteria": criteria}
                  for qid, criteria in SCORE_CASES.items()},
}
code, body, ms = call("POST", "/v1/systemone", req)
check("score request returns 200", code == 200, f"HTTP {code}: {str(body)[:200]}")
if code == 200:
    for qid, criteria in SCORE_CASES.items():
        legend = body["answers"][qid]["legend"]
        ok = all(
            type(legend[str(i)]) is type(c) and json.dumps(legend[str(i)], sort_keys=True) == json.dumps(c, sort_keys=True)
            for i, c in enumerate(criteria)
        )
        check(f"legend types preserved: {qid}", ok,
              f"criteria={json.dumps(criteria)[:80]} legend={json.dumps(legend)[:80]}")
        check(f"score/probabilities present: {qid}",
              isinstance(body["answers"][qid]["score"], (int, float)) and isinstance(body["answers"][qid]["probabilities"], dict))
    print(f"       (one packed request, {len(SCORE_CASES)} score questions, {ms:.0f} ms round trip)")

print("\n=== 2. Noul + Choice shapes ===")
req = {
    "state": "Customer: Anna. Ticket #777: refund requested for a broken mug, purchased 2026-09-01.",
    "model": "kev-latest",
    "questions": {
        "billing": {"type": "noul", "instructions": "About billing?",
                    "criteria": {"true": "Charges or refunds", "false": "Not charges"}},
        "department": {"type": "choice", "instructions": "Route to which team?",
                       "criteria": {"returns": None, "shipping": "delivery issues", "billing": "charges"}},
        "urgency": {"type": "score", "instructions": "How urgent?",
                    "criteria": ["can wait", "this week", "today"]},
    },
}
code, body, ms = call("POST", "/v1/systemone", req)
check("mixed request returns 200", code == 200, f"HTTP {code}: {str(body)[:200]}")
if code == 200:
    n, c, s = body["answers"]["billing"], body["answers"]["department"], body["answers"]["urgency"]
    check("noul shape", n.get("type") == "noul" and isinstance(n.get("noul"), (int, float)), str(n))
    check("choice shape", c.get("type") == "choice" and c.get("choice") in ("returns", "shipping", "billing")
          and isinstance(c.get("probabilities"), dict) and "confidence" in c, str(c)[:120])
    check("score(3 str levels) legend", s.get("legend") == {"0": "can wait", "1": "this week", "2": "today"}, str(s)[:120])
    check("usage fields", "input_tokens" in body.get("usage", {}) and "output_tokens" in body.get("usage", {}))
    check("latency_ms present", "latency_ms" in body, str(body.get("latency_ms")))

print("\n=== 3. GET /v1/models ===")
code, body, ms = call("GET", "/v1/models", auth=True)   # the SF gateway requires the key on every path
check("/v1/models returns 200", code == 200, f"HTTP {code}")
if code == 200:
    names = [m.get("name") or m.get("id") for m in body.get("models", [])]   # gateway card uses `id`, kev.serve uses `name`
    check("kev-latest listed", "kev-latest" in names, str(names))

print("\n=== 4. Validation: invalid request -> 422 ===")
code, body, ms = call("POST", "/v1/systemone", {"state": "x", "model": "kev-latest",
                                                "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["only-one"]}}})
check("score with 1 level rejected 422", code == 422, f"HTTP {code}: {str(body)[:120]}")

code, body, ms = call("POST", "/v1/systemone", {"state": "x", "model": "kev-latest", "questions": {}})
check("empty questions rejected 422", code == 422, f"HTTP {code}: {str(body)[:120]}")

print("\n=== summary ===")
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for name, detail in failures:
        print(f"  - {name}: {detail}")
    sys.exit(1)
print("ALL PASS")
