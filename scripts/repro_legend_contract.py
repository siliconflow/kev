"""OpenRouter legend contract repro: string/object/array/mixed score levels.

Mirrors the exact code path kev.serve uses for /v1/systemone: SystemOneRequest ->
api.to_record -> (model probs) -> api.to_answers -> response JSON. We substitute
fixed probabilities for the model, then assert the JSON round-trip keeps the
caller's criteria types in legend[i].
"""
import json
import sys

sys.path.insert(0, '.')
from kev.api import SystemOneRequest, to_record, to_answers

REQUEST = {
    "state": "Support ticket: app crashes on export.",
    "model": "kev-latest",
    "questions": {
        "string_levels": {"type": "score", "instructions": "Severity?", "criteria": [
            "low", "medium", "high"]},
        "object_levels": {"type": "score", "instructions": "Severity?", "criteria": [
            {"what": "cosmetic", "examples": ["typo", "spacing"]},
            {"what": "functional", "examples": ["broken export"]}]},
        "array_levels": {"type": "score", "instructions": "Severity?", "criteria": [
            ["cosmetic", "no functional impact"], ["blocks the workflow"]]},
        "mixed_levels": {"type": "score", "instructions": "Severity?", "criteria": [
            "no issue",
            {"what": "cosmetic", "examples": ["typo"]},
            ["severe", "data loss"]]},
    },
}

CRITERIA = {qid: q["criteria"] for qid, q in REQUEST["questions"].items()}

req = SystemOneRequest.model_validate(REQUEST)
rec, meta = to_record(req)
probs = [[0.2, 0.5, 0.3] if len(m["legend"]) == 3 else [0.4, 0.6] for m in meta]
answers = to_answers(probs, meta)

results = {}
for qid in REQUEST["questions"]:
    legend = answers[qid]["legend"]
    ok_types = all(
        type(legend[str(i)]) is type(c) and legend[str(i)] == c
        for i, c in enumerate(CRITERIA[qid])
    )
    results[qid] = "PASS" if ok_types else "FAIL"

print(f"{'case':<15} result")
for k, v in results.items():
    print(f"{k:<15} {v}")

# JSON round-trip must keep types (this is what OpenRouter receives)
wire = json.loads(json.dumps({"answers": answers}))
for qid, crit in CRITERIA.items():
    legend = wire["answers"][qid]["legend"]
    for i, c in enumerate(crit):
        assert json.dumps(legend[str(i)], sort_keys=True) == json.dumps(c, sort_keys=True), \
            f"{qid} level {i} changed on the wire: {legend[str(i)]!r} != {c!r}"

print("\nwire round-trip: types preserved for all cases")
print("\n--- sample mixed_levels legend ---")
print(json.dumps(wire["answers"]["mixed_levels"]["legend"], indent=2))
sys.exit(0 if all(v == "PASS" for v in results.values()) else 1)
