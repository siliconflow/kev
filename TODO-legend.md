# TODO: score `legend` must echo criteria levels with original JSON types

**Status:** fixed on main via PR #1 (f7ff45f); re-applied on top of the 2026-09-25 upstream sync (same code, upstream's `question_keys` shape). Upstream still carries the bug in `to_record`. · **Severity:** blocks OpenRouter listing
**Reported:** 2026-09-25 by OpenRouter (relayed via Feishu, 王煜青) · **Repro cases:** string_levels PASS / object_levels, array_levels, mixed_levels FAIL

## Problem

`POST /v1/systemone` score questions: when a `criteria` level is a JSON object or
array, the `legend[i]` echoed in the response was flattened into a multi-line
string by `render()` (`kev/api.py`, `to_record`):

- object level `{"what": ..., "examples": [...]}` -> `"what: ...\nexamples:\n - ..."`
- array level `["cosmetic", "no functional impact"]` -> `"- cosmetic\n- no functional impact"`

OpenRouter requires `legend[i]` to be byte-identical in type and structure to the
original `criteria[i]` (or no legend at all). String levels pass; object/array/
mixed levels FAIL against the live service (HTTP 200 but the type is lost).

## Fix

`to_record` now deep-copies the original criteria levels into the legend
(`copy.deepcopy(x)`); the rendered text keeps feeding the model prompt
(`opts`), only the response echo keeps JSON types. Tracked here because the
repo has issues disabled; the PR description references this file.

## Also affected

- Upstream `jaredpalmer/kev` main has the same pattern (`m["legend"] =
  dict(zip(m["keys"], opts))` in `to_record`) — worth an upstream issue when we
  next sync; it predates the fork and is not an SF-introduced regression.
- `Open-Jev` is NOT affected: its legend is `json.loads(json.dumps(criteria))`,
  types preserved.
- `jev-proxy` is not affected (no score/legend in its proxy layer).
