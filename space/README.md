---
title: Kev
emoji: 🎚️
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
short_description: Typed questions in, calibrated probabilities out
startup_duration_timeout: 1h
license: apache-2.0
tags:
  - decision-model
  - calibration
  - typesafe
  - qwen3.5
models:
  - jaredpalmer/kev-4b
  - jaredpalmer/kev-0.8b
  - Qwen/Qwen3.5-4B-Base
  - Qwen/Qwen3.5-0.8B-Base
datasets:
  - jaredpalmer/kev-suites
---

# Kev

Kev is a family of small decision models built on Qwen3.5. You give it one document (the *state*) and a set of typed
questions; it returns a probability for every option of every question, from one forward pass. No text is generated.

This Space runs [`jaredpalmer/kev-4b`](https://huggingface.co/jaredpalmer/kev-4b) and
[`jaredpalmer/kev-0.8b`](https://huggingface.co/jaredpalmer/kev-0.8b) on ZeroGPU. Pick a model, or **Both** to compare
them on the same request.

## What it does

| type | criteria | answer |
|---|---|---|
| `choice` | `{name: description}` | argmax name, probability per name, confidence |
| `noul` | optional `{"true": …, "false": …}` | `p(true)` |
| `score` | ordered list of level descriptions | expected level, legend, probability per level |

The request and response are TypeSafe's public `/v1/systemone` contract. Each question only sees the state and
itself; a secret written into one question is invisible to its siblings (try the *Isolation probe* example). Option
boundaries cannot be forged from user text (*Boundary forgery*).

The options next to the **Decide** button mirror the opt-in flags of `kev.serve`:

- *Calibrated probabilities*: one temperature (T = 2.0) fitted on the in-distribution development set for the Qwen3.5
  family. It leaves the argmax unchanged and brings out-of-domain ECE from 0.12 to 0.05 on Kev-4B.
- *date_facts*: Kev cannot subtract dates by itself. This appends the day count between every pair of absolute dates
  in the state before the model reads it (the *Return window* example shows the difference).
- *Option-order stability*: re-run the first Choice question under shuffled option orders and report whether the
  argmax flips.

## How it is built

- Backbone: `Qwen/Qwen3.5-4B-Base` / `Qwen/Qwen3.5-0.8B-Base` (revision pinned by each checkpoint), vocab head discarded.
- Adapter: LoRA r=16 on the attention, MLP and Gated DeltaNet projections, merged into the base weights in fp32. This is
  the path every number on the model cards was measured with.
- Readout: a pointer head scores each question's `<decide>` token against its option spans.
- Isolation: on these hybrid backbones each question runs as its own causal row continuing from the shared state.

`kev/model.py` and `kev/api.py` are copied verbatim from [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)
at publish time, so the Space runs the same encoder and API code as the repo's server.

## API and MCP

The `decide` endpoint is exposed over the Gradio API and as an MCP tool (`mcp_server=True`):

```python
from gradio_client import Client
c = Client("jaredpalmer/kev")
rendered, response, report = c.predict(
    "Shoes arrived two weeks late and in the wrong size.",
    '{"department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"returns": null, "shipping": null, "billing": null}}}',
    "Kev-4B", False, False, False, 4,
    api_name="/decide",
)
print(response["answers"])
```

## Caveats

Raw probabilities are usable but not perfectly calibrated out of domain (Kev-4B: raw ECE 0.12, 0.05 calibrated;
Kev-0.8B is a sub-1B model and noticeably weaker out of domain). Product-shaped questions with no training analogue
are not guaranteed. Measure on your own inputs. Model cards with every number:
[Kev-4B](https://huggingface.co/jaredpalmer/kev-4b), [Kev-0.8B](https://huggingface.co/jaredpalmer/kev-0.8b),
[Kev-9B](https://huggingface.co/jaredpalmer/kev-9b).

## Credits

Model and code by [Jared Palmer](https://github.com/jaredpalmer/kev), Apache-2.0. The first Space for Kev-4B was built
by [multimodalart](https://huggingface.co/multimodalart) at `hugging-apps/kev-4b-decision-demo`; this one follows its
layout. Base models by Qwen, Apache-2.0.
