"""Rebuild the long request of deploy_check.json (run from the repo root at this commit):
    python runs/kev-deploy-2ea5660/make_long_state.py jaredpalmer/kev-27b 70000 long_request.json
A one-line ticket, then the repo's own model cards (public English prose) repeated in order and cut to <tokens> tokens.
The same construction with jaredpalmer/kev-4b and 70000 / 12000 tokens is the state runs/space-republish-2ea5660.json sent."""
import glob, json, sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
target = int(sys.argv[2])
docs = [open(p).read() for p in sorted(glob.glob("docs/model-cards/*.md"))]
filler = ""; i = 0
while len(tok(filler)["input_ids"]) < target:
    filler += docs[i % len(docs)] + "\n\n"; i += 1
ids = tok(filler)["input_ids"][:target]
state = "Customer message: I was charged twice for order 8812. Please refund the duplicate charge.\n\n" + tok.decode(ids)
questions = {"team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                      "criteria": {"returns": "Exchanges, refunds for returned items", "shipping": "Delivery, delays", "billing": "Charges, payments, duplicate charges"}},
             "duplicate": {"type": "noul", "instructions": "Does the customer report being charged twice?"}}
json.dump({"state": state, "model": "kev-latest", "questions": questions}, open(sys.argv[3], "w"))
print("state tokens (tokenizer, no markup):", len(tok(state)["input_ids"]), "chars", len(state))
