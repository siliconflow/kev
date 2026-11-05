import json, glob
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("jaredpalmer/kev-27b")
# filler: the repo's own model cards (public English prose), repeated in order until ~16k tokens
docs = [open(p).read() for p in sorted(glob.glob("docs/model-cards/*.md"))]
ticket = ("Customer message (ticket #4411): Hi, I was charged twice on my card for order 8812 last Tuesday. "
          "The shoes themselves arrived on time and fit fine. Please refund the duplicate charge.\n\n"
          "Attached below: the full knowledge-base export the agent had open.\n\n")
needle = ("\n\nKB 7.3 Duplicate charges. Refunds for duplicate card charges are processed only by the Payments Operations "
          "office in Dublin; the Austin and Singapore offices cannot issue them.\n\n")
filler = ""
i = 0
while len(tok(filler)["input_ids"]) < 15600:
    filler += docs[i % len(docs)] + "\n\n"; i += 1
ids = tok(filler)["input_ids"][:15600]
filler = tok.decode(ids)
cut = int(len(filler) * 0.6); cut = filler.index("\n", cut)
state = ticket + filler[:cut] + needle + filler[cut:]
n = len(tok(state)["input_ids"])
req = {"state": state, "model": "kev-latest", "questions": {
  "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
           "criteria": {"returns": "Exchanges, refunds for returned items", "shipping": "Delivery, delays", "billing": "Charges, payments, duplicate charges"}},
  "office": {"type": "choice", "instructions": "According to the knowledge base, which office processes refunds for duplicate card charges?",
             "criteria": {"dublin": "Dublin", "austin": "Austin", "singapore": "Singapore"}},
  "duplicate": {"type": "noul", "instructions": "Does the customer report being charged twice?"}}}
json.dump(req, open("runs/kev-deploy-27b-v2/long_request.json", "w"))
print("state tokens (tokenizer, no markup):", n, "chars", len(state))
