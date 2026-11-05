"""Score-head order sensitivity, report-only (jaredpalmer/kev#161).

Kev's order tooling targets Choice (`/v1/systemone/permute`, `--perm_kl`, kev.evaluate's Permutation / IIA,
`kev.benchmark --rotations`); nothing measures order sensitivity on Score. This script ports the two checks
jaredpalmer/kev#161 ran by hand, in the style of laya's research/eval/presentation_checks.py
(NandhaKishorM/laya#259), against the same predictor interface kev.benchmark uses (a request record in,
probabilities out), so a checkpoint (LocalPredictor) and an endpoint (RemotePredictor) are measured the same way.
A checkpoint is scored at kev.suite.SERVING_CONTEXT, the window kev.serve admits and the one #161 measured in.
LocalPredictor's own default is the 384-token training context, which would refuse a state the endpoint accepts.

- first-slot rate: the same Score question is asked over every order of its K levels (all K! orders when that is
  at most --orders-max, else that many orders sampled without building K!) on --states states. The decision is
  the argmax. An exact tie for the maximum splits the win evenly across the tied slots and counts in tie_rate,
  so a flat head (the 1/K null) measures 1/K rather than 1 from awarding the tie to slot 0. No positional
  preference means 1/K (0.333 at K=3); read it next to tie_rate.
- identical-option control: all K levels share one text, so no answer is more right than another and any spread
  is presentation noise. Every permutation of those levels is the same request, so this is one predictor call
  per state, not K! calls. Slot 0's value minus the mean over slots, in logits when the predictor reports them
  (LocalPredictor does, at the temperature it loads — pass --raw-logits for temperature 1.0, what #161 calls
  "raw"; the issue's numbers are logit-space), else in probabilities.

One caveat the numbers need: Score levels are an ordered scale, so reordering them changes how the question reads as
well as where the answer sits — that is why --rotations leaves Score alone. A high first-slot rate therefore mixes
position bias with order-as-meaning, and only the identical-option control, where the texts are interchangeable,
isolates position. Read the two together, and not the rate as pure bias. The report repeats this next to the numbers.

Report-only on purpose: #98 declined to grow kev.benchmark's modes, so this never touches it. It writes one
JSON report (--out) and prints the same numbers. It is not a gate: it names what a checkpoint or endpoint does,
not what to ship.
"""
import argparse
import itertools
import json
import math
import random
import sys

from kev.api import MAX_OPTIONS, question_keys
from kev.suite import SERVING_CONTEXT, read_jsonl, write_json

QID = "score_q"
INSTRUCTIONS = "Answer the question using the scale defined by the levels."
ORDERED_SCALE_CAVEAT = (
    "Score levels are an ordered scale: reordering them changes how the question reads, not only where the answer "
    "sits, so the first-slot rate mixes position bias with order-as-meaning (why --rotations leaves Score alone). "
    "Only the identical-option control, whose texts are interchangeable, isolates position. Read the two together "
    "and not the rate as pure bias.")
DEFAULT_LEVEL_TEXTS = "low,medium,high"
DEFAULT_STATES = (
    "Ticket {i}: the customer cannot sign in and a deadline is tomorrow morning.",
    "Ticket {i}: a scheduled report arrived an hour late; nothing else is affected.",
    "Ticket {i}: the payment webhook dropped events for the last ten minutes in production.",
    "Ticket {i}: a user asks whether dark mode can be enabled on their account.",
    "Ticket {i}: the nightly backup finished with two files unreadable; the archive passed checksum.",
)


def level_orders(k, orders_max):
    """-> the orders the levels are asked in.

    Every permutation when K! <= orders_max (that list is then at most orders_max long). Otherwise a deterministic
    sample of orders_max distinct permutations (random.Random(0)), one permutation at a time, so the sample path
    does not materialise K!. Two runs of the same command measure the same orders.
    """
    if k < 1:
        raise ValueError(f"need at least one level, got {k}")
    if orders_max < 1:
        return []
    if math.factorial(k) <= orders_max:
        return list(itertools.permutations(range(k)))
    rng = random.Random(0)
    seen, orders = set(), []
    while len(orders) < orders_max:
        order = tuple(rng.sample(range(k), k))
        if order not in seen:
            seen.add(order)
            orders.append(order)
    return orders


def score_request(state, criteria, label=0):
    """A labelled one-question Score request in the serving format kev.benchmark feeds predictors. The label and
    src are required by the internal record (materialize) and ignored here: report-only, no accuracy is claimed."""
    return {"state": state, "questions": {QID: {"type": "score", "instructions": INSTRUCTIONS,
                                                "criteria": list(criteria), "label": label,
                                                "src": "presentation"}}}


def build_states(n, data=None):
    """-> the first n states to ask over. With data, the first n rows' "state" fields (fewer when the file is
    shorter); otherwise deterministic templates cycled to n. --states applies to both. Two runs over the same
    flags ask the same states."""
    if n < 1:
        raise ValueError(f"need at least one state, got {n}")
    if data is None:
        return [DEFAULT_STATES[i % len(DEFAULT_STATES)].format(i=i) for i in range(n)]
    states = []
    for row in data:
        state = row.get("state") if isinstance(row, dict) else None
        if not isinstance(state, str) or not state:
            raise ValueError("--data rows must each carry a non-empty string \"state\"")
        states.append(state)
    return states[:n]


def _read(answer, qid, criteria, space):
    """The question's values in option order, or None when this space is absent or not keyed by question_keys."""
    values = answer.get(space) if isinstance(answer, dict) else None
    if not isinstance(values, dict) or qid not in values or not isinstance(values[qid], dict):
        return None
    keyed = values[qid]
    keys = question_keys("score", criteria)
    if any(key not in keyed for key in keys):
        return None
    return [float(keyed[key]) for key in keys]


def _decide(predictor, state, criteria, spaces):
    """The first space the answer actually carries, in option order. NaN and infinities are refused: max() does not
    order NaN, and an infinity makes the identical-option difference NaN."""
    answer = predictor(score_request(state, criteria))
    for space in spaces:
        values = _read(answer, QID, criteria, space)
        if values is None:
            continue
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"predictor returned a non-finite {space} value for {QID}")
        return values, space
    raise ValueError(f"predictor returned neither {' nor '.join(spaces)} for {QID}")


def _split_win(values):
    """How one answer awards its win. A unique maximum takes 1. An exact tie for the maximum splits 1 evenly
    across the tied slots, so the lowest index does not win the tie (what max(..., key=) would do). A flat head,
    every slot equal, therefore awards 1/K to each slot.
    -> (shares, tied) with sum(shares) == 1; tied when more than one slot shares the maximum."""
    top = max(values)
    winners = [i for i, value in enumerate(values) if value == top]
    share = 1.0 / len(winners)
    shares = [0.0] * len(values)
    for i in winners:
        shares[i] = share
    return shares, len(winners) > 1


def first_slot_rate(predictor, criteria, states, orders):
    """The share of win-mass on slot 0 over states x orders, plus the per-slot shares (each should be 1/K under
    no positional preference), the expected rate, and tie_rate (the share of answers whose maximum was shared).

    The decision is read from probabilities when the predictor reports them, else from logits (a unique argmax
    is the same in both). An exact tie splits the win; it is not awarded to slot 0. A flat head, the 1/K null,
    therefore measures first_slot_rate == 1/K and tie_rate == 1, not first_slot_rate == 1.
    """
    if not states:
        raise ValueError("no states to ask")
    if not orders:
        raise ValueError("no orders to ask")
    k = len(criteria)
    wins = [0.0] * k
    n_ties = 0
    for state in states:
        for order in orders:
            criteria_order = [criteria[i] for i in order]
            values, _ = _decide(predictor, state, criteria_order, ("probabilities", "logits"))
            shares, tied = _split_win(values)
            wins = [win + share for win, share in zip(wins, shares)]
            if tied:
                n_ties += 1
    n = len(states) * len(orders)
    return {"first_slot_rate": wins[0] / n, "tie_rate": n_ties / n, "expected": 1 / k, "n_decisions": n,
            "n_ties": n_ties, "per_slot_rates": [win / n for win in wins], "n_orders": len(orders), "n_levels": k}


def identical_option_control(predictor, level_text, k, states):
    """Slot 0's value minus the mean over slots when every level carries the same text, averaged over states.

    One predictor call per state. Identical texts make every permutation the same request, so enumerating K!
    would rescore that request K! times and could not change the number. 0 is a flat head; the larger the
    magnitude, the more the first-listed slot is favoured on an unanswerable question. Logits when the predictor
    reports them, probabilities otherwise.
    """
    if not states:
        raise ValueError("no states to ask")
    if k < 1:
        raise ValueError(f"need at least one level, got {k}")
    criteria = [level_text] * k
    total, slots, space_used = 0.0, [0.0] * k, None
    for state in states:
        values, space_used = _decide(predictor, state, criteria, ("logits", "probabilities"))
        total += values[0] - sum(values) / k
        slots = [slot + value for slot, value in zip(slots, values)]
    n = len(states)
    return {"control": total / n, "space": space_used, "n_decisions": n,
            "per_slot_means": [slot / n for slot in slots], "n_levels": k}


def measure(predictor, level_texts, states, orders_max=120):
    """-> the report: both checks over the level set.

    The first-slot check honours orders_max (level_orders samples past K! without building it). The
    identical-option control does not take orders: one call per state, because identical texts make every
    permutation the same request.
    """
    if not level_texts:
        raise ValueError("need at least one level")
    if not states:
        raise ValueError("no states to ask")
    k = len(level_texts)
    orders = level_orders(k, orders_max)
    return {"levels": list(level_texts), "n_states": len(states),
            "first_slot": first_slot_rate(predictor, level_texts, states, orders),
            "identical_option": identical_option_control(predictor, level_texts[0], k, states),
            "caveat": ORDERED_SCALE_CAVEAT}


def resolve_level_texts(levels, texts):
    """The level texts the flags name.

    --level-texts, when given, is the question (and a --levels that disagrees with its length is an error, so the two
    flags cannot silently measure different K). With no texts, --levels is the count: the three urgency words at 3
    (the default, and #161's set), otherwise "level 1" .. "level N". Omitting both is those three words."""
    if texts is not None:
        chosen = [text.strip() for text in texts.split(",") if text.strip()]
        if not chosen:
            raise SystemExit("--level-texts must name at least one level")
        if levels is not None and levels != len(chosen):
            raise SystemExit(f"--levels is {levels} but --level-texts names {len(chosen)}")
        return chosen
    count = 3 if levels is None else levels
    if count == 3:
        return [text.strip() for text in DEFAULT_LEVEL_TEXTS.split(",")]
    return [f"level {i + 1}" for i in range(count)]


def build_predictor(args):
    """The predictor the flags name: LocalPredictor(--run) or RemotePredictor(--remote), exactly what
    kev.benchmark scores with. A checkpoint is loaded at SERVING_CONTEXT so a state the endpoint would score is not
    refused at the 384-token training window. Imported lazily so the checks themselves stay importable without a
    checkpoint."""
    if bool(args.run) == bool(args.remote):
        raise SystemExit("pass exactly one of --run <dir|hub-id> or --remote <url>")
    if args.raw_logits and not args.run:
        raise SystemExit("--raw-logits scores a checkpoint at temperature 1.0; a remote endpoint returns probabilities")
    from kev.device import default_device
    if args.run:
        from kev.checkpoint import LoadOptions
        from kev.predictors import LocalPredictor
        temperature = 1.0 if args.raw_logits else None
        return LocalPredictor(args.run, args.device or default_device(), opts=LoadOptions(temperature=temperature),
                              context=SERVING_CONTEXT)
    from kev.predictors import RemotePredictor
    return RemotePredictor(args.remote, model=args.model, api_key=args.api_key)


def main(argv=None):
    parse = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parse.add_argument("--run", help="a run directory or Hub id[@rev] to score in-process")
    parse.add_argument("--remote", help="a TypeSafe System One-compatible endpoint's base URL")
    parse.add_argument("--model", default="kev-latest", help="the model id a --remote request names")
    parse.add_argument("--api-key", default="local", help="the bearer token a --remote request sends")
    parse.add_argument("--device", help="what LocalPredictor loads onto (default: kev.device.default_device())")
    parse.add_argument("--raw-logits", action="store_true",
                       help="score --run at head temperature 1.0 (the issue's \"raw\" column) instead of the checkpoint's")
    parse.add_argument("--levels", type=int, default=None,
                       help="how many levels, when --level-texts is omitted (default 3: low, medium, high; "
                            "any other count is level 1 .. N). With --level-texts, the texts decide and a different count is an error")
    parse.add_argument("--level-texts", default=None,
                       help=f"comma-separated level texts (default, for 3 levels: {DEFAULT_LEVEL_TEXTS})")
    parse.add_argument("--states", type=int, default=20,
                       help="how many states to ask over (the first this many rows of --data)")
    parse.add_argument("--data", help="a JSONL of records whose \"state\" fields replace the built-in templates")
    parse.add_argument("--orders-max", type=int, default=120,
                       help="first-slot check: at most this many of the K! orders (all of them when K! fits; "
                            "otherwise a deterministic sample, K! is not built). The identical-option control is "
                            "one request per state and does not use this")
    parse.add_argument("--out", help="write the JSON report here (kev.suite.write_json)")
    args = parse.parse_args(argv)

    if args.levels is not None and args.levels < 1:
        raise SystemExit("--levels must be >= 1")
    if args.states < 1:
        raise SystemExit("--states must be >= 1")
    if args.orders_max < 1:
        raise SystemExit("--orders-max must be >= 1")
    level_texts = resolve_level_texts(args.levels, args.level_texts)
    if not 1 <= len(level_texts) <= MAX_OPTIONS:
        raise SystemExit(f"a Score question has 1..{MAX_OPTIONS} levels, got {len(level_texts)}")
    if len(set(level_texts)) != len(level_texts):
        raise SystemExit("--level-texts must be distinct")

    # Slice and check --data before loading a checkpoint: a bad file, including an empty one, fails before the weights do.
    states = build_states(args.states, read_jsonl(args.data) if args.data else None)
    if not states:
        raise SystemExit("--data has no rows")
    predictor = build_predictor(args)
    report = measure(predictor, level_texts, states, orders_max=args.orders_max)
    report["target"] = args.run or args.remote
    report["raw_logits"] = bool(args.raw_logits and args.run)
    if args.run:
        report["context"] = {key: SERVING_CONTEXT[key] for key in ("max_state", "max_branch", "max_packed")}
    print(json.dumps(report, indent=2))
    if args.out:
        write_json(args.out, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
