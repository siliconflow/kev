"""How many attempts does Modal give a spawned call that times out? The measurement behind kev.budget.FULL_FT_RETRIES.

Round 22's full-weight trial was spawned with modal.Retries(max_retries=2) and got two attempts, not three: each attempt
hit its 28,800 s timeout and then outlived the 30 s cancellation grace ("failed to respond to cancellation for too long:
30 seconds - killing task"), and the second timeout was final (FunctionTimeoutError, call status TIMEOUT). This probe
reproduces that in seconds on CPU: every variant times out after 10 s; an "unresponsive" attempt ignores the cancellation
(SIGUSR1, which Modal raises InputCancellation from) and is killed 30 s later, as a full-weight trial is (its trainer and
its volume commits keep running); a "graceful" one needs 15 s to clean up, inside the grace. Attempts are counted in a
modal.Dict (deleted at the end) and printed with the gaps between their starts.

    uv run modal run scripts/modal_retry_probe.py

Measured 2026-09-27 (modal 1.5.5, ephemeral app, spawn + get; three runs, retries on the decorator or through
with_options alike):
    Retries(2), responsive                          3 attempts, every run
    Retries(2), unresponsive (killed)               2 in three of four calls; 3 in one, its third starting 4 s after
                                                    its second, while that one ran
    Retries(4), responsive / unresponsive           5 / 4 (the fourth started 6-7 s after the third, while it ran)
    Retries(2), graceful 15 s cleanup               3
    Retries(2), single_use_containers, unresponsive 3, the third starting 4-5 s after the second, while it ran
    retries=0, unresponsive                         1, and nothing after the kill
A killed attempt is charged twice (its timeout, then the killed task), and the retry the kill triggers can start next to
an attempt already running: either an attempt is lost (round 22) or two attempts of one full-weight trial train the same
directory at once. Hence full-weight trials spawn with retries=0 and are continued by kev.rounds watch
(modal_app.continue_full_trial), one call per attempt, the next only after the last ended.
"""
import signal
import time

import modal

app = modal.App("kev-retry-probe")
image = modal.Image.debian_slim()
DICT = "kev-retry-probe"
TWO, FOUR = (modal.Retries(max_retries=n, initial_delay=1.0, backoff_coefficient=1.0) for n in (2, 4))
SMALL = {"image": image, "cpu": 0.25, "memory": 256, "timeout": 10}


def attempt(tag, mode):
    """One attempt: record its start, then sleep past the timeout, ignoring the cancellation ("unresponsive"), cleaning up
    for 15 s after it ("graceful") or letting it interrupt the sleep ("responsive")."""
    d = modal.Dict.from_name(DICT)
    d[tag] = d.get(tag, []) + [time.time()]
    if mode == "unresponsive": signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    try:
        time.sleep(60)
    finally:
        if mode == "graceful": signal.signal(signal.SIGUSR1, signal.SIG_IGN); time.sleep(15)


@app.function(**SMALL, retries=TWO)
def two(tag, mode): attempt(tag, mode)


@app.function(**SMALL, retries=0)
def none(tag, mode): attempt(tag, mode)


@app.function(**SMALL, retries=FOUR)
def four(tag, mode): attempt(tag, mode)


@app.function(**SMALL, retries=TWO, single_use_containers=True)
def single(tag, mode): attempt(tag, mode)


@app.local_entrypoint()
def main():
    state = modal.Dict.from_name(DICT, create_if_missing=True)
    state.clear()   # creates it now (from_name is lazy) and forgets an interrupted run
    variants = {"retries2-responsive": (two, "responsive"), "retries2-unresponsive": (two, "unresponsive"),
                "with_options-retries2-unresponsive": (none.with_options(retries=TWO), "unresponsive"),
                "retries4-responsive": (four, "responsive"), "retries4-unresponsive": (four, "unresponsive"),
                "retries2-graceful": (two, "graceful"), "single_use-retries2-unresponsive": (single, "unresponsive"),
                "retries0-unresponsive": (none, "unresponsive")}
    try:
        calls = {tag: fn.spawn(tag, mode) for tag, (fn, mode) in variants.items()}
        for tag, call in calls.items():
            try: out = call.get(timeout=900)
            except Exception as error: out = type(error).__name__   # noqa: BLE001 - FunctionTimeoutError expected
            starts = state.get(tag, [])
            print(f"{tag}: {out}; attempts {len(starts)}, starts {[round(b - a) for a, b in zip(starts, starts[1:])]} s apart", flush=True)
        time.sleep(45)   # past the last kill: an attempt the kill brings starts by now
        print({tag: len(state.get(tag, [])) for tag in variants}, flush=True)
    finally:
        modal.Dict.objects.delete(DICT)
