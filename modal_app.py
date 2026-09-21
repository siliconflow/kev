"""Run kev studies on Modal: one GPU container per trial, results pulled back into runs/.

    KEV_GPU=T4 uv run modal run modal_app.py::smoke                        # ~2 min end to end on a T4 (free tier)
    uv run modal run modal_app.py::study --suite evals/decision-v1 \\
        --plan experiments/mbp-comparison.json --name mbp-comparison-v1     # N trials in parallel on H100s
    uv run modal run modal_app.py::evaluate --run jaredpalmer/kev-0.5b \\
        --suite evals/transfer-v1 --name transfer-kev-v01-h100             # score a Hub checkpoint

The same `kev.experiment.execute_trial` runs here and on the MBP; only the device differs. Every trial records
the local git commit (KEV_GIT_COMMIT), the suite hash, and the hashes of the kev/*.py files that were shipped, and
`kev.experiment --aggregate` ranks the study locally afterwards so the ledger is produced by one code path.

Volumes: kev-hf-cache (base weights, downloaded once), kev-runs (trial outputs). Secrets: none required; set
KEV_HF_SECRET=<modal secret name> to attach a Secret carrying HF_TOKEN for gated bases.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "kev-research"
ROOT = Path(__file__).resolve().parent
RUNS_MOUNT, HF_MOUNT = "/runs", "/hf"
GPU = os.environ.get("KEV_GPU", "H100")   # H100 needs a payment method on the workspace; KEV_GPU=T4 for the free tier

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_sync(uv_project_dir=str(ROOT), groups=[])           # exact locked deps; Linux torch wheels are the CUDA build
    # Gated DeltaNet kernels for the Qwen3.5 hybrid backbones (transformers falls back to slow reference code without them)
    # fla refuses its gated chunk backward on Hopper with Triton 3.4-3.7.0 (incorrect results, fla#640); torch 2.8 pins 3.4
    .uv_pip_install("flash-linear-attention", "triton>=3.7.1")
    .env({"HF_HOME": HF_MOUNT, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("kev")
    .add_local_file(ROOT / "uv.lock", "/root/uv.lock")
    .add_local_file(ROOT / "pyproject.toml", "/root/pyproject.toml")
    .add_local_dir(ROOT / "evals", "/root/evals")
    .add_local_dir(ROOT / "scripts", "/root/scripts")
)
hf_cache = modal.Volume.from_name("kev-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("kev-runs", create_if_missing=True)
secrets = [modal.Secret.from_name(os.environ["KEV_HF_SECRET"])] if os.environ.get("KEV_HF_SECRET") else []


def local_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def local_source_hashes():
    sys.path.insert(0, str(ROOT))
    from kev.experiment import source_hashes
    return source_hashes()


@app.function(image=image, cpu=1, memory=1024, timeout=120)
def remote_source_hashes():
    """Hashes of kev/*.py inside the deployed image: the launcher compares them with the checkout before spawning."""
    from kev.experiment import source_hashes
    return source_hashes()


@app.function(image=image, gpu=GPU, cpu=4, memory=(65536, 196608), max_containers=8, retries=0, timeout=14400,   # a 35B-A3B bf16 checkpoint (70 GB) is staged through host memory while loading; the old 48 GB cap stalled the container
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_trial(study, index, label, config, suite, expected_sources, git_commit, existing=None, transfer=None):
    """One trial in one container. `existing` is a checkpoint path on the runs volume or a Hub id (legacy scoring)."""
    import torch
    from kev.experiment import execute_trial, source_hashes

    os.environ["KEV_GIT_COMMIT"] = git_commit
    if source_hashes() != expected_sources:
        raise RuntimeError("container received different kev/*.py than the launcher hashed")
    out = Path(RUNS_MOUNT) / study / f"{index:02d}-{label}"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite remote trial: {out}")
    print(f"[{label}] {torch.cuda.get_device_name(0)} torch {torch.__version__} config={json.dumps(config)}", flush=True)
    try:
        report, _ = execute_trial(config or {}, Path("/root") / suite, out, expected_sources, "cuda", existing, Path("/root") / transfer if transfer else None)
    finally:
        runs_volume.commit()
        hf_cache.commit()
    return {"label": label, "objective": report["objective"], "clean_acc": report["clean"]["acc"],
            "wall_seconds": report["wall_seconds"], "gates": report["gates"]["checks"]}


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 49152), retries=0, timeout=3600,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_locked_test(trial_path, name, suites, git_commit, redo_interrupted=False):
    """Read the locked test partitions ONCE for a promoted trial. Writes /runs/locked/<name>/... ; refuses to rerun."""
    import json
    from kev.benchmark import LocalPredictor, evaluate_records
    from kev.suite import digest, load_split, write_json
    os.environ["KEV_GIT_COMMIT"] = git_commit
    trial = Path(RUNS_MOUNT) / trial_path
    out = Path(RUNS_MOUNT) / "locked" / name
    runs_volume.reload()
    summary = None
    if out.exists():
        # an interrupted read may finish the suites it never touched; a suite that was read is never read again
        prior = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {"suites": {}}
        if all(label in prior["suites"] for label in suites):
            raise FileExistsError(f"locked test already read for {name}; a second read is not allowed")
        interrupted = []
        for label in list(suites):
            if label in prior["suites"]:
                suites.pop(label)
            elif (out / label).exists():
                # a read that crashed before any aggregate was produced: no number was ever observed, so completing it does
                # not enable selection on the test; it must be requested explicitly and is recorded
                if not redo_interrupted: raise RuntimeError(f"{label} partition was touched but not summarised; pass redo_interrupted to complete it")
                import shutil; shutil.rmtree(out / label); interrupted.append(label)
        summary = {**prior, "resumed_for": sorted(suites), "interrupted_reads_redone": interrupted}
    out.mkdir(parents=True, exist_ok=True)
    result = json.loads((trial / "result.json").read_text())
    if not result["gates"]["passed"] and not name.endswith("-ungated"):
        raise RuntimeError("trial did not pass its gates; name the read '<name>-ungated' to record an exploratory read")
    temperature = result.get("temperature", 1.0)
    predictor = LocalPredictor(str(trial / "checkpoint"), "cuda")
    summary = summary or {"trial": trial_path, "trial_result_sha256": digest(trial / "result.json"), "temperature": temperature, "git_commit": git_commit, "suites": {}}
    try:
        for label, suite in suites.items():
            records = load_split(Path("/root") / suite, "test", allow_test=True)
            report, _ = evaluate_records(records, predictor, out / label, temperature, heldout_sources=tuple(r["_meta"]["source"] for r in records))
            summary["suites"][label] = {"suite": suite, "suite_sha256": digest(Path("/root") / suite / "manifest.json"), "clean": report["clean"], "tasks": report["tasks"],
                                        "paired_flip": report["paired_flip"], "variants": report["variants"], "permutation": report["permutation"], "coverage": report["coverage"]}
            print(f"[{name}] {label} test: acc {report['clean']['acc']:.3f} brier {report['clean']['brier']:.3f}", flush=True)
    finally:
        write_json(out / "summary.json", summary)
        runs_volume.commit()
    return summary


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 65536), retries=0, timeout=3600,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_base_probe(base, suite, name, tasks="all"):
    """Untrained baseline: the base model's zero-shot letter-logit readout on a frozen suite's development partition
    (scripts/base_mmlu_probe.py). Writes benchmark-compatible rows/report under /runs/probes/<name>."""
    import subprocess as sp
    out = Path(RUNS_MOUNT) / "probes" / name
    if out.exists():
        raise FileExistsError(f"probe {name} exists")
    try:
        sp.run([sys.executable, "/root/scripts/base_mmlu_probe.py", "--base", base, "--suite", f"/root/{suite}", "--tasks", tasks, "--device", "cuda", "--out", str(out)],
               check=True, cwd="/root", env={**os.environ, "PYTHONPATH": "/root"})
    finally:
        runs_volume.commit(); hf_cache.commit()
    return json.loads((out / "report.json").read_text())["clean"]


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 65536), retries=0, timeout=3600,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_anchors(base, suite, name, revision=None):
    """Frozen-base zero-shot targets for a suite's training partition -> /runs/anchors/<name>.json (kev.anchors)."""
    from kev.anchors import build
    out = Path(RUNS_MOUNT) / "anchors" / f"{name}.json"
    if out.exists():
        raise FileExistsError(f"anchors {name} exist")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        meta = build(base, Path("/root") / suite, out, device="cuda", revision=revision)
    finally:
        runs_volume.commit(); hf_cache.commit()
    return meta


@app.local_entrypoint()
def anchors(base: str, suite: str, name: str, revision: str = "", gpu: str = GPU):
    call = run_anchors.with_options(gpu=gpu).spawn(base, suite, name, revision or None)
    print(f"spawned anchors {name}: call {call.object_id}; result lands at /runs/anchors/{name}.json on the volume")


@app.local_entrypoint()
def base_probe(base: str, name: str, suite: str = "evals/v4/transfer-v4", tasks: str = "all", gpu: str = GPU):
    """e.g. --base Qwen/Qwen3-30B-A3B-Base --name qwen3-30b-a3b-transfer-v4"""
    target = ROOT / "runs/probes" / name
    if target.exists():
        raise FileExistsError(f"{target} exists")
    call = run_base_probe.with_options(gpu=gpu).spawn(base, suite, name, tasks)
    print(f"spawned probe {name}: call {call.object_id}; pull with: modal volume get kev-runs /probes/{name} runs/probes/")


@app.function(image=image, cpu=1, memory=2048, retries=0, timeout=10 * 3600, volumes={RUNS_MOUNT: runs_volume})
def run_study(name, suite, jobs, gpu):
    """Server-side fan-out: runs every trial of a study and records the outcome on the volume. Spawned by `launch_detached`
    so the study survives the local client disconnecting; pull results later with `modal run modal_app.py::pull`."""
    import time
    study_dir = Path(RUNS_MOUNT) / name; study_dir.mkdir(parents=True, exist_ok=True)
    lock = study_dir / "study.lock"
    runs_volume.reload()
    if lock.exists():   # a second execution of the same spawn (retry/duplicate) must not launch or overwrite anything
        return {"study": name, "duplicate_execution": True, "started_by": lock.read_text()}
    lock.write_text(json.dumps({"started": time.time(), "trials": len(jobs)})); runs_volume.commit()
    fn = run_trial.with_options(gpu=gpu, retries=0, max_containers=8)
    started = time.time()
    results = list(fn.starmap(jobs, return_exceptions=True))
    summary = {"study": name, "suite": suite, "gpu": gpu, "wall_seconds": time.time() - started,
               "trials": [{"label": j[2], "ok": not isinstance(r, Exception), "result": None if isinstance(r, Exception) else r, "error": str(r) if isinstance(r, Exception) else None}
                          for j, r in zip(jobs, results)]}
    (Path(RUNS_MOUNT) / name).mkdir(parents=True, exist_ok=True)
    (Path(RUNS_MOUNT) / name / "launch.json").write_text(json.dumps(summary, indent=1))
    runs_volume.commit()
    return summary


def launch_detached(suite, plan_path, name, gpu, existing=(), transfer=None, budget=20.0, timeout=1800):
    """Validate locally, then spawn run_study and return immediately. Same admission checks as launch()."""
    from kev.experiment import load_plan
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name): raise ValueError("study name must be a simple unique identifier")
    if (ROOT / "runs" / name).exists(): raise FileExistsError("choose a new study name; existing results are immutable")
    if not 60 <= timeout <= 14400 or not 0 < budget <= 250: raise ValueError("timeout must be 60..14400 seconds and study budget <= $250")
    trials = load_plan(ROOT / suite, ROOT / plan_path) if plan_path else []
    rates = {"H100": 3.95, "H200": 4.54, "B200": 6.25, "T4": .59}
    upper = (rates[gpu] + 2 * .04730 + 48 * .008) * timeout / 3600 * (len(trials) + len(existing))
    if upper > budget: raise ValueError(f"timeout-based compute bound ${upper:.2f} exceeds budget ${budget:.2f}")
    commit, sources = local_git_commit(), local_source_hashes()
    if subprocess.run(["git", "status", "--porcelain", "kev", "evals"], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        print("warning: kev/ or evals/ has uncommitted changes; provenance records the last commit, not the working tree", flush=True)
    entries = [(None, p) for p in existing] + [(t, None) for t in trials]
    jobs = [(name, i, Path(ex).name if ex else f"trial-{i}", cfg or {}, suite, sources, commit, ex, transfer) for i, (cfg, ex) in enumerate(entries)]
    # Every trial is its own independent function call on the *deployed* app (modal deploy modal_app.py): no long-lived
    # parent whose loss would cancel children, nothing tied to this client. Results land on the volume; `pull` collects them.
    try:
        target = modal.Function.from_name(APP_NAME, "run_trial"); target.hydrate()
        deployed_sources = modal.Function.from_name(APP_NAME, "remote_source_hashes").remote()
        if deployed_sources != sources:
            changed = sorted(k for k in set(deployed_sources) | set(sources) if deployed_sources.get(k) != sources.get(k))
            raise SystemExit(f"deployed app has different kev/*.py than this checkout ({', '.join(changed)}); run `uv run modal deploy modal_app.py` first")
    except SystemExit:
        raise
    except Exception as error:
        raise SystemExit(f"deployed app not usable ({type(error).__name__}: {str(error)[:120]}); run `uv run modal deploy modal_app.py` first - spawns on the ephemeral app die with this client")
    fn = target.with_options(gpu=gpu, timeout=timeout, retries=0)
    calls = [fn.spawn(*job) for job in jobs]
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / f"{name}.spawn.json").write_text(json.dumps({"name": name, "calls": {j[2]: c.object_id for j, c in zip(jobs, calls)}, "bound_usd": round(upper, 2), "timeout": timeout}))
    print(f"spawned study {name}: {len(jobs)} independent trial(s) on {gpu}, bound ${upper:.2f}. Pull later: modal run modal_app.py::pull --name {name}", flush=True)


def pull_study(study):
    """Download a study directory from the runs volume into runs/<study> and rank it."""
    target = ROOT / "runs" / study
    if target.exists():
        raise FileExistsError(f"refusing to overwrite local study: {target}")
    target.parent.mkdir(exist_ok=True)
    # `modal volume get <vol> /<study> runs/` recreates runs/<study>/... locally, checkpoints included (gitignored)
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", "kev-runs", f"/{study}", str(target.parent)], check=True)
    subprocess.run([sys.executable, "-m", "kev.experiment", "--aggregate", "--out", str(target)], check=True, cwd=ROOT)
    return target


def launch(suite, plan_path, name, gpu, existing=(), transfer=None, budget=20.0, timeout=1800):
    from kev.experiment import load_plan

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("study name must be a simple unique identifier")
    if (ROOT / "runs" / name).exists():
        raise FileExistsError("choose a new study name; existing results are immutable")
    if not 60 <= timeout <= 14400 or not 0 < budget <= 250:   # overnight authorization: $500 total, tracked in PLAN.md
        raise ValueError("timeout must be 60..14400 seconds and study budget <= $250")
    trials = load_plan(ROOT / suite, ROOT / plan_path) if plan_path else []
    rates = {"H100": 3.95, "H200": 4.54, "B200": 6.25, "T4": .59}
    if gpu not in rates:
        raise ValueError("no verified cost bound for this GPU")
    upper = (rates[gpu] + 2 * .04730 + 48 * .008) * timeout / 3600 * (len(trials) + len(existing))
    if upper > budget:
        raise ValueError(f"timeout-based compute bound ${upper:.2f} exceeds budget ${budget:.2f}")
    print(f"Compute admission bound ${upper:.2f}; excludes image build, startup, and storage; no automatic retries.", flush=True)
    commit, sources = local_git_commit(), local_source_hashes()
    if subprocess.run(["git", "status", "--porcelain", "kev", "evals"], cwd=ROOT, capture_output=True, text=True).stdout.strip():
        print("warning: kev/ or evals/ has uncommitted changes; provenance records the last commit, not the working tree", flush=True)
    entries = [(None, p) for p in existing] + [(t, None) for t in trials]
    jobs = [(name, i, Path(ex).name if ex else f"trial-{i}", cfg or {}, suite, sources, commit, ex, transfer) for i, (cfg, ex) in enumerate(entries)]
    fn = run_trial.with_options(gpu=gpu, timeout=timeout, retries=0, max_containers=8)
    print(f"launching {len(jobs)} trial(s) on {gpu} for study {name}", flush=True)
    results = list(fn.starmap(jobs, return_exceptions=True))
    for job, result in zip(jobs, results):
        print(job[2], result if isinstance(result, Exception) else json.dumps(result), flush=True)
    failures = [r for r in results if isinstance(r, Exception)]
    if len(failures) == len(results):
        raise SystemExit(f"all {len(results)} trial(s) failed; nothing to pull")
    target = pull_study(name)
    print(f"study pulled to {target}; {len(failures)} failure(s)", flush=True)
    if failures:
        raise SystemExit(1)


@app.local_entrypoint()
def study(suite: str, plan: str, name: str, gpu: str = GPU, existing: str = "", transfer: str = "", budget: float = 20.0, timeout: int = 1800, detached: bool = True):
    """detached (default): spawn the fan-out server-side and return; `pull --name` afterwards. detached=False keeps the old
    attached behaviour (pulls automatically, but dies with the local client)."""
    if detached: launch_detached(suite, plan, name, gpu, [e for e in existing.split(",") if e], transfer or None, budget, timeout)
    else: launch(suite, plan, name, gpu, [e for e in existing.split(",") if e], transfer or None, budget, timeout)


@app.function(image=image, gpu=GPU, cpu=2, memory=(32768, 49152), retries=0, timeout=7200,
              volumes={RUNS_MOUNT: runs_volume, HF_MOUNT: hf_cache}, secrets=secrets)
def run_resume(study, trial, suite, transfer, expected_sources, git_commit):
    """Finish calibration/development/transfer scoring for an interrupted trial whose checkpoint is complete."""
    from kev.experiment import execute_trial
    os.environ["KEV_GIT_COMMIT"] = git_commit
    out = Path(RUNS_MOUNT) / study / trial
    runs_volume.reload()
    try:
        report, _ = execute_trial(None, Path("/root") / suite, out, expected_sources, "cuda", None, Path("/root") / transfer if transfer else None, resume=True)
    finally:
        runs_volume.commit()
    return {"trial": trial, "objective": report["objective"], "transfer_acc": (report.get("transfer") or {}).get("clean", {}).get("acc")}


@app.local_entrypoint()
def resume(study: str, suite: str, transfer: str = "evals/v4/transfer-v4", gpu: str = GPU):
    """Spawn evaluation for every trial in a study that has checkpoint/head.pt but no result.json."""
    entries = json.loads(subprocess.run([sys.executable, "-m", "modal", "volume", "ls", "kev-runs", f"/{study}", "--json"], capture_output=True, text=True, cwd=ROOT).stdout or "[]")
    trials = sorted(Path(e["filename"]).name for e in entries if e.get("type") == "dir")
    fn = modal.Function.from_name(APP_NAME, "run_resume").with_options(gpu=gpu)
    for t in trials:
        files = {Path(e["filename"]).name for e in json.loads(subprocess.run([sys.executable, "-m", "modal", "volume", "ls", "kev-runs", f"/{study}/{t}", "--json"], capture_output=True, text=True, cwd=ROOT).stdout or "[]")}
        ck = {Path(e["filename"]).name for e in json.loads(subprocess.run([sys.executable, "-m", "modal", "volume", "ls", "kev-runs", f"/{study}/{t}/checkpoint", "--json"], capture_output=True, text=True, cwd=ROOT).stdout or "[]")} if "checkpoint" in files else set()
        if "head.pt" in ck and "result.json" not in files:
            c = fn.spawn(study, t, suite, transfer, local_source_hashes(), local_git_commit()); print(f"resuming {study}/{t}: call {c.object_id}")
        else:
            print(f"skip {study}/{t}: {'has result' if 'result.json' in files else 'no finished checkpoint'}")


@app.local_entrypoint()
def pull(name: str):
    """Pull a finished (or partially finished) study from the volume and rank it."""
    entries = json.loads(subprocess.run([sys.executable, "-m", "modal", "volume", "ls", "kev-runs", f"/{name}", "--json"], capture_output=True, text=True, cwd=ROOT).stdout or "[]")
    if not any(e.get("filename", "").endswith("launch.json") for e in entries):
        print(f"{name}: launch.json not on the volume yet (study still running or never finished); pulling what exists", flush=True)
    target = pull_study(name)
    print(f"pulled {target}", flush=True)


@app.local_entrypoint()
def locked_test(trial: str, name: str, decision: str = "evals/v4/decision-v4", transfer: str = "evals/v4/transfer-v4", gpu: str = GPU, redo_interrupted: bool = False):
    """One locked-test read for a promoted trial (path under the runs volume, e.g. v4-4b-baseline/01-trial-1)."""
    target = ROOT / "runs/locked" / name
    if (target / "summary.json").exists() and all(k in json.loads((target / "summary.json").read_text())["suites"] for k in ("decision", "transfer")):
        raise FileExistsError(f"{target} is complete; the locked test is read once per candidate")
    fn = modal.Function.from_name(APP_NAME, "run_locked_test").with_options(gpu=gpu)
    summary = fn.remote(trial, name, {"decision": decision, "transfer": transfer}, local_git_commit(), redo_interrupted)
    import shutil
    if target.exists(): shutil.rmtree(target)   # local copy only; the volume is the record
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "modal", "volume", "get", "kev-runs", f"/locked/{name}", str(target.parent)], check=True)
    print(json.dumps({k: {"acc": v["clean"]["acc"], "brier": v["clean"]["brier"]} for k, v in summary["suites"].items()}, indent=1))


@app.local_entrypoint()
def smoke(gpu: str = GPU):
    launch("evals/smoke-v1", "experiments/smoke.json", "smoke", gpu)


@app.local_entrypoint()
def evaluate(run: str, suite: str, name: str, gpu: str = GPU, transfer: str = ""):
    """Score an existing checkpoint (Hub id, or a path under the runs volume) on a suite's development partition."""
    launch(suite, None, name, gpu, [run], transfer or None)
