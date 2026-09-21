import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from kev.data import ALL_REPOS, ALL_SOURCES, EVAL_ONLY, REPOS, SOURCES, TRAINABLE, TRANSFER_REPOS, TRANSFER_SOURCES, build, dataset_ref, materialize, source_seed
from kev.model import encode, load_tokenizer

SPLITS = ("train", "calibration", "development", "test")
BASES = ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B-Base")
# Frozen suites are mirrored on the Hub. Manifests (with the sha256 of every partition) and the development/test
# partitions live in git; large training partitions are fetched from this dataset on first use and verified against
# the manifest, so the suite hash and every provenance record stay unchanged.
SUITES_DATASET = "jaredpalmer/kev-suites"
SUITES_REVISION = "a957287d1c502a4e2e3b9d9d1325c2c6f27f181c"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record_digest(record):
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_split(directory, split, allow_test=False):
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if split == "test" and not allow_test:
        raise ValueError("locked test requires explicit --allow-test; never use it for search")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / f"{split}.jsonl"
    if not path.exists():
        fetch_partition(directory, path.name)
    if digest(path) != manifest["files"][path.name]["sha256"]:
        raise ValueError(f"suite checksum mismatch: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != manifest["files"][path.name]["records"]:
        raise ValueError("suite record count mismatch")
    return records


def fetch_partition(directory, filename):
    """Download one partition of a frozen suite from the Hub mirror into place. The caller verifies the sha256."""
    import shutil
    from huggingface_hub import hf_hub_download
    directory = Path(directory).resolve()
    evals_root = next((p for p in directory.parents if p.name == "evals"), None)
    if evals_root is None:
        raise FileNotFoundError(f"{directory / filename} is missing and is not under an evals/ tree")
    relative = directory.relative_to(evals_root) / filename
    cached = hf_hub_download(SUITES_DATASET, str(relative), repo_type="dataset", revision=SUITES_REVISION)
    shutil.copyfile(cached, directory / filename)
    print(f"fetched {relative} from {SUITES_DATASET}@{SUITES_REVISION[:10]}", flush=True)


def case_copy(record, variant):
    result = copy.deepcopy(record)
    result["_meta"]["group_id"] = record["_meta"].get("group_id", record["_meta"]["id"])   # bootstrap unit (pair for contrastive)
    result["_meta"]["parent_id"] = record["_meta"]["id"]                                   # the clean record this variant perturbs
    result["_meta"]["id"] += "/" + variant
    result["_meta"]["variant"] = variant
    return result


def contrast_cases(record, seed=0):
    candidates = [(qid, q) for qid, q in record["questions"].items() if q["type"] == "choice" and len(q["criteria"]) >= 3]
    if not candidates:
        return []
    qid, q = candidates[0]
    nk = "none_of_these"
    if nk in q["criteria"]:
        raise ValueError("reserved contrast option collision")
    out = []
    for variant in ("none_present", "none_absent"):
        r = case_copy(record, variant)
        r["questions"] = {qid: copy.deepcopy(q)}
        rq = r["questions"][qid]
        rq["criteria"][nk] = "None of these options describes the answer"
        if variant == "none_absent":
            rq["criteria"].pop(rq["label"])
            rq["label"] = nk
        keys = list(rq["criteria"])
        random.Random(source_seed(seed, record["_meta"]["id"])).shuffle(keys)
        rq["criteria"] = {k: rq["criteria"][k] for k in keys}
        r["_meta"]["none_key"] = nk
        out.append(r)
    r = case_copy(record, "permuted")
    rng = random.Random(source_seed(seed, record["_meta"]["id"]))
    for q in r["questions"].values():
        if q["type"] == "choice":
            keys = list(q["criteria"])
            rng.shuffle(keys)
            q["criteria"] = {k: q["criteria"][k] for k in keys}
    out.append(r)
    return out


def select_unique(records, count, seen, tokenizers, report):
    selected = []
    for record in sorted(records, key=lambda r: r["_meta"]["row_sha256"]):
        report["considered"] += 1
        key = record["_meta"]["text_sha256"]
        if key in seen:
            report["duplicate_state"] += 1
            continue
        try:
            rec = materialize(record)
            for tokenizer in tokenizers:
                enc = encode(tokenizer, rec, max_branch=960, strict=True)
                if len(enc["ids"]) > 2048:
                    raise ValueError("packed request exceeds 2048 tokens")
        except ValueError:
            report["context_rejected"] += 1
            continue
        record["_meta"].update(group_id=record["_meta"]["id"], variant="clean")
        seen.add(key)
        selected.append(record)
        report["accepted"] += 1
        if len(selected) == count:
            return selected
    raise ValueError(f"only {len(selected)}/{count} records fit the common context policy")


def training_state_hashes(suite_dir):
    """Normalized state hashes of a frozen suite's training + calibration partitions, for exact-match contamination checks."""
    hashes = set()
    for split in ("train", "calibration"):
        for record in load_split(suite_dir, split):
            hashes.add(record["_meta"]["text_sha256"])
    return hashes


def freeze(directory, train=300, calibration=40, development=80, test=80, seed=20260918, holdout=("mnli", "sst5"),
           sources=None, repos=None, exclude_states_from=None, contrastive_pairs=0, contrastive_holdout_families=(), contrastive_eval_pairs=40):
    from huggingface_hub import HfApi

    sources = SOURCES if sources is None else sources
    repos = REPOS if repos is None else repos
    eval_only = set(holdout) >= set(sources)
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite frozen suite {directory}")
    hub = HfApi()
    revisions = {repo: hub.dataset_info(*dataset_ref(repo)[:1], revision=dataset_ref(repo)[1]).sha for repo in set(repos.values())}
    base_revisions = {base: hub.model_info(base).sha for base in BASES}
    tokenizers = [load_tokenizer(base, revision=revision) for base, revision in base_revisions.items()]
    trainable_here = [x for x in sources if x not in holdout]
    if set(trainable_here) & set(EVAL_ONLY):
        raise ValueError(f"eval-only sources cannot be trainable: {sorted(set(trainable_here) & set(EVAL_ONLY))}")
    manifest = {
        "version": 2, "seed": seed, "holdout_sources": list(holdout), "eval_only": eval_only,
        "trainable_sources": trainable_here, "eval_only_sources": [x for x in sources if x in holdout],
        "policy": {"trainable": list(TRAINABLE), "eval_only": list(EVAL_ONLY)},
        "dataset_revisions": revisions, "base_revisions": base_revisions,
        "context": {"max_state": 384, "max_branch": 1024, "max_packed": 2048, "truncate": False},
        "selection": "Normalized exact-state deduplication across partitions; common tokenizer context admission; no fuzzy decontamination or pretraining-contamination claim.",
        "legacy_checkpoints": "Training/calibration overlap for pre-manifest checkpoints is unknown; exploratory only.",
        "objective": "Negative macro-average clean development NLL, equal weight per task; raw probabilities.",
        "excluded_training_states_from": str(exclude_states_from) if exclude_states_from else None,
        "files": {}, "admission": {},
    }
    partitions = {split: [] for split in SPLITS}
    seen = set(training_state_hashes(exclude_states_from)) if exclude_states_from else set()
    manifest["excluded_training_state_hashes"] = len(seen)
    for source in sources:
        report = Counter()
        test_pool = build(max(3 * (development + test), 600), "test", seed, only=[source], revisions=revisions, sources=sources, repos=repos)
        chosen = select_unique(test_pool, development + test, seen, tokenizers, report)
        partitions["development"].extend(chosen[:development])
        partitions["test"].extend(chosen[development:])
        if source not in holdout:
            train_pool = build(max(3 * (train + calibration), 800), "train", seed, only=[source], revisions=revisions, sources=sources, repos=repos)
            chosen = select_unique(train_pool, train + calibration, seen, tokenizers, report)
            partitions["calibration"].extend(chosen[:calibration])
            partitions["train"].extend(chosen[calibration:])
        manifest["admission"][source] = dict(report)
        print(f"froze {source}: {dict(report)}", flush=True)
    if contrastive_pairs:
        from kev.contrastive import FAMILIES, generate
        families = list(FAMILIES)
        held = list(contrastive_holdout_families)
        unknown = set(held) - set(families)
        if unknown: raise ValueError(f"unknown contrastive families: {sorted(unknown)}")
        trainable_fams = [f for f in families if f not in held]
        # training/calibration pairs and development/test pairs come from disjoint seeds within trainable families;
        # held-out families appear only in development/test (never trained anywhere)
        train_recs, rep_train = generate(contrastive_pairs, seed=f"{seed}-train", families=trainable_fams) if trainable_fams and not eval_only else ([], {})
        # eval-only suites carry the held-out families only; trainable suites carry their trainable families only
        eval_families = (held or families) if eval_only else trainable_fams
        eval_recs, rep_eval = generate(contrastive_eval_pairs, seed=f"{seed}-eval", families=eval_families)
        for r in train_recs + eval_recs:
            r["_meta"].update(group_id=r["_meta"]["family_id"], variant="clean")   # siblings are one bootstrap unit
            if r["_meta"]["text_sha256"] in seen: raise ValueError("contrastive state collides with an existing state")
        seen.update(r["_meta"]["text_sha256"] for r in train_recs + eval_recs)
        n_cal = 2 * max(len(train_recs) // 20, 1) if train_recs else 0
        partitions["calibration"].extend(train_recs[:n_cal]); partitions["train"].extend(train_recs[n_cal:])
        # stratified by family: alternate pairs (siblings adjacent) between development and test
        for i in range(0, len(eval_recs), 2):
            partitions["development" if (i // 2) % 2 == 0 else "test"].extend(eval_recs[i : i + 2])
        manifest["contrastive"] = {"pairs_per_family_train": contrastive_pairs, "pairs_per_family_eval": contrastive_eval_pairs,
                                   "eval_families": eval_families, "trainable_families": trainable_fams, "eval_only_families": held,
                                   "train_report": rep_train, "eval_report": rep_eval,
                                   "note": "labels from rule evaluators; ablation + invariance checks passed for every kept pair; no LLM"}
        if trainable_fams and not eval_only: manifest["trainable_sources"].append("contrastive")
        else: manifest["eval_only_sources"].append("contrastive")
        print(f"contrastive: {len(train_recs)} train/cal records, {len(eval_recs)} dev/test records; held-out families {held}", flush=True)
    for split in ("development", "test"):
        extras = []
        per_source = Counter()
        for record in partitions[split]:
            source = record["_meta"]["source"]
            if per_source[source] < 12:
                variants = contrast_cases(record, seed)
                for variant in variants:
                    for tok in tokenizers:
                        encode(tok, materialize(variant), strict=True)
                extras.extend(variants)
                per_source[source] += bool(variants)
        partitions[split].extend(extras)
    directory.mkdir(parents=True)
    for split, records in partitions.items():
        path = directory / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        manifest["files"][path.name] = {"sha256": digest(path), "records": len(records),
                                        "questions": sum(len(r["questions"]) for r in records)}
    manifest["code_hashes"] = {name: digest(Path(__file__).parent / name) for name in ("data.py", "api.py", "model.py", "suite.py")}
    write_json(directory / "manifest.json", manifest)
    print(json.dumps(manifest["files"], indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--calibration", type=int, default=40)
    ap.add_argument("--development", type=int, default=80)
    ap.add_argument("--test", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--transfer", action="store_true", help="eval-only suite from the eight transfer-v1 sources (no train/calibration partitions)")
    ap.add_argument("--sources", help="comma-separated source names (any of ALL_SOURCES); trainable ones get train/calibration partitions")
    ap.add_argument("--holdout", help="comma-separated sources kept eval-only within --sources (eval-only policy sources are always held out)")
    ap.add_argument("--exclude-states-from", help="frozen suite whose train+calibration states must not appear here")
    ap.add_argument("--contrastive-pairs", type=int, default=0, help="programmatic contrastive pairs per trainable family for training (0 = none)")
    ap.add_argument("--contrastive-holdout", default="", help="comma-separated contrastive families kept eval-only")
    a = ap.parse_args()
    cfam = tuple(x for x in a.contrastive_holdout.split(",") if x)
    if min(a.development, a.test) < 1:
        ap.error("split sizes must be positive")
    if a.transfer:
        freeze(a.out, 0, 0, a.development, a.test, a.seed, holdout=tuple(TRANSFER_SOURCES), sources=TRANSFER_SOURCES,
               repos=TRANSFER_REPOS, exclude_states_from=a.exclude_states_from, contrastive_pairs=a.contrastive_pairs, contrastive_holdout_families=cfam)
    elif a.sources:
        names = [x for x in a.sources.split(",") if x]
        unknown = set(names) - set(ALL_SOURCES)
        if unknown: ap.error(f"unknown sources: {sorted(unknown)}")
        holdout = tuple(x for x in names if x in EVAL_ONLY or x in (a.holdout or "").split(","))
        if any(x not in holdout for x in names) and min(a.train, a.calibration) < 1:
            ap.error("train and calibration sizes must be positive when a trainable source is included")
        freeze(a.out, a.train, a.calibration, a.development, a.test, a.seed, holdout=holdout,
               sources={k: ALL_SOURCES[k] for k in names}, repos={k: ALL_REPOS[k] for k in names}, exclude_states_from=a.exclude_states_from,
               contrastive_pairs=a.contrastive_pairs, contrastive_holdout_families=cfam)
    else:
        if min(a.train, a.calibration) < 1: ap.error("split sizes must be positive")
        freeze(a.out, a.train, a.calibration, a.development, a.test, a.seed, exclude_states_from=a.exclude_states_from)


if __name__ == "__main__":
    main()
