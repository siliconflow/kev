"""Stage a release checkpoint on the runs volume: copy a candidate's checkpoint directory to a new release directory,
never over an existing one, and check that the copy's weights hash (kev.checkpoint.Checkpoint.weights_sha256: the sha256
over every shard's `name:sha256` line, or the adapter file's sha256) equals the source's. The source is only read.
modal_app.py::release_copy runs this on a CPU container, so a 27B's ~51 GB never leaves Modal:

    KEV_APP_NAME=kev-release uv run modal run modal_app.py::release_copy \\
        --src /runs/r23-wise/27b-k-w85/checkpoint --dst /runs/release/kev-27b-r23/checkpoint \\
        --expect d27af6ab2be16824166ac639907b4dba40979ff338599c2872721aa6c5072022

The release temperature is written into the copy's head.pt afterwards with scripts/calibrate_checkpoint.py, on the rows
the round registered for its pool, and the copy is published with kev.publish (modal_app.py::release_publish).
"""
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kev.checkpoint import Checkpoint  # noqa: E402
from kev.suite import digest  # noqa: E402

SIDECARS = ("interpolation.json", "provenance.json", "result.json", "training_config.json", "training_metrics.json")   # a trial's or interpolation's record beside checkpoint/


def weights_sha256(path, threads=8):
    """Checkpoint.weights_sha256 with the shards hashed in parallel (hashlib releases the GIL); the same value."""
    checkpoint = Checkpoint(str(path))
    if not checkpoint.full: return checkpoint.weights_sha256()
    shards = checkpoint.shards()
    with ThreadPoolExecutor(threads) as pool: digests = list(pool.map(digest, shards))
    import hashlib
    return hashlib.sha256("".join(f"{p.name}:{d}\n" for p, d in zip(shards, digests)).encode()).hexdigest()


def copy_checkpoint(src, dst, expect=None, log=print):
    """Copy the checkpoint directory `src` to `dst` (refused if `dst` exists), plus the record files beside it
    (SIDECARS, into dst's parent), and verify the weights hash. -> {src, dst, weights_sha256, files, bytes, head_sha256}.
    Raises if the hashes differ, or if `expect` is given and the source's hash is not it."""
    src, dst = Path(src), Path(dst)
    if dst.exists() or dst.parent.exists() and any(dst.parent.iterdir()):
        raise FileExistsError(f"{dst.parent} exists and is not empty; a release directory is written once")
    source = weights_sha256(src)
    if expect and source != expect: raise ValueError(f"{src}: weights sha256 {source} is not the expected {expect}")
    log(f"source {src}: weights sha256 {source}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=False, ignore=shutil.ignore_patterns("resume", ".*"))
    for name in SIDECARS:
        if (src.parent / name).exists(): shutil.copy2(src.parent / name, dst.parent / name)
    copied = weights_sha256(dst)
    if copied != source: raise ValueError(f"{dst}: weights sha256 {copied} differs from the source's {source}")
    files = sorted(p.name for p in dst.iterdir())
    log(f"copy {dst}: weights sha256 {copied} (equal), {len(files)} files")
    return {"src": str(src), "dst": str(dst), "weights_sha256": copied, "files": files,
            "bytes": sum(p.stat().st_size for p in dst.iterdir()), "head_sha256": {"src": digest(src / "head.pt"), "dst": digest(dst / "head.pt")},
            "sidecars": [n for n in SIDECARS if (dst.parent / n).exists()]}
