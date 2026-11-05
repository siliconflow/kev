"""Build the GitHub release assets for a Kev family release: one tarball per adapter checkpoint plus SHA256SUMS.txt.

Each tarball is `<name>/` holding the Hub snapshot of the checkpoint at an exact revision (adapter, head.pt, tokenizer
files, the trial's result.json / provenance.json / training_config.json / training_metrics.json), the release's model card
as README.md, and the locked-test read as locked_test.json (`runs/locked/<name>/summary.json`). This is the layout the
`kev-family` release's tarballs had (they were assembled by hand; this script makes it reproducible). Full-weight
checkpoints (Kev-27B, 51 GB) are not assets: GitHub caps an asset at 2 GB, so they stay on the Hub.

    uv run python scripts/build_release_assets.py --release docs/releases/kev-1.0-assets.json --out /tmp/kev-1.0-assets
    uv run python scripts/build_release_assets.py --release docs/releases/kev-1.0-assets.json --out /tmp/x --dry-run

The spec has the release's `release_date` (YYYY-MM-DD) and lists, per asset, `name` (tarball stem), `repo`, `revision`
(full or short commit), `card`, `locked`, and the `expect` sha256 of `adapter_model.safetensors` and `head.pt`; a
downloaded file that does not hash to `expect` is refused, and so is a card that still holds a `{{PLACEHOLDER}}`. The
tarballs are deterministic (sorted members, uid/gid 0, every member's mtime and the gzip header's set to one time), so
rebuilding from the same inputs gives the same SHA256SUMS.txt. That time is SOURCE_DATE_EPOCH when it is set (the
reproducible-builds.org convention), else midnight UTC of `release_date`: an extracted checkpoint's files then carry the
release date, which `kev.serve` reports in `/v1/models` (Checkpoint.release_date). Needs network for the Hub download;
`--dry-run` checks the spec and the local files only.
"""
import argparse
import datetime
import gzip
import io
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev.suite import digest, read_json, write_json  # noqa: E402

PLACEHOLDER = re.compile(r"\{\{[A-Z0-9_]+\}\}")
SKIP = {".gitattributes"}


def card_problems(text):
    """Placeholders left in a card ({{VALIDATED_CONTEXT_4B}} etc.), sorted and unique."""
    return sorted(set(PLACEHOLDER.findall(text)))


def source_date_epoch(spec, env=os.environ):
    """The mtime of every member and of the gzip header: SOURCE_DATE_EPOCH if set, else midnight UTC of the spec's
    release_date. (Kev 1.0's first tarballs used 0, which extracts as 1969-12-31 or 1970-01-01.)"""
    if env.get("SOURCE_DATE_EPOCH"): return int(env["SOURCE_DATE_EPOCH"])
    midnight = datetime.datetime.combine(datetime.date.fromisoformat(spec["release_date"]), datetime.time(), datetime.timezone.utc)
    return int(midnight.timestamp())


def deterministic_tar(src_dir, out_path, mtime):
    """Write src_dir as <src_dir.name>/... into a gzip tarball whose bytes depend only on file names, contents and
    `mtime` (an int, seconds since the epoch)."""
    src_dir, buf = Path(src_dir), io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        def add(path, arcname):
            info = tar.gettarinfo(str(path), arcname)
            info.mtime, info.uid, info.gid, info.uname, info.gname = mtime, 0, 0, "", ""
            info.mode = 0o755 if path.is_dir() else 0o644
            if path.is_dir(): tar.addfile(info)
            else:
                with path.open("rb") as f: tar.addfile(info, f)
        add(src_dir, src_dir.name)
        for path in sorted(p for p in src_dir.rglob("*")):
            add(path, f"{src_dir.name}/{path.relative_to(src_dir).as_posix()}")
    with Path(out_path).open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=mtime) as gz:
        gz.write(buf.getvalue())
    return digest(out_path)


def write_sums(sums, out_dir):
    """SHA256SUMS.txt in `shasum -a 256 -c` format, one line per asset, sorted by file name."""
    path = Path(out_dir) / "SHA256SUMS.txt"
    path.write_text("".join(f"{sha}  {name}\n" for name, sha in sorted(sums.items())), encoding="utf-8")
    return path


def check_asset(asset, root=ROOT):
    """Local problems with one asset spec: missing keys or files, placeholders left in its card."""
    problems = [f"{asset.get('name')}: missing {k}" for k in ("name", "repo", "revision", "card", "locked", "expect") if k not in asset]
    if problems: return problems
    card, locked = root / asset["card"], root / asset["locked"]
    if not card.is_file(): problems.append(f"{asset['name']}: card {asset['card']} not found")
    elif card_problems(card.read_text(encoding="utf-8")):
        problems.append(f"{asset['name']}: card {asset['card']} still has {', '.join(card_problems(card.read_text(encoding='utf-8')))}")
    if not locked.is_file(): problems.append(f"{asset['name']}: locked read {asset['locked']} not found")
    return problems


def stage(asset, work, root=ROOT, download=None):
    """Download the Hub snapshot at the pinned revision into work/<name>, check the expected hashes, add the card and the
    locked read. Returns the staged directory."""
    if download is None:
        from huggingface_hub import snapshot_download
        download = lambda repo, revision, local_dir: snapshot_download(repo, revision=revision, local_dir=local_dir)
    dst = Path(work) / asset["name"]
    if dst.exists(): shutil.rmtree(dst)
    tmp = Path(work) / f".{asset['name']}-download"
    download(asset["repo"], asset["revision"], str(tmp))
    dst.mkdir(parents=True)
    for path in sorted(tmp.iterdir()):
        if path.name in SKIP or path.name.startswith(".") or path.name == "README.md" or path.is_dir(): continue
        shutil.copy2(path, dst / path.name)
    shutil.rmtree(tmp, ignore_errors=True)
    for name, want in asset["expect"].items():
        got = digest(dst / name)
        if got != want: raise SystemExit(f"{asset['name']}: {name} sha256 {got} != expected {want}")
    shutil.copyfile(root / asset["card"], dst / "README.md")
    shutil.copyfile(root / asset["locked"], dst / "locked_test.json")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", required=True, help="asset spec JSON (docs/releases/kev-1.0-assets.json)")
    ap.add_argument("--out", required=True, help="output directory for the tarballs and SHA256SUMS.txt")
    ap.add_argument("--dry-run", action="store_true", help="check the spec and the local files; download nothing")
    a = ap.parse_args()
    spec = read_json(Path(a.release))
    problems = [p for asset in spec["assets"] for p in check_asset(asset)]
    if "release_date" not in spec: problems.insert(0, "spec: missing release_date")
    if problems:
        for p in problems: print(f"REFUSED {p}")
        raise SystemExit(1)
    mtime = source_date_epoch(spec)
    if a.dry_run:
        for asset in spec["assets"]: print(f"ok {asset['name']}: {asset['repo']}@{asset['revision']}")
        print(f"ok mtime {mtime} ({datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc).isoformat()})")
        return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    work = out / ".stage"; work.mkdir(exist_ok=True)
    sums, manifest = {}, {"release": spec["release"], "source_date_epoch": mtime, "assets": []}
    for asset in spec["assets"]:
        staged = stage(asset, work)
        tarball = out / f"{asset['name']}.tar.gz"
        sums[tarball.name] = deterministic_tar(staged, tarball, mtime)
        manifest["assets"].append({"file": tarball.name, "sha256": sums[tarball.name], "repo": asset["repo"],
                                   "revision": asset["revision"], "files": {p.name: digest(p) for p in sorted(staged.iterdir())}})
        print(f"{tarball.name} {sums[tarball.name]}")
    shutil.rmtree(work, ignore_errors=True)
    write_sums(sums, out)
    write_json(out / "manifest.json", manifest)
    print(f"wrote {out / 'SHA256SUMS.txt'} and {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
