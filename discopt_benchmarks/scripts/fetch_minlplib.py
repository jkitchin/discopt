#!/usr/bin/env python3
"""
Fetch MINLPLib (.nl archive + instancedata.csv) into a local cache directory.

Cache layout:

    <cache_dir>/
      <version>/
        instancedata.csv
        nl/
          ex1221.nl
          ...
        manifest.json   # version, urls, sha256 of downloaded files, fetch timestamp

The cache_dir defaults to ``~/.cache/discopt/minlplib``. It is NOT vendored in
the git repo: ~1700 .nl files and ~50 MB total.

A version is "current" by default (fetches latest from minlplib.org). To pin a
version, write a ``versions.json`` next to this script with a record of the
sha256 hashes of the desired snapshot and pass ``--version <tag>``.

Usage:
    # Fetch everything fresh (~50 MB download)
    python -m discopt_benchmarks.scripts.fetch_minlplib

    # Fetch a specific subset (skip the full archive)
    python -m discopt_benchmarks.scripts.fetch_minlplib --instances ex1221,nvs03

    # Force re-download even if cache exists
    python -m discopt_benchmarks.scripts.fetch_minlplib --force

    # Verify-only against a pinned manifest
    python -m discopt_benchmarks.scripts.fetch_minlplib --verify
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


DEFAULT_BASE_URL = "https://www.minlplib.org"
DEFAULT_VERSION = "current"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "discopt" / "minlplib"

_INSTANCEDATA_FILENAME = "instancedata.csv"
_NL_ARCHIVE_FILENAME = "nl.zip"
_MANIFEST_FILENAME = "manifest.json"


@dataclass
class Manifest:
    version: str
    base_url: str
    fetched_at: str
    instancedata_sha256: str = ""
    nl_archive_sha256: str = ""
    nl_file_count: int = 0
    pinned: bool = False
    per_instance_sha256: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def get_cache_dir(env_override: bool = True) -> Path:
    """Return the cache directory, honoring DISCOPT_MINLPLIB_CACHE env var."""
    if env_override:
        env = os.environ.get("DISCOPT_MINLPLIB_CACHE")
        if env:
            return Path(env).expanduser()
    return DEFAULT_CACHE_DIR


def get_version_dir(cache_dir: Path, version: str) -> Path:
    return cache_dir / version


def get_nl_dir(cache_dir: Path, version: str) -> Path:
    return get_version_dir(cache_dir, version) / "nl"


def get_instancedata_path(cache_dir: Path, version: str) -> Path:
    return get_version_dir(cache_dir, version) / _INSTANCEDATA_FILENAME


def get_manifest_path(cache_dir: Path, version: str) -> Path:
    return get_version_dir(cache_dir, version) / _MANIFEST_FILENAME


def find_nl_file(name: str, cache_dir: Path | None = None, version: str = DEFAULT_VERSION) -> Path | None:
    """Resolve an instance name to a cached .nl path, or None if missing."""
    cache_dir = cache_dir or get_cache_dir()
    candidate = get_nl_dir(cache_dir, version) / f"{name}.nl"
    return candidate if candidate.exists() else None


def load_manifest(cache_dir: Path, version: str) -> Manifest | None:
    path = get_manifest_path(cache_dir, version)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return Manifest(**data)


def save_manifest(manifest: Manifest, cache_dir: Path, version: str) -> None:
    path = get_manifest_path(cache_dir, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path | None = None) -> bytes:
    """Fetch a URL into memory; optionally tee to ``dest``."""
    req = urllib.request.Request(url, headers={"User-Agent": "discopt-benchmarks"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return data


def _load_pinned_versions() -> dict:
    """Load pinned-version sha256 records from versions.json next to this script."""
    versions_file = Path(__file__).parent / "versions.json"
    if not versions_file.exists():
        return {}
    with open(versions_file) as f:
        return json.load(f)


def fetch(
    cache_dir: Path | None = None,
    version: str = DEFAULT_VERSION,
    base_url: str = DEFAULT_BASE_URL,
    instances: list[str] | None = None,
    force: bool = False,
    skip_archive: bool = False,
) -> Manifest:
    """Fetch instancedata.csv plus the .nl archive (or selected instances).

    Returns the resulting Manifest. Idempotent: skips re-downloads when files
    already exist and ``force`` is False.
    """
    cache_dir = cache_dir or get_cache_dir()
    version_dir = get_version_dir(cache_dir, version)
    nl_dir = get_nl_dir(cache_dir, version)
    version_dir.mkdir(parents=True, exist_ok=True)
    nl_dir.mkdir(parents=True, exist_ok=True)

    pinned = _load_pinned_versions().get(version)

    existing = load_manifest(cache_dir, version)
    manifest = existing or Manifest(
        version=version,
        base_url=base_url,
        fetched_at=datetime.now().isoformat(),
        pinned=bool(pinned),
    )

    # ── instancedata.csv ──
    inst_path = get_instancedata_path(cache_dir, version)
    if force or not inst_path.exists():
        url = f"{base_url}/{_INSTANCEDATA_FILENAME}"
        print(f"[fetch] {url}")
        data = _download(url, inst_path)
        manifest.instancedata_sha256 = _sha256_bytes(data)
        if pinned and pinned.get("instancedata_sha256"):
            expected = pinned["instancedata_sha256"]
            if manifest.instancedata_sha256 != expected:
                raise RuntimeError(
                    f"instancedata.csv sha256 mismatch for version '{version}':\n"
                    f"  expected {expected}\n"
                    f"  got      {manifest.instancedata_sha256}"
                )
    else:
        manifest.instancedata_sha256 = _sha256_file(inst_path)

    # ── nl files ──
    if instances:
        # Per-instance fetch (cheap when caller only needs a handful)
        for name in instances:
            dest = nl_dir / f"{name}.nl"
            if dest.exists() and not force:
                continue
            url = f"{base_url}/nl/{name}.nl"
            print(f"[fetch] {url}")
            try:
                data = _download(url, dest)
                manifest.per_instance_sha256[name] = _sha256_bytes(data)
            except Exception as e:
                print(f"  ! failed: {e}", file=sys.stderr)
    elif not skip_archive:
        # Bulk archive fetch
        archive_url = f"{base_url}/{_NL_ARCHIVE_FILENAME}"
        archive_dest = version_dir / _NL_ARCHIVE_FILENAME
        if force or not archive_dest.exists():
            print(f"[fetch] {archive_url} (this may take a minute)")
            archive_bytes = _download(archive_url, archive_dest)
        else:
            archive_bytes = archive_dest.read_bytes()

        manifest.nl_archive_sha256 = _sha256_bytes(archive_bytes)
        if pinned and pinned.get("nl_archive_sha256"):
            expected = pinned["nl_archive_sha256"]
            if manifest.nl_archive_sha256 != expected:
                raise RuntimeError(
                    f"nl.zip sha256 mismatch for version '{version}':\n"
                    f"  expected {expected}\n"
                    f"  got      {manifest.nl_archive_sha256}"
                )

        print(f"[extract] {archive_dest} -> {nl_dir}")
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            count = 0
            for member in zf.namelist():
                if not member.endswith(".nl"):
                    continue
                # Flatten: strip any directory prefix
                target_name = Path(member).name
                target = nl_dir / target_name
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
        manifest.nl_file_count = count

    if not manifest.nl_file_count:
        manifest.nl_file_count = sum(1 for _ in nl_dir.glob("*.nl"))

    manifest.fetched_at = datetime.now().isoformat()
    save_manifest(manifest, cache_dir, version)
    print(
        f"[done] version={version} dir={version_dir} "
        f"nl_files={manifest.nl_file_count}"
    )
    return manifest


def verify(cache_dir: Path | None = None, version: str = DEFAULT_VERSION) -> bool:
    """Verify cached files match the pinned sha256, if pinned."""
    cache_dir = cache_dir or get_cache_dir()
    pinned = _load_pinned_versions().get(version)
    if not pinned:
        print(f"[verify] no pinned record for version '{version}' — nothing to check")
        return True

    inst_path = get_instancedata_path(cache_dir, version)
    if pinned.get("instancedata_sha256"):
        if not inst_path.exists():
            print(f"[verify] FAIL: {inst_path} missing")
            return False
        got = _sha256_file(inst_path)
        if got != pinned["instancedata_sha256"]:
            print(f"[verify] FAIL: instancedata.csv sha256 {got} != pinned {pinned['instancedata_sha256']}")
            return False

    archive_path = get_version_dir(cache_dir, version) / _NL_ARCHIVE_FILENAME
    if pinned.get("nl_archive_sha256") and archive_path.exists():
        got = _sha256_file(archive_path)
        if got != pinned["nl_archive_sha256"]:
            print(f"[verify] FAIL: nl.zip sha256 {got} != pinned {pinned['nl_archive_sha256']}")
            return False

    print(f"[verify] OK: version '{version}' matches pinned manifest")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION,
                        help="Version tag for the snapshot (default: current)")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help="Base URL for MINLPLib (default: %(default)s)")
    parser.add_argument("--instances", type=str, default=None,
                        help="Comma-separated instance names to fetch individually (skips full archive)")
    parser.add_argument("--skip-archive", action="store_true",
                        help="Skip the bulk nl.zip download (instancedata.csv only)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files already exist")
    parser.add_argument("--verify", action="store_true",
                        help="Verify cached files against pinned sha256 and exit")
    args = parser.parse_args()

    cache_dir = args.cache_dir or get_cache_dir()

    if args.verify:
        ok = verify(cache_dir=cache_dir, version=args.version)
        sys.exit(0 if ok else 1)

    instances = None
    if args.instances:
        instances = [s.strip() for s in args.instances.split(",") if s.strip()]

    fetch(
        cache_dir=cache_dir,
        version=args.version,
        base_url=args.base_url,
        instances=instances,
        force=args.force,
        skip_archive=args.skip_archive,
    )


if __name__ == "__main__":
    main()
