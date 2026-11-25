"""Download the blueprint tables that ``parciver play nlhe`` needs (``parciver fetch-blueprint``).

The tables are too large for git (the strategy is 1.27 GB), so they are published as assets of a
GitHub release, gzip-compressed (about 0.7 GB to download). ``assets.json`` beside this file lists
each file with the SHA-256 and size of its UNCOMPRESSED contents; a download is decompressed on the
fly, hashed, and only moved into place if the hash matches. Nothing here executes downloaded
content: the files are numeric tables read by the Rust engine.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Callable

from .play_nlhe import home

MANIFEST = Path(__file__).with_name("assets.json")
CHUNK = 1 << 20


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verified(path: Path, entry: dict) -> bool:
    return path.is_file() and path.stat().st_size == entry["bytes"] and _sha256(path) == entry["sha256"]


def _download(url: str, dest: Path, entry: dict, say: Callable[[str], None]) -> None:
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    h, got = hashlib.sha256(), 0
    req = urllib.request.Request(url, headers={"User-Agent": "parciver-fetch"})
    with urllib.request.urlopen(req) as resp, part.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        src = gzip.GzipFile(fileobj=resp) if entry.get("gzip", True) else resp
        last = -1
        while chunk := src.read(CHUNK):
            out.write(chunk)
            h.update(chunk)
            got += len(chunk)
            if total and sys.stdout.isatty():
                done = got * 100 // entry["bytes"]
                if done != last:
                    last = done
                    print(f"\r  {dest.name}: {done}%", end="", flush=True)
    if sys.stdout.isatty():
        print()
    if got != entry["bytes"] or h.hexdigest() != entry["sha256"]:
        part.unlink(missing_ok=True)
        raise SystemExit(f"{dest.name}: the download does not match the published checksum "
                         f"(got {got} bytes, sha256 {h.hexdigest()[:16]}...); nothing was installed")
    os.replace(part, dest)


def fetch(url: str | None = None, say: Callable[[str], None] = print) -> int:
    m = load_manifest()
    base = (url or os.environ.get("PARCIVER_ASSET_URL") or m.get("base_url") or "").rstrip("/")
    if not base:
        say("No download location is configured: this checkout has no release to fetch from yet.")
        say("Pass --url <directory URL holding the .gz files> or set PARCIVER_ASSET_URL.")
        return 2
    root = home()
    todo = []
    for e in m["files"]:
        dest = root / e["dest"]
        if verified(dest, e):
            say(f"  {e['dest']}: already installed and verified")
        else:
            todo.append((e, dest))
    if not todo:
        say(f"All {len(m['files'])} files are in place under {root}.")
        return 0
    need = sum(e["bytes"] for e, _ in todo)
    probe = root
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    say(f"Downloading {len(todo)} file(s), {need / 1e9:.2f} GB once unpacked, to {root}")
    if free < need * 1.1:
        say(f"Not enough free disk space ({free / 1e9:.1f} GB free).")
        return 2
    for e, dest in todo:
        _download(f"{base}/{e['name']}", dest, e, say)
        say(f"  {e['dest']}: verified (sha256 {e['sha256'][:12]}...)")
    say("Done. Play with:  parciver play nlhe")
    return 0
