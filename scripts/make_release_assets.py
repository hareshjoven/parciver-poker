#!/usr/bin/env python
"""Build the GitHub-release assets for `parciver fetch-blueprint` and write src/parciver/assets.json.

    python scripts/make_release_assets.py --out C:/path/to/release-assets

Gzips the v2-8B blueprint (runs/bp-8b-stage2/blueprint.bin) and the three bucket tables the play
layer reads (data/abstraction/{flop,turn,river}_buckets.npy) into OUT, and records each file's
size and SHA-256 (of the uncompressed bytes) in the manifest. Upload OUT's .gz files to a release,
then set "base_url" in the manifest (or pass --url to fetch-blueprint). Never uploads anything.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("blueprint.bin", ROOT / "runs" / "bp-8b-stage2" / "blueprint.bin"),
    ("abstraction/flop_buckets.npy", ROOT / "data" / "abstraction" / "flop_buckets.npy"),
    ("abstraction/turn_buckets.npy", ROOT / "data" / "abstraction" / "turn_buckets.npy"),
    ("abstraction/river_buckets.npy", ROOT / "data" / "abstraction" / "river_buckets.npy"),
)
CHUNK = 1 << 22


def pack(src: Path, dst: Path) -> tuple[int, str]:
    h, n = hashlib.sha256(), 0
    with src.open("rb") as f, gzip.GzipFile(dst, "wb", compresslevel=6, mtime=0) as g:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            n += len(chunk)
            g.write(chunk)
    return n, h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--version", default="blueprint-v1")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files = []
    for dest, src in SOURCES:
        name = dest.replace("/", "_") + ".gz"
        n, sha = pack(src, a.out / name)
        files.append({"name": name, "dest": dest, "bytes": n, "sha256": sha, "gz_bytes": (a.out / name).stat().st_size})
        print(f"{dest}: {n} bytes -> {files[-1]['gz_bytes']} gz, sha256 {sha[:16]}...")
    manifest = {"version": a.version, "base_url": None,
                "blueprint": "v2-8B (200 bb), the blueprint behind the SLUMBOT-2 match in RESULTS.md",
                "files": files}
    out = ROOT / "src" / "parciver" / "assets.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
