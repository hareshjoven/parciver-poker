#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Build the GitHub-release assets for `parciver fetch-blueprint` / `fetch-nets` and write src/parciver/assets.json.

    python scripts/make_release_assets.py --out C:/path/to/release-assets                 # both groups
    python scripts/make_release_assets.py --out C:/path/to/release-assets --only nets     # just the flop nets

blueprint group: gzips the v2-8B blueprint (runs/bp-8b-stage2/blueprint.bin) and the three bucket tables the play
layer reads (data/abstraction/{flop,turn,river}_buckets.npy) into OUT.
nets group: copies the two flop value nets (N0 = the net the SLUMBOT-2 match was played with, N1 = the net the
live configuration uses now) into OUT as flopnet_N0.pt / flopnet_N1.pt, uncompressed (float32 weights barely
compress). Each net is first loaded with ``torch.load(weights_only=True)``: a file that does not pass is refused.
Each file's size and SHA-256 go in the manifest. Upload OUT's files to a release, then set "base_url" (blueprint)
or "nets"."base_url" in the manifest. Never uploads anything.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "src" / "parciver" / "assets.json"
SOURCES = (
    ("blueprint.bin", ROOT / "runs" / "bp-8b-stage2" / "blueprint.bin"),
    ("abstraction/flop_buckets.npy", ROOT / "data" / "abstraction" / "flop_buckets.npy"),
    ("abstraction/turn_buckets.npy", ROOT / "data" / "abstraction" / "turn_buckets.npy"),
    ("abstraction/river_buckets.npy", ROOT / "data" / "abstraction" / "river_buckets.npy"),
)
NETS = (  # (name, source, the sha256 the records pin: refused if the file differs)
    ("N0", ROOT / "runs" / "turn-net" / "final-w2048-d5-e200" / "net.pt",
     "a8bf00be2fd1a3c7b0d1b0796809256cd939305b68723e541ef2cd029238ef65"),
    ("N1", ROOT / "runs" / "turn-net" / "d2b-w2048-wd0.1" / "net.pt",
     "54bae7494d830143daa5a06d6dfd7a805e99b28f1c3b5dfd9ec2533de7f97d72"),
)
NETS_TAG = "flopnets-v1"
CHUNK = 1 << 22


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def pack(src: Path, dst: Path) -> tuple[int, str]:
    h, n = hashlib.sha256(), 0
    with src.open("rb") as f, gzip.GzipFile(dst, "wb", compresslevel=6, mtime=0) as g:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            n += len(chunk)
            g.write(chunk)
    return n, h.hexdigest()


def check_net(path: Path) -> dict:
    """The net's blob under the safe loader: only tensors and plain numbers may be in it."""
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=True)
    if set(blob) - {"width", "depth", "state", "meta", "canonical"}:
        raise SystemExit(f"{path}: unexpected keys {sorted(blob)}")
    return blob


def build_blueprint(out: Path, version: str, old: dict) -> dict:
    files = []
    for dest, src in SOURCES:
        name = dest.replace("/", "_") + ".gz"
        n, sha = pack(src, out / name)
        files.append({"name": name, "dest": dest, "bytes": n, "sha256": sha, "gz_bytes": (out / name).stat().st_size})
        print(f"{dest}: {n} bytes -> {files[-1]['gz_bytes']} gz, sha256 {sha[:16]}...")
    return {"version": version, "base_url": old.get("base_url"),
            "blueprint": "v2-8B (200 bb), the blueprint behind the SLUMBOT-2 match in RESULTS.md", "files": files}


def build_nets(out: Path, old: dict) -> dict:
    files = []
    for tag, src, want in NETS:
        got = sha256(src)
        if got != want:
            raise SystemExit(f"{src} is not {tag} (sha256 {got[:16]}..., the records pin {want[:16]}...)")
        blob = check_net(src)
        name = f"flopnet_{tag}.pt"
        shutil.copyfile(src, out / name)
        files.append({"name": name, "dest": f"nets/{tag}.pt", "bytes": src.stat().st_size, "sha256": got,
                      "gzip": False, "width": blob["width"], "depth": blob["depth"]})
        print(f"{tag}: {files[-1]['bytes']} bytes, sha256 {got[:16]}..., loads with weights_only=True")
    return {"version": NETS_TAG, "base_url": old.get("base_url"),
            "about": "flop value nets: N0 = the SLUMBOT-2 match's net, N1 = the live configuration's net (RESULTS.md)",
            "files": files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--version", default="blueprint-v1")
    ap.add_argument("--only", choices=("blueprint", "nets"), help="build one group; the other keeps its manifest entry")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    if a.only != "nets":
        nets = manifest.get("nets") or {}
        manifest = build_blueprint(a.out, a.version, manifest)
        if nets:
            manifest["nets"] = nets
    if a.only != "blueprint":
        manifest["nets"] = build_nets(a.out, manifest.get("nets") or {})
        if not manifest["nets"].get("base_url") and manifest.get("base_url"):
            manifest["nets"]["base_url"] = manifest["base_url"].rsplit("/", 1)[0] + "/" + NETS_TAG
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
