#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Build the Phase-3b card-abstraction bucket tables.

Runs the full pipeline for one or more streets: enumerate/index every canonical
(hole, board) situation -> extract its equity-distribution feature (the exact,
multi-hour pass; chunked + checkpointed, so it is resumable) -> cluster into `k`
buckets -> persist the bucket table + centroids + metadata -> gate.

Examples
--------
  # Quick end-to-end smoke on a small slice (proves the whole path, seconds).
  # --limit needs its own --out: the default directory holds the real tables.
  python scripts/build_buckets.py --street flop --k 8 --limit 2000 --bins 30 --out C:/tmp/buckets-smoke
  python scripts/build_buckets.py --street river --k 8 --limit 200000 --out C:/tmp/buckets-smoke

  # The real runs (measured: river ~12 min, turn ~1 h, flop ~4.2 h). The tables in
  # data/abstraction were built with k 10000; the river caps k at its 1950
  # distinct equities, and its meta (older than `k_requested`) records 1950:
  python scripts/build_buckets.py --street river --k 1950
  python scripts/build_buckets.py --street turn  --k 10000
  python scripts/build_buckets.py --street flop  --k 10000

Extraction is resumable: re-running the same command after an interruption
continues from the last checkpoint rather than restarting. A street's bucket
table, once built, is kept: the same command reports it, another k, bins or
k-means seed, restarts or iteration cap stops (every blueprint built on the
table reads it) unless --force.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a bare script (python scripts/build_buckets.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parciver import bucketing  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--street", default="all", choices=["flop", "turn", "river", "all"],
                    help="street to build (default: all)")
    ap.add_argument("--k", type=int, default=5000, help="number of buckets (default 5000)")
    ap.add_argument("--bins", type=int, default=30, help="equity histogram bins for flop/turn (default 30)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="situations per extraction chunk (default: per-street sensible value)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap situations per street (smoke tests only)")
    ap.add_argument("--out", type=str, default=None, help="output dir (default: repo data/abstraction)")
    ap.add_argument("--n-iter", type=int, default=100, help="max Lloyd iterations")
    ap.add_argument("--n-init", type=int, default=3, help="k-means++ restarts (flop/turn)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="rebuild a bucket table that exists (every blueprint built on it is then invalid)")
    args = ap.parse_args()
    if args.limit is not None and not args.out:
        ap.error("--limit is for smoke tests and needs its own --out "
                 "(the default directory holds the production tables)")

    streets = ["flop", "turn", "river"] if args.street == "all" else [args.street]
    # Extraction chunk defaults: river is cheap per situation (big chunk), flop
    # is heavy (small chunk keeps checkpoints frequent).
    default_chunk = {"flop": 2048, "turn": 16384, "river": 262144}
    out_dir = Path(args.out) if args.out else None

    all_ok = True
    for street in streets:
        chunk = args.chunk if args.chunk is not None else default_chunk[street]
        print(f"\n=== building {street} (k={args.k}, bins={args.bins}, chunk={chunk}, "
              f"limit={args.limit}) ===", flush=True)
        gate = bucketing.build_street(
            street, k=args.k, bins=args.bins, out_dir=out_dir, chunk=chunk,
            limit=args.limit, n_iter=args.n_iter, n_init=args.n_init, seed=args.seed, overwrite=args.force,
        )
        all_ok = all_ok and gate.ok

    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
