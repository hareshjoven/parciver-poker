#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""GPU subgame solver speed: ms per DCFR iteration on the same trees and
inputs as the Rust solver's `print_solver_speed` (search.rs), plus the
exploitability reached, so the two can be compared number for number.

    .venv/Scripts/python.exe scripts/gpu_search_bench.py [--iters 100] [--tf32] [--solver fused|torch|both]

Rust side (idle machine):
    cd engine; cargo test --no-default-features --release --lib -- --ignored --nocapture print_solver_speed
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from parciver import gpu_search as gs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--tf32", action="store_true", help="allow TF32 tensor-core matmuls (float32 inputs)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--solver", choices=["fused", "torch", "both"], default="both",
                    help="fused: Triton kernels + CUDA graph (gpu_fused); torch: plain torch ops (gpu_search)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    b = gs.board(7)
    cases = [
        ("river", b, gs.spec()),
        ("turn (agent menus)", b[:4], gs.small_turn_spec()),
        ("turn (full menus)", b[:4], gs.turn_spec()),
    ]
    print(f"device {args.device}" + (f" ({torch.cuda.get_device_name()})" if args.device == "cuda" else "")
          + f", float32, TF32 {'on' if args.tf32 else 'off'}")
    kinds = ["fused", "torch"] if args.solver == "both" else [args.solver]
    for (name, cards, sp), kind in [(c, k) for c in cases for k in kinds]:
        hands = gs.Hands(cards)
        tree = gs.Tree(sp, hands)
        ranges = [gs.range_(hands.n, 31), gs.range_(hands.n, 32)]
        torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
        t0 = time.perf_counter()
        if kind == "fused":
            from parciver import gpu_fused
            s = gpu_fused.FusedSolver(hands, tree, ranges, device=args.device, dtype=torch.float32)
        else:
            s = gs.GpuSolver(hands, tree, ranges, device=args.device, dtype=torch.float32)
        setup = time.perf_counter() - t0
        for _ in range(3):
            s.step()
        sync()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            s.step()
        sync()
        ms = 1000 * (time.perf_counter() - t0) / args.iters
        pot = float(sum(sp.stake))
        e = s.exploitability()
        mem = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else float("nan")
        print(f"{name:20s} {kind:5s}: {tree.slots:6d} decision nodes, {len(tree.nodes):6d} nodes, {s.max_depth + 1:2d} levels | "
              f"{ms:7.2f} ms/iter | exploitability {100 * e / pot:.3f}% pot after {s.iter} iters | "
              f"setup {setup:.1f}s | peak GPU memory {mem:.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
