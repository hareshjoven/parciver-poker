# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Single sustained full-game `fixed_response` solve — the profiling target.

Runs one long external-sampling best-response solve on the full abstract game so
a sampling profiler (samply) gets a dense flamegraph of the Rust hot loop
(`parciver_engine::mccfr` traverse: infoset-key build, FxHash/shard lookup,
Mutex lock-wait, regret matching/update).

Usage (see HANDOFF.md "compute-optimization / flamegraph"):
    # build with symbols+frame pointers first:
    #   CARGO_PROFILE_RELEASE_DEBUG=true RUSTFLAGS="-C force-frame-pointers=yes" \
    #     .venv/Scripts/python.exe -m maturin develop --release -m engine/Cargo.toml
    # then, e.g.:
    #   samply record -- .venv/Scripts/python.exe scripts/flame_solve.py 16 2000000
    #   samply record -- .venv/Scripts/python.exe scripts/flame_solve.py 1 400000
Args: [threads] [iterations] [family] [strength]  (defaults 16, 2_000_000, nit, 0.35)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
    family = sys.argv[3] if len(sys.argv) > 3 else "nit"
    strength = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35

    import parciver_engine as pe
    from parciver.rmapss import step34

    saved = pe.SavedBlueprint(str(step34.ABSTRACTION_DIR), str(step34.BLUEPRINT))
    anchor = saved.s32_policy()
    opponent = anchor.leak(family, strength)
    print(f"solving fixed_response: threads={threads} iters={iters:,} "
          f"opp={family}({strength})", flush=True)
    opponent.fixed_response(0, iters, "linear", 90_001, threads)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
