"""FLOPSEARCH D4 M3, as amended by the author (7 Oct 21:44): the M2 arms through
search_ab.py, three runs in one process, in order:

  x15        each arm against the x1.5 sizes bot, 5,000 deals   (m3-x15.json)
  realistic  each arm against the realistic sizes bot, 5,000    (m3-realistic.json)
  lbr        each arm's LBR, 2,500 deals                        (m3-lbr.json)

Each run is search_ab.py's own (its checks, saves after every chunk, resume on
rerun with the same configuration); this only fixes the arguments and the order.
One process, so one PID for the watcher (scripts/watch_m3.py). A run that stops
stops the rest. Outputs and the log go to runs/flop-eval/.

    .venv/Scripts/python.exe scripts/m3_run.py [--threads 4]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import search_ab as ab  # noqa: E402

OUT = ROOT / "runs" / "flop-eval"
NET = "runs/turn-net/final-w2048-d5-e200/net.pt"
# M2's arms (FLOPSEARCH "M2 protocol"); the cache changes no value (M2: max(4, 2 x threads) = 8).
ARM_A = {"from_flop": True, "from_turn": True, "flop_iters": 500, "solver": "flop", "net": NET,
         "solver_kw": {"gamma": 5, "cache": 8}}
ARM_B = {"from_turn": True, "solver": "gpu"}
RUNS = (("x15", ["--row", "baseline_sizes_x1.5", "--deals", "5000", "--chunk", "100"]),
        ("realistic", ["--row", "baseline_realistic_sizes", "--deals", "5000", "--chunk", "100"]),
        ("lbr", ["--lbr", "--deals", "2500", "--chunk", "50"]))


def stop(why: str) -> None:
    print(f"stopped: {why}", flush=True)
    raise SystemExit(why)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    os.chdir(ROOT)  # the arguments below are relative to the repository
    # search_ab.py hashes the sources on disk at each run's start, but this process imported them once: an edit
    # (or an engine rebuild) after this point would key run 2 or 3 to code that is not the code playing.
    arms = [ab.arm(json.dumps(x)) for x in (ARM_A, ARM_B)]
    pinned = (ab.sources(arms), ab.engine_digest())
    for name, extra in RUNS:
        print(f"== M3 {name}", flush=True)
        if (ab.sources(arms), ab.engine_digest()) != pinned:
            stop(f"the sources or the engine build changed on disk since M3 started; {name} not started")
        sys.argv = ["search_ab.py", "--benchmark", "runs/scorecard/scorecard-v2-200bb/benchmark.json",
                    "--blueprint", "runs/blueprint-20260927-141645/blueprint.bin",
                    "--a", json.dumps(ARM_A), "--b", json.dumps(ARM_B), "--threads", str(args.threads),
                    "--out", str(OUT / f"m3-{name}.json"), *extra]
        try:
            ab.main()
        except SystemExit as e:  # search_ab.py's refusals: into the log the watcher reads, then stop
            if e.code not in (None, 0):
                stop(f"{name}: {e.code}")
            raise
        except BaseException as e:
            traceback.print_exc(file=sys.stdout)
            stop(f"{name}: {type(e).__name__}: {e}")
        gc.collect()  # the run's solvers and their GPU tables go before the next run builds its own
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    print("== M3 done", flush=True)


if __name__ == "__main__":
    main()
