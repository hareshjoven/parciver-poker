"""VERIFY.md V4 (pre-registered 3 Oct; amendment of 4 Oct before any data): K2 x live turn search.
The 40 recorded real turn starts (`C:/tmp/d1/recs40.pkl`, 200bb, bb 100): the production `FusedSolver` (float32,
gamma 2, CUDA graph; all strategy sums kept, which leaves its iterates unchanged) against the generic `GpuSolver`
in float64 (gamma 2), both at 100 and 400 iterations. Every average strategy is scored by the float64 NumPy
reference walk (`records/d1-tail/ref.py`, `ref2.measure`: dense terminals), in % of the pot.
Reading (VERIFY.md V4, unchanged): K2 reaches live search if, at either budget, fused's mean exploitability is more
than 25% above the generic's, or any spot is more than 2% pot above the generic's. Columns `Q3` (the candidate
float64-sum terminals, KERNELS.md Q3) are descriptive only: no reading rests on them.
usage: v4_turn_search.py HEAD_SRC Q3_SRC OUT.jsonl [FIRST_N]  (FIRST_N: a smoke run on the first N spots)"""
import ctypes

k32 = ctypes.windll.kernel32
k32.GetCurrentProcess.restype = ctypes.c_void_p
k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
assert k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BelowNormal before anything spawns

import importlib.util  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "d1-tail"))
from ref2 import measure  # noqa: E402  (its walk takes the solver objects it is given)

RECS = r"C:\tmp\d1\recs40.pkl"
STACK, BB, BUDGETS = 20000, 100, (100, 400)
FLAT = ("kind", "player", "slot", "stake", "view", "child_start", "children")


def package(src: str, alias: str):
    init = Path(src) / "parciver" / "__init__.py"
    spec = importlib.util.spec_from_file_location(alias, init, submodule_search_locations=[str(init.parent)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return {m: importlib.import_module(f"{alias}.{m}") for m in ("gpu_search", "gpu_fused", "turn_values")}


def score(sums, P, hands, tree, sit, slots, ks) -> float:
    ref = P["gpu_search"].RefSolver(hands, tree, list(sit.ranges))
    for d in slots:
        ref.sum[d] = sums[d, :ks[d]].copy()
    return measure(ref, hands, sit.ranges, sit.pot)[1]


def main():
    head_src, q3_src, out_path = sys.argv[1:4]
    first = int(sys.argv[4]) if len(sys.argv) > 4 else None
    assert torch.cuda.is_available() and not torch.backends.cuda.matmul.allow_tf32
    torch.set_num_threads(8)
    H, Q = package(head_src, "parciver_head"), package(q3_src, "parciver_q3")
    recs = pickle.load(open(RECS, "rb"))[:first]
    out = open(out_path, "w")

    def emit(d):
        print(json.dumps(d), flush=True)
        out.write(json.dumps(d) + "\n")
        out.flush()

    emit(dict(spots=len(recs), head=head_src, q3=q3_src, gpu=torch.cuda.get_device_name(), torch=torch.__version__))
    t0, rows = time.time(), []
    for i, r in enumerate(recs):
        sit = H["turn_values"].root_of(dict(r["job"], board=r["board"], ranges=r["ranges"]), STACK, BB)
        assert sit.half == r["half"] and sit.first == r["first"]
        hands = H["gpu_search"].Hands(list(sit.board))
        assert np.array_equal(hands.strength.reshape(-1), np.asarray(r["strength"], dtype=float).reshape(-1))
        assert np.array_equal(hands.alive.reshape(-1), np.asarray(r["alive"], dtype=float).reshape(-1))
        tree = H["gpu_search"].Tree(sit.spec(), hands)
        flat = H["gpu_fused"].flatten(tree)
        for k in FLAT:  # the rebuilt tree is the recorded one
            assert np.array_equal(np.asarray(flat[k]).reshape(-1), np.asarray(r["job"][k]).reshape(-1)), (i, k)
        assert flat["norm"] == r["job"]["norm"]
        slots = sorted(x.slot for x in tree.nodes if x.kind == "D")
        ks = {x.slot: len(x.acts) for x in tree.nodes if x.kind == "D"}
        row = dict(spot=i, half=sit.half, spr=sit.stack / sit.pot)
        for name, P in (("fused", H), ("Q3", Q)):
            h = P["gpu_search"].Hands(list(sit.board))
            f = P["gpu_fused"].FusedSolver(h, flat, list(sit.ranges), dtype=torch.float32, gamma=2)
            assert f.use_graph
            for n in BUDGETS:
                while f.iter < n:
                    f.step()
                torch.cuda.synchronize()
                row[f"{name}_{n}"] = 100.0 * score(f.sum.double().cpu().numpy(), H, hands, tree, sit, slots, ks)
            del f
            torch.cuda.empty_cache()
        g = H["gpu_search"].GpuSolver(hands, tree, list(sit.ranges), device="cuda", dtype=torch.float64)
        done = 0
        for n in BUDGETS:
            for _ in range(n - done):
                g.step()
            done = n
            torch.cuda.synchronize()
            row[f"generic_{n}"] = 100.0 * score(g.sum.double().cpu().numpy(), H, hands, tree, sit, slots, ks)
        del g
        torch.cuda.empty_cache()
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        emit(row)

    def band(x):
        return "<4" if x < 4 else "4-16" if x < 16 else ">=16"

    summary = dict(done=True, spots=len(rows), seconds=round(time.time() - t0, 1))
    reaches = False
    for n in BUDGETS:
        fm, gm, qm = (statistics.mean(r[f"{k}_{n}"] for r in rows) for k in ("fused", "generic", "Q3"))
        worst = max(r[f"fused_{n}"] - r[f"generic_{n}"] for r in rows)
        hit = fm > 1.25 * gm or worst > 2.0
        reaches |= hit
        summary[str(n)] = dict(fused_mean=fm, generic_mean=gm, ratio=fm / gm, worst_spot_excess=worst, reaches=hit,
                               Q3_mean=qm, Q3_worst_spot_excess=max(r[f"Q3_{n}"] - r[f"generic_{n}"] for r in rows),
                               by_spr={b: dict(spots=len(s), fused=statistics.mean(r[f"fused_{n}"] for r in s),
                                               generic=statistics.mean(r[f"generic_{n}"] for r in s),
                                               Q3=statistics.mean(r[f"Q3_{n}"] for r in s))
                                       for b in ("<4", "4-16", ">=16")
                                       for s in [[r for r in rows if band(r["spr"]) == b]] if s})
    summary["reading"] = "REACHES" if reaches else "STANDS"
    emit(summary)


if __name__ == "__main__":
    main()
