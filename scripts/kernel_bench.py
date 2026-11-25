#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""The kernel work's measurements (KERNELS.md): the profile (Step 1), the
bit-identity check (rule 1), the precision gate (rule 2) and the interleaved
speed gate (rule 3).

  profile    P1 a live solve's stages; P2 a turn/river iteration's kernels;
             P3 a flop iteration's parts; P4 the precision baseline.
  identity   --old SRC: old and new `FusedSolver` hold the same bits after 1,
             2, 3 and 100 iterations on the bench trees under every option;
             both return the recorded jobs' recorded answers; the flop solver
             likewise on three spots.
  precision  [--old SRC]: float32 error against the float64 oracles: terminal
             kernels on real terminal reaches, leaf evaluators on real leaf
             reaches, and the whole solve's float64-evaluated exploitability.
  speed      [--old SRC]: old and new in alternating order, 12 rounds, with
             rule 3's verdict per measure. Without --old both sides are this
             checkout: an A/A run, the noise of the measure.
  designs    the candidates' NumPy models (`parciver.kernel_designs`) on the
             recorded jobs, on the CPU: K1's rows, K2 and K3b against the dense
             float64 matrices, K4's invariant. Float32 numbers are previews.
  selftest   the harness's own arithmetic on the CPU (no GPU).

`--terminals64`: precision and speed build both sides' float32 solvers with
`terminals64` (KERNELS Q4's gates: Q4 against Q3), and precision also reports
the sides' agreement on the terminal values (bit-equal share, largest ulps);
profile builds its float32 solvers with it (the live config, Q4): P1's, P2's
and P4's. P2 also measures `g`, the GPU's cost per launch (KERNELS K10's
pre-gate).

`--old SRC` is a `src` directory holding the reference `parciver` package
(`--new SRC` likewise the candidate, for identity, precision and speed: this
checkout's by default):

    git archive HEAD src/parciver | tar -x -C C:/tmp/kb/old     (SRC = C:/tmp/kb/old/src)

Every command but `selftest`, `designs` and `precision --device cpu` needs the GPU to
itself and refuses to start while it is busier than `--max-busy` (5% by default; this
desktop never drops below ~30%: KERNELS.md rule 3, amendment of 4 Oct) or while another
process of this Python holds it. No dashboard."""
from __future__ import annotations

import argparse
import copy
import gc
import importlib
import importlib.util
import inspect
import json
import math
import os
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
JOBS = "C:/tmp/gpu_jobs.pkl"  # 53 real (job, answer) pairs of a searched LBR run
STARTS = "C:/tmp/d1/recs40.pkl"  # 40 real turn starts
SPOTS = str(ROOT / "runs" / "flop-eval" / "spots-540000.json")  # 40 flop-round roots
NET = "C:/tmp/audit/d2prev/turn-net/prev/net.pt"  # the preview net (w1024 d3)
CHECKS = (1, 2, 3, 100)  # rule 1's iterations
T64: dict = {}  # --terminals64: the float32 solvers' option, both sides
AT = (1, 10, 100)  # rule 2's iterations
MODULES = (("gs", "gpu_search"), ("gf", "gpu_fused"), ("fs", "flop_search"), ("vn", "value_net"), ("fl", "flop_leaves"))


# ---------------------------------------------------------------- packages, the GPU, timing


def package(src: str | None, alias: str) -> SimpleNamespace:
    """The `parciver` package under `src` (this checkout's if None), imported
    as `alias` so two versions live in one process: its modules."""
    base = Path(src) if src else ROOT / "src"
    init = base / "parciver" / "__init__.py"
    if not init.exists():
        raise SystemExit(f"no parciver package under {base}")
    spec = importlib.util.spec_from_file_location(alias, init, submodule_search_locations=[str(init.parent)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    ns = SimpleNamespace(name=alias, src=str(base))
    for short, name in MODULES:
        if (init.parent / f"{name}.py").exists():
            setattr(ns, short, importlib.import_module(f"{alias}.{name}"))
        else:
            setattr(ns, short, None)  # flop_leaves in an older tree
    return ns


def gpu_busy() -> float | None:
    """The GPU's utilisation before this process uses it, % (the largest of 5
    samples over a second); None without nvidia-smi."""
    try:
        seen = []
        for _ in range(5):
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            seen.append(float(out.split()[0]))
            time.sleep(0.2)
        return max(seen)
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def gpu_peers() -> list[int] | None:
    """Other processes of this Python holding a GPU context (D1's generator, a
    second benchmark), which the utilisation samples miss between their
    launches; None without nvidia-smi. Windows lists every desktop app as a
    GPU client, so only this interpreter's processes count."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    me = os.path.normcase(os.path.realpath(getattr(sys, "_base_executable", sys.executable)))
    peers = []
    for line in out.splitlines():
        pid, _, name = (x.strip() for x in line.partition(","))
        if pid.isdigit() and int(pid) != os.getpid() and os.path.normcase(os.path.realpath(name)) == me:
            peers.append(int(pid))
    return peers


def need_gpu(a) -> None:
    busy, peers = gpu_busy(), gpu_peers()
    a.busy_before = busy
    if busy is None:
        print("nvidia-smi not found: the GPU's idleness is unchecked")
    elif busy > a.max_busy and not a.busy_ok:
        raise SystemExit(f"the GPU is busy ({busy:.0f}%, limit {a.max_busy:g}%): these numbers need it idle (D1 still running?)")
    elif peers and not a.busy_ok:
        raise SystemExit(f"processes {peers} of this Python hold the GPU: these numbers need it alone (D1 still running?)")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest":
        raise SystemExit("TF32 is allowed in this process: not the production setting (KERNELS.md rule 6)")
    print(f"device {torch.cuda.get_device_name()}, torch {torch.__version__}, GPU utilisation before start "
          f"{'unknown' if busy is None else f'{busy:.0f}%'}")


def still_alone(a, out: dict) -> int:
    """After timings: 0 if the GPU was still this process's alone, else 2 with
    the timings marked void (a job that started part-way skews the rounds)."""
    sync()
    time.sleep(1.0)  # past the utilisation counter's window over this process's last kernels
    busy, peers = gpu_busy(), gpu_peers()
    out["gpu_busy"] = dict(before=a.busy_before, after=busy, limit=a.max_busy)
    if (busy is None or busy <= a.max_busy) and not peers or a.busy_ok:
        return 0
    out["void"] = f"at the end the GPU was {busy}% busy, held by {peers}"
    print(f"VOID: {out['void']}: run these timings again with the GPU alone")
    return 2


def sync() -> None:
    torch.cuda.synchronize()


def timed(f) -> float:
    """Seconds of `f()` including everything it queued on the GPU."""
    sync()
    t = time.perf_counter()
    f()
    sync()
    return time.perf_counter() - t


def capture(step):
    """`step()`'s kernels as a CUDA graph (recorded, not run). Every kernel
    must be compiled already; no garbage collection while capturing, as
    `FusedSolver.step`."""
    g = torch.cuda.CUDAGraph()
    sync()
    enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.graph(g):
            step()
    finally:
        if enabled:
            gc.enable()
    return g


def abba(sides: dict, rounds: int) -> list[dict]:
    """Each side's measure per round, the order alternating (old new, new old, ...)."""
    names = list(sides)
    rows = []
    for r in range(rounds):
        row = {}
        for k in (names if r % 2 == 0 else names[::-1]):
            row[k] = sides[k]()
        rows.append(row)
    return rows


def verdict(old: list[float], new: list[float]) -> dict:
    """KERNELS.md rule 3 on paired measures (smaller is better)."""
    ratio = [o / n for o, n in zip(old, new)]
    need = math.ceil(len(ratio) * 10 / 12)
    med = statistics.median(ratio)
    wins, losses = sum(x > 1 for x in ratio), sum(x < 1 for x in ratio)
    return dict(rounds=len(ratio), median_ratio=med, new_wins=wins, old_wins=losses,
                faster=bool(med >= 1.02 and wins >= need), not_slower=bool(med >= 0.99 and losses < need))


def stat(errors) -> dict:
    e = torch.as_tensor(errors, dtype=torch.float64)
    return dict(nodes=int(e.numel()), mean=float(e.mean()), max=float(e.max()))


# ---------------------------------------------------------------- inputs


def load_jobs(a) -> list:
    return pickle.load(open(a.jobs, "rb"))


def load_starts(a) -> list:
    """The recorded turn starts as jobs (no frozen slots), each with its
    `half`: a player's share of the pot at the start, chips."""
    return [dict(r["job"], board=r["board"], strength=r["strength"], alive=r["alive"], ranges=r["ranges"], frozen=[],
                 want=[], half=r["half"]) for r in pickle.load(open(a.starts, "rb"))]


def load_spots(a) -> list:
    return json.loads(Path(a.spots).read_text())["spots"]


def hands_of(P, job):
    return P.gs.Hands.from_arrays(job["board"], job["strength"], job["alive"])


def solver_of(P, job, hands=None, **kw):
    """A `FusedSolver` on a recorded job, its frozen slots frozen (float32: with `T64`)."""
    if kw.get("dtype", torch.float32) == torch.float32:
        kw = {**T64, **kw}
    s = P.gf.FusedSolver(hands or hands_of(P, job), job, job["ranges"], **kw)
    for slot, strategy in job["frozen"]:
        s.freeze(slot, np.asarray(strategy))
    return s


def takes(P, name: str) -> bool:
    return name in inspect.signature(P.gf.FusedSolver.__init__).parameters


def bench_trees(P):
    """`gpu_search_bench.py`'s three trees: (name, hands, tree, ranges)."""
    b = P.gs.board(7)
    for name, cards, sp in (("river", b, P.gs.spec()), ("turn (agent menus)", b[:4], P.gs.small_turn_spec()),
                            ("turn (full menus)", b[:4], P.gs.turn_spec())):
        hands = P.gs.Hands(cards)
        yield name, hands, P.gs.Tree(sp, hands), [P.gs.range_(hands.n, 31), P.gs.range_(hands.n, 32)]


def dense_values(net):
    """`flop_search.net_values` in the net's own dtype (that one feeds
    float32): over a float64 copy of the net, the leaf oracle."""

    def f(ranges, board, live, half, stack):
        L, T = ranges.shape[:2]
        wd = net.mlp[0].weight.dtype
        onehot = torch.zeros(T, 52, dtype=wd, device=ranges.device)
        onehot.scatter_(1, board, 1.0)
        h, s = half.double(), stack.double()
        scal = torch.stack([h / (h + s), s / (2 * h)], -1).to(wd)
        with torch.no_grad():
            v = net(ranges.reshape(L * T, 2, -1).to(wd), onehot.repeat(L, 1), scal.repeat_interleave(T, 0),
                    live.repeat(L, 1))
        return v.view(L, T, 2, -1).to(ranges.dtype)

    return f


def leaf_factory(P, name: str, board, net, device):
    """A leaf evaluator by name: `dense` (`NetLeaves` over `net_values`), a
    class of `flop_leaves` (`FactoredLeaves`), or a class and its last
    stage's dtype (`FactoredLeaves:float64`, N1b; `FactoredLeaves:float64+eps`,
    N1c: the zero-sum term's ranges in it too)."""
    if name == "dense":
        return P.fs.NetLeaves(board, P.fs.net_values(net), device)
    cls, _, last = name.partition(":")
    last, _, extra = last.partition("+")
    if not has_leaves(P, name):
        raise SystemExit(f"{P.src} has no flop_leaves.{cls}")
    kw = dict(last=getattr(torch, last)) if last else {}
    # Checked before the leaves are built (their tables go on the device): a
    # last stage in the net's own dtype is N1 under N1b's or N1c's name.
    wd = net.mlp[0].weight.dtype
    if last and kw["last"] == wd:
        raise SystemExit(f"{name!r}: the net is {last} already, so this is plain N1 (`{cls}`)")
    if extra == "eps":
        if not last:
            raise SystemExit(f"{name!r}: +eps needs a last-stage dtype other than the net's {wd} "
                             f"(e.g. {cls}:float64+eps)")
        if "last_eps" not in inspect.signature(getattr(P.fl, cls)).parameters:
            raise SystemExit(f"{P.src}'s flop_leaves.{cls} has no last_eps (N1c)")
        kw["last_eps"] = True
    elif extra:
        raise SystemExit(f"unknown leaf option {extra!r} in {name!r}")
    return getattr(P.fl, cls)(board, net, device, **kw)


def has_leaves(P, name: str) -> bool:
    return name == "dense" or (P.fl is not None and hasattr(P.fl, name.partition(":")[0]))


def flop_solver(P, job, leaves, device="cuda"):
    s = P.fs.FlopSolver(job, leaves, P.fs.runout_equity(job["board"], device), device=device)
    for slot, strategy in job["frozen"]:
        s.freeze(slot, strategy)
    return s


# ---------------------------------------------------------------- rule 2: the oracles


class Dense:
    """A job's terminal values from the dense card-removal matrices in float64
    (`C @ r` x the amount won, `W_v @ r` x the smaller stake), rows in the
    solver's terminal order (folds, then showdowns by view)."""

    def __init__(self, hands, plan, device="cuda"):
        f = dict(dtype=torch.float64, device=device)
        self.F = plan.F
        self.compat = torch.as_tensor(hands.compat, **f)
        self.strength, self.alive = torch.as_tensor(hands.strength, **f), torch.as_tensor(hands.alive, **f)
        self.amt = torch.as_tensor(plan.amt, **f)[:, :plan.F]
        self.view = torch.as_tensor(np.asarray(plan.sd_view, dtype=np.int64)[:plan.S], device=device)
        self.m = torch.as_tensor(np.asarray(plan.sd_m, dtype=np.float64)[:plan.S], **f)

    def __call__(self, p: int, rq, amounts32: bool = False):
        """`amounts32`: times the amounts as a float32 solver holds them."""
        r, F = rq.double(), self.F
        amt, m = (self.amt.float().double(), self.m.float().double()) if amounts32 else (self.amt, self.m)
        out = torch.zeros_like(r)
        out[:F] = (r[:F] @ self.compat) * amt[p][:, None]  # compat is symmetric
        for v in self.view.unique().tolist():
            at = torch.nonzero(self.view == v).flatten()
            s, live = self.strength[v], self.alive[v]
            w = torch.sign(s[:, None] - s[None, :]) * self.compat * live[:, None] * live[None, :]
            out[F + at] = (r[F + at] @ w.T) * m[at][:, None]
        return out

    def scale(self, p: int, rq):
        """Per row: the opponent's mass x the chips at stake."""
        return rq.double().sum(1) * torch.cat([self.amt[p].abs(), self.m])


def terminal_rows(P, job, hands, plan, at=AT):
    """The float32 terminal reach rows of `job`'s solve as each traversal's
    terminal kernels read them: (iteration, traverser, rows [F + S, n])."""
    s = solver_of(P, job, hands, plan=plan, graph=False)
    for it in range(1, max(at) + 1):
        s.begin()
        for p in (0, 1):
            s.traverse(p)
            if it in at:
                yield it, p, s.rq[s.f_lo:].clone()


def ordered(x):
    """float32 values as integers in their order, one step per ulp (-0 and +0 both 0)."""
    i = x.contiguous().view(torch.int32).to(torch.int64)
    return torch.where(i < 0, -(i & 0x7FFFFFFF), i)


def terminal_precision(base, sides: dict, jobs, at=AT) -> dict:
    """Rule 2, terminal kernels: each side's float32 error on the base
    package's reach rows. `sides`: {name: package}. Also the holdings outside
    Q3's per-holding bound (one rounding of the dense value on the solvers'
    float32 amounts: |v - d| <= 2^-24 |d| + 1e-12 x scale)."""
    errors = {k: [] for k in sides}
    outside = {k: [0, 0] for k in sides}
    agree = {n: [0, 0, 0] for n in at}  # two sides: per iteration, [bit-equal, values, largest ulps]
    for job, _ in jobs:
        hands = hands_of(base, job)
        plan = base.gf.host_plan(hands, job, job["ranges"])
        dense = Dense(hands, plan)
        solvers = {k: solver_of(P, job, graph=False) for k, P in sides.items()}
        for s in solvers.values():
            assert s.term_nodes == plan.term_nodes, "the sides order their terminal rows differently"
        for it, p, rq in terminal_rows(base, job, hands, plan, at):
            want, scale = dense(p, rq), dense.scale(p, rq)
            want32 = dense(p, rq, amounts32=True)
            bound = 2.0 ** -24 * want32.abs() + 1e-12 * scale[:, None]
            ok = scale > 0
            got32 = {}
            for k, s in solvers.items():
                got32[k] = g32 = s.terminal_values(p, rq)
                got = g32.double()
                err = (got - want).abs().amax(1)
                errors[k].append((err[ok] / scale[ok]).cpu())
                outside[k][0] += int(((got - want32).abs() > bound).sum())
                outside[k][1] += got.numel()
            if len(got32) == 2:
                x, y = got32.values()
                ag = agree[it]
                ag[0] += int((x.contiguous().view(torch.int32) == y.contiguous().view(torch.int32)).sum())
                ag[1] += x.numel()
                ag[2] = max(ag[2], int((ordered(x) - ordered(y)).abs().max()))
        del solvers, dense
    out = {k: dict(stat(torch.cat(v)), outside=outside[k][0], holdings=outside[k][1]) for k, v in errors.items()}
    if len(sides) == 2:
        out["agreement"] = {str(n): dict(equal=e, values=v, max_ulps=u) for n, (e, v, u) in agree.items()}
    return out


def leaf_inputs(P, net, job, device, at=AT) -> list:
    """The float32 leaf reaches of `job`'s flop solve (dense net leaves,
    eager) at the given iterations: [(reach [L, 2, n], half, stack)], two per
    iteration (one per traversal)."""
    leaves = leaf_factory(P, "dense", job["board"], net, device)
    seen = []

    def tap(both, half, stack):
        if s.iter in at:
            seen.append((both.clone(), half, stack))
        return leaves(both, half, stack)

    s = flop_solver(P, job, tap, device)
    s.solve(max(at), graph=False)
    return seen


def leaf_precision(base, sides: dict, net, spots, device, at=AT) -> dict:
    """Rule 2, flop leaves: each evaluator's float32 error against the dense
    forward over a float64 copy of the net, relative to the leaf's largest
    value. `sides`: {name: (package, evaluator name)}."""
    net64 = copy.deepcopy(net).double()
    errors = {k: [] for k in sides}
    for job in spots:
        board = job["board"]
        oracle = base.fs.NetLeaves(board, dense_values(net64), device, torch.float64)
        evaluators = {k: leaf_factory(P, name, board, net, device) for k, (P, name) in sides.items()}
        for both, half, stack in leaf_inputs(base, net, job, device, at):
            want = oracle(both.double(), half.double(), stack.double())
            scale = want.abs().amax((1, 2))
            ok = scale > 0
            for k, f in evaluators.items():
                err = (f(both, half, stack).double() - want).abs().amax((1, 2))
                errors[k].append((err[ok] / scale[ok]).cpu())
    return {k: stat(torch.cat(v)) for k, v in errors.items()}


def evaluate64(P, job, s32) -> tuple[float, float]:
    """(exploitability, player 0's value) of `s32`'s float32 strategy sums,
    evaluated by a float64 solver of package `P`: chips per dealt pair."""
    s64 = solver_of(P, job, dtype=torch.float64, graph=False)
    s64.sum.copy_(s32.sum)
    return s64.exploitability(), s64.value(0)


def solve_precision(base, sides: dict, starts, budgets=(100, 400)) -> dict:
    """Rule 2, whole solve: the mean float64-evaluated exploitability (% of
    the pot) of each side's float32 solve at each budget, and the float32
    readouts' error against float64 (Q2), in pots."""
    out = {k: dict(expl={n: [] for n in budgets}, readout_expl=[], readout_value=[]) for k in sides}
    for job in starts:
        pot = 2.0 * job["half"]
        for k, P in sides.items():
            s = solver_of(P, job)
            for n in budgets:
                while s.iter < n:
                    s.step()
                e64, v64 = evaluate64(base, job, s)
                out[k]["expl"][n].append(100.0 * e64 / pot)
                out[k]["readout_expl"].append(abs(s.exploitability() - e64) / pot)
                out[k]["readout_value"].append(abs(s.value(0) - v64) / pot)
            del s
    return {k: dict(expl={n: float(np.mean(v)) for n, v in r["expl"].items()},
                    readout_expl_max=float(np.max(r["readout_expl"])), readout_value_max=float(np.max(r["readout_value"])))
            for k, r in out.items()}


# ---------------------------------------------------------------- P1: a live solve's stages

STAGES = ("hands", "plan", "construct", "freeze", "first step + capture", "replays", "readback")


def stages(P, job, iters: int | None = None) -> dict:
    """Seconds of each stage of `SubgameGpu.__call__` on a cold board cache.
    `hands` and `plan` run outside the GPU lock in a live search."""
    gf = P.gf
    want = list(job["want"])
    t = [time.perf_counter()]

    def mark():
        sync()
        t.append(time.perf_counter())

    hands = hands_of(P, job)
    mark()
    plan = gf.host_plan(hands, job, job["ranges"])
    mark()
    s = gf.FusedSolver(hands, job, job["ranges"], plan=plan, **T64, **(dict(sums=want) if takes(P, "sums") else {}))
    mark()
    for slot, strategy in job["frozen"]:
        s.freeze(slot, np.asarray(strategy))
    mark()
    s.step()
    mark()
    n = int(job["iters"]) if iters is None else iters
    for _ in range(n - 1):
        s.step()
    mark()
    avg = s.average(want).to(torch.float64).cpu().numpy()
    [avg[i, :plan.dk[w]].reshape(-1).tolist() for i, w in enumerate(want)]
    mark()
    return dict(zip(STAGES, np.diff(t)))


def profile_stages(P, jobs) -> dict:
    out = {}
    for job, _ in jobs[:3]:  # compile the kernels of both streets' shapes
        stages(P, job)
    for label, iters in (("own iterations", None), ("400 iterations", 400)):
        for street, cards in (("turn", 4), ("river", 5)):
            rows = [stages(P, job, iters) for job, _ in jobs if len(job["board"]) == cards]
            mean = {k: 1e3 * float(np.mean([r[k] for r in rows])) for k in STAGES}
            locked = sum(mean[k] for k in STAGES[2:])
            out[f"{street}, {label}"] = dict(solves=len(rows), ms=mean, locked_ms=locked)
            print(f"P1 {street}, {label}: {len(rows)} solves, {locked:.1f} ms in the lock")
            for k in STAGES:
                print(f"    {k:>22}: {mean[k]:8.2f} ms  {mean[k] / locked:6.1%} of locked")
    return out


# ---------------------------------------------------------------- P2: an iteration's kernels


class _Launch:
    def __init__(self, tap, name, kernel):
        self.tap, self.name, self.kernel = tap, name, kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]
        return lambda *args, **kw: self.tap.run(self.name, lambda: launch(*args, **kw))


class _Torch:
    """`torch` for a tapped module: its `matmul` is a launch (the fold sums')."""

    def __init__(self, tap):
        self._tap = tap

    def __getattr__(self, k):
        return getattr(torch, k)

    def matmul(self, *args, **kw):
        return self._tap.run("matmul", lambda: torch.matmul(*args, **kw))


class Tap:
    """Wraps a `gpu_fused` module's kernel launches and its fold matmul, so a
    solver step can be recorded with some launches left out, or run with a
    CUDA event on each side of every launch. Labels: `prep`, and
    `<kernel>.p<traversal>.<k>` for the k-th such launch of that traversal
    (`down.p0.3`: the first traversal's fourth level down). A skip is a label
    or a prefix of labels (`down`, `up.p1`)."""

    KERNELS = ("_prep", "_discount", "_down", "_up", "_fold", "_showdown",
               "_fold64", "_showdown64", "_fold_ff", "_showdown_ff")  # terminals64: Q3's, Q4's

    def __init__(self, gf):
        self.gf, self.skip, self.events, self.labels = gf, (), None, []
        self._saved = {k: getattr(gf, k) for k in self.KERNELS + ("torch",) if hasattr(gf, k)}  # older packages lack some
        for k in self.KERNELS:
            if k in self._saved:
                setattr(gf, k, _Launch(self, k[1:], self._saved[k]))
        gf.torch = _Torch(self)
        self._reset()

    def close(self):
        for k, v in self._saved.items():
            setattr(self.gf, k, v)

    def _reset(self):
        self.traversal, self.count, self.last, self.labels = 0, {}, None, []

    def _label(self, name: str) -> str:
        if name == "prep":
            self._reset()
            return name
        if name == "down" and self.last == "up":
            self.traversal += 1
        self.last = name
        k = self.count.get((name, self.traversal), 0)
        self.count[name, self.traversal] = k + 1
        return f"{name}.p{self.traversal}.{k}"

    def run(self, name: str, launch):
        label = self._label(name)
        self.labels.append(label)
        if any(label == s or label.startswith(s + ".") for s in self.skip):
            return None
        if self.events is None:
            return launch()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        out = launch()
        b.record()
        self.events.append((label, a, b))
        return out


def replay_ms(graphs: dict, reps: int, rounds: int, before=lambda: None) -> dict:
    """Median ms per replay of each graph, the graphs taken in alternating order."""
    names = list(graphs)
    seen = {k: [] for k in names}

    def batch(g):
        for _ in range(reps):
            g.replay()

    for r in range(rounds):
        for k in (names if r % 2 == 0 else names[::-1]):
            before()
            seen[k].append(1e3 * timed(lambda: batch(graphs[k])) / reps)
    return {k: statistics.median(v) for k, v in seen.items()}


def profile_kernels(P, job, levels: bool, reps: int = 200, rounds: int = 7, warm: int = 10, event_iters: int = 50) -> dict:
    """P2 on one job. Primary: leave-one-out on replayed graphs (the full
    iteration's graph against the same graph without one class of launches:
    GPU time in context, no Python). Secondary, as first registered: CUDA
    events around each launch in eager mode, which also count the host's
    launch latency whenever the GPU queue is empty (an upper bound)."""
    tap = Tap(P.gf)
    try:
        s = solver_of(P, job, graph=False, **(dict(sums=list(job["want"])) if takes(P, "sums") else {}))
        for _ in range(warm):
            s._step_device()
        sync()
        every = list(dict.fromkeys(tap.labels))  # one step's launches, in order
        kinds = list(dict.fromkeys(x if x == "prep" else ".".join(x.split(".")[:2]) for x in every))
        classes = kinds + (["down", "up"] if levels else [])
        if levels:
            classes += [x for x in every if x.split(".")[0] in ("down", "up")]
        graphs = {}
        for c in ["full"] + classes + ["prep only"]:
            tap.skip = () if c == "full" else tuple(k for k in kinds if k != "prep") if c == "prep only" else (c,)
            graphs[c] = capture(s._step_device)
        tap.skip = ()
        ms = replay_ms(graphs, reps, rounds)
        tap.events = []
        for _ in range(event_iters):
            s._step_device()
        sync()
        events: dict = {}
        for label, a, b in tap.events:
            kind = label if label == "prep" else ".".join(label.split(".")[:2])
            for key in {kind, label} if levels else {kind}:
                events[key] = events.get(key, 0.0) + a.elapsed_time(b) / event_iters
        tap.events = None
    finally:
        tap.close()
    full = ms["full"]
    shares = {c: (full - ms[c]) / full for c in classes}
    disjoint = sum(shares[c] for c in kinds)
    print(f"P2 {len(job['board'])}-card job, {s.N} rows x {s.n} hands, {len(s.levels)} levels: "
          f"{full:.4f} ms per replayed iteration; prep-only graph {ms['prep only']:.4f} ms (replay overhead)")
    for c in classes:
        print(f"    {c:>14}: leave-one-out {full - ms[c]:8.4f} ms {shares[c]:6.1%}   events {events.get(c, float('nan')):8.4f} ms")
    print(f"    the {len(kinds)} disjoint classes sum to {disjoint:.1%}; the events sum to "
          f"{sum(events.get(c, 0.0) for c in kinds):.4f} ms (eager: host latency included)")
    return dict(rows=s.N, hands=s.n, levels=len(s.levels), ms=ms, shares=shares, events=events, disjoint=disjoint)


_NOOP = None


def launch_cost(reps: int = 200, rounds: int = 7, few: int = 20, many: int = 200, check: int = 1000) -> dict:
    """`g`, the GPU's cost per launch (KERNELS K10's pre-gate): replayed graphs
    of `many` and of `few` launches of a one-program kernel that does nothing,
    timed as P2's graphs; `g = (t_many - t_few) / (many - few)`. A one-program
    launch has no grid to fill or drain, so `g` is a lower bound on a wave
    launch's cost. Descriptive beside it: the slope from `many` to `check`
    launches (if the host's replay submission bounds the `few` graph, `g` is
    low and this slope higher), and the intercept `t_few - few x g`, the
    per-replay overhead (compare P2's prep-only graph)."""
    global _NOOP
    import triton

    if _NOOP is None:
        @triton.jit
        def _noop(x_ptr):
            pass

        _NOOP = _noop
    x = torch.zeros(1, device="cuda")
    _NOOP[(1,)](x)  # compile before capturing
    graphs = {k: capture(lambda k=k: [_NOOP[(1,)](x) for _ in range(k)]) for k in (few, many, check)}
    ms = replay_ms(graphs, reps, rounds)
    g = (ms[many] - ms[few]) / (many - few)
    g_check = (ms[check] - ms[many]) / (check - many)
    intercept = ms[few] - few * g
    print(f"P2 launch cost g: {1e3 * g:.3f} us per launch (graphs of {many} and {few} one-program no-op launches: "
          f"{ms[many]:.4f} and {ms[few]:.4f} ms per replay); {many} -> {check} launches {1e3 * g_check:.3f} us; "
          f"intercept {1e3 * intercept:.2f} us per replay")
    return dict(g_ms=g, g_check_ms=g_check, intercept_ms=intercept, ms={str(k): v for k, v in ms.items()},
                few=few, many=many, check=check)


# ---------------------------------------------------------------- P3: a flop iteration's parts

_MISSING = object()


def profile_flop(P, net, job, leaves_name: str = "dense", reps: int = 200, rounds: int = 7) -> dict:
    """P3 on one spot: the iteration's graph against the same graph with one
    part replaced by a stored result (leave-one-out, as P2)."""
    leaves = leaf_factory(P, leaves_name, job["board"], net, "cuda")
    s = flop_solver(P, job, leaves)
    s.solve(10, graph=False)
    sig = s.current()
    reach = s._reach(sig)
    both = reach[:, s.leaf].transpose(0, 1)
    leaf_out = leaves(both, s.leaf_half, s.leaf_stack).clone()
    stubs = {"current": dict(current=lambda: sig), "reach": dict(_reach=lambda strat: reach),
             "terminals": dict(_terminals=lambda p, r: None),
             "leaves": dict(leaves=lambda b, h, st, players=(0, 1): leaf_out[:, players[0]:players[-1] + 1]),  # a view
             "up": dict(_up=lambda p, strat, mode: None)}
    net_out = {}
    real_evaluate = getattr(leaves, "evaluate", None)
    if real_evaluate is not None:  # the dense path: the net's forward alone
        leaves.evaluate = lambda *x: net_out.setdefault("v", real_evaluate(*x))
        leaves(both, s.leaf_half, s.leaf_stack)
        leaves.evaluate = real_evaluate

    def step():
        s._begin_device()
        for p in (0, 1):
            s.traverse(p)

    def without(patch: dict):
        saved = {k: s.__dict__.get(k, _MISSING) for k in patch}
        for k, v in patch.items():
            setattr(s, k, v)
        try:
            return capture(step)
        finally:
            for k, v in saved.items():
                delattr(s, k) if v is _MISSING else setattr(s, k, v)

    graphs = {"full": capture(step)}
    for name, patch in stubs.items():
        graphs[name] = without(patch)
    if real_evaluate is not None:
        leaves.evaluate = lambda *x: net_out["v"]
        graphs["net"] = capture(step)
        leaves.evaluate = real_evaluate
    # The discount tables hold 1,024 iterations and a replay reads the next: rewind before each batch.
    ms = replay_ms(graphs, reps, rounds, before=lambda: s._t.fill_(10))
    full = ms["full"]
    part = {k: full - v for k, v in ms.items() if k != "full"}
    part["folds + runouts"] = part["terminals"] - part["leaves"]
    if "net" in part:
        part["leaf preparation + reduction"] = part["leaves"] - part["net"]
    part["rest (sum update, discount)"] = full - part["current"] - part["reach"] - part["terminals"] - part["up"]
    kernels = None
    try:
        from torch.profiler import ProfilerActivity, profile

        s._t.fill_(10)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step()
            sync()
        kernels = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    except Exception as e:  # the count is a side note
        print(f"    (no kernel count: {e!r})")
    print(f"P3 {leaves_name} leaves, {len(s.leaf)} leaves, {s.S} slots: {full:.4f} ms per replayed iteration"
          + (f", {kernels} GPU kernels" if kernels is not None else ""))
    for k, v in part.items():
        print(f"    {k:>30}: {v:8.4f} ms {v / full:6.1%}")
    return dict(ms=ms, parts=part, kernels=kernels)


# ---------------------------------------------------------------- commands


def cmd_profile(a) -> int:
    need_gpu(a)
    P = package(None, "parciver_new")
    parts = a.parts.split(",")
    out = {}
    jobs = load_jobs(a)
    if "p1" in parts:
        out["p1"] = profile_stages(P, jobs)
    if "p2" in parts:
        turn = sorted((j for j, _ in jobs if len(j["board"]) == 4), key=lambda j: len(j["kind"]))
        river = sorted((j for j, _ in jobs if len(j["board"]) == 5), key=lambda j: len(j["kind"]))
        picks = [turn[0], turn[len(turn) // 2], turn[-1], river[len(river) // 2]]
        out["p2"] = [profile_kernels(P, job, a.levels) for job in picks]
        out["g"] = launch_cost()
    if "p3" in parts:
        net, _ = P.vn.load(a.net)
        spots = load_spots(a)
        out["p3"] = [profile_flop(P, net, job, name) for job in spots[:5] for name in a.leaves.split(",")]
    if "p4" in parts:
        out["p4"] = precision(a, P, {"current": P}, {"current": (P, "dense")})
    code = still_alone(a, out)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
    return code


def precision(a, base, sides: dict, leaf_sides: dict) -> dict:
    out = {}
    parts = a.precision_parts.split(",")
    if "terminal" in parts:
        out["terminal"] = terminal_precision(base, sides, load_jobs(a))
        for n, g in out["terminal"].pop("agreement", {}).items():
            print(f"agreement at iteration {n}: {g['equal']} of {g['values']} values bit-equal "
                  f"({g['equal'] / g['values']:.6%}), largest difference {g['max_ulps']} ulps")
            out.setdefault("agreement", {})[n] = g
        for k, v in out["terminal"].items():
            print(f"rule 2 terminals, {k}: mean {v['mean']:.3e}  max {v['max']:.3e}  ({v['nodes']} rows); "
                  f"{v['outside']} of {v['holdings']} holdings outside Q3's per-holding bound")
    if "leaf" in parts:
        net, _ = base.vn.load(a.net, a.device)
        spots = load_spots(a)[:a.spots_n] if a.spots_n else load_spots(a)
        out["leaf"] = leaf_precision(base, leaf_sides, net, spots, a.device)
        for k, v in out["leaf"].items():
            print(f"rule 2 leaves, {k}: mean {v['mean']:.3e}  max {v['max']:.3e}  ({v['nodes']} leaves, {a.device})")
    if "solve" in parts:
        out["solve"] = solve_precision(base, sides, load_starts(a))
        for k, v in out["solve"].items():
            print(f"rule 2 whole solve, {k}: float64 exploitability "
                  + ", ".join(f"{e:.4f}% pot at {n}" for n, e in v["expl"].items())
                  + f"; float32 readouts off by at most {v['readout_expl_max']:.2e} (exploitability) and "
                    f"{v['readout_value_max']:.2e} (value) pots")
    return out


def cmd_precision(a) -> int:
    if a.device == "cuda":
        need_gpu(a)
    elif a.precision_parts != "leaf":
        raise SystemExit("on the CPU only the leaf part runs (--precision-parts leaf)")
    new = package(a.new, "parciver_new")
    old = package(a.old, "parciver_old") if a.old else new
    sides = {"old": old, "new": new} if a.old else {"current": new}
    leaf_sides = {f"{side} {name}": (P, name) for side, P in sides.items() for name in a.leaves.split(",")
                  if has_leaves(P, name)}
    out = precision(a, old, sides, leaf_sides)
    if "leaf" in out:
        ref = out["leaf"]["old dense" if a.old else "current dense"]
        for k, v in out["leaf"].items():
            if not k.endswith(" dense"):
                print(f"rule 2 leaves, {k} against the dense path: no worse in mean {v['mean'] <= ref['mean']}, "
                      f"in max {v['max'] <= ref['max']}; strictly lower in both {v['mean'] < ref['mean'] and v['max'] < ref['max']}")
    if a.old:
        for part in ("terminal",):
            if part in out:
                o, n = out[part]["old"], out[part]["new"]
                print(f"rule 2 {part}: new <= old in mean {n['mean'] <= o['mean']}, in max {n['max'] <= o['max']}")
        if "solve" in out:
            for n, e in out["solve"]["old"]["expl"].items():
                rel = abs(out["solve"]["new"]["expl"][n] - e) / e
                print(f"rule 2 whole solve at {n}: relative difference of the means {rel:.3%} (bound 0.5%): {rel <= 0.005}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
    return 0


OPTIONS = [("default", {}), ("eager", dict(graph=False)), ("gamma 5", dict(gamma=5)), ("dcfr+", dict(plus=True, gamma=4)),
           ("postflop-solver", dict(gamma=3, shift=2, restart=True)),
           ("float16 regrets", dict(regret_dtype=torch.float16)), ("bfloat16 regrets", dict(regret_dtype=torch.bfloat16)),
           ("kept sums", "sums"), ("frozen slots", "frozen")]


def cmd_identity(a) -> int:
    need_gpu(a)
    old, new = package(a.old, "parciver_old"), package(a.new, "parciver_new")
    fields = a.fields.split(",")
    bad = 0
    for (name, ho, to, ranges), (_, hn, tn, _) in zip(bench_trees(old), bench_trees(new)):
        dec = [x for x in to.nodes if x.kind == "D"]
        other = next(x for x in dec if x.player != dec[0].player)
        for label, opt in OPTIONS:
            kept, kw_old, kw_new, frozen = None, {}, {}, []
            if opt == "sums":  # the kept slots' sums equal a full solve's (the old side keeps them all)
                kept = [0, other.slot, len(dec) - 1]
                kw_new = dict(sums=kept)
            elif opt == "frozen":
                for x in (dec[0], other):
                    st = np.random.default_rng(x.slot).random((len(x.children), ho.n)) + 0.05
                    frozen.append((x.slot, st / st.sum(0)))
            else:
                kw_old = kw_new = opt
            lacking = [side for side, P, kw in (("old", old, kw_old), ("new", new, kw_new)) if not all(takes(P, k) for k in kw)]
            if lacking:
                print(f"{name:20s} {label:18s}: the {' and '.join(lacking)} package has no such option, skipped")
                continue
            so = old.gf.FusedSolver(ho, to, ranges, dtype=torch.float32, **kw_old)
            sn = new.gf.FusedSolver(hn, tn, ranges, dtype=torch.float32, **kw_new)
            for s in (so, sn):
                for slot, st in frozen:
                    s.freeze(slot, st)
            diff = []
            for n in CHECKS:
                while so.iter < n:
                    so.step()
                    sn.step()
                for f in fields:
                    x, y = getattr(so, f), getattr(sn, f)
                    if kept is not None and f == "sum":
                        x, y = x[kept], y[kept]
                    if not torch.equal(x, y):
                        diff.append(f"{f}@{n}")
            bad += bool(diff)
            print(f"{name:20s} {label:18s}: {'IDENTICAL' if not diff else 'DIFFERENT ' + ' '.join(diff)}", flush=True)
            del so, sn
    jobs = load_jobs(a)
    for side, P in (("old", old), ("new", new)):
        g = P.gf.SubgameGpu()
        wrong = sum(g(job) != rec for job, rec in jobs)
        bad += wrong
        print(f"{len(jobs)} recorded jobs, {side}: {'the recorded answers' if not wrong else f'{wrong} DIFFER'}", flush=True)
    if Path(a.net).exists() and "flop" not in a.skip:
        for i, job in enumerate(load_spots(a)[:3]):
            pair = []
            for P in (old, new):
                net, _ = P.vn.load(a.net)
                pair.append(flop_solver(P, job, leaf_factory(P, "dense", job["board"], net, "cuda")))
            diff = []
            for n in (1, 2, 3, 50):
                for s in pair:
                    s.solve(n - s.iter)
                diff += [f"{f}@{n}" for f in ("regret", "sum") if not torch.equal(getattr(pair[0], f), getattr(pair[1], f))]
            bad += bool(diff)
            print(f"flop spot {i}: {'IDENTICAL' if not diff else 'DIFFERENT ' + ' '.join(diff)}", flush=True)
    print("RESULT: " + ("same bits" if bad == 0 else f"{bad} differences"))
    return 1 if bad else 0


def cmd_speed(a) -> int:
    need_gpu(a)
    new = package(a.new, "parciver_new")
    old = package(a.old, "parciver_old") if a.old else package(a.new, "parciver_same")
    sides = {"old": old, "new": new}
    measures = a.measures.split(",")
    out = {}

    def report(name, rows, unit, per):
        o, n = [r["old"] for r in rows], [r["new"] for r in rows]
        v = verdict(o, n)
        out[name] = dict(old=o, new=n, **v)
        print(f"{name}: old {1e3 * statistics.median(o) / per:.4f} {unit}, new {1e3 * statistics.median(n) / per:.4f}; "
              f"ratio old/new {v['median_ratio']:.4f}, new ahead in {v['new_wins']}/{v['rounds']} rounds -> "
              f"{'FASTER' if v['faster'] else 'not slower' if v['not_slower'] else 'SLOWER'}", flush=True)

    if "live" in measures:
        jobs = load_jobs(a)
        runs = {}
        for side, P in sides.items():
            g = P.gf.SubgameGpu(**T64)
            wrong = sum(g(job) != rec for job, rec in jobs)  # also compiles every kernel
            if wrong:
                print(f"live, {side}: {wrong} answers differ from the recorded ones (expected only for an oracle candidate)")

            def run(g=g):
                before = g.seconds
                for job, _ in jobs:
                    g._hands.clear()  # a new board per deal
                    g(job)
                return g.seconds - before

            runs[side] = run
        report("live solve (53 recorded jobs, locked)", abba(runs, a.rounds), "ms/solve", len(jobs))
    if "iter" in measures:
        for (name, ho, to, ranges), (_, hn, tn, _) in zip(bench_trees(old), bench_trees(new)):
            runs = {}
            for side, P, h, t in (("old", old, ho, to), ("new", new, hn, tn)):
                s = P.gf.FusedSolver(h, t, ranges, dtype=torch.float32, **T64)
                for _ in range(3):
                    s.step()

                def run(s=s):
                    return timed(lambda: [s.step() for _ in range(a.iters)])

                runs[side] = run
            report(f"iteration, {name}", abba(runs, a.rounds), "ms/iter", a.iters)
            del runs
    if "flop" in measures:
        spots = load_spots(a)
        iters = min(a.flop_iters, (1000 - 2) // a.rounds)  # one graph per solver: stay inside its discount tables
        runs = {}
        for side, P, name in (("old", old, a.old_leaves), ("new", new, a.new_leaves)):
            net, _ = P.vn.load(a.net)
            solvers = [flop_solver(P, job, leaf_factory(P, name, job["board"], net, "cuda")) for job in spots]
            for s in solvers:
                s.solve(2)  # a real step and the capture

            def run(solvers=solvers):
                return timed(lambda: [s.solve(iters) for s in solvers])

            runs[side] = run
        report(f"flop iteration ({a.old_leaves} -> {a.new_leaves} leaves, {len(spots)} spots)", abba(runs, a.rounds),
               "ms/iter", iters * len(spots))
    code = still_alone(a, out)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
    return code


def cmd_designs(a) -> int:
    """KERNELS.md's CPU work before D1 ends: the designs on real subgames.
    Rows: each job's terminal reaches under its frozen strategies and (a) the
    uniform strategy, which is iteration 1's exactly, (b) a skewed random one."""
    P = package(None, "parciver_new")
    kd = importlib.import_module("parciver_new.kernel_designs")
    jobs = [job for job, _ in load_jobs(a)]
    jobs = jobs[:a.first] if a.first else jobs
    acc: dict = {}
    bad64, alive_breaks = 0.0, 0
    for i, job in enumerate(jobs):
        hands = hands_of(P, job)
        plan = P.gf.host_plan(hands, job, job["ranges"])
        street = "turn" if len(job["board"]) == 4 else "river"
        for name, strat in (("uniform", kd.uniform(plan)), ("skewed", kd.skewed(plan, i))):
            strat = np.array(strat)
            for slot, fixed in job["frozen"]:
                strat[slot] = 0
                strat[slot, :plan.dk[slot]] = np.asarray(fixed, dtype=np.float32).reshape(-1, plan.n)
            for p in (0, 1):
                rq, _, written = kd.down(plan, p, strat)
                ra, rrow, fewer = kd.down(plan, p, strat, alias=True)
                assert np.array_equal(ra[rrow], rq), "K1: an aliased row differs"
                rf, rs = rq[plan.f_lo:plan.f_lo + plan.F], rq[plan.s_lo:]
                view = np.asarray(plan.sd_view)[:plan.S]
                alive_breaks += int(np.count_nonzero(rs[np.asarray(plan.alive)[view] == 0]))
                sf, ss = kd.scales(plan, p, rf, rs)
                wf, ws = kd.dense_fold(hands, plan, p, rf), kd.dense_showdown(hands, plan, rs)
                row = acc.setdefault((street, name), dict(written=[], fewer=[], fold=[], k2=[], sd=[], k3b=[], fold64=[],
                                                          sd64=[]))
                row["written"].append(written)
                row["fewer"].append(fewer)
                row["fold"].append(kd.errors(kd.fold_today(plan, p, rf), wf, sf))
                row["k2"].append(kd.errors(kd.fold_cards(plan, p, rf), wf, sf))
                row["sd"].append(kd.errors(kd.showdown(plan, rs), ws, ss))
                row["k3b"].append(kd.errors(kd.showdown(plan, rs, cards=True), ws, ss))
                row["fold64"].append(kd.errors(kd.fold_cards(plan, p, rf, sums=np.float64), wf, sf))
                row["sd64"].append(kd.errors(kd.showdown(plan, rs, cards=True, sums=np.float64), ws, ss))
                if name == "uniform" and p == 0:  # rule 5 on the real subgame, float64
                    for got, want in ((kd.fold_cards(plan, p, rf, np.float64), wf),
                                      (kd.showdown(plan, rs, np.float64, cards=True), ws)):
                        bad64 = max(bad64, float(np.abs(got - want).max() / max(1.0, np.abs(want).max())))
        print(f"  job {i + 1}/{len(jobs)} ({street}, {plan.N} rows)", flush=True)
    out = {}
    for (street, name), row in sorted(acc.items()):
        w, f = np.mean(row["written"]), np.mean(row["fewer"])
        e = {k: np.concatenate(row[k]) for k in ("fold", "k2", "sd", "k3b", "fold64", "sd64")}
        out[f"{street} {name}"] = dict(rows_written=w, rows_written_k1=f, **{k: stat(v) for k, v in e.items()})
        print(f"{street}, {name} strategy ({len(row['written'])} traversals)")
        print(f"    K1: {w:.0f} -> {f:.0f} rows written per down pass ({1 - f / w:.1%} fewer)")
        for label, old, new, wide in (("folds, today -> K2 -> Q3", "fold", "k2", "fold64"),
                                      ("showdowns, today -> K3b -> Q3", "sd", "k3b", "sd64")):
            print(f"    {label}: float32 preview mean {e[old].mean():.3e} -> {e[new].mean():.3e} -> {e[wide].mean():.3e}, "
                  f"max {e[old].max():.3e} -> {e[new].max():.3e} -> {e[wide].max():.3e}")
    print(f"K2 and K3b in float64 against the dense matrices: worst {bad64:.2e} of max(1, |want|) (rule 5: 1e-12)")
    print(f"K4: {alive_breaks} showdown reach entries nonzero on a dead hand (the invariant needs 0)")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
    return 0 if bad64 <= 1e-12 and alive_breaks == 0 else 1


def cmd_selftest(a) -> int:
    """The harness's own logic, no GPU: rule 3's verdict, the ABBA order, the
    tap's labels and skips, two copies of the package, and `Dense` against
    `Hands`' NumPy matrices."""
    v = verdict([1.05] * 12, [1.0] * 12)
    assert v["faster"] and v["not_slower"] and v["new_wins"] == 12
    v = verdict([1.05] * 9 + [0.99] * 3, [1.0] * 12)  # 9 of 12: not enough for faster
    assert not v["faster"] and v["not_slower"]
    v = verdict([1.0] * 12, [1.0] * 12)
    assert not v["faster"] and v["not_slower"]
    v = verdict([0.995] * 10 + [1.0] * 2, [1.0] * 12)  # inside 1%, but the old code ahead in 10 rounds
    assert not v["not_slower"]
    assert not verdict([0.97] * 12, [1.0] * 12)["not_slower"]
    order = []
    abba({"old": lambda: order.append("o") or 1.0, "new": lambda: order.append("n") or 1.0}, 4)
    assert "".join(order) == "onnoonno"

    class Kernel:
        def __init__(self, log, name):
            self.log, self.name = log, name

        def __getitem__(self, grid):
            return lambda *x, **kw: self.log.append(self.name)

    log: list = []
    fake = SimpleNamespace(torch=torch, **{k: Kernel(log, k[1:]) for k in Tap.KERNELS})

    def step(levels=3):  # the launches of `FusedSolver._step_device`
        fake._prep[(1,)]()
        for _ in (0, 1):
            for _ in range(levels):
                fake._down[(1, 1)]()
            fake.torch.matmul(torch.ones(2, 2), torch.ones(2, 2))
            fake._fold[(1, 1)]()
            fake._showdown[(1,)]()
            for _ in range(levels):
                fake._up[(1, 1)]()

    tap = Tap(fake)
    step()
    assert tap.labels == (["prep"] + [f"down.p0.{k}" for k in range(3)] + ["matmul.p0.0", "fold.p0.0", "showdown.p0.0"]
                          + [f"up.p0.{k}" for k in range(3)] + [f"down.p1.{k}" for k in range(3)]
                          + ["matmul.p1.0", "fold.p1.0", "showdown.p1.0"] + [f"up.p1.{k}" for k in range(3)])
    assert log == ["prep"] + (["down"] * 3 + ["fold", "showdown"] + ["up"] * 3) * 2
    for skip, left in ((("down",), 11), (("down.p1",), 14), (("up.p0.1",), 16), (("fold", "showdown", "matmul"), 13)):
        log.clear()  # 17 kernel launches in a full step (the matmul is torch's own)
        tap.skip = skip
        step()
        assert len(log) == left, (skip, log)
    log.clear()  # Q4's terminals: their own labels, and `fold` does not skip `fold_ff`
    tap.skip = ("fold", "showdown")
    fake._prep[(1,)]()
    for _ in (0, 1):
        fake._down[(1, 1)]()
        fake._fold_ff[(1, 1)]()
        fake._showdown_ff[(1,)]()
        fake._up[(1, 1)]()
    assert tap.labels == ["prep"] + [f"{k}.p{t}.0" for t in (0, 1) for k in ("down", "fold_ff", "showdown_ff", "up")]
    assert log == ["prep"] + ["down", "fold_ff", "showdown_ff", "up"] * 2, log
    tap.close()
    assert fake.torch is torch and isinstance(fake._down, Kernel) and isinstance(fake._fold_ff, Kernel)
    old = SimpleNamespace(torch=torch, **{k: Kernel(log, k[1:]) for k in Tap.KERNELS if "64" not in k and "ff" not in k})
    Tap(old).close()  # a package without the terminals64 kernels: nothing added, everything restored
    assert not any(hasattr(old, k) for k in ("_fold64", "_fold_ff")) and old.torch is torch
    one, two = package(None, "parciver_a"), package(None, "parciver_b")
    assert one.gf is not two.gf and one.gf.FusedSolver is not two.gf.FusedSolver and takes(one, "gamma")
    net = one.vn.ValueNet(16, 2).eval()  # leaf names: N1, N1b, N1c; never N1 under another name
    for name, last, eps in (("FactoredLeaves", torch.float32, False), ("FactoredLeaves:float64", torch.float64, False),
                            ("FactoredLeaves:float64+eps", torch.float64, True)):
        leaves = leaf_factory(one, name, (3, 22, 41), net, "cpu")
        assert (leaves.ld, leaves.last_eps) == (last, eps), name
    for name in ("FactoredLeaves:float32", "FactoredLeaves:+eps", "FactoredLeaves:float32+eps", "FactoredLeaves:float64+x"):
        try:
            leaf_factory(one, name, (3, 22, 41), net, "cpu")
        except SystemExit:
            continue
        raise AssertionError(f"{name} was accepted")
    try:
        import parciver_engine  # noqa: F401
    except ImportError:
        print("selftest: no parciver_engine, Dense not checked")
    else:
        gs = one.gs
        rng = np.random.default_rng(0)
        for cards in (5, 4):
            hands = gs.Hands(gs.board(3)[:cards])
            tree = gs.Tree((gs.spec if cards == 5 else gs.small_turn_spec)(), hands)
            plan = one.gf.host_plan(hands, tree, [gs.range_(hands.n, 1), gs.range_(hands.n, 2)])
            pick = rng.choice(len(plan.term_nodes), size=min(40, len(plan.term_nodes)), replace=False)
            rq = np.zeros((len(plan.term_nodes), hands.n))
            for j in pick:
                rq[j] = rng.random(hands.n) * (rng.random(hands.n) < 0.7) * hands.alive[tree.nodes[plan.term_nodes[j]].view]
            dense = Dense(hands, plan, device="cpu")
            for p in (0, 1):
                got = dense(p, torch.as_tensor(rq)).numpy()
                scale = dense.scale(p, torch.as_tensor(rq)).numpy()
                for j in pick:
                    x = tree.nodes[plan.term_nodes[j]]
                    if x.folded is not None:
                        amount = -float(x.stake[p]) if x.folded == p else float(x.stake[1 - p])
                        want = (hands.compat @ rq[j]) * amount
                    else:
                        amount = float(min(x.stake))
                        want = (hands.showdown_matrix(x.view) @ rq[j]) * amount
                    assert np.abs(got[j] - want).max() <= 1e-12 * max(1.0, np.abs(want).max()), (cards, j)
                    assert math.isclose(scale[j], rq[j].sum() * abs(amount), rel_tol=1e-12)
    print("selftest: ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", default=JOBS)
    ap.add_argument("--starts", default=STARTS)
    ap.add_argument("--spots", default=SPOTS)
    ap.add_argument("--net", default=NET)
    ap.add_argument("--json", help="write the numbers here")
    ap.add_argument("--new", help="identity, precision, speed: the candidate's src (this checkout's if omitted)")
    ap.add_argument("--terminals64", action="store_true",
                    help="precision, speed: both sides' float32 solvers with terminals64 (KERNELS Q4); "
                         "profile: its float32 solvers (P1, P2, P4)")
    ap.add_argument("--busy-ok", action="store_true", help="run on a busy GPU (a dry run: its numbers do not count)")
    ap.add_argument("--max-busy", type=float, default=5.0,
                    help="the utilisation %% other programs may hold before and after a run (KERNELS rule 3 amendment, 4 Oct)")
    ap.add_argument("--leaves", default="dense", help="leaf evaluators, comma-separated: dense, FactoredLeaves, FactoredLeaves:float64, FactoredLeaves:float64+eps")
    ap.add_argument("--precision-parts", default="terminal,leaf,solve")
    ap.add_argument("--device", default="cuda", help="cpu: the leaf precision part alone, as a preview")
    ap.add_argument("--spots-n", type=int, default=0, help="the first N flop spots only")
    sub = ap.add_subparsers(dest="cmd", required=True)  # no abbreviations: `speed --new X` would be --new-leaves
    p = sub.add_parser("profile", allow_abbrev=False)
    p.add_argument("--parts", default="p1,p2,p3,p4")
    p.add_argument("--levels", action="store_true", help="P2 per tree level as well")
    i = sub.add_parser("identity", allow_abbrev=False)
    i.add_argument("--old", required=True)
    i.add_argument("--fields", default="regret,sum,val,rq", help="K1 leaves aliased rq rows unwritten: regret,sum,val")
    i.add_argument("--skip", default="", help="flop: no flop-solver check")
    c = sub.add_parser("precision", allow_abbrev=False)
    c.add_argument("--old")
    s = sub.add_parser("speed", allow_abbrev=False)
    s.add_argument("--old")
    s.add_argument("--measures", default="live,iter,flop")
    s.add_argument("--rounds", type=int, default=12)
    s.add_argument("--iters", type=int, default=100, help="iterations per round on a bench tree")
    s.add_argument("--flop-iters", type=int, default=50, help="iterations per spot and round")
    s.add_argument("--old-leaves", default="dense")
    s.add_argument("--new-leaves", default="dense")
    d = sub.add_parser("designs", allow_abbrev=False)
    d.add_argument("--first", type=int, default=0, help="the first N recorded jobs only")
    sub.add_parser("selftest", allow_abbrev=False)
    a = ap.parse_args()
    if a.terminals64:
        if a.cmd not in ("precision", "speed", "profile"):
            raise SystemExit("--terminals64 is for precision, speed and profile")
        T64["terminals64"] = True
    return dict(profile=cmd_profile, identity=cmd_identity, precision=cmd_precision, speed=cmd_speed,
                designs=cmd_designs, selftest=cmd_selftest)[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
