# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""LIVESEARCH B1 on the live bot (LIVESEARCH.md, Amendment 3): a more precise turn search.

  stage1  the turn schedule (Stage 1): the 8 registered arms on recs40's 40 real turn starts and a random-range
          twin each (rng 77, as plus_curve.py), float32 with Q4's terminals, exploitability (% of the pot) at
          25-500 iterations; the rule at 100 and 400 iterations; if A*(400) is not dcfr_g2, the guard on
          starts300's 300 fresh starts.  -> records/livesearch/stage1.json (B1's schedule)
  smoke   the pipeline check before P2: 50 deals of seed 777,003 at 1 and at 4 threads, which must agree deal for
          deal, each chunk's arms proven.  -> runs/livesearch/smoke.json (scratch)
  p2      P2: B1 against B0, M2's protocol (the 200bb scorecard's baseline deals, mirrored, AIVAT reported), in
          chunks of 100 deals, resumed on rerun; B0's turn jobs in deals 0-299 kept for P1.
          -> runs/livesearch/p2-state.json and p1-jobs-<first>.pkl; at the end records/livesearch/p2.json
  p1      P1: each kept job solved from scratch as B0 and as B1, its exploitability (% of the pot), and a
          board-clustered bootstrap of the difference.  -> records/livesearch/p1.json

B0 is the live bot as scripts/slumbot_run.py configures it, with turn 100 iterations and the default schedule
(the live bot before B1); B1 the same with turn 400 iterations and Stage 1's schedule
(`SubgameGpu(turn_schedule=...)`). Each arm has its own solvers. Run from main, the play code committed:
    .venv/Scripts/python.exe -I scripts/livesearch.py stage1
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REC = ROOT / "records" / "livesearch"
RUNS = ROOT / "runs" / "livesearch"
RUNNER = ROOT / "scripts" / "slumbot_run.py"
BENCHMARK = ROOT / "runs" / "scorecard" / "scorecard-v2-200bb" / "benchmark.json"
RECS40 = Path("C:/tmp/d1/recs40.pkl")  # Stage 1's 40 recorded real turn starts (D3's)
RECS40_SHA256 = "e310f5f212966aa2166bf144c72cfc7eae041eecd89662d3eafaefdb7325b06f"
STARTS300 = ROOT / "records" / "q4-gpu" / "starts300.pkl"  # KERNELS Q4's 300 fresh turn starts (git-ignored)
STARTS300_SHA256 = "6109cb69c6a187eb367afbc2af0a4d99150b5cf20f9193d3f8b43a3de3a69a86"
KERNELS_ROWS = ROOT / "records" / "q4-gpu" / "retest-rows.jsonl"  # their float64-evaluated Q4 rows
# What decides a record's numbers: a resumed P2 refuses a change in any of these.
PLAY_CODE = ("src", "engine", "scripts/livesearch.py", "scripts/slumbot_run.py")

# Stage 1: plus_curve.py's arms (FusedSolver's schedule options), its checkpoints with 400 added (Amendment 3).
ARMS = {"dcfr_g2": dict(gamma=2), "dcfr_g5": dict(gamma=5), "shift1_g5": dict(gamma=5, shift=1),
        "plus_g2": dict(gamma=2, plus=True), "plus_g4": dict(gamma=4, plus=True), "plus_g5": dict(gamma=5, plus=True),
        "postflop_solver": dict(gamma=3, shift=2, restart=True), "pfs_g5": dict(gamma=5, shift=2, restart=True)}
ORDER = ("dcfr_g5", "shift1_g5", "plus_g5", "plus_g4", "plus_g2", "pfs_g5", "postflop_solver")  # least new code first
CHECK = (25, 50, 100, 150, 200, 300, 400, 500)
BUDGETS = (100, 400)
MARGIN, TIE = 0.05, 0.03  # ties: within 3% of the lowest; the arm chosen must be >= 5% below dcfr_g2
TWIN_SEED, STACK, BB = 77, 20000, 100
GUARD_ITERS = 400

# Stage 2.
B0_TURN, B1_TURN = 100, 400
DEALS, CHUNK, THREADS, BLOCKS = 10_000, 100, 4, 50
P1_DEALS = 300  # B0's turn jobs in deals 0..299 (P2's first three chunks)
SMOKE_SEED, SMOKE_DEALS = 777_003, 50
BOOT, BOOT_SEED = 10_000, 0


# ---------------------------------------------------------------- provenance


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def need_sha(path: Path, want: str) -> str:
    if not path.exists():
        raise SystemExit(f"{path} is missing")
    got = sha256(path)
    if got != want:
        raise SystemExit(f"{path}: sha256 {got}, registered {want}")
    return got


def play_code() -> dict:
    """The play code's git object ids at HEAD; refused if any of it is changed and not committed (tracked files,
    or untracked .py/.rs/.toml files)."""
    out = git("status", "--porcelain", "-z", "--untracked-files=all", "--", *PLAY_CODE)
    dirty = [x for x in out.split("\0") if x and (not x.startswith("??") or x.endswith((".py", ".rs", ".toml")))]
    if dirty:
        raise SystemExit(f"uncommitted play code: {dirty[:5]}: commit it first (every record holds the commit)")
    return {p: git("rev-parse", f"HEAD:{p}").strip() for p in PLAY_CODE}


def provenance() -> dict:
    import torch
    import triton

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest":
        raise SystemExit("TF32 is allowed in this process: not the live setting")
    return dict(commit=git("rev-parse", "HEAD").strip(), play_code=play_code(), torch=torch.__version__,
                triton=triton.__version__, python=sys.version.split()[0], device=torch.cuda.get_device_name())


def gpu_state() -> dict:
    """nvidia-smi's view: utilisation (%), memory used (MiB), the compute apps; `peers`: those that are this
    Python, other than this process (a compute job beside the run; kernel_bench.gpu_peers)."""
    def smi(*q):
        try:
            return subprocess.run(["nvidia-smi", *q], capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.TimeoutExpired):
            return ""

    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import kernel_bench as kb

    util = mem = None
    line = smi("--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits").strip()
    try:
        util, mem = (float(x) for x in line.splitlines()[0].split(","))
    except (IndexError, ValueError):  # no answer, or one this does not parse: recorded as unknown
        pass
    apps = [x.strip() for x in smi("--query-compute-apps=pid,process_name", "--format=csv,noheader").splitlines()
            if x.strip()]
    return dict(utc=utc(), util=util, mem_mib=mem, apps=apps, peers=kb.gpu_peers())


def busy(g: dict) -> bool:
    """Another process of this Python held the GPU, or nvidia-smi could not say: it did not run (peers None) or
    gave no utilisation (it ran and failed, and then lists no processes)."""
    return g["peers"] is None or bool(g["peers"]) or g["util"] is None


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save(path: Path, obj, indent=None) -> None:
    from parciver.solver_watch import write_atomic

    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps(obj, indent=indent))


def fresh(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists: a record is written once (a VOID run's is moved aside as *-void-<n>)")


# ---------------------------------------------------------------- Stage 1


def situations(recs, tv, gs) -> list:
    """plus_curve.py's situations, in its order: each recorded start's round root, then its twin with random ranges
    (rng 77: player 0's range, then player 1's)."""
    rng = np.random.default_rng(TWIN_SEED)
    out = []
    for r in recs:
        s = tv.root_of(dict(r["job"], board=r["board"], ranges=r["ranges"]), STACK, BB)
        h = gs.Hands(list(s.board))
        tree = gs.Tree(s.spec(), h)
        order = tv.strength_order(h)
        twin = tv.Situation(s.board, s.half, s.stack, s.first, s.bb,
                            np.stack([tv.random_range(order, rng), tv.random_range(order, rng)]))
        out += [("real", s, h, tree), ("random", twin, h, tree)]
    return out


def root_key(s) -> tuple:
    return (tuple(int(c) for c in s.board), int(s.half), int(s.first),
            tuple(np.asarray(r, dtype=np.float64).tobytes() for r in s.ranges))


def curve(gf, hands, tree, ranges, pot: float, kw: dict, check=CHECK, plan=None) -> list[float]:
    """Exploitability (% of `pot`) of a float32 solve with Q4's terminals and schedule `kw` at each checkpoint."""
    import torch

    s = gf.FusedSolver(hands, tree, list(ranges), dtype=torch.float32, plan=plan, terminals64=True, **kw)
    if not s.ff:
        raise SystemExit("the solver is not running Q4's terminals")
    out = []
    for it in range(1, max(check) + 1):
        s.step()
        if it in check:
            out.append(100.0 * s.exploitability() / pot)
    del s
    return out


def tables(rows: list) -> dict:
    """Mean and median exploitability per arm and checkpoint over the real starts as registered (40, the 9
    re-solves counted twice), the distinct real roots, the random twins, and all."""
    subsets = {"real": [r for r in rows if r["kind"] == "real"],
               "real_distinct": [r for r in rows if r["kind"] == "real" and r["same_as"] == r["start"]],
               "random": [r for r in rows if r["kind"] == "random"], "all": rows}
    return {name: dict(n=len(sub), **{nm: dict(mean=[float(np.mean([r["expl"][nm][j] for r in sub])) for j in range(len(CHECK))],
                                               median=[float(np.median([r["expl"][nm][j] for r in sub])) for j in range(len(CHECK))])
                                      for nm in ARMS})
            for name, sub in subsets.items()}


def pick(table: dict, budget: int) -> dict:
    """The rule at one budget, on the real starts' means, read as written (Amendment 3): among the arms within 3%
    of the lowest mean, the first in ORDER is chosen; it replaces dcfr_g2 only if its mean is at least 5% below
    dcfr_g2's; else dcfr_g2 stays."""
    j = CHECK.index(budget)
    m = {nm: table[nm]["mean"][j] for nm in ARMS}
    low = min(m.values())
    tied = [nm for nm in ORDER if m[nm] <= (1 + TIE) * low]
    first = tied[0] if tied else None
    ok = first is not None and m[first] <= (1 - MARGIN) * m["dcfr_g2"]
    return dict(budget=budget, means=m, lowest=min(m, key=m.get), tied=tied, first=first,
                pick=first if ok else "dcfr_g2")


def resolved(schedule: dict | None) -> tuple:
    """(gamma, plus, shift, restart) as FusedSolver resolves `schedule` (shift's default: 1 with plus, else 0)."""
    kw = schedule or {}
    plus = bool(kw.get("plus", False))
    shift = kw.get("shift")
    return int(kw.get("gamma", 2)), plus, (1 if plus else 0) if shift is None else int(shift), bool(kw.get("restart", False))


def run_key(street: int, sched: tuple) -> str:
    g, plus, shift, restart = sched
    return f"{street}:gamma{g}:plus{int(plus)}:shift{shift}:restart{int(restart)}"


def pot_of(job: dict) -> float:
    """KERNELS Q4's pot: twice `half`, half of the smallest terminal stake sum (the check-down's, the pot at the
    round's start)."""
    kind = np.asarray(job["kind"])
    stake = np.asarray(job["stake"], dtype=np.int64).reshape(-1, 2)
    return 2.0 * (int(stake[kind == 0].sum(1).min()) // 2)


def guard(gf, gs, star: str) -> dict:
    """A*(400) against dcfr_g2 on starts300's 300 fresh starts, 400 iterations each, the recorded jobs as KERNELS
    solved them (no frozen slots); dcfr_g2's rows beside KERNELS' float64-evaluated Q4 rows (descriptive)."""
    need_sha(STARTS300, STARTS300_SHA256)
    starts = pickle.load(open(STARTS300, "rb"))
    if len(starts) != 300:
        raise SystemExit(f"{len(starts)} starts in {STARTS300}, registered 300")
    kernels = {}
    for line in KERNELS_ROWS.read_text().splitlines():
        row = json.loads(line)
        kernels[row["i"]] = row["q4@400"]
    rows, t0 = [], time.time()
    for i, r in enumerate(starts):
        job = dict(r["job"], board=r["board"], strength=r["strength"], alive=r["alive"], ranges=r["ranges"],
                   frozen=[], want=[])
        hands = gs.Hands.from_arrays(job["board"], job["strength"], job["alive"])
        plan = gf.host_plan(hands, job, job["ranges"])
        pot = 2.0 * r["half"]
        e = {nm: curve(gf, hands, job, job["ranges"], pot, ARMS[nm], check=(GUARD_ITERS,), plan=plan)[0]
             for nm in (star, "dcfr_g2")}
        rows.append(dict(i=i, board=[int(c) for c in r["board"]], pot=pot, expl=e, kernels_q4=kernels.get(i)))
        if (i + 1) % 25 == 0:
            print(f"guard: {i + 1}/300 starts, {time.time() - t0:.0f} s", flush=True)
    mean = {nm: float(np.mean([x["expl"][nm] for x in rows])) for nm in (star, "dcfr_g2")}
    diff = [abs(x["expl"]["dcfr_g2"] - x["kernels_q4"]) for x in rows if x["kernels_q4"] is not None]
    return dict(arm=star, iters=GUARD_ITERS, starts=len(rows), starts_sha256=STARTS300_SHA256, mean=mean,
                passed=mean[star] < mean["dcfr_g2"], dcfr_g2_vs_kernels_max_abs=max(diff) if diff else None, rows=rows)


def cmd_stage1(a) -> int:
    from parciver import gpu_fused as gf
    from parciver import gpu_search as gs
    from parciver import turn_values as tv

    out = REC / "stage1.json"
    fresh(out)
    prov = provenance()
    need_sha(RECS40, RECS40_SHA256)
    need_sha(STARTS300, STARTS300_SHA256)  # the guard's, checked before ~10 minutes of solves
    if not KERNELS_ROWS.exists():
        raise SystemExit(f"{KERNELS_ROWS} is missing")
    gpu0 = gpu_state()
    if busy(gpu0):
        raise SystemExit(f"the GPU's other users: {gpu0['peers']} (None: unknown): Stage 1 needs it alone")
    recs = pickle.load(open(RECS40, "rb"))
    if len(recs) != 40:
        raise SystemExit(f"{len(recs)} starts in {RECS40}, registered 40")
    sits = situations(recs, tv, gs)
    seen, rows, t0 = {}, [], time.time()
    for i, (kind, s, h, tree) in enumerate(sits):
        plan = gf.host_plan(h, tree, list(s.ranges))
        row = dict(i=i, kind=kind, start=i // 2, board=[int(c) for c in s.board], pot=int(s.pot))
        if kind == "real":
            row["same_as"] = seen.setdefault(root_key(s), i // 2)  # the first start with this root
        row["expl"] = {nm: curve(gf, h, tree, s.ranges, s.pot, kw, plan=plan) for nm, kw in ARMS.items()}
        rows.append(row)
        print(f"{i + 1}/{len(sits)} {kind}, {time.time() - t0:.0f} s: "
              + " ".join(f"{nm} {row['expl'][nm][CHECK.index(400)]:.3f}" for nm in ARMS), flush=True)
    tab = tables(rows)
    rule = {str(n): pick(tab["real"], n) for n in BUDGETS}
    star = rule["400"]["pick"]
    g = guard(gf, gs, star) if star != "dcfr_g2" else None
    b1 = star if g is None or g["passed"] else "dcfr_g2"
    gpu1 = gpu_state()
    result = dict(registered="LIVESEARCH.md Stage 1 as amended (Amendment 3)", **prov, recs40=str(RECS40),
                  recs40_sha256=RECS40_SHA256, arms=ARMS, order=ORDER, check=CHECK, margin=MARGIN, tie=TIE,
                  twin_seed=TWIN_SEED, terminals="Q4 (float32, terminals64)", tables=tab, rule=rule, guard=g,
                  b1=dict(arm=b1, schedule=ARMS[b1]), b2_note=None if rule["100"]["pick"] == "dcfr_g2" else
                  f"A*(100) is {rule['100']['pick']}: B2 goes to IDEAS", rows=rows, seconds=time.time() - t0,
                  gpu_start=gpu0, gpu_end=gpu1, void=busy(gpu1))
    save(out, result, indent=1)
    for n in BUDGETS:
        r = rule[str(n)]
        print(f"A*({n}) = {r['pick']} (lowest {r['lowest']}; within 3% {r['tied']}, first {r['first']}); real means "
              + " ".join(f"{nm} {v:.4f}" for nm, v in r["means"].items()))
    if g is not None:
        print(f"guard on starts300 at {GUARD_ITERS}: {star} {g['mean'][star]:.4f}% vs dcfr_g2 "
              f"{g['mean']['dcfr_g2']:.4f}%: {'passed' if g['passed'] else 'FAILED, B1 uses dcfr_g2'}")
    print(f"B1's turn schedule: {b1} {ARMS[b1]}" + ("  (VOID: the GPU was not ours alone at the end)" if result["void"] else ""))
    return 2 if result["void"] else 0


def b1_settings() -> tuple[str, dict, str]:
    path = REC / "stage1.json"
    if not path.exists():
        raise SystemExit(f"no {path}: Stage 1 comes first")
    st = json.loads(path.read_text(encoding="utf-8"))
    if st.get("void"):
        raise SystemExit(f"{path} is VOID")
    arm, schedule = st["b1"]["arm"], st["b1"]["schedule"]
    if ARMS.get(arm) != schedule:
        raise SystemExit(f"{path}: B1's schedule {schedule} is not arm {arm}'s")
    return arm, schedule, sha256(path)


# ---------------------------------------------------------------- Stage 2: the arms


def runner():
    """scripts/slumbot_run.py as a module: the live bot's configuration."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("slumbot_run", RUNNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Tap:
    """An arm's solver as the engine calls it: counts each job by street and iterations, and while `keep` is set
    keeps a copy of each turn job (P1's record). Thread-safe."""

    def __init__(self, solver):
        self.solver = solver
        self.keep = False
        self.seen: Counter = Counter()
        self.jobs: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, job: dict):
        street = len(job["board"])
        kept = copy.deepcopy(dict(job)) if self.keep and street == 4 else None
        with self._lock:
            self.seen[f"{street}:{int(job['iters'])}"] += 1
            if kept is not None:
                self.jobs.append(kept)
        return self.solver(job)

    def take(self) -> tuple[dict, list]:
        with self._lock:
            got = dict(self.seen), self.jobs
            self.seen, self.jobs = Counter(), []
        return got


class Arm:
    """B0 or B1: the live runner's bot (its blueprint, translation, search and flop solver) with its own solvers,
    turn `turn_iters` iterations and `schedule` on the turn; `net`: another leaf net than the runner's (D2b)."""

    def __init__(self, name: str, sr, bp, turn_iters: int, schedule: dict | None, threads: int, net=None):
        from parciver import value_net as vn
        from parciver.flop_search import FlopSearch
        from parciver.gpu_fused import SubgameGpu
        from parciver.solver_watch import Watched

        self.name, self.schedule = name, schedule
        self.search_kw = dict(sr.SEARCH, turn_iters=turn_iters)
        self.iters = {3: self.search_kw["flop_iters"], 4: turn_iters, 5: self.search_kw["river_iters"]}
        net, _ = vn.load(sr.NET if net is None else net)
        self.turn = SubgameGpu(terminals64=sr.TERMINALS64, turn_schedule=schedule)
        # The table cache changes no value (M2's): room for every thread's flop.
        self.search = FlopSearch(net, gamma=sr.FLOP_GAMMA, turn=self.turn, leaves=sr.LEAVES, cache=max(4, 2 * threads))
        self.tap = Tap(self.search)
        self.watch = Watched(self.tap)
        base = bp.play_agent()
        if base.translation != "snap":
            raise SystemExit(f"the play agent's translation is {base.translation!r}, not the live 'snap'")
        self.agent = base.with_search(solver=self.watch, **self.search_kw)
        self.mark()

    def mark(self) -> None:
        """A chunk starts: its counts and times are those after this."""
        self.watch.take()
        for v in self.watch.seconds.values():
            v.clear()
        self.tap.take()
        self.at = dict(flop=len(self.search.seconds), turn=len(self.turn.times.get(4, [])),
                       river=len(self.turn.times.get(5, [])), schedules=dict(self.turn.schedules),
                       symmetric=self.search.symmetric)

    def chunk(self) -> tuple[dict, dict, list]:
        """(counts, in-lock seconds by street, kept jobs) since `mark`."""
        calls, failures = self.watch.take()
        seen, jobs = self.tap.take()
        times = dict(flop=self.search.seconds[self.at["flop"]:], turn=self.turn.times.get(4, [])[self.at["turn"]:],
                     river=self.turn.times.get(5, [])[self.at["river"]:])
        ran = {run_key(k[0], k[1:]): n - self.at["schedules"].get(k, 0) for k, n in sorted(self.turn.schedules.items())}
        counts = dict(calls=calls, failures=failures, last_error=self.watch.last_error if failures else None,
                      seen=dict(sorted(seen.items())), **{k: len(v) for k, v in times.items()},
                      ran={k: n for k, n in ran.items() if n}, symmetric=self.search.symmetric - self.at["symmetric"])
        return counts, times, jobs


def prove(name: str, c: dict, iters: dict, schedule: dict | None) -> list[str]:
    """A chunk's proof of its arm (Amendment 3), as a list of what is wrong: every job at the arm's iterations on
    its street (as the engine sent it); every turn solve run by the solver with the arm's schedule (`schedule`;
    None: the default) and every river solve with the default, as the solver objects themselves report it; the
    solvers' solves equal to the jobs; at least one flop and one turn solve; no failed solve."""
    bad, by = [], Counter()
    for k, n in c["seen"].items():
        street, it = (int(x) for x in k.split(":"))
        by[street] += n
        if iters.get(street) != it:
            bad.append(f"{name}: {n} jobs on a {street}-card board at {it} iterations, not {iters.get(street)}")
    for street, key in ((3, "flop"), (4, "turn"), (5, "river")):
        if c[key] != by[street]:
            bad.append(f"{name}: {c[key]} {key} solves for {by[street]} {key} jobs")
    if sum(by.values()) != c["calls"]:
        bad.append(f"{name}: {c['calls']} calls for {sum(by.values())} jobs")
    if c["flop"] < 1 or c["turn"] < 1:
        bad.append(f"{name}: {c['flop']} flop and {c['turn']} turn solves (at least one each)")
    want = {4: run_key(4, resolved(schedule)), 5: run_key(5, resolved(None))}
    ran = Counter()
    for k, n in c["ran"].items():
        street = int(k.split(":")[0])
        ran[street] += n
        if k != want.get(street):
            bad.append(f"{name}: {n} solves ran {k}, not {want.get(street, 'in the turn solver')}")
    for street, key in ((4, "turn"), (5, "river")):
        if ran[street] != c[key]:
            bad.append(f"{name}: {ran[street]} of {c[key]} {key} solves have a recorded schedule")
    if c["failures"]:
        bad.append(f"{name}: {c['failures']} of {c['calls']} search solves failed (last: {c['last_error']})")
    return bad


def build_arms(threads: int, schedule: dict):
    import parciver_engine as pe

    sr = runner()
    bp = pe.SavedBlueprint(str(sr.ABSTRACTION), str(sr.BLUEPRINT))
    b0 = Arm("B0", sr, bp, B0_TURN, None, threads)
    b1 = Arm("B1", sr, bp, B1_TURN, schedule, threads)
    return sr, pe, b0, b1


def play(b0: Arm, b1: Arm, stakes, seed: int, first: int, deals: int, threads: int) -> tuple[dict, dict]:
    """Deals first..first+deals-1, B1 (the engine's A) against B0, AIVAT on; refused (SystemExit, nothing kept) if
    a search failed or an arm is not proven."""
    import torch

    b0.mark()
    b1.mark()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    res = dict(b1.agent.match_against("agent", deals, *stakes, other=b0.agent, seed=seed, threads=threads,
                                      first_deal=first, aivat=True))
    dt = time.time() - t
    peak = torch.cuda.max_memory_allocated() / 2**20
    (c1, t1, _), (c0, t0, jobs) = b1.chunk(), b0.chunk()
    bad = prove("B1", c1, b1.iters, b1.schedule) + prove("B0", c0, b0.iters, b0.schedule)
    if any(res["a_search_failed"]) or any(res["b_search_failed"]):
        bad.append(f"decisions reached a failed search (B1 {res['a_search_failed']}, B0 {res['b_search_failed']})")
    for key in ("per_deal_bb", "per_deal_bb_aivat"):
        if len(res.get(key, ())) != deals:
            bad.append(f"the engine returned {len(res.get(key, ()))} {key} values for {deals} deals")
    if bad:
        raise SystemExit(f"deals {first}..{first + deals - 1} of seed {seed}: " + "; ".join(bad)
                         + ". Nothing of this chunk was saved.")
    return res, dict(seconds=dt, b0=c0, b1=c1, times=dict(b0=t0, b1=t1), jobs=jobs, peak_mib=peak)


def stakes_of(m: dict) -> tuple[int, int, int]:
    st = m["stakes"]
    return st["sb"], st["bb"], st["stack"]


def cmd_smoke(a) -> int:
    import gc

    from parciver import scorecard as sc

    prov = provenance()
    arm, schedule, stage1 = b1_settings()
    m = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    runs = {}
    for threads in (1, THREADS):
        # Fresh arms for each run: an agent's search cache (keyed by the public state and the hand's seed, one per
        # deal and seat, so it is reused only within a hand) would answer the replayed deals' searches from the
        # first run's solves, with no solve at all (the first smoke run, 15 Oct: its 4-thread run's proof refused
        # it).
        _, pe, b0, b1 = build_arms(THREADS, schedule)
        res, info = play(b0, b1, stakes_of(m), SMOKE_SEED, 0, SMOKE_DEALS, threads)
        runs[str(threads)] = dict(per_deal_bb=list(res["per_deal_bb"]), per_deal_bb_aivat=list(res["per_deal_bb_aivat"]),
                                  seconds=info["seconds"], peak_mib=info["peak_mib"], b0=info["b0"], b1=info["b1"])
        print(f"smoke at {threads} thread(s): {info['seconds']:.0f} s, B1 {np.mean(res['per_deal_bb']):+.3f} bb/deal; "
              f"B0 solves {info['b0']['flop']}/{info['b0']['turn']}/{info['b0']['river']}, "
              f"B1 {info['b1']['flop']}/{info['b1']['turn']}/{info['b1']['river']} (flop/turn/river)", flush=True)
        del b0, b1, res, info
        gc.collect()
    one, many = runs["1"], runs[str(THREADS)]
    same = one["per_deal_bb"] == many["per_deal_bb"] and one["per_deal_bb_aivat"] == many["per_deal_bb_aivat"]
    # Descriptive, not the registered test: with no cache clear (50 deals insert far fewer than its 4,096 entries)
    # each arm makes the same solves at either thread count.
    solves = {k: [one[k][s] for s in ("flop", "turn", "river")] == [many[k][s] for s in ("flop", "turn", "river")]
              for k in ("b0", "b1")}
    save(RUNS / "smoke.json", dict(**prov, engine=sc.sha256_file(pe.parciver_engine.__file__), b1_arm=arm,
                                   b1_schedule=schedule, stage1_sha256=stage1, seed=SMOKE_SEED, deals=SMOKE_DEALS,
                                   threads=[1, THREADS], identical=same, same_solves=solves, runs=runs))
    print(f"smoke: 1 and {THREADS} threads {'identical deal for deal' if same else 'DIFFER'}; the same solves by "
          f"street: B0 {solves['b0']}, B1 {solves['b1']}")
    return 0 if same else 1


# ---------------------------------------------------------------- P2


def stats(x: list) -> dict | None:
    if not x:
        return None
    x = np.asarray(x)
    return dict(n=len(x), median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(x.max()))


def p2_config(threads: int, sr, pe, m: dict, row: dict, arm: str, schedule: dict, stage1: str, prov: dict) -> dict:
    """What a resumed P2 must match (Amendment 3): the deals, both arms' settings, the inputs' sha256, the engine
    build, the play code's git objects, torch and triton. The commit itself is not compared (a docs commit
    during the run changes it), only the play code."""
    from parciver import scorecard as sc

    return dict(benchmark=m["benchmark_id"], manifest_hash=m["manifest_hash"], row="baseline", seed=row["seed"],
                deals=DEALS, hands_per_deal=m["hands_per_deal"], chunk=CHUNK, threads=threads, blocks=BLOCKS,
                aivat=True, p1_deals=P1_DEALS, blueprint=str(sr.BLUEPRINT), blueprint_sha256=sc.sha256_file(sr.BLUEPRINT),
                net=str(sr.NET), net_sha256=sc.sha256_file(sr.NET), leaves=sr.LEAVES, flop_gamma=sr.FLOP_GAMMA,
                terminals64=sr.TERMINALS64, translation="snap",
                b0=dict(search=dict(sr.SEARCH, turn_iters=B0_TURN), turn_schedule=None),
                b1=dict(search=dict(sr.SEARCH, turn_iters=B1_TURN), turn_schedule=schedule, arm=arm),
                stage1_sha256=stage1, engine=sc.sha256_file(pe.parciver_engine.__file__), play_code=prov["play_code"],
                torch=prov["torch"], triton=prov["triton"])


def smoke_passed(prov: dict, stage1: str, engine: str) -> bool:
    """The smoke run showed 1 and 4 threads identical on this play code, engine build and Stage 1 record."""
    path = RUNS / "smoke.json"
    if not path.exists():
        return False
    s = json.loads(path.read_text(encoding="utf-8"))
    return (s.get("identical") is True and s.get("threads") == [1, THREADS] and s.get("play_code") == prov["play_code"]
            and s.get("engine") == engine and s.get("stage1_sha256") == stage1)


def resume(path: Path, cfg: dict) -> dict:
    """P2's saved state if its configuration is `cfg`, an empty one if none is saved; else refused."""
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if state and state.get("config") != cfg:
        old = state.get("config") or {}
        diff = sorted(k for k in set(cfg) | set(old) if cfg.get(k) != old.get(k))
        raise SystemExit(f"{path} holds another configuration (differs in {diff}): not resumed")
    state["config"] = cfg
    return state


def cmd_p2(a) -> int:
    import parciver_engine as pe
    from parciver import scorecard as sc
    from parciver.solver_watch import persist

    fresh(REC / "p2.json")
    prov = provenance()
    arm, schedule, stage1 = b1_settings()
    if a.threads not in (1, THREADS):
        raise SystemExit(f"P2 runs at {THREADS} threads, or at 1 if the smoke run's differ (Amendment 3)")
    path = RUNS / "p2-state.json"
    # A new P2 at 4 threads needs the smoke run; a resume is pinned to the same play code and engine below.
    if a.threads != 1 and not path.exists() and not smoke_passed(prov, stage1, sc.sha256_file(pe.parciver_engine.__file__)):
        raise SystemExit(f"no smoke run shows 1 and {THREADS} threads identical on this code and engine: run "
                         "`smoke`, or P2 at --threads 1 (Amendment 3)")
    m = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    row = next(r for r in m["rows"] if r["name"] == "baseline")
    sr = runner()
    if stakes_of(m) != (50, 100, 20000) or Path(m["abstraction"]["dir"]).resolve() != Path(sr.ABSTRACTION).resolve():
        raise SystemExit(f"{BENCHMARK} is not the 200bb benchmark on {sr.ABSTRACTION}")
    if row["deals"] < DEALS:
        raise SystemExit(f"the baseline row has {row['deals']} deals, P2 needs {DEALS}")
    cfg = p2_config(a.threads, sr, pe, m, row, arm, schedule, stage1, prov)
    state = resume(path, cfg)
    if "smoke_sha256" not in state:
        state["smoke_sha256"] = sha256(RUNS / "smoke.json") if a.threads != 1 else None
    state.setdefault("sessions", []).append(dict(started=utc(), commit=prov["commit"]))
    run = state.setdefault("run", dict(per_deal_bb=[], per_deal_bb_aivat=[], aivat_mass_error=0.0, a_sources=[0] * 4,
                                       b_sources=[0] * 4, seconds=0.0, chunks=[]))
    hpd = m["hands_per_deal"]
    if len(run["per_deal_bb"]) < DEALS:
        _, _, b0, b1 = build_arms(a.threads, schedule)
        while len(run["per_deal_bb"]) < DEALS:
            first = len(run["per_deal_bb"])
            keep = first < P1_DEALS
            b0.tap.keep = keep
            g0 = gpu_state()
            res, info = play(b0, b1, stakes_of(m), row["seed"], first, CHUNK, a.threads)
            b0.tap.keep = False
            g1 = gpu_state()
            jobs = info.pop("jobs")
            chunk = dict(first=first, deals=CHUNK, seconds=info["seconds"], ended=utc(), b0=info["b0"], b1=info["b1"],
                         peak_mib=info["peak_mib"], gpu_start=g0, gpu_end=g1, shared=busy(g0) or busy(g1),
                         times=info["times"])
            if keep:
                f = RUNS / f"p1-jobs-{first:05d}.pkl"
                tmp = f.with_suffix(".tmp")
                with open(tmp, "wb") as fh:
                    pickle.dump(jobs, fh, protocol=pickle.HIGHEST_PROTOCOL)
                persist(tmp, f)
                chunk["p1_jobs"] = dict(file=f.name, jobs=len(jobs), sha256=sha256(f))
            del jobs
            run["per_deal_bb"] += list(res["per_deal_bb"])
            run["per_deal_bb_aivat"] += list(res["per_deal_bb_aivat"])
            run["aivat_mass_error"] = max(run["aivat_mass_error"], float(res["aivat_mass_error"]))
            run["a_sources"] = [x + y for x, y in zip(run["a_sources"], res["a_sources"])]
            run["b_sources"] = [x + y for x, y in zip(run["b_sources"], res["b_sources"])]
            run["seconds"] += info["seconds"]
            run["chunks"].append(chunk)
            save(path, state)
            done = len(run["per_deal_bb"])
            mean, se = sc.mean_se(run["per_deal_bb"], hpd)
            print(f"p2: {done}/{DEALS} deals, B1 - B0 {mean:+.2f} ± {se:.2f} bb/100 so far, "
                  f"{run['seconds'] / done:.2f} s/deal, eta {run['seconds'] / done * (DEALS - done) / 3600:.1f} h"
                  + ("  (GPU shared or unknown: this chunk's latency is left out)" if chunk["shared"] else ""), flush=True)
    return p2_finish(state, DEALS, hpd)


def p2_finish(state: dict, deals: int, hpd: int) -> int:
    from parciver import scorecard as sc

    run = state["run"]
    s = sc.summarize(sc.to_blocks(run["per_deal_bb"][:deals], BLOCKS, hpd))
    s_aivat = sc.summarize(sc.to_blocks(run["per_deal_bb_aivat"][:deals], BLOCKS, hpd))
    down = s["hi"] < 0
    # Latency: chunks with the GPU shared (or unknown) and chunks that copied P1's jobs (a CPU cost beside the
    # solves) are left out.
    clean = [c for c in run["chunks"] if not c["shared"] and "p1_jobs" not in c]
    latency = {arm: {street: stats([t for c in clean for t in c["times"][arm][street]]) for street in ("flop", "turn", "river")}
               for arm in ("b0", "b1")}
    solves = {arm: {street: sum(c[arm][street] for c in run["chunks"]) / deals for street in ("flop", "turn", "river")}
              for arm in ("b0", "b1")}
    share = lambda src: src[3] / max(1, sum(src))  # [blueprint, missing_key, off_tree, search]
    summary = dict(s, down=down, aivat=s_aivat, aivat_mass_error=run["aivat_mass_error"], s_per_deal=run["seconds"] / deals,
                   latency_in_lock=latency, latency_chunks=len(clean), shared_chunks=[c["first"] for c in run["chunks"] if c["shared"]],
                   solves_per_deal=solves, b1_search_share=share(run["a_sources"]), b0_search_share=share(run["b_sources"]))
    state["summary"] = summary
    save(RUNS / "p2-state.json", state)
    fresh(REC / "p2.json")
    record = dict(registered="LIVESEARCH.md Amendment 3, P2", config=state["config"], sessions=state["sessions"],
                  summary=summary, chunks=[{k: v for k, v in c.items() if k != "times"} for c in run["chunks"]],
                  per_deal_bb=run["per_deal_bb"], per_deal_bb_aivat=run["per_deal_bb_aivat"])
    save(REC / "p2.json", record)
    print(f"P2: B1 - B0 = {s['mean']:+.2f} bb/100, 95% [{s['lo']:+.2f}, {s['hi']:+.2f}] over {deals} deals: "
          f"{'DOWN' if down else 'not DOWN'}; AIVAT (reported) {s_aivat['mean']:+.2f} [{s_aivat['lo']:+.2f}, {s_aivat['hi']:+.2f}]")
    return 0


# ---------------------------------------------------------------- P1


def board_key(board) -> tuple:
    """A turn board as a set: the flop's cards sorted, then the turn card."""
    b = [int(c) for c in board]
    return (*sorted(b[:3]), b[3])


def job_digest(job: dict) -> str:
    """A content hash of a job (its order in P1 does not depend on the threads' timing)."""
    h = hashlib.sha256()
    for k in sorted(job):
        h.update(k.encode())
        v = job[k]
        if k == "frozen":
            for slot, st in v:
                h.update(np.int64(slot).tobytes())
                h.update(np.ascontiguousarray(np.asarray(st, dtype=np.float64)).tobytes())
        else:
            x = np.ascontiguousarray(np.asarray(v))
            h.update(str((x.dtype.str, x.shape)).encode())
            h.update(x.tobytes())
    return h.hexdigest()


def board_bootstrap(d, boards: list, n: int = BOOT, seed: int = BOOT_SEED) -> dict:
    """The mean of `d` and its 95% percentile interval over `n` resamples of the boards (each with all its jobs)."""
    d = np.asarray(d, dtype=np.float64)
    keys = sorted(set(boards))
    g = np.array([keys.index(b) for b in boards])
    sums = np.bincount(g, weights=d, minlength=len(keys))
    counts = np.bincount(g, minlength=len(keys)).astype(np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), size=(n, len(keys)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), jobs=len(d), boards=len(keys), resamples=n, seed=seed)


def solve_expl(gf, gs, job: dict, iters: int, schedule: dict | None) -> float:
    """A job solved from scratch (float32, Q4's terminals, `schedule`, its frozen slots held, every slot summed):
    its exploitability in its own subgame, % of the pot."""
    import torch

    hands = gs.Hands.from_arrays(job["board"], job["strength"], job["alive"])
    s = gf.FusedSolver(hands, job, job["ranges"], dtype=torch.float32, terminals64=True, **(schedule or {}))
    if not s.ff:
        raise SystemExit("the solver is not running Q4's terminals")
    for slot, strategy in job["frozen"]:
        s.freeze(slot, np.asarray(strategy))
    for _ in range(iters):
        s.step()
    e = s.exploitability()
    del s
    return 100.0 * e / pot_of(job)


def cmd_p1(a) -> int:
    from parciver import gpu_fused as gf
    from parciver import gpu_search as gs

    out = REC / "p1.json"
    fresh(out)
    prov = provenance()
    state = json.loads((RUNS / "p2-state.json").read_text(encoding="utf-8"))
    cfg = state["config"]
    # P1 solves as P2's arms did: the same solver code (src), torch and triton.
    same = {k: (cfg["play_code"]["src"], prov["play_code"]["src"]) if k == "src" else (cfg[k], prov[k])
            for k in ("src", "torch", "triton")}
    if any(x != y for x, y in same.values()):
        raise SystemExit(f"P1 must run P2's solver code, torch and triton: {same}")
    kept = [c for c in state["run"]["chunks"] if c["first"] < P1_DEALS]
    if [c["first"] for c in kept] != list(range(0, P1_DEALS, cfg["chunk"])) or not all("p1_jobs" in c for c in kept):
        raise SystemExit(f"P2's chunks of deals 0..{P1_DEALS - 1} are not all saved with their jobs")
    for c in kept:
        need_sha(RUNS / c["p1_jobs"]["file"], c["p1_jobs"]["sha256"])
    schedule = cfg["b1"]["turn_schedule"]
    gpu0 = gpu_state()
    if busy(gpu0):
        raise SystemExit(f"the GPU's other users: {gpu0['peers']} (None: unknown): P1 needs it alone")
    # One file in memory at a time; each job is solved from scratch, so the order of the solves changes no value.
    # The rows are then sorted by board and content hash: the statistic does not depend on the threads' timing.
    rows, t0, total = [], time.time(), sum(c["p1_jobs"]["jobs"] for c in kept)
    for c in kept:
        with open(RUNS / c["p1_jobs"]["file"], "rb") as fh:
            jobs = pickle.load(fh)
        if len(jobs) != c["p1_jobs"]["jobs"]:
            raise SystemExit(f"{c['p1_jobs']['file']}: {len(jobs)} jobs, its record says {c['p1_jobs']['jobs']}")
        if any(len(j["board"]) != 4 or int(j["iters"]) != B0_TURN for j in jobs):
            raise SystemExit(f"{c['p1_jobs']['file']}: jobs that are not B0's turn jobs")
        for job in jobs:
            e0 = solve_expl(gf, gs, job, B0_TURN, None)
            e1 = solve_expl(gf, gs, job, B1_TURN, schedule)
            rows.append(dict(board=list(board_key(job["board"])), digest=job_digest(job), pot=pot_of(job),
                             frozen=len(job["frozen"]), nodes=len(job["kind"]), b0=e0, b1=e1, d=e1 - e0))
            if len(rows) % 50 == 0:
                print(f"p1: {len(rows)}/{total} jobs, {time.time() - t0:.0f} s", flush=True)
        del jobs
    rows.sort(key=lambda r: (tuple(r["board"]), r["digest"]))
    boot = board_bootstrap([r["d"] for r in rows], [tuple(r["board"]) for r in rows])
    holds = boot["hi"] < 0
    gpu1 = gpu_state()
    result = dict(registered="LIVESEARCH.md Amendment 3, P1", **prov, p2_config=cfg,
                  jobs_files=[c["p1_jobs"] for c in kept], b0=dict(iters=B0_TURN, schedule=None),
                  b1=dict(iters=B1_TURN, schedule=schedule), bootstrap=boot, holds=holds,
                  b0_mean=float(np.mean([r["b0"] for r in rows])), b1_mean=float(np.mean([r["b1"] for r in rows])),
                  b0_median=float(np.median([r["b0"] for r in rows])), b1_median=float(np.median([r["b1"] for r in rows])),
                  b1_lower=int(sum(r["d"] < 0 for r in rows)), rows=rows, seconds=time.time() - t0, gpu_start=gpu0,
                  gpu_end=gpu1, void=busy(gpu1))
    save(out, result, indent=1)
    print(f"P1: B1 - B0 = {boot['mean']:+.4f}% of the pot, 95% [{boot['lo']:+.4f}, {boot['hi']:+.4f}] over "
          f"{boot['jobs']} jobs on {boot['boards']} boards (B0 {result['b0_mean']:.3f}%, B1 {result['b1_mean']:.3f}%): "
          f"{'HOLDS' if holds else 'does NOT hold'}" + ("  (VOID: the GPU was not ours alone at the end)" if result["void"] else ""))
    return 2 if result["void"] else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stage1")
    sub.add_parser("smoke")
    p2 = sub.add_parser("p2")
    p2.add_argument("--threads", type=int, default=THREADS, help="4 as registered; 1 if the smoke run's differ")
    sub.add_parser("p1")
    a = ap.parse_args()
    return {"stage1": cmd_stage1, "smoke": cmd_smoke, "p2": cmd_p2, "p1": cmd_p1}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
