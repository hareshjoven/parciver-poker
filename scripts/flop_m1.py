"""FLOPSEARCH.md D4 M1: loss against the full flop oracle on real flop decisions
(the rules are in the D4 protocol amendment).

  record    deal seeded hands (seed 550000) and play v2-2B's preflop against
            itself; the first 40 that reach the flop with chips behind are the
            spots: the preflop actions, the flop and the round's starting
            ranges as the flop search sees them (`m1-spots.json`). The hero
            alternates: even spots OOP (seat 1), odd spots IP (seat 0).
  strategy  a candidate X at every hero flop node of the oracle, from the
            agent's `options` at that node's history (`m1-x-<cand>.npz`):
            bp (v2-2B, no search), net (flop search, value net leaves), eq
            (flop search, check-down leaves); FLOP-SYM's sym (C2) and orbit
            (C1): the net's suit-symmetric leaves (records/flop-sym/PREREG.md).
            A search candidate also saves every flop job it solved
            (`m1-jobs-<cand>.pkl`, FLOP-SYM's speed jobs).
  solve     the oracle at each spot with X frozen at the hero's flop nodes
            (`star`: nothing frozen, V*), to the iteration rule; one row per
            spot appended to `m1-v-<cand>.json` (resumable).
  report    Loss(X_net) - Loss(X_bp) = V(X_bp) - V(X_net) in pot units, with
            the paired 95% CI and the oracle's value brackets; for sym and
            orbit, FLOP-SYM's test: Loss(X_C) - Loss(X_net), non-inferior when
            the CI's upper end + e <= SYM_MARGIN.
  A sym / orbit solve copies a rainbow spot's row from m1-v-net.json when its
  X there is bit-identical to X_net's (the same game; FLOP-SYM).

LEAF-NET (records/leaf-net/PREREG.md, Gate B): `leafnet` is the sym leaves with
the grid's pick P (`strategy --cand leafnet --net <P>`; every spot solved, no
copies). `symcheck` is X_sym solved again under the current engine:
  symcheck-init  m1-x-symcheck.* = byte copies of m1-x-sym.* (no strategy run)
  enginecheck    the two check spots (lowest `sec` among m1-v-sym's solved rows
                 and among its copied rows); once `solve --cand symcheck --spots`
                 has them, compares their rows to m1-v-sym's bit for bit (all
                 fields but sec and copied) -> records/leaf-net/enginecheck.json
The report pairs leafnet with sym when the check passed, else with symcheck
(all 40 spots solved again), z 1.96, -> records/leaf-net/m1-report-leafnet.json.

D2b (records/d2b/PREREG.md, Gate B), in runs/flop-eval-d2b on the first 160 spots:
`d2b0` is the sym leaves with N0, `d2b1` with N1, the net Gate A passed.
  record --spots 160  the spots, read by no net and no label
  d2b-init    the first 40 spots equal m1-spots.json's bit for bit, the engine
              the engine check's -> d2b-carry.json (no strategy needs the GPU yet)
  strategy    d2b0: spots 0-39 are m1-x-sym.npz's arrays (0-1 made again and
              compared, the guard on the code that made them); d2b1: all new
  solve       d2b0: starts from m1-v-sym.json's 40 rows, each marked `carried`
  report-d2b  Loss(X_d2b1) - Loss(X_d2b0), z 1.96, a verdict at 160 spots only
              -> records/d2b/m1-report.json

D3 (records/d3/PREREG.md 5.5, Gate B), in runs/flop-eval-d3 on the same 160 spots (seed 550000, bit for bit):
`d3n1` is the sym leaves with N1 (D2b's X_d2b1, carried), `d3n2` with N2. Only X_N2's 160 solves are new.
  d3-init     the 160 spots equal runs/flop-eval-d2b's bit for bit, D2b adopted N1, the engine X_N1's rows were solved
              under is this build, X_N1's files are the ones D2b wrote -> d3-carry.json
  strategy    d3n1: all 160 spots are m1-x-d2b1.npz's arrays (0-1 made again and compared); a difference writes
              d3-remake-failed.json and stops, and `--resolve-all` then makes and solves all of X_N1 again; d3n2: all new
  solve       d3n1: starts from m1-v-d2b1.json's 160 rows, each marked `carried` (not in resolve mode)
  report-d3   Loss(X_d3n2) - Loss(X_d3n1), z 1.96, a verdict at 160 spots only -> records/d3/m1-report.json

Paths default to runs/flop-eval. No dashboard (the author paused them)."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from flop_eval import ABS, STAKES, V2, FlopRecorder, pot_of

M1_SEED = 550000
# Flop: the search's sizes with three raises (v2-2B's 3-bets exact); turn, river.
MENUS = [([0.5, 1.0], [1.0], 3), ([0.75], [], 1), ([0.75], [], 1)]
FLOP_ITERS, GAMMA = 500, 5.0  # the flop search's settings (D3 calibration)
# The oracle's rule (D4 protocol amendment): DCFR with sums weighted by
# ORACLE_GAMMA, checked every CHECK_EVERY iterations, stopped at TOLERANCE
# (exploitability, % of the pot) or ITERS. ORACLE_GAMMA and ITERS come from the
# amendment's calibration on other seeds; `solve` refuses to run until they are set.
ORACLE_GAMMA: float | None = 5.0  # D4 calibration result: gamma 5 beat gamma 2
ITERS: int | None = 1000  # D4 calibration result: e_1000 = 0.092% of the pot
CHECK_EVERY, TOLERANCE = 100, 0.2
ORACLE_ERROR = 0.001  # e = max(0.1% of the pot, e_1000 = 0.092%), in pots: (a) needs the CI's upper end + e < 0
PRECISION = "f32"  # the stored regrets and sums (half of float64's memory)
HEADROOM = 10 * 2**30  # commit left free beside a solve (the generator's GPU driver needs it)
SCRATCH = 2**30  # measured: a 2bb solve peaks 1.3 GiB above its store, 0.4 GiB of it the tree
# Deal suit-isomorphic turn and river cards once when the ranges and X are exactly
# unchanged by the flop's suit swaps (the same game; otherwise the full one).
ISOMORPHISM = True  # FLOPSEARCH D4 isomorphism amendment: both rules passed (30 Sep)
CANDIDATES = ("star", "bp", "net", "eq", "sym", "orbit", "symcheck", "leafnet")
D2B = ("d2b0", "d2b1")  # records/d2b/PREREG.md: not read by `report`
D3 = ("d3n1", "d3n2")  # records/d3/PREREG.md: not read by `report`
LEAVES = {"net": "dense", "sym": "sym", "orbit": "orbit", "symcheck": "sym", "leafnet": "sym", "d2b0": "sym", "d2b1": "sym",
          "d3n1": "sym", "d3n2": "sym"}  # the net's leaves
SYM_MARGIN = 0.0025  # FLOP-SYM's non-inferiority margin (pots), fixed before any data
SYM_Z = 2.241  # FLOP-SYM's CI: Bonferroni over C2 and C1 (one-sided 1.25% each)
LEAFNET_Z = 1.96  # LEAF-NET Gate B: one candidate reaches M1 (records/leaf-net/PREREG.md)
LEAFNET_MARGIN = 0.0025  # LEAF-NET Gate B's margin (FLOP-SYM's)
N0_SHA256 = "a8bf00be2fd1a3c7b0d1b0796809256cd939305b68723e541ef2cd029238ef65"  # the live net X_sym was made with
N1_SHA256 = "54bae7494d830143daa5a06d6dfd7a805e99b28f1c3b5dfd9ec2533de7f97d72"  # D2b's adopted net, the live flop net
ROW_FIELDS = ("br", "expl", "lo", "hi", "value", "profile", "iters", "curve", "isomorphism", "frozen")  # enginecheck
REPO = Path(__file__).resolve().parents[1]
LEAF_RECORDS = REPO / "records" / "leaf-net"
D2B_RECORDS = REPO / "records" / "d2b"
D2B_OLD = REPO / "runs" / "flop-eval"  # m1-spots.json, m1-x-sym.npz, m1-v-sym.json, m1-v-star.json
D2B_SPOTS, D2B_CARRIED, D2B_REMADE = 160, 40, 2  # spots; the first of them carried; the first of those made again
D2B_Z, D2B_MARGIN = 1.96, 0.0025  # Gate B (LEAF-NET's)
D3_RECORDS = REPO / "records" / "d3"
D3_OLD = REPO / "runs" / "flop-eval-d2b"  # m1-spots.json, m1-x-d2b1.npz, m1-v-d2b1.json (X_N1), d2b-carry.json
D3_SPOTS, D3_REMADE = 160, 2  # spots (all carried); the first of them made again


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def below_normal():
    """Run below normal priority (the turn-data generator keeps the machine)."""
    if sys.platform == "win32":
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)


def free_commit() -> int | None:
    """Bytes the system can still commit (its limit minus its charge)."""
    if sys.platform != "win32":
        return None

    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
            (f, ctypes.c_uint64) for f in ("phys", "avail_phys", "commit_limit", "avail_commit", "virt", "avail_virt", "ext")]

    st = Status()
    st.length = ctypes.sizeof(st)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.avail_commit


def wait_for_memory(o) -> None:
    """Before the store is allocated (its first freeze or run): wait until the
    store, the solve's buffers and HEADROOM fit in the free commit (the tree is
    already charged). The D1 generator crashed on 29 Sep when an oracle took the
    machine to its commit limit."""
    need = (8 if PRECISION == "f32" else 16) * o.size[2] * len(o.holdings) + SCRATCH + HEADROOM
    waited = False
    while (free := free_commit()) is not None and free < need:
        if not waited:
            print(f"  waiting for memory: {need / 2**30:.1f} GiB needed, {free / 2**30:.1f} free", flush=True)
            waited = True
        time.sleep(60)


def oracle(pe, spot: dict, threads: int = 0, isomorphism: bool = False):
    half = spot["pot"] // 2
    return pe.FlopOracle(spot["board"], (half, half), tuple(spot["total"][p] - half for p in (0, 1)), spot["first"],
                         STAKES[1], MENUS, tuple(spot["ranges"]), threads=threads, precision=PRECISION,
                         isomorphism=isomorphism)


def frozen_oracle(pe, spot: dict, threads: int, fixed: list, isomorphism: bool = False):
    """The oracle with each `(path, strategy)` in `fixed` frozen, isomorphic when
    asked and the ranges and every strategy allow it (the engine checks exactly;
    a refusal falls back to the full game). Waits for memory before the store."""
    for iso in dict.fromkeys((isomorphism, False)):
        try:
            o = oracle(pe, spot, threads, iso)
            wait_for_memory(o)
            for path, strategy in fixed:
                o.freeze(path, strategy)
            return o
        except ValueError:
            if not iso:
                raise
    raise AssertionError("unreachable")


def rainbow(board) -> bool:
    """Whether only the identity among the suit permutations maps the flop onto itself."""
    from parciver import value_net as vn

    return sum(set(vn.CARD_PERM[k, list(board)].tolist()) == set(board) for k in range(len(vn.SUIT_PERMS))) == 1


def history(nodes: dict, path: tuple) -> list:
    """The flop actions along `path`, as `PlayAgent` takes them."""
    out, put = [], [0, 0]
    for k in range(len(path)):
        player, acts = nodes[path[:k]]
        kind, add = acts[path[k]]
        if kind == "b":
            put[player] += add
            out.append(("b", add))
        elif kind == "c":
            out.append(("c", 0) if put[1 - player] > put[player] else ("x", 0))
            put[player] = put[1 - player]
        else:
            raise ValueError(f"a fold on the path {path}")
    return out


def to_oracle(opts: list, acts: list) -> tuple[list, float]:
    """One holding's `options` as a distribution over the node's oracle
    actions: bets by their chips, else the nearest oracle bet in log size
    (ties to the smaller). Also the probability that needed the nearest."""
    row, moved = [0.0] * len(acts), 0.0
    bets = [(a, i) for i, (k, a) in enumerate(acts) if k == "b"]
    for kind, add, p in opts:
        if kind == "fold":
            i = next(i for i, (k, _) in enumerate(acts) if k == "f")
        elif kind in ("check", "call"):
            i = next(i for i, (k, _) in enumerate(acts) if k == "c")
        else:
            exact = [i for a, i in bets if a == add]
            if exact:
                i = exact[0]
            else:
                i = min(bets, key=lambda b: (abs(math.log(add / b[0])), b[0]))[1]
                moved += p
        row[i] += p
    return row, moved


def cmd_record(a) -> int:
    import parciver_engine as pe

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "m1-spots.json"
    if path.exists():
        print(f"{path} exists")
        return 1
    sb, bb, stack = STAKES
    agent = pe.SavedBlueprint(ABS, V2).play_agent()
    rec = FlopRecorder()
    searcher = agent.with_search(from_flop=True, solver=rec)
    spots, deal = [], 0
    while len(spots) < a.spots:
        rng = np.random.default_rng([a.seed, deal])
        cards = [int(c) for c in rng.permutation(52)[:7]]
        holes, board = [tuple(sorted(cards[0:2])), tuple(sorted(cards[2:4]))], sorted(cards[4:7])
        actions, put, acted, p, over = [], [sb, bb], [False, False], 0, False
        while not over:
            opts = agent.options(p, [], actions, sb, bb, (stack, stack), [holes[p]], seed=deal)[0]
            u, acc = rng.random(), 0.0
            kind, add, _ = opts[-1]
            for kind_i, add_i, prob in opts:
                acc += prob
                if u < acc:
                    kind, add = kind_i, add_i
                    break
            if kind == "fold":
                actions.append(("f", 0))
                break
            if kind == "bet":
                put[p] += add
                acted = [False, False]
                actions.append(("b", add))
            else:
                put[p] = put[1 - p]
                actions.append(("c", 0) if kind == "call" else ("x", 0))
            acted[p] = True
            over = acted[0] and acted[1] and put[0] == put[1]
            p = 1 - p
        deal += 1
        if not over or max(put) >= stack:
            continue  # a fold, or all-in before the flop: no flop decision
        before = len(rec.jobs)
        searcher.decide(1, holes[1], board, actions, sb, bb, (stack, stack), seed=deal - 1)
        if len(rec.jobs) != before + 1:
            print(f"  deal {deal - 1}: no flop job recorded, skipped")
            continue
        job = rec.jobs[-1]
        pot = int(pot_of(job))
        if pot != 2 * put[0] or list(job["board"]) != board:
            raise RuntimeError(f"deal {deal - 1}: the job ({job['board']}, pot {pot}) is not the hand's")
        spots.append(dict(deal=deal - 1, holes=holes, board=board, actions=actions, pot=pot, total=list(job["total"]),
                          first=int(job["player"][0]), hero=1 if len(spots) % 2 == 0 else 0,
                          ranges=[list(map(float, r)) for r in job["ranges"]]))
    path.write_text(json.dumps(dict(seed=a.seed, stakes=STAKES, deals=deal, spots=spots)))
    pots = [s["pot"] / bb for s in spots]
    print(f"saved {len(spots)} spots ({deal} deals) to {path}; pot median {np.median(pots):.1f} bb, "
          f"range {min(pots):.0f}-{max(pots):.0f} bb")
    return 0


class Strict:
    """A search solver that keeps its failures: exceptions and answers the
    engine refuses (`solver_watch.malformed`). Engines before Finding 16's
    engine half answer a failed search from the blueprint without a word
    (play.rs `options_many`), which would store blueprint rows under a search
    candidate's name; later ones raise RuntimeError from `options`."""

    def __init__(self, solver):
        from parciver.solver_watch import malformed

        self.solver, self.errors, self.malformed = solver, [], malformed
        self.jobs = []  # the flop jobs, in order (FLOP-SYM's speed test)

    @property
    def solves(self) -> int:
        return self.solver.solves

    def __call__(self, job):
        if len(job["board"]) == 3:
            self.jobs.append(job)
        try:
            out = self.solver(job)
        except BaseException as e:
            self.errors.append(repr(e))
            raise
        bad = self.malformed(job, out)
        if bad:
            self.errors.append(bad)
        return out


def make_agent(pe, cand: str, net_path: str | None):
    agent = pe.SavedBlueprint(ABS, V2).play_agent()
    if cand == "bp":
        return agent, None
    from parciver import flop_search as fs
    from parciver import value_net as vn

    net = None
    if cand in LEAVES:
        if not net_path:
            raise SystemExit(f"--net is required for X_{cand}")
        net, _ = vn.load(net_path)
    search = Strict(fs.FlopSearch(net, gamma=GAMMA, leaves=LEAVES.get(cand, "dense")))  # M1's leaves, or FLOP-SYM's
    return agent.with_search(from_flop=True, flop_iters=FLOP_ITERS, solver=search), search


def cmd_strategy(a) -> int:
    import parciver_engine as pe

    if a.cand == "symcheck":
        raise SystemExit("symcheck plays no strategy: `symcheck-init` copies X_sym's files")
    if a.cand == "leafnet":
        if not a.net:
            raise SystemExit("leafnet needs --net (P, the Gate A pick)")
        gate_a_passed(sha256(a.net))
    if a.cand in D2B:
        if a.seeds:
            raise SystemExit(f"{a.cand} plays seed 0 alone, as X_sym did: no --seeds")
        d2b_strategy_guard(a.cand, a.net, Path(a.out))
    if a.cand in D3:
        if a.seeds:
            raise SystemExit(f"{a.cand} plays seed 0 alone, as X_sym and X_N1 did: no --seeds")
        d3_strategy_guard(a.cand, a.net, Path(a.out), getattr(a, "resolve_all", False))
    elif getattr(a, "resolve_all", False):
        raise SystemExit("--resolve-all is X_d3n1's alone")
    out = Path(a.out)
    path, meta_path = out / f"m1-x-{a.cand}.npz", out / f"m1-x-{a.cand}.json"
    if path.exists():
        print(f"{path} exists")
        return 1
    spots = json.loads((out / "m1-spots.json").read_text())["spots"]
    sb, bb, stack = STAKES
    agent, search = make_agent(pe, a.cand, a.net)
    bp_agent = agent if a.cand == "bp" else make_agent(pe, "bp", None)[0]
    seeds = [int(s) for s in (a.seeds or ("0,1,2,3" if a.cand == "bp" else "0")).split(",")]
    arrays, meta = {}, []
    old_x, carried_to, old_name, remade_n = None, 0, None, 0  # the old X's file, its spots carried, its name, the first of them made again
    if a.cand == "d2b0":
        old_x, carried_to, old_name, remade_n = np.load(D2B_OLD / "m1-x-sym.npz"), D2B_CARRIED, "m1-x-sym.npz", D2B_REMADE
    elif a.cand == "d3n1" and d3_mode(out) == "carry":
        if engine_sha(pe) != d3_carry(out)["engine"]:
            raise SystemExit(f"this build ({engine_sha(pe)}) is not the one X_N1's carried rows were solved under: nothing made")
        old_x, carried_to, old_name, remade_n = np.load(D3_OLD / "m1-x-d2b1.npz"), D3_SPOTS, "m1-x-d2b1.npz", D3_REMADE
    t0 = time.time()
    for i, spot in enumerate(spots):
        if old_x is not None and remade_n <= i < carried_to:
            arrays.update({k: old_x[k] for k in old_x.files if k.startswith(f"{i}/")})
            meta.append(dict(spot=i, carried=old_name))
            continue
        o = oracle(pe, spot)
        holdings = o.holdings
        flop = o.flop_nodes()
        del o
        nodes = {tuple(p): (player, acts) for p, player, acts in flop}
        hero_nodes = [tuple(p) for p, player, _ in flop if player == spot["hero"]]
        pre = [tuple(x) for x in spot["actions"]]
        weight = np.asarray(spot["ranges"][spot["hero"]], dtype=np.float64)
        row = dict(spot=i, nodes=len(hero_nodes), unreached=0, moved=0.0, moved_selfplay=0.0, seed_diff=0, sources={})
        x_at: dict = {}
        # v2-2B's own play at the opponent's nodes: weights `moved_selfplay`
        # (the moved mass in v2-2B self-play from this spot, card removal
        # between the two hands ignored).
        villain = 1 - spot["hero"]
        opp_w = np.asarray(spot["ranges"][villain], dtype=np.float64)
        opp_at = {}
        for p_, (player, acts) in nodes.items():
            if player == villain:
                o_ = bp_agent.options(villain, spot["board"], pre + history(nodes, p_), sb, bb, (stack, stack), holdings)
                opp_at[p_] = np.array([to_oracle(r, acts)[0] for r in o_]).T
        for path_ in hero_nodes:  # pre-order: a node's own earlier nodes come first
            acts = nodes[path_][1]
            # The hero's own reach: X's probabilities of its actions on the path.
            reach = weight.copy()
            for k in range(len(path_)):
                if path_[:k] in x_at:
                    reach *= x_at[path_[:k]][path_[k]]
            if not reach.any():  # X never plays to here: any strategy (no effect)
                x_at[path_] = np.full((len(acts), len(holdings)), 1.0 / len(acts))
                row["unreached"] += 1
                arrays[f"{i}/" + ",".join(map(str, path_))] = x_at[path_]
                continue
            actions = pre + history(nodes, path_)
            per_seed = []
            for seed in seeds:
                try:
                    opts = agent.options(spot["hero"], spot["board"], actions, sb, bb, (stack, stack), holdings, seed=seed)
                except RuntimeError as e:  # a failed search: keep the cause `Strict` recorded
                    first = search.errors[0] if search is not None and search.errors else "none recorded"
                    raise SystemExit(f"spot {i}: {e} (first error: {first}); nothing saved") from e
                rows = [to_oracle(o_, acts) for o_ in opts]
                per_seed.append(np.array([r for r, _ in rows]).T)  # [acts, holdings]
                # Probability moved to the nearest size, weighted by the hero's reach
                # (summed over nodes: moved actions per hand, the opponent's reach ignored).
                moved = np.array([m for _, m in rows])
                row["moved"] += float(moved @ reach / weight.sum()) / len(seeds)
                opp = opp_w.copy()
                for k in range(len(path_)):
                    if path_[:k] in opp_at:
                        opp *= opp_at[path_[:k]][path_[k]]
                row["moved_selfplay"] += float(moved @ reach / weight.sum() * opp.sum() / opp_w.sum()) / len(seeds)
            if any(not np.array_equal(per_seed[0], x) for x in per_seed[1:]):
                row["seed_diff"] += 1
            if a.cand == "bp":
                src = agent.decide(spot["hero"], holdings[0], spot["board"], actions, sb, bb, (stack, stack),
                                   seed=seeds[0])["source"]
                row["sources"][src] = row["sources"].get(src, 0) + 1
            x_at[path_] = np.mean(per_seed, axis=0)
            arrays[f"{i}/" + ",".join(map(str, path_))] = x_at[path_]
        if search is not None and search.errors:
            raise SystemExit(f"spot {i}: {len(search.errors)} failed searches (first: {search.errors[0]}); "
                             f"their rows are the blueprint's, nothing saved")
        if old_x is not None and i < remade_n:
            remade = {k: v for k, v in arrays.items() if k.startswith(f"{i}/")}
            if sorted(remade) != sorted(k for k in old_x.files if k.startswith(f"{i}/")) or not all(
                    v.dtype == old_x[k].dtype and np.array_equal(v, old_x[k]) for k, v in remade.items()):
                if a.cand == "d3n1":  # PREREG 5.5: X_N1 is made and solved again, all of it; this record is the only way to ask
                    (out / "d3-remake-failed.json").write_text(json.dumps(dict(spot=i, engine=engine_sha(pe)), indent=1))
                    raise SystemExit(f"spot {i}: X_N1 made now differs from m1-x-d2b1.npz: nothing saved; {out / 'd3-remake-failed.json'} "
                                     f"written; all of X_N1 is made and solved again, with `strategy --cand d3n1 --resolve-all` "
                                     f"(records/d3/PREREG.md 5.5)")
                raise SystemExit(f"spot {i}: X_N0 made now differs from m1-x-sym.npz: nothing saved; the carried "
                                 f"strategies are not this code's (records/d2b/PREREG.md, VOID and bugs)")
            arrays.update({k: old_x[k] for k in remade})  # m1-x-sym.npz's own bytes, as spots 2-39: -0.0 passes as 0.0 above
            row["remade_equal"] = True
        meta.append(row)
        print(f"  spot {i + 1}/{len(spots)}  {row['nodes']} hero nodes ({row['unreached']} unreached)  "
              f"seed-dependent {row['seed_diff']}  moved {row['moved']:.4f} (self-play {row['moved_selfplay']:.5f})  "
              f"{row['sources']}  "
              f"{time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(path, **arrays)
    if search is not None:
        import pickle

        (out / f"m1-jobs-{a.cand}.pkl").write_bytes(pickle.dumps(search.jobs))
    meta_path.write_text(json.dumps(dict(cand=a.cand, net=a.net, net_sha256=sha256(a.net) if a.net else None, seeds=seeds, flop_iters=FLOP_ITERS, gamma=GAMMA,
                                         solves=search.solves if search else 0,
                                         symmetric=getattr(search.solver, "symmetric", 0) if search else 0,
                                         drift=getattr(search.solver, "drift", 0.0) if search else 0.0,
                                         spots=meta, **({"mode": d3_mode(out)} if a.cand in D3 else {})), indent=1))
    print(f"saved {len(arrays)} hero-node strategies to {path}")
    return 0


def cmd_solve(a) -> int:
    import parciver_engine as pe

    from parciver.solver_watch import write_atomic

    if ORACLE_GAMMA is None or ITERS is None:
        raise SystemExit("set ORACLE_GAMMA and ITERS from the D4 calibration result first")
    below_normal()
    out = Path(a.out)
    spots = json.loads((out / "m1-spots.json").read_text())["spots"]
    x_path = out / f"m1-x-{a.cand}.npz"
    x = None if a.cand == "star" else np.load(x_path)
    path = out / f"m1-v-{a.cand}.json"
    # As a file holds it (tuples are lists there). `inputs`: the files the rows
    # were solved from, so a resume or the report never mixes rows of a
    # regenerated X or other spots.
    inputs = dict(spots=sha256(out / "m1-spots.json"), x=None if x is None else sha256(x_path))
    if a.cand in ("sym", "orbit"):  # the files a copied rainbow row comes from
        inputs.update(x_net=sha256(out / "m1-x-net.npz"), v_net=sha256(out / "m1-v-net.json"))
    if a.cand in ("symcheck", "leafnet") + D2B + D3:  # LEAF-NET's engine pairing: a resume under another build is refused
        inputs.update(engine=engine_sha(pe), threads=a.threads)
    if a.cand in D2B:  # both spots files and the old files as d2b-init checked them, or no solve
        carry = d2b_carry(out)
    if a.cand == "d2b0":  # the old files' sha256 are in the header as well as in d2b-carry.json
        inputs.update(carry=sha256(out / "d2b-carry.json"), old_files=carry["files"])
    mode = None
    if a.cand in D3:  # d3-init's files as it checked them, or no solve; the carried rows' build, unless X_N1 is made again
        carry, mode = d3_carry(out), d3_mode(out)
        if mode == "carry" and inputs["engine"] != carry["engine"]:
            raise SystemExit(f"this build ({inputs['engine']}) is not the one X_N1's carried rows were solved under ({carry['engine']})")
    if a.cand == "d3n1":
        inputs.update(mode=mode, carry=sha256(out / "d3-carry.json"), old_files=carry["files"])
        if mode == "resolve":
            inputs.update(failed=sha256(out / "d3-remake-failed.json"))
    header = json.loads(json.dumps(dict(cand=a.cand, menus=MENUS, rule=[ORACLE_GAMMA, CHECK_EVERY, TOLERANCE, ITERS],
                                        precision=PRECISION, inputs=inputs)))
    rows = []
    if a.cand == "d2b0" and not path.exists():
        rows = d2b_carried_rows(out, x)
        write_atomic(path, json.dumps(dict(header, rows=rows), indent=1))
    if a.cand == "d3n1" and mode == "carry" and not path.exists():
        rows = d3_carried_rows(out, x, header)
        write_atomic(path, json.dumps(dict(header, rows=rows), indent=1))
    if path.exists():
        old = json.loads(path.read_text())
        rows = old.pop("rows")
        if "inputs" not in old and old == {k: v for k, v in header.items() if k != "inputs"}:
            # Begun before the inputs were recorded (bp, 30 Sep): resumed, and
            # left without them (they cannot be checked for the rows it holds).
            print(f"  {path.name}: inputs unrecorded (an older file), resumed under its own header", flush=True)
            header = old
        elif old != header:
            raise SystemExit(f"{path} holds rows solved under other settings or inputs ({old}, now {header})")
    done = {r["spot"] for r in rows}
    net_rows = {}
    if a.cand in ("sym", "orbit"):
        net_rows = {r["spot"]: r for r in json.loads((out / "m1-v-net.json").read_text())["rows"]}
        x_net = np.load(out / "m1-x-net.npz")
    for i, spot in enumerate(spots):
        if i in done or (a.spots and i not in a.spots):
            continue
        if i in net_rows and rainbow(spot["board"]):
            if same_x(x, x_net, i):
                rows.append(dict(net_rows[i], copied="net"))  # a rainbow spot, X bit-identical: the same game
                write_atomic(path, json.dumps(dict(header, rows=rows), indent=1))
                print(f"  {a.cand} spot {i + 1}/{len(spots)}  rainbow, X bit-identical to X_net: its row copied", flush=True)
                continue
        t0 = time.time()
        fixed = []
        if x is not None:
            for key in (k for k in x.files if k.startswith(f"{i}/")):
                tail = key.split("/", 1)[1]
                fixed.append(([int(v) for v in tail.split(",")] if tail else [], x[key].reshape(-1).tolist()))
        o = frozen_oracle(pe, spot, a.threads, fixed, ISOMORPHISM)
        frozen = len(fixed)
        pot, hero, curve = spot["pot"], spot["hero"], []
        while True:
            o.run(CHECK_EVERY, ORACLE_GAMMA)
            br = o.best_responses()
            expl = 100.0 * (br[0] + br[1]) / pot
            curve.append((o.iterations, expl, (br[hero] - br[1 - hero]) / 2 / pot))
            if expl <= TOLERANCE or o.iterations >= ITERS:
                break
        # The game value to the hero lies in [-br[1 - hero], br[hero]].
        lo, hi = -br[1 - hero] / pot, br[hero] / pot
        rows.append(dict(spot=i, pot=pot, hero=hero, frozen=frozen, isomorphism=o.isomorphism, iters=o.iterations,
                         br=list(br), expl=expl,
                         lo=lo, hi=hi, value=(lo + hi) / 2, profile=o.value(hero) / pot, curve=curve,
                         sec=time.time() - t0))
        del o
        write_atomic(path, json.dumps(dict(header, rows=rows), indent=1))  # a kill or a power cut keeps the finished rows
        r = rows[-1]
        print(f"  {a.cand} spot {i + 1}/{len(spots)}  pot {pot / 100:.0f}bb  {r['iters']} iters  expl {expl:.3f}%  "
              f"V {100 * r['value']:+.3f}% [{100 * lo:+.3f}, {100 * hi:+.3f}]  {r['sec']:.0f}s", flush=True)
    return 0


def cmd_symcheck_init(a) -> int:
    """m1-x-symcheck.{npz,json} = byte copies of X_sym's (the json's cand
    renamed): the same X, solved again under the current engine."""
    out = Path(a.out)
    if (out / "m1-x-symcheck.npz").exists():
        print("m1-x-symcheck.npz exists")
        return 1
    (out / "m1-x-symcheck.npz").write_bytes((out / "m1-x-sym.npz").read_bytes())
    meta = json.loads((out / "m1-x-sym.json").read_text())
    (out / "m1-x-symcheck.json").write_text(json.dumps(dict(meta, cand="symcheck", copy_of="m1-x-sym.npz"), indent=1))
    print("m1-x-symcheck.npz: a byte copy of m1-x-sym.npz")
    return 0


def engine_sha(pe) -> str:
    """sha256 of the loaded engine binary (as rating.py's engine identity)."""
    f = getattr(getattr(pe, "parciver_engine", None), "__file__", None)
    if not f or Path(f).suffix not in (".pyd", ".so"):
        raise SystemExit(f"cannot locate the loaded engine binary of {pe.__file__}")
    return sha256(f)


def gate_a_passed(p_sha: str) -> dict:
    """records/leaf-net/gate-a.json, refused unless Gate A passed for the net
    whose sha256 is `p_sha` (Gate B runs only after Gate A, on its P)."""
    path = LEAF_RECORDS / "gate-a.json"
    g = json.loads(path.read_text()) if path.exists() else {}
    if not g.get("proceed") or g.get("sha256", {}).get("P") != p_sha:
        raise SystemExit(f"{path}: Gate A has not passed for this net (proceed {g.get('proceed')}, "
                         f"P {g.get('sha256', {}).get('P')}, this net {p_sha})")
    return g


def check_spots(rows: list[dict]) -> tuple[int, int]:
    """LEAF-NET's engine-check spots: the lowest `sec` among the solved rows
    and among the copied rows of m1-v-sym.json."""
    solved = [r for r in rows if not r.get("copied")]
    copied = [r for r in rows if r.get("copied")]
    if not solved or not copied:
        raise SystemExit(f"m1-v-sym.json: {len(solved)} solved and {len(copied)} copied rows; the check needs one of each")
    return min(solved, key=lambda r: r["sec"])["spot"], min(copied, key=lambda r: r["sec"])["spot"]


def cmd_enginecheck(a) -> int:
    out = Path(a.out)
    sym = {r["spot"]: r for r in json.loads((out / "m1-v-sym.json").read_text())["rows"]}
    spots = check_spots(list(sym.values()))
    p = out / "m1-v-symcheck.json"
    have = {r["spot"]: r for r in json.loads(p.read_text())["rows"]} if p.exists() else {}
    if not all(s in have for s in spots):
        print(f"solve the check spots first: flop_m1.py solve --cand symcheck --spots {spots[0]},{spots[1]} (--threads 12)")
        return 2
    if sha256(out / "m1-x-symcheck.npz") != sha256(out / "m1-x-sym.npz"):
        raise SystemExit("m1-x-symcheck.npz is not a byte copy of m1-x-sym.npz")
    diff = {s: [k for k in ROW_FIELDS if have[s][k] != sym[s][k]] for s in spots}
    res = dict(spots=list(spots), fields=list(ROW_FIELDS), differ=diff, passed=not any(diff.values()),
               baseline="sym" if not any(diff.values()) else "symcheck",
               engine=json.loads(p.read_text())["inputs"].get("engine"),
               files={f: sha256(out / f) for f in ("m1-v-sym.json", "m1-x-sym.npz")})
    LEAF_RECORDS.mkdir(parents=True, exist_ok=True)
    (LEAF_RECORDS / "enginecheck.json").write_text(json.dumps(res, indent=1))
    print(f"engine check on spots {spots}: {'PASS (baseline sym)' if res['passed'] else f'FAIL {diff}: solve all 40 as symcheck'}")
    return 0 if res["passed"] else 1


def leafnet_report(out: Path, got: dict, heads: dict, spots: list) -> dict | None:
    """LEAF-NET Gate B: Loss(X_leafnet) - Loss(baseline), the baseline sym
    (enginecheck passed) or symcheck (all 40 spots solved again)."""
    chk = LEAF_RECORDS / "enginecheck.json"
    if "leafnet" not in got or not chk.exists():
        return None
    check = json.loads(chk.read_text())
    base = check["baseline"]
    if base not in got:
        print(f"LEAF-NET: the baseline {base} has no rows yet")
        return None
    same_game(out, {c: heads[c] for c in ("star", base, "leafnet") if c in heads})
    if base == "sym" and any(sha256(out / f) != h for f, h in check["files"].items()):
        raise SystemExit("m1-v-sym.json or m1-x-sym.npz changed after the engine check")
    if heads["leafnet"]["inputs"].get("engine") != check["engine"]:
        raise SystemExit("X_leafnet was solved under another engine build than the engine check's")
    if base == "symcheck" and heads["symcheck"]["inputs"].get("engine") != check["engine"]:
        raise SystemExit("X_symcheck's rows were solved under another engine build than the engine check's")
    lf = json.loads((REPO / "runs" / "turn-net" / "leafnet.json").read_text())
    p_path = REPO / "runs" / "turn-net" / lf["pick"] / "net.pt"
    meta = json.loads((out / "m1-x-leafnet.json").read_text())
    if meta.get("net_sha256") != sha256(p_path) or sha256(REPO / meta["net"]) != meta["net_sha256"]:
        raise SystemExit("X_leafnet was not made with the grid's pick P, or that net file has changed since")
    gate_a_passed(meta["net_sha256"])
    if json.loads((out / f"m1-x-{base}.json").read_text()).get("net_sha256") != N0_SHA256:
        raise SystemExit(f"X_{base} was not made with N0")
    common = sorted(set(got["leafnet"]) & set(got[base]))
    if any(got["leafnet"][i].get("copied") for i in common):
        raise SystemExit("X_leafnet has a copied row: every leafnet spot is solved")
    if len(common) < 2:
        print(f"LEAF-NET: {len(common)} spot(s) solved, nothing to report yet")
        return None
    d = np.array([got[base][i]["value"] - got["leafnet"][i]["value"] for i in common])  # Loss(leafnet) - Loss(base)
    m, lo, hi = paired(d, LEAFNET_Z)
    final = len(common) == len(spots)
    res = dict(baseline=base, pick=lf["pick"], net_sha256=meta["net_sha256"], spots=len(common), final=final, mean=m,
               ci=[lo, hi], z=LEAFNET_Z, oracle_error=ORACLE_ERROR, margin=LEAFNET_MARGIN,
               non_inferior=hi + ORACLE_ERROR <= LEAFNET_MARGIN if final else None,
               superior=hi + ORACLE_ERROR < 0 if final else None, per_spot={int(i): float(x) for i, x in zip(common, d)},
               isomorphism=sum(1 for i in common if got["leafnet"][i]["isomorphism"]),
               hours=sum(got["leafnet"][i]["sec"] for i in common) / 3600,
               solves=meta.get("solves"), symmetric=meta.get("symmetric"), drift=meta.get("drift"))
    verdict = (("PASS" if res["non_inferior"] else "FAIL") if final
               else f"interim ({len(common)}/{len(spots)} spots, no verdict)")
    print(f"LEAF-NET Gate B: Loss(X_leafnet) - Loss(X_{base}) = {100 * m:+.3f}% pot, CI (z {LEAFNET_Z}) "
          f"[{100 * lo:+.3f}, {100 * hi:+.3f}] on {len(common)} spots; "
          + (f"upper end + e = {100 * (hi + ORACLE_ERROR):+.3f}% (<= {100 * LEAFNET_MARGIN:.2f}%): " if final else "")
          + verdict)
    return res


def d2b_gate_a(n1_sha: str | None = None) -> dict:
    """records/d2b/gate-a.json, refused unless Gate A passed for N1 (`n1_sha`,
    when given) against N0, the net X_sym was made with."""
    path = D2B_RECORDS / "gate-a.json"
    g = json.loads(path.read_text()) if path.exists() else {}
    sha = g.get("sha256", {})
    if not g.get("proceed") or sha.get("N0") != N0_SHA256 or (n1_sha is not None and sha.get("N1") != n1_sha):
        raise SystemExit(f"{path}: Gate A has not passed for this net (proceed {g.get('proceed')}, {sha}, this net {n1_sha})")
    return g


def d2b_carry(out: Path) -> dict:
    """d2b-carry.json, refused if a file it holds the sha256 of has changed since d2b-init."""
    carry = json.loads((out / "d2b-carry.json").read_text())
    if {f: sha256(D2B_OLD / f) for f in carry["files"]} != carry["files"]:
        raise SystemExit("m1-v-sym.json or m1-x-sym.npz changed after d2b-init")
    if sha256(out / "m1-spots.json") != carry["new_spots"] or sha256(D2B_OLD / "m1-spots.json") != carry["spots"]:
        raise SystemExit("a m1-spots.json changed after d2b-init: the first 40 spots are not the ones it compared")
    return carry


def d2b_strategy_guard(cand: str, net: str | None, out: Path) -> None:
    if not net:
        raise SystemExit(f"{cand} needs --net")
    if not (out / "d2b-carry.json").exists():
        raise SystemExit(f"{out}: run d2b-init first")
    d2b_carry(out)
    if cand == "d2b1":
        d2b_gate_a(sha256(net))
    else:
        d2b_gate_a()
        if sha256(net) != N0_SHA256:
            raise SystemExit(f"{net} is not N0")


def d2b_x_settings(meta: dict) -> bool:
    """A strategy's seeds and search settings are the ones D2b's are made with."""
    return meta.get("seeds") == [0] and meta.get("flop_iters") == FLOP_ITERS and meta.get("gamma") == GAMMA


def cmd_d2b_init(a) -> int:
    """The carried rows' conditions: the first 40 spots bit for bit, the engine
    the engine check's, the old files the ones it checked."""
    import parciver_engine as pe

    out = Path(a.out)
    new, old = (json.loads((p / "m1-spots.json").read_text()) for p in (out, D2B_OLD))
    if (new["seed"], new["stakes"]) != (old["seed"], old["stakes"]) or len(new["spots"]) != D2B_SPOTS:
        raise SystemExit(f"{out}: seed {new['seed']}, {len(new['spots'])} spots: not the registered {D2B_SPOTS} of seed {old['seed']}")
    if new["spots"][:D2B_CARRIED] != old["spots"]:
        raise SystemExit("the first 40 spots are not m1-spots.json's: D2b stops here (PREREG, Gate B)")
    check = json.loads((LEAF_RECORDS / "enginecheck.json").read_text())
    engine = engine_sha(pe)
    if not check.get("passed") or check.get("baseline") != "sym" or check.get("engine") != engine:
        raise SystemExit(f"the engine check does not cover this build ({engine}): {check}")
    files = {f: sha256(D2B_OLD / f) for f in ("m1-v-sym.json", "m1-x-sym.npz")}
    if files != check["files"]:
        raise SystemExit("m1-v-sym.json or m1-x-sym.npz changed after the engine check")
    x_sym = json.loads((D2B_OLD / "m1-x-sym.json").read_text())
    if x_sym.get("net_sha256") != N0_SHA256:
        raise SystemExit("X_sym was not made with N0")
    if not d2b_x_settings(x_sym):
        raise SystemExit(f"X_sym's settings are not the ones X_d2b0 and X_d2b1 are made with: {x_sym}")
    res = dict(engine=engine, files=files, spots=sha256(D2B_OLD / "m1-spots.json"), new_spots=sha256(out / "m1-spots.json"),
               compared=D2B_CARRIED)
    (out / "d2b-carry.json").write_text(json.dumps(res, indent=1))
    print(f"the first {D2B_CARRIED} of {D2B_SPOTS} spots are m1-spots.json's bit for bit; engine {engine[:8]}; {out / 'd2b-carry.json'}")
    return 0


def d2b_carried_rows(out: Path, x) -> list[dict]:
    """m1-v-sym.json's rows for the first 40 spots, marked `carried`, refused
    unless the files are the ones d2b-init checked and X is theirs there."""
    d2b_carry(out)
    old_x = np.load(D2B_OLD / "m1-x-sym.npz")
    if not all(same_x(x, old_x, i) for i in range(D2B_CARRIED)):
        raise SystemExit("X_d2b0's first 40 spots are not m1-x-sym.npz's")
    rows = json.loads((D2B_OLD / "m1-v-sym.json").read_text())["rows"]
    if sorted(r["spot"] for r in rows) != list(range(D2B_CARRIED)):
        raise SystemExit("m1-v-sym.json does not hold the first 40 spots")
    return [dict(r, carried="m1-v-sym") for r in sorted(rows, key=lambda r: r["spot"])]


def d2b_board_class(board) -> str:
    """Suits on the flop, and whether a rank pairs (cards are 4 * rank + suit)."""
    suits = {c % 4 for c in board}
    return {1: "monotone", 2: "two-tone", 3: "rainbow"}[len(suits)] + (" paired" if len({c // 4 for c in board}) < 3 else "")


def d2b_summary(d: np.ndarray) -> dict:
    m, lo, hi = paired(d, D2B_Z)
    return dict(spots=len(d), mean=m, ci=[lo, hi], sd=float(d.std(ddof=1)))


def d2b_guards(out: Path, got: dict, heads: dict) -> dict:
    """Everything the report rests on, or SystemExit. Returns gate-a.json."""
    g = d2b_gate_a()
    carry = d2b_carry(out)
    same_game(out, heads)
    if {h["inputs"]["engine"] for h in heads.values()} != {carry["engine"]}:
        raise SystemExit("the two candidates' rows were not solved under the engine check's build")
    if heads["d2b0"]["inputs"]["carry"] != sha256(out / "d2b-carry.json"):
        raise SystemExit("d2b-carry.json changed after X_d2b0's solve began")
    for c, want in (("d2b0", N0_SHA256), ("d2b1", g["sha256"]["N1"])):
        meta = json.loads((out / f"m1-x-{c}.json").read_text())
        if meta.get("net_sha256") != want or sha256(REPO / meta["net"]) != want:
            raise SystemExit(f"X_{c} was not made with its registered net, or that net file has changed since")
        if not d2b_x_settings(meta):
            raise SystemExit(f"X_{c} was made with other seeds or search settings: {[meta.get(k) for k in ('seeds', 'flop_iters', 'gamma')]}")
    old_rows = {r["spot"]: r for r in json.loads((D2B_OLD / "m1-v-sym.json").read_text())["rows"]}
    old_x, x0 = np.load(D2B_OLD / "m1-x-sym.npz"), np.load(out / "m1-x-d2b0.npz")
    for i, r in got["d2b0"].items():
        if (i < D2B_CARRIED) != (r.get("carried") == "m1-v-sym") or (i >= D2B_CARRIED and r.get("copied")):
            raise SystemExit(f"X_d2b0 spot {i}: carried or copied where the registration says not")
        if i < D2B_CARRIED and ({k: v for k, v in r.items() if k != "carried"} != old_rows[i] or not same_x(x0, old_x, i)):
            raise SystemExit(f"X_d2b0 spot {i}: no longer m1-v-sym.json's row, or m1-x-sym.npz's arrays")
    if any(r.get("carried") or r.get("copied") for r in got["d2b1"].values()):
        raise SystemExit("X_d2b1 has a carried or copied row: every spot is solved")
    return g


def d2b_report(out: Path) -> dict | None:
    """Gate B: D = Loss(X_d2b1) - Loss(X_d2b0) = V0 - V1 per spot, in pots. Pass
    iff the CI's upper end + e <= the margin, with all 160 in. Reported: the old
    40 and new 120 apart, the five largest D, and on the old 40 (the only spots
    with a V*) each X's Loss, to say whose the tail is."""
    got, heads = {}, {}
    for c in D2B:
        p = out / f"m1-v-{c}.json"
        if not p.exists():
            print(f"D2b Gate B: {p.name} is not there yet")
            return None
        f = json.loads(p.read_text())
        got[c] = {r["spot"]: r for r in f.pop("rows")}
        heads[c] = f
    g = d2b_guards(out, got, heads)
    spots = json.loads((out / "m1-spots.json").read_text())["spots"]
    common = sorted(set(got["d2b0"]) & set(got["d2b1"]))
    if len(common) < 2:
        print(f"D2b Gate B: {len(common)} spot(s) in both, nothing to report yet")
        return None
    d = np.array([got["d2b0"][i]["value"] - got["d2b1"][i]["value"] for i in common])
    m, lo, hi = paired(d, D2B_Z)
    final = len(common) == len(spots) == D2B_SPOTS
    at = np.array(common)
    star = json.loads((D2B_OLD / "m1-v-star.json").read_text())
    if json.dumps(star["rule"]) != json.dumps(heads["d2b0"]["rule"]) or star["inputs"]["spots"] != sha256(D2B_OLD / "m1-spots.json"):
        raise SystemExit("m1-v-star.json is not this rule's or these spots'")
    star = {r["spot"]: r for r in star["rows"]}
    old = [i for i in common if i < D2B_CARRIED]
    loss = {c: {i: star[i]["value"] - got[c][i]["value"] for i in old} for c in D2B}
    top = sorted(common, key=lambda i: -(got["d2b0"][i]["value"] - got["d2b1"][i]["value"]))[:5]
    res = dict(n1=g["n1"], sha256=g["sha256"], spots=len(common), final=final, mean=m, ci=[lo, hi], z=D2B_Z,
               sd=float(d.std(ddof=1)), oracle_error=ORACLE_ERROR, margin=D2B_MARGIN,
               non_inferior=hi + ORACLE_ERROR <= D2B_MARGIN if final else None,
               superior=hi + ORACLE_ERROR < 0 if final else None,
               old40=d2b_summary(d[at < D2B_CARRIED]) if (at < D2B_CARRIED).sum() >= 2 else None,
               new120=d2b_summary(d[at >= D2B_CARRIED]) if (at >= D2B_CARRIED).sum() >= 2 else None,
               largest=[dict(spot=i, d=got["d2b0"][i]["value"] - got["d2b1"][i]["value"], pot=spots[i]["pot"],
                             board=d2b_board_class(spots[i]["board"]), old=i < D2B_CARRIED) for i in top],
               loss_old40={c: dict(spots=len(old), mean=float(np.mean(list(loss[c].values()))) if old else None,
                                   top=sorted(loss[c].values(), reverse=True)[:3]) for c in D2B},
               per_spot={int(i): float(x) for i, x in zip(common, d)},
               isomorphism={c: sum(1 for i in common if got[c][i]["isomorphism"]) for c in D2B},
               hours={c: sum(r["sec"] for r in got[c].values() if not r.get("carried")) / 3600 for c in D2B},
               strategies={c: {k: json.loads((out / f"m1-x-{c}.json").read_text()).get(k)
                               for k in ("solves", "symmetric", "drift")} for c in D2B})
    verdict = (("PASS" if res["non_inferior"] else "FAIL") if final else f"interim ({len(common)}/{len(spots)} spots, no verdict)")
    print(f"D2b Gate B: Loss(X_N1) - Loss(X_N0) = {100 * m:+.3f}% pot, CI (z {D2B_Z}) [{100 * lo:+.3f}, {100 * hi:+.3f}] on "
          f"{len(common)} spots; " + (f"upper end + e = {100 * (hi + ORACLE_ERROR):+.3f}% (<= {100 * D2B_MARGIN:.2f}%): " if final else "")
          + verdict)
    return res


def cmd_report_d2b(a) -> int:
    res = d2b_report(Path(a.out))
    if res is None:
        return 2
    D2B_RECORDS.mkdir(parents=True, exist_ok=True)
    (D2B_RECORDS / "m1-report.json").write_text(json.dumps(res, indent=1))
    return 0 if res["non_inferior"] else 1


def d3_mode(out: Path) -> str:
    """`carry`: X_N1's rows and strategies are runs/flop-eval-d2b's (spots 0-1 made again and compared); `resolve`: a re-made
    spot differed, so all of X_N1 is made and solved again (records/d3/PREREG.md 5.5). The mode is the failure record's existence;
    a record is refused unless it names a re-made spot and the build the carried rows were solved under (a failure under
    another build is not the registered fallback)."""
    path = out / "d3-remake-failed.json"
    if not path.exists():
        return "carry"
    carry_path = out / "d3-carry.json"
    if not carry_path.exists():
        raise SystemExit(f"{out}: {path.name} without d3-carry.json: run d3-init")
    try:
        rec, want = json.loads(path.read_text()), json.loads(carry_path.read_text())["engine"]
    except (ValueError, KeyError):
        raise SystemExit(f"{path} or d3-carry.json is not readable") from None
    if not isinstance(rec, dict) or type(rec.get("spot")) is not int or rec["spot"] not in range(D3_REMADE) or rec.get("engine") != want:
        raise SystemExit(f"{path}: not a failed re-make of spot 0-{D3_REMADE - 1} under the carried rows' build ({want}): {rec}")
    return "resolve"


def d3_gate_a(n2_sha: str | None = None) -> dict:
    """records/d3/gate-a.json, refused unless Gate A passed for N2 (`n2_sha`, when given) against N1."""
    path = D3_RECORDS / "gate-a.json"
    g = json.loads(path.read_text()) if path.exists() else {}
    sha = g.get("sha256", {})
    if not g.get("proceed") or sha.get("N1") != N1_SHA256 or not sha.get("N2") or (n2_sha is not None and sha.get("N2") != n2_sha):
        raise SystemExit(f"{path}: Gate A has not passed for this net (proceed {g.get('proceed')}, {sha}, this net {n2_sha})")
    return g


def d3_not_old(out: Path) -> None:
    if Path(out).resolve() == D3_OLD.resolve():
        raise SystemExit("--out is runs/flop-eval-d2b itself: D3 writes its own folder and only reads that one")


def d3_carry(out: Path) -> dict:
    """d3-carry.json, refused if a file it holds the sha256 of has changed since d3-init."""
    d3_not_old(out)
    carry = json.loads((out / "d3-carry.json").read_text())
    if {f: sha256(D3_OLD / f) for f in carry["files"]} != carry["files"]:
        raise SystemExit("an X_N1 file of runs/flop-eval-d2b changed after d3-init")
    if sha256(out / "m1-spots.json") != carry["new_spots"] or sha256(D3_OLD / "m1-spots.json") != carry["spots"]:
        raise SystemExit("a m1-spots.json changed after d3-init: the 160 spots are not the ones it compared")
    return carry


def d3_strategy_guard(cand: str, net: str | None, out: Path, resolve_all: bool) -> None:
    d3_not_old(out)
    if not net:
        raise SystemExit(f"{cand} needs --net")
    if not (out / "d3-carry.json").exists():
        raise SystemExit(f"{out}: run d3-init first")
    d3_carry(out)
    failed = d3_mode(out) == "resolve"
    if cand == "d3n1":
        if sha256(net) != N1_SHA256:
            raise SystemExit(f"{net} is not N1")
        d3_gate_a()
        if failed and not resolve_all:
            raise SystemExit(f"{out / 'd3-remake-failed.json'}: a re-made spot of X_N1 differed: all of X_N1 is made again, with --resolve-all")
        if resolve_all and not failed:
            raise SystemExit("--resolve-all is for after a re-made spot of X_N1 differed (d3-remake-failed.json): none did")
    else:
        if resolve_all:
            raise SystemExit("--resolve-all is X_N1's alone")
        d3_gate_a(sha256(net))
        if not (out / "m1-x-d3n1.json").exists() and not failed:  # its mode is stamped into X_N2's file
            raise SystemExit("X_N1's re-made spots 0-1 are not decided yet: make X_d3n1 (strategy --cand d3n1) before X_d3n2")


def cmd_d3_init(a) -> int:
    """The carried rows' conditions: all 160 spots bit for bit, D2b adopted N1, the engine D2b's rows were solved under,
    the old files the ones D2b wrote, every row of X_N1 solved (none carried or copied)."""
    import parciver_engine as pe

    out = Path(a.out)
    d3_not_old(out)
    new, old = (json.loads((p / "m1-spots.json").read_text()) for p in (out, D3_OLD))
    if (new["seed"], new["stakes"]) != (old["seed"], old["stakes"]) or len(new["spots"]) != D3_SPOTS or new["spots"] != old["spots"]:
        raise SystemExit(f"{out}: not the registered {D3_SPOTS} spots of seed {old['seed']} in {D3_OLD}, bit for bit: D3 stops here (PREREG 5.5)")
    d2b_gate_a(N1_SHA256)
    engine = engine_sha(pe)
    d2b = json.loads((D3_OLD / "d2b-carry.json").read_text())
    v1 = json.loads((D3_OLD / "m1-v-d2b1.json").read_text())
    if engine != d2b["engine"] or v1["inputs"].get("engine") != engine:
        raise SystemExit(f"X_N1's rows were solved under another engine build ({v1['inputs'].get('engine')}) than this one ({engine}): "
                         f"they are not carried (PREREG 5.5)")
    x1 = json.loads((D3_OLD / "m1-x-d2b1.json").read_text())
    if x1.get("net_sha256") != N1_SHA256 or not d2b_x_settings(x1):
        raise SystemExit(f"X_N1 (m1-x-d2b1.json) was not made with N1 under D2b's settings: {x1.get('net_sha256')}, {x1.get('seeds')}")
    files = {f: sha256(D3_OLD / f) for f in ("m1-v-d2b1.json", "m1-x-d2b1.npz", "m1-x-d2b1.json")}
    if v1["inputs"]["x"] != files["m1-x-d2b1.npz"] or v1["inputs"]["spots"] != sha256(D3_OLD / "m1-spots.json"):
        raise SystemExit("m1-v-d2b1.json was not solved from this m1-x-d2b1.npz and these spots")
    rows = v1["rows"]
    if sorted(r["spot"] for r in rows) != list(range(D3_SPOTS)) or any(r.get("carried") or r.get("copied") for r in rows):
        raise SystemExit("m1-v-d2b1.json does not hold 160 solved spots (carried or copied rows, or a spot missing)")
    res = dict(engine=engine, files=files, spots=sha256(D3_OLD / "m1-spots.json"), new_spots=sha256(out / "m1-spots.json"), compared=D3_SPOTS)
    (out / "d3-carry.json").write_text(json.dumps(res, indent=1))
    print(f"the {D3_SPOTS} spots are runs/flop-eval-d2b's bit for bit; engine {engine[:8]}; {out / 'd3-carry.json'}")
    return 0


def d3_carried_rows(out: Path, x, header: dict) -> list[dict]:
    """m1-v-d2b1.json's 160 rows, marked `carried`, refused unless the files are the ones d3-init checked, X is theirs
    and the rows were solved under this rule."""
    d3_carry(out)
    old_x = np.load(D3_OLD / "m1-x-d2b1.npz")
    if not all(same_x(x, old_x, i) for i in range(D3_SPOTS)):
        raise SystemExit("X_N1's strategies are not m1-x-d2b1.npz's")
    f = json.loads((D3_OLD / "m1-v-d2b1.json").read_text())
    if any(json.dumps(f.get(k)) != json.dumps(header[k]) for k in ("menus", "rule", "precision")):
        raise SystemExit("m1-v-d2b1.json was solved under another rule, menus or precision than this one")
    rows = f["rows"]
    if sorted(r["spot"] for r in rows) != list(range(D3_SPOTS)):
        raise SystemExit("m1-v-d2b1.json does not hold the 160 spots")
    return [dict(r, carried="m1-v-d2b1") for r in sorted(rows, key=lambda r: r["spot"])]


def d3_guards(out: Path, got: dict, heads: dict) -> dict:
    """Everything the report rests on, or SystemExit. Returns gate-a.json."""
    g = d3_gate_a()
    carry = d3_carry(out)
    mode = d3_mode(out)
    same_game(out, heads)
    engines = {h["inputs"]["engine"] for h in heads.values()}
    if (engines != {carry["engine"]}) if mode == "carry" else (len(engines) != 1):
        raise SystemExit("the two candidates' rows were not solved under one engine build (the carried rows' build, in carry mode)")
    h1 = heads["d3n1"]["inputs"]
    if h1.get("mode") != mode or h1.get("carry") != sha256(out / "d3-carry.json") or (
            mode == "resolve" and h1.get("failed") != sha256(out / "d3-remake-failed.json")):
        raise SystemExit("X_N1's rows were solved under another mode, or d3-carry.json / d3-remake-failed.json changed since")
    metas = {}
    for c, want in (("d3n1", N1_SHA256), ("d3n2", g["sha256"]["N2"])):
        meta = metas[c] = json.loads((out / f"m1-x-{c}.json").read_text())
        if meta.get("net_sha256") != want or sha256(REPO / meta["net"]) != want:
            raise SystemExit(f"X_{c} was not made with its registered net, or that net file has changed since")
        if not d2b_x_settings(meta):
            raise SystemExit(f"X_{c} was made with other seeds or search settings: {[meta.get(k) for k in ('seeds', 'flop_iters', 'gamma')]}")
    if any(m.get("mode") != mode for m in metas.values()):
        raise SystemExit("an X file was made under another mode than the one now in force")
    x_rows = {r["spot"]: r for r in metas["d3n1"]["spots"]}
    if mode == "carry":
        old_rows = {r["spot"]: r for r in json.loads((D3_OLD / "m1-v-d2b1.json").read_text())["rows"]}
        old_x, x1 = np.load(D3_OLD / "m1-x-d2b1.npz"), np.load(out / "m1-x-d3n1.npz")
        for i, r in got["d3n1"].items():
            if r.get("carried") != "m1-v-d2b1" or r.get("copied"):
                raise SystemExit(f"X_d3n1 spot {i}: not a carried row where the registration says carried")
            if {k: v for k, v in r.items() if k != "carried"} != old_rows[i] or not same_x(x1, old_x, i):
                raise SystemExit(f"X_d3n1 spot {i}: no longer m1-v-d2b1.json's row, or m1-x-d2b1.npz's arrays")
        if any((x_rows[i].get("remade_equal") is not True) if i < D3_REMADE else (x_rows[i].get("carried") != "m1-x-d2b1.npz")
               for i in range(D3_SPOTS)):
            raise SystemExit("X_d3n1: spots 0-1 were not made again and found equal, or the others were not carried")
    else:
        if any(r.get("carried") or r.get("copied") for r in got["d3n1"].values()) or any(
                r.get("carried") or r.get("remade_equal") for r in x_rows.values()):
            raise SystemExit("X_d3n1 (mode resolve) has a carried or copied row: every spot is made and solved again")
    if any(r.get("carried") or r.get("copied") for r in got["d3n2"].values()):
        raise SystemExit("X_d3n2 has a carried or copied row: every spot is solved")
    return g


def d3_report(out: Path) -> dict | None:
    """Gate B: D = Loss(X_N2) - Loss(X_N1) = V1 - V2 per spot, in pots (V1 = X_d3n1's row, V2 = X_d3n2's). Pass iff the CI's
    upper end + e <= the margin, with all 160 in. Reported: the original M1 spots (the first 40) and the other 120 apart, the
    five largest D, and on the first 40 (the only spots with a V*) each X's Loss, to say whose the tail is."""
    got, heads = {}, {}
    for c in D3:
        p = out / f"m1-v-{c}.json"
        if not p.exists():
            print(f"D3 Gate B: {p.name} is not there yet")
            return None
        f = json.loads(p.read_text())
        got[c] = {r["spot"]: r for r in f.pop("rows")}
        heads[c] = f
    g = d3_guards(out, got, heads)
    spots = json.loads((out / "m1-spots.json").read_text())["spots"]
    common = sorted(set(got["d3n1"]) & set(got["d3n2"]))
    if len(common) < 2:
        print(f"D3 Gate B: {len(common)} spot(s) in both, nothing to report yet")
        return None
    d = np.array([got["d3n1"][i]["value"] - got["d3n2"][i]["value"] for i in common])
    m, lo, hi = paired(d, D2B_Z)
    final = len(common) == len(spots) == D3_SPOTS
    at = np.array(common)
    star = json.loads((D2B_OLD / "m1-v-star.json").read_text())
    if json.dumps(star["rule"]) != json.dumps(heads["d3n1"]["rule"]) or star["inputs"]["spots"] != sha256(D2B_OLD / "m1-spots.json"):
        raise SystemExit("m1-v-star.json is not this rule's or these spots'")
    star = {r["spot"]: r for r in star["rows"]}
    old = [i for i in common if i < D2B_CARRIED]
    loss = {c: {i: star[i]["value"] - got[c][i]["value"] for i in old} for c in D3}
    top = sorted(common, key=lambda i: -(got["d3n1"][i]["value"] - got["d3n2"][i]["value"]))[:5]
    res = dict(n2=g["n2"], sha256=g["sha256"], mode=d3_mode(out), spots=len(common), final=final, mean=m, ci=[lo, hi], z=D2B_Z,
               sd=float(d.std(ddof=1)), oracle_error=ORACLE_ERROR, margin=D2B_MARGIN,
               non_inferior=hi + ORACLE_ERROR <= D2B_MARGIN if final else None,
               superior=hi + ORACLE_ERROR < 0 if final else None,
               old40=d2b_summary(d[at < D2B_CARRIED]) if (at < D2B_CARRIED).sum() >= 2 else None,
               new120=d2b_summary(d[at >= D2B_CARRIED]) if (at >= D2B_CARRIED).sum() >= 2 else None,
               largest=[dict(spot=i, d=got["d3n1"][i]["value"] - got["d3n2"][i]["value"], pot=spots[i]["pot"],
                             board=d2b_board_class(spots[i]["board"]), old=i < D2B_CARRIED) for i in top],
               loss_old40={c: dict(spots=len(old), mean=float(np.mean(list(loss[c].values()))) if old else None,
                                   top=sorted(loss[c].values(), reverse=True)[:3]) for c in D3},
               per_spot={int(i): float(x) for i, x in zip(common, d)},
               isomorphism={c: sum(1 for i in common if got[c][i]["isomorphism"]) for c in D3},
               hours={c: sum(r["sec"] for r in got[c].values() if not r.get("carried")) / 3600 for c in D3},
               strategies={c: {k: json.loads((out / f"m1-x-{c}.json").read_text()).get(k)
                               for k in ("solves", "symmetric", "drift")} for c in D3})
    verdict = (("PASS" if res["non_inferior"] else "FAIL") if final else f"interim ({len(common)}/{len(spots)} spots, no verdict)")
    print(f"D3 Gate B: Loss(X_N2) - Loss(X_N1) = {100 * m:+.3f}% pot, CI (z {D2B_Z}) [{100 * lo:+.3f}, {100 * hi:+.3f}] on "
          f"{len(common)} spots; " + (f"upper end + e = {100 * (hi + ORACLE_ERROR):+.3f}% (<= {100 * D2B_MARGIN:.2f}%): " if final else "")
          + verdict)
    return res


def cmd_report_d3(a) -> int:
    d3_not_old(Path(a.out))
    res = d3_report(Path(a.out))
    if res is None:
        return 2
    D3_RECORDS.mkdir(parents=True, exist_ok=True)
    (D3_RECORDS / "m1-report.json").write_text(json.dumps(res, indent=1))
    return 0 if res["non_inferior"] else 1


def same_x(x, y, i: int) -> bool:
    """Spot i's strategies equal in value and dtype in the two X files (so -0.0 equals 0.0)."""
    mine, theirs = sorted(k for k in x.files if k.startswith(f"{i}/")), sorted(k for k in y.files if k.startswith(f"{i}/"))
    return mine == theirs and all(x[k].dtype == y[k].dtype and np.array_equal(x[k], y[k]) for k in mine)


def paired(d: np.ndarray, z: float = 1.96) -> tuple[float, float, float]:
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    return m, m - z * se, m + z * se


def same_game(out: Path, heads: dict) -> None:
    """Refuse to pair candidates' rows solved under other settings, from other
    spots, or from an X that has since been regenerated (files with recorded
    inputs; an older file without them is checked on its settings alone)."""
    bad = [k for k in ("menus", "rule", "precision") if len({json.dumps(h.get(k)) for h in heads.values()}) > 1]
    if len({h["inputs"]["spots"] for h in heads.values() if "inputs" in h}) > 1:
        bad.append("spots")
    for c, h in heads.items():
        x = h.get("inputs", {}).get("x")
        if x is not None and (not (out / f"m1-x-{c}.npz").exists() or sha256(out / f"m1-x-{c}.npz") != x):
            bad.append(f"m1-x-{c}.npz")
    if bad:
        raise SystemExit(f"the m1-v files differ in {bad}: their rows cannot be paired")


def cmd_report(a) -> int:
    out = Path(a.out)
    got, heads = {}, {}
    for c in CANDIDATES:
        p = out / f"m1-v-{c}.json"
        if p.exists():
            f = json.loads(p.read_text())
            got[c] = {r["spot"]: r for r in f.pop("rows")}
            heads[c] = f
    same_game(out, {c: h for c, h in heads.items() if c not in ("sym", "orbit", "symcheck", "leafnet")})  # FLOPSEARCH's
    spots = json.loads((out / "m1-spots.json").read_text())["spots"]
    report, flopsym = {}, {}  # FLOPSEARCH's (m1-report.json, unchanged) and FLOP-SYM's (m1-report-flopsym.json)
    if "bp" in got and "net" in got:
        common = sorted(set(got["bp"]) & set(got["net"]))
        d = np.array([got["bp"][i]["value"] - got["net"][i]["value"] for i in common])
        slack = np.array([(got["bp"][i]["hi"] - got["bp"][i]["lo"] + got["net"][i]["hi"] - got["net"][i]["lo"]) / 2
                          for i in common])
        m, lo, hi = paired(d)
        report["primary"] = dict(spots=len(common), mean=m, ci=[lo, hi], oracle_error=ORACLE_ERROR,
                                 below_zero=hi + ORACLE_ERROR < 0, mean_bracket_halfwidth=float(slack.mean()))
        print(f"M1: Loss(X_net) - Loss(X_bp) = {100 * m:+.3f}% pot, 95% CI [{100 * lo:+.3f}, {100 * hi:+.3f}] "
              f"on {len(common)} spots; upper end + e ({100 * ORACLE_ERROR:.1f}%) = {100 * (hi + ORACLE_ERROR):+.3f}%; "
              f"the difference's oracle bracket half-width {100 * slack.mean():.3f}% (mean)")
    if "star" in got:
        for c in ("bp", "net", "eq", "sym", "orbit", "symcheck", "leafnet"):
            if c in got:
                common = sorted(set(got["star"]) & set(got[c]))
                if len(common) < 2:
                    continue
                loss = np.array([got["star"][i]["value"] - got[c][i]["value"] for i in common])
                m, lo, hi = paired(loss)
                if c in ("bp", "net", "eq"):
                    report[f"loss_{c}"] = dict(spots=len(common), mean=m, ci=[lo, hi])
                elif c in ("sym", "orbit"):
                    flopsym[f"loss_{c}"] = dict(spots=len(common), mean=m, ci=[lo, hi])
                print(f"  Loss(X_{c}) = {100 * m:+.3f}% pot [{100 * lo:+.3f}, {100 * hi:+.3f}] on {len(common)} spots")
    (out / "m1-report.json").write_text(json.dumps(report, indent=1))  # FLOPSEARCH's, before FLOP-SYM's checks
    same_game(out, {c: h for c, h in heads.items() if c not in ("symcheck", "leafnet")})  # FLOP-SYM's (LEAF-NET: its own)
    for c in ("sym", "orbit"):
        if c in got and "net" in got:
            meta = json.loads((out / f"m1-x-{c}.json").read_text())
            if meta.get("net") != json.loads((out / "m1-x-net.json").read_text()).get("net") or meta.get("net_sha256") != sha256(Path(__file__).resolve().parents[1] / meta["net"]):
                raise SystemExit(f"X_{c} was made with another net than X_net's (path), or the net file has changed since")
            common = sorted(set(got[c]) & set(got["net"]))
            if len(common) < 2:
                print(f"FLOP-SYM {c}: {len(common)} spot(s) solved, nothing to report yet")
                continue
            copied = [i for i in common if got[c][i].get("copied")]
            if copied:  # re-checked now: the net files unchanged, each copied spot's X still bit-identical, the row the same
                ins = heads[c]["inputs"]
                x_c, x_net = np.load(out / f"m1-x-{c}.npz"), np.load(out / "m1-x-net.npz")
                if (ins["x_net"] != sha256(out / "m1-x-net.npz") or ins["v_net"] != sha256(out / "m1-v-net.json")
                        or not all(rainbow(spots[i]["board"]) and same_x(x_c, x_net, i)
                                   and {k: v for k, v in got[c][i].items() if k != "copied"} == got["net"][i] for i in copied)):
                    raise SystemExit(f"X_{c}: a copied rainbow row no longer matches X_net's files")
            d = np.array([got["net"][i]["value"] - got[c][i]["value"] for i in common])  # Loss(X_c) - Loss(X_net)
            m, lo, hi = paired(d, SYM_Z)
            final = len(common) == len(spots)  # a verdict only on all 40 spots: nothing is decided at an interim look
            flopsym[f"flopsym_{c}"] = dict(spots=len(common), final=final, mean=m, ci=[lo, hi], z=SYM_Z, oracle_error=ORACLE_ERROR,
                                           margin=SYM_MARGIN, non_inferior=hi + ORACLE_ERROR <= SYM_MARGIN if final else None,
                                           superior=hi + ORACLE_ERROR < 0 if final else None, copied=copied,
                                           isomorphism=sum(1 for i in common if got[c][i]["isomorphism"]),
                                           hours=sum(got[c][i]["sec"] for i in common if not got[c][i].get("copied")) / 3600,
                                           solves=meta.get("solves"), symmetric=meta.get("symmetric"), drift=meta.get("drift"))
            verdict = (("PASS" if hi + ORACLE_ERROR <= SYM_MARGIN else "FAIL") if final
                       else f"interim ({len(common)}/{len(spots)} spots, no verdict)")
            rule = f"upper end + e = {100 * (hi + ORACLE_ERROR):+.3f}% (non-inferior if <= {100 * SYM_MARGIN:.2f}%): " if final else ""
            print(f"FLOP-SYM {c}: Loss(X_{c}) - Loss(X_net) = {100 * m:+.3f}% pot, CI (z {SYM_Z}) [{100 * lo:+.3f}, {100 * hi:+.3f}] "
                  f"on {len(common)} spots; {rule}{verdict}; symmetric solves {meta.get('symmetric')}/"
                  f"{meta.get('solves')}, largest G-average change {meta.get('drift')}")
    if flopsym:
        (out / "m1-report-flopsym.json").write_text(json.dumps(flopsym, indent=1))
    leaf = leafnet_report(out, got, heads, spots)
    if leaf is not None:
        if "star" in got:
            common = sorted(set(got["star"]) & set(got["leafnet"]))
            if len(common) >= 2:
                m, lo, hi = paired(np.array([got["star"][i]["value"] - got["leafnet"][i]["value"] for i in common]))
                leaf["loss_leafnet"] = dict(spots=len(common), mean=m, ci=[lo, hi])
        LEAF_RECORDS.mkdir(parents=True, exist_ok=True)
        (LEAF_RECORDS / "m1-report-leafnet.json").write_text(json.dumps(leaf, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs" / "flop-eval"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--seed", type=int, default=M1_SEED)
    r.add_argument("--spots", type=int, default=40)
    s = sub.add_parser("strategy")
    s.add_argument("--cand", choices=CANDIDATES[1:] + D2B + D3, required=True)
    s.add_argument("--net")
    s.add_argument("--resolve-all", action="store_true", help="d3n1 only, after d3-remake-failed.json: make X_N1 again in full")
    s.add_argument("--seeds", help="translation seeds, averaged (default 0,1,2,3 for bp, 0 for a search)")
    v = sub.add_parser("solve")
    v.add_argument("--cand", choices=CANDIDATES + D2B + D3, required=True)
    v.add_argument("--threads", type=int, default=12)  # leaves the generator's feeder threads
    v.add_argument("--spots", type=lambda t: {int(x) for x in t.split(",")}, default=None)
    sub.add_parser("report")
    sub.add_parser("symcheck-init")
    sub.add_parser("enginecheck")
    sub.add_parser("d2b-init")
    sub.add_parser("report-d2b")
    sub.add_parser("d3-init")
    sub.add_parser("report-d3")
    a = ap.parse_args()
    return {"record": cmd_record, "strategy": cmd_strategy, "solve": cmd_solve, "report": cmd_report,
            "symcheck-init": cmd_symcheck_init, "enginecheck": cmd_enginecheck, "d2b-init": cmd_d2b_init,
            "report-d2b": cmd_report_d2b, "d3-init": cmd_d3_init, "report-d3": cmd_report_d3}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
