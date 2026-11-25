# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""R-MAPSS Step 2 — the inter-opponent interaction KERNEL (RMAPSS.md §34).

The research bet (hypothesis H2): against ADAPTIVE opponents, modeling
inter-opponent interaction (features B/C/D/O) earns held-out EV above the best
STATIC-model exploiter (A2.5 = myopic exact-BR-refit each block). Refinement R1
makes this the ONLY regime where a kernel can beat exact BR — vs fixed opponents
BR is already the ceiling.

Everything here is EXACT (full traversal) and game-agnostic (runs on
``KuhnPokerN(3)`` and ``LeducHoldemN(3)``). The pre-registered contract is §34.

Design (all controllers share ONE finite Hero menu, so they are strictly
comparable; §34.3):

* **The lifted game M(ρ):** ``m`` blocks; block EV is exact. Between blocks the
  opponents run the locked rule ρ (§34.a) — DENSE lagged-best-response: each
  opponent seat j sets its next policy to an ε-smoothed exact BR to the CURRENT
  block strategies of ALL other seats (Hero + co-opponent). Realisation always
  uses this true dense ρ.
* **Controllers = depth-d receding-horizon MPC over the menu**, differing only in
  the adaptation MODEL used for planning:
    - **A2.5** (baseline): depth-1 ⇒ picks the menu policy maximising IMMEDIATE EV
      ⇒ the exact myopic BR. No shaping.
    - **A4** (cascade, STAR model): depth-d planning with a model that has Hero→opp
      edges only (co-opponents frozen ⇒ no opp→opp propagation).
    - **A5** (cascade, DENSE interaction graph): depth-d planning with the true
      dense model (opp→opp edges). ``A5 − A4`` is the exact signature of the
      multiplayer-externality edge (the whole point of choosing ρ dense).
    - **ρ-ceiling**: A5's dense model at FULL horizon with optimal (not truncated)
      lookahead ⇒ the achievable optimal exploitation of ρ over the menu.
* **§28 controls:** (a) realised EV ≤ Σ_t BR_value(P_t) along each controller's own
  trajectory (a hard mathematical bound — a violation is a bug); (b) A4 plans with
  a WRONG model, so its planned window value exceeds its realised value — a
  measurable hallucination signal that A5 (correct model) does not show.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Hashable

from ..games.base import Game
from ..games.kuhn_multi import KuhnPokerN
from ..games.leduc_multi import LeducHoldemN
from .fast_tree import FastTree

Policy = dict[str, list[float]]

CHECK_CALL, BET_RAISE = "c", "r"  # 'b' is Kuhn's bet token; handled generically below


# ==========================================================================
# infoset structure (per seat): keys and the legal-action tokens at each
# ==========================================================================

def collect_seat_infosets(game: Game) -> tuple[dict[int, list[str]], dict[str, tuple]]:
    """Return (keys_by_seat, actions_by_key) over the whole game tree."""
    keys_by_seat: dict[int, list[str]] = defaultdict(list)
    actions_by_key: dict[str, tuple] = {}
    seen: set[str] = set()

    def walk(s: Hashable) -> None:
        if game.is_terminal(s):
            return
        if game.is_chance(s):
            for a, _ in game.chance_outcomes(s):
                walk(game.next_state(s, a))
            return
        k = game.infoset_key(s)
        if k not in seen:
            seen.add(k)
            acts = tuple(game.legal_actions(s))
            keys_by_seat[game.current_player(s)].append(k)
            actions_by_key[k] = acts
        for a in game.legal_actions(s):
            walk(game.next_state(s, a))

    walk(game.initial_state())
    return keys_by_seat, actions_by_key


def _uniform(n: int) -> list[float]:
    return [1.0 / n] * n


# ==========================================================================
# archetype policies (used for opponent inits AND the Hero shaping menu)
# ==========================================================================

def archetype_policy(keys: list[str], actions_by_key: dict[str, tuple], kind: str,
                     *, eq: Policy | None = None) -> Policy:
    """A fixed policy over ``keys`` following a behavioural archetype. Works for
    Kuhn (tokens c/b/f) and Leduc (tokens c/r/f) by inspecting the action tuple."""
    pol: Policy = {}
    for k in keys:
        acts = actions_by_key[k]
        na = len(acts)
        facing_bet = "f" in acts  # fold is only legal when facing a bet
        if kind == "eq":
            pol[k] = list(eq[k]) if eq and k in eq else _uniform(na)
            continue
        if kind == "uniform":
            pol[k] = _uniform(na)
            continue
        # index helpers
        def idx(tok: str) -> int | None:
            return acts.index(tok) if tok in acts else None
        aggressive = idx(BET_RAISE) if BET_RAISE in acts else idx("b")
        callcheck = idx(CHECK_CALL)
        fold = idx("f")
        oh = [0.0] * na
        if kind == "passive" or kind == "station":   # never fold, never raise
            oh[callcheck] = 1.0
        elif kind == "aggro" or kind == "maniac":     # bet/raise if possible, else call
            oh[aggressive if aggressive is not None else callcheck] = 1.0
        elif kind == "nit":                            # fold to bets, never raise
            oh[fold if (facing_bet and fold is not None) else callcheck] = 1.0
        else:
            raise ValueError(f"unknown archetype {kind!r}")
        pol[k] = oh
    return pol


def complete(game_keys: list[str], actions_by_key: dict[str, tuple], partial: Policy) -> Policy:
    """Fill any missing (off-path) keys with uniform so a profile is total."""
    out = dict(partial)
    for k in game_keys:
        if k not in out:
            out[k] = _uniform(len(actions_by_key[k]))
    return out


# ==========================================================================
# ρ dynamics + block evaluation (with interning + caches for Leduc speed)
# ==========================================================================

@dataclass
class Lifted:
    """The lifted multi-hand game under ρ for a fixed game + Hero seat.

    Holds the precompiled ``FastTree`` (10x the game-driven walk), the seat
    structure, the CFR equilibrium reference, the Hero menu, and memo caches so the
    many exact evaluations are computed once per distinct (policy, profile). The
    tree, eq, menu and eps-INDEPENDENT caches (intern/ev/br) are shared across
    opponent families; the eps-DEPENDENT ``_step_cache`` is per-family."""
    ft: FastTree
    hero: int
    eps: float
    keys_by_seat: dict[int, list[str]]
    actions_by_key: dict[str, tuple]
    eq: Policy
    hero_menu: dict[str, object]  # name -> Policy or "br" sentinel

    _intern: dict = field(default_factory=dict)         # canon-repr -> id
    _by_id: dict = field(default_factory=dict)          # id -> Policy
    _ev_cache: dict = field(default_factory=dict)       # (hero_id, opp_id) -> float
    _br_cache: dict = field(default_factory=dict)       # opp_id -> (policy_id, value)
    _step_cache: dict = field(default_factory=dict)     # (opp_id, hero_id, mode, freeze_id) -> opp_id

    # -- interning --
    def intern(self, pol: Policy) -> int:
        canon = tuple(sorted((k, tuple(round(p, 9) for p in v)) for k, v in pol.items()))
        pid = self._intern.get(canon)
        if pid is None:
            pid = len(self._intern)
            self._intern[canon] = pid
            self._by_id[pid] = pol
        return pid

    def pol(self, pid: int) -> Policy:
        return self._by_id[pid]

    @property
    def opp_seats(self) -> list[int]:
        return [s for s in self.keys_by_seat if s != self.hero]

    @property
    def opp_keys(self) -> list[str]:
        return [k for s in self.opp_seats for k in self.keys_by_seat[s]]

    # -- block reward: Hero EV when Hero plays hero_pol and opps play opp_profile --
    def block_ev(self, hero_id: int, opp_id: int) -> float:
        key = (hero_id, opp_id)
        v = self._ev_cache.get(key)
        if v is None:
            profile = {**self.pol(hero_id), **self.pol(opp_id)}
            v = self.ft.on_policy_value(profile, self.hero)
            self._ev_cache[key] = v
        return v

    # -- exact myopic BR to a fixed opp profile (the A2.5 move) + its value --
    def br_now(self, opp_id: int) -> tuple[int, float]:
        cached = self._br_cache.get(opp_id)
        if cached is None:
            opp = self.pol(opp_id)
            br = self.ft.best_response_strategy(opp, self.hero)
            br = complete(self.keys_by_seat[self.hero], self.actions_by_key, br)
            bid = self.intern(br)
            val = self.ft.best_response_value(opp, self.hero)
            self._br_cache[opp_id] = (bid, val)
            cached = (bid, val)
        return cached

    # -- one opponent-adaptation step (dense = true ρ; star = A4's wrong model) --
    def _smoothed_br(self, seat: int, others: Policy) -> Policy:
        br = self.ft.best_response_strategy(others, seat)
        out: Policy = {}
        for k in self.keys_by_seat[seat]:
            acts = self.actions_by_key[k]
            na = len(acts)
            oh = br.get(k)
            if oh is None:
                out[k] = _uniform(na)
            else:
                out[k] = [(1.0 - self.eps) * oh[i] + self.eps / na for i in range(na)]
        return out

    def step(self, opp_id: int, hero_id: int, mode: str, freeze_id: int | None) -> int:
        """Next opponent profile after one block. ``mode`` 'dense' uses the current
        co-opponent policies (true ρ, opp->opp edges); 'star' freezes co-opponents
        at ``freeze_id`` (Hero->opp edges only)."""
        ck = (opp_id, hero_id, mode, freeze_id if mode == "star" else None)
        nid = self._step_cache.get(ck)
        if nid is not None:
            return nid
        cur = self.pol(opp_id)
        hero = self.pol(hero_id)
        coopp_src = self.pol(freeze_id) if (mode == "star" and freeze_id is not None) else cur
        new_opp: Policy = {}
        for j in self.opp_seats:
            others: Policy = dict(hero)
            for k in self.opp_seats:
                if k == j:
                    continue
                src = coopp_src if mode == "star" else cur
                for key in self.keys_by_seat[k]:
                    others[key] = src[key]
            new_opp.update(self._smoothed_br(j, others))
        nid = self.intern(new_opp)
        self._step_cache[ck] = nid
        return nid

    # -- resolve a menu entry to a concrete Hero policy id at a given opp state --
    def menu_ids(self, opp_id: int) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for name, val in self.hero_menu.items():
            if val == "br":
                hid = self.br_now(opp_id)[0]
            else:
                hid = self.intern(val)  # fixed policy (interned once)
            out.append((name, hid))
        return out


# ==========================================================================
# depth-d MPC planner (shared by A2.5 / A4 / A5 / ceiling)
# ==========================================================================

class Planner:
    """Finite-horizon lookahead over the Hero menu with a given adaptation model.

    ``value(opp_id, depth)`` = best achievable cumulative Hero EV over the next
    ``depth`` blocks using the model (dense/star). depth-1 = the immediate-EV
    maximiser = the myopic BR (A2.5). depth=m with dense model & no truncation =
    the optimal exploitation of ρ over the menu (the ceiling)."""

    def __init__(self, lifted: Lifted, mode: str, freeze_id: int | None = None):
        self.L = lifted
        self.mode = mode
        self.freeze_id = freeze_id
        self._memo: dict = {}

    def value(self, opp_id: int, depth: int) -> tuple[float, int]:
        """Return (cumulative value, best first Hero policy id)."""
        if depth <= 0:
            return 0.0, -1
        memo_key = (opp_id, depth)
        hit = self._memo.get(memo_key)
        if hit is not None:
            return hit
        best_val = -1e18
        best_hid = -1
        for _name, hid in self.L.menu_ids(opp_id):
            immediate = self.L.block_ev(hid, opp_id)
            if depth > 1:
                nxt = self.L.step(opp_id, hid, self.mode, self.freeze_id)
                future, _ = self.value(nxt, depth - 1)
            else:
                future = 0.0
            tot = immediate + future
            if tot > best_val:
                best_val, best_hid = tot, hid
        self._memo[memo_key] = (best_val, best_hid)
        return best_val, best_hid


# ==========================================================================
# realisation of a controller over the lifted game (always TRUE dense ρ)
# ==========================================================================

@dataclass
class Rollout:
    controller: str
    total_ev: float                 # sum of block EVs (realised, exact)
    per_block: list[float]
    traj_br_ceiling: float          # Σ_t BR_value(P_t) along the realised trajectory
    planned_window: float           # block-1 planner window value (for §28)
    realized_window: float          # realised EV over that same window
    depth: int

    @property
    def mean_ev(self) -> float:
        return self.total_ev / len(self.per_block)


def realise(lifted: Lifted, init_opp_id: int, m: int, controller: str, depth: int) -> Rollout:
    """Play the lifted game for ``m`` blocks. Hero picks each block with a depth-d
    receding-horizon MPC (model from the controller); opponents ALWAYS respond with
    the true dense ρ. The DENSE controllers (A2.5/A5) reuse ONE memoised planner
    across blocks (so the full-horizon rollout is a single DP); the STAR controller
    (A4) needs a fresh planner per block because its frozen co-opponent reference
    changes each block."""
    L = lifted
    mode = "star" if controller == "A4" else "dense"
    shared = Planner(L, mode) if mode == "dense" else None
    opp_id = init_opp_id
    per_block: list[float] = []
    traj_ceiling = 0.0
    planned_window = 0.0
    for t in range(m):
        d = min(depth, m - t)
        planner = shared if shared is not None else Planner(L, mode, freeze_id=opp_id)
        window_val, hid = planner.value(opp_id, d)
        if t == 0:
            planned_window = window_val
        per_block.append(L.block_ev(hid, opp_id))
        traj_ceiling += L.br_now(opp_id)[1]          # BR to this block's opp = per-block ceiling
        opp_id = L.step(opp_id, hid, "dense", None)  # TRUE dense ρ, regardless of model
    total = sum(per_block)
    win = min(depth, m)
    realized_window = sum(per_block[:win])
    return Rollout(controller, total, per_block, traj_ceiling, planned_window, realized_window, depth)


def ceiling_value(lifted: Lifted, init_opp_id: int, m: int) -> tuple[float, Rollout]:
    """Optimal exploitation of ρ over the menu: dense model, full-horizon optimal
    lookahead. Also realise the optimal sequence to get its exact per-block EVs."""
    planner = Planner(lifted, "dense")
    # replay the optimal policy (planner is optimal for the remaining horizon).
    opp_id = init_opp_id
    per_block: list[float] = []
    traj_ceiling = 0.0
    for t in range(m):
        _, hid = planner.value(opp_id, m - t)
        per_block.append(lifted.block_ev(hid, opp_id))
        traj_ceiling += lifted.br_now(opp_id)[1]
        opp_id = lifted.step(opp_id, hid, "dense", None)
    total = sum(per_block)
    ro = Rollout("ceiling", total, per_block, traj_ceiling, total, total, m)
    return total, ro


# ==========================================================================
# opponent families (adaptive: an initial profile that then runs ρ)
# ==========================================================================

@dataclass
class Family:
    label: str
    init_kinds: tuple[str, ...]   # per opponent seat, in seat order
    split: str                     # "fit" | "held"
    eps: float                     # ρ smoothing for this family


def default_families() -> list[Family]:
    """Diverse adaptive families. HELD-OUT families are never used to choose menu
    or depth (R4); the GO metric is measured on them. Distinct inits AND distinct
    ε give a transfer test (§R)."""
    return [
        # --- FIT (tune menu/depth here) ---
        Family("eq|eq",         ("eq", "eq"),           "fit",  0.10),
        Family("uniform|uniform", ("uniform", "uniform"), "fit", 0.10),
        Family("station|aggro", ("station", "aggro"),   "fit",  0.10),
        # --- HELD-OUT (GO measured here) ---
        Family("nit|station",   ("nit", "station"),     "held", 0.10),
        Family("aggro|nit",     ("aggro", "nit"),       "held", 0.10),
        Family("eq|uniform",    ("eq", "uniform"),      "held", 0.10),
        # --- transfer family: different ε, unseen init combo (§R) ---
        Family("station|nit@.2", ("station", "nit"),    "held", 0.20),
    ]


# ==========================================================================
# the experiment
# ==========================================================================

@dataclass
class FamilyResult:
    label: str
    split: str
    best_depth: int
    ev: dict[str, float]        # controller -> mean per-block realised EV (at best_depth)
    ceiling: float              # mean per-block ceiling EV (dense model, full horizon)
    upper_bound: float          # mean per-block valid upper bound (hard tripwire)
    headroom: float             # ceiling - A2.5
    a5_minus_a25: float
    a5_minus_a4: float
    a5_full_eq_ceiling: bool    # realise(A5, m) == ceiling  (planner sound + model correct)
    depths: dict[str, list[tuple[int, float]]]  # controller -> [(depth, mean_ev)]


@dataclass
class Step2Report:
    game: str
    role: str                   # "filter" (Kuhn) | "decisive" (Leduc)
    hero: int
    m: int
    depths: list[int]
    delta: float
    best_depth: int = 1
    results: list[FamilyResult] = field(default_factory=list)
    gates: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, float] = field(default_factory=dict)
    verdict: str = ""           # GO | KILL | INCONCLUSIVE (decisive) / FILTER-* (Kuhn)

    @property
    def all_pass(self) -> bool:
        return all(self.gates.values())


@dataclass
class Shared:
    """Game-level structures built ONCE and shared across all opponent families:
    the precompiled tree, seat structure, CFR eq reference, the Hero menu, and the
    eps-independent intern/ev/br caches. Only ρ smoothing (eps) varies per family."""
    ft: FastTree
    hero: int
    keys_by_seat: dict[int, list[str]]
    actions_by_key: dict[str, tuple]
    eq: Policy
    menu: dict[str, object]
    intern: dict = field(default_factory=dict)
    by_id: dict = field(default_factory=dict)
    ev_cache: dict = field(default_factory=dict)
    br_cache: dict = field(default_factory=dict)


def build_shared(game: Game, hero: int, cfr_iters: int) -> Shared:
    ft = FastTree(game)
    keys_by_seat, actions_by_key = collect_seat_infosets(game)
    eq = ft.solve_cfr(cfr_iters, "cfr+")
    eq = {k: eq.get(k, _uniform(len(actions_by_key[k]))) for k in actions_by_key}
    hero_keys = keys_by_seat[hero]
    eq_hero = complete(hero_keys, actions_by_key, {k: eq[k] for k in hero_keys})
    menu = {
        "br": "br",  # exact BR to current opp (makes depth-1 == A2.5)
        "eq": eq_hero,
        "passive": archetype_policy(hero_keys, actions_by_key, "passive"),
        "aggro": archetype_policy(hero_keys, actions_by_key, "aggro"),
        "nit": archetype_policy(hero_keys, actions_by_key, "nit"),  # tight/fold-leaning image
    }
    return Shared(ft, hero, keys_by_seat, actions_by_key, eq, menu)


def lifted_for(sh: Shared, eps: float) -> Lifted:
    """A per-family Lifted sharing the tree + eq-independent caches (fresh step
    cache, since ρ transitions depend on eps)."""
    return Lifted(sh.ft, sh.hero, eps, sh.keys_by_seat, sh.actions_by_key, sh.eq, sh.menu,
                  _intern=sh.intern, _by_id=sh.by_id, _ev_cache=sh.ev_cache, _br_cache=sh.br_cache)


def _init_opp_profile(lifted: Lifted, fam: Family) -> int:
    """Build a family's initial opponent profile (block 1) from its per-seat kinds."""
    prof: Policy = {}
    for idx, seat in enumerate(lifted.opp_seats):
        kind = fam.init_kinds[idx] if idx < len(fam.init_kinds) else "uniform"
        prof.update(archetype_policy(lifted.keys_by_seat[seat], lifted.actions_by_key,
                                     kind, eq=lifted.eq))
    return lifted.intern(prof)


def run_step2(*, game_name: str = "3p-kuhn", hero: int = 0, m: int = 8,
              depths: tuple[int, ...] = (1, 2, 3), delta: float = 0.02,
              cfr_iters: int = 6000, on_family=None) -> Step2Report:
    """Run the Step-2 kernel experiment + apply the pre-registered go/no-go (RMAPSS
    Sec.34). Per-block mean EV for A2.5, A4(depth), A5(depth), the rho-ceiling and the
    valid upper bound, on every adaptive family. The planning depth is selected on
    FIT families and applied to HELD-OUT ones (no peeking, R4). Kuhn is the within-
    round FILTER (a NO there is not a kill, R3); Leduc is DECISIVE.
    """
    if game_name == "3p-kuhn":
        game: Game = KuhnPokerN(3)
        role = "filter"
    elif game_name == "3p-leduc":
        game = LeducHoldemN(3)
        role = "decisive"
    else:
        raise ValueError(f"unknown game {game_name!r}")

    families = default_families()
    rep = Step2Report(game=game_name, role=role, hero=hero, m=m,
                      depths=list(depths), delta=delta)

    # Build the tree + eq + menu + shared caches ONCE (families vary only in eps/init).
    shared = build_shared(game, hero, cfr_iters)

    # ---- pass 1: realise every controller at every depth, per family ----
    raw = []  # (fam, a25, {d:a4}, {d:a5}, a5_full_ro, ceil_ro, ub_mean)
    for fam in families:
        lifted = lifted_for(shared, fam.eps)
        init = _init_opp_profile(lifted, fam)
        a25 = realise(lifted, init, m, "A25", 1)
        a4 = {d: realise(lifted, init, m, "A4", d) for d in depths}
        a5 = {d: realise(lifted, init, m, "A5", d) for d in depths}
        _, ceil_ro = ceiling_value(lifted, init, m)
        a5_full = realise(lifted, init, m, "A5", m)  # full-horizon A5 == ceiling (consistency)
        ub = max(a25.traj_br_ceiling, ceil_ro.traj_br_ceiling,
                 *(a5[d].traj_br_ceiling for d in depths)) / m
        raw.append((fam, a25, a4, a5, a5_full, ceil_ro, ub))

    # ---- select the planning depth on FIT families (max mean A5 EV) ----
    fit = [r for r in raw if r[0].split == "fit"]
    best_depth = max(depths, key=lambda d: sum(r[3][d].mean_ev for r in fit) / len(fit))
    rep.best_depth = best_depth

    # ---- pass 2: build per-family results at the selected depth ----
    for i, (fam, a25, a4, a5, a5_full, ceil_ro, ub) in enumerate(raw):
        a5b, a4b = a5[best_depth], a4[best_depth]
        fr = FamilyResult(
            label=fam.label, split=fam.split, best_depth=best_depth,
            ev={"A2.5": a25.mean_ev, "A4": a4b.mean_ev, "A5": a5b.mean_ev},
            ceiling=ceil_ro.mean_ev, upper_bound=ub,
            headroom=ceil_ro.mean_ev - a25.mean_ev,
            a5_minus_a25=a5b.mean_ev - a25.mean_ev,
            a5_minus_a4=a5b.mean_ev - a4b.mean_ev,
            a5_full_eq_ceiling=abs(a5_full.mean_ev - ceil_ro.mean_ev) < 1e-9,
            depths={"A4": [(d, a4[d].mean_ev) for d in depths],
                    "A5": [(d, a5[d].mean_ev) for d in depths]},
        )
        rep.results.append(fr)
        if on_family is not None:
            on_family(i, fam, fr, a25, a4, a5, ceil_ro)

    held = [r for r in rep.results if r.split == "held"]

    # ---- Sec.28 hard controls (must hold everywhere, or it is a bug) ----
    rep.gates["ctl: realised EV <= trajectory-BR upper bound (all)"] = all(
        max(r.ev.values()) <= r.upper_bound + 1e-9 for r in rep.results)
    rep.gates["ctl: A5 full-horizon == rho-ceiling (planner sound, model correct)"] = all(
        r.a5_full_eq_ceiling for r in rep.results)
    rep.gates["ctl: ceiling >= A2.5 (shaping never hurts the oracle)"] = all(
        r.ceiling >= r.ev["A2.5"] - 1e-9 for r in rep.results)
    rep.gates["ctl: A5 >= A2.5 (correct-model MPC never below myopic)"] = all(
        r.ev["A5"] >= r.ev["A2.5"] - 1e-9 for r in rep.results)

    # ---- go/no-go inputs (HELD-OUT only) ----
    mean_headroom = sum(r.headroom for r in held) / len(held)
    mean_a5_a25 = sum(r.a5_minus_a25 for r in held) / len(held)
    mean_a5_a4 = sum(r.a5_minus_a4 for r in held) / len(held)
    n_transfer = sum(1 for r in held if r.a5_minus_a25 > delta)
    rep.detail.update({
        "mean headroom (held)": mean_headroom,
        "mean A5-A2.5 (held)": mean_a5_a25,
        "mean A5-A4 (held)": mean_a5_a4,
        "held families beating A2.5 by delta": float(n_transfer),
        "held families": float(len(held)),
        "selected planning depth": float(best_depth),
    })

    # ---- verdict (Sec.34.4), game-role-aware ----
    controls_ok = (rep.gates["ctl: realised EV <= trajectory-BR upper bound (all)"]
                   and rep.gates["ctl: A5 full-horizon == rho-ceiling (planner sound, model correct)"])
    headroom_exists = mean_headroom > delta
    shaping_edge = mean_a5_a25 > delta          # A4/A5 vs myopic: dynamic shaping helps
    inter_opp_edge = mean_a5_a4 > 1e-9          # opp->opp edges contribute (the multiplayer bet)
    transfers = n_transfer >= 1

    if not controls_ok:
        rep.verdict = "CONTROLS-FAILED"
    elif not headroom_exists:
        rep.verdict = "INCONCLUSIVE"
    elif role == "filter":
        # Kuhn: report the shaping edge; the opp->opp edge is EXPECTED ~0 in a single
        # betting round (R3), so its absence is NOT a kill. Machinery must be sound.
        rep.verdict = "FILTER-PASS" if shaping_edge else "FILTER-FLAT"
    else:
        # Leduc: the decisive game. GO needs the multiplayer edge + transfer + depth.
        if shaping_edge and inter_opp_edge and transfers:
            rep.verdict = "GO"
        else:
            rep.verdict = "KILL"

    rep.gates[f"VERDICT[{role}] = {rep.verdict}"] = True  # the outcome, recorded (not pass/fail)
    return rep
