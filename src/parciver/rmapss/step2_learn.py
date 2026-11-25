# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""R-MAPSS Step 2 FOLLOW-UP §35.B — the NO-REGRET-LEARNER decider (last shot at H2).

§35.A (the rich-bait decider) OVERTURNED §34.5: against dense lagged-BR followers a rich
bait search finds LARGE 1-step shaping headroom, but it is DOMINANTLY single-opponent
(``A4 − A2.5``, the known H1/Stackelberg effect); the MULTIPLAYER slice ``A5 − A4`` is
certified nonzero yet SMALL (held mean +0.011 Leduc / +0.009 Kuhn, below δ=0.02, only some
families). So H2 was left "open" — not killed, not greenlit.

**The bet (§35.B.LOCK).** Under dense lagged-BR the opponent is MEMORYLESS (each block it
jumps fully to a smoothed BR), so the current opponent state summarises the future and myopic
BR is ρ-optimal (§34.5 Leduc). A **no-regret learner carries cumulative regret = MEMORY**, so
a bait's influence PERSISTS and COMPOUNDS, and Hero→A shaping propagates into B's regrets over
successive blocks — the opp→opp cascade gains a dynamical channel it structurally lacked. This
is the last pre-registered shot at a clean, exact-controlled ``A5 − A4 > δ``.

What changes vs :mod:`step2` / :mod:`step2_bait`: ONLY the opponent adaptation rule ρ and the
opponent-state abstraction. The Hero machinery (finite menu, the depth-d receding-horizon
:class:`~parciver.rmapss.step2.Planner`, ``realise``, ``ceiling_value``, the rich-bait search
``decide_state`` / coordinate ascent, ``eval_g1``) is REUSED byte-for-byte — a
:class:`LearnLifted` presents the exact same interface as ``Lifted`` (``pol`` / ``block_ev`` /
``br_now`` / ``step`` / ``menu_ids`` / ``intern``), but opponent ids are NEGATIVE and address a
continuous regret STATE (per-seat cumulative counterfactual regret) rather than a policy.

**The learner rule ρ (§35.B.LOCK item 1).** Each opponent seat *j* is a no-regret learner with
cumulative counterfactual regret ``R_j``; its block policy is ``link(R_j)`` (what it actually
plays). Between blocks it does ONE EXACT counterfactual-regret update against the CLAIRVOYANT
joint field {Hero, co-opponents} — the per-player CFR regret computed by :func:`cf_regrets` on
the FastTree arrays (``fast_tree.py`` stays byte-identical to the certified oracle; the update
is cross-checked against ``FastTree.solve_cfr`` in the tests). Because the field is clairvoyant
(exact current strategies, not sampled frequencies), ρ is a DETERMINISTIC map — load-bearing
for the exact control. Three no-regret links share ONE regret accumulation and differ only in
floor/link:

* ``rm``   (VANILLA regret-matching, primary FIT rule): ``π = relu_normalize(R)``.
* ``rm+``  (held-out transfer): ``R ← max(R + r, 0)`` each block; ``π = relu_normalize(R)``.
* ``hedge`` (held-out transfer; Hedge / multiplicative-weights, explicit step size η):
  ``π = softmax(η · R)``.

**A4 vs A5 (§35.B.LOCK item 2).** A2.5 = depth-1 exact myopic BR (no shaping). A4 (STAR) plans
each learner updating against {Hero-plan, co-opps FROZEN at plan-start} — no opp→opp
propagation in-plan. A5 (DENSE) plans against the evolving joint field (co-opps also update) —
correctly specified. Realisation ALWAYS uses the true dense clairvoyant update. ``A5 − A4``
(realised, held-out) = the exact EV of correctly anticipating the emergent cascade = the H2
signal.

**Exact control (§35.B.LOCK item 3) — finite-menu lower bound + exact per-trajectory controls,
NO discretization.** The naive §34.2 upper bound ``U`` breaks (RM's reachable regret states are
a continuum, not a finite vertex family), but the ``A5 − A4 > δ`` claim is a LOWER-BOUND /
existence claim (as §35.A), so no continuous-control upper bound — hence no discretization — is
needed. The finite-menu backward-induction value is an ACHIEVABLE lower bound (ρ deterministic
+ finite Hero menu ⇒ finite reachable tree; the DP is exact, it just stops collapsing, so m is
kept small). Soundness comes from controls that carry over UNCHANGED and EXACT: (i) realised ≤
trajectory-BR ceiling ``Σ_t BR_value(P_t)`` along each controller's own path; (ii) A5
predicted-window == realised-window (correct-model self-consistency; A4 differs — the
measurable hallucination signal); (iii) composed g1 == certified ``eval.exploitability`` oracle
per candidate (checked in tests); (iv) headroom ≥ 0; (v) immediate ≤ BR_value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


from ..games.base import Game
from ..games.kuhn_multi import KuhnPokerN
from ..games.leduc_multi import LeducHoldemN
from .fast_tree import FastTree
from .step2 import (
    Shared, build_shared, archetype_policy, complete, Planner, realise, ceiling_value,
)
from .step2_bait import (
    BaitConfig, decide_family,
)

Policy = dict[str, list[float]]
Regret = dict[int, dict[str, list[float]]]   # seat -> key -> cumulative regret vector
_TOL = 1e-9


# ==========================================================================
# no-regret links (all three share ONE regret accumulation)
# ==========================================================================

def rm_policy(regret: list[float]) -> list[float]:
    """Regret-matching: π ∝ max(regret, 0); uniform if all ≤ 0. Byte-identical to the
    ``rm`` used inside the certified ``FastTree.solve_cfr``."""
    pos = [x if x > 0.0 else 0.0 for x in regret]
    s = sum(pos)
    if s > 0.0:
        return [x / s for x in pos]
    u = 1.0 / len(regret)
    return [u] * len(regret)


def hedge_policy(regret: list[float], eta: float) -> list[float]:
    """Hedge / multiplicative-weights on cumulative regret: π ∝ exp(η·R). Numerically
    stabilised by subtracting the max."""
    m = max(regret)
    w = [math.exp(eta * (x - m)) for x in regret]
    s = sum(w)
    return [x / s for x in w]


def link_policy(rule: str, regret: list[float], eta: float) -> list[float]:
    if rule in ("rm", "rm+"):
        return rm_policy(regret)
    if rule == "hedge":
        return hedge_policy(regret, eta)
    raise ValueError(f"unknown learner rule {rule!r}")


# ==========================================================================
# exact per-player counterfactual regret over the flat tree
# ==========================================================================

def cf_regrets(ft: FastTree, profile: Policy, targets: set[int]) -> dict[int, dict[str, list[float]]]:
    """Exact instantaneous counterfactual regret r(a) = cf_reach·(v_a − v) for every
    ``target`` player at each of its infosets, given ALL players follow ``profile``
    (uniform default for a missing key). This is byte-for-byte the regret increment of
    ``FastTree.solve_cfr`` (validated against it in the tests), just taking explicit
    strategies instead of the internal RM strategy — so one synchronous update over the
    opponent seats == one certified CFR iteration for those seats."""
    n = ft.n
    reg: dict[int, dict[str, list[float]]] = {p: {} for p in targets}

    def traverse(node: int, reach: list[float], reach_c: float) -> list[float]:
        k = ft.kind[node]
        if k == 0:                       # terminal
            return list(ft.tutil[node])
        if k == 1:                       # chance
            val = [0.0] * n
            for c, p in ft.cchildren[node]:
                cv = traverse(c, reach, reach_c * p)
                for i in range(n):
                    val[i] += p * cv[i]
            return val
        player = ft.dplayer[node]
        key = ft.dkey[node]
        na = ft.nactions[node]
        ch = ft.dchildren[node]
        probs = profile.get(key)
        if probs is None:
            probs = [1.0 / na] * na
        avals: list[list[float]] = []
        val = [0.0] * n
        for a in range(na):
            cr = list(reach)
            cr[player] *= probs[a]
            cv = traverse(ch[a], cr, reach_c)
            avals.append(cv)
            for i in range(n):
                val[i] += probs[a] * cv[i]
        if player in targets:
            cf = reach_c
            for j in range(n):
                if j != player:
                    cf *= reach[j]
            d = reg[player].setdefault(key, [0.0] * na)
            vp = val[player]
            for a in range(na):
                d[a] += cf * (avals[a][player] - vp)
        return val

    traverse(ft.root, [1.0] * n, 1.0)
    return reg


# ==========================================================================
# LearnLifted — the lifted game under no-regret-learner ρ
# ==========================================================================

@dataclass
class LearnState:
    regret: Regret                    # seat -> key -> cumulative regret
    policy: Policy                    # derived opp policy (link of regret), all opp keys


class LearnLifted:
    """The multi-hand lifted game whose opponents are no-regret learners.

    Presents the same duck-typed interface the §35.A machinery expects. Hero-policy ids
    are NON-NEGATIVE (interned like ``Lifted``); opponent-STATE ids are NEGATIVE and address
    a :class:`LearnState` (continuous per-seat cumulative regret). ``pol(id)`` branches on
    the sign, so ``Planner`` / ``realise`` / ``eval_g1`` / ``decide_state`` run unchanged."""

    def __init__(self, sh: Shared, rule: str = "rm", eta: float = 1.0):
        self.ft = sh.ft
        self.hero = sh.hero
        self.keys_by_seat = sh.keys_by_seat
        self.actions_by_key = sh.actions_by_key
        self.eq = sh.eq
        self.hero_menu = sh.menu
        self.rule = rule
        self.eta = eta
        self.eps = 0.0                            # interface parity (unused by learners)

        # hero-policy interning (non-negative ids)
        self._intern: dict = {}
        self._by_id: dict = {}
        # opponent regret-state registry (negative ids: -(k+1))
        self._state_intern: dict = {}
        self._states: list[LearnState] = []
        # caches
        self._ev_cache: dict = {}                 # (hero_id, state_id) -> float
        self._br_cache: dict = {}                 # state_id -> (hero_id, value)
        self._step_cache: dict = {}               # (state_id, hero_id, mode, freeze) -> state_id

    # -- seats --
    @property
    def opp_seats(self) -> list[int]:
        return [s for s in self.keys_by_seat if s != self.hero]

    @property
    def opp_keys(self) -> list[str]:
        return [k for s in self.opp_seats for k in self.keys_by_seat[s]]

    # -- hero-policy interning (mirrors Lifted.intern) --
    def intern(self, pol: Policy) -> int:
        canon = tuple(sorted((k, tuple(round(p, 9) for p in v)) for k, v in pol.items()))
        pid = self._intern.get(canon)
        if pid is None:
            pid = len(self._intern)
            self._intern[canon] = pid
            self._by_id[pid] = pol
        return pid

    # -- opponent regret-state interning (negative ids) --
    def _derive_policy(self, regret: Regret) -> Policy:
        pol: Policy = {}
        for seat in self.opp_seats:
            for k in self.keys_by_seat[seat]:
                pol[k] = link_policy(self.rule, regret[seat][k], self.eta)
        return pol

    def intern_state(self, regret: Regret) -> int:
        canon = tuple((seat, tuple(sorted((k, tuple(round(x, 9) for x in v))
                                          for k, v in regret[seat].items())))
                      for seat in self.opp_seats)
        sid = self._state_intern.get(canon)
        if sid is not None:
            return sid
        pol = self._derive_policy(regret)
        neg = -(len(self._states) + 1)
        self._state_intern[canon] = neg
        self._states.append(LearnState(regret, pol))
        return neg

    def _state(self, sid: int) -> LearnState:
        return self._states[-sid - 1]

    def state_policy(self, sid: int) -> Policy:
        return self._state(sid).policy

    def state_regret(self, sid: int) -> Regret:
        return self._state(sid).regret

    # -- pol(): sign-branch (hero policy vs opponent-state policy) --
    def pol(self, pid: int) -> Policy:
        if pid < 0:
            return self.state_policy(pid)
        return self._by_id[pid]

    # -- block reward: Hero EV when Hero plays hero_id vs opponent state --
    def block_ev(self, hero_id: int, opp_id: int) -> float:
        key = (hero_id, opp_id)
        v = self._ev_cache.get(key)
        if v is None:
            profile = {**self.pol(hero_id), **self.state_policy(opp_id)}
            v = self.ft.on_policy_value(profile, self.hero)
            self._ev_cache[key] = v
        return v

    # -- exact myopic Hero BR to the opponent state's policy + value --
    def br_now(self, opp_id: int) -> tuple[int, float]:
        cached = self._br_cache.get(opp_id)
        if cached is None:
            opp = self.state_policy(opp_id)
            br = self.ft.best_response_strategy(opp, self.hero)
            br = complete(self.keys_by_seat[self.hero], self.actions_by_key, br)
            bid = self.intern(br)
            val = self.ft.best_response_value(opp, self.hero)
            self._br_cache[opp_id] = (bid, val)
            cached = (bid, val)
        return cached

    # -- one learner-adaptation step (dense = true ρ; star = A4's wrong model) --
    def step(self, opp_id: int, hero_id: int, mode: str, freeze_id: int | None) -> int:
        """Next opponent STATE after one block. All opp seats update simultaneously by an
        exact counterfactual-regret step. 'dense' uses each other's CURRENT strategies
        (true ρ, opp→opp edges); 'star' uses the co-opponents' FROZEN strategies from
        ``freeze_id`` (Hero→opp edges only — A4's misspecified model)."""
        ck = (opp_id, hero_id, mode, freeze_id if mode == "star" else None)
        nid = self._step_cache.get(ck)
        if nid is not None:
            return nid
        hero_pol = self.pol(hero_id)
        cur = self.state_regret(opp_id)
        cur_pol = self.state_policy(opp_id)
        froz_pol = self.state_policy(freeze_id) if (mode == "star" and freeze_id is not None) else None

        new_regret: Regret = {}
        if mode == "dense":
            # one synchronous CFR step for all opp seats against {hero, all-current}
            profile = {**hero_pol, **cur_pol}
            inc = cf_regrets(self.ft, profile, set(self.opp_seats))
        for j in self.opp_seats:
            if mode == "star":
                # seat j responds to {hero, j=current, co-opps=frozen}
                prof_j = dict(hero_pol)
                for other in self.opp_seats:
                    src = cur_pol if other == j else froz_pol
                    for key in self.keys_by_seat[other]:
                        prof_j[key] = src[key]
                inc_j = cf_regrets(self.ft, prof_j, {j})[j]
            else:
                inc_j = inc[j]
            rj: dict[str, list[float]] = {}
            for k in self.keys_by_seat[j]:
                na = len(self.actions_by_key[k])
                base = cur[j][k]
                delta = inc_j.get(k, [0.0] * na)
                if self.rule == "rm+":
                    rj[k] = [max(base[a] + delta[a], 0.0) for a in range(na)]
                else:
                    rj[k] = [base[a] + delta[a] for a in range(na)]
            new_regret[j] = rj
        nid = self.intern_state(new_regret)
        self._step_cache[ck] = nid
        return nid

    # -- resolve a menu entry to a concrete Hero policy id at a given opp state --
    def menu_ids(self, opp_id: int) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for name, val in self.hero_menu.items():
            if val == "br":
                hid = self.br_now(opp_id)[0]
            else:
                hid = self.intern(val)
            out.append((name, hid))
        return out


# ==========================================================================
# learner families + initial regret state
# ==========================================================================

@dataclass
class LearnFamily:
    """An adaptive family: per-seat archetype INIT, a learner RULE, and the two split
    axes. ``split`` is the INIT split (fit/held); ``rule_split`` is the learner-rule split
    (fit = RM; held = RM+/Hedge). A GO on the gate needs ``rule_split == 'held'`` transfer."""
    label: str
    init_kinds: tuple[str, ...]
    split: str                    # init split: "fit" | "held"
    rule: str                     # "rm" | "rm+" | "hedge"
    eta: float                    # Hedge step size (ignored by rm/rm+)
    rule_split: str               # "fit" (rm) | "held" (rm+/hedge)


def default_learn_families() -> list[LearnFamily]:
    """The §35.B.LOCK family set. FIT = RM × {eq|eq, uniform|uniform, station|aggro}.
    HELD (init transfer, RM) = RM × {nit|station, aggro|nit, eq|uniform, station|nit}.
    HELD (learner transfer) = RM+ and Hedge on a couple of unseen inits — the un-fit
    learner families required by the gate."""
    return [
        # --- FIT: kernel menu + depth selected here (RM only) ---
        LearnFamily("rm:eq|eq",           ("eq", "eq"),        "fit",  "rm",    1.0, "fit"),
        LearnFamily("rm:uniform|uniform", ("uniform", "uniform"), "fit", "rm",  1.0, "fit"),
        LearnFamily("rm:station|aggro",   ("station", "aggro"), "fit", "rm",    1.0, "fit"),
        # --- HELD-OUT init families (still RM) — the init transfer test ---
        LearnFamily("rm:nit|station",     ("nit", "station"),  "held", "rm",    1.0, "fit"),
        LearnFamily("rm:aggro|nit",       ("aggro", "nit"),    "held", "rm",    1.0, "fit"),
        LearnFamily("rm:eq|uniform",      ("eq", "uniform"),   "held", "rm",    1.0, "fit"),
        # --- HELD-OUT LEARNER families — the un-fit-learner transfer test (the gate) ---
        LearnFamily("rm+:aggro|nit",      ("aggro", "nit"),    "held", "rm+",   1.0, "held"),
        LearnFamily("rm+:station|nit",    ("station", "nit"),  "held", "rm+",   1.0, "held"),
        LearnFamily("hedge1:aggro|nit",   ("aggro", "nit"),    "held", "hedge", 1.0, "held"),
        LearnFamily("hedge.5:eq|uniform", ("eq", "uniform"),   "held", "hedge", 0.5, "held"),
    ]


def leduc_learn_families() -> list[LearnFamily]:
    """A trimmed family set for the DECISIVE (and expensive) Leduc run: keep 2 FIT, 2 held
    inits, and 2 un-fit learner families so the gate (transfer to ≥1 un-fit learner family)
    is still testable while the per-family escalation stays tractable."""
    return [
        LearnFamily("rm:eq|eq",         ("eq", "eq"),       "fit",  "rm",    1.0, "fit"),
        LearnFamily("rm:station|aggro", ("station", "aggro"), "fit", "rm",   1.0, "fit"),
        LearnFamily("rm:aggro|nit",     ("aggro", "nit"),   "held", "rm",    1.0, "fit"),
        LearnFamily("rm:eq|uniform",    ("eq", "uniform"),  "held", "rm",    1.0, "fit"),
        LearnFamily("rm+:aggro|nit",    ("aggro", "nit"),   "held", "rm+",   1.0, "held"),
        LearnFamily("hedge1:aggro|nit", ("aggro", "nit"),   "held", "hedge", 1.0, "held"),
    ]


def leduc_confirm_families() -> list[LearnFamily]:
    """§37.3 loose-end confirmation set: 1 FIT (for depth selection) + 3 HELD (1 RM, 2 un-fit
    learners RM+/Hedge). Used with a RICHER menu + deeper planning than the decisive run to
    test whether a stronger kernel reveals shaping headroom / a multiplayer edge on Leduc HELD
    families that the bounded depth-2 kernel missed."""
    return [
        LearnFamily("rm:eq|eq",         ("eq", "eq"),       "fit",  "rm",    1.0, "fit"),
        LearnFamily("rm:aggro|nit",     ("aggro", "nit"),   "held", "rm",    1.0, "fit"),
        LearnFamily("rm+:aggro|nit",    ("aggro", "nit"),   "held", "rm+",   1.0, "held"),
        LearnFamily("hedge1:aggro|nit", ("aggro", "nit"),   "held", "hedge", 1.0, "held"),
    ]


def init_regret_for(L: LearnLifted, fam: LearnFamily) -> int:
    """Block-1 learner state: cumulative regret set to the archetype policy values, so the
    block-1 policy ``link(R)`` equals the archetype (exactly for rm/rm+ since π≥0 sums to 1;
    a softened archetype for Hedge). Gives a diverse, sensibly-scaled O(1) start that the
    O(1)-per-block cf-regret updates then move."""
    regret: Regret = {}
    for idx, seat in enumerate(L.opp_seats):
        kind = fam.init_kinds[idx] if idx < len(fam.init_kinds) else "uniform"
        arch = archetype_policy(L.keys_by_seat[seat], L.actions_by_key, kind, eq=L.eq)
        regret[seat] = {k: list(arch[k]) for k in L.keys_by_seat[seat]}
    return L.intern_state(regret)


# ==========================================================================
# escalation: rich-menu A5−A4 decomposition under learner ρ (m blocks, exact controls)
# ==========================================================================

@dataclass
class LearnFamilyEval:
    label: str
    split: str
    rule_split: str
    a25: float
    a4: float
    a5: float
    headroom_lb: float            # (finite-menu achievable lower bound) ceiling − A2.5
    a5_minus_a25: float
    a5_minus_a4: float
    traj_ceiling: float           # Σ_t BR_value(P_t)/m along A5's realised path
    ctl_traj_ok: bool             # realised ≤ trajectory-BR ceiling
    ctl_a5_selfconsistent: bool   # A5 predicted-window == realised-window
    ctl_ceiling_ge_a25: bool
    ctl_a5_ge_a25: bool
    depths: dict[str, list[tuple[int, float]]]


def eval_family(L: LearnLifted, init_id: int, m: int, depths: tuple[int, ...],
                do_ceiling: bool, ctl_h: int) -> tuple[dict, float]:
    """Realise A2.5, A4(d), A5(d) for each depth under the true dense learner ρ, plus a
    dedicated FULL-horizon (depth==ctl_h) A5/A4 rollout for the model-correctness control,
    plus the finite-menu achievable ceiling (optional; costly on Leduc). Returns raw
    rollouts + the ceiling mean."""
    a25 = realise(L, init_id, m, "A25", 1)
    a4 = {d: realise(L, init_id, m, "A4", d) for d in depths}
    a5 = {d: realise(L, init_id, m, "A5", d) for d in depths}
    # self-consistency control: full-horizon (no truncation) so re-planning cannot muddy it.
    a5_ctl = realise(L, init_id, ctl_h, "A5", ctl_h)
    a4_ctl = realise(L, init_id, ctl_h, "A4", ctl_h)
    ceil_mean = None
    if do_ceiling:
        _, ceil_ro = ceiling_value(L, init_id, m)
        ceil_mean = ceil_ro.mean_ev
    rolls = {"a25": a25, "a4": a4, "a5": a5, "a5_ctl": a5_ctl, "a4_ctl": a4_ctl}
    return rolls, (ceil_mean if ceil_mean is not None else a25.mean_ev)


# ==========================================================================
# the §35.B decider (one game)
# ==========================================================================

def _aggression(L: LearnLifted, pol: Policy, seat: int) -> float:
    """Mean probability a seat puts on the aggressive (bet/raise) action across its
    infosets — a scalar strategy fingerprint for the learning-trajectory panel."""
    tot, n = 0.0, 0
    for k in L.keys_by_seat[seat]:
        acts = L.actions_by_key[k]
        ai = acts.index("r") if "r" in acts else (acts.index("b") if "b" in acts else None)
        if ai is not None:
            tot += pol[k][ai]
            n += 1
    return tot / n if n else 0.0


def trace_trajectory(L: LearnLifted, init_id: int, m: int, depth: int) -> list[dict]:
    """Replay the A5 controller under the true dense learner ρ and record, per block, the
    watchable learning signals: Hero's block EV, the per-block exact-BR ceiling, each
    learner's cumulative-regret L1 norm, each learner's aggression, and the opp→opp
    coupling strength (how far a 2-step dense successor diverges from the star successor).
    Viz-only (not gating)."""
    planner = Planner(L, "dense")
    opp_id = init_id
    blocks: list[dict] = []
    for t in range(m):
        d = min(depth, m - t)
        _, hid = planner.value(opp_id, d)
        pol = L.state_policy(opp_id)
        reg = L.state_regret(opp_id)
        s1 = L.step(opp_id, hid, "dense", None)
        dense2 = L.state_policy(L.step(s1, hid, "dense", None))
        star2 = L.state_policy(L.step(s1, hid, "star", opp_id))
        coupling = max((abs(a - b) for k in L.opp_keys
                        for a, b in zip(dense2[k], star2[k])), default=0.0)
        blocks.append({
            "t": t,
            "hero_ev": round(L.block_ev(hid, opp_id), 6),
            "br_ceiling": round(L.br_now(opp_id)[1], 6),
            "regret_norm": [round(sum(abs(x) for k in L.keys_by_seat[s] for x in reg[s][k]), 4)
                            for s in L.opp_seats],
            "aggression": [round(_aggression(L, pol, s), 4) for s in L.opp_seats],
            "coupling": round(coupling, 6),
        })
        opp_id = L.step(opp_id, hid, "dense", None)
    return blocks


@dataclass
class LearnReport:
    game: str
    role: str
    m: int
    delta: float
    best_depth: int
    families: list[LearnFamilyEval] = field(default_factory=list)
    gates: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, float] = field(default_factory=dict)
    verdict: str = ""             # GO | KILL | FILTER-* | CONTROLS-FAILED
    bait_headroom_held: float = 0.0
    trajectories: dict[str, list[dict]] = field(default_factory=dict)

    @property
    def all_pass(self) -> bool:
        return all(self.gates.values())


def _enrich_menu(sh: Shared, discovered: dict[str, list[Policy]], max_baits: int) -> Shared:
    hero_keys = sh.keys_by_seat[sh.hero]
    seen: set[tuple] = set()
    enriched = dict(sh.menu)
    idx = 0
    for pols in discovered.values():
        for pol in pols:
            canon = tuple(sorted((k, tuple(round(p, 6) for p in pol[k])) for k in hero_keys))
            if canon in seen:
                continue
            seen.add(canon)
            enriched[f"bait{idx}"] = complete(hero_keys, sh.actions_by_key, pol)
            idx += 1
            if idx >= max_baits:
                break
        if idx >= max_baits:
            break
    return Shared(sh.ft, sh.hero, sh.keys_by_seat, sh.actions_by_key, sh.eq, enriched)


def run_learn_decider(*, game_name: str = "3p-kuhn", hero: int = 0, m: int = 4,
                      depths: tuple[int, ...] = (1, 2, 3), delta: float = 0.02,
                      cfr_iters: int = 4000, cfg: BaitConfig | None = None,
                      max_baits: int = 8, do_ceiling: bool = True,
                      ctl_horizon: int | None = None,
                      families: list[LearnFamily] | None = None,
                      on_family=None, on_state=None, on_eval=None,
                      on_traj=None) -> LearnReport:
    """Run the §35.B no-regret-learner decider on one game and apply the §35.C gate.

    Flow (mirrors §35.A): (1) per family, run the rich-bait search UNDER THE LEARNER ρ to
    discover shaping baits; (2) enrich the Hero menu with the discovered baits; (3) realise
    A2.5 / A4(d) / A5(d) over ``m`` blocks under the true dense learner ρ; select the
    planning depth on FIT×RM families; (4) apply the gate on ``A5 − A4`` over HELD families
    with the un-fit-learner transfer requirement. Kuhn is the within-round FILTER (a NO is
    not a kill; R3); Leduc is DECISIVE."""
    if game_name == "3p-kuhn":
        game: Game = KuhnPokerN(3)
        role = "filter"
    elif game_name == "3p-leduc":
        game = LeducHoldemN(3)
        role = "decisive"
    else:
        raise ValueError(f"unknown game {game_name!r}")

    cfg = cfg or BaitConfig()
    fams = families or default_learn_families()
    ctl_h = ctl_horizon if ctl_horizon is not None else min(m, 3)
    sh = build_shared(game, hero, cfr_iters)

    # ---- pass 1: rich-bait search under the learner ρ (discovers baits per family) ----
    discovered: dict[str, list[Policy]] = {}
    bait_headroom: dict[str, float] = {}
    for fam in fams:
        L = LearnLifted(sh, rule=fam.rule, eta=fam.eta)
        init = init_regret_for(L, fam)
        fr = decide_family(
            L, init, m, cfg, min(cfg_n_states(cfg, m), m),
            on_state=(lambda t, sd, f=fam: on_state(f, t, sd)) if on_state else None,
            on_eval=(lambda g, f=fam: on_eval(f, g)) if on_eval else None)
        if fr.top_baits:
            discovered[fam.label] = fr.top_baits
        bait_headroom[fam.label] = fr.headroom
        if on_family is not None:
            on_family(fam, fr)

    # ---- enrich the menu with discovered baits (best-first, capped for DP tractability) ----
    sh_rich = _enrich_menu(sh, discovered, max_baits)

    # ---- pass 2: realise controllers at every depth under the true dense learner ρ ----
    raw = []  # (fam, rolls dict, ceil_mean)
    for fam in fams:
        L = LearnLifted(sh_rich, rule=fam.rule, eta=fam.eta)
        init = init_regret_for(L, fam)
        rolls, ceil_mean = eval_family(L, init, m, depths, do_ceiling, ctl_h)
        raw.append((fam, rolls, ceil_mean))
        if on_traj is not None:
            on_traj(fam, rolls)

    # ---- select planning depth on FIT×RM families (max mean A5 EV) ----
    fitrm = [r for r in raw if r[0].split == "fit" and r[0].rule_split == "fit"]
    if not fitrm:
        fitrm = raw
    best_depth = max(depths, key=lambda d: sum(r[1]["a5"][d].mean_ev for r in fitrm) / len(fitrm))

    rep = LearnReport(game=game_name, role=role, m=m, delta=delta, best_depth=best_depth)

    # ---- learning-trajectory traces for the dashboard (§35.D): one FIT-RM family +
    # one un-fit-learner HELD family, replayed under the true dense learner ρ ----
    def _pick(pred):
        return next((f for f in fams if pred(f)), None)
    repr_fams = [f for f in (_pick(lambda f: f.split == "fit" and f.rule_split == "fit"),
                             _pick(lambda f: f.rule_split == "held"),
                             _pick(lambda f: f.split == "held" and f.rule_split == "fit")) if f]
    seen_lbl: set[str] = set()
    for fam in repr_fams:
        if fam.label in seen_lbl:
            continue
        seen_lbl.add(fam.label)
        Lt = LearnLifted(sh_rich, rule=fam.rule, eta=fam.eta)
        rep.trajectories[fam.label] = trace_trajectory(Lt, init_regret_for(Lt, fam),
                                                        m, best_depth)

    a4_mispredict_gaps: list[float] = []
    for fam, rolls, ceil_mean in raw:
        a25 = rolls["a25"]
        a5b = rolls["a5"][best_depth]
        a4b = rolls["a4"][best_depth]
        a5_ctl = rolls["a5_ctl"]
        a4_ctl = rolls["a4_ctl"]
        ctl_traj = (a5b.total_ev <= a5b.traj_br_ceiling + 1e-7
                    and a4b.total_ev <= a4b.traj_br_ceiling + 1e-7
                    and a25.total_ev <= a25.traj_br_ceiling + 1e-7)
        # model-correctness: A5's full-horizon plan must equal what really happens.
        ctl_a5_self = abs(a5_ctl.planned_window - a5_ctl.realized_window) < 1e-6
        a4_mispredict_gaps.append(a4_ctl.planned_window - a4_ctl.realized_window)
        fe = LearnFamilyEval(
            label=fam.label, split=fam.split, rule_split=fam.rule_split,
            a25=a25.mean_ev, a4=a4b.mean_ev, a5=a5b.mean_ev,
            headroom_lb=ceil_mean - a25.mean_ev,
            a5_minus_a25=a5b.mean_ev - a25.mean_ev,
            a5_minus_a4=a5b.mean_ev - a4b.mean_ev,
            traj_ceiling=a5b.traj_br_ceiling / m,
            ctl_traj_ok=ctl_traj,
            ctl_a5_selfconsistent=ctl_a5_self,
            ctl_ceiling_ge_a25=(ceil_mean >= a25.mean_ev - 1e-7),
            ctl_a5_ge_a25=(a5b.mean_ev >= a25.mean_ev - 1e-7),
            depths={"A4": [(d, rolls["a4"][d].mean_ev) for d in depths],
                    "A5": [(d, rolls["a5"][d].mean_ev) for d in depths]},
        )
        rep.families.append(fe)

    held = [f for f in rep.families if f.split == "held"]
    held_unfit = [f for f in held if f.rule_split == "held"]

    # ---- §28 exact controls (must hold everywhere) ----
    rep.gates["ctl: realised EV <= trajectory-BR ceiling (all)"] = all(
        f.ctl_traj_ok for f in rep.families)
    rep.gates["ctl: A5 predicted-window == realised-window (correct-model self-consistency)"] = all(
        f.ctl_a5_selfconsistent for f in rep.families)
    rep.gates["ctl: finite-menu ceiling >= A2.5 (shaping never hurts the oracle)"] = all(
        f.ctl_ceiling_ge_a25 for f in rep.families)
    rep.gates["ctl: A5 >= A2.5 (correct-model MPC never below myopic)"] = all(
        f.ctl_a5_ge_a25 for f in rep.families)

    # ---- gate inputs (HELD-OUT only) ----
    mean_a5_a25 = sum(f.a5_minus_a25 for f in held) / len(held)
    mean_a5_a4 = sum(f.a5_minus_a4 for f in held) / len(held)
    max_a5_a4 = max(f.a5_minus_a4 for f in held)
    n_a5a4_over_delta = sum(1 for f in held if f.a5_minus_a4 > delta)
    unfit_over_delta = sum(1 for f in held_unfit if f.a5_minus_a4 > delta)
    mean_headroom = sum(f.headroom_lb for f in held) / len(held)
    depth_earns = _depth_earns_marginal(raw, best_depth, depths)
    rep.bait_headroom_held = max((bait_headroom[f.label] for f in
                                  [ff for ff in fams if ff.split == "held"]), default=0.0)
    rep.detail.update({
        "mean A5-A2.5 (held)": mean_a5_a25,
        "mean A5-A4 (held)": mean_a5_a4,
        "max A5-A4 (held)": max_a5_a4,
        "held families A5-A4 > delta": float(n_a5a4_over_delta),
        "un-fit-learner families A5-A4 > delta": float(unfit_over_delta),
        "mean finite-menu headroom lb (held)": mean_headroom,
        "selected planning depth": float(best_depth),
        "depth earns marginal over depth-1": float(depth_earns),
        "held families": float(len(held)),
        "un-fit-learner held families": float(len(held_unfit)),
        "max bait headroom (held)": rep.bait_headroom_held,
        "mean A4 mispredict gap (planned-realised, informational)":
            sum(a4_mispredict_gaps) / len(a4_mispredict_gaps),
    })

    # ---- verdict (§35.C), role-aware ----
    controls_ok = all(v for k, v in rep.gates.items() if k.startswith("ctl:"))
    inter_opp = mean_a5_a4 > delta
    transfers = unfit_over_delta >= 1
    if not controls_ok:
        rep.verdict = "CONTROLS-FAILED"
    elif role == "filter":
        # Kuhn: report whether the multiplayer edge clears δ; a NO is NOT a kill (R3).
        rep.verdict = "FILTER-GO" if (inter_opp and transfers) else "FILTER-FLAT"
    else:
        # Leduc: decisive. GO needs A5−A4 > δ, transfer to an un-fit learner, depth earns.
        if inter_opp and transfers and depth_earns:
            rep.verdict = "GO"
        else:
            rep.verdict = "KILL"

    rep.gates[f"VERDICT[{role}] = {rep.verdict}"] = True
    return rep


def _depth_earns_marginal(raw, best_depth: int, depths: tuple[int, ...]) -> bool:
    """Does the selected depth beat depth-1 on A5 over HELD families (feature C / H3)?"""
    if best_depth == 1 or 1 not in depths:
        return best_depth > 1
    held = [r for r in raw if r[0].split == "held"]
    if not held:
        return False
    d1 = sum(r[1]["a5"][1].mean_ev for r in held) / len(held)
    db = sum(r[1]["a5"][best_depth].mean_ev for r in held) / len(held)
    return db - d1 > 1e-6


def cfg_n_states(cfg: BaitConfig, m: int) -> int:
    """Number of trajectory states to probe in the bait search (all m by default)."""
    return m
