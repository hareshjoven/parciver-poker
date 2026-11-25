# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""R-MAPSS Step 2 FOLLOW-UP §35.A — the RICH-BAIT DECIDER.

§34.5 found H2 (the multiplayer inter-opponent edge) NOT SUPPORTED under the
pre-registered dense lagged-BR ρ: in 3p Leduc the omniscient ρ-ceiling could not
beat the myopic exact-BR-refit anywhere. But that ceiling was computed over a
MENU-restricted controller (only 5 Hero archetypes {br, eq, passive, aggro, nit}),
so it is only a LOWER BOUND on the true optimal exploitation. Stackelberg-vs-best-
responder theory predicts shaping SHOULD generically beat myopic against lagged-BR
followers — so the flat §34.5 result MIGHT be a menu artifact rather than a real null.

This module removes that confound (RMAPSS.md §35.A). It replaces the 5-archetype menu
with a genuinely RICH bait search and recomputes a much tighter ceiling of the
1-step-ahead Stackelberg objective::

    g1(h | P) = u_0(h, P)                      # immediate EV of playing bait h vs opp state P
              + BR_value( ρ(h, P) )            # + exact BR-value against the opp state ρ pushes P to

The myopic baseline is h = BR(P): it earns the best immediate EV, BR_value(P), then
best-responds again next block. Since BR(P) is always IN the candidate set,
``headroom = max_h g1(h) − g1(BR) ≥ 0`` exactly; headroom > δ means a bait beats
myopic over the 2-block window ⇒ the §34.5 null was a menu artifact.

Rich candidate set (§35.A "large sample of pure + random mixtures + coordinate ascent
from the myopic-BR seed"):
  * the myopic BR seed + the §34 archetypes + uniform (continuity with §34);
  * ``n_pure`` random PURE Hero strategies;
  * ``n_mixed`` random MIXED strategies (Dirichlet, varied concentration);
  * coordinate ascent on g1 from the BR seed (Kuhn: all infosets; Leduc: the
    reached-&-influential subset, prob-ordered, eval-capped — the full 1401-infoset
    sweep is intractable at ~0.4 s/candidate).

The **future-exploitability screen** is the decisive necessary condition and the
cheap Leduc lever. Headroom > δ requires
``BR_value(ρ(h,P)) − BR_value(ρ(BR,P)) > δ`` for some h (the future gain must exceed δ
before it can even cover the immediate sacrifice, which is ≥ 0). §34.5's verified
mechanism claim was precisely that BR ALSO leaves opponents in the most-exploitable
next state; this screen tests that claim over the RICH h-space. If the richest search
cannot find a reachable successor more exploitable than the myopic one by δ, no bait
can pay — decisive at the 1-step level, and cheaply so.

Everything is EXACT (full traversal via :class:`FastTree`) and game-agnostic. All the
§28 anti-hallucination controls carry over: every candidate's immediate EV ≤ the exact
BR value (BR maximises immediate — a violation is a bug), headroom ≥ 0 (BR seed in set),
and g1 ≤ a valid 2-block upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..games.base import Game
from ..games.kuhn_multi import KuhnPokerN
from ..games.leduc_multi import LeducHoldemN
from .step2 import (
    Lifted, Shared, build_shared, lifted_for, realise, ceiling_value,
    archetype_policy, complete, _init_opp_profile, default_families, Family,
)

Policy = dict[str, list[float]]
_TOL = 1e-9


# ==========================================================================
# rich Hero-policy candidate generation
# ==========================================================================

def _onehot(a: int, na: int) -> list[float]:
    oh = [0.0] * na
    oh[a] = 1.0
    return oh


def random_pure(hero_keys: list[str], actions_by_key: dict[str, tuple],
                rng: np.random.Generator) -> Policy:
    """A uniformly random PURE Hero strategy (one action per infoset)."""
    return {k: _onehot(int(rng.integers(len(actions_by_key[k]))), len(actions_by_key[k]))
            for k in hero_keys}


def random_mixed(hero_keys: list[str], actions_by_key: dict[str, tuple],
                 rng: np.random.Generator, conc: float) -> Policy:
    """A random MIXED Hero strategy: Dirichlet(conc) per infoset. Small conc ⇒
    near-pure; large conc ⇒ near-uniform. Sampling a spread of conc explores the
    simplex interior the pure vertices miss."""
    out: Policy = {}
    for k in hero_keys:
        na = len(actions_by_key[k])
        out[k] = [float(x) for x in rng.dirichlet([conc] * na)]
    return out


# ==========================================================================
# the 1-step-ahead Stackelberg objective (exact) and its decomposition
# ==========================================================================

@dataclass
class G1:
    """One candidate's exact 1-step-ahead evaluation from an opponent state."""
    hid: int
    immediate: float          # u_0(h, P)  — the block-1 EV of the bait
    future: float             # BR_value(ρ(h, P)) — block-2 EV against the pushed opp
    succ_id: int              # interned id of the successor opp profile ρ(h, P)
    tier: str                 # provenance: 'br' | 'archetype' | 'pure' | 'mixed' | 'coord'

    @property
    def total(self) -> float:
        return self.immediate + self.future


def eval_g1(L: Lifted, hid: int, opp_id: int, tier: str) -> G1:
    """Exact g1(h | P). ``immediate`` = on-policy EV of the bait; the opponents run
    the TRUE dense ρ to a successor; ``future`` = Hero's exact BR value there."""
    immediate = L.block_ev(hid, opp_id)
    succ = L.step(opp_id, hid, "dense", None)
    future = L.br_now(succ)[1]
    return G1(hid, immediate, future, succ, tier)


# ==========================================================================
# coordinate ascent on g1 from the myopic-BR seed
# ==========================================================================

def _reach_prob_by_key(L: Lifted, hero_id: int, opp_id: int) -> dict[str, float]:
    """Reach probability of each Hero infoset under the profile {hero, opp} — the
    opponents' + chance contribution AND Hero's own play. Used to prioritise the
    Leduc coordinate-ascent set (off-path infosets can't shape anything)."""
    ft = L.ft
    profile = {**L.pol(hero_id), **L.pol(opp_id)}
    reach: dict[str, float] = {}

    def walk(node: int, p: float) -> None:
        if p == 0.0:
            return
        k = ft.kind[node]
        if k == 0:  # terminal
            return
        if k == 1:  # chance
            for c, pr in ft.cchildren[node]:
                walk(c, p * pr)
            return
        key = ft.dkey[node]
        if ft.dplayer[node] == L.hero:
            reach[key] = reach.get(key, 0.0) + p
        na = ft.nactions[node]
        probs = profile.get(key)
        ch = ft.dchildren[node]
        if probs is None:
            for c in ch:
                walk(c, p / na)
        else:
            for a in range(na):
                if probs[a] > 0.0:
                    walk(ch[a], p * probs[a])

    walk(ft.root, 1.0)
    return reach


def coordinate_ascent(L: Lifted, opp_id: int, seed_hid: int, *,
                      ascent_keys: list[str], eval_budget: int,
                      on_eval: Callable[[G1], None] | None = None) -> tuple[Policy, G1, int]:
    """Greedy coordinate ascent on g1 starting from the BR seed. At each infoset in
    ``ascent_keys`` (prob-ordered) try every pure action, keep the best; sweep until
    no improvement or ``eval_budget`` g1-evals are spent. Returns (best policy, its
    G1, evals used)."""
    h = dict(L.pol(seed_hid))
    best = eval_g1(L, seed_hid, opp_id, "coord")
    if on_eval:
        on_eval(best)
    evals = 0
    improved = True
    while improved and evals < eval_budget:
        improved = False
        for k in ascent_keys:
            na = len(L.actions_by_key[k])
            if na < 2:
                continue
            cur = h[k]
            local_best = best
            local_oh = cur
            for a in range(na):
                oh = _onehot(a, na)
                if oh == cur:
                    continue
                h[k] = oh
                hid = L.intern(h)
                g = eval_g1(L, hid, opp_id, "coord")
                evals += 1
                if on_eval:
                    on_eval(g)
                if g.total > local_best.total + _TOL:
                    local_best, local_oh = g, oh
                if evals >= eval_budget:
                    break
            h[k] = local_oh
            if local_best.total > best.total + _TOL:
                best = local_best
                improved = True
            if evals >= eval_budget:
                break
    return dict(h), best, evals


# ==========================================================================
# per-state decision (the DECIDER at one opponent state P)
# ==========================================================================

@dataclass
class StateDecision:
    state_ix: int
    myopic_immediate: float
    myopic_future: float
    myopic_val: float                 # 2-block myopic (BR now + BR next)
    best_total: float                 # best g1 found over the rich set
    best_immediate: float
    best_future: float
    best_tier: str
    headroom: float                   # best_total − myopic_val   (≥ 0)
    screen_max_future: float          # richest reachable BR-value successor
    screen_gap: float                 # screen_max_future − myopic_future (necessary cond)
    n_candidates: int
    n_distinct_succ: int              # distinct opp' induced (interaction richness)
    lb_pure: float                    # headroom using {br,archetype,pure} only
    lb_mixed: float                   # + mixed
    lb_coord: float                   # + coordinate ascent  (== headroom)
    control_immediate_ok: bool        # every candidate immediate ≤ BR_value(P)+tol
    control_seed_ok: bool             # BR-seed g1 == myopic_val (seed consistency)
    best_bait: Policy = field(default_factory=dict, repr=False)


@dataclass
class BaitConfig:
    n_pure: int = 400
    n_mixed: int = 400
    coord: bool = True
    coord_budget: int = 4000          # max g1-evals in ascent (per state)
    coord_max_keys: int = 0           # 0 = all reached keys; else top-N by reach prob
    reach_floor: float = 1e-6         # ignore Hero infosets reached below this prob
    seed: int = 12345


def decide_state(L: Lifted, opp_id: int, state_ix: int, cfg: BaitConfig,
                 rng: np.random.Generator,
                 on_eval: Callable[[G1], None] | None = None) -> StateDecision:
    """Run the rich-bait search from opponent state ``opp_id`` and return the tightest
    1-step shaping headroom found, with the ablation lower bounds and §28 controls."""
    hero_keys = L.keys_by_seat[L.hero]
    br_hid, br_val = L.br_now(opp_id)

    # myopic 2-block value: BR now, opponents adapt, BR again.
    myo = eval_g1(L, br_hid, opp_id, "br")
    myopic_val = myo.total

    succ_ids: set[int] = set()
    max_immediate = -1e18

    def record(g: G1) -> None:
        nonlocal max_immediate
        succ_ids.add(g.succ_id)
        if g.immediate > max_immediate:
            max_immediate = g.immediate
        if on_eval:
            on_eval(g)

    # --- tier 1: BR seed + archetypes + uniform ---
    cands: list[G1] = [myo]
    record(myo)
    base_menu_pols = {
        "eq": complete(hero_keys, L.actions_by_key, {k: L.eq[k] for k in hero_keys}),
        "passive": archetype_policy(hero_keys, L.actions_by_key, "passive"),
        "aggro": archetype_policy(hero_keys, L.actions_by_key, "aggro"),
        "nit": archetype_policy(hero_keys, L.actions_by_key, "nit"),
        "uniform": archetype_policy(hero_keys, L.actions_by_key, "uniform"),
    }
    for pol in base_menu_pols.values():
        g = eval_g1(L, L.intern(pol), opp_id, "archetype")
        cands.append(g)
        record(g)

    # --- tier 2: random pure ---
    for _ in range(cfg.n_pure):
        g = eval_g1(L, L.intern(random_pure(hero_keys, L.actions_by_key, rng)), opp_id, "pure")
        cands.append(g)
        record(g)

    # --- tier 3: random mixed (spread of Dirichlet concentration) ---
    concs = [0.1, 0.3, 1.0, 3.0]
    for i in range(cfg.n_mixed):
        conc = concs[i % len(concs)]
        g = eval_g1(L, L.intern(random_mixed(hero_keys, L.actions_by_key, rng, conc)), opp_id, "mixed")
        cands.append(g)
        record(g)

    # progressive-tier lower bounds on headroom (each tier can only tighten)
    def hr(pool: list[G1]) -> float:
        return max(g.total for g in pool) - myopic_val
    lb_pure = hr([g for g in cands if g.tier in ("br", "archetype", "pure")])
    lb_mixed = hr([g for g in cands if g.tier in ("br", "archetype", "pure", "mixed")])

    # --- tier 4: coordinate ascent from the BR seed ---
    best = max(cands, key=lambda g: g.total)
    coord_best: G1 | None = None
    if cfg.coord:
        reach = _reach_prob_by_key(L, br_hid, opp_id)
        keys = [k for k in hero_keys if reach.get(k, 0.0) > cfg.reach_floor
                and len(L.actions_by_key[k]) > 1]
        keys.sort(key=lambda k: reach.get(k, 0.0), reverse=True)
        if cfg.coord_max_keys and len(keys) > cfg.coord_max_keys:
            keys = keys[:cfg.coord_max_keys]
        _, coord_best, _ = coordinate_ascent(
            L, opp_id, br_hid, ascent_keys=keys, eval_budget=cfg.coord_budget, on_eval=record)
        if coord_best.total > best.total:
            best = coord_best
    lb_coord = best.total - myopic_val

    screen_max_future = max(g.future for g in cands + ([coord_best] if coord_best else []))
    return StateDecision(
        state_ix=state_ix,
        myopic_immediate=myo.immediate, myopic_future=myo.future, myopic_val=myopic_val,
        best_total=best.total, best_immediate=best.immediate, best_future=best.future,
        best_tier=best.tier, headroom=best.total - myopic_val,
        screen_max_future=screen_max_future, screen_gap=screen_max_future - myo.future,
        n_candidates=len(cands) + (1 if coord_best else 0), n_distinct_succ=len(succ_ids),
        lb_pure=lb_pure, lb_mixed=lb_mixed, lb_coord=lb_coord,
        control_immediate_ok=(max_immediate <= br_val + 1e-7),
        control_seed_ok=(abs(myo.immediate - br_val) < 1e-7),
        best_bait=dict(L.pol(best.hid)),
    )


# ==========================================================================
# per-family trajectory probing (probe every state on the myopic path)
# ==========================================================================

@dataclass
class BaitFamilyResult:
    label: str
    split: str
    eps: float
    per_state: list[StateDecision]
    headroom: float                 # max over states of 1-step headroom
    screen_gap: float               # max over states of the future-exploitability gap
    best_state_ix: int
    control_immediate_ok: bool
    control_seed_ok: bool
    top_baits: list[Policy] = field(default_factory=list, repr=False)  # for escalation


def decide_family(L: Lifted, init_id: int, m: int, cfg: BaitConfig,
                  n_states: int, on_state=None, on_eval=None) -> BaitFamilyResult:
    """Walk the myopic (A2.5) trajectory and run the rich-bait decider at each of the
    first ``n_states`` states. Family headroom = the tightest place shaping could help.
    Records the best baits found for the escalation menu."""
    fam_rng = np.random.default_rng(cfg.seed)
    per_state: list[StateDecision] = []
    top_baits: list[Policy] = []
    opp_id = init_id
    n = min(n_states, m)
    for t in range(n):
        sd = decide_state(L, opp_id, t, cfg, fam_rng, on_eval=on_eval)
        per_state.append(sd)
        if sd.headroom > _TOL:
            top_baits.append(sd.best_bait)
        if on_state is not None:
            on_state(t, sd)
        # advance along the MYOPIC path (the baseline we are trying to beat)
        br_hid, _ = L.br_now(opp_id)
        opp_id = L.step(opp_id, br_hid, "dense", None)
    best_ix = max(range(len(per_state)), key=lambda i: per_state[i].headroom)
    return BaitFamilyResult(
        label="", split="", eps=L.eps, per_state=per_state,
        headroom=max(s.headroom for s in per_state),
        screen_gap=max(s.screen_gap for s in per_state),
        best_state_ix=best_ix,
        control_immediate_ok=all(s.control_immediate_ok for s in per_state),
        control_seed_ok=all(s.control_seed_ok for s in per_state),
        top_baits=top_baits,
    )


# ==========================================================================
# escalation: multi-block rich ceiling + star-vs-dense (A5−A4) decomposition
# ==========================================================================

@dataclass
class Escalation:
    game: str
    depth: int
    per_family: list[dict]          # label/split/A2.5/A4/A5/ceiling/headroom/a5_minus_a4
    mean_headroom_held: float
    mean_a5_minus_a4_held: float
    controls_ok: bool


def escalate(game: Game, hero: int, cfr_iters: int, m: int, depth: int,
             discovered_baits: dict[str, list[Policy]], families: list[Family],
             max_baits: int = 24) -> Escalation:
    """Only run if §35.A found headroom. Enrich the Hero menu with the discovered
    baits and reuse step2's exact realise()/ceiling_value() — 'the existing A4/A5
    machinery with the rich candidate set' (§35.A). ``A5 − A4`` with this rich menu
    is the multiplayer (H2) signal; ceiling − A2.5 is the total rich headroom.

    ``max_baits`` caps the enriched-menu size: the full-horizon ρ-ceiling DP branches
    ``menu`` ways per block, so on Leduc (m=4, ~60k-node evals) a large menu is
    intractable — keep it small there (the discovered baits are added best-first)."""
    sh = build_shared(game, hero, cfr_iters)
    hero_keys = sh.keys_by_seat[hero]
    # collapse discovered baits to distinct policies, cap the count (DP cost ∝ menu size)
    seen: set[tuple] = set()
    enriched = dict(sh.menu)
    idx = 0
    for pols in discovered_baits.values():
        for pol in pols:
            canon = tuple(sorted((k, tuple(round(p, 6) for p in pol[k])) for k in hero_keys))
            if canon in seen:
                continue
            seen.add(canon)
            enriched[f"bait{idx}"] = complete(hero_keys, sh.actions_by_key, pol)
            idx += 1
            if idx >= max_baits:  # keep the DP tractable
                break
        if idx >= max_baits:
            break
    sh_rich = Shared(sh.ft, hero, sh.keys_by_seat, sh.actions_by_key, sh.eq, enriched,
                     intern=sh.intern, by_id=sh.by_id, ev_cache=sh.ev_cache, br_cache=sh.br_cache)

    per_family: list[dict] = []
    controls_ok = True
    for fam in families:
        L = lifted_for(sh_rich, fam.eps)
        init = _init_opp_profile(L, fam)
        a25 = realise(L, init, m, "A25", 1)
        a4 = realise(L, init, m, "A4", depth)
        a5 = realise(L, init, m, "A5", depth)
        _, ceil = ceiling_value(L, init, m)
        a5_full = realise(L, init, m, "A5", m)
        controls_ok = controls_ok and (a5_full.mean_ev <= ceil.mean_ev + 1e-7) \
            and (ceil.mean_ev >= a25.mean_ev - 1e-7) and (a5.mean_ev >= a25.mean_ev - 1e-7)
        per_family.append({
            "label": fam.label, "split": fam.split,
            "A2.5": a25.mean_ev, "A4": a4.mean_ev, "A5": a5.mean_ev, "ceiling": ceil.mean_ev,
            "headroom": ceil.mean_ev - a25.mean_ev, "a5_minus_a4": a5.mean_ev - a4.mean_ev,
        })
    held = [r for r in per_family if r["split"] == "held"]
    return Escalation(
        game=game.__class__.__name__, depth=depth, per_family=per_family,
        mean_headroom_held=sum(r["headroom"] for r in held) / len(held),
        mean_a5_minus_a4_held=sum(r["a5_minus_a4"] for r in held) / len(held),
        controls_ok=controls_ok,
    )


# ==========================================================================
# the §35.A decider (one game)
# ==========================================================================

@dataclass
class BaitReport:
    game: str
    role: str
    m: int
    delta: float
    cfg: BaitConfig
    n_states: int
    results: list[BaitFamilyResult] = field(default_factory=list)
    gates: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, float] = field(default_factory=dict)
    verdict: str = ""               # NULL-REAL | MENU-ARTIFACT | CONTROLS-FAILED
    escalation: Escalation | None = None

    @property
    def all_pass(self) -> bool:
        return all(self.gates.values())


def run_decider(*, game_name: str = "3p-kuhn", hero: int = 0, m: int = 8,
                delta: float = 0.02, cfr_iters: int = 6000, depth: int = 3,
                n_states: int | None = None, cfg: BaitConfig | None = None,
                do_escalate: bool = True, max_baits: int = 24, on_family=None,
                on_state=None, on_eval=None) -> BaitReport:
    """Run the §35.A rich-bait decider on one game and apply the pre-registered gate.

    Gate (§35.A): if the rich-bait tighter ceiling STILL == A2.5 within δ on held-out
    families ⇒ ``NULL-REAL`` (the §34.5 null is genuine, not a menu artifact). If
    headroom > δ appears ⇒ ``MENU-ARTIFACT`` ⇒ escalate to the multi-block rich ceiling
    and decompose the single-opponent (star) vs multiplayer (A5−A4) contributions.
    Kuhn is the within-round filter; Leduc is decisive (a NO in Kuhn is not a kill, R3).
    """
    if game_name == "3p-kuhn":
        game: Game = KuhnPokerN(3)
        role = "filter"
    elif game_name == "3p-leduc":
        game = LeducHoldemN(3)
        role = "decisive"
    else:
        raise ValueError(f"unknown game {game_name!r}")

    cfg = cfg or BaitConfig()
    n_states = n_states if n_states is not None else m
    rep = BaitReport(game=game_name, role=role, m=m, delta=delta, cfg=cfg, n_states=n_states)

    sh = build_shared(game, hero, cfr_iters)
    families = default_families()
    for fam in families:
        L = lifted_for(sh, fam.eps)
        init = _init_opp_profile(L, fam)
        fr = decide_family(
            L, init, m, cfg, n_states,
            on_state=(lambda t, sd, f=fam: on_state(f, t, sd)) if on_state else None,
            on_eval=(lambda g, f=fam: on_eval(f, g)) if on_eval else None)
        fr.label, fr.split = fam.label, fam.split
        rep.results.append(fr)
        if on_family is not None:
            on_family(fam, fr)

    held = [r for r in rep.results if r.split == "held"]

    # ---- §28 controls (must hold everywhere) ----
    rep.gates["ctl: every candidate immediate EV <= exact BR value (BR maximises immediate)"] = \
        all(r.control_immediate_ok for r in rep.results)
    rep.gates["ctl: BR-seed g1 == myopic 2-block value (seed consistency)"] = \
        all(r.control_seed_ok for r in rep.results)
    rep.gates["ctl: headroom >= 0 (BR seed in candidate set)"] = \
        all(s.headroom >= -1e-9 for r in rep.results for s in r.per_state)

    # ---- decider inputs (HELD-OUT only) ----
    max_headroom_held = max(r.headroom for r in held)
    mean_headroom_held = sum(r.headroom for r in held) / len(held)
    max_screen_gap_held = max(r.screen_gap for r in held)
    n_families_with_headroom = sum(1 for r in held if r.headroom > delta)
    rep.detail.update({
        "max headroom (held)": max_headroom_held,
        "mean headroom (held)": mean_headroom_held,
        "max future-exploitability screen gap (held)": max_screen_gap_held,
        "held families with headroom > delta": float(n_families_with_headroom),
        "held families": float(len(held)),
        "candidates/state (approx)": float(cfg.n_pure + cfg.n_mixed + 6),
    })

    controls_ok = all(rep.gates.values())
    if not controls_ok:
        rep.verdict = "CONTROLS-FAILED"
    elif max_headroom_held > delta:
        rep.verdict = "MENU-ARTIFACT"
        if do_escalate:
            discovered = {r.label: r.top_baits for r in rep.results if r.top_baits}
            rep.escalation = escalate(game, hero, cfr_iters, m, depth, discovered, families,
                                      max_baits=max_baits)
    else:
        rep.verdict = "NULL-REAL"

    rep.gates[f"VERDICT[{role}] = {rep.verdict}"] = True
    return rep
