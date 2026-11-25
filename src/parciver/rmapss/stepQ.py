"""R-MAPSS Step Q — fragility as a transferable-EV predictor (RMAPSS.md §38).

Pre-registered in §38 (locked WITH the author, session 15, BEFORE any result). Tests
hypothesis **H-Q**: does a CHEAP, best-response-free, Hero-observable fragility metric
of an approximate-equilibrium opponent PREDICT the exact extractable *transferable*
EV, out-of-sample (held-out opponents), and BEYOND the trivial distance-from-equilibrium
baseline?

Everything is exact (full traversal) and engine-independent (3p Kuhn / 3p Leduc are
pure-Python) — a positive held-out number is a real edge with no confidence interval
(R4). The module REUSES the certified step-1 machinery byte-for-byte
(`observe_marginal` / `deviation_signal` / `blend` / `safety_gated_alpha` /
`clairvoyant_team_min`) and `eval.exploitability` (`best_response_value` /
`best_response_strategy` / `on_policy_value` / `nash_conv_multiplayer`).

§38.5 circularity guard (LOAD-BEARING): NO Hero best response — its strategy or its
value — enters ANY cheap predictor. Predictors live on a BR-free code path; the only
place `best_response_*` for Hero is called is the exact targets (`Y_ceil`, `Y_cap`'s
aggregate-marginal seed) and the deliberate "exact-BR predictor" CONTROL, which must
score ρ≈1 (proof the pipeline can see a perfect predictor). The opponent's own BR
(the `nashconv_opp` reference) is a different quantity and is reported, not gated.

Targets (§38.1 + §38.DEV1, user-locked; the original single aggregate-blind ``Y_cap`` was
degenerate, so the pre-registered SECONDARY was promoted to the gated primary):
  * PRIMARY (gated): the exact ceiling ``Y_ceil`` = per-opponent exact BR gain (extractable EV).
  * REPORTED: the alpha=1 realized transferable capture ``Y_cap`` — a deployable identity-blind
    exploiter that reads only the public card-free marginal and BRs full-strength (<= Y_ceil).

Gate (§38.4): a CHEAP metric (not the ``dist_eq`` baseline) with |ρ_held| ≥ ρ* AND
|ρ_held| − |ρ_held(dist_eq)| ≥ Δρ AND sign(ρ_fit)==sign(ρ_held) (sign-stability) AND
sign holds on the un-fit transfer family — in BOTH Kuhn and Leduc. Defaults ρ*=0.70,
Δρ=0.15 (locked with the author).
"""
from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field
from typing import Callable, Hashable

from ..eval.exploitability import (
    best_response_strategy,
    best_response_value,
    nash_conv_multiplayer,
    on_policy_value,
)
from ..solvers.cfr_multi import MultiCFR
from .step1 import (
    blend,
    clairvoyant_team_min,
    collect_infosets,
    deviation_signal,
    safety_gated_alpha,
)

Strategy = dict[str, list[float]]


# ==========================================================================
# game-agnostic card-free marginal (step1.observe_marginal is Kuhn-key-specific)
# ==========================================================================
# Hero cannot see opponents' private cards, so the OBSERVABLE opponent model marginalises
# infosets that share the same PUBLIC history. The public grouping key differs by game:
# Kuhn key = "card:history" (public = after ':'); Leduc key = "private|board|betlog"
# (public = after the first '|'). This handles both; unknown formats fall back to the full
# key (no marginalisation), which is safe (just a finer grouping).

def _public_of(key: str) -> str:
    if ":" in key:
        return key.split(":", 1)[1]
    if "|" in key:
        return key.split("|", 1)[1]
    return key


def observe_marginal(game, hero_strategy: Strategy, opp_profile: Strategy,
                     seat_keys: dict[int, list[str]], hero: int = 0):
    """Game-agnostic port of step1.observe_marginal: the reach-weighted, card-free
    (public-history-conditioned) opponent marginal Hero can OBSERVE. Returns
    ``(flat, marg_by_node, mass_by_node)`` keyed by ``(seat, public_history)``."""
    n = game.num_players
    counts: dict[tuple[int, str], list[float]] = {}

    def walk(state, reach):
        if reach == 0.0 or game.is_terminal(state):
            return
        if game.is_chance(state):
            for a, p in game.chance_outcomes(state):
                walk(game.next_state(state, a), reach * p)
            return
        player = game.current_player(state)
        actions = game.legal_actions(state)
        key = game.infoset_key(state)
        probs = (hero_strategy if player == hero else opp_profile).get(
            key, [1.0 / len(actions)] * len(actions))
        if player != hero:
            c = counts.setdefault((player, _public_of(key)), [0.0] * len(actions))
            for i in range(len(actions)):
                c[i] += reach * probs[i]
        for i, a in enumerate(actions):
            if probs[i] > 0.0:
                walk(game.next_state(state, a), reach * probs[i])

    walk(game.initial_state(), 1.0)

    marg_by_node: dict[tuple[int, str], list[float]] = {}
    mass_by_node: dict[tuple[int, str], float] = {}
    for node, c in counts.items():
        s = sum(c)
        mass_by_node[node] = s
        marg_by_node[node] = [x / s for x in c] if s > 0 else [1.0 / len(c)] * len(c)

    flat: Strategy = {}
    for seat in range(n):
        if seat == hero:
            continue
        for key in seat_keys[seat]:
            node = (seat, _public_of(key))
            # unreached public nodes fall back to the opponent's own policy (correct arity)
            flat[key] = marg_by_node.get(node) or opp_profile.get(key)
    return flat, marg_by_node, mass_by_node


# ==========================================================================
# game-agnostic reach + policy helpers (exact, full traversal)
# ==========================================================================

def infoset_reach(game, full_profile: Strategy, hero: int = 0) -> dict[str, float]:
    """Reach mass of every infoset key under the joint ``full_profile`` (chance
    included). Game-agnostic: only uses the ``Game`` interface. Used to weight the
    per-infoset fragility contributions the way Hero actually encounters them."""
    reach: dict[str, float] = {}

    def walk(state: Hashable, r: float) -> None:
        if r == 0.0 or game.is_terminal(state):
            return
        if game.is_chance(state):
            for a, p in game.chance_outcomes(state):
                walk(game.next_state(state, a), r * p)
            return
        key = game.infoset_key(state)
        reach[key] = reach.get(key, 0.0) + r
        actions = game.legal_actions(state)
        probs = full_profile.get(key, [1.0 / len(actions)] * len(actions))
        for i, a in enumerate(actions):
            if probs[i] > 0.0:
                walk(game.next_state(state, a), r * probs[i])

    walk(game.initial_state(), 1.0)
    return reach


def opp_keys_by_seat(seat_keys: dict[int, list[str]], hero: int = 0) -> dict[int, list[str]]:
    return {s: ks for s, ks in seat_keys.items() if s != hero}


# ==========================================================================
# population (§38.3) — game-agnostic near-equilibrium ladder + matched-dist_eq
# ==========================================================================
# The crux: variance must live among agents that look *similarly* non-GTO but leak
# DIFFERENT EV. We perturb the exact equilibrium per infoset toward named DIRECTIONS
# (structural over action indices, so it works for any action count) at graded ε, and
# we include matched-ε pairs in DIFFERENT directions (≈equal dist_eq, different leak).

def _direction_target(n_actions: int, direction: str, key: str) -> list[float]:
    """A target action distribution for a named perturbation direction. Structural
    (index-based), heuristic-free (§D): 'fold' = mass on the first legal action (check/
    fold), 'aggro' = mass on the last (bet/raise/call-up), 'uniform', 'rand' = a fixed
    key-seeded pseudo-random simplex point (a direction uncorrelated with fold/aggro)."""
    if direction == "fold":
        v = [0.0] * n_actions
        v[0] = 1.0
        return v
    if direction == "aggro":
        v = [0.0] * n_actions
        v[-1] = 1.0
        return v
    if direction == "uniform":
        return [1.0 / n_actions] * n_actions
    if direction == "rand":
        # DETERMINISTIC seed (zlib.crc32 of the key) — NOT builtin hash(), which is
        # randomised by PYTHONHASHSEED and made the "rand" opponents (hence the whole
        # population + every ρ) non-reproducible across processes. Fix, session 15.
        rng = random.Random(zlib.crc32(("rand:" + key).encode("utf-8")))
        w = [rng.random() + 1e-6 for _ in range(n_actions)]
        s = sum(w)
        return [x / s for x in w]
    raise ValueError(f"unknown direction {direction!r}")


def make_seat_policy(game, seat_keys_for_seat: list[str], eq: Strategy,
                     direction: str, eps: float) -> Strategy:
    """(1-eps)·eq + eps·direction-target, per infoset (eps in [0,1])."""
    pol: Strategy = {}
    for key in seat_keys_for_seat:
        base = list(eq.get(key, []))
        if not base:
            continue
        tgt = _direction_target(len(base), direction, key)
        pol[key] = [(1.0 - eps) * base[i] + eps * tgt[i] for i in range(len(base))]
    return pol


@dataclass
class OppQ:
    label: str
    klass: str            # "eq" | "near" | "gross"
    split: str            # "fit" | "held" | "transfer"
    profile: Strategy     # opponent seats only (1..N-1)


def build_pop_q(game, eq: Strategy, seat_keys: dict[int, list[str]]) -> list[OppQ]:
    """Near-equilibrium ladder + matched-dist_eq different-direction pairs, split
    FIT / HELD / TRANSFER (R4 + §R). Two opponent seats (1,2) for the 3p games."""
    okeys = opp_keys_by_seat(seat_keys)
    s_ids = sorted(okeys)          # e.g. [1, 2]
    s1, s2 = s_ids[0], s_ids[1]

    def pol(seat, direction, eps):
        return make_seat_policy(game, okeys[seat], eq, direction, eps)

    def pair(label, klass, split, d1, e1, d2, e2):
        prof = {**pol(s1, d1, e1), **pol(s2, d2, e2)}
        return OppQ(label, klass, split, prof)

    pop: list[OppQ] = []
    # --- equilibrium anchors (capture must be ~0; |ρ| not driven by these) ---
    pop.append(pair("eq|eq", "eq", "fit", "uniform", 0.0, "uniform", 0.0))
    # near-eq (10% toward uniform) is NOT a true equilibrium — its own klass so the anchor
    # gate does not hold it to ~0 (§38.DEV2 F2); still excluded from ρ like the true eq.
    pop.append(pair("nearEq(.1u)|nearEq(.1u)", "near_eq", "held", "uniform", 0.1, "uniform", 0.1))

    # --- FIT: a graded ladder across directions & eps (fitting the ρ sign here) ---
    fit_specs = [
        ("fold.35|fold.35", "near", "fold", .35, "fold", .35),
        ("aggro.35|aggro.35", "near", "aggro", .35, "aggro", .35),
        ("fold.6|aggro.6", "near", "fold", .6, "aggro", .6),
        ("rand.5|rand.5", "near", "rand", .5, "rand", .5),
        ("uniform.6|fold.4", "near", "uniform", .6, "fold", .4),
        ("fold.85|fold.85", "gross", "fold", .85, "fold", .85),
        ("aggro.85|aggro.85", "gross", "aggro", .85, "aggro", .85),
        ("fold1|aggro1", "gross", "fold", 1.0, "aggro", 1.0),
    ]
    for lbl, kl, d1, e1, d2, e2 in fit_specs:
        pop.append(pair(lbl, kl, "fit", d1, e1, d2, e2))

    # --- HELD: unseen ladder rungs + MATCHED-dist_eq different-direction pairs ---
    held_specs = [
        ("fold.5|fold.5", "near", "fold", .5, "fold", .5),
        ("aggro.5|aggro.5", "near", "aggro", .5, "aggro", .5),      # matched eps, diff dir
        ("rand.5|fold.5", "near", "rand", .5, "fold", .5),          # matched eps, diff dir
        ("uniform.5|uniform.5", "near", "uniform", .5, "uniform", .5),
        ("fold.7|aggro.3", "near", "fold", .7, "aggro", .3),
        ("aggro1|aggro1", "gross", "aggro", 1.0, "aggro", 1.0),
        ("fold1|fold1", "gross", "fold", 1.0, "fold", 1.0),
    ]
    for lbl, kl, d1, e1, d2, e2 in held_specs:
        pop.append(pair(lbl, kl, "held", d1, e1, d2, e2))

    # --- TRANSFER: an un-fit family (mixed directions, own eps grid) for §R ---
    transfer_specs = [
        ("T:rand.7|aggro.7", "near", "rand", .7, "aggro", .7),
        ("T:uniform.4|rand.4", "near", "uniform", .4, "rand", .4),
        ("T:fold.9|rand.9", "gross", "fold", .9, "rand", .9),
        ("T:rand1|rand1", "gross", "rand", 1.0, "rand", 1.0),
    ]
    for lbl, kl, d1, e1, d2, e2 in transfer_specs:
        pop.append(pair(lbl, kl, "transfer", d1, e1, d2, e2))
    return pop


# ==========================================================================
# CHEAP, BR-FREE fragility metrics (§38.2) — NO Hero best response anywhere here
# ==========================================================================

def m_dist_eq(game, opp: Strategy, seat_keys, ctx) -> float:
    """BASELINE: reach-weighted L1 distance of the opponent marginal from the eq
    marginal (= step1.deviation_signal). The trivial 'looks non-GTO' signal to beat."""
    _, marg_obs, mass_obs = observe_marginal(game, ctx["hero_eq"], opp, seat_keys)
    return deviation_signal(marg_obs, mass_obs, ctx["eq_marg"])


def m_entropy(game, opp: Strategy, seat_keys, ctx) -> float:
    """Reach-weighted mean Shannon entropy (nats) of the opponent's per-infoset action
    distributions (the true card-conditioned policy), weighted by infoset reach under
    (Hero=eq, opp). Direction is NOT pre-assigned (§38.4 sign-stability decides it)."""
    full = {**ctx["hero_eq"], **opp}
    reach = infoset_reach(game, full)
    num = den = 0.0
    for seat, keys in opp_keys_by_seat(seat_keys).items():
        for key in keys:
            p = opp.get(key)
            if not p:
                continue
            r = reach.get(key, 0.0)
            if r <= 0.0:
                continue
            h = -sum(pi * math.log(pi) for pi in p if pi > 0.0)
            num += r * h
            den += r
    return num / den if den > 0 else 0.0


def _seat_values_under_profile(game, full_profile: Strategy, seat: int):
    """For every ``seat`` infoset, the REACH-WEIGHTED per-action continuation value to
    ``seat`` under the fixed joint profile, plus the infoset's own reach. An infoset
    aggregates many states (opponents' hidden cards differ), so per-action values are
    averaged over those states weighted by state reach = E[continuation | at infoset].
    One traversal; no best response — the profile is fixed."""
    qacc: dict[str, list[float]] = {}      # key -> sum_state reach * per-action child value
    reach: dict[str, float] = {}           # key -> own reach

    def value(state: Hashable, r: float) -> float:
        if game.is_terminal(state):
            return game.terminal_utility_for(state, seat)
        if game.is_chance(state):
            return sum(p * value(game.next_state(state, a), r * p)
                       for a, p in game.chance_outcomes(state))
        actions = game.legal_actions(state)
        key = game.infoset_key(state)
        player = game.current_player(state)
        probs = full_profile.get(key, [1.0 / len(actions)] * len(actions))
        cv = [value(game.next_state(state, a), r * probs[i]) for i, a in enumerate(actions)]
        if player == seat:
            reach[key] = reach.get(key, 0.0) + r
            q = qacc.setdefault(key, [0.0] * len(actions))
            for i in range(len(actions)):
                q[i] += r * cv[i]
        return sum(probs[i] * cv[i] for i in range(len(actions)))

    value(game.initial_state(), 1.0)
    child_vals = {k: ([q[i] / reach[k] for i in range(len(q))] if reach[k] > 1e-15 else list(q))
                  for k, q in qacc.items()}
    return child_vals, reach


def m_mistake(game, opp: Strategy, seat_keys, ctx) -> float:
    """Reach-weighted LOCAL one-step suboptimality (§38.2): the opponent's regret from
    not playing its locally-best action against a FIXED reference field (Hero=eq anchor
    + the coopponent's own policy), holding its OWN continuation fixed. Myopic (no
    recursion, no Hero BR) — the cheap regret building block, distinct from full BR."""
    full = {**ctx["hero_eq"], **opp}
    total = 0.0
    for seat in opp_keys_by_seat(seat_keys):
        child_vals, node_reach = _seat_values_under_profile(game, full, seat)
        for key, cv in child_vals.items():
            p = opp.get(key)
            if not p:
                continue
            r = node_reach.get(key, 0.0)
            if r <= 0.0:
                continue
            expected = sum(p[i] * cv[i] for i in range(len(cv)))
            regret = max(cv) - expected      # ≥ 0
            total += r * regret
    return total


def m_pert(game, opp: Strategy, seat_keys, ctx, *, k: int = 8, delta: float = 0.05,
           seed: int = 12345) -> float:
    """Local exploitation-surface steepness (§38.2): finite-difference magnitude of
    Hero-EV's response to K small RANDOM perturbations of Hero's play around the anchor,
    vs this opponent. A first-order slope — no argmax, no BR, only on-policy values."""
    hero_eq = ctx["hero_eq"]
    hero_keys = ctx["hero_keys"]
    base = on_policy_value(game, {**hero_eq, **opp}, 0)
    rng = random.Random(seed)
    acc = 0.0
    for _ in range(k):
        pert: Strategy = {}
        for key in hero_keys:
            d = hero_eq.get(key)
            if not d:
                continue
            noise = [rng.gauss(0.0, 1.0) for _ in d]
            m = sum(noise) / len(noise)
            step = [delta * (noise[i] - m) for i in range(len(d))]  # tangent to simplex
            v = [max(0.0, d[i] + step[i]) for i in range(len(d))]
            s = sum(v)
            pert[key] = [x / s for x in v] if s > 0 else list(d)
        ev = on_policy_value(game, {**pert, **opp}, 0)
        acc += abs(ev - base)
    return acc / k / delta


def m_nashconv_opp(game, opp: Strategy, seat_keys, ctx) -> float:
    """REFERENCE ONLY (not a cheap candidate, §38.2): the opponents' own exact
    exploitability against their field (Hero=eq + each opponent's coopponent fixed) —
    sum over non-hero seats of that seat's own BR gain. Uses the OPPONENT's BR (a
    different quantity from Hero's BR); reported to bound residual signal, never gated."""
    hero_eq = ctx["hero_eq"]
    total = 0.0
    for seat in opp_keys_by_seat(seat_keys):
        field = {**hero_eq, **opp}                 # everyone fixed
        br = best_response_value(game, field, seat)
        onp = on_policy_value(game, field, seat)
        total += max(0.0, br - onp)
    return total


# The registry of CHEAP candidates (dist_eq is the baseline; the rest are candidates).
CHEAP_METRICS: dict[str, Callable] = {
    "dist_eq": m_dist_eq,       # baseline
    "entropy": m_entropy,
    "mistake_mass": m_mistake,
    "pert_sens": m_pert,
}
BASELINE_METRIC = "dist_eq"


# ==========================================================================
# targets (§38.1) — EXACT; and the FIT-tuned blind exploiter E*
# ==========================================================================

def aggregate_fit_marginal(game, hero_eq: Strategy, fit_opps: list[OppQ], seat_keys):
    """Reach-weighted average of the FIT opponents' observed (card-free) marginals — the
    pooled-FIT opponent model E* best-responds to. Blind to any held opponent."""
    acc: dict[str, list[float]] = {}
    cnt: dict[str, float] = {}
    for opp in fit_opps:
        flat, _, _ = observe_marginal(game, hero_eq, opp.profile, seat_keys)
        for key, dist in flat.items():
            a = acc.setdefault(key, [0.0] * len(dist))
            for i in range(len(dist)):
                a[i] += dist[i]
            cnt[key] = cnt.get(key, 0.0) + 1.0
    agg: Strategy = {}
    for key, a in acc.items():
        c = cnt[key]
        s = sum(a)
        agg[key] = [x / s for x in a] if s > 0 else [x / c for x in a]
    return agg


def build_exploiter(game, hero_eq: Strategy, hero_keys: list[str], fit_opps: list[OppQ],
                    seat_keys, *, budget: float, alpha_max: float = 1.0):
    """E* (§38.1): confidence-free, safety-gated α-blend of the CFR/eq anchor with the
    BR to the pooled-FIT AGGREGATE marginal. α is tuned on FIT (max mean FIT capture
    s.t. the §28 exploitability budget). Fixed and BLIND to held identity thereafter."""
    agg = aggregate_fit_marginal(game, hero_eq, fit_opps, seat_keys)
    a_seed = best_response_strategy(game, agg, 0)     # BR to the pooled model (not to any real opp)
    wc_anchor = clairvoyant_team_min(game, hero_eq)
    alpha_safe = safety_gated_alpha(game, hero_eq, a_seed, hero_keys,
                                    safe_floor=wc_anchor, budget=budget, alpha_max=alpha_max)
    # tune α on FIT: max mean FIT capture over a grid capped by the safe α (honest FIT fit)
    def fit_mean_capture(alpha: float) -> float:
        estar = blend(hero_eq, a_seed, alpha, hero_keys)
        vals = []
        for opp in fit_opps:
            base = on_policy_value(game, {**hero_eq, **opp.profile}, 0)
            vals.append(on_policy_value(game, {**estar, **opp.profile}, 0) - base)
        return sum(vals) / len(vals) if vals else 0.0

    best_a, best_v = 0.0, fit_mean_capture(0.0)
    grid = [i / 20.0 for i in range(1, 21)]
    for a in grid:
        if a > alpha_safe + 1e-9:
            break
        v = fit_mean_capture(a)
        if v > best_v + 1e-12:
            best_a, best_v = a, v
    estar = blend(hero_eq, a_seed, best_a, hero_keys)
    return estar, best_a, alpha_safe, wc_anchor


def realized_capture(game, ctx, opp: Strategy, seat_keys) -> float:
    """Reported realized transferable capture (§38.DEV1): an identity-BLIND exploiter
    that reads only the opponent's PUBLIC card-free marginal and best-responds
    full-strength (α=1) — a deployable exploiter, NOT the exact BR to the true
    card-conditioned policy (that gap is R2). ≤ ``Y_ceil`` by construction. The BR here
    is part of the TARGET (reported), never a cheap predictor, so the circularity guard
    is untouched (cheap metrics remain Hero-BR-free)."""
    hero_eq = ctx["hero_eq"]
    marg, _, _ = observe_marginal(game, hero_eq, opp, seat_keys)
    a2 = best_response_strategy(game, marg, 0)
    base = on_policy_value(game, {**hero_eq, **opp}, 0)
    return on_policy_value(game, {**a2, **opp}, 0) - base


# ==========================================================================
# rank correlation (Spearman) — no scipy dependency
# ==========================================================================

def _rankdata(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0        # average rank for ties (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 1e-18 or vb <= 1e-18:
        return 0.0
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return _pearson(_rankdata(xs), _rankdata(ys))


# ==========================================================================
# the experiment (§38) — one game
# ==========================================================================

@dataclass
class OppRow:
    label: str
    klass: str
    split: str
    metrics: dict[str, float]
    y_ceil: float          # GATED PRIMARY target (§38.DEV1): exact per-opponent BR gain
    y_cap: float           # REPORTED: α=1 observed-marginal realized capture (≤ y_ceil)
    ev_base: float


@dataclass
class MetricRho:
    name: str
    rho_fit: float          # vs the gated primary target Y_ceil
    rho_held: float
    rho_transfer: float
    rho_cap_held: float     # vs the reported realized-capture target (secondary)
    is_baseline: bool


@dataclass
class StepQReport:
    game_name: str
    eq_nashconv: float
    alpha_star: float
    alpha_safe: float
    rows: list[OppRow] = field(default_factory=list)
    metric_rho: dict[str, MetricRho] = field(default_factory=dict)
    control_rho_perfect_held: float = 0.0   # perfect-predictor pipeline sanity (→ ~1)
    gates: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, float] = field(default_factory=dict)
    verdict: str = ""

    @property
    def all_gates_pass(self) -> bool:
        return all(self.gates.values())


def solve_eq(game, cfr_iters: int):
    """The CFR+ average strategy — the eq reference the population and metrics anchor to.
    Slow on 3p Leduc (pure Python), so the runner caches it (see scripts/rmapss_stepQ.py)."""
    solver = MultiCFR(game, variant="cfr+")
    solver.run(cfr_iters)
    return solver.average_strategy()


def run_stepQ(game, game_name: str, *, cfr_iters: int = 20000, budget: float = 0.02,
              rho_star: float = 0.70, delta_rho: float = 0.15, pert_k: int = 8,
              eq: Strategy | None = None, on_row: Callable | None = None) -> StepQReport:
    seat_keys = collect_infosets(game)
    hero_keys = seat_keys[0]

    if eq is None:
        eq = solve_eq(game, cfr_iters)
    hero_eq = {k: eq[k] for k in hero_keys}
    eq_nc = nash_conv_multiplayer(game, eq)

    # eq marginal — the card-free reference dist_eq measures against.
    okeys_flat = [k for s, ks in opp_keys_by_seat(seat_keys).items() for k in ks]
    eq_opp = {k: eq[k] for k in okeys_flat}
    _, eq_marg, _ = observe_marginal(game, hero_eq, eq_opp, seat_keys)

    ctx = {"hero_eq": hero_eq, "hero_keys": hero_keys, "eq_marg": eq_marg}

    pop = build_pop_q(game, eq, seat_keys)
    fit_opps = [o for o in pop if o.split == "fit"]

    # E* — the pooled-FIT, blind, safety-gated exploiter (the Y_cap engine).
    estar, alpha_star, alpha_safe, wc_anchor = build_exploiter(
        game, hero_eq, hero_keys, fit_opps, seat_keys, budget=budget)

    rep = StepQReport(game_name=game_name, eq_nashconv=eq_nc,
                      alpha_star=alpha_star, alpha_safe=alpha_safe)

    # per-opponent rows: cheap metrics (BR-free) + the exact targets.
    #   GATED PRIMARY = Y_ceil (exact per-opponent BR gain, §38.DEV1);
    #   REPORTED      = Y_cap  (α=1 observed-marginal realized capture, ≤ Y_ceil).
    for i, opp in enumerate(pop):
        base = on_policy_value(game, {**hero_eq, **opp.profile}, 0)
        y_ceil = best_response_value(game, opp.profile, 0) - base
        y_cap = realized_capture(game, ctx, opp.profile, seat_keys)
        mets: dict[str, float] = {}
        for name, fn in CHEAP_METRICS.items():
            if name == "pert_sens":
                mets[name] = fn(game, opp.profile, seat_keys, ctx, k=pert_k)
            else:
                mets[name] = fn(game, opp.profile, seat_keys, ctx)
        mets["nashconv_opp*"] = m_nashconv_opp(game, opp.profile, seat_keys, ctx)  # reference
        row = OppRow(label=opp.label, klass=opp.klass, split=opp.split,
                     metrics=mets, y_ceil=y_ceil, y_cap=y_cap, ev_base=base)
        rep.rows.append(row)
        if on_row is not None:
            on_row(i, opp, row)

    # correlations by split; target selectable (Y_ceil gated, Y_cap reported). eq anchors
    # excluded (both X and Y ≈ 0 there → they would only INFLATE a positive ρ; excluding
    # them is the conservative choice).
    def rho_for(name: str, split: str, target: str = "y_ceil") -> float:
        xs, ys = [], []
        for r in rep.rows:
            if r.split == split and r.klass not in ("eq", "near_eq"):
                xs.append(r.metrics[name])
                ys.append(getattr(r, target))
        return spearman(xs, ys)

    all_metric_names = list(CHEAP_METRICS) + ["nashconv_opp*"]
    for name in all_metric_names:
        rep.metric_rho[name] = MetricRho(
            name=name,
            rho_fit=rho_for(name, "fit"),
            rho_held=rho_for(name, "held"),
            rho_transfer=rho_for(name, "transfer"),
            rho_cap_held=rho_for(name, "held", target="y_cap"),
            is_baseline=(name == BASELINE_METRIC),
        )

    # pipeline sanity (§38.DEV1): a PERFECT predictor (the target itself) must score ρ≈1
    # — confirms the Spearman path can detect a perfect monotone relation. Under the
    # Y_ceil target the load-bearing anti-circularity guard is instead the zero-Hero-BR
    # property of the cheap metrics (certified in the test suite), not this sanity.
    def rho_perfect(split: str) -> float:
        xs = [r.y_ceil for r in rep.rows if r.split == split and r.klass != "eq"]
        return spearman(xs, xs)
    rep.control_rho_perfect_held = rho_perfect("held")

    # ---- gates (§38.4 / §38.5, target = Y_ceil) ----
    base_rho_held = abs(rep.metric_rho[BASELINE_METRIC].rho_held)
    cheap_candidates = [m for n, m in rep.metric_rho.items()
                        if n in CHEAP_METRICS and n != BASELINE_METRIC]

    def passes(m: MetricRho) -> bool:
        signed_stable = (m.rho_fit == 0.0) or (m.rho_held == 0.0) or \
                        (math.copysign(1, m.rho_fit) == math.copysign(1, m.rho_held))
        transfer_stable = (m.rho_transfer == 0.0) or (m.rho_held == 0.0) or \
                          (math.copysign(1, m.rho_transfer) == math.copysign(1, m.rho_held))
        return (abs(m.rho_held) >= rho_star and
                abs(m.rho_held) - base_rho_held >= delta_rho and
                signed_stable and transfer_stable)

    winners = [m.name for m in cheap_candidates if passes(m)]

    # §28 invariants (anti-hallucination)
    rep.gates["Y_cap <= Y_ceil (all opponents, R2)"] = all(
        r.y_cap <= r.y_ceil + 1e-9 for r in rep.rows)
    # §28 on the GATED target: the exact ceiling against a (near-)equilibrium opponent
    # cannot exceed what the eq REFERENCE's own residual imperfection allows — Hero's BR
    # gain vs the eq pair is bounded by the reference's NashConv. So the anchor gate is
    # RELATIVE to eq_nashconv (correct, not a weakening): a rough Leduc eq (slow pure-Python
    # CFR) leaves the anchors slightly exploitable by exactly that residual. NB the α=1
    # REPORTED capture can be negative vs a true eq (an unsafe exploiter that deviates from
    # GTO gets punished) — correct, not a violation, so it is not gated here.
    eq_anchor_tol = max(budget, eq_nc) + 5e-3
    rep.gates["Y_ceil ~ eq-residual vs equilibrium anchors"] = all(
        r.y_ceil <= eq_anchor_tol for r in rep.rows if r.klass == "eq")
    rep.gates["E* alpha within safety budget"] = alpha_star <= alpha_safe + 1e-9
    rep.gates["pipeline sanity: perfect predictor rho_held ~1"] = \
        rep.control_rho_perfect_held >= 0.999
    # the H-Q GO condition for THIS game (both-games AND is applied by the runner)
    rep.gates[f"H-Q GO (some cheap metric beats baseline, this game={game_name})"] = len(winners) > 0

    rep.detail["baseline dist_eq |rho_held|"] = base_rho_held
    for m in cheap_candidates:
        rep.detail[f"{m.name} rho_held"] = m.rho_held
        rep.detail[f"{m.name} rho_fit"] = m.rho_fit
        rep.detail[f"{m.name} rho_transfer"] = m.rho_transfer
        rep.detail[f"{m.name} rho_cap_held"] = m.rho_cap_held
    rep.detail["reference nashconv_opp* rho_held"] = rep.metric_rho["nashconv_opp*"].rho_held
    rep.detail["max Y_cap (held)"] = max((r.y_cap for r in rep.rows if r.split == "held"), default=0.0)
    rep.detail["max Y_ceil (held)"] = max((r.y_ceil for r in rep.rows if r.split == "held"), default=0.0)
    rep.detail["winners"] = float(len(winners))
    rep.verdict = (f"GO-in-{game_name}: " + ",".join(winners)) if winners else f"FLAT-in-{game_name}"
    return rep
