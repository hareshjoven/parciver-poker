"""R-MAPSS Step 1 — multiplayer exploitation spine (3p Kuhn, tabular/exact).

Implements the Step-1 contract of RMAPSS.md §33:

* a diverse population of fixed 3p opponents (feature I), split FIT / HELD-OUT (R4);
* an INDEPENDENT opponent model (feature H) — the marginal, public-history-
  conditioned action model Hero can actually observe (it cannot see opponents'
  cards). Best-responding to this marginal (A2) is generally weaker than the exact
  best response to the true card-conditioned policies (A2.5 = the ceiling); that
  gap is exactly refinement R2;
* a SAFETY-GATED deviation (features K/L, α-blend): π_final = (1-α)·π_CFR + α·π_BR,
  with α the largest value (up to the confidence cap) whose worst-case value stays
  within an exploitability budget of the CFR anchor's worst case;
* the §28 exact control: the exact best response to the fixed pair is the EV
  ceiling, and Hero's worst case is measured against an omniscient coordinated
  adversary (a conservative, exact upper bound on real-opponent exploitability).

Everything is exact (full traversal) — no sampling noise — so "out-of-sample"
means a HELD-OUT opponent set (R4), and a positive held-out marginal EV is a real
edge with no confidence interval needed.

R1/R2 note: against FIXED opponents the exact BR is the ceiling and captures all
interaction, so there is nothing here for an interaction KERNEL to add — that is
Step 2 (adaptive opponents). Step 1 validates the spine + safety gate + controls.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Hashable

from ..games.kuhn_multi import KuhnPokerN
from ..solvers.cfr_multi import MultiCFR
from ..eval.exploitability import (
    best_response_strategy,
    best_response_value,
    on_policy_value,
)

Strategy = dict[str, list[float]]


# --------------------------------------------------------------------------
# game helpers
# --------------------------------------------------------------------------

def _history_of(key: str) -> str:
    return key.split(":", 1)[1]


def collect_infosets(game) -> dict[int, list[str]]:
    """Infoset keys grouped by acting seat."""
    out: dict[int, list[str]] = defaultdict(list)
    seen: set[str] = set()

    def walk(s: Hashable):
        if game.is_terminal(s):
            return
        if game.is_chance(s):
            for a, _ in game.chance_outcomes(s):
                walk(game.next_state(s, a))
            return
        k = game.infoset_key(s)
        if k not in seen:
            seen.add(k)
            out[game.current_player(s)].append(k)
        for a in game.legal_actions(s):
            walk(game.next_state(s, a))

    walk(game.initial_state())
    return out


# --------------------------------------------------------------------------
# opponent population (feature I)
# --------------------------------------------------------------------------
# Kuhn actions are always binary: pre-bet (CHECK_CALL=idx0, BET=idx1); facing a
# bet (FOLD=idx0, CHECK_CALL/call=idx1). A key faces a bet iff its history
# already contains a 'b'.

def opponent_seat_policy(seat_keys: list[str], kind: str, *, eq: Strategy | None = None,
                         eps: float = 0.0) -> Strategy:
    pol: Strategy = {}
    for key in seat_keys:
        facing = "b" in _history_of(key)
        if kind == "equilibrium":
            base = list(eq[key])
        elif kind == "station":      # never folds, never bets
            base = [0.0, 1.0] if facing else [1.0, 0.0]
        elif kind == "maniac":       # always bets / always calls
            base = [0.0, 1.0]
        elif kind == "nit":          # folds to any bet, never bets
            base = [1.0, 0.0]
        elif kind == "uniform":
            base = [0.5, 0.5]
        else:
            raise ValueError(f"unknown opponent kind {kind!r}")
        if eps > 0.0 and eq is not None:  # blend toward equilibrium (near-eq noise)
            base = [(1.0 - eps) * base[i] + eps * eq[key][i] for i in range(2)]
        pol[key] = base
    return pol


@dataclass
class OppPair:
    label: str
    seat1: Strategy
    seat2: Strategy
    klass: str  # "equilibrium" | "suboptimal"
    split: str  # "fit" | "held"

    @property
    def profile(self) -> Strategy:
        return {**self.seat1, **self.seat2}


def build_population(game, eq: Strategy) -> list[OppPair]:
    keys = collect_infosets(game)
    k1, k2 = keys[1], keys[2]

    def s1(kind, eps=0.0):
        return opponent_seat_policy(k1, kind, eq=eq, eps=eps)

    def s2(kind, eps=0.0):
        return opponent_seat_policy(k2, kind, eq=eq, eps=eps)

    pop: list[OppPair] = []
    # --- equilibrium / near-equilibrium (deviation must MATCH CFR here) ---
    pop.append(OppPair("eq,eq", s1("equilibrium"), s2("equilibrium"), "equilibrium", "fit"))
    # near-equilibrium = eq perturbed 10% toward uniform (genuinely off, still
    # card-dependent -> the marginal model is slightly misspecified here too).
    pop.append(OppPair("nearEq(10%off)", s1("uniform", .9), s2("uniform", .9), "equilibrium", "held"))
    # --- suboptimal FIT (deviation must BEAT CFR here) ---
    pop.append(OppPair("station,station", s1("station"), s2("station"), "suboptimal", "fit"))
    pop.append(OppPair("maniac,maniac", s1("maniac"), s2("maniac"), "suboptimal", "fit"))
    pop.append(OppPair("nit,nit", s1("nit"), s2("nit"), "suboptimal", "fit"))
    pop.append(OppPair("station,maniac", s1("station"), s2("maniac"), "suboptimal", "fit"))
    pop.append(OppPair("uniform,uniform", s1("uniform"), s2("uniform"), "suboptimal", "fit"))
    pop.append(OppPair("nit,station", s1("nit"), s2("station"), "suboptimal", "fit"))
    # --- suboptimal HELD-OUT (unseen combos -> transfer test, R4) ---
    pop.append(OppPair("maniac,nit", s1("maniac"), s2("nit"), "suboptimal", "held"))
    pop.append(OppPair("station,uniform", s1("station"), s2("uniform"), "suboptimal", "held"))
    pop.append(OppPair("nit,maniac", s1("nit"), s2("maniac"), "suboptimal", "held"))
    pop.append(OppPair("uniform,maniac", s1("uniform"), s2("maniac"), "suboptimal", "held"))
    pop.append(OppPair("noisyStation.2", s1("station", .2), s2("nit", .2), "suboptimal", "held"))
    return pop


# --------------------------------------------------------------------------
# independent (marginal) opponent model (feature H, refinement R2)
# --------------------------------------------------------------------------

def observe_marginal(game, hero_strategy: Strategy, opp_profile: Strategy,
                     seat_keys: dict[int, list[str]], hero: int = 0):
    """The marginal, public-history-conditioned opponent model Hero can OBSERVE
    (it never sees opponents' cards). Returns ``(flat, marg_by_node, mass_by_node)``:

    * ``flat`` — a flattened opponent profile where every ``card:history`` key maps
      to the SAME reach-weighted marginal action distribution for its
      ``(seat, history)`` (the infinite-data limit of Hero's tally);
    * ``marg_by_node[(seat, hist)]`` — that marginal distribution;
    * ``mass_by_node[(seat, hist)]`` — the reach mass at that node (how often Hero
      encounters it), used to weight the deviation-from-equilibrium signal.
    """
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
        probs = (hero_strategy if player == hero else opp_profile).get(key, [1.0 / len(actions)] * len(actions))
        if player != hero:
            c = counts.setdefault((player, _history_of(key)), [0.0] * len(actions))
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
        marg_by_node[node] = [x / s for x in c] if s > 0 else [0.5, 0.5]

    flat: Strategy = {}
    for seat in range(n):
        if seat == hero:
            continue
        for key in seat_keys[seat]:
            node = (seat, _history_of(key))
            flat[key] = marg_by_node.get(node, [0.5, 0.5])
    return flat, marg_by_node, mass_by_node


def deviation_signal(marg_obs, mass_obs, marg_eq) -> float:
    """Reach-weighted mean L1 distance between the OBSERVED opponent marginal and
    the EQUILIBRIUM marginal — Hero's (card-free) evidence that the opponents are
    detectably non-equilibrium. ~0 for equilibrium opponents (so the deviation
    switches off), large for gross distortions. This is the confidence signal that
    schedules α (confidence-scheduled restricted response)."""
    num = 0.0
    den = 0.0
    for node, m in mass_obs.items():
        if m <= 0:
            continue
        po = marg_obs[node]
        pe = marg_eq.get(node, [0.5] * len(po))
        l1 = sum(abs(po[i] - pe[i]) for i in range(len(po)))
        num += m * l1
        den += m
    return num / den if den > 0 else 0.0


# --------------------------------------------------------------------------
# exact safety control (§28): worst case vs an omniscient coordinated adversary
# --------------------------------------------------------------------------

def clairvoyant_team_min(game, hero_strategy: Strategy, hero: int = 0) -> float:
    """Hero's worst-case EV against an OMNISCIENT, coordinated adversary that
    controls every non-Hero seat and minimises Hero's EV. Because that adversary
    has perfect information, per-node minimisation is exact (a DP). This is a
    conservative, EXACT upper bound on Hero's exploitability by any real,
    decentralised opponents — the safety yardstick for the α-gate.
    """
    def walk(state):
        if game.is_terminal(state):
            return game.terminal_utility_for(state, hero)
        if game.is_chance(state):
            return sum(p * walk(game.next_state(state, a)) for a, p in game.chance_outcomes(state))
        actions = game.legal_actions(state)
        if game.current_player(state) == hero:
            key = game.infoset_key(state)
            probs = hero_strategy.get(key, [1.0 / len(actions)] * len(actions))
            return sum(probs[i] * walk(game.next_state(state, a)) for i, a in enumerate(actions))
        return min(walk(game.next_state(state, a)) for a in actions)

    return walk(game.initial_state())


def blend(hero_cfr: Strategy, br: Strategy, alpha: float, hero_keys: list[str]) -> Strategy:
    out: Strategy = {}
    for k in hero_keys:
        base = hero_cfr.get(k)
        tgt = br.get(k, base)
        if base is None:
            base = tgt
        out[k] = [(1.0 - alpha) * base[i] + alpha * tgt[i] for i in range(len(base))]
    return out


def safety_gated_alpha(game, hero_cfr: Strategy, br: Strategy, hero_keys: list[str], *,
                       safe_floor: float, budget: float, alpha_max: float = 1.0,
                       step: float = 0.01) -> float:
    """Largest α in [0, alpha_max] whose worst-case value stays within `budget` of
    the CFR anchor's worst case. Worst case is monotone non-increasing in α, so
    scanning down from alpha_max returns the exploitability-bounded α."""
    a = alpha_max
    while a > 1e-9:
        pi = blend(hero_cfr, br, a, hero_keys)
        if clairvoyant_team_min(game, pi) >= safe_floor - budget - 1e-12:
            return a
        a -= step
    return 0.0


# --------------------------------------------------------------------------
# per-pair evaluation + the Step-1 experiment
# --------------------------------------------------------------------------

@dataclass
class PairResult:
    label: str
    klass: str
    split: str
    ev_a0: float          # CFR baseline EV vs the true pair
    ev_a2: float          # independent (marginal) model BR, realized vs true pair
    ev_final: float       # safety-gated α-blend, realized vs true pair
    ev_ceiling: float     # exact BR to the true pair (A2.5 ceiling, §28 control)
    alpha: float
    worstcase_final: float
    worstcase_cfr: float
    exploit_gain: float   # ev_final - ev_a0
    marginal_gap: float   # ev_ceiling - ev_a2 (R2: what the marginal model leaves)
    safety_ok: bool


@dataclass
class Step1Report:
    results: list[PairResult] = field(default_factory=list)
    gates: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, float] = field(default_factory=dict)
    eq_nashconv: float = 0.0

    @property
    def all_pass(self) -> bool:
        return all(self.gates.values())


def run_step1(*, cfr_iters: int = 8000, budget: float = 0.02, alpha_max: float = 1.0,
              kappa: float = 0.35, margin_sub: float = 0.02, tol_eq: float = 5e-3,
              on_metric=None) -> Step1Report:
    """Run the full Step-1 spine + controls over the population and evaluate the
    gates. `on_metric(i, pair, res)` is an optional callback for telemetry.

    α is the confidence-scheduled restricted response: α = min(α_conf, α_safe),
    where α_conf = min(1, deviation_signal / kappa) switches the deviation OFF
    against opponents that look equilibrium-like (so CFR is matched there), and
    α_safe caps the worst-case exploitability within `budget` of the CFR anchor.
    """
    from ..eval.exploitability import nash_conv_multiplayer

    game = KuhnPokerN(3)
    seat_keys = collect_infosets(game)
    hero_keys = seat_keys[0]

    # A0 anchor: CFR (no >=3p guarantee; NashConv checked exactly).
    solver = MultiCFR(game, variant="cfr+")
    solver.run(cfr_iters)
    eq = solver.average_strategy()
    hero_cfr = {k: eq[k] for k in hero_keys}
    eq_nc = nash_conv_multiplayer(game, eq)
    wc_cfr = clairvoyant_team_min(game, hero_cfr)

    # The equilibrium marginal — Hero's card-free reference for "what an
    # equilibrium opponent looks like", against which it detects deviation.
    eq_opp = {k: eq[k] for k in seat_keys[1] + seat_keys[2]}
    _, eq_marg, _ = observe_marginal(game, hero_cfr, eq_opp, seat_keys)

    pop = build_population(game, eq)
    rep = Step1Report(eq_nashconv=eq_nc)

    for i, pair in enumerate(pop):
        true_opp = pair.profile
        ev_a0 = on_policy_value(game, {**hero_cfr, **true_opp}, 0)
        ev_ceiling = best_response_value(game, true_opp, 0)

        # independent (marginal) model, observed under Hero=CFR, then BR to it.
        marg, marg_obs, mass_obs = observe_marginal(game, hero_cfr, true_opp, seat_keys)
        a2_strat = best_response_strategy(game, marg, 0)
        ev_a2 = on_policy_value(game, {**hero_cfr, **a2_strat, **true_opp}, 0)

        # confidence-scheduled, safety-gated deviation toward the marginal-model BR.
        signal = deviation_signal(marg_obs, mass_obs, eq_marg)
        alpha_conf = min(1.0, signal / kappa)
        alpha_safe = safety_gated_alpha(game, hero_cfr, a2_strat, hero_keys,
                                        safe_floor=wc_cfr, budget=budget, alpha_max=alpha_max)
        alpha = min(alpha_conf, alpha_safe)
        pi_final = blend(hero_cfr, a2_strat, alpha, hero_keys)
        ev_final = on_policy_value(game, {**pi_final, **true_opp}, 0)
        wc_final = clairvoyant_team_min(game, pi_final)

        res = PairResult(
            label=pair.label, klass=pair.klass, split=pair.split,
            ev_a0=ev_a0, ev_a2=ev_a2, ev_final=ev_final, ev_ceiling=ev_ceiling,
            alpha=alpha, worstcase_final=wc_final, worstcase_cfr=wc_cfr,
            exploit_gain=ev_final - ev_a0, marginal_gap=ev_ceiling - ev_a2,
            safety_ok=wc_final >= wc_cfr - budget - 1e-9,
        )
        rep.results.append(res)
        if on_metric is not None:
            on_metric(i, pair, res)

    # ---- gates (RMAPSS §33.G) ----
    sub = [r for r in rep.results if r.klass == "suboptimal"]
    eqp = [r for r in rep.results if r.klass == "equilibrium"]
    sub_held = [r for r in sub if r.split == "held"]

    rep.gates["safety: within exploitability budget (all pairs)"] = all(r.safety_ok for r in rep.results)
    rep.gates["beats CFR vs suboptimal (all)"] = all(r.exploit_gain > margin_sub for r in sub)
    rep.gates["beats CFR vs suboptimal (HELD-OUT, R4)"] = all(r.exploit_gain > margin_sub for r in sub_held)
    rep.gates["matches CFR vs equilibrium (|gain|<=tol)"] = all(abs(r.exploit_gain) <= tol_eq for r in eqp)
    rep.gates["marginal BR <= exact BR ceiling (R2)"] = all(
        r.ev_a2 <= r.ev_ceiling + 1e-9 for r in rep.results)
    rep.gates["never worse than CFR by > budget (all pairs)"] = all(
        r.ev_final >= r.ev_a0 - budget - 1e-9 for r in rep.results)

    rep.detail["mean exploit gain (suboptimal)"] = sum(r.exploit_gain for r in sub) / len(sub)
    rep.detail["mean exploit gain (suboptimal HELD-OUT)"] = sum(r.exploit_gain for r in sub_held) / len(sub_held)
    rep.detail["mean ceiling gain (suboptimal)"] = sum(r.ev_ceiling - r.ev_a0 for r in sub) / len(sub)
    rep.detail["mean marginal gap A2.5-A2 (suboptimal)"] = sum(r.marginal_gap for r in sub) / len(sub)
    rep.detail["max |exploit gain| (equilibrium)"] = max(abs(r.exploit_gain) for r in eqp)
    rep.detail["eq NashConv (exact)"] = eq_nc
    return rep
