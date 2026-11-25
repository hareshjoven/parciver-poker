"""R-MAPSS Step 3 (S3.1) — HU exploitation spine on `nlhe_small` (real reduced NLHE).

This ports the validated Step-1 spine (RMAPSS.md §40) from 3p Kuhn onto a *real*
(reduced) heads-up No-Limit Hold'em game — `NlheConfig::small`: a 7-card deck,
2 bb effective stacks, flop-only, **456 infosets**, on the lossless `Exact` card
key so its equilibrium is the true equilibrium and **exact best-response
exploitability is computable and certified** (B3). It is the vehicle chosen with
the author (§40.0) to establish the exploitation spine where the safety control is
EXACT before S3.2 carries it to the full 100bb game (where exploitability is only
an abstraction proxy).

The spine, per fixed opponent:

* observe the opponent's Hero-visible action marginal (`nlhe_small_observed_
  marginal` — the model Hero can estimate from watching actions, never the hole);
* best-respond to that marginal (`nlhe_small_best_response_strategy`) — the blend
  target;
* deviate from the CFR-equilibrium anchor toward that BR by
  `π_final = (1−α)·π_anchor + α·π_BR`, with the **confidence-scheduled,
  safety-gated** `α = min(α_conf, α_safe)`:
    - `α_conf = min(1, deviation_signal / κ)` switches the deviation OFF against
      opponents that look equilibrium-like (`step1.deviation_signal`, scale-free);
    - `α_safe` is the largest α whose **exact** worst-case exploitability
      (`nlhe_small_exploitability`, the certified yardstick — NOT step1's Python
      clairvoyant walk) stays within `budget` of the anchor's.

§28 controls (anti-hallucination): every claimed exploit is checked against the
exact BR ceiling (`nlhe_small_best_response_value`), and the safety yardstick is
the certified exact exploitability. §S: this is the restricted-Nash-response /
confidence-scheduled safe-exploitation spine (StratFormer, Confidence-Scheduled
Restricted Responses, AlphaExploitem, Ganzfried) implemented on real rules — cited,
no novelty claimed. Reuses `step1.blend` / `step1.deviation_signal` verbatim.

Policy evaluation is exact (full traversal via the certified Rust oracle), while
the MCCFR anchor is sampled. The historical "held" population has now been
inspected: these are regression results, not fresh confirmation (RMAPSS §41).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from math import isfinite

import parciver_engine as pe

from .step1 import blend, deviation_signal

Strategy = dict[str, list[float]]

# Confidence scale κ (scale-free: deviation_signal is an L1 of probability
# distributions, so the Kuhn value carries directly, §40 — NOT a per-game knob).
KAPPA = 0.35
# EXPLORATORY thresholds: session 17 lowered C_SUB from .10 to .08 AFTER
# inspecting results and reclassified the hole-blind opponent (RMAPSS §41).
# Retained for regression continuity, NOT a pre-registered confirmation.
# Scale is now calibrated on FIT opponents only; HELD never sets the budget.
PROTOCOL = "exploratory-fit-v1"
C_SUB = 0.08   # beats anchor vs suboptimal HELD-OUT by > 8% of the mean ceiling gain
C_TOL = 0.05   # matches / declines: |gain| <= 5% of S (near-eq AND hidden classes)
C_BUD = 0.10   # safety: exact exploitability may rise by <= 10% of S
# Coarse α scan for the exact-exploitability safety gate (each exploitability call
# is ~0.26 s; monotone-ish in α, and the gate re-checks the actual α used).
ALPHA_GRID = (1.0, 0.85, 0.7, 0.55, 0.4, 0.28, 0.18, 0.1, 0.05, 0.0)

# kind, epsilon, label, class, split. These opponents have already been inspected;
# "held" describes their original role, not a fresh confirmation population.
POPULATION_SPEC = (
    ("eq", 0.0, "eq", "equilibrium", "fit"),
    ("eq", 0.10, "near_eq(.10)", "equilibrium", "held"),
    ("eq", 0.20, "near_eq(.20)", "equilibrium", "held"),
    ("nit", 0.0, "nit", "suboptimal", "fit"),
    ("station", 0.0, "station", "suboptimal", "fit"),
    ("maniac", 0.0, "maniac", "suboptimal", "fit"),
    ("board_blind", 0.0, "board_blind", "hidden", "held"),
    ("nit", 0.25, "nit(.25)", "suboptimal", "held"),
    ("station", 0.25, "station(.25)", "suboptimal", "held"),
    ("maniac", 0.25, "maniac(.25)", "suboptimal", "held"),
)
CONFIRMATION_PROTOCOL = "confirmation-v1"
CONFIRMATION_SEEDS = (7, 19, 41)
CONFIRMATION_ITERS = 2_000_000
CONFIRMATION_SPEC = (
    ("eq", 0.0, "eq", "equilibrium", "fit"),
    ("nit", 0.0, "nit", "suboptimal", "fit"),
    ("station", 0.0, "station", "suboptimal", "fit"),
    ("maniac", 0.0, "maniac", "suboptimal", "fit"),
    ("eq", 0.05, "near_eq(.05)", "equilibrium", "held"),
    ("eq", 0.15, "near_eq(.15)", "equilibrium", "held"),
    ("board_blind", 0.0, "board_blind", "hidden", "boundary"),
) + tuple((kind, eps, f"{kind}({eps:.2f})", "suboptimal", "held")
          for kind in ("nit", "station", "maniac") for eps in (0.15, 0.35, 0.45))


def protocol_settings(protocol: str):
    """Locked population and threshold fractions; §42 approved before new results."""
    if protocol == PROTOCOL:
        return POPULATION_SPEC, C_SUB, C_TOL, C_BUD
    if protocol == CONFIRMATION_PROTOCOL:
        return CONFIRMATION_SPEC, 0.10, 0.05, 0.10
    raise ValueError(f"unknown S3.1 protocol {protocol!r}")


# --------------------------------------------------------------------------
# engine-backed game helpers (all exact, via the certified Rust oracle)
# --------------------------------------------------------------------------

def infoset_structure() -> dict[str, tuple[int, list[str]]]:
    """`{infoset_key: (acting_player, [action category, ...])}` on nlhe_small."""
    return pe.nlhe_small_infoset_structure()


def split_keys(struct: dict[str, tuple[int, list[str]]]) -> tuple[list[str], list[str]]:
    """Hero (seat 0) vs opponent (seat 1) infoset keys."""
    p0 = sorted(k for k, (pl, _) in struct.items() if pl == 0)
    p1 = sorted(k for k, (pl, _) in struct.items() if pl == 1)
    return p0, p1


def solve_anchor(iters: int = 300_000, variant: str = "linear", seed: int = 0) -> Strategy:
    """The CFR-equilibrium anchor `π_anchor` (approximate; MCCFR is far faster than
    full DCFR on nlhe_small — ~8 it/s — and the spine's controls hold for ANY
    anchor). A larger `iters` tightens the anchor's exploitability."""
    return pe.solve_nlhe_small_mccfr(iters, variant, seed)


# --------------------------------------------------------------------------
# opponent population (feature I): behavioural archetypes over opponent keys
# --------------------------------------------------------------------------
# nlhe_small's 2 bb stacks collapse the action set to {fold, check, call, allin}
# (a pot-size open always meets or exceeds the short stack => it is the all-in),
# so the archetypes are clean shove/call/fold distortions.

def _onehot(n: int, i: int) -> list[float]:
    v = [0.0] * n
    v[i] = 1.0
    return v


def _archetype_row(cats: list[str], kind: str) -> list[float]:
    """A fixed action distribution for one infoset given its action categories."""
    n = len(cats)
    if kind == "nit":            # folds to any bet, never puts chips in voluntarily
        for c in ("fold", "check", "call"):
            if c in cats:
                return _onehot(n, cats.index(c))
        return _onehot(n, 0)
    if kind == "station":        # never folds, never raises: the least-aggressive continue
        for c in ("check", "call", "allin"):  # allin only when it is the forced call
            if c in cats:
                return _onehot(n, cats.index(c))
        return _onehot(n, 0)
    if kind == "maniac":         # maximally aggressive: jam whenever possible, never fold
        for c in ("allin", "bet", "call", "check"):
            if c in cats:
                return _onehot(n, cats.index(c))
        return _onehot(n, 0)
    raise ValueError(f"unknown archetype {kind!r}")


def _blend_uniform(pol: Strategy, eps: float) -> Strategy:
    if eps <= 0.0:
        return pol
    out: Strategy = {}
    for k, row in pol.items():
        n = len(row)
        u = 1.0 / n
        out[k] = [(1.0 - eps) * row[i] + eps * u for i in range(n)]
    return out


def build_opponent(struct, opp_keys: list[str], anchor: Strategy, kind: str, *,
                   eps: float = 0.0, board_blind_flat: Strategy | None = None) -> Strategy:
    """A fixed opponent strategy over `opp_keys`. `kind`:
    ``eq`` (the anchor), ``board_blind`` (the anchor's card-blind observed marginal,
    ignoring its own hole), or an archetype (``nit``/``station``/``maniac``);
    ``eps`` blends the result toward uniform (near-eq noise / softened archetypes)."""
    if kind == "eq":
        pol = {k: list(anchor[k]) for k in opp_keys if k in anchor}
    elif kind == "board_blind":
        assert board_blind_flat is not None
        pol = {}
        for k in opp_keys:
            if k in board_blind_flat:
                pol[k] = list(board_blind_flat[k])
            elif k in anchor:
                pol[k] = list(anchor[k])
    else:
        pol = {k: _archetype_row(struct[k][1], kind) for k in opp_keys}
    return _blend_uniform(pol, eps)


@dataclass
class Opp:
    label: str
    strat: Strategy
    klass: str   # "equilibrium" | "suboptimal" | "hidden" (legacy boundary label)
    split: str   # "fit" | "held"


def build_population(struct, opp_keys: list[str], anchor: Strategy,
                     board_blind_flat: Strategy, *, protocol: str = PROTOCOL) -> list[Opp]:
    def mk(kind, eps=0.0):
        return build_opponent(struct, opp_keys, anchor, kind, eps=eps,
                              board_blind_flat=board_blind_flat)

    # Legacy label "board_blind" retained for log compatibility. This opponent
    # actually retains the board and drops private-hole conditioning. Its small
    # nonzero signal is a limitation of this model, not proof of indistinguishability.
    return [Opp(label, mk(kind, eps), klass, split)
            for kind, eps, label, klass, split in protocol_settings(protocol)[0]]


# --------------------------------------------------------------------------
# exact safety control (§28): worst-case exploitability yardstick (Rust oracle)
# --------------------------------------------------------------------------

def _eq_reference(marg_obs: dict, marg_eq: dict) -> dict:
    """The equilibrium observed-marginal reference aligned to `marg_obs`'s nodes,
    filling any node the equilibrium never reaches with a same-length uniform (so
    `step1.deviation_signal`'s fallback never mis-sizes a 3-action node)."""
    ref = {}
    for node, po in marg_obs.items():
        ref[node] = marg_eq[node] if node in marg_eq else [1.0 / len(po)] * len(po)
    return ref


def safety_gated_alpha(anchor: Strategy, anchor_hero: Strategy, br: Strategy,
                       hero_keys: list[str], *, base_expl: float, budget: float,
                       grid: tuple[float, ...] = ALPHA_GRID,
                       anchor_br0: float | None = None) -> tuple[float, float]:
    """Largest α in `grid` whose EXACT worst-case exploitability of the resulting
    full profile stays within `budget` of the anchor's. Returns ``(alpha, expl)``.

    The yardstick is the certified `nlhe_small_exploitability` on the profile with
    Hero's seat replaced by the blend (opponent seat = the equilibrium anchor) — an
    opponent-independent measure. It is the two-seat MEAN: with the opponent
    fixed, Hero's unilateral worst-case loss increase is twice this increase.
    """
    # BR_0 depends only on the unchanged opponent seat. Cache that half of
    # (BR_0 + BR_1)/2; no approximation or change to the exact oracle.
    if anchor_br0 is None:
        anchor_br0 = pe.nlhe_small_best_response_value(anchor, 0)
    for a in grid:
        pi = blend(anchor_hero, br, a, hero_keys)
        expl = (anchor_br0 + pe.nlhe_small_best_response_value({**anchor, **pi}, 1)) / 2.0
        if expl <= base_expl + budget + 1e-12:
            return a, expl
    pi = blend(anchor_hero, br, 0.0, hero_keys)
    return 0.0, pe.nlhe_small_exploitability({**anchor, **pi})


# --------------------------------------------------------------------------
# per-opponent evaluation + the S3.1 experiment
# --------------------------------------------------------------------------

@dataclass
class OppResult:
    label: str
    klass: str
    split: str
    ev_a0: float          # anchor baseline EV vs the true opponent
    ev_a2: float          # observed-marginal-model BR, realized vs true opponent
    ev_final: float       # safety-gated α-blend, realized vs true opponent
    ev_ceiling: float     # exact BR to the true opponent (§28 control ceiling)
    alpha: float
    alpha_conf: float
    alpha_safe: float
    signal: float
    expl_final: float     # exact exploitability of Hero's deviated profile
    exploit_gain: float   # ev_final - ev_a0
    ceiling_gain: float   # ev_ceiling - ev_a0 (the available headroom)
    marginal_gap: float   # ev_ceiling - ev_a2 (R2: what the marginal model leaves)
    safety_ok: bool


@dataclass
class Step3Report:
    results: list[OppResult] = field(default_factory=list)
    gates: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, float] = field(default_factory=dict)
    base_expl: float = 0.0
    scale_S: float = 0.0
    delta_sub: float = 0.0
    tol_eq: float = 0.0
    budget: float = 0.0
    verdict: str = "running"
    protocol: str = PROTOCOL

    @property
    def all_pass(self) -> bool:
        return bool(self.gates) and all(self.gates.values())


def run_step3(*, anchor_iters: int = 300_000, anchor_seed: int = 0, kappa: float = KAPPA,
              on_metric=None, on_thresholds=None, anchor: Strategy | None = None,
              alpha_grid: tuple[float, ...] = ALPHA_GRID,
              protocol: str = PROTOCOL) -> Step3Report:
    """Run the full S3.1 spine + §28 controls over the population and evaluate the
    exploratory regression gates. `on_metric(i, opp, res)` is an optional telemetry
    callback. Pass a precomputed `anchor` to skip the (slow) MCCFR solve.

    Two passes: pass 1 measures the exact-BR ceiling gain per opponent to fix the
    FIT-only EV scale S and budget; pass 2 evaluates all opponents at that budget.
    Historical results informed these thresholds; this is NOT confirmation.
    """
    _spec, c_sub, c_tol, c_bud = protocol_settings(protocol)
    if protocol == CONFIRMATION_PROTOCOL and (
            anchor is not None or anchor_iters != CONFIRMATION_ITERS
            or anchor_seed not in CONFIRMATION_SEEDS or kappa != KAPPA
            or tuple(alpha_grid) != ALPHA_GRID):
        raise ValueError("confirmation-v1 requires the locked anchor budget, seeds, kappa and full grid")
    if not isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    if (not alpha_grid or any(not isfinite(a) or not 0 <= a <= 1 for a in alpha_grid)
            or tuple(sorted(alpha_grid, reverse=True)) != tuple(alpha_grid)):
        raise ValueError("alpha_grid must be nonempty, descending and within [0, 1]")
    if anchor is None:
        anchor = solve_anchor(anchor_iters, "linear", anchor_seed)
    struct = infoset_structure()
    p0_keys, p1_keys = split_keys(struct)
    # Hero's anchor strategy, completed with uniform on any infoset the (sampled)
    # anchor is silent about — that IS the engine's default there, so on-policy and
    # exploitability values are unchanged, and `blend` always has a base row.
    anchor_hero = {}
    for k in p0_keys:
        if k in anchor:
            anchor_hero[k] = list(anchor[k])
        else:
            n = len(struct[k][1])
            anchor_hero[k] = [1.0 / n] * n

    base_expl = pe.nlhe_small_exploitability(anchor)
    # the card-blind equilibrium (opponent's observed marginal under eq×eq): the
    # board_blind archetype AND the eq reference for the confidence signal.
    board_blind_flat, eq_marg, _eq_mass = pe.nlhe_small_observed_model(anchor, anchor, 0)
    anchor_br0 = pe.nlhe_small_best_response_value(anchor, 0)

    pop = build_population(struct, p1_keys, anchor, board_blind_flat, protocol=protocol)
    rep = Step3Report(base_expl=base_expl, protocol=protocol)

    def baseline(opp):
        opp_full = {**anchor, **opp.strat}          # p1 = archetype, p0 = anchor (ignored at p1)
        ev_a0 = pe.nlhe_small_on_policy_value({**anchor_hero, **opp.strat}, 0)
        ev_ceiling = pe.nlhe_small_best_response_value(opp_full, 0)
        return {"opp": opp, "opp_full": opp_full, "ev_a0": ev_a0, "ev_ceiling": ev_ceiling}

    # Freeze and emit FIT calibration BEFORE computing any HELD outcomes.
    fit_rows = {opp.label: baseline(opp) for opp in pop if opp.split == "fit"}
    rep.scale_S = _fit_scale(list(fit_rows.values()))
    rep.delta_sub = c_sub * rep.scale_S
    rep.tol_eq = c_tol * rep.scale_S
    rep.budget = c_bud * rep.scale_S
    if on_thresholds is not None:
        on_thresholds(rep)

    # -- pass 2: confidence-scheduled, safety-gated deviation (real S-scaled budget) --
    for i, opp in enumerate(pop):
        row = fit_rows.get(opp.label)
        if row is None:
            row = baseline(opp)
        opp_full = row["opp_full"]
        ev_a0, ev_ceiling = row["ev_a0"], row["ev_ceiling"]

        # observed (marginal) opponent model + BR to it
        flat, marg_obs, mass_obs = pe.nlhe_small_observed_model(anchor, opp_full, 0)
        a2 = pe.nlhe_small_best_response_strategy(flat, 0)
        ev_a2 = pe.nlhe_small_on_policy_value({**anchor_hero, **a2, **opp.strat}, 0)

        # α_conf: confidence from the Hero-observable deviation-from-eq signal
        signal = deviation_signal(marg_obs, mass_obs, _eq_reference(marg_obs, eq_marg))
        alpha_conf = min(1.0, signal / kappa)
        # α_safe: largest α keeping exact exploitability within the S-scaled budget
        alpha_safe, _e = safety_gated_alpha(anchor, anchor_hero, a2, p0_keys,
                                            base_expl=base_expl, budget=rep.budget,
                                            grid=alpha_grid, anchor_br0=anchor_br0)
        alpha = min(alpha_conf, alpha_safe)
        pi_final = blend(anchor_hero, a2, alpha, p0_keys)
        ev_final = pe.nlhe_small_on_policy_value({**pi_final, **opp.strat}, 0)
        expl_final = pe.nlhe_small_exploitability({**anchor, **pi_final})

        res = OppResult(
            label=opp.label, klass=opp.klass, split=opp.split,
            ev_a0=ev_a0, ev_a2=ev_a2, ev_final=ev_final, ev_ceiling=ev_ceiling,
            alpha=alpha, alpha_conf=alpha_conf, alpha_safe=alpha_safe, signal=signal,
            expl_final=expl_final,
            exploit_gain=ev_final - ev_a0, ceiling_gain=ev_ceiling - ev_a0,
            marginal_gap=ev_ceiling - ev_a2,
            safety_ok=expl_final <= base_expl + rep.budget + 1e-9,
        )
        rep.results.append(res)
        if on_metric is not None:
            on_metric(i, opp, res)

    _finalize(rep)
    return rep


def _finalize(rep: Step3Report) -> None:
    sub = [r for r in rep.results if r.klass == "suboptimal"]
    eqp = [r for r in rep.results if r.klass == "equilibrium"]
    hidden = [r for r in rep.results if r.klass == "hidden"]
    sub_held = [r for r in sub if r.split == "held"]

    try:
        spec, c_sub, c_tol, c_bud = protocol_settings(rep.protocol)
    except ValueError:
        spec, c_sub, c_tol, c_bud = (), 0.0, 0.0, 0.0
    expected = sorted((label, klass, split) for _, _, label, klass, split in spec)
    actual = sorted((r.label, r.klass, r.split) for r in rep.results)
    numeric = [f.name for f in fields(OppResult) if f.name not in ("label", "klass", "split", "safety_ok")]
    valid = (bool(spec) and actual == expected
             and all(isfinite(v) for v in (rep.base_expl, rep.scale_S, rep.delta_sub, rep.tol_eq, rep.budget))
             and rep.base_expl >= -1e-9 and rep.scale_S > 0
             and 0 <= rep.tol_eq < rep.delta_sub and rep.budget >= 0
             and abs(rep.delta_sub - c_sub * rep.scale_S) <= 1e-12
             and abs(rep.tol_eq - c_tol * rep.scale_S) <= 1e-12
             and abs(rep.budget - c_bud * rep.scale_S) <= 1e-12
             and all(all(isfinite(getattr(r, k)) for k in numeric) for r in rep.results)
             and all(0 <= r.alpha <= min(r.alpha_conf, r.alpha_safe) <= 1 for r in rep.results))
    rep.gates.clear()
    rep.gates["complete population and finite measurements"] = valid
    rep.gates["safety: exact exploitability within budget (all opponents)"] = bool(rep.results) and all(
        r.expl_final <= rep.base_expl + rep.budget + 1e-9 for r in rep.results)
    rep.gates["beats anchor vs suboptimal HELD-OUT (> delta_sub)"] = bool(sub_held) and all(
        r.exploit_gain > rep.delta_sub for r in sub_held)
    rep.gates["matches anchor vs near-eq (|gain| <= tol)"] = bool(eqp) and all(
        abs(r.exploit_gain) <= rep.tol_eq for r in eqp)
    # Exploratory boundary criterion changed after results; see §41.DEV1.
    rep.gates["matches hole-blind model boundary (|gain| <= tol)"] = bool(hidden) and all(
        abs(r.exploit_gain) <= rep.tol_eq for r in hidden)
    rep.gates["observed-model BR <= exact BR ceiling (no hallucination, §28)"] = bool(rep.results) and all(
        max(r.ev_a2, r.ev_final, r.ev_a0) <= r.ev_ceiling + 1e-9 for r in rep.results)

    rep.detail["mean exploit gain (suboptimal)"] = _mean(r.exploit_gain for r in sub)
    rep.detail["mean exploit gain (suboptimal HELD-OUT)"] = _mean(r.exploit_gain for r in sub_held)
    rep.detail["mean ceiling gain (suboptimal FIT) = S"] = rep.scale_S
    if hidden:
        rep.detail["hidden ceiling gain (foregone, R2 boundary)"] = _mean(r.ceiling_gain for r in hidden)
        rep.detail["hidden realized gain (near 0 = correct decline)"] = _mean(r.exploit_gain for r in hidden)
    rep.detail["mean marginal gap (ceiling - A2, suboptimal)"] = _mean(r.marginal_gap for r in sub)
    rep.detail["max |exploit gain| (equilibrium)"] = max((abs(r.exploit_gain) for r in eqp), default=0.0)
    rep.detail["max exploitability increase (all)"] = max(
        (r.expl_final - rep.base_expl for r in rep.results), default=0.0)
    rep.detail["anchor exploitability"] = rep.base_expl
    if not rep.all_pass:
        rep.verdict = "FAIL (gate breach)"
    elif rep.protocol == CONFIRMATION_PROTOCOL:
        rep.verdict = "PASS (confirmation seed; all three seeds required)"
    else:
        rep.verdict = "PASS (exploratory regression; confirmation pending)"


def _fit_scale(rows: list[dict]) -> float:
    gains = [r["ev_ceiling"] - r["ev_a0"] for r in rows
             if r["opp"].klass == "suboptimal" and r["opp"].split == "fit"]
    if not gains or any(not isfinite(g) or g < -1e-9 for g in gains):
        raise ValueError("calibration needs finite suboptimal FIT ceiling gains")
    scale = sum(gains) / len(gains)
    if scale <= 0:
        raise ValueError("FIT ceiling-gain scale must be positive")
    return scale


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
