// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
//! Read-only analysis of a SAVED blueprint (average strategy), for the dashboard's
//! three summary panels — action-frequency, bet-sizing, and head-to-head. It never
//! touches a running solve: the sidecar hands it a `blueprint.bin` snapshot the
//! trainer wrote to disk, and everything here is a fresh Monte-Carlo walk of the
//! real game using that strategy. Compute lives in Rust (like the solver) so the
//! sidecar stays a thin driver.
//!
//! Two products, both from playing the blueprint through the certified NLHE engine
//! (so action legality and bet sizing come from the game, never re-derived):
//!   * **self-play stats** — deal a hand, play the blueprint against ITSELF, and at
//!     every decision node add the strategy's action mix to that street's
//!     accumulator (each node is visited in proportion to its reach under the
//!     blueprint, so the normalized totals are the reach-weighted average action /
//!     bet-size distributions per street);
//!   * **head-to-head** — play the blueprint against a passive calling-station
//!     baseline (never folds: checks, calls any bet, calls a shove all-in), both
//!     seats equally, and score the blueprint's winnings in bb/100. A calling
//!     station is maximally exploitable by value betting, so a real blueprint beats
//!     it by a clear positive margin — the sanity signal the panel shows.

use rustc_hash::FxHashMap;

use crate::game::Game;
use crate::nlhe::{ActLabel, Nlhe, NlheState};

/// Deterministic SplitMix64 — the engine's shared generator, local here so the
/// analysis draws a reproducible stream independent of the solver's.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

/// The five action categories the dashboard's action-frequency panel bins into,
/// in its display order: fold, check, call, raise/bet, all-in.
pub(crate) const N_CATS: usize = 5;
/// The three bet-size buckets: 0.5x pot, 1.0x pot, all-in.
pub(crate) const N_SIZES: usize = 3;
const N_STREETS: usize = 4; // P, F, T, R

/// Reach-weighted per-street action and bet-size mass from blueprint self-play. Not
/// yet normalized — the caller divides each street's row by its own total (so an
/// unvisited street stays all-zero and is reported as absent).
pub(crate) struct BlueprintStats {
    /// `[street][category]` — fold / check / call / raise / all-in.
    pub action: [[f64; N_CATS]; N_STREETS],
    /// `[street][size]` — 0.5pot / 1.0pot / all-in (over aggressive actions only).
    pub bet: [[f64; N_SIZES]; N_STREETS],
}

impl BlueprintStats {
    fn zero() -> Self {
        BlueprintStats {
            action: [[0.0; N_CATS]; N_STREETS],
            bet: [[0.0; N_SIZES]; N_STREETS],
        }
    }
}

#[inline]
fn street_index(tag: u8) -> usize {
    match tag {
        b'P' => 0,
        b'F' => 1,
        b'T' => 2,
        b'R' => 3,
        _ => 0,
    }
}

#[inline]
fn category_index(label: ActLabel) -> usize {
    match label {
        ActLabel::Fold => 0,
        ActLabel::Check => 1,
        ActLabel::Call => 2,
        ActLabel::Bet(_) => 3,
        ActLabel::AllIn => 4,
    }
}

/// Bet-size bucket for a sized bet's pot-fraction. The blueprint grid uses {0.5, 1.0};
/// any other fraction falls into the 1.0-pot bucket (none occur in practice).
#[inline]
fn size_index(frac: f64) -> usize {
    if (frac - 0.5).abs() < 1e-9 {
        0
    } else {
        1
    }
}

/// Look up the blueprint's action distribution for an infoset, or `None` to mean
/// "unvisited — play uniformly over the `n` legal actions" (the solver's own default
/// for a never-reached node).
#[inline]
fn strategy_at<'a>(strat: &'a FxHashMap<Box<[u8]>, Vec<f32>>, key: &[u8]) -> Option<&'a [f32]> {
    strat.get(key).map(|v| v.as_slice())
}

/// Sample an action index from `probs` (or uniform over `n` if `probs` is `None`).
#[inline]
fn sample_action(probs: Option<&[f32]>, n: usize, rng: &mut Rng) -> usize {
    let r = rng.next_f64();
    match probs {
        Some(p) => {
            let mut acc = 0.0f64;
            for (i, &pi) in p.iter().enumerate() {
                acc += pi as f64;
                if r < acc {
                    return i;
                }
            }
            n - 1
        }
        None => ((r * n as f64) as usize).min(n - 1),
    }
}

/// Sample a chance outcome index in proportion to its probability.
#[inline]
fn sample_outcome(probs: &[f64], rng: &mut Rng) -> usize {
    let r = rng.next_f64();
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return i;
        }
    }
    probs.len() - 1
}

/// Play `hands` self-play hands and accumulate the reach-weighted per-street action
/// and bet-size mass (see module docs). `key_buf` is a reused scratch buffer.
pub(crate) fn self_play_stats(
    game: &Nlhe,
    strat: &FxHashMap<Box<[u8]>, Vec<f32>>,
    hands: u64,
    seed: u64,
) -> BlueprintStats {
    let mut stats = BlueprintStats::zero();
    let mut rng = Rng::new(seed ^ 0x5150_5F53_5441_5453); // "PLPSTATS"-ish salt
    let mut key = Vec::<u8>::with_capacity(48);

    for _ in 0..hands {
        let mut s: NlheState = game.initial_state();
        loop {
            if game.is_terminal(&s) {
                break;
            }
            if game.is_chance(&s) {
                s = game.sample_chance_with(&s, |probs| sample_outcome(probs, &mut rng));
                continue;
            }
            let n = game.num_actions(&s);
            key.clear();
            game.infoset_key_into(&s, &mut key);
            let st = street_index(key[0]);
            let probs = strategy_at(strat, &key);
            let labels = game.labeled_actions(&s);
            let uniform = 1.0 / n as f64;
            for (i, &label) in labels.iter().enumerate() {
                let p = probs.map(|pr| pr[i] as f64).unwrap_or(uniform);
                stats.action[st][category_index(label)] += p;
                match label {
                    ActLabel::Bet(frac) => stats.bet[st][size_index(frac)] += p,
                    ActLabel::AllIn => stats.bet[st][2] += p,
                    _ => {}
                }
            }
            let idx = sample_action(probs, n, &mut rng);
            s = game.next_state(&s, idx);
        }
    }
    stats
}

/// The passive calling-station baseline: never folds — checks when it can, otherwise
/// calls, otherwise calls a shove by going all-in. Returns the index (in the legal
/// list) of its chosen action. Maximally exploitable by value betting.
fn calling_station_index(labels: &[ActLabel]) -> usize {
    if let Some(i) = labels.iter().position(|l| matches!(l, ActLabel::Check)) {
        return i;
    }
    if let Some(i) = labels.iter().position(|l| matches!(l, ActLabel::Call)) {
        return i;
    }
    if let Some(i) = labels.iter().position(|l| matches!(l, ActLabel::AllIn)) {
        return i;
    }
    0 // unreachable in a well-formed node, but never panic
}

/// Head-to-head bb/100: play the blueprint against the calling station for `hands`
/// hands, alternating which seat the blueprint occupies so position averages out,
/// and return the blueprint's win rate in big blinds per 100 hands. Utilities are
/// player-0-view, so the blueprint negates when it sits in seat 1.
pub(crate) fn head_to_head(
    game: &Nlhe,
    strat: &FxHashMap<Box<[u8]>, Vec<f32>>,
    hands: u64,
    seed: u64,
) -> f64 {
    if hands == 0 {
        return 0.0;
    }
    let mut rng = Rng::new(seed ^ 0x4832_485F_4241_5345); // "H2H_BASE"-ish salt
    let mut key = Vec::<u8>::with_capacity(48);
    let mut net: f64 = 0.0;

    for h in 0..hands {
        let bp_seat = (h & 1) as usize; // blueprint plays seat 0 then seat 1, alternating
        let mut s: NlheState = game.initial_state();
        let u = loop {
            if game.is_terminal(&s) {
                break game.terminal_utility(&s); // player-0 view
            }
            if game.is_chance(&s) {
                s = game.sample_chance_with(&s, |probs| sample_outcome(probs, &mut rng));
                continue;
            }
            let n = game.num_actions(&s);
            let idx = if game.current_player(&s) == bp_seat {
                key.clear();
                game.infoset_key_into(&s, &mut key);
                sample_action(strategy_at(strat, &key), n, &mut rng)
            } else {
                calling_station_index(&game.labeled_actions(&s))
            };
            s = game.next_state(&s, idx);
        };
        net += if bp_seat == 0 { u } else { -u };
    }

    let bb = game.big_blind() as f64;
    (net / hands as f64) / bb * 100.0
}

/// Read-only match result for two saved policies. `stddev_bb_per_hand` is the
/// sample standard deviation of individual hand returns in big blinds; callers
/// retain the per-hand unit so later protocol code can form paired/block
/// intervals without confusing it with bb/100.
pub(crate) struct PolicyMatchStats {
    pub hands: u64,
    pub bb_per_100: f64,
    pub stddev_bb_per_hand: f64,
}

/// Play two saved average strategies against each other, alternating which seat
/// `hero` occupies so position averages out. This is deliberately a thin,
/// read-only primitive: it neither changes either policy nor applies any leak,
/// response, confidence, or safety logic.
pub(crate) fn head_to_head_policies(
    game: &Nlhe,
    hero: &FxHashMap<Box<[u8]>, Vec<f32>>,
    opponent: &FxHashMap<Box<[u8]>, Vec<f32>>,
    hands: u64,
    seed: u64,
) -> Result<PolicyMatchStats, String> {
    if hands == 0 {
        return Err("policy match requires at least one hand".to_string());
    }
    let mut rng = Rng::new(seed ^ 0x5032_505F_4D41_5443); // "P2P_MATC"-ish salt
    let mut key = Vec::<u8>::with_capacity(48);
    let mut sum_bb = 0.0f64;
    let mut sum_sq_bb = 0.0f64;
    let bb = game.big_blind() as f64;

    for h in 0..hands {
        let hero_seat = (h & 1) as usize;
        let mut s: NlheState = game.initial_state();
        let u = loop {
            if game.is_terminal(&s) {
                break game.terminal_utility(&s);
            }
            if game.is_chance(&s) {
                s = game.sample_chance_with(&s, |probs| sample_outcome(probs, &mut rng));
                continue;
            }
            let n = game.num_actions(&s);
            key.clear();
            game.infoset_key_into(&s, &mut key);
            let policy = if game.current_player(&s) == hero_seat {
                hero
            } else {
                opponent
            };
            let probs = strategy_at(policy, &key);
            if let Some(row) = probs {
                if row.len() != n {
                    return Err(format!(
                        "saved policy row has {} actions but game requires {n} at {}",
                        row.len(),
                        String::from_utf8_lossy(&key)
                    ));
                }
            }
            let idx = sample_action(probs, n, &mut rng);
            s = game.next_state(&s, idx);
        };
        let hero_chips = if hero_seat == 0 { u } else { -u };
        let x = hero_chips / bb;
        sum_bb += x;
        sum_sq_bb += x * x;
    }

    let n = hands as f64;
    let mean = sum_bb / n;
    let variance = if hands > 1 {
        ((sum_sq_bb - sum_bb * sum_bb / n) / (n - 1.0)).max(0.0)
    } else {
        0.0
    };
    Ok(PolicyMatchStats {
        hands,
        bb_per_100: mean * 100.0,
        stddev_bb_per_hand: variance.sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mccfr::{MCCFRSolver, Variant};
    use crate::nlhe::NlheConfig;

    /// Build a strategy map (byte keys -> f32 probs) from a solver's average
    /// strategy, the same shape `load_average_strategy` returns.
    fn strat_from_solver(solver: &MCCFRSolver) -> FxHashMap<Box<[u8]>, Vec<f32>> {
        solver
            .average_strategy()
            .into_iter()
            .map(|(k, v)| {
                (
                    k.into_bytes().into_boxed_slice(),
                    v.into_iter().map(|x| x as f32).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn self_play_and_head_to_head_are_well_formed() {
        // Small flop-only game (Exact abstraction, no bucket tables needed). Solve a
        // little, then analyze: every visited street's action mix must normalize to a
        // proper distribution, and head-to-head must be a finite number.
        let game = Nlhe::new(NlheConfig::small());
        let mut solver = MCCFRSolver::new(Variant::Linear, 1);
        solver.run(&game, 100_000);
        let strat = strat_from_solver(&solver);

        let stats = self_play_stats(&game, &strat, 20_000, 7);
        let mut any_street = false;
        for st in 0..N_STREETS {
            let total: f64 = stats.action[st].iter().sum();
            if total > 0.0 {
                any_street = true;
                // Every category mass is non-negative and the row is a distribution.
                for &m in &stats.action[st] {
                    assert!(m >= 0.0);
                }
                // Bet mass never exceeds the raise+all-in action mass on that street.
                let bet_total: f64 = stats.bet[st].iter().sum();
                let aggressive = stats.action[st][3] + stats.action[st][4];
                assert!(
                    bet_total <= aggressive + 1e-9,
                    "bet mass exceeds aggressive mass"
                );
            }
        }
        assert!(any_street, "no streets were visited in self-play");

        let hh = head_to_head(&game, &strat, 20_000, 11);
        assert!(hh.is_finite(), "head-to-head bb/100 not finite: {hh}");

        let p2p = head_to_head_policies(&game, &strat, &strat, 20_000, 13).unwrap();
        let again = head_to_head_policies(&game, &strat, &strat, 20_000, 13).unwrap();
        assert_eq!(p2p.hands, 20_000);
        assert!(p2p.bb_per_100.is_finite());
        assert!(p2p.stddev_bb_per_hand.is_finite() && p2p.stddev_bb_per_hand >= 0.0);
        assert_eq!(
            p2p.bb_per_100, again.bb_per_100,
            "fixed seed must replay exactly"
        );
        assert!(head_to_head_policies(&game, &strat, &strat, 0, 13).is_err());
    }
}
