// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
//! N-player general-sum CFR (vanilla / CFR+ / Linear / DCFR), the multiplayer
//! sibling of [`crate::cfr`] for the R-MAPSS track (RMAPSS.md §33, Step 0.2).
//!
//! It is a single traversal that returns a PER-PLAYER value vector and updates
//! every player's regrets against their OWN utility (`terminal_utility_for`), so
//! it is correct for general-sum games with any number of players. Restricted to
//! the validated SIMULTANEOUS-update path, exactly like `cfr.rs`.
//!
//! Two honesty notes:
//!   * For `n = 2` zero-sum, the update reduces algebraically to `cfr.rs` (player
//!     1's `av1 - v1 = -(av0 - v0)`, and the counterfactual reach is the same), so
//!     `KuhnN(2)` solved here reaches the known Kuhn value -1/18 — a behavioural
//!     cross-check in the tests.
//!   * For `n ≥ 3` general-sum, CFR has NO equilibrium guarantee (Gibson 2013).
//!     We do NOT assert convergence to Nash. The average strategy is used as an
//!     approximate, low-NashConv profile (empirically true in 3p Kuhn — measured,
//!     not assumed) to seed the opponent population and the A0 baseline. NashConv
//!     is always checked EXACTLY and independently by `crate::exploit`.

use std::collections::HashMap;

use rustc_hash::FxHashMap;

use crate::cfr::Variant;
use crate::game::Game;

struct Node {
    regret_sum: Vec<f64>,
    strategy_sum: Vec<f64>,
}

impl Node {
    fn new(num_actions: usize) -> Self {
        Node {
            regret_sum: vec![0.0; num_actions],
            strategy_sum: vec![0.0; num_actions],
        }
    }
}

pub struct NpCFRSolver {
    variant: Variant,
    nodes: FxHashMap<String, Node>,
    iterations: u64,
    num_players: usize,
}

impl NpCFRSolver {
    pub fn new(variant: Variant, num_players: usize) -> Self {
        NpCFRSolver {
            variant,
            nodes: FxHashMap::default(),
            iterations: 0,
            num_players,
        }
    }

    #[allow(dead_code)]
    pub fn num_infosets(&self) -> usize {
        self.nodes.len()
    }

    /// Positive-regret matching; uniform when no regret is positive. Identical to
    /// the `cfr.rs` helper.
    fn regret_matching(regret_sum: &[f64]) -> Vec<f64> {
        let n = regret_sum.len();
        let mut out = Vec::with_capacity(n);
        let mut total = 0.0;
        for &r in regret_sum {
            let p = if r > 0.0 { r } else { 0.0 };
            out.push(p);
            total += p;
        }
        if total > 0.0 {
            for x in out.iter_mut() {
                *x /= total;
            }
        } else {
            let u = 1.0 / n as f64;
            for x in out.iter_mut() {
                *x = u;
            }
        }
        out
    }

    fn strategy_weight(&self, t: u64) -> f64 {
        let gamma = if let Some(g) = self.variant.gamma {
            g
        } else if self.variant.linear_averaging {
            1.0
        } else {
            0.0
        };
        (t as f64).powf(gamma)
    }

    /// Per-player expected value below `state` under the current strategy, also
    /// accumulating regrets and average-strategy weight for every acting player.
    fn traverse<G: Game>(
        &mut self,
        game: &G,
        state: &G::State,
        reach: &[f64],
        reach_chance: f64,
    ) -> Vec<f64> {
        let n = self.num_players;
        if game.is_terminal(state) {
            return (0..n)
                .map(|i| game.terminal_utility_for(state, i))
                .collect();
        }
        if game.is_chance(state) {
            let mut value = vec![0.0; n];
            for (child, prob) in game.chance_outcomes(state) {
                let cv = self.traverse(game, &child, reach, reach_chance * prob);
                for i in 0..n {
                    value[i] += prob * cv[i];
                }
            }
            return value;
        }

        let player = game.current_player(state);
        let na = game.num_actions(state);
        let key = game.infoset_key(state);

        let regret_snapshot = {
            let node = self
                .nodes
                .entry(key.clone())
                .or_insert_with(|| Node::new(na));
            node.regret_sum.clone()
        };
        let strategy = Self::regret_matching(&regret_snapshot);

        let mut action_values: Vec<Vec<f64>> = Vec::with_capacity(na);
        let mut value = vec![0.0; n];
        for a in 0..na {
            let child = game.next_state(state, a);
            let mut child_reach = reach.to_vec();
            child_reach[player] *= strategy[a];
            let cv = self.traverse(game, &child, &child_reach, reach_chance);
            for i in 0..n {
                value[i] += strategy[a] * cv[i];
            }
            action_values.push(cv);
        }

        // Counterfactual reach for the acting player EXCLUDES its own reach.
        let cf_reach: f64 = reach_chance
            * (0..n)
                .filter(|&j| j != player)
                .map(|j| reach[j])
                .product::<f64>();
        let own_reach = reach[player];
        let weight = self.strategy_weight(self.iterations);

        let node = self.nodes.get_mut(&key).unwrap();
        for a in 0..na {
            node.regret_sum[a] += cf_reach * (action_values[a][player] - value[player]);
            node.strategy_sum[a] += weight * own_reach * strategy[a];
        }
        value
    }

    fn apply_discounts(&mut self) {
        let v = self.variant;
        if v.floor_regrets_at_zero {
            for node in self.nodes.values_mut() {
                for r in node.regret_sum.iter_mut() {
                    if *r < 0.0 {
                        *r = 0.0;
                    }
                }
            }
            return;
        }
        if v.alpha.is_none() && v.beta.is_none() {
            return; // vanilla
        }
        let t = self.iterations as f64;
        let scale_pos = match v.alpha {
            Some(a) => {
                let ta = t.powf(a);
                ta / (ta + 1.0)
            }
            None => 1.0,
        };
        let scale_neg = match v.beta {
            Some(b) => {
                let tb = t.powf(b);
                tb / (tb + 1.0)
            }
            None => 1.0,
        };
        for node in self.nodes.values_mut() {
            for r in node.regret_sum.iter_mut() {
                *r *= if *r > 0.0 { scale_pos } else { scale_neg };
            }
        }
    }

    pub fn run<G: Game>(&mut self, game: &G, iterations: u64) {
        assert_eq!(
            game.num_players(),
            self.num_players,
            "solver/game player-count mismatch"
        );
        let root = game.initial_state();
        let reach = vec![1.0; self.num_players];
        for _ in 0..iterations {
            self.iterations += 1;
            self.traverse(game, &root, &reach, 1.0);
            self.apply_discounts();
        }
    }

    /// The time-average strategy keyed by infoset — the converged object.
    pub fn average_strategy(&self) -> HashMap<String, Vec<f64>> {
        let mut out = HashMap::with_capacity(self.nodes.len());
        for (key, node) in &self.nodes {
            let total: f64 = node.strategy_sum.iter().sum();
            let probs = if total > 0.0 {
                node.strategy_sum.iter().map(|x| x / total).collect()
            } else {
                let na = node.strategy_sum.len();
                vec![1.0 / na as f64; na]
            };
            out.insert(key.clone(), probs);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::{DCFR, VANILLA};
    use crate::exploit::nash_conv_n;
    use crate::kuhn_n::{KuhnN, KuhnNState};

    /// KuhnN(2) solved by the N-player CFR must reach the known Kuhn value -1/18
    /// to player 0 — a behavioural proof that the generalisation reduces to the
    /// certified 2p game.
    #[test]
    fn two_player_kuhn_reaches_minus_one_eighteenth() {
        let game = KuhnN::new(2);
        let mut solver = NpCFRSolver::new(DCFR, 2);
        solver.run(&game, 5000);
        let strat = solver.average_strategy();

        fn walk(g: &KuhnN, s: &KuhnNState, strat: &HashMap<String, Vec<f64>>) -> f64 {
            if g.is_terminal(s) {
                return g.terminal_utility(s);
            }
            if g.is_chance(s) {
                return g
                    .chance_outcomes(s)
                    .into_iter()
                    .map(|(c, p)| p * walk(g, &c, strat))
                    .sum();
            }
            let key = g.infoset_key(s);
            let probs = &strat[&key];
            (0..g.num_actions(s))
                .map(|i| probs[i] * walk(g, &g.next_state(s, i), strat))
                .sum()
        }
        let value = walk(&game, &game.initial_state(), &strat);
        assert!(
            (value + 1.0 / 18.0).abs() < 2e-3,
            "KuhnN(2) value {value} != -1/18"
        );
    }

    /// Empirical (NOT theoretical — no ≥3p guarantee): CFR in 3p Kuhn drives the
    /// exact, independently-computed NashConv low. Regression bound; the exact
    /// value is printed so drift is visible.
    #[test]
    fn three_player_kuhn_cfr_reaches_low_nashconv() {
        let game = KuhnN::new(3);
        let mut solver = NpCFRSolver::new(VANILLA, 3);
        solver.run(&game, 20000);
        let strat = solver.average_strategy();
        let none = |_: &_| None::<String>;
        let nc = nash_conv_n(&game, &strat, none);
        println!("3p Kuhn CFR NashConv = {nc}");
        assert!(nc < 0.05, "3p Kuhn CFR NashConv {nc} unexpectedly high");
        assert!(nc >= -1e-9, "NashConv must be non-negative, got {nc}");
    }
}
