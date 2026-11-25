//! Tabular CFR family (vanilla / CFR+ / Linear / DCFR), a faithful port of
//! `parciver.solvers.cfr` restricted to the validated SIMULTANEOUS-update path
//! (`alternating=False` in the Python solver, its default).
//!
//! All four variants are the same full-tree traversal differing only in how
//! regret and the average strategy are weighted over time, so they share one
//! parameterised solver. Discount parameters follow Brown & Sandholm (AAAI
//! 2019): `alpha` on positive regret, `beta` on negative regret, `gamma` on the
//! average-strategy weight.
//!
//! Correctness is not asserted here by construction — it is the Python
//! exploitability oracle consuming `average_strategy()` that certifies this
//! reproduces a true equilibrium of the same game.

use std::collections::HashMap;

use rustc_hash::FxHashMap;

use crate::game::Game;

/// Weighting scheme for one CFR variant. Mirrors `parciver.solvers.cfr.Variant`.
#[derive(Clone, Copy)]
pub struct Variant {
    /// Python-side variant name; part of the API surface, kept for later phases.
    #[allow(dead_code)]
    pub name: &'static str,
    pub alpha: Option<f64>,
    pub beta: Option<f64>,
    pub gamma: Option<f64>,
    pub floor_regrets_at_zero: bool,
    pub linear_averaging: bool,
}

pub const VANILLA: Variant = Variant {
    name: "cfr",
    alpha: None,
    beta: None,
    gamma: None,
    floor_regrets_at_zero: false,
    linear_averaging: false,
};

pub const CFR_PLUS: Variant = Variant {
    name: "cfr+",
    alpha: None,
    beta: None,
    gamma: None,
    floor_regrets_at_zero: true,
    linear_averaging: true,
};

pub const LINEAR: Variant = Variant {
    name: "linear-cfr",
    alpha: Some(1.0),
    beta: Some(1.0),
    gamma: Some(1.0),
    floor_regrets_at_zero: false,
    linear_averaging: false,
};

pub const DCFR: Variant = Variant {
    name: "dcfr",
    alpha: Some(1.5),
    beta: Some(0.0),
    gamma: Some(2.0),
    floor_regrets_at_zero: false,
    linear_averaging: false,
};

/// Resolve a variant by its Python-side name, or `None` if unknown.
pub fn variant_by_name(name: &str) -> Option<Variant> {
    match name {
        "cfr" => Some(VANILLA),
        "cfr+" => Some(CFR_PLUS),
        "linear-cfr" => Some(LINEAR),
        "dcfr" => Some(DCFR),
        _ => None,
    }
}

/// Per-infoset accumulators.
struct Node {
    regret_sum: Vec<f64>,
    strategy_sum: Vec<f64>,
    player: usize,
}

impl Node {
    fn new(num_actions: usize, player: usize) -> Self {
        Node {
            regret_sum: vec![0.0; num_actions],
            strategy_sum: vec![0.0; num_actions],
            player,
        }
    }
}

pub struct CFRSolver {
    variant: Variant,
    nodes: FxHashMap<String, Node>,
    iterations: u64,
    /// Per-player update clock. Under simultaneous updates both advance every
    /// iteration, so the discount/averaging schedules the paper states against
    /// the per-player clock reduce to the global iteration count — but keeping
    /// the array documents the intent and eases a later alternating port.
    clock: [u64; 2],
}

impl CFRSolver {
    pub fn new(variant: Variant) -> Self {
        CFRSolver {
            variant,
            nodes: FxHashMap::default(),
            iterations: 0,
            clock: [0, 0],
        }
    }

    /// Infoset count after solving; used by tests and future diagnostics.
    #[allow(dead_code)]
    pub fn num_infosets(&self) -> usize {
        self.nodes.len()
    }

    /// Positive-regret matching. Uniform when no regret is positive, which keeps
    /// the early iterate well defined.
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

    /// Weight applied to this iteration's contribution to the average strategy:
    /// an INCREASING t**gamma so later, near-equilibrium iterates dominate.
    /// vanilla -> gamma 0 (uniform), CFR+/Linear -> 1, DCFR -> 2.
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

    /// Expected value to player 0 below `state`. Updates both players' regrets
    /// and average strategy (simultaneous CFR).
    fn traverse<G: Game>(
        &mut self,
        game: &G,
        state: &G::State,
        reach0: f64,
        reach1: f64,
        reach_chance: f64,
    ) -> f64 {
        if game.is_terminal(state) {
            return game.terminal_utility(state);
        }
        if game.is_chance(state) {
            let mut value = 0.0;
            for (child, prob) in game.chance_outcomes(state) {
                value += prob * self.traverse(game, &child, reach0, reach1, reach_chance * prob);
            }
            return value;
        }

        let player = game.current_player(state);
        let n = game.num_actions(state);
        let key = game.infoset_key(state);

        // Snapshot regret to build the strategy, then drop the borrow so the
        // recursion below can re-borrow the node map mutably.
        let regret_snapshot = {
            let node = self
                .nodes
                .entry(key.clone())
                .or_insert_with(|| Node::new(n, player));
            node.regret_sum.clone()
        };
        let strategy = Self::regret_matching(&regret_snapshot);

        let mut action_values = vec![0.0; n];
        let mut value = 0.0;
        for i in 0..n {
            let child = game.next_state(state, i);
            let child_value = if player == 0 {
                self.traverse(game, &child, reach0 * strategy[i], reach1, reach_chance)
            } else {
                self.traverse(game, &child, reach0, reach1 * strategy[i], reach_chance)
            };
            action_values[i] = child_value;
            value += strategy[i] * child_value;
        }

        // Counterfactual reach EXCLUDES the acting player's own contribution;
        // including it is the most common CFR bug. Utilities are player-0-view,
        // so player 1 maximises -u (the sign flip).
        let cf_reach = (if player == 0 { reach1 } else { reach0 }) * reach_chance;
        let sign = if player == 0 { 1.0 } else { -1.0 };
        let own_reach = if player == 0 { reach0 } else { reach1 };
        let weight = self.strategy_weight(self.clock[player]);

        let node = self.nodes.get_mut(&key).unwrap();
        for i in 0..n {
            node.regret_sum[i] += cf_reach * sign * (action_values[i] - value);
            node.strategy_sum[i] += weight * own_reach * strategy[i];
        }
        value
    }

    /// Re-weight accumulated regret after a full traversal. Under simultaneous
    /// updates both players' nodes are touched every iteration.
    fn apply_discounts(&mut self) {
        let v = self.variant;
        for p in 0..2usize {
            if v.floor_regrets_at_zero {
                // The "+" in CFR+: clamp negatives, no alpha/beta scaling.
                for node in self.nodes.values_mut() {
                    if node.player == p {
                        for r in node.regret_sum.iter_mut() {
                            if *r < 0.0 {
                                *r = 0.0;
                            }
                        }
                    }
                }
                continue;
            }
            if v.alpha.is_none() && v.beta.is_none() {
                continue; // vanilla: no discounting
            }
            let t = self.clock[p] as f64;
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
                if node.player == p {
                    for r in node.regret_sum.iter_mut() {
                        *r *= if *r > 0.0 { scale_pos } else { scale_neg };
                    }
                }
            }
        }
    }

    /// Run `iterations` full-traversal CFR passes.
    pub fn run<G: Game>(&mut self, game: &G, iterations: u64) {
        let root = game.initial_state();
        for _ in 0..iterations {
            self.iterations += 1;
            self.clock[0] += 1;
            self.clock[1] += 1;
            self.traverse(game, &root, 1.0, 1.0, 1.0);
            self.apply_discounts();
        }
    }

    /// The converged object: the time-AVERAGE strategy, keyed by infoset. Not
    /// the current strategy — reading that out instead is a classic wrong
    /// shortcut, since the CFR guarantee is on the average iterate.
    pub fn average_strategy(&self) -> HashMap<String, Vec<f64>> {
        let mut out = HashMap::with_capacity(self.nodes.len());
        for (key, node) in &self.nodes {
            let total: f64 = node.strategy_sum.iter().sum();
            let probs = if total > 0.0 {
                node.strategy_sum.iter().map(|x| x / total).collect()
            } else {
                let n = node.strategy_sum.len();
                vec![1.0 / n as f64; n]
            };
            out.insert(key.clone(), probs);
        }
        out
    }
}
