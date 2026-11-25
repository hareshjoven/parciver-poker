//! Kuhn poker — the correctness oracle with a known game value of -1/18.
//! A faithful port of `parciver.games.kuhn`, including its exact legal-action
//! ORDER and `card:history` infoset-key format so the Python oracle can score
//! strategies solved here.

use crate::game::Game;

const NUM_CARDS: u8 = 3;
/// Exact game value to player 0 under optimal play, for the Rust unit test.
#[allow(dead_code)]
pub const GAME_VALUE: f64 = -1.0 / 18.0;

#[derive(Clone)]
pub struct KuhnState {
    /// Dealt private cards, order (player0, player1). Grows via chance nodes.
    dealt: Vec<u8>,
    /// Betting history in the shared encoding: 'c' check/call, 'b' bet, 'f' fold.
    history: String,
}

pub struct Kuhn;

impl Kuhn {
    pub fn new() -> Self {
        Kuhn
    }
}

impl Game for Kuhn {
    type State = KuhnState;

    fn initial_state(&self) -> KuhnState {
        KuhnState {
            dealt: Vec::new(),
            history: String::new(),
        }
    }

    fn is_chance(&self, s: &KuhnState) -> bool {
        s.dealt.len() < 2
    }

    fn chance_outcomes(&self, s: &KuhnState) -> Vec<(KuhnState, f64)> {
        // Remaining ranks in ascending order, matching Python's
        // `for c in range(num_cards) if c not in dealt`.
        let remaining: Vec<u8> = (0..NUM_CARDS).filter(|c| !s.dealt.contains(c)).collect();
        let p = 1.0 / remaining.len() as f64;
        remaining
            .into_iter()
            .map(|c| {
                let mut dealt = s.dealt.clone();
                dealt.push(c);
                (
                    KuhnState {
                        dealt,
                        history: s.history.clone(),
                    },
                    p,
                )
            })
            .collect()
    }

    fn is_terminal(&self, s: &KuhnState) -> bool {
        if s.dealt.len() != 2 {
            return false;
        }
        matches!(s.history.as_str(), "cc" | "cbf" | "cbc" | "bf" | "bc")
    }

    fn terminal_utility(&self, s: &KuhnState) -> f64 {
        match s.history.as_str() {
            "cbf" => -1.0,
            "bf" => 1.0,
            other => {
                // Showdown: higher card takes the pot.
                let pot = match other {
                    "cc" => 1.0,
                    "cbc" | "bc" => 2.0,
                    _ => unreachable!("non-terminal history in terminal_utility"),
                };
                if s.dealt[0] > s.dealt[1] {
                    pot
                } else {
                    -pot
                }
            }
        }
    }

    fn current_player(&self, s: &KuhnState) -> usize {
        s.history.len() % 2
    }

    fn num_actions(&self, _s: &KuhnState) -> usize {
        // Kuhn always offers two actions: {fold, call} facing a bet, else
        // {check, bet}.
        2
    }

    fn next_state(&self, s: &KuhnState, action: usize) -> KuhnState {
        // Legal-action order mirrors Python exactly:
        //   facing a bet -> (FOLD='f', CHECK_CALL='c')
        //   otherwise    -> (CHECK_CALL='c', BET='b')
        let facing_bet = s.history.ends_with('b');
        let ch = if facing_bet {
            if action == 0 {
                'f'
            } else {
                'c'
            }
        } else if action == 0 {
            'c'
        } else {
            'b'
        };
        let mut history = s.history.clone();
        history.push(ch);
        KuhnState {
            dealt: s.dealt.clone(),
            history,
        }
    }

    fn infoset_key(&self, s: &KuhnState) -> String {
        let player = self.current_player(s);
        format!("{}:{}", s.dealt[player], s.history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::{CFRSolver, DCFR};
    use crate::game::count_infosets;

    #[test]
    fn infoset_count_is_12() {
        assert_eq!(count_infosets(&Kuhn::new()), 12);
    }

    #[test]
    fn dcfr_reaches_game_value() {
        // On-policy value under the average strategy, computed by a tiny local
        // tree walk, must land on -1/18.
        let game = Kuhn::new();
        let mut solver = CFRSolver::new(DCFR);
        solver.run(&game, 5000);
        let strat = solver.average_strategy();

        fn walk(
            game: &Kuhn,
            s: &KuhnState,
            strat: &std::collections::HashMap<String, Vec<f64>>,
        ) -> f64 {
            if game.is_terminal(s) {
                return game.terminal_utility(s);
            }
            if game.is_chance(s) {
                return game
                    .chance_outcomes(s)
                    .into_iter()
                    .map(|(c, p)| p * walk(game, &c, strat))
                    .sum();
            }
            let key = game.infoset_key(s);
            let probs = &strat[&key];
            (0..game.num_actions(s))
                .map(|i| probs[i] * walk(game, &game.next_state(s, i), strat))
                .sum()
        }

        let value = walk(&game, &game.initial_state(), &strat);
        assert!((value - GAME_VALUE).abs() < 2e-3, "value {value} != -1/18");
    }
}
