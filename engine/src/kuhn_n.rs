// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
//! N-player Kuhn poker — the multiplayer foundation for the R-MAPSS research
//! track (RMAPSS.md §33, Step 0.2).
//!
//! Standard n-player Kuhn (the OpenSpiel / Szafron-Gibson-Sturtevant 2013
//! formulation): a deck of `n+1` cards ranked `0..n`, each of the `n` players is
//! dealt one private card (one card stays undealt), everyone antes 1, and there
//! is a SINGLE betting round with one bet size of 1 and NO raises.
//!
//! Betting. Seats act in order `0,1,..,n-1`. Facing no bet a seat may check
//! (`'c'`) or bet (`'b'`); once a seat bets, the OTHER `n-1` seats respond in
//! wrap order `(bettor+1)%n .. (bettor+n-1)%n`, each folding (`'f'`) or calling
//! (`'c'`), and the hand ends. If all `n` seats check, it is a showdown among
//! all of them. Highest card among the non-folded players takes the whole pot.
//!
//! Two facts make this a faithful, cheap generalisation of the certified 2p
//! engine and drive the whole implementation:
//!   * the acting seat is always `history.len() % n` — seat = position mod n, and
//!     the bettor is never asked to act again — so the turn rule is the 2p rule
//!     with `2` replaced by `n`; and
//!   * with `n = 2` the terminal set, legal-action order, and `card:history`
//!     infoset keys are BYTE-IDENTICAL to [`crate::kuhn::Kuhn`], so `KuhnN::new(2)`
//!     is a drop-in that the existing Kuhn gates re-certify (see the tests and
//!     `phase_audit.py`).
//!
//! Utilities are general-sum per player (`terminal_utility_for`); the crate-wide
//! player-0 `terminal_utility` is `terminal_utility_for(_, 0)`. Chip conservation
//! makes the profile zero-sum: `Σ_i u_i = 0` at every terminal (a gate).

use crate::game::Game;

#[derive(Clone)]
pub struct KuhnNState {
    /// Dealt private cards, one per seat in seat order. Grows via chance nodes.
    dealt: Vec<u8>,
    /// Betting history: `'c'` check/call, `'b'` bet, `'f'` fold — the shared
    /// encoding of the 2p engine.
    history: String,
}

pub struct KuhnN {
    num_players: usize,
    num_cards: u8,
}

impl KuhnN {
    /// `num_players`-seat Kuhn on a deck of `num_players + 1` cards.
    pub fn new(num_players: usize) -> Self {
        assert!(num_players >= 2, "Kuhn needs at least 2 players");
        KuhnN {
            num_players,
            num_cards: (num_players + 1) as u8,
        }
    }

    /// Index of the (unique) first bet in the history, if any. There are no
    /// raises, so a `'b'` can only be the opening bet, at a pre-bet seat.
    fn bet_pos(history: &str) -> Option<usize> {
        history.as_bytes().iter().position(|&t| t == b'b')
    }
}

impl Game for KuhnN {
    type State = KuhnNState;

    fn initial_state(&self) -> KuhnNState {
        KuhnNState {
            dealt: Vec::new(),
            history: String::new(),
        }
    }

    fn num_players(&self) -> usize {
        self.num_players
    }

    fn is_chance(&self, s: &KuhnNState) -> bool {
        s.dealt.len() < self.num_players
    }

    fn chance_outcomes(&self, s: &KuhnNState) -> Vec<(KuhnNState, f64)> {
        // Remaining ranks in ascending order, matching the 2p engine's
        // `for c in range(num_cards) if c not in dealt`.
        let remaining: Vec<u8> = (0..self.num_cards)
            .filter(|c| !s.dealt.contains(c))
            .collect();
        let p = 1.0 / remaining.len() as f64;
        remaining
            .into_iter()
            .map(|c| {
                let mut dealt = s.dealt.clone();
                dealt.push(c);
                (
                    KuhnNState {
                        dealt,
                        history: s.history.clone(),
                    },
                    p,
                )
            })
            .collect()
    }

    fn is_terminal(&self, s: &KuhnNState) -> bool {
        if s.dealt.len() != self.num_players {
            return false;
        }
        let len = s.history.len();
        match Self::bet_pos(&s.history) {
            // All seats checked.
            None => len == self.num_players,
            // The bettor plus the n-1 responders have all acted.
            Some(p) => len == p + self.num_players,
        }
    }

    fn terminal_utility(&self, s: &KuhnNState) -> f64 {
        self.terminal_utility_for(s, 0)
    }

    fn terminal_utility_for(&self, s: &KuhnNState, player: usize) -> f64 {
        let n = self.num_players;
        let toks = s.history.as_bytes();
        let mut contrib = vec![1.0f64; n]; // antes
        let mut folded = vec![false; n];
        if let Some(p) = Self::bet_pos(&s.history) {
            contrib[p] += 1.0; // the bettor
                               // Response tokens cover every non-bettor seat exactly once, in wrap
                               // order starting after the bettor.
            for (k, &t) in toks[p + 1..].iter().enumerate() {
                let seat = (p + 1 + k) % n;
                if t == b'c' {
                    contrib[seat] += 1.0; // called
                } else {
                    folded[seat] = true; // 'f' folded
                }
            }
        }
        let pot: f64 = contrib.iter().sum();
        // Highest card among the non-folded seats wins the pot. Cards are
        // distinct (dealt from one deck), so there is no tie. The bettor never
        // folds and an all-check line folds nobody, so a winner always exists.
        let mut winner = 0usize;
        let mut best = -1i32;
        for i in 0..n {
            if !folded[i] && (s.dealt[i] as i32) > best {
                best = s.dealt[i] as i32;
                winner = i;
            }
        }
        let mut u = -contrib[player];
        if player == winner {
            u += pot;
        }
        u
    }

    fn current_player(&self, s: &KuhnNState) -> usize {
        s.history.len() % self.num_players
    }

    fn num_actions(&self, _s: &KuhnNState) -> usize {
        2
    }

    fn next_state(&self, s: &KuhnNState, action: usize) -> KuhnNState {
        // Legal-action order mirrors the 2p engine exactly:
        //   facing a bet -> (FOLD='f', CHECK_CALL='c')
        //   otherwise    -> (CHECK_CALL='c', BET='b')
        let facing_bet = Self::bet_pos(&s.history).is_some();
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
        KuhnNState {
            dealt: s.dealt.clone(),
            history,
        }
    }

    fn infoset_key(&self, s: &KuhnNState) -> String {
        let player = self.current_player(s);
        format!("{}:{}", s.dealt[player], s.history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::count_infosets;
    use crate::kuhn::Kuhn;

    /// Collect the set of infoset keys a game exposes, via the public trait only.
    fn collect_keys<G: Game>(game: &G) -> std::collections::BTreeSet<String> {
        fn walk<G: Game>(game: &G, s: &G::State, out: &mut std::collections::BTreeSet<String>) {
            if game.is_terminal(s) {
                return;
            }
            if game.is_chance(s) {
                for (c, _) in game.chance_outcomes(s) {
                    walk(game, &c, out);
                }
                return;
            }
            out.insert(game.infoset_key(s));
            for a in 0..game.num_actions(s) {
                walk(game, &game.next_state(s, a), out);
            }
        }
        let mut out = std::collections::BTreeSet::new();
        walk(game, &game.initial_state(), &mut out);
        out
    }

    #[test]
    fn two_player_reduces_to_certified_kuhn() {
        // Structural: same infoset count AND the exact same set of `card:history`
        // infoset keys as the certified 2p engine — byte-identical infoset
        // structure, which is what lets the existing Kuhn gates re-certify this.
        assert_eq!(count_infosets(&KuhnN::new(2)), 12);
        assert_eq!(collect_keys(&KuhnN::new(2)), collect_keys(&Kuhn::new()));
        // (The -1/18 behavioural game-value check for KuhnN(2) lives in the
        // cfr_np module, where the N-player solver is available.)
    }

    #[test]
    fn three_player_infoset_count_is_48() {
        // 12 public decision histories × 4 possible acting-player cards.
        assert_eq!(count_infosets(&KuhnN::new(3)), 48);
    }

    #[test]
    fn zero_sum_at_every_terminal() {
        fn walk(g: &KuhnN, s: &KuhnNState) {
            if g.is_terminal(s) {
                let sum: f64 = (0..g.num_players())
                    .map(|i| g.terminal_utility_for(s, i))
                    .sum();
                assert!(sum.abs() < 1e-12, "not zero-sum at {}: {sum}", s.history);
                return;
            }
            if g.is_chance(s) {
                for (c, _) in g.chance_outcomes(s) {
                    walk(g, &c);
                }
                return;
            }
            for a in 0..g.num_actions(s) {
                walk(g, &g.next_state(s, a));
            }
        }
        for n in 2..=4 {
            walk(&KuhnN::new(n), &KuhnN::new(n).initial_state());
        }
    }
}
