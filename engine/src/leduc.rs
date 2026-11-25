//! Leduc hold'em — the primary CFR correctness gate (288 infosets, perfect
//! recall). A faithful port of `parciver.games.leduc`, matching its betting
//! logic, legal-action ORDER, and `private|board|bet_log` infoset-key format
//! exactly so the Python exploitability oracle can score strategies solved here.

use crate::game::Game;

/// 6-card deck: two suits x three ranks (J=0, Q=1, K=2).
const DECK: [u8; 6] = [0, 0, 1, 1, 2, 2];
const BET_SIZE: [i32; 2] = [2, 4]; // round 0, round 1
const MAX_RAISES: usize = 2;
const ANTE: i32 = 1;

#[derive(Clone)]
pub struct LeducState {
    /// Cards in order (player0, player1, [public]); grows via chance nodes.
    dealt: Vec<u8>,
    round: u8, // 0 pre-flop, 1 post-flop
    /// Action string for the CURRENT round only (drives betting logic).
    round_history: String,
    committed: [i32; 2],
    folded: Option<usize>,
    /// Full cross-round action log ("/"-separated) — what the infoset key uses,
    /// giving the game PERFECT RECALL. Dropping round-0 history here would
    /// silently create an imperfect-recall game and break CFR's guarantee.
    bet_log: String,
}

pub struct Leduc;

impl Leduc {
    pub fn new() -> Self {
        Leduc
    }

    /// True when the current betting round has closed: a call closes it, and a
    /// lone opening check ("c") does not. Equivalent to Python's `_round_over`.
    fn round_over(round_history: &str) -> bool {
        round_history.ends_with('c') && round_history != "c"
    }

    fn num_raises(round_history: &str) -> usize {
        round_history.matches('r').count()
    }

    fn showdown_winner(dealt: &[u8]) -> Option<usize> {
        let (p0, p1, board) = (dealt[0], dealt[1], dealt[2]);
        let p0_pair = p0 == board;
        let p1_pair = p1 == board;
        if p0_pair != p1_pair {
            return Some(if p0_pair { 0 } else { 1 });
        }
        if p0 == p1 {
            return None; // split pot
        }
        Some(if p0 > p1 { 0 } else { 1 })
    }
}

impl Game for Leduc {
    type State = LeducState;

    fn initial_state(&self) -> LeducState {
        LeducState {
            dealt: Vec::new(),
            round: 0,
            round_history: String::new(),
            committed: [ANTE, ANTE],
            folded: None,
            bet_log: String::new(),
        }
    }

    fn is_chance(&self, s: &LeducState) -> bool {
        if s.dealt.len() < 2 {
            return true; // deal both private cards
        }
        s.round == 1 && s.dealt.len() < 3 // deal the public card
    }

    fn chance_outcomes(&self, s: &LeducState) -> Vec<(LeducState, f64)> {
        // Remaining deck as a multiset, collapsed to distinct values (ascending)
        // with weight n/total — exactly Python's weighted outcomes.
        let mut remaining = DECK.to_vec();
        for &c in &s.dealt {
            if let Some(pos) = remaining.iter().position(|&x| x == c) {
                remaining.remove(pos);
            }
        }
        let total = remaining.len() as f64;
        let mut counts: [usize; 3] = [0, 0, 0];
        for &c in &remaining {
            counts[c as usize] += 1;
        }
        (0..3u8)
            .filter(|&v| counts[v as usize] > 0)
            .map(|v| {
                let mut dealt = s.dealt.clone();
                dealt.push(v);
                (
                    LeducState { dealt, ..s.clone() },
                    counts[v as usize] as f64 / total,
                )
            })
            .collect()
    }

    fn is_terminal(&self, s: &LeducState) -> bool {
        if s.folded.is_some() {
            return true;
        }
        s.round == 1 && s.dealt.len() == 3 && Self::round_over(&s.round_history)
    }

    fn terminal_utility(&self, s: &LeducState) -> f64 {
        // Utility to player 0 is chips won (+) or lost (-); the loser forfeits
        // what they committed.
        if let Some(folder) = s.folded {
            let winner = 1 - folder;
            let gain = s.committed[folder] as f64;
            return if winner == 0 { gain } else { -gain };
        }
        match Self::showdown_winner(&s.dealt) {
            None => 0.0, // split pot: everyone gets their commitment back
            Some(winner) => {
                let gain = s.committed[1 - winner] as f64;
                if winner == 0 {
                    gain
                } else {
                    -gain
                }
            }
        }
    }

    fn current_player(&self, s: &LeducState) -> usize {
        s.round_history.len() % 2
    }

    fn num_actions(&self, s: &LeducState) -> usize {
        let facing_bet = s.round_history.ends_with('r');
        if facing_bet {
            if Self::num_raises(&s.round_history) < MAX_RAISES {
                3 // fold, call, raise
            } else {
                2 // fold, call (raise cap reached)
            }
        } else {
            2 // check, bet
        }
    }

    fn next_state(&self, s: &LeducState, action: usize) -> LeducState {
        // Legal-action order mirrors Python exactly:
        //   facing a bet, under the cap -> (FOLD='f', CHECK_CALL='c', BET_RAISE='r')
        //   facing a bet, at the cap    -> (FOLD='f', CHECK_CALL='c')
        //   not facing a bet            -> (CHECK_CALL='c', BET_RAISE='r')
        let facing_bet = s.round_history.ends_with('r');
        let ch = if facing_bet {
            match action {
                0 => 'f',
                1 => 'c',
                _ => 'r',
            }
        } else if action == 0 {
            'c'
        } else {
            'r'
        };

        let player = self.current_player(s);
        let mut committed = s.committed;
        let mut bet_log = s.bet_log.clone();
        bet_log.push(ch);

        if ch == 'f' {
            let mut round_history = s.round_history.clone();
            round_history.push('f');
            return LeducState {
                dealt: s.dealt.clone(),
                round: s.round,
                round_history,
                committed,
                folded: Some(player),
                bet_log,
            };
        }

        if ch == 'r' {
            // Match the opponent, then add one bet increment on top.
            let opp = 1 - player;
            let call_amount = committed[opp] - committed[player];
            committed[player] += call_amount + BET_SIZE[s.round as usize];
        } else {
            // 'c': call to match (a check when already level is a no-op).
            let opp = 1 - player;
            if committed[opp] > committed[player] {
                committed[player] = committed[opp];
            }
        }

        let mut round_history = s.round_history.clone();
        round_history.push(ch);

        // Advance to round 1 when round-0 betting closes; the "/" separator
        // keeps the two rounds' logs distinct for perfect recall.
        if s.round == 0 && Self::round_over(&round_history) {
            bet_log.push('/');
            return LeducState {
                dealt: s.dealt.clone(),
                round: 1,
                round_history: String::new(),
                committed,
                folded: None,
                bet_log,
            };
        }

        LeducState {
            dealt: s.dealt.clone(),
            round: s.round,
            round_history,
            committed,
            folded: None,
            bet_log,
        }
    }

    fn infoset_key(&self, s: &LeducState) -> String {
        let player = self.current_player(s);
        let private = s.dealt[player];
        // Board shows once revealed (round 1), else "?".
        if s.dealt.len() >= 3 {
            format!("{}|{}|{}", private, s.dealt[2], s.bet_log)
        } else {
            format!("{}|?|{}", private, s.bet_log)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::count_infosets;

    #[test]
    fn infoset_count_is_288() {
        assert_eq!(count_infosets(&Leduc::new()), 288);
    }

    #[test]
    fn pair_beats_high_card() {
        // p0 has J(0), p1 has K(2), board pairs p0's J -> p0 wins despite the
        // lower private card.
        assert_eq!(Leduc::showdown_winner(&[0, 2, 0]), Some(0));
    }
}
