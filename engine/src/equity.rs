// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
//! All-in equity: the probability a holding beats a uniform-random opponent
//! holding once the board runs out.
//!
//! Equity is the scalar feature the river card-abstraction buckets on, and the
//! building block of the turn/flop equity-distribution histograms. It is
//! computed on the already-certified [`crate::hand_eval`] evaluator, so this
//! module only has to enumerate correctly.
//!
//! Two paths:
//!   * [`equity_vs_random`] — exact, enumerating every board runout and every
//!     opponent holding. Cheap on the river/turn/flop (≤ ~1M matchups); a full
//!     preflop enumeration is ~2.1e9 matchups per hand, so use the sampler there.
//!   * [`equity_vs_random_mc`] — a seeded, deterministic Monte-Carlo sampler for
//!     any street, including preflop.
//!
//! Both count wins/ties/losses into a [`Tally`]; equity counts a tie as half a
//! win, matching the split-pot convention.

use crate::cards::Card;
// The equity/feature workload is ~1e12 evaluations; use the fast single-pass
// evaluator, which is certified bit-identical to the reference `eval7`. Aliased
// so the enumeration code below is unchanged and its correctness gates still hold.
use crate::hand_eval::eval7_fast as eval7;
use crate::hand_eval::{eval7_masks, suit_masks};

/// Win/tie/loss counts for a hero holding against a distribution of opponents.
#[derive(Clone, Copy, Default, Debug)]
pub struct Tally {
    pub wins: u64,
    pub ties: u64,
    pub losses: u64,
}

impl Tally {
    #[inline]
    pub fn total(&self) -> u64 {
        self.wins + self.ties + self.losses
    }

    /// Equity in `[0, 1]`; a tie counts as half a pot. Empty tally → 0.5.
    pub fn equity(&self) -> f64 {
        let t = self.total();
        if t == 0 {
            return 0.5;
        }
        (self.wins as f64 + 0.5 * self.ties as f64) / t as f64
    }
}

/// The 52-index cards not present in `hole`/`board`, as a `(Vec<Card>, used_mask)`.
fn remaining(hole: &[Card], board: &[Card]) -> (Vec<Card>, u64) {
    let mut used = 0u64;
    for c in hole.iter().chain(board) {
        used |= 1u64 << c.index();
    }
    let rem = (0..52u8)
        .filter(|i| used & (1u64 << i) == 0)
        .map(Card::from_index)
        .collect();
    (rem, used)
}

#[inline]
fn hand7(hole: [Card; 2], board5: &[Card; 5]) -> u32 {
    eval7(&[
        hole[0], hole[1], board5[0], board5[1], board5[2], board5[3], board5[4],
    ])
}

/// Exact all-in equity of `hole` versus a uniform-random opponent holding,
/// enumerating every remaining board card and every opponent hand.
///
/// `board` is 0..=5 known community cards. Cost grows fast as the board
/// shrinks; intended for flop/turn/river (preflop is ~2.1e9 matchups — sample
/// instead).
pub fn equity_vs_random(hole: [Card; 2], board: &[Card]) -> Tally {
    let (rem, _) = remaining(&hole, board);
    let need = 5 - board.len();
    let mut full = [Card::default(); 5];
    for (i, c) in board.iter().enumerate() {
        full[i] = *c;
    }
    let mut tally = Tally::default();
    complete_and_score(hole, &mut full, board.len(), need, &rem, 0, 0, &mut tally);
    tally
}

/// [`equity_vs_random`] of every holding at once on a turn or river board,
/// indexed `a * 52 + b` (both card orders; holdings touching the board stay
/// empty). The same integer counts as the per-holding enumeration, from one
/// strength sweep per river card with per-card counts for the blockers: a turn
/// board costs ~48 x 1081 evaluations instead of ~45k per holding.
pub fn equity_vs_random_all(board: &[Card]) -> Vec<Tally> {
    assert!(matches!(board.len(), 4 | 5), "equity_vs_random_all wants a turn or river board");
    let used = board.iter().fold(0u64, |m, c| m | 1u64 << c.index());
    let mut out = vec![Tally::default(); 52 * 52];
    let mut full = [Card::default(); 5];
    full[..board.len()].copy_from_slice(board);
    let rivers: Vec<Option<u8>> = if board.len() == 5 {
        vec![None]
    } else {
        (0..52u8).filter(|&c| used & (1u64 << c) == 0).map(Some).collect()
    };
    // Every holding live on the full board as strength << 12 | low card << 6 |
    // high card, so one u64 sort orders them by strength (the order inside a
    // group of equal strength does not matter to the counts).
    let mut live: Vec<u64> = Vec::with_capacity(1128);
    for river in rivers {
        let mut dead = used;
        if let Some(r) = river {
            full[4] = Card::from_index(r);
            dead |= 1u64 << r;
        }
        // The board's suit masks once; each holding ORs its two cards in.
        let board_masks = suit_masks(&full);
        live.clear();
        for a in 0..52u8 {
            if dead & (1u64 << a) != 0 {
                continue;
            }
            let mut with_a = board_masks;
            with_a[usize::from(a & 3)] |= 1 << (a >> 2);
            for b in a + 1..52 {
                if dead & (1u64 << b) == 0 {
                    let mut sm = with_a;
                    sm[usize::from(b & 3)] |= 1 << (b >> 2);
                    live.push(u64::from(eval7_masks(sm)) << 12 | u64::from(a) << 6 | u64::from(b));
                }
            }
        }
        live.sort_unstable();
        // Opponent holdings disjoint from a live holding.
        let left = 52 - u64::from(dead.count_ones()) - 2;
        let opponents = left * (left - 1) / 2;
        // Per-card counts in 64 slots indexed by a 6-bit card: no bounds check.
        let (mut below, mut below_card) = (0u64, [0u64; 64]);
        let mut same_card = [0u64; 64];
        let cards = |x: u64| ((x >> 6 & 63) as usize, (x & 63) as usize);
        let mut i = 0;
        while i < live.len() {
            let s = live[i] >> 12;
            let mut j = i;
            while j < live.len() && live[j] >> 12 == s {
                j += 1;
            }
            let group = &live[i..j];
            for &x in group {
                let (a, b) = cards(x);
                same_card[a] += 1;
                same_card[b] += 1;
            }
            let same = group.len() as u64;
            for &x in group {
                let (a, b) = cards(x);
                // Holdings sharing a card with this one never oppose it; the
                // holding itself is counted in both of its cards' ties.
                let wins = below - below_card[a] - below_card[b];
                let ties = same - same_card[a] - same_card[b] + 1;
                let t = &mut out[a * 52 + b];
                t.wins += wins;
                t.ties += ties;
                t.losses += opponents - wins - ties;
            }
            for &x in group {
                let (a, b) = cards(x);
                same_card[a] -= 1;
                same_card[b] -= 1;
                below_card[a] += 1;
                below_card[b] += 1;
            }
            below += same;
            i = j;
        }
    }
    for a in 0..52 {
        for b in a + 1..52 {
            out[b * 52 + a] = out[a * 52 + b];
        }
    }
    out
}

/// Recursively choose `need` board-completion cards from `rem[start..]`, filling
/// `full[fill..]`; `used_rem` masks which `rem` indices are already on the board
/// so the opponent draws from the rest. At a full board, score the hero against
/// every remaining opponent two-card holding.
fn complete_and_score(
    hole: [Card; 2],
    full: &mut [Card; 5],
    fill: usize,
    need: usize,
    rem: &[Card],
    start: usize,
    used_rem: u64,
    tally: &mut Tally,
) {
    if need == 0 {
        // The board's suit masks once; hero and every opponent OR two cards in.
        let board = suit_masks(full);
        let with = |mut sm: [u16; 4], c: Card| {
            sm[usize::from(c.suit & 3)] |= 1 << c.rank;
            sm
        };
        let hero = eval7_masks(with(with(board, hole[0]), hole[1]));
        // Opponent draws from rem indices not used by the board completion.
        for (i, &a) in rem.iter().enumerate() {
            if used_rem & (1u64 << i) != 0 {
                continue;
            }
            let with_a = with(board, a);
            for (j, &b) in rem.iter().enumerate().skip(i + 1) {
                if used_rem & (1u64 << j) != 0 {
                    continue;
                }
                let opp = eval7_masks(with(with_a, b));
                tally.wins += u64::from(hero > opp);
                tally.losses += u64::from(hero < opp);
                tally.ties += u64::from(hero == opp);
            }
        }
        return;
    }
    // Leave room for the remaining `need` picks.
    for i in start..=rem.len() - need {
        full[fill] = rem[i];
        complete_and_score(
            hole,
            full,
            fill + 1,
            need - 1,
            rem,
            i + 1,
            used_rem | (1u64 << i),
            tally,
        );
    }
}

/// Exact equity of `hole` against one specific `villain` holding over all board
/// runouts. Used both directly and as an exactness cross-check: hero-vs-villain
/// and villain-vs-hero equities over the same runouts sum to exactly 1.
pub fn equity_vs_hand(hole: [Card; 2], villain: [Card; 2], board: &[Card]) -> Tally {
    let mut used = 0u64;
    for c in hole.iter().chain(villain.iter()).chain(board) {
        used |= 1u64 << c.index();
    }
    let rem: Vec<Card> = (0..52u8)
        .filter(|i| used & (1u64 << i) == 0)
        .map(Card::from_index)
        .collect();
    let need = 5 - board.len();
    let mut full = [Card::default(); 5];
    for (i, c) in board.iter().enumerate() {
        full[i] = *c;
    }
    let mut tally = Tally::default();
    vs_hand_recurse(
        hole,
        villain,
        &mut full,
        board.len(),
        need,
        &rem,
        0,
        &mut tally,
    );
    tally
}

fn vs_hand_recurse(
    hole: [Card; 2],
    villain: [Card; 2],
    full: &mut [Card; 5],
    fill: usize,
    need: usize,
    rem: &[Card],
    start: usize,
    tally: &mut Tally,
) {
    if need == 0 {
        let hero = hand7(hole, full);
        let opp = hand7(villain, full);
        if hero > opp {
            tally.wins += 1;
        } else if hero < opp {
            tally.losses += 1;
        } else {
            tally.ties += 1;
        }
        return;
    }
    for i in start..=rem.len() - need {
        full[fill] = rem[i];
        vs_hand_recurse(hole, villain, full, fill + 1, need - 1, rem, i + 1, tally);
    }
}

/// Deterministic SplitMix64 PRNG — reproducible sampling with no external
/// dependency. Statistical quality is ample for equity Monte-Carlo.
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

    /// Uniform in `0..n` via Lemire's debiased multiply-shift.
    #[inline]
    fn bounded(&mut self, n: usize) -> usize {
        let n = n as u64;
        let mut x = self.next_u64();
        let mut m = (x as u128) * (n as u128);
        let mut l = m as u64;
        if l < n {
            let t = n.wrapping_neg() % n;
            while l < t {
                x = self.next_u64();
                m = (x as u128) * (n as u128);
                l = m as u64;
            }
        }
        (m >> 64) as usize
    }
}

/// Monte-Carlo all-in equity of `hole` versus a uniform-random opponent, for any
/// street including preflop. `seed` makes it fully reproducible. Each sample
/// deals the board completion and the opponent's two cards from the live deck
/// via a partial Fisher–Yates shuffle (sampling without replacement).
pub fn equity_vs_random_mc(hole: [Card; 2], board: &[Card], samples: u64, seed: u64) -> Tally {
    let (rem, _) = remaining(&hole, board);
    let need = 5 - board.len();
    let draw = need + 2; // board completion + opponent's two cards
    let mut deck = rem;
    let mut rng = Rng::new(seed);
    let mut full = [Card::default(); 5];
    for (i, c) in board.iter().enumerate() {
        full[i] = *c;
    }
    let mut tally = Tally::default();
    let len = deck.len();
    for _ in 0..samples {
        // Partial Fisher–Yates: first `draw` slots become a uniform sample
        // without replacement; the deck stays a valid permutation for reuse.
        for t in 0..draw {
            let j = t + rng.bounded(len - t);
            deck.swap(t, j);
        }
        for k in 0..need {
            full[board.len() + k] = deck[k];
        }
        let hero = hand7(hole, &full);
        let opp = eval7(&[
            deck[need],
            deck[need + 1],
            full[0],
            full[1],
            full[2],
            full[3],
            full[4],
        ]);
        if hero > opp {
            tally.wins += 1;
        } else if hero < opp {
            tally.losses += 1;
        } else {
            tally.ties += 1;
        }
    }
    tally
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::parse_card;

    fn c(s: &str) -> Card {
        Card::from_index(parse_card(s).unwrap())
    }

    #[test]
    fn river_vs_hand_complementarity_is_exact() {
        // Over identical runouts (here a full river: a single showdown) the two
        // players' equities sum to exactly 1.
        let hero = [c("Ac"), c("Ad")];
        let villain = [c("Kc"), c("Kd")];
        let board = [c("2h"), c("7s"), c("Ts"), c("Jd"), c("3c")];
        let a = equity_vs_hand(hero, villain, &board).equity();
        let b = equity_vs_hand(villain, hero, &board).equity();
        assert!((a + b - 1.0).abs() < 1e-12);
        assert_eq!(a, 1.0); // aces full? no — just a pair of aces beats kings here
    }

    #[test]
    fn turn_vs_hand_complementarity_is_exact() {
        // A turn board leaves 44 river runouts; complementarity must still hold
        // exactly across the distribution.
        let hero = [c("Ah"), c("Kh")];
        let villain = [c("Qs"), c("Qd")];
        let board = [c("2h"), c("7h"), c("Ts"), c("Jd")];
        let a = equity_vs_hand(hero, villain, &board).equity();
        let b = equity_vs_hand(villain, hero, &board).equity();
        assert!((a + b - 1.0).abs() < 1e-12);
    }

    #[test]
    fn board_plays_makes_every_hand_a_tie() {
        // A royal flush on the board: no two hole cards can change the result,
        // so every hero has exactly 0.5 equity (all ties).
        let board = [c("Ah"), c("Kh"), c("Qh"), c("Jh"), c("Th")];
        let t = equity_vs_random([c("2c"), c("3d")], &board);
        assert_eq!(t.wins, 0);
        assert_eq!(t.losses, 0);
        assert!(t.ties > 0);
        assert_eq!(t.equity(), 0.5);
    }

    #[test]
    fn nut_low_on_river_loses_to_almost_everything() {
        // 7-2 offsuit on a board that pairs nothing useful: worst realistic
        // holding, equity well under 0.2 against a random hand.
        let board = [c("Ah"), c("Kd"), c("Qs"), c("Jc"), c("9h")];
        let e = equity_vs_random([c("2c"), c("7d")], &board).equity();
        assert!(e < 0.25, "expected weak equity, got {e}");
    }

    /// `equity_vs_random` against the most literal enumeration there is: every
    /// runout, every opponent holding, both scored by the 21-subset reference
    /// `eval7` — shares no code with the suit-mask paths above.
    #[test]
    fn enumeration_matches_a_naive_count_on_the_reference_evaluator() {
        let mut state = 0x5eed_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (board_len, deals) in [(5, 12), (4, 6), (3, 1)] {
            for _ in 0..deals {
                let mut deck: Vec<u8> = (0..52).collect();
                for i in 0..2 + board_len {
                    let j = i + (next() % (52 - i as u64)) as usize;
                    deck.swap(i, j);
                }
                let hole = [Card::from_index(deck[0]), Card::from_index(deck[1])];
                let board: Vec<Card> = deck[2..2 + board_len].iter().map(|&i| Card::from_index(i)).collect();
                let rest = &deck[2 + board_len..];
                let runouts: Vec<Vec<u8>> = match board_len {
                    5 => vec![vec![]],
                    4 => rest.iter().map(|&r| vec![r]).collect(),
                    _ => (0..rest.len()).flat_map(|i| (i + 1..rest.len()).map(move |j| vec![rest[i], rest[j]])).collect(),
                };
                let (mut wins, mut ties, mut losses) = (0u64, 0u64, 0u64);
                for ro in &runouts {
                    let full: Vec<Card> = board.iter().copied().chain(ro.iter().map(|&i| Card::from_index(i))).collect();
                    let seven = |a: Card, b: Card| crate::hand_eval::eval7(&[a, b, full[0], full[1], full[2], full[3], full[4]]);
                    let hero = seven(hole[0], hole[1]);
                    let opp: Vec<Card> = rest.iter().filter(|x| !ro.contains(x)).map(|&i| Card::from_index(i)).collect();
                    for i in 0..opp.len() {
                        for j in i + 1..opp.len() {
                            match hero.cmp(&seven(opp[i], opp[j])) {
                                std::cmp::Ordering::Greater => wins += 1,
                                std::cmp::Ordering::Equal => ties += 1,
                                std::cmp::Ordering::Less => losses += 1,
                            }
                        }
                    }
                }
                let t = equity_vs_random(hole, &board);
                assert_eq!((t.wins, t.ties, t.losses), (wins, ties, losses), "{hole:?} on {board:?}");
            }
        }
    }

    #[test]
    fn batched_equity_matches_the_enumeration_exactly() {
        let mut z = 0x5EED_u64;
        let mut next = move || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        // Random boards plus boards full of ties (a royal board, a paired one).
        let mut boards: Vec<Vec<Card>> = vec![
            ["Ah", "Kh", "Qh", "Jh", "Th"].map(c).to_vec(),
            ["7c", "7d", "7h", "2s"].map(c).to_vec(),
        ];
        for len in [5, 5, 5, 4, 4] {
            let mut deck: Vec<u8> = (0..52).collect();
            for i in 0..len {
                let j = i + (next() % (52 - i as u64)) as usize;
                deck.swap(i, j);
            }
            boards.push(deck[..len].iter().map(|&i| Card::from_index(i)).collect());
        }
        for board in &boards {
            let all = equity_vs_random_all(board);
            let used = board.iter().fold(0u64, |m, x| m | 1u64 << x.index());
            let mut checked = 0;
            for a in 0..52u8 {
                for b in a + 1..52 {
                    let t = all[a as usize * 52 + b as usize];
                    let u = all[b as usize * 52 + a as usize];
                    assert_eq!((t.wins, t.ties, t.losses), (u.wins, u.ties, u.losses));
                    if used & (1u64 << a | 1u64 << b) != 0 {
                        assert_eq!(t.total(), 0);
                        continue;
                    }
                    // Every river holding; a sample on the turn (45k evaluations each).
                    if board.len() == 4 && next() % 16 != 0 {
                        continue;
                    }
                    let e = equity_vs_random([Card::from_index(a), Card::from_index(b)], board);
                    assert_eq!((t.wins, t.ties, t.losses), (e.wins, e.ties, e.losses), "{a} {b} on {board:?}");
                    assert_eq!(t.equity().to_bits(), e.equity().to_bits());
                    checked += 1;
                }
            }
            assert!(checked > 40, "only {checked} holdings checked");
        }
    }

    /// One holding's exact equity against the whole board's batch:
    ///   cargo test --release --lib -- --ignored --nocapture print_batched_equity_speed
    #[test]
    #[ignore]
    fn print_batched_equity_speed() {
        let hole = [c("Ac"), c("Ks")];
        for board in [["2h", "7s", "Ts", "Jd"].map(c).to_vec(), ["2h", "7s", "Ts", "Jd", "3c"].map(c).to_vec()] {
            let reps = if board.len() == 4 { 20 } else { 500 };
            let clock = std::time::Instant::now();
            let mut x = 0;
            for _ in 0..reps {
                x += equity_vs_random(hole, &board).wins;
            }
            let one = clock.elapsed().as_secs_f64() / reps as f64;
            let clock = std::time::Instant::now();
            for _ in 0..reps {
                x += equity_vs_random_all(&board)[0].wins;
            }
            let all = clock.elapsed().as_secs_f64() / reps as f64;
            println!("{} cards: one holding {:.3} ms, every holding {:.3} ms ({:.1} holdings' worth) [{x}]", board.len(), 1e3 * one, 1e3 * all, all / one);
        }
    }

    #[test]
    fn mc_matches_exact_on_the_flop() {
        // The sampler must converge to the exact enumeration.
        let hole = [c("Ac"), c("Ks")];
        let board = [c("Ad"), c("7h"), c("2c")];
        let exact = equity_vs_random(hole, &board).equity();
        let mc = equity_vs_random_mc(hole, &board, 400_000, 0xC0FFEE).equity();
        assert!((exact - mc).abs() < 5e-3, "exact {exact} vs mc {mc}");
    }

    #[test]
    fn mc_is_deterministic_for_a_fixed_seed() {
        let hole = [c("Ac"), c("Ad")];
        let a = equity_vs_random_mc(hole, &[], 10_000, 42);
        let b = equity_vs_random_mc(hole, &[], 10_000, 42);
        assert_eq!(a.wins, b.wins);
        assert_eq!(a.ties, b.ties);
        assert_eq!(a.losses, b.losses);
    }

    #[test]
    fn pocket_aces_preflop_beat_a_random_hand_about_85_percent() {
        // The canonical textbook number: AA is ~85% to win heads-up preflop.
        let e = equity_vs_random_mc([c("Ac"), c("Ad")], &[], 300_000, 12345).equity();
        assert!((e - 0.852).abs() < 0.01, "AA vs random equity was {e}");
    }
}
