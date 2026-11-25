// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
//! 7-card poker hand evaluator producing a TOTAL ORDER.
//!
//! `eval5`/`eval7` return a `u32` score where a numerically larger score is the
//! strictly stronger poker hand — category AND every tiebreak (kickers, the two
//! ranks of a full house, all five flush ranks) are packed in, so any two hands
//! can be compared by comparing their scores. That total order is what the old
//! Phase-0 evaluator lacked: it returned only a 0..8 category and could not tell
//! a pair of aces from a pair of deuces.
//!
//! `eval7` is the max over all 21 five-card subsets of a 7-card hand. That is
//! obviously correct (the best hand is the best of its 5-card sub-hands) and
//! allocation-free, which is what lets the exhaustive tests certify it against
//! the known category counts over all 133,784,560 seven-card hands.
//!
//! Score layout (high to low bits): category (4 bits) then up to five 4-bit
//! tiebreak ranks in descending significance.

use crate::cards::Card;

/// Category names indexed by `category(score)` (0=high card … 8=straight flush).
pub const CATEGORY_NAMES: [&str; 9] = [
    "high card",
    "one pair",
    "two pair",
    "three of a kind",
    "straight",
    "flush",
    "full house",
    "four of a kind",
    "straight flush",
];

/// The 0..9 hand category of a score.
#[inline]
pub fn category(score: u32) -> u8 {
    (score >> 20) as u8
}

#[inline]
fn encode(cat: u8, keys: &[u8]) -> u32 {
    let mut s = (cat as u32) << 20;
    for (i, &k) in keys.iter().enumerate() {
        s |= (k as u32) << (16 - 4 * i as u32);
    }
    s
}

/// If the 13-bit rank mask contains a straight, return its high card's rank
/// (the wheel A-2-3-4-5 returns 3, i.e. "five-high", so it ranks below a
/// six-high straight as it must).
#[inline]
fn straight_high(mask: u16) -> Option<u8> {
    debug_assert!(mask < 1 << 13, "a rank mask has 13 bits");
    // Bit r of `runs` is set iff ranks r-4..=r are all present (so only r >= 4
    // can be); the highest such r is the top card of the best straight.
    let m = u32::from(mask);
    let runs = m & (m << 1) & (m << 2) & (m << 3) & (m << 4);
    if runs != 0 {
        return Some((31 - runs.leading_zeros()) as u8);
    }
    // Wheel: A(12),2(0),3(1),4(2),5(3) -> five-high (rank 3).
    const WHEEL: u16 = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);
    (mask & WHEEL == WHEEL).then_some(3)
}

/// The window scan `straight_high` replaced (kept to certify it on every mask).
#[cfg(test)]
fn straight_high_scan(mask: u16) -> Option<u8> {
    // Normal straights: five consecutive set bits, top card from 6-high (top
    // rank 4) up to ace-high (top rank 12).
    for top in (4u8..=12).rev() {
        let window = 0b11111u16 << (top - 4);
        if mask & window == window {
            return Some(top);
        }
    }
    // Wheel: A(12),2(0),3(1),4(2),5(3) -> five-high (rank 3).
    const WHEEL: u16 = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);
    if mask & WHEEL == WHEEL {
        return Some(3);
    }
    None
}

/// Evaluate exactly five cards to a comparable score.
pub fn eval5(cards: &[Card; 5]) -> u32 {
    let mut rank_count = [0u8; 13];
    let mut suit_count = [0u8; 4];
    let mut mask = 0u16;
    for c in cards {
        rank_count[c.rank as usize] += 1;
        suit_count[c.suit as usize] += 1;
        mask |= 1 << c.rank;
    }
    let is_flush = suit_count.iter().any(|&x| x == 5);
    let straight = straight_high(mask);

    // Distinct ranks ordered by (count desc, rank desc). For no-pair hands this
    // is simply the five ranks high-to-low, which is exactly the flush/high-card
    // tiebreak; for made hands the leading entries are the meaningful ranks
    // (quad, trips, pairs) followed by kickers.
    let mut groups = [0u8; 5];
    let mut gi = 0usize;
    for target in [4u8, 3, 2, 1] {
        for r in (0..13usize).rev() {
            if rank_count[r] == target {
                groups[gi] = r as u8;
                gi += 1;
            }
        }
    }
    let c0 = rank_count[groups[0] as usize];
    let c1 = if gi > 1 {
        rank_count[groups[1] as usize]
    } else {
        0
    };

    if let Some(h) = straight {
        if is_flush {
            return encode(8, &[h]);
        }
    }
    if c0 == 4 {
        return encode(7, &[groups[0], groups[1]]);
    }
    if c0 == 3 && c1 == 2 {
        return encode(6, &[groups[0], groups[1]]);
    }
    if is_flush {
        return encode(5, &groups[0..5]);
    }
    if let Some(h) = straight {
        return encode(4, &[h]);
    }
    if c0 == 3 {
        return encode(3, &[groups[0], groups[1], groups[2]]);
    }
    if c0 == 2 && c1 == 2 {
        return encode(2, &[groups[0], groups[1], groups[2]]);
    }
    if c0 == 2 {
        return encode(1, &[groups[0], groups[1], groups[2], groups[3]]);
    }
    encode(0, &groups[0..5])
}

/// Evaluate a 7-card hand: the best of its 21 five-card sub-hands.
///
/// This is the *reference*: obviously correct and the thing the exhaustive
/// category-count gate certifies. [`eval7_fast`] is the hot-path replacement,
/// certified bit-identical to this.
pub fn eval7(cards: &[Card; 7]) -> u32 {
    let mut best = 0u32;
    let mut five = [Card::default(); 5];
    // Choose the two cards to EXCLUDE; the rest form the 5-card hand.
    for i in 0..7 {
        for j in (i + 1)..7 {
            let mut idx = 0;
            for k in 0..7 {
                if k != i && k != j {
                    five[idx] = cards[k];
                    idx += 1;
                }
            }
            let s = eval5(&five);
            if s > best {
                best = s;
            }
        }
    }
    best
}

/// Fast best-of-seven evaluator: the same packed score as [`eval7`], computed
/// from rank/suit histograms in a single pass instead of the 21-subset max.
///
/// For each poker category reachable from the seven cards it encodes the *best*
/// hand of that category (optimal kickers), then returns the maximum. Because
/// the score is a strict total order, the best hand of the winning category is
/// exactly the best 5-of-7 — so this is **bit-identical** to `eval7`, certified
/// over all 133,784,560 hands by `eval7_fast_matches_reference_exhaustive`.
///
/// The equity/feature workload is ~1e12 of these, so the allocation-free
/// single-pass form is what makes Phase 3b tractable.
///
/// Works on the four suit bitmasks alone: the ranks held at least 1/2/3/4
/// times are unions of their intersections, and "the highest ranks of a set,
/// minus some" is a leading-zero count — no rank arrays, no counters, no
/// bounds checks. Categories are tried best-first, which picks the same hand
/// as the max over all of them (a category always outranks any lower one's
/// kickers). The seven cards must be distinct, as in any real deal.
pub fn eval7_fast(cards: &[Card; 7]) -> u32 {
    debug_assert!(
        cards.iter().fold(0u64, |m, c| m | 1u64 << (4 * c.rank + c.suit)).count_ones() == 7,
        "eval7_fast needs seven distinct cards"
    );
    eval7_masks(suit_masks(cards))
}

/// Per suit, the bitmask of ranks held (the input of [`eval7_masks`]). A
/// caller scoring many holdings on one board builds the board's masks once and
/// ORs each holding's two cards in.
#[inline]
pub(crate) fn suit_masks(cards: &[Card]) -> [u16; 4] {
    let mut sm = [0u16; 4];
    for c in cards {
        sm[(c.suit & 3) as usize] |= 1 << c.rank;
    }
    sm
}

/// [`eval7_fast`] on the suit masks of seven distinct cards.
#[inline]
pub(crate) fn eval7_masks(sm: [u16; 4]) -> u32 {
    debug_assert!(sm.iter().map(|m| m.count_ones()).sum::<u32>() == 7, "eval7_masks needs seven distinct cards");
    let [s0, s1, s2, s3] = sm;
    let any = s0 | s1 | s2 | s3;
    let two = (s0 & s1) | (s2 & s3) | ((s0 | s1) & (s2 | s3));
    let three = (s0 & s1 & (s2 | s3)) | (s2 & s3 & (s0 | s1));
    let four = s0 & s1 & s2 & s3;
    // Highest rank of a set (0 for the empty set, as the old zeroed kickers
    // were), and the same with that rank removed from the set.
    let top = |m: u16| (15 - (m | 1).leading_zeros()) as u8;
    let pop = |m: &mut u16| {
        let r = top(*m);
        *m &= !(1 << r);
        r
    };
    let bit = |r: u8| 1u16 << r;
    // At most one suit can hold >= 5 of seven cards.
    let flush = sm.into_iter().find(|m| m.count_ones() >= 5);
    if let Some(h) = flush.and_then(straight_high) {
        return encode(8, &[h]);
    }
    if four != 0 {
        let q = top(four);
        return encode(7, &[q, top(any & !bit(q))]);
    }
    if three != 0 {
        let t = top(three);
        let others = two & !bit(t);
        if others != 0 {
            return encode(6, &[t, top(others)]);
        }
    }
    if let Some(mut f) = flush {
        return encode(5, &[pop(&mut f), pop(&mut f), pop(&mut f), pop(&mut f), pop(&mut f)]);
    }
    if let Some(h) = straight_high(any) {
        return encode(4, &[h]);
    }
    if three != 0 {
        let t = top(three);
        let mut k = any & !bit(t);
        return encode(3, &[t, pop(&mut k), pop(&mut k)]);
    }
    if two.count_ones() >= 2 {
        let mut p = two;
        let (p0, p1) = (pop(&mut p), pop(&mut p));
        return encode(2, &[p0, p1, top(any & !bit(p0) & !bit(p1))]);
    }
    if two != 0 {
        let p = top(two);
        let mut k = any & !bit(p);
        return encode(1, &[p, pop(&mut k), pop(&mut k), pop(&mut k)]);
    }
    if any.count_ones() >= 5 {
        let mut k = any;
        return encode(0, &[pop(&mut k), pop(&mut k), pop(&mut k), pop(&mut k), pop(&mut k)]);
    }
    0
}

/// The previous `eval7_fast` (rank histogram, per-category candidates, max),
/// kept as a second oracle and for the speed comparison.
#[cfg(test)]
fn eval7_fast_histogram(cards: &[Card; 7]) -> u32 {
    let mut rank_count = [0u8; 13];
    let mut suit_mask = [0u16; 4];
    let mut mask = 0u16;
    for c in cards {
        rank_count[c.rank as usize] += 1;
        suit_mask[c.suit as usize] |= 1 << c.rank;
        mask |= 1 << c.rank;
    }

    // Distinct ranks and pair/trip/quad ranks, each high-to-low.
    let mut distinct = [0u8; 13];
    let mut nd = 0usize;
    let mut pairs = [0u8; 13]; // count >= 2 (includes trips/quads)
    let mut npair = 0usize;
    let mut trips = [0u8; 13]; // count >= 3
    let mut ntrip = 0usize;
    let mut quad = 13u8; // sentinel = none
    for r in (0..13usize).rev() {
        let c = rank_count[r];
        if c >= 1 {
            distinct[nd] = r as u8;
            nd += 1;
        }
        if c >= 2 {
            pairs[npair] = r as u8;
            npair += 1;
        }
        if c >= 3 {
            trips[ntrip] = r as u8;
            ntrip += 1;
        }
        if c == 4 && quad == 13 {
            quad = r as u8;
        }
    }

    // Highest `n` distinct ranks excluding up to two forbidden ranks.
    let top_kickers = |out: &mut [u8], n: usize, ban0: u8, ban1: u8| {
        let mut k = 0usize;
        for i in 0..nd {
            let r = distinct[i];
            if r == ban0 || r == ban1 {
                continue;
            }
            out[k] = r;
            k += 1;
            if k == n {
                break;
            }
        }
    };

    let mut best = 0u32;
    let consider = |s: u32, best: &mut u32| {
        if s > *best {
            *best = s;
        }
    };

    // Flush / straight flush: at most one suit can hold >= 5 of seven cards.
    for s in 0..4 {
        if suit_mask[s].count_ones() >= 5 {
            let fmask = suit_mask[s];
            if let Some(h) = straight_high(fmask) {
                consider(encode(8, &[h]), &mut best);
            }
            let mut keys = [0u8; 5];
            let mut ki = 0;
            for r in (0..13usize).rev() {
                if fmask & (1 << r) != 0 {
                    keys[ki] = r as u8;
                    ki += 1;
                    if ki == 5 {
                        break;
                    }
                }
            }
            consider(encode(5, &keys), &mut best);
            break;
        }
    }

    // Four of a kind.
    if quad != 13 {
        let mut kick = [0u8; 1];
        top_kickers(&mut kick, 1, quad, 13);
        consider(encode(7, &[quad, kick[0]]), &mut best);
    }

    // Full house: highest trips + highest other pair-or-better.
    if ntrip >= 1 {
        let t = trips[0];
        for i in 0..npair {
            if pairs[i] != t {
                consider(encode(6, &[t, pairs[i]]), &mut best);
                break;
            }
        }
    }

    // Straight.
    if let Some(h) = straight_high(mask) {
        consider(encode(4, &[h]), &mut best);
    }

    // Three of a kind (dominated by a full house when a pair also exists — the
    // max handles that).
    if ntrip >= 1 {
        let t = trips[0];
        let mut k = [0u8; 2];
        top_kickers(&mut k, 2, t, 13);
        consider(encode(3, &[t, k[0], k[1]]), &mut best);
    }

    // Two pair.
    if npair >= 2 {
        let (p0, p1) = (pairs[0], pairs[1]);
        let mut k = [0u8; 1];
        top_kickers(&mut k, 1, p0, p1);
        consider(encode(2, &[p0, p1, k[0]]), &mut best);
    }

    // One pair.
    if npair >= 1 {
        let p = pairs[0];
        let mut k = [0u8; 3];
        top_kickers(&mut k, 3, p, 13);
        consider(encode(1, &[p, k[0], k[1], k[2]]), &mut best);
    }

    // High card.
    if nd >= 5 {
        consider(encode(0, &distinct[0..5]), &mut best);
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::parse_card;

    fn cards5(specs: [&str; 5]) -> [Card; 5] {
        specs.map(|s| Card::from_index(parse_card(s).unwrap()))
    }
    fn cards7(specs: [&str; 7]) -> [Card; 7] {
        specs.map(|s| Card::from_index(parse_card(s).unwrap()))
    }

    #[test]
    fn category_ordering_is_correct() {
        // One representative of each category, strictly increasing in strength.
        let hands = [
            cards5(["2c", "3d", "5h", "9s", "Kc"]), // high card
            cards5(["2c", "2d", "5h", "9s", "Kc"]), // pair
            cards5(["2c", "2d", "5h", "5s", "Kc"]), // two pair
            cards5(["2c", "2d", "2h", "9s", "Kc"]), // trips
            cards5(["2c", "3d", "4h", "5s", "6c"]), // straight
            cards5(["2c", "5c", "7c", "9c", "Kc"]), // flush
            cards5(["2c", "2d", "2h", "9s", "9c"]), // full house
            cards5(["2c", "2d", "2h", "2s", "Kc"]), // quads
            cards5(["2c", "3c", "4c", "5c", "6c"]), // straight flush
        ];
        for w in hands.windows(2) {
            assert!(eval5(&w[0]) < eval5(&w[1]));
        }
    }

    #[test]
    fn wheel_is_lowest_straight() {
        let wheel = cards5(["Ac", "2d", "3h", "4s", "5c"]);
        let six_high = cards5(["2c", "3d", "4h", "5s", "6c"]);
        assert_eq!(category(eval5(&wheel)), 4);
        assert!(eval5(&wheel) < eval5(&six_high));
    }

    #[test]
    fn kickers_resolve() {
        // Both pair of aces; second hand has the better kicker.
        let a = cards5(["Ac", "Ad", "Kh", "Qs", "2c"]);
        let b = cards5(["As", "Ah", "Kd", "Qc", "3c"]);
        assert_eq!(category(eval5(&a)), 1);
        assert!(eval5(&a) < eval5(&b));
    }

    #[test]
    fn eval7_picks_best_five() {
        // A royal flush hidden among seven cards beats a made full house.
        let royal = cards7(["Ah", "Kh", "Qh", "Jh", "Th", "Ac", "Ad"]);
        let boat = cards7(["Ah", "Ac", "Ad", "Kh", "Kc", "2d", "3s"]);
        assert_eq!(category(eval7(&royal)), 8);
        assert_eq!(category(eval7(&boat)), 6);
        assert!(eval7(&royal) > eval7(&boat));
    }

    #[test]
    fn eval7_fast_matches_reference_on_hand_cases() {
        // The tricky categories, checked bit-for-bit against the 21-subset max.
        let cases = [
            cards7(["Ah", "Kh", "Qh", "Jh", "Th", "Ac", "Ad"]), // royal flush
            cards7(["Ah", "Ac", "Ad", "As", "Kh", "Kc", "2d"]), // quads + pair
            cards7(["Ah", "Ac", "Ad", "Kh", "Kc", "Kd", "2d"]), // two trips -> boat
            cards7(["Ah", "Ac", "Kh", "Kc", "Qh", "Qc", "2d"]), // three pair -> two pair
            cards7(["6h", "5h", "4h", "3h", "2h", "Ah", "Kc"]), // 6-high straight flush + higher flush card
            cards7(["Ah", "2d", "3h", "4s", "5c", "Kh", "Kd"]), // wheel vs pair
            cards7(["9c", "9d", "9h", "5s", "5c", "2d", "3h"]), // trips + pair boat
            cards7(["2c", "5d", "8h", "Ts", "Jc", "Qd", "Kh"]), // pure high card
        ];
        for h in &cases {
            assert_eq!(eval7_fast(h), eval7(h), "mismatch on {h:?}");
        }
    }

    #[test]
    fn eval7_fast_matches_reference_on_random_sample() {
        // A large deterministic random sample: eval7_fast must be bit-identical
        // to the certified 21-subset eval7. (The exhaustive proof is the ignored
        // test below; this runs every build as a cheap guard.)
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            // xorshift64*
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        for _ in 0..3_000_000u32 {
            // Draw 7 distinct cards via partial Fisher-Yates over 0..52.
            let mut deck: [u8; 52] = core::array::from_fn(|i| i as u8);
            for t in 0..7usize {
                let j = t + (next() as usize) % (52 - t);
                deck.swap(t, j);
            }
            let hand: [Card; 7] = core::array::from_fn(|i| Card::from_index(deck[i]));
            assert_eq!(eval7_fast(&hand), eval7(&hand));
        }
    }

    /// Exhaustive bit-equality of `eval7_fast` against the reference `eval7`
    /// over all C(52,7) = 133,784,560 hands. Heavy (drives the slow reference),
    /// so `#[ignore]`d; certify with
    /// `cargo test --release -- --ignored eval7_fast_matches_reference_exhaustive`.
    #[test]
    #[ignore]
    fn eval7_fast_matches_reference_exhaustive() {
        let idx: Vec<Card> = (0..52u8).map(Card::from_index).collect();
        let mut checked = 0u64;
        for a in 0..52 {
            for b in a + 1..52 {
                for c in b + 1..52 {
                    for d in c + 1..52 {
                        for e in d + 1..52 {
                            for f in e + 1..52 {
                                for g in f + 1..52 {
                                    let hand =
                                        [idx[a], idx[b], idx[c], idx[d], idx[e], idx[f], idx[g]];
                                    assert_eq!(eval7_fast(&hand), eval7(&hand));
                                    checked += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 133_784_560);
    }

    /// Exhaustive bit-equality of the suit-mask `eval7_fast` against the
    /// histogram form it replaced, over all 133,784,560 hands (both are fast,
    /// so this one takes seconds in release):
    /// `cargo test --release -- --ignored eval7_fast_matches_the_histogram_form_exhaustive`.
    #[test]
    #[ignore]
    fn eval7_fast_matches_the_histogram_form_exhaustive() {
        let idx: Vec<Card> = (0..52u8).map(Card::from_index).collect();
        let mut checked = 0u64;
        for a in 0..52 {
            for b in a + 1..52 {
                for c in b + 1..52 {
                    for d in c + 1..52 {
                        for e in d + 1..52 {
                            for f in e + 1..52 {
                                for g in f + 1..52 {
                                    let hand = [idx[a], idx[b], idx[c], idx[d], idx[e], idx[f], idx[g]];
                                    assert_eq!(eval7_fast(&hand), eval7_fast_histogram(&hand), "{hand:?}");
                                    checked += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 133_784_560);
    }

    /// Histogram vs suit-mask `eval7_fast`, interleaved, on a fixed stream of
    /// random hands:
    #[test]
    fn straight_high_matches_the_window_scan_on_every_rank_mask() {
        for mask in 0u16..1 << 13 {
            assert_eq!(straight_high(mask), straight_high_scan(mask), "mask {mask:013b}");
        }
    }

    /// `cargo test --release --lib -- --ignored --nocapture print_eval7_speed`.
    #[test]
    #[ignore]
    fn print_eval7_speed() {
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let hands: Vec<[Card; 7]> = (0..2_000_000)
            .map(|_| {
                let mut deck: [u8; 52] = core::array::from_fn(|i| i as u8);
                for t in 0..7usize {
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    let j = t + (state.wrapping_mul(0x2545_F491_4F6C_DD1D) as usize) % (52 - t);
                    deck.swap(t, j);
                }
                core::array::from_fn(|i| Card::from_index(deck[i]))
            })
            .collect();
        let time = |f: fn(&[Card; 7]) -> u32| {
            let clock = std::time::Instant::now();
            let s = hands.iter().fold(0u32, |acc, h| acc ^ f(std::hint::black_box(h)));
            (1e9 * clock.elapsed().as_secs_f64() / hands.len() as f64, s)
        };
        let (mut old, mut new) = (Vec::new(), Vec::new());
        for _ in 0..5 {
            let (o, so) = time(eval7_fast_histogram);
            let (n, sn) = time(eval7_fast);
            assert_eq!(so, sn);
            old.push(o);
            new.push(n);
        }
        old.sort_by(f64::total_cmp);
        new.sort_by(f64::total_cmp);
        println!("eval7_fast: histogram {:.1} ns/hand, suit masks {:.1} ns/hand ({:.2}x)", old[2], new[2], old[2] / new[2]);
    }

    #[test]
    fn five_card_category_counts_are_exact() {
        // The definitive gate: over all C(52,5) = 2,598,960 hands the category
        // histogram must match the known combinatorics exactly.
        let mut counts = [0u64; 9];
        for a in 0..52u8 {
            for b in a + 1..52 {
                for c in b + 1..52 {
                    for d in c + 1..52 {
                        for e in d + 1..52 {
                            let hand = [
                                Card::from_index(a),
                                Card::from_index(b),
                                Card::from_index(c),
                                Card::from_index(d),
                                Card::from_index(e),
                            ];
                            counts[category(eval5(&hand)) as usize] += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(
            counts,
            [
                1_302_540, // high card
                1_098_240, // one pair
                123_552,   // two pair
                54_912,    // three of a kind
                10_200,    // straight
                5_108,     // flush
                3_744,     // full house
                624,       // four of a kind
                40,        // straight flush
            ]
        );
    }

    /// The full 7-card gate over all C(52,7) = 133,784,560 hands. Correct but
    /// heavy (~2.8B eval5 calls), so it is `#[ignore]`d out of routine runs;
    /// certify with `cargo test --release -- --ignored seven_card`.
    #[test]
    #[ignore]
    fn seven_card_category_counts_are_exact() {
        let mut counts = [0u64; 9];
        let idx: Vec<Card> = (0..52u8).map(Card::from_index).collect();
        for a in 0..52 {
            for b in a + 1..52 {
                for c in b + 1..52 {
                    for d in c + 1..52 {
                        for e in d + 1..52 {
                            for f in e + 1..52 {
                                for g in f + 1..52 {
                                    let hand =
                                        [idx[a], idx[b], idx[c], idx[d], idx[e], idx[f], idx[g]];
                                    counts[category(eval7(&hand)) as usize] += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            counts,
            [
                23_294_460, // high card
                58_627_800, // one pair
                31_433_400, // two pair
                6_461_620,  // three of a kind
                6_180_020,  // straight
                4_047_644,  // flush
                3_473_184,  // full house
                224_848,    // four of a kind
                41_584,     // straight flush
            ]
        );
    }
}
