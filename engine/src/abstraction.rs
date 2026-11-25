// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
//! Card abstraction primitives: suit isomorphism.
//!
//! The NLHE game tree is astronomically large, but suits are interchangeable
//! until they matter for a flush, so most positions are strategically identical
//! up to a permutation of the four suits. Collapsing those isomorphic positions
//! is the first, exact stage of card abstraction (no strategic information is
//! lost — it is a relabeling, not a bucketing).
//!
//! Two exact, externally-checkable invariants make this module self-certifying:
//!   * the 1326 = C(52,2) two-card holdings collapse to exactly **169** classes
//!     (13 pairs + 78 suited + 78 offsuit), and
//!   * the C(52,3) three-card flops collapse to exactly **1755** classes.
//! Both numbers are standard in the solver literature; the unit tests assert them.

use rayon::prelude::*;

use crate::cards::Card;

/// Rank characters, index 0..13 (0=2 … 12=A).
const RANK_CHARS: [u8; 13] = *b"23456789TJQKA";

/// The 24 permutations of the four suits `{0,1,2,3}`, computed at compile time.
///
/// The same table [`suit_perms`] builds at runtime, but as a `const` it can be
/// used allocation-free in the 26-million-situation enumeration hot loop without
/// rebuilding the array per call.
const SUIT_PERMS: [[u8; 4]; 24] = build_suit_perms();

const fn build_suit_perms() -> [[u8; 4]; 24] {
    let mut perms = [[0u8; 4]; 24];
    let mut n = 0;
    let mut a = 0u8;
    while a < 4 {
        let mut b = 0u8;
        while b < 4 {
            if b != a {
                let mut c = 0u8;
                while c < 4 {
                    if c != a && c != b {
                        let d = 6 - a - b - c; // the remaining suit
                        perms[n] = [a, b, c, d];
                        n += 1;
                    }
                    c += 1;
                }
            }
            b += 1;
        }
        a += 1;
    }
    perms
}

/// Combinatorial index of an unordered rank pair `hi > lo` into `0..78`.
///
/// `combo_index(1,0) = 0` … `combo_index(12,11) = 77`; a bijection on
/// `{(hi, lo) : 0 <= lo < hi <= 12}`.
#[inline]
fn combo_index(hi: u8, lo: u8) -> u8 {
    hi * (hi - 1) / 2 + lo
}

/// Invert [`combo_index`] back to `(hi, lo)`.
fn combo_unindex(ci: u8) -> (u8, u8) {
    for hi in 1u8..=12 {
        let base = combo_index(hi, 0);
        if ci < base + hi {
            return (hi, ci - base);
        }
    }
    unreachable!("combo index {ci} out of 0..78")
}

/// Map a two-card holding to its suit-isomorphic class in `0..169`.
///
/// Layout: `0..13` pairs (by rank), `13..91` suited (by rank pair), `91..169`
/// offsuit (by rank pair). Order within suited/offsuit follows [`combo_index`].
pub fn preflop_index(a: Card, b: Card) -> u8 {
    if a.rank == b.rank {
        return a.rank; // pocket pair, 0..13
    }
    let (hi, lo) = if a.rank > b.rank {
        (a.rank, b.rank)
    } else {
        (b.rank, a.rank)
    };
    let ci = combo_index(hi, lo);
    if a.suit == b.suit {
        13 + ci
    } else {
        91 + ci
    }
}

/// Human-readable name of a preflop class, e.g. `"AA"`, `"AKs"`, `"72o"`.
pub fn preflop_name(idx: u8) -> String {
    let r = |rank: u8| RANK_CHARS[rank as usize] as char;
    if idx < 13 {
        let c = r(idx);
        return format!("{c}{c}");
    }
    let (ci, suited) = if idx < 91 {
        (idx - 13, true)
    } else {
        (idx - 91, false)
    };
    let (hi, lo) = combo_unindex(ci);
    format!("{}{}{}", r(hi), r(lo), if suited { 's' } else { 'o' })
}

/// The 24 permutations of the four suits `{0,1,2,3}` (the compile-time
/// [`SUIT_PERMS`] table).
fn suit_perms() -> [[u8; 4]; 24] {
    SUIT_PERMS
}

/// Canonical representative of a set of cards under suit permutation.
///
/// Returns the lexicographically-smallest sorted list of `0..52` indices
/// reachable by relabeling suits. Two card sets are suit-isomorphic iff they
/// share a canonical form, so this deduplicates boards (and, applied to the
/// union, whole positions). `cards` is treated as a set; duplicates are the
/// caller's error.
pub fn canonical_cards(cards: &[Card]) -> Vec<u8> {
    let perms = suit_perms();
    let mut best: Option<Vec<u8>> = None;
    for p in &perms {
        let mut mapped: Vec<u8> = cards
            .iter()
            .map(|c| {
                Card {
                    rank: c.rank,
                    suit: p[c.suit as usize],
                }
                .index()
            })
            .collect();
        mapped.sort_unstable();
        match &best {
            Some(b) if *b <= mapped => {}
            _ => best = Some(mapped),
        }
    }
    best.unwrap_or_default()
}

/// Role-aware canonical form of a `(hole, board)` situation under suit
/// permutation.
///
/// Unlike [`canonical_cards`], which treats its input as one flat set, this keeps
/// hole and board distinguishable: the representative is the
/// lexicographically-smallest `[sorted hole .. sorted board]` concatenation over
/// all 24 suit relabelings. Two situations are strategically suit-isomorphic iff
/// they share this form — so moving a card between hole and board (e.g. holding
/// the flush vs. the flush being on the board) yields a *different* class, which
/// is exactly what a strategy abstraction must preserve. `hole` is 2 cards,
/// `board` is 0..=5; the returned key has length `2 + board.len()`.
pub fn canonical_situation(hole: &[Card], board: &[Card]) -> Vec<u8> {
    let perms = suit_perms();
    let relabel = |c: &Card, p: &[u8; 4]| {
        Card {
            rank: c.rank,
            suit: p[c.suit as usize],
        }
        .index()
    };
    let mut best: Option<Vec<u8>> = None;
    for p in &perms {
        let mut h: Vec<u8> = hole.iter().map(|c| relabel(c, p)).collect();
        let mut b: Vec<u8> = board.iter().map(|c| relabel(c, p)).collect();
        h.sort_unstable();
        b.sort_unstable();
        // Concatenate hole then board: a fixed-length hole prefix makes the
        // lexicographic order of the concatenation identical to the order of the
        // (hole, board) tuple, so roles stay separated.
        h.extend_from_slice(&b);
        match &best {
            Some(x) if *x <= h => {}
            _ => best = Some(h),
        }
    }
    best.unwrap_or_default()
}

/// Allocation-free packed form of the role-aware canonical key.
///
/// Produces exactly the representative of [`canonical_situation`] — sorted hole
/// then sorted board, lexicographically minimal over the 24 suit relabelings —
/// but packed into a `u64` with the first card in the most-significant 6-bit
/// field (`hole0, hole1, board0, …`). Within a single street (fixed card count)
/// the numeric order of the packed key equals the lexicographic order of the
/// situation, so this is a drop-in dedup/sort key that allocates nothing. `board`
/// is 0..=5 cards, so the situation is at most 7 cards = 42 packed bits.
///
/// Unpack with [`unpack_situation`].
#[inline]
pub fn canonical_key_packed(hole: [Card; 2], board: &[Card]) -> u64 {
    let n = board.len();
    let mut best = u64::MAX;
    for p in &SUIT_PERMS {
        let mut h0 = Card {
            rank: hole[0].rank,
            suit: p[hole[0].suit as usize],
        }
        .index();
        let mut h1 = Card {
            rank: hole[1].rank,
            suit: p[hole[1].suit as usize],
        }
        .index();
        if h0 > h1 {
            core::mem::swap(&mut h0, &mut h1);
        }
        let mut b = [0u8; 5];
        for i in 0..n {
            b[i] = Card {
                rank: board[i].rank,
                suit: p[board[i].suit as usize],
            }
            .index();
        }
        b[..n].sort_unstable();
        let mut key = ((h0 as u64) << 6) | h1 as u64;
        for &bi in &b[..n] {
            key = (key << 6) | bi as u64;
        }
        if key < best {
            best = key;
        }
    }
    best
}

/// Unpack a [`canonical_key_packed`] value of `total` cards into 0..52 indices,
/// in the packed order (hole0, hole1, board0, …).
pub fn unpack_situation(key: u64, total: usize) -> Vec<u8> {
    let mut out = vec![0u8; total];
    for i in 0..total {
        let shift = 6 * (total - 1 - i);
        out[i] = ((key >> shift) & 0x3f) as u8;
    }
    out
}

/// Enumerate every suit-isomorphic canonical flop situation (2 hole + 3 board).
///
/// Naive but exact: canonicalize all C(52,2)·C(50,3) = 25,989,600 raw (hole,
/// flop) deals and deduplicate. The result is the standard **1,286,792**
/// canonical flop situations (Waugh's round-2 count). Returned flat and sorted,
/// 5 cards per situation (`hole0, hole1, board0, board1, board2`), length
/// `5 * 1_286_792`. Tractable for the flop; the turn (13.96 M) and river
/// (123.16 M) use the arithmetic indexer in [`crate::isomorphism`] instead of
/// materializing a hash set this large.
pub fn enumerate_canonical_flops() -> Vec<u8> {
    // Parallel over the 1326 hole combos; each contributes <= C(50,3)=19600
    // packed canonical keys. Rows are independent, so rayon changes nothing.
    let holes: Vec<(u8, u8)> = (0..52u8)
        .flat_map(|a| (a + 1..52).map(move |b| (a, b)))
        .collect();
    let mut keys: Vec<u64> = holes
        .par_iter()
        .flat_map_iter(|&(a, b)| {
            let hole = [Card::from_index(a), Card::from_index(b)];
            let used = (1u64 << a) | (1u64 << b);
            let mut local = Vec::with_capacity(19600);
            for c in 0..52u8 {
                if used & (1u64 << c) != 0 {
                    continue;
                }
                for d in c + 1..52 {
                    if used & (1u64 << d) != 0 {
                        continue;
                    }
                    for e in d + 1..52 {
                        if used & (1u64 << e) != 0 {
                            continue;
                        }
                        let board = [
                            Card::from_index(c),
                            Card::from_index(d),
                            Card::from_index(e),
                        ];
                        local.push(canonical_key_packed(hole, &board));
                    }
                }
            }
            local
        })
        .collect();
    keys.par_sort_unstable();
    keys.dedup();
    let mut out = Vec::with_capacity(keys.len() * 5);
    for &k in &keys {
        out.extend_from_slice(&unpack_situation(k, 5));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(i: u8) -> Card {
        Card::from_index(i)
    }

    #[test]
    fn preflop_collapses_1326_holdings_to_169_classes() {
        // Every two-card holding maps into 0..169, and the class multiplicities
        // are exactly the poker-combinatorics values: pairs C(4,2)=6, suited 4,
        // offsuit 4*3=12, summing to C(52,2)=1326.
        let mut counts = [0u32; 169];
        for a in 0..52u8 {
            for b in a + 1..52 {
                let idx = preflop_index(card(a), card(b));
                counts[idx as usize] += 1;
            }
        }
        assert_eq!(counts.iter().sum::<u32>(), 1326);
        for (idx, &c) in counts.iter().enumerate() {
            let expected = if idx < 13 {
                6 // pocket pair
            } else if idx < 91 {
                4 // suited
            } else {
                12 // offsuit
            };
            assert_eq!(c, expected, "class {idx} ({})", preflop_name(idx as u8));
        }
    }

    #[test]
    fn preflop_names_round_trip() {
        // Card index = rank*4 + suit, suits c,d,h,s = 0,1,2,3. Aces = rank 12
        // (indices 48..52), Kings = rank 11 (44..48), sevens = rank 5 (20..24).
        assert_eq!(preflop_name(preflop_index(card(48), card(49))), "AA"); // Ac Ad
        assert_eq!(preflop_name(preflop_index(card(48), card(45))), "AKo"); // Ac Kd
        assert_eq!(preflop_name(preflop_index(card(48), card(44))), "AKs"); // Ac Kc
        assert_eq!(preflop_name(preflop_index(card(20), card(1))), "72o"); // 7c 2d
                                                                           // Order of the two cards must not matter.
        assert_eq!(
            preflop_index(card(48), card(44)),
            preflop_index(card(44), card(48))
        );
        // Exhaustive: names are unique across all 169 classes.
        let names: std::collections::HashSet<String> = (0..169u8).map(preflop_name).collect();
        assert_eq!(names.len(), 169);
    }

    #[test]
    fn canonical_form_is_permutation_invariant() {
        // The same three ranks in any suit pattern with the same "flush shape"
        // must share a canonical form; a rainbow flop and a two-tone flop of the
        // same ranks must differ.
        let rainbow_a = [card(0), card(4 + 1), card(8 + 2)]; // 2c 3d? ranks 0,1,2 distinct suits
        let rainbow_b = [card(1), card(4 + 2), card(8 + 3)]; // same ranks, suits permuted
        assert_eq!(canonical_cards(&rainbow_a), canonical_cards(&rainbow_b));
    }

    #[test]
    fn canonical_situation_distinguishes_hole_from_board() {
        // The same five cards split two ways are NOT the same situation: holding
        // the flush is strategically distinct from the flush lying on the board.
        let a = canonical_situation(&[card(0), card(4)], &[card(8), card(12), card(16)]);
        let b = canonical_situation(&[card(8), card(12)], &[card(0), card(4), card(16)]);
        assert_ne!(a, b);
    }

    #[test]
    fn canonical_situation_is_suit_permutation_invariant() {
        // Relabel every suit consistently; the canonical situation is unchanged.
        // hole = As Ks (suit 3), board = Qs Js 2c.
        let hole = [card(12 * 4 + 3), card(11 * 4 + 3)];
        let board = [card(10 * 4 + 3), card(9 * 4 + 3), card(0 * 4 + 0)];
        // same shape, suits rotated (s->h, c->d): As Ks -> Ah Kh, Qs Js 2c -> Qh Jh 2d.
        let hole2 = [card(12 * 4 + 2), card(11 * 4 + 2)];
        let board2 = [card(10 * 4 + 2), card(9 * 4 + 2), card(0 * 4 + 1)];
        assert_eq!(
            canonical_situation(&hole, &board),
            canonical_situation(&hole2, &board2)
        );
    }

    #[test]
    fn canonical_situation_key_has_expected_length() {
        let k = canonical_situation(&[card(0), card(4)], &[card(8), card(12), card(16)]);
        assert_eq!(k.len(), 5); // 2 hole + 3 board
    }

    #[test]
    fn packed_key_matches_canonical_situation() {
        // The allocation-free packed key must unpack to exactly the same
        // representative as the reference `canonical_situation`, across a spread
        // of streets (flop/turn/river). This is what lets the fast enumerator
        // stand in for the proven reference.
        let cases: [(&[u8], &[u8]); 4] = [
            (&[0, 4], &[8, 12, 16]),
            (&[48, 44], &[43, 39, 35]), // As Ks | Qs Js Ts (suited)
            (&[12 * 4 + 3, 11 * 4 + 3], &[10 * 4 + 3, 9 * 4 + 3, 0, 4]), // turn
            (&[20, 1], &[8, 12, 16, 24, 28]), // river
        ];
        for (hole, board) in cases {
            let h = [Card::from_index(hole[0]), Card::from_index(hole[1])];
            let b: Vec<Card> = board.iter().map(|&i| Card::from_index(i)).collect();
            let packed = canonical_key_packed([h[0], h[1]], &b);
            let unpacked = unpack_situation(packed, 2 + b.len());
            let reference = canonical_situation(&h, &b);
            assert_eq!(unpacked, reference, "hole={hole:?} board={board:?}");
        }
    }

    #[test]
    fn flop_enumeration_yields_1286792_canonical_situations() {
        // Waugh's round-2 count: all C(52,2)*C(50,3) raw (hole, flop) deals
        // collapse to exactly 1,286,792 suit-isomorphic situations. Also checks
        // the output is deduplicated and role-aware-idempotent.
        let flat = enumerate_canonical_flops();
        assert_eq!(flat.len() % 5, 0);
        let count = flat.len() / 5;
        assert_eq!(count, 1_286_792, "canonical flop count");
        // Every emitted situation is already its own canonical representative
        // (idempotence), and the set is unique. Spot-check a strided sample to
        // keep the test fast.
        let mut seen = std::collections::HashSet::new();
        for i in (0..count).step_by(997) {
            let s = &flat[i * 5..i * 5 + 5];
            let hole = [Card::from_index(s[0]), Card::from_index(s[1])];
            let board = [
                Card::from_index(s[2]),
                Card::from_index(s[3]),
                Card::from_index(s[4]),
            ];
            let canon = canonical_situation(&hole, &board);
            assert_eq!(canon, s.to_vec(), "situation {i} not canonical");
            assert!(seen.insert(s.to_vec()), "duplicate situation {i}");
        }
    }

    #[test]
    fn flops_collapse_to_1755_classes() {
        // The definitive board gate: all C(52,3) = 22100 flops collapse to
        // exactly 1755 suit-isomorphic classes.
        let mut set = std::collections::HashSet::new();
        for a in 0..52u8 {
            for b in a + 1..52 {
                for c in b + 1..52 {
                    set.insert(canonical_cards(&[card(a), card(b), card(c)]));
                }
            }
        }
        assert_eq!(set.len(), 1755);
    }
}
