// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
//! Waugh hand-isomorphism indexer (Phase 3b).
//!
//! A dense, bijective index over the suit-isomorphic `(hole, board)` classes of
//! a street. The flop's 1,286,792 classes can be found by naive dedup
//! ([`crate::abstraction::enumerate_canonical_flops`]), but the turn's
//! 13,960,050 and the river's 123,156,254 cannot — enumerating and hashing that
//! many representatives is intractable. Kevin Waugh's algorithm ("A Fast and
//! Optimal Hand Isomorphism Algorithm", 2013) instead computes the index
//! *arithmetically*, in closed form, so the bucket table for a street is simply
//! an array indexed by `index(hand)` with no hash set ever materialized.
//!
//! # The construction
//!
//! Suits are interchangeable until a flush matters, and the *round* a card
//! arrives in is strategically meaningful (a flush card on the turn is not the
//! same as one on the flop), so the isomorphism is: two hands are equivalent iff
//! one is a suit permutation of the other that preserves which round each card
//! was dealt in. This is exactly the role-aware notion
//! [`crate::abstraction::canonical_situation`] enforces (round 0 = hole is a
//! distinct round from the board rounds), which is what lets the flop gate below
//! cross-check this indexer against that independent method.
//!
//! For a hand, each suit holds a set of ranks per round. The index is built from:
//!   * a **rank-set colex index** per (suit, round) — the position of a `k`-subset
//!     of the available ranks in colexicographic order;
//!   * mixed-radix combination of a suit's per-round colex indices into a single
//!     **column index** (ranks are dealt without replacement within a suit, so
//!     each round draws from the ranks that suit has not already used);
//!   * grouping suits with an identical per-round *configuration* (they are
//!     mutually interchangeable) and indexing each group's column indices as a
//!     **multiset** (combination-with-repetition), then combining groups and the
//!     per-configuration offset in mixed radix.
//!
//! Only the final round of each street's indexer is used, so the multi-round
//! offset bookkeeping of the original paper is unneeded here: a separate indexer
//! is built per street (`[2,3]`, `[2,3,1]`, `[2,3,1,1]`) and only the full hand
//! is ever indexed/unindexed.
//!
//! Every headline number is externally checkable: `size()` must equal the
//! literature counts, `index(unindex(i)) == i` over the whole flop and large
//! samples of turn/river, and — the strongest check — the *set* of classes the
//! flop indexer covers equals [`crate::abstraction::enumerate_canonical_flops`]
//! exactly.

use rustc_hash::FxHashMap;

const RANKS: usize = 13;
const SUITS: usize = 4;
/// Most rounds any indexer uses (the hot path uses 2: hole, board). Bounds the
/// stack arrays in [`HandIndexer::index`].
const MAX_ROUNDS: usize = 4;

/// Exact binomial coefficient C(n, k). `k` is tiny here (<= 7) and `n` is at most
/// a few thousand, so the running product (with u128 intermediates to avoid
/// overflow) is exact and cheap.
#[inline]
fn binom(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut num: u128 = 1;
    for i in 0..k {
        num = num * (n - i) as u128 / (i + 1) as u128;
    }
    num as u64
}

/// Colexicographic index of a rank set (a `<=13`-bit mask) among all subsets of
/// its own cardinality: `sum_j C(bit_j, j+1)` over the set bits in increasing
/// order (`j` 0-based). Ranges over `0..C(RANKS, popcount)`.
fn colex_index(mask: u32) -> u32 {
    let mut idx = 0u64;
    let mut j = 1u64;
    let mut m = mask;
    while m != 0 {
        let bit = m.trailing_zeros() as u64;
        idx += binom(bit, j);
        m &= m - 1;
        j += 1;
    }
    idx as u32
}

/// Remove the `used` rank bits from `ranks`, compacting the remaining bits down
/// into the space of available ranks (inverse: [`expand`]).
#[inline]
fn compact(ranks: u32, used: u32) -> u32 {
    let mut out = 0u32;
    let mut m = ranks;
    while m != 0 {
        let b = m.trailing_zeros();
        let below = used & ((1u32 << b) - 1);
        out |= 1u32 << (b - below.count_ones());
        m &= m - 1;
    }
    out
}

/// Inverse of [`compact`]: map each bit of `compacted` (a position among the
/// ranks not in `used`) back to its actual rank.
#[inline]
fn expand(compacted: u32, used: u32) -> u32 {
    let mut out = 0u32;
    let mut m = compacted;
    while m != 0 {
        let p = m.trailing_zeros();
        // The p-th (0-based) rank not present in `used`.
        let mut avail = 0u32;
        let mut rank = 0u32;
        loop {
            if used & (1u32 << rank) == 0 {
                if avail == p {
                    break;
                }
                avail += 1;
            }
            rank += 1;
        }
        out |= 1u32 << rank;
        m &= m - 1;
    }
    out
}

/// One suit-configuration group: a run of suits that share the same per-round
/// count vector (`spec`) and are therefore mutually interchangeable.
#[derive(Clone, Debug)]
struct Group {
    spec: Vec<u8>, // per-round card counts for each suit in this group
    g: usize,      // number of suits in the group
    mult: u64,     // multiset count C(m + g - 1, g) over the g suits' column values
}

/// One suit configuration: the multiset of the four suits' per-round count
/// vectors, split into equal-spec groups, with its dense offset.
#[derive(Clone, Debug)]
struct Config {
    groups: Vec<Group>, // equal-spec groups, in descending order
    offset: u64,        // cumulative offset (sum of earlier configs' sizes)
}

/// Dense bijective indexer over the suit-isomorphic classes of one street.
pub struct HandIndexer {
    rounds: usize,
    cards_per_round: Vec<u8>,
    total_cards: usize,
    index_to_set: Vec<Vec<u32>>, // [size] -> colex index -> rank-set mask
    configs: Vec<Config>,
    offsets: Vec<u64>, // sorted config offsets, for unindex binary search
    /// Four suits' per-round counts packed 4 bits each ([`pack_specs`]) -> config.
    spec_key_to_config: FxHashMap<u64, usize>,
    total_size: u64,
}

impl HandIndexer {
    /// Build the indexer for a street given its cards-per-round profile, e.g.
    /// `[2, 3]` flop, `[2, 3, 1]` turn, `[2, 3, 1, 1]` river.
    pub fn new(cards_per_round: &[u8]) -> HandIndexer {
        let rounds = cards_per_round.len();
        assert!(rounds <= MAX_ROUNDS, "at most {MAX_ROUNDS} rounds");
        assert!(
            cards_per_round.iter().all(|&c| c < 16),
            "per-round counts must fit the 4-bit spec packing"
        );
        let total_cards: usize = cards_per_round.iter().map(|&c| c as usize).sum();

        // Colex inverse tables: index_to_set[size][colex] = rank-set mask.
        let mut index_to_set = vec![Vec::new(); RANKS + 1];
        for size in 0..=RANKS {
            index_to_set[size] = vec![0u32; binom(RANKS as u64, size as u64) as usize];
        }
        for mask in 0u32..(1u32 << RANKS) {
            let size = mask.count_ones() as usize;
            index_to_set[size][colex_index(mask) as usize] = mask;
        }

        // Enumerate suit configurations (multisets of per-suit specs) canonically
        // — suit specs non-increasing so each configuration appears once.
        let mut spec_tuples: Vec<Vec<Vec<u8>>> = Vec::new();
        let remaining: Vec<u8> = cards_per_round.to_vec();
        let mut current: Vec<Vec<u8>> = Vec::new();
        enumerate_configs(0, &remaining, None, &mut current, &mut spec_tuples);

        let mut configs = Vec::with_capacity(spec_tuples.len());
        let mut offsets = Vec::with_capacity(spec_tuples.len());
        let mut spec_key_to_config = FxHashMap::default();
        let mut offset = 0u64;
        for specs4 in spec_tuples {
            // Group runs of equal specs.
            let mut groups: Vec<Group> = Vec::new();
            let mut i = 0;
            while i < specs4.len() {
                let spec = specs4[i].clone();
                let mut g = 1;
                while i + g < specs4.len() && specs4[i + g] == spec {
                    g += 1;
                }
                let m = column_count(&spec);
                let mult = binom(m + g as u64 - 1, g as u64);
                groups.push(Group { spec, g, mult });
                i += g;
            }
            let size: u64 = groups.iter().map(|gr| gr.mult).product();
            let key = pack_specs(specs4.iter().map(|s| s.as_slice()));
            spec_key_to_config.insert(key, configs.len());
            offsets.push(offset);
            configs.push(Config { groups, offset });
            offset += size;
        }
        let total_size = offset;

        HandIndexer {
            rounds,
            cards_per_round: cards_per_round.to_vec(),
            total_cards,
            index_to_set,
            configs,
            offsets,
            spec_key_to_config,
            total_size,
        }
    }

    /// Total number of suit-isomorphic classes for this street.
    #[inline]
    pub fn size(&self) -> u64 {
        self.total_size
    }

    /// Number of cards in a full hand for this street (hole + board).
    #[inline]
    pub fn total_cards(&self) -> usize {
        self.total_cards
    }

    /// Per-round rank sets of each suit for a concrete hand. `cards` is the full
    /// hand as 0..52 indices, ordered by round (hole first). Returns
    /// `rank_sets[suit][round]` as bitmasks.
    #[cfg(test)]
    fn suit_round_sets(&self, cards: &[u8]) -> [Vec<u32>; SUITS] {
        let mut sets: [Vec<u32>; SUITS] = [
            vec![0; self.rounds],
            vec![0; self.rounds],
            vec![0; self.rounds],
            vec![0; self.rounds],
        ];
        let mut pos = 0;
        for (r, &n) in self.cards_per_round.iter().enumerate() {
            for _ in 0..n {
                let c = cards[pos];
                let rank = (c >> 2) as usize;
                let suit = (c & 3) as usize;
                sets[suit][r] |= 1u32 << rank;
                pos += 1;
            }
        }
        sets
    }

    /// Column index of one suit: mixed-radix combination of its per-round colex
    /// indices, drawing each round from the ranks the suit has not yet used.
    fn column_index(&self, rank_sets: &[u32], spec: &[u8]) -> u64 {
        let mut idx = 0u64;
        let mut used = 0u32;
        for r in 0..self.rounds {
            let size = spec[r] as u64;
            if size == 0 {
                continue; // radix C(avail,0)=1, contributes nothing
            }
            let avail = RANKS as u64 - used.count_ones() as u64;
            let ci = colex_index(compact(rank_sets[r], used)) as u64;
            idx = idx * binom(avail, size) + ci;
            used |= rank_sets[r];
        }
        idx
    }

    /// Reconstruct a suit's per-round rank-set masks from its column index.
    fn column_unindex(&self, mut col: u64, spec: &[u8]) -> Vec<u32> {
        // Radices are determined by the spec alone (available *count* per round
        // is fixed), so decode least-significant round first.
        let mut prefix = 0u64;
        let mut radix = vec![1u64; self.rounds];
        for r in 0..self.rounds {
            let size = spec[r] as u64;
            radix[r] = binom(RANKS as u64 - prefix, size);
            prefix += size;
        }
        let mut ci = vec![0u64; self.rounds];
        for r in (0..self.rounds).rev() {
            ci[r] = col % radix[r];
            col /= radix[r];
        }
        // Expand each round's colex index into actual ranks, skipping used ranks.
        let mut out = vec![0u32; self.rounds];
        let mut used = 0u32;
        for r in 0..self.rounds {
            let size = spec[r] as usize;
            if size == 0 {
                continue;
            }
            let compacted = self.index_to_set[size][ci[r] as usize];
            let ranks = expand(compacted, used);
            out[r] = ranks;
            used |= ranks;
        }
        out
    }

    /// Dense index of a concrete hand (`cards`: full hand as 0..52 indices,
    /// ordered hole-then-board). Any suit permutation of the same hand yields the
    /// same index.
    pub fn index(&self, cards: &[u8]) -> u64 {
        // Allocation-free: this runs at every hold'em decision node (via
        // `bucket_of`). Same arithmetic as [`HandIndexer::index_reference`], with
        // stack arrays in place of the per-suit `Vec`s.
        let rounds = self.rounds;
        let mut sets = [[0u32; MAX_ROUNDS]; SUITS];
        let mut pos = 0;
        for (r, &n) in self.cards_per_round.iter().enumerate() {
            for _ in 0..n {
                let c = cards[pos];
                sets[(c & 3) as usize][r] |= 1u32 << (c >> 2);
                pos += 1;
            }
        }
        // (spec, column index) per suit; spec padded with zeros past `rounds`,
        // which leaves the lexicographic order of equal-length specs unchanged.
        let mut suits = [([0u8; MAX_ROUNDS], 0u64); SUITS];
        for (s, suit) in suits.iter_mut().enumerate() {
            for r in 0..rounds {
                suit.0[r] = sets[s][r].count_ones() as u8;
            }
            suit.1 = self.column_index(&sets[s][..rounds], &suit.0[..rounds]);
        }
        // Canonical suit order: spec descending, then column ascending. Ties are
        // identical (spec, column) pairs, so an unstable sort is equivalent.
        suits.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        let key = pack_specs(suits.iter().map(|(spec, _)| &spec[..rounds]));
        let config = &self.configs[self.spec_key_to_config[&key]];

        let mut acc = 0u64;
        let mut si = 0usize;
        for group in &config.groups {
            // Multiset index of the group's (ascending) columns, as `multiset_index`.
            let mut msi = 0u64;
            for k in 0..group.g {
                msi += binom(suits[si + k].1 + k as u64, k as u64 + 1);
            }
            acc = acc * group.mult + msi;
            si += group.g;
        }
        config.offset + acc
    }

    /// The original `Vec`-based [`HandIndexer::index`], kept as the test oracle the
    /// allocation-free version is checked against.
    #[cfg(test)]
    fn index_reference(&self, cards: &[u8]) -> u64 {
        let sets = self.suit_round_sets(cards);
        // (spec, column index) per suit.
        let mut suits: Vec<(Vec<u8>, u64)> = (0..SUITS)
            .map(|s| {
                let spec: Vec<u8> = (0..self.rounds)
                    .map(|r| sets[s][r].count_ones() as u8)
                    .collect();
                let col = self.column_index(&sets[s], &spec);
                (spec, col)
            })
            .collect();
        // Canonical suit order: spec descending, then column ascending.
        suits.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        let specs4: Vec<Vec<u8>> = suits.iter().map(|(s, _)| s.clone()).collect();
        let cfg = self.spec_key_to_config[&pack_specs(flatten_specs(&specs4).chunks(self.rounds))];
        let config = &self.configs[cfg];

        let mut acc = 0u64;
        let mut si = 0usize;
        for group in &config.groups {
            // The next `g` suits belong to this group; their columns are already
            // ascending (from the sort), ready for the multiset index.
            let cols: Vec<u64> = (0..group.g).map(|k| suits[si + k].1).collect();
            let msi = multiset_index(&cols);
            acc = acc * group.mult + msi;
            si += group.g;
        }
        config.offset + acc
    }

    /// Inverse of [`index`]: a canonical representative hand (full 0..52 indices,
    /// ordered hole-then-board, ascending within each round) for a dense index.
    pub fn unindex(&self, index: u64) -> Vec<u8> {
        // Find the configuration whose [offset, offset+size) contains `index`.
        let cfg = match self.offsets.binary_search(&index) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let config = &self.configs[cfg];
        let mut acc = index - config.offset;

        // Decode per-group multiset indices (mixed radix, reverse group order).
        let mut msis = vec![0u64; config.groups.len()];
        for gi in (0..config.groups.len()).rev() {
            let mult = config.groups[gi].mult;
            msis[gi] = acc % mult;
            acc /= mult;
        }

        // Assign canonical suit labels group by group, ascending column within
        // each group, and rebuild each suit's per-round rank sets.
        let mut suit_sets: Vec<Vec<u32>> = Vec::with_capacity(SUITS);
        for (gi, group) in config.groups.iter().enumerate() {
            let cols = multiset_unindex(msis[gi], group.g);
            for &col in &cols {
                suit_sets.push(self.column_unindex(col, &group.spec));
            }
        }

        // Emit cards per round, ascending within a round.
        let mut out = Vec::with_capacity(self.total_cards);
        for r in 0..self.rounds {
            let mut round_cards: Vec<u8> = Vec::new();
            for (suit, sets) in suit_sets.iter().enumerate() {
                let mut m = sets[r];
                while m != 0 {
                    let rank = m.trailing_zeros() as u8;
                    round_cards.push(rank * 4 + suit as u8);
                    m &= m - 1;
                }
            }
            round_cards.sort_unstable();
            out.extend_from_slice(&round_cards);
        }
        out
    }
}

/// Column count for a spec: how many distinct column indices a suit with this
/// per-round count vector can take (ranks drawn without replacement across
/// rounds).
fn column_count(spec: &[u8]) -> u64 {
    let mut m = 1u64;
    let mut prefix = 0u64;
    for &c in spec {
        m *= binom(RANKS as u64 - prefix, c as u64);
        prefix += c as u64;
    }
    m
}

/// Multiset (combination-with-repetition) index of column values. `cols` is
/// sorted ascending; `index = sum_i C(cols[i] + i, i+1)`, ranging over
/// `0..C(m + g - 1, g)`.
#[cfg(test)]
fn multiset_index(cols: &[u64]) -> u64 {
    cols.iter()
        .enumerate()
        .map(|(i, &a)| binom(a + i as u64, i as u64 + 1))
        .sum()
}

/// Inverse of [`multiset_index`]: recover the `g` ascending column values.
fn multiset_unindex(mut msi: u64, g: usize) -> Vec<u64> {
    // Colex combinatorial number system on b_i = cols_i + i (strictly increasing).
    let mut b = vec![0u64; g];
    for i in (0..g).rev() {
        let mut v = i as u64;
        while binom(v + 1, i as u64 + 1) <= msi {
            v += 1;
        }
        b[i] = v;
        msi -= binom(v, i as u64 + 1);
    }
    (0..g).map(|i| b[i] - i as u64).collect()
}

/// Pack the four suits' per-round count vectors into one lookup key, 4 bits per
/// count in suit-then-round order. Lossless: every indexer's specs have the same
/// length (`rounds <= MAX_ROUNDS`) and counts `< 16` (asserted in `new`).
#[inline]
fn pack_specs<'a>(specs4: impl Iterator<Item = &'a [u8]>) -> u64 {
    let mut key = 0u64;
    for spec in specs4 {
        for &c in spec {
            key = (key << 4) | c as u64;
        }
    }
    key
}

/// Flatten the four suit specs into a single lookup key.
#[cfg(test)]
fn flatten_specs(specs4: &[Vec<u8>]) -> Vec<u8> {
    let mut key = Vec::with_capacity(specs4.len() * specs4.first().map_or(0, |s| s.len()));
    for s in specs4 {
        key.extend_from_slice(s);
    }
    key
}

/// Recursively enumerate the four suits' per-round count vectors, non-increasing
/// (so each configuration is produced once), exactly consuming `remaining`.
fn enumerate_configs(
    suit: usize,
    remaining: &[u8],
    prev: Option<&Vec<u8>>,
    current: &mut Vec<Vec<u8>>,
    out: &mut Vec<Vec<Vec<u8>>>,
) {
    if suit == SUITS {
        if remaining.iter().all(|&x| x == 0) {
            out.push(current.clone());
        }
        return;
    }
    for spec in gen_specs(remaining) {
        // Non-increasing: this suit's spec must not exceed the previous suit's.
        if let Some(p) = prev {
            if spec.as_slice() > p.as_slice() {
                continue;
            }
        }
        let next: Vec<u8> = remaining.iter().zip(&spec).map(|(&r, &s)| r - s).collect();
        current.push(spec.clone());
        enumerate_configs(suit + 1, &next, Some(&spec), current, out);
        current.pop();
    }
}

/// All per-round count vectors with `0 <= v[r] <= remaining[r]` and total `<= 13`
/// (a suit cannot hold more than 13 ranks).
fn gen_specs(remaining: &[u8]) -> Vec<Vec<u8>> {
    let mut res: Vec<Vec<u8>> = vec![Vec::new()];
    for &rem in remaining {
        let mut next = Vec::new();
        for base in &res {
            for v in 0..=rem {
                let mut nv = base.clone();
                nv.push(v);
                next.push(nv);
            }
        }
        res = next;
    }
    res.retain(|v| v.iter().map(|&x| x as u32).sum::<u32>() <= RANKS as u32);
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{canonical_key_packed, enumerate_canonical_flops};
    use crate::cards::Card;

    // The bucket abstraction keys on equity, which is a pure function of the
    // card *set*, so within the board the dealing order (flop vs turn vs river)
    // is strategically irrelevant — only hole-vs-board matters. Hence two rounds
    // always: hole (2) and the whole board (3/4/5). Distinguishing the turn card
    // as its own round would create spurious classes with identical features.
    fn flop() -> HandIndexer {
        HandIndexer::new(&[2, 3])
    }
    fn turn() -> HandIndexer {
        HandIndexer::new(&[2, 4])
    }
    fn river() -> HandIndexer {
        HandIndexer::new(&[2, 5])
    }

    #[test]
    fn street_sizes_match_the_literature() {
        assert_eq!(flop().size(), 1_286_792, "flop");
        assert_eq!(turn().size(), 13_960_050, "turn");
        assert_eq!(river().size(), 123_156_254, "river");
    }

    fn packed_of(cards: &[u8]) -> u64 {
        // Role-aware canonical key over the same (hole=first 2, board=rest) split.
        let hole = [Card::from_index(cards[0]), Card::from_index(cards[1])];
        let board: Vec<Card> = cards[2..].iter().map(|&i| Card::from_index(i)).collect();
        canonical_key_packed(hole, &board)
    }

    #[test]
    fn flop_indexer_covers_exactly_the_naive_canonical_set() {
        // The definitive cross-check: the set of classes the arithmetic indexer
        // enumerates equals the independently-derived naive canonical flop set,
        // and the index is a bijection onto 0..size.
        let idx = flop();
        let n = idx.size();
        let mut from_indexer = std::collections::HashSet::with_capacity(n as usize);
        for i in 0..n {
            let cards = idx.unindex(i);
            // Round-trips exactly.
            assert_eq!(idx.index(&cards), i, "round-trip at {i}");
            from_indexer.insert(packed_of(&cards));
        }
        assert_eq!(from_indexer.len(), n as usize, "indexer is not bijective");

        let naive = enumerate_canonical_flops();
        let mut from_naive = std::collections::HashSet::with_capacity(n as usize);
        for s in naive.chunks(5) {
            from_naive.insert(packed_of(s));
        }
        assert_eq!(
            from_indexer, from_naive,
            "indexer set != naive canonical set"
        );
    }

    /// Deterministic small PRNG for sampling (no rand dependency, matching the
    /// crate's SplitMix64 style elsewhere).
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Deal `total_cards` distinct random cards.
    fn random_hand(total: usize, rng: &mut u64) -> Vec<u8> {
        let mut used = 0u64;
        let mut hand = Vec::with_capacity(total);
        while hand.len() < total {
            let c = (splitmix64(rng) % 52) as u8;
            if used & (1u64 << c) == 0 {
                used |= 1u64 << c;
                hand.push(c);
            }
        }
        hand
    }

    fn suit_permute(cards: &[u8], perm: &[u8; 4]) -> Vec<u8> {
        cards
            .iter()
            .map(|&c| {
                let rank = c >> 2;
                let suit = (c & 3) as usize;
                rank * 4 + perm[suit]
            })
            .collect()
    }

    #[test]
    fn allocation_free_index_matches_reference_implementation() {
        // The hot-path `index` must number every class exactly as the original
        // Vec-based implementation did — the bucket tables were built against it.
        for idx in [flop(), turn(), river(), HandIndexer::new(&[2, 3, 1, 1])] {
            let total = idx.total_cards();
            let mut rng = 0x0bad_5eed_1234_5678u64;
            for _ in 0..200_000 {
                let hand = random_hand(total, &mut rng);
                assert_eq!(idx.index(&hand), idx.index_reference(&hand));
            }
        }
    }

    #[test]
    fn turn_and_river_round_trip_and_match_independent_canonicalizer() {
        // Size is exact (arithmetic); here we check, on a large random sample,
        // that (a) index/unindex round-trips, (b) the index is suit-permutation
        // invariant, and (c) the indexer's class agrees with the independent
        // role-aware canonicalizer used for the fully-verified flop gate.
        for idx in [turn(), river()] {
            let total = idx.total_cards();
            let mut rng = 0x1234_5678_9abc_def0u64;
            for _ in 0..20_000 {
                let hand = random_hand(total, &mut rng);
                let i = idx.index(&hand);
                assert!(i < idx.size());
                let rep = idx.unindex(i);
                // Unindex gives the same class, and re-indexing is stable.
                assert_eq!(idx.index(&rep), i, "round-trip");
                // Independent canonicalizer agrees on the class.
                assert_eq!(
                    packed_of(&rep),
                    packed_of(&hand),
                    "indexer class != canonical_key_packed"
                );
                // Suit-permutation invariance.
                let perm = [1u8, 2, 3, 0];
                let permuted = suit_permute(&hand, &perm);
                assert_eq!(idx.index(&permuted), i, "not suit-invariant");
            }
        }
    }
}

#[cfg(test)]
mod exhaustive_numbering {
    use super::HandIndexer;
    use rayon::prelude::*;

    /// Order-sensitive checksum of `index` over every (hole pair, board set) in a
    /// fixed enumeration order.
    fn checksum(idx: &HandIndexer, board_n: usize) -> (u64, u64) {
        let holes: Vec<(u8, u8)> = (0..52u8).flat_map(|a| (a + 1..52).map(move |b| (a, b))).collect();
        let per: Vec<(u64, u64)> = holes.par_iter().map(|&(a, b)| {
            let rest: Vec<u8> = (0..52u8).filter(|&c| c != a && c != b).collect();
            let mut pos: Vec<usize> = (0..board_n).collect();
            let mut h = 0xcbf29ce484222325u64 ^ ((a as u64) << 8 | b as u64);
            let mut n = 0u64;
            let mut cards = vec![a, b];
            cards.resize(2 + board_n, 0);
            loop {
                for (k, &p) in pos.iter().enumerate() { cards[2 + k] = rest[p]; }
                let i = idx.index(&cards);
                h = (h ^ i).wrapping_mul(0x100000001b3).rotate_left(7);
                n += 1;
                // next combination of board_n from rest.len()
                let m = rest.len();
                let mut k = board_n;
                loop {
                    if k == 0 { return (h, n); }
                    k -= 1;
                    if pos[k] != k + m - board_n { break; }
                }
                pos[k] += 1;
                for j in k + 1..board_n { pos[j] = pos[j - 1] + 1; }
            }
        }).collect();
        per.iter().fold((0u64, 0u64), |(h, n), &(x, c)| ((h.rotate_left(13) ^ x).wrapping_mul(0x9E3779B97F4A7C15), n + c))
    }

    /// Exhaustive numbering gate: the bucket tables were built against the ORIGINAL
    /// Vec-based indexer, so `index` must reproduce its class number for every one
    /// of the 3.14e9 (hole, board) sets. These hashes were produced by the pre-
    /// optimization engine (6a5246a^) and matched by the allocation-free one; any
    /// numbering change moves them. ~2-10 min on 4-16 cores, so run on demand:
    /// `cargo test --release --lib exhaustive_index -- --ignored`.
    #[test]
    #[ignore]
    fn exhaustive_index_matches_original_numbering() {
        for (name, board_n, hands, hash) in [
            ("flop", 3usize, 25_989_600u64, 0x9e26_21b3_92c2_b10du64),
            ("turn", 4, 305_377_800, 0x601f_0575_496a_f8b6),
            ("river", 5, 2_809_475_760, 0x3395_4e94_82b5_2495),
        ] {
            let idx = HandIndexer::new(&[2, board_n as u8]);
            assert_eq!(checksum(&idx, board_n), (hash, hands), "{name} numbering changed");
        }
    }
}
