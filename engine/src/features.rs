// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
//! Equity-distribution feature extraction (Phase 3b).
//!
//! Card abstraction buckets hands by *strategic* similarity, not by current
//! strength. The potential-aware feature (Ganzfried–Sandholm; the
//! Libratus/Pluribus lineage) represents a hand-on-a-board as the **distribution
//! of its equity one street later**: as the next community card runs out, the
//! hand's all-in equity-vs-random moves around, and the *shape* of that movement
//! separates a made hand (tight, high) from a draw (bimodal — busts low or hits
//! high) even when their current equity is identical.
//!
//! For each situation we bin those next-street equities into a fixed-width
//! histogram over `[0, 1]`; the GPU k-means in `parciver.clustering` then
//! clusters the histograms under Earth-Mover's distance. This is the branchy
//! card-chasing half of the pipeline, so it lives in Rust; every equity here is
//! the already-certified [`crate::equity::equity_vs_random`], so this module only
//! has to enumerate the next card and bin — the arithmetic it delegates is proven.
//!
//! Street features:
//!   * **flop**  (2 hole + 3 board): distribution over the 47 turn cards of the
//!     *turn* equity (which itself still integrates over the river).
//!   * **turn**  (2 hole + 4 board): distribution over the 46 river cards of the
//!     *river* equity (an exact showdown vs every opponent holding).
//!   * **river** (2 hole + 5 board): a single scalar equity — a degenerate
//!     one-point distribution, so the river clusters on equity directly.

use rayon::prelude::*;

use crate::cards::Card;
use crate::equity::equity_vs_random;

/// Bin an equity in `[0, 1]` into `0..bins` (right edge folds into the last bin).
#[inline]
fn bin_of(equity: f64, bins: usize) -> usize {
    let b = (equity * bins as f64) as usize;
    if b >= bins {
        bins - 1
    } else {
        b
    }
}

/// Mask of the 0..52 indices occupied by `hole` and `board`.
#[inline]
fn used_mask(hole: [Card; 2], board: &[Card]) -> u64 {
    let mut used = 0u64;
    for c in hole.iter().chain(board) {
        used |= 1u64 << c.index();
    }
    used
}

/// Flop feature: the normalized histogram (length `bins`) of the hero's turn
/// equity across every possible turn card. `flop` is exactly 3 board cards.
///
/// Each turn card yields a 4-card board whose exact equity-vs-random still
/// enumerates the river, so this is the genuine potential-aware distribution,
/// not a one-card approximation.
pub fn flop_equity_histogram(hole: [Card; 2], flop: [Card; 3], bins: usize) -> Vec<f32> {
    let used = used_mask(hole, &flop);
    let mut hist = vec![0.0f32; bins];
    let mut board = [flop[0], flop[1], flop[2], Card::default()];
    let mut n = 0u32;
    for t in 0..52u8 {
        if used & (1u64 << t) != 0 {
            continue;
        }
        board[3] = Card::from_index(t);
        let eq = equity_vs_random(hole, &board).equity();
        hist[bin_of(eq, bins)] += 1.0;
        n += 1;
    }
    let inv = 1.0 / n as f32;
    for h in hist.iter_mut() {
        *h *= inv;
    }
    hist
}

/// Turn feature: the normalized histogram (length `bins`) of the hero's river
/// equity across every possible river card. `turn_board` is exactly 4 board
/// cards. Each river card completes the board, so its equity is an exact
/// showdown against every remaining opponent holding.
pub fn turn_equity_histogram(hole: [Card; 2], turn_board: [Card; 4], bins: usize) -> Vec<f32> {
    let used = used_mask(hole, &turn_board);
    let mut hist = vec![0.0f32; bins];
    let mut board = [
        turn_board[0],
        turn_board[1],
        turn_board[2],
        turn_board[3],
        Card::default(),
    ];
    let mut n = 0u32;
    for r in 0..52u8 {
        if used & (1u64 << r) != 0 {
            continue;
        }
        board[4] = Card::from_index(r);
        let eq = equity_vs_random(hole, &board).equity();
        hist[bin_of(eq, bins)] += 1.0;
        n += 1;
    }
    let inv = 1.0 / n as f32;
    for h in hist.iter_mut() {
        *h *= inv;
    }
    hist
}

/// River feature: the scalar all-in equity of `hole` on the complete 5-card
/// `river_board`. The clustering treats this as a one-bin distribution.
pub fn river_equity(hole: [Card; 2], river_board: [Card; 5]) -> f64 {
    equity_vs_random(hole, &river_board).equity()
}

/// Batch flop histograms, one row per `(hole, flop)` situation, in parallel over
/// the hardware threads. `holes` is `2 * count` card indices, `flops` is
/// `3 * count`; the returned flat vector is `count * bins`, row-major (row `i`
/// is `out[i*bins .. (i+1)*bins]`). Rows are independent, so rayon never changes
/// a value.
pub fn flop_histograms_batch(
    holes: &[Card],
    flops: &[Card],
    count: usize,
    bins: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; count * bins];
    out.par_chunks_mut(bins).enumerate().for_each(|(i, row)| {
        let hole = [holes[2 * i], holes[2 * i + 1]];
        let flop = [flops[3 * i], flops[3 * i + 1], flops[3 * i + 2]];
        let h = flop_equity_histogram(hole, flop, bins);
        row.copy_from_slice(&h);
    });
    out
}

/// Batch turn histograms; layout mirrors [`flop_histograms_batch`] with a 4-card
/// board per situation (`turn_boards` is `4 * count`).
pub fn turn_histograms_batch(
    holes: &[Card],
    turn_boards: &[Card],
    count: usize,
    bins: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; count * bins];
    out.par_chunks_mut(bins).enumerate().for_each(|(i, row)| {
        let hole = [holes[2 * i], holes[2 * i + 1]];
        let tb = [
            turn_boards[4 * i],
            turn_boards[4 * i + 1],
            turn_boards[4 * i + 2],
            turn_boards[4 * i + 3],
        ];
        let h = turn_equity_histogram(hole, tb, bins);
        row.copy_from_slice(&h);
    });
    out
}

/// Batch river equities, one scalar per `(hole, river_board)` situation, in
/// parallel. `holes` is `2 * count`, `river_boards` is `5 * count`.
pub fn river_equities_batch(holes: &[Card], river_boards: &[Card], count: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; count];
    out.par_iter_mut().enumerate().for_each(|(i, e)| {
        let hole = [holes[2 * i], holes[2 * i + 1]];
        let rb = [
            river_boards[5 * i],
            river_boards[5 * i + 1],
            river_boards[5 * i + 2],
            river_boards[5 * i + 3],
            river_boards[5 * i + 4],
        ];
        *e = river_equity(hole, rb) as f32;
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::parse_card;

    fn c(s: &str) -> Card {
        Card::from_index(parse_card(s).unwrap())
    }

    /// Expected value (mean equity) of a normalized histogram, in [0, 1]:
    /// sum over bins of (bin center) * mass.
    fn hist_mean(hist: &[f32]) -> f64 {
        let bins = hist.len();
        hist.iter()
            .enumerate()
            .map(|(b, &m)| ((b as f64 + 0.5) / bins as f64) * m as f64)
            .sum()
    }

    #[test]
    fn flop_histogram_is_a_normalized_distribution() {
        let h = flop_equity_histogram([c("Ac"), c("Ad")], [c("Ah"), c("7d"), c("2c")], 50);
        assert_eq!(h.len(), 50);
        let total: f32 = h.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "mass {total}");
        assert!(h.iter().all(|&m| m >= 0.0));
    }

    #[test]
    fn made_set_dominates_a_dead_hand_on_the_flop() {
        // Flopped top set (AA on A72) sits high; 72o on a broadway flop is nearly
        // drawing dead and sits low. The distribution means must order that way.
        let set = flop_equity_histogram([c("Ac"), c("Ad")], [c("Ah"), c("7d"), c("2c")], 50);
        let trash = flop_equity_histogram([c("7c"), c("2d")], [c("Ah"), c("Kd"), c("Qs")], 50);
        let (m_set, m_trash) = (hist_mean(&set), hist_mean(&trash));
        assert!(m_set > 0.85, "set mean {m_set}");
        assert!(m_trash < 0.35, "trash mean {m_trash}");
        assert!(m_set > m_trash);
    }

    #[test]
    fn flop_extraction_is_deterministic() {
        // Exact enumeration → bit-identical across calls (no sampling).
        let a = flop_equity_histogram([c("Ts"), c("Js")], [c("Qs"), c("9d"), c("2c")], 30);
        let b = flop_equity_histogram([c("Ts"), c("Js")], [c("Qs"), c("9d"), c("2c")], 30);
        assert_eq!(a, b);
    }

    #[test]
    fn turn_histogram_is_normalized_and_orders_correctly() {
        // Nut flush on the turn vs a weak no-pair hand.
        let nuts =
            turn_equity_histogram([c("Ah"), c("Kh")], [c("Qh"), c("Jh"), c("2h"), c("3d")], 50);
        let weak =
            turn_equity_histogram([c("7c"), c("2d")], [c("Ah"), c("Kd"), c("Qs"), c("Jc")], 50);
        assert!((nuts.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(hist_mean(&nuts) > hist_mean(&weak));
        assert!(hist_mean(&nuts) > 0.9, "flush mean {}", hist_mean(&nuts));
    }

    #[test]
    fn batch_matches_single_row_by_row() {
        // The rayon batch must reproduce the single-situation function exactly.
        let holes = [c("Ac"), c("Ad"), c("7c"), c("2d")];
        let flops = [
            c("Ah"),
            c("7d"),
            c("2s"), // situation 0 board
            c("Ah"),
            c("Kd"),
            c("Qs"), // situation 1 board
        ];
        let bins = 40;
        let batch = flop_histograms_batch(&holes, &flops, 2, bins);
        let row0 =
            flop_equity_histogram([holes[0], holes[1]], [flops[0], flops[1], flops[2]], bins);
        let row1 =
            flop_equity_histogram([holes[2], holes[3]], [flops[3], flops[4], flops[5]], bins);
        assert_eq!(&batch[0..bins], &row0[..]);
        assert_eq!(&batch[bins..2 * bins], &row1[..]);
    }

    #[test]
    fn river_equity_matches_the_equity_engine() {
        // River feature is just the certified scalar equity.
        let e = river_equity(
            [c("Ac"), c("Ad")],
            [c("Ah"), c("7d"), c("2c"), c("Ts"), c("Js")],
        );
        assert!(
            e > 0.9,
            "top set on the river should crush a random hand, got {e}"
        );
    }
}
