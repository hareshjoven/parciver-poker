//! `parciver_engine` — the fast Rust half of the poker bot, exposed to Python
//! via PyO3.
//!
//! The Python side owns orchestration, the exploitability oracle, and the
//! GPU/neural work; this crate owns the branchy game-tree traversal and the CFR
//! hot loop, where systems-language speed actually pays off. The solvers here
//! return the AVERAGE strategy keyed by infoset string, in the exact format the
//! Python games/oracle use, so `parciver.eval.exploitability` can certify a
//! Rust-solved strategy is a genuine equilibrium of the Python-defined game.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// mimalloc as the global allocator: node-table growth allocates from every
/// solver thread at once, which the Windows system heap serializes.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

extern "C" {
    /// mimalloc's collection of the calling thread's heap (its library is
    /// linked for the global allocator); `force` also returns every freed
    /// page to the system now, not on a later call.
    fn mi_collect(force: bool);
}

mod abstraction;
mod analysis;
mod atomic_file;
mod bucketing;
mod cards;
mod cfr;
mod cfr_np;
mod equity;
mod exploit;
mod features;
mod game;
mod hand_eval;
mod isomorphism;
mod kuhn;
mod kuhn_n;
mod leduc;
mod mccfr;
// The NLHE engine is validated by its own Rust tests and now keys infosets on
// the B2 card abstraction, but it is not yet reachable from the crate root (the
// Python solve entry point lands in B3). Suppress the interim dead-code warnings
// until then; this attribute comes off when B3 exposes the solver.
#[allow(dead_code)]
mod nlhe;
mod play;
mod arena;
mod lbr;
mod search;
mod subgame;
mod s32;

use bucketing::Abstraction;
use cards::Card;
use cfr::{variant_by_name, CFRSolver};
use cfr_np::NpCFRSolver;
use game::count_infosets;
use game::Game;
use isomorphism::HandIndexer;
use kuhn::Kuhn;
use kuhn_n::KuhnN;
use leduc::Leduc;
use mccfr::{FixedOpponentMCCFR, MCCFRSolver};
use nlhe::{ActLabel, Nlhe, NlheConfig, NlheState};

/// Solve Leduc hold'em with full-traversal CFR and return the average strategy
/// as `{infoset_key: [prob, ...]}`. `variant` is one of "cfr", "cfr+",
/// "linear-cfr", "dcfr".
#[pyfunction]
#[pyo3(signature = (iterations, variant = "dcfr"))]
fn solve_leduc(
    py: Python<'_>,
    iterations: u64,
    variant: &str,
) -> PyResult<HashMap<String, Vec<f64>>> {
    let v = variant_by_name(variant)
        .ok_or_else(|| PyValueError::new_err(format!("unknown CFR variant: {variant:?}")))?;
    // Release the GIL for the pure-Rust hot loop.
    Ok(py.detach(|| {
        let game = Leduc::new();
        let mut solver = CFRSolver::new(v);
        solver.run(&game, iterations);
        solver.average_strategy()
    }))
}

/// Solve Kuhn poker with full-traversal CFR. See `solve_leduc`.
#[pyfunction]
#[pyo3(signature = (iterations, variant = "dcfr"))]
fn solve_kuhn(
    py: Python<'_>,
    iterations: u64,
    variant: &str,
) -> PyResult<HashMap<String, Vec<f64>>> {
    let v = variant_by_name(variant)
        .ok_or_else(|| PyValueError::new_err(format!("unknown CFR variant: {variant:?}")))?;
    Ok(py.detach(|| {
        let game = Kuhn::new();
        let mut solver = CFRSolver::new(v);
        solver.run(&game, iterations);
        solver.average_strategy()
    }))
}

/// Solve Leduc hold'em with external-sampling MCCFR and return the average
/// strategy as `{infoset_key: [prob, ...]}`. `variant` is one of "mccfr"
/// (vanilla), "mccfr+" (RM+ with linear averaging), "linear" (Linear CFR). `seed`
/// makes the sampled run fully reproducible. This is the sampled hot loop that
/// scales to the abstracted hold'em tree; on Leduc it is validated against the
/// same Python exploitability oracle as the full-traversal `solve_leduc`.
#[pyfunction]
#[pyo3(signature = (iterations, variant = "linear", seed = 0, threads = 1))]
fn solve_leduc_mccfr(
    py: Python<'_>,
    iterations: u64,
    variant: &str,
    seed: u64,
    threads: usize,
) -> PyResult<HashMap<String, Vec<f64>>> {
    let v = mccfr::variant_by_name(variant)
        .ok_or_else(|| PyValueError::new_err(format!("unknown MCCFR variant: {variant:?}")))?;
    // Release the GIL for the pure-Rust hot loop. `threads` defaults to 1 (the exact
    // deterministic serial path); `> 1` parallelizes, `0` uses all cores — this is
    // what lets the phase audit score the PARALLEL solver against the oracle too.
    Ok(py.detach(|| {
        let game = Leduc::new();
        let mut solver = MCCFRSolver::new(v, seed);
        solver.set_threads(resolve_threads(threads));
        solver.run(&game, iterations);
        solver.average_strategy()
    }))
}

/// Solve Kuhn poker with external-sampling MCCFR. See `solve_leduc_mccfr`.
#[pyfunction]
#[pyo3(signature = (iterations, variant = "linear", seed = 0, threads = 1))]
fn solve_kuhn_mccfr(
    py: Python<'_>,
    iterations: u64,
    variant: &str,
    seed: u64,
    threads: usize,
) -> PyResult<HashMap<String, Vec<f64>>> {
    let v = mccfr::variant_by_name(variant)
        .ok_or_else(|| PyValueError::new_err(format!("unknown MCCFR variant: {variant:?}")))?;
    Ok(py.detach(|| {
        let game = Kuhn::new();
        let mut solver = MCCFRSolver::new(v, seed);
        solver.set_threads(resolve_threads(threads));
        solver.run(&game, iterations);
        solver.average_strategy()
    }))
}

/// Number of information sets in the Rust Leduc tree (canonically 288). A cheap
/// structural cross-check against the Python engine.
#[pyfunction]
fn leduc_infoset_count() -> usize {
    count_infosets(&Leduc::new())
}

/// Number of information sets in the Rust Kuhn tree (canonically 12).
#[pyfunction]
fn kuhn_infoset_count() -> usize {
    count_infosets(&Kuhn::new())
}

// --- N-player Kuhn foundation (R-MAPSS Phase 4, Step 0.2) ------------------
//
// The multiplayer sibling of the 2p Kuhn oracle. `KuhnN::new(2)` reproduces the
// certified 2p game byte-for-byte; `KuhnN::new(3)` is the first exactly-solvable
// multiplayer game for the R-MAPSS track. All exploitation claims on it are
// controlled by the EXACT best response below (§28).

/// Solve n-player Kuhn with full-traversal general-sum CFR. `variant` one of
/// "cfr"/"cfr+"/"linear-cfr"/"dcfr". For n>=3 CFR has NO equilibrium guarantee
/// (Gibson 2013); the returned average strategy is an approximate, low-NashConv
/// profile whose exact NashConv is checked by `kuhn_n_nashconv`.
#[pyfunction]
#[pyo3(signature = (num_players, iterations, variant = "cfr"))]
fn solve_kuhn_n(
    py: Python<'_>,
    num_players: usize,
    iterations: u64,
    variant: &str,
) -> PyResult<HashMap<String, Vec<f64>>> {
    let v = variant_by_name(variant)
        .ok_or_else(|| PyValueError::new_err(format!("unknown CFR variant: {variant:?}")))?;
    Ok(py.detach(|| {
        let game = KuhnN::new(num_players);
        let mut solver = NpCFRSolver::new(v, num_players);
        solver.run(&game, iterations);
        solver.average_strategy()
    }))
}

/// Number of information sets in n-player Kuhn (2p -> 12, 3p -> 48).
#[pyfunction]
fn kuhn_n_infoset_count(num_players: usize) -> usize {
    count_infosets(&KuhnN::new(num_players))
}

/// Exact TOTAL NashConv of `strategy` on n-player Kuhn (Σ_i [BR_i − u_i]). Zero
/// exactly at a Nash equilibrium; the independent hallucination/convergence
/// control. Computed with no signature memo (exact, like the Kuhn/Leduc gates).
#[pyfunction]
fn kuhn_n_nashconv(py: Python<'_>, num_players: usize, strategy: HashMap<String, Vec<f64>>) -> f64 {
    py.detach(|| {
        exploit::nash_conv_n(&KuhnN::new(num_players), &strategy, |_s| {
            Option::<String>::None
        })
    })
}

/// Per-player NashConv (each seat's gain from unilaterally best-responding), one
/// entry per seat in seat order.
#[pyfunction]
fn kuhn_n_nashconv_per_player(
    py: Python<'_>,
    num_players: usize,
    strategy: HashMap<String, Vec<f64>>,
) -> Vec<f64> {
    py.detach(|| {
        exploit::nash_conv_per_player(&KuhnN::new(num_players), &strategy, |_s| {
            Option::<String>::None
        })
    })
}

/// Exact best-response EV for `player` vs the fixed others in `strategy`
/// (`strategy` is consulted only at the OTHER players' infosets). The §28
/// exploitation CEILING against a fixed opponent profile.
#[pyfunction]
fn kuhn_n_best_response_value(
    py: Python<'_>,
    num_players: usize,
    strategy: HashMap<String, Vec<f64>>,
    player: usize,
) -> f64 {
    py.detach(|| {
        exploit::best_response_value(&KuhnN::new(num_players), &strategy, player, &|_s| {
            Option::<String>::None
        })
    })
}

/// Exact EV to `player` when ALL players follow `strategy`. The on-policy value
/// used to measure a Hero deviation's realized EV vs a fixed opponent pair.
#[pyfunction]
fn kuhn_n_on_policy_value(
    py: Python<'_>,
    num_players: usize,
    strategy: HashMap<String, Vec<f64>>,
    player: usize,
) -> f64 {
    py.detach(|| {
        exploit::on_policy_value(&KuhnN::new(num_players), &strategy, player, |_s| {
            Option::<String>::None
        })
    })
}

// --- hand evaluator -------------------------------------------------------

fn to_cards<const N: usize>(indices: &[u8]) -> PyResult<[Card; N]> {
    if indices.len() != N {
        return Err(PyValueError::new_err(format!(
            "expected {N} card indices, got {}",
            indices.len()
        )));
    }
    let mut out = [Card::default(); N];
    for (o, &i) in out.iter_mut().zip(indices) {
        if i >= 52 {
            return Err(PyValueError::new_err(format!(
                "card index {i} out of 0..52"
            )));
        }
        *o = Card::from_index(i);
    }
    Ok(out)
}

/// Evaluate a 7-card hand (list of seven 0..52 card indices) to a comparable
/// score; a larger score is the stronger hand. Use `hand_category(score)` for
/// the 0..8 category.
#[pyfunction]
fn eval7(cards: Vec<u8>) -> PyResult<u32> {
    Ok(hand_eval::eval7(&to_cards::<7>(&cards)?))
}

/// Evaluate a 5-card hand (list of five 0..52 card indices) to a comparable score.
#[pyfunction]
fn eval5(cards: Vec<u8>) -> PyResult<u32> {
    Ok(hand_eval::eval5(&to_cards::<5>(&cards)?))
}

/// The 0..8 hand category of an `eval5`/`eval7` score (0=high card … 8=straight flush).
#[pyfunction]
fn hand_category(score: u32) -> u8 {
    hand_eval::category(score)
}

/// Human-readable name for a 0..8 hand category.
#[pyfunction]
fn hand_category_name(category: u8) -> PyResult<&'static str> {
    hand_eval::CATEGORY_NAMES
        .get(category as usize)
        .copied()
        .ok_or_else(|| PyValueError::new_err(format!("category {category} out of 0..9")))
}

/// Parse a two-char card string like "As", "Td", "2c" into its 0..52 index.
#[pyfunction]
fn parse_card(card: &str) -> PyResult<u8> {
    cards::parse_card(card).map_err(PyValueError::new_err)
}

// --- card abstraction: suit isomorphism -----------------------------------

/// Validate that `indices` are all in 0..52, then convert to `Card`s.
fn validate_cards(indices: &[u8]) -> PyResult<Vec<Card>> {
    for &i in indices {
        if i >= 52 {
            return Err(PyValueError::new_err(format!(
                "card index {i} out of 0..52"
            )));
        }
    }
    Ok(indices.iter().map(|&i| Card::from_index(i)).collect())
}

/// Split card lists for the equity API: check indices are valid, `hole` is
/// exactly two cards, `board` is 0..=5 cards, and nothing is duplicated.
fn parse_hole_board(hole: &[u8], board: &[u8]) -> PyResult<([Card; 2], Vec<Card>)> {
    if hole.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "hole must be exactly 2 cards, got {}",
            hole.len()
        )));
    }
    if board.len() > 5 {
        return Err(PyValueError::new_err(format!(
            "board must be 0..=5 cards, got {}",
            board.len()
        )));
    }
    let mut seen = 0u64;
    for &i in hole.iter().chain(board) {
        if i >= 52 {
            return Err(PyValueError::new_err(format!(
                "card index {i} out of 0..52"
            )));
        }
        let bit = 1u64 << i;
        if seen & bit != 0 {
            return Err(PyValueError::new_err(format!("duplicate card index {i}")));
        }
        seen |= bit;
    }
    let h = validate_cards(hole)?;
    Ok(([h[0], h[1]], validate_cards(board)?))
}

/// Suit-isomorphic preflop class (0..169) of a two-card holding.
#[pyfunction]
fn preflop_index(a: u8, b: u8) -> PyResult<u8> {
    let cards = validate_cards(&[a, b])?;
    Ok(abstraction::preflop_index(cards[0], cards[1]))
}

/// Human-readable name of a preflop class, e.g. "AA", "AKs", "72o".
#[pyfunction]
fn preflop_name(idx: u8) -> PyResult<String> {
    if idx >= 169 {
        return Err(PyValueError::new_err(format!(
            "preflop class {idx} out of 0..169"
        )));
    }
    Ok(abstraction::preflop_name(idx))
}

/// Canonical suit-isomorphic representative of a card set, as sorted 0..52
/// indices. Suit-isomorphic sets share a canonical form.
#[pyfunction]
fn canonical_cards(cards: Vec<u8>) -> PyResult<Vec<u8>> {
    Ok(abstraction::canonical_cards(&validate_cards(&cards)?))
}

// --- equity ----------------------------------------------------------------

/// Exact all-in equity (tie = half) of `hole` vs a uniform-random opponent on
/// `board` (0..=5 cards). Enumerates every runout and opponent hand; use the
/// Monte-Carlo variant preflop where the exact count is ~2.1e9.
#[pyfunction]
fn equity_vs_random(py: Python<'_>, hole: Vec<u8>, board: Vec<u8>) -> PyResult<f64> {
    let (h, b) = parse_hole_board(&hole, &board)?;
    Ok(py.detach(|| equity::equity_vs_random(h, &b).equity()))
}

/// Exact all-in equity of `hole` vs one specific `villain` holding on `board`.
#[pyfunction]
fn equity_vs_hand(
    py: Python<'_>,
    hole: Vec<u8>,
    villain: Vec<u8>,
    board: Vec<u8>,
) -> PyResult<f64> {
    let (h, b) = parse_hole_board(&hole, &board)?;
    if villain.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "villain must be exactly 2 cards, got {}",
            villain.len()
        )));
    }
    // Reject any overlap between villain and hole/board.
    let mut seen = 0u64;
    for &i in hole.iter().chain(&board) {
        seen |= 1u64 << i;
    }
    for &i in &villain {
        if i >= 52 {
            return Err(PyValueError::new_err(format!(
                "card index {i} out of 0..52"
            )));
        }
        if seen & (1u64 << i) != 0 {
            return Err(PyValueError::new_err(format!(
                "villain card {i} collides with hole/board"
            )));
        }
        seen |= 1u64 << i;
    }
    let v = validate_cards(&villain)?;
    Ok(py.detach(|| equity::equity_vs_hand(h, [v[0], v[1]], &b).equity()))
}

/// Monte-Carlo all-in equity of `hole` vs a uniform-random opponent on `board`,
/// for any street incl. preflop. Deterministic given `seed`.
#[pyfunction]
#[pyo3(signature = (hole, board, samples, seed = 0))]
fn equity_vs_random_mc(
    py: Python<'_>,
    hole: Vec<u8>,
    board: Vec<u8>,
    samples: u64,
    seed: u64,
) -> PyResult<f64> {
    let (h, b) = parse_hole_board(&hole, &board)?;
    Ok(py.detach(|| equity::equity_vs_random_mc(h, &b, samples, seed).equity()))
}

// --- card abstraction: equity-distribution features (Phase 3b) --------------

/// Convert a flat `0..52` index slice of an expected length into `Card`s.
fn cards_exact(indices: &[u8], expect: usize, what: &str) -> PyResult<Vec<Card>> {
    if indices.len() != expect {
        return Err(PyValueError::new_err(format!(
            "{what} must be exactly {expect} cards, got {}",
            indices.len()
        )));
    }
    validate_cards(indices)
}

/// Reject duplicate card indices across the given slices (a situation must use
/// distinct cards).
fn reject_duplicates(groups: &[&[u8]]) -> PyResult<()> {
    let mut seen = 0u64;
    for g in groups {
        for &i in *g {
            let bit = 1u64 << i;
            if seen & bit != 0 {
                return Err(PyValueError::new_err(format!("duplicate card index {i}")));
            }
            seen |= bit;
        }
    }
    Ok(())
}

/// Flop feature: normalized histogram (length `bins`) of the turn equity over
/// every turn card. `hole` is 2 cards, `flop` is 3.
#[pyfunction]
fn flop_equity_histogram(
    py: Python<'_>,
    hole: Vec<u8>,
    flop: Vec<u8>,
    bins: usize,
) -> PyResult<Vec<f32>> {
    if bins == 0 {
        return Err(PyValueError::new_err("bins must be >= 1"));
    }
    let h = cards_exact(&hole, 2, "hole")?;
    let f = cards_exact(&flop, 3, "flop")?;
    reject_duplicates(&[&hole, &flop])?;
    Ok(py.detach(|| features::flop_equity_histogram([h[0], h[1]], [f[0], f[1], f[2]], bins)))
}

/// Turn feature: normalized histogram of the river equity over every river card.
/// `hole` is 2 cards, `turn_board` is 4.
#[pyfunction]
fn turn_equity_histogram(
    py: Python<'_>,
    hole: Vec<u8>,
    turn_board: Vec<u8>,
    bins: usize,
) -> PyResult<Vec<f32>> {
    if bins == 0 {
        return Err(PyValueError::new_err("bins must be >= 1"));
    }
    let h = cards_exact(&hole, 2, "hole")?;
    let b = cards_exact(&turn_board, 4, "turn_board")?;
    reject_duplicates(&[&hole, &turn_board])?;
    Ok(py.detach(|| features::turn_equity_histogram([h[0], h[1]], [b[0], b[1], b[2], b[3]], bins)))
}

/// Every turn-and-river runout of a flop, for all-in and check-down values:
/// little-endian u32 hand strengths, row-major `[1176 runouts][1176 holdings]`.
/// Runouts are the pairs t < r of the 49 unseen cards in order, holdings the
/// flop search's order (`FlatSubgame`'s); a holding that uses a runout card
/// has strength 0.
#[pyfunction]
fn flop_runouts<'py>(py: Python<'py>, board: Vec<u8>) -> PyResult<Bound<'py, PyBytes>> {
    cards_exact(&board, 3, "board")?;
    reject_duplicates(&[&board])?;
    let b = [board[0], board[1], board[2]];
    let s = py.detach(|| search::Hands::flop_runout_strengths(&b));
    let bytes: Vec<u8> = s.iter().flat_map(|x| x.to_le_bytes()).collect();
    Ok(PyBytes::new(py, &bytes))
}

/// River feature: scalar all-in equity on a complete 5-card board.
#[pyfunction]
fn river_equity(py: Python<'_>, hole: Vec<u8>, river_board: Vec<u8>) -> PyResult<f64> {
    let h = cards_exact(&hole, 2, "hole")?;
    let b = cards_exact(&river_board, 5, "river_board")?;
    reject_duplicates(&[&hole, &river_board])?;
    Ok(py.detach(|| features::river_equity([h[0], h[1]], [b[0], b[1], b[2], b[3], b[4]])))
}

/// Batch flop histograms in parallel. `holes` is `2*count` indices, `flops` is
/// `3*count`; returns a flat `count*bins` row-major vector. Duplicate cards
/// *within* a situation are the caller's responsibility (not re-checked per row
/// for speed); this is fed canonical enumerations.
#[pyfunction]
fn flop_histograms_batch(
    py: Python<'_>,
    holes: Vec<u8>,
    flops: Vec<u8>,
    bins: usize,
) -> PyResult<Vec<f32>> {
    if bins == 0 {
        return Err(PyValueError::new_err("bins must be >= 1"));
    }
    if holes.len() % 2 != 0 {
        return Err(PyValueError::new_err(
            "holes length must be a multiple of 2",
        ));
    }
    let count = holes.len() / 2;
    if flops.len() != 3 * count {
        return Err(PyValueError::new_err(format!(
            "flops length {} != 3*count {}",
            flops.len(),
            3 * count
        )));
    }
    let hc = validate_cards(&holes)?;
    let fc = validate_cards(&flops)?;
    Ok(py.detach(|| features::flop_histograms_batch(&hc, &fc, count, bins)))
}

/// Batch turn histograms in parallel. `holes` is `2*count`, `turn_boards` is
/// `4*count`; returns a flat `count*bins` row-major vector.
#[pyfunction]
fn turn_histograms_batch(
    py: Python<'_>,
    holes: Vec<u8>,
    turn_boards: Vec<u8>,
    bins: usize,
) -> PyResult<Vec<f32>> {
    if bins == 0 {
        return Err(PyValueError::new_err("bins must be >= 1"));
    }
    if holes.len() % 2 != 0 {
        return Err(PyValueError::new_err(
            "holes length must be a multiple of 2",
        ));
    }
    let count = holes.len() / 2;
    if turn_boards.len() != 4 * count {
        return Err(PyValueError::new_err(format!(
            "turn_boards length {} != 4*count {}",
            turn_boards.len(),
            4 * count
        )));
    }
    let hc = validate_cards(&holes)?;
    let bc = validate_cards(&turn_boards)?;
    Ok(py.detach(|| features::turn_histograms_batch(&hc, &bc, count, bins)))
}

/// Batch river equities in parallel. `holes` is `2*count`, `river_boards` is
/// `5*count`; returns a flat `count` vector of scalar equities.
#[pyfunction]
fn river_equities_batch(
    py: Python<'_>,
    holes: Vec<u8>,
    river_boards: Vec<u8>,
) -> PyResult<Vec<f32>> {
    if holes.len() % 2 != 0 {
        return Err(PyValueError::new_err(
            "holes length must be a multiple of 2",
        ));
    }
    let count = holes.len() / 2;
    if river_boards.len() != 5 * count {
        return Err(PyValueError::new_err(format!(
            "river_boards length {} != 5*count {}",
            river_boards.len(),
            5 * count
        )));
    }
    let hc = validate_cards(&holes)?;
    let bc = validate_cards(&river_boards)?;
    Ok(py.detach(|| features::river_equities_batch(&hc, &bc, count)))
}

/// Role-aware suit-isomorphic canonical key of a `(hole, board)` situation, as
/// sorted `0..52` indices `[hole.. board..]`. Distinguishes hole from board.
#[pyfunction]
fn canonical_situation(hole: Vec<u8>, board: Vec<u8>) -> PyResult<Vec<u8>> {
    if hole.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "hole must be exactly 2 cards, got {}",
            hole.len()
        )));
    }
    if board.len() > 5 {
        return Err(PyValueError::new_err(format!(
            "board must be 0..=5 cards, got {}",
            board.len()
        )));
    }
    let h = validate_cards(&hole)?;
    let b = validate_cards(&board)?;
    Ok(abstraction::canonical_situation(&h, &b))
}

/// Enumerate every suit-isomorphic canonical flop situation (2 hole + 3 board)
/// by exact naive dedup, as a flat sorted `count*5` vector. There are exactly
/// 1,286,792 of them. (Turn/river are far too many to materialize this way — use
/// the dense indexer instead.)
#[pyfunction]
fn enumerate_canonical_flops(py: Python<'_>) -> Vec<u8> {
    py.detach(abstraction::enumerate_canonical_flops)
}

// --- dense hand-isomorphism indexer (Waugh) --------------------------------

/// The three street indexers (flop `[2,3]`, turn `[2,4]`, river `[2,5]`), built
/// once. The board is one round (equity is a card-*set* function, so dealing
/// order within the board is irrelevant), the hole another — so the classes are
/// exactly the role-aware canonical `(hole, board)` situations.
fn street_indexers() -> &'static [HandIndexer; 3] {
    static INDEXERS: OnceLock<[HandIndexer; 3]> = OnceLock::new();
    INDEXERS.get_or_init(|| {
        [
            HandIndexer::new(&[2, 3]),
            HandIndexer::new(&[2, 4]),
            HandIndexer::new(&[2, 5]),
        ]
    })
}

/// Validate a street selector (0=flop, 1=turn, 2=river) and return its indexer.
fn indexer_for(street: usize) -> PyResult<&'static HandIndexer> {
    street_indexers().get(street).ok_or_else(|| {
        PyValueError::new_err(format!("street {street} out of 0..3 (flop/turn/river)"))
    })
}

/// Number of suit-isomorphic canonical situations for a street (0=flop, 1=turn,
/// 2=river): 1,286,792 / 13,960,050 / 123,156,254.
#[pyfunction]
fn canonical_index_size(street: usize) -> PyResult<u64> {
    Ok(indexer_for(street)?.size())
}

/// Number of cards in a full hand for a street (5/6/7).
#[pyfunction]
fn street_card_count(street: usize) -> PyResult<usize> {
    Ok(indexer_for(street)?.total_cards())
}

/// Dense index (0..size) of a concrete hand on a street. `cards` is the full hand
/// as 0..52 indices, hole first then board; any suit permutation maps to the same
/// index. Used at play time and by the correctness gates.
#[pyfunction]
fn canonical_index(street: usize, cards: Vec<u8>) -> PyResult<u64> {
    let idx = indexer_for(street)?;
    let expect = idx.total_cards();
    if cards.len() != expect {
        return Err(PyValueError::new_err(format!(
            "street {street} needs {expect} cards, got {}",
            cards.len()
        )));
    }
    let mut seen = 0u64;
    for &c in &cards {
        if c >= 52 {
            return Err(PyValueError::new_err(format!(
                "card index {c} out of 0..52"
            )));
        }
        if seen & (1u64 << c) != 0 {
            return Err(PyValueError::new_err(format!("duplicate card index {c}")));
        }
        seen |= 1u64 << c;
    }
    Ok(idx.index(&cards))
}

/// Canonical representative hands for a contiguous block of dense indices
/// `[start, start+count)` on a street, in parallel. Returns raw `bytes` of a flat
/// `count * street_card_count` row-major `u8` array (each row is one hand, hole
/// then board, ascending within hole and within board) — `bytes` so numpy can
/// wrap it without a per-element Python conversion, which matters for the 123 M
/// river rows. This is how the driver materializes situations for feature
/// extraction without ever building a hash set of all classes.
#[pyfunction]
fn canonical_unindex_batch<'py>(
    py: Python<'py>,
    street: usize,
    start: u64,
    count: u64,
) -> PyResult<Bound<'py, PyBytes>> {
    let idx = indexer_for(street)?;
    let size = idx.size();
    if start > size || count > size - start {
        return Err(PyValueError::new_err(format!(
            "block [{start}, {}) out of range 0..{size}",
            start + count
        )));
    }
    let tc = idx.total_cards();
    let out = py.detach(|| {
        let mut out = vec![0u8; count as usize * tc];
        out.par_chunks_mut(tc).enumerate().for_each(|(i, row)| {
            let cards = idx.unindex(start + i as u64);
            row.copy_from_slice(&cards);
        });
        out
    });
    Ok(PyBytes::new(py, &out))
}

// --- card-abstraction bucket tables (Phase 3c, B2) -------------------------

/// The loaded flop/turn/river bucket tables, addressed by the same
/// `canonical_index` the driver used. Constructed from a `data/abstraction`
/// directory; `bucket_of` reproduces `parciver.bucketing.bucket_of` exactly
/// (preflop = the 169 `preflop_index` classes), which is the Rust half the NLHE
/// game keys its information sets on.
#[pyclass(name = "Abstraction")]
struct PyAbstraction {
    inner: Arc<Abstraction>,
}

#[pymethods]
impl PyAbstraction {
    /// Load the bucket tables from `dir` (a `data/abstraction` directory holding
    /// `{flop,turn,river}_buckets.npy`). Raises if a table is missing, malformed,
    /// or the wrong length for its street.
    #[new]
    fn new(dir: &str) -> PyResult<Self> {
        let inner = Abstraction::load(Path::new(dir)).map_err(PyValueError::new_err)?;
        Ok(PyAbstraction {
            inner: Arc::new(inner),
        })
    }

    /// Bucket id of a `(hole, board)` situation: the preflop class `0..169` for an
    /// empty board, else the postflop bucket. `hole` is exactly 2 cards; `board`
    /// is 0/3/4/5 cards. Any suit permutation maps to the same bucket.
    fn bucket_of(&self, hole: Vec<u8>, board: Vec<u8>) -> PyResult<u32> {
        if hole.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "hole must be exactly 2 cards, got {}",
                hole.len()
            )));
        }
        if !matches!(board.len(), 0 | 3 | 4 | 5) {
            return Err(PyValueError::new_err(format!(
                "board must be 0, 3, 4, or 5 cards, got {}",
                board.len()
            )));
        }
        let mut seen = 0u64;
        for &c in hole.iter().chain(&board) {
            if c >= 52 {
                return Err(PyValueError::new_err(format!(
                    "card index {c} out of 0..52"
                )));
            }
            if seen & (1u64 << c) != 0 {
                return Err(PyValueError::new_err(format!("duplicate card index {c}")));
            }
            seen |= 1u64 << c;
        }
        let h = [Card::from_index(hole[0]), Card::from_index(hole[1])];
        let b: Vec<Card> = board.iter().map(|&i| Card::from_index(i)).collect();
        Ok(self.inner.bucket_of(h, &b))
    }
}

// --- exact best-response exploitability + small NLHE gate (Phase 3c, B3) ---

/// Exact per-player exploitability of `strategy` on Rust Kuhn, via the same
/// best-response engine used for the small NLHE gate. META-GATE: this must equal
/// the independent Python oracle to ~1e-9 (Kuhn/Leduc are the anchors that certify
/// the Rust best response before it is trusted on the small NLHE game).
#[pyfunction]
fn kuhn_exploitability(py: Python<'_>, strategy: HashMap<String, Vec<f64>>) -> f64 {
    py.detach(|| exploit::exploitability(&Kuhn::new(), &strategy, |_s| Option::<String>::None))
}

/// Exact per-player exploitability of `strategy` on Rust Leduc. See
/// `kuhn_exploitability`.
#[pyfunction]
fn leduc_exploitability(py: Python<'_>, strategy: HashMap<String, Vec<f64>>) -> f64 {
    py.detach(|| exploit::exploitability(&Leduc::new(), &strategy, |_s| Option::<String>::None))
}

/// Solve the small B3 cross-validation NLHE game (`NlheConfig::small`, a 7-card
/// deck / 2 bb / lean grid, on the lossless Exact card key) with full-traversal
/// CFR. Returns the average strategy as `{infoset_key: [prob, ...]}`.
#[pyfunction]
#[pyo3(signature = (iterations, variant = "dcfr"))]
fn solve_nlhe_small(
    py: Python<'_>,
    iterations: u64,
    variant: &str,
) -> PyResult<HashMap<String, Vec<f64>>> {
    let v = variant_by_name(variant)
        .ok_or_else(|| PyValueError::new_err(format!("unknown CFR variant: {variant:?}")))?;
    Ok(py.detach(|| {
        let game = Nlhe::new(NlheConfig::small());
        let mut solver = CFRSolver::new(v);
        solver.run(&game, iterations);
        solver.average_strategy()
    }))
}

/// Solve the small B3 NLHE game with external-sampling MCCFR. See
/// `solve_nlhe_small`; `variant` is "mccfr"/"mccfr+"/"linear", `seed` makes it
/// reproducible.
#[pyfunction]
#[pyo3(signature = (iterations, variant = "linear", seed = 0))]
fn solve_nlhe_small_mccfr(
    py: Python<'_>,
    iterations: u64,
    variant: &str,
    seed: u64,
) -> PyResult<HashMap<String, Vec<f64>>> {
    let v = mccfr::variant_by_name(variant)
        .ok_or_else(|| PyValueError::new_err(format!("unknown MCCFR variant: {variant:?}")))?;
    Ok(py.detach(|| {
        let game = Nlhe::new(NlheConfig::small());
        let mut solver = MCCFRSolver::new(v, seed);
        solver.run(&game, iterations);
        solver.average_strategy()
    }))
}

/// Train one sampled response against an immutable opponent on the exact small
/// NLHE game. This is the S3.2 response solver's load-bearing exact-control
/// bridge: its realized value can be compared directly with the certified exact
/// best response before the same algorithm is used inside the full abstraction.
#[pyfunction]
#[pyo3(signature = (opponent_strategy, responder, iterations, variant = "linear", seed = 0, threads = 1))]
fn nlhe_small_fixed_response(
    py: Python<'_>,
    opponent_strategy: HashMap<String, Vec<f64>>,
    responder: usize,
    iterations: u64,
    variant: &str,
    seed: u64,
    threads: usize,
) -> PyResult<HashMap<String, Vec<f64>>> {
    if responder >= 2 {
        return Err(PyValueError::new_err("responder must be 0 or 1"));
    }
    if iterations == 0 {
        return Err(PyValueError::new_err("iterations must be positive"));
    }
    let selected = mccfr::variant_by_name(variant)
        .ok_or_else(|| PyValueError::new_err(format!("unknown MCCFR variant: {variant:?}")))?;
    Ok(py.detach(move || {
        let game = Nlhe::new(NlheConfig::small());
        let mut solver = FixedOpponentMCCFR::new(selected, seed, responder);
        solver.set_threads(threads.max(1));
        solver.run(&game, &opponent_strategy, iterations);
        solver.average_strategy()
    }))
}

/// Exact per-player exploitability of `strategy` on the small B3 NLHE game. Uses
/// the state-signature memo (collapsing the engine's one-card-at-a-time dealing
/// orderings) so the full best response stays tractable.
#[pyfunction]
fn nlhe_small_exploitability(py: Python<'_>, strategy: HashMap<String, Vec<f64>>) -> f64 {
    py.detach(|| {
        let game = Nlhe::new(NlheConfig::small());
        exploit::exploitability(&game, &strategy, |s| Some(Nlhe::state_signature(s)))
    })
}

/// Number of information sets in the small B3 NLHE game — a structural size
/// cross-check (kept modest so full CFR + exact exploitability stay tractable).
#[pyfunction]
fn nlhe_small_infoset_count() -> usize {
    count_infosets(&Nlhe::new(NlheConfig::small()))
}

// --- exact exploitation-spine primitives on the small NLHE game (RMAPSS §40) ---
// Thin, additive, read-only exports mirroring the `kuhn_n_*` ones, so the Python
// H1 spine (`rmapss.step1`) can run on real (reduced) NLHE with EXACT controls:
// best-response value & STRATEGY vs a fixed opponent (the ceiling / blend target),
// realized on-policy value, and the Hero-observable opponent marginal. No change
// to `nlhe.rs`'s 2p betting core.

/// Exact best-response EV for `player` vs the fixed others in `strategy` — the
/// exploitation CEILING against a fixed opponent (the §28 control).
#[pyfunction]
fn nlhe_small_best_response_value(
    py: Python<'_>,
    strategy: HashMap<String, Vec<f64>>,
    player: usize,
) -> f64 {
    py.detach(|| {
        let game = Nlhe::new(NlheConfig::small());
        exploit::best_response_value(&game, &strategy, player, &|s| {
            Some(Nlhe::state_signature(s))
        })
    })
}

/// The exact best-response STRATEGY for `player` vs the fixed others in `strategy`
/// (one-hot per infoset). Realizing it reproduces `nlhe_small_best_response_value`;
/// the spine blends toward it.
#[pyfunction]
fn nlhe_small_best_response_strategy(
    py: Python<'_>,
    strategy: HashMap<String, Vec<f64>>,
    player: usize,
) -> HashMap<String, Vec<f64>> {
    py.detach(|| {
        let game = Nlhe::new(NlheConfig::small());
        exploit::best_response_strategy(&game, &strategy, player, &|s| {
            Some(Nlhe::state_signature(s))
        })
    })
}

/// Realized EV to `player` when ALL players follow `strategy` (missing infoset =>
/// uniform) — used to measure a Hero deviation's realized EV vs a fixed opponent.
#[pyfunction]
fn nlhe_small_on_policy_value(
    py: Python<'_>,
    strategy: HashMap<String, Vec<f64>>,
    player: usize,
) -> f64 {
    py.detach(|| {
        let game = Nlhe::new(NlheConfig::small());
        exploit::on_policy_value(&game, &strategy, player, |s| Some(Nlhe::state_signature(s)))
    })
}

/// Shared reach-weighted traversal for the Hero-observed opponent model on the
/// small NLHE game. Returns `(counts, key_to_obs)`: `counts[obs_key]` is the
/// reach-weighted action tally at each Hero-observable opponent node
/// (`observable_key` = street|board|history), and `key_to_obs` maps every
/// opponent infoset key to its observable context. Both the flat per-key marginal
/// export and the per-node (marginal + mass) export derive from this single walk,
/// so they can never drift. Mirrors `parciver.rmapss.step1.observe_marginal`.
fn nlhe_small_observed_counts(
    hero_strategy: &HashMap<String, Vec<f64>>,
    opp_strategy: &HashMap<String, Vec<f64>>,
    hero: usize,
) -> (HashMap<String, Vec<f64>>, HashMap<String, String>) {
    let game = Nlhe::new(NlheConfig::small());
    let mut counts: HashMap<String, Vec<f64>> = HashMap::new();
    let mut key_to_obs: HashMap<String, String> = HashMap::new();

    fn walk(
        game: &Nlhe,
        hero_strategy: &HashMap<String, Vec<f64>>,
        opp_strategy: &HashMap<String, Vec<f64>>,
        hero: usize,
        s: &NlheState,
        reach: f64,
        counts: &mut HashMap<String, Vec<f64>>,
        key_to_obs: &mut HashMap<String, String>,
    ) {
        if reach == 0.0 || game.is_terminal(s) {
            return;
        }
        if game.is_chance(s) {
            for (child, p) in game.chance_outcomes(s) {
                walk(
                    game,
                    hero_strategy,
                    opp_strategy,
                    hero,
                    &child,
                    reach * p,
                    counts,
                    key_to_obs,
                );
            }
            return;
        }
        let p = game.current_player(s);
        let n = game.num_actions(s);
        let key = game.infoset_key(s);
        let src = if p == hero {
            hero_strategy
        } else {
            opp_strategy
        };
        let probs = src
            .get(&key)
            .cloned()
            .unwrap_or_else(|| vec![1.0 / n as f64; n]);
        if p != hero {
            let obs = game.observable_key(s);
            let c = counts.entry(obs.clone()).or_insert_with(|| vec![0.0; n]);
            for i in 0..n {
                c[i] += reach * probs[i];
            }
            key_to_obs.entry(key).or_insert(obs);
        }
        for i in 0..n {
            if probs[i] > 0.0 {
                let child = game.next_state(s, i);
                walk(
                    game,
                    hero_strategy,
                    opp_strategy,
                    hero,
                    &child,
                    reach * probs[i],
                    counts,
                    key_to_obs,
                );
            }
        }
    }

    walk(
        &game,
        hero_strategy,
        opp_strategy,
        hero,
        &game.initial_state(),
        1.0,
        &mut counts,
        &mut key_to_obs,
    );
    (counts, key_to_obs)
}

/// The Hero-OBSERVED opponent marginal on the small NLHE game: with Hero (=`hero`)
/// playing `hero_strategy` and the opponent playing `opp_strategy`, walk the tree
/// reach-weighted and return a FLAT opponent model mapping EACH opponent infoset
/// key to the action marginal of its Hero-observable context (`observable_key`:
/// street|board|history) — what Hero can estimate from watching actions without
/// seeing the opponent's hole. Mirrors `parciver.rmapss.step1.observe_marginal`.
#[pyfunction]
fn nlhe_small_observed_marginal(
    py: Python<'_>,
    hero_strategy: HashMap<String, Vec<f64>>,
    opp_strategy: HashMap<String, Vec<f64>>,
    hero: usize,
) -> HashMap<String, Vec<f64>> {
    py.detach(|| {
        let (counts, key_to_obs) = nlhe_small_observed_counts(&hero_strategy, &opp_strategy, hero);
        let mut marg: HashMap<String, Vec<f64>> = HashMap::new();
        for (obs, c) in &counts {
            let total: f64 = c.iter().sum();
            let v = if total > 0.0 {
                c.iter().map(|x| x / total).collect()
            } else {
                vec![1.0 / c.len() as f64; c.len()]
            };
            marg.insert(obs.clone(), v);
        }
        let mut flat: HashMap<String, Vec<f64>> = HashMap::new();
        for (key, obs) in &key_to_obs {
            if let Some(v) = marg.get(obs) {
                flat.insert(key.clone(), v.clone());
            }
        }
        flat
    })
}

/// The Hero-observed opponent marginal returned per OBSERVABLE NODE rather than
/// flattened per opponent infoset: `(marg_by_obs, mass_by_obs)`, both keyed by
/// `observable_key`, where `mass_by_obs` is the reach mass Hero encounters at that
/// node. Feeds `step1.deviation_signal` (the reach-weighted L1 between the observed
/// and equilibrium marginals) — the confidence signal that schedules α. Same walk
/// as `nlhe_small_observed_marginal`, exposing the mass the flat form discards.
#[pyfunction]
fn nlhe_small_observed_marginal_nodes(
    py: Python<'_>,
    hero_strategy: HashMap<String, Vec<f64>>,
    opp_strategy: HashMap<String, Vec<f64>>,
    hero: usize,
) -> (HashMap<String, Vec<f64>>, HashMap<String, f64>) {
    py.detach(|| {
        let (counts, _key_to_obs) = nlhe_small_observed_counts(&hero_strategy, &opp_strategy, hero);
        let mut marg: HashMap<String, Vec<f64>> = HashMap::new();
        let mut mass: HashMap<String, f64> = HashMap::new();
        for (obs, c) in &counts {
            let total: f64 = c.iter().sum();
            mass.insert(obs.clone(), total);
            let v = if total > 0.0 {
                c.iter().map(|x| x / total).collect()
            } else {
                vec![1.0 / c.len() as f64; c.len()]
            };
            marg.insert(obs.clone(), v);
        }
        (marg, mass)
    })
}

/// Flat model, node marginals and node reach masses from ONE observation walk.
/// The separate exports remain available as parity controls and for old callers.
#[pyfunction]
fn nlhe_small_observed_model(
    py: Python<'_>,
    hero_strategy: HashMap<String, Vec<f64>>,
    opp_strategy: HashMap<String, Vec<f64>>,
    hero: usize,
) -> (
    HashMap<String, Vec<f64>>,
    HashMap<String, Vec<f64>>,
    HashMap<String, f64>,
) {
    py.detach(|| {
        let (counts, key_to_obs) = nlhe_small_observed_counts(&hero_strategy, &opp_strategy, hero);
        let mut marg = HashMap::new();
        let mut mass = HashMap::new();
        for (obs, c) in counts {
            let total: f64 = c.iter().sum();
            let v = if total > 0.0 {
                c.iter().map(|x| x / total).collect()
            } else {
                vec![1.0 / c.len() as f64; c.len()]
            };
            mass.insert(obs.clone(), total);
            marg.insert(obs, v);
        }
        let flat = key_to_obs
            .into_iter()
            .map(|(key, obs)| (key, marg[&obs].clone()))
            .collect();
        (flat, marg, mass)
    })
}

/// The infoset STRUCTURE of the small NLHE game: each infoset key mapped to
/// `(acting player, [action category, ...])`, category ∈
/// {"fold","check","call","bet","allin"} and index-aligned with the legal-action
/// list / `next_state`. Lets the Python spine split Hero(0) vs opponent(1) keys and
/// build behavioural archetypes (a nit folds, a station calls, a maniac jams)
/// without hard-coding action indices. Derived from `labeled_actions`, so it can
/// never drift from the betting engine. Read-only; no change to the 2p core.
#[pyfunction]
fn nlhe_small_infoset_structure() -> HashMap<String, (usize, Vec<String>)> {
    fn cat(l: ActLabel) -> String {
        match l {
            ActLabel::Fold => "fold",
            ActLabel::Check => "check",
            ActLabel::Call => "call",
            ActLabel::Bet(_) => "bet",
            ActLabel::AllIn => "allin",
        }
        .to_string()
    }
    fn walk(game: &Nlhe, s: &NlheState, out: &mut HashMap<String, (usize, Vec<String>)>) {
        if game.is_terminal(s) {
            return;
        }
        if game.is_chance(s) {
            for (child, _) in game.chance_outcomes(s) {
                walk(game, &child, out);
            }
            return;
        }
        let key = game.infoset_key(s);
        if !out.contains_key(&key) {
            let cats = game.labeled_actions(s).iter().copied().map(cat).collect();
            out.insert(key, (game.current_player(s), cats));
        }
        for i in 0..game.num_actions(s) {
            walk(game, &game.next_state(s, i), out);
        }
    }
    let game = Nlhe::new(NlheConfig::small());
    let mut out: HashMap<String, (usize, Vec<String>)> = HashMap::new();
    walk(&game, &game.initial_state(), &mut out);
    out
}

// --- full 100bb HU blueprint solver (Phase 3c, B4) -------------------------

/// Incremental external-sampling MCCFR on the REAL 52-card heads-up game
/// ([`NlheConfig::pluribus_100bb`]) keyed on the Phase-3b bucket abstraction — the
/// blueprint solve. Exposed as a stateful object (not a one-shot function) so the
/// Python trainer can run it in chunks and read telemetry between chunks:
/// throughput, the average-regret convergence proxy, infoset coverage, and the 169
/// preflop opening nodes for the live range grid. Exact exploitability is
/// intractable on this game (that is why B3 built the small-game bridge), so
/// convergence is judged by the proxy + range-sanity, watched live on the dashboard.
#[pyclass]
struct Blueprint {
    game: Nlhe,
    solver: MCCFRSolver,
    /// The bet grid (`NlheConfig::grid`), stored in every file this writes.
    grid: String,
    /// The bucket tables the game is keyed on (the game holds them too).
    abs: Arc<Abstraction>,
    /// Their fingerprint (`Abstraction::fingerprint`), stored in every
    /// checkpoint this writes: computed at the first save (a resume has it).
    abstraction: std::sync::OnceLock<u64>,
}

/// The bet grid named `name` (see `NlheConfig::grid`), or a Python error.
fn grid_config(name: &str) -> PyResult<NlheConfig> {
    NlheConfig::grid(name).ok_or_else(|| {
        PyValueError::new_err(format!("unknown bet grid {name:?} (one of {:?})", NlheConfig::GRIDS))
    })
}

/// The grid a saved file names; files from before the tag existed are "100bb".
fn grid_name(tag: &str) -> String {
    if tag.is_empty() { "100bb".to_string() } else { tag.to_string() }
}

/// Resolve a requested thread count: `0` means "use every logical core" (queried
/// from the OS, falling back to 1 if unavailable); any other value is passed
/// through. Keeps the "just use the whole box" default one integer away from an
/// explicit count for the speedup sweep.
fn resolve_threads(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    }
}

/// Collapse an opening-node action mix `[fold, call, raise, (all-in)…]` into the
/// grid's `[fold, call, raise]` (everything past call folds into "raise"). Robust
/// to the exact raise/all-in action count.
fn collapse_open(p: &[f64]) -> Vec<f64> {
    let fold = p.first().copied().unwrap_or(0.0);
    let call = p.get(1).copied().unwrap_or(0.0);
    let raise: f64 = p.iter().skip(2).sum();
    vec![fold, call, raise]
}

#[pymethods]
impl Blueprint {
    /// Build the blueprint solver: load the bucket abstraction from
    /// `abstraction_dir`, wire it into the 100bb HU game, and create an MCCFR solver
    /// (`variant` = "linear"/"mccfr"/"mccfr+", `seed` reproducible). `threads` sets
    /// the parallel worker count for the solve; `0` (the default) uses every logical
    /// core, `1` is the deterministic serial path.
    #[new]
    #[pyo3(signature = (abstraction_dir, variant = "linear", seed = 0, threads = 0, grid = "100bb"))]
    fn new(abstraction_dir: &str, variant: &str, seed: u64, threads: usize, grid: &str) -> PyResult<Self> {
        let v = mccfr::variant_by_name(variant)
            .ok_or_else(|| PyValueError::new_err(format!("unknown MCCFR variant: {variant:?}")))?;
        let cfg = grid_config(grid)?;
        let abs = Arc::new(Abstraction::load(Path::new(abstraction_dir)).map_err(PyValueError::new_err)?);
        let game = Nlhe::with_abstraction(cfg, Arc::clone(&abs));
        let mut solver = MCCFRSolver::new(v, seed);
        solver.set_threads(resolve_threads(threads));
        Ok(Blueprint { game, solver, grid: grid.to_string(), abs, abstraction: Default::default() })
    }

    /// Resume a solve from a full-state checkpoint written by
    /// [`Blueprint::save_checkpoint`]: rebuild the game from `abstraction_dir` and
    /// restore the solver's regret/strategy accumulators, iteration count, and seed
    /// from `checkpoint_path`. `threads` is applied fresh (it is a runtime knob, not
    /// solve state). This is what makes a multi-day 100bb solve restart-safe.
    /// Refuses a checkpoint trained on other bucket tables (one that records
    /// them: version 3, see `MCCFRSolver::save_state_tagged`).
    #[staticmethod]
    #[pyo3(signature = (abstraction_dir, checkpoint_path, threads = 0))]
    fn resume(abstraction_dir: &str, checkpoint_path: &str, threads: usize) -> PyResult<Self> {
        let abs = Arc::new(Abstraction::load(Path::new(abstraction_dir)).map_err(PyValueError::new_err)?);
        let abstraction = abs.fingerprint();
        let (mut solver, tag) = MCCFRSolver::load_state_tagged(Path::new(checkpoint_path), Some(abstraction))
            .map_err(|e| PyValueError::new_err(format!("blueprint resume failed: {e}")))?;
        // The checkpoint names its grid (untagged ones are "100bb").
        let grid = grid_name(&tag);
        let cfg = grid_config(&grid)?;
        let game = Nlhe::with_abstraction(cfg, Arc::clone(&abs));
        solver.set_threads(resolve_threads(threads));
        Ok(Blueprint { game, solver, grid, abs, abstraction: abstraction.into() })
    }

    /// The bet grid this solve plays (`NlheConfig::grid`).
    #[getter]
    fn grid(&self) -> &str {
        &self.grid
    }

    /// The MCCFR variant this solve runs ("linear", "mccfr" or "mccfr+"); a
    /// resumed solve runs its checkpoint's, whatever was asked for.
    #[getter]
    fn variant(&self) -> &'static str {
        self.solver.variant().name()
    }

    /// Run `iterations` more MCCFR iterations (GIL released), accumulating into the
    /// solver. Call repeatedly; the sampling stream is continuous across calls.
    fn run(&mut self, py: Python<'_>, iterations: u64) {
        let game = &self.game;
        let solver = &mut self.solver;
        py.detach(move || solver.run(game, iterations));
    }

    /// Total iterations run so far.
    fn iterations(&self) -> u64 {
        self.solver.iterations()
    }

    /// Number of information sets discovered so far (coverage).
    fn num_infosets(&self) -> usize {
        self.solver.num_infosets()
    }

    /// The variant-normalized average-regret convergence proxy (trends to 0).
    fn avg_regret(&self, py: Python<'_>) -> f64 {
        py.detach(|| self.solver.avg_regret())
    }

    /// Infoset counts by street `[P, F, T, R]`.
    fn infosets_by_street(&self, py: Python<'_>) -> [u64; 4] {
        py.detach(|| self.solver.infosets_by_street())
    }

    /// The 13x13 preflop range grid: 169 rows of `[fold, call, raise]`, one per
    /// canonical starting hand (`preflop_index` order), read from the SB opening
    /// infosets `P|<class>|`. An unvisited hand is `[0, 0, 0]` (the contract's empty
    /// cell). This is the iconic live visualization the dashboard animates.
    fn preflop_range_grid(&self, py: Python<'_>) -> Vec<Vec<f64>> {
        py.detach(|| {
            (0..169u32)
                .map(|h| {
                    let key = format!("P|{h}|");
                    match self.solver.average_strategy_for(&key) {
                        Some(p) => collapse_open(&p),
                        None => vec![0.0, 0.0, 0.0],
                    }
                })
                .collect()
        })
    }

    /// Persist the current average strategy (the blueprint) to `path` as a compact
    /// binary file. This is the durable PLAY artifact (normalized average, f32); it
    /// is NOT resumable — use [`Blueprint::save_checkpoint`] for that.
    fn save(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.solver.save_average_strategy_tagged(Path::new(path), Some(&self.grid)))
            .map_err(|e| PyValueError::new_err(format!("blueprint save failed: {e}")))
    }

    /// Persist the FULL solver state (per-infoset f64 regret + strategy sums, plus
    /// iteration count and seed) to `path`, so the solve can be resumed exactly with
    /// [`Blueprint::resume`]. Larger than [`Blueprint::save`]; the trainer writes it
    /// periodically for restart safety on multi-day runs.
    fn save_checkpoint(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| {
            let abstraction = *self.abstraction.get_or_init(|| self.abs.fingerprint());
            self.solver.save_state_tagged(Path::new(path), Some((&self.grid, abstraction)))
        })
            .map_err(|e| PyValueError::new_err(format!("blueprint checkpoint failed: {e}")))
    }

    /// The number of parallel worker threads this solve is using.
    fn threads(&self) -> usize {
        self.solver.threads()
    }
}

/// Read-only analysis of a SAVED blueprint, for the dashboard's three summary panels
/// (action-frequency, bet-sizing, head-to-head). Loads the abstraction ONCE, then
/// `analyze(blueprint_path)` reads a `blueprint.bin` snapshot the trainer wrote to
/// disk and returns the panel payloads. It never touches a running solve — the
/// sidecar constructs one of these and calls `analyze` on a timer, so the three
/// panels can be filled WITHOUT restarting or slowing the trainer.
#[pyclass]
struct BlueprintAnalyzer {
    abs: Arc<Abstraction>,
}

/// A validated, read-only saved average strategy wired to the game of the bet
/// grid it names (untagged files: the 100bb grid).
/// Unlike [`Blueprint`], this object has no regrets, RNG state or training method:
/// it can only report metadata and play Monte-Carlo matches against another saved
/// policy. S3.2 builds its paired evaluator on this non-mutating boundary.
#[pyclass]
struct SavedBlueprint {
    game: Arc<Nlhe>,
    strategy: Arc<FxHashMap<Box<[u8]>, Vec<f32>>>,
    source_path: String,
    grid: String,
}

/// Immutable S3.2 policy composition. It can wrap a saved tabular blueprint, a
/// locked leak transform, a response, a blend, or a fitted public model without
/// materializing transformed copies of the 13M-row anchor.
#[pyclass]
struct S32Policy {
    game: Arc<Nlhe>,
    policy: s32::Policy,
    label: String,
}

/// Public-only sampled action counts. Keys contain public board/history/position;
/// private holes and abstraction buckets never enter this object.
#[pyclass]
struct S32Observation {
    game: Arc<Nlhe>,
    data: Arc<s32::Observation>,
}

/// Play-time agent: a real heads-up hand in, a legal real action out. See
/// `engine/src/play.rs` for the translation and fallback rules.
#[pyclass]
struct PlayAgent {
    agent: Arc<play::Agent>,
}

/// A Python callable as the search's subgame solver (`PlayAgent.with_search`).
struct PySubgameSolver(Py<PyAny>);

impl subgame::SubgameSolver for PySubgameSolver {
    fn solve(&self, job: &search::FlatSubgame) -> Result<Vec<Vec<f64>>, String> {
        Python::attach(|py| -> PyResult<Vec<Vec<f64>>> {
            let d = PyDict::new(py);
            d.set_item("board", job.board.iter().map(|&c| u32::from(c)).collect::<Vec<u32>>())?;
            d.set_item("kind", job.kind.iter().map(|&k| u32::from(k)).collect::<Vec<u32>>())?;
            d.set_item("player", &job.player)?;
            d.set_item("slot", &job.slot)?;
            d.set_item("stake", job.stake.iter().map(|s| (s[0], s[1])).collect::<Vec<_>>())?;
            d.set_item("total", (job.total[0], job.total[1]))?;
            d.set_item("view", &job.view)?;
            d.set_item("child_start", &job.child_start)?;
            d.set_item("children", &job.children)?;
            d.set_item("norm", job.norm)?;
            d.set_item("strength", &job.strength)?;
            d.set_item("alive", &job.alive)?;
            d.set_item("ranges", (&job.ranges[0], &job.ranges[1]))?;
            d.set_item("frozen", job.frozen.clone())?;
            d.set_item("iters", job.iters)?;
            d.set_item("want", &job.want)?;
            self.0.call1(py, (d,))?.extract::<Vec<Vec<f64>>>(py)
        })
        .map_err(|e| format!("external subgame solver: {e}"))
    }
}

fn parse_real_act(kind: &str, amount: i64) -> PyResult<nlhe::RealAct> {
    use nlhe::RealAct;
    Ok(match kind {
        "f" | "fold" => RealAct::Fold,
        "x" | "check" => RealAct::Check,
        "c" | "call" => RealAct::Call,
        "b" | "r" | "bet" | "raise" => RealAct::Add(amount),
        "a" | "allin" => RealAct::Add(play::ALL_IN),
        other => return Err(PyValueError::new_err(format!("unknown action kind {other:?}"))),
    })
}

#[pymethods]
impl PlayAgent {
    /// An agent over an in-memory strategy on a table-free game: "small" (the B3
    /// exact game) or "100bb_exact" (the real grid keyed on exact canonical cards).
    /// For tests and demos; the real bot comes from `SavedBlueprint.play_agent()`.
    #[staticmethod]
    #[pyo3(signature = (strategy, game = "small"))]
    fn from_strategy(strategy: HashMap<String, Vec<f64>>, game: &str) -> PyResult<Self> {
        let cfg = match game {
            "small" => NlheConfig::small(),
            "100bb_exact" => NlheConfig::pluribus_100bb(),
            "200bb_exact" => NlheConfig::deep_200bb(),
            other => return Err(PyValueError::new_err(format!("unknown game {other:?}"))),
        };
        let table: play::Strategy = strategy
            .into_iter()
            .map(|(k, row)| (k.into_bytes().into_boxed_slice(), row.into_iter().map(|p| p as f32).collect()))
            .collect();
        Ok(PlayAgent {
            agent: Arc::new(play::Agent::new(Arc::new(Nlhe::new(cfg)), Arc::new(table))),
        })
    }

    /// A uniform-random policy over a real bet grid (exact canonical cards): a
    /// table-free agent for tests and demos of the play layer and search.
    #[staticmethod]
    #[pyo3(signature = (grid = "100bb"))]
    fn uniform(grid: &str) -> PyResult<Self> {
        let game = Arc::new(Nlhe::new(grid_config(grid)?));
        Ok(PlayAgent { agent: Arc::new(play::Agent::from_policy(game, s32::Policy::uniform())) })
    }

    /// The same policy with real-time search: every river decision (and every
    /// turn decision when `from_turn`, every flop decision when `from_flop`) is
    /// re-solved as an exact-card subgame of the current round, with both ranges
    /// from Bayes over the hand's actions (source "search"). Preflop stays on the
    /// policy. A flop subgame is the flop round alone: its leaves need `solver`
    /// (a value net, FLOPSEARCH.md) unless `flop_checkdown`, which solves it with
    /// the built-in solver and values each leaf as a check-down to the river. The searcher's
    /// own range is its exact one (its real decisions, search included) unless
    /// `exact_own_range=False`, which reproduces the first version's
    /// policy-modelled own range. The round is solved from its start (the
    /// opponent's actions this round solved, the searcher's own frozen) unless
    /// `solve_round=False`, which reproduces the earlier versions' forced round.
    ///
    /// `solver`: a Python callable that solves each subgame instead of the
    /// built-in CPU solver (e.g. `parciver.gpu_fused.SubgameGpu()`). It gets a
    /// dict describing the subgame (`search::FlatSubgame`'s fields) and returns
    /// the average strategy of each slot in `want`. Matches then call it from
    /// their worker threads (one at a time, under the GIL). Where it fails (an
    /// exception or an answer the engine refuses), a turn or river subgame is
    /// solved by the built-in solver instead unless `rescue=False`; a failed
    /// flop search, or any where `rescue=False`, leaves the decision to the
    /// policy. A failed search is tried once per public state.
    #[pyo3(signature = (from_turn = false, river_iters = 200, turn_iters = 100, exact_own_range = true, solve_round = true, solver = None, from_flop = false, flop_iters = 100, flop_checkdown = false, rescue = true))]
    #[allow(clippy::too_many_arguments)]
    fn with_search(
        &self,
        from_turn: bool,
        river_iters: u32,
        turn_iters: u32,
        exact_own_range: bool,
        solve_round: bool,
        solver: Option<Py<PyAny>>,
        from_flop: bool,
        flop_iters: u32,
        flop_checkdown: bool,
        rescue: bool,
    ) -> PyResult<Self> {
        if from_flop && solver.is_none() && !flop_checkdown {
            return Err(PyValueError::new_err("from_flop needs a solver (its leaves need the value net) or flop_checkdown=True"));
        }
        let external = solver.map(|s| subgame::External(Arc::new(PySubgameSolver(s))));
        let cfg = subgame::SearchCfg {
            from_flop,
            flop_iters,
            flop_checkdown,
            from_turn,
            river_iters,
            turn_iters,
            exact_own_range,
            solve_round,
            external,
            rescue,
        };
        Ok(PlayAgent { agent: Arc::new(self.agent.with_search(cfg)) })
    }

    /// The same agent reading a real bet or raise above the largest grid size
    /// per `translation`: "snap" (the default: as that size unless it commits at
    /// least half the bettor's stack) or "standin" (always between that size
    /// and all-in; an all-in reading of a bet that is not all-in only stands in
    /// for the response, and unless that is a fold the hand continues as if the
    /// bet were the largest size; TRANSLATE.md).
    fn with_translation(&self, translation: &str) -> PyResult<Self> {
        let o = play::Oversize::parse(translation)
            .ok_or_else(|| PyValueError::new_err("translation must be \"snap\" or \"standin\""))?;
        Ok(PlayAgent { agent: Arc::new(self.agent.with_oversize(o)) })
    }

    /// "snap" or "standin" (`with_translation`).
    #[getter]
    fn translation(&self) -> &'static str {
        match self.agent.oversize() {
            play::Oversize::Snap => "snap",
            play::Oversize::Standin => "standin",
        }
    }

    /// The same agent reading a real preflop open (the small blind's first
    /// raise) below the smallest grid size per `reading`: "snap" (the default:
    /// as that size) or "limp" (records/preflop-open F2: between the limp and
    /// that size, read as the limp with probability (1 - x) / (1 + x) against
    /// the pot-size open, x the open's pot fraction; under a limp reading the
    /// big blind's real call of the open is its abstract check).
    fn with_open_reading(&self, reading: &str) -> PyResult<Self> {
        let o = play::OpenBelow::parse(reading).ok_or_else(|| PyValueError::new_err("reading must be \"snap\" or \"limp\""))?;
        Ok(PlayAgent { agent: Arc::new(self.agent.with_open_below(o)) })
    }

    /// "snap" or "limp" (`with_open_reading`).
    #[getter]
    fn open_reading(&self) -> &'static str {
        match self.agent.open_below() {
            play::OpenBelow::Snap => "snap",
            play::OpenBelow::Limp => "limp",
        }
    }

    /// Decide one action. `actions` is every action so far in order, as
    /// `(kind, chips_added)`: kind "f"/"x"/"c" (amount ignored), "b" (bet/raise:
    /// the chips ADDED by this action), "a" (all-in). Seat 0 is the button/SB.
    /// Returns a dict with the action ("fold"/"check"/"call"/"bet"), `add` (chips to
    /// put in; for "bet" the total added this action), `all_in`, `source`
    /// ("blueprint"/"missing_key"/"off_tree"/"search"), the abstract `key`, `labels`/`probs`
    /// at that key, the translated `abstract_history` and any `off_tree_reason` (on a
    /// "search" decision: why the built-in solver solved it in place of the external one,
    /// and/or that a failed search decided an earlier decision of the agent's in the hand
    /// that the search read).
    #[pyo3(signature = (hero, hole, board, actions, sb, bb, stacks, seed = 0))]
    #[allow(clippy::too_many_arguments)]
    fn decide(
        &self,
        py: Python<'_>,
        hero: usize,
        hole: (u8, u8),
        board: Vec<u8>,
        actions: Vec<(String, i64)>,
        sb: i64,
        bb: i64,
        stacks: (i64, i64),
        seed: u64,
    ) -> PyResult<Py<PyDict>> {
        let acts = actions
            .iter()
            .map(|(k, a)| parse_real_act(k, *a))
            .collect::<PyResult<Vec<_>>>()?;
        let spec = play::HandSpec {
            hero,
            hole: [hole.0, hole.1],
            board,
            sb,
            bb,
            stacks: [stacks.0, stacks.1],
            actions: acts,
            seed,
        };
        // No GIL while it runs (as `options`): a searching agent's solver takes it
        // back, from whichever thread calls it.
        let d = py.detach(|| self.agent.decide(&spec)).map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        let kind = match d.action {
            nlhe::RealAct::Fold => "fold",
            nlhe::RealAct::Check => "check",
            nlhe::RealAct::Call => "call",
            nlhe::RealAct::Add(_) => "bet",
        };
        out.set_item("action", kind)?;
        out.set_item("add", d.add)?;
        out.set_item("all_in", d.all_in)?;
        out.set_item("source", d.source.as_str())?;
        out.set_item("key", d.key)?;
        out.set_item("labels", d.labels)?;
        out.set_item("probs", d.probs)?;
        out.set_item("abstract_history", d.abstract_history)?;
        out.set_item("off_tree_reason", d.off_tree_reason)?;
        Ok(out.into())
    }

    /// Every holding's decision distribution at one public state (`decide`'s
    /// arguments, with `holes` for the hole cards): per holding, a list of
    /// `(kind, add, prob)`, kind "fold"/"check"/"call"/"bet" and `add` the chips
    /// a bet adds. Where the agent searches it is the solve's strategy (one
    /// solve for all holdings), else the blueprint's (off-grid sizes read with
    /// `seed`'s translation). Where a failed search decided them it raises
    /// RuntimeError: they are then the built-in solver's or the blueprint's,
    /// not the search's (`options_failure` returns them with the failure). So
    /// does a search that read an earlier decision of the agent's in the hand
    /// (for its range, or its strategy frozen this round) that a failed search
    /// decided.
    #[pyo3(signature = (hero, board, actions, sb, bb, stacks, holes, seed = 0))]
    #[allow(clippy::too_many_arguments)]
    fn options(
        &self,
        py: Python<'_>,
        hero: usize,
        board: Vec<u8>,
        actions: Vec<(String, i64)>,
        sb: i64,
        bb: i64,
        stacks: (i64, i64),
        holes: Vec<(u8, u8)>,
        seed: u64,
    ) -> PyResult<Vec<Vec<(String, i64, f64)>>> {
        match self.options_failure(py, hero, board, actions, sb, bb, stacks, holes, seed)? {
            (opts, None) => Ok(opts),
            (_, Some(k)) => Err(PyRuntimeError::new_err(
                [
                    "external solver failed, built-in solver used (here or at an earlier decision in the hand)",
                    "search failed, blueprint used (here or at an earlier decision in the hand)",
                ][k],
            )),
        }
    }

    /// `options`, and whether a failed search decided them, as an index into
    /// `match_against`'s `a_search_failed`: 0 the built-in solver used in
    /// place of the external one, 1 the blueprint used, here or at an earlier
    /// decision of the agent's in the hand that the search read; `None` where
    /// no search failed. Returns `(options, failed)`.
    #[pyo3(signature = (hero, board, actions, sb, bb, stacks, holes, seed = 0))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn options_failure(
        &self,
        py: Python<'_>,
        hero: usize,
        board: Vec<u8>,
        actions: Vec<(String, i64)>,
        sb: i64,
        bb: i64,
        stacks: (i64, i64),
        holes: Vec<(u8, u8)>,
        seed: u64,
    ) -> PyResult<(Vec<Vec<(String, i64, f64)>>, Option<usize>)> {
        let acts = actions
            .iter()
            .map(|(k, a)| parse_real_act(k, *a))
            .collect::<PyResult<Vec<_>>>()?;
        let holes: Vec<[u8; 2]> = holes.into_iter().map(|(a, b)| [a, b]).collect();
        let Some(&first) = holes.first() else { return Ok((Vec::new(), None)) };
        let spec = play::HandSpec { hero, hole: first, board, sb, bb, stacks: [stacks.0, stacks.1], actions: acts, seed };
        // No GIL while it runs: a searching agent's solver takes it back.
        let (opts, failed) = py.detach(|| self.agent.options_many_failure(&spec, &holes)).map_err(PyValueError::new_err)?;
        let opts = opts
            .into_iter()
            .map(|o| {
                o.into_iter()
                    .map(|(a, p)| match a {
                        nlhe::RealAct::Fold => ("fold".to_string(), 0, p),
                        nlhe::RealAct::Check => ("check".to_string(), 0, p),
                        nlhe::RealAct::Call => ("call".to_string(), 0, p),
                        nlhe::RealAct::Add(d) => ("bet".to_string(), d, p),
                    })
                    .collect()
            })
            .collect();
        Ok((opts, failed))
    }

    /// Head-to-head match on the real-chip engine. `opponent` is "agent" (the
    /// `other` PlayAgent, default this one), "perturbed" (`other` or this agent
    /// with bet sizes scaled by `perturb`), "realistic" (`other` or this agent
    /// with each size redrawn from realistic menus), "station", "pot_raiser",
    /// "min_raiser" or "random". With `mirror` every deal is played in both seats
    /// (duplicate); otherwise this agent sits in `seat`. Returns this agent's
    /// `bb_per_100` with its standard error `se`, `hands`, both sides'
    /// decision-source counts `[blueprint, missing_key, off_tree, search]`
    /// (`a_sources`, `b_sources`) and how many of their decisions a failed
    /// search reached (here or at an earlier decision in the hand that the
    /// search read), `[built-in solver used in place of the external one,
    /// blueprint used]` (`a_search_failed`, `b_search_failed`). `first_deal` plays deals `first_deal..first_deal + deals` of the match
    /// defined by `seed`, so a long match can run in chunks with results
    /// identical to one call. With `luck` (real 52-card games), also
    /// `per_deal_bb_cv`: `per_deal_bb` less the luck of the board deals
    /// (VARRED.md), an unbiased score with smaller variance. With `aivat`
    /// (both bots unmodified agents: `opponent="agent"`), also
    /// `per_deal_bb_cv` and `per_deal_bb_aivat`: that less the luck of both
    /// bots' sampled actions as well (records/rating-live/PREREG.md).
    #[pyo3(signature = (opponent, deals, sb, bb, stack, mirror = true, seat = 0, seed = 0, threads = 0, other = None, perturb = 1.0, first_deal = 0, luck = false, aivat = false))]
    #[allow(clippy::too_many_arguments)]
    fn match_against(
        &self,
        py: Python<'_>,
        opponent: &str,
        deals: u64,
        sb: i64,
        bb: i64,
        stack: i64,
        mirror: bool,
        seat: usize,
        seed: u64,
        threads: usize,
        other: Option<PyRef<'_, PlayAgent>>,
        perturb: f64,
        first_deal: u64,
        luck: bool,
        aivat: bool,
    ) -> PyResult<Py<PyDict>> {
        let opp_agent = other.map(|o| Arc::clone(&o.agent)).unwrap_or_else(|| Arc::clone(&self.agent));
        let b = match opponent {
            "agent" => arena::Bot::Agent(opp_agent),
            "perturbed" => arena::Bot::Perturbed(opp_agent, perturb),
            "realistic" => arena::Bot::Realistic(opp_agent),
            "station" => arena::Bot::Station,
            "pot_raiser" => arena::Bot::PotRaiser,
            "min_raiser" => arena::Bot::MinRaiser,
            "random" => arena::Bot::Random,
            o => return Err(PyValueError::new_err(format!("unknown opponent {o:?}"))),
        };
        if seat > 1 || sb <= 0 || bb < sb || stack <= bb || deals < 2 {
            return Err(PyValueError::new_err("need seat in {0,1}, 0 < sb <= bb < stack, deals >= 2"));
        }
        let a = arena::Bot::Agent(Arc::clone(&self.agent));
        let table = arena::Table::new(self.agent.cfg(), sb, bb, stack);
        let run = || arena::run_match(&table, &a, &b, first_deal, deals, mirror, seat, seed, luck, aivat);
        let stats = py
            .detach(|| {
                if threads == 0 {
                    run()
                } else {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .map_err(|e| e.to_string())?
                        .install(run)
                }
            })
            .map_err(PyValueError::new_err)?;
        let (mean, se) = stats.bb_per_100(if mirror { 2.0 } else { 1.0 });
        let out = PyDict::new(py);
        out.set_item("opponent", b.name())?;
        out.set_item("bb_per_100", mean)?;
        out.set_item("se", se)?;
        out.set_item("hands", stats.hands)?;
        out.set_item("a_sources", stats.a_sources.to_vec())?;
        out.set_item("b_sources", stats.b_sources.to_vec())?;
        out.set_item("a_search_failed", stats.a_search_failed.to_vec())?;
        out.set_item("b_search_failed", stats.b_search_failed.to_vec())?;
        out.set_item("per_deal_bb", stats.per_deal_bb)?;
        if luck || aivat {
            out.set_item("per_deal_bb_cv", stats.per_deal_bb_cv)?;
        }
        if aivat {
            out.set_item("per_deal_bb_aivat", stats.per_deal_bb_aivat)?;
            out.set_item("aivat_mass_error", stats.aivat_mass_error)?;
        }
        Ok(out.into())
    }

    /// Local best response against this agent: a lower bound on its
    /// exploitability (see engine/src/lbr.rs). Returns LBR's `bb_per_100` against
    /// the agent with standard error `se` (mirrored seats unless `mirror=False`,
    /// then LBR sits in `seat`). `first_deal` as in `match_against`.
    /// `lbr_sizes="grid"` restricts LBR's bets to the agent's own grid sizes;
    /// `lbr_peek=False` stops LBR from reading how the agent's seeded
    /// translation mapped its bets (diagnostics; the defaults are the
    /// standard LBR). `per_deal_preflop_bb` is the part of each deal's
    /// `per_deal_bb` from hands whose betting ended before the flop: LBR only
    /// checks or calls preflop, so that part is not exploitation (e.g. calling
    /// a preflop shove with any two cards); the rest measures postflop play.
    /// `search_failed` counts the agent's decisions a failed search reached,
    /// as `match_against`'s `a_search_failed`; `model_search_failed` the
    /// queries of LBR's model of the agent (its range updates, and the agent's
    /// answer to each bet LBR weighs, often at states never played) that read
    /// one, in the same two parts. `lbr_open_to` > 0 (LBR-open,
    /// records/preflop-open): LBR's first action in the small blind is a raise
    /// to that many chips, with every hand, or with `lbr_opener` (an agent)
    /// where that agent's root action for LBR's hand (sampled on LBR's own
    /// stream) is its pot-size open, its fold, limp and all-in kept; later
    /// preflop decisions are standard LBR's. `open_counts` is then `[LBR's
    /// raises to lbr_open_to, the agent's folds to them, its raises of them]`.
    #[pyo3(signature = (deals, sb, bb, stack, mirror = true, seat = 0, seed = 0, threads = 0, first_deal = 0, lbr_sizes = "fixed", lbr_peek = true, lbr_open_to = 0, lbr_opener = None))]
    #[allow(clippy::too_many_arguments)]
    fn lbr(
        &self,
        py: Python<'_>,
        deals: u64,
        sb: i64,
        bb: i64,
        stack: i64,
        mirror: bool,
        seat: usize,
        seed: u64,
        threads: usize,
        first_deal: u64,
        lbr_sizes: &str,
        lbr_peek: bool,
        lbr_open_to: i64,
        lbr_opener: Option<PyRef<'_, PlayAgent>>,
    ) -> PyResult<Py<PyDict>> {
        if seat > 1 || sb <= 0 || bb < sb || stack <= bb || deals < 2 {
            return Err(PyValueError::new_err("need seat in {0,1}, 0 < sb <= bb < stack, deals >= 2"));
        }
        if lbr_open_to < 0 || (lbr_open_to == 0 && lbr_opener.is_some()) {
            return Err(PyValueError::new_err("lbr_opener needs lbr_open_to > 0"));
        }
        let sizes = lbr::Sizes::parse(lbr_sizes).ok_or_else(|| PyValueError::new_err("lbr_sizes must be \"fixed\" or \"grid\""))?;
        let agent = Arc::clone(&self.agent);
        let opener = lbr_opener.map(|o| Arc::clone(&o.agent));
        let run = || {
            let open = (lbr_open_to > 0).then(|| lbr::Open { to: lbr_open_to, opener: opener.as_deref() });
            lbr::run_lbr(&agent, sb, bb, stack, first_deal, deals, mirror, seat, seed, sizes, lbr_peek, open)
        };
        let rows = py
            .detach(|| {
                if threads == 0 {
                    run()
                } else {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .map_err(|e| e.to_string())?
                        .install(run)
                }
            })
            .map_err(PyValueError::new_err)?;
        let preflop: Vec<f64> = rows.iter().map(|r| r.preflop).collect();
        let failed = rows.iter().fold([0u64; 2], |s, r| [s[0] + r.failed[0], s[1] + r.failed[1]]);
        let model_failed = rows.iter().fold([0u64; 2], |s, r| [s[0] + r.model_failed[0], s[1] + r.model_failed[1]]);
        let open_counts = rows.iter().fold([0u64; 3], |s, r| [s[0] + r.open[0], s[1] + r.open[1], s[2] + r.open[2]]);
        let stats = arena::MatchStats { per_deal_bb: rows.iter().map(|r| r.net).collect(), ..Default::default() };
        let (mean, se) = stats.bb_per_100(if mirror { 2.0 } else { 1.0 });
        let out = PyDict::new(py);
        out.set_item("bb_per_100", mean)?;
        out.set_item("se", se)?;
        out.set_item("hands", deals * if mirror { 2 } else { 1 })?;
        out.set_item("per_deal_bb", stats.per_deal_bb)?;
        out.set_item("per_deal_preflop_bb", preflop)?;
        out.set_item("search_failed", failed.to_vec())?;
        out.set_item("model_search_failed", model_failed.to_vec())?;
        if lbr_open_to > 0 {
            out.set_item("open_counts", open_counts.to_vec())?;
        }
        Ok(out.into())
    }

    /// Replay LBR deals (the indices of a `lbr(...)` run with the same stakes and
    /// `seed`; both seats per deal) with every action recorded. One dict per hand:
    /// `deal`, `lbr_seat`, `cards` (seat 0's hole, seat 1's, then the board, as
    /// 0..52 indices: rank*4+suit, suits cdhs), `net_bb` (LBR's), and `steps`:
    /// `street` (board cards dealt), `actor` ("lbr"/"agent"), `action`, `add`,
    /// `pot`, `owed`, LBR's `lbr_equity` against the agent's range, and for the
    /// agent `source`, `options` [(action, add, p)] and, on the turn and river,
    /// `diag`: `own_tv` (distance between the search model's range for the agent
    /// and its true range), `eq_model` / `eq_actual` (the agent's equity against
    /// the modelled opponent range / LBR's actual hand), `opp_eff` / `own_eff`
    /// (effective range sizes).
    /// `lbr_sizes` and `lbr_peek` as in `lbr`.
    #[pyo3(signature = (deals, sb, bb, stack, seed = 0, threads = 0, lbr_sizes = "fixed", lbr_peek = true))]
    #[allow(clippy::too_many_arguments)]
    fn lbr_trace(
        &self,
        py: Python<'_>,
        deals: Vec<u64>,
        sb: i64,
        bb: i64,
        stack: i64,
        seed: u64,
        threads: usize,
        lbr_sizes: &str,
        lbr_peek: bool,
    ) -> PyResult<Vec<Py<PyDict>>> {
        if sb <= 0 || bb < sb || stack <= bb {
            return Err(PyValueError::new_err("need 0 < sb <= bb < stack"));
        }
        let sizes = lbr::Sizes::parse(lbr_sizes).ok_or_else(|| PyValueError::new_err("lbr_sizes must be \"fixed\" or \"grid\""))?;
        let agent = Arc::clone(&self.agent);
        let run = || lbr::trace_lbr(&agent, sb, bb, stack, &deals, seed, sizes, lbr_peek);
        let hands = py
            .detach(|| {
                if threads == 0 {
                    run()
                } else {
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .map_err(|e| e.to_string())?
                        .install(run)
                }
            })
            .map_err(PyValueError::new_err)?;
        let act_parts = |a: nlhe::RealAct| -> (&'static str, i64) {
            match a {
                nlhe::RealAct::Fold => ("fold", 0),
                nlhe::RealAct::Check => ("check", 0),
                nlhe::RealAct::Call => ("call", 0),
                nlhe::RealAct::Add(d) => ("bet", d),
            }
        };
        let mut out = Vec::with_capacity(hands.len());
        for h in hands {
            let d = PyDict::new(py);
            d.set_item("deal", h.deal)?;
            d.set_item("lbr_seat", h.lbr_seat)?;
            d.set_item("cards", h.cards.iter().map(|&c| u32::from(c)).collect::<Vec<u32>>())?;
            d.set_item("net_bb", h.net / bb as f64)?;
            let mut steps = Vec::with_capacity(h.steps.len());
            for s in h.steps {
                let st = PyDict::new(py);
                let (kind, add) = act_parts(s.act);
                st.set_item("street", s.board_len)?;
                st.set_item("actor", if s.lbr { "lbr" } else { "agent" })?;
                st.set_item("action", kind)?;
                st.set_item("add", add)?;
                st.set_item("pot", s.pot)?;
                st.set_item("owed", s.owed)?;
                if !s.lbr_equity.is_nan() {
                    st.set_item("lbr_equity", s.lbr_equity)?;
                }
                if let Some(src) = s.source {
                    st.set_item("source", src.as_str())?;
                    let opts: Vec<(&str, i64, f64)> = s
                        .options
                        .iter()
                        .map(|&(a, p)| {
                            let (k, v) = act_parts(a);
                            (k, v, p)
                        })
                        .collect();
                    st.set_item("options", opts)?;
                }
                if let Some(g) = s.diag {
                    let dg = PyDict::new(py);
                    dg.set_item("own_tv", g.own_tv)?;
                    dg.set_item("eq_model", g.eq_model)?;
                    dg.set_item("eq_actual", g.eq_actual)?;
                    dg.set_item("opp_eff", g.opp_eff)?;
                    dg.set_item("own_eff", g.own_eff)?;
                    st.set_item("diag", dg)?;
                }
                steps.push(st.unbind());
            }
            d.set_item("steps", steps)?;
            out.push(d.unbind());
        }
        Ok(out)
    }
}

#[pymethods]
impl SavedBlueprint {
    /// Load the frozen card abstraction and a compact `blueprint.bin`. The binary
    /// reader validates the row count, keys, probability rows, duplicates and EOF;
    /// protocol-specific SHA-256 validation remains in Python so expected digests
    /// stay visible in pre-registration metadata.
    #[new]
    fn new(py: Python<'_>, abstraction_dir: &str, blueprint_path: &str) -> PyResult<Self> {
        let abstraction_dir = abstraction_dir.to_string();
        let blueprint_path = blueprint_path.to_string();
        let source_path = blueprint_path.clone();
        let (game, strategy, grid) = py
            .detach(move || -> Result<_, String> {
                let tag = mccfr::read_strategy_tag(Path::new(&blueprint_path))
                    .map_err(|e| format!("saved blueprint load failed: {e}"))?;
                let grid = grid_name(&tag);
                let cfg = NlheConfig::grid(&grid).ok_or_else(|| format!("blueprint names an unknown bet grid {grid:?}"))?;
                let abs = Abstraction::load(Path::new(&abstraction_dir))?;
                let game = Nlhe::with_abstraction(cfg, Arc::new(abs));
                let strategy = mccfr::load_average_strategy(Path::new(&blueprint_path))
                    .map_err(|e| format!("saved blueprint load failed: {e}"))?;
                Ok((game, strategy, grid))
            })
            .map_err(PyValueError::new_err)?;
        Ok(SavedBlueprint {
            game: Arc::new(game),
            strategy: Arc::new(strategy),
            source_path,
            grid,
        })
    }

    /// The bet grid the blueprint was trained on (`NlheConfig::grid`).
    #[getter]
    fn grid(&self) -> &str {
        &self.grid
    }

    fn num_infosets(&self) -> usize {
        self.strategy.len()
    }

    fn source_path(&self) -> &str {
        &self.source_path
    }

    /// The play-time agent over this blueprint (shares the loaded table).
    fn play_agent(&self) -> PlayAgent {
        PlayAgent {
            agent: Arc::new(play::Agent::new(Arc::clone(&self.game), Arc::clone(&self.strategy))),
        }
    }

    /// Zero-copy protocol policy view over this validated saved artifact.
    fn s32_policy(&self) -> S32Policy {
        S32Policy {
            game: Arc::clone(&self.game),
            policy: s32::Policy::tabular(Arc::clone(&self.strategy)),
            label: self.source_path.clone(),
        }
    }

    /// Train one in-abstraction sampled response to this immutable policy. The
    /// returned object contains responder rows only; blend it into a complete
    /// anchor with `blend_response` before evaluation so every unvisited row uses
    /// the anchor rather than the engine's generic uniform fallback.
    #[pyo3(signature = (responder, iterations, variant = "linear", seed = 0, threads = 1))]
    fn fixed_response(
        &self,
        py: Python<'_>,
        responder: usize,
        iterations: u64,
        variant: &str,
        seed: u64,
        threads: usize,
    ) -> PyResult<Self> {
        if responder >= 2 {
            return Err(PyValueError::new_err("responder must be 0 or 1"));
        }
        if iterations == 0 {
            return Err(PyValueError::new_err("iterations must be positive"));
        }
        let selected = mccfr::variant_by_name(variant)
            .ok_or_else(|| PyValueError::new_err(format!("unknown MCCFR variant: {variant:?}")))?;
        let game = Arc::clone(&self.game);
        let fixed = Arc::clone(&self.strategy);
        let response = py.detach(move || {
            let mut solver = FixedOpponentMCCFR::new(selected, seed, responder);
            solver.set_threads(threads.max(1));
            solver.run(game.as_ref(), fixed.as_ref(), iterations);
            solver
                .average_strategy()
                .into_iter()
                .map(|(key, row)| {
                    (
                        key.into_bytes().into_boxed_slice(),
                        row.into_iter().map(|p| p as f32).collect(),
                    )
                })
                .collect::<FxHashMap<_, _>>()
        });
        Ok(Self {
            game: Arc::clone(&self.game),
            strategy: Arc::new(response),
            source_path: format!(
                "response(responder={responder},iterations={iterations},seed={seed})"
            ),
            grid: self.grid.clone(),
        })
    }

    /// Blend responder rows into this complete anchor. Keys absent from the
    /// response retain the anchor exactly, including unvisited response infosets.
    #[pyo3(signature = (response, alpha))]
    fn blend_response(
        &self,
        py: Python<'_>,
        response: PyRef<'_, SavedBlueprint>,
        alpha: f64,
    ) -> PyResult<Self> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(PyValueError::new_err(
                "alpha must be finite and within [0, 1]",
            ));
        }
        if response.grid != self.grid {
            return Err(PyValueError::new_err(format!(
                "cannot blend a {} response into a {} anchor",
                response.grid, self.grid
            )));
        }
        let base = Arc::clone(&self.strategy);
        let target = Arc::clone(&response.strategy);
        let blended = py
            .detach(move || -> Result<_, String> {
                let mut out = FxHashMap::with_capacity_and_hasher(base.len(), Default::default());
                for (key, anchor_row) in base.iter() {
                    let row = if let Some(response_row) = target.get(key.as_ref()) {
                        if response_row.len() != anchor_row.len() {
                            return Err(format!(
                                "response row length mismatch at {}",
                                String::from_utf8_lossy(key)
                            ));
                        }
                        anchor_row
                            .iter()
                            .zip(response_row)
                            .map(|(a, b)| ((1.0 - alpha) * *a as f64 + alpha * *b as f64) as f32)
                            .collect()
                    } else {
                        anchor_row.clone()
                    };
                    out.insert(key.clone(), row);
                }
                Ok(out)
            })
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            game: Arc::clone(&self.game),
            strategy: Arc::new(blended),
            source_path: format!("blend({},alpha={alpha})", self.source_path),
            grid: self.grid.clone(),
        })
    }

    /// Seat-balanced policy-vs-policy Monte Carlo. Returns the mean in bb/100 and
    /// the sample standard deviation of per-hand returns in bb. This primitive is
    /// intentionally unpaired; the locked S3.2 layer adds common-random-number
    /// paired differences and block intervals later in the build order.
    #[pyo3(signature = (opponent, hands, seed = 0))]
    fn evaluate_against(
        &self,
        py: Python<'_>,
        opponent: PyRef<'_, SavedBlueprint>,
        hands: u64,
        seed: u64,
    ) -> PyResult<Py<PyDict>> {
        // Both tables are played on this blueprint's abstract tree.
        if opponent.grid != self.grid {
            return Err(PyValueError::new_err(format!(
                "abstract head-to-head needs one bet grid ({} vs {}); seat both as play agents instead",
                self.grid, opponent.grid
            )));
        }
        let game = Arc::clone(&self.game);
        let hero = Arc::clone(&self.strategy);
        let opp = Arc::clone(&opponent.strategy);
        let stats = py
            .detach(move || analysis::head_to_head_policies(&game, &hero, &opp, hands, seed))
            .map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        out.set_item("hands", stats.hands)?;
        out.set_item("bb_per_100", stats.bb_per_100)?;
        out.set_item("stddev_bb_per_hand", stats.stddev_bb_per_hand)?;
        Ok(out.unbind())
    }
}

#[pymethods]
impl S32Policy {
    /// This policy as a real-table play agent (translation + fallback as in
    /// `SavedBlueprint.play_agent`): leaks, blends and responses can be seated.
    fn play_agent(&self) -> PlayAgent {
        PlayAgent {
            agent: Arc::new(play::Agent::from_policy(Arc::clone(&self.game), self.policy.clone())),
        }
    }

    fn label(&self) -> &str {
        &self.label
    }

    #[staticmethod]
    fn uniform_like(reference: PyRef<'_, S32Policy>) -> Self {
        Self {
            game: Arc::clone(&reference.game),
            policy: s32::Policy::uniform(),
            label: "legal-action-uniform".to_string(),
        }
    }

    #[pyo3(signature = (family, strength))]
    fn leak(&self, family: &str, strength: f64) -> PyResult<Self> {
        let family = s32::LeakFamily::parse(family)
            .ok_or_else(|| PyValueError::new_err("family must be nit, station or maniac"))?;
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            return Err(PyValueError::new_err(
                "strength must be finite and within [0, 1]",
            ));
        }
        Ok(Self {
            game: Arc::clone(&self.game),
            policy: self.policy.leak(family, strength),
            label: format!("{}({strength})", self.label),
        })
    }

    #[pyo3(signature = (target, alpha))]
    fn blend(&self, target: PyRef<'_, S32Policy>, alpha: f64) -> PyResult<Self> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(PyValueError::new_err(
                "alpha must be finite and within [0, 1]",
            ));
        }
        Ok(Self {
            game: Arc::clone(&self.game),
            policy: self.policy.blend(&target.policy, alpha),
            label: format!("blend({}, {}, {alpha})", self.label, target.label),
        })
    }

    #[pyo3(signature = (responder, iterations, variant = "linear", seed = 0, threads = 1))]
    fn fixed_response(
        &self,
        py: Python<'_>,
        responder: usize,
        iterations: u64,
        variant: &str,
        seed: u64,
        threads: usize,
    ) -> PyResult<Self> {
        if responder >= 2 {
            return Err(PyValueError::new_err("responder must be 0 or 1"));
        }
        if iterations == 0 {
            return Err(PyValueError::new_err("iterations must be positive"));
        }
        let selected = mccfr::variant_by_name(variant)
            .ok_or_else(|| PyValueError::new_err(format!("unknown MCCFR variant: {variant:?}")))?;
        let game = Arc::clone(&self.game);
        let fixed = self.policy.clone();
        let strategy = py.detach(move || {
            let mut solver = FixedOpponentMCCFR::new(selected, seed, responder);
            solver.set_threads(threads.max(1));
            solver.run(game.as_ref(), &fixed, iterations);
            solver
                .average_strategy()
                .into_iter()
                .map(|(key, row)| {
                    (
                        key.into_bytes().into_boxed_slice(),
                        row.into_iter().map(|p| p as f32).collect(),
                    )
                })
                .collect::<FxHashMap<_, _>>()
        });
        Ok(Self {
            game: Arc::clone(&self.game),
            policy: s32::Policy::tabular(Arc::new(strategy)),
            label: format!(
                "response({},responder={responder},iterations={iterations},seed={seed})",
                self.label
            ),
        })
    }

    /// One `fixed_response` solve, returned as one policy per reach threshold in
    /// `min_reach` (see `FixedOpponentMCCFR::average_strategy_min_reach`): each
    /// keeps only the response infosets whose estimated reach probability is at
    /// least that threshold, so `anchor.blend(policy, 1.0)` falls back to the
    /// anchor on the rest. A threshold `<= 0` keeps every infoset and equals
    /// `fixed_response` exactly. Returns `(policies, kept_infosets, total_infosets)`.
    #[pyo3(signature = (responder, iterations, variant, seed, threads, min_reach))]
    #[allow(clippy::too_many_arguments)]
    fn fixed_response_gated(
        &self,
        py: Python<'_>,
        responder: usize,
        iterations: u64,
        variant: &str,
        seed: u64,
        threads: usize,
        min_reach: Vec<f64>,
    ) -> PyResult<(Vec<Self>, Vec<usize>, usize)> {
        if responder >= 2 {
            return Err(PyValueError::new_err("responder must be 0 or 1"));
        }
        if iterations == 0 {
            return Err(PyValueError::new_err("iterations must be positive"));
        }
        if min_reach.is_empty() || min_reach.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err("min_reach must be a non-empty list of finite thresholds"));
        }
        let selected = mccfr::variant_by_name(variant)
            .ok_or_else(|| PyValueError::new_err(format!("unknown MCCFR variant: {variant:?}")))?;
        let game = Arc::clone(&self.game);
        let fixed = self.policy.clone();
        let thresholds = min_reach.clone();
        let (tables, total) = py.detach(move || {
            let mut solver = FixedOpponentMCCFR::new(selected, seed, responder);
            solver.set_threads(threads.max(1));
            solver.run(game.as_ref(), &fixed, iterations);
            let tables: Vec<_> = thresholds
                .iter()
                .map(|&tau| {
                    solver
                        .average_strategy_min_reach(tau)
                        .into_iter()
                        .map(|(key, row)| {
                            (
                                key.into_boxed_slice(),
                                row.into_iter().map(|p| p as f32).collect(),
                            )
                        })
                        .collect::<FxHashMap<_, _>>()
                })
                .collect();
            (tables, solver.num_infosets())
        });
        let kept = tables.iter().map(|table| table.len()).collect();
        let policies = tables
            .into_iter()
            .zip(&min_reach)
            .map(|(strategy, tau)| Self {
                game: Arc::clone(&self.game),
                policy: s32::Policy::tabular(Arc::new(strategy)),
                label: format!(
                    "response({},responder={responder},iterations={iterations},seed={seed},min_reach={tau})",
                    self.label
                ),
            })
            .collect();
        Ok((policies, kept, total))
    }

    /// Sample only opponent public actions while the receiver plays Hero. Seats
    /// alternate each hand; neither terminal utility nor private cards are stored.
    #[pyo3(signature = (opponent, hands, seed = 0))]
    fn observe(
        &self,
        py: Python<'_>,
        opponent: PyRef<'_, S32Policy>,
        hands: u64,
        seed: u64,
    ) -> PyResult<S32Observation> {
        if hands == 0 {
            return Err(PyValueError::new_err(
                "observation requires at least one hand",
            ));
        }
        let game = Arc::clone(&self.game);
        let hero = self.policy.clone();
        let opponent_policy = opponent.policy.clone();
        let data = py.detach(move || {
            s32::sample_observation(game.as_ref(), &hero, &opponent_policy, hands, seed)
        });
        Ok(S32Observation {
            game: Arc::clone(&self.game),
            data: Arc::new(data),
        })
    }

    /// S3.3 fixed, independently seeded observation blocks.  Blocks are returned
    /// in seed order regardless of worker scheduling.
    #[pyo3(signature = (opponent, hands_per_block, seeds, threads = 1))]
    fn observe_blocks(
        &self,
        py: Python<'_>,
        opponent: PyRef<'_, S32Policy>,
        hands_per_block: u64,
        seeds: Vec<u64>,
        threads: usize,
    ) -> PyResult<Vec<S32Observation>> {
        if threads == 0 || threads > 16 {
            return Err(PyValueError::new_err("threads must be within 1..=16"));
        }
        let game = Arc::clone(&self.game);
        let hero = self.policy.clone();
        let opponent_policy = opponent.policy.clone();
        let data = py
            .detach(move || {
                s32::sample_observation_blocks(
                    game.as_ref(),
                    &hero,
                    &opponent_policy,
                    hands_per_block,
                    &seeds,
                    threads,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok(data
            .into_iter()
            .map(|block| S32Observation {
                game: Arc::clone(&self.game),
                data: Arc::new(block),
            })
            .collect())
    }

    /// One continuous observation stream split at the frozen hand boundary.
    #[pyo3(signature = (opponent, fit_hands, validation_hands, seed = 0))]
    fn observe_split(
        &self,
        py: Python<'_>,
        opponent: PyRef<'_, S32Policy>,
        fit_hands: u64,
        validation_hands: u64,
        seed: u64,
    ) -> PyResult<(S32Observation, S32Observation)> {
        if fit_hands == 0 || validation_hands == 0 {
            return Err(PyValueError::new_err(
                "fit_hands and validation_hands must both be positive",
            ));
        }
        let game = Arc::clone(&self.game);
        let hero = self.policy.clone();
        let opponent_policy = opponent.policy.clone();
        let (fit, validation) = py.detach(move || {
            s32::sample_observation_split(
                game.as_ref(),
                &hero,
                &opponent_policy,
                fit_hands,
                validation_hands,
                seed,
            )
        });
        Ok((
            S32Observation {
                game: Arc::clone(&self.game),
                data: Arc::new(fit),
            },
            S32Observation {
                game: Arc::clone(&self.game),
                data: Arc::new(validation),
            },
        ))
    }

    /// Paired candidate-minus-anchor bb/100 with fixed consecutive blocks. This
    /// returns measurements only; confidence corrections and protocol gates stay
    /// in Python where their frozen constants are visible.
    #[pyo3(signature = (anchor, opponent, hands, seed = 0, block_size = 10000))]
    fn paired_evaluate(
        &self,
        py: Python<'_>,
        anchor: PyRef<'_, S32Policy>,
        opponent: PyRef<'_, S32Policy>,
        hands: u64,
        seed: u64,
        block_size: u64,
    ) -> PyResult<Py<PyDict>> {
        let game = Arc::clone(&self.game);
        let candidate = self.policy.clone();
        let anchor_policy = anchor.policy.clone();
        let opponent_policy = opponent.policy.clone();
        let stats = py
            .detach(move || {
                s32::paired_evaluate(
                    game.as_ref(),
                    &candidate,
                    &anchor_policy,
                    &opponent_policy,
                    hands,
                    seed,
                    block_size,
                )
            })
            .map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        out.set_item("hands", stats.hands)?;
        out.set_item("block_size", stats.block_size)?;
        out.set_item("candidate_bb_per_100", stats.candidate_bb_per_100)?;
        out.set_item("anchor_bb_per_100", stats.anchor_bb_per_100)?;
        out.set_item("gain_bb_per_100", stats.gain_bb_per_100)?;
        out.set_item(
            "difference_stddev_bb_per_hand",
            stats.difference_stddev_bb_per_hand,
        )?;
        out.set_item("block_gains_bb_per_100", stats.block_gains_bb_per_100)?;
        Ok(out.unbind())
    }

    /// S3.3 fixed-block evaluator with worker-count invariant RNG streams and
    /// ordered reduction.
    #[pyo3(signature = (anchor, opponent, hands, seed = 0, block_size = 10000, threads = 1))]
    fn paired_evaluate_blocks(
        &self,
        py: Python<'_>,
        anchor: PyRef<'_, S32Policy>,
        opponent: PyRef<'_, S32Policy>,
        hands: u64,
        seed: u64,
        block_size: u64,
        threads: usize,
    ) -> PyResult<Py<PyDict>> {
        if threads == 0 || threads > 16 {
            return Err(PyValueError::new_err("threads must be within 1..=16"));
        }
        let game = Arc::clone(&self.game);
        let candidate = self.policy.clone();
        let anchor_policy = anchor.policy.clone();
        let opponent_policy = opponent.policy.clone();
        let stats = py
            .detach(move || {
                s32::paired_evaluate_blocks(
                    game.as_ref(),
                    &candidate,
                    &anchor_policy,
                    &opponent_policy,
                    hands,
                    seed,
                    block_size,
                    threads,
                )
            })
            .map_err(PyValueError::new_err)?;
        let out = PyDict::new(py);
        out.set_item("hands", stats.hands)?;
        out.set_item("block_size", stats.block_size)?;
        out.set_item("candidate_bb_per_100", stats.candidate_bb_per_100)?;
        out.set_item("anchor_bb_per_100", stats.anchor_bb_per_100)?;
        out.set_item("gain_bb_per_100", stats.gain_bb_per_100)?;
        out.set_item(
            "difference_stddev_bb_per_hand",
            stats.difference_stddev_bb_per_hand,
        )?;
        out.set_item("block_gains_bb_per_100", stats.block_gains_bb_per_100)?;
        Ok(out.unbind())
    }
}

#[pymethods]
impl S32Observation {
    fn decisions(&self) -> u64 {
        self.data.decisions()
    }

    fn deterministic_fingerprint(&self) -> u64 {
        self.data.deterministic_fingerprint()
    }

    /// S3.4 fingerprint includes the additive position-shape counts while the
    /// older deterministic fingerprint remains frozen for S3.3 parity.
    fn support_fingerprint(&self) -> u64 {
        self.data.support_fingerprint()
    }

    #[pyo3(signature = (five_level = true))]
    fn count_table_bytes(&self, five_level: bool) -> usize {
        self.data.count_table_bytes(five_level)
    }

    #[pyo3(signature = (limit = 20))]
    fn exact_key_sample(&self, limit: usize) -> Vec<String> {
        self.data
            .exact_keys()
            .take(limit)
            .map(str::to_string)
            .collect()
    }

    /// Full-table schema check for the public-only S3.2 observation keys.
    fn public_keys_valid(&self) -> bool {
        self.data.public_keys_valid()
    }

    /// Integer-exact merge used to assemble independently sampled S3.3 blocks.
    fn merged(&self, other: PyRef<'_, S32Observation>) -> PyResult<S32Observation> {
        let data = self
            .data
            .merged(other.data.as_ref())
            .map_err(PyValueError::new_err)?;
        Ok(S32Observation {
            game: Arc::clone(&self.game),
            data: Arc::new(data),
        })
    }

    /// Bulk integer-exact merge of many blocks. Identical result to folding
    /// `merged` left-to-right, but clones the accumulator once instead of once
    /// per block (the assembly of 200 fit/reference blocks is otherwise
    /// quadratic in accumulator size).
    #[staticmethod]
    fn merge_all(blocks: Vec<PyRef<'_, S32Observation>>) -> PyResult<S32Observation> {
        let first = blocks
            .first()
            .ok_or_else(|| PyValueError::new_err("merge_all needs at least one block"))?;
        let game = Arc::clone(&first.game);
        let refs: Vec<&s32::Observation> = blocks.iter().map(|block| block.data.as_ref()).collect();
        let data = s32::Observation::merge_all(&refs).map_err(PyValueError::new_err)?;
        Ok(S32Observation {
            game,
            data: Arc::new(data),
        })
    }

    #[pyo3(signature = (reference, min_count = 200, prior_strength = 20.0))]
    fn fit_model(
        &self,
        reference: PyRef<'_, S32Observation>,
        min_count: u64,
        prior_strength: f64,
    ) -> PyResult<S32Policy> {
        if min_count == 0 {
            return Err(PyValueError::new_err("min_count must be positive"));
        }
        if !prior_strength.is_finite() || prior_strength < 0.0 {
            return Err(PyValueError::new_err(
                "prior_strength must be finite and non-negative",
            ));
        }
        Ok(S32Policy {
            game: Arc::clone(&self.game),
            policy: s32::Policy::public_model(
                self.data.as_ref(),
                reference.data.as_ref(),
                min_count,
                prior_strength,
            ),
            label: format!("public-model(min={min_count},prior={prior_strength})"),
        })
    }

    fn score_model(&self, py: Python<'_>, model: PyRef<'_, S32Policy>) -> PyResult<Py<PyDict>> {
        let score = model
            .policy
            .score(self.data.as_ref())
            .ok_or_else(|| PyValueError::new_err("policy is not a fitted public model"))?;
        let out = PyDict::new(py);
        out.set_item("decisions", score.decisions)?;
        out.set_item("coverage", score.coverage)?;
        out.set_item("model_log_loss", score.model_log_loss)?;
        out.set_item("reference_log_loss", score.reference_log_loss)?;
        out.set_item("deviation_l1", score.deviation_l1)?;
        Ok(out.unbind())
    }

    #[pyo3(signature = (reference, exact_strength = 200.0, backoff_strength = 50.0, context_strength = 20.0))]
    fn fit_hierarchical_model(
        &self,
        reference: PyRef<'_, S32Observation>,
        exact_strength: f64,
        backoff_strength: f64,
        context_strength: f64,
    ) -> PyResult<S32Policy> {
        if [exact_strength, backoff_strength, context_strength]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(PyValueError::new_err(
                "hierarchical strengths must be finite and positive",
            ));
        }
        Ok(S32Policy {
            game: Arc::clone(&self.game),
            policy: s32::Policy::hierarchical_public_model(
                self.data.as_ref(),
                reference.data.as_ref(),
                exact_strength,
                backoff_strength,
                context_strength,
            ),
            label: format!(
                "hierarchical-public-model(exact={exact_strength},backoff={backoff_strength},context={context_strength})"
            ),
        })
    }

    fn score_hierarchical_model(
        &self,
        py: Python<'_>,
        model: PyRef<'_, S32Policy>,
    ) -> PyResult<Py<PyDict>> {
        let score = model
            .policy
            .score_hierarchical(self.data.as_ref())
            .ok_or_else(|| {
                PyValueError::new_err("policy is not a fitted hierarchical public model")
            })?;
        let out = PyDict::new(py);
        out.set_item("decisions", score.decisions)?;
        out.set_item("effective_weight_sum", score.effective_weight_sum)?;
        out.set_item("zero_weight_decisions", score.zero_weight_decisions)?;
        out.set_item(
            "legacy_supported_decisions",
            score.legacy_supported_decisions,
        )?;
        out.set_item("model_nll_sum", score.model_nll_sum)?;
        out.set_item("reference_nll_sum", score.reference_nll_sum)?;
        out.set_item("legacy_nll_sum", score.legacy_nll_sum)?;
        out.set_item("sparse_decisions", score.sparse_decisions)?;
        out.set_item("sparse_model_nll_sum", score.sparse_model_nll_sum)?;
        out.set_item("sparse_reference_nll_sum", score.sparse_reference_nll_sum)?;
        out.set_item("model_brier_sum", score.model_brier_sum)?;
        out.set_item("reference_brier_sum", score.reference_brier_sum)?;
        out.set_item("deviation_l1_sum", score.deviation_l1_sum)?;
        out.set_item("ece_count", score.ece_count)?;
        out.set_item("ece_confidence_sum", score.ece_confidence_sum)?;
        out.set_item("ece_accuracy_sum", score.ece_accuracy_sum)?;
        Ok(out.unbind())
    }

    #[pyo3(signature = (reference, exact_strength = 200.0, backoff_strength = 50.0, context_strength = 20.0, position_strength = 10.0, global_strength = 5.0))]
    fn fit_five_level_model(
        &self,
        reference: PyRef<'_, S32Observation>,
        exact_strength: f64,
        backoff_strength: f64,
        context_strength: f64,
        position_strength: f64,
        global_strength: f64,
    ) -> PyResult<S32Policy> {
        if [
            exact_strength,
            backoff_strength,
            context_strength,
            position_strength,
            global_strength,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(PyValueError::new_err(
                "five-level strengths must be finite and positive",
            ));
        }
        Ok(S32Policy {
            game: Arc::clone(&self.game),
            policy: s32::Policy::five_level_public_model(
                self.data.as_ref(),
                reference.data.as_ref(),
                exact_strength,
                backoff_strength,
                context_strength,
                position_strength,
                global_strength,
            ),
            label: format!(
                "five-level-public-model(exact={exact_strength},backoff={backoff_strength},context={context_strength},position={position_strength},global={global_strength})"
            ),
        })
    }

    fn score_five_level_model(
        &self,
        py: Python<'_>,
        model: PyRef<'_, S32Policy>,
    ) -> PyResult<Py<PyDict>> {
        let score = model
            .policy
            .score_five_level(self.data.as_ref())
            .ok_or_else(|| {
                PyValueError::new_err("policy is not a fitted five-level public model")
            })?;
        let out = PyDict::new(py);
        out.set_item("decisions", score.decisions)?;
        out.set_item("effective_weight_sum", score.effective_weight_sum)?;
        out.set_item("zero_weight_decisions", score.zero_weight_decisions)?;
        out.set_item(
            "three_level_zero_decisions",
            score.three_level_zero_decisions,
        )?;
        out.set_item(
            "recovered_three_level_zero_decisions",
            score.recovered_three_level_zero_decisions,
        )?;
        out.set_item(
            "legacy_supported_decisions",
            score.legacy_supported_decisions,
        )?;
        out.set_item("model_nll_sum", score.model_nll_sum)?;
        out.set_item("reference_nll_sum", score.reference_nll_sum)?;
        out.set_item("shadow_nll_sum", score.shadow_nll_sum)?;
        out.set_item("legacy_nll_sum", score.legacy_nll_sum)?;
        out.set_item("sparse_decisions", score.sparse_decisions)?;
        out.set_item("sparse_model_nll_sum", score.sparse_model_nll_sum)?;
        out.set_item("sparse_reference_nll_sum", score.sparse_reference_nll_sum)?;
        out.set_item("model_brier_sum", score.model_brier_sum)?;
        out.set_item("reference_brier_sum", score.reference_brier_sum)?;
        out.set_item("deviation_l1_sum", score.deviation_l1_sum)?;
        out.set_item("ece_count", score.ece_count)?;
        out.set_item("ece_confidence_sum", score.ece_confidence_sum)?;
        out.set_item("ece_accuracy_sum", score.ece_accuracy_sum)?;
        Ok(out.unbind())
    }
}

#[pymethods]
impl BlueprintAnalyzer {
    #[new]
    fn new(abstraction_dir: &str) -> PyResult<Self> {
        let abs = Abstraction::load(Path::new(abstraction_dir)).map_err(PyValueError::new_err)?;
        Ok(BlueprintAnalyzer { abs: Arc::new(abs) })
    }

    /// Read the average-strategy blueprint at `blueprint_path` and return a dict
    /// `{action_freq, bet_size_freq, hh_bb_per_100}`: the per-street action mix
    /// (`f`/`x`/`c`/`r`/`A`) and bet-size mix (`0.5pot`/`1.0pot`/`allin`), each
    /// normalized and including only visited streets, plus the head-to-head win rate
    /// in bb/100 against a calling-station baseline. The compute (Monte-Carlo
    /// self-play + head-to-head) runs with the GIL released.
    #[pyo3(signature = (blueprint_path, self_play_hands = 20000, hh_hands = 20000, seed = 0))]
    fn analyze(
        &self,
        py: Python<'_>,
        blueprint_path: &str,
        self_play_hands: u64,
        hh_hands: u64,
        seed: u64,
    ) -> PyResult<Py<PyDict>> {
        let (stats, hh) = py
            .detach(|| -> std::io::Result<_> {
                // The game of the grid the blueprint names.
                let grid = grid_name(&mccfr::read_strategy_tag(Path::new(blueprint_path))?);
                let cfg = NlheConfig::grid(&grid).ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, format!("unknown bet grid {grid:?}"))
                })?;
                let game = Nlhe::with_abstraction(cfg, Arc::clone(&self.abs));
                let strat = mccfr::load_average_strategy(Path::new(blueprint_path))?;
                let stats = analysis::self_play_stats(&game, &strat, self_play_hands, seed);
                let hh = analysis::head_to_head(&game, &strat, hh_hands, seed);
                Ok((stats, hh))
            })
            .map_err(|e| PyValueError::new_err(format!("blueprint analysis failed: {e}")))?;

        const STREETS: [&str; 4] = ["P", "F", "T", "R"];
        const CATS: [&str; analysis::N_CATS] = ["f", "x", "c", "r", "A"];
        const SIZES: [&str; analysis::N_SIZES] = ["0.5pot", "1.0pot", "allin"];

        let action_freq = PyDict::new(py);
        let bet_size_freq = PyDict::new(py);
        for (si, st) in STREETS.iter().enumerate() {
            let row = &stats.action[si];
            let total: f64 = row.iter().sum();
            if total > 0.0 {
                let m = PyDict::new(py);
                for (ci, c) in CATS.iter().enumerate() {
                    m.set_item(c, row[ci] / total)?;
                }
                action_freq.set_item(st, m)?;
            }
            let brow = &stats.bet[si];
            let btotal: f64 = brow.iter().sum();
            if btotal > 0.0 {
                let m = PyDict::new(py);
                for (bi, b) in SIZES.iter().enumerate() {
                    m.set_item(b, brow[bi] / btotal)?;
                }
                bet_size_freq.set_item(st, m)?;
            }
        }

        let out = PyDict::new(py);
        out.set_item("action_freq", action_freq)?;
        out.set_item("bet_size_freq", bet_size_freq)?;
        out.set_item("hh_bb_per_100", hh)?;
        Ok(out.unbind())
    }
}

/// FLOPSEARCH.md D4's oracle: the full flop game (the flop, turn and river
/// rounds over every turn and river card, exact card removal) solved by the
/// search's DCFR on the CPU, its arithmetic in float64 and its regrets and
/// strategy sums stored in float64 or float32. Far too big to play with (about
/// a million decisions at 200bb; 16 or 8 bytes per action and holding); it
/// scores flop strategies offline. Flop nodes are addressed by `flop_nodes`'
/// paths.
#[pyclass]
struct FlopOracle {
    inner: AnyOracle,
    /// Declared after `inner`, so dropped after it (`OraclePool`).
    pool: OraclePool,
}

/// The oracle's thread pool, which when dropped after the oracle hands the
/// oracle's freed memory back to the system. mimalloc returns freed pages only
/// on a later call from the thread that owns them (frees from other threads
/// wait for it), so the old store's gigabytes stayed committed for good in an
/// idle caller: flop_m1 waiting for memory before its next spot stalled on 29
/// Sep. So each pool thread collects its own heap, then the dropping thread
/// purges once they are done (mimalloc lets one thread purge at a time).
struct OraclePool(rayon::ThreadPool);

impl std::ops::Deref for OraclePool {
    type Target = rayon::ThreadPool;

    fn deref(&self) -> &rayon::ThreadPool {
        &self.0
    }
}

impl Drop for OraclePool {
    fn drop(&mut self) {
        // SAFETY: mimalloc's collection takes no pointers.
        self.0.broadcast(|_| unsafe { mi_collect(true) });
        unsafe { mi_collect(true) };
    }
}

/// The oracle in either store precision.
enum AnyOracle {
    F64(search::Oracle<f64>),
    F32(search::Oracle<f32>),
}

/// `$body` with `$o` bound to the oracle, whatever its precision.
macro_rules! each_oracle {
    ($inner:expr, $o:ident => $body:expr) => {
        match $inner {
            AnyOracle::F64($o) => $body,
            AnyOracle::F32($o) => $body,
        }
    };
}

fn py_menu((bet_fracs, raise_fracs, max_raises): (Vec<f64>, Vec<f64>, u8)) -> search::Menu {
    search::Menu { bet_fracs, raise_fracs, max_raises }
}

#[pymethods]
impl FlopOracle {
    /// The flop round from its start: `board` (3 cards), each player's `stake`
    /// in the pot and `stacks` behind, the `first` to act, the `min_bet`, the
    /// flop, turn and river `menus` as `(bet_fracs, raise_fracs, max_raises)`,
    /// both `ranges` over `holdings`; `threads` for the solve (0: all);
    /// `precision` of the stored regrets and strategy sums, "f64" or "f32"
    /// (half the memory); `isomorphism`: deal the turn and river cards that
    /// only swap suits the flop cannot tell apart once (the same game, less
    /// time and memory), which needs both ranges and every frozen strategy
    /// unchanged by those swaps (a ValueError otherwise).
    #[new]
    #[pyo3(signature = (board, stake, stacks, first, min_bet, menus, ranges, threads = 0, precision = "f64", isomorphism = false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        board: Vec<u8>,
        stake: (i64, i64),
        stacks: (i64, i64),
        first: usize,
        min_bet: i64,
        menus: Vec<(Vec<f64>, Vec<f64>, u8)>,
        ranges: (Vec<f64>, Vec<f64>),
        threads: usize,
        precision: &str,
        isomorphism: bool,
    ) -> PyResult<Self> {
        let b: [u8; 3] = board.try_into().map_err(|_| PyValueError::new_err("board must have 3 cards"))?;
        if precision != "f64" && precision != "f32" {
            return Err(PyValueError::new_err("precision must be \"f64\" or \"f32\""));
        }
        if b.iter().any(|&c| c >= 52) || b[0] == b[1] || b[0] == b[2] || b[1] == b[2] {
            return Err(PyValueError::new_err("board cards must be distinct and below 52"));
        }
        let [flop, turn, river]: [(Vec<f64>, Vec<f64>, u8); 3] =
            menus.try_into().map_err(|_| PyValueError::new_err("menus: flop, turn and river"))?;
        if first > 1 {
            return Err(PyValueError::new_err("first must be 0 or 1"));
        }
        let spec = search::RoundSpec {
            stake: [stake.0, stake.1],
            stacks: [stacks.0, stacks.1],
            first,
            min_bet,
            menu: py_menu(flop),
            river: Some(py_menu(river)),
            forced: Vec::new(),
        };
        let turn = py_menu(turn);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ranges = [ranges.0, ranges.1];
        let inner = py
            .detach(|| {
                pool.install(|| match precision {
                    "f32" => search::Oracle::zeroed(b, &spec, &turn, ranges, isomorphism).map(AnyOracle::F32),
                    _ => search::Oracle::zeroed(b, &spec, &turn, ranges, isomorphism).map(AnyOracle::F64),
                })
            })
            .map_err(PyValueError::new_err)?;
        Ok(FlopOracle { inner, pool: OraclePool(pool) })
    }

    /// The holdings the ranges and strategies are over (`FlatSubgame`'s order).
    #[getter]
    fn holdings(&self) -> Vec<(u8, u8)> {
        each_oracle!(&self.inner, o => o.hands.combos.iter().map(|h| (h[0], h[1])).collect())
    }

    /// "f64" or "f32": the stored regrets' and strategy sums' precision.
    #[getter]
    fn precision(&self) -> &'static str {
        match self.inner {
            AnyOracle::F64(_) => "f64",
            AnyOracle::F32(_) => "f32",
        }
    }

    /// Whether suit-isomorphic cards are dealt once (`isomorphism`).
    #[getter]
    fn isomorphism(&self) -> bool {
        each_oracle!(&self.inner, o => o.isomorphic())
    }

    /// DCFR iterations run so far.
    #[getter]
    fn iterations(&self) -> u64 {
        each_oracle!(&self.inner, o => o.iterations())
    }

    /// (nodes, decision slots, actions over all slots).
    #[getter]
    fn size(&self) -> (usize, usize, usize) {
        each_oracle!(&self.inner, o => o.size())
    }

    /// The flop round's decision nodes in pre-order: `(path, player, actions)`,
    /// the path the action indices from the root, each action `(kind, add)`
    /// with kind "f" (fold), "c" (check or call) or "b" (bet or raise adding
    /// `add` chips; all-in included).
    fn flop_nodes(&self) -> Vec<(Vec<usize>, usize, Vec<(String, i64)>)> {
        each_oracle!(&self.inner, o => o.flop_nodes())
            .into_iter()
            .map(|(path, player, acts)| {
                let acts = acts
                    .into_iter()
                    .map(|a| match a {
                        search::Act::Fold => ("f".to_string(), 0),
                        search::Act::CheckCall => ("c".to_string(), 0),
                        search::Act::Bet(d) => ("b".to_string(), d),
                    })
                    .collect();
                (path, player, acts)
            })
            .collect()
    }

    /// Play `strategy` (action-major, actions x holdings, each holding's
    /// column a distribution) at the flop node at `path`, never updated.
    fn freeze(&mut self, path: Vec<usize>, strategy: Vec<f64>) -> PyResult<()> {
        each_oracle!(&mut self.inner, o => o.freeze(&path, strategy)).map_err(PyValueError::new_err)
    }

    /// Average strategy at the flop node at `path` (action-major).
    fn average(&mut self, path: Vec<usize>) -> PyResult<Vec<f64>> {
        each_oracle!(&mut self.inner, o => o.average(&path)).map_err(PyValueError::new_err)
    }

    /// Run `iters` more DCFR iterations, the strategy sums weighted
    /// `(t / (t + 1)) ^ gamma` (the search's 2 by default).
    #[pyo3(signature = (iters, gamma = 2.0))]
    fn run(&mut self, py: Python<'_>, iters: u64, gamma: f64) -> PyResult<()> {
        if !(gamma.is_finite() && gamma >= 0.0) {
            return Err(PyValueError::new_err("gamma must be finite and non-negative"));
        }
        let FlopOracle { inner, pool } = self;
        py.detach(|| pool.install(|| each_oracle!(inner, o => o.run(iters, gamma))));
        Ok(())
    }

    /// Each player's best-response value against the average strategy
    /// (frozen nodes as frozen), chips per dealt pair: the game value to
    /// player 0 lies in `[-br[1], br[0]]`, and the sum is the exploitability.
    fn best_responses(&mut self, py: Python<'_>) -> (f64, f64) {
        let FlopOracle { inner, pool } = self;
        let [a, b] = py.detach(|| pool.install(|| each_oracle!(inner, o => o.best_responses())));
        (a, b)
    }

    /// Value to `p` of the average-strategy profile, chips per dealt pair.
    fn value(&mut self, py: Python<'_>, p: usize) -> PyResult<f64> {
        if p > 1 {
            return Err(PyValueError::new_err("p must be 0 or 1"));
        }
        let FlopOracle { inner, pool } = self;
        Ok(py.detach(|| pool.install(|| each_oracle!(inner, o => o.value(p)))))
    }
}

#[pymodule]
fn parciver_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_leduc, m)?)?;
    m.add_function(wrap_pyfunction!(solve_kuhn, m)?)?;
    m.add_function(wrap_pyfunction!(solve_leduc_mccfr, m)?)?;
    m.add_function(wrap_pyfunction!(solve_kuhn_mccfr, m)?)?;
    m.add_function(wrap_pyfunction!(leduc_infoset_count, m)?)?;
    m.add_function(wrap_pyfunction!(kuhn_infoset_count, m)?)?;
    m.add_function(wrap_pyfunction!(solve_kuhn_n, m)?)?;
    m.add_function(wrap_pyfunction!(kuhn_n_infoset_count, m)?)?;
    m.add_function(wrap_pyfunction!(kuhn_n_nashconv, m)?)?;
    m.add_function(wrap_pyfunction!(kuhn_n_nashconv_per_player, m)?)?;
    m.add_function(wrap_pyfunction!(kuhn_n_best_response_value, m)?)?;
    m.add_function(wrap_pyfunction!(kuhn_n_on_policy_value, m)?)?;
    m.add_function(wrap_pyfunction!(eval7, m)?)?;
    m.add_function(wrap_pyfunction!(eval5, m)?)?;
    m.add_function(wrap_pyfunction!(hand_category, m)?)?;
    m.add_function(wrap_pyfunction!(hand_category_name, m)?)?;
    m.add_function(wrap_pyfunction!(parse_card, m)?)?;
    m.add_function(wrap_pyfunction!(preflop_index, m)?)?;
    m.add_function(wrap_pyfunction!(preflop_name, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_cards, m)?)?;
    m.add_function(wrap_pyfunction!(equity_vs_random, m)?)?;
    m.add_function(wrap_pyfunction!(equity_vs_hand, m)?)?;
    m.add_function(wrap_pyfunction!(equity_vs_random_mc, m)?)?;
    m.add_function(wrap_pyfunction!(flop_equity_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(turn_equity_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(river_equity, m)?)?;
    m.add_function(wrap_pyfunction!(flop_runouts, m)?)?;
    m.add_function(wrap_pyfunction!(flop_histograms_batch, m)?)?;
    m.add_function(wrap_pyfunction!(turn_histograms_batch, m)?)?;
    m.add_function(wrap_pyfunction!(river_equities_batch, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_situation, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_canonical_flops, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_index_size, m)?)?;
    m.add_function(wrap_pyfunction!(street_card_count, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_index, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_unindex_batch, m)?)?;
    m.add_function(wrap_pyfunction!(kuhn_exploitability, m)?)?;
    m.add_function(wrap_pyfunction!(leduc_exploitability, m)?)?;
    m.add_function(wrap_pyfunction!(solve_nlhe_small, m)?)?;
    m.add_function(wrap_pyfunction!(solve_nlhe_small_mccfr, m)?)?;
    m.add_function(wrap_pyfunction!(nlhe_small_fixed_response, m)?)?;
    m.add_function(wrap_pyfunction!(nlhe_small_exploitability, m)?)?;
    m.add_function(wrap_pyfunction!(nlhe_small_infoset_count, m)?)?;
    m.add_function(wrap_pyfunction!(nlhe_small_best_response_value, m)?)?;
    m.add_function(wrap_pyfunction!(nlhe_small_best_response_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(nlhe_small_on_policy_value, m)?)?;
    m.add_function(wrap_pyfunction!(nlhe_small_observed_marginal, m)?)?;
    m.add_function(wrap_pyfunction!(nlhe_small_observed_marginal_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(nlhe_small_observed_model, m)?)?;
    m.add_function(wrap_pyfunction!(nlhe_small_infoset_structure, m)?)?;
    m.add_class::<PyAbstraction>()?;
    m.add_class::<Blueprint>()?;
    m.add_class::<BlueprintAnalyzer>()?;
    m.add_class::<SavedBlueprint>()?;
    m.add_class::<PlayAgent>()?;
    m.add_class::<FlopOracle>()?;
    m.add_class::<S32Policy>()?;
    m.add_class::<S32Observation>()?;
    m.add(
        "__all__",
        vec![
            "solve_leduc",
            "solve_kuhn",
            "solve_leduc_mccfr",
            "solve_kuhn_mccfr",
            "leduc_infoset_count",
            "kuhn_infoset_count",
            "solve_kuhn_n",
            "kuhn_n_infoset_count",
            "kuhn_n_nashconv",
            "kuhn_n_nashconv_per_player",
            "kuhn_n_best_response_value",
            "kuhn_n_on_policy_value",
            "eval7",
            "eval5",
            "hand_category",
            "hand_category_name",
            "parse_card",
            "preflop_index",
            "preflop_name",
            "canonical_cards",
            "equity_vs_random",
            "equity_vs_hand",
            "equity_vs_random_mc",
            "flop_equity_histogram",
            "turn_equity_histogram",
            "river_equity",
            "flop_runouts",
            "flop_histograms_batch",
            "turn_histograms_batch",
            "river_equities_batch",
            "canonical_situation",
            "enumerate_canonical_flops",
            "canonical_index_size",
            "street_card_count",
            "canonical_index",
            "canonical_unindex_batch",
            "kuhn_exploitability",
            "leduc_exploitability",
            "solve_nlhe_small",
            "solve_nlhe_small_mccfr",
            "nlhe_small_fixed_response",
            "nlhe_small_exploitability",
            "nlhe_small_infoset_count",
            "nlhe_small_best_response_value",
            "nlhe_small_best_response_strategy",
            "nlhe_small_on_policy_value",
            "nlhe_small_observed_marginal",
            "nlhe_small_observed_marginal_nodes",
            "nlhe_small_observed_model",
            "nlhe_small_infoset_structure",
            "Abstraction",
            "Blueprint",
            "BlueprintAnalyzer",
            "SavedBlueprint",
            "PlayAgent",
            "FlopOracle",
            "S32Policy",
            "S32Observation",
        ],
    )?;
    Ok(())
}
