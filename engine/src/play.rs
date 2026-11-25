//! The play-time agent: turns a REAL heads-up hand (real blinds, stacks and
//! arbitrary bet sizes) into a decision from a blueprint solved on the abstract
//! 100bb grid.
//!
//! How a decision is made (stateless: the whole hand is replayed every call, so a
//! caller only has to pass what happened, and the same inputs always give the
//! same translation):
//!
//! 1. The real hand is replayed on a real-chip copy of the betting engine
//!    ([`Nlhe::apply_real`]) — the same blinds / min-raise / street-closure rules
//!    as the abstract tree, with the table's real amounts. Every action is
//!    validated, so an impossible history is an error, not a silent guess.
//! 2. In lockstep, each real action is TRANSLATED into the abstract tree:
//!    fold/check/call map directly; a bet or raise maps to the grid size whose
//!    chip amount matches exactly, otherwise to one of the two neighbouring grid
//!    sizes by the pseudo-harmonic mapping (Ganzfried & Sandholm, IJCAI 2013) on
//!    pot fractions. A real all-in maps to the abstract all-in. The opponent's
//!    randomized mapping is seeded by `(seed, action index)`, so re-asking the
//!    same hand reproduces it; the agent's own past actions map deterministically.
//! 3. At the agent's turn the abstract infoset key (`street | bucket | history`)
//!    is looked up in the blueprint, an action is sampled from its row, and it is
//!    converted back into real chips (a pot fraction of the REAL pot, clamped to
//!    the real min-raise and stack).
//!
//! Where the grid cannot represent the real hand — the most common case is a
//! non-all-in 3-bet, which the grid only has as all-in — the abstract line either
//! ends early or falls out of step with the real one. From then on the hand is
//! OFF-TREE and the agent uses [`fallback`]: check when free, otherwise call when
//! the hand's all-in equity against a random holding covers the pot odds, else
//! fold (never raise). The same fallback covers a key the blueprint never visited. Every decision reports its
//! source so the evaluation harness can measure how often this happens; real-time
//! search is the intended replacement for it.

use std::sync::Arc;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::cards::Card;
use crate::equity;
use crate::subgame::{self, SearchCache, SearchCfg};
use crate::game::Game;
use crate::nlhe::{ActLabel, Nlhe, NlheConfig, NlheState, RealAct};
use crate::s32::Policy;

/// Monte-Carlo samples for the fallback's equity on preflop/flop (turn and river
/// are enumerated exactly).
const FALLBACK_EQUITY_SAMPLES: u64 = 800;

/// A real bet this large a fraction of the bettor's remaining stack may map to
/// the abstract all-in (as the upper pseudo-harmonic neighbour); smaller real
/// bets never do, because an abstract all-in ends the abstract betting while the
/// real hand would continue (`Oversize::Snap`).
const ALLIN_NEIGHBOUR_COMMIT: f64 = 0.5;

/// How a real bet or raise above the largest grid size at its node is read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Oversize {
    /// As that size, unless it commits at least `ALLIN_NEIGHBOUR_COMMIT` of the
    /// bettor's stack: then between that size and all-in. A hand that goes on
    /// past an all-in reading leaves the tree.
    #[default]
    Snap,
    /// Always between the largest grid size and all-in (pseudo-harmonic). An
    /// all-in reading of a bet that is not all-in only stands in for the
    /// response: unless the response is a fold, the bet is replayed as the
    /// largest grid size and the response is read there (TRANSLATE.md).
    Standin,
}

impl Oversize {
    pub(crate) fn parse(s: &str) -> Option<Self> {
        match s {
            "snap" => Some(Oversize::Snap),
            "standin" => Some(Oversize::Standin),
            _ => None,
        }
    }
}

/// How a real preflop open below the smallest grid size is read
/// (records/preflop-open/FINDING.md, fix F2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum OpenBelow {
    /// As that size (sizes below the grid snap up).
    #[default]
    Snap,
    /// Between the limp (x = 0) and that size, pseudo-harmonically:
    /// P(limp) = (1 - x) / (1 + x) against the pot-size open. Under a limp
    /// reading the big blind's real call of the open is its abstract check.
    Limp,
}

impl OpenBelow {
    pub(crate) fn parse(s: &str) -> Option<Self> {
        match s {
            "snap" => Some(OpenBelow::Snap),
            "limp" => Some(OpenBelow::Limp),
            _ => None,
        }
    }
}

pub(crate) type Strategy = FxHashMap<Box<[u8]>, Vec<f32>>;

/// `RealAct::Add(ALL_IN)` means "all-in" without the caller tracking stacks; it is
/// resolved to the actor's remaining stack during the replay.
pub(crate) const ALL_IN: i64 = i64::MAX;

/// One action of a hand at its public node (`Agent::action_steps`).
pub(crate) struct ActionStep {
    pub actor: usize,
    /// Board cards out when the action was taken.
    pub board_len: usize,
    /// What the actor owed and its stack before the action.
    pub owed: i64,
    pub stack: i64,
    /// The action in canonical form.
    pub act: RealAct,
}

/// One spelling per real action at a node where the actor owes `owed` and
/// holds `stack`: all-in as the stack, adding no more than the price (even
/// all-in for less) as a call, a free call as a check.
pub(crate) fn canonical_act(act: RealAct, owed: i64, stack: i64) -> RealAct {
    match act {
        RealAct::Add(ALL_IN) => RealAct::Add(stack),
        RealAct::Add(d) if d <= owed => RealAct::Call,
        RealAct::Call if owed == 0 => RealAct::Check,
        other => other,
    }
}

/// Where a decision's action came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Source {
    /// Sampled from the blueprint row at the translated infoset.
    Blueprint,
    /// Translated infoset exists in the tree but the blueprint has no row for it.
    MissingKey,
    /// The real hand has left the abstract tree.
    OffTree,
    /// Real-time subgame search (turn/river), not the blueprint row.
    Search,
}

impl Source {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Source::Blueprint => "blueprint",
            Source::MissingKey => "missing_key",
            Source::OffTree => "off_tree",
            Source::Search => "search",
        }
    }
}

/// A real hand as seen by the agent at its decision.
#[derive(Clone)]
pub(crate) struct HandSpec {
    /// Agent's seat: 0 = button / small blind, 1 = big blind.
    pub hero: usize,
    pub hole: [u8; 2],
    /// Board cards dealt so far (0, 3, 4 or 5).
    pub board: Vec<u8>,
    pub sb: i64,
    pub bb: i64,
    /// Starting stacks (before blinds); the effective stack (the smaller) is
    /// what is in play heads-up.
    pub stacks: [i64; 2],
    /// Every action so far, in order (both players).
    pub actions: Vec<RealAct>,
    /// Per-hand seed for the randomized translation and the action sample.
    pub seed: u64,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Decision {
    pub action: RealAct,
    /// Chips the action puts in (0 for fold/check; the call amount for a call).
    pub add: i64,
    /// Whether the action leaves the agent all-in.
    pub all_in: bool,
    pub source: Source,
    /// Abstract infoset key used (blueprint / missing-key cases).
    pub key: Option<String>,
    /// Blueprint action labels and probabilities at that key (blueprint case).
    pub labels: Vec<String>,
    pub probs: Vec<f64>,
    /// Abstract betting history the real hand was translated into.
    pub abstract_history: String,
    /// Where the translation left the tree, if it did; why the search failed
    /// where the blueprint decided in its place; why the external solver was
    /// not used where the built-in one searched instead (`subgame::RESCUED`).
    pub off_tree_reason: Option<String>,
    /// The agent's full policy here as real actions with probabilities (the
    /// blueprint row mapped to chips, identical actions merged; a fallback is a
    /// single action with probability 1). `action` is a sample from it.
    pub options: Vec<(RealAct, f64)>,
}

/// A real action's place in the grid (see `Agent::translate_dist`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Translation {
    Exact(usize),
    Between { lo: usize, hi: usize, pl: f64 },
}

/// Opponent actions whose translation between two grid sizes is fixed instead
/// of drawn from the hand's seed: action `i` (bit `i` of `mask`) is read as the
/// lower size iff bit `i` of `lower` is set. Play never sets it; LBR's
/// peek-free model (lbr.rs, V2LEAK.md) marginalizes over it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Force {
    pub mask: u64,
    pub lower: u64,
}

/// The replayed hand at the hero's turn (see `Agent::locate`).
struct Located {
    real_game: Nlhe,
    real: NlheState,
    abs: NlheState,
    /// The abstract hand is at the same decision as the real one.
    on_tree: bool,
    off_reason: Option<String>,
    /// When the last action is the opponent's and falls between two grid
    /// sizes: the chance it is read as the lower one.
    last_split: Option<f64>,
}

/// What the blueprint reads at a public state that no holding changes: the
/// hand located with some holding and, on the tree, the real action each
/// abstract action maps to.
struct PublicNode {
    base: Located,
    reals: Vec<RealAct>,
}

/// A policy row's action probabilities: its non-negative part normalized,
/// uniform when nothing is positive.
fn normalized(row: &[f64]) -> Vec<f64> {
    let total: f64 = row.iter().map(|&p| p.max(0.0)).sum();
    if total > 0.0 {
        row.iter().map(|&p| p.max(0.0) / total).collect()
    } else {
        vec![1.0 / row.len() as f64; row.len()]
    }
}

/// Probabilities per real action, in first-seen order: actions that map to
/// the same real one summed.
fn merged(acts: impl IntoIterator<Item = RealAct>, probs: &[f64]) -> Vec<(RealAct, f64)> {
    let mut options: Vec<(RealAct, f64)> = Vec::with_capacity(probs.len());
    for (a, &p) in acts.into_iter().zip(probs) {
        match options.iter_mut().find(|(b, _)| *b == a) {
            Some(o) => o.1 += p,
            None => options.push((a, p)),
        }
    }
    options
}

pub(crate) struct Agent {
    game: Arc<Nlhe>,
    /// Any protocol policy: a saved table, or a leak / blend / response built on
    /// one. A row it does not have is a missing key (fallback).
    policy: Policy,
    /// Real-time search for turn/river decisions (`None`: blueprint only).
    search: Option<SearchCfg>,
    /// Solved subgames by public state (see `subgame::search_decide`).
    cache: SearchCache,
    /// How bets above the largest grid size are read.
    oversize: Oversize,
    /// How a preflop open below the smallest grid size is read.
    open_below: OpenBelow,
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in [0, 1) from a hashed stream position.
pub(crate) fn unit(seed: u64, stream: u64, index: u64) -> f64 {
    let z = splitmix(splitmix(seed ^ stream.wrapping_mul(0xA24B_AED4_963E_E407)) ^ index);
    (z >> 11) as f64 / (1u64 << 53) as f64
}

const STREAM_TRANSLATE: u64 = 1;
const STREAM_SAMPLE: u64 = 2;
pub(crate) const STREAM_SEARCH: u64 = 3;

/// Pseudo-harmonic mapping: probability that a pot-fraction bet `x` with
/// `a <= x <= b` is treated as the smaller size `a` (Ganzfried & Sandholm 2013).
pub(crate) fn prob_lower(a: f64, b: f64, x: f64) -> f64 {
    if x <= a {
        return 1.0;
    }
    if x >= b {
        return 0.0;
    }
    ((b - x) * (1.0 + a)) / ((b - a) * (1.0 + x))
}

fn label_name(l: ActLabel) -> String {
    match l {
        ActLabel::Fold => "fold".into(),
        ActLabel::Check => "check".into(),
        ActLabel::Call => "call".into(),
        ActLabel::Bet(f) => format!("bet{f}"),
        ActLabel::AllIn => "allin".into(),
    }
}

thread_local! {
    /// Equity vs a random hand depends only on the cards, and range-tracking
    /// callers (LBR) ask for the same (hole, board) thousands of times.
    static EQUITY_CACHE: std::cell::RefCell<FxHashMap<u128, f64>> = std::cell::RefCell::new(FxHashMap::default());
    /// The turn/river board this thread last read (`board_equity`).
    static LAST_BOARD: std::cell::RefCell<Option<(u64, BoardCell)>> = const { std::cell::RefCell::new(None) };
}

/// One turn/river board's exact equities: how many holdings were asked for,
/// and every holding's (`a * 52 + b`) once the board is worth a batch.
#[derive(Default)]
struct BoardEquity {
    asks: std::sync::atomic::AtomicUsize,
    all: std::sync::OnceLock<Vec<f64>>,
}

type BoardCell = Arc<BoardEquity>;

/// Boards kept by `board_equity` (21 KB each).
const BOARD_EQUITY_LIMIT: usize = 256;

/// Holdings asked for on one board before the whole board is computed.
const BOARD_EQUITY_BATCH_AFTER: usize = 2;

/// Exact equity of `hole` on a turn or river board, or `None` for the caller
/// to enumerate it alone. One holding's exact turn equity is ~45k
/// evaluations and range tracking asks for most holdings on a board, so from
/// the third holding on every holding is computed together, once per board
/// (`equity_vs_random_all`: the same counts, so the same value); a lone
/// decision still pays for its own holding only.
fn board_equity(hole: [u8; 2], bm: u64, board: &[u8]) -> Option<f64> {
    use std::sync::atomic::Ordering;
    static BOARDS: std::sync::LazyLock<std::sync::Mutex<FxHashMap<u64, BoardCell>>> = std::sync::LazyLock::new(Default::default);
    let i = hole[0] as usize * 52 + hole[1] as usize;
    let last = LAST_BOARD.with(|l| match &*l.borrow() {
        Some((k, cell)) if *k == bm => cell.all.get().map(|e| e[i]),
        _ => None,
    });
    if last.is_some() {
        return last;
    }
    let cell = {
        let mut m = BOARDS.lock().unwrap_or_else(|e| e.into_inner());
        if m.len() >= BOARD_EQUITY_LIMIT && !m.contains_key(&bm) {
            m.clear();
        }
        Arc::clone(m.entry(bm).or_default())
    };
    if cell.all.get().is_none() && cell.asks.fetch_add(1, Ordering::Relaxed) < BOARD_EQUITY_BATCH_AFTER {
        return None;
    }
    // Threads asking for the same board wait for the one computing it.
    let e = cell.all.get_or_init(|| {
        let b: Vec<Card> = board.iter().map(|&c| Card::from_index(c)).collect();
        equity::equity_vs_random_all(&b).iter().map(equity::Tally::equity).collect()
    })[i];
    LAST_BOARD.with(|l| *l.borrow_mut() = Some((bm, cell)));
    Some(e)
}

/// Equity of `hole` against a uniformly random holding on `board` (exact on the
/// turn and river, Monte-Carlo before). The Monte-Carlo seed is a function of the
/// cards alone, so the value is deterministic whichever thread computes it.
fn equity_vs_random_cached(hole: [u8; 2], board: &[u8]) -> f64 {
    let hm = (1u64 << hole[0]) | (1u64 << hole[1]);
    let bm = board.iter().fold(0u64, |m, &c| m | 1u64 << c);
    let key = (hm as u128) << 64 | bm as u128;
    if let Some(e) = EQUITY_CACHE.with(|c| c.borrow().get(&key).copied()) {
        return e;
    }
    if board.len() >= 4 {
        if let Some(e) = board_equity(hole, bm, board) {
            return e;
        }
    }
    let h = [Card::from_index(hole[0]), Card::from_index(hole[1])];
    let b: Vec<Card> = board.iter().map(|&c| Card::from_index(c)).collect();
    let e = if b.len() >= 4 {
        equity::equity_vs_random(h, &b).equity()
    } else {
        equity::equity_vs_random_mc(h, &b, FALLBACK_EQUITY_SAMPLES, splitmix(hm ^ splitmix(bm))).equity()
    };
    EQUITY_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() > 1 << 20 {
            c.clear();
        }
        c.insert(key, e);
    });
    e
}

/// Off-tree / missing-key policy: check when free; otherwise call when the hand's
/// equity against a uniformly random holding is at least the price
/// `to_call / (pot + to_call)`, else fold. Never raises. Equity is only defined
/// for real 52-card hold'em; reduced test games fall back to calling a price of at
/// most one third.
pub(crate) fn fallback(game: &Nlhe, real: &NlheState, hole: [u8; 2], board: &[u8]) -> RealAct {
    let c = Nlhe::owed(real);
    if c == 0 {
        return RealAct::Check;
    }
    let pay = c.min(Nlhe::stack_of(real, Nlhe::acting(real)));
    let price = pay as f64 / (Nlhe::pot(real) + pay) as f64;
    let cfg = game.cfg();
    let equity = if cfg.deck.len() == 52 && cfg.board_cards == 5 {
        equity_vs_random_cached(hole, board)
    } else {
        1.0 / 3.0
    };
    if equity >= price {
        RealAct::Call
    } else {
        RealAct::Fold
    }
}

impl Agent {
    pub(crate) fn new(game: Arc<Nlhe>, strategy: Arc<Strategy>) -> Self {
        Self::from_policy(game, Policy::tabular(strategy))
    }

    /// An agent playing any protocol policy over `game` (the policy's own game).
    pub(crate) fn from_policy(game: Arc<Nlhe>, policy: Policy) -> Self {
        Agent { game, policy, search: None, cache: SearchCache::default(), oversize: Oversize::Snap, open_below: OpenBelow::Snap }
    }

    /// The same policy, searching turn/river decisions per `cfg` (the policy
    /// still supplies preflop/flop play and the range model).
    pub(crate) fn with_search(&self, cfg: SearchCfg) -> Self {
        Agent { search: Some(cfg), ..self.with_oversize(self.oversize) }
    }

    /// The same agent reading bets above the largest grid size per `oversize`.
    pub(crate) fn with_oversize(&self, oversize: Oversize) -> Self {
        Agent {
            game: Arc::clone(&self.game),
            policy: self.policy.clone(),
            search: self.search.clone(),
            cache: SearchCache::default(),
            oversize,
            open_below: self.open_below,
        }
    }

    pub(crate) fn oversize(&self) -> Oversize {
        self.oversize
    }

    /// The same agent reading a preflop open below the smallest grid size per
    /// `open_below`.
    pub(crate) fn with_open_below(&self, open_below: OpenBelow) -> Self {
        Agent { open_below, ..self.with_oversize(self.oversize) }
    }

    pub(crate) fn open_below(&self) -> OpenBelow {
        self.open_below
    }

    /// The search configuration, when this agent searches.
    pub(crate) fn search_cfg(&self) -> Option<&SearchCfg> {
        self.search.as_ref()
    }

    /// The solved subgame behind this agent's decision at `spec`: `None` where
    /// it does not search, or its search fails (the blueprint decides there).
    pub(crate) fn searched(&self, spec: &HandSpec) -> Option<Arc<subgame::Solved>> {
        let cfg = self.search.as_ref()?;
        if !subgame::applies(cfg, spec) {
            return None;
        }
        subgame::solved(self, cfg, &self.cache, spec).ok()
    }

    #[cfg(test)]
    pub(crate) fn search_cache_len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// The abstract game's configuration (deck, board length, grid).
    pub(crate) fn cfg(&self) -> &NlheConfig {
        self.game.cfg()
    }

    /// The real-chip twin of the abstract game: same deck and board length, the
    /// hand's own blinds and effective stack.
    pub(crate) fn real_game(&self, spec: &HandSpec) -> Nlhe {
        let base = self.game.cfg();
        Nlhe::new(NlheConfig { sb: spec.sb, bb: spec.bb, stack: spec.stacks[0].min(spec.stacks[1]), ..base.clone() })
    }

    pub(crate) fn validate(&self, spec: &HandSpec) -> Result<(), String> {
        if spec.hero > 1 {
            return Err("hero must be seat 0 (button/SB) or 1 (BB)".into());
        }
        if spec.sb <= 0 || spec.bb < spec.sb {
            return Err(format!("bad blinds sb={} bb={}", spec.sb, spec.bb));
        }
        let eff = spec.stacks[0].min(spec.stacks[1]);
        if eff <= spec.bb {
            return Err(format!("effective stack {eff} must exceed the big blind"));
        }
        let cfg = self.game.cfg();
        if !matches!(spec.board.len(), 0 | 3 | 4 | 5) || spec.board.len() > cfg.board_cards {
            return Err(format!("board has {} cards", spec.board.len()));
        }
        self.check_cards(spec.hole, &spec.board)
    }

    /// `validate`'s card checks: every card in this game's deck, none twice.
    fn check_cards(&self, hole: [u8; 2], board: &[u8]) -> Result<(), String> {
        let cfg = self.game.cfg();
        let mut seen = 0u64;
        for &c in hole.iter().chain(board) {
            if !cfg.deck.contains(&c) {
                return Err(format!("card {c} not in this game's deck"));
            }
            if seen & (1u64 << c) != 0 {
                return Err(format!("card {c} appears twice"));
            }
            seen |= 1u64 << c;
        }
        Ok(())
    }

    /// Replay the hand and decide. Errors describe an impossible input (illegal
    /// action, missing board card, not the agent's turn) — never a guess.
    pub(crate) fn decide(&self, spec: &HandSpec) -> Result<Decision, String> {
        let mut d = self.decide_inner(spec)?;
        // Fill the chip accounting from the real replay of the chosen action.
        let real_game = self.real_game(spec);
        let (real, _) = self.replay_real(&real_game, spec)?;
        let p = Nlhe::acting(&real);
        let before = Nlhe::committed_of(&real, p);
        let after = real_game
            .apply_real(&real, d.action)
            .map_err(|e| format!("internal: agent chose an illegal action {:?}: {e}", d.action))?;
        d.add = Nlhe::committed_of(&after, p) - before;
        d.all_in = Nlhe::stack_of(&after, p) == 0;
        Ok(d)
    }

    /// Real-chip replay only (no translation): the state at the agent's turn.
    pub(crate) fn replay_real(&self, real_game: &Nlhe, spec: &HandSpec) -> Result<(NlheState, usize), String> {
        let mut real = real_game.initial_state();
        let used = spec.hole.iter().chain(&spec.board).fold(0u64, |m, &c| m | 1u64 << c);
        let mut dummies = self.game.cfg().deck.iter().copied().filter(|&c| used & (1u64 << c) == 0);
        let dummy = [dummies.next().ok_or("deck too small")?, dummies.next().ok_or("deck too small")?];
        let holes = if spec.hero == 0 { [spec.hole, dummy] } else { [dummy, spec.hole] };
        for seat in holes {
            for c in seat {
                real = real_game.deal_specific(&real, Card::from_index(c));
            }
        }
        let mut next = 0usize;
        for &act in spec.actions.iter() {
            while Nlhe::is_dealing(&real) {
                let card = *spec.board.get(next).ok_or("board card missing")?;
                next += 1;
                real = real_game.deal_specific(&real, Card::from_index(card));
            }
            let act = match act {
                RealAct::Add(ALL_IN) => RealAct::Add(Nlhe::stack_of(&real, Nlhe::acting(&real))),
                other => other,
            };
            real = real_game.apply_real(&real, act)?;
        }
        while Nlhe::is_dealing(&real) {
            let card = *spec.board.get(next).ok_or("board card missing")?;
            next += 1;
            real = real_game.deal_specific(&real, Card::from_index(card));
        }
        Ok((real, next))
    }

    /// Replay the hand on the real-chip game and, while it stays on the grid, on
    /// the abstract game (translating every real action): the state at the
    /// hero's turn.
    /// Both seats' hole cards for replaying `spec`: the hero's, and for the
    /// opponent (unknown) the first two deck cards the agent cannot see, which
    /// keeps the deal legal. Only the agent's key is ever read.
    fn seat_holes(&self, spec: &HandSpec) -> Result<[[u8; 2]; 2], String> {
        self.holes_for(spec.hero, spec.hole, &spec.board)
    }

    /// `seat_holes` of a spec with this hero, hole and board.
    fn holes_for(&self, hero: usize, hole: [u8; 2], board: &[u8]) -> Result<[[u8; 2]; 2], String> {
        let used = hole.iter().chain(board).fold(0u64, |m, &c| m | 1u64 << c);
        let mut dummies = self.game.cfg().deck.iter().copied().filter(|&c| used & (1u64 << c) == 0);
        let dummy = [
            dummies.next().ok_or("deck too small")?,
            dummies.next().ok_or("deck too small")?,
        ];
        Ok(if hero == 0 { [hole, dummy] } else { [dummy, hole] })
    }

    /// `locate(spec)` from `base`, the same public hand located with other
    /// hole cards: the betting and its translation do not depend on the cards,
    /// so only the holes are swapped in — no replay.
    fn relocate(&self, base: &Located, spec: &HandSpec) -> Result<Located, String> {
        self.validate(spec)?;
        let c = |s: [u8; 2]| [Card::from_index(s[0]), Card::from_index(s[1])];
        let [h0, h1] = self.seat_holes(spec)?;
        let holes = [c(h0), c(h1)];
        Ok(Located {
            real_game: self.real_game(spec),
            real: Nlhe::with_holes(&base.real, holes),
            abs: Nlhe::with_holes(&base.abs, holes),
            on_tree: base.on_tree,
            off_reason: base.off_reason.clone(),
            last_split: base.last_split,
        })
    }

    fn locate(&self, spec: &HandSpec) -> Result<Located, String> {
        self.locate_forced(spec, Force::default())
    }

    /// `locate` with the opponent's translations fixed where `force` says.
    fn locate_forced(&self, spec: &HandSpec, force: Force) -> Result<Located, String> {
        self.validate(spec)?;
        let real_game = self.real_game(spec);
        let abs_game: &Nlhe = &self.game;
        let holes = self.seat_holes(spec)?;

        let mut real = real_game.initial_state();
        let mut abs = abs_game.initial_state();
        for seat in holes {
            for c in seat {
                real = real_game.deal_specific(&real, Card::from_index(c));
                abs = abs_game.deal_specific(&abs, Card::from_index(c));
            }
        }

        let mut synced = true;
        let mut off_reason: Option<String> = None;
        let mut last_split = None;
        let mut board_next = 0usize;
        // The last action read as an all-in stand-in (`Oversize::Standin`): the
        // abstract state before it and the largest grid size there.
        let mut standin: Option<(NlheState, usize)> = None;

        for (i, &act) in spec.actions.iter().enumerate() {
            self.deal_board(&real_game, &mut real, &mut abs, &mut synced, spec, &mut board_next)?;
            if Nlhe::betting_street(&real).is_none() {
                return Err(format!("action {i} given after the hand ended"));
            }
            let actor = Nlhe::acting(&real);
            let act = match act {
                RealAct::Add(ALL_IN) => RealAct::Add(Nlhe::stack_of(&real, actor)),
                other => other,
            };
            last_split = None;
            if synced {
                if let Some((before, lo)) = standin.take() {
                    if act != RealAct::Fold {
                        abs = abs_game.next_state(&before, lo);
                    }
                }
                if Nlhe::betting_street(&abs) != Nlhe::betting_street(&real)
                    || Nlhe::acting(&abs) != actor
                {
                    synced = false;
                    off_reason = Some(format!("abstract line ended or diverged before action {i}"));
                } else {
                    match self.translate(&real_game, &real, &abs, act, actor == spec.hero, spec.seed, i as u64, force) {
                        Some((idx, split)) => {
                            standin = self.standin_of(&real, &abs, act, idx).map(|lo| (abs.clone(), lo));
                            abs = abs_game.next_state(&abs, idx);
                            last_split = split;
                        }
                        None => {
                            synced = false;
                            off_reason = Some(format!("action {i} ({act:?}) has no abstract counterpart"));
                        }
                    }
                }
            }
            real = real_game
                .apply_real(&real, act)
                .map_err(|e| format!("illegal action {i} ({act:?}): {e}"))?;
        }
        self.deal_board(&real_game, &mut real, &mut abs, &mut synced, spec, &mut board_next)?;
        if board_next != spec.board.len() {
            return Err(format!(
                "{} board cards given but the betting has only reached {}",
                spec.board.len(),
                board_next
            ));
        }
        if Nlhe::betting_street(&real).is_none() || Nlhe::acting(&real) != spec.hero {
            return Err("it is not the agent's turn".into());
        }

        let on_tree = synced
            && Nlhe::betting_street(&abs) == Nlhe::betting_street(&real)
            && Nlhe::acting(&abs) == spec.hero;
        Ok(Located { real_game, real, abs, on_tree, off_reason, last_split })
    }

    /// For each action in `spec.actions`: the seat that took it and how many
    /// board cards were out at the time.
    pub(crate) fn action_context(&self, spec: &HandSpec) -> Result<Vec<(usize, usize)>, String> {
        Ok(self.action_steps(spec)?.into_iter().map(|s| (s.actor, s.board_len)).collect())
    }

    /// `action_context` plus each action's public node: what the actor owed,
    /// its stack, and the action in canonical form (`canonical_act`).
    pub(crate) fn action_steps(&self, spec: &HandSpec) -> Result<Vec<ActionStep>, String> {
        self.validate(spec)?;
        let real_game = self.real_game(spec);
        let mut real = real_game.initial_state();
        let used = spec.hole.iter().chain(&spec.board).fold(0u64, |m, &c| m | 1u64 << c);
        let mut dummies = self.game.cfg().deck.iter().copied().filter(|&c| used & (1u64 << c) == 0);
        let dummy = [dummies.next().ok_or("deck too small")?, dummies.next().ok_or("deck too small")?];
        let holes = if spec.hero == 0 { [spec.hole, dummy] } else { [dummy, spec.hole] };
        for seat in holes {
            for c in seat {
                real = real_game.deal_specific(&real, Card::from_index(c));
            }
        }
        let mut next = 0usize;
        let mut out = Vec::with_capacity(spec.actions.len());
        for (i, &act) in spec.actions.iter().enumerate() {
            while Nlhe::is_dealing(&real) {
                let card = *spec.board.get(next).ok_or("board card missing")?;
                next += 1;
                real = real_game.deal_specific(&real, Card::from_index(card));
            }
            if Nlhe::betting_street(&real).is_none() {
                return Err(format!("action {i} given after the hand ended"));
            }
            let actor = Nlhe::acting(&real);
            let (owed, stack) = (Nlhe::owed(&real), Nlhe::stack_of(&real, actor));
            out.push(ActionStep { actor, board_len: next, owed, stack, act: canonical_act(act, owed, stack) });
            let act = match act {
                RealAct::Add(ALL_IN) => RealAct::Add(stack),
                other => other,
            };
            real = real_game.apply_real(&real, act).map_err(|e| format!("illegal action {i} ({act:?}): {e}"))?;
        }
        Ok(out)
    }

    /// Probability that the player to act (`spec.hero`, holding `spec.hole`)
    /// takes the real action `act` under this agent's policy — the likelihood
    /// used for Bayesian range tracking. An action the policy can output is
    /// scored exactly (every abstract action producing it); any other size by
    /// its translation into the grid (for a size between two grid sizes, the
    /// pseudo-harmonic mixture of their probabilities); fallback decisions
    /// score 1 for the fallback's action and 0 otherwise.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn action_likelihood(&self, spec: &HandSpec, act: RealAct) -> Result<f64, String> {
        self.likelihood_at(&self.locate(spec)?, spec, act)
    }

    /// `action_likelihood` for every holding in `holes` at `spec`'s public
    /// state (`spec.hole` is ignored). The public work — the replay, the
    /// action's real and abstract counterparts — is done once; each holding
    /// only reads its row.
    pub(crate) fn action_likelihoods(&self, spec: &HandSpec, act: RealAct, holes: &[[u8; 2]]) -> Result<Vec<f64>, String> {
        let Some(&first) = holes.first() else { return Ok(Vec::new()) };
        let node = self.public_node(spec, first)?;
        let (real_game, real) = (&node.base.real_game, &node.base.real);
        let act = canonical_act(act, Nlhe::owed(real), Nlhe::stack_of(real, spec.hero));
        let exact = node.reals.contains(&act);
        let translation = if node.base.on_tree && !exact { self.translate_dist(real_game, real, &node.base.abs, act) } else { None };
        holes
            .par_iter()
            .map_init(
                || (Vec::new(), Vec::new()),
                |(key, row), &h| {
                    if !self.row_at(&node, spec, h, key, row)? {
                        return Ok(if fallback(&self.game, real, h, &spec.board) == act { 1.0 } else { 0.0 });
                    }
                    let probs = normalized(row);
                    if exact {
                        let mut direct = 0.0;
                        for (&a, &p) in node.reals.iter().zip(&probs) {
                            if a == act {
                                direct += p;
                            }
                        }
                        return Ok(direct);
                    }
                    Ok(match translation {
                        None => 0.0,
                        Some(Translation::Exact(i)) => probs[i],
                        Some(Translation::Between { lo, hi, pl }) => pl * probs[lo] + (1.0 - pl) * probs[hi],
                    })
                },
            )
            .collect()
    }

    /// `spec`'s public state, located with the holding `hole`.
    fn public_node(&self, spec: &HandSpec, hole: [u8; 2]) -> Result<PublicNode, String> {
        self.public_node_forced(spec, hole, Force::default())
    }

    /// `public_node` with the opponent's translations fixed where `force` says.
    fn public_node_forced(&self, spec: &HandSpec, hole: [u8; 2], force: Force) -> Result<PublicNode, String> {
        let base = self.locate_forced(&HandSpec { hole, ..spec.clone() }, force)?;
        let reals = if base.on_tree {
            self.game.labeled_actions(&base.abs).iter().map(|&l| self.to_real(&base.real_game, &base.real, &base.abs, l)).collect()
        } else {
            Vec::new()
        };
        Ok(PublicNode { base, reals })
    }

    /// The blueprint's row for holding `h` at `node` (into `row`, the key
    /// built in `key`); `false` where the fallback decides instead (off the
    /// tree, or no row for the key). Errors as `relocate` would.
    fn row_at(&self, node: &PublicNode, spec: &HandSpec, h: [u8; 2], key: &mut Vec<u8>, row: &mut Vec<f64>) -> Result<bool, String> {
        self.check_cards(h, &spec.board)?;
        let [h0, h1] = self.holes_for(spec.hero, h, &spec.board)?;
        if !node.base.on_tree {
            return Ok(false);
        }
        let c = |s: [u8; 2]| [Card::from_index(s[0]), Card::from_index(s[1])];
        let abs = Nlhe::with_holes(&node.base.abs, [c(h0), c(h1)]);
        key.clear();
        self.game.infoset_key_into(&abs, key);
        row.clear();
        row.resize(node.reals.len(), 0.0);
        Ok(self.policy.try_fill(&self.game, &abs, key, row))
    }

    fn likelihood_at(&self, loc: &Located, spec: &HandSpec, act: RealAct) -> Result<f64, String> {
        let act = canonical_act(act, Nlhe::owed(&loc.real), Nlhe::stack_of(&loc.real, spec.hero));
        let fallback_score = || {
            let f = fallback(&self.game, &loc.real, spec.hole, &spec.board);
            if f == act {
                1.0
            } else {
                0.0
            }
        };
        if !loc.on_tree {
            return Ok(fallback_score());
        }
        let key = self.game.infoset_key(&loc.abs);
        let labels = self.game.labeled_actions(&loc.abs);
        let mut row = vec![0.0f64; labels.len()];
        if !self.policy.try_fill(&self.game, &loc.abs, key.as_bytes(), &mut row) {
            return Ok(fallback_score());
        }
        let probs = normalized(&row);
        let mut exact = false;
        let mut direct = 0.0;
        for (&l, &p) in labels.iter().zip(&probs) {
            if self.to_real(&loc.real_game, &loc.real, &loc.abs, l) == act {
                exact = true;
                direct += p;
            }
        }
        if exact {
            return Ok(direct);
        }
        Ok(match self.translate_dist(&loc.real_game, &loc.real, &loc.abs, act) {
            None => 0.0,
            Some(Translation::Exact(i)) => probs[i],
            Some(Translation::Between { lo, hi, pl }) => pl * probs[lo] + (1.0 - pl) * probs[hi],
        })
    }

    /// `decide` without the chip accounting (`add`/`all_in` left at 0): the fast
    /// path for callers that only need the policy, e.g. range tracking.
    pub(crate) fn decide_inner(&self, spec: &HandSpec) -> Result<Decision, String> {
        let mut search_error = None;
        if let Some(cfg) = &self.search {
            if subgame::applies(cfg, spec) {
                match subgame::search_decide(self, cfg, &self.cache, spec) {
                    Ok(d) => return Ok(d),
                    Err(e) => search_error = Some(format!("search failed, blueprint used: {e}")),
                }
            }
        }
        let mut d = self.blueprint_decide(spec)?;
        if search_error.is_some() {
            d.off_tree_reason = search_error;
        }
        Ok(d)
    }

    /// `decide_inner` for every holding in `holes` at `spec`'s public state
    /// (`spec.hole` is ignored). A searched state is solved once, then read
    /// per holding; a blueprint state is replayed once, not once per holding.
    pub(crate) fn decide_inner_many(&self, spec: &HandSpec, holes: &[[u8; 2]]) -> Result<Vec<Decision>, String> {
        let Some(&first) = holes.first() else { return Ok(Vec::new()) };
        let at = |h: [u8; 2]| HandSpec { hole: h, ..spec.clone() };
        if self.search.as_ref().is_some_and(|cfg| subgame::applies(cfg, spec)) {
            // One solve serves every holding: solve it before fanning out
            // (racing threads would each solve it).
            self.decide_inner(&at(first))?;
            return holes.par_iter().map(|&h| self.decide_inner(&at(h))).collect();
        }
        let base = self.locate(&at(first))?;
        holes
            .par_iter()
            .map(|&h| {
                let s = at(h);
                Ok(self.blueprint_at(self.relocate(&base, &s)?, &s))
            })
            .collect()
    }

    /// Each holding's `decide_inner` options (`spec.hole` is ignored): the
    /// numbers `decide_inner_many` gives, without a full `Decision` per
    /// holding. The public work — the replay, the abstract actions' real
    /// sizes, the solve — is done once; each holding only reads its row.
    pub(crate) fn options_many(&self, spec: &HandSpec, holes: &[[u8; 2]]) -> Result<Vec<Vec<(RealAct, f64)>>, String> {
        self.options_many_failure(spec, holes).map(|(options, _)| options)
    }

    /// `options_many`, and whether a failed search decided them, as an index
    /// into `subgame::search_failure`'s `[built-in solver used in place of
    /// the external one, blueprint used]`.
    #[allow(clippy::type_complexity)]
    pub(crate) fn options_many_failure(&self, spec: &HandSpec, holes: &[[u8; 2]]) -> Result<(Vec<Vec<(RealAct, f64)>>, Option<usize>), String> {
        let Some(&first) = holes.first() else { return Ok((Vec::new(), None)) };
        let at = |h: [u8; 2]| HandSpec { hole: h, ..spec.clone() };
        if let Some(cfg) = self.search.as_ref().filter(|cfg| subgame::applies(cfg, spec)) {
            let Ok(solved) = subgame::solved(self, cfg, &self.cache, &at(first)) else {
                return Ok((self.decide_inner_many(spec, holes)?.into_iter().map(|d| d.options).collect(), Some(1)));
            };
            let options = holes
                .par_iter()
                .map(|&h| match solved.row(h) {
                    Some(probs) => Ok(merged(solved.acts().iter().copied(), &probs)),
                    // Not in the subgame (the hole collides with the board):
                    // what `decide_inner` does there.
                    None => self.decide_inner(&at(h)).map(|d| d.options),
                })
                .collect::<Result<_, String>>()?;
            return Ok((options, solved.failure()));
        }
        Ok((self.options_at(&self.public_node(spec, first)?, spec, holes)?, None))
    }

    /// `options_many` with the opponent's translations fixed where `force`
    /// says (blueprint play only: where the agent would search it is an error).
    pub(crate) fn options_many_forced(
        &self,
        spec: &HandSpec,
        holes: &[[u8; 2]],
        force: Force,
    ) -> Result<Vec<Vec<(RealAct, f64)>>, String> {
        if force == Force::default() {
            return self.options_many(spec, holes);
        }
        let Some(&first) = holes.first() else { return Ok(Vec::new()) };
        if self.search.as_ref().is_some_and(|cfg| subgame::applies(cfg, spec)) {
            return Err("forced translations are not supported where the agent searches".into());
        }
        self.options_at(&self.public_node_forced(spec, first, force)?, spec, holes)
    }

    /// The blueprint options of every holding in `holes` at `node`.
    fn options_at(&self, node: &PublicNode, spec: &HandSpec, holes: &[[u8; 2]]) -> Result<Vec<Vec<(RealAct, f64)>>, String> {
        holes
            .par_iter()
            .map_init(
                || (Vec::new(), Vec::new()),
                |(key, row), &h| {
                    Ok(if self.row_at(node, spec, h, key, row)? {
                        merged(node.reals.iter().copied(), &normalized(row))
                    } else {
                        vec![(fallback(&self.game, &node.base.real, h, &spec.board), 1.0)]
                    })
                },
            )
            .collect()
    }

    /// When `spec`'s last action is the opponent's and, with the earlier
    /// translations fixed by `force`, falls between two grid sizes: the chance
    /// the agent reads it as the lower one.
    pub(crate) fn translation_split(&self, spec: &HandSpec, force: Force) -> Result<Option<f64>, String> {
        Ok(self.locate_forced(spec, force)?.last_split)
    }

    /// The blueprint's decision (translation, row lookup, fallback).
    fn blueprint_decide(&self, spec: &HandSpec) -> Result<Decision, String> {
        Ok(self.blueprint_at(self.locate(spec)?, spec))
    }

    fn blueprint_at(&self, loc: Located, spec: &HandSpec) -> Decision {
        let Located { real_game, real, abs, on_tree, off_reason, .. } = loc;
        let abs_game: &Nlhe = &self.game;
        let abstract_history = Nlhe::betting_history(&abs);
        if !on_tree {
            let reason = off_reason.unwrap_or_else(|| "abstract line ended or diverged".into());
            let action = fallback(&self.game, &real, spec.hole, &spec.board);
            return Decision {
                action,
                options: vec![(action, 1.0)],
                add: 0,
                all_in: false,
                source: Source::OffTree,
                key: None,
                labels: vec![],
                probs: vec![],
                abstract_history,
                off_tree_reason: Some(reason),
            };
        }

        let key = abs_game.infoset_key(&abs);
        let labels = abs_game.labeled_actions(&abs);
        let mut row = vec![0.0f64; labels.len()];
        if !self.policy.try_fill(abs_game, &abs, key.as_bytes(), &mut row) {
            {
                let action = fallback(&self.game, &real, spec.hole, &spec.board);
                return Decision {
                    action,
                    options: vec![(action, 1.0)],
                    add: 0,
                    all_in: false,
                    source: Source::MissingKey,
                    key: Some(key),
                    labels: labels.iter().map(|&l| label_name(l)).collect(),
                    probs: vec![],
                    abstract_history,
                    off_tree_reason: None,
                };
            }
        }
        let probs = normalized(&row);
        let r = unit(spec.seed, STREAM_SAMPLE, spec.actions.len() as u64);
        let mut acc = 0.0;
        let mut chosen = probs.len() - 1;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                chosen = i;
                break;
            }
        }
        let action = self.to_real(&real_game, &real, &abs, labels[chosen]);
        debug_assert!(real_game.apply_real(&real, action).is_ok(), "agent produced an illegal action");
        let options = merged(labels.iter().map(|&l| self.to_real(&real_game, &real, &abs, l)), &probs);
        Decision {
            action,
            options,
            add: 0,
            all_in: false,
            source: Source::Blueprint,
            key: Some(key),
            labels: labels.iter().map(|&l| label_name(l)).collect(),
            probs,
            abstract_history,
            off_tree_reason: None,
        }
    }

    /// Deal every board card the real hand has reached, mirroring into the
    /// abstract hand while it is still on the same street.
    fn deal_board(
        &self,
        real_game: &Nlhe,
        real: &mut NlheState,
        abs: &mut NlheState,
        synced: &mut bool,
        spec: &HandSpec,
        next: &mut usize,
    ) -> Result<(), String> {
        while Nlhe::is_dealing(real) {
            let card = *spec.board.get(*next).ok_or_else(|| {
                format!("the betting has reached board card {} but it was not given", *next + 1)
            })?;
            *next += 1;
            *real = real_game.deal_specific(real, Card::from_index(card));
            if *synced {
                if Nlhe::is_dealing(abs) && Nlhe::board_len(abs) + 1 == Nlhe::board_len(real) {
                    *abs = self.game.deal_specific(abs, Card::from_index(card));
                } else {
                    *synced = false;
                }
            }
        }
        Ok(())
    }

    /// How a real action at a synced node maps into the grid: one abstract
    /// action, or two neighbouring sizes with the pseudo-harmonic probability of
    /// the lower one; `None` when the grid has no counterpart.
    fn translate_dist(&self, real_game: &Nlhe, real: &NlheState, abs: &NlheState, act: RealAct) -> Option<Translation> {
        let labels = self.game.labeled_actions(abs);
        let find = |want: fn(&ActLabel) -> bool| labels.iter().position(want);
        let fold = find(|l| matches!(l, ActLabel::Fold));
        let check = find(|l| matches!(l, ActLabel::Check));
        let call = find(|l| matches!(l, ActLabel::Call));
        let allin = find(|l| matches!(l, ActLabel::AllIn));
        let p = Nlhe::acting(real);
        let c = Nlhe::owed(real);
        let stack = Nlhe::stack_of(real, p);
        let calls = |d: i64| d <= c || c >= stack;
        // `OpenBelow::Limp`: the big blind's real call of an open read as a
        // limp, where the abstract hand owes nothing, is its check.
        let limped = self.open_below == OpenBelow::Limp && Nlhe::betting_street(real) == Some(0) && Nlhe::owed(abs) == 0;
        let call = call.or_else(|| check.filter(|_| limped));
        match act {
            RealAct::Fold => fold.map(Translation::Exact),
            RealAct::Check => check.map(Translation::Exact),
            RealAct::Call => call.or(allin).map(Translation::Exact),
            RealAct::Add(d) if calls(d) => {
                if d >= stack {
                    // Calling (possibly for less) with the whole stack.
                    call.filter(|_| Nlhe::owed(abs) < Nlhe::stack_of(abs, p)).or(allin).or(call).map(Translation::Exact)
                } else {
                    call.or(allin).map(Translation::Exact)
                }
            }
            RealAct::Add(d) => {
                if d >= stack {
                    return allin.map(Translation::Exact);
                }
                let sized: Vec<(f64, usize)> = labels
                    .iter()
                    .enumerate()
                    .filter_map(|(i, l)| match l {
                        ActLabel::Bet(f) if f.is_finite() => Some((*f, i)),
                        _ => None,
                    })
                    .collect();
                // Exact grid amount at the real table (the common case for bots
                // playing the grid, and always for the agent's own bets).
                if let Some(&(_, i)) = sized.iter().find(|&&(f, _)| real_game.grid_delta(real, f) == d) {
                    return Some(Translation::Exact(i));
                }
                let pot = Nlhe::pot(real) as f64;
                let x = if c == 0 {
                    d as f64 / pot
                } else {
                    (d - c) as f64 / (pot + c as f64)
                };
                let mut cands = sized;
                if let Some(ai) = allin {
                    if self.oversize == Oversize::Standin
                        || d as f64 >= ALLIN_NEIGHBOUR_COMMIT * stack as f64
                        || cands.is_empty()
                    {
                        let (apot, ac) = (Nlhe::pot(abs) as f64, Nlhe::owed(abs));
                        let astack = Nlhe::stack_of(abs, p);
                        let af = if ac == 0 {
                            astack as f64 / apot
                        } else {
                            (astack - ac) as f64 / (apot + ac as f64)
                        };
                        cands.push((af, ai));
                    }
                }
                if cands.is_empty() {
                    return None;
                }
                cands.sort_by(|a, b| a.0.total_cmp(&b.0));
                // `OpenBelow::Limp`: the small blind's open below the smallest
                // grid size, read between the limp and that size.
                if self.open_below == OpenBelow::Limp && x < cands[0].0 && matches!(labels[cands[0].1], ActLabel::Bet(_)) {
                    let cfg = real_game.cfg();
                    let first = Nlhe::betting_street(real) == Some(0) && p == 0 && Nlhe::pot(real) == cfg.sb + cfg.bb;
                    if let Some(lo) = call.filter(|_| first) {
                        return Some(Translation::Between { lo, hi: cands[0].1, pl: prob_lower(0.0, cands[0].0, x) });
                    }
                }
                if x <= cands[0].0 {
                    return Some(Translation::Exact(cands[0].1));
                }
                let last = cands[cands.len() - 1];
                if x >= last.0 {
                    return Some(Translation::Exact(last.1));
                }
                let k = cands.iter().position(|&(f, _)| f > x).unwrap();
                let (lo, hi) = (cands[k - 1], cands[k]);
                let pl = prob_lower(lo.0, hi.0, x);
                Some(Translation::Between { lo: lo.1, hi: hi.1, pl })
            }
        }
    }

    /// Under `Oversize::Standin`, when abstract action `idx` at `abs` reads the
    /// real bet `act` (taken at `real`) as all-in although it is not one: the
    /// index of the largest grid size, which the bet is replayed as unless the
    /// response is a fold.
    fn standin_of(&self, real: &NlheState, abs: &NlheState, act: RealAct, idx: usize) -> Option<usize> {
        if self.oversize != Oversize::Standin {
            return None;
        }
        let RealAct::Add(d) = act else { return None };
        let (c, stack) = (Nlhe::owed(real), Nlhe::stack_of(real, Nlhe::acting(real)));
        if d <= c || c >= stack || d >= stack {
            return None; // a call, or a real all-in
        }
        let labels = self.game.labeled_actions(abs);
        if !matches!(labels.get(idx), Some(ActLabel::AllIn)) {
            return None;
        }
        labels
            .iter()
            .enumerate()
            .filter_map(|(i, l)| match l {
                ActLabel::Bet(f) if f.is_finite() => Some((*f, i)),
                _ => None,
            })
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, i)| i)
    }

    /// The abstract action index for a real action at a synced node, or `None`
    /// when the grid has no counterpart. Between two sizes, an opponent's action
    /// is randomized (seeded per hand, or as `force` fixes it) and the agent's
    /// own is deterministic; the second value is then the opponent's chance of
    /// being read as the lower size.
    #[allow(clippy::too_many_arguments)]
    fn translate(
        &self,
        real_game: &Nlhe,
        real: &NlheState,
        abs: &NlheState,
        act: RealAct,
        own: bool,
        seed: u64,
        index: u64,
        force: Force,
    ) -> Option<(usize, Option<f64>)> {
        match self.translate_dist(real_game, real, abs, act)? {
            Translation::Exact(i) => Some((i, None)),
            Translation::Between { lo, hi, pl } => {
                if own {
                    return Some((if pl >= 0.5 { lo } else { hi }, None));
                }
                let pick_lower = if index < 64 && force.mask >> index & 1 == 1 {
                    force.lower >> index & 1 == 1
                } else {
                    unit(seed, STREAM_TRANSLATE, index) < pl
                };
                Some((if pick_lower { lo } else { hi }, Some(pl)))
            }
        }
    }

    /// Convert a chosen abstract action at `abs` into a legal real action.
    fn to_real(&self, real_game: &Nlhe, real: &NlheState, abs: &NlheState, label: ActLabel) -> RealAct {
        let c = Nlhe::owed(real);
        let stack = Nlhe::stack_of(real, Nlhe::acting(real));
        match label {
            ActLabel::Fold if c > 0 => RealAct::Fold,
            ActLabel::Fold | ActLabel::Check | ActLabel::Call => {
                if c == 0 {
                    RealAct::Check
                } else {
                    RealAct::Call
                }
            }
            // Shoving when the stack does not exceed the price is just a call.
            ActLabel::AllIn if stack <= c => RealAct::Call,
            // So is calling a bet read as all-in that is not one: at the all-in
            // node the call is labelled AllIn (TRANSLATE.md: the agent calls).
            ActLabel::AllIn if c > 0 && Nlhe::owed(abs) >= Nlhe::stack_of(abs, Nlhe::acting(abs)) => RealAct::Call,
            ActLabel::AllIn => RealAct::Add(stack),
            ActLabel::Bet(f) => {
                let d = real_game.grid_delta(real, f).max(real_game.min_raise_add(real));
                if d >= stack {
                    RealAct::Add(stack)
                } else {
                    RealAct::Add(d)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_equity_is_the_exact_enumeration_on_the_turn_and_river() {
        let turn = [3u8, 17, 30, 44];
        let river = [3u8, 17, 30, 44, 51];
        // The first holdings on a board are enumerated alone, the rest read the
        // board's batch; alternating boards leave the thread's last board.
        for board in [&turn[..], &river[..], &turn[..]] {
            for hole in [[0u8, 1], [50, 12], [12, 50], [22, 23], [23, 22], [5, 40], [40, 6], [31, 2]] {
                let h = [Card::from_index(hole[0]), Card::from_index(hole[1])];
                let b: Vec<Card> = board.iter().map(|&c| Card::from_index(c)).collect();
                let exact = equity::equity_vs_random(h, &b).equity();
                assert_eq!(equity_vs_random_cached(hole, board).to_bits(), exact.to_bits(), "{hole:?} on {board:?}");
            }
        }
    }

    fn agent_100bb(strategy: Strategy) -> Agent {
        Agent::new(Arc::new(Nlhe::new(NlheConfig::pluribus_100bb())), Arc::new(strategy))
    }

    /// SplitMix stream for the randomized tests.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(1);
            splitmix(self.0)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn deal_cards(rng: &mut Rng) -> Vec<u8> {
        let mut deck: Vec<u8> = (0..52).collect();
        for i in (1..deck.len()).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            deck.swap(i, j);
        }
        deck.truncate(9);
        deck
    }

    #[test]
    fn pseudo_harmonic_endpoints_and_monotone() {
        assert_eq!(prob_lower(0.5, 1.0, 0.5), 1.0);
        assert_eq!(prob_lower(0.5, 1.0, 1.0), 0.0);
        let mut prev = 1.0;
        for i in 1..100 {
            let x = 0.5 + 0.5 * i as f64 / 100.0;
            let p = prob_lower(0.5, 1.0, x);
            assert!(p < prev && p > 0.0);
            prev = p;
        }
        // Published property: the harmonic midpoint of pot fractions maps 50/50,
        // e.g. between 0.5x and 1x pot it is x = 5/7 (not 0.75).
        assert!((prob_lower(0.5, 1.0, 5.0 / 7.0) - 0.5).abs() < 1e-12);
    }

    /// Play random abstract lines of the real 100bb grid; feeding the SAME lines
    /// as real chips (same blinds and stacks, and a 1/2-blind rescale) must
    /// translate back to exactly the abstract history, with the agent on-tree.
    #[test]
    fn on_grid_lines_translate_exactly_at_any_scale() {
        let g = Nlhe::new(NlheConfig::pluribus_100bb());
        let mut rng = Rng(11);
        let mut checked = 0;
        for _ in 0..3000 {
            let cards = deal_cards(&mut rng);
            let hero = rng.below(2) as usize;
            let mut s = g.initial_state();
            for &c in &cards[..4] {
                s = g.deal_specific(&s, Card::from_index(c));
            }
            let mut board = Vec::new();
            let mut real_acts = Vec::new();
            let mut steps = rng.below(12);
            loop {
                while Nlhe::is_dealing(&s) {
                    let c = cards[4 + board.len()];
                    board.push(c);
                    s = g.deal_specific(&s, Card::from_index(c));
                }
                if g.is_terminal(&s) {
                    break;
                }
                if steps == 0 && Nlhe::acting(&s) == hero {
                    break;
                }
                steps = steps.saturating_sub(1);
                let labels = g.labeled_actions(&s);
                let i = rng.below(labels.len() as u64) as usize;
                let before = Nlhe::committed_of(&s, Nlhe::acting(&s));
                let act = labels[i];
                let ns = g.next_state(&s, i);
                let added = Nlhe::committed_of(&ns, Nlhe::acting(&s)) - before;
                real_acts.push(match act {
                    ActLabel::Fold => RealAct::Fold,
                    ActLabel::Check => RealAct::Check,
                    ActLabel::Call => RealAct::Call,
                    ActLabel::Bet(_) | ActLabel::AllIn => RealAct::Add(added),
                });
                s = ns;
            }
            if g.is_terminal(&s) || Nlhe::acting(&s) != hero {
                continue;
            }
            let hole = [cards[2 * hero], cards[2 * hero + 1]];
            for scale in [1i64, 50] {
                // scale 50: blinds 1/2 with every amount divided by 50.
                let acts: Vec<RealAct> = real_acts
                    .iter()
                    .map(|a| match *a {
                        RealAct::Add(d) => RealAct::Add(d / scale),
                        other => other,
                    })
                    .collect();
                if scale != 1 && real_acts.iter().any(|a| matches!(a, RealAct::Add(d) if d % scale != 0)) {
                    continue; // rounding makes the rescaled line a different line
                }
                let spec = HandSpec {
                    hero,
                    hole,
                    board: board.clone(),
                    sb: 50 / scale,
                    bb: 100 / scale,
                    stacks: [10_000 / scale, 10_000 / scale],
                    actions: acts,
                    seed: 7,
                };
                for oversize in [Oversize::Snap, Oversize::Standin] {
                    let agent = agent_100bb(Strategy::default()).with_oversize(oversize);
                    let d = agent.decide(&spec).expect("valid on-grid hand");
                    assert_ne!(d.source, Source::OffTree, "on-grid line left the tree: {:?}", d.off_tree_reason);
                    assert_eq!(d.abstract_history, Nlhe::betting_history(&s));
                    assert_eq!(d.key.as_deref(), Some(g.infoset_key(&s).as_str()));
                }
                checked += 1;
            }
        }
        assert!(checked > 1000, "only {checked} decisions checked");
    }

    /// Random REAL hands with arbitrary legal sizes (min-raises, overbets,
    /// odd amounts, short stacks): every decision must succeed and be legal.
    #[test]
    fn arbitrary_real_hands_always_get_a_legal_action() {
        // A uniform policy has a row everywhere (blueprint path); an empty table
        // has none (missing-key path); off-grid play produces off-tree.
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let uniform = Agent::from_policy(Arc::clone(&game), Policy::uniform());
        let empty = Agent::new(Arc::clone(&game), Arc::new(Strategy::default()));
        let standin = uniform.with_oversize(Oversize::Standin);
        let mut rng = Rng(99);
        let mut sources = [0usize; 3];
        for _ in 0..4000 {
            let cards = deal_cards(&mut rng);
            let hero = rng.below(2) as usize;
            let bb = [2i64, 100, 20][rng.below(3) as usize];
            let stacks = [bb * (5 + rng.below(300) as i64), bb * (5 + rng.below(300) as i64)];
            let real_game = Nlhe::new(NlheConfig { sb: bb / 2, bb, stack: stacks[0].min(stacks[1]), ..NlheConfig::pluribus_100bb() });
            let mut s = real_game.initial_state();
            for &c in &cards[..4] {
                s = real_game.deal_specific(&s, Card::from_index(c));
            }
            let mut board = Vec::new();
            let mut acts = Vec::new();
            loop {
                while Nlhe::is_dealing(&s) {
                    let c = cards[4 + board.len()];
                    board.push(c);
                    s = real_game.deal_specific(&s, Card::from_index(c));
                }
                if real_game.is_terminal(&s) {
                    break;
                }
                if Nlhe::acting(&s) == hero && rng.below(3) == 0 {
                    let spec = HandSpec {
                        hero,
                        hole: [cards[2 * hero], cards[2 * hero + 1]],
                        board: board.clone(),
                        sb: bb / 2,
                        bb,
                        stacks,
                        actions: acts.clone(),
                        seed: rng.next(),
                    };
                    let agent = if rng.below(2) == 0 { &uniform } else { &empty };
                    let d = agent.decide(&spec).expect("valid real hand");
                    let total: f64 = d.options.iter().map(|o| o.1).sum();
                    assert!((total - 1.0).abs() < 1e-9, "options must be a distribution");
                    sources[d.source as usize] += 1;
                    real_game.apply_real(&s, d.action).expect("agent action must be legal");
                    let d = standin.decide(&spec).expect("valid real hand");
                    let total: f64 = d.options.iter().map(|o| o.1).sum();
                    assert!((total - 1.0).abs() < 1e-9, "options must be a distribution");
                    real_game.apply_real(&s, d.action).expect("agent action must be legal");
                }
                // Random legal real action with an arbitrary size.
                let c = Nlhe::owed(&s);
                let stack = Nlhe::stack_of(&s, Nlhe::acting(&s));
                let minr = real_game.min_raise_add(&s);
                let act = match rng.below(5) {
                    0 if c > 0 => RealAct::Fold,
                    0 | 1 => if c == 0 { RealAct::Check } else { RealAct::Call },
                    2 if minr < stack => RealAct::Add(minr),
                    3 if minr < stack => RealAct::Add(minr + rng.below((stack - minr) as u64) as i64),
                    _ => RealAct::Add(stack),
                };
                s = real_game.apply_real(&s, act).expect("generator makes legal actions");
                acts.push(act);
            }
        }
        assert!(sources.iter().all(|&n| n > 0), "every source exercised: {sources:?}");
    }

    #[test]
    fn non_allin_three_bet_is_off_tree_after_the_abstract_shove() {
        // SB opens to 3bb (off-grid, maps to the pot raise), BB 3-bets to 9bb:
        // the grid only has all-in there, so the abstract line shoves and the
        // flop is off-tree.
        let agent = agent_100bb(Strategy::default());
        let spec = HandSpec {
            hero: 1,
            hole: [0, 5],
            board: vec![10, 20, 30],
            sb: 50,
            bb: 100,
            stacks: [10_000, 10_000],
            actions: vec![RealAct::Add(250), RealAct::Add(800), RealAct::Call],
            seed: 3,
        };
        let d = agent.decide(&spec).unwrap();
        assert_eq!(d.source, Source::OffTree);
        assert!(d.off_tree_reason.is_some());
    }

    #[test]
    fn rejects_impossible_inputs() {
        let agent = agent_100bb(Strategy::default());
        let base = HandSpec {
            hero: 0,
            hole: [0, 1],
            board: vec![],
            sb: 50,
            bb: 100,
            stacks: [10_000, 10_000],
            actions: vec![],
            seed: 0,
        };
        assert!(agent.decide(&base).is_ok());
        let bad = |f: &dyn Fn(&mut HandSpec)| {
            let mut s = HandSpec { board: base.board.clone(), actions: base.actions.clone(), ..base };
            f(&mut s);
            agent.decide(&s).is_err()
        };
        assert!(bad(&|s| s.hero = 1), "not BB's turn preflop");
        assert!(bad(&|s| s.hole = [3, 3]), "duplicate card");
        assert!(bad(&|s| s.actions = vec![RealAct::Check]), "SB cannot check facing the BB");
        assert!(bad(&|s| s.actions = vec![RealAct::Add(120)]), "raise below the minimum");
        assert!(bad(&|s| s.actions = vec![RealAct::Call, RealAct::Check]), "flop not dealt");
        assert!(bad(&|s| s.board = vec![10, 11, 12]), "board ahead of the betting");
    }

    /// Random real hands (arbitrary blinds, stacks and sizes) at the hero's
    /// decisions, for the likelihood tests.
    fn random_decision_points(seed: u64, hands: usize) -> Vec<(HandSpec, RealAct)> {
        let mut rng = Rng(seed);
        let mut out = Vec::new();
        for _ in 0..hands {
            let cards = deal_cards(&mut rng);
            let bb = [2i64, 100][rng.below(2) as usize];
            let stacks = [bb * (20 + rng.below(200) as i64), bb * (20 + rng.below(200) as i64)];
            let real_game = Nlhe::new(NlheConfig { sb: bb / 2, bb, stack: stacks[0].min(stacks[1]), ..NlheConfig::pluribus_100bb() });
            let mut s = real_game.initial_state();
            for &c in &cards[..4] {
                s = real_game.deal_specific(&s, Card::from_index(c));
            }
            let (mut board, mut acts) = (Vec::new(), Vec::new());
            loop {
                while Nlhe::is_dealing(&s) {
                    let c = cards[4 + board.len()];
                    board.push(c);
                    s = real_game.deal_specific(&s, Card::from_index(c));
                }
                if real_game.is_terminal(&s) {
                    break;
                }
                let actor = Nlhe::acting(&s);
                let c = Nlhe::owed(&s);
                let stack = Nlhe::stack_of(&s, actor);
                let minr = real_game.min_raise_add(&s);
                let act = match rng.below(5) {
                    0 if c > 0 => RealAct::Fold,
                    0 | 1 => if c == 0 { RealAct::Check } else { RealAct::Call },
                    2 if minr < stack => RealAct::Add(minr),
                    3 if minr < stack => RealAct::Add(minr + rng.below((stack - minr) as u64) as i64),
                    _ => RealAct::Add(stack),
                };
                let spec = HandSpec {
                    hero: actor,
                    hole: [cards[2 * actor], cards[2 * actor + 1]],
                    board: board.clone(),
                    sb: bb / 2,
                    bb,
                    stacks,
                    actions: acts.clone(),
                    seed: rng.next(),
                };
                out.push((spec, act));
                s = real_game.apply_real(&s, act).expect("generator makes legal actions");
                acts.push(act);
            }
        }
        out
    }

    #[test]
    fn likelihood_of_every_option_is_its_probability() {
        // What range tracking of the agent's own actions relies on (the LBR path).
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let maniac = Agent::from_policy(Arc::clone(&game), Policy::uniform().leak(crate::s32::LeakFamily::Maniac, 0.5));
        let empty = Agent::new(Arc::clone(&game), Arc::new(Strategy::default()));
        let standin = maniac.with_oversize(Oversize::Standin);
        let mut checked = 0;
        for (i, (spec, _)) in random_decision_points(5, 400).into_iter().enumerate() {
            for agent in [if i % 2 == 0 { &maniac } else { &empty }, &standin] {
                let d = agent.decide_inner(&spec).unwrap();
                for &(a, p) in &d.options {
                    let l = agent.action_likelihood(&spec, a).unwrap();
                    assert!((l - p).abs() < 1e-12, "{a:?}: likelihood {l} vs option probability {p} ({:?})", d.source);
                    checked += 1;
                }
            }
        }
        assert!(checked > 2000);
    }

    /// Fixing the reading of the opponent's last action (`Force`) gives exactly
    /// the options of every seed that draws that reading, with the draw made
    /// at `translation_split`'s chance.
    #[test]
    fn forced_translation_reads_like_the_seeded_one() {
        for oversize in [Oversize::Snap, Oversize::Standin] {
            forced_translation_reads_like_the_seeded_one_with(oversize);
        }
    }

    fn forced_translation_reads_like_the_seeded_one_with(oversize: Oversize) {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let agent = Agent::from_policy(Arc::clone(&game), Policy::uniform().leak(crate::s32::LeakFamily::Maniac, 0.5))
            .with_oversize(oversize);
        let bits = |o: Vec<Vec<(RealAct, f64)>>| o.into_iter().flatten().map(|(a, p)| (a, p.to_bits())).collect::<Vec<_>>();
        let (mut between, mut distinct) = (0, 0);
        for (spec, _) in random_decision_points(9, 400) {
            if agent.translation_split(&spec, Force::default()).unwrap().is_none() {
                continue;
            }
            between += 1;
            let i = spec.actions.len() - 1;
            let holes = [spec.hole];
            for k in 0..40u64 {
                let spec = HandSpec { seed: splitmix(k), ..spec.clone() };
                // The earlier readings follow this seed in every call below.
                let pl = agent.translation_split(&spec, Force::default()).unwrap().expect("a split for every seed");
                let lower = unit(spec.seed, STREAM_TRANSLATE, i as u64) < pl;
                let force = |lower: bool| Force { mask: 1 << i, lower: if lower { 1 << i } else { 0 } };
                let seeded = bits(agent.options_many(&spec, &holes).unwrap());
                assert_eq!(bits(agent.options_many_forced(&spec, &holes, force(lower)).unwrap()), seeded);
                let other = bits(agent.options_many_forced(&spec, &holes, force(!lower)).unwrap());
                distinct += usize::from(other != seeded);
            }
        }
        assert!(between > 30, "only {between} spots between two sizes");
        assert!(distinct > 100, "the two readings differ in only {distinct} cases");
    }

    #[test]
    fn off_grid_likelihood_is_the_average_over_the_randomized_translation() {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let agent = Agent::from_policy(Arc::clone(&game), Policy::uniform().leak(crate::s32::LeakFamily::Maniac, 0.5));
        let mut between = 0;
        for (spec, act) in random_decision_points(8, 300) {
            let loc = agent.locate(&spec).unwrap();
            if !loc.on_tree {
                continue;
            }
            let Some(Translation::Between { lo, hi, .. }) = agent.translate_dist(&loc.real_game, &loc.real, &loc.abs, act) else {
                continue;
            };
            let probs = agent.decide_inner(&spec).unwrap().probs;
            // Monte Carlo over the per-hand translation seed.
            let k = 4000;
            let mean: f64 = (0..k)
                .map(|seed| {
                    let (i, _) = agent
                        .translate(
                            &loc.real_game,
                            &loc.real,
                            &loc.abs,
                            act,
                            false,
                            splitmix(seed),
                            spec.actions.len() as u64,
                            Force::default(),
                        )
                        .unwrap();
                    assert!(i == lo || i == hi);
                    probs[i]
                })
                .sum::<f64>()
                / k as f64;
            let l = agent.action_likelihood(&spec, act).unwrap();
            assert!((l - mean).abs() < 0.03, "likelihood {l} vs sampled translation {mean}");
            between += 1;
        }
        assert!(between > 30, "only {between} off-grid spots");
    }

    /// A flop spot on the 100bb grid: SB limps, BB checks, BB then bets `bet`
    /// into the 2bb pot and the SB (hero) may respond with `then`.
    fn flop_overbet(bet: i64, then: &[RealAct], seed: u64) -> HandSpec {
        let mut actions = vec![RealAct::Call, RealAct::Check, RealAct::Add(bet)];
        actions.extend_from_slice(then);
        let board = if then.is_empty() { vec![10, 20, 30] } else { vec![10, 20, 30, 40] };
        HandSpec { hero: 0, hole: [0, 5], board, sb: 50, bb: 100, stacks: [10_000, 10_000], actions, seed }
    }

    #[test]
    fn standin_reads_an_overbet_between_the_largest_size_and_allin() {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let snap = Agent::from_policy(Arc::clone(&game), Policy::uniform());
        let standin = snap.with_oversize(Oversize::Standin);
        let cfg = SearchCfg { from_flop: false, flop_iters: 1, flop_checkdown: false, from_turn: false, river_iters: 1, turn_iters: 1, exact_own_range: false, solve_round: false, external: None, rescue: true };
        assert_eq!(standin.with_search(cfg).oversize(), Oversize::Standin);
        assert_eq!(snap.oversize(), Oversize::Snap);
        // 2x pot (400 into 200, 4% of the stack): snap reads the 1x pot bet.
        let spec = flop_overbet(400, &[], 1);
        // The node before the bet: the grid's sizes there.
        let before = snap.locate(&HandSpec { hero: 1, hole: [1, 6], actions: spec.actions[..2].to_vec(), ..spec.clone() }).unwrap();
        let at = |a: &Agent| a.translate_dist(&before.real_game, &before.real, &before.abs, RealAct::Add(400)).unwrap();
        let bl = game.labeled_actions(&before.abs);
        let one = bl.iter().position(|l| *l == ActLabel::Bet(1.0)).unwrap();
        let allin = bl.iter().position(|l| *l == ActLabel::AllIn).unwrap();
        assert_eq!(at(&snap), Translation::Exact(one));
        let Translation::Between { lo, hi, pl } = at(&standin) else { panic!("standin must randomize") };
        assert_eq!((lo, hi), (one, allin));
        assert_eq!(pl.to_bits(), prob_lower(1.0, 9_900.0 / 200.0, 2.0).to_bits());
    }

    #[test]
    fn standin_call_of_an_allin_reading_stays_on_the_tree() {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let snap = Agent::from_policy(Arc::clone(&game), Policy::uniform());
        let standin = snap.with_oversize(Oversize::Standin);
        // The 1x pot line: bet 200, call, then BB checks the turn.
        let grid_line = snap.decide(&flop_overbet(200, &[RealAct::Call, RealAct::Check], 1)).unwrap();
        assert_eq!(grid_line.source, Source::Blueprint);
        // Whether the hero, facing the bet, reads it as all-in.
        let reads_allin = |a: &Agent, bet: i64, seed: u64| {
            let loc = a.locate(&flop_overbet(bet, &[], seed)).unwrap();
            assert!(loc.on_tree);
            Nlhe::owed(&loc.abs) >= Nlhe::stack_of(&loc.abs, 0)
        };
        let (mut read_allin, mut read_lower, mut snap_left) = (0, 0, 0);
        for bet in [400i64, 6_000] {
            for seed in 0..64u64 {
                let allin_reading = reads_allin(&standin, bet, seed);
                read_allin += usize::from(allin_reading);
                read_lower += usize::from(!allin_reading);
                // Called and the turn checked to the hero: on the tree, as after the 1x pot line.
                let d = standin.decide(&flop_overbet(bet, &[RealAct::Call, RealAct::Check], seed)).unwrap();
                assert_eq!(d.source, Source::Blueprint, "bet {bet} seed {seed}: {:?}", d.off_tree_reason);
                assert_eq!(d.abstract_history, grid_line.abstract_history);
                assert_eq!(d.key, grid_line.key);
                // Snap: 2x pot is always the 1x pot bet; the 60% stack bet
                // read as all-in and called leaves the tree, as before.
                let s = snap.decide(&flop_overbet(bet, &[RealAct::Call, RealAct::Check], seed)).unwrap();
                let snap_allin = reads_allin(&snap, bet, seed);
                assert!(bet != 400 || !snap_allin);
                snap_left += usize::from(snap_allin);
                assert_eq!(s.source == Source::OffTree, snap_allin);
            }
        }
        assert!(read_allin > 20 && read_lower > 20, "{read_allin} all-in readings, {read_lower} lower");
        assert!(snap_left > 40, "{snap_left}");
    }

    #[test]
    fn calling_an_allin_reading_is_a_real_call() {
        // Facing a bet read as all-in that is not one, the agent's call (the
        // all-in node's AllIn label) is a real call, never a shove over the bet.
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let snap = Agent::from_policy(Arc::clone(&game), Policy::uniform());
        let standin = snap.with_oversize(Oversize::Standin);
        let mut seen = [0, 0];
        for (k, a) in [&snap, &standin].into_iter().enumerate() {
            for bet in [400i64, 6_000] {
                for seed in 0..64u64 {
                    let spec = flop_overbet(bet, &[], seed);
                    let loc = a.locate(&spec).unwrap();
                    if Nlhe::owed(&loc.abs) < Nlhe::stack_of(&loc.abs, 0) {
                        continue; // read as a grid size
                    }
                    seen[k] += 1;
                    let d = a.decide(&spec).unwrap();
                    assert_eq!(d.source, Source::Blueprint);
                    assert_eq!(d.options, vec![(RealAct::Fold, 0.5), (RealAct::Call, 0.5)], "bet {bet} seed {seed}");
                    assert!(matches!(d.action, RealAct::Fold | RealAct::Call));
                    let node = a.public_node(&spec, spec.hole).unwrap();
                    assert_eq!(node.reals, vec![RealAct::Fold, RealAct::Call]);
                    assert_eq!(a.action_likelihood(&spec, RealAct::Call).unwrap(), 0.5);
                }
            }
        }
        assert!(seen[0] > 20 && seen[1] > 40, "all-in readings: snap {} standin {}", seen[0], seen[1]);
    }

    /// records/preflop-open: a small-blind open to 2bb or 2.5bb is read as
    /// the 3bb (pot-size) open under every translation mode; with F2
    /// (`OpenBelow::Limp`) between the limp and it, P(limp) = (1 - x) / (1 + x);
    /// the 3bb open itself is exact either way.
    #[test]
    fn an_open_below_the_grid_is_the_grid_open_or_with_f2_between_the_limp_and_it() {
        let game = Arc::new(Nlhe::new(NlheConfig::grid("200bb").unwrap()));
        let base = Agent::from_policy(Arc::clone(&game), Policy::uniform());
        let spec = |actions: Vec<RealAct>, board: Vec<u8>| HandSpec {
            hero: 1,
            hole: [0, 1],
            board,
            sb: 50,
            bb: 100,
            stacks: [20_000, 20_000],
            actions,
            seed: 7,
        };
        let history = |agent: &Agent, s: &HandSpec, force: Force| {
            let loc = agent.locate_forced(s, force).unwrap();
            assert!(loc.on_tree, "{:?}", s.actions);
            Nlhe::betting_history(&loc.abs)
        };
        let flop = vec![20u8, 33, 46];
        let lower = Force { mask: 1, lower: 1 };
        let upper = Force { mask: 1, lower: 0 };
        let open3 = history(&base, &spec(vec![RealAct::Add(250)], vec![]), Force::default());
        let open3_called = history(&base, &spec(vec![RealAct::Add(250), RealAct::Call], flop.clone()), Force::default());
        let limped = history(&base, &spec(vec![RealAct::Call, RealAct::Check], flop.clone()), Force::default());
        for oversize in [Oversize::Snap, Oversize::Standin] {
            let snap = base.with_oversize(oversize);
            let f2 = snap.with_open_below(OpenBelow::Limp);
            assert_eq!(f2.oversize(), oversize);
            assert_eq!(f2.with_oversize(Oversize::Snap).open_below(), OpenBelow::Limp); // so with_search (built on with_oversize) keeps it
            for (add, x) in [(150, 0.5), (200, 0.75)] {
                let open = spec(vec![RealAct::Add(add)], vec![]);
                assert_eq!(snap.translation_split(&open, Force::default()).unwrap(), None, "{oversize:?} {add}");
                assert_eq!(history(&snap, &open, Force::default()), open3, "{oversize:?} {add}");
                let pl = f2.translation_split(&open, Force::default()).unwrap().expect("F2 splits the open");
                assert!((pl - (1.0 - x) / (1.0 + x)).abs() < 1e-12, "{oversize:?} {add}: {pl}");
                // The big blind's call: the limped flop, or the 3bb open called.
                let called = spec(vec![RealAct::Add(add), RealAct::Call], flop.clone());
                assert_eq!(history(&f2, &called, lower), limped, "{oversize:?} {add}");
                assert_eq!(history(&f2, &called, upper), open3_called, "{oversize:?} {add}");
                assert_eq!(history(&snap, &called, Force::default()), open3_called, "{oversize:?} {add}");
                // Read as a limp, the big blind never folds: its check is a real call.
                let reals = f2.public_node_forced(&open, [0, 1], lower).unwrap().reals;
                assert!(reals.contains(&RealAct::Call) && !reals.contains(&RealAct::Fold), "{reals:?}");
            }
            let open = spec(vec![RealAct::Add(250)], vec![]);
            assert_eq!(f2.translation_split(&open, Force::default()).unwrap(), None);
            assert_eq!(history(&f2, &open, Force::default()), open3);
        }
    }

    /// F1 (the `200bb-f1` grid): the small blind's 2, 2.5 and 3 bb opens are on
    /// the grid, so each is read as its own open (not as 3 bb), in every
    /// translation mode, and F2 never fires on them.
    #[test]
    fn the_f1_grid_reads_its_small_opens_as_themselves() {
        let game = Arc::new(Nlhe::new(NlheConfig::grid("200bb-f1").unwrap()));
        let base = Agent::from_policy(Arc::clone(&game), Policy::uniform());
        let open = |add: i64| HandSpec {
            hero: 1,
            hole: [0, 1],
            board: Vec::new(),
            sb: 50,
            bb: 100,
            stacks: [20_000, 20_000],
            actions: vec![RealAct::Add(add)],
            seed: 7,
        };
        let mut root = game.initial_state();
        while game.is_chance(&root) {
            root = game.sample_chance_with(&root, |_| 0);
        }
        let labels: Vec<ActLabel> = game.labeled_actions(&root).iter().copied().collect();
        let grid_open = |f: f64| {
            let i = labels.iter().position(|l| *l == ActLabel::Bet(f)).unwrap();
            Nlhe::betting_history(&game.next_state(&root, i))
        };
        let mut seen = Vec::new();
        for oversize in [Oversize::Snap, Oversize::Standin] {
            let snap = base.with_oversize(oversize);
            let f2 = snap.with_open_below(OpenBelow::Limp);
            for (add, f) in [(150, 0.5), (200, 0.75), (250, 1.0)] {
                for agent in [&snap, &f2] {
                    assert_eq!(agent.translation_split(&open(add), Force::default()).unwrap(), None, "{oversize:?} {add}");
                    let loc = agent.locate_forced(&open(add), Force::default()).unwrap();
                    assert!(loc.on_tree, "{oversize:?} {add}");
                    let h = Nlhe::betting_history(&loc.abs);
                    assert_eq!(h, grid_open(f), "{oversize:?} {add}");
                    seen.push(h);
                }
            }
        }
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 3, "{seen:?}");
    }

    /// F2 reads only the small blind's first raise: every other action is
    /// located exactly as without it (a raise below the grid later in the
    /// hand, the big blind's raise of a limp, the agent's own actions).
    #[test]
    fn f2_changes_no_reading_but_the_small_blinds_open() {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let snap = Agent::from_policy(Arc::clone(&game), Policy::uniform().leak(crate::s32::LeakFamily::Maniac, 0.5));
        let f2 = snap.with_open_below(OpenBelow::Limp);
        let (mut checked, mut opens) = (0, 0);
        for (spec, _) in random_decision_points(12, 600) {
            // A first raise below the pot-size open (to 3bb: 2.5bb added).
            let open_below = matches!(spec.actions.first(), Some(&RealAct::Add(d)) if 2 * d > spec.bb && 2 * d < 5 * spec.bb);
            if open_below {
                opens += 1;
                continue;
            }
            let (a, b) = (snap.locate(&spec).unwrap(), f2.locate(&spec).unwrap());
            assert_eq!((Nlhe::betting_history(&a.abs), a.on_tree, a.last_split), (Nlhe::betting_history(&b.abs), b.on_tree, b.last_split), "{:?}", spec.actions);
            checked += 1;
        }
        assert!(checked > 300, "only {checked} spots checked ({opens} opens below the grid)");
    }
}
