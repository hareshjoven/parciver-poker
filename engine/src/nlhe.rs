// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
//! Heads-up No-Limit Texas Hold'em as a [`crate::game::Game`] — the real target
//! tree the blueprint is solved on (the 6-max generalization comes later, on this
//! proven base).
//!
//! Design decisions, and why:
//!
//! * **Integer chips.** All money is `i64` chips with 1 big blind = 100 chips
//!   (`BB`), small blind 50. Pot-fraction bet sizes round to whole chips. Integers
//!   make chip conservation and zero-sum EXACT — the two invariants the tests
//!   pin — where floats would accumulate rounding error in the pot.
//!
//! * **One card dealt per chance node.** A hold'em deal (2+2 hole, then 3/1/1
//!   board) has far too many joint outcomes to materialize, but the CFR trait
//!   deals via `chance_outcomes` (a materialized list). Dealing ONE card at a time
//!   keeps every chance node to <=52 uniform outcomes, so the existing interface
//!   and the external-sampling MCCFR work unchanged (they sample one outcome).
//!   Order within hole/board is irrelevant to hand value and is abstracted away by
//!   the eventual bucket key, so the extra chance depth costs nothing but is exact.
//!
//! * **Explicit phase.** The betting/dealing state machine is an explicit `Phase`
//!   rather than being re-derived from counters each call — deriving it is exactly
//!   the kind of place an off-by-one silently corrupts the tree.
//!
//! * **Payoffs are zero-sum by construction.** Only the matched portion of the two
//!   stacks is contested; any unmatched excess (a shove that the other calls for
//!   less) is refunded. `payoffs` returns `(net0, net1)` with `net0 + net1 == 0`
//!   identically; `terminal_utility` is just `net0`. The tests still assert it.
//!
//! Bet abstraction (the moderate / Pluribus-like grid): first-in bets
//! {0.5x, 1.0x pot}, first raise {1.0x pot}, deeper raises pruned to all-in only;
//! all-in and check/call/fold always available as legal. Sizes are a config knob,
//! and named grids ([`NlheConfig::grid`]) add sized re-raises (the 200 bb grid).
//!
//! `infoset_key` is `street | card token | betting history` (perfect recall on
//! betting). The card token comes from one of two [`CardAbstraction`] modes:
//!   * `Buckets` (B2) — the Phase-3b 52-card tables via
//!     [`crate::bucketing::Abstraction`] (preflop = the 169 exact classes,
//!     postflop = the bucket tables), the real blueprint game
//!     ([`Nlhe::with_abstraction`]);
//!   * `Exact` (B3) — the lossless suit-isomorphic canonical `(hole, board)`,
//!     which needs no tables and is correct on any deck ([`Nlhe::new`]).
//!
//! The deck is a config knob ([`NlheConfig::deck`]): the real game uses all 52
//! cards; [`NlheConfig::small`] uses a 7-card deck so the WHOLE tree is tractable
//! for full-traversal CFR + exact best-response exploitability — the NLHE analog
//! of the Kuhn/Leduc correctness gate (B3), the bridge before the big blueprint
//! solve.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::bucketing::Abstraction;
use crate::cards::Card;
use crate::game::Game;
use crate::hand_eval::{eval5, eval7_fast};

const NUM_RANKS_BOARD: [usize; 4] = [0, 3, 4, 5]; // cumulative board size by street

/// Bet-abstraction + stakes + deck configuration.
#[derive(Clone)]
pub struct NlheConfig {
    pub sb: i64,
    pub bb: i64,
    pub stack: i64,
    /// Pot-fraction sizes for an opening bet (facing no bet); all-in is separate.
    pub open_fracs: Vec<f64>,
    /// Pot-fraction sizes for the first raise facing a bet; all-in is separate.
    pub raise_fracs: Vec<f64>,
    /// Pot-fraction sizes for each of the next `max_reraises` raises of a street
    /// (a preflop 3-bet, 4-bet, ...; a postflop re-raise, ...). Beyond them only
    /// all-in remains. Empty / 0 is the original grid (deeper raises all-in only).
    pub reraise_fracs: Vec<f64>,
    pub max_reraises: u8,
    /// Pot-fraction sizes for the small blind's open (the first raise preflop), in
    /// place of `raise_fracs` there; empty (every grid before F1) means
    /// `raise_fracs`. The F1 grid (records/preflop-open/FINDING.md) opens to 2, 2.5
    /// and 3 bb here and is otherwise the 200 bb grid.
    pub preflop_open_fracs: Vec<f64>,
    /// The 0..52 card indices dealt from. The real game uses all 52; the small
    /// cross-validation game (B3) uses a reduced deck so the whole tree is
    /// tractable for full-traversal CFR + exact exploitability.
    pub deck: Vec<u8>,
    /// Number of community cards dealt before showdown: 5 for real hold'em
    /// (flop+turn+river), 3 for the small flop-only game (showdown on 2 hole + 3
    /// board, scored by `eval5`). Flop-only keeps the full tree small enough to
    /// full-traverse while still exercising two betting rounds and NL sizing.
    pub board_cards: usize,
}

impl NlheConfig {
    /// The chosen blueprint config: 100 bb effective, moderate/Pluribus-like grid,
    /// full 52-card deck.
    pub fn pluribus_100bb() -> Self {
        NlheConfig {
            sb: 50,
            bb: 100,
            stack: 10_000, // 100 bb
            open_fracs: vec![0.5, 1.0],
            raise_fracs: vec![1.0],
            reraise_fracs: vec![],
            max_reraises: 0,
            preflop_open_fracs: vec![],
            deck: (0..52).collect(),
            board_cards: 5,
        }
    }

    /// 200 bb effective with real re-raises: the 100 bb grid's opens and first
    /// raise, then pot-size raises for the next three raises of a street
    /// (preflop: a 3bb open, 9bb 3-bet, 27bb 4-bet, 81bb 5-bet), then all-in.
    /// The 100 bb grid has none (a 3-bet there is all-in), which real opponents
    /// punish; 200 bb is the GTO Wizard benchmark's depth.
    pub fn deep_200bb() -> Self {
        NlheConfig { stack: 20_000, reraise_fracs: vec![1.0], max_reraises: 3, ..Self::pluribus_100bb() }
    }

    /// F1 (records/preflop-open/FINDING.md, "Candidate fixes"): the 200 bb grid
    /// with the small blind's open at 0.5, 0.75 and 1.0 pot (to 2, 2.5 and 3 bb),
    /// so the common 2-2.5 bb opens are on the grid instead of read as 3 bb. Every
    /// other node is the 200 bb grid's.
    pub fn deep_200bb_f1() -> Self {
        NlheConfig { preflop_open_fracs: vec![0.5, 0.75, 1.0], ..Self::deep_200bb() }
    }

    /// The named bet grids a blueprint can be trained on. The name is stored in
    /// the blueprint and checkpoint files; files from before the tag existed
    /// ("") were trained on "100bb".
    pub const GRIDS: [&'static str; 3] = ["100bb", "200bb", "200bb-f1"];

    pub fn grid(name: &str) -> Option<Self> {
        match name {
            "" | "100bb" => Some(Self::pluribus_100bb()),
            "200bb" => Some(Self::deep_200bb()),
            "200bb-f1" => Some(Self::deep_200bb_f1()),
            _ => None,
        }
    }

    /// A deliberately TINY heads-up NLHE instance for the B3 correctness bridge: a
    /// 7-card deck (Qc Qd Qh Kc Kd Ac Ad), a 2 bb stack with one pot-sized open
    /// in the grid, and a FLOP-ONLY board (preflop + flop, showdown
    /// on 2 hole + 3 board via `eval5`). Small enough to solve with full-traversal
    /// CFR AND to score with exact best-response exploitability — the NLHE analog
    /// of the Kuhn/Leduc gate — while still exercising two betting rounds, NL bet
    /// sizing, and postflop card infosets. Cards are real 0..52 indices, and it is
    /// solved on the lossless `CardAbstraction::Exact` key, so its equilibrium is
    /// the game's true equilibrium (exploitability → 0 is meaningful).
    pub fn small() -> Self {
        // A 7-card deck: ranks Q=10, K=11, A=12 with suits c=0,d=1,h=2, minus Kh
        // and Ah (7 cards). Small enough (4 hole + 3 flop = 7 -> the flop is the forced
        // complement of the holes) that the whole flop-only tree can be traversed.
        let deck = vec![
            10 * 4,
            10 * 4 + 1,
            10 * 4 + 2, // Qc Qd Qh
            11 * 4,
            11 * 4 + 1, // Kc Kd
            12 * 4,
            12 * 4 + 1, // Ac Ad
        ];
        NlheConfig {
            sb: 1,
            bb: 2,
            stack: 4, // 2 bb effective
            open_fracs: vec![1.0],
            raise_fracs: vec![],
            reraise_fracs: vec![],
            max_reraises: 0,
            preflop_open_fracs: vec![],
            deck,
            board_cards: 3, // flop-only
        }
    }
}

/// How `infoset_key` turns the acting player's cards into a card token.
pub enum CardAbstraction {
    /// Lossless: the suit-isomorphic canonical `(hole, board)` as the token. Needs
    /// no tables and is correct on any deck, so it is what the small
    /// cross-validation game (B3) and the betting-engine tests use.
    Exact,
    /// The Phase-3b 52-card bucket tables — the real blueprint abstraction (B2).
    Buckets(Arc<Abstraction>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    DealHole,
    PreBet,
    DealFlop,
    FlopBet,
    DealTurn,
    TurnBet,
    DealRiver,
    RiverBet,
    Showdown,
}

/// One legal action. `Bet` carries the exact chips ADDED this action (call amount
/// plus the raise-by, or the bet amount when there is nothing to call).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Action {
    Fold,
    Check,
    Call,
    Bet(i64),
    AllIn,
}

/// Capacity of an [`ActionList`]: fold, check/call, the configured sized bets and
/// all-in. Matches the single-digit history token (`history_token`) and the MCCFR
/// `MAX_ACTIONS` stack buffers, which already require fewer than 10 actions.
const MAX_LEGAL: usize = 10;

/// A per-node list (legal actions or their labels) in a fixed-capacity stack
/// array. `num_actions`, `next_state` and the leak policy rebuild these on every
/// hot-loop visit, so they must not allocate; the list derefs to a slice, which
/// is all any caller needs.
#[derive(Clone, Copy)]
pub(crate) struct NodeList<T: Copy> {
    items: [T; MAX_LEGAL],
    len: usize,
}

impl<T: Copy> NodeList<T> {
    fn new(fill: T) -> Self {
        NodeList {
            items: [fill; MAX_LEGAL],
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, item: T) {
        assert!(self.len < MAX_LEGAL, "more than {MAX_LEGAL} legal actions");
        self.items[self.len] = item;
        self.len += 1;
    }
}

impl<T: Copy> std::ops::Deref for NodeList<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        &self.items[..self.len]
    }
}

type ActionList = NodeList<Action>;

/// Semantic label for a legal action, in the SAME order as [`Nlhe::legal_actions`]
/// (so a sampled action index maps straight to its label). Used by the read-only
/// blueprint analysis (`crate::analysis`) to bin actions into the dashboard's
/// per-street action-frequency (fold/check/call/raise/all-in) and bet-sizing
/// (pot-fraction) panels — never on the solve hot path.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum ActLabel {
    Fold,
    Check,
    Call,
    /// A sized bet/raise, tagged with the pot-fraction that produced it (e.g. 0.5, 1.0).
    Bet(f64),
    AllIn,
}

/// The dealt board (0, 3, 4 or 5 cards) inline, so cloning a state copies it
/// instead of allocating. Derefs to `&[Card]`.
#[derive(Clone, Copy)]
struct Board {
    cards: [Card; 5],
    len: u8,
}

impl Board {
    fn new() -> Self {
        Board {
            cards: [Card::default(); 5],
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, card: Card) {
        assert!((self.len as usize) < 5, "board already has five cards");
        self.cards[self.len as usize] = card;
        self.len += 1;
    }
}

impl std::ops::Deref for Board {
    type Target = [Card];

    #[inline]
    fn deref(&self) -> &[Card] {
        &self.cards[..self.len as usize]
    }
}

/// Capacity of the inline betting history. Postflop a street allows at most two
/// sized bets, then only fold/call/all-in, so a street is at most ~6 tokens and a
/// whole hand (4 streets + 3 separators) under 30 bytes; 64 leaves ample margin.
const HISTORY_CAP: usize = 64;

/// The compact ASCII betting history inline, so cloning a state copies it instead
/// of allocating. Displays (and yields bytes) exactly like the `String` it replaces.
#[derive(Clone, Copy)]
struct History {
    bytes: [u8; HISTORY_CAP],
    len: u8,
}

impl History {
    fn new() -> Self {
        History {
            bytes: [0; HISTORY_CAP],
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, token: char) {
        debug_assert!(token.is_ascii());
        assert!((self.len as usize) < HISTORY_CAP, "betting history exceeds {HISTORY_CAP} bytes");
        self.bytes[self.len as usize] = token as u8;
        self.len += 1;
    }

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    fn as_str(&self) -> &str {
        // Only ASCII tokens are ever pushed.
        std::str::from_utf8(self.as_bytes()).expect("history is ASCII")
    }
}

impl std::fmt::Display for History {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone)]
pub struct NlheState {
    phase: Phase,
    hole: [[Card; 2]; 2],
    n_hole: u8,
    board: Board,
    /// Chips remaining behind, and total chips put in this hand (pot = sum).
    stack: [i64; 2],
    committed: [i64; 2],
    /// Chips put in during the CURRENT street (drives to-call and closure).
    street_contrib: [i64; 2],
    /// Has each player taken a voluntary action since the last aggression this
    /// street? (Blind posting is not a voluntary action — this is what gives the
    /// big blind its preflop option.)
    acted: [bool; 2],
    all_in: [bool; 2],
    to_act: usize,
    /// Size of the last full raise increment this street (for the min-raise rule).
    last_raise: i64,
    /// Number of bet/raise actions this street (0 = no bet yet). Preflop starts at
    /// 1 because the big blind is the opening bet.
    street_bets: u8,
    folded: Option<usize>,
    /// Once either player is all-in and matched, remaining streets are dealt with
    /// no further betting (straight to showdown).
    no_more_betting: bool,
    /// Compact action history for the infoset key (perfect recall on betting).
    history: History,
    /// Each seat's card bucket for the current betting street, filled by the
    /// `Buckets` game numbered `bucket_game` when the street's cards are dealt (0 =
    /// not filled). Only a cache: the cards decide the bucket, so `infoset_key_into`
    /// uses it only when the game keying the state is the one that filled it, and
    /// otherwise recomputes. Saves recomputing `bucket_of` at every decision node.
    buckets: [u32; 2],
    bucket_game: u64,
}

/// Numbers each `Buckets` game for `NlheState::bucket_game`; 0 is never handed
/// out, and no number is reused within the process.
static NEXT_BUCKET_GAME: AtomicU64 = AtomicU64::new(1);

pub struct Nlhe {
    cfg: NlheConfig,
    /// How infoset keys map the acting player's cards to a token — a lossless
    /// canonical key (the small/exact game) or the Phase-3b buckets (the real
    /// blueprint game).
    card_abs: CardAbstraction,
    /// This game's number in `NlheState::bucket_game` (0 for the `Exact` game,
    /// which fills no bucket cache).
    bucket_game: u64,
}

impl Nlhe {
    /// The betting engine on the lossless `CardAbstraction::Exact` card key. Works
    /// on any deck without loading tables, so this is the small cross-validation
    /// game and what the betting-engine tests use.
    pub fn new(cfg: NlheConfig) -> Self {
        Nlhe {
            cfg,
            card_abs: CardAbstraction::Exact,
            bucket_game: 0,
        }
    }

    /// The full 52-card blueprint game: the betting engine keyed on the Phase-3b
    /// bucket abstraction, so `infoset_key` returns
    /// `street | acting-player bucket | betting history`.
    pub fn with_abstraction(cfg: NlheConfig, abs: Arc<Abstraction>) -> Self {
        Nlhe {
            cfg,
            card_abs: CardAbstraction::Buckets(abs),
            bucket_game: NEXT_BUCKET_GAME.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// A subtree-determining signature of a state, with each player's hole and the
    /// board SORTED so states that differ only in the one-card-at-a-time dealing
    /// order (which never affects value or any downstream infoset key) collapse to
    /// one signature. This lets the exact best-response memoize away that
    /// duplication. It encodes every field the subtree value depends on — a
    /// mismatch here would make two genuinely different subtrees share a memo slot,
    /// so it is deliberately exhaustive.
    pub fn state_signature(s: &NlheState) -> String {
        let mut h0 = [s.hole[0][0].index(), s.hole[0][1].index()];
        h0.sort_unstable();
        let mut h1 = [s.hole[1][0].index(), s.hole[1][1].index()];
        h1.sort_unstable();
        let mut board: Vec<u8> = s.board.iter().map(|c| c.index()).collect();
        board.sort_unstable();
        format!(
            "{}|{:?}|{:?}|{}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{}|{}|{}|{}|{}|{}",
            s.phase as u8,
            h0,
            h1,
            s.n_hole,
            board,
            s.stack,
            s.committed,
            s.street_contrib,
            s.acted,
            s.all_in,
            s.to_act,
            s.last_raise,
            s.street_bets,
            match s.folded {
                Some(p) => p as i64,
                None => -1,
            },
            s.no_more_betting,
            s.history,
        )
    }

    /// The Hero-OBSERVABLE context at a decision node: `street | public board |
    /// betting history` — the infoset key with the ACTING player's private hole
    /// dropped. Hero sees the board and the betting sequence, not the opponent's
    /// hole, so the observed opponent model (the exploitation spine's H-feature,
    /// RMAPSS §40) is the reach-weighted action marginal aggregated over this key.
    /// Legal actions depend only on the betting state (history+street), so every
    /// state sharing this key has the same action count — the marginal is well
    /// defined. Distinct from any `infoset_key` (own `B`-tagged namespace).
    pub fn observable_key(&self, s: &NlheState) -> String {
        let mut board: Vec<u8> = s.board.iter().map(|c| c.index()).collect();
        board.sort_unstable();
        format!("{}|B{:?}|{}", phase_street_tag(s.phase), board, s.history)
    }

    /// The two public-only keys frozen for S3.2's tabular observation model:
    /// exact `(street, sorted board, history, position)` and its
    /// `(street, history, position)` backoff. Neither key reads either private
    /// holding or its abstraction bucket.
    pub(crate) fn public_model_keys(&self, s: &NlheState) -> (String, String) {
        let (exact, backoff, _) = self.public_hierarchy_keys(s);
        (exact, backoff)
    }

    /// S3.3's third, coarsest public context adds only the legal-action shape to
    /// street and position.  It deliberately excludes board cards, private cards,
    /// buckets, payoffs and opponent labels.  S3.2 continues to consume only the
    /// first two values through `public_model_keys` above.
    pub(crate) fn public_hierarchy_keys(&self, s: &NlheState) -> (String, String, String) {
        let street = phase_street_tag(s.phase);
        let position = s.to_act;
        let mut board: Vec<u8> = s.board.iter().map(|c| c.index()).collect();
        board.sort_unstable();
        let shape: String = self
            .labeled_actions(s)
            .iter()
            .map(|label| match label {
                ActLabel::Fold => 'f',
                ActLabel::Check => 'x',
                ActLabel::Call => 'c',
                ActLabel::Bet(_) => 'r',
                ActLabel::AllIn => 'A',
            })
            .collect();
        (
            format!("{street}|B{board:?}|{}|P{position}", s.history),
            format!("{street}|{}|P{position}", s.history),
            format!("{street}|A{shape}|P{position}"),
        )
    }

    fn current_bet(s: &NlheState) -> i64 {
        s.street_contrib[0].max(s.street_contrib[1])
    }

    /// Chips the player to act must add to match the current bet (uncapped by
    /// stack; a short stack calls all-in for less).
    fn to_call(s: &NlheState) -> i64 {
        Self::current_bet(s) - s.street_contrib[s.to_act]
    }

    /// A player still owes action if, not folded and not all-in, they either have
    /// not matched the current bet or have not yet had their voluntary turn.
    fn owes_action(s: &NlheState, p: usize) -> bool {
        s.folded.is_none()
            && !s.all_in[p]
            && (s.street_contrib[p] < Self::current_bet(s) || !s.acted[p])
    }

    fn betting_closed(s: &NlheState) -> bool {
        !Self::owes_action(s, 0) && !Self::owes_action(s, 1)
    }

    /// Pot-fraction sizes available at this node (all-in handled separately).
    fn sized_fracs(&self, s: &NlheState) -> &[f64] {
        if Self::to_call(s) == 0 {
            &self.cfg.open_fracs
        } else if s.street_bets <= 1 {
            if s.phase == Phase::PreBet && !self.cfg.preflop_open_fracs.is_empty() {
                &self.cfg.preflop_open_fracs
            } else {
                &self.cfg.raise_fracs
            }
        } else if s.street_bets <= 1 + self.cfg.max_reraises {
            &self.cfg.reraise_fracs
        } else {
            &[]
        }
    }

    /// Chips to ADD for a bet/raise of pot-fraction `f`, enforcing the min-raise.
    fn sized_delta(&self, s: &NlheState, f: f64) -> i64 {
        let pot = s.committed[0] + s.committed[1];
        let c = Self::to_call(s);
        if c == 0 {
            // Opening bet: at least one big blind.
            let raw = (f * pot as f64).round() as i64;
            raw.max(self.cfg.bb)
        } else {
            // Raise: raise-by at least the last full raise increment (>= bb).
            let raw_raise = (f * (pot + c) as f64).round() as i64;
            let min_raise = self.cfg.last_raise_floor(s);
            c + raw_raise.max(min_raise)
        }
    }

    /// The legal actions in the SAME order `next_state` indexes them.
    fn legal_actions(&self, s: &NlheState) -> ActionList {
        let mut acts = ActionList::new(Action::Fold);
        let p = s.to_act;
        let stackp = s.stack[p];
        let c = Self::to_call(s);

        if c > 0 {
            acts.push(Action::Fold);
        }
        if c == 0 {
            acts.push(Action::Check);
        } else if c < stackp {
            // A full call is possible without going all-in; if c >= stackp the
            // only way to call is all-in, represented by AllIn below (no dup).
            acts.push(Action::Call);
        }

        for &f in self.sized_fracs(s) {
            let delta = self.sized_delta(s, f);
            // Legal as a non-all-in bet/raise only if it leaves chips behind; an
            // amount that meets or exceeds the stack is exactly the all-in below.
            if delta < stackp {
                acts.push(Action::Bet(delta));
            }
        }

        if stackp > 0 {
            acts.push(Action::AllIn);
        }
        acts
    }

    /// Semantic labels for the legal actions, index-aligned with
    /// [`Nlhe::legal_actions`] / `next_state`. Derived FROM `legal_actions` (each
    /// `Bet(delta)` is matched back to the pot-fraction that produced it), so the two
    /// can never drift in order or legality. Read-only helper for `crate::analysis`.
    pub(crate) fn labeled_actions(&self, s: &NlheState) -> NodeList<ActLabel> {
        let fracs = self.sized_fracs(s);
        let mut labels = NodeList::new(ActLabel::Fold);
        for &a in self.legal_actions(s).iter() {
            labels.push(match a {
                Action::Fold => ActLabel::Fold,
                Action::Check => ActLabel::Check,
                Action::Call => ActLabel::Call,
                Action::AllIn => ActLabel::AllIn,
                Action::Bet(delta) => {
                    let frac = fracs
                        .iter()
                        .copied()
                        .find(|&f| self.sized_delta(s, f) == delta)
                        .unwrap_or(f64::NAN);
                    ActLabel::Bet(frac)
                }
            });
        }
        labels
    }

    /// The big blind in chips — the unit for head-to-head bb/100 scoring.
    pub(crate) fn big_blind(&self) -> i64 {
        self.cfg.bb
    }

    /// Apply an action (identified by its index in the legal-action list), returning
    /// the resulting state (with any phase transition). The history records the
    /// action INDEX, not a semantic letter: two different bet SIZES (e.g. the 0.5x
    /// and 1.0x pot opens) must map to different history tokens, or their lines
    /// collapse to one infoset key with inconsistent legal-action counts. The index
    /// is card-independent (legal actions depend only on the betting state), so this
    /// keeps the key card-abstracted while distinguishing every distinct line.
    fn apply(&self, s: &NlheState, action: Action, action_idx: usize) -> NlheState {
        let mut ns = s.clone();
        let p = s.to_act;
        ns.history.push(history_token(action_idx));

        match action {
            Action::Fold => {
                ns.folded = Some(p);
                return ns; // terminal; no further transition
            }
            Action::Check => {
                ns.acted[p] = true;
            }
            Action::Call => {
                let amount = Self::to_call(s).min(s.stack[p]);
                Self::commit(&mut ns, p, amount);
                ns.acted[p] = true;
            }
            Action::Bet(delta) => {
                let before_bet = Self::current_bet(s);
                Self::commit(&mut ns, p, delta);
                ns.acted[p] = true;
                let raise_by = ns.street_contrib[p] - before_bet;
                if raise_by > 0 {
                    ns.last_raise = raise_by;
                    ns.street_bets += 1;
                }
            }
            Action::AllIn => {
                let before_bet = Self::current_bet(s);
                Self::commit(&mut ns, p, s.stack[p]);
                ns.acted[p] = true;
                let raise_by = ns.street_contrib[p] - before_bet;
                if raise_by > 0 {
                    // An all-in that raises resets the min-raise reference.
                    ns.last_raise = raise_by.max(ns.last_raise);
                    ns.street_bets += 1;
                }
            }
        }

        ns.to_act = 1 - p;
        if Self::betting_closed(&ns) {
            self.close_street(&mut ns);
        }
        ns
    }

    /// Move `amount` chips from a player's stack into the pot.
    fn commit(ns: &mut NlheState, p: usize, amount: i64) {
        ns.stack[p] -= amount;
        ns.committed[p] += amount;
        ns.street_contrib[p] += amount;
        if ns.stack[p] == 0 {
            ns.all_in[p] = true;
        }
    }

    /// Betting on the current street just closed (and nobody folded): advance to
    /// the next dealing phase (or showdown), flagging a run-out if anyone is all-in.
    fn close_street(&self, ns: &mut NlheState) {
        if ns.all_in[0] || ns.all_in[1] {
            ns.no_more_betting = true;
        }
        ns.phase = match ns.phase {
            Phase::PreBet => Phase::DealFlop,
            // Flop-only games (board_cards == 3) go straight to showdown after
            // the flop betting closes; full games continue to the turn.
            Phase::FlopBet if self.cfg.board_cards == 3 => Phase::Showdown,
            Phase::FlopBet => Phase::DealTurn,
            Phase::TurnBet => Phase::DealRiver,
            Phase::RiverBet => Phase::Showdown,
            other => other, // unreachable for a Bet phase
        };
    }

    /// Enter a betting street: reset per-street state and seat the first actor
    /// (button/SB acts first preflop; the big blind acts first on later streets).
    fn enter_bet(ns: &mut NlheState, phase: Phase, first_actor: usize) {
        ns.phase = phase;
        ns.street_contrib = [0, 0];
        ns.acted = [false, false];
        ns.last_raise = 0;
        ns.street_bets = 0;
        ns.to_act = first_actor;
        ns.history.push('/');
    }

    /// The set of cards already dealt, as a used-index bitmask.
    fn used_mask(s: &NlheState) -> u64 {
        let mut m = 0u64;
        for i in 0..s.n_hole as usize {
            m |= 1u64 << s.hole[i / 2][i % 2].index();
        }
        for c in s.board.iter() {
            m |= 1u64 << c.index();
        }
        m
    }

    /// Deal one specific card into the next open slot, advancing the phase when a
    /// dealing round completes.
    fn deal_card(&self, s: &NlheState, card: Card) -> NlheState {
        let mut ns = s.clone();
        match ns.phase {
            Phase::DealHole => {
                let i = ns.n_hole as usize;
                ns.hole[i / 2][i % 2] = card;
                ns.n_hole += 1;
                if ns.n_hole == 4 {
                    // Blinds are already posted (at initial_state); open betting.
                    ns.phase = Phase::PreBet;
                    ns.to_act = 0;
                    ns.acted = [false, false];
                }
            }
            Phase::DealFlop | Phase::DealTurn | Phase::DealRiver => {
                ns.board.push(card);
                let target = NUM_RANKS_BOARD[ns.street_index_for_deal()];
                if ns.board.len() == target {
                    self.after_board_complete(&mut ns);
                }
            }
            _ => unreachable!("deal_card in a non-dealing phase"),
        }
        self.fill_buckets(&mut ns);
        ns
    }

    /// Fill `ns`'s bucket cache when this deal opened a betting street (the only
    /// nodes ever keyed); otherwise mark it empty. Called after every dealt card,
    /// the only way the cards of a state change besides `with_holes`.
    fn fill_buckets(&self, ns: &mut NlheState) {
        match &self.card_abs {
            CardAbstraction::Buckets(abs)
                if matches!(
                    ns.phase,
                    Phase::PreBet | Phase::FlopBet | Phase::TurnBet | Phase::RiverBet
                ) =>
            {
                ns.buckets = [
                    abs.bucket_of(ns.hole[0], &ns.board),
                    abs.bucket_of(ns.hole[1], &ns.board),
                ];
                ns.bucket_game = self.bucket_game;
            }
            _ => ns.bucket_game = 0,
        }
    }

    /// After a board street is fully dealt, either open its betting round or, on a
    /// run-out, deal straight through to showdown.
    fn after_board_complete(&self, ns: &mut NlheState) {
        match ns.phase {
            Phase::DealFlop => {
                if ns.no_more_betting {
                    // On a run-out, skip straight to showdown in a flop-only game.
                    ns.phase = if self.cfg.board_cards == 3 {
                        Phase::Showdown
                    } else {
                        Phase::DealTurn
                    };
                } else {
                    Self::enter_bet(ns, Phase::FlopBet, 1);
                }
            }
            Phase::DealTurn => {
                if ns.no_more_betting {
                    ns.phase = Phase::DealRiver;
                } else {
                    Self::enter_bet(ns, Phase::TurnBet, 1);
                }
            }
            Phase::DealRiver => {
                if ns.no_more_betting {
                    ns.phase = Phase::Showdown;
                } else {
                    Self::enter_bet(ns, Phase::RiverBet, 1);
                }
            }
            _ => unreachable!(),
        }
    }

    /// Winner at showdown: 0, 1, or None for a chopped pot. Scores 2 hole + the
    /// full board — five cards via `eval5` on the flop-only game, seven via
    /// `eval7_fast` on the real game — so both use the certified evaluator.
    fn showdown_winner(s: &NlheState) -> Option<usize> {
        let score = |p: usize| match s.board.len() {
            3 => eval5(&[
                s.hole[p][0],
                s.hole[p][1],
                s.board[0],
                s.board[1],
                s.board[2],
            ]),
            5 => eval7_fast(&[
                s.hole[p][0],
                s.hole[p][1],
                s.board[0],
                s.board[1],
                s.board[2],
                s.board[3],
                s.board[4],
            ]),
            n => unreachable!("showdown with {n} board cards"),
        };
        let (a, b) = (score(0), score(1));
        if a > b {
            Some(0)
        } else if b > a {
            Some(1)
        } else {
            None
        }
    }

    /// Net chips to each player at a terminal state. Only the matched portion of
    /// the two commitments is contested; unmatched excess is refunded. Zero-sum by
    /// construction: `net0 + net1 == 0`.
    pub fn payoffs(&self, s: &NlheState) -> (f64, f64) {
        let matched = s.committed[0].min(s.committed[1]);
        let refund = [s.committed[0] - matched, s.committed[1] - matched];
        // Chips each player receives from the contested pot (2 * matched).
        let gain = if let Some(folder) = s.folded {
            let mut g = [0i64, 0i64];
            g[1 - folder] = 2 * matched;
            g
        } else {
            match Self::showdown_winner(s) {
                Some(w) => {
                    let mut g = [0i64, 0i64];
                    g[w] = 2 * matched;
                    g
                }
                None => [matched, matched], // chop
            }
        };
        let net0 = (gain[0] + refund[0] - s.committed[0]) as f64;
        let net1 = (gain[1] + refund[1] - s.committed[1]) as f64;
        (net0, net1)
    }
}

/// Helper on the state so `sized_delta` can read the min-raise floor.
impl NlheState {
    /// Which board-deal street index (1=flop,2=turn,3=river) is being dealt.
    fn street_index_for_deal(&self) -> usize {
        match self.phase {
            Phase::DealFlop => 1,
            Phase::DealTurn => 2,
            Phase::DealRiver => 3,
            _ => 0,
        }
    }
}

/// A real-table action with an arbitrary chip amount — what an opponent (or the
/// table) actually does, as opposed to an index into the abstract grid. `Add` is
/// the chips ADDED by this action (a bet, a raise's call-plus-raise-by, or an
/// all-in when it equals the stack).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RealAct {
    Fold,
    Check,
    Call,
    Add(i64),
}

/// Real-chip play support for the play agent (`crate::play`). These reuse the
/// same betting state machine as the abstract tree (blinds, min-raise, street
/// closure, run-outs), so a real hand and its abstract translation can never
/// disagree about whose turn it is or when a street ends, except where the
/// translation itself changes the action.
impl Nlhe {
    pub(crate) fn cfg(&self) -> &NlheConfig {
        &self.cfg
    }

    /// Betting street 0..=3 when a player is to act, else `None` (dealing,
    /// showdown or folded).
    pub(crate) fn betting_street(s: &NlheState) -> Option<usize> {
        if s.folded.is_some() {
            return None;
        }
        match s.phase {
            Phase::PreBet => Some(0),
            Phase::FlopBet => Some(1),
            Phase::TurnBet => Some(2),
            Phase::RiverBet => Some(3),
            _ => None,
        }
    }

    pub(crate) fn is_dealing(s: &NlheState) -> bool {
        s.folded.is_none()
            && matches!(
                s.phase,
                Phase::DealHole | Phase::DealFlop | Phase::DealTurn | Phase::DealRiver
            )
    }

    pub(crate) fn acting(s: &NlheState) -> usize {
        s.to_act
    }

    pub(crate) fn pot(s: &NlheState) -> i64 {
        s.committed[0] + s.committed[1]
    }

    pub(crate) fn owed(s: &NlheState) -> i64 {
        Self::to_call(s)
    }

    pub(crate) fn stack_of(s: &NlheState, p: usize) -> i64 {
        s.stack[p]
    }

    pub(crate) fn committed_of(s: &NlheState, p: usize) -> i64 {
        s.committed[p]
    }

    /// Chips `p` has put in on the current street (blinds count preflop).
    pub(crate) fn street_contrib_of(s: &NlheState, p: usize) -> i64 {
        s.street_contrib[p]
    }

    pub(crate) fn board_len(s: &NlheState) -> usize {
        s.board.len()
    }

    pub(crate) fn betting_history(s: &NlheState) -> String {
        s.history.to_string()
    }

    /// Chips to add for the grid bet/raise of pot-fraction `f` at this node — the
    /// exact amount the abstract tree would use (min-bet / min-raise enforced).
    pub(crate) fn grid_delta(&self, s: &NlheState, f: f64) -> i64 {
        self.sized_delta(s, f)
    }

    /// Pot fractions of the grid's sized bets/raises at this node (all-in aside).
    pub(crate) fn grid_fracs(&self, s: &NlheState) -> &[f64] {
        self.sized_fracs(s)
    }

    /// Smallest legal non-all-in raise (chips to add) at this node.
    pub(crate) fn min_raise_add(&self, s: &NlheState) -> i64 {
        Self::to_call(s) + self.cfg.last_raise_floor(s)
    }

    /// Deal one given card into the next open slot (hole seat 0 card 0, seat 0
    /// card 1, seat 1 card 0, seat 1 card 1, then the board).
    pub(crate) fn deal_specific(&self, s: &NlheState, card: Card) -> NlheState {
        self.deal_card(s, card)
    }

    /// `s` with both seats' hole cards replaced: exactly the state replaying
    /// the same deal and actions with these holes gives, since nothing but
    /// the holes themselves depends on them (betting, board and history do not).
    pub(crate) fn with_holes(s: &NlheState, holes: [[Card; 2]; 2]) -> NlheState {
        debug_assert_eq!(s.n_hole, 4, "holes swapped in before the deal finished");
        let mut ns = s.clone();
        ns.hole = holes;
        ns.bucket_game = 0; // the cached buckets were the old holes'
        ns
    }

    /// Apply a real action with an arbitrary chip amount under the tree's rules.
    /// Unlike `apply`, no history token is recorded (real hands are not keyed, and
    /// a min-raise war can be arbitrarily long), so only street separators appear.
    pub(crate) fn apply_real(&self, s: &NlheState, act: RealAct) -> Result<NlheState, String> {
        if Self::betting_street(s).is_none() {
            return Err("no player is to act".into());
        }
        let p = s.to_act;
        let c = Self::to_call(s);
        let stackp = s.stack[p];
        let mut ns = s.clone();
        match act {
            RealAct::Fold => {
                if c == 0 {
                    return Err("fold with nothing to call (check instead)".into());
                }
                ns.folded = Some(p);
                return Ok(ns);
            }
            RealAct::Check => {
                if c != 0 {
                    return Err(format!("check facing a bet of {c}"));
                }
                ns.acted[p] = true;
            }
            RealAct::Call => {
                if c == 0 {
                    return Err("call with nothing to call (check instead)".into());
                }
                Self::commit(&mut ns, p, c.min(stackp));
                ns.acted[p] = true;
            }
            RealAct::Add(d) => {
                if d <= 0 || d > stackp {
                    return Err(format!("add {d} outside 1..={stackp}"));
                }
                if d <= c {
                    // Matching (or all-in for less): this is a call.
                    if d < c && d < stackp {
                        return Err(format!("add {d} is less than the {c} to call"));
                    }
                    return self.apply_real(s, RealAct::Call);
                }
                let raise_by = d - c;
                if d < stackp && raise_by < self.cfg.last_raise_floor(s) {
                    return Err(format!(
                        "raise by {raise_by} below the minimum {}",
                        self.cfg.last_raise_floor(s)
                    ));
                }
                let before_bet = Self::current_bet(s);
                Self::commit(&mut ns, p, d);
                ns.acted[p] = true;
                let by = ns.street_contrib[p] - before_bet;
                if by > 0 {
                    ns.last_raise = if d == stackp { by.max(ns.last_raise) } else { by };
                    ns.street_bets += 1;
                }
            }
        }
        ns.to_act = 1 - p;
        if Self::betting_closed(&ns) {
            self.close_street(&mut ns);
        }
        Ok(ns)
    }
}

impl NlheConfig {
    /// Minimum raise INCREMENT this street: the last full raise, or one big blind
    /// if there has been no raise yet.
    fn last_raise_floor(&self, s: &NlheState) -> i64 {
        s.last_raise.max(self.bb)
    }
}

/// History token for the action at index `idx` in the legal-action list. A single
/// digit (legal-action counts are small: fold, check/call, up to two sized bets
/// (three at F1's preflop root), all-in), so the compact `street|bucket|history`
/// key stays short. Distinct per action at a node — the property that keeps
/// different bet sizes on different lines.
fn history_token(idx: usize) -> char {
    debug_assert!(idx < 10, "unexpectedly many legal actions: {idx}");
    char::from(b'0' + (idx as u8))
}

impl Game for Nlhe {
    type State = NlheState;

    fn initial_state(&self) -> NlheState {
        let mut s = NlheState {
            phase: Phase::DealHole,
            hole: [[Card::default(); 2]; 2],
            n_hole: 0,
            board: Board::new(),
            stack: [self.cfg.stack, self.cfg.stack],
            committed: [0, 0],
            street_contrib: [0, 0],
            acted: [false, false],
            all_in: [false, false],
            to_act: 0,
            last_raise: self.cfg.bb, // preflop min-raise reference is the BB
            street_bets: 1,          // the big blind is the opening bet
            folded: None,
            no_more_betting: false,
            history: History::new(),
            buckets: [0, 0],
            bucket_game: 0,
        };
        // Post blinds: player 0 = button/SB, player 1 = big blind.
        Nlhe::commit(&mut s, 0, self.cfg.sb);
        Nlhe::commit(&mut s, 1, self.cfg.bb);
        s
    }

    fn is_terminal(&self, s: &NlheState) -> bool {
        s.folded.is_some() || s.phase == Phase::Showdown
    }

    fn terminal_utility(&self, s: &NlheState) -> f64 {
        self.payoffs(s).0
    }

    fn is_chance(&self, s: &NlheState) -> bool {
        matches!(
            s.phase,
            Phase::DealHole | Phase::DealFlop | Phase::DealTurn | Phase::DealRiver
        )
    }

    fn chance_outcomes(&self, s: &NlheState) -> Vec<(NlheState, f64)> {
        let used = Nlhe::used_mask(s);
        let remaining: Vec<Card> = self
            .cfg
            .deck
            .iter()
            .copied()
            .filter(|i| used & (1u64 << i) == 0)
            .map(Card::from_index)
            .collect();
        let p = 1.0 / remaining.len() as f64;
        remaining
            .into_iter()
            .map(|card| (self.deal_card(s, card), p))
            .collect()
    }

    /// Same undealt cards, same order and same uniform probability as
    /// `chance_outcomes`, but only the chosen card is dealt — the sampled hot loops
    /// no longer clone a state per remaining card to keep one.
    fn sample_chance_with<F: FnOnce(&[f64]) -> usize>(&self, s: &NlheState, choose: F) -> NlheState {
        let used = Nlhe::used_mask(s);
        let mut remaining = [Card::default(); 52];
        let mut len = 0;
        for &i in self.cfg.deck.iter().filter(|&&i| used & (1u64 << i) == 0) {
            remaining[len] = Card::from_index(i);
            len += 1;
        }
        let probabilities = [1.0 / len as f64; 52];
        let idx = choose(&probabilities[..len]);
        self.deal_card(s, remaining[idx])
    }

    fn current_player(&self, s: &NlheState) -> usize {
        s.to_act
    }

    fn num_actions(&self, s: &NlheState) -> usize {
        self.legal_actions(s).len()
    }

    fn next_state(&self, s: &NlheState, action: usize) -> NlheState {
        let acts = self.legal_actions(s);
        self.apply(s, acts[action], action)
    }

    /// `num_actions` and `next_state` from one `legal_actions` list.
    fn with_children<R>(
        &self,
        s: &NlheState,
        f: impl FnOnce(usize, &dyn Fn(usize) -> NlheState) -> R,
    ) -> R {
        let acts = self.legal_actions(s);
        f(acts.len(), &|a| self.apply(s, acts[a], a))
    }

    fn infoset_key(&self, s: &NlheState) -> String {
        // Thin wrapper over `infoset_key_into` (below) so the two can never drift:
        // one definition of the key bytes, materialized as a `String` for the
        // Python oracle / blueprint save / structural walks. The hot loop never
        // takes this path — it writes bytes into a reused buffer via `_into`.
        let mut buf = Vec::with_capacity(32);
        self.infoset_key_into(s, &mut buf);
        // The key is ASCII by construction (tag char, decimal bucket / dotted
        // canonical cards, digit history), so this is a lossless, exact round-trip.
        String::from_utf8(buf).expect("nlhe infoset key is valid ASCII")
    }

    fn infoset_key_into(&self, s: &NlheState, buf: &mut Vec<u8>) {
        // `street | card token | betting history` (perfect recall on betting), byte
        // for byte the old `format!("{}|{}|{}", ...)` — appended directly to `buf`
        // with no intermediate `String`. The card token collapses strategically-
        // identical holdings: Buckets mode uses the Phase-3b abstraction (preflop =
        // the 169 exact classes, postflop = the bucket tables), Exact mode the
        // lossless suit-isomorphic canonical (hole, board).
        let p = s.to_act;
        buf.push(phase_street_tag(s.phase) as u8);
        buf.push(b'|');
        match &self.card_abs {
            CardAbstraction::Buckets(abs) => {
                let bucket = if s.bucket_game == self.bucket_game {
                    s.buckets[p]
                } else {
                    abs.bucket_of(s.hole[p], &s.board)
                };
                debug_assert_eq!(bucket, abs.bucket_of(s.hole[p], &s.board), "stale bucket cache");
                push_u32_decimal(buf, bucket)
            }
            // Exact mode is the small-game B3 path, not the 100bb hot loop, so the
            // one allocation here is immaterial; reusing `exact_card_token` keeps its
            // bytes provably identical to `infoset_key`'s former output.
            CardAbstraction::Exact => {
                buf.extend_from_slice(exact_card_token(s.hole[p], &s.board).as_bytes())
            }
        }
        buf.push(b'|');
        buf.extend_from_slice(s.history.as_bytes());
    }
}

/// Append the base-10 digits of `n` to `buf` without allocating an intermediate
/// `String` — the byte-for-byte equivalent of `n.to_string()` for the bucket id in
/// the infoset key. Buckets are small (<= 9999 across all streets), so a 10-digit
/// scratch is always ample.
#[inline]
fn push_u32_decimal(buf: &mut Vec<u8>, mut n: u32) {
    if n == 0 {
        buf.push(b'0');
        return;
    }
    let mut tmp = [0u8; 10];
    let mut i = tmp.len();
    while n > 0 {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    buf.extend_from_slice(&tmp[i..]);
}

/// Lossless suit-isomorphic card token: the canonical `(hole, board)` as sorted
/// 0..52 indices, hole and board separated by ':' so the roles stay distinct
/// (holding the flush is not the flush on board). Used by the Exact game.
fn exact_card_token(hole: [Card; 2], board: &[Card]) -> String {
    let canon = crate::abstraction::canonical_situation(&[hole[0], hole[1]], board);
    let part = |cs: &[u8]| cs.iter().map(u8::to_string).collect::<Vec<_>>().join(".");
    format!("{}:{}", part(&canon[..2]), part(&canon[2..]))
}

/// Single-char street tag for the infoset key. `infoset_key` is only ever called
/// at a betting node (never at a chance or terminal node), so only the four
/// betting phases are reachable.
fn phase_street_tag(phase: Phase) -> char {
    match phase {
        Phase::PreBet => 'P',
        Phase::FlopBet => 'F',
        Phase::TurnBet => 'T',
        Phase::RiverBet => 'R',
        other => panic!("infoset_key in non-betting phase {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal SplitMix64 for the randomized rollout (same generator the rest of
    /// the engine uses).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    fn game() -> Nlhe {
        Nlhe::new(NlheConfig::pluribus_100bb())
    }

    #[test]
    fn blinds_posted_at_start() {
        let g = game();
        let s = g.initial_state();
        assert_eq!(s.committed, [50, 100]);
        assert_eq!(s.stack, [9950, 9900]);
        assert!(g.is_chance(&s)); // first action is dealing hole cards
    }

    #[test]
    fn public_model_keys_exclude_private_cards() {
        let g = game();
        let aces_vs_kings = deal_holes(&g, [48, 49, 44, 45]);
        let low_cards = deal_holes(&g, [0, 1, 4, 5]);
        assert_ne!(g.infoset_key(&aces_vs_kings), g.infoset_key(&low_cards));
        assert_eq!(
            g.public_model_keys(&aces_vs_kings),
            g.public_model_keys(&low_cards),
            "public observation key must be invariant to both private holdings"
        );
    }

    /// Deal a fixed set of hole+board cards, returning the state at the point
    /// preflop betting opens (all 4 hole cards dealt).
    fn deal_holes(g: &Nlhe, cards: [u8; 4]) -> NlheState {
        let mut s = g.initial_state();
        for &c in &cards {
            s = g.deal_card(&s, Card::from_index(c));
        }
        assert_eq!(s.phase, Phase::PreBet);
        s
    }

    fn act(g: &Nlhe, s: &NlheState, a: Action) -> NlheState {
        let acts = g.legal_actions(s);
        let i = acts.iter().position(|x| *x == a).expect("action not legal");
        g.next_state(s, i)
    }

    #[test]
    fn sb_fold_preflop_loses_small_blind() {
        let g = game();
        let s = deal_holes(&g, [0, 1, 2, 3]);
        // SB (player 0) faces the big blind: fold is legal, and folding loses the
        // 50 chips already matched.
        let t = act(&g, &s, Action::Fold);
        assert!(g.is_terminal(&t));
        let (n0, n1) = g.payoffs(&t);
        assert_eq!((n0, n1), (-50.0, 50.0));
    }

    #[test]
    fn preflop_limp_check_advances_to_flop() {
        let g = game();
        let s = deal_holes(&g, [0, 1, 2, 3]);
        let s = act(&g, &s, Action::Call); // SB limps to 100
        assert_eq!(s.phase, Phase::PreBet); // BB still has the option
        let s = act(&g, &s, Action::Check); // BB checks
        assert_eq!(s.phase, Phase::DealFlop);
        assert_eq!(s.committed, [100, 100]);
    }

    #[test]
    fn legal_action_grid_matches_config() {
        let g = game();
        let s = deal_holes(&g, [0, 1, 2, 3]);
        // Preflop SB facing the BB: fold, call, one pot-sized raise, all-in.
        let acts = g.legal_actions(&s);
        assert!(acts.contains(&Action::Fold));
        assert!(acts.contains(&Action::Call));
        assert!(acts.contains(&Action::AllIn));
        assert_eq!(
            acts.iter().filter(|a| matches!(a, Action::Bet(_))).count(),
            1,
            "one pot-sized preflop raise expected"
        );

        // Postflop first-in (limp/check to a flop): check, two bets, all-in.
        let s = act(&g, &s, Action::Call);
        let s = act(&g, &s, Action::Check);
        let mut s = s;
        for _ in 0..3 {
            s = g.chance_outcomes(&s)[0].0.clone();
        }
        assert_eq!(s.phase, Phase::FlopBet);
        let acts = g.legal_actions(&s);
        assert!(acts.contains(&Action::Check));
        assert!(acts.contains(&Action::AllIn));
        assert_eq!(
            acts.iter().filter(|a| matches!(a, Action::Bet(_))).count(),
            2,
            "0.5x and 1.0x pot bets expected"
        );
    }

    #[test]
    fn all_in_call_runs_out_board_to_showdown() {
        let g = game();
        // Aces vs kings all-in preflop: deal a board that keeps aces ahead.
        // hole p0 = Ac Ad (indices for A: rank 12 -> 48..51); p1 = Kc Kd.
        let ac = 48;
        let ad = 49;
        let kc = 44;
        let kd = 45;
        let mut s = g.initial_state();
        for &c in &[ac, ad, kc, kd] {
            s = g.deal_card(&s, Card::from_index(c));
        }
        let s = act(&g, &s, Action::AllIn); // SB shoves 100bb
                                            // BB's call requires its entire remaining stack, so the legal action is
                                            // the all-in call (not a plain Call — that would be a duplicate).
        let mut s = act(&g, &s, Action::AllIn);
        assert!(s.no_more_betting);
        // Run out five community cards (all forced chance).
        while g.is_chance(&s) {
            s = g.chance_outcomes(&s)[0].0.clone();
        }
        assert!(g.is_terminal(&s));
        assert_eq!(s.committed, [10_000, 10_000]);
        let (n0, n1) = g.payoffs(&s);
        assert_eq!(n0 + n1, 0.0);
        assert!(n0 == 10_000.0 || n1 == 10_000.0 || n0 == 0.0);
    }

    #[test]
    fn chip_conservation_and_zero_sum_over_random_playouts() {
        // The heavy net: many random legal playouts must always conserve chips and
        // pay off zero-sum. Catches essentially every betting-logic bug.
        let g = game();
        let total = 2 * g.cfg.stack;
        let mut rng = Rng(0xA11CE);
        for _ in 0..200_000 {
            let mut s = g.initial_state();
            loop {
                // Chips are never created or destroyed, only moved stack->pot.
                assert_eq!(
                    s.stack[0] + s.stack[1] + s.committed[0] + s.committed[1],
                    total,
                    "chip conservation violated"
                );
                if g.is_terminal(&s) {
                    break;
                }
                if g.is_chance(&s) {
                    let outs = g.chance_outcomes(&s);
                    s = outs[rng.below(outs.len())].0.clone();
                } else {
                    let n = g.num_actions(&s);
                    assert!(n >= 1, "decision node with no legal actions");
                    s = g.next_state(&s, rng.below(n));
                }
            }
            let (n0, n1) = g.payoffs(&s);
            assert_eq!(n0 + n1, 0.0, "payoff not zero-sum");
            // Neither player can lose more than they staked.
            assert!(n0 >= -(g.cfg.stack as f64) && n1 >= -(g.cfg.stack as f64));
        }
    }

    #[test]
    fn phase_street_tag_maps_each_betting_street() {
        assert_eq!(phase_street_tag(Phase::PreBet), 'P');
        assert_eq!(phase_street_tag(Phase::FlopBet), 'F');
        assert_eq!(phase_street_tag(Phase::TurnBet), 'T');
        assert_eq!(phase_street_tag(Phase::RiverBet), 'R');
    }

    /// End-to-end check of the abstracted infoset key against the real Phase-3b
    /// tables: the key is `street | acting-player bucket | history`, and it tracks
    /// whose turn it is. Ignored by default because it loads the ~277 MB baked
    /// bucket tables; run with `cargo test -- --ignored infoset_key`.
    #[test]
    #[ignore = "loads the ~277MB baked bucket tables; run explicitly"]
    fn infoset_key_is_street_bucket_history_for_acting_player() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/abstraction");
        if !dir.join("river_buckets.npy").exists() {
            eprintln!("baked bucket tables absent at {dir:?}; skipping");
            return;
        }
        let abs = Arc::new(Abstraction::load(&dir).expect("load abstraction"));
        let g = Nlhe::with_abstraction(NlheConfig::pluribus_100bb(), abs);

        // Preflop: p0 = Ac Ad (indices 48,49), p1 = Kc Kd (44,45). SB (p0) acts.
        let s = deal_holes(&g, [48, 49, 44, 45]);
        let aa = crate::abstraction::preflop_index(Card::from_index(48), Card::from_index(49));
        let kk = crate::abstraction::preflop_index(Card::from_index(44), Card::from_index(45));
        assert_eq!(g.infoset_key(&s), format!("P|{aa}|"), "SB (AA) preflop key");

        // After the SB calls, it is the BB's turn: same street, BB's bucket, and
        // the history now carries the call as its action index (Call is index 1 in
        // the preflop legal list [Fold, Call, Bet, AllIn]).
        let s2 = act(&g, &s, Action::Call);
        assert_eq!(s2.phase, Phase::PreBet);
        assert_eq!(
            g.infoset_key(&s2),
            format!("P|{kk}|1"),
            "BB (KK) preflop key"
        );
        assert_ne!(
            aa, kk,
            "the two holdings must fall in different preflop classes"
        );
    }

    /// The per-street bucket cache is only a cache: along random 200bb deals and
    /// lines, every decision node's key from the game that filled the cache equals
    /// the key a second game on the same tables recomputes (its number differs, so
    /// it never reads the cache), and both seats' cached buckets equal `bucket_of`.
    /// `with_holes` must drop the cache. Ignored by default: loads the tables.
    #[test]
    #[ignore = "loads the ~277MB baked bucket tables; run explicitly"]
    fn bucket_cache_matches_recomputed_keys() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/abstraction");
        if !dir.join("river_buckets.npy").exists() {
            eprintln!("baked bucket tables absent at {dir:?}; skipping");
            return;
        }
        let abs = Arc::new(Abstraction::load(&dir).expect("load abstraction"));
        let g = Nlhe::with_abstraction(NlheConfig::deep_200bb(), Arc::clone(&abs));
        let fresh = Nlhe::with_abstraction(NlheConfig::deep_200bb(), Arc::clone(&abs));
        assert_ne!(g.bucket_game, fresh.bucket_game);
        let mut rng = Rng(0x00B0_C4E7);
        let mut keyed = [0usize; 4];
        for _ in 0..100_000 {
            let mut s = g.initial_state();
            while !g.is_terminal(&s) {
                if g.is_chance(&s) {
                    s = g.sample_chance_with(&s, |p| rng.below(p.len()));
                    continue;
                }
                assert_eq!(s.bucket_game, g.bucket_game, "betting node without a filled cache");
                for p in 0..2 {
                    assert_eq!(s.buckets[p], abs.bucket_of(s.hole[p], &s.board));
                }
                assert_eq!(g.infoset_key(&s), fresh.infoset_key(&s));
                keyed[s.board.len().saturating_sub(2)] += 1;
                if s.board.len() == 5 && keyed[3] % 7 == 0 {
                    let swapped = Nlhe::with_holes(&s, [s.hole[1], s.hole[0]]);
                    assert_eq!(swapped.bucket_game, 0, "with_holes kept the old holes' buckets");
                    let want = abs.bucket_of(s.hole[1 - s.to_act], &s.board);
                    let key = g.infoset_key(&swapped);
                    assert_eq!(key.split('|').nth(1), Some(want.to_string().as_str()));
                }
                s = g.next_state(&s, rng.below(g.num_actions(&s)));
            }
        }
        assert!(keyed.iter().all(|&n| n > 1_000), "every street keyed often: {keyed:?}");
    }
}

#[cfg(test)]
mod history_bound {
    use super::*;
    use crate::game::Game;

    /// The inline `History` buffer hard-asserts at HISTORY_CAP bytes. Walk the whole
    /// 100bb betting tree (betting never depends on the cards, so one fixed deal
    /// suffices) and require the longest history to stay well inside the cap —
    /// fails here, not mid-solve, if a bet grid change ever makes the tree deeper.
    #[test]
    fn pluribus_100bb_history_fits_inline_buffer() {
        fn walk(g: &Nlhe, s: &NlheState, max: &mut usize, nodes: &mut u64) {
            *max = (*max).max(s.history.len as usize);
            if g.is_terminal(s) {
                return;
            }
            if g.is_chance(s) {
                return walk(g, &g.sample_chance_with(s, |_| 0), max, nodes);
            }
            *nodes += 1;
            for a in 0..g.num_actions(s) {
                walk(g, &g.next_state(s, a), max, nodes);
            }
        }
        // Node counts cross-checked by an independent re-implementation of the
        // rules (the 200 bb grid's count was predicted before it was built).
        for (name, want) in [("100bb", 3986u64), ("200bb", 11_276)] {
            let g = Nlhe::new(NlheConfig::grid(name).unwrap());
            let (mut max, mut nodes) = (0usize, 0u64);
            walk(&g, &g.initial_state(), &mut max, &mut nodes);
            assert_eq!(nodes, want, "{name}: betting tree changed; re-check the bound");
            assert!(2 * max <= HISTORY_CAP, "{name}: longest history {max} B vs cap {HISTORY_CAP} B");
        }
        // F1: the small blind's open has three sizes instead of one, and every
        // subtree under a given first action is the 200 bb grid's. So the tree is
        // the 200 bb tree plus the subtrees under the two new opens, and the
        // subtree under the 3 bb open, the limp and the fold are node for node
        // the 200 bb grid's.
        let deep = Nlhe::new(NlheConfig::grid("200bb").unwrap());
        let f1 = Nlhe::new(NlheConfig::grid("200bb-f1").unwrap());
        let root = |g: &Nlhe| {
            let mut s = g.initial_state();
            while g.is_chance(&s) {
                s = g.sample_chance_with(&s, |_| 0);
            }
            s
        };
        let count = |g: &Nlhe, s: &NlheState| {
            let (mut max, mut nodes) = (0usize, 0u64);
            walk(g, s, &mut max, &mut nodes);
            nodes
        };
        let (rd, rf) = (root(&deep), root(&f1));
        let ld: Vec<ActLabel> = deep.labeled_actions(&rd).iter().copied().collect();
        let lf: Vec<ActLabel> = f1.labeled_actions(&rf).iter().copied().collect();
        assert_eq!(ld.len() + 2, lf.len(), "{ld:?} vs {lf:?}");
        let mut extra = 0u64;
        for (i, l) in lf.iter().enumerate() {
            let sub = count(&f1, &f1.next_state(&rf, i));
            match ld.iter().position(|d| d == l) {
                Some(j) => assert_eq!(sub, count(&deep, &deep.next_state(&rd, j)), "{l:?}"),
                None => extra += sub,
            }
        }
        assert_eq!(count(&f1, &rf), count(&deep, &rd) + extra);
        assert!(extra > 0);
        assert!(NlheConfig::grid("").is_some() && NlheConfig::grid("300bb").is_none());
        for name in NlheConfig::GRIDS {
            assert!(NlheConfig::grid(name).is_some());
        }
    }

    #[test]
    fn deep_grid_has_sized_three_four_and_five_bets() {
        let g = Nlhe::new(NlheConfig::deep_200bb());
        let mut s = g.initial_state();
        while g.is_chance(&s) {
            s = g.sample_chance_with(&s, |_| 0);
        }
        // Each step: the labels at the node, then take the (only) sized raise.
        let mut adds = Vec::new();
        for expect_sized in [true, true, true, true, false] {
            let labels: Vec<ActLabel> = g.labeled_actions(&s).iter().copied().collect();
            let sized: Vec<usize> =
                labels.iter().enumerate().filter(|(_, l)| matches!(l, ActLabel::Bet(_))).map(|(i, _)| i).collect();
            assert!(labels.iter().any(|l| matches!(l, ActLabel::AllIn)));
            if !expect_sized {
                assert!(sized.is_empty(), "a 6-bet is all-in only: {labels:?}");
                break;
            }
            assert_eq!(sized.len(), 1, "{labels:?}");
            let p = s.to_act;
            let before = s.committed[p];
            s = g.next_state(&s, sized[0]);
            adds.push(s.committed[p]);
            assert!(s.committed[p] > before);
        }
        // Total committed after the open, 3-bet, 4-bet, 5-bet (pot-size raises).
        assert_eq!(adds, vec![300, 900, 2700, 8100]);
        // F1: the small blind opens to 2, 2.5 or 3 bb (pot-fraction 0.5, 0.75,
        // 1.0 of the raise); the big blind's 3-bet after each is pot-size.
        let g = Nlhe::new(NlheConfig::grid("200bb-f1").unwrap());
        let mut s = g.initial_state();
        while g.is_chance(&s) {
            s = g.sample_chance_with(&s, |_| 0);
        }
        let labels: Vec<ActLabel> = g.labeled_actions(&s).iter().copied().collect();
        assert_eq!(
            labels,
            vec![ActLabel::Fold, ActLabel::Call, ActLabel::Bet(0.5), ActLabel::Bet(0.75), ActLabel::Bet(1.0), ActLabel::AllIn]
        );
        let mut opens = Vec::new();
        for (i, l) in labels.iter().enumerate() {
            if let ActLabel::Bet(_) = l {
                let t = g.next_state(&s, i);
                let three = g.labeled_actions(&t).iter().position(|l| matches!(l, ActLabel::Bet(_))).unwrap();
                assert_eq!(g.labeled_actions(&t).iter().filter(|l| matches!(l, ActLabel::Bet(_))).count(), 1);
                let u = g.next_state(&t, three);
                opens.push((t.committed[0], u.committed[1]));
            }
        }
        // (the open, the 3-bet): 3x the open, as a pot-size raise of it.
        assert_eq!(opens, vec![(200, 600), (250, 750), (300, 900)]);
        // A limp is the 200 bb grid's: the big blind may raise 0.5 or 1.0 pot.
        let call = labels.iter().position(|l| *l == ActLabel::Call).unwrap();
        let t = g.next_state(&s, call);
        let lt: Vec<ActLabel> = g.labeled_actions(&t).iter().copied().collect();
        assert_eq!(lt, vec![ActLabel::Check, ActLabel::Bet(0.5), ActLabel::Bet(1.0), ActLabel::AllIn]);
        // The 100 bb grid: the 3-bet is all-in only.
        let g = Nlhe::new(NlheConfig::pluribus_100bb());
        let mut s = g.initial_state();
        while g.is_chance(&s) {
            s = g.sample_chance_with(&s, |_| 0);
        }
        let open = g.labeled_actions(&s).iter().position(|l| matches!(l, ActLabel::Bet(_))).unwrap();
        s = g.next_state(&s, open);
        assert!(!g.labeled_actions(&s).iter().any(|l| matches!(l, ActLabel::Bet(_))));
    }
}
