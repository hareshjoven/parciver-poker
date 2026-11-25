//! Real-time subgame search: an exact-card, range-vs-range solver for the last
//! betting rounds of a hand — the river alone, or the turn followed by every
//! possible river card and the river betting round.
//!
//! Unlike the blueprint, nothing here is abstracted about the cards: every
//! two-card holding consistent with the board is its own "hand" in both players'
//! ranges, and card removal between the two holdings (and the river card) is
//! exact. The betting tree is built from real chip amounts with a configurable
//! (richer-than-blueprint) size menu, and any bets already made this round can be
//! forced into it exactly, so the opponent's real size needs no translation.
//! A forced action is either the only branch at its node (`Forced::Only`: its
//! actor's range is conditioned on it outside the solve), one branch among the
//! menu's (`Forced::Branch`: the actor's choice is solved, the real line runs
//! through it), or a fixed mixed strategy (`Forced::Frozen`: the actor's real
//! per-holding probabilities, frozen in the solver).
//!
//! The solver is Discounted CFR (Brown & Sandholm 2019; alpha=1.5, beta=0,
//! gamma=2) over range vectors with alternating updates. Terminal values are
//! computed in O(n) per node after a one-off strength sort, with blockers handled
//! by per-card inclusion-exclusion; `showdown_values_naive` is the O(n^2) oracle
//! the tests check it against. Exploitability (both players' best-response
//! values against the average strategy) is computed exactly on the same tree.
//!
//! Turn solves: when the turn round ends without a fold, a chance node deals each
//! of the 48 unseen river cards. All vectors stay indexed by the TURN holdings;
//! the holdings that contain the river card are dead in that branch (zero reach,
//! zero value). Every pair of disjoint holdings sees exactly 44 river cards, so
//! the chance node averages its branches with weight 1/44. The 48 river subtrees
//! own disjoint, contiguous ranges of the regret storage and are solved in
//! parallel; results are summed in card order, so the numbers do not depend on
//! the thread count.
//!
//! Flop solves (the depth-limited flop search, FLOPSEARCH.md D3): the flop round
//! only. Where it ends without a fold there is a `Leaf` (chips behind; the
//! external solver values it with the turn-start value net) or a `Runout` (a
//! player is all-in: showdown over every turn and river card). This crate's
//! `Solver` has no net: it values a leaf as a check-down, like a runout, which
//! makes it the exact solver of the game with no betting after the flop (the
//! oracle for the external solver's leaf plumbing, and the check-down ablation).
//! Every pair of disjoint holdings sees C(45, 2) = 990 runouts.
//!
//! The full flop game (`Tree::build_full`, FLOPSEARCH.md D4's oracle): the flop
//! round, then a chance node over the 49 turn cards (each disjoint pair sees
//! 45), the turn round, a chance node over the 48 river cards (44) and the
//! river round, scored on the runout view of the turn and river pair. It is
//! far too big to play with; it scores the flop search offline.

use std::cell::RefCell;
use std::ops::Range;

use rayon::prelude::*;

use crate::cards::Card;
use crate::hand_eval::eval7_fast;

/// All two-card holdings that do not touch the board, plus one showdown "view"
/// per complete five-card board the solve can reach.
pub(crate) struct Hands {
    pub combos: Vec<[u8; 2]>,
    /// The board the solve starts from (3 cards for a flop solve, 4 for a
    /// turn solve, 5 for river).
    pub board: Vec<u8>,
    views: Vec<View>,
    /// The leading views a flop runout averages over: all of them, except in
    /// `flop_full`'s hands, which add one masking view per turn card after
    /// the runouts.
    runouts: usize,
}

/// One complete five-card board: which holdings are live on it and their
/// showdown strength order.
struct View {
    /// The river card this view adds (`None` for a river solve).
    #[cfg_attr(not(test), allow(dead_code))]
    card: Option<u8>,
    /// Live holdings in ascending strength order.
    order: Vec<usize>,
    /// Strength per holding (0 for dead holdings).
    strength: Vec<u32>,
    /// Holdings that contain `card`, or a runout card (empty for a river solve).
    dead: Vec<usize>,
}

fn holdings_off(dead: u64) -> Vec<[u8; 2]> {
    let mut combos = Vec::with_capacity(1128);
    for a in 0..52u8 {
        for b in a + 1..52 {
            if dead & (1u64 << a) == 0 && dead & (1u64 << b) == 0 {
                combos.push([a, b]);
            }
        }
    }
    combos
}

fn make_view(combos: &[[u8; 2]], board: &[u8; 5], card: Option<u8>) -> View {
    make_view_off(combos, board, card, card.map_or(0, |x| 1u64 << x))
}

/// `make_view` with the holdings that touch `off` (a card mask) dead.
fn make_view_off(combos: &[[u8; 2]], board: &[u8; 5], card: Option<u8>, off: u64) -> View {
    let c = |i: u8| Card::from_index(i);
    let mut strength = vec![0u32; combos.len()];
    let mut order = Vec::with_capacity(combos.len());
    let mut dead = Vec::new();
    for (i, h) in combos.iter().enumerate() {
        if off & (1u64 << h[0] | 1u64 << h[1]) != 0 {
            dead.push(i);
            continue;
        }
        strength[i] = eval7_fast(&[
            c(h[0]),
            c(h[1]),
            c(board[0]),
            c(board[1]),
            c(board[2]),
            c(board[3]),
            c(board[4]),
        ]);
        order.push(i);
    }
    order.sort_by_key(|&i| strength[i]);
    View { card, order, strength, dead }
}

impl Hands {
    pub(crate) fn river(board: &[u8; 5]) -> Hands {
        let combos = holdings_off(board.iter().fold(0u64, |m, &c| m | 1u64 << c));
        let views = vec![make_view(&combos, board, None)];
        Hands { combos, board: board.to_vec(), runouts: views.len(), views }
    }

    /// Turn holdings (1128) with one view per unseen river card (48).
    pub(crate) fn turn(board: &[u8; 4]) -> Hands {
        let used = board.iter().fold(0u64, |m, &c| m | 1u64 << c);
        let combos = holdings_off(used);
        let views = (0..52u8)
            .filter(|&c| used & (1u64 << c) == 0)
            .map(|c| make_view(&combos, &[board[0], board[1], board[2], board[3], c], Some(c)))
            .collect::<Vec<_>>();
        Hands { combos, board: board.to_vec(), runouts: views.len(), views }
    }

    /// Flop holdings (1176). With `runouts`, one view per unordered turn and
    /// river pair (1176, in `(turn, river)` ascending order), which this
    /// crate's `Solver` needs; an external solver does not (`flop_runouts`).
    pub(crate) fn flop(board: &[u8; 3], runouts: bool) -> Hands {
        let used = board.iter().fold(0u64, |m, &c| m | 1u64 << c);
        let combos = holdings_off(used);
        let views = if runouts {
            let pairs: Vec<[u8; 2]> = holdings_off(used);
            pairs
                .par_iter()
                .map(|&[t, r]| make_view_off(&combos, &[board[0], board[1], board[2], t, r], None, 1u64 << t | 1u64 << r))
                .collect()
        } else {
            Vec::new()
        };
        Hands { combos, board: board.to_vec(), runouts: views.len(), views }
    }

    /// Flop holdings for the full flop game (`Tree::build_full`, the D4
    /// oracle): `flop(board, true)`'s 1176 runout views, then one view per
    /// unseen turn card (ascending) that only kills the holdings holding it.
    pub(crate) fn flop_full(board: &[u8; 3]) -> Hands {
        let mut hands = Hands::flop(board, true);
        let used = board.iter().fold(0u64, |m, &c| m | 1u64 << c);
        let n = hands.len();
        for t in (0..52u8).filter(|&c| used & (1u64 << c) == 0) {
            let dead = (0..n).filter(|&h| hands.combos[h].contains(&t)).collect();
            hands.views.push(View { card: Some(t), order: Vec::new(), strength: vec![0; n], dead });
        }
        hands
    }

    /// Showdown strength per holding on each runout view, view-major (0 for
    /// the holdings a runout card kills).
    pub(crate) fn view_strengths(&self) -> Vec<u32> {
        self.views.iter().flat_map(|v| v.strength.iter().copied()).collect()
    }

    /// `flop(board, true).view_strengths()` for an external solver
    /// (`flop_runouts`), built `apart`: the flop search asks for it from a
    /// solver callback under its locks, on a match worker.
    pub(crate) fn flop_runout_strengths(board: &[u8; 3]) -> Vec<u32> {
        apart(|| Hands::flop(board, true).view_strengths())
    }
}

/// Runs `f` on a rayon pool that runs nothing else, entered from a plain
/// thread. A rayon worker that waits on rayon work steals its pool's other
/// jobs meanwhile: a match worker building runouts inside the flop search's
/// callback ran another deal, whose search re-entered the callback on the same
/// thread and blocked on the lock that thread held (D4's M2, 7 Oct). A thread
/// outside every pool waits without running anything. Never call it from the
/// runouts pool itself (its workers would wait on their own pool).
fn apart<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    let pool = POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new().thread_name(|i| format!("runouts-{i}")).build().expect("the runouts pool")
    });
    std::thread::scope(|s| s.spawn(|| pool.install(f)).join()).unwrap_or_else(|e| std::panic::resume_unwind(e))
}

impl Hands {

    pub(crate) fn len(&self) -> usize {
        self.combos.len()
    }

    /// Number of complete boards (1 for a river solve, 48 for a turn solve).
    pub(crate) fn view_count(&self) -> usize {
        self.views.len()
    }

    /// Sum of `reach` over the opponent holdings compatible with each holding
    /// (sharing no card with it).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn compatible_mass(&self, reach: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.len()];
        self.compatible_mass_into(reach, &mut out);
        out
    }

    /// `compatible_mass` written into `out`. Per-card sums here and in the
    /// showdown sweep sit in 64 slots indexed `c & 63`: a card is always < 52,
    /// so the mask changes nothing but drops the bounds check.
    fn compatible_mass_into(&self, reach: &[f64], out: &mut [f64]) {
        let mut card = [0.0f64; 64];
        let mut total = 0.0;
        for (h, &r) in self.combos.iter().zip(reach) {
            card[h[0] as usize & 63] += r;
            card[h[1] as usize & 63] += r;
            total += r;
        }
        for ((o, h), &r) in out.iter_mut().zip(&self.combos).zip(reach) {
            *o = total - card[h[0] as usize & 63] - card[h[1] as usize & 63] + r;
        }
    }

    /// Per live holding `h` on `view`: sum over compatible live opponent holdings
    /// `j` of `reach[j] * sign(strength[h] - strength[j])` — the showdown value
    /// per chip at stake (0 for dead holdings). O(n) sweeps over the strength
    /// order with per-card sums for the blockers.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn showdown_values(&self, view: usize, reach: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.len()];
        self.showdown_values_into(view, reach, &mut out);
        out
    }

    /// `showdown_values` written into `out`.
    fn showdown_values_into(&self, view: usize, reach: &[f64], out: &mut [f64]) {
        let v = &self.views[view];
        out.fill(0.0);
        let m = v.order.len();
        // Ascending sweep: mass strictly weaker than each group (wins).
        let mut card = [0.0f64; 64];
        let mut total = 0.0;
        let mut i = 0;
        while i < m {
            let s = v.strength[v.order[i]];
            let mut j = i;
            while j < m && v.strength[v.order[j]] == s {
                j += 1;
            }
            for &h in &v.order[i..j] {
                let [a, b] = self.combos[h];
                out[h] += total - card[a as usize & 63] - card[b as usize & 63];
            }
            for &h in &v.order[i..j] {
                let [a, b] = self.combos[h];
                card[a as usize & 63] += reach[h];
                card[b as usize & 63] += reach[h];
                total += reach[h];
            }
            i = j;
        }
        // Descending sweep: mass strictly stronger (losses).
        let mut card = [0.0f64; 64];
        let mut total = 0.0;
        let mut i = m;
        while i > 0 {
            let s = v.strength[v.order[i - 1]];
            let mut j = i;
            while j > 0 && v.strength[v.order[j - 1]] == s {
                j -= 1;
            }
            for &h in &v.order[j..i] {
                let [a, b] = self.combos[h];
                out[h] -= total - card[a as usize & 63] - card[b as usize & 63];
            }
            for &h in &v.order[j..i] {
                let [a, b] = self.combos[h];
                card[a as usize & 63] += reach[h];
                card[b as usize & 63] += reach[h];
                total += reach[h];
            }
            i = j;
        }
    }

    #[cfg(test)]
    fn showdown_values_naive(&self, view: usize, reach: &[f64]) -> Vec<f64> {
        let v = &self.views[view];
        let live = |h: usize| !v.dead.contains(&h);
        (0..self.len())
            .map(|h| {
                if !live(h) {
                    return 0.0;
                }
                let [a, b] = self.combos[h];
                (0..self.len())
                    .filter(|&j| {
                        let [c, d] = self.combos[j];
                        live(j) && a != c && a != d && b != c && b != d
                    })
                    .map(|j| reach[j] * (v.strength[h].cmp(&v.strength[j]) as i32) as f64)
                    .sum()
            })
            .collect()
    }
}

/// One betting action in the subgame tree, as chips ADDED by the actor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Act {
    Fold,
    CheckCall,
    Bet(i64),
}

impl Act {
    /// Tree order: fold, check/call, bets ascending.
    fn order(&self) -> i64 {
        match *self {
            Act::Fold => -2,
            Act::CheckCall => -1,
            Act::Bet(d) => d,
        }
    }
}

/// An action already taken this round, and how the subgame treats it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Forced {
    /// The only branch at its node.
    Only(Act),
    /// The menu's actions at its node plus this one (added when off the menu);
    /// the real line continues through it, the others are solved subtrees.
    Branch(Act),
    /// Exactly these actions at its node (the real one among them), played
    /// with a strategy frozen in the solver (`Solver::freeze`).
    Frozen(Act, Vec<Act>),
}

impl Forced {
    pub(crate) fn act(&self) -> Act {
        match self {
            Forced::Only(a) | Forced::Branch(a) | Forced::Frozen(a, _) => *a,
        }
    }
}

/// One river card's branch below a chance node.
struct Kid {
    view: usize,
    node: usize,
    /// The regret-storage slots of this branch's subtree (contiguous).
    slots: Range<usize>,
    /// The cards this branch also stands for (`Tree::perms` indices; empty
    /// without isomorphism): each deals the same subtree with its holdings
    /// permuted, so its values are this branch's read through the permutation.
    images: Vec<u8>,
}

enum Node {
    Terminal {
        /// `Some(p)`: player p folded.
        folded: Option<usize>,
        /// Each player's total stake in the pot at this terminal.
        stake: [i64; 2],
        /// The board a showdown is scored on (unused for folds).
        view: usize,
    },
    Decision {
        player: usize,
        acts: Vec<Act>,
        children: Vec<usize>,
        /// Index into the regret / strategy-sum storage.
        slot: usize,
        /// The storage slots of each child's subtree (contiguous, ascending,
        /// right after `slot`), so the children can be solved in parallel.
        kid_slots: Vec<Range<usize>>,
    },
    Chance {
        kids: Vec<Kid>,
        /// River cards each pair of disjoint holdings can see (44).
        norm: f64,
    },
    /// The end of a flop round with chips behind (see the module doc).
    Leaf { stake: [i64; 2] },
    /// The end of a flop round with a player all-in.
    Runout { stake: [i64; 2] },
}

/// Turn and river runouts each pair of disjoint flop holdings can see.
pub(crate) const RUNOUTS_PER_PAIR: f64 = 990.0;

/// Bet sizes of one betting round.
#[derive(Clone)]
pub(crate) struct Menu {
    /// Pot fractions for a bet facing no bet.
    pub bet_fracs: Vec<f64>,
    /// Pot fractions (of the pot after calling) for a raise.
    pub raise_fracs: Vec<f64>,
    /// Bets/raises allowed in the round before only fold/call/all-in remain.
    pub max_raises: u8,
}

/// Betting configuration of a subgame.
#[derive(Clone)]
pub(crate) struct RoundSpec {
    /// Each player's stake in the pot when the round starts (equal heads-up
    /// after a completed street, but not required).
    pub stake: [i64; 2],
    /// Chips behind at round start.
    pub stacks: [i64; 2],
    /// First to act this round (and on the river, heads-up postflop).
    pub first: usize,
    /// Minimum bet / raise increment (the big blind).
    pub min_bet: i64,
    /// Sizes for the current round.
    pub menu: Menu,
    /// Sizes for the river rounds of a turn solve (required iff it is one).
    pub river: Option<Menu>,
    /// Actions already taken this round, in order (see `Forced`).
    pub forced: Vec<Forced>,
}

#[derive(Clone, Copy)]
struct Bet {
    stake: [i64; 2],
    stacks: [i64; 2],
    to_act: usize,
    last_raise: i64,
    raises: u8,
    acted: [bool; 2],
}

impl Bet {
    fn round_start(stake: [i64; 2], stacks: [i64; 2], first: usize) -> Bet {
        Bet { stake, stacks, to_act: first, last_raise: 0, raises: 0, acted: [false, false] }
    }
}

pub(crate) struct Tree {
    nodes: Vec<Node>,
    slots: usize,
    /// Action count per slot.
    slot_acts: Vec<usize>,
    /// Node index of the first decision after the forced prefix (where the
    /// searching player acts now).
    pub decision_root: usize,
    /// The real line: for each forced action, its node and the index of the
    /// real action among that node's actions.
    pub forced_path: Vec<(usize, usize)>,
    /// Each player's chips at the round's start, in the pot and behind.
    total: [i64; 2],
    /// `Some` for the full flop game (`build_full`).
    full: Option<Full>,
    /// Holding permutations of the suit swaps a chance node's `Kid::images`
    /// apply (`build_full` with isomorphism; empty otherwise).
    perms: Vec<Vec<u32>>,
}

/// The full flop game's cards (`Tree::build_full`).
struct Full {
    /// Turn round sizes.
    turn: Menu,
    /// Each unseen card, ascending, with its masking view.
    cards: Vec<(u8, usize)>,
    /// The runout view of each unordered turn and river pair, at `52 * t + r`.
    pair: Vec<usize>,
    /// While building: the turn card of the round being grown (`None` in the
    /// flop round).
    turn_card: Option<u8>,
    /// Each suit's class: suits with the same ranks on the flop share one
    /// (the smallest such suit) and may be swapped. Without isomorphism every
    /// suit is its own class.
    class: [u8; 4],
    /// `Tree::perms` index of the swap of suits `s` and `t` (same class only).
    swap: [[u8; 4]; 4],
}

impl Full {
    /// A dealt card's `Kid::images`, or `None` when another card of its orbit
    /// stands for it. The orbit is the card's rank in each suit of its class,
    /// less `fixed` (the turn card's suit, which the river deal keeps): the
    /// swaps that keep the flop, and the turn card, where they were.
    fn images(&self, card: u8, fixed: Option<u8>) -> Option<Vec<u8>> {
        let s = card & 3;
        if fixed == Some(s) {
            return Some(Vec::new());
        }
        let mates = (0..4u8).filter(|&t| t != s && Some(t) != fixed && self.class[t as usize] == self.class[s as usize]);
        let mut images = Vec::new();
        for t in mates {
            if t < s {
                return None;
            }
            images.push(self.swap[s as usize][t as usize]);
        }
        Some(images)
    }
}

impl Tree {
    pub(crate) fn build(spec: &RoundSpec, hands: &Hands) -> Result<Tree, String> {
        let turn = hands.board.len() == 4;
        if turn != spec.river.is_some() {
            return Err("a river menu is required for a turn solve and only for one".into());
        }
        Self::build_with(spec, hands, None)
    }

    /// The full flop game (FLOPSEARCH.md D4, the oracle): the flop round (sizes
    /// `spec.menu`), then each turn card, the turn round (`turn`), each river
    /// card and the river round (`spec.river`). A flop round ending all-in is a
    /// `Runout`, as in the flop search. `hands`: `Hands::flop_full`. Every pair
    /// of disjoint holdings sees 45 turn cards, then 44 river cards.
    ///
    /// With `iso`, a deal that only swaps suits the flop cannot tell apart
    /// (the same ranks in each) is not grown again: its canonical card's
    /// branch stands for it (`Kid::images`). The solve is the same game only
    /// when both ranges, and every frozen strategy, are unchanged by those
    /// swaps (`Oracle` checks); then its iterates are the full tree's.
    pub(crate) fn build_full(spec: &RoundSpec, turn: &Menu, hands: &Hands, iso: bool) -> Result<Tree, String> {
        if hands.board.len() != 3 || spec.river.is_none() {
            return Err("a full flop tree needs a flop board and a river menu".into());
        }
        let used = hands.board.iter().fold(0u64, |m, &c| m | 1u64 << c);
        let pairs = holdings_off(used);
        let unseen: Vec<u8> = (0..52u8).filter(|&c| used & (1u64 << c) == 0).collect();
        if hands.runouts != pairs.len() || hands.views.len() != pairs.len() + unseen.len() {
            return Err("a full flop tree needs Hands::flop_full".into());
        }
        let mut pair = vec![usize::MAX; 52 * 52];
        for (v, &[t, r]) in pairs.iter().enumerate() {
            pair[52 * t as usize + r as usize] = v;
            pair[52 * r as usize + t as usize] = v;
        }
        let cards = unseen.iter().enumerate().map(|(i, &c)| (c, pairs.len() + i)).collect();
        let mut class = [0u8, 1, 2, 3];
        let mut swap = [[u8::MAX; 4]; 4];
        let mut perms = Vec::new();
        if iso {
            let mut ranks = [0u16; 4];
            for &c in &hands.board {
                ranks[(c & 3) as usize] |= 1 << (c >> 2);
            }
            for s in 0..4 {
                class[s] = (0..4).find(|&t| ranks[t] == ranks[s]).expect("s itself") as u8;
            }
            let mut index = vec![u32::MAX; 52 * 52];
            for (h, &[a, b]) in hands.combos.iter().enumerate() {
                index[52 * a as usize + b as usize] = h as u32;
            }
            for s in 0..4u8 {
                for t in s + 1..4 {
                    if class[s as usize] != class[t as usize] {
                        continue;
                    }
                    let flip = |c: u8| match c & 3 {
                        x if x == s => c - s + t,
                        x if x == t => c - t + s,
                        _ => c,
                    };
                    let perm: Vec<u32> = hands
                        .combos
                        .iter()
                        .map(|&[a, b]| {
                            let (a, b) = (flip(a).min(flip(b)), flip(a).max(flip(b)));
                            index[52 * a as usize + b as usize]
                        })
                        .collect();
                    if perm.contains(&u32::MAX) {
                        return Err("a suit swap moved a flop card".into());
                    }
                    swap[s as usize][t as usize] = perms.len() as u8;
                    swap[t as usize][s as usize] = perms.len() as u8;
                    perms.push(perm);
                }
            }
        }
        let mut tree = Self::build_with(spec, hands, Some(Full { turn: turn.clone(), cards, pair, turn_card: None, class, swap }))?;
        tree.perms = perms;
        Ok(tree)
    }

    fn build_with(spec: &RoundSpec, hands: &Hands, full: Option<Full>) -> Result<Tree, String> {
        let total = [spec.stake[0] + spec.stacks[0], spec.stake[1] + spec.stacks[1]];
        let mut t = Tree {
            nodes: Vec::new(),
            slots: 0,
            slot_acts: Vec::new(),
            decision_root: 0,
            forced_path: Vec::new(),
            total,
            full,
            perms: Vec::new(),
        };
        let start = Bet::round_start(spec.stake, spec.stacks, spec.first);
        let mut root_marker = None;
        let view = if hands.board.len() == 5 { Some(0) } else { None };
        t.grow(spec, &spec.menu, hands, start, Some(0), view, &mut root_marker)?;
        t.decision_root = root_marker.ok_or("forced actions end the round")?;
        debug_assert_eq!(t.forced_path.len(), spec.forced.len());
        Ok(t)
    }

    /// Actions at a decision node.
    pub(crate) fn acts_at(&self, node: usize) -> Option<&[Act]> {
        self.decision(node).map(|(_, acts, _)| acts)
    }

    fn legal(spec: &RoundSpec, menu: &Menu, b: &Bet) -> Vec<Act> {
        let p = b.to_act;
        let q = 1 - p;
        let c = b.stake[q] - b.stake[p];
        let stack = b.stacks[p];
        let mut acts = Vec::new();
        if c > 0 {
            acts.push(Act::Fold);
        }
        acts.push(Act::CheckCall);
        if stack > c && b.stacks[q] > 0 {
            let pot = (b.stake[0] + b.stake[1]) as f64;
            let floor = b.last_raise.max(spec.min_bet);
            let mut adds: Vec<i64> = Vec::new();
            if b.raises < menu.max_raises {
                let fracs = if c == 0 { &menu.bet_fracs } else { &menu.raise_fracs };
                for &f in fracs {
                    let add = if c == 0 {
                        ((f * pot).round() as i64).max(floor)
                    } else {
                        c + ((f * (pot + c as f64)).round() as i64).max(floor)
                    };
                    if add < stack {
                        adds.push(add);
                    }
                }
            }
            adds.push(stack); // all-in always available
            adds.sort_unstable();
            adds.dedup();
            acts.extend(adds.into_iter().map(Act::Bet));
        }
        acts
    }

    fn apply(b: &Bet, act: Act) -> (Bet, Option<Option<usize>>) {
        let p = b.to_act;
        let q = 1 - p;
        let c = b.stake[q] - b.stake[p];
        let mut n = *b;
        match act {
            Act::Fold => return (n, Some(Some(p))),
            Act::CheckCall => {
                let pay = c.min(b.stacks[p]);
                n.stake[p] += pay;
                n.stacks[p] -= pay;
                n.acted[p] = true;
            }
            Act::Bet(add) => {
                n.stake[p] += add;
                n.stacks[p] -= add;
                n.acted[p] = true;
                let by = n.stake[p] - b.stake[q];
                if by > 0 {
                    n.last_raise = by.max(b.last_raise);
                    n.raises += 1;
                    n.acted[q] = false;
                }
            }
        }
        n.to_act = q;
        let owes = |x: usize| n.stacks[x] > 0 && (n.stake[x] < n.stake[1 - x] || !n.acted[x]);
        if !owes(0) && !owes(1) {
            return (n, Some(None)); // round over -> showdown or next card
        }
        if n.stacks[q] == 0 && n.stake[p] >= n.stake[q] {
            return (n, Some(None));
        }
        (n, None)
    }

    /// Grow one betting round from `b`. `forced`: `Some(depth)` while on the
    /// real line of the round whose actions `spec.forced` prescribes (`None`
    /// off it). `view`: the board a showdown at the end of this round is
    /// scored on (`None` = a turn round, which ends in a chance node instead,
    /// or a flop round, which ends in a leaf or a runout).
    #[allow(clippy::too_many_arguments)]
    fn grow(
        &mut self,
        spec: &RoundSpec,
        menu: &Menu,
        hands: &Hands,
        b: Bet,
        forced: Option<usize>,
        view: Option<usize>,
        root: &mut Option<usize>,
    ) -> Result<usize, String> {
        let idx = self.nodes.len();
        let mut acts = Self::legal(spec, menu, &b);
        // The real action at this node, when it is on the real line.
        let mut real = None;
        match forced.and_then(|d| spec.forced.get(d).map(|f| (d, f))) {
            Some((depth, f)) => {
                let p = b.to_act;
                let c = b.stake[1 - p] - b.stake[p];
                let norm = |a: Act| match a {
                    Act::Bet(add) if add >= b.stacks[p] => Act::Bet(b.stacks[p]),
                    Act::Bet(add) if add <= c => Act::CheckCall,
                    other => other,
                };
                let legal = |a: Act| match a {
                    Act::Fold => c > 0,
                    Act::CheckCall => true,
                    Act::Bet(add) => add > c && add <= b.stacks[p] && b.stacks[1 - p] > 0,
                };
                let forced_act = match f.act() {
                    // A real bet of no more than the price is a call.
                    Act::Bet(add) if add <= c => Act::CheckCall,
                    Act::Bet(add) if add >= b.stacks[p] => Act::Bet(b.stacks[p]),
                    other => other,
                };
                if !legal(forced_act) {
                    return Err(format!("forced action {forced_act:?} is illegal at depth {depth}"));
                }
                acts = match f {
                    Forced::Only(_) => vec![forced_act],
                    Forced::Branch(_) => {
                        acts.push(forced_act);
                        acts
                    }
                    Forced::Frozen(_, given) => {
                        let mut v: Vec<Act> = given.iter().map(|&a| norm(a)).collect();
                        if let Some(bad) = v.iter().find(|&&a| !legal(a)) {
                            return Err(format!("frozen action {bad:?} is illegal at depth {depth}"));
                        }
                        v.push(forced_act);
                        v
                    }
                };
                acts.sort_by_key(Act::order);
                acts.dedup();
                real = Some(acts.iter().position(|&a| a == forced_act).expect("the real action is present"));
            }
            None if forced == Some(spec.forced.len()) => *root = Some(idx),
            None => {}
        }
        if let Some(i) = real {
            self.forced_path.push((idx, i));
        }
        let slot = self.slots;
        self.slots += 1;
        self.slot_acts.push(acts.len());
        self.nodes.push(Node::Decision { player: b.to_act, acts: acts.clone(), children: Vec::new(), slot, kid_slots: Vec::new() });
        let mut kids = Vec::with_capacity(acts.len());
        let mut spans = Vec::with_capacity(acts.len());
        for (i, &a) in acts.iter().enumerate() {
            let lo = self.slots;
            let (nb, end) = Self::apply(&b, a);
            // Only the real action keeps the child on the real line.
            let on_line = real == Some(i);
            if on_line && end.is_some() {
                return Err("forced actions end the round".into());
            }
            let child = match end {
                Some(Some(f)) => self.terminal(Some(f), nb.stake, 0),
                Some(None) => match view {
                    Some(v) => self.terminal(None, nb.stake, v),
                    None if hands.board.len() == 3 => {
                        let all_in = nb.stacks[0] == 0 || nb.stacks[1] == 0;
                        match self.full.as_ref().map(|f| f.turn_card) {
                            Some(Some(t)) => self.river_chance(spec, hands, nb, t, root)?,
                            Some(None) if !all_in => self.turn_chance(spec, hands, nb, root)?,
                            _ => {
                                self.nodes.push(if all_in { Node::Runout { stake: nb.stake } } else { Node::Leaf { stake: nb.stake } });
                                self.nodes.len() - 1
                            }
                        }
                    }
                    None => self.chance(spec, hands, nb, root)?,
                },
                None => {
                    let next = if on_line { forced.map(|d| d + 1) } else { None };
                    self.grow(spec, menu, hands, nb, next, view, root)?
                }
            };
            kids.push(child);
            spans.push(lo..self.slots);
        }
        if let Node::Decision { children, kid_slots, .. } = &mut self.nodes[idx] {
            *children = kids;
            *kid_slots = spans;
        }
        Ok(idx)
    }

    fn terminal(&mut self, folded: Option<usize>, stake: [i64; 2], view: usize) -> usize {
        self.nodes.push(Node::Terminal { folded, stake, view });
        self.nodes.len() - 1
    }

    /// The end of a turn round without a fold: deal each river card, then play
    /// the river round (or go straight to showdown if someone is all-in).
    fn chance(&mut self, spec: &RoundSpec, hands: &Hands, b: Bet, root: &mut Option<usize>) -> Result<usize, String> {
        let idx = self.nodes.len();
        let norm = (hands.view_count() - 4) as f64;
        self.nodes.push(Node::Chance { kids: Vec::new(), norm });
        let menu = spec.river.as_ref().ok_or("turn solve without a river menu")?;
        let all_in = b.stacks[0] == 0 || b.stacks[1] == 0;
        let mut kids = Vec::with_capacity(hands.view_count());
        for view in 0..hands.view_count() {
            let lo = self.slots;
            let node = if all_in {
                self.terminal(None, b.stake, view)
            } else {
                let start = Bet::round_start(b.stake, b.stacks, spec.first);
                self.grow(spec, menu, hands, start, None, Some(view), root)?
            };
            kids.push(Kid { view, node, slots: lo..self.slots, images: Vec::new() });
        }
        if let Node::Chance { kids: k, .. } = &mut self.nodes[idx] {
            *k = kids;
        }
        Ok(idx)
    }

    /// A full flop tree's flop round ending with chips behind: deal each turn
    /// card, then play the turn round.
    fn turn_chance(&mut self, spec: &RoundSpec, hands: &Hands, b: Bet, root: &mut Option<usize>) -> Result<usize, String> {
        let full = self.full.as_ref().ok_or("not a full flop tree")?;
        let (menu, cards) = (full.turn.clone(), full.cards.clone());
        let idx = self.nodes.len();
        self.nodes.push(Node::Chance { kids: Vec::new(), norm: (cards.len() - 4) as f64 });
        let mut kids = Vec::with_capacity(cards.len());
        for (t, view) in cards {
            let Some(images) = self.full.as_ref().ok_or("not a full flop tree")?.images(t, None) else { continue };
            let lo = self.slots;
            self.full.as_mut().ok_or("not a full flop tree")?.turn_card = Some(t);
            let start = Bet::round_start(b.stake, b.stacks, spec.first);
            let node = self.grow(spec, &menu, hands, start, None, None, root);
            self.full.as_mut().ok_or("not a full flop tree")?.turn_card = None;
            kids.push(Kid { view, node: node?, slots: lo..self.slots, images });
        }
        if let Node::Chance { kids: k, .. } = &mut self.nodes[idx] {
            *k = kids;
        }
        Ok(idx)
    }

    /// A full flop tree's turn round (turn card `t`) ending without a fold:
    /// deal each river card, then play the river round (or go straight to
    /// showdown if someone is all-in).
    fn river_chance(&mut self, spec: &RoundSpec, hands: &Hands, b: Bet, t: u8, root: &mut Option<usize>) -> Result<usize, String> {
        let full = self.full.as_ref().ok_or("not a full flop tree")?;
        let rivers = full.cards.iter().filter(|&&(r, _)| r != t);
        let norm = (rivers.clone().count() - 4) as f64;
        // Each canonical river card's runout view and images (the swaps that
        // keep the turn card's suit).
        let deals: Vec<(usize, Vec<u8>)> = rivers
            .filter_map(|&(r, _)| full.images(r, Some(t & 3)).map(|im| (full.pair[52 * t as usize + r as usize], im)))
            .collect();
        let menu = spec.river.as_ref().ok_or("a full flop tree without a river menu")?;
        let idx = self.nodes.len();
        self.nodes.push(Node::Chance { kids: Vec::new(), norm });
        let all_in = b.stacks[0] == 0 || b.stacks[1] == 0;
        let mut kids = Vec::with_capacity(deals.len());
        for (view, images) in deals {
            let lo = self.slots;
            let node = if all_in {
                self.terminal(None, b.stake, view)
            } else {
                let start = Bet::round_start(b.stake, b.stacks, spec.first);
                self.grow(spec, menu, hands, start, None, Some(view), root)?
            };
            kids.push(Kid { view, node, slots: lo..self.slots, images });
        }
        if let Node::Chance { kids: k, .. } = &mut self.nodes[idx] {
            *k = kids;
        }
        Ok(idx)
    }

    /// The first betting round's decision nodes in pre-order, each with the
    /// action indices that lead to it from the root (empty for the root).
    pub(crate) fn round_decisions(&self) -> Vec<(Vec<usize>, usize)> {
        let mut out = Vec::new();
        let mut stack = vec![(Vec::new(), 0usize)];
        while let Some((path, node)) = stack.pop() {
            if let Node::Decision { children, .. } = &self.nodes[node] {
                for (a, &c) in children.iter().enumerate().rev() {
                    let mut p = path.clone();
                    p.push(a);
                    stack.push((p, c));
                }
                out.push((path, node));
            }
        }
        out
    }

    /// Total actions over all decision slots.
    pub(crate) fn action_count(&self) -> usize {
        self.slot_acts.iter().sum()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn decision_count(&self) -> usize {
        self.slots
    }

    /// The subgame in flat form for a solver outside this crate. `frozen`:
    /// (decision node, strategy) as `Solver::freeze` takes them; `want`: the
    /// slots whose average strategies the solver returns.
    pub(crate) fn flatten(
        &self,
        hands: &Hands,
        ranges: &[Vec<f64>; 2],
        frozen: &[(usize, Vec<f64>)],
        iters: u32,
        want: Vec<usize>,
    ) -> Result<FlatSubgame, String> {
        if self.full.is_some() {
            return Err("the full flop game has no flat form".into());
        }
        let n = hands.len();
        let mut f = FlatSubgame {
            board: hands.board.clone(),
            kind: Vec::with_capacity(self.nodes.len()),
            player: Vec::with_capacity(self.nodes.len()),
            slot: Vec::with_capacity(self.nodes.len()),
            stake: Vec::with_capacity(self.nodes.len()),
            view: Vec::with_capacity(self.nodes.len()),
            child_start: vec![0],
            children: Vec::new(),
            norm: 1.0,
            strength: Vec::with_capacity(hands.views.len() * n),
            alive: Vec::with_capacity(hands.views.len() * n),
            ranges: ranges.clone(),
            frozen: Vec::with_capacity(frozen.len()),
            iters,
            want,
            total: self.total,
        };
        for node in &self.nodes {
            match node {
                Node::Terminal { folded, stake, view } => {
                    f.kind.push(0);
                    f.player.push(folded.map_or(-1, |p| p as i64));
                    f.slot.push(-1);
                    f.stake.push(*stake);
                    f.view.push(*view as i64);
                }
                Node::Decision { player, children, slot, .. } => {
                    f.kind.push(1);
                    f.player.push(*player as i64);
                    f.slot.push(*slot as i64);
                    f.stake.push([0, 0]);
                    f.view.push(0);
                    f.children.extend(children.iter().map(|&c| c as i64));
                }
                Node::Chance { kids, norm } => {
                    f.kind.push(2);
                    f.player.push(-1);
                    f.slot.push(-1);
                    f.stake.push([0, 0]);
                    f.view.push(0);
                    f.norm = *norm;
                    debug_assert!(kids.iter().enumerate().all(|(v, k)| k.view == v));
                    f.children.extend(kids.iter().map(|k| k.node as i64));
                }
                Node::Leaf { stake } | Node::Runout { stake } => {
                    f.kind.push(if matches!(node, Node::Leaf { .. }) { 3 } else { 4 });
                    f.player.push(-1);
                    f.slot.push(-1);
                    f.stake.push(*stake);
                    f.view.push(0);
                }
            }
            f.child_start.push(f.children.len() as i64);
        }
        for v in &hands.views {
            f.strength.extend(&v.strength);
            let mut alive = vec![1u32; n];
            for &h in &v.dead {
                alive[h] = 0;
            }
            f.alive.extend(alive);
        }
        for (node, strategy) in frozen {
            let (_, acts, slot) = self.decision(*node).ok_or("frozen node is not a decision")?;
            if strategy.len() != acts.len() * n {
                return Err(format!("frozen strategy has {} entries, the node needs {}", strategy.len(), acts.len() * n));
            }
            f.frozen.push((slot, strategy.clone()));
        }
        Ok(f)
    }

    /// Action count of a decision slot.
    pub(crate) fn slot_len(&self, slot: usize) -> usize {
        self.slot_acts[slot]
    }
}

/// A subgame for an external solver (`Tree::flatten`): the tree in node order
/// (a parent before its children; node 0 is the root), the card views and
/// both ranges over `holdings_off(board)` in its order, the frozen slots, the
/// iteration count (DCFR as `Solver::step`) and the slots to report.
pub(crate) struct FlatSubgame {
    pub board: Vec<u8>,
    /// 0 terminal, 1 decision, 2 chance, 3 flop leaf, 4 flop runout.
    pub kind: Vec<u8>,
    /// Decision: the actor; terminal: the folder, -1 for a showdown; chance: -1.
    pub player: Vec<i64>,
    /// Decision slot, -1 for other nodes.
    pub slot: Vec<i64>,
    /// Terminal, leaf and runout stakes.
    pub stake: Vec<[i64; 2]>,
    /// Showdown board (view index).
    pub view: Vec<i64>,
    /// Children of node i: `children[child_start[i]..child_start[i + 1]]`
    /// (a decision's in action order, a chance node's in view order).
    pub child_start: Vec<i64>,
    pub children: Vec<i64>,
    /// Every chance node's normalizer (river cards a disjoint pair can see).
    pub norm: f64,
    /// View-major showdown strength and liveness per holding.
    pub strength: Vec<u32>,
    pub alive: Vec<u32>,
    pub ranges: [Vec<f64>; 2],
    /// (slot, action-major strategy) played fixed.
    pub frozen: Vec<(usize, Vec<f64>)>,
    pub iters: u32,
    pub want: Vec<usize>,
    /// Each player's chips at the round's start, in the pot and behind (a
    /// leaf's chips behind are `total - stake`).
    pub total: [i64; 2],
}

impl Tree {

    /// `(player, actions, slot)` of a decision node.
    pub(crate) fn decision(&self, node: usize) -> Option<(usize, &[Act], usize)> {
        match &self.nodes[node] {
            Node::Decision { player, acts, slot, .. } => Some((*player, acts, *slot)),
            _ => None,
        }
    }
}

/// A stored regret or strategy sum: `f64` (the search) or `f32` (the
/// oracle's full flop game, half the memory). Arithmetic is in `f64` either
/// way; an `f32` store rounds each result once, as it is written back, and an
/// `f64` store is the plain `f64` code.
pub(crate) trait Real: Copy + Send + Sync + 'static {
    fn get(self) -> f64;
    fn put(x: f64) -> Self;
}

impl Real for f64 {
    #[inline(always)]
    fn get(self) -> f64 {
        self
    }
    #[inline(always)]
    fn put(x: f64) -> f64 {
        x
    }
}

impl Real for f32 {
    #[inline(always)]
    fn get(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn put(x: f64) -> f32 {
        x as f32
    }
}

/// Regrets and strategy sums of one decision slot (action-major, `k * n`).
struct Slot<T = f64> {
    regret: Vec<T>,
    sum: Vec<T>,
    /// A fixed strategy played instead of regret matching (never updated).
    frozen: Option<Vec<f64>>,
}

impl<T: Real> Slot<T> {
    fn zeroed(k: usize, n: usize) -> Self {
        Slot { regret: vec![T::put(0.0); k * n], sum: vec![T::put(0.0); k * n], frozen: None }
    }
}

/// DCFR state over a tree and two ranges; `T` the store's precision.
pub(crate) struct Solver<'a, T = f64> {
    hands: &'a Hands,
    tree: &'a Tree,
    ranges: [Vec<f64>; 2],
    store: Vec<Slot<T>>,
    iter: u64,
}

const ALPHA: f64 = 1.5;
const GAMMA: f64 = 2.0;

/// Read-only context of one iteration, shared by the parallel river branches.
struct Ctx<'a> {
    hands: &'a Hands,
    tree: &'a Tree,
    /// Strategy-sum discount of this iteration.
    sum_w: f64,
    /// Positive-regret discount of this iteration (negative regrets are
    /// halved), applied as the iteration's first traversal reaches each slot.
    regret_w: Option<f64>,
}

/// Subtrees holding at least this many decision slots are solved in parallel
/// with their siblings; below it, forking costs more than it saves. Results do
/// not depend on it (every child writes only its own values and slots, and the
/// parent combines them in action order).
const PAR_MIN_SLOTS: usize = 8;

#[cfg_attr(not(test), allow(dead_code))]
fn terminal_values(hands: &Hands, folded: Option<usize>, stake: [i64; 2], view: usize, p: usize, opp: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; hands.len()];
    terminal_values_into(hands, folded, stake, view, p, opp, &mut out);
    out
}

/// `terminal_values` written into `out`.
fn terminal_values_into(hands: &Hands, folded: Option<usize>, stake: [i64; 2], view: usize, p: usize, opp: &[f64], out: &mut [f64]) {
    let q = 1 - p;
    match folded {
        Some(f) => {
            let amount = if f == p { -(stake[p] as f64) } else { stake[q] as f64 };
            hands.compatible_mass_into(opp, out);
            out.iter_mut().for_each(|m| *m *= amount);
        }
        None => {
            let m = stake[0].min(stake[1]) as f64;
            hands.showdown_values_into(view, opp, out);
            out.iter_mut().for_each(|v| *v *= m);
        }
    }
}

/// A flop leaf or runout's values for the player facing `reach_q`: the
/// showdown over every runout view, averaged over the 990 each disjoint pair
/// sees. Views are summed in fixed chunks, then the chunks in order, so the
/// result does not depend on the thread count.
fn runout_values_into(hands: &Hands, stake: [i64; 2], reach_q: &[f64], out: &mut [f64]) {
    const CHUNK: usize = 64;
    let n = hands.len();
    let views: Vec<usize> = (0..hands.runouts).collect();
    let parts: Vec<Vec<f64>> = views
        .par_chunks(CHUNK)
        .map(|vs| {
            let mut acc = vec![0.0; n];
            let mut v = vec![0.0; n];
            for &view in vs {
                hands.showdown_values_into(view, reach_q, &mut v);
                for (a, x) in acc.iter_mut().zip(&v) {
                    *a += x;
                }
            }
            acc
        })
        .collect();
    out.fill(0.0);
    for part in &parts {
        for (o, x) in out.iter_mut().zip(part) {
            *o += x;
        }
    }
    let m = stake[0].min(stake[1]) as f64 / RUNOUTS_PER_PAIR;
    out.iter_mut().for_each(|o| *o *= m);
}

#[cfg_attr(not(test), allow(dead_code))]
fn masked(reach: &[f64], dead: &[usize]) -> Vec<f64> {
    let mut r = reach.to_vec();
    for &h in dead {
        r[h] = 0.0;
    }
    r
}

/// Regret-matching strategy of a slot, written into `out` (`k * n`); `tot`
/// is scratch of length `n`. Each holding's total sums its actions in order.
///
/// The per-holding loops here and in `cfr` walk rows of exactly `n` with
/// `chunks_exact`/`zip` rather than indexing `a * n + h`: same arithmetic in
/// the same order, but no bounds check per element, so they vectorize.
fn current_into<T: Real>(regret: &[T], k: usize, n: usize, out: &mut [f64], tot: &mut [f64]) {
    let tot = &mut tot[..n];
    tot.fill(0.0);
    for row in regret[..k * n].chunks_exact(n) {
        for (t, r) in tot.iter_mut().zip(row) {
            *t += r.get().max(0.0);
        }
    }
    let uniform = 1.0 / k as f64;
    for (o, r) in out[..k * n].chunks_exact_mut(n).zip(regret.chunks_exact(n)) {
        for ((o, r), &t) in o.iter_mut().zip(r).zip(tot.iter()) {
            *o = if t > 0.0 { r.get().max(0.0) / t } else { uniform };
        }
    }
}

thread_local! {
    /// `cfr`'s scratch, reused across visits instead of allocated (and zeroed)
    /// per visit: one pool for decision nodes (~(3k+1)n values) and one for
    /// chance nodes (3n per card), so a small buffer never grows to the other
    /// kind's size. Every visit writes its buffer in full before reading it.
    static SCRATCH: [RefCell<Vec<Vec<f64>>>; 2] = const { [RefCell::new(Vec::new()), RefCell::new(Vec::new())] };
}

const DECISION_POOL: usize = 0;
const CHANCE_POOL: usize = 1;

/// A scratch buffer of at least `len` from `pool` (contents unspecified);
/// buffers only grow, so reuse never rewrites a tail. Use `[..len]` of it.
fn take_scratch(pool: usize, len: usize) -> Vec<f64> {
    let mut v = SCRATCH.with(|s| s[pool].borrow_mut().pop()).unwrap_or_default();
    if v.len() < len {
        v.resize(len, 0.0);
    }
    v
}

fn give_scratch(pool: usize, v: Vec<f64>) {
    SCRATCH.with(|s| s[pool].borrow_mut().push(v));
}

/// Split `store` (holding slots `base..`) into disjoint ascending slot ranges.
fn split_ranges<T>(mut store: &mut [Slot<T>], mut base: usize, ranges: impl Iterator<Item = Range<usize>>) -> Vec<&mut [Slot<T>]> {
    let mut parts = Vec::new();
    for r in ranges {
        let (_, tail) = std::mem::take(&mut store).split_at_mut(r.start - base);
        let (mine, tail) = tail.split_at_mut(r.len());
        parts.push(mine);
        store = tail;
        base = r.end;
    }
    parts
}

/// Split `store` (holding slots `base..`) into the kids' disjoint slot ranges.
fn split_for<'s, T>(store: &'s mut [Slot<T>], base: usize, kids: &[Kid]) -> Vec<&'s mut [Slot<T>]> {
    split_ranges(store, base, kids.iter().map(|k| k.slots.clone()))
}

/// Counterfactual values for `p` at `node`, written into `out` (length n);
/// updates regrets (for p) and strategy sums (for the opponent). `store`
/// holds the slots from `base` on: exactly the subtree's own slots.
///
/// Each decision node splits `store` into its own slot and one block per
/// child subtree (`kid_slots`), so large children run in parallel; each
/// writes only its own values and slots, and the parent combines them in
/// action order, so the numbers do not depend on the thread count. A node
/// takes one scratch buffer per visit from its thread's pool, nothing per
/// action.
#[allow(clippy::too_many_arguments)]
fn cfr<T: Real>(cx: &Ctx, store: &mut [Slot<T>], base: usize, node: usize, p: usize, reach_q: &[f64], reach_p: &[f64], out: &mut [f64]) {
    let n = cx.hands.len();
    match &cx.tree.nodes[node] {
        Node::Terminal { folded, stake, view } => terminal_values_into(cx.hands, *folded, *stake, *view, p, reach_q, out),
        Node::Leaf { stake } | Node::Runout { stake } => runout_values_into(cx.hands, *stake, reach_q, out),
        Node::Chance { kids, norm } => {
            let parts = split_for(store, base, kids);
            // Per river card: both reaches with its holdings masked, then its values.
            let mut scratch = take_scratch(CHANCE_POOL, kids.len() * 3 * n);
            let buf = &mut scratch[..kids.len() * 3 * n];
            parts.into_par_iter().zip(kids.par_iter()).zip(buf.par_chunks_mut(3 * n)).for_each(|((part, kid), b)| {
                let dead = &cx.hands.views[kid.view].dead;
                let (rq, rest) = b.split_at_mut(n);
                let (rp, v) = rest.split_at_mut(n);
                rq.copy_from_slice(reach_q);
                rp.copy_from_slice(reach_p);
                for &h in dead {
                    rq[h] = 0.0;
                    rp[h] = 0.0;
                }
                cfr(cx, part, kid.slots.start, kid.node, p, rq, rp, v);
                for &h in dead {
                    v[h] = 0.0;
                }
            });
            out.fill(0.0);
            for (b, kid) in buf.chunks(3 * n).zip(kids) {
                let v = &b[2 * n..];
                for (o, v) in out.iter_mut().zip(v) {
                    *o += v;
                }
                // An image card's value for holding h is this card's for the
                // holding the swap maps h to.
                for &g in &kid.images {
                    for (o, &j) in out.iter_mut().zip(&cx.tree.perms[g as usize]) {
                        *o += v[j as usize];
                    }
                }
            }
            out.iter_mut().for_each(|x| *x /= norm);
            give_scratch(CHANCE_POOL, scratch);
        }
        Node::Decision { player, acts, children, slot, kid_slots } => {
            let k = acts.len();
            let (head, tail) = store.split_at_mut(*slot - base + 1);
            let me = &mut head[*slot - base];
            let parts = split_ranges(tail, *slot + 1, kid_slots.iter().cloned());
            let frozen = me.frozen.is_some();
            // Scratch: each child's reach, each child's values, and (unless the
            // node is frozen) the current strategy plus its per-holding totals.
            let len = if frozen { 2 * k * n } else { (3 * k + 1) * n };
            let mut scratch = take_scratch(DECISION_POOL, len);
            let buf = &mut scratch[..len];
            let (reach, rest) = buf.split_at_mut(k * n);
            let (vals, rest) = rest.split_at_mut(k * n);
            // A frozen node is played as given: no regrets, no strategy sum.
            let sigma: &[f64] = match &me.frozen {
                Some(f) => f,
                None => {
                    if let Some(w) = cx.regret_w {
                        // The discount here, while the slot is in cache, rather than
                        // as a pass over the whole store: this traversal reaches every
                        // slot once, before anything reads its regrets.
                        me.regret.iter_mut().for_each(|x| *x = T::put(x.get() * if x.get() > 0.0 { w } else { 0.5 }));
                    }
                    let (sig, tot) = rest.split_at_mut(k * n);
                    current_into(&me.regret, k, n, sig, tot);
                    sig
                }
            };
            let mine = *player == p;
            let acting = &(if mine { reach_p } else { reach_q })[..n];
            for (r, s) in reach.chunks_exact_mut(n).zip(sigma.chunks_exact(n)) {
                for ((r, &x), &s) in r.iter_mut().zip(acting).zip(s) {
                    *r = x * s;
                }
            }
            if !mine && !frozen {
                for (s, r) in me.sum.iter_mut().zip(reach.iter()) {
                    *s = T::put(s.get() * cx.sum_w + r);
                }
            }
            let reach: &[f64] = reach;
            let solve = |a: usize, part: &mut [Slot<T>], v: &mut [f64]| {
                let r = &reach[a * n..(a + 1) * n];
                let lo = kid_slots[a].start;
                if mine {
                    cfr(cx, part, lo, children[a], p, reach_q, r, v)
                } else {
                    cfr(cx, part, lo, children[a], p, r, reach_p, v)
                }
            };
            let span = kid_slots.last().map_or(0, |r| r.end) - (*slot + 1);
            if span >= PAR_MIN_SLOTS {
                parts.into_par_iter().zip(vals.par_chunks_mut(n)).enumerate().for_each(|(a, (part, v))| solve(a, part, v));
            } else {
                for (a, (part, v)) in parts.into_iter().zip(vals.chunks_mut(n)).enumerate() {
                    solve(a, part, v);
                }
            }
            out.fill(0.0);
            let out = &mut out[..n];
            if mine {
                for (s, v) in sigma.chunks_exact(n).zip(vals.chunks_exact(n)) {
                    for ((o, &s), &v) in out.iter_mut().zip(s).zip(v) {
                        *o += s * v;
                    }
                }
                if !frozen {
                    for (r, v) in me.regret.chunks_exact_mut(n).zip(vals.chunks_exact(n)) {
                        for ((r, &v), &o) in r.iter_mut().zip(v).zip(out.iter()) {
                            *r = T::put(r.get() + (v - o));
                        }
                    }
                }
            } else {
                for v in vals.chunks_exact(n) {
                    for (o, &v) in out.iter_mut().zip(v) {
                        *o += v;
                    }
                }
            }
            give_scratch(DECISION_POOL, scratch);
        }
    }
}

impl<'a> Solver<'a> {
    pub(crate) fn new(hands: &'a Hands, tree: &'a Tree, ranges: [Vec<f64>; 2]) -> Self {
        Solver::zeroed(hands, tree, ranges)
    }
}

impl<'a, T: Real> Solver<'a, T> {
    /// `new` with the store in `T`.
    fn zeroed(hands: &'a Hands, tree: &'a Tree, ranges: [Vec<f64>; 2]) -> Self {
        let n = hands.len();
        let store = tree.slot_acts.iter().map(|&k| Slot::zeroed(k, n)).collect();
        Solver { hands, tree, ranges, store, iter: 0 }
    }

    /// Fix the strategy at `node` (action-major `k * n`, each holding's column
    /// summing to 1) before solving.
    pub(crate) fn freeze(&mut self, node: usize, strategy: Vec<f64>) -> Result<(), String> {
        let n = self.hands.len();
        let (_, acts, slot) = self.tree.decision(node).ok_or("frozen node is not a decision")?;
        let k = acts.len();
        if strategy.len() != k * n {
            return Err(format!("frozen strategy has {} entries, the node needs {}", strategy.len(), k * n));
        }
        for h in 0..n {
            let tot: f64 = (0..k).map(|a| strategy[a * n + h]).sum();
            if (tot - 1.0).abs() > 1e-6 || (0..k).any(|a| strategy[a * n + h].is_nan() || strategy[a * n + h] < 0.0) {
                return Err(format!("frozen strategy of holding {h} is not a distribution (sum {tot})"));
            }
        }
        self.store[slot].frozen = Some(strategy);
        Ok(())
    }

    /// Average strategy at a slot (action-major, `k * n`).
    pub(crate) fn average(&self, slot: usize) -> Vec<f64> {
        if let Some(f) = &self.store[slot].frozen {
            return f.clone();
        }
        let n = self.hands.len();
        let k = self.tree.slot_acts[slot];
        let s = &self.store[slot].sum;
        let mut out = vec![0.0; k * n];
        for h in 0..n {
            let tot: f64 = (0..k).map(|a| s[a * n + h].get()).sum();
            for a in 0..k {
                out[a * n + h] = if tot > 0.0 { s[a * n + h].get() / tot } else { 1.0 / k as f64 };
            }
        }
        out
    }

    /// One DCFR iteration (both players, alternating).
    pub(crate) fn step(&mut self) {
        self.step_gamma(GAMMA);
    }

    /// `step` with the strategy sums weighted `(t / (t + 1)) ^ gamma`.
    pub(crate) fn step_gamma(&mut self, gamma: f64) {
        self.iter += 1;
        let t = self.iter as f64;
        let pos = t.powf(ALPHA) / (t.powf(ALPHA) + 1.0);
        let sum_w = (t / (t + 1.0)).powf(gamma);
        let mut root_v = vec![0.0; self.hands.len()];
        for p in 0..2 {
            let cx = Ctx { hands: self.hands, tree: self.tree, sum_w, regret_w: (p == 0).then_some(pos) };
            let [r0, r1] = &self.ranges;
            let (rq, rp) = if p == 0 { (r1, r0) } else { (r0, r1) };
            cfr(&cx, &mut self.store, 0, 0, p, rq, rp, &mut root_v);
        }
    }

    /// Best-response counterfactual values for `p` against the average strategy.
    #[cfg_attr(not(test), allow(dead_code))]
    fn best_response(&self, node: usize, p: usize, reach_q: &[f64]) -> Vec<f64> {
        let n = self.hands.len();
        match &self.tree.nodes[node] {
            Node::Terminal { folded, stake, view } => terminal_values(self.hands, *folded, *stake, *view, p, reach_q),
            Node::Leaf { stake } | Node::Runout { stake } => {
                let mut out = vec![0.0; n];
                runout_values_into(self.hands, *stake, reach_q, &mut out);
                out
            }
            Node::Chance { kids, norm } => {
                let vals: Vec<Vec<f64>> = kids
                    .par_iter()
                    .map(|kid| {
                        let dead = &self.hands.views[kid.view].dead;
                        let mut v = self.best_response(kid.node, p, &masked(reach_q, dead));
                        for &h in dead {
                            v[h] = 0.0;
                        }
                        v
                    })
                    .collect();
                let mut out = vec![0.0; n];
                for (v, kid) in vals.iter().zip(kids) {
                    for h in 0..n {
                        out[h] += v[h] / norm;
                    }
                    for &g in &kid.images {
                        for (o, &j) in out.iter_mut().zip(&self.tree.perms[g as usize]) {
                            *o += v[j as usize] / norm;
                        }
                    }
                }
                out
            }
            Node::Decision { player, acts, children, slot, .. } => {
                let k = acts.len();
                if *player == p && self.store[*slot].frozen.is_some() {
                    // A frozen node is not a choice for the best responder.
                    let f = self.average(*slot);
                    let mut out = vec![0.0; n];
                    for a in 0..k {
                        let v = self.best_response(children[a], p, reach_q);
                        for h in 0..n {
                            out[h] += f[a * n + h] * v[h];
                        }
                    }
                    out
                } else if *player == p {
                    let mut best = vec![f64::NEG_INFINITY; n];
                    for &c in children {
                        let v = self.best_response(c, p, reach_q);
                        for h in 0..n {
                            best[h] = best[h].max(v[h]);
                        }
                    }
                    best
                } else {
                    let avg = self.average(*slot);
                    let mut out = vec![0.0; n];
                    for a in 0..k {
                        let cr: Vec<f64> = (0..n).map(|h| reach_q[h] * avg[a * n + h]).collect();
                        let v = self.best_response(children[a], p, &cr);
                        for h in 0..n {
                            out[h] += v[h];
                        }
                    }
                    out
                }
            }
        }
    }

    /// Total compatible joint mass of the two ranges (normalizes values to
    /// "per dealt pair of holdings").
    #[cfg_attr(not(test), allow(dead_code))]
    fn joint_mass(&self) -> f64 {
        let m = self.hands.compatible_mass(&self.ranges[1]);
        self.ranges[0].iter().zip(&m).map(|(r, m)| r * m).sum()
    }

    /// Exploitability of the average strategy in chips per dealt pair:
    /// (BR value of player 0 + BR value of player 1), >= 0, 0 at equilibrium.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn exploitability(&self) -> f64 {
        let z = self.joint_mass();
        (0..2)
            .map(|p| {
                let v = self.best_response(0, p, &self.ranges[1 - p]);
                self.ranges[p].iter().zip(&v).map(|(r, v)| r * v).sum::<f64>()
            })
            .sum::<f64>()
            / z
    }

    /// Value to `p` of the average-strategy profile, chips per dealt pair.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn value(&self, p: usize) -> f64 {
        fn walk<T: Real>(s: &Solver<T>, node: usize, p: usize, reach: [Vec<f64>; 2]) -> f64 {
            let n = s.hands.len();
            match &s.tree.nodes[node] {
                Node::Terminal { folded, stake, view } => {
                    let v = terminal_values(s.hands, *folded, *stake, *view, p, &reach[1 - p]);
                    reach[p].iter().zip(&v).map(|(r, v)| r * v).sum()
                }
                Node::Leaf { stake } | Node::Runout { stake } => {
                    let mut v = vec![0.0; n];
                    runout_values_into(s.hands, *stake, &reach[1 - p], &mut v);
                    reach[p].iter().zip(&v).map(|(r, v)| r * v).sum()
                }
                Node::Chance { kids, norm } => {
                    // An image deal is worth its canonical card's (the reaches are
                    // unchanged by the swap).
                    kids.iter()
                        .map(|kid| {
                            let dead = &s.hands.views[kid.view].dead;
                            walk(s, kid.node, p, [masked(&reach[0], dead), masked(&reach[1], dead)]) * (1 + kid.images.len()) as f64
                        })
                        .sum::<f64>()
                        / norm
                }
                Node::Decision { player, acts, children, slot, .. } => {
                    let avg = s.average(*slot);
                    (0..acts.len())
                        .map(|a| {
                            let mut r = reach.clone();
                            for h in 0..n {
                                r[*player][h] *= avg[a * n + h];
                            }
                            walk(s, children[a], p, r)
                        })
                        .sum()
                }
            }
        }
        walk(self, 0, p, self.ranges.clone()) / self.joint_mass()
    }
}

/// The full flop game's solve (FLOPSEARCH.md D4's oracle), owning its cards
/// and tree (a `Solver` borrows them) so a caller can run it in parts; `T`
/// the store's precision (`f32` halves the memory).
pub(crate) struct Oracle<T = f64> {
    pub hands: Hands,
    pub tree: Tree,
    ranges: [Vec<f64>; 2],
    store: Vec<Slot<T>>,
    iter: u64,
}

#[cfg(test)]
impl Oracle {
    /// `spec`: the flop round from its start (no forced actions), with the
    /// river menu; `turn`: the turn menu; `ranges` over `Hands::flop_full`'s
    /// holdings.
    pub(crate) fn new(board: [u8; 3], spec: &RoundSpec, turn: &Menu, ranges: [Vec<f64>; 2]) -> Result<Oracle, String> {
        Oracle::zeroed(board, spec, turn, ranges, false)
    }
}

impl<T: Real> Oracle<T> {
    /// `new` with the store in `T`; `iso`: deal suit-isomorphic cards once
    /// (`Tree::build_full`), which needs both ranges, and every strategy
    /// frozen later, unchanged by the flop's suit swaps (checked exactly).
    pub(crate) fn zeroed(board: [u8; 3], spec: &RoundSpec, turn: &Menu, ranges: [Vec<f64>; 2], iso: bool) -> Result<Self, String> {
        if !spec.forced.is_empty() {
            return Err("the oracle solves the flop round from its start".into());
        }
        let hands = Hands::flop_full(&board);
        let n = hands.len();
        if ranges.iter().any(|r| r.len() != n || r.iter().any(|x| !x.is_finite() || *x < 0.0)) {
            return Err(format!("ranges must be {n} non-negative numbers each"));
        }
        let tree = Tree::build_full(spec, turn, &hands, iso)?;
        let o = Oracle { hands, tree, ranges, store: Vec::new(), iter: 0 };
        for (p, r) in o.ranges.iter().enumerate() {
            if !o.symmetric(r, 1) {
                return Err(format!("player {p}'s range changes under a suit swap of the flop; solve without isomorphism"));
            }
        }
        Ok(o)
    }

    /// Whether `x` (`k` action rows of the holdings) is unchanged by every
    /// suit swap the tree deals once (true without isomorphism).
    fn symmetric(&self, x: &[f64], k: usize) -> bool {
        let n = self.hands.len();
        self.tree.perms.iter().all(|perm| (0..k).all(|a| (0..n).all(|h| x[a * n + h] == x[a * n + perm[h] as usize])))
    }

    /// Whether the tree deals suit-isomorphic cards once.
    pub(crate) fn isomorphic(&self) -> bool {
        !self.tree.perms.is_empty()
    }

    /// Run `f` on a `Solver` over this state. The store is allocated here on
    /// first use, so reading the tree (`flop_nodes`, `size`) costs none of it.
    fn with<R>(&mut self, f: impl FnOnce(&mut Solver<T>) -> R) -> R {
        if self.store.is_empty() {
            let n = self.hands.len();
            self.store = self.tree.slot_acts.iter().map(|&k| Slot::zeroed(k, n)).collect();
        }
        let mut s = Solver {
            hands: &self.hands,
            tree: &self.tree,
            ranges: std::mem::take(&mut self.ranges),
            store: std::mem::take(&mut self.store),
            iter: self.iter,
        };
        let out = f(&mut s);
        let Solver { ranges, store, iter, .. } = s;
        (self.ranges, self.store, self.iter) = (ranges, store, iter);
        out
    }

    /// The decision node reached from the root by these action indices.
    fn node_at(&self, path: &[usize]) -> Result<usize, String> {
        let mut node = 0;
        for &a in path {
            node = match &self.tree.nodes[node] {
                Node::Decision { children, .. } if a < children.len() => children[a],
                _ => return Err(format!("no action {a} on the path {path:?}")),
            };
        }
        self.tree.decision(node).map(|_| node).ok_or_else(|| format!("the path {path:?} does not end at a decision"))
    }

    /// The flop round's decision nodes in pre-order: the path to each (see
    /// `Tree::round_decisions`), its actor and its actions.
    pub(crate) fn flop_nodes(&self) -> Vec<(Vec<usize>, usize, Vec<Act>)> {
        self.tree
            .round_decisions()
            .into_iter()
            .map(|(path, node)| {
                let (player, acts, _) = self.tree.decision(node).expect("a decision");
                (path, player, acts.to_vec())
            })
            .collect()
    }

    /// Play a fixed strategy at the flop node at `path` (`Solver::freeze`).
    pub(crate) fn freeze(&mut self, path: &[usize], strategy: Vec<f64>) -> Result<(), String> {
        let node = self.node_at(path)?;
        let k = self.tree.decision(node).expect("a decision").1.len();
        if strategy.len() == k * self.hands.len() && !self.symmetric(&strategy, k) {
            return Err(format!("the strategy at {path:?} changes under a suit swap of the flop; solve without isomorphism"));
        }
        self.with(|s| s.freeze(node, strategy))
    }

    /// The average strategy at the flop node at `path` (action-major).
    pub(crate) fn average(&mut self, path: &[usize]) -> Result<Vec<f64>, String> {
        let (_, _, slot) = self.tree.decision(self.node_at(path)?).expect("a decision");
        Ok(self.with(|s| s.average(slot)))
    }

    /// `iters` more DCFR iterations, strategy sums weighted with `gamma`
    /// (`Solver::step_gamma`; the search's is 2).
    pub(crate) fn run(&mut self, iters: u64, gamma: f64) {
        self.with(|s| (0..iters).for_each(|_| s.step_gamma(gamma)));
    }

    pub(crate) fn iterations(&self) -> u64 {
        self.iter
    }

    /// Each player's best-response value against the average strategy (its
    /// frozen nodes played as frozen), chips per dealt pair. The game value
    /// to player 0 lies in `[-br[1], br[0]]`; the sum is the exploitability.
    pub(crate) fn best_responses(&mut self) -> [f64; 2] {
        self.with(|s| {
            let z = s.joint_mass();
            [0, 1].map(|p| {
                let v = s.best_response(0, p, &s.ranges[1 - p]);
                s.ranges[p].iter().zip(&v).map(|(r, v)| r * v).sum::<f64>() / z
            })
        })
    }

    /// Value to `p` of the average-strategy profile, chips per dealt pair.
    pub(crate) fn value(&mut self, p: usize) -> f64 {
        self.with(|s| s.value(p))
    }

    /// (nodes, decision slots, actions over all slots).
    pub(crate) fn size(&self) -> (usize, usize, usize) {
        (self.tree.nodes.len(), self.tree.slots, self.tree.action_count())
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::*;
    use super::*;

    #[test]
    fn apart_runs_nothing_of_the_callers_pool() {
        // One worker, two jobs: it runs the newer first (LIFO). Waiting inside
        // `apart`, it must not run the older one; a rayon wait would (M2's
        // deadlock). And the runouts are the same strengths.
        use std::sync::atomic::{AtomicBool, Ordering};
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let (older_ran, seen) = (AtomicBool::new(false), AtomicBool::new(true));
        pool.scope(|s| {
            s.spawn(|_| older_ran.store(true, Ordering::SeqCst));
            s.spawn(|_| {
                let ran = apart(|| {
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    older_ran.load(Ordering::SeqCst)
                });
                seen.store(ran, Ordering::SeqCst);
            });
        });
        assert!(!seen.load(Ordering::SeqCst), "the waiting worker ran the caller's pool's other job");
        assert!(older_ran.load(Ordering::SeqCst));
        let b = [3u8, 22, 41];
        let s = pool.install(|| Hands::flop_runout_strengths(&b));
        assert_eq!(s, Hands::flop(&b, true).view_strengths());
        assert_eq!(s.len(), 1176 * 1176);
    }

    fn random_range(n: usize, seed: u64, zero_frac: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let u = unit(splitmix(seed ^ i as u64));
                if u < zero_frac {
                    0.0
                } else {
                    u
                }
            })
            .collect()
    }

    #[test]
    fn fast_showdown_and_blockers_match_the_naive_oracle() {
        for seed in 0..6 {
            let hands = Hands::river(&board(seed));
            assert_eq!((hands.len(), hands.view_count()), (1081, 1));
            let reach = random_range(hands.len(), seed + 100, 0.3);
            let fast = hands.showdown_values(0, &reach);
            let slow = hands.showdown_values_naive(0, &reach);
            for (a, b) in fast.iter().zip(&slow) {
                assert!((a - b).abs() < 1e-9, "{a} vs {b}");
            }
            let mass = hands.compatible_mass(&reach);
            for (h, m) in mass.iter().enumerate() {
                let [a, b] = hands.combos[h];
                let naive: f64 = hands
                    .combos
                    .iter()
                    .zip(&reach)
                    .filter(|(j, _)| !j.contains(&a) && !j.contains(&b))
                    .map(|(_, r)| r)
                    .sum();
                assert!((m - naive).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn turn_views_match_the_naive_oracle() {
        let b = board(3);
        let hands = Hands::turn(&[b[0], b[1], b[2], b[3]]);
        assert_eq!((hands.len(), hands.view_count()), (1128, 48));
        let reach = random_range(hands.len(), 9, 0.3);
        for view in [0, 17, 47] {
            let v = &hands.views[view];
            let card = v.card.unwrap();
            assert!(!b[..4].contains(&card));
            assert_eq!(v.dead.len(), 47);
            assert_eq!(v.order.len(), 1128 - 47);
            let fast = hands.showdown_values(view, &reach);
            let slow = hands.showdown_values_naive(view, &reach);
            for (a, b) in fast.iter().zip(&slow) {
                assert!((a - b).abs() < 1e-9, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn river_solve_converges_to_equilibrium() {
        for seed in 0..3 {
            let hands = Hands::river(&board(seed + 7));
            let tree = Tree::build(&spec(vec![]), &hands).unwrap();
            let ranges = [random_range(hands.len(), seed, 0.5), random_range(hands.len(), seed + 50, 0.5)];
            let mut solver = Solver::new(&hands, &tree, ranges);
            for _ in 0..300 {
                solver.step();
            }
            let pot = 2000.0;
            let e = solver.exploitability();
            assert!(e >= -1e-9, "exploitability {e} negative");
            assert!(e / pot < 0.005, "seed {seed}: exploitability {:.4}% of pot after 300 iters", 100.0 * e / pot);
            // Zero-sum: the two players' values cancel.
            assert!((solver.value(0) + solver.value(1)).abs() < 1e-6);
        }
    }

    #[test]
    fn best_response_dominates_the_profile_for_each_player() {
        let hands = Hands::river(&board(21));
        let tree = Tree::build(&spec(vec![]), &hands).unwrap();
        let ranges = [random_range(hands.len(), 1, 0.6), random_range(hands.len(), 2, 0.6)];
        let mut solver = Solver::new(&hands, &tree, ranges);
        for _ in 0..20 {
            solver.step();
        }
        let z = solver.joint_mass();
        for p in 0..2 {
            let v = solver.best_response(0, p, &solver.ranges[1 - p]);
            let br: f64 = solver.ranges[p].iter().zip(&v).map(|(r, v)| r * v).sum::<f64>() / z;
            assert!(br >= solver.value(p) - 1e-6, "player {p}: BR {br} < profile {}", solver.value(p));
        }
    }

    #[test]
    fn forced_prefix_is_respected() {
        let hands = Hands::river(&board(1));
        let tree = Tree::build(&spec(vec![Act::CheckCall, Act::Bet(700)]), &hands).unwrap();
        // Root forced check (BB), then SB forced bet of 700; the searching player
        // (BB) decides facing that bet.
        match &tree.nodes[tree.decision_root] {
            Node::Decision { player, acts, .. } => {
                assert_eq!(*player, 1);
                assert!(acts.contains(&Act::Fold) && acts.contains(&Act::CheckCall));
            }
            _ => panic!("decision root is not a decision"),
        }
        // Folding when nothing is owed is not a legal forced action.
        assert!(Tree::build(&spec(vec![Act::Fold]), &hands).is_err());
        // A forced line that closes the round leaves nothing to search.
        assert!(Tree::build(&spec(vec![Act::CheckCall, Act::CheckCall]), &hands).is_err());
    }

    #[test]
    fn branch_and_frozen_forced_actions_grow_the_whole_round() {
        let hands = Hands::river(&board(1));
        // Pot 2000: BB's menu at the root is check, 660, 1500, 3000, all-in 9000.
        let root_acts = vec![Act::CheckCall, Act::Bet(660), Act::Bet(1500), Act::Bet(3000), Act::Bet(9000)];
        let mk = |forced: Vec<Forced>| RoundSpec { forced, ..spec(vec![]) };
        // BB checked (frozen), SB bet 2000 (off the menu, solved), BB decides.
        let tree = Tree::build(&mk(vec![Forced::Frozen(Act::CheckCall, root_acts.clone()), Forced::Branch(Act::Bet(2000))]), &hands)
            .unwrap();
        assert_eq!(tree.forced_path.len(), 2);
        let (n0, i0) = tree.forced_path[0];
        assert_eq!(n0, 0);
        assert_eq!(tree.acts_at(n0).unwrap(), &root_acts[..]);
        assert_eq!(i0, 0);
        let (n1, i1) = tree.forced_path[1];
        let acts1 = tree.acts_at(n1).unwrap();
        assert_eq!(acts1, &[Act::CheckCall, Act::Bet(660), Act::Bet(1500), Act::Bet(2000), Act::Bet(3000), Act::Bet(9000)]);
        assert_eq!(acts1[i1], Act::Bet(2000));
        match tree.decision(tree.decision_root) {
            Some((player, acts, _)) => {
                assert_eq!(player, 1);
                assert!(acts.contains(&Act::Fold) && acts.contains(&Act::CheckCall));
            }
            None => panic!("decision root is not a decision"),
        }
        // Every branch is grown, so the tree holds the full round and more.
        let full = Tree::build(&spec(vec![]), &hands).unwrap();
        let only = Tree::build(&spec(vec![Act::CheckCall, Act::Bet(2000)]), &hands).unwrap();
        assert!(tree.decision_count() > full.decision_count());
        assert!(only.decision_count() < full.decision_count());
        // An on-menu real size is not duplicated; an illegal frozen action is refused.
        let on_menu = Tree::build(&mk(vec![Forced::Branch(Act::CheckCall), Forced::Branch(Act::Bet(1500))]), &hands).unwrap();
        let (n, i) = on_menu.forced_path[1];
        assert_eq!(on_menu.acts_at(n).unwrap().iter().filter(|&&a| a == Act::Bet(1500)).count(), 1);
        assert_eq!(on_menu.acts_at(n).unwrap()[i], Act::Bet(1500));
        assert!(Tree::build(&mk(vec![Forced::Frozen(Act::CheckCall, vec![Act::Fold, Act::CheckCall])]), &hands).is_err());
        // A real line that closes the round still leaves nothing to search.
        assert!(Tree::build(&mk(vec![Forced::Branch(Act::CheckCall), Forced::Branch(Act::CheckCall)]), &hands).is_err());
    }

    #[test]
    fn a_frozen_equilibrium_strategy_keeps_the_game_value() {
        // Freezing one player's root strategy at an (approximate) equilibrium
        // leaves the value of the rest unchanged: re-solving below it must
        // reproduce the full solve's value, and the frozen strategy is played
        // exactly as given.
        let hands = Hands::river(&board(3));
        let ranges = [range(hands.len(), 11), range(hands.len(), 12)];
        let tree = Tree::build(&spec(vec![]), &hands).unwrap();
        let mut full = Solver::new(&hands, &tree, ranges.clone());
        for _ in 0..400 {
            full.step();
        }
        let (_, _, slot) = tree.decision(tree.decision_root).unwrap();
        let sigma = full.average(slot);
        let mut again = Solver::new(&hands, &tree, ranges);
        assert!(again.freeze(tree.decision_root, sigma[1..].to_vec()).is_err(), "wrong length");
        let mut bad = sigma.clone();
        bad[0] += 0.5;
        assert!(again.freeze(tree.decision_root, bad).is_err(), "not a distribution");
        again.freeze(tree.decision_root, sigma.clone()).unwrap();
        for _ in 0..400 {
            again.step();
        }
        assert_eq!(again.average(slot), sigma);
        let (v_full, v_again) = (full.value(1), again.value(1));
        // Each value is within its profile's exploitability of the game's value.
        let tol = 2.0 * (full.exploitability() + again.exploitability());
        assert!((v_full - v_again).abs() < tol, "value {v_again} vs {v_full} (tol {tol})");
        assert!(again.exploitability() >= -1e-9);
    }

    // ------------------------------------------------------------- turn

    fn turn_hands(seed: u64) -> Hands {
        let b = board(seed);
        Hands::turn(&[b[0], b[1], b[2], b[3]])
    }

    #[test]
    fn turn_forced_line_closing_the_turn_is_rejected() {
        let hands = turn_hands(2);
        assert!(Tree::build(&turn_spec(vec![Act::CheckCall, Act::CheckCall]), &hands).is_err());
        assert!(Tree::build(&turn_spec(vec![]), &Hands::river(&board(2))).is_err(), "river menu on a river solve");
        let mut no_menu = turn_spec(vec![]);
        no_menu.river = None;
        assert!(Tree::build(&no_menu, &hands).is_err(), "turn solve without a river menu");
        let tree = Tree::build(&turn_spec(vec![Act::CheckCall]), &hands).unwrap();
        match &tree.nodes[tree.decision_root] {
            Node::Decision { player, .. } => assert_eq!(*player, 0),
            _ => panic!("decision root is not a decision"),
        }
    }

    /// Exact value of a hand pair under the average profile: walks every river
    /// card and scores showdowns with `eval7_fast` directly (no views, no
    /// masking), so it checks the solver's chance and card-removal bookkeeping.
    fn pair_ev(s: &Solver, node: usize, hold: [[u8; 2]; 2], idx: [usize; 2], river: Option<u8>) -> f64 {
        let n = s.hands.len();
        match &s.tree.nodes[node] {
            Node::Terminal { folded: Some(f), stake, .. } => {
                if *f == 0 {
                    -(stake[0] as f64)
                } else {
                    stake[1] as f64
                }
            }
            Node::Terminal { folded: None, stake, .. } => {
                let mut b: Vec<u8> = s.hands.board.clone();
                b.extend(river);
                let c = |i: u8| Card::from_index(i);
                let st = |h: [u8; 2]| eval7_fast(&[c(h[0]), c(h[1]), c(b[0]), c(b[1]), c(b[2]), c(b[3]), c(b[4])]);
                let sign = st(hold[0]).cmp(&st(hold[1])) as i32 as f64;
                sign * stake[0].min(stake[1]) as f64
            }
            Node::Leaf { stake } | Node::Runout { stake } => {
                // Every turn and river pair off both holdings, scored directly.
                let dead = s.hands.board.iter().chain(&hold[0]).chain(&hold[1]).fold(0u64, |m, &c| m | 1u64 << c);
                let c = |i: u8| Card::from_index(i);
                let b = &s.hands.board;
                let (mut total, mut count) = (0.0, 0);
                for [t, r] in holdings_off(dead) {
                    let st = |h: [u8; 2]| eval7_fast(&[c(h[0]), c(h[1]), c(b[0]), c(b[1]), c(b[2]), c(t), c(r)]);
                    total += st(hold[0]).cmp(&st(hold[1])) as i32 as f64;
                    count += 1;
                }
                assert_eq!(count, 990);
                total / 990.0 * stake[0].min(stake[1]) as f64
            }
            Node::Chance { kids, .. } => {
                let live: Vec<&Kid> = kids
                    .iter()
                    .filter(|k| {
                        let c = s.hands.views[k.view].card.unwrap();
                        !hold[0].contains(&c) && !hold[1].contains(&c)
                    })
                    .collect();
                assert_eq!(live.len(), 44);
                live.iter()
                    .map(|k| pair_ev(s, k.node, hold, idx, s.hands.views[k.view].card))
                    .sum::<f64>()
                    / 44.0
            }
            Node::Decision { player, acts, children, slot, .. } => {
                let avg = s.average(*slot);
                (0..acts.len())
                    .map(|a| {
                        let pr = avg[a * n + idx[*player]];
                        if pr == 0.0 {
                            0.0
                        } else {
                            pr * pair_ev(s, children[a], hold, idx, river)
                        }
                    })
                    .sum()
            }
        }
    }

    fn sparse_range(n: usize, seed: u64, keep: usize) -> Vec<f64> {
        let mut r = vec![0.0; n];
        let mut s = seed;
        for _ in 0..keep {
            s = splitmix(s);
            r[(s % n as u64) as usize] = 0.1 + unit(splitmix(s ^ 77));
        }
        r
    }

    fn brute_value(s: &Solver) -> f64 {
        let (r0, r1) = (&s.ranges[0], &s.ranges[1]);
        let (mut num, mut den) = (0.0, 0.0);
        for (i, h) in s.hands.combos.iter().enumerate() {
            if r0[i] == 0.0 {
                continue;
            }
            for (j, q) in s.hands.combos.iter().enumerate() {
                if r1[j] == 0.0 || h.contains(&q[0]) || h.contains(&q[1]) {
                    continue;
                }
                let w = r0[i] * r1[j];
                num += w * pair_ev(s, 0, [*h, *q], [i, j], None);
                den += w;
            }
        }
        num / den
    }

    #[test]
    fn turn_all_in_value_matches_brute_force_equity() {
        let hands = turn_hands(5);
        let mut sp = turn_spec(vec![]);
        sp.stacks = [0, 0];
        let tree = Tree::build(&sp, &hands).unwrap();
        let ranges = [sparse_range(hands.len(), 1, 25), sparse_range(hands.len(), 2, 25)];
        let solver = Solver::new(&hands, &tree, ranges);
        let (v, b) = (solver.value(0), brute_value(&solver));
        assert!(b.abs() > 1.0, "degenerate test: brute value {b}");
        assert!((v - b).abs() < 1e-9, "solver {v} vs brute force {b}");
        assert!((solver.value(0) + solver.value(1)).abs() < 1e-9);
    }

    #[test]
    fn turn_profile_value_matches_the_per_pair_oracle() {
        let hands = turn_hands(11);
        let tree = Tree::build(&small_turn_spec(), &hands).unwrap();
        let ranges = [sparse_range(hands.len(), 3, 14), sparse_range(hands.len(), 4, 14)];
        let mut solver = Solver::new(&hands, &tree, ranges);
        for _ in 0..8 {
            solver.step();
        }
        let (v, b) = (solver.value(0), brute_value(&solver));
        assert!((v - b).abs() < 1e-6 * (1.0 + b.abs()), "solver {v} vs per-pair oracle {b}");
        assert!((solver.value(0) + solver.value(1)).abs() < 1e-6);
    }

    #[test]
    fn turn_solve_converges_and_best_response_dominates() {
        let hands = turn_hands(13);
        let tree = Tree::build(&small_turn_spec(), &hands).unwrap();
        let ranges = [random_range(hands.len(), 5, 0.5), random_range(hands.len(), 6, 0.5)];
        let mut solver = Solver::new(&hands, &tree, ranges);
        for _ in 0..3 {
            solver.step();
        }
        let early = solver.exploitability();
        for _ in 3..150 {
            solver.step();
        }
        let e = solver.exploitability();
        let pot = 2000.0;
        assert!(e >= -1e-9 && e < early, "exploitability {early} -> {e}");
        assert!(e / pot < 0.01, "exploitability {:.3}% of pot after 150 iters", 100.0 * e / pot);
        let z = solver.joint_mass();
        for p in 0..2 {
            let v = solver.best_response(0, p, &solver.ranges[1 - p]);
            let br: f64 = solver.ranges[p].iter().zip(&v).map(|(r, v)| r * v).sum::<f64>() / z;
            assert!(br >= solver.value(p) - 1e-6);
        }
    }

    #[test]
    fn turn_solve_does_not_depend_on_the_thread_count() {
        let hands = turn_hands(17);
        let tree = Tree::build(&small_turn_spec(), &hands).unwrap();
        let ranges = [random_range(hands.len(), 7, 0.5), random_range(hands.len(), 8, 0.5)];
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut s = Solver::new(&hands, &tree, ranges.clone());
                for _ in 0..5 {
                    s.step();
                }
                (s.average(tree.decision_root), s.exploitability())
            })
        };
        let (a, b) = (run(1), run(4));
        assert!(a.0 == b.0 && a.1 == b.1, "results differ between 1 and 4 threads");
    }

    fn flop_hands(seed: u64, runouts: bool) -> Hands {
        let b = board(seed);
        Hands::flop(&[b[0], b[1], b[2]], runouts)
    }

    /// The small turn tree's flop menu, with these chips behind.
    fn flop_spec(stacks: [i64; 2]) -> RoundSpec {
        RoundSpec { river: None, stacks, ..small_turn_spec() }
    }

    #[test]
    fn flop_round_ends_in_leaves_and_runouts() {
        let hands = flop_hands(3, false);
        assert_eq!((hands.len(), hands.view_count()), (1176, 0));
        assert!(Tree::build(&small_turn_spec(), &hands).is_err(), "river menu on a flop solve");
        let tree = Tree::build(&flop_spec([3000, 3000]), &hands).unwrap();
        let total = [4000, 4000];
        assert_eq!(tree.total, total);
        let (mut leaves, mut runouts) = (0, 0);
        for node in &tree.nodes {
            match node {
                Node::Leaf { stake } => {
                    leaves += 1;
                    assert!(stake[0] == stake[1] && stake[0] < total[0], "leaf {stake:?}");
                }
                Node::Runout { stake } => {
                    runouts += 1;
                    assert!(stake[0] == total[0] || stake[1] == total[1], "runout {stake:?}");
                }
                Node::Terminal { folded, .. } => assert!(folded.is_some(), "a showdown on the flop"),
                Node::Chance { .. } => panic!("a chance node in a flop round"),
                Node::Decision { .. } => {}
            }
        }
        assert!(leaves > 2 && runouts > 2, "{leaves} leaves, {runouts} runouts");
        let n = hands.len();
        let f = tree.flatten(&hands, &[vec![1.0; n], vec![1.0; n]], &[], 1, vec![0]).unwrap();
        assert_eq!(f.total, total);
        assert!(f.strength.is_empty() && f.alive.is_empty(), "an external flop solve gets no views");
        for (i, &k) in f.kind.iter().enumerate() {
            let kids = &f.children[f.child_start[i] as usize..f.child_start[i + 1] as usize];
            assert!(kids.iter().all(|&c| c as usize > i), "a child before its parent");
            if k >= 3 {
                assert!(kids.is_empty() && f.player[i] == -1 && f.slot[i] == -1);
            }
        }
        assert_eq!(f.kind.iter().filter(|&&k| k == 3).count(), leaves);
        assert_eq!(f.kind.iter().filter(|&&k| k == 4).count(), runouts);
    }

    #[test]
    fn flop_runout_views_score_every_turn_and_river_pair() {
        let b = board(21);
        let hands = Hands::flop(&[b[0], b[1], b[2]], true);
        let n = hands.len();
        let s = hands.view_strengths();
        assert_eq!((hands.view_count(), s.len()), (n, n * n));
        let pairs = holdings_off(b[..3].iter().fold(0u64, |m, &c| m | 1u64 << c));
        let c = |i: u8| Card::from_index(i);
        let mut z = 5u64;
        for _ in 0..2000 {
            z = splitmix(z);
            let (v, h) = ((z % n as u64) as usize, ((z >> 32) % n as u64) as usize);
            let ([t, r], x) = (pairs[v], hands.combos[h]);
            let want = if x.contains(&t) || x.contains(&r) {
                0
            } else {
                eval7_fast(&[c(x[0]), c(x[1]), c(b[0]), c(b[1]), c(b[2]), c(t), c(r)])
            };
            assert_eq!(s[v * n + h], want, "runout {t} {r}, holding {x:?}");
        }
        // Each holding is live on the C(47, 2) runouts off its cards.
        for h in [0, n / 2, n - 1] {
            assert_eq!((0..n).filter(|&v| s[v * n + h] != 0).count(), 1081);
        }
    }

    #[test]
    fn flop_all_in_value_matches_brute_force_equity() {
        let hands = flop_hands(5, true);
        let tree = Tree::build(&flop_spec([0, 0]), &hands).unwrap();
        let ranges = [sparse_range(hands.len(), 1, 20), sparse_range(hands.len(), 2, 20)];
        let solver = Solver::new(&hands, &tree, ranges);
        let (v, b) = (solver.value(0), brute_value(&solver));
        assert!(b.abs() > 1.0, "degenerate test: brute value {b}");
        assert!((v - b).abs() < 1e-9, "solver {v} vs brute force {b}");
        assert!((solver.value(0) + solver.value(1)).abs() < 1e-9);
    }

    #[test]
    fn flop_profile_value_matches_the_per_pair_oracle() {
        // Leaves (a check-down here) and runouts both reached.
        let hands = flop_hands(11, true);
        let tree = Tree::build(&flop_spec([3000, 3000]), &hands).unwrap();
        let ranges = [sparse_range(hands.len(), 3, 10), sparse_range(hands.len(), 4, 10)];
        let mut solver = Solver::new(&hands, &tree, ranges);
        for _ in 0..6 {
            solver.step();
        }
        let (v, b) = (solver.value(0), brute_value(&solver));
        assert!((v - b).abs() < 1e-6 * (1.0 + b.abs()), "solver {v} vs per-pair oracle {b}");
        assert!((solver.value(0) + solver.value(1)).abs() < 1e-6);
    }

    #[test]
    fn flop_solve_converges_and_is_thread_count_independent() {
        let hands = flop_hands(13, true);
        let tree = Tree::build(&flop_spec([3000, 3000]), &hands).unwrap();
        let ranges = [random_range(hands.len(), 5, 0.5), random_range(hands.len(), 6, 0.5)];
        let run = |threads: usize, iters: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut s = Solver::new(&hands, &tree, ranges.clone());
                for _ in 0..iters {
                    s.step();
                }
                (s.average(tree.decision_root), s.exploitability())
            })
        };
        let (a, b) = (run(1, 3), run(4, 3));
        assert!(a.0 == b.0 && a.1 == b.1, "results differ between 1 and 4 threads");
        let e = run(8, 100).1;
        assert!(e >= -1e-9 && e < a.1, "exploitability {} -> {e}", a.1);
        assert!(e / 2000.0 < 0.01, "exploitability {:.3}% of pot after 100 iters", 100.0 * e / 2000.0);
    }

    fn oracle_menus() -> (Menu, Menu, Menu) {
        let m = |bet: &[f64], raise: &[f64], max_raises| Menu { bet_fracs: bet.to_vec(), raise_fracs: raise.to_vec(), max_raises };
        (m(&[0.5, 1.0], &[1.0], 2), m(&[0.75], &[], 1), m(&[0.75], &[], 1))
    }

    /// FLOPSEARCH.md D4's oracle game at 200bb: the flop search's flop menu,
    /// bet 0.75 pot or all-in on the turn and river (raise all-in).
    fn oracle_spec(half: i64, stack: i64) -> (RoundSpec, Menu) {
        let (flop, turn, river) = oracle_menus();
        (RoundSpec { stake: [half, half], stacks: [stack - half; 2], first: 1, min_bet: 100, menu: flop, river: Some(river), forced: vec![] }, turn)
    }

    #[test]
    fn full_flop_tree_matches_the_independent_counter() {
        // Decisions and actions from scripts/flop_tree_size.py's counter (its
        // own betting rules, gpu_search.Tree's) for the same menus: 2bb and
        // 18bb pots at 200bb.
        let b = board(3);
        let hands = Hands::flop_full(&[b[0], b[1], b[2]]);
        for (half, decisions, actions) in [(100, 511_580, 1_278_956), (900, 379_476, 920_276)] {
            let (spec, turn) = oracle_spec(half, 20_000);
            let tree = Tree::build_full(&spec, &turn, &hands, false).unwrap();
            assert_eq!((tree.decision_count(), tree.action_count()), (decisions, actions), "pot {}", 2 * half);
        }
        // The chance nodes: 49 turn cards (norm 45), each masking its holdings;
        // 48 river cards (norm 44) on the turn and river pair's runout view.
        let (spec, turn) = oracle_spec(900, 20_000);
        let tree = Tree::build_full(&spec, &turn, &hands, false).unwrap();
        let pairs = holdings_off(b[..3].iter().fold(0u64, |m, &c| m | 1u64 << c));
        let (mut turns, mut rivers) = (0, 0);
        for node in &tree.nodes {
            let Node::Chance { kids, norm } = node else { continue };
            if kids.len() == 49 {
                turns += 1;
                assert_eq!(*norm, 45.0);
                for k in kids {
                    let v = &hands.views[k.view];
                    let t = v.card.unwrap();
                    assert_eq!(v.dead, (0..hands.len()).filter(|&h| hands.combos[h].contains(&t)).collect::<Vec<_>>());
                }
            } else {
                rivers += 1;
                assert_eq!((kids.len(), *norm), (48, 44.0));
                // Runout views of (t, r) for one turn card t and 48 distinct rivers.
                assert!(kids.iter().all(|k| k.view < hands.runouts));
                let ps: Vec<[u8; 2]> = kids.iter().map(|k| pairs[k.view]).collect();
                let t = if ps[0][0] == ps[1][0] || ps[0][0] == ps[1][1] { ps[0][0] } else { ps[0][1] };
                let mut rs: Vec<u8> = ps.iter().map(|p| if p[0] == t { p[1] } else { assert_eq!(p[1], t); p[0] }).collect();
                rs.dedup();
                assert!(rs.len() == 48 && rs.windows(2).all(|w| w[0] < w[1]) && !rs.contains(&t));
            }
        }
        assert!(turns > 2 && rivers > 49 * turns, "{turns} turn and {rivers} river chance nodes");
        assert!(tree.flatten(&hands, &[vec![1.0; 1176], vec![1.0; 1176]], &[], 1, vec![0]).is_err(), "no flat form");
        assert!(Tree::build_full(&spec, &turn, &Hands::flop(&[b[0], b[1], b[2]], true), false).is_err(), "needs flop_full hands");
    }

    #[test]
    fn full_flop_runouts_equal_the_nested_chance_average() {
        let b = board(8);
        let hands = Hands::flop_full(&[b[0], b[1], b[2]]);
        let n = hands.len();
        let (spec, turn) = oracle_spec(300, 20_000);
        let tree = Tree::build_full(&spec, &turn, &hands, false).unwrap();
        let full = tree.full.as_ref().unwrap();
        let reach = random_range(n, 9, 0.3);
        let stake = [2000, 2000];
        let mut want = vec![0.0; n];
        runout_values_into(&hands, stake, &reach, &mut want);
        // Turn card t (masked), then river r on the pair's view, at 1/45 and 1/44.
        let mut got = vec![0.0; n];
        for &(t, tv) in &full.cards {
            let rt = masked(&reach, &hands.views[tv].dead);
            let mut acc = vec![0.0; n];
            for &(r, _) in full.cards.iter().filter(|&&(r, _)| r != t) {
                let v = full.pair[52 * t as usize + r as usize];
                let rr = masked(&rt, &hands.views[v].dead);
                let sd = terminal_values(&hands, None, stake, v, 0, &rr);
                for &h in &hands.views[v].dead {
                    assert_eq!(sd[h], 0.0);
                }
                acc.iter_mut().zip(&sd).for_each(|(a, s)| *a += s / 44.0);
            }
            for &h in &hands.views[tv].dead {
                acc[h] = 0.0;
            }
            got.iter_mut().zip(&acc).for_each(|(g, a)| *g += a / 45.0);
        }
        for h in 0..n {
            assert!((got[h] - want[h]).abs() <= 1e-9 * (1.0 + want[h].abs()), "holding {h}: {} vs {}", got[h], want[h]);
        }
    }

    #[test]
    fn full_flop_turn_subtrees_are_turn_solves() {
        // With the flop round frozen, each turn subtree below the check-check
        // line is its own turn solve from the reach there: same regrets and
        // strategy sums, bit for bit, as the turn solver on that turn board.
        let b = board(12);
        let flop = [b[0], b[1], b[2]];
        let (spec, turn) = oracle_spec(200, 800);
        let mut o = Oracle::new(flop, &spec, &turn, [random_range(1176, 13, 0.3), random_range(1176, 14, 0.3)]).unwrap();
        let n = o.hands.len();
        let mut sigma = Vec::new();
        for (i, (path, _, acts)) in o.flop_nodes().into_iter().enumerate() {
            let k = acts.len();
            let mut s = vec![0.0; k * n];
            for h in 0..n {
                let w: Vec<f64> = (0..k).map(|a| 0.05 + unit(splitmix((i * 7919 + h * 31 + a) as u64))).collect();
                let tot: f64 = w.iter().sum();
                (0..k).for_each(|a| s[a * n + h] = w[a] / tot);
            }
            if path.is_empty() || path == [0] {
                sigma.push(s.clone());
            }
            o.freeze(&path, s).unwrap();
        }
        let iters = 4;
        o.run(iters, GAMMA);
        // The check-check line: root (player 1) checks, player 0 checks.
        let a = o.node_at(&[0]).unwrap();
        let Node::Decision { children, .. } = &o.tree.nodes[a] else { panic!() };
        let Node::Chance { kids, .. } = &o.tree.nodes[children[0]] else { panic!("check-check is not a turn deal") };
        let reach = [
            (0..n).map(|h| o.ranges[0][h] * sigma[1][h]).collect::<Vec<f64>>(),
            (0..n).map(|h| o.ranges[1][h] * sigma[0][h]).collect::<Vec<f64>>(),
        ];
        let (menu, river) = (turn.clone(), spec.river.clone());
        for kid in kids.iter().step_by(12) {
            let t = o.hands.views[kid.view].card.unwrap();
            let th = Hands::turn(&[flop[0], flop[1], flop[2], t]);
            let map: Vec<usize> = th.combos.iter().map(|x| o.hands.combos.iter().position(|y| y == x).unwrap()).collect();
            let ts = RoundSpec { stake: [200, 200], stacks: [600, 600], menu: menu.clone(), river: river.clone(), ..spec.clone() };
            let tt = Tree::build(&ts, &th).unwrap();
            let mut s = Solver::new(&th, &tt, [map.iter().map(|&h| reach[0][h]).collect(), map.iter().map(|&h| reach[1][h]).collect()]);
            for _ in 0..iters {
                s.step();
            }
            assert_eq!(kid.slots.len(), tt.slots, "turn {t}: subtree size");
            for (i, want) in s.store.iter().enumerate() {
                let got = &o.store[kid.slots.start + i];
                let k = tt.slot_acts[i];
                for a in 0..k {
                    for (j, &h) in map.iter().enumerate() {
                        let (g, w) = ((got.regret[a * n + h], got.sum[a * n + h]), (want.regret[a * th.len() + j], want.sum[a * th.len() + j]));
                        assert!(g == w, "turn {t}, slot {i}, action {a}, holding {j}: {g:?} vs {w:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn full_flop_oracle_converges_and_brackets_the_value() {
        let b = board(15);
        let (spec, turn) = oracle_spec(200, 800);
        let ranges = [random_range(1176, 16, 0.3), random_range(1176, 17, 0.3)];
        let mut o = Oracle::new([b[0], b[1], b[2]], &spec, &turn, ranges.clone()).unwrap();
        assert!(o.store.is_empty() && !o.flop_nodes().is_empty() && o.store.is_empty(), "the store waits for a solve");
        o.run(10, GAMMA);
        assert_eq!(o.store.len(), o.tree.slots);
        let e10: f64 = o.best_responses().iter().sum();
        o.run(40, GAMMA);
        let br = o.best_responses();
        let v0 = o.value(0);
        assert!(br[0] + br[1] >= -1e-9 && br[0] + br[1] < e10, "exploitability {e10} -> {}", br[0] + br[1]);
        assert!(-br[1] - 1e-9 <= v0 && v0 <= br[0] + 1e-9, "profile value {v0} outside [{}, {}]", -br[1], br[0]);
        assert!((v0 + o.value(1)).abs() < 1e-9);
        assert!(100.0 * (br[0] + br[1]) / 400.0 < 5.0, "exploitability {:.2}% of the pot", 100.0 * (br[0] + br[1]) / 400.0);
        // Another averaging weight solves the same game.
        let mut g = Oracle::new([b[0], b[1], b[2]], &spec, &turn, ranges.clone()).unwrap();
        g.run(50, 5.0);
        let gb = g.best_responses();
        assert!(gb[0] + gb[1] >= -1e-9 && 100.0 * (gb[0] + gb[1]) / 400.0 < 5.0, "gamma 5: {gb:?}");
        assert!(br[0].min(gb[0]) + 1e-9 >= (-br[1]).max(-gb[1]), "the brackets overlap: {br:?} {gb:?}");
        drop(g);
        // The same solve in an f32 store (each stored result rounded once).
        let mut h = Oracle::<f32>::zeroed([b[0], b[1], b[2]], &spec, &turn, ranges.clone(), false).unwrap();
        h.run(10, GAMMA);
        h.run(40, GAMMA);
        let hb = h.best_responses();
        let hv = h.value(0);
        let root = (o.average(&[]).unwrap(), h.average(&[]).unwrap());
        let gap = root.0.iter().zip(&root.1).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        // Measured: 2e-4 chips (5e-7 of the pot) and 4e-4 in probability.
        let close = |a: f64, b: f64| (a - b).abs() < 1e-5 * 400.0;
        assert!(close(hb[0], br[0]) && close(hb[1], br[1]) && close(hv, v0) && gap < 5e-3, "f32 {hb:?} {hv} vs f64 {br:?} {v0}, root strategy gap {gap}");
        drop(h);
        // A frozen root: its strategy comes back as given; the best responder
        // (player 1, the root's actor) may not deviate there.
        let nodes = o.flop_nodes();
        assert!(nodes[0].0.is_empty() && nodes[0].1 == 1);
        let k = nodes[0].2.len();
        let fixed: Vec<f64> = (0..k * 1176).map(|i| if i / 1176 == k - 1 { 1.0 } else { 0.0 }).collect(); // always all-in
        let mut f = Oracle::new([b[0], b[1], b[2]], &spec, &turn, ranges).unwrap();
        f.freeze(&[], fixed.clone()).unwrap();
        f.run(50, GAMMA);
        assert_eq!(f.average(&[]).unwrap(), fixed);
        let fb = f.best_responses();
        assert!(fb[1] < br[1] - 1e-3, "shoving every hand should cost player 1: {} vs {}", fb[1], br[1]);
        assert!(f.freeze(&[9, 9, 9], vec![]).is_err() && f.average(&[0, 0]).is_err(), "paths off the flop round");
    }

    /// A random range, constant on each orbit of `tree`'s suit swaps.
    fn symmetric_range(tree: &Tree, n: usize, seed: u64) -> Vec<f64> {
        let base = random_range(n, seed, 0.3);
        (0..n)
            .map(|h| {
                let mut orbit = vec![h];
                let mut i = 0;
                while i < orbit.len() {
                    for p in &tree.perms {
                        let j = p[orbit[i]] as usize;
                        if !orbit.contains(&j) {
                            orbit.push(j);
                        }
                    }
                    i += 1;
                }
                base[*orbit.iter().min().expect("h itself")]
            })
            .collect()
    }

    #[test]
    fn full_flop_isomorphism_solves_the_same_game() {
        let (spec, turn) = oracle_spec(200, 800);
        // Ranks A 12, K 11, Q 10, 7 5; suit = card & 3. Two-tone (suits 2 and 3
        // alike), monotone (1, 2, 3), paired (0 and 1: an ace each) and rainbow
        // (none): canonical turn cards, and the share of river deals grown.
        for (flop, turns, share) in [([48u8, 44, 21], 36, 0.65), ([48, 44, 40], 23, 0.33), ([48, 49, 22], 37, 0.67), ([48, 45, 22], 49, 1.0)] {
            let hands = Hands::flop_full(&flop);
            let n = hands.len();
            let iso = Tree::build_full(&spec, &turn, &hands, true).unwrap();
            let full = Tree::build_full(&spec, &turn, &hands, false).unwrap();
            // Every chance node still accounts for all its deals.
            let mut first = None;
            for node in &iso.nodes {
                let Node::Chance { kids, norm } = node else { continue };
                let deals: usize = kids.iter().map(|k| 1 + k.images.len()).sum();
                assert_eq!(deals as f64, norm + 4.0, "flop {flop:?}");
                first.get_or_insert(kids.len());
            }
            assert_eq!(first, Some(turns), "flop {flop:?}: canonical turn cards");
            assert!((iso.slots as f64) < share * full.slots as f64 + 0.5, "flop {flop:?}: {} of {} slots", iso.slots, full.slots);
            let ranges = [symmetric_range(&iso, n, 21), symmetric_range(&iso, n, 22)];
            let mut a = Oracle::<f64>::zeroed(flop, &spec, &turn, ranges.clone(), true).unwrap();
            let mut b = Oracle::new(flop, &spec, &turn, ranges.clone()).unwrap();
            assert_eq!(a.isomorphic(), turns < 49);
            // A frozen root, constant on the orbits, as the M1 candidates are.
            let k = a.flop_nodes()[0].2.len();
            let x: Vec<f64> = (0..k).flat_map(|i| symmetric_range(&iso, n, 30 + i as u64)).map(|w| w + 0.1).collect();
            let x: Vec<f64> = (0..k * n).map(|i| x[i] / (0..k).map(|j| x[j * n + i % n]).sum::<f64>()).collect();
            a.freeze(&[], x.clone()).unwrap();
            b.freeze(&[], x.clone()).unwrap();
            // The same iterates summed in another order, so equal up to rounding.
            // Regret matching amplifies it once rounding tips an exactly tied
            // regret: on these boards the gap is ~1e-14 through iteration 2 and
            // ~1e-4 chips from iteration 3 or 4 on. The full game against itself
            // with both ranges tripled (the same game, other rounding) jumps the
            // same way (FLOPSEARCH.md, D4 isomorphism amendment). Compare at 2.
            let iters = if turns == 49 { 20 } else { 2 };
            a.run(iters, 5.0);
            b.run(iters, 5.0);
            let (ab, bb) = (a.best_responses(), b.best_responses());
            let (av, bv) = (a.value(0), b.value(0));
            if turns == 49 {
                // Nothing to share: the same tree, the same numbers.
                assert_eq!(iso.slots, full.slots);
                assert!(ab == bb && av == bv, "rainbow: {ab:?} {av} vs {bb:?} {bv}");
                continue;
            }
            let close = |x: f64, y: f64| (x - y).abs() < 1e-9 * 400.0;
            assert!(close(ab[0], bb[0]) && close(ab[1], bb[1]) && close(av, bv), "flop {flop:?}: {ab:?} {av} vs {bb:?} {bv}");
            let path = [0usize];
            let gap = a.average(&path).unwrap().iter().zip(&b.average(&path).unwrap()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
            assert!(gap < 1e-9, "flop {flop:?}: strategy gap {gap}");
            // Asymmetric input is refused.
            let mut bad = ranges.clone();
            let h = (0..n).find(|&h| iso.perms[0][h] as usize != h).unwrap();
            bad[1][h] += 0.5;
            assert!(Oracle::<f64>::zeroed(flop, &spec, &turn, bad.clone(), true).is_err());
            assert!(Oracle::<f64>::zeroed(flop, &spec, &turn, bad, false).is_ok());
            let mut skew = x.clone();
            let (s, t) = (h, iso.perms[0][h] as usize);
            (skew[s], skew[n + s]) = (skew[s] + skew[n + s], 0.0);
            let mut c = Oracle::<f64>::zeroed(flop, &spec, &turn, ranges.clone(), true).unwrap();
            assert!(c.freeze(&[], skew).is_err(), "holding {s} against {t}");
        }
    }
}

#[cfg(test)]
mod convergence_probe {
    use super::tests_support::*;
    use super::*;

    fn probe(hands: &Hands, spec: &RoundSpec, iters: &[u64]) {
        let tree = Tree::build(spec, hands).unwrap();
        let ranges = [range(hands.len(), 0), range(hands.len(), 50)];
        let mut solver = Solver::new(hands, &tree, ranges);
        let t = std::time::Instant::now();
        let pot = (spec.stake[0] + spec.stake[1]) as f64;
        for it in 1..=*iters.last().unwrap() {
            solver.step();
            if iters.contains(&it) {
                let solve = t.elapsed().as_secs_f64();
                println!(
                    "iter {it:5}: exploitability {:.4}% pot  ({solve:.2}s, {} decision nodes)",
                    100.0 * solver.exploitability() / pot,
                    tree.decision_count()
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn print_river_convergence() {
        probe(&Hands::river(&board(7)), &spec(vec![]), &[10, 30, 100, 300, 1000]);
    }

    #[test]
    #[ignore]
    fn print_turn_convergence() {
        let b = board(7);
        let hands = Hands::turn(&[b[0], b[1], b[2], b[3]]);
        println!("small menu (turn 0.5/1.0 pot, river 0.75 pot):");
        probe(&hands, &small_turn_spec(), &[10, 30, 100, 200]);
        println!("full menu (turn and river 0.33/0.75/1.5 pot):");
        probe(&hands, &turn_spec(vec![]), &[10, 30, 100, 200]);
    }
}

/// The solver as it was before the buffer-reusing, parallel `cfr`: one
/// allocation per vector, children in sequence. The new one must reproduce
/// it bit for bit.
#[cfg(test)]
mod reference {
    use super::tests_support::*;
    use super::*;

    fn current(regret: &[f64], k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; k * n];
        for h in 0..n {
            let mut tot = 0.0;
            for a in 0..k {
                tot += regret[a * n + h].max(0.0);
            }
            for a in 0..k {
                out[a * n + h] = if tot > 0.0 { regret[a * n + h].max(0.0) / tot } else { 1.0 / k as f64 };
            }
        }
        out
    }

    fn cfr(cx: &Ctx, store: &mut [Slot], base: usize, node: usize, p: usize, reach_q: &[f64], reach_p: &[f64]) -> Vec<f64> {
        let n = cx.hands.len();
        match &cx.tree.nodes[node] {
            Node::Terminal { folded, stake, view } => terminal_values(cx.hands, *folded, *stake, *view, p, reach_q),
            Node::Leaf { stake } | Node::Runout { stake } => {
                let mut out = vec![0.0; n];
                runout_values_into(cx.hands, *stake, reach_q, &mut out);
                out
            }
            Node::Chance { kids, norm } => {
                let parts = split_for(store, base, kids);
                let vals: Vec<Vec<f64>> = parts
                    .into_iter()
                    .zip(kids)
                    .map(|(part, kid)| {
                        let dead = &cx.hands.views[kid.view].dead;
                        let (rq, rp) = (masked(reach_q, dead), masked(reach_p, dead));
                        let mut v = cfr(cx, part, kid.slots.start, kid.node, p, &rq, &rp);
                        for &h in dead {
                            v[h] = 0.0;
                        }
                        v
                    })
                    .collect();
                let mut out = vec![0.0; n];
                for v in &vals {
                    for h in 0..n {
                        out[h] += v[h];
                    }
                }
                out.iter_mut().for_each(|x| *x /= norm);
                out
            }
            Node::Decision { player, acts, children, slot, .. } => {
                let k = acts.len();
                let s = *slot - base;
                if let Some(f) = store[s].frozen.clone() {
                    let mut node_v = vec![0.0; n];
                    for a in 0..k {
                        let v = if *player == p {
                            let own: Vec<f64> = (0..n).map(|h| reach_p[h] * f[a * n + h]).collect();
                            let v = cfr(cx, store, base, children[a], p, reach_q, &own);
                            (0..n).map(|h| f[a * n + h] * v[h]).collect::<Vec<f64>>()
                        } else {
                            let child_reach: Vec<f64> = (0..n).map(|h| reach_q[h] * f[a * n + h]).collect();
                            cfr(cx, store, base, children[a], p, &child_reach, reach_p)
                        };
                        for h in 0..n {
                            node_v[h] += v[h];
                        }
                    }
                    return node_v;
                }
                let sigma = current(&store[s].regret, k, n);
                if *player == p {
                    let mut vals = Vec::with_capacity(k);
                    let mut node_v = vec![0.0; n];
                    for a in 0..k {
                        let own: Vec<f64> = (0..n).map(|h| reach_p[h] * sigma[a * n + h]).collect();
                        let v = cfr(cx, store, base, children[a], p, reach_q, &own);
                        for h in 0..n {
                            node_v[h] += sigma[a * n + h] * v[h];
                        }
                        vals.push(v);
                    }
                    let r = &mut store[s].regret;
                    for a in 0..k {
                        for h in 0..n {
                            r[a * n + h] += vals[a][h] - node_v[h];
                        }
                    }
                    node_v
                } else {
                    {
                        let sum = &mut store[s].sum;
                        for a in 0..k {
                            for h in 0..n {
                                sum[a * n + h] = sum[a * n + h] * cx.sum_w + reach_q[h] * sigma[a * n + h];
                            }
                        }
                    }
                    let mut node_v = vec![0.0; n];
                    for a in 0..k {
                        let child_reach: Vec<f64> = (0..n).map(|h| reach_q[h] * sigma[a * n + h]).collect();
                        let v = cfr(cx, store, base, children[a], p, &child_reach, reach_p);
                        for h in 0..n {
                            node_v[h] += v[h];
                        }
                    }
                    node_v
                }
            }
        }
    }

    fn step(s: &mut Solver) {
        s.iter += 1;
        let t = s.iter as f64;
        let pos = t.powf(ALPHA) / (t.powf(ALPHA) + 1.0);
        for slot in s.store.iter_mut() {
            for x in slot.regret.iter_mut() {
                *x *= if *x > 0.0 { pos } else { 0.5 };
            }
        }
        let cx = Ctx { hands: s.hands, tree: s.tree, sum_w: (t / (t + 1.0)).powf(GAMMA), regret_w: None };
        for p in 0..2 {
            let (rq, rp) = (s.ranges[1 - p].clone(), s.ranges[p].clone());
            cfr(&cx, &mut s.store, 0, 0, p, &rq, &rp);
        }
    }

    fn state(s: &Solver) -> Vec<(Vec<f64>, Vec<f64>)> {
        s.store.iter().map(|x| (x.regret.clone(), x.sum.clone())).collect()
    }

    /// Run the reference and the new solver side by side (the new one on 1
    /// and on 8 threads) and require identical regrets and strategy sums.
    fn check(hands: &Hands, tree: &Tree, freeze: Option<(usize, Vec<f64>)>, iters: usize) {
        let ranges = [range(hands.len(), 31), range(hands.len(), 32)];
        let make = || {
            let mut s = Solver::new(hands, tree, ranges.clone());
            if let Some((node, f)) = &freeze {
                s.freeze(*node, f.clone()).unwrap();
            }
            s
        };
        let mut old = make();
        for _ in 0..iters {
            step(&mut old);
        }
        let want = state(&old);
        for threads in [1, 8] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let got = pool.install(|| {
                let mut s = make();
                for _ in 0..iters {
                    s.step();
                }
                state(&s)
            });
            assert!(got == want, "{threads} threads: the solver no longer reproduces the reference");
        }
    }

    /// Old vs new solver speed on the agent's river and turn menus,
    /// interleaved (old, new on 1 thread, new on all threads) so machine
    /// drift hits every variant alike. Run it on an otherwise idle machine:
    ///   cargo test --no-default-features --release --lib -- --ignored --nocapture print_solver_speed
    #[test]
    #[ignore]
    fn print_solver_speed() {
        let b = board(7);
        let river = Hands::river(&b);
        let turn = Hands::turn(&[b[0], b[1], b[2], b[3]]);
        let cases = [
            ("river", &river, Tree::build(&spec(vec![]), &river).unwrap(), 40),
            ("turn", &turn, Tree::build(&small_turn_spec(), &turn).unwrap(), 6),
        ];
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let all = rayon::ThreadPoolBuilder::new().build().unwrap();
        for (street, hands, tree, iters) in &cases {
            let ranges = [range(hands.len(), 31), range(hands.len(), 32)];
            let time = |pool: &rayon::ThreadPool, new: bool| {
                pool.install(|| {
                    let mut s = Solver::new(hands, tree, ranges.clone());
                    let clock = std::time::Instant::now();
                    for _ in 0..*iters {
                        if new {
                            s.step();
                        } else {
                            step(&mut s);
                        }
                    }
                    1000.0 * clock.elapsed().as_secs_f64() / *iters as f64
                })
            };
            let mut ms = [0.0f64; 3];
            let rounds = 3;
            for _ in 0..rounds {
                ms[0] += time(&one, false) / rounds as f64;
                ms[1] += time(&one, true) / rounds as f64;
                ms[2] += time(&all, true) / rounds as f64;
            }
            println!(
                "{street:5} ({} decision nodes): old {:.3} ms/iter | new 1 thread {:.3} ({:.2}x) | new {} threads {:.3} ({:.2}x)",
                tree.decision_count(),
                ms[0],
                ms[1],
                ms[0] / ms[1],
                all.current_num_threads(),
                ms[2],
                ms[0] / ms[2]
            );
        }
    }

    #[test]
    fn new_solver_matches_the_reference_bit_for_bit() {
        // River, with parallel subtrees at the top.
        let hands = Hands::river(&board(4));
        let tree = Tree::build(&spec(vec![]), &hands).unwrap();
        assert!(tree.decision_count() > 2 * PAR_MIN_SLOTS, "tree too small to exercise the parallel path");
        check(&hands, &tree, None, 12);
        // A frozen node on the real line, and a solved branch below it.
        let mk = RoundSpec {
            forced: vec![
                Forced::Frozen(Act::CheckCall, vec![Act::CheckCall, Act::Bet(660), Act::Bet(9000)]),
                Forced::Branch(Act::Bet(2000)),
            ],
            ..spec(vec![])
        };
        let tree = Tree::build(&mk, &hands).unwrap();
        let (node, _) = tree.forced_path[0];
        let k = tree.acts_at(node).unwrap().len();
        let n = hands.len();
        let mut f = vec![0.0; k * n];
        for h in 0..n {
            let w = [0.2 + 0.6 * unit(splitmix(h as u64)), 0.1, 0.0];
            let tot: f64 = w.iter().sum();
            for a in 0..k {
                f[a * n + h] = w[a] / tot;
            }
        }
        check(&hands, &tree, Some((node, f)), 12);
        // Turn: the chance node and the river subtrees below it.
        let b = board(9);
        let hands = Hands::turn(&[b[0], b[1], b[2], b[3]]);
        let tree = Tree::build(&small_turn_spec(), &hands).unwrap();
        check(&hands, &tree, None, 4);
    }
}

#[cfg(test)]
mod tests_support {
    use super::*;
    pub fn splitmix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn unit(x: u64) -> f64 {
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn board(seed: u64) -> [u8; 5] {
        let mut deck: Vec<u8> = (0..52).collect();
        let mut s = seed;
        for i in 0..5 {
            s = splitmix(s);
            let j = i + (s % (52 - i as u64)) as usize;
            deck.swap(i, j);
        }
        [deck[0], deck[1], deck[2], deck[3], deck[4]]
    }
    pub fn range(n: usize, seed: u64) -> Vec<f64> {
        (0..n).map(|i| unit(splitmix(seed ^ i as u64))).collect()
    }
    fn menu(bet: &[f64], raise: &[f64], max_raises: u8) -> Menu {
        Menu { bet_fracs: bet.to_vec(), raise_fracs: raise.to_vec(), max_raises }
    }
    pub fn spec(forced: Vec<Act>) -> RoundSpec {
        RoundSpec {
            stake: [1000, 1000],
            stacks: [9000, 9000],
            first: 1,
            min_bet: 100,
            menu: menu(&[0.33, 0.75, 1.5], &[1.0], 3),
            river: None,
            forced: forced.into_iter().map(Forced::Only).collect(),
        }
    }
    pub fn turn_spec(forced: Vec<Act>) -> RoundSpec {
        RoundSpec { river: Some(menu(&[0.33, 0.75, 1.5], &[1.0], 3)), ..spec(forced) }
    }
    /// A small turn tree for tests: few sizes, so each iteration is quick.
    pub fn small_turn_spec() -> RoundSpec {
        RoundSpec {
            menu: menu(&[0.5, 1.0], &[1.0], 2),
            river: Some(menu(&[0.75], &[], 1)),
            ..spec(vec![])
        }
    }
}
