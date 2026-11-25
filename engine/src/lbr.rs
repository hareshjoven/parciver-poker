//! Local best response (LBR; Lisý & Bowling, AAAI-17 workshop "Equilibrium
//! approximation quality of current no-limit poker bots") against the play
//! agent: a cheap, sound LOWER BOUND on how exploitable the bot is.
//!
//! LBR is itself a legal strategy, so what it wins against the fixed agent can
//! never exceed the true best-response value — the property the small-game test
//! checks exactly. At each of its decisions LBR:
//!
//! * keeps the agent's RANGE: a weight per possible agent holding, updated by
//!   Bayes' rule with the agent's own action probabilities each time it acts
//!   (the agent's full real-action distribution, `Decision::options`);
//! * scores fold / check-call / a few bet sizes assuming that after its action
//!   the hand is checked down to showdown, with the agent's fold probability to
//!   each bet read from the same policy:
//!   `U(call) = wp * (pot + c) - c`,
//!   `U(bet d) = fp * pot + (1 - fp) * (wp' * (pot + 2d - c) - d)`,
//!   where `wp` is LBR's equity against the range and `wp'` against the part of
//!   the range that does not fold (`pot` includes both players' chips, `c` is
//!   what LBR owes, `d` what it adds);
//! * plays the best, never folding when checking is free.
//!
//! Preflop, LBR only checks or calls (as in the original paper's setting), so the
//! bound measures postflop play. The agent's randomized bet translation is
//! seeded per hand; LBR evaluates the agent with that same seed, i.e. it sees how
//! the agent read its bet — a slightly stronger adversary than a real opponent.
//! Without that peek (`peek = false`) LBR tracks the agent's holding jointly
//! with each possible reading of its off-grid bets, weighted by the agent's
//! translation probabilities (a diagnostic, see V2LEAK.md).
//!
//! LBR-open (`Open`, records/preflop-open/FINDING.md) replaces LBR's first
//! action in the small blind with a raise to a fixed size: with every hand,
//! or where an opener policy's root action is its grid open (its fold, limp
//! and all-in kept). A fixed preflop policy followed by LBR is still a legal
//! strategy, so the bound stays a bound.

use std::sync::Arc;

use rayon::prelude::*;

use crate::cards::Card;
use crate::game::Game;
use crate::hand_eval::{eval5, eval7_fast};
use crate::nlhe::{Nlhe, NlheState, RealAct};
use crate::play::{Agent, Force, HandSpec, Source};
use crate::subgame::{beliefs, search_failure};

/// Pot fractions LBR considers betting (plus all-in).
const BET_FRACS: [f64; 3] = [0.5, 1.0, 2.0];

/// Which bet sizes LBR considers after the flop.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(crate) enum Sizes {
    /// `BET_FRACS` of the pot, plus all-in: the standard LBR.
    #[default]
    Fixed,
    /// Only the agent's own grid sizes at the node, plus all-in, so every LBR
    /// bet maps to the agent's tree exactly: no randomized translation and no
    /// off-tree hands (a diagnostic, see V2LEAK.md).
    Grid,
}

impl Sizes {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "fixed" => Some(Sizes::Fixed),
            "grid" => Some(Sizes::Grid),
            _ => None,
        }
    }
}
/// LBR-open: LBR's first action in the small blind.
#[derive(Clone, Copy)]
pub(crate) struct Open<'a> {
    /// The chips the small blind raises to (its own blind included).
    pub to: i64,
    /// `None`: raise with every hand. Else this policy's root action for LBR's
    /// hand, sampled on LBR's own stream, with its grid open (pot-size)
    /// replaced by the raise to `to`.
    pub opener: Option<&'a Agent>,
}

/// The stream of LBR's hand seed the `Open` opener samples from (its rollouts
/// use the seed itself).
const OPEN_STREAM: u64 = 0x09E4_0BE4_5EED_0001;

/// Sampled board completions when two or more cards are still to come.
const ROLLOUTS: usize = 128;

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        splitmix(self.0)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// Showdown score of `hole` on a full board (3-card board: `eval5`, 5: `eval7`).
fn score(hole: [u8; 2], board: &[u8]) -> u32 {
    let c = |i: u8| Card::from_index(i);
    match board.len() {
        3 => eval5(&[c(hole[0]), c(hole[1]), c(board[0]), c(board[1]), c(board[2])]),
        5 => eval7_fast(&[
            c(hole[0]),
            c(hole[1]),
            c(board[0]),
            c(board[1]),
            c(board[2]),
            c(board[3]),
            c(board[4]),
        ]),
        n => unreachable!("showdown on {n} board cards"),
    }
}

struct Ctx<'a> {
    agent: &'a Agent,
    game: Nlhe,
    board_cards: usize,
    deck: Vec<u8>,
    sb: i64,
    bb: i64,
    stack: i64,
    sizes: Sizes,
    /// Evaluate the agent with its realized translation seed (standard LBR).
    peek: bool,
    /// LBR-open (`None`: the standard LBR).
    open: Option<Open<'a>>,
    /// Tests: range updates holding by holding (the oracle for `probs_of`).
    #[cfg(test)]
    per_holding: bool,
    /// Tests: LBR's model of the agent uses the translation seed XOR this
    /// (the agent itself plays with the true seed).
    #[cfg(test)]
    model_seed_xor: u64,
    /// Tests: every value LBR gives an action, and the most paths it tracked.
    #[cfg(test)]
    probe: std::sync::Mutex<(Vec<u64>, usize)>,
    /// Tests: the model's queries of the agent's policy at a public state where
    /// the agent searches (`probs_of` calls; each may count one failure).
    #[cfg(test)]
    search_queries: std::sync::atomic::AtomicU64,
}

/// LBR's equity against every holding in the range (0 weight -> ignored).
fn equities(ctx: &Ctx, me: [u8; 2], board: &[u8], range: &[([u8; 2], f64)], rng: &mut Rng) -> Vec<f64> {
    let need = ctx.board_cards - board.len();
    let dead = me.iter().chain(board).fold(0u64, |m, &c| m | 1u64 << c);
    let live: Vec<u8> = ctx.deck.iter().copied().filter(|&c| dead & (1u64 << c) == 0).collect();
    // Board completions: all of them for one card to come, else a shared sample.
    let mut runouts: Vec<Vec<u8>> = Vec::new();
    if need == 0 {
        runouts.push(Vec::new());
    } else if need == 1 {
        runouts.extend(live.iter().map(|&c| vec![c]));
    } else {
        for _ in 0..ROLLOUTS {
            let mut pick = live.clone();
            for k in 0..need {
                let j = k + rng.below((pick.len() - k) as u64) as usize;
                pick.swap(k, j);
            }
            pick.truncate(need);
            runouts.push(pick);
        }
    }
    // Per runout: the full board, its cards, and LBR's score (the same for
    // every holding).
    let fulls: Vec<(Vec<u8>, u64, u32)> = runouts
        .iter()
        .map(|r| {
            let full = [board, r].concat();
            let mine = score(me, &full);
            (full, r.iter().fold(0u64, |m, &c| m | 1u64 << c), mine)
        })
        .collect();
    range
        .iter()
        .map(|&(h, w)| {
            if w <= 0.0 {
                return 0.0;
            }
            let hm = (1u64 << h[0]) | (1u64 << h[1]);
            let (mut won, mut n) = (0.0, 0.0);
            for (full, rm, a) in &fulls {
                if hm & rm != 0 {
                    continue;
                }
                let b = score(h, full);
                won += if *a > b { 1.0 } else if *a == b { 0.5 } else { 0.0 };
                n += 1.0;
            }
            if n > 0.0 { won / n } else { 0.5 }
        })
        .collect()
}

fn agent_spec(ctx: &Ctx, seat: usize, hole: [u8; 2], board: &[u8], actions: &[RealAct], seed: u64) -> HandSpec {
    HandSpec {
        hero: seat,
        hole,
        board: board.to_vec(),
        sb: ctx.sb,
        bb: ctx.bb,
        stacks: [ctx.stack, ctx.stack],
        actions: actions.to_vec(),
        seed,
    }
}

/// Failed searches LBR's model of the agent read (its range updates, and the
/// agent's answer to each bet LBR weighs): one per query of the agent's
/// policy at a public state, as `[built-in solver used in place of the
/// external one, blueprint used]` (`subgame::search_failure`).
type ModelFailed = [u64; 2];

/// Each holding's probability of taking `act` at this node, one
/// `decide_inner` per holding; a failed search any of them read counts once.
/// A holding the agent cannot decide for is an error (not a probability 0).
#[allow(clippy::too_many_arguments)]
fn probs_one_by_one(
    ctx: &Ctx,
    seat: usize,
    hs: &[[u8; 2]],
    board: &[u8],
    actions: &[RealAct],
    seed: u64,
    act: RealAct,
    failed: &mut ModelFailed,
) -> Result<Vec<f64>, String> {
    let mut reached = None;
    let probs = hs
        .iter()
        .map(|&h| {
            let spec = agent_spec(ctx, seat, h, board, actions, seed);
            let d = ctx.agent.decide_inner(&spec)?;
            reached = reached.or(search_failure(ctx.agent, &spec, &d));
            Ok(d.options.iter().filter(|(a, _)| *a == act).map(|(_, p)| p).sum())
        })
        .collect::<Result<_, String>>()?;
    if let Some(k) = reached {
        failed[k] += 1;
    }
    Ok(probs)
}

/// `probs_one_by_one` with one `options_many_failure` (the same options and
/// counts, the public work done once), or holding by holding if it fails.
#[allow(clippy::too_many_arguments)]
fn probs_of(
    ctx: &Ctx,
    seat: usize,
    hs: &[[u8; 2]],
    board: &[u8],
    actions: &[RealAct],
    seed: u64,
    act: RealAct,
    failed: &mut ModelFailed,
) -> Result<Vec<f64>, String> {
    let Some(&first) = hs.first() else { return Ok(Vec::new()) };
    let spec = agent_spec(ctx, seat, first, board, actions, seed);
    #[cfg(test)]
    {
        if ctx.agent.search_cfg().is_some_and(|c| crate::subgame::applies(c, &spec)) {
            ctx.search_queries.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if ctx.per_holding {
            return probs_one_by_one(ctx, seat, hs, board, actions, seed, act, failed);
        }
    }
    match ctx.agent.options_many_failure(&spec, hs) {
        Ok((os, reached)) => {
            if let Some(k) = reached {
                failed[k] += 1;
            }
            Ok(os.iter().map(|o| o.iter().filter(|(a, _)| *a == act).map(|(_, p)| p).sum()).collect())
        }
        Err(_) => probs_one_by_one(ctx, seat, hs, board, actions, seed, act, failed),
    }
}

/// The holdings in `range` with positive weight.
fn live(range: &[([u8; 2], f64)]) -> Vec<[u8; 2]> {
    range.iter().filter(|r| r.1 > 0.0).map(|r| r.0).collect()
}

/// LBR's belief without the peek: per combination of readings of its off-grid
/// bets (`Force`), a weight for each holding of `range` (same order).
type Paths = Vec<(Force, Vec<f64>)>;

/// The holdings of `range` with positive weight in `w`.
fn live_in(range: &[([u8; 2], f64)], w: &[f64]) -> Vec<[u8; 2]> {
    range.iter().zip(w).filter(|(_, &x)| x > 0.0).map(|(r, _)| r.0).collect()
}

/// Each holding's weight in `range`: its total over the paths.
fn marginal(paths: &Paths, range: &mut [([u8; 2], f64)]) {
    for (k, r) in range.iter_mut().enumerate() {
        r.1 = paths.iter().map(|(_, w)| w[k]).sum();
    }
}

/// `force` extended with action `i` read as the lower, and as the higher size.
fn read_as(force: Force, i: usize) -> Result<(Force, Force), String> {
    if i >= 64 {
        return Err(format!("peek-free LBR fixes readings of the first 64 actions only, not action {i}"));
    }
    let bit = 1u64 << i;
    Ok((Force { mask: force.mask | bit, lower: force.lower | bit }, Force { mask: force.mask | bit, lower: force.lower & !bit }))
}

/// `probs_of` with the agent's readings of LBR's bets fixed by `force` (none
/// fixed: `probs_of` itself; fixed ones are read off the blueprint only).
#[allow(clippy::too_many_arguments)]
fn forced_probs_of(
    ctx: &Ctx,
    seat: usize,
    hs: &[[u8; 2]],
    board: &[u8],
    actions: &[RealAct],
    seed: u64,
    force: Force,
    act: RealAct,
    failed: &mut ModelFailed,
) -> Result<Vec<f64>, String> {
    if force == Force::default() {
        return probs_of(ctx, seat, hs, board, actions, seed, act, failed);
    }
    let Some(&first) = hs.first() else { return Ok(Vec::new()) };
    let os = ctx.agent.options_many_forced(&agent_spec(ctx, seat, first, board, actions, seed), hs, force)?;
    Ok(os.iter().map(|o| o.iter().filter(|(a, _)| *a == act).map(|(_, p)| p).sum()).collect())
}

/// Each holding's probability of folding to LBR's bet (the last of `actions`)
/// on the path `force`, averaged over the agent's readings of that bet.
#[allow(clippy::too_many_arguments)]
fn unseen_fold_probs(
    ctx: &Ctx,
    seat: usize,
    hs: &[[u8; 2]],
    board: &[u8],
    actions: &[RealAct],
    seed: u64,
    force: Force,
    failed: &mut ModelFailed,
) -> Result<Vec<f64>, String> {
    let Some(&first) = hs.first() else { return Ok(Vec::new()) };
    let Some(pl) = ctx.agent.translation_split(&agent_spec(ctx, seat, first, board, actions, seed), force)? else {
        return forced_probs_of(ctx, seat, hs, board, actions, seed, force, RealAct::Fold, failed);
    };
    let (lo, hi) = read_as(force, actions.len() - 1)?;
    let lo = forced_probs_of(ctx, seat, hs, board, actions, seed, lo, RealAct::Fold, failed)?;
    let hi = forced_probs_of(ctx, seat, hs, board, actions, seed, hi, RealAct::Fold, failed)?;
    Ok(lo.iter().zip(&hi).map(|(a, b)| pl * a + (1.0 - pl) * b).collect())
}

/// After LBR's bet (the last of `actions`): each path on which the agent
/// reads it between two grid sizes becomes two, one per reading, weighted
/// by the reading's probability.
fn split_paths(ctx: &Ctx, paths: &mut Paths, seat: usize, range: &[([u8; 2], f64)], board: &[u8], actions: &[RealAct], seed: u64) -> Result<(), String> {
    let mut out = Vec::with_capacity(paths.len());
    for (force, w) in paths.drain(..) {
        let Some(&first) = live_in(range, &w).first() else { continue };
        match ctx.agent.translation_split(&agent_spec(ctx, seat, first, board, actions, seed), force)? {
            None => out.push((force, w)),
            Some(pl) => {
                let (lo, hi) = read_as(force, actions.len() - 1)?;
                if pl > 0.0 {
                    out.push((lo, w.iter().map(|x| x * pl).collect()));
                }
                if pl < 1.0 {
                    out.push((hi, w.iter().map(|x| x * (1.0 - pl)).collect()));
                }
            }
        }
    }
    *paths = out;
    #[cfg(test)]
    {
        let mut probe = ctx.probe.lock().unwrap();
        probe.1 = probe.1.max(paths.len());
    }
    Ok(())
}

/// LBR's action at its decision, and its equity against the agent's range
/// (NaN preflop, where it does not compute one). `paths`: the belief without
/// the peek (`range` is then its marginal).
#[allow(clippy::too_many_arguments)]
fn lbr_action(
    ctx: &Ctx,
    s: &NlheState,
    me: [u8; 2],
    agent_seat: usize,
    board: &[u8],
    actions: &[RealAct],
    range: &[([u8; 2], f64)],
    paths: Option<&Paths>,
    agent_seed: u64,
    rng: &mut Rng,
    failed: &mut ModelFailed,
) -> Result<(RealAct, f64), String> {
    let c = Nlhe::owed(s);
    let check_call = if c == 0 { RealAct::Check } else { RealAct::Call };
    if Nlhe::betting_street(s) == Some(0) {
        return Ok((check_call, f64::NAN));
    }
    let pot = Nlhe::pot(s) as f64;
    let stack = Nlhe::stack_of(s, Nlhe::acting(s));
    let total: f64 = range.iter().map(|r| r.1).sum();
    if total <= 0.0 {
        return Ok((check_call, f64::NAN));
    }
    let eq = equities(ctx, me, board, range, rng);
    let wp = range.iter().zip(&eq).map(|(r, e)| r.1 * e).sum::<f64>() / total;
    let pay = c.min(stack) as f64;
    let mut best = (RealAct::Check, f64::NEG_INFINITY);
    let mut consider = |a: RealAct, u: f64| {
        #[cfg(test)]
        ctx.probe.lock().unwrap().0.push(u.to_bits());
        if u > best.1 {
            best = (a, u);
        }
    };
    consider(check_call, wp * (pot + pay) - pay);
    if c > 0 {
        consider(RealAct::Fold, 0.0);
    }
    // Candidate bet sizes (chips added), legal, deduplicated, plus all-in.
    let min_r = ctx.game.min_raise_add(s);
    let mut adds: Vec<i64> = match ctx.sizes {
        Sizes::Fixed => BET_FRACS
            .iter()
            .map(|f| {
                let raw = if c == 0 {
                    (f * pot).round() as i64
                } else {
                    c + (f * (pot + c as f64)).round() as i64
                };
                raw.max(min_r)
            })
            .filter(|&d| d < stack)
            .collect(),
        Sizes::Grid => ctx.game.grid_fracs(s).iter().map(|&f| ctx.game.grid_delta(s, f)).filter(|&d| d < stack).collect(),
    };
    if stack > c {
        adds.push(stack);
    }
    adds.sort_unstable();
    adds.dedup();
    let mut next_actions = actions.to_vec();
    let hs = live(range);
    let path_hs: Vec<Vec<[u8; 2]>> = paths.map_or_else(Vec::new, |ps| ps.iter().map(|(_, w)| live_in(range, w)).collect());
    for d in adds {
        next_actions.push(RealAct::Add(d));
        let (mut fold_w, mut cont_w, mut cont_eq) = (0.0, 0.0, 0.0);
        match paths {
            None => {
                let mut fps = probs_of(ctx, agent_seat, &hs, board, &next_actions, agent_seed, RealAct::Fold, failed)?.into_iter();
                for ((_, w), e) in range.iter().zip(&eq) {
                    if *w <= 0.0 {
                        continue;
                    }
                    let fp = fps.next().expect("one probability per live holding");
                    fold_w += w * fp;
                    cont_w += w * (1.0 - fp);
                    cont_eq += w * (1.0 - fp) * e;
                }
            }
            Some(ps) => {
                for ((force, pw), phs) in ps.iter().zip(&path_hs) {
                    let mut fps = unseen_fold_probs(ctx, agent_seat, phs, board, &next_actions, agent_seed, *force, failed)?.into_iter();
                    for (w, e) in pw.iter().zip(&eq) {
                        if *w <= 0.0 {
                            continue;
                        }
                        let fp = fps.next().expect("one probability per live holding");
                        fold_w += w * fp;
                        cont_w += w * (1.0 - fp);
                        cont_eq += w * (1.0 - fp) * e;
                    }
                }
            }
        }
        next_actions.pop();
        let fp = fold_w / total;
        let wpc = if cont_w > 0.0 { cont_eq / cont_w } else { 0.0 };
        let df = d as f64;
        // What the agent can still put in caps the called amount.
        let agent_stack = Nlhe::stack_of(s, agent_seat) as f64;
        // The agent calls at most its stack; LBR's unmatched excess is refunded.
        let called = (df - c as f64).min(agent_stack);
        let paid = c as f64 + called;
        let u = fp * pot + (1.0 - fp) * (wpc * (pot + paid + called) - paid);
        consider(RealAct::Add(d), u);
    }
    Ok((best.0, wp))
}

/// LBR-open's first action in the small blind at the root `s`, for LBR's
/// hand `me` (`seed`: LBR's hand seed).
fn open_action(ctx: &Ctx, open: Open, s: &NlheState, me: [u8; 2], seed: u64) -> Result<RealAct, String> {
    let raise = RealAct::Add(open.to - ctx.sb);
    let Some(opener) = open.opener else { return Ok(raise) };
    let spec = HandSpec { hero: 0, hole: me, board: Vec::new(), sb: ctx.sb, bb: ctx.bb, stacks: [ctx.stack, ctx.stack], actions: Vec::new(), seed: 0 };
    let d = opener.decide_inner(&spec)?;
    if d.source != Source::Blueprint {
        return Err(format!("the opener's root decision for {me:?} is not its blueprint's ({:?})", d.source));
    }
    let u = (splitmix(seed ^ OPEN_STREAM) >> 11) as f64 / (1u64 << 53) as f64;
    let mut acc = 0.0;
    let mut chosen = d.options.last().ok_or("the opener has no root options")?.0;
    for &(a, p) in &d.options {
        acc += p;
        if u < acc {
            chosen = a;
            break;
        }
    }
    let stack = Nlhe::stack_of(s, 0);
    match chosen {
        RealAct::Add(x) if x < stack => {
            let grid = ctx.game.grid_delta(s, 1.0);
            if x != grid {
                return Err(format!("the opener's root raise adds {x}, not the pot-size open's {grid}"));
            }
            Ok(raise)
        }
        other => Ok(other),
    }
}

/// One hand of LBR (in `lbr_seat`) vs the agent. Returns LBR's net chips.
/// With `trace`, every action is also recorded; recording never changes play.
fn play_hand(ctx: &Ctx, lbr_seat: usize, cards: &[u8], seed: u64, trace: Option<&mut Vec<Step>>) -> Result<f64, String> {
    play_hand_split(ctx, lbr_seat, cards, seed, trace).map(|h| h.net)
}

/// A hand of `play_hand_split`.
#[derive(Clone, Copy, Debug)]
struct Hand {
    /// LBR's net chips.
    net: f64,
    /// The betting ended before the flop (a preflop fold, or all-in before it).
    preflop: bool,
    /// The agent's decisions a failed search reached, as `[built-in solver
    /// used in place of the external one, blueprint used]`
    /// (`subgame::search_failure`).
    failed: [u64; 2],
    /// The same for LBR's model of the agent (`ModelFailed`).
    model_failed: ModelFailed,
    /// LBR-open: `[LBR raised as its first action, the agent folded to it,
    /// the agent raised it]`.
    open: [u64; 3],
}

/// `play_hand` with what `Hand` records.
fn play_hand_split(ctx: &Ctx, lbr_seat: usize, cards: &[u8], seed: u64, mut trace: Option<&mut Vec<Step>>) -> Result<Hand, String> {
    let g = &ctx.game;
    let agent_seat = 1 - lbr_seat;
    let agent_seed = splitmix(seed ^ 0xA5A5);
    // The seed LBR's model of the agent is given.
    #[cfg(not(test))]
    let model_seed = agent_seed;
    #[cfg(test)]
    let model_seed = agent_seed ^ ctx.model_seed_xor;
    let mut rng = Rng(seed);
    let mut s = g.initial_state();
    for &c in &cards[..4] {
        s = g.deal_specific(&s, Card::from_index(c));
    }
    let me = [cards[2 * lbr_seat], cards[2 * lbr_seat + 1]];
    let agent_hole = [cards[2 * agent_seat], cards[2 * agent_seat + 1]];
    // Agent range: every holding not colliding with LBR's cards (board later).
    let mut range: Vec<([u8; 2], f64)> = Vec::new();
    for (i, &a) in ctx.deck.iter().enumerate() {
        for &b in &ctx.deck[i + 1..] {
            if !me.contains(&a) && !me.contains(&b) {
                range.push(([a, b], 1.0));
            }
        }
    }
    let mut paths: Option<Paths> = (!ctx.peek).then(|| vec![(Force::default(), range.iter().map(|r| r.1).collect())]);
    let mut board: Vec<u8> = Vec::new();
    let mut actions: Vec<RealAct> = Vec::new();
    // Board cards out at the last action.
    let mut last_board = 0;
    let mut search_failed = [0u64; 2];
    let mut model_failed = ModelFailed::default();
    let mut open_counts = [0u64; 3];
    let opening = ctx.open.filter(|_| lbr_seat == 0);
    loop {
        while Nlhe::is_dealing(&s) {
            let c = cards[4 + board.len()];
            board.push(c);
            s = g.deal_specific(&s, Card::from_index(c));
            for r in range.iter_mut() {
                if r.0.contains(&c) {
                    r.1 = 0.0;
                }
            }
            for (_, w) in paths.iter_mut().flatten() {
                for (x, r) in w.iter_mut().zip(&range) {
                    if r.0.contains(&c) {
                        *x = 0.0;
                    }
                }
            }
        }
        if g.is_terminal(&s) {
            break;
        }
        let seat = Nlhe::acting(&s);
        let (pot, owed) = (Nlhe::pot(&s), Nlhe::owed(&s));
        let act = if seat == agent_seat {
            let spec = agent_spec(ctx, agent_seat, agent_hole, &board, &actions, agent_seed);
            let d = ctx.agent.decide_inner(&spec)?;
            if let Some(k) = search_failure(ctx.agent, &spec, &d) {
                search_failed[k] += 1;
            }
            if let Some(t) = trace.as_deref_mut() {
                let diag = if ctx.board_cards == 5 && board.len() >= 4 {
                    Some(diagnose(ctx, &spec, me, &range)?)
                } else {
                    None
                };
                t.push(Step {
                    board_len: board.len(),
                    lbr: false,
                    act: d.action,
                    pot,
                    owed,
                    lbr_equity: f64::NAN,
                    source: Some(d.source),
                    options: d.options.clone(),
                    diag,
                });
            }
            // Bayes: weight each holding by its probability of this action.
            match paths.as_mut() {
                None => {
                    let mut ps = probs_of(ctx, agent_seat, &live(&range), &board, &actions, model_seed, d.action, &mut model_failed)?.into_iter();
                    for r in range.iter_mut() {
                        if r.1 > 0.0 {
                            r.1 *= ps.next().expect("one probability per live holding");
                        }
                    }
                }
                Some(paths) => {
                    for (force, w) in paths.iter_mut() {
                        let hs = live_in(&range, w);
                        let mut ps = forced_probs_of(ctx, agent_seat, &hs, &board, &actions, model_seed, *force, d.action, &mut model_failed)?.into_iter();
                        for x in w.iter_mut() {
                            if *x > 0.0 {
                                *x *= ps.next().expect("one probability per live holding");
                            }
                        }
                    }
                    paths.retain(|(_, w)| w.iter().any(|&x| x > 0.0));
                    marginal(paths, &mut range);
                }
            }
            if range.iter().all(|r| r.1 <= 0.0) {
                return Err("agent range vanished (policy inconsistent with its own action)".into());
            }
            if let (Some(o), [RealAct::Add(x)]) = (opening, actions.as_slice()) {
                if *x == o.to - ctx.sb {
                    open_counts[1] += u64::from(d.action == RealAct::Fold);
                    open_counts[2] += u64::from(matches!(d.action, RealAct::Add(_)));
                }
            }
            d.action
        } else {
            let (a, wp) = match opening {
                Some(o) if actions.is_empty() => {
                    let a = open_action(ctx, o, &s, me, seed)?;
                    open_counts[0] += u64::from(a == RealAct::Add(o.to - ctx.sb));
                    (a, f64::NAN)
                }
                _ => lbr_action(ctx, &s, me, agent_seat, &board, &actions, &range, paths.as_ref(), model_seed, &mut rng, &mut model_failed)?,
            };
            if let Some(t) = trace.as_deref_mut() {
                t.push(Step {
                    board_len: board.len(),
                    lbr: true,
                    act: a,
                    pot,
                    owed,
                    lbr_equity: wp,
                    source: None,
                    options: Vec::new(),
                    diag: None,
                });
            }
            a
        };
        s = g.apply_real(&s, act).map_err(|e| format!("illegal {act:?}: {e}"))?;
        actions.push(act);
        last_board = board.len();
        if let (Some(paths), true, RealAct::Add(_)) = (paths.as_mut(), seat != agent_seat, act) {
            split_paths(ctx, paths, agent_seat, &range, &board, &actions, model_seed)?;
            marginal(paths, &mut range);
        }
    }
    let (n0, n1) = g.payoffs(&s);
    Ok(Hand { net: if lbr_seat == 0 { n0 } else { n1 }, preflop: last_board == 0, failed: search_failed, model_failed, open: open_counts })
}

/// One deal of `run_lbr`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LbrDeal {
    /// LBR's net in big blinds.
    pub net: f64,
    /// The part of `net` from hands whose betting ended before the flop.
    pub preflop: f64,
    /// `Hand::failed`, summed over the deal's hands.
    pub failed: [u64; 2],
    /// `Hand::model_failed`, summed over the deal's hands.
    pub model_failed: [u64; 2],
    /// `Hand::open`, summed over the deal's hands.
    pub open: [u64; 3],
}

/// LBR's winnings against the agent: per deal (both seats per deal when
/// `mirror`, else LBR in `lbr_seat`), as `LbrDeal`. Preflop LBR only checks
/// or calls, so the preflop part is not exploitation (e.g. calling a preflop
/// shove with any two cards); the rest measures postflop play.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_lbr(
    agent: &Arc<Agent>,
    sb: i64,
    bb: i64,
    stack: i64,
    first: u64,
    deals: u64,
    mirror: bool,
    lbr_seat: usize,
    seed: u64,
    sizes: Sizes,
    peek: bool,
    open: Option<Open>,
) -> Result<Vec<LbrDeal>, String> {
    if let Some(o) = open {
        if o.to < 2 * bb || o.to >= stack {
            return Err(format!("LBR-open raises to {}: need 2 x bb <= to < stack", o.to));
        }
    }
    let ctx = Ctx { sizes, peek, open, ..make_ctx(agent, sb, bb, stack) };
    let preflop = |h: &Hand| if h.preflop { h.net } else { 0.0 };
    let add = |x: [u64; 2], y: [u64; 2]| [x[0] + y[0], x[1] + y[1]];
    // Deal `i` depends only on `(seed, i)`: chunks of a longer run are identical.
    (first..first + deals)
        .into_par_iter()
        .map(|i| {
            let (cards, hand_seed) = deal(&ctx, seed, i);
            // Net and preflop part in chips.
            let (net, pre, failed, model_failed, open) = if mirror {
                let a = play_hand_split(&ctx, 0, &cards, hand_seed, None)?;
                let b = play_hand_split(&ctx, 1, &cards, hand_seed, None)?;
                let open = [a.open[0] + b.open[0], a.open[1] + b.open[1], a.open[2] + b.open[2]];
                (a.net + b.net, preflop(&a) + preflop(&b), add(a.failed, b.failed), add(a.model_failed, b.model_failed), open)
            } else {
                let a = play_hand_split(&ctx, lbr_seat, &cards, hand_seed, None)?;
                (a.net, preflop(&a), a.failed, a.model_failed, a.open)
            };
            Ok(LbrDeal { net: net / bb as f64, preflop: pre / bb as f64, failed, model_failed, open })
        })
        .collect()
}

fn make_ctx(agent: &Agent, sb: i64, bb: i64, stack: i64) -> Ctx<'_> {
    let base = agent.cfg();
    Ctx {
        agent,
        game: Nlhe::new(crate::nlhe::NlheConfig { sb, bb, stack, ..base.clone() }),
        board_cards: base.board_cards,
        deck: base.deck.clone(),
        sb,
        bb,
        stack,
        sizes: Sizes::Fixed,
        peek: true,
        open: None,
        #[cfg(test)]
        per_holding: false,
        #[cfg(test)]
        model_seed_xor: 0,
        #[cfg(test)]
        probe: Default::default(),
        #[cfg(test)]
        search_queries: Default::default(),
    }
}

/// Deal `i` of a run seeded `seed`: both hole cards then the board, and the
/// hand's seed.
fn deal(ctx: &Ctx, seed: u64, i: u64) -> (Vec<u8>, u64) {
    let need = 4 + ctx.board_cards;
    let mut rng = Rng(splitmix(seed ^ splitmix(i)));
    let mut d = ctx.deck.clone();
    for k in 0..need {
        let j = k + rng.below((d.len() - k) as u64) as usize;
        d.swap(k, j);
    }
    d.truncate(need);
    (d, rng.next())
}

/// One action of a traced hand.
pub(crate) struct Step {
    /// Board cards dealt when the action was taken.
    pub board_len: usize,
    /// Taken by LBR (else by the agent).
    pub lbr: bool,
    pub act: RealAct,
    /// Chips in the pot and owed by the actor, before the action.
    pub pot: i64,
    pub owed: i64,
    /// LBR's equity against the agent's range (LBR actions after preflop).
    pub lbr_equity: f64,
    /// Agent actions: where the action came from and the full distribution.
    pub source: Option<Source>,
    pub options: Vec<(RealAct, f64)>,
    /// Agent turn/river decisions: how the search's range model saw the spot.
    pub diag: Option<Diag>,
}

/// The search model's view of an agent decision, next to the truth.
pub(crate) struct Diag {
    /// Total-variation distance between the agent's own range as its
    /// decision sees it (`subgame::beliefs`: the solve's where it searches,
    /// the model's elsewhere) and its true range (LBR's exact tracking with
    /// the agent's real action probabilities), both over the holdings LBR's
    /// cards leave possible.
    pub own_tv: f64,
    /// The agent's equity (river: exact; turn: over every river card) against
    /// the opponent range its decision assumes, and against LBR's actual hand.
    pub eq_model: f64,
    pub eq_actual: f64,
    /// exp(entropy) of the modelled opponent range and of the agent's true range.
    pub opp_eff: f64,
    pub own_eff: f64,
}

/// A replayed LBR hand.
pub(crate) struct TracedHand {
    pub deal: u64,
    pub lbr_seat: usize,
    /// Seat 0's hole cards, seat 1's, then the five board cards.
    pub cards: Vec<u8>,
    pub steps: Vec<Step>,
    /// LBR's net, in chips.
    pub net: f64,
}

/// Replay deals of `run_lbr` (same `seed`, same deal indices) with every
/// action recorded: both seats per deal. Results equal `run_lbr`'s exactly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn trace_lbr(
    agent: &Arc<Agent>,
    sb: i64,
    bb: i64,
    stack: i64,
    deals: &[u64],
    seed: u64,
    sizes: Sizes,
    peek: bool,
) -> Result<Vec<TracedHand>, String> {
    let ctx = Ctx { sizes, peek, ..make_ctx(agent, sb, bb, stack) };
    let per_deal = deals
        .par_iter()
        .map(|&i| {
            let (cards, hand_seed) = deal(&ctx, seed, i);
            (0..2)
                .map(|lbr_seat| {
                    let mut steps = Vec::new();
                    let net = play_hand(&ctx, lbr_seat, &cards, hand_seed, Some(&mut steps))?;
                    Ok(TracedHand { deal: i, lbr_seat, cards: cards.clone(), steps, net })
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(per_deal.into_iter().flatten().collect())
}

fn entropy_size(w: &[f64]) -> f64 {
    let total: f64 = w.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    (-w.iter().filter(|&&x| x > 0.0).map(|&x| x / total * (x / total).ln()).sum::<f64>()).exp()
}

/// The agent's equity against weighted opponent holdings on a turn or river
/// board (every river card when one is to come).
fn equity_against(me: [u8; 2], board: &[u8], deck: &[u8], opp: &[([u8; 2], f64)]) -> f64 {
    let used = |c: u8| me.contains(&c) || board.contains(&c);
    let rivers: Vec<u8> = if board.len() == 5 { vec![u8::MAX] } else { deck.iter().copied().filter(|&c| !used(c)).collect() };
    let mut full = board.to_vec();
    let (mut won, mut total) = (0.0, 0.0);
    for &(h, w) in opp {
        if w <= 0.0 || used(h[0]) || used(h[1]) {
            continue;
        }
        for &r in &rivers {
            if r != u8::MAX && h.contains(&r) {
                continue;
            }
            full.truncate(board.len());
            if r != u8::MAX {
                full.push(r);
            }
            let (a, b) = (score(me, &full), score(h, &full));
            won += w * if a > b { 1.0 } else if a == b { 0.5 } else { 0.0 };
            total += w;
        }
    }
    if total > 0.0 { won / total } else { f64::NAN }
}

/// How the agent's decision at `spec` sees the ranges (`subgame::beliefs`),
/// given LBR's cards and the agent's true range.
fn diagnose(ctx: &Ctx, spec: &HandSpec, lbr_hole: [u8; 2], true_range: &[([u8; 2], f64)]) -> Result<Diag, String> {
    let board = &spec.board;
    let live: Vec<u8> = ctx.deck.iter().copied().filter(|c| !board.contains(c)).collect();
    let mut combos = Vec::with_capacity(live.len() * live.len() / 2);
    for (i, &a) in live.iter().enumerate() {
        for &b in &live[i + 1..] {
            combos.push(if a < b { [a, b] } else { [b, a] });
        }
    }
    let modelled = beliefs(ctx.agent, spec, &combos)?;
    let (me, opp) = (spec.hero, 1 - spec.hero);
    // Own range: modelled vs true, over holdings LBR's cards leave possible.
    let mut truth = vec![0.0; 52 * 52];
    for &(h, w) in true_range {
        let (a, b) = if h[0] < h[1] { (h[0], h[1]) } else { (h[1], h[0]) };
        truth[a as usize * 52 + b as usize] = w;
    }
    let (mut model_own, mut true_own) = (Vec::new(), Vec::new());
    for (k, h) in combos.iter().enumerate() {
        if !lbr_hole.contains(&h[0]) && !lbr_hole.contains(&h[1]) {
            model_own.push(modelled[me][k]);
            true_own.push(truth[h[0] as usize * 52 + h[1] as usize]);
        }
    }
    let (sm, st): (f64, f64) = (model_own.iter().sum(), true_own.iter().sum());
    let own_tv = if sm > 0.0 && st > 0.0 {
        0.5 * model_own.iter().zip(&true_own).map(|(m, t)| (m / sm - t / st).abs()).sum::<f64>()
    } else {
        f64::NAN
    };
    let opp_range: Vec<([u8; 2], f64)> = combos
        .iter()
        .zip(&modelled[opp])
        .filter(|(h, _)| !spec.hole.contains(&h[0]) && !spec.hole.contains(&h[1]))
        .map(|(h, &w)| (*h, w))
        .collect();
    let opp_w: Vec<f64> = opp_range.iter().map(|r| r.1).collect();
    Ok(Diag {
        own_tv,
        eq_model: equity_against(spec.hole, board, &ctx.deck, &opp_range),
        eq_actual: equity_against(spec.hole, board, &ctx.deck, &[(lbr_hole, 1.0)]),
        opp_eff: entropy_size(&opp_w),
        own_eff: entropy_size(&true_own),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nlhe::NlheConfig;
    use crate::play::Strategy;
    use crate::s32::{LeakFamily, Policy};
    use crate::subgame::SearchCfg;

    /// With the range updates batched (`probs_of`), every hand plays out
    /// exactly as with one `decide_inner` per holding, and the model's failed
    /// searches count the same: exactly one per query of the agent's policy
    /// at a public state where it searches. With the peek and without it (on
    /// the agent's grid, so no reading is forced).
    #[test]
    fn batched_range_updates_play_every_hand_as_before() {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let cfg = |external: Option<crate::subgame::External>, rescue: bool| SearchCfg { from_flop: false, flop_iters: 1, flop_checkdown: false, from_turn: false, river_iters: 5, turn_iters: 5, exact_own_range: true, solve_round: true, external, rescue };
        let no_gpu = || Some(crate::subgame::External(Arc::new(NoGpu)));
        // A calling station reaches the river, where the search solves.
        let station = || Agent::from_policy(Arc::clone(&game), Policy::uniform().leak(LeakFamily::Station, 0.8));
        let agents = [
            ("station + search", station().with_search(cfg(None, true)), None),
            ("station + failed search", station().with_search(cfg(no_gpu(), false)), Some(1)),
            ("station + rescued search", station().with_search(cfg(no_gpu(), true)), Some(0)),
            // Every decision falls back: hand-dependent at every node.
            ("empty table", Agent::new(Arc::clone(&game), Arc::new(Strategy::default())), None),
        ];
        use std::sync::atomic::Ordering::Relaxed;
        for (name, agent, fails) in &agents {
            for (peek, sizes) in [(true, Sizes::Fixed), (false, Sizes::Grid)] {
                let mut ctx = Ctx { peek, sizes, ..make_ctx(agent, 50, 100, 10_000) };
                let (mut rivers, mut queries) = (0u64, 0u64);
                let parts = |h: Hand| (h.net.to_bits(), h.preflop, h.failed, h.model_failed);
                for i in 0..8 {
                    let (cards, seed) = deal(&ctx, 11, i);
                    for seat in 0..2 {
                        ctx.per_holding = true;
                        let want = play_hand_split(&ctx, seat, &cards, seed, None).map(parts);
                        ctx.per_holding = false;
                        let before = ctx.search_queries.load(Relaxed);
                        let mut steps = Vec::new();
                        let got = play_hand_split(&ctx, seat, &cards, seed, Some(&mut steps)).map(parts);
                        let q = ctx.search_queries.load(Relaxed) - before;
                        assert_eq!(got, want, "{name}, peek {peek}: deal {i} seat {seat}");
                        rivers += u64::from(steps.iter().any(|s| s.board_len == 5 && !s.lbr));
                        queries += q;
                        let exact = match fails {
                            None => [0, 0],
                            Some(0) => [q, 0],
                            Some(_) => [0, q],
                        };
                        assert_eq!(got.unwrap().3, exact, "{name}, peek {peek}: deal {i} seat {seat}, {q} queries where it searches");
                    }
                }
                assert!(rivers >= 6, "{name}, peek {peek}: only {rivers} hands reached an agent river decision");
                // The model also reads the agent's answers to the bets LBR weighs.
                if agent.search_cfg().is_some() {
                    assert!(queries > rivers, "{name}, peek {peek}: {queries} queries, {rivers} river hands");
                } else {
                    assert_eq!(queries, 0, "{name}, peek {peek}");
                }
            }
        }
    }

    /// `equities` with LBR's score computed once per runout gives exactly
    /// what scoring both hands for every (holding, runout) pair gave.
    #[test]
    fn equities_match_scoring_every_pair() {
        fn per_pair(ctx: &Ctx, me: [u8; 2], board: &[u8], range: &[([u8; 2], f64)], rng: &mut Rng) -> Vec<f64> {
            let need = ctx.board_cards - board.len();
            let dead = me.iter().chain(board).fold(0u64, |m, &c| m | 1u64 << c);
            let live: Vec<u8> = ctx.deck.iter().copied().filter(|&c| dead & (1u64 << c) == 0).collect();
            let mut runouts: Vec<Vec<u8>> = Vec::new();
            if need == 0 {
                runouts.push(Vec::new());
            } else if need == 1 {
                runouts.extend(live.iter().map(|&c| vec![c]));
            } else {
                for _ in 0..ROLLOUTS {
                    let mut pick = live.clone();
                    for k in 0..need {
                        let j = k + rng.below((pick.len() - k) as u64) as usize;
                        pick.swap(k, j);
                    }
                    pick.truncate(need);
                    runouts.push(pick);
                }
            }
            let mut full = board.to_vec();
            range
                .iter()
                .map(|&(h, w)| {
                    if w <= 0.0 {
                        return 0.0;
                    }
                    let hm = (1u64 << h[0]) | (1u64 << h[1]);
                    let (mut won, mut n) = (0.0, 0.0);
                    for r in &runouts {
                        if r.iter().any(|&c| hm & (1u64 << c) != 0) {
                            continue;
                        }
                        full.truncate(board.len());
                        full.extend_from_slice(r);
                        let (a, b) = (score(me, &full), score(h, &full));
                        won += if a > b { 1.0 } else if a == b { 0.5 } else { 0.0 };
                        n += 1.0;
                    }
                    if n > 0.0 { won / n } else { 0.5 }
                })
                .collect()
        }
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let agent = Agent::new(Arc::clone(&game), Arc::new(Strategy::default()));
        let ctx = make_ctx(&agent, 50, 100, 10_000);
        for i in 0..6 {
            let (cards, seed) = deal(&ctx, 5, i);
            let me = [cards[0], cards[1]];
            for len in [3, 4, 5] {
                let board = &cards[4..4 + len];
                let used = |c: u8| me.contains(&c) || board.contains(&c);
                let mut range = Vec::new();
                for (x, &a) in ctx.deck.iter().enumerate() {
                    for &b in &ctx.deck[x + 1..] {
                        if !used(a) && !used(b) {
                            // Some holdings out of the range, the rest weighted.
                            let k = range.len() as u64;
                            range.push(([a, b], if splitmix(seed ^ k) % 5 == 0 { 0.0 } else { (k % 9 + 1) as f64 / 9.0 }));
                        }
                    }
                }
                let got = equities(&ctx, me, board, &range, &mut Rng(seed));
                let want = per_pair(&ctx, me, board, &range, &mut Rng(seed));
                assert_eq!(got.iter().map(|e| e.to_bits()).collect::<Vec<_>>(), want.iter().map(|e| e.to_bits()).collect::<Vec<_>>(), "deal {i}, {len} board cards");
            }
        }
    }

    /// With `Sizes::Grid` every LBR bet after the flop is one of the agent's
    /// grid amounts at that node (or all-in), so the agent never translates it;
    /// the standard sizes do bet off the grid.
    #[test]
    fn grid_sizes_keep_every_lbr_bet_on_the_agents_grid() {
        let game = Arc::new(Nlhe::new(NlheConfig::grid("200bb").unwrap()));
        // Every decision falls back to pot odds: its folds depend on the bet size.
        let agent = Arc::new(Agent::new(Arc::clone(&game), Arc::new(Strategy::default())));
        let ctx = make_ctx(&agent, 50, 100, 20_000);
        let deals: Vec<u64> = (0..60).collect();
        for (sizes, want_off) in [(Sizes::Grid, false), (Sizes::Fixed, true)] {
            let hands = trace_lbr(&agent, 50, 100, 20_000, &deals, 9, sizes, true).unwrap();
            let (mut bets, mut off) = (0, 0);
            for h in &hands {
                let mut s = ctx.game.initial_state();
                for &c in &h.cards[..4] {
                    s = ctx.game.deal_specific(&s, Card::from_index(c));
                }
                let mut dealt = 4;
                for st in &h.steps {
                    while Nlhe::is_dealing(&s) {
                        s = ctx.game.deal_specific(&s, Card::from_index(h.cards[dealt]));
                        dealt += 1;
                    }
                    if let (true, RealAct::Add(d)) = (st.lbr && st.board_len > 0, st.act) {
                        let stack = Nlhe::stack_of(&s, Nlhe::acting(&s));
                        let on = d == stack || ctx.game.grid_fracs(&s).iter().any(|&f| ctx.game.grid_delta(&s, f) == d);
                        bets += 1;
                        off += usize::from(!on);
                    }
                    s = ctx.game.apply_real(&s, st.act).unwrap();
                }
            }
            assert!(bets > 20, "{sizes:?}: only {bets} LBR bets");
            assert_eq!(off > 0, want_off, "{sizes:?}: {off} of {bets} LBR bets off the grid");
        }
    }

    /// Without the peek LBR's model never reads the agent's translation seed:
    /// handing it a wrong seed changes none of its action values (with the
    /// peek it does), and on the agent's own grid, where nothing is
    /// translated, it is the standard LBR.
    #[test]
    fn peek_free_lbr_never_reads_the_translation_seed() {
        // Grid sizes between LBR's (0.5, 1, 2 pot), so most of its bets are read
        // at random. Uniform rows over the grid's actions: with short stacks
        // the two readings leave different raises open, so the agent's fold
        // probability depends on the reading.
        let cfg = NlheConfig { stack: 3_000, open_fracs: vec![0.25, 0.75, 3.0], raise_fracs: vec![0.25, 1.5], ..NlheConfig::grid("200bb").unwrap() };
        let game = Arc::new(Nlhe::new(cfg));
        let agent = Agent::from_policy(Arc::clone(&game), Policy::uniform());
        let mut ctx = make_ctx(&agent, 50, 100, 3_000);
        let (mut peek_reads, mut changed, mut split, mut hands) = (0, 0, 0, 0);
        for i in 0..60 {
            let (cards, seed) = deal(&ctx, 5, i);
            for seat in 0..2 {
                // LBR's net and every action value it computed, and its most paths.
                let mut play = |sizes, peek, xor| {
                    ctx.sizes = sizes;
                    ctx.peek = peek;
                    ctx.model_seed_xor = xor;
                    let net = play_hand(&ctx, seat, &cards, seed, None).map(f64::to_bits);
                    let (values, paths) = std::mem::take(&mut *ctx.probe.lock().unwrap());
                    (net, values, paths)
                };
                let blind = play(Sizes::Fixed, false, 0);
                assert!(blind.0.is_ok(), "deal {i} seat {seat}: {:?}", blind.0);
                assert_eq!(play(Sizes::Fixed, false, 0x5EED), blind, "deal {i} seat {seat}");
                let peek = play(Sizes::Fixed, true, 0);
                peek_reads += usize::from(play(Sizes::Fixed, true, 0x5EED) != peek);
                changed += usize::from(peek.1 != blind.1);
                split += usize::from(blind.2 > 1);
                let grid = play(Sizes::Grid, false, 0);
                assert!(grid.2 <= 1, "deal {i} seat {seat}: a grid bet was split");
                let grid_peek = play(Sizes::Grid, true, 0);
                assert_eq!((grid.0, grid.1), (grid_peek.0, grid_peek.1), "deal {i} seat {seat}");
                hands += 1;
            }
        }
        assert!(split >= 10, "LBR's bets were split in only {split} of {hands} hands");
        assert!(peek_reads >= 10, "a wrong seed changed the standard LBR in only {peek_reads} of {hands} hands");
        assert!(changed >= 10, "the peek mattered in only {changed} of {hands} hands");
    }

    /// `run_lbr`'s preflop part is exactly the net of the traced hands whose
    /// last action came before the flop, and its total is the traced total.
    #[test]
    fn preflop_part_is_the_hands_that_ended_before_the_flop() {
        // Uniform rows: the agent folds and shoves preflop often, and plays on.
        let game = Arc::new(Nlhe::new(NlheConfig::grid("200bb").unwrap()));
        let agent = Arc::new(Agent::from_policy(Arc::clone(&game), Policy::uniform()));
        let deals: Vec<u64> = (0..80).collect();
        let hands = trace_lbr(&agent, 50, 100, 20_000, &deals, 3, Sizes::Fixed, true).unwrap();
        let split = run_lbr(&agent, 50, 100, 20_000, 0, 80, true, 0, 3, Sizes::Fixed, true, None).unwrap();
        let (mut pre_deals, mut post_deals) = (0, 0);
        for (i, &LbrDeal { net: total, preflop: pre, failed, model_failed, .. }) in split.iter().enumerate() {
            assert_eq!((failed, model_failed), ([0, 0], [0, 0]), "deal {i}: an agent without search");
            let h: Vec<&TracedHand> = hands.iter().filter(|h| h.deal == i as u64).collect();
            assert_eq!(h.len(), 2);
            let ended = |t: &TracedHand| t.steps.last().unwrap().board_len == 0;
            let part = |t: &TracedHand| if ended(t) { t.net } else { 0.0 };
            assert_eq!(total.to_bits(), ((h[0].net + h[1].net) / 100.0).to_bits(), "deal {i}");
            assert_eq!(pre.to_bits(), ((part(h[0]) + part(h[1])) / 100.0).to_bits(), "deal {i}");
            pre_deals += usize::from(h.iter().any(|t| ended(t)));
            post_deals += usize::from(h.iter().any(|t| !ended(t)));
        }
        assert!(pre_deals >= 10 && post_deals >= 10, "{pre_deals} deals with a preflop-ended hand, {post_deals} with a postflop one");
    }

    /// A holding the agent cannot decide for stops LBR's model (an error), not
    /// a probability 0 of the action.
    #[test]
    fn a_holding_the_agent_cannot_decide_for_is_an_error() {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let agent = Agent::from_policy(Arc::clone(&game), Policy::uniform());
        let ctx = make_ctx(&agent, 50, 100, 10_000);
        let mut failed = ModelFailed::default();
        let mut probs = |hs: &[[u8; 2]]| probs_one_by_one(&ctx, 0, hs, &[], &[], 0, RealAct::Call, &mut failed).map(|p| p.len());
        assert_eq!(probs(&[[0, 1], [2, 3]]), Ok(2));
        assert!(probs(&[[0, 1], [3, 3]]).is_err(), "a duplicate card");
    }

    /// An external solver that always fails.
    struct NoGpu;

    impl crate::subgame::SubgameSolver for NoGpu {
        fn solve(&self, _: &crate::search::FlatSubgame) -> Result<Vec<Vec<f64>>, String> {
            Err("no GPU".into())
        }
    }

    /// LBR counts the agent's decisions a failed search reached, as a match
    /// does: without a rescue the blueprint decides there and LBR plays as
    /// against the agent without search; rescued, the built-in search decides,
    /// every time in place of the external one. Each such decision counts
    /// once; LBR's model reads the failure at least there (its range update)
    /// and at the agent's answers to the river bets it weighs.
    #[test]
    fn lbr_counts_the_decisions_a_failed_search_reached() {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        // A calling station reaches the river, where the search solves.
        let station = Agent::from_policy(Arc::clone(&game), Policy::uniform().leak(LeakFamily::Station, 0.8));
        let cfg = |external: Option<crate::subgame::External>, rescue: bool| SearchCfg { from_flop: false, flop_iters: 1, flop_checkdown: false, from_turn: false, river_iters: 5, turn_iters: 5, exact_own_range: true, solve_round: true, external, rescue };
        let no_gpu = || Some(crate::subgame::External(Arc::new(NoGpu)));
        let deals: Vec<u64> = (0..8).collect();
        let run = |a: Agent| {
            let a = Arc::new(a);
            let r = run_lbr(&a, 50, 100, 10_000, 0, 8, true, 0, 11, Sizes::Fixed, true, None).expect("LBR runs");
            let bits: Vec<(u64, u64)> = r.iter().map(|x| (x.net.to_bits(), x.preflop.to_bits())).collect();
            let sum = |f: fn(&LbrDeal) -> [u64; 2]| r.iter().fold([0u64; 2], |s, x| [s[0] + f(x)[0], s[1] + f(x)[1]]);
            // The agent's river decisions (where it searches), and those the search decided.
            let steps: Vec<Step> = trace_lbr(&a, 50, 100, 10_000, &deals, 11, Sizes::Fixed, true).unwrap().into_iter().flat_map(|h| h.steps).collect();
            let rivers = steps.iter().filter(|s| !s.lbr && s.board_len == 5).count() as u64;
            let searched = steps.iter().filter(|s| !s.lbr && s.source == Some(Source::Search)).count() as u64;
            (bits, sum(|x| x.failed), sum(|x| x.model_failed), rivers, searched)
        };
        let blueprint = run(station.with_oversize(station.oversize()));
        let built_in = run(station.with_search(cfg(None, true)));
        let failed = run(station.with_search(cfg(no_gpu(), false)));
        let rescued = run(station.with_search(cfg(no_gpu(), true)));
        for (name, r) in [("blueprint", &blueprint), ("built-in", &built_in)] {
            assert_eq!((r.1, r.2), ([0, 0], [0, 0]), "{name}");
        }
        assert_ne!(built_in.0, blueprint.0, "the search changed no hand");
        assert_eq!(failed.0, blueprint.0);
        assert!(failed.3 > 0 && failed.4 == 0, "{} river decisions, {} searched", failed.3, failed.4);
        assert_eq!(failed.1, [0, failed.3]);
        assert!(failed.2[0] == 0 && failed.2[1] > failed.3, "{:?}", failed.2);
        assert_eq!(rescued.0, built_in.0);
        assert!(rescued.3 > 0 && rescued.4 == rescued.3, "{} river decisions, {} searched", rescued.3, rescued.4);
        assert_eq!(rescued.1, [rescued.3, 0]);
        assert!(rescued.2[0] > rescued.3 && rescued.2[1] == 0, "{:?}", rescued.2);
        // A mirrored deal is its two hands: LBR in each seat on its own adds up to it.
        for a in [station.with_search(cfg(no_gpu(), false)), station.with_search(cfg(no_gpu(), true))] {
            let a = Arc::new(a);
            let lbr = |mirror, seat| run_lbr(&a, 50, 100, 10_000, 0, 8, mirror, seat, 11, Sizes::Fixed, true, None).expect("LBR runs");
            let (both, seats) = (lbr(true, 0), [lbr(false, 0), lbr(false, 1)]);
            let add = |x: [u64; 2], y: [u64; 2]| [x[0] + y[0], x[1] + y[1]];
            for (i, m) in both.iter().enumerate() {
                let (x, y) = (&seats[0][i], &seats[1][i]);
                assert_eq!((m.failed, m.model_failed), (add(x.failed, y.failed), add(x.model_failed, y.model_failed)), "deal {i}");
                assert!((m.net - (x.net + y.net)).abs() < 1e-9 && (m.preflop - (x.preflop + y.preflop)).abs() < 1e-9, "deal {i}");
            }
            assert!((0..2).all(|s| seats[s].iter().any(|d| d.failed != [0, 0])), "a seat's hands reached no failed search");
        }
    }

    /// LBR against the real blueprint, one thread, batched range updates
    /// against per-holding ones, interleaved (same deals, same results):
    ///   cargo test --release --no-default-features --lib -- --ignored --nocapture print_lbr_speed
    #[test]
    #[ignore]
    fn print_lbr_speed() {
        let Some(agent) = crate::subgame::tests::real_blueprint_agent() else { return };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| {
            let mut ctx = make_ctx(&agent, 50, 100, 10_000);
            let deals: Vec<(Vec<u8>, u64)> = (0..30).map(|i| deal(&ctx, 3, i)).collect();
            for round in 0..3 {
                for per_holding in [true, false] {
                    ctx.per_holding = per_holding;
                    let clock = std::time::Instant::now();
                    let mut net = 0.0;
                    for (cards, seed) in &deals {
                        for seat in 0..2 {
                            net += play_hand(&ctx, seat, cards, *seed, None).unwrap();
                        }
                    }
                    let ms = 1000.0 * clock.elapsed().as_secs_f64() / deals.len() as f64;
                    println!("round {round} {:11}: {ms:7.1} ms/deal (LBR net {net})", if per_holding { "per-holding" } else { "batched" });
                }
            }
        });
    }

    /// LBR-open (records/preflop-open): every hand raises to `to` as LBR's
    /// first small-blind action; the big blind's LBR is the standard one; at
    /// the grid's own open (3bb) F2 changes nothing, with the peek or without;
    /// the standard LBR (which never raises preflop) is the same with F2.
    #[test]
    fn lbr_open_raises_first_and_at_the_grid_open_f2_changes_nothing() {
        let game = Arc::new(Nlhe::new(NlheConfig::grid("200bb").unwrap()));
        let agent = Arc::new(Agent::from_policy(Arc::clone(&game), Policy::uniform()));
        let f2 = Arc::new(agent.with_open_below(crate::play::OpenBelow::Limp));
        let nets = |a: &Arc<Agent>, peek: bool, open: Option<Open>| -> Vec<(u64, [u64; 3])> {
            run_lbr(a, 50, 100, 20_000, 0, 24, true, 0, 9, Sizes::Fixed, peek, open).unwrap().iter().map(|d| (d.net.to_bits(), d.open)).collect()
        };
        for peek in [true, false] {
            let open3 = Some(Open { to: 300, opener: None });
            let snap = nets(&agent, peek, open3);
            assert_eq!(nets(&f2, peek, open3), snap, "peek {peek}");
            assert!(snap.iter().all(|(_, o)| o[0] == 1 && o[1] + o[2] <= 1), "peek {peek}: {snap:?}");
            let standard = nets(&agent, peek, None);
            assert_eq!(nets(&f2, peek, None), standard, "peek {peek}");
            assert!(standard.iter().all(|(_, o)| *o == [0, 0, 0]));
            // LBR's first action in the big blind is not replaced: its seat-1
            // hands are the standard ones.
            let seat1 = |a: &Arc<Agent>, open| run_lbr(a, 50, 100, 20_000, 0, 24, false, 1, 9, Sizes::Fixed, peek, open).unwrap().iter().map(|d| d.net.to_bits()).collect::<Vec<_>>();
            assert_eq!(seat1(&agent, open3), seat1(&agent, None), "peek {peek}");
        }
        assert!(run_lbr(&agent, 50, 100, 20_000, 0, 2, true, 0, 9, Sizes::Fixed, true, Some(Open { to: 150, opener: None })).is_err());
    }

    /// Without the peek, LBR-open's model never reads F2's translation seed
    /// (a wrong seed changes none of its action values; with the peek it
    /// does), and F2's split of the open is tracked as two paths.
    #[test]
    fn peek_free_lbr_open_never_reads_f2s_seed() {
        let game = Arc::new(Nlhe::new(NlheConfig::grid("200bb").unwrap()));
        let agent = Agent::from_policy(Arc::clone(&game), Policy::uniform()).with_open_below(crate::play::OpenBelow::Limp);
        let mut ctx = Ctx { open: Some(Open { to: 200, opener: None }), ..make_ctx(&agent, 50, 100, 20_000) };
        let (mut peek_reads, mut split, mut hands) = (0, 0, 0);
        // Only postflop decisions leave values: most hands end preflop.
        for i in 0..100 {
            let (cards, seed) = deal(&ctx, 6, i);
            let mut play = |peek, xor| {
                ctx.peek = peek;
                ctx.model_seed_xor = xor;
                let net = play_hand(&ctx, 0, &cards, seed, None).map(f64::to_bits);
                let (values, paths) = std::mem::take(&mut *ctx.probe.lock().unwrap());
                (net, values, paths)
            };
            let blind = play(false, 0);
            assert!(blind.0.is_ok(), "deal {i}: {:?}", blind.0);
            assert_eq!(play(false, 0x5EED), blind, "deal {i}");
            split += usize::from(blind.2 > 1);
            let peek = play(true, 0);
            peek_reads += usize::from(play(true, 0x5EED) != peek);
            hands += 1;
        }
        assert_eq!(split, hands, "every open to 2bb is split: only {split} of {hands} hands");
        assert!(peek_reads >= 5, "a wrong seed changed the peeking LBR in only {peek_reads} of {hands} hands");
    }

    /// The `bp` opener: the opener's root action for LBR's hand, its grid
    /// open replaced by the raise to `to`, its fold, limp and all-in kept; the
    /// draw is on LBR's own stream, so the same with F2 and against any agent.
    #[test]
    fn the_bp_opener_plays_the_openers_root_with_its_open_resized() {
        let game = Arc::new(Nlhe::new(NlheConfig::grid("200bb").unwrap()));
        let opener = Agent::from_policy(Arc::clone(&game), Policy::uniform());
        let agent = Agent::from_policy(Arc::clone(&game), Policy::uniform().leak(crate::s32::LeakFamily::Station, 0.8));
        let f2 = agent.with_open_below(crate::play::OpenBelow::Limp);
        let open = Some(Open { to: 250, opener: Some(&opener) });
        let (a, b) = (Ctx { open, ..make_ctx(&agent, 50, 100, 20_000) }, Ctx { open, ..make_ctx(&f2, 50, 100, 20_000) });
        let mut seen = [0usize; 4];
        for i in 0..400 {
            let (cards, seed) = deal(&a, 4, i);
            let me = [cards[0], cards[1]];
            let mut s = a.game.initial_state();
            for &c in &cards[..4] {
                s = a.game.deal_specific(&s, Card::from_index(c));
            }
            let act = open_action(&a, open.unwrap(), &s, me, seed).unwrap();
            assert_eq!(open_action(&b, open.unwrap(), &s, me, seed).unwrap(), act);
            seen[match act {
                RealAct::Fold => 0,
                RealAct::Call => 1,
                RealAct::Add(200) => 2,
                RealAct::Add(19_950) => 3,
                other => panic!("deal {i}: {other:?}"),
            }] += 1;
        }
        // Uniform rows: a quarter each.
        assert!(seen.iter().all(|&n| (70..=130).contains(&n)), "{seen:?}");
    }
}
