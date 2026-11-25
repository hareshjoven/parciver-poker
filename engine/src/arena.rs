// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
//! Head-to-head match runner for the play agent: real hands, real rules, real
//! chips — the same boundary the bot will face at a table.
//!
//! Every hand is dealt from a seeded shuffle of the game's deck and played on the
//! real-chip engine (`Nlhe::apply_real`), each bot seeing only its own cards, the
//! board so far and the action list. With `mirror`, each deal is played twice
//! with the seats swapped (duplicate poker): card luck cancels, so the reported
//! mean has a far smaller standard error for the same number of hands. Scores are
//! bot A's net winnings in big blinds per 100 hands.
//!
//! Baseline opponents are deliberately simple and deterministic in intent:
//! a calling station, a pot-raiser, a min-raiser (the worst case for the grid,
//! whose smallest raise is pot), a random-size player (stress), and the agent
//! itself with its bet sizes rescaled off the grid (`Perturbed`), which measures
//! what translation costs against an opponent that plays the blueprint's
//! strategy at sizes the blueprint does not use, and the agent with each size
//! redrawn from the sizes people and other solvers use (`Realistic`).

use std::sync::Arc;

use rayon::prelude::*;

use crate::cards::Card;
use crate::equity;
use crate::game::Game;
use crate::nlhe::{Nlhe, NlheConfig, NlheState, RealAct};
use crate::play::{Agent, HandSpec, Source};
use crate::subgame;

pub(crate) enum Bot {
    Agent(Arc<Agent>),
    /// The agent, but every bet/raise it makes is rescaled by this factor
    /// (clamped to the legal range), putting its sizes off the grid.
    Perturbed(Arc<Agent>, f64),
    /// The agent, but every bet/raise it makes (short of all-in) is redrawn
    /// from a realistic size menu near its own size (`realistic_add`): what the
    /// grid's translation meets at a real table.
    Realistic(Arc<Agent>),
    Station,
    PotRaiser,
    MinRaiser,
    Random,
}

impl Bot {
    pub(crate) fn name(&self) -> String {
        match self {
            Bot::Agent(_) => "agent".into(),
            Bot::Perturbed(_, f) => format!("agent_sizes_x{f}"),
            Bot::Realistic(_) => "agent_realistic_sizes".into(),
            Bot::Station => "station".into(),
            Bot::PotRaiser => "pot_raiser".into(),
            Bot::MinRaiser => "min_raiser".into(),
            Bot::Random => "random".into(),
        }
    }
}

#[derive(Default, Clone)]
pub(crate) struct MatchStats {
    /// Bot A's net in big blinds, one entry per deal (sum over both seatings when
    /// mirrored).
    pub per_deal_bb: Vec<f64>,
    /// With `luck`: `per_deal_bb` less the board deals' luck (VARRED.md), same
    /// mean, smaller variance; else empty.
    pub per_deal_bb_cv: Vec<f64>,
    /// With `aivat`: `per_deal_bb_cv` less the luck of both bots' sampled
    /// actions as well (records/rating-live/PREREG.md), same mean; else empty.
    pub per_deal_bb_aivat: Vec<f64>,
    /// With `aivat`: the largest |1 - sum of a policy| met (a search's row from
    /// a float32 solver is not normalized to f64 precision; the terms use the
    /// normalized policy, which moves each term's mean by at most
    /// 2 * this * the largest |V|, a stack); else 0.
    pub aivat_mass_error: f64,
    /// Bot A agent decisions by source: blueprint, missing key, off-tree, search.
    pub a_sources: [u64; 4],
    pub b_sources: [u64; 4],
    /// Bot A's (B's) decisions where the search failed: [built-in solver used
    /// in place of the external one, blueprint used], here or at an earlier
    /// decision of the bot's in the hand that its search read
    /// (`subgame::search_failure`).
    pub a_search_failed: [u64; 2],
    pub b_search_failed: [u64; 2],
    pub hands: u64,
}

impl MatchStats {
    /// Mean and standard error of A's winnings in bb per 100 HANDS.
    pub(crate) fn bb_per_100(&self, hands_per_deal: f64) -> (f64, f64) {
        let n = self.per_deal_bb.len() as f64;
        if n < 2.0 {
            return (f64::NAN, f64::NAN);
        }
        let mean = self.per_deal_bb.iter().sum::<f64>() / n;
        let var = self.per_deal_bb.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let scale = 100.0 / hands_per_deal;
        (mean * scale, (var / n).sqrt() * scale)
    }
}

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

pub(crate) struct Table {
    pub cfg: NlheConfig,
    pub game: Nlhe,
}

impl Table {
    /// Real table: the given blinds and stack on `base`'s deck and board length.
    pub(crate) fn new(base: &NlheConfig, sb: i64, bb: i64, stack: i64) -> Self {
        let cfg = NlheConfig { sb, bb, stack, ..base.clone() };
        Table { game: Nlhe::new(cfg.clone()), cfg }
    }
}

/// Seat-relative view a bot decides from.
struct View<'a> {
    seat: usize,
    hole: [u8; 2],
    board: &'a [u8],
    actions: &'a [RealAct],
    state: &'a NlheState,
    seed: u64,
}

fn baseline(table: &Table, bot: &Bot, v: &View, rng: &mut Rng) -> RealAct {
    let s = v.state;
    let c = Nlhe::owed(s);
    let stack = Nlhe::stack_of(s, Nlhe::acting(s));
    let min_r = table.game.min_raise_add(s);
    let check_call = if c == 0 { RealAct::Check } else { RealAct::Call };
    match bot {
        Bot::Station => check_call,
        Bot::PotRaiser => {
            let pot = Nlhe::pot(s);
            let d = (c + (pot + c)).max(min_r);
            if d >= stack { RealAct::Add(stack) } else { RealAct::Add(d) }
        }
        Bot::MinRaiser => {
            if min_r < stack { RealAct::Add(min_r) } else { check_call }
        }
        Bot::Random => match rng.below(6) {
            0 if c > 0 => RealAct::Fold,
            0..=2 => check_call,
            3 if min_r < stack => RealAct::Add(min_r),
            4 if min_r < stack => RealAct::Add(min_r + rng.below((stack - min_r) as u64) as i64),
            _ => RealAct::Add(stack),
        },
        Bot::Agent(_) | Bot::Perturbed(..) | Bot::Realistic(_) => unreachable!(),
    }
}

/// Postflop bets as pot fractions, grouped by the size they replace.
const REAL_BETS: [(f64, &[f64]); 4] = [
    (0.45, &[0.25, 0.33, 0.4]),
    (0.9, &[0.5, 0.6, 0.66, 0.75]),
    (1.4, &[0.85, 1.0, 1.25]),
    (f64::INFINITY, &[1.5, 2.0, 2.5]),
];
/// Raises (and preflop opens) as raise-to multiples of the bet faced this
/// street (preflop: at least the big blind), grouped by the size they replace:
/// 2-2.5bb opens and min-ish raises, 3x, 4x and 5-6x.
const REAL_RAISES: [(f64, &[f64]); 4] = [
    (2.6, &[2.0, 2.2, 2.5]),
    (3.6, &[2.75, 3.0, 3.5]),
    (5.0, &[4.0, 4.5]),
    (f64::INFINITY, &[5.0, 6.0]),
];

/// A realistic replacement for the bet/raise `add` (chips added, short of
/// all-in): a size drawn from the menu group of `add`'s own size, so the
/// strategy's intent (small, medium, large, overbet) is kept while the exact
/// size is one the grid does not use. Clamped to the legal range.
fn realistic_add(table: &Table, s: &NlheState, add: i64, rng: &mut Rng) -> i64 {
    let p = Nlhe::acting(s);
    let owed = Nlhe::owed(s);
    let stack = Nlhe::stack_of(s, p);
    let pick = |menu: &[(f64, &[f64])], x: f64, rng: &mut Rng| {
        let group = menu.iter().find(|(upto, _)| x < *upto).map_or(menu[menu.len() - 1].1, |g| g.1);
        group[rng.below(group.len() as u64) as usize]
    };
    let preflop = Nlhe::betting_street(s) == Some(0);
    let y = if owed == 0 && !preflop {
        let pot = Nlhe::pot(s) as f64;
        (pick(&REAL_BETS, add as f64 / pot, rng) * pot).round() as i64
    } else {
        let mine = Nlhe::street_contrib_of(s, p);
        let faced = Nlhe::street_contrib_of(s, 1 - p).max(table.cfg.bb) as f64;
        let to = pick(&REAL_RAISES, (mine + add) as f64 / faced, rng) * faced;
        to.round() as i64 - mine
    };
    y.max(table.game.min_raise_add(s)).min(stack)
}

/// An agent decision's source and its index in `MatchStats::a_search_failed`
/// where the search failed.
type Origin = (Source, Option<usize>);

/// A decision: the action, the agent's decision origin when there is one, and
/// the policy the action was drawn from when it is exactly that (an unmodified
/// agent's `Decision::options`).
type Decided = (RealAct, Option<Origin>, Option<Vec<(RealAct, f64)>>);

/// One decision (see `Decided`).
fn decide(table: &Table, bot: &Bot, v: &View, rng: &mut Rng) -> Result<Decided, String> {
    let agent = match bot {
        Bot::Agent(a) | Bot::Perturbed(a, _) | Bot::Realistic(a) => a,
        _ => return Ok((baseline(table, bot, v, rng), None, None)),
    };
    let spec = HandSpec {
        hero: v.seat,
        hole: v.hole,
        board: v.board.to_vec(),
        sb: table.cfg.sb,
        bb: table.cfg.bb,
        stacks: [table.cfg.stack, table.cfg.stack],
        actions: v.actions.to_vec(),
        seed: v.seed,
    };
    // Only the action and its source are read: no chip accounting.
    let d = agent.decide_inner(&spec)?;
    let failed = subgame::search_failure(agent, &spec, &d);
    let mut act = d.action;
    if let (Bot::Perturbed(_, f), RealAct::Add(x)) = (bot, act) {
        let s = v.state;
        let c = Nlhe::owed(s);
        let stack = Nlhe::stack_of(s, Nlhe::acting(s));
        if x < stack && x > c {
            let raise = ((x - c) as f64 * f).round() as i64;
            let y = (c + raise).max(table.game.min_raise_add(s));
            act = RealAct::Add(y.min(stack));
        }
    }
    if let (Bot::Realistic(_), RealAct::Add(x)) = (bot, act) {
        let s = v.state;
        if x < Nlhe::stack_of(s, Nlhe::acting(s)) && x > Nlhe::owed(s) {
            act = RealAct::Add(realistic_add(table, s, x, rng));
        }
    }
    let policy = matches!(bot, Bot::Agent(_)).then_some(d.options);
    Ok((act, Some((d.source, failed)), policy))
}

/// Agent decisions of bots A (0) and B (1) as `MatchStats` counts them.
#[derive(Default)]
struct Tally {
    sources: [[u64; 4]; 2],
    search_failed: [[u64; 2]; 2],
    /// `MatchStats::aivat_mass_error` over the deal.
    mass_error: f64,
}

/// Seat 0's all-in edge 2 eq - 1 against seat 1 (ties half) on each board
/// length of one deal: computed on first use, shared by the deal's two seatings.
struct Luck<'a> {
    cards: &'a [u8],
    edge: [Option<f64>; 6],
}

impl<'a> Luck<'a> {
    fn new(cards: &'a [u8]) -> Self {
        Luck { cards, edge: [None; 6] }
    }

    /// Seat 0's edge with the first `n` board cards dealt (exact enumeration).
    fn edge(&mut self, n: usize) -> f64 {
        let cards = self.cards;
        *self.edge[n].get_or_insert_with(|| {
            let c = |i: usize| Card::from_index(cards[i]);
            let board: Vec<Card> = (0..n).map(|k| c(4 + k)).collect();
            let t = equity::equity_vs_hand([c(0), c(1)], [c(2), c(3)], &board);
            (t.wins as f64 - t.losses as f64) / t.total() as f64
        })
    }
}

/// The largest |1 - sum| of a policy `aivat` accepts (a float32 solver's
/// rows miss by ~1e-7); beyond it the match is an error.
const MASS_TOLERANCE: f64 = 1e-4;

/// A's value of `s` for the deal's actual holes (records/rating-live/PREREG.md):
/// its net at a terminal, else A's all-in edge on the board so far (`edge0` is
/// seat 0's) times the matched chips.
fn value(g: &Nlhe, s: &NlheState, a_seat: usize, edge0: f64) -> f64 {
    if g.is_terminal(s) {
        let (n0, n1) = g.payoffs(s);
        return if a_seat == 0 { n0 } else { n1 };
    }
    let sign = if a_seat == 0 { 1.0 } else { -1.0 };
    sign * edge0 * Nlhe::committed_of(s, 0).min(Nlhe::committed_of(s, 1)) as f64
}

/// Play one hand. `a_seat` is bot A's seat. Returns A's net chips; with
/// `luck`, A's net less the luck of the board deals (VARRED.md: each deal's
/// change in A's all-in value edge * matched chips, which has mean 0 given what
/// came before); with `luck` and `aivat`, that less the luck of every sampled
/// action as well (each decision's V(chosen) - sum_a pi(a) V(a), `value`,
/// mean 0 given what came before: both bots must be unmodified agents, whose
/// action is a fresh draw from `pi`). Tallies the decisions of A and B.
#[allow(clippy::too_many_arguments)]
fn play_hand(
    table: &Table,
    bots: [&Bot; 2],
    a_seat: usize,
    cards: &[u8],
    seed: u64,
    tally: &mut Tally,
    mut luck: Option<&mut Luck>,
    aivat: bool,
) -> Result<(f64, f64, f64), String> {
    let g = &table.game;
    let mut s = g.initial_state();
    for &c in &cards[..4] {
        s = g.deal_specific(&s, Card::from_index(c));
    }
    let seat_bot = |seat: usize| if seat == a_seat { bots[0] } else { bots[1] };
    let mut board: Vec<u8> = Vec::new();
    let mut actions: Vec<RealAct> = Vec::new();
    let mut rng = Rng(seed ^ 0xD1B5_4A32_D192_ED03);
    // A's sign on seat 0's edge: the two seatings' terms are exact negations.
    let sign = if a_seat == 0 { 1.0 } else { -1.0 };
    let (mut drawn, mut acted) = (0.0, 0.0);
    loop {
        if Nlhe::is_dealing(&s) {
            let (before, matched) = (board.len(), Nlhe::committed_of(&s, 0).min(Nlhe::committed_of(&s, 1)));
            while Nlhe::is_dealing(&s) {
                let c = cards[4 + board.len()];
                board.push(c);
                s = g.deal_specific(&s, Card::from_index(c));
            }
            if let Some(l) = luck.as_deref_mut() {
                drawn += sign * matched as f64 * (l.edge(board.len()) - l.edge(before));
            }
        }
        if g.is_terminal(&s) {
            break;
        }
        let seat = Nlhe::acting(&s);
        let v = View {
            seat,
            hole: [cards[2 * seat], cards[2 * seat + 1]],
            board: &board,
            actions: &actions,
            state: &s,
            // Constant per (hand, seat): the agent's translation of earlier
            // actions must not re-randomize between its decisions in one hand.
            seed: splitmix(seed ^ (seat as u64 + 1)),
        };
        let (act, origin, policy) = decide(table, seat_bot(seat), &v, &mut rng)?;
        let next = g
            .apply_real(&s, act)
            .map_err(|e| format!("bot {} made an illegal action {act:?}: {e}", seat_bot(seat).name()))?;
        if aivat {
            let Some(pi) = policy else {
                return Err(format!("aivat needs unmodified agents on both sides, not {}", seat_bot(seat).name()));
            };
            let Some(l) = luck.as_deref_mut() else {
                return Err("aivat needs the board's luck".into());
            };
            let edge = l.edge(board.len());
            let total: f64 = pi.iter().map(|o| o.1).sum();
            // The action need not have p > 0: a search row short of 1 gives the
            // missing mass to its last action, p == 0 or not (within the bound).
            if !((total - 1.0).abs() <= MASS_TOLERANCE) || !pi.iter().any(|o| o.0 == act) {
                return Err(format!("the action {act:?} is not a draw from its policy {pi:?}"));
            }
            tally.mass_error = tally.mass_error.max((total - 1.0).abs());
            let mut mean = 0.0;
            for &(o, p) in pi.iter().filter(|o| o.1 > 0.0) {
                let t = g.apply_real(&s, o).map_err(|e| format!("a policy action {o:?} is illegal: {e}"))?;
                mean += p / total * value(g, &t, a_seat, edge);
            }
            acted += value(g, &next, a_seat, edge) - mean;
        }
        if let Some((src, failed)) = origin {
            let bot = usize::from(seat != a_seat);
            tally.sources[bot][src as usize] += 1;
            if let Some(k) = failed {
                tally.search_failed[bot][k] += 1;
            }
        }
        s = next;
        actions.push(act);
    }
    let (n0, n1) = g.payoffs(&s);
    let net = if a_seat == 0 { n0 } else { n1 };
    Ok((net, net - drawn, net - drawn - acted))
}

/// Deal `i` of the match `seed`: its `need` cards (seat 0's hole, seat 1's, then
/// the board) and its hand seed.
fn deal(deck: &[u8], need: usize, seed: u64, i: u64) -> (Vec<u8>, u64) {
    let mut rng = Rng(splitmix(seed ^ splitmix(i)));
    let mut d = deck.to_vec();
    for k in 0..need {
        let j = k + rng.below((d.len() - k) as u64) as usize;
        d.swap(k, j);
    }
    d.truncate(need);
    (d, rng.next())
}

/// Play `deals` deals (twice each with seats swapped when `mirror`; otherwise A
/// sits in `a_seat`). Deterministic in `seed`, parallel over deals. Deal `i`
/// depends only on `(seed, i)`, so deals `first..first + deals` of a longer
/// match can be played in chunks with identical results. With `luck`, also
/// `MatchStats::per_deal_bb_cv`; with `aivat`, that and
/// `MatchStats::per_deal_bb_aivat` (real 52-card games only; `aivat` needs
/// unmodified agents on both sides).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_match(
    table: &Table,
    a: &Bot,
    b: &Bot,
    first: u64,
    deals: u64,
    mirror: bool,
    a_seat: usize,
    seed: u64,
    luck: bool,
    aivat: bool,
) -> Result<MatchStats, String> {
    let deck = table.cfg.deck.clone();
    let need = 4 + table.cfg.board_cards;
    let luck = luck || aivat;
    if luck && (deck.len() != 52 || table.cfg.board_cards != 5) {
        return Err("luck needs the real 52-card deck and a 5-card board".into());
    }
    if aivat && ![a, b].iter().all(|x| matches!(x, Bot::Agent(_))) {
        return Err(format!("aivat needs unmodified agents on both sides, not {} and {}", a.name(), b.name()));
    }
    let bb = table.cfg.bb as f64;
    type DealResult = Result<(f64, f64, f64, Tally), String>;
    let results: Vec<DealResult> = (first..first + deals)
        .into_par_iter()
        .map(|i| {
            let (d, hand_seed) = deal(&deck, need, seed, i);
            let cards = &d[..];
            let mut tally = Tally::default();
            let mut l = Luck::new(cards);
            let mut hand = |seat| play_hand(table, [a, b], seat, cards, hand_seed, &mut tally, luck.then_some(&mut l), aivat);
            let (net, cv, av) = if mirror {
                let (x, y) = (hand(0)?, hand(1)?);
                (x.0 + y.0, x.1 + y.1, x.2 + y.2)
            } else {
                hand(a_seat)?
            };
            Ok((net / bb, cv / bb, av / bb, tally))
        })
        .collect();
    let mut stats = MatchStats::default();
    for r in results {
        let (x, cv, av, t) = r?;
        stats.per_deal_bb.push(x);
        if luck {
            stats.per_deal_bb_cv.push(cv);
        }
        if aivat {
            stats.per_deal_bb_aivat.push(av);
            stats.aivat_mass_error = stats.aivat_mass_error.max(t.mass_error);
        }
        for (k, (a, b)) in t.sources[0].iter().zip(&t.sources[1]).enumerate() {
            stats.a_sources[k] += a;
            stats.b_sources[k] += b;
        }
        for (k, (a, b)) in t.search_failed[0].iter().zip(&t.search_failed[1]).enumerate() {
            stats.a_search_failed[k] += a;
            stats.b_search_failed[k] += b;
        }
    }
    stats.hands = deals * if mirror { 2 } else { 1 };
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s32::Policy;

    fn table() -> Table {
        Table::new(&NlheConfig::pluribus_100bb(), 50, 100, 10_000)
    }

    /// Every size `realistic_add` can give for `add` at `s` over many draws.
    fn sizes(t: &Table, s: &NlheState, add: i64) -> std::collections::BTreeSet<i64> {
        let mut rng = Rng(7);
        (0..200).map(|_| realistic_add(t, s, add, &mut rng)).collect()
    }

    #[test]
    fn realistic_sizes_keep_the_intent_on_real_menus() {
        let t = table();
        let g = &t.game;
        let mut s = g.initial_state();
        for c in [0u8, 1, 2, 3] {
            s = g.deal_specific(&s, Card::from_index(c));
        }
        // SB opens to 2.5bb (adds 200 over its 50): a 2-2.5bb open.
        assert_eq!(sizes(&t, &s, 200), [150, 170, 200].into());
        // Limped pot, flop: BB bets half pot (100 into 200): 25-75% pot group.
        s = g.apply_real(&s, RealAct::Call).unwrap();
        s = g.apply_real(&s, RealAct::Check).unwrap();
        for c in [10u8, 20, 30] {
            s = g.deal_specific(&s, Card::from_index(c));
        }
        assert_eq!(sizes(&t, &s, 100), [100, 120, 132, 150].into());
        // Pot-size overbet intent (1.5x) draws from the overbet group.
        assert_eq!(sizes(&t, &s, 300), [300, 400, 500].into());
        // BB bets 100; SB raises to 3x (adds 300): the 2.75-3.5x group.
        s = g.apply_real(&s, RealAct::Add(100)).unwrap();
        assert_eq!(sizes(&t, &s, 300), [275, 300, 350].into());
    }

    #[test]
    fn a_realistic_sizing_match_is_legal_and_deterministic() {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let agent = Arc::new(Agent::from_policy(game, Policy::uniform()));
        let (a, b) = (Bot::Agent(Arc::clone(&agent)), Bot::Realistic(Arc::clone(&agent)));
        let t = table();
        let run = |b: &Bot| run_match(&t, &a, b, 0, 40, true, 0, 11, false, false).expect("every action legal");
        let (x, y) = (run(&b), run(&b));
        assert_eq!(x.per_deal_bb, y.per_deal_bb);
        assert_eq!(x.hands, 80);
        // The resized sizes change the play: not the unmodified agent's match.
        assert_ne!(x.per_deal_bb, run(&Bot::Agent(agent)).per_deal_bb);
    }

    /// An external solver that always fails.
    struct NoGpu;

    impl subgame::SubgameSolver for NoGpu {
        fn solve(&self, _: &crate::search::FlatSubgame) -> Result<Vec<Vec<f64>>, String> {
            Err("no GPU".into())
        }
    }

    #[test]
    fn a_match_counts_the_decisions_a_failed_search_reached() {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let plain = Agent::from_policy(game, Policy::uniform());
        let cfg = |external: Option<subgame::External>, rescue: bool| subgame::SearchCfg {
            from_flop: false,
            flop_iters: 1,
            flop_checkdown: false,
            from_turn: false,
            river_iters: 3,
            turn_iters: 3,
            exact_own_range: true,
            solve_round: true,
            external,
            rescue,
        };
        let no_gpu = || Some(subgame::External(Arc::new(NoGpu)));
        let t = table();
        let b = Bot::Agent(Arc::new(plain.with_oversize(plain.oversize())));
        let run = |a: Agent| run_match(&t, &Bot::Agent(Arc::new(a)), &b, 0, 30, true, 0, 11, false, false).expect("every action legal");
        let blueprint = run(plain.with_oversize(plain.oversize()));
        let built_in = run(plain.with_search(cfg(None, true)));
        let failed = run(plain.with_search(cfg(no_gpu(), false)));
        let rescued = run(plain.with_search(cfg(no_gpu(), true)));
        assert!(built_in.a_sources[3] > 0 && built_in.a_search_failed == [0, 0]);
        // No rescue: the blueprint decides wherever the search would have, as
        // an agent without search plays.
        assert_eq!((&failed.per_deal_bb, failed.a_sources), (&blueprint.per_deal_bb, blueprint.a_sources));
        assert!(failed.a_search_failed[0] == 0 && failed.a_search_failed[1] > 0);
        // Rescued: the built-in search decides, every time in place of the external one.
        assert_eq!((&rescued.per_deal_bb, rescued.a_sources), (&built_in.per_deal_bb, built_in.a_sources));
        assert_eq!(rescued.a_search_failed, [built_in.a_sources[3], 0]);
        for s in [&blueprint, &built_in, &failed, &rescued] {
            assert_eq!(s.b_search_failed, [0, 0]);
        }
    }

    /// Seat 0's edge 2 eq - 1 on `board`, by `Tally::equity` (not `Luck`'s formula).
    fn edge_on(cards: &[u8], board: &[u8]) -> f64 {
        let c = |i: u8| Card::from_index(i);
        let b: Vec<Card> = board.iter().map(|&i| c(i)).collect();
        2.0 * equity::equity_vs_hand([c(cards[0]), c(cards[1])], [c(cards[2]), c(cards[3])], &b).equity() - 1.0
    }

    #[test]
    fn luck_changes_no_score_and_cancels_in_self_play() {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let agent = Arc::new(Agent::from_policy(game, Policy::uniform()));
        let (a, t) = (Bot::Agent(Arc::clone(&agent)), table());
        for b in [Bot::Station, Bot::Random, Bot::Realistic(Arc::clone(&agent))] {
            let off = run_match(&t, &a, &b, 0, 30, true, 0, 5, false, false).unwrap();
            let on = run_match(&t, &a, &b, 0, 30, true, 0, 5, true, false).unwrap();
            assert_eq!((&on.per_deal_bb, on.a_sources, on.hands), (&off.per_deal_bb, off.a_sources, off.hands));
            assert!(off.per_deal_bb_cv.is_empty() && on.per_deal_bb_cv.len() == 30);
            assert_ne!(on.per_deal_bb_cv, on.per_deal_bb, "the luck terms apply");
        }
        // Mirrored identical play: both scores cancel exactly, deal by deal.
        let same = run_match(&t, &a, &a, 0, 30, true, 0, 5, true, false).unwrap();
        assert!(same.per_deal_bb.iter().chain(&same.per_deal_bb_cv).all(|&x| x == 0.0));
        // Not a real deck: refused.
        let short = Table::new(&NlheConfig::small(), 1, 2, 4);
        assert!(run_match(&short, &Bot::Station, &Bot::Station, 0, 2, true, 0, 5, true, false).is_err());
    }

    #[test]
    fn checked_down_and_preflop_all_in_hands_score_their_preflop_all_in_value() {
        // Stations limp and check down (1 bb matched); a pot-raiser with 2 bb
        // stacks is all in preflop against a station (2 bb matched). Either way
        // the board luck is all of the luck: what is left is the preflop
        // all-in value, (2 eq - 1) * matched, for A in either seat.
        let shallow = Table::new(&NlheConfig::pluribus_100bb(), 50, 100, 200);
        for (t, a, matched) in [(table(), Bot::Station, 1.0), (shallow, Bot::PotRaiser, 2.0)] {
            for seat in [0, 1] {
                let m = run_match(&t, &a, &Bot::Station, 0, 12, false, seat, 21, true, false).unwrap();
                let sign = if seat == 0 { 1.0 } else { -1.0 };
                for (i, (&u, &cv)) in m.per_deal_bb.iter().zip(&m.per_deal_bb_cv).enumerate() {
                    let (cards, _) = deal(&t.cfg.deck, 9, 21, i as u64);
                    assert!([-matched, 0.0, matched].contains(&u), "a showdown for {matched} bb: {u}");
                    let want = sign * matched * edge_on(&cards, &[]);
                    assert!((cv - want).abs() < 1e-9, "deal {i} seat {seat}: {cv} vs {want}");
                }
            }
        }
    }

    #[test]
    fn a_deal_s_term_takes_the_chips_matched_before_the_deal() {
        // The other tests' bots ignore their cards, so any chips would keep the
        // terms' mean 0. A river-searching agent bets the river by its cards:
        // played against many river cards, a correct river term (the chips
        // matched before the river) leaves A's luck a function of who wins (the
        // edge after is 1, 0 or -1), while the chips at the end vary. (Against
        // a station, which never folds, most hands reach the river.)
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let river = subgame::SearchCfg {
            from_flop: false,
            flop_iters: 1,
            flop_checkdown: false,
            from_turn: false,
            river_iters: 3,
            turn_iters: 1,
            exact_own_range: true,
            solve_round: true,
            external: None,
            rescue: true,
        };
        let a = Bot::Agent(Arc::new(Agent::from_policy(game, Policy::uniform()).with_search(river)));
        let t = table();
        let mut varied = 0;
        for i in 0..40 {
            let (cards, seed) = deal(&t.cfg.deck, 9, 3, i);
            for seat in [0, 1] {
                // (edge after the river, |A's net|, A's luck) for every 4th river card.
                let all: Vec<(f64, f64, f64)> = t.cfg.deck.iter().filter(|c| !cards[..8].contains(c)).step_by(4).map(|&r| {
                    let mut c = cards.clone();
                    c[8] = r;
                    let mut l = Luck::new(&c);
                    let (net, cv, _) = play_hand(&t, [&a, &Bot::Station], seat, &c, seed, &mut Tally::default(), Some(&mut l), false).unwrap();
                    (l.edge(5), net.abs(), net - cv)
                }).collect();
                let class = |w: f64| all.iter().filter(move |x| x.0 == w);
                for w in [-1.0, 0.0, 1.0] {
                    if let Some(first) = class(w).next() {
                        assert!(class(w).all(|x| (x.2 - first.2).abs() < 1e-9), "deal {i} seat {seat}: luck varies within edge {w}");
                    }
                }
                // The hands where the river play changes the chips at the end
                // within a class: where chips taken after the deal would show.
                if [-1.0, 1.0].iter().any(|&w| class(w).any(|x| x.1 != class(w).next().unwrap().1)) {
                    varied += 1;
                    // The river's chips: half the luck between a win and a loss,
                    // at most what any showdown's winner takes.
                    if let (Some(w), Some(l)) = (class(1.0).next(), class(-1.0).next()) {
                        let m = (w.2 - l.2) / 2.0 * if seat == 0 { 1.0 } else { -1.0 };
                        assert!(m > 0.0 && (m - m.round()).abs() < 1e-9, "deal {i} seat {seat}: chips {m}");
                        assert!(all.iter().filter(|x| x.0 != 0.0).all(|x| m <= x.1 + 1e-9), "deal {i} seat {seat}: {m}");
                    }
                }
            }
            if varied >= 3 {
                break;
            }
        }
        assert!(varied >= 3, "only {varied} hands with river play that varies the chips");
    }

    #[test]
    fn aivat_changes_no_score_and_cancels_in_self_play() {
        // Tests 1, 2, 7 of records/rating-live/PREREG.md.
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let uniform = Arc::new(Agent::from_policy(Arc::clone(&game), Policy::uniform()));
        let leaky = Arc::new(Agent::from_policy(game, Policy::uniform().leak(crate::s32::LeakFamily::Maniac, 0.5)));
        let (a, b, t) = (Bot::Agent(Arc::clone(&uniform)), Bot::Agent(leaky), table());
        let off = run_match(&t, &a, &b, 0, 30, true, 0, 5, true, false).unwrap();
        let on = run_match(&t, &a, &b, 0, 30, true, 0, 5, true, true).unwrap();
        let only = run_match(&t, &a, &b, 0, 30, true, 0, 5, false, true).unwrap();
        assert_eq!((&on.per_deal_bb, &on.per_deal_bb_cv, on.a_sources), (&off.per_deal_bb, &off.per_deal_bb_cv, off.a_sources));
        assert_eq!((&only.per_deal_bb_cv, &only.per_deal_bb_aivat), (&on.per_deal_bb_cv, &on.per_deal_bb_aivat));
        assert!(off.per_deal_bb_aivat.is_empty() && on.per_deal_bb_aivat.len() == 30);
        assert_ne!(on.per_deal_bb_aivat, on.per_deal_bb_cv, "the action terms apply");
        let same = run_match(&t, &a, &a, 0, 30, true, 0, 5, false, true).unwrap();
        assert!(same.per_deal_bb.iter().chain(&same.per_deal_bb_aivat).all(|&x| x == 0.0));
        // Not an unmodified agent on both sides: refused.
        for other in [Bot::Station, Bot::Random, Bot::Realistic(Arc::clone(&uniform)), Bot::Perturbed(uniform, 1.5)] {
            assert!(run_match(&t, &a, &other, 0, 4, true, 0, 5, false, true).is_err(), "{}", other.name());
        }
    }

    #[test]
    fn a_deterministic_policy_has_no_action_terms() {
        // Test 5: an empty table: every decision is the fallback, one action
        // with probability 1, so only the board deals' terms are left.
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let fixed = Bot::Agent(Arc::new(Agent::from_policy(Arc::clone(&game), Policy::tabular(Arc::new(Default::default())))));
        let other = Bot::Agent(Arc::new(Agent::from_policy(game, Policy::tabular(Arc::new(Default::default())).leak(crate::s32::LeakFamily::Nit, 1.0))));
        let m = run_match(&table(), &fixed, &other, 0, 20, true, 0, 9, false, true).unwrap();
        assert_eq!(m.per_deal_bb_aivat, m.per_deal_bb_cv);
        assert_eq!(m.a_sources[1] + m.b_sources[1], m.a_sources.iter().chain(&m.b_sources).sum::<u64>(), "all fallbacks");
    }

    #[test]
    fn a_showdown_s_value_is_its_payoff() {
        // Test 4: `value` reads a terminal's payoff; at a showdown the formula
        // for other states (the edge on the full board times the matched chips)
        // gives the same, so the terms telescope into the score.
        let t = table();
        let g = &t.game;
        let mut rng = Rng(5);
        let (mut all_in, mut checked) = (0, 0);
        for i in 0..300 {
            let (cards, _) = deal(&t.cfg.deck, 9, 13, i);
            let mut s = g.initial_state();
            for &c in &cards[..4] {
                s = g.deal_specific(&s, Card::from_index(c));
            }
            let (mut n, mut before_check) = (0, None);
            while !g.is_terminal(&s) {
                if Nlhe::is_dealing(&s) {
                    s = g.deal_specific(&s, Card::from_index(cards[4 + n]));
                    n += 1;
                    continue;
                }
                let (stack, owed) = (Nlhe::stack_of(&s, Nlhe::acting(&s)), Nlhe::owed(&s));
                let act = match rng.below(4) {
                    0 if owed < stack => RealAct::Add(g.min_raise_add(&s).min(stack)),
                    1 if owed < stack => RealAct::Add(stack),
                    _ if owed == 0 => RealAct::Check,
                    _ => RealAct::Call,
                };
                before_check = (act == RealAct::Check).then(|| s.clone());
                s = g.apply_real(&s, act).unwrap();
            }
            assert_eq!(n, 5, "no folds: a showdown");
            let m = Nlhe::committed_of(&s, 0).min(Nlhe::committed_of(&s, 1));
            all_in += usize::from(m == t.cfg.stack);
            let edge = Luck::new(&cards).edge(5);
            let (n0, n1) = g.payoffs(&s);
            for (seat, pay) in [(0, n0), (1, n1)] {
                let sign = if seat == 0 { 1.0 } else { -1.0 };
                assert_eq!(value(g, &s, seat, edge), pay);
                // Checked down on the river: the state before the last check is
                // not terminal, and its value is already the payoff.
                if let Some(b) = &before_check {
                    checked += 1;
                    assert!(!g.is_terminal(b) && (value(g, b, seat, edge) - pay).abs() < 1e-9, "deal {i} seat {seat}");
                }
                assert!((sign * edge * m as f64 - pay).abs() < 1e-9, "deal {i} seat {seat}: {} vs {pay}", sign * edge * m as f64);
            }
        }
        // Equal stacks: both seats have put in the same at every showdown. All-in
        // run-outs and bet-and-called showdowns are both covered.
        assert!(all_in > 0 && all_in < 300, "{all_in} all-in showdowns of 300");
        assert!(checked > 0, "no river checked down");
    }

    /// An external solver that answers every wanted slot uniformly, each row
    /// scaled by `.0` (a float32 solver's rows miss 1 by ~1e-7).
    struct Scaled(f64);

    impl subgame::SubgameSolver for Scaled {
        fn solve(&self, job: &crate::search::FlatSubgame) -> Result<Vec<Vec<f64>>, String> {
            let n = job.ranges[0].len();
            Ok(job
                .want
                .iter()
                .map(|&w| {
                    let i = job.slot.iter().position(|&x| x == w as i64).expect("a wanted slot");
                    let k = (job.child_start[i + 1] - job.child_start[i]) as usize;
                    vec![self.0 / k as f64; k * n]
                })
                .collect())
        }
    }

    fn river(external: Option<subgame::External>) -> subgame::SearchCfg {
        subgame::SearchCfg {
            from_flop: false,
            flop_iters: 1,
            flop_checkdown: false,
            from_turn: false,
            river_iters: 3,
            turn_iters: 1,
            exact_own_range: true,
            solve_round: true,
            external,
            rescue: false,
        }
    }

    #[test]
    fn aivat_takes_an_external_solver_s_unnormalized_rows() {
        // The audit's blocker: the GPU path's rows are not normalized to f64
        // precision. Within MASS_TOLERANCE the match runs on the normalized
        // policy and reports the largest miss; beyond it, an error.
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let plain = Agent::from_policy(game, Policy::uniform());
        let b = Bot::Agent(Arc::new(plain.with_oversize(plain.oversize())));
        let t = table();
        // The deals and seed of a_match_counts_the_decisions_a_failed_search_reached,
        // where the search is known to decide.
        let run = |scale: f64| {
            let a = Bot::Agent(Arc::new(plain.with_search(river(Some(subgame::External(Arc::new(Scaled(scale))))))));
            run_match(&t, &a, &b, 0, 30, true, 0, 11, false, true)
        };
        let (exact, off) = (run(1.0).unwrap(), run(1.0 + 1e-7).unwrap());
        let short = run(1.0 - 1e-7).unwrap();
        assert!(short.aivat_mass_error > 5e-8 && short.aivat_mass_error < 2e-7, "{}", short.aivat_mass_error);
        assert!(exact.a_sources[3] > 0 && exact.a_search_failed == [0, 0], "{:?} {:?}", exact.a_sources, exact.a_search_failed);
        assert!(exact.aivat_mass_error < 1e-12, "{}", exact.aivat_mass_error);
        assert!(off.aivat_mass_error > 5e-8 && off.aivat_mass_error < 2e-7, "{}", off.aivat_mass_error);
        let err = run(1.0 + 1e-3).err().expect("an error");
        assert!(err.contains("is not a draw from its policy"), "{err}");
    }

    #[test]
    fn a_deal_s_action_terms_have_mean_zero_over_the_draws() {
        // VARRED2's planned test: for fixed cards, over many hand seeds (the
        // draws), the action terms (the board-luck score less the aivat score)
        // average 0, for mixed policies on both sides, blueprint-style and
        // searched. A biased draw or a policy that is not the one drawn from
        // moves the mean; the cards are fixed, so the board terms cannot hide it.
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let uniform = Agent::from_policy(Arc::clone(&game), Policy::uniform());
        let maniac = Bot::Agent(Arc::new(Agent::from_policy(game, Policy::uniform().leak(crate::s32::LeakFamily::Maniac, 0.5))));
        let t = table();
        let pairs = [(Bot::Agent(Arc::new(uniform.with_oversize(uniform.oversize()))), 600), (Bot::Agent(Arc::new(uniform.with_search(river(None)))), 150)];
        for (k, (a, seeds)) in pairs.iter().enumerate() {
            for i in 0..3 {
                let (cards, _) = deal(&t.cfg.deck, 9, 41, i);
                let mut l = Luck::new(&cards);
                let mut tally = Tally::default();
                let mut terms = Vec::new();
                for j in 0..*seeds {
                    for seat in [0, 1] {
                        let (_, cv, av) = play_hand(&t, [a, &maniac], seat, &cards, splitmix(j), &mut tally, Some(&mut l), true).unwrap();
                        terms.push(cv - av);
                    }
                }
                // The searching pair's draws include searched decisions.
                assert!(k == 0 || tally.sources[0][3] > 0, "deal {i}: no search decided");
                let n = terms.len() as f64;
                let mean = terms.iter().sum::<f64>() / n;
                let se = (terms.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
                assert!(se > 0.0 && mean.abs() < 4.0 * se, "pair {k} deal {i}: mean {mean} se {se}");
            }
        }
    }

    #[test]
    fn the_edge_is_a_martingale_over_every_deal() {
        // Given the cards so far, the mean of seat 0's edge over every next
        // deal (the flop, the turn card, the river card) is the edge before,
        // exactly (as integers): what makes every luck term mean 0.
        let t = table();
        for i in 0..6 {
            let (cards, _) = deal(&t.cfg.deck, 9, 77, i);
            let mut l = Luck::new(&cards);
            // The flop's 17,296 boards once; the turn and river every time.
            for (n, k) in [(0, 3), (3, 1), (4, 1)].into_iter().filter(|&(n, _)| n > 0 || i == 0) {
                let tally = |b: &[u8]| {
                    let c = |i: u8| Card::from_index(i);
                    let b: Vec<Card> = b.iter().map(|&i| c(i)).collect();
                    let x = equity::equity_vs_hand([c(cards[0]), c(cards[1])], [c(cards[2]), c(cards[3])], &b);
                    (x.wins as i128 - x.losses as i128, x.total() as i128)
                };
                let seen = &cards[..4 + n];
                let unseen: Vec<u8> = t.cfg.deck.iter().copied().filter(|c| !seen.contains(c)).collect();
                let (d0, t0) = tally(&cards[4..4 + n]);
                assert_eq!(l.edge(n), d0 as f64 / t0 as f64);
                let (mut d, mut tot, mut mean, mut count) = (0i128, 0i128, 0.0, 0.0);
                let mut next = |extra: &[u8]| {
                    let b: Vec<u8> = cards[4..4 + n].iter().chain(extra).copied().collect();
                    let (x, y) = tally(&b);
                    (d, tot, mean, count) = (d + x, tot + y, mean + x as f64 / y as f64, count + 1.0);
                };
                if k == 1 {
                    unseen.iter().for_each(|&c| next(&[c]));
                } else {
                    for (p, &x) in unseen.iter().enumerate() {
                        for (q, &y) in unseen.iter().enumerate().skip(p + 1) {
                            unseen[q + 1..].iter().for_each(|&z| next(&[x, y, z]));
                        }
                    }
                }
                assert_eq!(d * t0, d0 * tot, "deal {i}, {n} board cards");
                assert!((mean / count - l.edge(n)).abs() < 1e-12);
            }
        }
    }
}
