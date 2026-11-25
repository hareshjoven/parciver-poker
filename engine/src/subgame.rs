//! Real-time search, part 2: from a real hand to a subgame and back to an
//! action — both players' ranges, the subgame of the current betting round,
//! and the solved strategy's action for the hero's holding.
//!
//! Ranges are Bayesian: every holding starts at weight 1 and is multiplied, for
//! each action its player has taken this hand, by the probability of taking
//! that action with that holding.
//!
//! * The OPPONENT is modelled as playing the agent's policy — the blueprint's
//!   view of how its lines are played (`Agent::action_likelihood`: exact for
//!   sizes the policy can output, the translation mixture for off-grid sizes),
//!   each factor floored at `RANGE_FLOOR`, so an action the model deems
//!   impossible (an off-grid line, a fallback spot) down-weights holdings
//!   instead of erasing the range. This is an approximation: real opponents do
//!   not play the blueprint.
//! * The SEARCHER's own range is exact when `SearchCfg::exact_own_range`: its
//!   actual decision distribution at each of its past nodes (`decide_inner`:
//!   search where it searched, the seeded translation it really used), which
//!   is what an observer who knows its strategy — LBR — tracks. Otherwise (the
//!   first version) it is modelled like the opponent's, which misstates the
//!   range wherever search chose an earlier action.
//!
//! What the subgame does with the current round's actions
//! (`SearchCfg::solve_round`):
//!
//! * Solved round (default): the subgame starts at the round's first action
//!   with both ranges as they stood then. The opponent's actions this round
//!   are SOLVED — each is one branch among the menu's at its node (its real
//!   size added when off the menu) — and the searcher's own earlier actions
//!   this round play its real strategy there, frozen. The opponent's bet is
//!   then read through the solve's equilibrium, not through the blueprint's
//!   likelihood of it. The first versions did the latter and lost to LBR
//!   exactly there: a pure best response to a model LBR's bets do not follow
//!   (scorecard-v1-quick: 94% of LBR's gain came from the search folding to
//!   its river bets, holding 65% equity where the model gave it 21%).
//! * Forced round (the first versions): the round's real actions are the only
//!   branches and both ranges are conditioned on them.
//!
//! Either way the ranges at the subgame's start are modelled, not solved for:
//! "unsafe" subgame solving (no gadget); the scorecard's LBR row is the check.

use std::sync::{Arc, Mutex};

use rustc_hash::FxHashMap;

use crate::nlhe::{Nlhe, RealAct};
use crate::play::{canonical_act, unit, Agent, Decision, HandSpec, Source, ALL_IN, STREAM_SEARCH};
use crate::search::{Act, FlatSubgame, Forced, Hands, Menu, RoundSpec, Solver, Tree};

/// Minimum likelihood factor per observed action.
pub(crate) const RANGE_FLOOR: f64 = 0.01;

/// Both players' ranges over `combos` (holdings that avoid the board) given
/// every action in `spec.actions`. Holdings are not filtered by the hero's own
/// cards: the solver handles card removal between the two ranges exactly.
pub(crate) fn ranges_at(agent: &Agent, spec: &HandSpec, combos: &[[u8; 2]]) -> Result<[Vec<f64>; 2], String> {
    ranges_failure(agent, spec, combos).map(|(ranges, _)| ranges)
}

/// `ranges_at`, and whether a failed search decided one of the searcher's
/// decisions its exact range is read from (`search_failure`'s index).
fn ranges_failure(agent: &Agent, spec: &HandSpec, combos: &[[u8; 2]]) -> Result<([Vec<f64>; 2], Option<usize>), String> {
    let steps = agent.action_steps(spec)?;
    let exact_own = agent.search_cfg().is_some_and(|c| c.exact_own_range);
    let at = |i: usize, p: usize, h: [u8; 2]| HandSpec {
        hero: p,
        hole: h,
        board: spec.board[..steps[i].board_len].to_vec(),
        actions: spec.actions[..i].to_vec(),
        ..spec.clone()
    };
    let modelled = |p: usize| -> Result<Vec<f64>, String> {
        let mut w = vec![1.0; combos.len()];
        for i in (0..steps.len()).filter(|&i| steps[i].actor == p) {
            let l = agent.action_likelihoods(&at(i, p, spec.hole), spec.actions[i], combos)?;
            for (w, l) in w.iter_mut().zip(l) {
                *w *= (1.0 - RANGE_FLOOR) * l + RANGE_FLOOR;
            }
        }
        Ok(w)
    };
    let mut out = [Vec::new(), Vec::new()];
    let mut failed = None;
    for (p, range) in out.iter_mut().enumerate() {
        if p == spec.hero && exact_own {
            (*range, failed) = own_range(agent, spec, combos, &steps, &at)?;
        } else {
            *range = modelled(p)?;
        }
        // An exact range that no holding explains (only possible for a
        // hand history the agent could not have played) falls back to the model.
        // A failed search behind that range still counts: it may be why.
        if p == spec.hero && exact_own && range.iter().all(|&w| w <= 0.0) {
            *range = modelled(p)?;
        }
    }
    Ok((out, failed))
}

/// The searcher's exact range: the product of its real action probabilities,
/// and whether a failed search decided one of them (`search_failure`'s index).
fn own_range(
    agent: &Agent,
    spec: &HandSpec,
    combos: &[[u8; 2]],
    steps: &[crate::play::ActionStep],
    at: &(dyn Fn(usize, usize, [u8; 2]) -> HandSpec + Sync),
) -> Result<(Vec<f64>, Option<usize>), String> {
    let p = spec.hero;
    let mut w = vec![1.0; combos.len()];
    let mut failed = None;
    for i in (0..steps.len()).filter(|&i| steps[i].actor == p) {
        let st = &steps[i];
        let (options, f) = agent.options_many_failure(&at(i, p, spec.hole), combos)?;
        failed = failed.max(f);
        for (w, o) in w.iter_mut().zip(&options) {
            // A holding that never takes the real line stays at zero.
            if *w > 0.0 {
                *w *= o
                    .iter()
                    .filter(|(a, _)| canonical_act(*a, st.owed, st.stack) == st.act)
                    .map(|(_, q)| q)
                    .sum::<f64>();
            }
        }
    }
    Ok((w, failed))
}

/// What the agent searches and how long.
#[derive(Clone, Debug)]
pub(crate) struct SearchCfg {
    /// Also search flop decisions: the flop round, its leaves valued by the
    /// external solver's turn-start value net (FLOPSEARCH.md). Needs `external`
    /// unless `flop_checkdown`.
    pub from_flop: bool,
    pub flop_iters: u32,
    /// Solve flop subgames with the built-in `Solver`, which values each leaf
    /// as a check-down to the river (no more betting): the X_eq ablation, and
    /// the exact oracle of the external solver's leaf plumbing.
    pub flop_checkdown: bool,
    /// Also search turn decisions (a turn + every-river-card subgame), not
    /// only river decisions.
    pub from_turn: bool,
    pub river_iters: u32,
    pub turn_iters: u32,
    /// The searcher's own range from its real decisions (see the module doc);
    /// false reproduces the first version, which modelled it from the policy.
    pub exact_own_range: bool,
    /// Solve the current round from its start (see the module doc); false
    /// reproduces the first versions, which forced its real actions.
    pub solve_round: bool,
    /// Solve subgames with this instead of the built-in `Solver` (the GPU
    /// solver, from Python). Same tree, ranges, frozen nodes and iterations.
    pub external: Option<External>,
    /// Where `external` fails on a turn or river subgame, solve that subgame
    /// with the built-in `Solver` (the same tree, ranges, frozen nodes and
    /// iterations): the search still decides. A flop subgame has no built-in
    /// equivalent (its leaves need the value net): the blueprint decides there.
    pub rescue: bool,
}

/// A subgame solver outside this crate: returns the average strategy
/// (action-major, `acts * holdings`) of each slot in `job.want`.
pub(crate) trait SubgameSolver: Send + Sync {
    fn solve(&self, job: &FlatSubgame) -> Result<Vec<Vec<f64>>, String>;
}

#[derive(Clone)]
pub(crate) struct External(pub Arc<dyn SubgameSolver>);

impl std::fmt::Debug for External {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("External")
    }
}

/// River subgame sizes: richer than the blueprint grid.
fn river_menu() -> Menu {
    Menu { bet_fracs: vec![0.33, 0.75, 1.5], raise_fracs: vec![1.0], max_raises: 3 }
}

/// Flop subgame sizes: the turn search's (FLOPSEARCH.md, D3 amendment).
fn flop_menu() -> Menu {
    turn_menu()
}

/// Turn subgame sizes, and the (coarse) river rounds inside a turn subgame.
fn turn_menu() -> Menu {
    Menu { bet_fracs: vec![0.5, 1.0], raise_fracs: vec![1.0], max_raises: 2 }
}

fn turn_river_menu() -> Menu {
    Menu { bet_fracs: vec![0.75], raise_fracs: vec![], max_raises: 1 }
}

/// A solved subgame: the searcher's average strategy at its decision for
/// every holding. It depends only on public information (board, actions,
/// stakes, the per-hand seed), never on the hero's cards, so one solve serves
/// every holding — which range tracking (LBR) relies on.
pub(crate) struct Solved {
    combos: Vec<[u8; 2]>,
    acts: Vec<RealAct>,
    /// Action-major `acts.len() * combos.len()`.
    avg: Vec<f64>,
    /// Both players' ranges at the decision as the solve sees them: its
    /// starting ranges carried along the real line by the strategies there.
    reach: [Vec<f64>; 2],
    /// Where the solve's time went (for the timing harness).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) timing: Timing,
    /// Why the external solver's answer was not used, where the built-in
    /// solver solved the subgame instead (`SearchCfg::rescue`).
    rescued: Option<String>,
    /// Whether a failed search decided one of the searcher's earlier
    /// decisions in the hand that this solve read (for its starting range, or
    /// its strategy frozen this round): `search_failure`'s index.
    earlier: Option<usize>,
}

impl Solved {
    /// The real actions of the solved decision.
    pub(crate) fn acts(&self) -> &[RealAct] {
        &self.acts
    }

    /// The solved strategy of `hole` (either card order), in `acts` order;
    /// `None` for a holding outside the subgame.
    pub(crate) fn row(&self, hole: [u8; 2]) -> Option<Vec<f64>> {
        let hole = if hole[0] < hole[1] { hole } else { [hole[1], hole[0]] };
        let h = self.combos.binary_search(&hole).ok()?;
        let n = self.combos.len();
        Some((0..self.acts.len()).map(|a| self.avg[a * n + h]).collect())
    }

    /// Whether a failed search decided it, as `search_failure`'s index: the
    /// built-in solver in place of the external one (here or at one of the
    /// searcher's earlier decisions in the hand), or the blueprint (there).
    pub(crate) fn failure(&self) -> Option<usize> {
        self.rescued.as_ref().map(|_| 0).max(self.earlier)
    }
}

/// Wall seconds of one solve's phases. Nested solves (the searcher's earlier
/// searched decisions, for its own range) count in the phase that needed them.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Timing {
    /// The searcher's real strategy at its earlier nodes this round.
    pub own: f64,
    /// Card views and the betting tree.
    pub build: f64,
    /// Both players' ranges at the subgame's start.
    pub ranges: f64,
    /// Solver set-up and every DCFR iteration.
    pub solve: f64,
}

/// Solves by public key, failed ones too (a failed search is tried once per
/// public state; one the built-in solver rescued is kept as that solve);
/// cleared when it reaches `CACHE_LIMIT`.
pub(crate) type SearchCache = Mutex<FxHashMap<Vec<i64>, Result<Arc<Solved>, String>>>;
const CACHE_LIMIT: usize = 4096;

/// How a decision's `off_tree_reason` starts where the built-in solver solved
/// the subgame in place of the external one.
pub(crate) const RESCUED: &str = "external solver failed, built-in solver used: ";

/// How a searched decision's `off_tree_reason` ends where a failed search
/// decided one of the searcher's earlier decisions in the hand (its range,
/// or its strategy frozen this round, comes from them), by `search_failure`'s index.
pub(crate) const EARLIER: [&str; 2] = [
    "the searcher's earlier decision in this hand: external solver failed, built-in solver used",
    "the searcher's earlier decision in this hand: search failed, blueprint used",
];

pub(crate) fn applies(cfg: &SearchCfg, spec: &HandSpec) -> bool {
    match spec.board.len() {
        5 => true,
        4 => cfg.from_turn,
        3 => cfg.from_flop,
        _ => false,
    }
}

/// Whether a failed search reached `agent`'s decision `d` at `spec`, as an
/// index into `[built-in solver used in place of the external one (a search
/// decision), blueprint used]`; `None` where no search failed. A search
/// decision also carries a failure at one of the searcher's earlier
/// decisions in the hand that its solve read (`EARLIER`).
pub(crate) fn search_failure(agent: &Agent, spec: &HandSpec, d: &Decision) -> Option<usize> {
    if d.source == Source::Search {
        let r = d.off_tree_reason.as_deref().unwrap_or_default();
        r.starts_with(RESCUED).then_some(0).max((0..2).find(|&k| r.ends_with(EARLIER[k])))
    } else {
        agent.search_cfg().is_some_and(|c| applies(c, spec)).then_some(1)
    }
}

/// Everything `solve` reads but the hero's cards. A valid hand's hero is the
/// player to act, but a solve for the wrong one fails, and that failure must
/// not stand for the right one's.
fn public_key(spec: &HandSpec) -> Vec<i64> {
    let mut k = vec![spec.sb, spec.bb, spec.stacks[0], spec.stacks[1], spec.seed as i64, spec.hero as i64];
    k.extend(spec.board.iter().map(|&c| i64::from(c)));
    k.push(-1);
    for a in &spec.actions {
        match *a {
            RealAct::Fold => k.push(-2),
            RealAct::Check => k.push(-3),
            RealAct::Call => k.push(-4),
            RealAct::Add(d) => k.extend([-5, d]),
        }
    }
    k
}

/// The solved subgame at `spec`'s public state, or why it could not be solved
/// (from the cache when it was already tried).
pub(crate) fn solved(agent: &Agent, cfg: &SearchCfg, cache: &SearchCache, spec: &HandSpec) -> Result<Arc<Solved>, String> {
    let key = public_key(spec);
    if let Some(s) = cache.lock().map_err(|_| "search cache poisoned")?.get(&key) {
        return s.clone();
    }
    let s = solve(agent, cfg, spec).map(Arc::new);
    // A hand refused for its own cards says nothing of the public state.
    if s.is_ok() || agent.validate(spec).is_ok() {
        let mut c = cache.lock().map_err(|_| "search cache poisoned")?;
        if c.len() >= CACHE_LIMIT {
            c.clear();
        }
        c.insert(key, s.clone());
    }
    s
}

/// Both players' ranges over `combos` at `spec` as the agent's decision there
/// sees them: the solve's (`Solved::reach`) where it searches, the model's
/// (`ranges_at`) where it does not or the search fails (the blueprint decides).
pub(crate) fn beliefs(agent: &Agent, spec: &HandSpec, combos: &[[u8; 2]]) -> Result<[Vec<f64>; 2], String> {
    if let Some(s) = agent.searched(spec) {
        let mut out = [vec![0.0; combos.len()], vec![0.0; combos.len()]];
        for (k, h) in combos.iter().enumerate() {
            let h = if h[0] < h[1] { *h } else { [h[1], h[0]] };
            let i = s.combos.binary_search(&h).map_err(|_| "holding outside the solved subgame")?;
            out[0][k] = s.reach[0][i];
            out[1][k] = s.reach[1][i];
        }
        return Ok(out);
    }
    ranges_at(agent, spec, combos)
}

/// The searched decision for `spec`.
pub(crate) fn search_decide(agent: &Agent, cfg: &SearchCfg, cache: &SearchCache, spec: &HandSpec) -> Result<Decision, String> {
    let solved = solved(agent, cfg, cache, spec)?;
    let probs = solved.row(spec.hole).ok_or("hole cards collide with the board")?;
    let r = unit(spec.seed, STREAM_SEARCH, spec.actions.len() as u64);
    let mut acc = 0.0;
    let mut chosen = probs.len() - 1;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            chosen = i;
            break;
        }
    }
    let mut options: Vec<(RealAct, f64)> = Vec::with_capacity(probs.len());
    for (&a, &p) in solved.acts.iter().zip(&probs) {
        match options.iter_mut().find(|(b, _)| *b == a) {
            Some(o) => o.1 += p,
            None => options.push((a, p)),
        }
    }
    Ok(Decision {
        action: solved.acts[chosen],
        options,
        add: 0,
        all_in: false,
        source: Source::Search,
        key: None,
        labels: solved.acts.iter().map(|a| format!("{a:?}")).collect(),
        probs,
        abstract_history: String::new(),
        off_tree_reason: {
            let mut why: Vec<String> = solved.rescued.iter().map(|e| format!("{RESCUED}{e}")).collect();
            why.extend(solved.earlier.map(|k| EARLIER[k].to_string()));
            (!why.is_empty()).then(|| why.join("; "))
        },
    })
}

/// Build and solve the subgame of the current betting round.
fn solve(agent: &Agent, cfg: &SearchCfg, spec: &HandSpec) -> Result<Solved, String> {
    let flop = spec.board.len() == 3;
    let external = if flop && cfg.flop_checkdown { None } else { cfg.external.as_ref() };
    if flop && external.is_none() && !cfg.flop_checkdown {
        return Err("the flop search needs the external solver (its leaves need the value net)".into());
    }
    let ctx = agent.action_context(spec)?;
    let turn = spec.board.len() == 4;
    // The current round's actions start at the first one taken on this board.
    let r = ctx.iter().position(|c| c.1 == spec.board.len()).unwrap_or(ctx.len());
    let real_game = agent.real_game(spec);
    let at_start = HandSpec { actions: spec.actions[..r].to_vec(), ..spec.clone() };
    let (start, dealt) = agent.replay_real(&real_game, &at_start)?;
    if dealt != spec.board.len() {
        return Err(format!("{} board cards given, the betting reached {dealt}", spec.board.len()));
    }
    let stake = [Nlhe::committed_of(&start, 0), Nlhe::committed_of(&start, 1)];
    let stacks = [Nlhe::stack_of(&start, 0), Nlhe::stack_of(&start, 1)];
    let first = Nlhe::acting(&start);
    let mut timing = Timing::default();
    let clock = std::time::Instant::now();
    let b = &spec.board;
    let hands = match b.len() {
        // The built-in solver values the leaves over every turn and river card.
        3 => Hands::flop(&[b[0], b[1], b[2]], external.is_none()),
        4 => Hands::turn(&[b[0], b[1], b[2], b[3]]),
        _ => Hands::river(&[b[0], b[1], b[2], b[3], b[4]]),
    };
    let mut s = start;
    // The round's real actions: (index in spec.actions, actor, action).
    let mut line = Vec::new();
    for (i, &a) in spec.actions.iter().enumerate().skip(r) {
        let actor = Nlhe::acting(&s);
        let (owed, stack) = (Nlhe::owed(&s), Nlhe::stack_of(&s, actor));
        let a = match a {
            RealAct::Add(ALL_IN) => RealAct::Add(stack),
            RealAct::Fold => return Err("a fold before the searcher's decision".into()),
            other => other,
        };
        line.push((i, actor, tree_act(a, owed, stack)));
        s = real_game.apply_real(&s, a)?;
    }
    if Nlhe::betting_street(&s).is_none() || Nlhe::acting(&s) != spec.hero {
        return Err("it is not the searcher's turn".into());
    }
    timing.build = clock.elapsed().as_secs_f64();
    // The searcher's real strategy at each of its earlier nodes this round, per holding.
    let clock = std::time::Instant::now();
    let mut own: Vec<Option<Strategy>> = Vec::with_capacity(line.len());
    let mut earlier = None;
    for &(i, actor, _) in &line {
        own.push(if cfg.solve_round && actor == spec.hero {
            let (strategy, failed) = own_strategy(agent, &HandSpec { actions: spec.actions[..i].to_vec(), ..spec.clone() }, &hands.combos)?;
            earlier = earlier.max(failed);
            Some(strategy)
        } else {
            None
        });
    }
    timing.own = clock.elapsed().as_secs_f64();
    let clock = std::time::Instant::now();
    let forced = line
        .iter()
        .zip(&own)
        .map(|(&(_, actor, a), o)| match o {
            _ if !cfg.solve_round => Forced::Only(a),
            Some(per) => {
                let mut acts: Vec<Act> = Vec::new();
                for &(x, _) in per.iter().flatten() {
                    if !acts.contains(&x) {
                        acts.push(x);
                    }
                }
                Forced::Frozen(a, acts)
            }
            None => {
                debug_assert_ne!(actor, spec.hero);
                Forced::Branch(a)
            }
        })
        .collect();
    let round = RoundSpec {
        stake,
        stacks,
        first,
        min_bet: spec.bb,
        menu: if flop {
            flop_menu()
        } else if turn {
            turn_menu()
        } else {
            river_menu()
        },
        river: turn.then(turn_river_menu),
        forced,
    };
    let tree = Tree::build(&round, &hands)?;
    let (player, acts, slot) = tree.decision(tree.decision_root).ok_or("subgame root is not a decision")?;
    if player != spec.hero {
        return Err("subgame root belongs to the other player".into());
    }
    let owed = Nlhe::owed(&s);
    let acts: Vec<RealAct> = acts
        .iter()
        .map(|a| match *a {
            Act::Fold => RealAct::Fold,
            Act::CheckCall if owed == 0 => RealAct::Check,
            Act::CheckCall => RealAct::Call,
            Act::Bet(d) => RealAct::Add(d),
        })
        .collect();
    for &a in &acts {
        real_game.apply_real(&s, a).map_err(|e| format!("subgame action {a:?} is illegal at the table: {e}"))?;
    }
    timing.build += clock.elapsed().as_secs_f64();
    // Solved round: the ranges at the round's start; forced: at the decision.
    let clock = std::time::Instant::now();
    let (ranges, failed) = ranges_failure(agent, if cfg.solve_round { &at_start } else { spec }, &hands.combos)?;
    earlier = earlier.max(failed);
    timing.ranges = clock.elapsed().as_secs_f64();
    let clock = std::time::Instant::now();
    let n = hands.combos.len();
    let mut frozen = Vec::new();
    for (&(node, _), o) in tree.forced_path.iter().zip(&own) {
        if let Some(per) = o {
            let node_acts = tree.acts_at(node).ok_or("frozen node is not a decision")?;
            let mut strategy = vec![0.0; node_acts.len() * n];
            for (h, opts) in per.iter().enumerate() {
                for &(a, p) in opts {
                    let k = node_acts.iter().position(|&x| x == a).ok_or("frozen action missing from its node")?;
                    strategy[k * n + h] += p;
                }
            }
            frozen.push((node, strategy));
        }
    }
    let iters = if flop {
        cfg.flop_iters
    } else if turn {
        cfg.turn_iters
    } else {
        cfg.river_iters
    };
    // Average strategies: the decision's, then each real-line node's.
    let mut want = vec![slot];
    for &(node, _) in &tree.forced_path {
        want.push(tree.decision(node).ok_or("real-line node is not a decision")?.2);
    }
    let built_in = |frozen: Vec<(usize, Vec<f64>)>| -> Result<Vec<Vec<f64>>, String> {
        let mut solver = Solver::new(&hands, &tree, ranges.clone());
        for (node, strategy) in frozen {
            solver.freeze(node, strategy)?;
        }
        for _ in 0..iters {
            solver.step();
        }
        Ok(want.iter().map(|&s| solver.average(s)).collect())
    };
    let mut rescued = None;
    let avgs = match external {
        Some(ext) => {
            let answer = tree.flatten(&hands, &ranges, &frozen, iters, want.clone()).and_then(|job| ext.0.solve(&job)).and_then(|avgs| {
                let ok = avgs.len() == want.len()
                    && avgs.iter().zip(&want).all(|(a, &s)| a.len() == tree.slot_len(s) * n && a.iter().all(|x| x.is_finite() && *x >= 0.0));
                if ok {
                    Ok(avgs)
                } else {
                    Err("the external solver returned malformed strategies".into())
                }
            });
            match answer {
                Ok(avgs) => avgs,
                Err(e) if cfg.rescue && !flop => {
                    let avgs = built_in(frozen).map_err(|b| format!("{e}; the built-in solver failed too: {b}"))?;
                    rescued = Some(e);
                    avgs
                }
                Err(e) => return Err(e),
            }
        }
        None => built_in(frozen)?,
    };
    timing.solve = clock.elapsed().as_secs_f64();
    let mut reach = ranges;
    for (&(node, i), strat) in tree.forced_path.iter().zip(&avgs[1..]) {
        let player = tree.decision(node).ok_or("real-line node is not a decision")?.0;
        for (h, w) in reach[player].iter_mut().enumerate() {
            *w *= strat[i * n + h];
        }
    }
    let avg = avgs.into_iter().next().ok_or("no strategy at the decision")?;
    Ok(Solved { combos: hands.combos.clone(), acts, avg, reach, timing, rescued, earlier })
}

/// A decision distribution per holding, in tree actions.
type Strategy = Vec<Vec<(Act, f64)>>;

/// A real action as the subgame tree spells it, where its actor owes `owed`
/// and holds `stack`.
fn tree_act(a: RealAct, owed: i64, stack: i64) -> Act {
    match canonical_act(a, owed, stack) {
        RealAct::Fold => Act::Fold,
        RealAct::Check | RealAct::Call => Act::CheckCall,
        RealAct::Add(d) => Act::Bet(d.min(stack)),
    }
}

/// The agent's real decision distribution at `node` (its own earlier decision
/// this round) for every holding in `combos`, in tree actions, and whether a
/// failed search decided it (`search_failure`'s index). After a cache clear
/// it is solved again, and that solve may fail where the played one did not.
fn own_strategy(agent: &Agent, node: &HandSpec, combos: &[[u8; 2]]) -> Result<(Strategy, Option<usize>), String> {
    let (owed, stack) = {
        let real_game = agent.real_game(node);
        let (st, _) = agent.replay_real(&real_game, node)?;
        (Nlhe::owed(&st), Nlhe::stack_of(&st, Nlhe::acting(&st)))
    };
    let (options, failed) = agent.options_many_failure(node, combos)?;
    let strategy = options
        .iter()
        .map(|o| o.iter().map(|&(a, p)| (tree_act(a, owed, stack), p)).collect())
        .collect();
    Ok((strategy, failed))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::nlhe::{Nlhe, NlheConfig, RealAct};
    use crate::play::Strategy;
    use crate::s32::Policy;

    fn holdings(board: &[u8]) -> Vec<[u8; 2]> {
        let used = board.iter().fold(0u64, |m, &c| m | 1u64 << c);
        let mut v = Vec::new();
        for a in 0..52u8 {
            for b in a + 1..52 {
                if used & (1u64 << a) == 0 && used & (1u64 << b) == 0 {
                    v.push([a, b]);
                }
            }
        }
        v
    }

    fn spec(hero: usize, hole: [u8; 2], board: Vec<u8>, actions: Vec<RealAct>) -> HandSpec {
        HandSpec { hero, hole, board, sb: 50, bb: 100, stacks: [10_000, 10_000], actions, seed: 3 }
    }

    #[test]
    fn ranges_are_the_floored_product_of_per_street_likelihoods() {
        // Uniform policy: every likelihood is 1/k at its node, so the range is
        // an explicit product; the flop bet is scored on the flop board.
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let agent = Agent::from_policy(Arc::clone(&game), Policy::uniform());
        let board = vec![0u8, 17, 34];
        // SB limps, BB checks, flop: BB bets half pot (a grid size), SB to act.
        let acts = vec![RealAct::Call, RealAct::Check, RealAct::Add(100)];
        let s = spec(0, [50, 51], board.clone(), acts);
        let combos = holdings(&board);
        let [r0, r1] = ranges_at(&agent, &s, &combos).unwrap();
        let f = |k: f64| (1.0 - RANGE_FLOOR) / k + RANGE_FLOOR;
        let labels = |sp: &HandSpec| agent.decide_inner(sp).unwrap().labels.len() as f64;
        let k_sb = labels(&spec(0, [50, 51], vec![], vec![]));
        let k_bb_pre = labels(&spec(1, [50, 51], vec![], vec![RealAct::Call]));
        let k_bb_flop = labels(&spec(1, [50, 51], board.clone(), vec![RealAct::Call, RealAct::Check]));
        for h in 0..combos.len() {
            assert!((r0[h] - f(k_sb)).abs() < 1e-12, "SB range {} vs {}", r0[h], f(k_sb));
            assert!((r1[h] - f(k_bb_pre) * f(k_bb_flop)).abs() < 1e-12);
        }
    }

    #[test]
    fn ranges_follow_a_hand_dependent_policy() {
        // A table whose row at the SB's first decision depends on the holding:
        // the SB's range after a limp is exactly its (floored) call probability.
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let probe = Agent::new(Arc::clone(&game), Arc::new(Strategy::default()));
        let combos = holdings(&[]);
        let mut table = Strategy::default();
        for h in &combos {
            let d = probe.decide_inner(&spec(0, *h, vec![], vec![])).unwrap();
            let key = d.key.unwrap();
            let w = (key.bytes().map(u32::from).sum::<u32>() % 97) as f32 / 97.0;
            let k = d.labels.len();
            let mut row = vec![(1.0 - w) / (k - 1) as f32; k];
            row[d.labels.iter().position(|l| l == "call").unwrap()] = w;
            table.insert(key.into_bytes().into_boxed_slice(), row);
        }
        let agent = Agent::new(Arc::clone(&game), Arc::new(table));
        let s = spec(1, [50, 51], vec![], vec![RealAct::Call]);
        let [r0, r1] = ranges_at(&agent, &s, &combos).unwrap();
        let mut distinct = std::collections::BTreeSet::new();
        for (h, hole) in combos.iter().enumerate() {
            let d = agent.decide_inner(&spec(0, *hole, vec![], vec![])).unwrap();
            let p: f64 = d.options.iter().filter(|o| o.0 == RealAct::Call).map(|o| o.1).sum();
            let want = (1.0 - RANGE_FLOOR) * p + RANGE_FLOOR;
            assert!((r0[h] - want).abs() < 1e-9, "{hole:?}: {} vs {want}", r0[h]);
            assert_eq!(r1[h], 1.0);
            distinct.insert((r0[h] * 1e6) as i64);
        }
        assert!(distinct.len() > 20, "test policy is not hand-dependent");
    }

    #[test]
    fn an_action_the_model_never_takes_leaves_the_floor_not_zero() {
        // The empty table falls back everywhere and the fallback never raises.
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let agent = Agent::new(Arc::clone(&game), Arc::new(Strategy::default()));
        let s = spec(1, [50, 51], vec![], vec![RealAct::Add(250)]);
        let combos = holdings(&[]);
        let [r0, _] = ranges_at(&agent, &s, &combos).unwrap();
        assert!(r0.iter().all(|&w| (w - RANGE_FLOOR).abs() < 1e-15));
    }

    // ------------------------------------------------------------- search in the agent

    fn splitmix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn searcher(from_turn: bool, iters: u32) -> Agent {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        Agent::from_policy(game, Policy::uniform()).with_search(SearchCfg { from_flop: false, flop_iters: 1, flop_checkdown: false, from_turn, river_iters: iters, turn_iters: iters, exact_own_range: true, solve_round: true, external: None, rescue: true })
    }

    /// Random real hands with arbitrary blinds, stacks and sizes; returns the
    /// decision points whose board has `board_len` cards.
    fn decision_points(seed: u64, hands: usize, board_len: usize) -> Vec<HandSpec> {
        use crate::cards::Card;
        use crate::game::Game;
        let mut z = seed;
        let mut next = move || {
            z = splitmix(z);
            z
        };
        let mut out = Vec::new();
        for _ in 0..hands {
            let mut deck: Vec<u8> = (0..52).collect();
            for i in 0..9 {
                let j = i + (next() % (52 - i as u64)) as usize;
                deck.swap(i, j);
            }
            let bb = [2i64, 100][(next() % 2) as usize];
            let stacks = [bb * (10 + (next() % 200) as i64), bb * (10 + (next() % 200) as i64)];
            let g = Nlhe::new(NlheConfig { sb: bb / 2, bb, stack: stacks[0].min(stacks[1]), ..NlheConfig::pluribus_100bb() });
            let mut s = g.initial_state();
            for &c in &deck[..4] {
                s = g.deal_specific(&s, Card::from_index(c));
            }
            let (mut board, mut acts) = (Vec::new(), Vec::new());
            loop {
                while Nlhe::is_dealing(&s) {
                    board.push(deck[4 + board.len()]);
                    s = g.deal_specific(&s, Card::from_index(*board.last().unwrap()));
                }
                if g.is_terminal(&s) {
                    break;
                }
                let actor = Nlhe::acting(&s);
                if board.len() == board_len {
                    out.push(HandSpec {
                        hero: actor,
                        hole: [deck[2 * actor], deck[2 * actor + 1]],
                        board: board.clone(),
                        sb: bb / 2,
                        bb,
                        stacks,
                        actions: acts.clone(),
                        seed: next(),
                    });
                }
                let (c, stack, minr) = (Nlhe::owed(&s), Nlhe::stack_of(&s, actor), g.min_raise_add(&s));
                // Mostly passive so hands reach the later streets.
                let act = match next() % 8 {
                    0 if c > 0 && board.len() == 5 => RealAct::Fold,
                    0..=4 => if c == 0 { RealAct::Check } else { RealAct::Call },
                    5 if minr < stack => RealAct::Add(minr),
                    6 if minr < stack => RealAct::Add(minr + (next() % (stack - minr) as u64) as i64),
                    _ if board.len() == 5 => RealAct::Add(stack),
                    _ => if c == 0 { RealAct::Check } else { RealAct::Call },
                };
                s = g.apply_real(&s, act).unwrap();
                acts.push(act);
            }
        }
        out
    }

    /// `ranges_at` as it was: every holding replays the whole hand for every
    /// action (the oracle for the batched version).
    fn ranges_reference(agent: &Agent, spec: &HandSpec, combos: &[[u8; 2]]) -> Result<[Vec<f64>; 2], String> {
        let steps = agent.action_steps(spec)?;
        let exact_own = agent.search_cfg().is_some_and(|c| c.exact_own_range);
        let at = |i: usize, p: usize, h: [u8; 2]| HandSpec {
            hero: p,
            hole: h,
            board: spec.board[..steps[i].board_len].to_vec(),
            actions: spec.actions[..i].to_vec(),
            ..spec.clone()
        };
        let one = |p: usize, h: [u8; 2], own: bool| -> Result<f64, String> {
            let mut w = 1.0;
            for i in (0..steps.len()).filter(|&i| steps[i].actor == p) {
                if own {
                    let st = &steps[i];
                    let d = agent.decide_inner(&at(i, p, h))?;
                    w *= d.options.iter().filter(|(a, _)| canonical_act(*a, st.owed, st.stack) == st.act).map(|(_, q)| q).sum::<f64>();
                    if w <= 0.0 {
                        break;
                    }
                } else {
                    let l = agent.action_likelihood(&at(i, p, h), spec.actions[i])?;
                    w *= (1.0 - RANGE_FLOOR) * l + RANGE_FLOOR;
                }
            }
            Ok(w)
        };
        let mut out = [Vec::new(), Vec::new()];
        for (p, range) in out.iter_mut().enumerate() {
            let own = p == spec.hero && exact_own;
            *range = combos.iter().map(|&h| one(p, h, own)).collect::<Result<_, _>>()?;
            if own && range.iter().all(|&w| w <= 0.0) {
                *range = combos.iter().map(|&h| one(p, h, false)).collect::<Result<_, _>>()?;
            }
        }
        Ok(out)
    }

    /// A table with a pseudo-random row for about half the keys `probe` (an
    /// empty table) meets at every decision of `specs`, for every holding:
    /// rows that depend on the hand, and keys that miss and fall back.
    fn half_table(probe: &Agent, specs: &[HandSpec]) -> Strategy {
        let mut table = Strategy::default();
        for spec in specs {
            let steps = probe.action_steps(spec).unwrap();
            for i in 0..=steps.len() {
                let (p, board_len) = steps.get(i).map_or((spec.hero, spec.board.len()), |s| (s.actor, s.board_len));
                for &h in &holdings(&spec.board) {
                    let s = HandSpec { hero: p, hole: h, board: spec.board[..board_len].to_vec(), actions: spec.actions[..i].to_vec(), ..spec.clone() };
                    let Some(key) = probe.decide_inner(&s).ok().and_then(|d| d.key.map(|k| (k, d.labels.len()))) else { continue };
                    let z = key.0.bytes().fold(0u64, |z, b| splitmix(z ^ u64::from(b)));
                    if z % 2 == 0 {
                        // Some zeros, and now and then a row of nothing but.
                        let row = (0..key.1 as u64).map(|k| if z % 11 == 0 { 0.0 } else { (splitmix(z.wrapping_add(k)) % 5) as f32 }).collect();
                        table.insert(key.0.into_bytes().into_boxed_slice(), row);
                    }
                }
            }
        }
        table
    }

    fn bits(options: &[(RealAct, f64)]) -> Vec<(RealAct, u64)> {
        options.iter().map(|&(a, p)| (a, p.to_bits())).collect()
    }

    #[test]
    fn batched_ranges_and_decisions_match_the_per_holding_replay() {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let cfg = SearchCfg { from_flop: false, flop_iters: 1, flop_checkdown: false, from_turn: false, river_iters: 5, turn_iters: 5, exact_own_range: true, solve_round: true, external: None, rescue: true };
        let points: Vec<(usize, Vec<HandSpec>)> =
            [3, 4, 5].map(|n| (n, decision_points(40 + n as u64, 6, n).into_iter().take(4).collect())).into();
        let empty = Agent::new(Arc::clone(&game), Arc::new(Strategy::default()));
        let table = Arc::new(half_table(&empty, &points.iter().flat_map(|p| p.1.clone()).collect::<Vec<_>>()));
        let agents = [
            ("uniform + search", Agent::from_policy(Arc::clone(&game), Policy::uniform()).with_search(cfg.clone())),
            // Every decision falls back: hand-dependent (equity) at every node,
            // and each decision carries its hand-dependent key.
            ("empty table + search", empty.with_search(cfg.clone())),
            ("half table + search", Agent::new(Arc::clone(&game), Arc::clone(&table)).with_search(cfg)),
            ("half table", Agent::new(Arc::clone(&game), table)),
            ("empty table", empty),
        ];
        let mut checked = 0;
        for (name, agent) in &agents {
            // Holdings whose decision read a row, and whose key missed.
            let (mut rows, mut misses) = (0, 0);
            for (_, specs) in &points {
                for spec in specs {
                    let combos = holdings(&spec.board);
                    let at = |h: [u8; 2]| HandSpec { hole: h, ..spec.clone() };
                    let want = ranges_reference(agent, spec, &combos);
                    let got = ranges_at(agent, spec, &combos);
                    assert_eq!(got, want, "{name}: ranges differ at {:?} / {:?}", spec.board, spec.actions);
                    let many = agent.decide_inner_many(spec, &combos).unwrap();
                    let options = agent.options_many(spec, &combos).unwrap();
                    for ((h, d), o) in combos.iter().zip(&many).zip(&options) {
                        let one = agent.decide_inner(&at(*h)).unwrap();
                        assert_eq!(d, &one, "{name}: decision differs for {h:?}");
                        assert_eq!(bits(o), bits(&one.options), "{name}: options differ for {h:?}");
                        rows += usize::from(one.source == Source::Blueprint);
                        misses += usize::from(one.source == Source::MissingKey);
                    }
                    // Every kind of real action, on the grid and off it.
                    let acts = [RealAct::Fold, RealAct::Check, RealAct::Call, RealAct::Add(ALL_IN), RealAct::Add(3 * spec.bb), RealAct::Add(7 * spec.bb + 13)];
                    for act in acts {
                        let many = agent.action_likelihoods(spec, act, &combos).unwrap();
                        for (h, l) in combos.iter().zip(&many).step_by(5) {
                            let one = agent.action_likelihood(&at(*h), act).unwrap();
                            assert_eq!(l.to_bits(), one.to_bits(), "{name}: likelihood of {act:?} differs for {h:?}");
                        }
                    }
                    // A holding on the board is an error either way.
                    let off = (0..52u8).find(|c| !spec.board.contains(c)).unwrap();
                    let bad = [combos[0], [spec.board[0], off]];
                    assert!(agent.decide_inner_many(spec, &bad).is_err() && agent.options_many(spec, &bad).is_err(), "{name}");
                    assert!(agent.action_likelihoods(spec, RealAct::Call, &bad).is_err(), "{name}");
                    checked += 1;
                }
            }
            if *name == "half table" {
                assert!(rows > 1000 && misses > 1000, "{name}: {rows} rows read, {misses} keys missed");
            }
        }
        assert!(checked >= 60, "only {checked} decision points checked");
    }

    #[test]
    fn river_search_engages_and_is_legal_on_arbitrary_real_hands() {
        let agent = searcher(false, 20);
        let points = decision_points(1, 120, 5);
        assert!(points.len() > 80, "only {} river decisions", points.len());
        for spec in &points {
            let d = agent.decide(spec).expect("valid hand");
            assert_eq!(d.source, Source::Search, "{:?}", d.off_tree_reason);
            let total: f64 = d.options.iter().map(|o| o.1).sum();
            assert!((total - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn turn_search_engages_and_is_legal() {
        let agent = searcher(true, 3);
        let points = decision_points(2, 12, 4);
        assert!(points.len() > 6, "only {} turn decisions", points.len());
        for spec in &points {
            let d = agent.decide(spec).expect("valid hand");
            assert_eq!(d.source, Source::Search, "{:?}", d.off_tree_reason);
        }
    }

    /// An external solver that checks the flat subgame's invariants and plays
    /// uniformly (`malformed`: returns one strategy too few).
    struct Uniform {
        malformed: bool,
        jobs: Mutex<(usize, usize)>, // (subgames, frozen slots seen)
    }

    impl SubgameSolver for Uniform {
        fn solve(&self, job: &FlatSubgame) -> Result<Vec<Vec<f64>>, String> {
            let nodes = job.kind.len();
            let n = job.ranges[0].len();
            let views = match job.board.len() {
                3 => 0, // an external flop solver gets no runout views
                4 => 48,
                _ => 1,
            };
            assert_eq!(n, holdings(&job.board).len());
            assert_eq!((job.strength.len(), job.alive.len()), (views * n, views * n));
            assert_eq!(job.child_start.len(), nodes + 1);
            assert_eq!(job.kind[0], 1, "the root is a decision");
            let kids = |i: usize| job.children[job.child_start[i] as usize..job.child_start[i + 1] as usize].to_vec();
            let mut slot_node = vec![usize::MAX; nodes];
            for i in 0..nodes {
                let ks = kids(i);
                assert!(ks.iter().all(|&c| c as usize > i), "a child precedes its parent");
                match job.kind[i] {
                    0 => assert!(ks.is_empty() && job.player[i] >= -1 && job.player[i] <= 1),
                    1 => {
                        assert!(!ks.is_empty() && (job.player[i] == 0 || job.player[i] == 1));
                        slot_node[job.slot[i] as usize] = i;
                    }
                    2 => assert_eq!((ks.len(), job.norm, job.board.len()), (48, 44.0, 4)),
                    k => {
                        // A flop leaf (chips behind, equal stakes) or runout (someone all-in).
                        let (st, tot) = (job.stake[i], job.total);
                        assert!(ks.is_empty() && job.board.len() == 3 && job.player[i] == -1);
                        assert!(st[0] <= tot[0] && st[1] <= tot[1] && st[0] > 0 && st[1] > 0);
                        let all_in = st[0] == tot[0] || st[1] == tot[1];
                        match k {
                            3 => assert!(!all_in && st[0] == st[1], "leaf {st:?} of {tot:?}"),
                            4 => assert!(all_in, "runout {st:?} of {tot:?}"),
                            _ => panic!("unknown node kind {k}"),
                        }
                    }
                }
            }
            let slots = job.slot.iter().filter(|&&s| s >= 0).count();
            assert!(slot_node[..slots].iter().all(|&i| i < nodes), "slots are 0..decisions");
            for (s, strat) in &job.frozen {
                let k = kids(slot_node[*s]).len();
                assert_eq!(strat.len(), k * n);
                for h in 0..n {
                    assert!(((0..k).map(|a| strat[a * n + h]).sum::<f64>() - 1.0).abs() < 1e-9);
                }
            }
            let mut j = self.jobs.lock().unwrap();
            j.0 += 1;
            j.1 += job.frozen.len();
            let mut out: Vec<Vec<f64>> = job
                .want
                .iter()
                .map(|&s| {
                    let k = kids(slot_node[s]).len();
                    vec![1.0 / k as f64; k * n]
                })
                .collect();
            if self.malformed {
                out.pop();
            }
            Ok(out)
        }
    }

    #[test]
    fn an_external_solver_gets_the_whole_subgame_and_decides() {
        for (board_len, from_turn) in [(5, false), (4, true), (3, false)] {
            let ext = Arc::new(Uniform { malformed: false, jobs: Mutex::new((0, 0)) });
            let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
            let cfg = SearchCfg {
                from_flop: board_len == 3,
                flop_iters: 3,
                flop_checkdown: false,
                from_turn,
                river_iters: 3,
                turn_iters: 3,
                exact_own_range: true,
                solve_round: true,
                external: Some(External(ext.clone())),
                rescue: true,
            };
            let agent = Agent::from_policy(game, Policy::uniform()).with_search(cfg);
            let points = decision_points(7, 20, board_len);
            for spec in &points {
                let d = agent.decide(spec).expect("valid hand");
                assert_eq!(d.source, Source::Search, "{:?}", d.off_tree_reason);
                assert!(d.off_tree_reason.is_none(), "rescued: {:?}", d.off_tree_reason); // the external solver's answer
                assert!(d.probs.iter().all(|&p| (p - 1.0 / d.probs.len() as f64).abs() < 1e-12));
                assert_eq!(agent.options_many_failure(spec, &holdings(&spec.board)).unwrap().1, None, "board {board_len}");
            }
            let (jobs, frozen) = *ext.jobs.lock().unwrap();
            assert!(jobs > 5, "board {board_len}: only {jobs} subgames");
            if board_len == 3 {
                // Flop search off (the default): the same spots are not searched.
                let plain = Agent::from_policy(Arc::new(Nlhe::new(NlheConfig::pluribus_100bb())), Policy::uniform())
                    .with_search(SearchCfg { from_flop: false, external: Some(External(ext.clone())), ..agent.search_cfg().unwrap().clone() });
                assert!(points.iter().all(|s| plain.decide(s).unwrap().source != Source::Search));
                // Without an external solver the flop search fails: the blueprint decides.
                let cpu = Agent::from_policy(Arc::new(Nlhe::new(NlheConfig::pluribus_100bb())), Policy::uniform())
                    .with_search(SearchCfg { external: None, ..agent.search_cfg().unwrap().clone() });
                let d = cpu.decide(&points[0]).unwrap();
                assert_ne!(d.source, Source::Search);
                assert!(d.off_tree_reason.unwrap().contains("external solver"));
                // Check-down leaves: the built-in solver searches the flop; the
                // external one is not asked.
                let before = ext.jobs.lock().unwrap().0;
                let eq = Agent::from_policy(Arc::new(Nlhe::new(NlheConfig::pluribus_100bb())), Policy::uniform())
                    .with_search(SearchCfg { flop_checkdown: true, flop_iters: 2, ..agent.search_cfg().unwrap().clone() });
                for spec in &points[..3] {
                    let d = eq.decide(spec).unwrap();
                    assert_eq!(d.source, Source::Search, "{:?}", d.off_tree_reason);
                    assert!((d.probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                }
                assert_eq!(ext.jobs.lock().unwrap().0, before);
            }
            if board_len == 5 {
                assert!(frozen > 0, "no frozen node reached the external solver");
            }
        }
    }

    /// A search config with `ext` as the external solver, from the turn when
    /// `board_len` is 4, the flop when 3.
    fn external_cfg(board_len: usize, ext: &Arc<Uniform>, rescue: bool) -> SearchCfg {
        SearchCfg {
            from_flop: board_len == 3,
            flop_iters: 3,
            flop_checkdown: false,
            from_turn: board_len == 4,
            river_iters: 3,
            turn_iters: 3,
            exact_own_range: true,
            solve_round: true,
            external: Some(External(ext.clone())),
            rescue,
        }
    }

    fn without_reason(d: Decision) -> Decision {
        Decision { off_tree_reason: None, ..d }
    }

    #[test]
    fn a_failed_search_is_tried_once_per_public_state_and_the_blueprint_decides() {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let plain = Agent::from_policy(game, Policy::uniform());
        let mut earlier_seen = 0;
        for board_len in [5, 4, 3] {
            for spec in decision_points(7, 20, board_len).iter().take(4) {
                // An answer the engine refuses; no rescue (on the flop there is none).
                let ext = Arc::new(Uniform { malformed: true, jobs: Mutex::new((0, 0)) });
                let agent = plain.with_search(external_cfg(board_len, &ext, board_len == 3));
                let combos = holdings(&spec.board);
                let options = agent.options_many(spec, &combos).unwrap();
                // Reported as the blueprint's (`PlayAgent.options` raises on it).
                assert_eq!(agent.options_many_failure(spec, &combos).unwrap().1, Some(1), "board {board_len}");
                // One solve per public state the search met: this decision and
                // the searcher's earlier ones this round (its own strategy there).
                let earlier = agent.action_context(spec).unwrap().iter().filter(|&&(actor, len)| actor == spec.hero && len == board_len).count();
                earlier_seen += earlier;
                let calls = || ext.jobs.lock().unwrap().0;
                assert_eq!(calls(), 1 + earlier, "board {board_len}");
                let many = agent.decide_inner_many(spec, &combos).unwrap();
                let want = plain.decide_inner_many(spec, &combos).unwrap();
                for (((h, d), o), w) in combos.iter().zip(many).zip(&options).zip(want) {
                    let one = agent.decide_inner(&HandSpec { hole: *h, ..spec.clone() }).unwrap();
                    let reason = one.off_tree_reason.as_deref().unwrap_or_default();
                    assert!(reason.starts_with("search failed, blueprint used: ") && reason.contains("malformed"), "{reason}");
                    assert_eq!(bits(o), bits(&w.options), "{h:?}");
                    assert_eq!(d, one, "{h:?}");
                    let probs = |d: &Decision| d.probs.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
                    assert_eq!(probs(&d), probs(&w), "{h:?}");
                    assert_eq!(without_reason(d), without_reason(w), "{h:?}");
                }
                assert!(agent.searched(spec).is_none());
                assert_eq!(beliefs(&agent, spec, &combos).unwrap(), ranges_at(&agent, spec, &combos).unwrap());
                assert_eq!(calls(), 1 + earlier, "board {board_len}: a failed search was tried again");
            }
        }
        assert!(earlier_seen > 0, "no searcher's earlier decision in the round was met");
    }

    /// `Uniform`, but its first answer is malformed: the first subgame solved fails.
    struct FailsFirst(Uniform, std::sync::atomic::AtomicUsize);

    impl SubgameSolver for FailsFirst {
        fn solve(&self, job: &FlatSubgame) -> Result<Vec<Vec<f64>>, String> {
            let mut out = self.0.solve(job)?;
            if self.1.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                out.pop();
            }
            Ok(out)
        }
    }

    #[test]
    fn a_failed_search_at_the_searchers_earlier_decision_is_reported_at_this_one() {
        // A fresh agent solves the searcher's earlier decisions again (as
        // after a cache clear); the earliest is solved first, and fails. This
        // decision's own solve succeeds and still decides. The earlier decision
        // is this round's (its strategy frozen), or only the turn's for a river
        // decision (its range: a checked-down hand, which every holding
        // explains; the uniform blueprint's hands may hold bets off the
        // search's menu, and then the range is the model's, which reads no search).
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        let checked_down = vec![spec(1, [50, 51], vec![0, 17, 34, 7, 22], [RealAct::Call].into_iter().chain([RealAct::Check; 5]).collect())];
        for (board_len, from_turn, round) in [(5, false, true), (4, true, true), (5, true, false)] {
            let mut met = 0;
            for spec in &(if round { decision_points(7, 20, board_len) } else { checked_down.clone() }) {
                let context = Agent::from_policy(Arc::clone(&game), Policy::uniform()).action_context(spec).unwrap();
                let hero_at = |len: usize| context.iter().any(|&(actor, l)| actor == spec.hero && l == len);
                if hero_at(board_len) != round || (!round && !hero_at(4)) {
                    continue;
                }
                met += 1;
                let combos = holdings(&spec.board);
                for rescue in [true, false] {
                    let agent = |first_fails: bool| {
                        let ext = Arc::new(Uniform { malformed: false, jobs: Mutex::new((0, 0)) });
                        let cfg = SearchCfg { from_turn, ..external_cfg(board_len, &ext, rescue) };
                        let fails = FailsFirst(Uniform { malformed: false, jobs: Mutex::new((0, 0)) }, Default::default());
                        let cfg = if first_fails { SearchCfg { external: Some(External(Arc::new(fails))), ..cfg } } else { cfg };
                        Agent::from_policy(Arc::clone(&game), Policy::uniform()).with_search(cfg)
                    };
                    // The built-in solver there (rescue), else the blueprint.
                    let k = usize::from(!rescue);
                    let at = format!("board {board_len}, from_turn {from_turn}, round {round}, rescue {rescue}");
                    assert_eq!(agent(false).options_many_failure(spec, &combos).unwrap().1, None, "{at}");
                    assert_eq!(agent(true).options_many_failure(spec, &combos).unwrap().1, Some(k), "{at}");
                    let fresh = agent(true);
                    let d = fresh.decide_inner(spec).unwrap();
                    assert_eq!(d.source, Source::Search, "{at}: {:?}", d.off_tree_reason);
                    assert_eq!(search_failure(&fresh, spec, &d), Some(k), "{at}: {:?}", d.off_tree_reason);
                    assert_eq!(d.off_tree_reason.as_deref(), Some(EARLIER[k]), "{at}");
                    // Rescued here and failed earlier: the larger index.
                    let reread = |why: Option<String>| {
                        let d = fresh.decide_inner(spec).unwrap();
                        search_failure(&fresh, spec, &Decision { off_tree_reason: why, ..d })
                    };
                    assert_eq!(reread(None), None);
                    assert_eq!(reread(Some(format!("{RESCUED}x"))), Some(0));
                    assert_eq!(reread(Some(format!("{RESCUED}x; {}", EARLIER[0]))), Some(0));
                    assert_eq!(reread(Some(format!("{RESCUED}x; {}", EARLIER[1]))), Some(1));
                }
            }
            assert!(met > 0, "board {board_len}, from_turn {from_turn}, round {round}: no such decision met");
        }
    }

    #[test]
    fn a_hand_refused_for_its_own_cards_or_seat_does_not_stop_the_search() {
        let agent = searcher(false, 3);
        for spec in decision_points(7, 20, 5).iter().take(4) {
            let off = (0..52u8).find(|c| !spec.board.contains(c) && !spec.hole.contains(c)).unwrap();
            assert!(agent.decide_inner(&HandSpec { hole: [spec.board[0], off], ..spec.clone() }).is_err());
            // The wrong seat: its solve fails (whatever the blueprint makes of it).
            let _ = agent.decide_inner(&HandSpec { hero: 1 - spec.hero, ..spec.clone() });
            let d = agent.decide_inner(spec).unwrap();
            assert_eq!(d.source, Source::Search, "{:?}", d.off_tree_reason);
        }
    }

    #[test]
    fn a_rescued_search_is_the_built_in_search_bit_for_bit() {
        let game = Arc::new(Nlhe::new(NlheConfig::pluribus_100bb()));
        for board_len in [5, 4] {
            let ext = Arc::new(Uniform { malformed: true, jobs: Mutex::new((0, 0)) });
            let cfg = external_cfg(board_len, &ext, true);
            let rescued = Agent::from_policy(Arc::clone(&game), Policy::uniform()).with_search(cfg.clone());
            let built_in = Agent::from_policy(Arc::clone(&game), Policy::uniform()).with_search(SearchCfg { external: None, ..cfg });
            for spec in decision_points(7, 20, board_len).iter().take(4) {
                let combos = holdings(&spec.board);
                let got = rescued.decide_inner_many(spec, &combos).unwrap();
                let want = built_in.decide_inner_many(spec, &combos).unwrap();
                for (d, w) in got.into_iter().zip(want) {
                    let reason = d.off_tree_reason.as_deref().unwrap_or_default();
                    assert!(reason.starts_with(RESCUED) && reason.contains("malformed"), "{reason}");
                    assert_eq!(d.source, Source::Search);
                    assert_eq!(bits(&d.options), bits(&w.options));
                    assert_eq!(d.probs.iter().map(|p| p.to_bits()).collect::<Vec<_>>(), w.probs.iter().map(|p| p.to_bits()).collect::<Vec<_>>());
                    assert_eq!(without_reason(d), w);
                }
                assert_eq!(beliefs(&rescued, spec, &combos).unwrap(), beliefs(&built_in, spec, &combos).unwrap());
                // The same options; only the rescued ones are reported as a failure.
                let (got, failed) = rescued.options_many_failure(spec, &combos).unwrap();
                let (want, none) = built_in.options_many_failure(spec, &combos).unwrap();
                assert_eq!((failed, none), (Some(0), None), "board {board_len}");
                assert_eq!(got.len(), want.len());
                assert!(got.iter().zip(&want).all(|(g, w)| bits(g) == bits(w)));
            }
            assert!(ext.jobs.lock().unwrap().0 > 0, "board {board_len}: the external solver was never asked");
        }
        // The flop has no built-in equivalent: the blueprint decides there.
        let ext = Arc::new(Uniform { malformed: true, jobs: Mutex::new((0, 0)) });
        let agent = Agent::from_policy(game, Policy::uniform()).with_search(external_cfg(3, &ext, true));
        let spec = &decision_points(7, 20, 3)[0];
        let d = agent.decide(spec).expect("valid hand");
        assert_ne!(d.source, Source::Search);
        assert!(d.off_tree_reason.unwrap().contains("malformed"));
        assert_eq!(agent.options_many_failure(spec, &holdings(&spec.board)).unwrap().1, Some(1));
    }

    /// Where a searched decision's time goes, on the real blueprint, at 1
    /// thread and at all threads. Run it on an otherwise idle machine:
    ///   cargo test --no-default-features --release --lib -- --ignored --nocapture print_search_timing
    /// PARCIVER_BLUEPRINT and PARCIVER_ABSTRACTION override the default paths.
    /// The real blueprint's agent (PARCIVER_BLUEPRINT / PARCIVER_ABSTRACTION,
    /// else the 100bb v1 blueprint), or `None` when its files are absent.
    pub(crate) fn real_blueprint_agent() -> Option<Agent> {
        use std::path::PathBuf;
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        let path = |var: &str, default: &str| std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| root.join(default));
        let bp = path("PARCIVER_BLUEPRINT", "runs/blueprint-20260914-144924/blueprint.bin");
        let abs_dir = path("PARCIVER_ABSTRACTION", "data/abstraction");
        if !bp.exists() || !abs_dir.join("river_buckets.npy").exists() {
            eprintln!("blueprint or abstraction absent ({bp:?}, {abs_dir:?}); skipping");
            return None;
        }
        let tag = crate::mccfr::read_strategy_tag(&bp).unwrap();
        let cfg = NlheConfig::grid(&crate::grid_name(&tag)).unwrap();
        let abs = crate::bucketing::Abstraction::load(&abs_dir).unwrap();
        let game = Arc::new(Nlhe::with_abstraction(cfg, Arc::new(abs)));
        println!("blueprint {bp:?}");
        Some(Agent::new(game, Arc::new(crate::mccfr::load_average_strategy(&bp).unwrap())))
    }

    /// The batched range oracle on the real blueprint (bucketed keys, real rows):
    ///   cargo test --no-default-features --release --lib -- --ignored batched_ranges_on_the_real_blueprint
    #[test]
    #[ignore]
    fn batched_ranges_on_the_real_blueprint() {
        let Some(base) = real_blueprint_agent() else { return };
        let cfg = SearchCfg { from_flop: false, flop_iters: 1, flop_checkdown: false, from_turn: false, river_iters: 5, turn_iters: 5, exact_own_range: true, solve_round: true, external: None, rescue: true };
        let agent = base.with_search(cfg);
        let mut checked = 0;
        for board_len in [3, 4, 5] {
            for spec in decision_points(90 + board_len as u64, 30, board_len).iter().filter(|s| s.bb == 100).take(5) {
                let combos = holdings(&spec.board);
                assert_eq!(ranges_at(&agent, spec, &combos), ranges_reference(&agent, spec, &combos));
                checked += 1;
            }
        }
        assert!(checked >= 10, "only {checked} decision points checked");
    }

    #[test]
    #[ignore]
    fn print_search_timing() {
        let Some(base) = real_blueprint_agent() else { return };
        for (street, board_len, count) in [("river", 5, 24), ("turn", 4, 6)] {
            let cfg = SearchCfg { from_flop: false, flop_iters: 1, flop_checkdown: false, from_turn: board_len == 4, river_iters: 200, turn_iters: 100, exact_own_range: true, solve_round: true, external: None, rescue: true };
            let points: Vec<HandSpec> =
                decision_points(7, 600, board_len).into_iter().filter(|s| s.bb == 100).take(count).collect();
            for threads in [1, 0] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                let (wall, sum, solved) = pool.install(|| {
                    // A fresh agent: nothing cached from the other thread count.
                    let agent = base.with_search(cfg.clone());
                    let clock = std::time::Instant::now();
                    let mut sum = Timing::default();
                    let mut solved = 0;
                    for spec in &points {
                        if let Some(s) = agent.searched(spec) {
                            solved += 1;
                            sum.own += s.timing.own;
                            sum.build += s.timing.build;
                            sum.ranges += s.timing.ranges;
                            sum.solve += s.timing.solve;
                        }
                    }
                    (clock.elapsed().as_secs_f64(), sum, solved)
                });
                let ms = |x: f64| 1000.0 * x / solved.max(1) as f64;
                println!(
                    "{street:5} {:>2} threads, {solved}/{} solved: {:8.1} ms/decision = own {:.1} + build {:.1} + ranges {:.1} + solve {:.1} (+ rest {:.1})",
                    pool.current_num_threads(),
                    points.len(),
                    ms(wall),
                    ms(sum.own),
                    ms(sum.build),
                    ms(sum.ranges),
                    ms(sum.solve),
                    ms(wall - sum.own - sum.build - sum.ranges - sum.solve),
                );
            }
        }
    }

    /// Range building alone, one thread, as a river solve asks for it (at the
    /// round's start): the first pass fills the fallback's equity cache, the
    /// second reads it.
    #[test]
    #[ignore]
    fn print_ranges_timing() {
        let Some(base) = real_blueprint_agent() else { return };
        let cfg = SearchCfg { from_flop: false, flop_iters: 1, flop_checkdown: false, from_turn: false, river_iters: 200, turn_iters: 100, exact_own_range: true, solve_round: true, external: None, rescue: true };
        let agent = base.with_search(cfg);
        let starts: Vec<HandSpec> = decision_points(7, 600, 5)
            .into_iter()
            .filter(|s| s.bb == 100)
            .take(24)
            .map(|spec| {
                let ctx = agent.action_context(&spec).unwrap();
                let r = ctx.iter().position(|c| c.1 == spec.board.len()).unwrap_or(ctx.len());
                HandSpec { actions: spec.actions[..r].to_vec(), ..spec }
            })
            .collect();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| {
            for pass in ["cold", "warm"] {
                let mut each = Vec::new();
                let clock = std::time::Instant::now();
                for s in &starts {
                    let c = std::time::Instant::now();
                    ranges_at(&agent, s, &holdings(&s.board)).unwrap();
                    each.push(format!("{:.0}", 1000.0 * c.elapsed().as_secs_f64()));
                }
                let ms = 1000.0 * clock.elapsed().as_secs_f64() / starts.len() as f64;
                println!("ranges {pass}: {ms:.1} ms/decision [{}]", each.join(" "));
            }
        });
    }

    #[test]
    fn one_solve_serves_every_holding_at_a_public_state() {
        let agent = searcher(false, 20);
        let spec = decision_points(3, 20, 5).into_iter().next().unwrap();
        let first = agent.decide_inner(&spec).unwrap();
        let used: Vec<u8> = spec.board.clone();
        let mut n = 0;
        for a in 0..52u8 {
            for b in a + 1..52 {
                if used.contains(&a) || used.contains(&b) || n > 40 {
                    continue;
                }
                let d = agent.decide_inner(&HandSpec { hole: [a, b], ..spec.clone() }).unwrap();
                assert_eq!(d.labels, first.labels);
                n += 1;
            }
        }
        assert_eq!(agent_cache_len(&agent), 1);
    }

    #[test]
    fn exact_own_range_is_the_searchers_real_strategy() {
        // Limped and checked to the river; BB checks the river (a searched
        // decision), SB bets pot, BB decides again. BB's range at the second
        // decision must carry the search's check probability per holding.
        let board = vec![0u8, 9, 22, 31, 44];
        let combos = holdings(&board);
        let mut acts = vec![RealAct::Call, RealAct::Check, RealAct::Check, RealAct::Check, RealAct::Check, RealAct::Check];
        let first = spec(1, [50, 51], board.clone(), acts.clone());
        let agent = searcher(false, 40);
        let check: Vec<f64> = combos
            .iter()
            .map(|&h| {
                let d = agent.decide_inner(&HandSpec { hole: h, ..first.clone() }).unwrap();
                assert_eq!(d.source, Source::Search);
                d.options.iter().filter(|o| o.0 == RealAct::Check).map(|o| o.1).sum()
            })
            .collect();
        acts.extend([RealAct::Check, RealAct::Add(200)]);
        let second = spec(1, [50, 51], board.clone(), acts);
        let [opp, own] = ranges_at(&agent, &second, &combos).unwrap();
        // Uniform policy before the river: the same factor for every holding,
        // so own / check is one constant wherever the search checks at all.
        let ratios: Vec<f64> = own.iter().zip(&check).filter(|(_, &c)| c > 1e-6).map(|(o, c)| o / c).collect();
        assert!(ratios.len() > 100);
        for r in &ratios {
            assert!((r / ratios[0] - 1.0).abs() < 1e-9, "{r} vs {}", ratios[0]);
        }
        for (o, c) in own.iter().zip(&check) {
            if *c <= 1e-6 {
                assert!(*o <= ratios[0] * 1e-6 + 1e-300);
            }
        }
        let distinct: std::collections::BTreeSet<i64> = check.iter().map(|c| (c * 1e6) as i64).collect();
        assert!(distinct.len() > 10, "search's check strategy is not hand-dependent");
        // The first version models the own range from the (uniform) policy:
        // flat across holdings. The opponent's range is modelled either way.
        let v1 = Agent::from_policy(Arc::new(Nlhe::new(NlheConfig::pluribus_100bb())), Policy::uniform())
            .with_search(SearchCfg { from_flop: false, flop_iters: 1, flop_checkdown: false, from_turn: false, river_iters: 40, turn_iters: 40, exact_own_range: false, solve_round: false, external: None, rescue: true });
        let [opp1, own1] = ranges_at(&v1, &second, &combos).unwrap();
        assert!(own1.iter().all(|&w| (w - own1[0]).abs() < 1e-12));
        assert_eq!(opp, opp1);
    }

    #[test]
    fn solving_the_round_defends_an_off_menu_river_bet_near_mdf() {
        // Limped and checked to the river; BB checks the river (searched, then
        // frozen), SB bets pot (200 into 200: off the 0.33/0.75/1.5 menu), BB
        // decides. With the round solved, SB's bet is read through the solve's
        // equilibrium: BB continues about pot / (pot + bet) = 1/2 of its range.
        let board = vec![0u8, 9, 22, 31, 44];
        let combos = holdings(&board);
        let mut acts = vec![RealAct::Call, RealAct::Check, RealAct::Check, RealAct::Check, RealAct::Check, RealAct::Check];
        acts.extend([RealAct::Check, RealAct::Add(200)]);
        let at = spec(1, [50, 51], board.clone(), acts);
        let agent = searcher(false, 300);
        let [_, own] = beliefs(&agent, &at, &combos).unwrap();
        // The solve's own range on the real line is the true one (the exact
        // own range: the frozen check strategy is the one really used).
        let [_, exact] = ranges_at(&agent, &at, &combos).unwrap();
        let (so, se): (f64, f64) = (own.iter().sum(), exact.iter().sum());
        assert!(so > 0.0 && se > 0.0);
        let tv: f64 = 0.5 * own.iter().zip(&exact).map(|(o, e)| (o / so - e / se).abs()).sum::<f64>();
        assert!(tv < 1e-9, "own range on the real line differs from the exact one: TV {tv}");
        let mut cont = 0.0;
        for (h, &w) in combos.iter().zip(&own) {
            let d = agent.decide_inner(&HandSpec { hole: *h, ..at.clone() }).unwrap();
            assert_eq!(d.source, Source::Search, "{:?}", d.off_tree_reason);
            let fold: f64 = d.options.iter().filter(|o| o.0 == RealAct::Fold).map(|o| o.1).sum();
            cont += w * (1.0 - fold);
        }
        let cont = cont / so;
        assert!((0.35..=0.65).contains(&cont), "BB continues {cont:.3} of its range facing a pot bet");
    }

    fn agent_cache_len(agent: &Agent) -> usize {
        agent.search_cache_len()
    }

    #[test]
    fn facing_a_river_shove_the_nuts_calls_and_air_folds() {
        use crate::cards::Card;
        use crate::hand_eval::eval7_fast;
        let agent = searcher(false, 300);
        let board = vec![0u8, 9, 22, 31, 44];
        let c = |i: u8| Card::from_index(i);
        let strength = |h: [u8; 2]| {
            eval7_fast(&[c(h[0]), c(h[1]), c(board[0]), c(board[1]), c(board[2]), c(board[3]), c(board[4])])
        };
        let combos = holdings(&board);
        let best = *combos.iter().max_by_key(|&&h| strength(h)).unwrap();
        let worst = *combos.iter().min_by_key(|&&h| strength(h)).unwrap();
        // Limped to the river, checked through; on the river BB shoves, SB decides.
        let mut acts = vec![RealAct::Call, RealAct::Check];
        acts.extend([RealAct::Check, RealAct::Check, RealAct::Check, RealAct::Check]);
        acts.push(RealAct::Add(9_900));
        let call_prob = |hole: [u8; 2]| {
            let d = agent.decide_inner(&spec(0, hole, board.clone(), acts.clone())).unwrap();
            assert_eq!(d.source, Source::Search, "{:?}", d.off_tree_reason);
            d.options.iter().filter(|o| o.0 == RealAct::Call).map(|o| o.1).sum::<f64>()
        };
        assert!(call_prob(best) > 0.99, "nuts call prob {}", call_prob(best));
        assert!(call_prob(worst) < 0.01, "worst-hand call prob {}", call_prob(worst));
    }
}
