//! Extensive-form game interface, the Rust mirror of `parciver.games.base.Game`.
//!
//! The solver moves through states only via this trait, so one CFR
//! implementation serves every game. Monomorphised generics make the
//! abstraction zero-cost — the branchy tree traversal this buys is exactly
//! where Rust beats the array-oriented Python/numpy path.
//!
//! Two hard rules mirror the Python contract and must not drift, or the
//! Python exploitability oracle (which cross-validates this engine) breaks:
//!   * utilities are always from player 0's perspective, and
//!   * `infoset_key` strings and legal-action ORDER match the Python games
//!     byte for byte, so a strategy solved here can be scored by that oracle.

pub trait Game {
    /// Concrete, cheaply clonable game state.
    type State: Clone;

    fn initial_state(&self) -> Self::State;

    fn is_terminal(&self, s: &Self::State) -> bool;
    /// Payoff to player 0 at a terminal state.
    fn terminal_utility(&self, s: &Self::State) -> f64;

    /// Number of players. Defaults to 2 (the zero-sum games this crate was built
    /// on); N-player games override it. Kept as a defaulted method so every
    /// existing 2p game (Kuhn, Leduc, NLHE) is UNCHANGED and every green 2p gate
    /// stays byte-identical.
    fn num_players(&self) -> usize {
        2
    }

    /// Payoff to `player` at a terminal state. The default is the 2-player
    /// zero-sum relation derived from the player-0 payoff, so it reproduces the
    /// old hard-coded `if p==0 { u } else { -u }` EXACTLY (byte-identical floats)
    /// for every existing game. N-player general-sum games override this with the
    /// real per-player payoff.
    fn terminal_utility_for(&self, s: &Self::State, player: usize) -> f64 {
        let u = self.terminal_utility(s);
        if player == 0 {
            u
        } else {
            -u
        }
    }

    fn is_chance(&self, s: &Self::State) -> bool;
    /// Materialised (child_state, probability) pairs; probabilities sum to 1.
    fn chance_outcomes(&self, s: &Self::State) -> Vec<(Self::State, f64)>;

    /// Sample ONE chance child: `choose` receives the outcome probabilities in
    /// [`Game::chance_outcomes`] order and returns the chosen index, and the child
    /// at that index is returned. Exactly `chance_outcomes(s)[choose(probs)].0` —
    /// same probabilities, same order — so a sampled walk is byte-identical either
    /// way. The default materializes every child; games whose chance nodes are wide
    /// (hold'em deals up to 52 cards) override it to build only the chosen child.
    fn sample_chance_with<F: FnOnce(&[f64]) -> usize>(
        &self,
        s: &Self::State,
        choose: F,
    ) -> Self::State {
        let mut outcomes = self.chance_outcomes(s);
        let probabilities: Vec<f64> = outcomes.iter().map(|(_, p)| *p).collect();
        let idx = choose(&probabilities);
        outcomes.swap_remove(idx).0
    }

    /// Acting player (`0..num_players()`) at a decision node.
    fn current_player(&self, s: &Self::State) -> usize;
    /// Number of legal actions at a decision node.
    fn num_actions(&self, s: &Self::State) -> usize;
    /// Child after taking the `action`-th legal action. `action` indexes the
    /// legal-action list in the SAME order the Python game exposes it.
    fn next_state(&self, s: &Self::State, action: usize) -> Self::State;

    /// A decision node's action count and its children from ONE pass over the
    /// legal actions: `f(n, child)` gets `num_actions(s)` and a `child(a)` that
    /// returns exactly `next_state(s, a)`. The default calls those two methods; a
    /// game whose legal-action list costs real work (NLHE) overrides it to build
    /// the list once per node instead of once more per child.
    fn with_children<R>(
        &self,
        s: &Self::State,
        f: impl FnOnce(usize, &dyn Fn(usize) -> Self::State) -> R,
    ) -> R {
        f(self.num_actions(s), &|a| self.next_state(s, a))
    }

    /// Information-set identifier for the acting player. Must equal the Python
    /// game's key for the corresponding state exactly.
    fn infoset_key(&self, s: &Self::State) -> String;

    /// Write the infoset key BYTES into `buf` (which the caller has cleared). The
    /// bytes must be exactly `infoset_key(s).as_bytes()`. The MCCFR hot loop calls
    /// this into a reused scratch buffer so a repeat visit allocates no `String` —
    /// the dominant per-node cost on the abstracted hold'em tree (13 M+ keys). The
    /// default delegates to [`Game::infoset_key`] (an allocation per call); games on
    /// the hot path override it to append bytes directly.
    fn infoset_key_into(&self, s: &Self::State, buf: &mut Vec<u8>) {
        buf.extend_from_slice(self.infoset_key(s).as_bytes());
    }
}

/// Count distinct information sets by exhaustively walking the tree.
///
/// A structural gate that is independent of any solver: Kuhn has 12, canonical
/// Leduc has 288. A mismatch means the Rust engine and the Python engine have
/// diverged (usually an infoset-key or perfect-recall bug), which would
/// silently corrupt training long before exploitability caught it.
pub fn count_infosets<G: Game>(game: &G) -> usize {
    use std::collections::HashSet;
    let mut keys: HashSet<String> = HashSet::new();
    fn walk<G: Game>(game: &G, s: &G::State, keys: &mut std::collections::HashSet<String>) {
        if game.is_terminal(s) {
            return;
        }
        if game.is_chance(s) {
            for (child, _) in game.chance_outcomes(s) {
                walk(game, &child, keys);
            }
            return;
        }
        keys.insert(game.infoset_key(s));
        for i in 0..game.num_actions(s) {
            walk(game, &game.next_state(s, i), keys);
        }
    }
    walk(game, &game.initial_state(), &mut keys);
    keys.len()
}
