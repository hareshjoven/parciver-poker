"""N-player Leduc hold'em — the decisive game for the R-MAPSS kernel (Step 2).

The multiplayer generalisation of ``parciver.games.leduc`` (RMAPSS.md §33/§34,
Step 2/3). Leduc is the first game in the toy progression with TWO betting rounds
and a persistent public belief state, so it is the first game that can exhibit
cross-street cascades and multi-street intervention (refinement R3) — 3p Kuhn only
tests the within-round form.

Rules (n players). A 6-card deck: two suits x three ranks (J,Q,K), two of each
rank (same deck as 2p Leduc, which supports up to n=5 since it needs n+1 cards).
Everyone antes 1 and is dealt one private card. Two betting rounds; ONE public
card is revealed before round 1. Fixed bet size (2 in round 0, 4 in round 1), cap
of two raises per round. Seats act in ascending order starting from seat 0 in both
rounds; folded seats are skipped. At showdown among the non-folded seats: a seat
that pairs the public card wins (at most one can, since only two copies of a rank
exist and the board holds one); else the highest private card wins; equal ranks
split the pot.

Byte-for-byte reduction: ``LeducHoldemN(2)`` produces the SAME infoset keys, deal
outcomes, and player-0 utilities as the certified ``parciver.games.leduc``
(288 infosets, value -0.086, exploitability -> 0), which is the correctness gate.
For n>=3 there is no zero-sum-per-pair structure, but the game is zero-sum OVERALL
(chips conserved) and general-sum for exploitation purposes; correctness is gated
by chip conservation + total zero-sum + the exact-BR/brute-force control (§28),
never by a solver fixpoint (CFR has no >=3p guarantee — R5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

from .base import Game

CHECK_CALL, BET_RAISE, FOLD = "c", "r", "f"

_DECK = (0, 0, 1, 1, 2, 2)  # ranks 0,1,2 == J,Q,K; two copies each
_BET_SIZE = (2, 4)          # round 0, round 1
_MAX_RAISES = 2             # per round
_ANTE = 1


@dataclass(frozen=True)
class _State:
    """Immutable n-player Leduc state.

    ``dealt`` holds cards in order (seat0, seat1, ..., seat_{n-1}, [public]).
    ``committed`` is total chips per seat (ante + all bets). ``folded`` is the set
    of folded seats. ``to_act`` / ``need`` drive the betting round (``need`` = how
    many active seats still owe an action before the round closes; reset to
    num_active-1 after any bet/raise). ``bet_log`` is the full action string across
    rounds (rounds joined by "/") and is the perfect-recall public signal the
    infoset key uses. ``num_raises`` counts aggression in the current round.
    """

    dealt: tuple[int, ...]
    round: int
    committed: tuple[int, ...]
    folded: frozenset
    to_act: int
    need: int
    num_raises: int
    bet_log: str = ""


class LeducHoldemN(Game):
    """n-player Leduc hold'em as an extensive-form game (general-sum for n>=3)."""

    def __init__(self, num_players: int = 3) -> None:
        if num_players < 2:
            raise ValueError("Leduc needs at least 2 players")
        if num_players + 1 > len(_DECK):
            raise ValueError(f"deck of {len(_DECK)} cards supports at most {len(_DECK) - 1} players")
        self.num_players = num_players

    # ---- helpers --------------------------------------------------------
    def _active(self, state: _State) -> list[int]:
        return [i for i in range(self.num_players) if i not in state.folded]

    def _level(self, state: _State) -> int:
        """The amount an active seat must have committed this round to stay in."""
        return max(state.committed)

    def _next_active(self, state: _State, after: int) -> int:
        n = self.num_players
        for step in range(1, n + 1):
            seat = (after + step) % n
            if seat not in state.folded:
                return seat
        return after  # unreachable while >=1 active

    def _first_active_from0(self, folded: frozenset) -> int:
        n = self.num_players
        for seat in range(n):
            if seat not in folded:
                return seat
        return 0

    # ---- chance ---------------------------------------------------------
    def initial_state(self) -> Hashable:
        n = self.num_players
        return _State(dealt=(), round=0, committed=(_ANTE,) * n, folded=frozenset(),
                      to_act=0, need=n, num_raises=0, bet_log="")

    def is_chance(self, state: _State) -> bool:
        n = self.num_players
        if len(state.dealt) < n:          # deal private cards
            return True
        if state.round == 1 and len(state.dealt) < n + 1:  # deal the public card
            return True
        return False

    def chance_outcomes(self, state: _State) -> Sequence[tuple[Hashable, float]]:
        remaining = list(_DECK)
        for c in state.dealt:
            remaining.remove(c)
        counts: dict[int, int] = {}
        for c in remaining:
            counts[c] = counts.get(c, 0) + 1
        total = len(remaining)
        return [(c, n / total) for c, n in sorted(counts.items())]

    # ---- terminal -------------------------------------------------------
    def is_terminal(self, state: _State) -> bool:
        n = self.num_players
        if len(state.folded) >= n - 1:     # all but one folded
            return True
        # Showdown: round 1 betting closed (need==0) after the public card is out.
        return state.round == 1 and len(state.dealt) == n + 1 and state.need == 0

    def terminal_utility(self, state: _State) -> float:
        return self.terminal_utility_for(state, 0)

    def _winners(self, state: _State) -> list[int]:
        active = self._active(state)
        if len(active) == 1:
            return active
        board = state.dealt[self.num_players]
        pairers = [i for i in active if state.dealt[i] == board]
        if pairers:              # at most one seat can pair a given board rank
            return pairers
        best = max(state.dealt[i] for i in active)
        return [i for i in active if state.dealt[i] == best]

    def terminal_utility_for(self, state: _State, player: int) -> float:
        pot = float(sum(state.committed))
        winners = self._winners(state)
        share = pot / len(winners) if winners else 0.0
        u = -float(state.committed[player])
        if player in winners:
            u += share
        return u

    # ---- decisions ------------------------------------------------------
    def current_player(self, state: _State) -> int:
        return state.to_act

    def _num_raises_cap_ok(self, state: _State) -> bool:
        return state.num_raises < _MAX_RAISES

    def legal_actions(self, state: _State) -> Sequence[Hashable]:
        facing_bet = state.committed[state.to_act] < self._level(state)
        if facing_bet:
            if self._num_raises_cap_ok(state):
                return (FOLD, CHECK_CALL, BET_RAISE)
            return (FOLD, CHECK_CALL)
        # not facing a bet: check, or open (bet/raise) if the cap allows
        if self._num_raises_cap_ok(state):
            return (CHECK_CALL, BET_RAISE)
        return (CHECK_CALL,)

    def next_state(self, state: _State, action) -> _State:
        n = self.num_players
        # -- chance: deal a card, initialise betting when a round begins --
        if self.is_chance(state):
            dealt = state.dealt + (action,)
            if len(dealt) == n:                 # last private dealt -> round 0 betting
                return _State(dealt, 0, state.committed, state.folded,
                              to_act=self._first_active_from0(state.folded),
                              need=len(self._active(state)), num_raises=0, bet_log=state.bet_log)
            if len(dealt) == n + 1:             # public dealt -> round 1 betting
                st = _State(dealt, 1, state.committed, state.folded, 0, 0, 0, state.bet_log)
                return _State(dealt, 1, state.committed, state.folded,
                              to_act=self._first_active_from0(state.folded),
                              need=len(self._active(st)), num_raises=0, bet_log=state.bet_log)
            return _State(dealt, state.round, state.committed, state.folded,
                          state.to_act, state.need, state.num_raises, state.bet_log)

        # -- a betting action by seat `to_act` --
        seat = state.to_act
        committed = list(state.committed)
        folded = state.folded
        num_raises = state.num_raises
        need = state.need
        level = self._level(state)
        new_log = state.bet_log + action

        if action == FOLD:
            folded = folded | {seat}
            need -= 1
        elif action == CHECK_CALL:
            committed[seat] = level        # match (a check when already == level)
            need -= 1
        elif action == BET_RAISE:
            committed[seat] = level + _BET_SIZE[state.round]
            num_raises += 1
            need = len(self._active(state)) - 1   # everyone else active owes a response
        else:
            raise ValueError(f"illegal action {action!r}")

        committed_t = tuple(committed)
        num_active = n - len(folded)

        # hand ends immediately if all but one folded
        if num_active <= 1:
            return _State(state.dealt, state.round, committed_t, folded,
                          to_act=seat, need=0, num_raises=num_raises, bet_log=new_log)

        if need == 0:
            # round closes
            if state.round == 0:
                # advance to round 1: append the separator, wait for the public card
                return _State(state.dealt, 1, committed_t, folded,
                              to_act=0, need=0, num_raises=0, bet_log=new_log + "/")
            # round 1 closed -> terminal (need==0, dealt==n+1)
            return _State(state.dealt, 1, committed_t, folded,
                          to_act=seat, need=0, num_raises=num_raises, bet_log=new_log)

        # round continues: pass action to the next active seat
        nxt = self._next_active(_State(state.dealt, state.round, committed_t, folded,
                                       seat, need, num_raises, new_log), seat)
        return _State(state.dealt, state.round, committed_t, folded,
                      to_act=nxt, need=need, num_raises=num_raises, bet_log=new_log)

    def infoset_key(self, state: _State) -> str:
        player = state.to_act
        private = state.dealt[player]
        board = state.dealt[self.num_players] if len(state.dealt) >= self.num_players + 1 else "?"
        return f"{private}|{board}|{state.bet_log}"
