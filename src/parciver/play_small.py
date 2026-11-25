# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Play Kuhn or Leduc poker against a CFR-solved opponent, in the terminal.

Nothing to download: the opponent is solved on the spot by the same CFR code the
validation ladder uses, and the solution's EXACT exploitability is printed, so you
know how close to unbeatable the thing you are playing is. In Kuhn and Leduc
"unbeatable" is a precise statement: an exploitability of 0.01 chips per hand means
no opponent, however clever, can win more than 0.01 chips per hand from it in the
long run.

    parciver play kuhn
    parciver play leduc --hands 20

The opponent plays its solved MIXED strategy: facing the same situation twice it
can act differently, which is the point. Run with ``--peek`` to see, after every
hand, the probabilities it used.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from .eval.exploitability import exploitability
from .games.base import Game
from .games.kuhn import KuhnPoker
from .games.leduc import LeducHoldem
from .solvers.cfr import DCFR, CFRSolver

RANKS = "JQK"


@dataclass(frozen=True)
class SmallGame:
    """How to solve, show and describe one of the small games."""

    name: str
    title: str
    make: Callable[[], Game]
    iterations: int  # default solve budget
    rules: str


GAMES: dict[str, SmallGame] = {
    "kuhn": SmallGame(
        "kuhn", "Kuhn poker", KuhnPoker, 3000,
        "3 cards (J < Q < K). Each player antes 1 and gets one card. The first player may check or bet 1;\n"
        "the other may then bet (if checked to) or fold/call. Higher card wins at showdown.",
    ),
    "leduc": SmallGame(
        "leduc", "Leduc hold'em", LeducHoldem, 600,
        "6 cards (two each of J, Q, K). Ante 1, one private card each, a bet round (bets of 2, up to two\n"
        "raises), then a public card, then a bet round (bets of 4). Pairing the public card wins;\n"
        "otherwise the higher card wins.",
    ),
}


def _card(c: int) -> str:
    return RANKS[c]


def _facing_bet(game_name: str, state) -> bool:
    if game_name == "kuhn":
        return state[1].endswith("b")
    return state.round_history.endswith("r")


def _action_label(game_name: str, state, action: str) -> str:
    facing = _facing_bet(game_name, state)
    if action == "f":
        return "fold"
    if action == "c":
        return "call" if facing else "check"
    return "raise" if facing else "bet"


def _dealt(game_name: str, state) -> tuple[int, ...]:
    return state[0] if game_name == "kuhn" else state.dealt


def _pot(game_name: str, state) -> int | None:
    return None if game_name == "kuhn" else sum(state.committed)


def _board(game_name: str, state) -> str | None:
    if game_name == "kuhn":
        return None
    return _card(state.dealt[2]) if len(state.dealt) == 3 else None


def _engine():
    try:
        import parciver_engine
    except ImportError:
        return None
    return parciver_engine


def solve(spec: SmallGame, iterations: int | None = None) -> tuple[dict[str, list[float]], float, float, str]:
    """Solve the game: (average strategy, exact exploitability, seconds, which solver).

    Uses the Rust engine when it is built (an order of magnitude more iterations in a
    fraction of the time); otherwise the pure-Python reference solver. Both are DCFR and
    both are scored by the same independent exact best-response code.
    """
    game = spec.make()
    t0 = time.perf_counter()
    eng = _engine()
    if eng is not None:
        strategy = getattr(eng, f"solve_{spec.name}")(iterations or spec.iterations * 8, "dcfr")
        which = "Rust engine"
    else:
        solver = CFRSolver(game, variant=DCFR)
        solver.run(iterations or spec.iterations)
        strategy = solver.average_strategy()
        which = "pure Python"
    secs = time.perf_counter() - t0
    return strategy, float(exploitability(game, strategy)), secs, which


def _sample(rng: random.Random, outcomes: Sequence[tuple[object, float]]):
    r, acc = rng.random(), 0.0
    for outcome, p in outcomes:
        acc += p
        if r < acc:
            return outcome
    return outcomes[-1][0]


def play_hand(
    spec: SmallGame,
    game: Game,
    strategy: dict[str, list[float]],
    human: int,
    rng: random.Random,
    ask: Callable[[str], str],
    say: Callable[[str], None],
    peek: bool,
) -> float | None:
    """Play one hand; return the human's winnings in chips, or None if the human quit."""
    name = spec.name
    state = game.initial_state()
    bot_moves: list[str] = []
    log: list[str] = []  # the hand so far in words, from the human's seat
    while not game.is_terminal(state):
        if game.is_chance(state):
            state = game.next_state(state, _sample(rng, game.chance_outcomes(state)))
            if _board(name, state) and not any(x.startswith("board") for x in log):
                log.append(f"board {_board(name, state)}")
            continue
        player = game.current_player(state)
        legal = list(game.legal_actions(state))
        if player == human:
            labels = {_action_label(name, state, a): a for a in legal}
            hole = _card(_dealt(name, state)[human])
            board = _board(name, state)
            pot = _pot(name, state)
            ctx = f"  your card: {hole}"
            if board:
                ctx += f"   board: {board}"
            if pot is not None:
                ctx += f"   pot: {pot}"
            say(ctx + "   so far: " + (" > ".join(log) if log else "you act first"))
            while True:
                reply = ask(f"  your move ({'/'.join(labels)}, q to quit)? ").strip().lower()
                if reply in ("q", "quit", "exit"):
                    return None
                match = [lab for lab in labels if lab == reply or (reply and lab.startswith(reply))]
                if len(match) == 1:
                    log.append(f"you {match[0]}")
                    state = game.next_state(state, labels[match[0]])
                    break
                say("  please type one of: " + ", ".join(labels))
        else:
            key = game.infoset_key(state)
            probs = strategy.get(key) or [1.0 / len(legal)] * len(legal)
            action = legal[_pick(rng, probs)]
            bot_moves.append(
                f"{_card(_dealt(name, state)[player])}: "
                + ", ".join(f"{_action_label(name, state, a)} {p:.0%}" for a, p in zip(legal, probs))
                + f"  -> {_action_label(name, state, action)}"
            )
            say(f"  opponent {_action_label(name, state, action)}s")
            log.append(f"opponent {_action_label(name, state, action)}s")
            state = game.next_state(state, action)

    dealt = _dealt(name, state)
    winnings = float(game.terminal_utility_for(state, human))
    bot = 1 - human
    say(f"  showdown/result: you had {_card(dealt[human])}, opponent had {_card(dealt[bot])}"
        + (f", board {_board(name, state)}" if _board(name, state) else ""))
    say(f"  you {'win' if winnings > 0 else 'lose' if winnings < 0 else 'split'} {abs(winnings):g} chip(s)")
    if peek and bot_moves:
        say("  opponent's mixed strategy at its decisions (holding: probabilities -> action taken):")
        for m in bot_moves:
            say("    " + m)
    return winnings


def _pick(rng: random.Random, probs: Sequence[float]) -> int:
    r, acc = rng.random(), 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1


def run(
    game_name: str,
    iterations: int | None = None,
    hands: int | None = None,
    seed: int | None = None,
    peek: bool = False,
    ask: Callable[[str], str] | None = None,
    say: Callable[[str], None] = print,
) -> int:
    ask = ask or input
    spec = GAMES[game_name]
    game = spec.make()
    say(f"{spec.title}: solving the opponent with discounted CFR ...")
    strategy, expl, secs, which = solve(spec, iterations)
    say(f"  solved in {secs:.1f}s ({which}); exact exploitability {expl:.5f} chips/hand "
        f"({len(strategy)} information sets)")
    say("  meaning: even a perfect counter-strategy wins at most that per hand (seats alternated) "
        "against this bot.")
    say("")
    say(spec.rules)
    say("")
    rng = random.Random(seed)
    total, played = 0.0, 0
    while hands is None or played < hands:
        human = played % 2  # alternate seats so nobody keeps the first-move edge
        say(f"--- hand {played + 1} (you are player {human + 1}, "
            f"{'acting first' if human == 0 else 'acting second'}) ---")
        result = play_hand(spec, game, strategy, human, rng, ask, say, peek)
        if result is None:
            break
        played += 1
        total += result
        say(f"  session: {total:+g} chips over {played} hand(s)\n")
    if played:
        say(f"Final: {total:+g} chips over {played} hand(s) ({total / played:+.3f} per hand).")
        say(f"Short samples are noisy. Even a perfect counter-strategy gains at most {expl:.4f} chips/hand "
            "on this bot, so over a long run you cannot beat it by more than luck.")
    return 0
