"""Play heads-up no-limit hold'em against the parciver blueprint, in the terminal (CPU only).

    parciver fetch-blueprint        # once: ~0.7 GB download, 1.5 GB unpacked
    parciver play nlhe              # blinds 50/100, 200 big blinds deep, seats alternate

The opponent is the v2-8B blueprint: a strategy solved offline by Monte Carlo CFR over a
card- and bet-abstracted game (see ARCHITECTURE.md). At the table it reads your real bets by
nearest-size translation. It does NOT run the real-time GPU search that the benchmark numbers
in RESULTS.md use; add ``--search`` to re-solve the turn and river on the CPU (slower, a few
seconds a decision), or see the README for the full GPU configuration.

Rules are Slumbot's (the same code that checks the live benchmark): 50/100 blinds, 20,000-chip
stacks reset every hand, the small blind acts first preflop and the big blind first afterwards.

At your turn type one of::

    f            fold
    c  (or k)    call / check
    b 600        bet or raise TO 600 chips in total on this street (also: r 600)
    h  p         half-pot / pot-sized bet or raise
    a            all in
    q            quit
"""
from __future__ import annotations

import os
import random
import re
import time
from pathlib import Path
from typing import Callable

from . import slumbot as S

RANKS, SUITS = S.RANKS, S.SUITS
STREET_NAMES = ("preflop", "flop", "turn", "river")
_TOKEN = re.compile(r"k|c|f|b\d+")
BLUEPRINT_FILE = "blueprint.bin"
BUCKET_FILES = ("flop_buckets.npy", "turn_buckets.npy", "river_buckets.npy")


def home() -> Path:
    """Where `parciver fetch-blueprint` keeps its tables (override with $PARCIVER_HOME)."""
    return Path(os.environ.get("PARCIVER_HOME") or Path.home() / ".parciver")


def locate_assets(blueprint: str | None = None, abstraction: str | None = None) -> tuple[Path, Path] | None:
    """(abstraction directory, blueprint file), or None if the tables are not on this machine."""
    cands = []
    if blueprint and abstraction:
        cands.append((Path(abstraction), Path(blueprint)))
    cands.append((home() / "abstraction", home() / BLUEPRINT_FILE))
    root = Path(__file__).resolve().parents[2]  # a source checkout: data/ beside src/
    cands.append((root / "data" / "abstraction", root / "data" / "blueprint" / BLUEPRINT_FILE))
    for a, b in cands:
        if b.is_file() and all((a / f).is_file() for f in BUCKET_FILES):
            return a, b
    return None


def card_name(c: int) -> str:
    return S.card_name(c)


def cards_str(cs) -> str:
    return " ".join(card_name(c) for c in cs) if cs else "-"


def describe(history: str, human: int) -> list[str]:
    """The hand so far as one line per street, from the human's seat ("you raise to 300")."""
    s = S.parse(history)
    tokens = _TOKEN.findall(history)
    lines: dict[int, list[str]] = {}
    bet_to = {0: S.BB}  # the big blind counts as the first bet preflop
    for tok, who, st in zip(tokens, s.movers, s.streets):
        name = "you" if who == human else "opponent"
        s_ = "" if who == human else "s"
        if tok == "k":
            txt = f"{name} check{s_}"
        elif tok == "c":
            txt = f"{name} call{s_}"
        elif tok == "f":
            txt = f"{name} fold{s_}"
        else:
            verb = "raise" if bet_to.get(st, 0) > 0 else "bet"
            bet_to[st] = int(tok[1:])
            txt = f"{name} {verb}{s_} to {tok[1:]}"
        lines.setdefault(st, []).append(txt)
    return [f"{STREET_NAMES[st]}: " + ", ".join(v) for st, v in sorted(lines.items())]


def parse_move(state: S.State, history: str, text: str) -> str | None:
    """The history token for what the human typed, or None if it is not a move (the caller reprompts)."""
    p = state.pos
    t = text.strip().lower().replace(",", "")
    word, _, arg = t.partition(" ")
    to_call = state.to_call(p)
    if word in ("f", "fold"):
        return "f" if to_call > 0 else None  # nothing to fold to: the caller says so
    if word in ("c", "k", "x", "call", "check"):
        return "c" if to_call > 0 else "k"
    pot = state.total[0] + state.total[1]
    all_in_to = state.street[p] + (S.STACK - state.total[p])
    least = state.street_last_bet_to + max(state.last_bet_size, S.BB)
    if word in ("a", "allin", "all-in", "all"):
        to = all_in_to
    elif word in ("h", "half", "p", "pot"):
        frac = 0.5 if word in ("h", "half") else 1.0
        to = state.street[p] + to_call + int(frac * (pot + to_call))
    elif word in ("b", "bet", "r", "raise") and arg.isdigit():
        to = int(arg)
    elif word.isdigit() and not arg:
        to = int(word)
    else:
        return None
    to = max(min(to, all_in_to), min(least, all_in_to)) if word in ("h", "half", "p", "pot") else to
    return f"b{to}"


def prompt_for(state: S.State, hole, board) -> str:
    p = state.pos
    pot = state.total[0] + state.total[1]
    behind = S.STACK - state.total[p]
    to_call = state.to_call(p)
    ctx = f"  [{STREET_NAMES[state.st]}] you: {cards_str(hole)}   board: {cards_str(board)}   pot: {pot}   "
    ctx += f"to call: {to_call}   behind: {behind}"
    return ctx


def load_agent(abstraction: Path, blueprint: Path, search: bool, say: Callable[[str], None]):
    import parciver_engine as pe

    say(f"loading the blueprint ({blueprint.stat().st_size / 1e9:.2f} GB) ...")
    t0 = time.perf_counter()
    agent = pe.SavedBlueprint(str(abstraction), str(blueprint)).play_agent()
    if search:
        agent = agent.with_search(from_turn=True, turn_iters=400, river_iters=2000)
    say(f"  ready in {time.perf_counter() - t0:.0f}s")
    return agent


def play_hand(agent, rng: random.Random, human: int, ask: Callable[[str], str], say: Callable[[str], None],
              peek: bool) -> int | None:
    """One hand; the human's winnings in chips, or None if they quit. `human` is a Slumbot position (0 = big blind)."""
    deck = list(range(52))
    rng.shuffle(deck)
    cards = {0: deck[0:2], 1: deck[2:4]}
    full_board = deck[4:9]
    hole = cards[human]
    bot = 1 - human
    h = ""
    bot_moves: list[str] = []
    say(f"  your cards: {cards_str(hole)}   (you are the {'big' if human == 0 else 'small'} blind)")
    shown = -1
    while True:
        s = S.parse(h)
        if s.over:
            break
        board = full_board[:S.BOARD_CARDS[s.st]]
        if s.st != shown:
            shown = s.st
            if s.st > 0:
                say(f"  --- {STREET_NAMES[s.st]}: board {cards_str(board)}   pot {s.total[0] + s.total[1]}")
        if s.pos == human:
            say(prompt_for(s, hole, board))
            lines = describe(h, human) if h else []
            if lines:
                say("    so far: " + "  |  ".join(lines))
            while True:
                reply = ask("  your move (f / c / b <to> / h / p / a / q)? ")
                if reply.strip().lower() in ("q", "quit", "exit"):
                    return None
                x = parse_move(s, h, reply)
                if x is not None:
                    try:
                        S.parse(h + x)
                    except S.ParseError as e:
                        least = s.street_last_bet_to + max(s.last_bet_size, S.BB)
                        say(f"  not allowed ({e}); a raise must be to at least {least} and at most "
                            f"{s.street[s.pos] + S.STACK - s.total[s.pos]}")
                        continue
                    break
                if reply.strip().lower() in ("f", "fold"):
                    say("  there is no bet to fold to: check with c")
                else:
                    say("  type f (fold), c (call/check), b <to>, h (half pot), p (pot), a (all in) or q")
            h = S.extend(h, x)
            continue
        t0 = time.perf_counter()
        d = agent.decide(S.engine_seat(bot), tuple(cards[bot]), board, s.acts, S.SB, S.BB,
                         (S.STACK, S.STACK), seed=rng.getrandbits(62))
        x = S.incr(s, d["action"], d["add"])
        verb = {"f": "folds", "k": "checks", "c": "calls"}.get(x[0]) or (
            ("raises" if s.last_bet_size > 0 else "bets") + f" to {x[1:]}")
        say(f"  opponent {verb}" + (f"   ({time.perf_counter() - t0:.1f}s)" if time.perf_counter() - t0 > 1 else ""))
        if peek:
            mix = ", ".join(f"{lab} {p:.0%}" for lab, p in zip(d["labels"], d["probs"]))
            bot_moves.append(f"{STREET_NAMES[s.st]} {cards_str(cards[bot])}: {mix}  -> {x}  [{d['source']}]")
        h = S.extend(h, x)

    s = S.parse(h)
    if s.all_in_call or s.folded is None:
        say(f"  board: {cards_str(full_board)}")
    if s.folded is None:
        say(f"  showdown: you {cards_str(hole)}  opponent {cards_str(cards[bot])}")
    winnings = S.payoff(s, human, cards, full_board)
    say(f"  you {'win' if winnings > 0 else 'lose' if winnings < 0 else 'split'} {abs(winnings)} chips "
        f"({winnings / S.BB:+.1f} bb)")
    if peek and bot_moves:
        say("  opponent's strategy at its decisions (abstract action: probability -> action taken):")
        for m in bot_moves:
            say("    " + m)
    return winnings


def run(hands: int | None = None, seed: int | None = None, peek: bool = False, search: bool = False,
        blueprint: str | None = None, abstraction: str | None = None,
        ask: Callable[[str], str] | None = None, say: Callable[[str], None] = print, agent=None) -> int:
    ask = ask or input
    if agent is None:
        found = locate_assets(blueprint, abstraction)
        if found is None:
            say("The blueprint tables are not on this machine. Run:  parciver fetch-blueprint")
            say(f"(they go to {home()}; ~1.5 GB, checked against published sha256 sums)")
            return 2
        agent = load_agent(*found, search, say)
    rng = random.Random(seed)
    say("")
    say("Heads-up no-limit hold'em, blinds 50/100, 20,000-chip stacks (200 bb). Seats alternate.")
    say("The opponent plays a fixed mixed strategy: it will not tilt, bluff-read you or adapt.")
    say("")
    total, played = 0, 0
    while hands is None or played < hands:
        human = (played + 1) % 2  # 1 = small blind on the first hand
        say(f"--- hand {played + 1} ---")
        result = play_hand(agent, rng, human, ask, say, peek)
        if result is None:
            break
        played += 1
        total += result
        say(f"  session: {total:+d} chips ({total / S.BB:+.1f} bb) over {played} hand(s)\n")
    if played:
        say(f"Final: {total / S.BB:+.1f} bb over {played} hand(s) ({total / S.BB / played * 100:+.0f} bb/100). "
            "Poker is high-variance: a session this short says almost nothing about who is stronger.")
    return 0
