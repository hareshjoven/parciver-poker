"""GTO Wizard researcher API <-> play agent (pure translation; no network).

The API (https://researcher.gtowizard.com, client: github.com/gtowizard-ai/researcher-api-client)
plays heads-up no-limit hands against GTO Wizard AI: `POST /hands` starts one, `POST
/hands/{id}/act` sends {"action": "f"|"k"|"c"|"b", "amount": int|None}; every response carries
the game (`starting_stack`, `blinds`) and the state: `players` (name, stack, position,
hole_cards), `board_cards`, `action_history` ('f' fold, 'k' check, 'c' call, 'bX' bet to a
cumulative X on the round, '_' end of round), `legal_actions`, `raise_range` {min, max},
`total_pot`/`common_pot`, and at the end `winnings` and `aivat_score` (chips, our view).

Here: rebuild the engine's action list (chips ADDED per action; seat 0 = small blind /
button, seat 1 = big blind) from `action_history`, and turn the agent's decision back into
an API action. Two conventions the documentation leaves open are settled from the state
itself, never assumed: whether a preflop 'bX' counts the posted blind (the replay must
reproduce the server's `total_pot`), and which seat we hold (the player to act when the
server hands us the state). Any disagreement raises `Desync` instead of playing on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_CARD = re.compile(r"([2-9TJQKA])([cdhs])", re.IGNORECASE)
# Basis of preflop 'bX' amounts: the round total including the posted blind, or excluding it.
BASES = ("inclusive", "exclusive")


class Desync(RuntimeError):
    """The server's state and our reconstruction disagree."""


def parse_cards(s: str | None) -> list[int]:
    """Card indices (rank*4 + suit, suits cdhs) from any spelling ('AsKd', 'As Kd', 'As,Kd')."""
    if not s:
        return []
    return [_RANKS.index(r.upper()) * 4 + _SUITS.index(u.lower()) for r, u in _CARD.findall(s)]


def card_names(cs: list[int]) -> str:
    return "".join(f"{_RANKS[c >> 2]}{_SUITS[c & 3]}" for c in cs)


@dataclass
class Replay:
    """The hand so far, as the engine sees it."""

    actions: list[tuple[str, int]] = field(default_factory=list)  # (kind, chips added)
    contributed: list[int] = field(default_factory=lambda: [0, 0])  # this hand, per seat
    round_bet: list[int] = field(default_factory=lambda: [0, 0])  # this round, per seat
    to_act: int | None = 0  # seat to act next; None once the round or hand is closed
    street: int = 0  # 0 preflop .. 3 river
    folded: bool = False


def replay(history: list[str], blinds: tuple[int, int], stack: int, basis: str = "inclusive") -> Replay:
    """Rebuild the hand from the API's `action_history`."""
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}")
    sb, bb = blinds
    r = Replay(contributed=[sb, bb], round_bet=[sb, bb])
    for tok in history:
        if tok == "_":
            r.street += 1
            r.round_bet = [0, 0]
            r.to_act = 1  # the big blind acts first after the flop
            continue
        if r.folded or r.to_act is None:
            raise Desync(f"action {tok!r} after the hand or round closed ({history})")
        p = r.to_act
        owed = r.round_bet[1 - p] - r.round_bet[p]
        if tok == "f":
            r.actions.append(("f", 0))
            r.folded = True
            r.to_act = None
            continue
        if tok == "k":
            if owed != 0:
                raise Desync(f"check facing a bet of {owed} ({history})")
            r.actions.append(("x", 0))
            # Checks close the round when the second player checks.
            closes = (r.street == 0 and p == 1) or (r.street > 0 and p == 0)
        elif tok == "c":
            add = min(owed, stack - r.contributed[p])
            if add <= 0:
                raise Desync(f"call with nothing to call ({history})")
            r.actions.append(("c", 0))
            r.round_bet[p] += add
            r.contributed[p] += add
            # A call closes the round, except the small blind completing preflop.
            closes = not (r.street == 0 and p == 0 and r.round_bet[0] == bb and len(r.actions) == 1)
        elif tok.startswith("b"):
            try:
                x = round(float(tok[1:]))
            except ValueError as e:
                raise Desync(f"unreadable bet {tok!r}") from e
            target = x + (blinds[p] if r.street == 0 and basis == "exclusive" else 0)
            add = target - r.round_bet[p]
            if add <= owed or r.contributed[p] + add > stack:
                raise Desync(f"bet {tok!r} is not a legal raise for seat {p} ({history}, basis {basis})")
            r.actions.append(("b", add))
            r.round_bet[p] += add
            r.contributed[p] += add
            closes = False
        else:
            raise Desync(f"unknown history token {tok!r}")
        r.to_act = None if closes else 1 - p
    return r


@dataclass
class Spot:
    """Our decision point, reconstructed and checked against the server's state."""

    seat: int
    hole: list[int]
    board: list[int]
    replay: Replay
    basis: str
    sb: int
    bb: int
    stack: int
    # The preflop bet convention is pinned down (not just compatible) at this spot.
    certain: bool = False


def _pot_matches(r: Replay, state: dict) -> bool:
    v = state.get("total_pot")
    return v is not None and abs(float(v) - sum(r.contributed)) < 0.5


_SEAT_OF_POSITION = {"sb": 0, "btn": 0, "button": 0, "bu": 0, "d": 0, "dealer": 0, "bb": 1}


def hero_index(players: list[dict]) -> int:
    """Our entry in `players`: the one whose hole cards we see (GTO Wizard's are hidden)."""
    seen = [i for i, p in enumerate(players) if p.get("hole_cards")]
    if len(seen) == 1:
        return seen[0]
    named = [i for i, p in enumerate(players) if "wizard" not in str(p.get("name", "")).lower()]
    if len(named) == 1:
        return named[0]
    raise Desync(f"cannot tell which player is us: {players}")


def spot(response: dict, basis: str | None = None) -> Spot:
    """Our decision point from a server response where it is our turn. `basis` fixes the
    preflop bet convention once known; None tries both and keeps those consistent with the
    server's pot, our stack, seat and raise range (`Spot.certain` when exactly one fits)."""
    game, state = response["game"], response["game_state"]
    sb, bb = (round(float(x)) for x in game["blinds"])
    stack = round(float(game["starting_stack"]))
    history = list(state.get("action_history") or [])
    board = parse_cards(state.get("board_cards"))
    players = state["players"]
    me = players[hero_index(players)]
    hole = parse_cards(me.get("hole_cards"))
    if len(hole) != 2:
        raise Desync(f"hole cards {me.get('hole_cards')!r}")
    pos = _SEAT_OF_POSITION.get(str(me.get("position", "")).strip().lower())
    rr = state.get("raise_range")
    errors, fits = [], []
    for b in [basis] if basis else list(BASES):
        try:
            r = replay(history, (sb, bb), stack, b)
        except Desync as e:
            errors.append(f"{b}: {e}")
            continue
        # A new street the history has not marked yet ('_' only after its first action).
        if r.to_act is None and not r.folded and len(board) > (0, 3, 4, 5)[r.street]:
            r.street += 1
            r.round_bet = [0, 0]
            r.to_act = 1
        if r.to_act is None:
            errors.append(f"{b}: nobody to act after {history}")
            continue
        p = r.to_act
        if len(board) != (0, 3, 4, 5)[r.street]:
            errors.append(f"{b}: {len(board)} board cards on street {r.street}")
            continue
        if not _pot_matches(r, state):
            errors.append(f"{b}: pot {sum(r.contributed)} vs server total_pot={state.get('total_pot')} "
                          f"common_pot={state.get('common_pot')}")
            continue
        if pos is not None and pos != p:
            errors.append(f"{b}: we are {me.get('position')!r} but seat {p} is to act")
            continue
        # Our stack as reported (chips behind, or the starting stack) must fit the replay.
        behind = stack - r.contributed[p]
        if me.get("stack") is not None and min(abs(float(me["stack"]) - v) for v in (behind, stack)) >= 0.5:
            errors.append(f"{b}: our stack {me['stack']} vs replay {behind}")
            continue
        # The largest bet is all-in: our round total plus what is behind, in this basis.
        if rr:
            top = r.round_bet[p] + behind - ((sb, bb)[p] if r.street == 0 and b == "exclusive" else 0)
            if abs(float(rr["max"]) - top) >= 0.5:
                errors.append(f"{b}: raise_range max {rr['max']} vs all-in {top}")
                continue
        fits.append(Spot(p, hole, board, r, b, sb, bb, stack))
    if not fits:
        raise Desync("; ".join(errors) or "no basis fits")
    fits[0].certain = basis is not None or len(fits) == 1
    return fits[0]


def to_request(s: Spot, decision: dict, legal: list[str], raise_range: dict | None) -> tuple[dict, str | None]:
    """The API action for the agent's decision, and a note when it had to be adjusted."""
    kind, note = decision["action"], None
    legal = list(legal)

    def first_legal(*opts: str) -> str:
        for o in opts:
            if o in legal:
                return o
        raise Desync(f"none of {opts} legal in {legal}")

    if kind == "fold":
        a = first_legal("f", "k")
        return {"action": a, "amount": None}, None if a == "f" else "fold -> check (free)"
    if kind == "check":
        a = first_legal("k", "c")
        return {"action": a, "amount": None}, None if a == "k" else "check -> call"
    if kind == "call":
        a = first_legal("c", "k")
        return {"action": a, "amount": None}, None if a == "c" else "call -> check"
    if kind != "bet":
        raise ValueError(f"unknown decision {kind!r}")
    if "b" not in legal or not raise_range:
        return {"action": first_legal("c", "k"), "amount": None}, "bet not legal -> call/check"
    p = s.seat
    target = s.replay.round_bet[p] + int(decision["add"])
    if s.replay.street == 0 and s.basis == "exclusive":
        target -= (s.sb, s.bb)[p]
    lo, hi = round(float(raise_range["min"])), round(float(raise_range["max"]))
    if decision.get("all_in"):
        if target != hi:
            note = f"all-in {target} -> server max {hi}"
        target = hi
    elif target < lo:
        note, target = f"bet {target} below min {lo}", lo
    elif target > hi:
        note, target = f"bet {target} above max {hi}", hi
    return {"action": "b", "amount": int(target)}, note
