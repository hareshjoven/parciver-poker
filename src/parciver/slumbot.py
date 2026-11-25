# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Slumbot's public API as the engine's play layer sees it (records/slumbot/PREREG.md).

Slumbot (slumbot.com) plays heads-up NLHE, blinds 50/100, 20,000-chip stacks reset every hand. A hand's
history is one string: `k` check, `c` call, `f` fold, `b<n>` a bet or raise to `n` chips put in ON THAT
STREET, `/` between streets; an all-in call may be followed by the streets it skipped (`b20000c///`).
`client_pos` 0 is the big blind (second to act preflop, first after). Positions here are Slumbot's: 0 the
big blind, 1 the small blind; the engine's seats are the other way round (0 the button / small blind).

- `parse` reads a history and returns the state (whose turn, the street, both players' chips) and the
  engine's actions, `(kind, chips added)` as `PlayAgent.options` takes them. It agrees with
  `ParseAction` of Slumbot's `sample_api.py` on every prefix of a legal history (the test), and refuses
  more: anything after the hand ended.
- `incr` turns one of the agent's options into the string Slumbot takes.
- `MockSlumbot` serves the API offline (a random legal opponent, real cards and payoffs) for the
  pre-registered checks; `Http` is the live transport. Both answer `post(endpoint, body) -> dict`.
"""
from __future__ import annotations

import http.client
import json
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

SB, BB, STACK = 50, 100, 20000
STREETS = 4
BOARD_CARDS = (0, 3, 4, 5)  # board cards dealt by each street
HOST = "https://slumbot.com/slumbot/api/"
_INCR = re.compile(r"^(k|c|f|b\d+)$")


class ParseError(ValueError):
    """A history Slumbot's rules do not allow (or one this parser does not take)."""


@dataclass
class State:
    """A parsed history. `pos` is the position to act (-1 once the hand is over); `street`/`total` are each
    position's chips this street / this hand; `last_bet_size` is the increment of the last bet or raise
    still to be answered (0 when none). The six `ParseAction` fields keep its meaning, including
    `last_bettor` (-1 after a call; unchanged by checks and by an all-in call)."""

    st: int = 0
    pos: int = 1
    street_last_bet_to: int = BB
    total_last_bet_to: int = BB
    last_bet_size: int = BB - SB
    last_bettor: int = 0
    street: list[int] = field(default_factory=lambda: [BB, SB])
    total: list[int] = field(default_factory=lambda: [BB, SB])
    acts: list[tuple[str, int]] = field(default_factory=list)
    movers: list[int] = field(default_factory=list)  # the position of each act
    streets: list[int] = field(default_factory=list)  # the street of each act
    folded: int | None = None
    all_in_call: bool = False
    over: bool = False

    def fields(self) -> dict:
        """The fields `ParseAction` returns."""
        return dict(st=self.st, pos=self.pos, street_last_bet_to=self.street_last_bet_to,
                    total_last_bet_to=self.total_last_bet_to, last_bet_size=self.last_bet_size,
                    last_bettor=self.last_bettor)

    def to_call(self, p: int) -> int:
        return self.street_last_bet_to - self.street[p]


def parse(action: str) -> State:
    """The state after `action` (Slumbot's history string), or ParseError."""
    s = State()
    closes = False  # the next check or call ends the street
    i, n = 0, len(action)

    def end_street() -> None:
        nonlocal i, closes
        if s.st < STREETS - 1 and i < n:  # a '/' or the end of the string
            if action[i] != "/":
                raise ParseError("missing slash")
            i += 1
        if s.st == STREETS - 1:
            s.pos, s.over = -1, True
        else:
            s.pos, s.st = 0, s.st + 1
        s.street_last_bet_to, s.street, closes = 0, [0, 0], False

    while i < n:
        if s.over:
            raise ParseError("action after the hand ended")
        c, p = action[i], s.pos
        i += 1
        if c == "k":
            if s.last_bet_size > 0:
                raise ParseError("illegal check")
            s.acts.append(("check", 0))
            s.movers.append(p)
            s.streets.append(s.st)
            if closes:
                end_street()
            else:
                s.pos, closes = 1 - p, True
        elif c == "c":
            if s.last_bet_size == 0:
                raise ParseError("illegal call")
            add = s.to_call(p)
            s.street[p] += add
            s.total[p] += add
            s.acts.append(("call", 0))
            s.movers.append(p)
            s.streets.append(s.st)
            if s.total_last_bet_to == STACK:  # an all-in called: no slashes, or one for each street left
                if i != n:
                    for _ in range(s.st, STREETS - 1):
                        if i == n:
                            raise ParseError("missing slash (end of string)")
                        if action[i] != "/":
                            raise ParseError("missing slash")
                        i += 1
                if i != n:
                    raise ParseError("extra characters at end of action")
                s.st, s.pos, s.last_bet_size, s.all_in_call, s.over = STREETS - 1, -1, 0, True, True
                return s
            if closes:
                end_street()
            else:
                s.pos, closes = 1 - p, True
            s.last_bet_size, s.last_bettor = 0, -1
        elif c == "f":
            if s.last_bet_size == 0:
                raise ParseError("illegal fold")
            if i != n:
                raise ParseError("extra characters at end of action")
            s.acts.append(("fold", 0))
            s.movers.append(p)
            s.streets.append(s.st)
            s.pos, s.folded, s.over = -1, p, True
            return s
        elif c == "b":
            j = i
            while i < n and "0" <= action[i] <= "9":
                i += 1
            if i == j:
                raise ParseError("missing bet size")
            to = int(action[j:i])
            size = to - s.street_last_bet_to
            remaining = STACK - s.total_last_bet_to
            least = min(max(s.last_bet_size, BB) if s.last_bet_size > 0 else BB, remaining)
            if size < least:
                raise ParseError("bet too small")
            if size > remaining:
                raise ParseError("bet too big")
            add = to - s.street[p]
            s.street[p] = to
            s.total[p] += add
            s.acts.append(("bet", add))
            s.movers.append(p)
            s.streets.append(s.st)
            s.last_bet_size, s.street_last_bet_to = size, to
            s.total_last_bet_to += size
            s.last_bettor, s.pos, closes = p, 1 - p, True
        else:
            raise ParseError(f"unexpected character {c!r}")
    return s


def incr(state: State, kind: str, add: int) -> str:
    """The string Slumbot takes for the option `(kind, add)` of the player to act."""
    if kind == "fold":
        return "f"
    if kind == "check":
        return "k"
    if kind == "call":
        return "c"
    if kind == "bet":
        return f"b{state.street[state.pos] + add}"
    raise ValueError(f"unknown option kind {kind!r}")


def engine_seat(pos: int) -> int:
    """The engine's seat (0 button / small blind) of Slumbot position `pos` (0 big blind)."""
    return 1 - pos


def payoff(state: State, pos: int, cards: dict[int, list[int]] | None, board: list[int]) -> int | None:
    """Position `pos`'s chips won (+) or lost (-) in a finished hand: the folder loses what it put in, else
    the better seven cards win the other's chips (equal totals; a tie is 0). None where a showdown's cards
    are not all known. `cards` maps each position to its two card indices."""
    import parciver_engine as pe

    if not state.over:
        raise ValueError("the hand is not over")
    other = 1 - pos
    if state.folded is not None:
        return -state.total[pos] if state.folded == pos else state.total[other]
    if state.total[0] != state.total[1]:
        raise ValueError(f"a showdown with unequal chips {state.total}")
    if not cards or len(cards.get(0, ())) != 2 or len(cards.get(1, ())) != 2 or len(board) != 5:
        return None
    a, b = pe.eval7(cards[pos] + board), pe.eval7(cards[other] + board)
    return state.total[other] if a > b else -state.total[pos] if a < b else 0


def all_in_equity(state: State, pos: int, cards: dict[int, list[int]], board: list[int]) -> float | None:
    """Position `pos`'s equity over the cards still to come where both players were all in before the river
    was dealt and both hands are known (descriptive: the all-in-adjusted win rate), else None."""
    import parciver_engine as pe

    if not state.all_in_call or len(cards.get(0, ())) != 2 or len(cards.get(1, ())) != 2:
        return None
    called = state.streets[-1]  # the street the all-in was called on: its board had BOARD_CARDS of it
    if called >= STREETS - 1:
        return None
    return float(pe.equity_vs_hand(tuple(cards[pos]), tuple(cards[1 - pos]), board[:BOARD_CARDS[called]]))


# ---------------------------------------------------------------- transports

class Http:
    """The live API: one JSON POST at a time. A network error or an HTTP 5xx is retried with the same body
    (`retries` times, backing off); what still fails raises `Lost`. So does a retry answered with
    `error_msg`: the first try may have reached the server, and the hand cannot be resumed. Every request
    and response goes to `log` (a callable taking one dict) before this returns. `retried` counts the
    retries so far."""

    def __init__(self, log, host: str = HOST, timeout: float = 60.0, retries: int = 5):
        self.log, self.host, self.timeout, self.retries = log, host, timeout, retries
        self.retried = 0

    def _answer(self, endpoint: str, attempt: int, out: dict) -> dict:
        if attempt and "error_msg" in out:
            raise Lost(f"{endpoint}: a retry was refused ({out['error_msg']}); the first try may have reached the server")
        return out

    def post(self, endpoint: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(self.retries + 1):
            self.retried += attempt > 0
            t0 = time.time()
            rec = {"utc": _utc(t0), "endpoint": endpoint, "body": body, "attempt": attempt}
            try:
                req = urllib.request.Request(self.host + endpoint, data=data, method="POST",
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    text = r.read().decode(errors="replace")
                    rec["status"] = r.status
            except urllib.error.HTTPError as e:
                try:
                    text = e.read().decode(errors="replace")
                except (OSError, http.client.HTTPException):
                    text = ""
                rec["status"] = e.code
                if e.code < 500:
                    rec.update(seconds=time.time() - t0, response_text=text)
                    self.log(rec)
                    try:
                        out = json.loads(text)
                    except ValueError:
                        out = {"error_msg": f"HTTP {e.code}: {text[:200]}"}
                    if not isinstance(out, dict):
                        out = {"error_msg": f"HTTP {e.code}: {text[:200]}"}
                    elif "error_msg" not in out:
                        out["error_msg"] = f"HTTP {e.code}"
                    return self._answer(endpoint, attempt, out)
            except (urllib.error.URLError, OSError, http.client.HTTPException) as e:  # a cut answer too
                rec.update(seconds=time.time() - t0, error=f"{type(e).__name__}: {e}")
                self.log(rec)
                time.sleep(min(60, 2 ** (attempt + 1)))
                continue
            rec["seconds"] = time.time() - t0
            try:
                out = json.loads(text)
            except ValueError:
                rec["response_text"] = text
                self.log(rec)
                time.sleep(min(60, 2 ** (attempt + 1)))
                continue
            rec["response"] = out
            self.log(rec)
            if rec["status"] >= 500:
                time.sleep(min(60, 2 ** (attempt + 1)))
                continue
            if not isinstance(out, dict):
                return self._answer(endpoint, attempt, {"error_msg": f"not a JSON object: {text[:200]}"})
            return self._answer(endpoint, attempt, out)
        raise Lost(f"{endpoint}: no answer after {self.retries + 1} tries")


class Lost(RuntimeError):
    """A request no retry of the same body could complete: its hand is lost."""


def _utc(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"


RANKS, SUITS = "23456789TJQKA", "cdhs"


def card_name(c: int) -> str:
    return RANKS[c >> 2] + SUITS[c & 3]


class MockSlumbot:
    """Slumbot's API offline, by the rules above: sessions by token, the client starting in the big blind
    and changing seat every hand, real cards and payoffs, and a random legal opponent (every bet size from
    the least to all-in). Each client action is checked by `parse` and by `referee` (Slumbot's
    `ParseAction`, when given) and refused with `error_msg` if either rejects it, if it is not one action,
    if it is not the client's turn, or if it raises over an all-in. `bot_net` is the opponent's chips won,
    kept apart from what the client is told."""

    def __init__(self, seed: int, referee=None, log=None):
        self.rng = random.Random(seed)
        self.referee, self.log = referee, log
        self.sessions: dict[str, dict] = {}
        self.bot_net = 0
        self.hands = 0
        self.refused: list[str] = []

    def post(self, endpoint: str, body: dict) -> dict:
        out = self._post(endpoint, body)
        if self.log is not None:
            self.log({"utc": _utc(time.time()), "endpoint": endpoint, "body": body, "response": out, "mock": True})
        return out

    def _post(self, endpoint: str, body: dict) -> dict:
        if endpoint == "new_hand":
            tok = body.get("token")
            if tok not in self.sessions:
                tok = f"mock-{self.rng.getrandbits(64):016x}"
                self.sessions[tok] = {"client_pos": 1, "hand": None}  # flipped to 0 below
            sess = self.sessions[tok]
            sess["client_pos"] = 1 - sess["client_pos"]
            deck = list(range(52))
            self.rng.shuffle(deck)
            sess["hand"] = {"action": "", "client": deck[:2], "bot": deck[2:4], "board": deck[4:9], "done": False}
            self.hands += 1
            return self._bot_then_answer(tok, "")
        if endpoint == "act":
            tok = body.get("token")
            sess = self.sessions.get(tok)
            if sess is None or sess["hand"] is None or sess["hand"]["done"]:
                return {"error_msg": "no hand in progress"}
            h, x = sess["hand"], body.get("incr", "")
            old = h["action"]
            why = self._refuse(old, x, sess["client_pos"])
            if why:
                self.refused.append(why)
                return {"error_msg": why}
            h["action"] = extend(old, x)
            return self._bot_then_answer(tok, old)
        return {"error_msg": f"unknown endpoint {endpoint}"}

    def _refuse(self, old: str, x: str, client_pos: int) -> str | None:
        if not isinstance(x, str) or not _INCR.match(x):
            return f"not one action: {x!r}"
        try:
            s = parse(old)
        except ParseError as e:
            return f"bad history {old!r}: {e}"
        if s.pos != client_pos:
            return "not your turn"
        if x.startswith("b") and STACK - s.total_last_bet_to == 0:
            return "raise over an all-in"
        try:
            parse(old + x)
        except ParseError as e:
            return f"illegal action {x!r} after {old!r}: {e}"
        if self.referee is not None:
            r = self.referee(old + x)
            if isinstance(r, set) or "error" in r:
                return f"ParseAction refuses {old + x!r}: {r}"
        return None

    def _bot_then_answer(self, tok: str, old: str) -> dict:
        sess = self.sessions[tok]
        h, cp = sess["hand"], sess["client_pos"]
        s = parse(h["action"])
        while not s.over and s.pos != cp:
            h["action"] = extend(h["action"], random_action(self.rng, s))
            s = parse(h["action"])
        out = {"old_action": old, "action": h["action"], "client_pos": cp,
               "hole_cards": [card_name(c) for c in h["client"]],
               "board": [card_name(c) for c in h["board"][:BOARD_CARDS[s.st]]], "token": tok}
        if s.over:
            h["done"] = True
            cards = {cp: h["client"], 1 - cp: h["bot"]}
            w = payoff(s, cp, cards, h["board"])
            self.bot_net += payoff(s, 1 - cp, cards, h["board"])
            out["winnings"] = w
            if s.folded is None:
                out["bot_hole_cards"] = [card_name(c) for c in h["bot"]]
                out["board"] = [card_name(c) for c in h["board"]]
        return out

def random_action(rng: random.Random, s: State, fold: float = 0.15) -> str:
    """A random legal action for the player to act at `s`: facing a bet, fold (`fold`), call, or raise; else
    check or bet. A size is the least (30%), all-in (10%), uniform up to the pot (40%) or up to all-in."""
    r = rng.random()
    remaining = STACK - s.total_last_bet_to
    if s.last_bet_size > 0:  # facing a bet (or the blind)
        if r < fold:
            return "f"
        if r < fold + 0.45 or remaining == 0:
            return "c"
    elif r < 0.5:
        return "k"
    least = min(max(s.last_bet_size, BB) if s.last_bet_size > 0 else BB, remaining)
    u = rng.random()
    pot = max(least, min(remaining, s.total[0] + s.total[1]))
    size = least if u < 0.3 else remaining if u < 0.4 else rng.randint(least, pot) if u < 0.8 else rng.randint(least, remaining)
    return f"b{s.street_last_bet_to + size}"


def legal_history(rng: random.Random) -> str:
    """A whole random legal history as the server writes it (every street's '/', an all-in call's skipped
    streets)."""
    fold = rng.choice([0.15, 0.02])
    h = ""
    while not (s := parse(h)).over:
        h = extend(h, random_action(rng, s, fold=fold))
    return h


def parser_check(parse_action, histories: int = 10_000, seed: int = 20261005) -> dict:
    """PREREG's parser check: `parse` agrees with Slumbot's `ParseAction` on every prefix of `histories` random
    legal histories (the six fields where both accept; both refuse elsewhere). `passed` also needs the sample
    to cover long hands, all-ins and prefixes both refuse."""
    def mine(a):
        try:
            return parse(a).fields()
        except ParseError:
            return "error"

    def theirs(a):
        return "error" if refuses(parse_action, a) else parse_action(a)

    rng = random.Random(seed)
    prefixes = refused = longest = allin = 0
    for _ in range(histories):
        h = legal_history(rng)
        longest = max(longest, len(h))
        allin += parse(h).all_in_call
        for k in range(len(h) + 1):
            a, b = mine(h[:k]), theirs(h[:k])
            if a != b:
                return dict(passed=False, disagree=dict(prefix=h[:k], history=h, ours=str(a), parse_action=str(b)))
            prefixes += 1
            refused += a == "error"
    covered = prefixes > 150_000 and refused > 10_000 and longest > 40 and allin > 500
    return dict(passed=covered, histories=histories, seed=seed, prefixes=prefixes, both_refuse=refused,
                longest=longest, all_in_calls=allin)


def extend(action: str, x: str) -> str:
    """The history after action `x`, as the server writes it: a '/' after a street that `x` ended before the
    river, and one for each street an all-in call skipped."""
    before, after = parse(action), parse(action + x)
    if after.all_in_call:
        return action + x + "/" * (STREETS - 1 - before.st)
    if not after.over and after.st > before.st:
        return action + x + "/"
    return action + x


# ---------------------------------------------------------------- playing a hand

class ProtocolError(RuntimeError):
    """The server's answer disagrees with the rules or with the hand so far, or one of our options is not legal
    there: the run stops (a bug, ours, or a misreading of the API)."""


class Stop(RuntimeError):
    """`options` would not decide (a failed search): the hand is abandoned and the run stops."""


_KNOWN = {"old_action", "action", "client_pos", "hole_cards", "board", "token", "winnings", "bot_hole_cards"}


def card_index(name) -> int:
    """The engine's index of a card named as Slumbot names it ("Ac" = 48)."""
    if not isinstance(name, str) or len(name) != 2 or name[0] not in RANKS or name[1] not in SUITS:
        raise ProtocolError(f"not a card: {name!r}")
    return RANKS.index(name[0]) * 4 + SUITS.index(name[1])


def sample(rows: list, u: float) -> int:
    """The index of the option that `u` (uniform in [0, 1)) picks by the options' probabilities."""
    ps = [r[-1] for r in rows]
    if not ps or min(ps) < 0 or abs(sum(ps) - 1.0) > 1e-4:
        raise ProtocolError(f"the options are not a distribution: {rows}")
    acc, total = 0.0, sum(ps)
    for i, p in enumerate(ps):
        acc += p / total
        if u < acc:
            return i
    return max(i for i, p in enumerate(ps) if p > 0)  # u within rounding of 1


def _token(r: dict, token: str | None) -> str | None:
    if not isinstance(r, dict) or "error_msg" in r:
        raise ProtocolError(f"the server refused: {r.get('error_msg') if isinstance(r, dict) else r!r}")
    return r.get("token") or token


def play_hand(post, token: str | None, options, rng: random.Random, rec: dict, live=None) -> str | None:
    """Play one hand. `post(endpoint, body) -> dict` is the transport. `options(hero, hole, board, actions)`
    returns our holding's options `[(kind, add, prob)]` at the engine's seat `hero` and a dict of facts to
    record (it raises `Stop` to stop). `rng` samples them. `rec` is filled as the hand goes, so a hand cut
    short keeps what happened, and `live(rec)` is called after each answer. Returns the newest token.

    Every answer is checked: the history parses and extends ours, our seat and cards stay, the board has the
    street's cards, and the winnings are what the cards and chips give (`check`: "ok"; "unchecked" where a
    showdown's cards were not shown and the winnings are 0 or all of either side's chips; else "mismatch")."""
    t0 = time.time()
    rec.update(utc=_utc(t0), decisions=[], think_seconds=0.0)
    if live is not None:
        live(rec)  # before the first request: a kill from here on leaves the hand on record
    r = post("new_hand", {"token": token} if token else {})
    rec["token"] = token = _token(r, token)  # the newest, for the next hand even if this one fails
    cp = r.get("client_pos")
    if cp not in (0, 1):
        raise ProtocolError(f"client_pos {cp!r}")
    hole = [card_index(c) for c in r.get("hole_cards") or []]
    if len(hole) != 2 or hole[0] == hole[1]:
        raise ProtocolError(f"hole cards {r.get('hole_cards')!r}")
    rec.update(client_pos=cp, seat="BB" if cp == 0 else "SB", hole=list(r["hole_cards"]))
    prev = None  # the actions up to and including our last one
    while True:
        h = r.get("action")
        if not isinstance(h, str):
            raise ProtocolError(f"no history in {r!r}")
        try:
            s = parse(h)
        except ParseError as e:
            raise ProtocolError(f"the server's history {h!r}: {e}") from None
        if prev is not None and s.acts[:len(prev)] != prev:
            raise ProtocolError(f"the history {h!r} does not extend ours {prev}")
        if cp in s.movers[len(prev or []):]:  # an action of ours we did not send: a retry applied twice
            raise Lost(f"the history {h!r} has an action of ours after our last {prev}")
        if r.get("client_pos") != cp or r.get("hole_cards") != rec["hole"]:
            raise ProtocolError(f"the seat or the cards changed mid-hand: {r!r}")
        board = [card_index(c) for c in r.get("board") or []]
        if len(set(board + hole)) != len(board) + 2:
            raise ProtocolError(f"repeated cards: {rec['hole']} {r.get('board')}")
        rec.update(action=h, board=list(r.get("board") or []))
        if "winnings" in r:
            break
        if s.over or s.pos != cp:
            raise ProtocolError(f"no winnings, and not our turn after {h!r}")
        if len(board) != BOARD_CARDS[s.st]:
            raise ProtocolError(f"{len(board)} board cards on street {s.st}")
        if live is not None:
            live(rec)
        rows, info = options(engine_seat(cp), hole, board, s.acts)
        i = sample(rows, rng.random())
        kind, add = rows[i][0], rows[i][1]
        x = incr(s, kind, add)
        try:
            after = parse(h + x)
        except ParseError as e:
            raise ProtocolError(f"our option {(kind, add)} as {x!r} after {h!r}: {e}") from None
        if after.acts[-1] != (kind, add) or after.movers[-1] != cp:
            raise ProtocolError(f"{x!r} after {h!r} reads back as {after.acts[-1]}, not {(kind, add)}")
        if kind == "bet" and after.total[cp] <= after.total[1 - cp]:
            raise ProtocolError(f"our raise {x!r} after {h!r} raises nothing")
        rec["decisions"].append(dict(street=s.st, at=h, options=[[k, a, round(p, 6)] for k, a, p in rows], pick=i,
                                     incr=x, **info))
        rec["think_seconds"] += info.get("seconds", 0.0)
        prev = after.acts
        r = post("act", {"token": token, "incr": x})
        rec["token"] = token = _token(r, token)
    w = r["winnings"]
    if not s.over or isinstance(w, bool) or not isinstance(w, int):
        raise ProtocolError(f"winnings {w!r} after {h!r}")
    cards = {cp: hole}
    if r.get("bot_hole_cards"):
        cards[1 - cp] = [card_index(c) for c in r["bot_hole_cards"]]
        rec["bot_hole"] = list(r["bot_hole_cards"])
        if len(set(cards[1 - cp] + hole + board)) != len(board) + 4:
            raise ProtocolError(f"repeated cards: {rec['hole']} {rec['bot_hole']} {rec['board']}")
    want = payoff(s, cp, cards, board)
    ok = w == want if want is not None else w in (0, s.total[1 - cp], -s.total[cp])
    rec.update(winnings=w, check="mismatch" if not ok else "ok" if want is not None else "unchecked",
               put_in=s.total[cp], pot=s.total[0] + s.total[1], street=s.st, showdown=s.folded is None,
               all_in=s.all_in_call)
    if want is not None and not ok:
        rec["want"] = want
        if live is not None:
            live(rec)  # on record at once: a mismatch stops the run, even one killed before this hand ends
    eq =all_in_equity(s, cp, cards, board) if len(cards) == 2 else None
    if eq is not None:
        rec.update(allin_equity=round(eq, 6), allin_adjusted=round(eq * (s.total[0] + s.total[1]) - s.total[cp], 3))
    extra = {k: v for k, v in r.items() if k not in _KNOWN}
    if extra:
        rec["extra"] = extra
    rec["seconds"] = time.time() - t0
    if live is not None:
        live(rec)
    return token


SAMPLE_SHA256 = "17ab1f6c1a25db1822cef7f34ed67b6516ac65ea24897adcc92d8f2db8f71505"  # sample_api.py, fetched 5 Oct 2026 16:38


def load_parse_action(path, sha256: str = SAMPLE_SHA256):
    """`ParseAction` of Slumbot's `sample_api.py` at `path`, checked against `sha256` (ValueError if it is
    another file; None if it is missing). Its module imports `requests`, which the venv lacks: a stand-in
    module is given for the import, and only `ParseAction` is used."""
    import hashlib
    import importlib.util
    import sys
    import types
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return None
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != sha256:
        raise ValueError(f"{path} has sha256 {got}, not the registered {sha256}")
    saved = sys.modules.get("requests")
    sys.modules["requests"] = types.ModuleType("requests")
    try:
        spec = importlib.util.spec_from_file_location("slumbot_sample_api", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is None:
            sys.modules.pop("requests", None)
        else:
            sys.modules["requests"] = saved
    return mod.ParseAction


def refuses(parse_action, action: str) -> bool:
    """Whether Slumbot's `ParseAction` refuses `action` (an illegal fold returns a set there)."""
    r = parse_action(action)
    return isinstance(r, set) or "error" in r
