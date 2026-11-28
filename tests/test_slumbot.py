"""The Slumbot client's protocol layer (src/parciver/slumbot.py; records/slumbot/PREREG.md).

The pre-registered parser check: `parse` agrees with `ParseAction` of Slumbot's own `sample_api.py` on every
prefix of 10,000 random legal histories (the six fields where both accept; both refuse elsewhere). The file
is Slumbot's, not ours: it is read from PARCIVER_SLUMBOT_SAMPLE (default C:/tmp/slumbot/sample_api.py), and
the test skips where it is missing. Only its ParseAction runs (its module imports `requests`, which the venv
lacks: a stand-in module is given for the import).
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
pytest.importorskip("parciver_engine")  # payoffs and hand evaluation run in the Rust engine
from parciver import slumbot as S  # noqa: E402

SAMPLE = Path(os.environ.get("PARCIVER_SLUMBOT_SAMPLE", "C:/tmp/slumbot/sample_api.py"))


def referee():
    """Slumbot's ParseAction, or a skip."""
    try:
        parse_action = S.load_parse_action(SAMPLE)
    except ValueError as e:
        pytest.fail(f"{e}: not the sample_api.py this check was registered with")
    if parse_action is None:
        pytest.skip(f"{SAMPLE} missing (Slumbot's sample_api.py)")
    return parse_action


legal_history = S.legal_history


def theirs(parse_action, action: str):
    return "error" if S.refuses(parse_action, action) else parse_action(action)


def test_parse_agrees_with_parse_action_on_every_prefix():
    out = S.parser_check(referee())  # the runner's mock run records the same check
    assert out["passed"], out
    # the sample covers what it should: long hands, all-ins, and prefixes both refuse (cut bet sizes)
    assert out["prefixes"] > 150_000 and out["both_refuse"] > 10_000, out


def test_documented_histories():
    s = S.parse("b200c/kk/kk/kb200")
    assert (s.st, s.pos, s.total, s.over) == (3, 0, [200, 400], False)
    assert s.acts == [("bet", 150), ("call", 0)] + [("check", 0)] * 5 + [("bet", 200)]
    s = S.parse("b20000c///")
    assert s.over and s.all_in_call and s.total == [20000, 20000] and s.acts == [("bet", 19950), ("call", 0)]
    s = S.parse("b200c/kb400")  # the docs' pot-size flop bet: 600 each if called
    assert s.pos == 0 and s.to_call(0) == 400 and S.parse("b200c/kb400c").total == [600, 600]
    for bad in ("k", "b150", "b20001", "b200c/c", "b200f/", "ck/k/kk/kk/kkk", "b200c/b100x", "ck/f", "ckf"):
        with pytest.raises(S.ParseError):
            S.parse(bad)


def test_every_option_round_trips_to_the_engine_action():
    """`incr` writes each option so that `parse` reads back the same engine action, by the same position."""
    rng = random.Random(7)
    for _ in range(3000):
        h = ""
        while not (s := S.parse(h)).over:
            opts = [("check", 0)] if s.last_bet_size == 0 else [("fold", 0), ("call", 0)]
            remaining = S.STACK - s.total_last_bet_to
            least = min(max(s.last_bet_size, S.BB) if s.last_bet_size > 0 else S.BB, remaining)
            if remaining:
                for size in {least, remaining, rng.randint(least, remaining)}:
                    opts.append(("bet", s.to_call(s.pos) + size))
            kind, add = rng.choice(opts)
            x = S.incr(s, kind, add)
            after = S.parse(h + x)
            assert after.acts[-1] == (kind, add) and after.movers[-1] == s.pos, (h, x)
            h = S.extend(h, x)


def test_payoffs_conserve_chips():
    rng = random.Random(3)
    for _ in range(2000):
        h = legal_history(rng)
        deck = list(range(52))
        rng.shuffle(deck)
        cards, board = {0: deck[:2], 1: deck[2:4]}, deck[4:9]
        s = S.parse(h)
        a, b = S.payoff(s, 0, cards, board), S.payoff(s, 1, cards, board)
        assert a + b == 0 and abs(a) <= min(s.total), (h, a, b)
        if s.folded is not None:
            assert a == (-s.total[0] if s.folded == 0 else s.total[1])


def test_mock_refuses_what_slumbot_would():
    m = S.MockSlumbot(seed=1, referee=None)
    r = m.post("new_hand", {})
    tok, h = r["token"], r["action"]
    assert r["client_pos"] == 0 and len(r["hole_cards"]) == 2  # a session starts in the big blind
    s = S.parse(h)
    if not s.over:
        for bad in ("b1", "x", "kk", "b20001", "/"):
            assert "error_msg" in m.post("act", {"token": tok, "incr": bad}), bad
    # the seat changes every hand
    seats = [m.post("new_hand", {"token": tok})["client_pos"] for _ in range(4)]
    assert seats == [1, 0, 1, 0]


def test_mock_hands_are_legal_for_parse_action():
    """The mock's own histories (the opponent's actions and its slashes) pass ParseAction."""
    parse_action = referee()
    m = S.MockSlumbot(seed=11, referee=parse_action)
    rng, tok, net = random.Random(5), None, 0
    for _ in range(500):
        r = m.post("new_hand", {"token": tok} if tok else {})
        tok = r["token"]
        while "winnings" not in r:
            s = S.parse(r["action"])
            assert s.pos == r["client_pos"] and len(r["board"]) == S.BOARD_CARDS[s.st]
            r = m.post("act", {"token": tok, "incr": S.random_action(rng, s)})
            assert "error_msg" not in r, r
        assert theirs(parse_action, r["action"]) != "error", r["action"]
        net += r["winnings"]
    assert net + m.bot_net == 0 and not m.refused


def history_of(acts) -> str:
    """The server's history string for the engine's actions."""
    h = ""
    for kind, add in acts:
        h = S.extend(h, S.incr(S.parse(h), kind, add))
    return h


def random_options(rng: random.Random):
    """A stand-in for the engine's options: every legal kind and three sizes, random probabilities."""
    def options(hero, hole, board, acts):
        s = S.parse(history_of(acts))
        assert S.engine_seat(s.pos) == hero and len(board) == S.BOARD_CARDS[s.st] and len(hole) == 2
        rows = [("check", 0)] if s.last_bet_size == 0 else [("fold", 0), ("call", 0)]
        remaining = S.STACK - s.total_last_bet_to
        least = min(max(s.last_bet_size, S.BB) if s.last_bet_size > 0 else S.BB, remaining)
        if remaining:
            rows += [("bet", s.to_call(s.pos) + z) for z in sorted({least, remaining, rng.randint(least, remaining)})]
        w = [rng.random() for _ in rows]
        return [(k, a, x / sum(w)) for (k, a), x in zip(rows, w)], {"seconds": 0.0}
    return options


def test_play_hand_against_the_mock():
    """The client's hand loop: chips conserved, every winnings checked, seats alternating, every decision kept."""
    parse_action = referee()
    m = S.MockSlumbot(seed=21, referee=parse_action)
    rng, tok, net, seats, checks, decisions = random.Random(8), None, 0, [], set(), 0
    options = random_options(random.Random(9))
    for _ in range(400):
        rec = {}
        tok = S.play_hand(m.post, tok, options, rng, rec)
        net += rec["winnings"]
        seats.append(rec["client_pos"])
        checks.add(rec["check"])
        decisions += len(rec["decisions"])
        assert not S.refuses(parse_action, rec["action"])
        assert len(rec["decisions"]) == sum(1 for p in S.parse(rec["action"]).movers if p == rec["client_pos"])
        if rec.get("allin_equity") is not None:
            assert rec["all_in"] and rec["showdown"] and 0.0 <= rec["allin_equity"] <= 1.0
    assert net + m.bot_net == 0 and not m.refused and checks == {"ok"} and decisions > 400
    assert seats == [0, 1] * 200


def test_play_hand_refuses_a_wrong_answer():
    """A server whose winnings disagree with the cards is caught, and a refused action stops the hand."""
    m = S.MockSlumbot(seed=4)
    real = m._bot_then_answer

    def lying(tok, old):
        out = real(tok, old)
        if "winnings" in out:
            out["winnings"] += 1
        return out

    m._bot_then_answer = lying
    rec = {}
    S.play_hand(m.post, None, random_options(random.Random(1)), random.Random(2), rec)
    assert rec["check"] == "mismatch" and rec["want"] == rec["winnings"] - 1
    with pytest.raises(S.ProtocolError):
        S.play_hand(lambda e, b: {"error_msg": "Illegal action"}, "t", random_options(random.Random(1)),
                    random.Random(2), {})


def test_play_hand_loses_a_hand_with_an_action_applied_twice():
    """A retried act the server applied twice puts an action of ours in the history that the engine did not
    choose: the hand is lost, never counted."""
    m = S.MockSlumbot(seed=7)
    real = m.post

    def twice(endpoint, body):
        out = real(endpoint, body)
        if endpoint == "act" and "error_msg" not in out and "winnings" not in out:
            again = real(endpoint, body)
            if "error_msg" not in again:
                return again
        return out

    options, rng, tok = random_options(random.Random(3)), random.Random(4), None
    lost = played = 0
    for _ in range(200):
        rec = {}
        try:
            tok = S.play_hand(twice, tok, options, rng, rec)
            played += 1
            assert len(rec["decisions"]) == sum(1 for p in S.parse(rec["action"]).movers if p == rec["client_pos"])
        except S.Lost:
            lost += 1
            tok = rec.get("token", tok)
    assert lost > 20 and played > 20, (lost, played)


class _Scripted(BaseHTTPRequestHandler):
    script: list = []

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        step = self.script.pop(0)
        if step == "drop":
            self.close_connection = True
            self.connection.close()
            return
        if step == "short":  # an answer cut mid-body (http.client.IncompleteRead)
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b'{"tok')
            self.wfile.flush()
            self.close_connection = True
            self.connection.close()
            return
        code, body = step
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_http_retries_and_losses(monkeypatch):
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Scripted)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log = []
    http = S.Http(log.append, host=f"http://127.0.0.1:{srv.server_address[1]}/", timeout=5, retries=2)
    try:
        _Scripted.script[:] = [(200, {"action": "b200", "token": "t"})]
        assert http.post("act", {"incr": "c"}) == {"action": "b200", "token": "t"} and http.retried == 0
        _Scripted.script[:] = [(500, {}), "drop", (200, {"token": "u"})]  # a 5xx and a dropped answer: retried
        assert http.post("new_hand", {}) == {"token": "u"} and http.retried == 2
        _Scripted.script[:] = ["short", (200, {"token": "v"})]  # a cut answer: retried
        assert http.post("new_hand", {}) == {"token": "v"} and http.retried == 3
        _Scripted.script[:] = [(400, {"error_msg": "Illegal action"})]  # the first try refused: the caller's to read
        assert http.post("act", {"incr": "x"})["error_msg"] == "Illegal action"
        _Scripted.script[:] = ["drop", (400, {"error_msg": "not your turn"})]  # a retry refused: lost
        with pytest.raises(S.Lost):
            http.post("act", {"incr": "c"})
        _Scripted.script[:] = ["drop", "drop", "drop"]
        with pytest.raises(S.Lost):
            http.post("act", {"incr": "c"})
    finally:
        srv.shutdown()
    assert all("endpoint" in r and "attempt" in r for r in log) and len(log) == 1 + 3 + 2 + 1 + 2 + 3
