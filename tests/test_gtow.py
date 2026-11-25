# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""GTO Wizard API connector: the state translation (parciver.gtow) and the runner
(scripts/gtow_bench.py) against an in-process fake server that follows the API as its
client documents it (github.com/gtowizard-ai/researcher-api-client). The fake can only
check our reading of the protocol for self-consistency; the runner's live checks (pot,
stack, seat) are what validate it against the real server.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parciver import gtow  # noqa: E402

STACK, SB, BB = 20_000, 50, 100


# ---------------------------------------------------------------- pure translation

def test_parse_cards_any_spelling():
    assert gtow.parse_cards("AsKd") == [51, 45]
    assert gtow.parse_cards("As Kd 2c") == [51, 45, 0]
    assert gtow.parse_cards("as,kd") == [51, 45]
    assert gtow.parse_cards("") == [] and gtow.parse_cards(None) == []
    assert gtow.card_names([51, 45, 0]) == "AsKd2c"


def test_replay_limp_check_and_postflop_order():
    r = gtow.replay(["c", "k", "_", "k"], (SB, BB), STACK)
    assert r.actions == [("c", 0), ("x", 0), ("x", 0)]
    assert r.contributed == [100, 100] and r.street == 1 and r.to_act == 0


def test_replay_raises_are_round_cumulative():
    # SB raises to 300, BB 3-bets to 900, SB calls; flop BB bets 600, SB raises to 2000.
    h = ["b300", "b900", "c", "_", "b600", "b2000"]
    r = gtow.replay(h, (SB, BB), STACK)
    assert r.actions == [("b", 250), ("b", 800), ("c", 0), ("b", 600), ("b", 2000)]
    assert r.contributed == [2900, 1500] and r.to_act == 1
    # The same history if preflop amounts exclude the posted blind.
    r2 = gtow.replay(h, (SB, BB), STACK, "exclusive")
    assert r2.actions[:2] == [("b", 300), ("b", 900)]


def test_replay_rejects_impossible_histories():
    with pytest.raises(gtow.Desync):
        gtow.replay(["k"], (SB, BB), STACK)  # SB cannot check preflop
    with pytest.raises(gtow.Desync):
        gtow.replay(["b50"], (SB, BB), STACK)  # not a raise
    with pytest.raises(gtow.Desync):
        gtow.replay(["c", "k", "_", "q"], (SB, BB), STACK)


def _response(history, board="", total=None, me_seat=0, legal=("f", "c", "b"), rr=None):
    r = gtow.replay(history, (SB, BB), STACK)
    if rr is None:
        street_in = r.round_bet[me_seat] if not board else 0
        rr = (200, street_in + STACK - r.contributed[me_seat])
    return {"hand_id": 1, "game": {"starting_stack": float(STACK), "blinds": [float(SB), float(BB)]},
            "game_state": {"action_history": history, "board_cards": board,
                           "total_pot": float(sum(r.contributed) if total is None else total),
                           "common_pot": 0.0, "legal_actions": list(legal),
                           "raise_range": {"min": rr[0], "max": rr[1]},
                           "players": [{"name": "GTO Wizard", "stack": 0.0, "position": ("BB", "SB")[me_seat],
                                        "hole_cards": None},
                                       {"name": "me", "stack": float(STACK - r.contributed[me_seat]),
                                        "position": ("SB", "BB")[me_seat], "hole_cards": "AsKd"}]}}


def test_spot_checks_pot_seat_and_street():
    s = gtow.spot(_response([], me_seat=0))
    assert (s.seat, s.hole, s.board, s.basis) == (0, [51, 45], [], "inclusive")
    # BB facing a raise to 300.
    s = gtow.spot(_response(["b300"], me_seat=1))
    assert s.seat == 1 and s.replay.round_bet == [300, 100]
    # A new street with no '_' yet: the board says it started.
    s = gtow.spot(_response(["c", "k"], board="2c3d4h", me_seat=1, legal=("k", "b")))
    assert s.replay.street == 1 and s.seat == 1
    with pytest.raises(gtow.Desync, match="pot"):
        gtow.spot(_response(["b300"], total=999, me_seat=1))
    with pytest.raises(gtow.Desync, match="seat"):
        gtow.spot(_response(["b300"], me_seat=0))  # the server says we are SB, but BB is to act


def test_to_request_amounts_clamps_and_fallbacks():
    s = gtow.spot(_response(["b300"], me_seat=1, rr=(500, 20000)))
    assert s.certain  # a preflop raise range pins the bet convention
    # BB (100 in) re-raises by adding 800: to 900 on the round.
    req, note = gtow.to_request(s, {"action": "bet", "add": 800, "all_in": False}, ["f", "c", "b"],
                                {"min": 500, "max": 20000})
    assert req == {"action": "b", "amount": 900} and note is None
    req, note = gtow.to_request(s, {"action": "bet", "add": 300, "all_in": False}, ["f", "c", "b"],
                                {"min": 500, "max": 20000})
    assert req == {"action": "b", "amount": 500} and "below min" in note
    req, _ = gtow.to_request(s, {"action": "bet", "add": 19900, "all_in": True}, ["f", "c", "b"],
                             {"min": 500, "max": 20000})
    assert req == {"action": "b", "amount": 20000}
    assert gtow.to_request(s, {"action": "fold", "add": 0}, ["f", "c"], None)[0]["action"] == "f"
    assert gtow.to_request(s, {"action": "call", "add": 200}, ["f", "c"], None)[0]["action"] == "c"
    req, note = gtow.to_request(s, {"action": "bet", "add": 800, "all_in": False}, ["f", "c"], None)
    assert req["action"] == "c" and note


# ---------------------------------------------------------------- fake server

class FakeHand:
    """A heads-up NLHE hand as the API presents it; the villain plays randomly."""

    def __init__(self, hid, rng, hero, basis="inclusive"):
        deck = list(range(52))
        rng.shuffle(deck)
        self.id, self.rng, self.hero, self.basis = hid, rng, hero, basis
        self.holes, self.board = [deck[0:2], deck[2:4]], deck[4:9]
        self.contrib, self.round, self.street, self.to_act = [SB, BB], [SB, BB], 0, 0
        self.hist, self.over, self.winner, self.last_raise, self.first = [], False, None, BB, True

    def owed(self, p):
        return self.round[1 - p] - self.round[p]

    def blind(self, p):
        return (SB, BB)[p] if self.street == 0 and self.basis == "exclusive" else 0

    def legal(self):
        p = self.to_act
        acts = ["f", "c"] if self.owed(p) > 0 else ["k"]
        if self.contrib[p] + self.owed(p) < STACK and self.contrib[1 - p] < STACK:
            acts.append("b")
        return acts

    def raise_range(self):
        p = self.to_act
        hi = self.round[p] + STACK - self.contrib[p]
        lo = min(self.round[1 - p] + max(self.last_raise, BB), hi)
        return {"min": float(lo - self.blind(p)), "max": float(hi - self.blind(p))}

    def apply(self, a, amount=None):
        p = self.to_act
        if a not in self.legal():
            raise ValueError(f"illegal {a}")
        if a == "f":
            self.hist.append("f")
            self.over, self.winner = True, 1 - p
            return
        if a == "k":
            self.hist.append("k")
            closes = (self.street == 0 and p == 1) or (self.street > 0 and p == 0)
        elif a == "c":
            add = min(self.owed(p), STACK - self.contrib[p])
            self.round[p] += add
            self.contrib[p] += add
            self.hist.append("c")
            closes = not (self.street == 0 and p == 0 and self.first)
        else:
            rr = self.raise_range()
            if not rr["min"] <= amount <= rr["max"]:
                raise ValueError(f"bet {amount} outside {rr}")
            to = int(amount) + self.blind(p)
            self.last_raise = max(to - self.round[1 - p], BB)
            self.contrib[p] += to - self.round[p]
            self.round[p] = to
            self.hist.append(f"b{int(amount)}")
            closes = False
        self.first = False
        if not closes:
            self.to_act = 1 - p
        elif STACK in self.contrib or self.street == 3:
            self.over, self.winner = True, self.rng.choice([0, 1, None])
            self.street = 3
        else:
            self.street += 1
            self.round, self.to_act, self.last_raise = [0, 0], 1, BB
            self.hist.append("_")

    def villain(self):
        while not self.over and self.to_act != self.hero:
            acts = self.legal()
            a = self.rng.choice(acts + ["c" if "c" in acts else "k"] * 2)
            amount = None
            if a == "b":
                rr = self.raise_range()
                amount = self.rng.choice([rr["min"], rr["max"], (rr["min"] + rr["max"]) // 2])
            self.apply(a, amount)

    def response(self, pot_error=0.0):
        me, vil = self.hero, 1 - self.hero
        n_board = (0, 3, 4, 5)[self.street]
        winnings = None
        if self.over:
            winnings = 0.0 if self.winner is None else float(
                self.contrib[vil] if self.winner == me else -self.contrib[me])
        players = [{"name": "GTO Wizard", "stack": float(STACK - self.contrib[vil]), "position": ("SB", "BB")[vil],
                    "hole_cards": gtow.card_names(self.holes[vil]) if self.over else None},
                   {"name": "researcher", "stack": float(STACK - self.contrib[me]), "position": ("SB", "BB")[me],
                    "hole_cards": gtow.card_names(self.holes[me])}]
        if self.id % 2:
            players.reverse()
        return {"hand_id": self.id,
                "game": {"game_id": 1, "game_name": "HUNL 200BB", "game_format": "HUNL", "starting_stack": 20000.0,
                         "blinds": [50.0, 100.0], "stack_reset_per_hand": True},
                "game_state": {"street": ["preflop", "flop", "turn", "river"][self.street],
                               "common_pot": float(sum(self.contrib) - sum(self.round)),
                               "total_pot": float(sum(self.contrib)) + pot_error,
                               "board_cards": gtow.card_names(self.board[:n_board]), "is_hand_over": self.over,
                               "players": players, "legal_actions": [] if self.over else self.legal(),
                               "raise_range": None if self.over or "b" not in self.legal() else self.raise_range(),
                               "action_history": list(self.hist), "has_gto_wizard_folded": self.winner == me,
                               "winnings": winnings, "aivat_score": winnings}}


class FakeServer:
    def __init__(self, basis="inclusive", busy_every=0, pot_error_at=None, key="k"):
        self.hands, self.rng, self.lock, self.next_id = {}, random.Random(7), threading.Lock(), 1
        self.basis, self.busy_every, self.pot_error_at, self.key, self.calls = basis, busy_every, pot_error_at, key, 0
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                if self.headers.get("User-Agent", "").startswith("Python-urllib"):  # as the real Cloudflare does
                    return self._send(403, {"title": "Error 1010: Access denied", "cloudflare_error": True})
                if self.headers.get("X-API-Key") != fake.key:
                    return self._send(401, {"detail": "bad key"})
                with fake.lock:
                    fake.calls += 1
                    if fake.busy_every and fake.calls % fake.busy_every == 0:
                        return self._send(503, {"detail": "busy"})
                    try:
                        code, out = fake.handle(self.path, body)
                    except ValueError as e:
                        code, out = 400, {"detail": str(e)}
                self._send(code, out)

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def handle(self, path, body):
        if path == "/hands":
            assert body == {"game_name": "HUNL 200BB"}
            hid = self.next_id
            self.next_id += 1
            h = self.hands[hid] = FakeHand(hid, random.Random(hid), hid % 2, self.basis)
            h.villain()
            return 200, h.response(self._pot_error(h))
        hid = int(path.split("/")[2])
        h = self.hands[hid]
        if h.over or h.to_act != h.hero:
            raise ValueError("not your turn")
        h.apply(body["action"], body.get("amount"))
        h.villain()
        return 200, h.response(self._pot_error(h))

    def _pot_error(self, h):
        return 7.0 if self.pot_error_at is not None and h.id >= self.pot_error_at and not h.over else 0.0

    def close(self):
        self.httpd.shutdown()


def _cli():
    spec = importlib.util.spec_from_file_location("gtow_bench_cli", ROOT / "scripts" / "gtow_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(cli, run_dir, hands, **kw):
    import argparse
    base = dict(candidate="x.bin", abstraction_dir="", label="t", hands=hands, run=str(run_dir), concurrency=3,
                search="none", river_iters=20, turn_iters=5, own_range="exact", round="solve", raw=2, every=1000, key="k",
                url="")
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def engine_agent():
    pe = pytest.importorskip("parciver_engine")
    return pe.PlayAgent.uniform()


@pytest.mark.parametrize("basis", ["inclusive", "exclusive"])
def test_runner_plays_full_hands_on_both_seats(tmp_path, engine_agent, basis, monkeypatch):
    cli = _cli()
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    srv = FakeServer(basis=basis, busy_every=9)
    try:
        args = _args(cli, tmp_path / "run", 24)
        assert cli.run(args, engine_agent, cli.Api(srv.url, "k")) == 0
    finally:
        srv.close()
    recs = [json.loads(line) for line in (tmp_path / "run" / "hands.jsonl").read_text().splitlines()]
    assert len(recs) == 24 and len({r["hand_id"] for r in recs}) == 24
    assert {r["seat"] for r in recs if r["seat"] is not None} == {0, 1}
    assert all(r["basis"] in (None, basis) for r in recs)
    assert sum(len(r["decisions"]) for r in recs) > 24
    assert all(r["winnings"] is not None for r in recs)
    assert len(list((tmp_path / "run" / "raw").glob("hand-*.json"))) == 2
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    want = sum(r["winnings"] / 100 for r in recs) / 24 * 100
    assert summary["hands"] == 24 and summary["winnings_bb100"] == pytest.approx(want)


def test_runner_resumes_to_the_target(tmp_path, engine_agent):
    cli = _cli()
    srv = FakeServer()
    try:
        assert cli.run(_args(cli, tmp_path / "run", 5), engine_agent, cli.Api(srv.url, "k")) == 0
        assert cli.run(_args(cli, tmp_path / "run", 9), engine_agent, cli.Api(srv.url, "k")) == 0
        with pytest.raises(SystemExit, match="different bot"):
            cli.run(_args(cli, tmp_path / "run", 12, search="river"), engine_agent, cli.Api(srv.url, "k"))
    finally:
        srv.close()
    assert len((tmp_path / "run" / "hands.jsonl").read_text().splitlines()) == 9


def test_runner_stops_on_a_pot_mismatch_with_the_transcript(tmp_path, engine_agent):
    cli = _cli()
    srv = FakeServer(pot_error_at=3)
    try:
        rc = cli.run(_args(cli, tmp_path / "run", 40, concurrency=1), engine_agent, cli.Api(srv.url, "k"))
    finally:
        srv.close()
    assert rc == 1
    failed = list((tmp_path / "run" / "raw").glob("failed-*.json"))
    assert len(failed) == 1 and "pot" in json.loads(failed[0].read_text())["error"]
    assert len((tmp_path / "run" / "hands.jsonl").read_text().splitlines()) < 40


def test_runner_stops_on_a_rejected_key(tmp_path, engine_agent):
    cli = _cli()
    srv = FakeServer(key="right")
    try:
        rc = cli.run(_args(cli, tmp_path / "run", 5), engine_agent, cli.Api(srv.url, "wrong"))
    finally:
        srv.close()
    assert rc == 1
    assert "rejected" in json.loads((tmp_path / "run" / "summary.json").read_text())["stopped"]


def test_key_file_in_any_windows_encoding(tmp_path, monkeypatch):
    cli = _cli()
    monkeypatch.delenv("GTOW_API_KEY", raising=False)
    for enc in ("utf-8", "utf-8-sig", "utf-16"):
        f = tmp_path / f"key-{enc}"
        f.write_bytes("abcDEF123\r\n".encode(enc))
        monkeypatch.setattr(cli, "KEY_FILE", f)
        assert cli.api_key(None) == "abcDEF123"
    assert cli.api_key("given") == "given"


def test_a_bad_certificate_stops_at_once_with_the_fix(tmp_path, monkeypatch):
    """A certificate the client can't verify is not retried (31 retries of an error that can't
    change): the run stops at once and says whether truststore or GTO Wizard fixes it."""
    import shutil
    import ssl
    import subprocess

    if shutil.which("openssl") is None:
        pytest.skip("needs the openssl command to make a certificate")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=127.0.0.1",
                    "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    cli = _cli()
    naps: list[float] = []
    me, real_sleep = threading.get_ident(), cli.time.sleep  # time.sleep is global: other tests' threads sleep too
    monkeypatch.setattr(cli.time, "sleep", lambda s: naps.append(s) if threading.get_ident() == me else real_sleep(s))
    try:
        api = cli.Api(f"https://127.0.0.1:{httpd.server_address[1]}", "k", timeout=10)
        with pytest.raises(cli.Fatal, match="certificate check failed") as err:
            api.post("/hands", {}, threading.Event())
    finally:
        httpd.shutdown()
    assert naps == []
    advice = "install --python .venv/Scripts/python.exe truststore" if api.verifier != cli.SYSTEM_VERIFIER else "GTO Wizard's to fix"
    assert advice in str(err.value)
