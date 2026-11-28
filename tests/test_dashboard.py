"""Tests for the dashboard producers and viewer server (Phase 3c, pre-B4).

The load-bearing contract is covered by ``test_telemetry.py``; these cover the
pieces built on top of it: the two telemetry producers emit schema-valid logs the
viewer can read (the real small-game driver AND the full-scale simulator), the
simulator's range model has the right shape, and the stdlib server actually serves
the SPA and replays the event log over SSE.
"""
from __future__ import annotations

import importlib.util
import json
import socket
import threading
import urllib.request
from pathlib import Path

import pytest

from parciver import telemetry as T

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    """Import a ``scripts/<name>.py`` producer as a module."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:  # scripts that import the Rust engine skip on a Python-only install
        if e.name == "parciver_engine":
            pytest.skip("needs the Rust engine (parciver_engine)")
        raise
    return mod


# --- simulator (replay_demo): pure model, no engine needed ------------------

def test_replay_hand_strength_orders_hands():
    replay = _load_script("replay_demo")
    assert replay.hand_strength("AA") > replay.hand_strength("KK") > replay.hand_strength("22")
    assert replay.hand_strength("AKs") > replay.hand_strength("72o")
    assert replay.hand_strength("JTs") > replay.hand_strength("J2o")  # suited connector beats junky offsuit


def test_replay_target_mix_is_a_distribution():
    replay = _load_script("replay_demo")
    for s in (0.05, 0.35, 0.5, 0.65, 0.95):
        m = replay.target_mix(s)
        assert abs(float(m.sum()) - 1.0) < 1e-9
        assert all(x >= 0 for x in m)
    # Premiums raise heavily, trash folds heavily.
    assert replay.target_mix(0.95)[2] > 0.85   # raise
    assert replay.target_mix(0.05)[0] > 0.85   # fold


def test_replay_demo_emits_valid_full_scale_log(tmp_path):
    pytest.importorskip("parciver_engine")
    replay = _load_script("replay_demo")
    replay.run(tmp_path, target_iters=1_000_000, duration=0.1, tick=0.05,
               checkpoint_every=1, fast=False, seed=3)
    events = T.read_events(tmp_path)  # read_events validates every line
    types = [e["type"] for e in events]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert events[-1]["reason"] == "completed"
    # A checkpoint carried the full 169-hand range grid the grid panel needs.
    grids = [e for e in events if e["type"] == "checkpoint" and e.get("range_grid")]
    assert grids and len(grids[-1]["range_grid"]["cells"]) == T.NUM_PREFLOP_HANDS
    # Head-to-head and the regret proxy were populated for their panels.
    ms = [e for e in events if e["type"] == "metrics"]
    assert any(e.get("hh_bb_per_100") is not None for e in ms)
    assert any(e.get("mean_abs_regret") is not None for e in ms)


# --- real driver (train_small): needs the Rust engine -----------------------

def test_train_small_emits_real_convergence(tmp_path):
    pytest.importorskip("parciver_engine")
    train = _load_script("train_small")
    train.run(tmp_path, variant="linear", seed=0, iters_max=8000, points=4,
              min_step_seconds=0.0)
    events = T.read_events(tmp_path)
    assert events[0]["type"] == "run_start" and events[-1]["type"] == "run_end"
    expl = [e["exploitability"] for e in events
            if e["type"] == "metrics" and e.get("exploitability") is not None]
    assert len(expl) >= 3
    # Exact exploitability must fall as the MCCFR budget grows (the whole point).
    assert expl[-1] < expl[0]
    assert events[-1]["summary"]["final_exploitability"] <= min(expl) + 1e-12


def test_train_small_helpers():
    pytest.importorskip("parciver_engine")
    train = _load_script("train_small")
    sched = train.geometric_schedule(200, 100000, 10)
    assert sched[0] == 200 and sched[-1] == 100000
    assert all(sched[i] < sched[i + 1] for i in range(len(sched) - 1))  # strictly increasing
    # L1 delta over shared keys.
    prev = {"a": [0.5, 0.5], "b": [1.0, 0.0]}
    cur = {"a": [0.5, 0.5], "b": [0.0, 1.0]}
    assert train.strategy_l1_delta(prev, cur) == pytest.approx(1.0)  # only b moved, by 2 -> mean 1.0
    assert train.strategy_l1_delta({}, cur) is None


# --- viewer server ----------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _read_one_sse_frame(host: str, port: int, timeout: float = 3.0) -> str:
    """Open /events over a raw socket and return the first complete SSE frame
    (terminated by a blank line), skipping the HTTP headers."""
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(b"GET /events HTTP/1.1\r\nHost: x\r\n\r\n")
        buf = b""
        # Read headers first (up to the header terminator).
        while b"\r\n\r\n" not in buf:
            buf += s.recv(4096)
        body = buf.split(b"\r\n\r\n", 1)[1]
        # Then read until the SSE frame terminator.
        while b"\n\n" not in body:
            body += s.recv(4096)
    return body.split(b"\n\n", 1)[0].decode("utf-8", "replace") + "\n"


def test_server_serves_spa_and_replays_log(tmp_path):
    from parciver.dashboard import server as dash

    # Seed a tiny valid run for the server to replay.
    with T.RunWriter(tmp_path) as w:
        cfg = {"hand_names": [f"H{i}" for i in range(T.NUM_PREFLOP_HANDS)],
               "abstraction": {"preflop": 169, "flop": 10000, "turn": 10000, "river": 1950}}
        w.start(variant="v", game="g", config=cfg)
        w.metrics(iteration=1, it_per_s=10.0, exploitability=0.1)
        w.end(iteration=1)

    port = _free_port()
    httpd = dash.DashboardServer(("127.0.0.1", port), tmp_path)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        # SPA + static asset.
        html = urllib.request.urlopen(base + "/", timeout=3).read().decode()
        assert "<title>parciver" in html
        js = urllib.request.urlopen(base + "/static/app.js", timeout=3).read()
        assert len(js) > 1000
        # Health + meta.
        meta = json.loads(urllib.request.urlopen(base + "/api/meta", timeout=3).read())
        assert meta["log_exists"] is True
        # SSE replay: read the first batched frame and confirm it carries the run.
        # Use a raw socket and stop at the frame terminator (the stream stays open
        # for live tailing, so a fill-the-buffer read() would block).
        frame = _read_one_sse_frame("127.0.0.1", port)
        assert frame.startswith("data: ")
        batch = json.loads(frame[len("data: "):].strip())
        assert batch[0]["type"] == "run_start"
        assert any(e["type"] == "run_end" for e in batch)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_blocks_path_traversal(tmp_path):
    """A raw request with a literal ``../`` (which a normalizing HTTP client would
    collapse before sending) must not escape the static directory."""
    from parciver.dashboard import server as dash
    port = _free_port()
    httpd = dash.DashboardServer(("127.0.0.1", port), tmp_path)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3) as s:
            s.sendall(b"GET /static/../../server.py HTTP/1.1\r\nHost: x\r\n"
                      b"Connection: close\r\n\r\n")
            resp = b""
            while len(resp) < 4096:
                b = s.recv(4096)
                if not b:
                    break
                resp += b
        status = resp.split(b"\r\n", 1)[0].decode("latin-1")
        assert "404" in status, status
    finally:
        httpd.shutdown()
        httpd.server_close()
