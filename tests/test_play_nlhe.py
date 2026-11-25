"""The No-Limit terminal game and the blueprint downloader, against a stub opponent (no 1.3 GB tables)."""
import gzip
import hashlib
from pathlib import Path

import pytest

from parciver import assets, play_nlhe as P
from parciver import slumbot as S

pytest.importorskip("parciver_engine")


class Passive:
    """Calls the small blind's completion and every bet, checks otherwise: a legal opponent that never raises."""

    def decide(self, hero, hole, board, acts, sb, bb, stacks, seed=0):
        facing = not acts or acts[-1][0] == "bet"
        kind = "call" if facing else "check"
        return {"action": kind, "add": 0, "labels": [kind], "probs": [1.0], "source": "stub"}


class Raiser(Passive):
    """Raises to 300 whenever it opens the betting preflop as the small blind, else as Passive."""

    def decide(self, hero, hole, board, acts, sb, bb, stacks, seed=0):
        if not acts and not board:
            return {"action": "bet", "add": 250, "labels": ["bet"], "probs": [1.0], "source": "stub"}
        return super().decide(hero, hole, board, acts, sb, bb, stacks, seed)


def session(moves, agent, **kw):
    out, it = [], iter(moves)

    def ask(prompt):
        out.append(prompt)
        return next(it)

    lines: list[str] = []
    rc = P.run(agent=agent, ask=ask, say=lines.append, **kw)
    return rc, lines, out


def test_a_checked_down_hand_is_played_to_a_showdown():
    rc, lines, _ = session(["c"] * 10, Passive(), hands=1, seed=1)
    text = "\n".join(lines)
    assert rc == 0 and "showdown" in text and "session:" in text and "Final:" in text
    won = [ln for ln in lines if ln.strip().startswith("you ") and " chips (" in ln]
    assert len(won) == 1 and ("300" in won[0] or "100" in won[0] or "split" in won[0] or "200" in won[0])


def test_folding_the_small_blind_loses_half_a_big_blind():
    rc, lines, _ = session(["f"], Passive(), hands=1, seed=1)  # hand 1 seats you as the small blind, first to act
    assert rc == 0 and any("lose 50 chips" in ln for ln in lines)


def test_an_illegal_raise_is_refused_then_the_hand_goes_on():
    rc, lines, _ = session(["b 120", "c", "c", "c", "c", "c", "c", "c"], Passive(), hands=1, seed=2)
    assert rc == 0 and any("not allowed" in ln and "at least 200" in ln for ln in lines)


def test_quit_ends_the_session_without_counting_the_hand():
    rc, lines, _ = session(["q"], Passive(), hands=3, seed=1)
    assert rc == 0 and not any("Final:" in ln for ln in lines)


def test_seats_alternate_and_a_raise_by_the_opponent_is_shown():
    rc, lines, _ = session(["f", "f"], Raiser(), hands=2, seed=4)
    text = "\n".join(lines)
    assert "you are the small blind" in text and "you are the big blind" in text
    assert "opponent raises to 300" in text


def test_peek_prints_the_opponents_strategy():
    _, lines, _ = session(["c"] * 10, Passive(), hands=1, seed=1, peek=True)
    assert any("opponent's strategy" in ln for ln in lines)


def test_fold_with_nothing_to_fold_to_is_explained():
    # hand 1 takes four calls/checks; hand 2 seats you as the big blind and the stub completes: nothing to fold to
    _, lines, _ = session(["c"] * 4 + ["f"] + ["c"] * 4 + ["q"], Passive(), hands=3, seed=1)
    assert any("no bet to fold to" in ln for ln in lines)


def test_parse_move_shorthands():
    s = S.parse("")  # the small blind to act: 50 to call, the least raise is to 200
    assert P.parse_move(s, "", "f") == "f"
    assert P.parse_move(s, "", "c") == "c"
    assert P.parse_move(s, "", "b 600") == "b600"
    assert P.parse_move(s, "", "r 600") == "b600"
    assert P.parse_move(s, "", "a") == f"b{S.STACK}"
    assert P.parse_move(s, "", "p") == "b300"  # call 50, then raise the pot (150 + 50): to 300
    assert P.parse_move(s, "", "h") == "b200"  # half of 200 added over the call is under the least raise: clamped
    assert P.parse_move(s, "", "nonsense") is None
    after = S.parse("c")  # the big blind may check but not fold
    assert P.parse_move(after, "c", "f") is None
    assert P.parse_move(after, "c", "c") == "k"


def test_describe_reads_the_history_from_the_players_seat():
    lines = P.describe("b300c/kb200c/", human=1)  # position 1 is the small blind
    assert lines[0] == "preflop: you raise to 300, opponent calls"
    assert lines[1] == "flop: opponent checks, you bet to 200, opponent calls"


def test_locate_assets(tmp_path, monkeypatch):
    monkeypatch.setenv("PARCIVER_HOME", str(tmp_path))
    assert P.locate_assets(None, None) is None
    (tmp_path / "abstraction").mkdir()
    for f in P.BUCKET_FILES:
        (tmp_path / "abstraction" / f).write_bytes(b"x")
    (tmp_path / P.BLUEPRINT_FILE).write_bytes(b"x")
    assert P.locate_assets(None, None) == (tmp_path / "abstraction", tmp_path / P.BLUEPRINT_FILE)


def _publish(tmp: Path) -> tuple[dict, Path]:
    """A two-file fake release: gzip files in tmp/rel and the manifest that describes them."""
    rel = tmp / "rel"
    rel.mkdir()
    files = []
    for dest, body in (("blueprint.bin", b"strategy" * 1000), ("abstraction/flop_buckets.npy", b"buckets" * 500)):
        name = dest.replace("/", "_") + ".gz"
        (rel / name).write_bytes(gzip.compress(body))
        files.append({"name": name, "dest": dest, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
    return {"version": "test", "base_url": None, "files": files}, rel


def test_fetch_installs_verifies_and_resumes(tmp_path, monkeypatch):
    manifest, rel = _publish(tmp_path)
    monkeypatch.setattr(assets, "load_manifest", lambda: manifest)
    monkeypatch.setenv("PARCIVER_HOME", str(tmp_path / "home"))
    msgs: list[str] = []
    assert assets.fetch(rel.as_uri(), say=msgs.append) == 0
    assert (tmp_path / "home" / "blueprint.bin").read_bytes() == b"strategy" * 1000
    msgs.clear()
    assert assets.fetch(rel.as_uri(), say=msgs.append) == 0  # second run: nothing to download
    assert any("already installed" in m for m in msgs)


def test_fetch_refuses_a_corrupted_download(tmp_path, monkeypatch):
    manifest, rel = _publish(tmp_path)
    manifest["files"][0]["sha256"] = "0" * 64
    monkeypatch.setattr(assets, "load_manifest", lambda: manifest)
    monkeypatch.setenv("PARCIVER_HOME", str(tmp_path / "home"))
    with pytest.raises(SystemExit, match="does not match"):
        assets.fetch(rel.as_uri(), say=lambda m: None)
    assert not (tmp_path / "home" / "blueprint.bin").exists()
    assert not list((tmp_path / "home").glob("*.part"))


def test_fetch_without_a_location_says_so(monkeypatch):
    monkeypatch.setattr(assets, "load_manifest", lambda: {"version": "t", "base_url": None, "files": []})
    monkeypatch.delenv("PARCIVER_ASSET_URL", raising=False)
    msgs: list[str] = []
    assert assets.fetch(None, say=msgs.append) == 2 and any("--url" in m for m in msgs)
