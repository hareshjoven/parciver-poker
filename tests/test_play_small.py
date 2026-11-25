# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""The terminal game against the solved Kuhn / Leduc opponent (parciver play)."""
from __future__ import annotations

import re

import pytest

from parciver import cli, play_small


def _first_option(prompt: str) -> str:
    return re.search(r"\((.*?), q to quit", prompt).group(1).split("/")[0]


def _session(game: str, hands: int, **kw) -> list[str]:
    lines: list[str] = []
    rc = play_small.run(game, iterations=kw.pop("iterations", 200), hands=hands, seed=7,
                        ask=_first_option, say=lines.append, **kw)
    assert rc == 0
    return lines


@pytest.mark.parametrize("game", ["kuhn", "leduc"])
def test_scripted_session_finishes_and_reports(game, monkeypatch):
    monkeypatch.setattr(play_small, "_engine", lambda: None)  # the pure-Python path, always available
    text = "\n".join(_session(game, hands=3))
    assert "exact exploitability" in text and "pure Python" in text
    assert text.count("--- hand") == 3
    assert re.search(r"Final: [+-]?\d", text)


def test_seats_alternate_and_peek_shows_the_mixed_strategy(monkeypatch):
    monkeypatch.setattr(play_small, "_engine", lambda: None)
    text = "\n".join(_session("kuhn", hands=2, peek=True))
    assert "you are player 1, acting first" in text and "you are player 2, acting second" in text
    assert "mixed strategy" in text


def test_quit_stops_without_a_result():
    lines: list[str] = []
    play_small.run("kuhn", iterations=50, hands=None, seed=1, ask=lambda _: "q", say=lines.append)
    assert not any(line.startswith("Final") for line in lines)


def test_bad_input_is_reprompted_not_crashed():
    junk = iter(["xyz", ""])
    lines: list[str] = []
    play_small.run("kuhn", iterations=50, hands=1, seed=2,
                   ask=lambda prompt: next(junk, None) or _first_option(prompt), say=lines.append)
    assert any("please type one of" in line for line in lines)


def test_solved_kuhn_is_close_to_nash():
    strategy, expl, _, _ = play_small.solve(play_small.GAMES["kuhn"], iterations=3000)
    assert expl < 0.01 and len(strategy) == 12


def test_cli_parses_and_runs(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _="": "q")
    assert cli.main(["play", "kuhn", "--iterations", "50", "--seed", "1"]) == 0
    assert "exploitability" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["play", "chess"])
