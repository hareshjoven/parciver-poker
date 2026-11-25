"""scripts/slumbot_offgrid.py's replay of a flop history and its off-grid test. CPU only."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("slumbot_offgrid", ROOT / "scripts" / "slumbot_offgrid.py")
og = importlib.util.module_from_spec(spec)
spec.loader.exec_module(og)


def test_half_pot_bet_is_on_the_grid():
    pot, owed, floor, bets, last, stack = og.replay_flop("b200c/kb200")
    assert (pot, owed, floor, bets, stack) == (600, 200, 200, 1, 19800)
    assert last == (200, False, False, 0.5)


def test_half_pot_raise_is_off_the_grid():
    pot, owed, floor, bets, last, stack = og.replay_flop("ck/kb100b300")
    assert (pot, owed, floor, bets, stack) == (600, 200, 200, 2, 19800)
    assert last == (300, False, True, 0.5)
    assert og.grid_amounts(pot, owed, floor, bets) == [1000]


def test_pot_raise_and_all_in_are_on_the_grid():
    assert og.replay_flop("b300c/kb300b1500")[4] == (1500, False, False, 1.0)
    assert og.replay_flop("b300c/kb19700")[4][1:3] == (True, False)


def test_grid_floors_and_cap():
    assert og.grid_amounts(150, 0, 100, 0) == [100, 150]  # the minimum bet is a big blind
    assert og.grid_amounts(300, 100, 1000, 1) == [1100]  # the raise floor
    assert len(og.grid_amounts(600, 200, 200, 4)) == 1  # the last re-raise the cap allows
    assert og.grid_amounts(600, 200, 200, 5) == []


def test_no_bet_to_answer():
    assert og.replay_flop("b200c/")[4] is None
    assert og.replay_flop("b200c/k")[4] is None
