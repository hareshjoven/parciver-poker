# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""scripts/slumbot_report.py's run checks under SLUMBOT-2 Amendment 1 (records/slumbot-2/PREREG.md): lost hands
excluded up to the limit, Slumbot's running totals chained hand by hand per session (restarting after a lost
hand), uneven seats allowed, seconds per hand summed over each session's own playing time. CPU only."""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rep = _load("slumbot_report_checks", ROOT / "scripts" / "slumbot_report.py")
T0 = datetime(2026, 10, 25, tzinfo=timezone.utc)


def _stamp(s):
    return (T0 + timedelta(seconds=s)).isoformat().replace("+00:00", "Z")


def _run(plan, pilot=2):
    """Hands from `plan`, one (session, winnings, lost, slumbot_counts_it) per hand, two seconds apart;
    Slumbot's totals advance by every hand it counts (a lost hand it counted moves them without our record)."""
    hands, tot, cur = [], {}, None
    for k, (s, w, lost, counted) in enumerate(plan, 1):
        if s != cur:
            cur, tot = s, dict(session_num_hands=0, session_total=0, session_baseline_total=0)
        h = dict(hand=k, session=s, phase="pilot" if k <= pilot else "counted", utc=_stamp(2 * k),
                 client_pos=(k + s) % 2)
        if counted:
            tot = dict(session_num_hands=tot["session_num_hands"] + 1, session_total=tot["session_total"] + w,
                       session_baseline_total=tot["session_baseline_total"] + w // 2)
        if lost:
            h.update(lost=True, reason="the client stopped mid-hand")  # no seconds: it never finished
        else:
            h.update(seconds=1.5, winnings=w, check="ok", extra=dict(tot, baseline_winnings=w // 2))
        hands.append(h)
    return hands


def _cfg(hands, lost_limit=2, pilot=2):
    return dict(hands=sum(1 for h in hands if h["phase"] == "counted" and not h.get("lost")), pilot=pilot,
                lost_limit=lost_limit)


DONE = dict(state="done", lost={})


def test_clean_run_chains():
    hands = _run([(1, 50, False, True)] * 6)
    assert rep.chain(hands) == ([], 0, 0)
    assert rep.check_run(None, _cfg(hands), hands, DONE) == []


def test_lost_mid_session_restarts_the_chain_counted_or_not():
    for counted in (True, False):  # Slumbot may or may not have counted the lost hand
        hands = _run([(1, 50, False, True), (1, 100, False, True), (1, -30, True, counted),
                      (1, 20, False, True), (1, -40, False, True)])
        assert rep.chain(hands) == ([], 1, 0)
        assert rep.check_run(None, _cfg(hands), hands, DONE) == []


def test_cut_hand_then_new_session_uneven_seats():
    hands = _run([(1, 50, False, True), (1, 100, False, True), (1, 70, False, True), (1, 0, True, False),
                  (2, -20, False, True), (2, 30, False, True)])
    assert rep.chain(hands) == ([], 0, 0)  # a new session starts from zero: chained, not after a lost hand
    counted = [h for h in hands if h["phase"] == "counted" and not h.get("lost")]
    assert sum(h["client_pos"] == 0 for h in counted) != sum(h["client_pos"] == 1 for h in counted)
    assert rep.check_run(None, _cfg(hands), hands, DONE) == []


def test_broken_chain_fails():
    hands = _run([(1, 50, False, True)] * 5)
    hands[3]["winnings"] += 1  # our record disagrees with Slumbot's running total
    bad, _, _ = rep.chain(hands)
    assert len(bad) == 1 and "hand 4" in bad[0]  # hand 5 advances from Slumbot's total, so only hand 4
    assert rep.check_run(None, _cfg(hands), hands, DONE)


def test_missing_totals_fail():
    hands = _run([(1, 50, False, True)] * 4)
    del hands[2]["extra"]
    assert "hand 3: no running totals" in rep.chain(hands)[0][0]


def test_lost_over_the_limit_fails():
    hands = _run([(1, 10, False, True), (1, 10, False, True)] + [(1, 10, True, False), (1, 10, False, True)] * 3)
    assert rep.check_run(None, _cfg(hands, lost_limit=3), hands, DONE) == []
    bad = rep.check_run(None, _cfg(hands, lost_limit=2), hands, DONE)
    assert bad == ["3 lost counted hands, above the limit 2"]


def test_lost_pilot_hand_counts_toward_the_pilot():
    hands = _run([(1, 10, True, False), (1, 10, False, True), (1, 10, False, True), (1, 10, False, True)])
    assert rep.check_run(None, _cfg(hands), hands, DONE) == []


def test_short_or_unfinished_run_fails():
    hands = _run([(1, 50, False, True)] * 5)
    cfg = _cfg(hands)
    assert rep.check_run(None, dict(cfg, hands=cfg["hands"] + 1), hands, DONE)
    assert rep.check_run(None, cfg, hands, dict(state="stopped", message="x"))


def test_session_seconds_skip_the_gap_between_sessions():
    hands = _run([(1, 1, False, True)] * 4 + [(1, 0, True, False)] + [(2, 1, False, True)] * 3)
    hands[5]["utc"] = _stamp(10_000)  # session 2 starts long after the reboot
    hands[6]["utc"], hands[7]["utc"] = _stamp(10_002), _stamp(10_004)
    events = [dict(event="end", session=2, utc=_stamp(10_005))]
    # session 1 (cut, no end event): its first counted hand's start (hand 3, 6 s) to its last hand's end (the cut
    # hand 5 started at 10 s and has no seconds); session 2: 10,000 s to its end event at 10,005 s
    assert rep.session_seconds(hands, events) == (10 - 6) + 5


def test_totals_restarting_from_zero_mid_session_are_checked_from_zero():
    hands = _run([(1, 50, False, True)] * 5)
    e = hands[3]["extra"]  # Slumbot's totals restart at hand 4 (a new token after a long pause)
    e.update(session_num_hands=1, session_total=50, session_baseline_total=25)
    for h in hands[4:]:
        h["extra"].update(session_num_hands=h["extra"]["session_num_hands"] - 3,
                          session_total=h["extra"]["session_total"] - 150,
                          session_baseline_total=h["extra"]["session_baseline_total"] - 75)
    assert rep.chain(hands) == ([], 0, 1)
    hands[3]["winnings"] += 1  # and our winnings still have to agree with them
    assert rep.chain(hands)[0]


def test_mismatch_on_a_lost_hand_fails():
    hands = _run([(1, 50, False, True)] * 4 + [(1, 50, True, True)] + [(1, 50, False, True)])
    assert rep.check_run(None, _cfg(hands), hands, DONE) == []
    hands[4].update(winnings=50, check="mismatch")  # killed after the mismatch, before the hand ended
    assert rep.check_run(None, _cfg(hands), hands, DONE) == ["lost hand(s) [5]: winnings disagree with the cards"]


def test_torn_line_is_skipped(tmp_path):
    f = tmp_path / "hands.jsonl"
    f.write_text('{"hand": 1}\n{"hand": 2, "pha\n{"hand": 2}\n', encoding="utf-8")
    assert rep.read_jsonl(f) == [{"hand": 1}, {"hand": 2}]


def test_hand_after_a_lost_one_has_its_count_checked():
    hands = _run([(1, 50, False, True)] * 3 + [(1, 50, True, True), (1, 50, True, False)] + [(1, 50, False, True)])
    assert rep.chain(hands) == ([], 1, 0)  # Slumbot counted one of the two lost hands
    hands[5]["extra"]["session_num_hands"] += 2  # more hands than we played and lost
    assert "Slumbot counts" in rep.chain(hands)[0][0]


def test_torn_lines_are_reported(tmp_path):
    f = tmp_path / "hands.jsonl"
    f.write_text('{"hand": 1}\n{"ha\n{"hand": 2}\n', encoding="utf-8")
    torn = []
    assert rep.read_jsonl(f, torn) == [{"hand": 1}, {"hand": 2}] and torn == [2]
