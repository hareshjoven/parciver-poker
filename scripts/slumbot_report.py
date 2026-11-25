"""The Slumbot reading (records/slumbot/PREREG.md, "Reading"), from a finished run folder.

    .venv/Scripts/python.exe scripts/slumbot_report.py runs/slumbot/<UTC> [--json records/slumbot/result.json]

Registered:
  - our win rate in bb/100 over the counted hands, from `winnings`, with a 95% normal interval on the per-hand
    values: WIN if the lower bound is above 0, LOSE if the upper bound is below 0, else UNRESOLVED;
  - the win rate by seat;
  - the all-in-adjusted win rate (descriptive): an all-in with both hands shown is scored at our exact equity
    over the board still to come, times the pot, less what we put in;
  - seconds per hand; the pilot's sigma.
Not registered (descriptive, labelled so): Slumbot's own "Baseline", which it returns with every hand.
slumbot.com: "Baseline numbers report player's relative performance to Slumbot playing in the same seat with
the same cards". `baseline_winnings` is already that difference, ours less Slumbot's own result in our seat,
not Slumbot's result itself: in every hand where Slumbot's small blind folds at once it is 0 (an identical
replay), and read as Slumbot's result it would make 27% of hands chops against our 0.7% (audit, 7 Oct). Over
an even seat split it estimates the same win rate as the raw one, with card luck partly cancelled; whether it
is less noisy is measured, not assumed.

Before any number, the run is checked from its own files: N counted hands that finished, hand numbers
contiguous, the pilot first and exactly its registered length, every payoff check "ok", the run done, and
Slumbot's running totals chained hand by hand within each session: every finished hand's `session_total`,
`session_baseline_total` and `session_num_hands` advance by its own winnings, baseline and one from the hand
before it (from zero at a session's first hand). Any failure exits 3. bb/100 equals the mean chips per hand
(bb = 100 chips).

SLUMBOT-2 Amendment 1 (records/slumbot-2/PREREG.md; before its first live hand): a restart is not a failure, so a
run may have several sessions and lost hands (a hand cut by a reboot or kill, or lost to the network). Lost hands
are excluded from every number; up to the run's `lost_limit` (0.5% of N) they are allowed. The chain restarts after
a lost hand, whose result Slumbot may have counted (the hand after it is reported as unchained; only Slumbot's hand
count is checked). Within a session, a hand whose totals restart from zero (a new token) is checked from zero and
reported. A lost hand whose winnings disagreed with the cards still fails. Seats may be uneven (each session starts
in the big blind) and are reported. Seconds per hand sum each session's own playing time, not the time between
sessions. Runs without lost hands read as before (the 5 Oct run: every number the same).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import parciver_engine as pe  # noqa: E402

from parciver import slumbot as S  # noqa: E402

Z = 1.959963984540054  # two-sided 95%
BOARD_AT = (0, 3, 4, 5)  # board cards seen by the end of each street


def utc(stamp: str) -> float:
    """Seconds since the epoch of the run's UTC stamps ("2026-10-05T10:15:18.529Z")."""
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


def read_jsonl(path: Path, torn: list | None = None) -> list[dict]:
    """The file's records; a line cut by a power loss is skipped (the runner's rule: its hand is logged again from
    status.json) and its line number appended to `torn`."""
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                if torn is not None:
                    torn.append(i)
    return out


def stats(xs: list[float]) -> dict:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    se = sd / math.sqrt(n)
    return dict(n=n, bb100=m, sd_bb=sd / 100, lo=m - Z * se, hi=m + Z * se, half=Z * se)


def verdict(s: dict) -> str:
    return "WIN" if s["lo"] > 0 else "LOSE" if s["hi"] < 0 else "UNRESOLVED"


def allin_street(action: str) -> int:
    """The street the last bet was called on: the last non-empty street of the history."""
    streets = action.split("/")
    return max(i for i, s in enumerate(streets) if s)


def adjusted(h: dict) -> tuple[float, bool]:
    """(the hand's all-in-adjusted winnings, whether it was adjusted). Only an all-in with both hands shown
    before the river is scored at equity; a river all-in's equity is its result."""
    if not (h["all_in"] and h["showdown"] and h.get("bot_hole")):
        return float(h["winnings"]), False
    st = allin_street(h["action"])
    hole = [S.card_index(c) for c in h["hole"]]
    bot = [S.card_index(c) for c in h["bot_hole"]]
    board = [S.card_index(c) for c in h["board"][:BOARD_AT[st]]]
    eq = pe.equity_vs_hand(hole, bot, board)
    return eq * h["pot"] - h["put_in"], st < 3


def chain(hands: list[dict]) -> tuple[list[str], int, int]:
    """Slumbot's running totals, hand by hand within each session (a lost hand breaks the chain): (what disagrees,
    how many finished hands follow a lost one and so have only Slumbot's hand count checked, how many restart
    Slumbot's totals from zero within a session: a new token, which sample_api.py says can come after a long pause;
    checked from zero)."""
    zero = dict(session_num_hands=0, session_total=0, session_baseline_total=0)

    def advances(e, q, w):
        return (e["session_num_hands"] == q["session_num_hands"] + 1 and e["session_total"] == q["session_total"] + w
                and e["session_baseline_total"] == q["session_baseline_total"] + e["baseline_winnings"])

    bad, unchained, restarted = [], 0, 0
    session, prev, broken = None, None, 0  # prev: this session's last finished hand; broken: lost hands since
    for h in hands:
        if h["session"] != session:  # a new session's totals start from zero
            session, prev, broken = h["session"], None, 0
        if h.get("lost") or "extra" not in h:
            if not h.get("lost"):
                bad.append(f"hand {h['hand']}: no running totals from Slumbot")
            broken += 1
            continue
        e = h["extra"]
        if broken:  # Slumbot may or may not have counted each lost hand: only its count is checked, then restart
            unchained += 1
            n = (prev["extra"] if prev else zero)["session_num_hands"]
            if not (n + 1 <= e["session_num_hands"] <= n + 1 + broken or advances(e, zero, h["winnings"])):
                bad.append(f"hand {h['hand']} (session {h['session']}): Slumbot counts {e['session_num_hands']} "
                           f"hands, {n} before {broken} lost one(s)")
        elif not advances(e, prev["extra"] if prev else zero, h["winnings"]):
            if prev and advances(e, zero, h["winnings"]):
                restarted += 1
            else:
                bad.append(f"hand {h['hand']} (session {h['session']}): Slumbot's running totals do not advance by "
                           f"this hand")
        prev, broken = h, 0
    return bad, unchained, restarted


def session_seconds(hands: list[dict], events: list[dict]) -> float:
    """The counted hands' playing time: per session, from its first counted hand's start to the session's end
    event, or, for a session cut without one (a reboot), to its last hand's end."""
    total = 0.0
    for s in sorted({h["session"] for h in hands if h["phase"] == "counted"}):
        hs = [h for h in hands if h["session"] == s]
        first = next(h for h in hs if h["phase"] == "counted")
        ends = [e for e in events if e.get("event") == "end" and e.get("session") == s]
        last = hs[-1]
        stop = utc(ends[-1]["utc"]) if ends else utc(last["utc"]) + last.get("seconds", 0.0)
        total += stop - utc(first["utc"])
    return total


def check_run(run: Path, cfg: dict, hands: list[dict], status: dict) -> list[str]:
    bad = []
    counted = [h for h in hands if h["phase"] == "counted" and not h.get("lost")]
    if len(counted) != cfg["hands"]:
        bad.append(f"{len(counted)} finished counted hands, registered {cfg['hands']}")
    lost = sum(1 for h in hands if h.get("lost") and h["phase"] == "counted")
    if lost > cfg["lost_limit"]:
        bad.append(f"{lost} lost counted hands, above the limit {cfg['lost_limit']}")
    if [h["hand"] for h in hands] != list(range(1, len(hands) + 1)):
        bad.append("hand numbers not contiguous from 1")
    if [h["phase"] for h in hands] != ["pilot"] * cfg["pilot"] + ["counted"] * (len(hands) - cfg["pilot"]):
        bad.append(f"the first {cfg['pilot']} hands are not the pilot, or a later hand is not counted")
    done = [h for h in hands if not h.get("lost")]
    if any(h.get("check") != "ok" for h in done):
        bad.append(f"{sum(h.get('check') != 'ok' for h in done)} payoff checks not ok")
    if any(h.get("check") == "mismatch" for h in hands if h.get("lost")):  # killed after its winnings disagreed
        bad.append(f"lost hand(s) {[h['hand'] for h in hands if h.get('lost') and h.get('check') == 'mismatch']}: "
                   f"winnings disagree with the cards")
    if status["state"] != "done":
        bad.append(f"run state {status['state']!r}: {status.get('message')}")
    return bad + chain(hands)[0]  # Slumbot's running totals cover a session's hands, pilot included


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="the run folder, runs/slumbot/<UTC>")
    p.add_argument("--json", help="also write the reading here")
    a = p.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "run.json").read_text(encoding="utf-8"))
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    torn = []
    hands = read_jsonl(run / "hands.jsonl", torn)
    events = read_jsonl(run / "events.jsonl")

    bad = check_run(run, cfg, hands, status)
    print(f"run {run.name}: commit {cfg['commit'][:7]}, blueprint sha256 {cfg['blueprint_sha256'][:12]}..., "
          f"search {cfg['search']}")
    if bad:
        for b in bad:
            print(f"  CHECK FAILED: {b}")
        return 3
    counted = [h for h in hands if h["phase"] == "counted" and not h.get("lost")]
    pilot = [h for h in hands if h["phase"] == "pilot" and not h.get("lost")]
    lost = {ph: sum(1 for h in hands if h.get("lost") and h["phase"] == ph) for ph in ("pilot", "counted")}
    sessions = sorted({h["session"] for h in hands})
    _, unchained, restarted = chain(hands)
    print(f"checks: {len(counted):,} counted hands after the {cfg['pilot']}-hand pilot, "
          + (f"lost {lost['counted']} counted and {lost['pilot']} pilot (excluded)" if any(lost.values()) else "0 lost")
          + f", hand numbers contiguous, every payoff check ok, seats {sum(h['client_pos'] == 0 for h in counted):,} BB"
          f" / {sum(h['client_pos'] == 1 for h in counted):,} SB; Slumbot's running totals chain hand by hand in "
          f"{len(sessions)} session(s)" + (f" ({unchained} hand(s) after a lost one not chained)" if unchained else "")
          + (f" ({restarted} restart(s) from zero within a session)" if restarted else "")
          + (f"; {len(torn)} torn line(s) in hands.jsonl skipped (lines {torn})" if torn else "")
          + f"; {status['retried']} request retries" + ("" if any(lost.values()) else ", all resumed"))

    raw = stats([h["winnings"] for h in counted])
    seat = {name: stats([h["winnings"] for h in counted if h["client_pos"] == pos])
            for name, pos in (("BB", 0), ("SB", 1))}
    adj_pairs = [adjusted(h) for h in counted]
    for h, (v, moved) in zip(counted, adj_pairs):  # a river all-in's equity must reproduce its result
        if h["all_in"] and h["showdown"] and not moved and abs(v - h["winnings"]) > 1e-9:
            print(f"  CHECK FAILED: hand {h['hand']} river all-in equity {v} vs winnings {h['winnings']}")
            return 3
    adj = stats([v for v, _ in adj_pairs])
    n_adj = sum(m for _, m in adj_pairs)
    base = stats([h["extra"]["baseline_winnings"] for h in counted])  # already ours less Slumbot's
    sec = [h["seconds"] for h in counted]
    think = [h["think_seconds"] for h in counted]
    solves = [d["seconds"] for h in counted for d in h["decisions"] if d.get("solves")]
    pil = stats([h["winnings"] for h in pilot]) if len(pilot) > 1 else None
    blocks = [stats([h["winnings"] for h in counted[i:i + 10000]])["bb100"] for i in range(0, len(counted), 10000)]

    def line(name, s, extra=""):
        return (f"  {name:<26} {s['bb100']:+8.2f} bb/100   95% [{s['lo']:+7.2f}, {s['hi']:+7.2f}]"
                f"   sd {s['sd_bb']:.2f} bb/hand{extra}")

    print(f"\nPRIMARY (registered): {verdict(raw)}")
    print(line(f"win rate, {raw['n']:,} hands", raw))
    print("by seat (registered):")
    for name, s in seat.items():
        print(line(f"{name}, {s['n']:,} hands", s))
    print("all-in adjusted (registered, descriptive):")
    print(line(f"{n_adj} pre-river all-ins scored", adj))
    print("Slumbot's Baseline (NOT registered; descriptive):")
    print(line("ours less Slumbot's seat", base, f"   (variance {(base['sd_bb'] / raw['sd_bb']) ** 2:.0%} of raw)"))
    print("by 10,000-hand block (descriptive): " + ", ".join(f"{b:+.1f}" for b in blocks))
    pilot_ev = next((e for e in events if e.get("event") == "pilot"), None)
    elapsed = session_seconds(hands, events)  # a hand's utc is its start; with the gaps between hands
    print(f"\nseconds per hand: {elapsed / len(counted):.2f} elapsed ({sum(sec) / len(sec):.2f} within hands, "
          f"network included), {sum(think) / len(think):.2f} thinking; {len(solves):,} searched decisions, "
          f"mean {sum(solves) / len(solves):.2f} s, max {max(solves):.1f} s; counted hands took {elapsed / 3600:.1f} h"
          + (f" (the pilot projected {pilot_ev['projected_hours']:.1f} h)" if pilot_ev else ""))
    if pil:
        print(f"pilot ({pil['n']} hands, discarded): sd {pil['sd_bb']:.2f} bb/hand "
              f"(PREREG assumed ~10; N = 50,000 gives +/- {raw['half']:.1f} bb/100 at the realised sd)")

    if a.json:
        out = dict(run=run.name, commit=cfg["commit"], prereg=cfg["prereg"], search=cfg["search"],
                   verdict=verdict(raw), raw=raw, by_seat=seat, allin_adjusted=dict(adj, adjusted_hands=n_adj),
                   baseline_unregistered=base, blocks_10k=blocks, retried=status["retried"], lost=lost["counted"],
                   lost_pilot=lost["pilot"], sessions=len(sessions), unchained=unchained,
                   restarted=restarted, torn_lines=torn,
                   seconds=dict(elapsed_mean=elapsed / len(counted), in_hand_mean=sum(sec) / len(sec),
                                think_mean=sum(think) / len(think), solve_mean=sum(solves) / len(solves),
                                solve_max=max(solves), counted_hours=elapsed / 3600),
                   pilot_sd_bb=pil["sd_bb"] if pil else None)
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
