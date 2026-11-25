#!/usr/bin/env python
"""Play the bot against GTO Wizard AI through the researcher API (heads-up NLHE, 200bb stacks
reset every hand, 50/100 blinds) and measure it with the server's AIVAT scores.

    # the API key: --key, or the GTOW_API_KEY environment variable, or a one-line file .gtow_key
    # in the repo root (gitignored)
    .venv/Scripts/python.exe scripts/gtow_bench.py --candidate runs/blueprint-20260914-144924/blueprint.bin --label v1 --hands 2500

    # resume an interrupted run (plays until the run has --hands finished hands)
    .venv/Scripts/python.exe scripts/gtow_bench.py --candidate <bin> --run runs/gtow/<run dir> --hands 2500

Every finished hand is appended to `<run>/hands.jsonl` (cards, history, our decisions with their
source, `winnings` and `aivat_score` in chips); the first `--raw` hands' full request/response
transcripts go to `<run>/raw/`. The reconstruction of each state is checked against the server
(pot, our stack, seat; see parciver.gtow): any disagreement, an engine error or a rejected action
stops the run and saves that hand's transcript to `<run>/raw/failed-<hand>.json`, rather than
playing on with a misread hand.

Certificates are checked with Windows' own verifier when the `truststore` package is installed
(`uv pip install --python .venv/Scripts/python.exe truststore`), otherwise with Python's copy of the
certificate store, which can reject a chain browsers accept ("certificate has expired").

The bot is trained for 100bb stacks; here stacks are 200bb, which the play agent handles by
translation, so this measures the bot out of its training depth.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parciver import gtow  # noqa: E402

DEFAULT_URL = "https://researcher.gtowizard.com"
GAME = "HUNL 200BB"
BUSY = {429, 502, 503, 504}
MAX_ATTEMPTS = 32
HEARTBEAT = 60.0  # seconds of no output before a "still running" line
KEY_FILE = ROOT / ".gtow_key"
# Cloudflare in front of the API bans urllib's default "Python-urllib/x.y" signature (error 1010).
USER_AGENT = "parciver-gtow-bench/1.0"


class Fatal(RuntimeError):
    """Stop the whole run."""


def _read_key_file(path: Path) -> str:
    raw = path.read_bytes()
    # PowerShell's `echo x > file` writes UTF-16 with a byte-order mark; editors may add a UTF-8 one.
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    return text.strip()


def api_key(arg: str | None) -> str:
    key = arg or os.environ.get("GTOW_API_KEY") or (_read_key_file(KEY_FILE) if KEY_FILE.exists() else "")
    if not key:
        raise SystemExit(f"no API key: pass --key, set GTOW_API_KEY, or put it in {KEY_FILE}")
    return key


SYSTEM_VERIFIER = "Windows' own certificate verifier (truststore)"


def ssl_context() -> tuple[ssl.SSLContext, str]:
    """The TLS context and how it checks certificates: the operating system's own verifier (the
    `truststore` package) when that is installed, as browsers do. Python's copy of the Windows
    certificate store lacks the roots Windows only fetches when its own verifier asks for them, so
    a chain browsers accept can fail there with "certificate has expired" (an old cross-signed path)."""
    try:
        import truststore
    except ImportError:
        return ssl.create_default_context(), "Python's certificate store"
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), SYSTEM_VERIFIER


def certificate_failure(url: str, err: ssl.SSLCertVerificationError, verifier: str) -> str:
    why = getattr(err, "verify_message", None) or str(err)
    head = f"TLS certificate check failed for {url} ({why}; checked with {verifier})"
    if verifier != SYSTEM_VERIFIER:
        return (f"{head}. Often Python's copy of the certificate store is out of date for this chain, not the "
                f"server. Fix: uv pip install --python .venv/Scripts/python.exe truststore (the bench then checks "
                f"certificates with Windows' own verifier, like a browser) and run again. If it still fails, "
                f"the server's certificate itself is out of date and only GTO Wizard can fix it.")
    return (f"{head}. Windows' own verifier rejects it too, so the server's certificate chain is out of date "
            f"(GTO Wizard's to fix; open {url} in a browser to see its dates) or security software on this PC "
            f"is intercepting HTTPS.")


class Api:
    def __init__(self, url: str, key: str, timeout: float = 180.0):
        self.url, self.key, self.timeout = url.rstrip("/"), key, timeout
        self.context, self.verifier = ssl_context()
        self.replies, self.last_reply = 0, time.time()  # for the "still waiting" heartbeat

    def post(self, path: str, body: dict, stop: threading.Event) -> dict:
        data = json.dumps(body).encode()
        delay = 2.0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            req = urllib.request.Request(self.url + path, data=data, method="POST",
                                         headers={"X-API-Key": self.key, "Content-Type": "application/json",
                                                  "Accept": "application/json", "User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self.context) as r:
                    out = json.loads(r.read().decode())
                self.replies, self.last_reply = self.replies + 1, time.time()
                return out
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")[:2000]
                if e.code == 403 and "cloudflare" in text.lower():
                    raise Fatal(f"403 from {path}: Cloudflare blocked the request itself, not the key "
                                f"(User-Agent {USER_AGENT!r}; ask GTO Wizard to allow it): {text}") from e
                if e.code in (401, 403):
                    raise Fatal(f"{e.code} from {path}: the API key was rejected ({text})") from e
                if e.code not in BUSY:
                    raise RuntimeError(f"{e.code} from {path}: {text}") from e
                why = f"{e.code} {' '.join(text.split())[:160]}".rstrip()  # the server's reason, on one line
            except urllib.error.URLError as e:
                if isinstance(e.reason, ssl.SSLCertVerificationError):  # retrying cannot fix it
                    raise Fatal(certificate_failure(self.url, e.reason, self.verifier)) from e
                why = f"URLError: {e.reason}"
            except (TimeoutError, ConnectionError) as e:
                why = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
            if stop.is_set():
                raise Fatal("stopping")
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"{path}: gave up after {attempt} attempts ({why})")
            wait = min(delay, 32.0) * (0.75 + 0.5 * random.random())
            print(f"[gtow] {path}: {why}; retry {attempt}/{MAX_ATTEMPTS - 1} in {wait:.0f}s", file=sys.stderr, flush=True)
            time.sleep(wait)
            delay *= 2
        raise AssertionError("unreachable")


def play_hand(api: Api, agent, stop: threading.Event, transcript: list, learned: dict) -> dict:
    """One hand; returns its record. Raises gtow.Desync / RuntimeError / Fatal. `learned`
    holds the server's preflop bet convention once a state has pinned it down."""
    resp = api.post("/hands", {"game_name": GAME}, stop)
    transcript.append({"response": resp})
    hand_id = resp["hand_id"]
    basis, seat, decisions = None, None, []
    t0 = time.time()
    while not resp["game_state"]["is_hand_over"]:
        st = resp["game_state"]
        s = gtow.spot(resp, learned.get("basis"))
        if s.certain and "basis" not in learned:
            learned["basis"] = s.basis
            print(f"[gtow] server convention: preflop bets are {s.basis} of the posted blind", flush=True)
        basis = s.basis
        if seat is None:
            seat = s.seat
        elif s.seat != seat:
            raise gtow.Desync(f"our seat changed from {seat} to {s.seat}")
        try:
            d = agent.decide(s.seat, tuple(s.hole), s.board, s.replay.actions, s.sb, s.bb, (s.stack, s.stack),
                             seed=int(hand_id) & 0xFFFFFFFFFFFF)
        except ValueError as e:
            raise gtow.Desync(f"engine rejected the reconstructed hand: {e}") from e
        req, note = gtow.to_request(s, d, st.get("legal_actions") or [], st.get("raise_range"))
        decisions.append({"street": s.replay.street, "action": d["action"], "add": d["add"],
                          "all_in": d["all_in"], "source": d["source"], "sent": req, "note": note})
        transcript.append({"request": req, "decision": {k: d[k] for k in ("action", "add", "all_in", "source")}})
        resp = api.post(f"/hands/{hand_id}/act", req, stop)
        transcript.append({"response": resp})
    st, game = resp["game_state"], resp["game"]
    players = st.get("players") or []
    me = None
    try:
        me = players[gtow.hero_index(players)]
    except gtow.Desync:
        pass
    return {
        "hand_id": hand_id,
        "seat": seat,
        "position": me.get("position") if me else None,
        "hole": me.get("hole_cards") if me else None,
        "villain_hole": next((p.get("hole_cards") for p in players if p is not me), None),
        "board": st.get("board_cards"),
        "history": st.get("action_history"),
        "winnings": st.get("winnings"),
        "aivat": st.get("aivat_score"),
        "bb": round(float(game["blinds"][1])),
        "basis": basis,
        "decisions": decisions,
        "seconds": round(time.time() - t0, 3),
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def stats(records: list[dict]) -> dict:
    def mean_se(xs):
        n = len(xs)
        if n == 0:
            return float("nan"), float("nan")
        m = sum(xs) / n
        se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else float("nan")
        return m * 100, se * 100

    won = [r["winnings"] / r["bb"] for r in records if r.get("winnings") is not None]
    aiv = [r["aivat"] / r["bb"] for r in records if r.get("aivat") is not None]
    src: dict[str, int] = {}
    notes = 0
    for r in records:
        for d in r["decisions"]:
            src[d["source"]] = src.get(d["source"], 0) + 1
            notes += d["note"] is not None
    (wm, ws), (am, as_) = mean_se(won), mean_se(aiv)
    return {"hands": len(records), "winnings_bb100": wm, "winnings_se": ws, "aivat_bb100": am, "aivat_se": as_,
            "sources": src, "adjusted_actions": notes}


def fmt(s: dict) -> str:
    def iv(m, se):
        return "n/a" if math.isnan(m) else f"{m:+.1f} ± {1.96 * se:.1f}" if not math.isnan(se) else f"{m:+.1f}"
    total = sum(s["sources"].values()) or 1
    share = ", ".join(f"{k} {v / total:.0%}" for k, v in sorted(s["sources"].items()))
    return (f"{s['hands']} hands | AIVAT {iv(s['aivat_bb100'], s['aivat_se'])} bb/100 | "
            f"raw winnings {iv(s['winnings_bb100'], s['winnings_se'])} bb/100 | {share}"
            + (f" | {s['adjusted_actions']} adjusted actions" if s["adjusted_actions"] else ""))


def build_agent(args):
    import parciver_engine as pe
    saved = pe.SavedBlueprint(str(Path(args.abstraction_dir)), str(Path(args.candidate)))
    agent = saved.play_agent()
    if args.search != "none":
        agent = agent.with_search(from_turn=args.search == "turn", river_iters=args.river_iters,
                                  turn_iters=args.turn_iters, exact_own_range=args.own_range == "exact",
                                  solve_round=args.round == "solve")
    return agent


def run(args, agent, api: Api) -> int:
    run_dir = Path(args.run) if args.run else (
        ROOT / "runs" / "gtow" / f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{args.label}")
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    config = {"label": args.label, "candidate": str(args.candidate), "search": args.search,
              "river_iters": args.river_iters, "turn_iters": args.turn_iters, "own_range": args.own_range,
              "round": args.round if args.search != "none" else None,
              "url": api.url, "game": GAME}
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        same = {k: old.get(k) for k in ("search", "river_iters", "turn_iters", "own_range", "round")}
        if same != {k: config[k] for k in same} or Path(old["candidate"]).name != Path(config["candidate"]).name:
            raise SystemExit(f"{run_dir} was played by a different bot ({old}); start a new run")
    else:
        cfg_path.write_text(json.dumps(config, indent=1))
    hands_path = run_dir / "hands.jsonl"
    records = [json.loads(line) for line in hands_path.read_text().splitlines() if line.strip()] \
        if hands_path.exists() else []
    todo = args.hands - len(records)
    print(f"[gtow] run {run_dir} | bot: {args.label} ({args.search} search) | {len(records)} done, {max(todo, 0)} to go")
    if todo <= 0:
        print("[gtow] " + fmt(stats(records)))
        return 0

    lock, stop, learned = threading.Lock(), threading.Event(), {}
    state = {"claimed": 0, "raw_left": max(args.raw - len(records), 0), "failed": None, "t0": time.time(),
             "started": len(records), "printed": time.time()}

    def worker():
        while not stop.is_set():
            with lock:
                if state["claimed"] >= todo:
                    return
                state["claimed"] += 1
                keep_raw = state["raw_left"] > 0
                state["raw_left"] -= keep_raw
            transcript: list = []
            try:
                rec = play_hand(api, agent, stop, transcript, learned)
            except Fatal as e:
                if not stop.is_set():
                    state["failed"] = str(e)
                stop.set()
                return
            except Exception as e:  # noqa: BLE001 - every failure stops the run with its transcript
                name = f"failed-{int(time.time() * 1000)}.json"
                (run_dir / "raw" / name).write_text(json.dumps({"error": f"{type(e).__name__}: {e}",
                                                                "transcript": transcript}, indent=1))
                with lock:
                    if state["failed"] is None:
                        state["failed"] = f"{type(e).__name__}: {e} (transcript raw/{name})"
                stop.set()
                return
            with lock:
                if keep_raw:
                    (run_dir / "raw" / f"hand-{rec['hand_id']}.json").write_text(json.dumps(transcript, indent=1))
                with hands_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                records.append(rec)
                n = len(records) - state["started"]
                if n == 1 or n % args.every == 0 or n == todo:
                    rate = n / max(time.time() - state["t0"], 1e-9)
                    print(f"[gtow] {fmt(stats(records))} | {rate:.2f} hands/s", flush=True)
                    state["printed"] = time.time()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    beat = started = time.time()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
            if time.time() - max(beat, state["printed"]) >= HEARTBEAT:
                beat = time.time()
                replies = (f"{api.replies} server replies, the last {beat - api.last_reply:.0f}s ago" if api.replies
                           else f"no server reply yet in {beat - started:.0f}s")
                print(f"[gtow] still running: {len(records) - state['started']} hands finished, {replies}", flush=True)
    except KeyboardInterrupt:
        print("\n[gtow] stopping after the hands in flight (Ctrl+C again to abort)…", flush=True)
        stop.set()
        for t in threads:
            t.join()
    summary = stats(records)
    (run_dir / "summary.json").write_text(json.dumps({**summary, "stopped": state["failed"], "config": config}, indent=1))
    print(f"[gtow] {fmt(summary)}\n[gtow] hands: {hands_path}")
    if state["failed"]:
        print(f"[gtow] STOPPED: {state['failed']}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True, help="blueprint.bin")
    ap.add_argument("--abstraction-dir", default=str(ROOT / "data" / "abstraction"))
    ap.add_argument("--label", default="bot")
    ap.add_argument("--hands", type=int, default=2500, help="finished hands the run should reach")
    ap.add_argument("--run", help="existing run dir to resume")
    ap.add_argument("--concurrency", type=int, default=5, help="hands in flight (server allows 20)")
    ap.add_argument("--search", choices=["none", "river", "turn"], default="none")
    ap.add_argument("--river-iters", type=int, default=200)
    ap.add_argument("--turn-iters", type=int, default=100)
    ap.add_argument("--own-range", choices=["exact", "model"], default="exact")
    ap.add_argument("--round", choices=["solve", "forced"], default="solve")
    ap.add_argument("--raw", type=int, default=10, help="hands whose full transcript is saved")
    ap.add_argument("--every", type=int, default=25, help="print progress every N hands")
    ap.add_argument("--key")
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args(argv)
    if not 1 <= args.concurrency <= 20:
        ap.error("--concurrency must be 1..20")
    api = Api(args.url, api_key(args.key))
    return run(args, build_agent(args), api)


if __name__ == "__main__":
    raise SystemExit(main())
