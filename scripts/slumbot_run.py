# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Play Slumbot (records/slumbot/PREREG.md; flop search: records/live-flop/PREREG.md; its leaves:
records/flop-sym/PREREG.md; turn search: LIVESEARCH.md, Amendment 3): v2-8B with live GPU search, one hand and
one request at a time.

    .venv/Scripts/python.exe scripts/slumbot_run.py --mock          # the pre-registered 1,000-hand mock run
    .venv/Scripts/python.exe scripts/slumbot_run.py                 # live: a new run (the pilot, then N counted)
    .venv/Scripts/python.exe scripts/slumbot_run.py --dir <run>     # resume a run (a new session)

The bot: `SavedBlueprint(v2-8B).play_agent()` (translation snap) `.with_search(from_flop=True, flop_iters=500,
from_turn=True, turn_iters=400, river_iters=2000, solver=Watched(FlopSearch(final net, gamma 5, sym leaves,
turn=SubgameGpu(terminals64=True, turn_schedule=dict(gamma=5, shift=1)))), rescue=False)`: PREREG's production
live search with V5's K2 MATTERS reading; since 11 Oct that is KERNELS Q4's terminal sums (the 5-7 Oct run was
Q3), since LIVE-FLOP (11 Oct) D4's adopted flop search (its flop decisions' p95 <= 3.0 s, checked by the mock
run), since FLOP-SYM (12 Oct; records/flop-sym/PREREG.md, M1 and speed passed) its suit-symmetric flop leaves
(C2, "sym"), and since LIVESEARCH B1 (LIVESEARCH.md, Amendment 3; adopted as P1 held and P2 was not DOWN) the
turn solved at 400 iterations (100 before) with Stage 1's schedule shift1_g5 (DCFR gamma 5, the regret discount
shifted by 1; the river keeps the default, DCFR gamma 2), its turn and river decisions' p95 <= 2 s each,
checked by the mock run; since RIVER R1 (records/river-live/PREREG.md; P1 held, P2 not DOWN) the river at 2,000
iterations (200 before); since BP-8B (records/bp-8b/PREREG.md "Adoption"; a GAIN, the live match not DOWN) the
blueprint v2-8B (v2-2B before). A failed search stops the run before its action is sent (the hand is abandoned,
recorded as lost), so `rescue` never decides a sent action; False only saves the CPU solve. Each hand's options
come from `options_failure` with the hand's seed, and our action is sampled with a generator seeded by it (the seed: sha256 of
"<run seed>:<hand number>").

Files in the run folder (append-only JSONL, one line per record, flushed as written):
  run.json        the configuration, fixed at the first start (commit, engine and blueprint sha256, search, seed, N)
  requests.jsonl  every request and response, as it happens
  hands.jsonl     one record per hand: its number, phase (pilot / counted / mock), session, client_pos, our cards,
                  Slumbot's where shown, the board, the history, winnings, the checked payoff, and our options and
                  pick at each decision. A hand that did not finish (no answer after the retries, a failed search,
                  a stop or crash mid-hand) has `lost` and the reason: excluded from the reading, and counted
  events.jsonl    sessions, the pilot's projection, the end and why
  status.json     the live state (the hand in play), replaced after each answer: the dashboard reads it
  mock_check.json (mock) PREREG's check: chips conserved, every action legal, nothing lost, (LIVE-FLOP) the
                  searched flop decisions' p95 <= FLOP_P95, and (LIVESEARCH) the searched turn and river
                  decisions' p95 <= TURN_P95 and RIVER_P95, each street on its own; or why not (exit 5). Also
                  (FLOP-SYM, reported) the share of flop solves that took the symmetric path, and (LIVESEARCH
                  Amendment 5, reported) the GPU's other clients over the run
  gpu_clients.jsonl  (LIVESEARCH.md Amendment 5; off with --no-gpu-clients) the GPU's clients every ~2.5 s, from
                  Windows' GPU Engine counters, by scripts/gpu_clients.ps1 in its own process: a slow run says
                  why. Reported only (each session's end event and mock_check.json carry a summary); it decides
                  nothing. A helper that cannot start or stopped by itself is a warning event, a failed counter
                  read an "err" line in the log; neither stops the run. The helper also exits once we are gone
A run stops by itself at N counted hands (live) or --hands (mock), on a failed search, a protocol error, a
payoff mismatch, or lost hands above 0.5% of N. To stop it by hand, create STOP in the run folder: the session
stops after an even number of its hands (each session starts in the big blind and changes seat every hand).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from parciver import slumbot as S  # noqa: E402
from parciver.solver_watch import Watched, write_atomic  # noqa: E402

# v2-8B since BP-8B (records/bp-8b/PREREG.md "Adoption"; a GAIN, the live match not DOWN); before: v2-2B,
# runs/blueprint-20260927-141645/blueprint.bin
BLUEPRINT = ROOT / "runs" / "bp-8b-stage2" / "blueprint.bin"  # v2-8B, 200 bb, sha256 fd27fe50...c7ae071
ABSTRACTION = ROOT / "data" / "abstraction"
# river_iters: 2,000 since RIVER R1 (records/river-live/PREREG.md "Adoption"; P1 held, P2 not DOWN); 200 before
SEARCH = dict(from_flop=True, flop_iters=500, from_turn=True, turn_iters=400, river_iters=2000, rescue=False)
# LIVESEARCH B1 (Amendment 3; Stage 1's shift1_g5, records/livesearch/stage1.json): FusedSolver's schedule on
# turn jobs only. None (before B1) is the default, DCFR gamma 2.
TURN_SCHEDULE = dict(gamma=5, shift=1)
TERMINALS64 = True  # records/verify/v5.json: K2 MATTERS
NET = ROOT / "runs" / "turn-net" / "d2b-w2048-wd0.1" / "net.pt"  # D2b's N1 (9 Nov 2026, the author's yes): the flop search's leaves; before it D2's final-w2048-d5-e200 (N0)
FLOP_GAMMA = 5.0  # D4 (FLOPSEARCH M2)
LEAVES = "sym"  # FLOP-SYM C2 (records/flop-sym/PREREG.md "Adoption"); before 12 Oct "dense"
FLOP_P95 = 3.0  # seconds, the flop decisions' bar (records/live-flop/PREREG.md, the author's call 11 Oct)
TURN_P95 = RIVER_P95 = 2.0  # seconds, each street's own bar (LIVESEARCH.md; gated since B1, Amendment 3)
# SEED 20261024 for SLUMBOT-2 (records/slumbot-2, B4, the author's okay 25 Oct); the 5 Oct run's was 20261005. It seeds
# our action sampling, the mapping of Slumbot's off-grid bets onto our grid, and the mock's dealer; Slumbot deals the cards. The parser check keeps its own seed.
N, PILOT, SEED = 50_000, 100, 20261024
LIMIT_HOURS, LOST_SHARE = 60.0, 0.005
SAMPLE = Path(os.environ.get("PARCIVER_SLUMBOT_SAMPLE", "C:/tmp/slumbot/sample_api.py"))
# what decides play (all of src/ and engine/ but the dashboard): a change between sessions of one run is refused
# unless --accept-change, and a live run needs the registered mock run at the same play code
PLAY_CODE = ("src", "engine", "scripts/slumbot_run.py", ":(exclude)src/parciver/dashboard")
GATES = ROOT / "records" / "livesearch" / "mock_check.json"  # the registered mock run's check, committed
# (records/flop-sym/mock_check.json before LIVESEARCH B1)
LOCK = ROOT / "runs" / "slumbot" / ".runner.lock"
# LIVESEARCH.md Amendment 5: the GPU's other clients, logged beside the run by a helper process (outside PLAY_CODE:
# reported only, it decides nothing). PID 4 is the kernel ("System"): its copy engine carries our transfers.
GPU_CLIENTS = ROOT / "scripts" / "gpu_clients.ps1"
SYSTEM_PIDS = (0, 4)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def uncommitted() -> list[str]:
    """Changes in the play code (PLAY_CODE) that are not committed: tracked files changed, and untracked code
    (.py, .rs, .toml)."""
    out = git("status", "--porcelain", "-z", "--untracked-files=all", "--", *PLAY_CODE)  # -z: paths unquoted
    return [x for x in out.split("\0") if x and (not x.startswith("??") or x.endswith((".py", ".rs", ".toml")))]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def hand_seed(seed: int, k: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{k}".encode()).digest()[:8], "little") >> 1


class Appender:
    """One append-only JSONL file, on disk per line (status.json, written after, never runs ahead of it)."""

    def __init__(self, path: Path):
        torn = False
        if path.exists() and path.stat().st_size:
            with open(path, "rb") as f:
                f.seek(-1, 2)
                torn = f.read(1) != b"\n"
        self.f = open(path, "a", encoding="utf-8", newline="\n")
        if torn:  # a line cut by a power loss stays a line of its own (read_jsonl skips it)
            self.f.write("\n")

    def __call__(self, rec: dict) -> None:
        self.f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass  # a line cut by a crash: the hand it held is recovered from status.json as lost
    return out


class Tally:
    """The run's counts, from hands.jsonl."""

    def __init__(self, hands: list[dict]):
        self.next = 1 + max((h["hand"] for h in hands), default=0)
        self.done = {p: 0 for p in ("pilot", "counted", "mock")}
        self.lost = dict(self.done)
        self.seconds = dict.fromkeys(self.done, 0.0)
        for h in hands:
            self.add(h)

    def add(self, h: dict) -> None:
        if h.get("lost"):
            self.lost[h["phase"]] += 1
        else:
            self.done[h["phase"]] += 1
            self.seconds[h["phase"]] += h.get("seconds", 0.0)


class GpuClients:
    """LIVESEARCH.md Amendment 5: GPU_CLIENTS in its own process, appending to the run's gpu_clients.jsonl after a
    header line of ours (the session and our PID). Reported only: a helper that cannot start or stopped early is a
    warning (`warn`), never a stop; it also exits by itself once this process is gone. `cmd`: tests."""

    def __init__(self, path: Path, session: int, warn, on: bool = True, cmd: list[str] | None = None):
        self.path, self.session, self.warn, self.on, self.proc = path, session, warn, on, None
        if not on:
            return
        try:
            if cmd is None and os.name != "nt":
                raise OSError("Windows' GPU Engine counters only")
            head = Appender(path)
            head(dict(start=S._utc(time.time()), session=session, pid=os.getpid()))
            head.f.close()
            self.proc = subprocess.Popen(
                cmd or ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                        str(GPU_CLIENTS), "-Out", str(path), "-RunnerPid", str(os.getpid())],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)  # BelowNormal however we were started
                | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
        except Exception as e:  # anything: the run goes on without the log
            warn(f"the GPU-client log did not start: {type(e).__name__}: {e}")

    def stop(self) -> None:
        """Ends the helper (again: a no-op)."""
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if (code := proc.poll()) is not None:
                self.warn(f"the GPU-client log had stopped by itself (exit {code})")
            else:
                proc.terminate()
                proc.wait(timeout=10)
        except Exception as e:
            self.warn(f"the GPU-client log did not stop cleanly: {type(e).__name__}: {e}")

    def summary(self, session: int | None = None) -> dict | None:
        return gpu_clients_summary(self.path, session) if self.on else None


def gpu_clients_summary(path: Path, session: int | None = None) -> dict:
    """Amendment 5's report from gpu_clients.jsonl (every session, or one): counter reads and failed reads; the
    share of reads with another client (a PID neither ours nor the kernel's) and the largest other-client total
    in one read, when and whose; the other clients by the reads they appear in (top 8); the kernel's (SYSTEM_PIDS:
    it pages and copies for every process) share of reads, median and max percent; our own median percent. Never
    raises: a line it cannot read is skipped (torn) or counted in `bad`."""
    reads = errors = bad = others = 0
    own, kernel, top, by, ours, cur = [], [], None, {}, None, None
    try:
        lines = path.read_bytes().decode("utf-8", errors="replace").splitlines() if path.exists() else []
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue  # a line cut by a kill (or a blank one)
        try:
            if "start" in r:
                cur, ours = r.get("session"), r.get("pid")
                continue
            if session is not None and cur != session:
                continue
            if "err" in r:
                errors += 1
                continue
            g = [(str(n), int(p), str(e), float(v)) for n, p, e, v in r["g"]]
            if not all(np.isfinite(c[3]) for c in g):
                raise ValueError("a percent that is not finite")
        except Exception:  # anything in a line: counted, never raised
            bad += 1
            continue
        reads += 1
        other = [c for c in g if c[1] != ours and c[1] not in SYSTEM_PIDS]
        kernel.append(sum(c[3] for c in g if c[1] in SYSTEM_PIDS))
        own.append(sum(c[3] for c in g if c[1] == ours))
        if other:
            others += 1
            total = round(sum(c[3] for c in other), 1)
            if top is None or total > top["percent"]:
                top = dict(percent=total, utc=r.get("t"), by=sorted(f"{n}({p}:{e})={v}" for n, p, e, v in other))
            for name in {c[0] for c in other}:
                by[name] = by.get(name, 0) + 1
    share = (lambda k: round(k / reads, 4) if reads else None)
    return dict(reads=reads, errors=errors, bad=bad, other_reads=others, other_share=share(others), other_max=top,
                others=dict(sorted(by.items(), key=lambda kv: (-kv[1], kv[0]))[:8]),
                system_share=share(sum(1 for x in kernel if x)),
                system_median=float(np.median(kernel)) if kernel else None, system_max=max(kernel, default=None),
                own_median=float(np.median(own)) if own else None)


def hold_lock():
    """One runner at a time (PREREG: one client): an OS lock on LOCK, freed when this process ends."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOCK, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another Slumbot runner is playing ({LOCK} is held): one client at a time") from None
    return f


def gates_refuse(engine_sha: str, blueprint_sha: str, net_sha: str) -> str | None:
    """Why a live run may not start (None if it may): PREREG's parser check and mock run of 1,000 hands must
    have passed, on the play code of HEAD and this engine, blueprint, net and leaves, and their record must be
    committed."""
    name = GATES.relative_to(ROOT).as_posix()
    try:
        g = json.loads(GATES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"{name} is missing: the registered mock run (--mock) comes first"
    if not g.get("passed") or not (g.get("parser") or {}).get("passed"):
        return f"{name}: the mock run or the parser check did not pass"
    if g.get("hands", 0) < 1000:
        return f"{name}: a mock run of {g.get('hands')} hands, not PREREG's 1,000"
    if g.get("engine_sha256") != engine_sha or g.get("blueprint_sha256") != blueprint_sha \
            or g.get("net_sha256") != net_sha or g.get("leaves", "dense") != LEAVES:
        return f"{name}: the mock run played another engine build, blueprint, net or leaves"
    if git("status", "--porcelain", "--", name).strip():
        return f"{name} is not committed"
    changed = [x for x in git("diff", "--name-only", g["commit"], "HEAD", "--", *PLAY_CODE).splitlines() if x]
    if changed:
        return f"play code changed since the mock run at {g['commit'][:10]}: {changed} (run the mock again)"
    return None


def build_agent():
    import parciver_engine as pe
    from parciver import value_net as vn
    from parciver.flop_search import FlopSearch
    from parciver.gpu_fused import SubgameGpu

    net, _ = vn.load(NET)
    # Q4 (KERNELS Q4 landed; the author's call, 11 Oct); the turn's schedule: LIVESEARCH B1
    turn = SubgameGpu(terminals64=TERMINALS64, turn_schedule=TURN_SCHEDULE)
    solver = Watched(FlopSearch(net, gamma=FLOP_GAMMA, turn=turn, leaves=LEAVES))  # flop jobs: FLOP-SYM's leaves
    base = pe.SavedBlueprint(str(ABSTRACTION), str(BLUEPRINT)).play_agent()
    if base.translation != "snap":
        raise SystemExit(f"the play agent's translation is {base.translation!r}, not the registered 'snap'")
    return base.with_search(solver=solver, **SEARCH), solver, pe


def engine_options(agent, solver, seed: int):
    """`play_hand`'s options: the engine's for our holding (one solve per searched decision), or Stop where a
    search failed (the solver's own count, or `options_failure`'s index)."""
    def options(hero, hole, board, acts):
        solver.take()
        before = {k: len(v) for k, v in solver.seconds.items()}
        t = time.perf_counter()
        rows, failed = agent.options_failure(hero, board, list(acts), S.SB, S.BB, (S.STACK, S.STACK),
                                             [tuple(hole)], seed=seed)
        dt = time.perf_counter() - t
        calls, failures = solver.take()
        if failures or failed is not None:
            what = ["the built-in solver replaced the external one", "the blueprint decided"][failed] \
                if failed is not None else "the solver failed"
            raise S.Stop(f"a failed search ({what}; {failures} of {calls} solves failed; last: {solver.last_error})")
        solve = sum(sum(v[before.get(k, 0):]) for k, v in solver.seconds.items())  # its calls alone (one thread)
        return rows[0], {"seconds": round(dt, 4), "solves": calls, "solve_seconds": round(solve, 4)}
    return options


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true", help="play the offline mock server (PREREG's mock run)")
    p.add_argument("--hands", type=int, help=f"hands to play: counted hands live (registered: {N}), all in --mock "
                                             f"(default 1000)")
    p.add_argument("--dir", help="resume this run folder")
    p.add_argument("--go", action="store_true", help="the author's go after a pilot that projects over 60 h")
    p.add_argument("--accept-change", action="store_true", help="resume although play code changed (recorded)")
    p.add_argument("--allow-dirty", action="store_true", help="--mock only, for development: play uncommitted code "
                                                               "(recorded; not the registered mock run)")
    p.add_argument("--no-gpu-clients", action="store_true", help="do not log the GPU's other clients beside the run "
                                                                  "(LIVESEARCH.md Amendment 5; recorded)")
    args = p.parse_args()

    if args.dir:
        run = Path(args.dir)
        cfg = json.loads((run / "run.json").read_text(encoding="utf-8"))
        mode = cfg["mode"]
        if args.mock != (mode == "mock") or (args.hands and args.hands != cfg["hands"]):
            raise SystemExit(f"{run} is a {mode} run of {cfg['hands']} hands: resume it without changing them")
    else:
        mode = "mock" if args.mock else "live"
        hands = args.hands or (1000 if args.mock else N)
        if mode == "live" and hands != N:
            raise SystemExit(f"N is registered at {N} (PREREG): no other count is a live run")
        if hands % 2:
            raise SystemExit("an even number of hands (each seat equally often)")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run = ROOT / "runs" / "slumbot" / (f"mock-{stamp}" if args.mock else stamp)
    lock = hold_lock()  # noqa: F841 (held until the process ends)
    dirty = uncommitted()
    if args.allow_dirty and mode != "mock":
        raise SystemExit("--allow-dirty is for --mock runs only")
    if dirty and not args.allow_dirty:
        raise SystemExit("uncommitted play code (PREREG: the run refuses it):\n  " + "\n  ".join(dirty))
    commit = git("rev-parse", "HEAD").strip()

    parse_action = S.load_parse_action(SAMPLE)
    if mode == "mock" and parse_action is None:
        raise SystemExit(f"the mock run checks every action with Slumbot's ParseAction: {SAMPLE} is missing")

    agent, solver, pe = build_agent()
    engine_sha = sha256(Path(pe.parciver_engine.__file__))
    net_sha = sha256(NET)
    if mode == "live" and (why := gates_refuse(engine_sha, sha256(BLUEPRINT), net_sha)):
        raise SystemExit(f"no live hand before PREREG's checks: {why}")
    if not args.dir:
        run.mkdir(parents=True)
        cfg = dict(mode=mode, hands=hands, pilot=0 if mode == "mock" else PILOT, seed=SEED, commit=commit,
                   engine_sha256=engine_sha, blueprint=str(BLUEPRINT), blueprint_sha256=sha256(BLUEPRINT),
                   abstraction=str(ABSTRACTION), translation="snap", search={**SEARCH, "terminals64": TERMINALS64,
                   "q3": False, "solver": f"Watched(FlopSearch({LEAVES}, turn=SubgameGpu))", "leaves": LEAVES,
                   "flop_gamma": FLOP_GAMMA, "turn_schedule": TURN_SCHEDULE,
                   "net": str(NET), "net_sha256": net_sha, "flop_p95_bar": FLOP_P95, "turn_p95_bar": TURN_P95,
                   "river_p95_bar": RIVER_P95},
                   chips=dict(sb=S.SB, bb=S.BB, stack=S.STACK),
                   host=None if mode == "mock" else S.HOST, limit_hours=LIMIT_HOURS,
                   lost_limit=int(LOST_SHARE * hands), sample_api_sha256=S.SAMPLE_SHA256,
                   prereg=["records/slumbot/PREREG.md", "records/live-flop/PREREG.md", "records/flop-sym/PREREG.md",
                           "LIVESEARCH.md", "records/river-live/PREREG.md", "records/bp-8b/PREREG.md",
                           "records/slumbot-2/PREREG.md"],
                   started=S._utc(time.time()), dirty=dirty or None)
        write_atomic(run / "run.json", json.dumps(cfg, indent=1))
    else:
        if engine_sha != cfg["engine_sha256"] or sha256(Path(cfg["blueprint"])) != cfg["blueprint_sha256"]:
            raise SystemExit(f"{run} was played on another engine or blueprint")
        if cfg["blueprint_sha256"] != sha256(BLUEPRINT):  # runs before BP-8B played v2-2B: never mix blueprints
            raise SystemExit(f"{run} was played on {cfg['blueprint']}, not {BLUEPRINT}")
        if cfg["search"].get("q3", True):  # runs before 11 Oct have no key and were Q3: never mix kernels
            raise SystemExit(f"{run} was played with Q3's terminal sums; this code runs Q4")
        if not cfg["search"].get("from_flop") or cfg["search"].get("net_sha256") != net_sha:
            raise SystemExit(f"{run} was played without flop search or with another net")
        if cfg["search"].get("leaves", "dense") != LEAVES:  # runs before 12 Oct have no key and were dense
            raise SystemExit(f"{run} was played with {cfg['search'].get('leaves', 'dense')} flop leaves, not {LEAVES}")
        # runs before LIVESEARCH B1 have turn_iters 100 and no turn_schedule key: never mix turn searches
        turn = (cfg["search"].get("turn_iters"), cfg["search"].get("turn_schedule"))
        if turn != (SEARCH["turn_iters"], TURN_SCHEDULE):
            raise SystemExit(f"{run} was played with turn {turn[0]} iterations and schedule {turn[1]}, not "
                             f"{SEARCH['turn_iters']} and {TURN_SCHEDULE}")
        if cfg["search"].get("river_iters") != SEARCH["river_iters"]:  # runs before RIVER R1 have 200
            raise SystemExit(f"{run} was played with river {cfg['search'].get('river_iters')} iterations, not "
                             f"{SEARCH['river_iters']}")
        changed = [x for x in git("diff", "--name-only", cfg["commit"], "HEAD", "--", *PLAY_CODE).splitlines() if x]
        if changed and not args.accept_change:
            raise SystemExit(f"play code changed since {cfg['commit'][:10]}: {changed} (--accept-change to resume anyway)")

    events, hands_log, req_log = (Appender(run / f) for f in ("events.jsonl", "hands.jsonl", "requests.jsonl"))
    hands = read_jsonl(run / "hands.jsonl")
    tally = Tally(hands)
    # a hand the last session left mid-way (a crash or a kill): recorded as lost
    try:
        last = json.loads((run / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        last = {}
    cur, last_hand = last.get("current"), (hands[-1] if hands else None)
    if cur and cur.get("hand", 0) >= tally.next:  # finished but not logged, or cut mid-hand
        cur.pop("token", None)
        if "seconds" not in cur and not cur.get("lost"):
            cur.update(lost=True, reason=f"the client stopped mid-hand ({last.get('state', '?')})")
        hands_log(cur)
        tally.add(cur)
        tally.next = cur["hand"] + 1
        last_hand = cur
    # a hand that stops the run, killed before the run's end was recorded: stop now
    ended = max((e.get("last_hand", -1) for e in read_jsonl(run / "events.jsonl") if e.get("event") == "end"),
                default=-1)
    recovered_stop = None
    if last_hand and (last_hand.get("stop") or last_hand.get("check") == "mismatch") and ended < last_hand["hand"]:
        recovered_stop = (f"hand {last_hand['hand']}, recovered: "
                          f"{last_hand.get('stop') or 'winnings disagree with the cards'}")
    tally.next = max(tally.next, int(last.get("next_hand") or 0))  # a number once used is never used again
    session = 1 + sum(1 for e in read_jsonl(run / "events.jsonl") if e.get("event") == "session")
    events(dict(event="session", session=session, utc=S._utc(time.time()), pid=os.getpid(), commit=commit,
                accept_change=args.accept_change or None, go=args.go or None, dirty=dirty or None,
                gpu_clients=False if args.no_gpu_clients else None))
    prior_events = read_jsonl(run / "events.jsonl")

    transport = S.MockSlumbot(SEED + session, referee=parse_action, log=req_log) if mode == "mock" \
        else S.Http(req_log)
    status = {"pid": os.getpid(), "session": session, "mode": mode, "state": "playing", "message": None}

    def phase() -> str:
        if mode == "mock":
            return "mock"
        return "pilot" if tally.done["pilot"] + tally.lost["pilot"] < cfg["pilot"] else "counted"

    def finished() -> bool:
        return tally.done["mock" if mode == "mock" else "counted"] >= cfg["hands"]

    def write_status(current=None, **kw) -> None:
        status.update(kw, utc=S._utc(time.time()), current=current, phase=phase(), done=tally.done,
                      lost=tally.lost, seconds=tally.seconds, next_hand=tally.next,
                      solver_calls=solver_calls, session_hands=session_hands, retried=getattr(transport, "retried", 0))
        write_atomic(run / "status.json", json.dumps(status, separators=(",", ":")))

    def warn(message: str) -> None:
        events(dict(event="warning", message=message, utc=S._utc(time.time()), session=session))
        print(f"[slumbot] warning: {message}", flush=True)

    def end(state: str, message: str, code: int) -> int:
        gpu.stop()
        events(dict(event="end", state=state, message=message, utc=S._utc(time.time()), session=session,
                    session_hands=session_hands, last_hand=tally.next - 1, gpu_clients=gpu.summary(session)))
        write_status(current=status.get("current"), state=state, message=message)  # a hand not yet logged stays
        print(f"[slumbot] {state}: {message}", flush=True)
        return code

    token, session_hands, solver_calls = None, 0, 0
    print(f"[slumbot] {mode} run {run} session {session}: {tally.done} done, {tally.lost} lost", flush=True)
    write_status()
    gpu = GpuClients(run / "gpu_clients.jsonl", session, warn, on=not args.no_gpu_clients)  # Amendment 5: reported only
    try:
        while True:
            if recovered_stop:
                return end("stopped", recovered_stop, 2)
            if finished():
                if mode == "mock":
                    gpu.stop()  # the log covers the hands (mock_check reports it)
                    ok, message = mock_check(run, cfg, transport, parse_action, tally, solver.solver)
                    return end("done" if ok else "failed", message, 0 if ok else 5)
                return end("done", f"{tally.done['counted']} counted hands", 0)
            if (run / "STOP").exists() and session_hands % 2 == 0:
                return end("stopped", f"STOP after {session_hands} hands of session {session}", 0)
            if mode == "live" and phase() == "counted":
                pilots = [e for e in prior_events if e.get("event") == "pilot"]
                if not pilots:
                    d = tally.done["pilot"]
                    per = tally.seconds["pilot"] / d if d else None  # every pilot hand lost: no projection
                    pilots = [dict(event="pilot", utc=S._utc(time.time()), hands=d, lost=tally.lost["pilot"],
                                   s_per_hand=per, projected_hours=per * cfg["hands"] / 3600 if d else None,
                                   limit_hours=LIMIT_HOURS)]
                    events(pilots[0])
                    prior_events.append(pilots[0])
                hours = pilots[-1]["projected_hours"]
                after = prior_events[prior_events.index(pilots[-1]) + 1:]
                went = any(e.get("event") == "session" and e.get("go") for e in after)  # a go given once it was seen
                if (hours is None or hours > LIMIT_HOURS) and not went:  # asked on every resume until then
                    told = "no pilot hand finished" if hours is None else f"the pilot projects {hours:.1f} h"
                    return end("paused", f"{told} for {cfg['hands']} hands (limit {LIMIT_HOURS:.0f} h):"
                                         f" the author decides (resume with --go to play on)", 3)
            if tally.lost["counted"] > cfg["lost_limit"]:
                return end("stopped", f"{tally.lost['counted']} lost counted hands, above 0.5% of N", 4)

            k, ph = tally.next, phase()
            seed = hand_seed(cfg["seed"], k)
            rec = {"hand": k, "phase": ph, "session": session, "seed": seed}
            rng = random.Random(seed)
            stop = None
            try:
                token = S.play_hand(transport.post, token, engine_options(agent, solver, seed), rng, rec,
                                    live=lambda r: write_status(current=r))
            except S.Lost as e:
                rec.update(lost=True, reason=f"lost: {e}")
            except (S.Stop, S.ProtocolError) as e:
                rec.update(lost=True, reason=f"{type(e).__name__}: {e}")
                stop = (f"{type(e).__name__}: {e}", 2)
            except KeyboardInterrupt:
                rec.update(lost=True, reason="interrupted mid-hand")
                stop = ("interrupted mid-hand", 130)
            except Exception as e:
                rec.update(lost=True, reason=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
                stop = (f"{type(e).__name__}: {e}", 1)
            token = rec.pop("token", token)  # the newest, even from a hand that failed
            if stop:
                rec["stop"] = stop[0]
                try:
                    write_status(current=rec)  # a kill before the log below still stops the resumed run
                except OSError:  # status.json held open (a reader, antivirus): the log below is the record
                    pass
            solver_calls += sum(d.get("solves", 0) for d in rec.get("decisions", []))
            hands_log(rec)
            tally.add(rec)
            tally.next = k + 1
            session_hands += 1
            write_status()
            if rec.get("check") == "mismatch":
                return end("stopped", f"hand {k}: winnings {rec['winnings']}, the cards and chips give {rec.get('want')}", 2)
            if stop:
                return end("stopped", stop[0], stop[1])
            if k % 50 == 0:
                d = tally.done[ph]
                print(f"[slumbot] hand {k} ({ph}): {d} done, {tally.lost[ph]} lost, "
                      f"{tally.seconds[ph] / max(1, d):.2f} s/hand", flush=True)
    except KeyboardInterrupt:  # between hands (one mid-hand is caught and recorded above)
        return end("stopped", "interrupted between hands", 130)
    finally:
        gpu.stop()


def cfg_dirty(run: Path) -> bool:
    """Whether any session of the run played uncommitted code (--allow-dirty)."""
    return any(e.get("dirty") for e in read_jsonl(run / "events.jsonl") if e.get("event") == "session")


def mock_check(run: Path, cfg: dict, mock: S.MockSlumbot, parse_action, tally: Tally, flop_search) -> tuple[bool, str]:
    """PREREG's checks before any counted hand: the mock run (chips conserved and every action legal: each
    hand's history passes ParseAction, and the mock refused none of ours) and the parser check. Written to
    mock_check.json; a live run needs it, committed as GATES, at its play code. `flop_search`: the run's
    FlopSearch (one session, so its counts cover the run): the symmetric share, reported."""
    print("[slumbot] the parser check (10,000 histories)", flush=True)
    parser = S.parser_check(parse_action)
    hands = [h for h in read_jsonl(run / "hands.jsonl") if not h.get("lost")]
    net = sum(h["winnings"] for h in hands)
    illegal = [h["hand"] for h in hands if S.refuses(parse_action, h["action"])]
    checks = {c: sum(1 for h in hands if h.get("check") == c) for c in ("ok", "unchecked", "mismatch")}
    single = len({h["session"] for h in hands}) == 1  # the mock's bot_net covers its own session only
    # LIVE-FLOP: the searched flop decisions' latency (each the whole options_failure call)
    sec = np.array([d["seconds"] for h in hands for d in h["decisions"] if d["street"] == 1 and d.get("solves", 0)])
    flop = dict(bar=FLOP_P95, n=len(sec)) | (dict(p95=float(np.percentile(sec, 95)), median=float(np.median(sec)),
                                                  mean=float(sec.mean()), max=float(sec.max())) if len(sec) else {})
    late = np.array([d["seconds"] for h in hands for d in h["decisions"] if d["street"] > 1 and d.get("solves", 0)])
    turn_river = dict(n=len(late)) | (dict(p95=float(np.percentile(late, 95)), median=float(np.median(late)),
                                           max=float(late.max())) if len(late) else {})  # pooled: reported
    # LIVESEARCH (Amendment 3): the searched turn and river decisions' latency, each street gated on its own
    by = {}
    for street, name, bar in ((2, "turn", TURN_P95), (3, "river", RIVER_P95)):
        x = np.array([d["seconds"] for h in hands for d in h["decisions"] if d["street"] == street and d.get("solves", 0)])
        by[name] = dict(bar=bar, n=len(x)) | (dict(p95=float(np.percentile(x, 95)), median=float(np.median(x)),
                                                   mean=float(x.mean()), max=float(x.max())) if len(x) else {})
    failed = [why for why, bad in (("code not committed (--allow-dirty)", cfg_dirty(run)),
                                   ("more than one session", not single),
                                   ("chips not conserved", single and net + mock.bot_net != 0),
                                   (f"{len(illegal)} illegal histories", illegal),
                                   (f"{len(mock.refused)} actions refused", mock.refused),
                                   (f"{checks['mismatch']} winnings mismatches", checks["mismatch"]),
                                   (f"{tally.lost['mock']} hands lost", tally.lost["mock"]),
                                   ("the parser check failed", not parser["passed"]),
                                   ("no searched flop decision", not len(sec)),
                                   (f"flop decisions' p95 {flop.get('p95', 0):.3f} s over {FLOP_P95} s",
                                    flop.get("p95", 0) > FLOP_P95),
                                   ("no searched turn decision", not by["turn"]["n"]),
                                   (f"turn decisions' p95 {by['turn'].get('p95', 0):.3f} s over {TURN_P95} s",
                                    by["turn"].get("p95", 0) > TURN_P95),
                                   ("no searched river decision", not by["river"]["n"]),
                                   (f"river decisions' p95 {by['river'].get('p95', 0):.3f} s over {RIVER_P95} s",
                                    by["river"].get("p95", 0) > RIVER_P95)) if bad]
    ok = not failed
    log = run / "gpu_clients.jsonl"  # none: --no-gpu-clients (reported as None)
    out = dict(commit=cfg["commit"], engine_sha256=cfg["engine_sha256"], blueprint_sha256=cfg["blueprint_sha256"],
               sample_api_sha256=S.SAMPLE_SHA256, run=run.name, finished=S._utc(time.time()), parser=parser,
               hands=len(hands), lost=tally.lost["mock"], net=net, bot_net=mock.bot_net if single else None,
               conserved=(net + mock.bot_net == 0) if single else None, refused=mock.refused, illegal=illegal,
               checks=checks, decisions=sum(len(h["decisions"]) for h in hands),
               solver_calls=sum(d.get("solves", 0) for h in hands for d in h["decisions"]),
               s_per_hand=sum(h["seconds"] for h in hands) / max(1, len(hands)), net_sha256=cfg["search"]["net_sha256"],
               flop=flop, turn=by["turn"], river=by["river"], turn_river=turn_river,
               turn_schedule=cfg["search"].get("turn_schedule"), turn_iters=cfg["search"]["turn_iters"],
               river_iters=cfg["search"]["river_iters"], leaves=flop_search.leaves,  # what ran (cfg's, by the resume guard)
               flop_solves=flop_search.solves, flop_symmetric=flop_search.symmetric,
               flop_symmetric_share=flop_search.symmetric / max(1, flop_search.solves), flop_drift=flop_search.drift,
               gpu_clients=gpu_clients_summary(log) if log.exists() else None,  # Amendment 5: reported, never gated
               failed=failed, passed=ok)
    write_atomic(run / "mock_check.json", json.dumps(out, indent=1))
    if not ok:
        return False, f"mock check FAILED: {'; '.join(failed)} (mock_check.json)"
    return True, (f"mock check PASSED: {len(hands)} hands, chips conserved, every history legal, none refused, "
                  f"{checks['ok']} winnings checked ({checks['unchecked']} unchecked), {out['decisions']} decisions; "
                  f"parser check: {parser['prefixes']:,} prefixes agree; flop decisions p95 {flop['p95']:.3f} s "
                  f"(<= {FLOP_P95}; median {flop['median']:.3f}, max {flop['max']:.3f}, n {flop['n']}); turn p95 "
                  f"{by['turn']['p95']:.3f} s (<= {TURN_P95}, n {by['turn']['n']}), river p95 {by['river']['p95']:.3f} s "
                  f"(<= {RIVER_P95}, n {by['river']['n']}); "
                  f"{out['flop_symmetric']}/{out['flop_solves']} flop solves symmetric")


if __name__ == "__main__":
    sys.exit(main())
