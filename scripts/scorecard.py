#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Bot scorecard: track whether the playing bot improves (see src/parciver/scorecard.py).

    # once: freeze a baseline bot (a completed blueprint.bin + its abstraction dir);
    # new benchmarks are scorecard-v2 (v1's rows + the realistic-sizing row)
    .venv/Scripts/python.exe scripts/scorecard.py init --baseline runs/blueprint-20260914-144924/blueprint.bin --abstraction-dir data/abstraction --out runs/scorecard/scorecard-v2

    # each checkpoint / bot version (opens a live dashboard; --no-browser to suppress)
    .venv/Scripts/python.exe scripts/scorecard.py evaluate --benchmark runs/scorecard/scorecard-v2/benchmark.json --candidate <blueprint.bin> --label v2 [--previous <receipt.json>] [--threads 16]

    # the same blueprint with real-time search on river (or turn+river) decisions
    ... evaluate ... --candidate <blueprint.bin> --label v1+search --search river [--river-iters 200] [--own-range exact|model] [--round solve|forced]

    # the table of every completed evaluation
    .venv/Scripts/python.exe scripts/scorecard.py history --benchmark runs/scorecard/scorecard-v2/benchmark.json

    # re-open the dashboard of the latest (or a given) evaluation
    .venv/Scripts/python.exe scripts/scorecard.py view --benchmark runs/scorecard/scorecard-v2/benchmark.json [--run <live-dir>]

    # split an older receipt's LBR into its postflop and preflop-ended parts
    # (reruns its LBR, which must reproduce exactly; writes <receipt>.lbr-split.json)
    .venv/Scripts/python.exe scripts/scorecard.py lbr-split --receipt <receipt.json> [--threads 16]

`--quick` on init makes a separate `scorecard-v2-quick` benchmark (in `<out>-quick/`) with 1/10 the
hands for fast iteration; its history never mixes with the full one.

Every row is played in `CHUNKS` consecutive chunks of its deals (`first_deal`), so
the dashboard can show progress and a running estimate inside a row; deal `i`
depends only on `(seed, i)`, so the result is identical to one call.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parciver import scorecard as sc  # noqa: E402

ABSTRACTION_FILES = ("flop_buckets.npy", "turn_buckets.npy", "river_buckets.npy")
CHUNKS = 10
DEFAULT_PORT = 8773
PAGE = "/static/scorecard.html"
# Seconds the dashboard keeps serving after the evaluation ends.
LINGER_S = 3.0


def _engine():
    import parciver_engine as pe
    return pe


def _git() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return None
    head = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {"commit": head, "dirty": None if dirty is None else bool(dirty)}


def _engine_identity() -> dict:
    pe = _engine()
    path = Path(pe.__file__).parent
    binaries = sorted(p for p in path.iterdir() if p.suffix in (".pyd", ".so"))
    return {"path": str(binaries[0]) if binaries else None,
            "sha256": sc.sha256_file(binaries[0]) if binaries else None}


def _abstraction_identity(abs_dir: Path) -> dict:
    files = {}
    for name in ABSTRACTION_FILES:
        f = abs_dir / name
        if not f.exists():
            raise sc.ScorecardError(f"abstraction file missing: {f}")
        files[name] = sc.sha256_file(f)
    return {"dir": str(abs_dir.resolve()), "sha256": files}


def _write_new(path: Path, obj: dict) -> None:
    """Atomic create: temp file + rename; never overwrite an existing file."""
    if path.exists():
        raise sc.ScorecardError(f"refusing to overwrite {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    if path.exists():
        tmp.unlink()
        raise sc.ScorecardError(f"refusing to overwrite {path}")
    os.replace(tmp, path)


def _safe(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)


def _chunked(call, deals: int, chunks: int, hands_per_deal: int, on_chunk=None, check=None) -> dict:
    """Play `deals` as consecutive `first_deal` chunks and merge them (identical
    to one call). `check(result, first_deal, deals)` on each chunk before it is
    kept; `on_chunk(done, parts, chunk_seconds)` after each chunk."""
    n = chunks if chunks > 1 and deals % chunks == 0 and deals // chunks >= 2 else 1
    size = deals // n
    parts = []
    for k in range(n):
        t = time.time()
        part = dict(call(first_deal=k * size, deals=size))
        if check:
            check(part, k * size, size)
        parts.append(part)
        if on_chunk:
            on_chunk((k + 1) * size, parts, time.time() - t)
    return sc.merge_chunks(parts, hands_per_deal)


def _no_failed_search(keys: tuple[str, ...], name: str, counted: bool):
    """A `_chunked` check: no decision of the chunk reached a failed search."""
    return lambda res, first, deals: sc.check_search_failures(res, keys, f"{name} deals {first}..{first + deals - 1}",
                                                              counted)


def measure(manifest: dict, cand_agent, base_agent, base_policy, threads: int, log=print,
            progress=None, chunks: int = CHUNKS, counted: bool = False) -> tuple[dict, dict]:
    """Run every row and LBR. `base_policy` builds the leak opponents (may be None
    only if the manifest has no leak rows). `progress(name, done, total, parts,
    chunk_seconds)` is called after every chunk of every row (LBR as "LBR").
    A chunk in which a decision reached a failed search stops the run;
    `counted`: the engine must count them (a search-enabled candidate)."""
    st = manifest["stakes"]
    hpd = manifest["hands_per_deal"]
    stakes = (st["sb"], st["bb"], st["stack"])
    out = {}
    for r in sc.rows_of(manifest):
        t = time.time()
        kw = dict(seed=r.seed, threads=threads)
        if r.kind == "agent":
            def call(first_deal, deals, kw=kw):
                return cand_agent.match_against("agent", deals, *stakes, other=base_agent,
                                                first_deal=first_deal, **kw)
        elif r.kind == "leak":
            opp = base_policy.leak(r.family, r.strength).play_agent()

            def call(first_deal, deals, kw=kw, opp=opp):
                return cand_agent.match_against("agent", deals, *stakes, other=opp,
                                                first_deal=first_deal, **kw)
        elif r.kind == "perturbed":
            def call(first_deal, deals, kw=kw, f=r.perturb):
                return cand_agent.match_against("perturbed", deals, *stakes, other=base_agent, perturb=f,
                                                first_deal=first_deal, **kw)
        elif r.kind == "realistic":
            def call(first_deal, deals, kw=kw):
                return cand_agent.match_against("realistic", deals, *stakes, other=base_agent,
                                                first_deal=first_deal, **kw)
        elif r.kind == "builtin":
            def call(first_deal, deals, kw=kw, name=r.builtin):
                return cand_agent.match_against(name, deals, *stakes, first_deal=first_deal, **kw)
        else:
            raise sc.ScorecardError(f"unknown row kind {r.kind}")
        hook = None
        if progress:
            def hook(done, parts, s, name=r.name, total=r.deals):
                progress(name, done, total, parts, s)
        res = _chunked(call, r.deals, chunks, hpd, hook, _no_failed_search(sc.MATCH_COUNTS, r.name, counted))
        out[r.name] = res
        log(f"  {r.name:>22s}: {res['bb_per_100']:+8.1f} ± {res['se']:.1f} bb/100  ({time.time() - t:.0f}s)")
    t = time.time()
    lb = manifest["lbr"]

    def call(first_deal, deals):
        return cand_agent.lbr(deals, *stakes, seed=lb["seed"], threads=threads, first_deal=first_deal)
    hook = None
    if progress:
        def hook(done, parts, s):
            progress("LBR", done, lb["deals"], parts, s)
    lbr = _chunked(call, lb["deals"], chunks, hpd, hook, _no_failed_search(sc.LBR_COUNTS, "LBR", counted))
    log(f"  {'LBR':>22s}: {lbr['bb_per_100']:+8.1f} ± {lbr['se']:.1f} bb/100  ({time.time() - t:.0f}s)")
    return out, lbr


# ---------------------------------------------------------------- live dashboard


def _free_port(start: int) -> int:
    for port in range(start, start + 30):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise sc.ScorecardError(f"no free port in {start}..{start + 29}")


def _serve(run_dir: Path, port: int, open_browser: bool, block: bool):
    """Serve the dashboard for `run_dir` from a background thread of THIS process
    (it stops with the script; `view` re-opens a finished run)."""
    from parciver.dashboard import server as dash_server
    port = _free_port(port)
    httpd = dash_server.serve(run_dir, port=port, open_browser=False, block=False)
    url = f"http://127.0.0.1:{port}{PAGE}"
    print(f"[dashboard] {url}", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    if block:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            httpd.shutdown()
    return httpd


class Live:
    """Telemetry for one evaluation: `events.jsonl` in the run directory, a
    2-second heartbeat (resources, the chunk in flight), and one checkpoint per
    chunk / finished row. Every number the dashboard shows comes from here."""

    def __init__(self, run_dir: Path, manifest: dict, threads: int):
        from parciver.resources import ResourceSampler
        from parciver.telemetry import RunWriter
        self.writer = RunWriter(run_dir)
        self.lock = threading.Lock()
        self.sampler = ResourceSampler(threads=threads or os.cpu_count())
        self.manifest = manifest
        self.total_units = sum(r.deals for r in sc.rows_of(manifest)) + manifest["lbr"]["deals"]
        self.done_units = 0
        self.rows_done_units = 0
        self.current: dict | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._beat, name="scorecard-heartbeat", daemon=True)

    def start(self, config: dict) -> None:
        with self.lock:
            depth = self.manifest["stakes"]["stack"] // self.manifest["stakes"]["bb"]
            self.writer.start(variant="scorecard", game=f"nlhe-{depth}bb-real-chips", config=config,
                              target_iters=self.total_units)
        self.thread.start()

    def _beat(self) -> None:
        while not self.stop.wait(2.0):
            fields = {"resources": self.sampler.sample()}
            cur = self.current
            if cur:
                fields["sc_current"] = {**cur, "elapsed_s": time.time() - cur["started"]}
            with self.lock:
                if self.stop.is_set():
                    break
                self.writer.metrics(self.done_units, **fields)

    def begin_chunk(self, row: str, expected_s: float | None) -> None:
        self.current = {"row": row, "started": time.time(), "expected_s": expected_s}

    def chunk(self, row: str, done: int, total: int, parts: list[dict], seconds: float) -> None:
        merged = sc.merge_chunks(parts, self.manifest["hands_per_deal"])
        self.done_units = self.rows_done_units + done
        if done == total:
            self.rows_done_units += total
        payload = {"row": row, "done": done, "total": total, "chunk_s": seconds,
                   "units_done": self.done_units, "units_total": self.total_units,
                   "partial": {"mean": merged["bb_per_100"], "se": merged["se"], "deals": done,
                               "sources": merged.get("a_sources")}}
        with self.lock:
            self.writer.checkpoint(self.done_units, sc_progress=payload)

    def row(self, payload: dict) -> None:
        with self.lock:
            self.writer.checkpoint(self.done_units, sc_row=payload)
            self.writer.milestone("note", f"{payload['name']}: {payload['mean']:+.1f} "
                                          f"[{payload['lo']:+.1f}, {payload['hi']:+.1f}] bb/100")

    def end(self, reason: str, summary: dict | None = None) -> None:
        self.stop.set()
        self.current = None
        with self.lock:
            self.writer.end(self.done_units, reason=reason, summary=summary)
            self.writer.close()


def _sidecar_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(receipt_path.stem + ".lbr-split.json")


def _load_receipt(path: Path) -> tuple[dict, list[float] | None]:
    """A receipt and its LBR preflop part (recorded, or from its `lbr-split` sidecar)."""
    r = json.loads(path.read_text(encoding="utf-8"))
    side = _sidecar_path(path)
    return r, sc.lbr_split_of(r, json.loads(side.read_text(encoding="utf-8")) if side.exists() else None)


def _receipts(manifest_path: Path, manifest: dict) -> list[tuple[dict, list[float] | None]]:
    """Every completed receipt of the benchmark, oldest first, with its LBR split."""
    out = []
    for p in sorted(manifest_path.parent.glob("receipt-*.json")):
        if p.name.endswith(".lbr-split.json"):
            continue
        r, split = _load_receipt(p)
        if r.get("manifest_hash") == manifest["manifest_hash"] and r.get("state") == "complete":
            out.append((r, split))
    out.sort(key=lambda rs: rs[0]["started_utc"])
    return out


def _history(manifest_path: Path, manifest: dict) -> list[dict]:
    return [sc.summary_of(r, split) for r, split in _receipts(manifest_path, manifest)]


def _next_row(manifest: dict, name: str) -> str | None:
    names = [r.name for r in sc.rows_of(manifest)] + ["LBR"]
    i = names.index(name)
    return names[i + 1] if i + 1 < len(names) else None


def run_evaluation(manifest: dict, manifest_path: Path, cand_agent, base_agent, base_policy, *, label: str,
                   candidate: dict, previous: dict | None, threads: int, live: Live | None,
                   verify=None, log=print, previous_split: list[float] | None = None) -> tuple[dict, Path]:
    """Measure, score and write the receipt, streaming progress to `live`.
    `verify()` runs after measuring and before the receipt is written.
    `previous_split`: the previous receipt's LBR preflop part if it predates it."""
    started = dt.datetime.now(dt.timezone.utc)
    t0 = time.time()
    blocks, hpd = manifest["blocks"], manifest["hands_per_deal"]
    prev_parts = {} if previous is None else sc.lbr_part_scores(previous, previous_split, hpd)

    def prev_blocks(name):
        if previous is None:
            return None
        s = previous["scores"]
        row = s["exploitability_lbr"] if name == "LBR" else s["rows"].get(name)
        return None if row is None else row["blocks"]

    def progress(name, done, total, parts, seconds):
        live.chunk(name, done, total, parts, seconds)
        if done < total:
            live.begin_chunk(name, seconds)
            return
        merged = sc.merge_chunks(parts, hpd)
        b = sc.to_blocks(merged["per_deal_bb"], blocks, hpd)
        s = sc.summarize(b)
        payload = {"name": name, "mean": s["mean"], "lo": s["lo"], "hi": s["hi"], "se": s["se"],
                   "sources": merged.get("a_sources"), "deals": total}
        pb = prev_blocks(name)
        if pb is not None:
            c = sc.paired_change(b, pb)
            payload["change"] = {**c, "status": sc.status(c["lo"], c["hi"], name != "LBR")}
        parts_b = sc.lbr_parts(merged["per_deal_bb"], merged.get("per_deal_preflop_bb"), blocks, hpd) \
            if name == "LBR" else None
        for part, pb in (parts_b or {}).items():
            s = sc.summarize(pb)
            payload[part] = {k: s[k] for k in ("mean", "lo", "hi", "se")}
            prev = prev_parts.get(f"exploitability_lbr_{part}")
            if part == "postflop" and prev is not None:
                c = sc.paired_change(pb, prev["blocks"])
                payload[part]["change"] = {**c, "status": sc.status(c["lo"], c["hi"], False)}
        live.row(payload)
        nxt = _next_row(manifest, name)
        if nxt:
            live.begin_chunk(nxt, None)

    if live:
        live.begin_chunk(sc.rows_of(manifest)[0].name, None)
    measurements, lbr = measure(manifest, cand_agent, base_agent, base_policy, threads, log=log,
                                progress=progress if live else None, counted=candidate.get("search") is not None)
    if verify:
        verify()
    scores = sc.score_receipt(manifest, measurements, lbr, measurements[sc.HEADLINE]["a_sources"])
    receipt = {
        "schema": sc.SCHEMA,
        "state": "complete",
        "benchmark_id": manifest["benchmark_id"],
        "manifest_hash": manifest["manifest_hash"],
        "stakes": manifest["stakes"],
        "candidate": {"label": label, **candidate},
        "baseline": manifest["baseline"],
        "abstraction": manifest["abstraction"],
        "engine": _engine_identity(),
        "git": _git(),
        "script": {"path": "scripts/scorecard.py", "sha256": sc.sha256_file(__file__)},
        "threads": threads,
        "started_utc": started.isoformat(),
        "elapsed_s": time.time() - t0,
        "raw": {"rows": measurements, "lbr": lbr},
        "scores": scores,
    }
    if previous is not None:
        receipt["change"] = sc.compare(receipt, previous, previous_split, hpd)
    path = manifest_path.parent / f"receipt-{started:%Y%m%dT%H%M%SZ}-{_safe(label)}.json"
    _write_new(path, receipt)
    return receipt, path


def live_config(manifest: dict, manifest_path: Path, *, candidate: dict, baseline_file: str,
                previous: dict | None, threads: int, previous_split: list[float] | None = None) -> dict:
    """The dashboard's run_start config: the benchmark, the candidate, the
    previous receipt's numbers and every completed evaluation so far."""
    return {
        "kind": "experiment", "study": "scorecard",
        "benchmark_id": manifest["benchmark_id"], "stakes": manifest["stakes"],
        "blocks": manifest["blocks"], "chunks": CHUNKS, "headline": sc.HEADLINE,
        "rows": [asdict(r) for r in sc.rows_of(manifest)], "lbr": manifest["lbr"],
        "candidate": candidate,
        "baseline": {"file": baseline_file, "sha256": manifest["baseline"]["sha256"][:12]},
        "previous": sc.summary_of(previous, previous_split) if previous else None,
        "history": _history(manifest_path, manifest),
        "threads": threads,
    }


def end_summary(receipt: dict, path: Path) -> dict:
    return {"scores": {k: v for k, v in receipt["scores"].items() if k != "rows"},
            "rows": sc.summary_of(receipt)["rows"], "change": receipt.get("change"),
            "receipt": str(path), "text": sc.render(receipt), "elapsed_s": receipt["elapsed_s"]}


# ---------------------------------------------------------------- commands


def cmd_init(args) -> int:
    out = Path(args.out)
    baseline = Path(args.baseline).resolve()
    rows, lbr, bid = sc.DEFAULT_ROWS, dict(sc.DEFAULT_LBR), sc.BENCHMARK_ID
    stakes = dict(sc.STAKES)
    if args.stack <= 0:
        raise sc.ScorecardError("--stack must be a positive number of big blinds")
    if args.stack != sc.STAKES["stack"] // sc.STAKES["bb"]:
        # Another depth is another benchmark: its own id, directory and history.
        stakes["stack"] = args.stack * stakes["bb"]
        bid += f"-{args.stack}bb"
        if not out.name.endswith(f"-{args.stack}bb"):
            out = out.with_name(out.name + f"-{args.stack}bb")
    if args.quick:
        rows = tuple(replace(r, deals=r.deals // 10) for r in rows)
        lbr["deals"] //= 10
        bid += "-quick"
        # Keep the quick benchmark out of the full one's directory.
        if not out.name.endswith("-quick"):
            out = out.with_name(out.name + "-quick")
    manifest_path = out / "benchmark.json"
    if manifest_path.exists():
        raise sc.ScorecardError(f"{manifest_path} exists; a benchmark is never re-initialized")
    manifest = sc.make_manifest(
        bid,
        {"path": str(baseline), "sha256": sc.sha256_file(baseline)},
        _abstraction_identity(Path(args.abstraction_dir)),
        rows=rows,
        lbr=lbr,
        stakes=stakes,
    )
    out.mkdir(parents=True, exist_ok=True)
    _write_new(manifest_path, manifest)
    print(f"initialized {bid} at {manifest_path} ({sc.stakes_text(stakes)})\n"
          f"baseline {baseline} ({manifest['baseline']['sha256'][:12]}…)")
    return 0


def _checked_manifest(manifest_path: Path) -> dict:
    """The benchmark manifest, with its frozen baseline and abstraction unchanged."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sc.check_manifest(manifest)
    base = manifest["baseline"]
    if sc.sha256_file(base["path"]) != base["sha256"]:
        raise sc.ScorecardError(f"baseline {base['path']} changed since init; start a new benchmark")
    abs_dir = Path(manifest["abstraction"]["dir"])
    if _abstraction_identity(abs_dir)["sha256"] != manifest["abstraction"]["sha256"]:
        raise sc.ScorecardError("abstraction files changed since init; start a new benchmark")
    return manifest


def _candidate_agent(manifest: dict, cand_path: Path, cand_hash: str, search: dict | None,
                     translation: str = "snap", open_reading: str = "snap"):
    """(baseline blueprint, candidate play agent, its description). `search`:
    None or {streets, river_iters, turn_iters, own_range, round} as in receipts
    (receipts from before own_range / round existed used model / forced).
    `translation`: how bets above the largest grid size are read ("snap", the
    default and all receipts from before it existed, or "standin"; TRANSLATE.md).
    `open_reading`: how a small-blind open below the grid is read ("snap", the
    default and all receipts from before it existed, or "limp": F2,
    records/preflop-open)."""
    pe = _engine()
    abs_dir, base = str(Path(manifest["abstraction"]["dir"])), manifest["baseline"]
    baseline = pe.SavedBlueprint(abs_dir, base["path"])
    candidate = baseline if cand_hash == base["sha256"] else pe.SavedBlueprint(abs_dir, str(cand_path))
    cand_agent = candidate.play_agent()
    agent_desc = "blueprint + translation + fallback (play.rs)"
    if translation != "snap":
        cand_agent = cand_agent.with_translation(translation)
        agent_desc = f"blueprint + translation ({translation}; TRANSLATE.md) + fallback (play.rs)"
    if open_reading != "snap":
        cand_agent = cand_agent.with_open_reading(open_reading)
        agent_desc += f" + open reading ({open_reading}; records/preflop-open)"
    if search is not None:
        own, rnd = search.get("own_range", "model"), search.get("round", "forced")
        cand_agent = cand_agent.with_search(from_turn=search["streets"] == "turn", river_iters=search["river_iters"],
                                            turn_iters=search["turn_iters"], exact_own_range=own == "exact",
                                            solve_round=rnd == "solve")
        agent_desc += (f" + real-time search ({search['streets']}; river_iters={search['river_iters']}, "
                       f"turn_iters={search['turn_iters']}; own range {own}; round {rnd}; subgame.rs)")
    return baseline, cand_agent, agent_desc


def cmd_evaluate(args) -> int:
    manifest_path = Path(args.benchmark)
    manifest = _checked_manifest(manifest_path)
    base = manifest["baseline"]
    previous, previous_split = None, None
    if args.previous:
        previous, previous_split = _load_receipt(Path(args.previous))
        if previous.get("manifest_hash") != manifest["manifest_hash"]:
            raise sc.ScorecardError("--previous is from a different benchmark definition")
    cand_path = Path(args.candidate).resolve()
    cand_hash = sc.sha256_file(cand_path)
    print(f"{manifest['benchmark_id']}: evaluating {args.label} ({cand_path.name}, {cand_hash[:12]}…)")
    search = None
    if args.search != "none":
        search = {"streets": args.search, "river_iters": args.river_iters, "turn_iters": args.turn_iters,
                  "own_range": args.own_range, "round": args.round}
    baseline, cand_agent, agent_desc = _candidate_agent(manifest, cand_path, cand_hash, search, args.translation,
                                                        args.open_reading)
    cand_info = {"path": str(cand_path), "sha256": cand_hash, "agent": agent_desc, "search": search}
    if args.translation != "snap":
        cand_info["translation"] = args.translation
    if args.open_reading != "snap":
        cand_info["open_reading"] = args.open_reading

    def verify():
        if sc.sha256_file(cand_path) != cand_hash:
            raise sc.ScorecardError("candidate file changed during evaluation; result discarded")

    run_dir = manifest_path.parent / f"live-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{_safe(args.label)}"
    live = Live(run_dir, manifest, args.threads)
    httpd = None if args.no_dashboard else _serve(run_dir, args.port, not args.no_browser, block=False)
    live.start(live_config(manifest, manifest_path,
                           candidate={"label": args.label, "file": cand_path.name, "sha256": cand_hash[:12],
                                      "agent": agent_desc, "search": search,
                                      "translation": args.translation, "open_reading": args.open_reading},
                           baseline_file=Path(base["path"]).name, previous=previous, threads=args.threads,
                           previous_split=previous_split))
    try:
        receipt, path = run_evaluation(manifest, manifest_path, cand_agent, baseline.play_agent(),
                                       baseline.s32_policy(), label=args.label, candidate=cand_info,
                                       previous=previous, threads=args.threads, live=live, verify=verify,
                                       previous_split=previous_split)
    except BaseException as e:
        live.end("interrupted" if isinstance(e, KeyboardInterrupt) else "error",
                 {"error": f"{type(e).__name__}: {e}"})
        raise
    live.end("completed", end_summary(receipt, path))
    if httpd is not None:
        time.sleep(LINGER_S)  # let open pages receive the final event before the server stops
    print()
    print(sc.render(receipt))
    print(f"\nreceipt: {path}")
    if httpd is not None:
        print(f"dashboard stops with this script; re-open it with:\n"
              f"  scorecard.py view --benchmark {manifest_path}")
    return 0


def cmd_history(args) -> int:
    manifest_path = Path(args.benchmark)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipts = _receipts(manifest_path, manifest)
    if not receipts:
        print("no completed evaluations yet")
        return 0
    print(f"{'date (UTC)':<17} {'label':<16} {'strength':>22} {'LBR (lower=better)':>22} "
          f"{'postflop LBR':>22} {'off-tree':>9}  change")
    for r, split in receipts:
        s = r["scores"]
        st, ex = s["strength"], s["exploitability_lbr"]
        post = sc.lbr_part_scores(r, split, manifest["hands_per_deal"]).get("exploitability_lbr_postflop")
        post = "—" if post is None else f"{post['mean']:+8.1f} [{post['lo']:+.1f},{post['hi']:+.1f}]"
        off = "—" if s["off_tree_share"] is None else f"{100 * s['off_tree_share']:.1f}%"
        ch = r.get("change")
        change = f"{ch['strength']['status']} (vs {ch['previous_label']})" if ch else "—"
        print(f"{r['started_utc'][:16]:<17} {r['candidate']['label']:<16} "
              f"{st['mean']:+8.1f} [{st['lo']:+.1f},{st['hi']:+.1f}] "
              f"{ex['mean']:+8.1f} [{ex['lo']:+.1f},{ex['hi']:+.1f}] {post:>22} {off:>9}  {change}")
    return 0


def cmd_lbr_split(args) -> int:
    """Backfill the LBR postflop / preflop split of a receipt that predates it:
    rerun its LBR (same agent, deals and seed), require the per-deal results to
    reproduce EXACTLY, and write the split to a sidecar (the receipt, and the
    hashes that refer to it, are never changed)."""
    path = Path(args.receipt)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("state") != "complete":
        raise sc.ScorecardError("receipt is not complete")
    side = _sidecar_path(path)
    if "per_deal_preflop_bb" in receipt["raw"]["lbr"] or side.exists():
        print(f"{path.name}: already split")
        return 0
    manifest_path = path.parent / "benchmark.json"
    manifest = _checked_manifest(manifest_path)
    if receipt.get("manifest_hash") != manifest["manifest_hash"]:
        raise sc.ScorecardError(f"receipt is not from {manifest_path}")
    cand = receipt["candidate"]
    cand_path = Path(cand["path"])
    if sc.sha256_file(cand_path) != cand["sha256"]:
        raise sc.ScorecardError(f"candidate {cand_path} changed since the receipt")
    _, cand_agent, _ = _candidate_agent(manifest, cand_path, cand["sha256"], cand.get("search"),
                                        cand.get("translation", "snap"), cand.get("open_reading", "snap"))
    st, lb, hpd = manifest["stakes"], manifest["lbr"], manifest["hands_per_deal"]
    t = time.time()

    def call(first_deal, deals):
        return cand_agent.lbr(deals, st["sb"], st["bb"], st["stack"], seed=lb["seed"], threads=args.threads,
                              first_deal=first_deal)
    lbr = _chunked(call, lb["deals"], CHUNKS, hpd,
                   check=_no_failed_search(sc.LBR_COUNTS, "LBR", cand.get("search") is not None))
    if lbr["per_deal_bb"] != receipt["raw"]["lbr"]["per_deal_bb"]:
        raise sc.ScorecardError("LBR did not reproduce the receipt exactly (engine or agent changed); no split written")
    split = {"receipt": path.name, "receipt_hash": sc.canonical_hash(receipt),
             "per_deal_preflop_bb": lbr["per_deal_preflop_bb"], "engine": _engine_identity(), "git": _git(),
             "script": {"path": "scripts/scorecard.py", "sha256": sc.sha256_file(__file__)},
             "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    parts = sc.lbr_part_scores(receipt, sc.lbr_split_of(receipt, split), hpd)
    _write_new(side, split)
    po, pr = parts["exploitability_lbr_postflop"], parts["exploitability_lbr_preflop"]
    ex = receipt["scores"]["exploitability_lbr"]
    print(f"{cand['label']}: LBR {ex['mean']:+.1f} [{ex['lo']:+.1f}, {ex['hi']:+.1f}] = postflop "
          f"{po['mean']:+.1f} [{po['lo']:+.1f}, {po['hi']:+.1f}] + preflop-ended {pr['mean']:+.1f} "
          f"[{pr['lo']:+.1f}, {pr['hi']:+.1f}] bb/100 (reproduced exactly, {time.time() - t:.0f}s)")
    print(side)
    return 0


def cmd_view(args) -> int:
    manifest_path = Path(args.benchmark)
    if args.run:
        run_dir = Path(args.run)
    else:
        runs = [d for d in manifest_path.parent.glob("live-*") if (d / "events.jsonl").exists()]
        if not runs:
            raise sc.ScorecardError(f"no live evaluation runs under {manifest_path.parent}")
        run_dir = max(runs, key=lambda d: (d / "events.jsonl").stat().st_mtime)
    print(f"[dashboard] replaying {run_dir.name} (Ctrl+C to stop)")
    _serve(run_dir, args.port, not args.no_browser, block=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--baseline", required=True)
    p.add_argument("--abstraction-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--stack", type=int, default=sc.STAKES["stack"] // sc.STAKES["bb"],
                   help="starting stack in big blinds; other than 100 makes a separate "
                        "benchmark (e.g. scorecard-v2-200bb in <out>-200bb/)")
    p = sub.add_parser("evaluate")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--previous")
    p.add_argument("--threads", type=int, default=0, help="0 = all cores (results do not depend on it)")
    p.add_argument("--search", choices=["none", "river", "turn"], default="none",
                   help="candidate plays with real-time search on river (or turn+river) decisions")
    p.add_argument("--river-iters", type=int, default=200)
    p.add_argument("--turn-iters", type=int, default=100)
    p.add_argument("--own-range", choices=["exact", "model"], default="exact",
                   help="search's own range: exact (its real decisions) or model (the first version's)")
    p.add_argument("--round", choices=["solve", "forced"], default="solve",
                   help="search's current round: solve (the opponent's actions this round are solved) "
                        "or forced (the first versions': conditioned on through the blueprint)")
    p.add_argument("--translation", choices=["snap", "standin"], default="snap",
                   help="how the candidate reads bets above its largest grid size (TRANSLATE.md)")
    p.add_argument("--open-reading", choices=["snap", "limp"], default="snap",
                   help="how the candidate reads a small-blind open below the grid: as the grid's open (snap) "
                        "or between the limp and it (limp: F2, records/preflop-open)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--no-browser", action="store_true", help="serve the dashboard but do not open it")
    p.add_argument("--no-dashboard", action="store_true", help="no dashboard server (events are still written)")
    p = sub.add_parser("history")
    p.add_argument("--benchmark", required=True)
    p = sub.add_parser("lbr-split")
    p.add_argument("--receipt", required=True)
    p.add_argument("--threads", type=int, default=0, help="0 = all cores (results do not depend on it)")
    p = sub.add_parser("view")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--run", help="a live-* run directory (default: the latest)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    try:
        return {"init": cmd_init, "evaluate": cmd_evaluate, "history": cmd_history,
                "view": cmd_view, "lbr-split": cmd_lbr_split}[args.cmd](args)
    except sc.ScorecardError as e:
        print(f"scorecard: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
