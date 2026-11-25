# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Training telemetry contract (Phase 3c, pre-B4) — the decoupling layer.

The blueprint solve (B4) is a run of hundreds of millions of MCCFR iterations we
want to *watch* live without ever slowing it down and without the viewer ever
being able to lag it. The mechanism is deliberately dumb and one-directional:

    training  ──append──▶  runs/<run_id>/events.jsonl  ◀──tail──  viewer

Training only ever **appends** small, versioned JSON events to an append-only log
(one JSON object per line — JSON Lines). A completely separate viewer process
tails that file and renders. There is no shared memory, no lock the trainer can
block on, no callback into the trainer: the trainer's only cost is serializing a
few hundred bytes every checkpoint and handing them to the OS write buffer. If the
viewer dies, stalls, or a hundred of them connect at once, the trainer neither
knows nor cares.

This module is the **contract** both sides agree on, and it is load-bearing: B4
emits to it, so a malformed event or a silent schema drift would corrupt the one
window we have into a multi-day solve. Therefore every event is:

  * **versioned** — a top-level ``v`` (``SCHEMA_VERSION``) on every line, so a
    future change is detectable rather than silently misread;
  * **validated on write** — :class:`RunWriter` runs :func:`validate_event` before
    a single byte hits the file, so the trainer fails loudly at the source rather
    than emitting garbage a reader discovers hours later;
  * **round-trippable** — :class:`RunReader` parses the same envelope back, and the
    test-suite + phase audit pin ``write -> read`` equality.

Design notes / why the shapes are what they are:

  * The **hot** stream (:meth:`RunWriter.metrics`) carries only scalars — iteration,
    throughput, the convergence signals, resource gauges — so it stays tiny and can
    be emitted often. The **heavy** periodic snapshot (:meth:`RunWriter.checkpoint`)
    carries the compact 13x13 preflop range grid and the per-street action/bet-size
    distributions, emitted rarely. Splitting them keeps ``events.jsonl`` small and
    the SSE push cheap even over a very long run.
  * Convergence is reported two ways because the game determines what is
    measurable. On the small B3 game exact best-response ``exploitability`` is
    cheap and is the ground-truth hero curve; on the full blueprint game it is
    intractable, so B4 reports proxies instead (``mean_abs_regret`` and the
    average-strategy ``strategy_l1_delta`` between checkpoints). Both are optional
    fields on the same ``metrics`` event; a producer emits whichever it has.
  * The range grid is stored as 169 action-mix rows aligned to a ``hand_names``
    array declared once in ``run_start`` (the 169 canonical hand names, e.g.
    ``"AA"``/``"AKs"``/``"72o"``). The viewer maps names to the 13x13 layout itself,
    so the contract never bakes in the engine's internal preflop-index ordering.

The module is intentionally pure standard library (json + os + time): the trainer
must be able to import it without pulling in numpy/torch, and the viewer server is
stdlib too. Resource sampling lives in :mod:`parciver.resources`, kept separate so
this contract has no platform-specific code.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

# Bump this only with a deliberate, documented migration. The viewer and the
# audit both assert events carry exactly this version.
SCHEMA_VERSION = 1

# The five event types, and the fields each requires beyond the common envelope
# (``v``, ``type``, ``seq``, ``ts``). "Required" means the validator rejects the
# event without it; every other field named in the docstrings is optional and may
# be omitted by a producer that does not have it.
EVENT_TYPES = ("run_start", "metrics", "checkpoint", "milestone", "run_end")

# Number of canonical starting-hand classes (13 pairs + 78 suited + 78 offsuit).
NUM_PREFLOP_HANDS = 169


class SchemaError(ValueError):
    """An event does not conform to the telemetry contract."""


# --------------------------------------------------------------------------
# Validation — the single source of truth for the contract's shape.
# --------------------------------------------------------------------------

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def _is_number(x: Any) -> bool:
    # bool is a subclass of int; a probability/flag mix-up is a real bug, so
    # exclude it from "number".
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_envelope(e: dict) -> None:
    _require(isinstance(e, dict), f"event must be a dict, got {type(e).__name__}")
    _require(e.get("v") == SCHEMA_VERSION, f"event v must be {SCHEMA_VERSION}, got {e.get('v')!r}")
    _require(e.get("type") in EVENT_TYPES, f"unknown event type {e.get('type')!r}")
    _require(isinstance(e.get("seq"), int) and not isinstance(e["seq"], bool) and e["seq"] >= 0,
             f"seq must be a non-negative int, got {e.get('seq')!r}")
    _require(_is_number(e.get("ts")), f"ts must be a number, got {e.get('ts')!r}")


def _check_config(cfg: Any) -> None:
    _require(isinstance(cfg, dict), "run_start.config must be a dict")
    # Generic experiment runs (the R-MAPSS research track: toy games, ablations,
    # exploitation sweeps) opt out of the blueprint-specific range-grid metadata
    # by declaring ``kind == "experiment"``. Everything else is passed through for
    # the status header. Blueprint runs (no ``kind``) keep the strict schema so a
    # range-grid panel always has its 169 hand_names + abstraction sizes.
    if cfg.get("kind") == "experiment":
        return
    names = cfg.get("hand_names")
    _require(
        isinstance(names, list) and len(names) == NUM_PREFLOP_HANDS and all(isinstance(n, str) for n in names),
        f"run_start.config.hand_names must be {NUM_PREFLOP_HANDS} strings",
    )
    abst = cfg.get("abstraction")
    _require(isinstance(abst, dict), "run_start.config.abstraction must be a dict")
    for street in ("preflop", "flop", "turn", "river"):
        _require(
            isinstance(abst.get(street), int) and not isinstance(abst[street], bool) and abst[street] > 0,
            f"run_start.config.abstraction.{street} must be a positive int",
        )


def _check_range_grid(grid: Any) -> None:
    _require(isinstance(grid, dict), "checkpoint.range_grid must be a dict")
    actions = grid.get("actions")
    _require(
        isinstance(actions, list) and len(actions) >= 1 and all(isinstance(a, str) for a in actions),
        "checkpoint.range_grid.actions must be a non-empty list of strings",
    )
    cells = grid.get("cells")
    _require(
        isinstance(cells, list) and len(cells) == NUM_PREFLOP_HANDS,
        f"checkpoint.range_grid.cells must be {NUM_PREFLOP_HANDS} rows",
    )
    width = len(actions)
    for i, row in enumerate(cells):
        _require(
            isinstance(row, list) and len(row) == width and all(_is_number(x) for x in row),
            f"checkpoint.range_grid.cells[{i}] must be {width} numbers",
        )
        s = float(sum(row))
        # An action mix is a probability distribution; allow a small numeric slack
        # and a legitimately-empty (never-visited) cell reported as all zeros.
        _require(
            abs(s - 1.0) < 1e-3 or s == 0.0,
            f"checkpoint.range_grid.cells[{i}] must sum to ~1 (or 0 if unvisited), got {s:.6f}",
        )


def _check_street_map(obj: Any, what: str, *, value_check) -> None:
    _require(isinstance(obj, dict), f"{what} must be a dict keyed by street")
    for k, v in obj.items():
        _require(k in ("P", "F", "T", "R"), f"{what} has unknown street key {k!r} (use P/F/T/R)")
        value_check(v, f"{what}[{k}]")


def validate_event(e: dict) -> None:
    """Raise :class:`SchemaError` unless ``e`` is a valid telemetry event.

    Checks the common envelope then the per-type required fields and the shape of
    the structured payloads (config, range grid, per-street maps). Optional fields,
    when present, are shape-checked too; absent optional fields are fine. This is
    the exact gate :class:`RunWriter` applies before writing, so anything on disk
    has passed it.
    """
    _check_envelope(e)
    t = e["type"]

    if t == "run_start":
        _require(isinstance(e.get("run_id"), str) and e["run_id"], "run_start.run_id must be a non-empty string")
        _require(isinstance(e.get("variant"), str), "run_start.variant must be a string")
        _require(isinstance(e.get("game"), str), "run_start.game must be a string")
        _check_config(e.get("config"))
        ti = e.get("target_iters", None)
        _require(ti is None or (isinstance(ti, int) and not isinstance(ti, bool) and ti > 0),
                 "run_start.target_iters must be a positive int or null")

    elif t == "metrics":
        _require(isinstance(e.get("iteration"), int) and not isinstance(e["iteration"], bool) and e["iteration"] >= 0,
                 "metrics.iteration must be a non-negative int")
        # All of these are optional, but must be numeric when present.
        for f in ("it_per_s", "elapsed_s", "exploitability", "mean_abs_regret",
                  "strategy_l1_delta", "hh_bb_per_100"):
            if f in e and e[f] is not None:
                _require(_is_number(e[f]), f"metrics.{f} must be a number when present")
        if "infosets" in e and e["infosets"] is not None:
            _require(isinstance(e["infosets"], int) and not isinstance(e["infosets"], bool) and e["infosets"] >= 0,
                     "metrics.infosets must be a non-negative int when present")
        if e.get("resources") is not None:
            _check_resources(e["resources"])

    elif t == "checkpoint":
        _require(isinstance(e.get("iteration"), int) and not isinstance(e["iteration"], bool) and e["iteration"] >= 0,
                 "checkpoint.iteration must be a non-negative int")
        if e.get("range_grid") is not None:
            _check_range_grid(e["range_grid"])
        if e.get("action_freq") is not None:
            _check_street_map(
                e["action_freq"], "checkpoint.action_freq",
                value_check=lambda v, w: _require(
                    isinstance(v, dict) and all(_is_number(x) for x in v.values()),
                    f"{w} must map action code -> number"),
            )
        if e.get("bet_size_freq") is not None:
            _check_street_map(
                e["bet_size_freq"], "checkpoint.bet_size_freq",
                value_check=lambda v, w: _require(
                    isinstance(v, dict) and all(_is_number(x) for x in v.values()),
                    f"{w} must map size label -> number"),
            )
        if e.get("infosets_by_street") is not None:
            _check_street_map(
                e["infosets_by_street"], "checkpoint.infosets_by_street",
                value_check=lambda v, w: _require(
                    isinstance(v, int) and not isinstance(v, bool) and v >= 0,
                    f"{w} must be a non-negative int"),
            )

    elif t == "milestone":
        _require(isinstance(e.get("kind"), str) and e["kind"], "milestone.kind must be a non-empty string")
        _require(isinstance(e.get("message"), str), "milestone.message must be a string")
        if "iteration" in e and e["iteration"] is not None:
            _require(isinstance(e["iteration"], int) and not isinstance(e["iteration"], bool) and e["iteration"] >= 0,
                     "milestone.iteration must be a non-negative int when present")

    elif t == "run_end":
        _require(isinstance(e.get("iteration"), int) and not isinstance(e["iteration"], bool) and e["iteration"] >= 0,
                 "run_end.iteration must be a non-negative int")
        _require(e.get("reason") in ("completed", "interrupted", "error"),
                 "run_end.reason must be one of completed/interrupted/error")
        if e.get("summary") is not None:
            _require(isinstance(e["summary"], dict), "run_end.summary must be a dict")


def _check_resources(r: Any) -> None:
    _require(isinstance(r, dict), "resources must be a dict")
    # Every field is optional (a sampler may not have a GPU, etc.) but numeric or
    # null when present.
    for f in ("cpu_pct", "ram_used_gb", "ram_total_gb", "gpu_pct",
              "gpu_mem_used_gb", "gpu_mem_total_gb"):
        if f in r and r[f] is not None:
            _require(_is_number(r[f]), f"resources.{f} must be a number when present")
    if "threads" in r and r["threads"] is not None:
        _require(isinstance(r["threads"], int) and not isinstance(r["threads"], bool) and r["threads"] >= 0,
                 "resources.threads must be a non-negative int when present")


# --------------------------------------------------------------------------
# Writer — the trainer's only interface to the log.
# --------------------------------------------------------------------------

def make_run_id(prefix: str = "run") -> str:
    """A sortable, filesystem-safe run id: ``<prefix>-YYYYmmdd-HHMMSS``."""
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


class RunWriter:
    """Append-only writer for one run's ``events.jsonl``.

    Assigns the monotonic ``seq`` and wall-clock ``ts`` for every event, validates
    it, and appends one JSON line. The file is opened in append mode and each line
    is flushed so a tailing viewer sees it promptly; the cost to the trainer is a
    serialize + buffered write, nothing it can block on.

    Use as a context manager so an unclean exit still records a ``run_end``::

        with RunWriter(run_dir) as w:
            w.start(variant="linear-mccfr", game="nlhe-small", config=cfg)
            ...
            w.metrics(iteration=i, it_per_s=r, exploitability=x)
    """

    def __init__(self, run_dir: str | os.PathLike, *, fsync: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"
        self._fsync = fsync
        self._seq = 0
        self._started = False
        self._ended = False
        self._start_ts: float | None = None
        # Line-buffered text append; newline="" keeps our explicit "\n" exact.
        self._fh = open(self.path, "a", encoding="utf-8", newline="")

    # -- low-level emit ---------------------------------------------------
    def _emit(self, type_: str, payload: dict) -> dict:
        if self._ended:
            raise RuntimeError("RunWriter already emitted run_end; no further events allowed")
        event = {"v": SCHEMA_VERSION, "type": type_, "seq": self._seq, "ts": time.time()}
        # payload never overrides the envelope keys.
        for k in ("v", "type", "seq", "ts"):
            payload.pop(k, None)
        event.update(payload)
        validate_event(event)  # fail loudly at the source
        self._fh.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False))
        self._fh.write("\n")
        self._fh.flush()
        if self._fsync:
            os.fsync(self._fh.fileno())
        self._seq += 1
        return event

    # -- typed events -----------------------------------------------------
    def start(
        self,
        *,
        run_id: str | None = None,
        variant: str,
        game: str,
        config: dict,
        target_iters: int | None = None,
        **extra: Any,
    ) -> dict:
        """Emit the ``run_start`` header (must be first). ``config`` must carry the
        169 ``hand_names`` and the four ``abstraction`` sizes (see
        :func:`_check_config`); anything else in ``config`` is passed through for the
        status header (stack, blinds, bet grid, deck size, ...)."""
        if self._started:
            raise RuntimeError("RunWriter.start already called")
        rid = run_id or self.run_dir.name
        self._start_ts = time.time()
        ev = self._emit("run_start", {
            "run_id": rid, "variant": variant, "game": game,
            "config": config, "target_iters": target_iters, **extra,
        })
        self._started = True
        return ev

    def metrics(self, iteration: int, **fields: Any) -> dict:
        """Emit a lean scalar ``metrics`` event. Common fields: ``it_per_s``,
        ``elapsed_s``, ``exploitability`` (exact, small game), ``mean_abs_regret``
        and ``strategy_l1_delta`` (proxies, big game), ``infosets``,
        ``hh_bb_per_100``, and a ``resources`` dict (see :mod:`parciver.resources`).
        ``elapsed_s`` defaults to time since ``start`` when omitted."""
        if "elapsed_s" not in fields and self._start_ts is not None:
            fields["elapsed_s"] = time.time() - self._start_ts
        return self._emit("metrics", {"iteration": iteration, **fields})

    def checkpoint(self, iteration: int, *, range_grid: dict | None = None,
                   action_freq: dict | None = None, bet_size_freq: dict | None = None,
                   infosets_by_street: dict | None = None, **fields: Any) -> dict:
        """Emit a periodic ``checkpoint`` snapshot: the compact 13x13 ``range_grid``,
        per-street ``action_freq`` / ``bet_size_freq`` distributions, and
        ``infosets_by_street``. ``checkpoint_path`` may point at the full model
        snapshot on disk (the viewer only logs it, never reads it)."""
        return self._emit("checkpoint", {
            "iteration": iteration, "range_grid": range_grid, "action_freq": action_freq,
            "bet_size_freq": bet_size_freq, "infosets_by_street": infosets_by_street, **fields,
        })

    def milestone(self, kind: str, message: str, iteration: int | None = None, **fields: Any) -> dict:
        """Emit a discrete ``milestone`` for the event log (checkpoint saved, phase
        change, note, warning)."""
        return self._emit("milestone", {"kind": kind, "message": message,
                                         "iteration": iteration, **fields})

    def end(self, iteration: int, *, reason: str = "completed",
            summary: dict | None = None, **fields: Any) -> dict:
        """Emit the terminal ``run_end`` — the signal that drives the viewer's
        completion banner. After this, no further events may be emitted."""
        ev = self._emit("run_end", {"iteration": iteration, "reason": reason,
                                     "summary": summary, **fields})
        self._ended = True
        return ev

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # If the body raised before emitting run_end, record why so the viewer's
        # banner reflects reality instead of hanging "in progress" forever.
        try:
            if self._started and not self._ended:
                reason = "error" if exc_type is not None else "interrupted"
                msg = f"{exc_type.__name__}: {exc}" if exc_type is not None else "writer closed without end()"
                self.end(self._last_iteration(), reason=reason, summary={"note": msg})
        finally:
            self.close()

    # Best-effort last-iteration for an abnormal exit; the writer does not track
    # iteration itself, so fall back to 0 (the summary note carries the reason).
    def _last_iteration(self) -> int:
        return 0


# --------------------------------------------------------------------------
# Reader — the viewer's interface to the log.
# --------------------------------------------------------------------------

def read_events(run_dir: str | os.PathLike, *, validate: bool = True) -> list[dict]:
    """Read and parse every complete event line from a run's log (a partial
    trailing line still being written is skipped). Raises on a malformed line when
    ``validate`` is set."""
    path = Path(run_dir)
    if path.is_dir():
        path = path / "events.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.endswith("\n"):
                break  # partial final line: writer mid-append
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if validate:
                validate_event(e)
            out.append(e)
    return out


class RunReader:
    """Tailing reader for one run's ``events.jsonl``.

    :meth:`tail` yields complete events as they are appended, resuming from a byte
    offset so a viewer that reconnects does not miss or re-emit lines. Only lines
    terminated by a newline are yielded, so a half-written final line is never
    parsed — the reader simply waits for the writer to finish it.
    """

    def __init__(self, run_dir: str | os.PathLike) -> None:
        path = Path(run_dir)
        self.path = path / "events.jsonl" if path.is_dir() or not path.suffix else path
        self._pos = 0
        self._buf = ""

    def _drain(self, validate: bool) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            fh.seek(self._pos)
            chunk = fh.read()
            self._pos = fh.tell()
        if not chunk:
            return
        self._buf += chunk
        # Split off all COMPLETE lines in one pass. (The old char-by-char
        # `self._buf = self._buf[nl+1:]` reslice was O(n^2) in the buffer size, so a
        # fresh viewer replaying a large log — tens of MB after a long run — stalled
        # for many seconds before the first frame. `split` is O(n).) The element
        # after the final newline is the partial remainder, kept for next time.
        if "\n" not in self._buf:
            return
        parts = self._buf.split("\n")
        self._buf = parts.pop()  # trailing remainder ("" if the chunk ended on a newline)
        for line in parts:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if validate:
                validate_event(e)
            yield e

    def read_new(self, *, validate: bool = True) -> list[dict]:
        """Return every complete event appended since the last call (or start)."""
        return list(self._drain(validate))

    def tail(self, *, poll: float = 0.1, stop=None, validate: bool = True) -> Iterator[dict]:
        """Yield events forever as they are appended, polling every ``poll`` seconds.

        Stops when ``stop`` (a zero-arg predicate) returns true, if given, or when a
        ``run_end`` event is seen and no ``stop`` was provided. Designed for the SSE
        server loop; the byte-offset resume makes it cheap to poll frequently.
        """
        while True:
            saw_end = False
            for e in self._drain(validate):
                yield e
                if e.get("type") == "run_end":
                    saw_end = True
            if stop is not None:
                if stop():
                    return
            elif saw_end:
                return
            time.sleep(poll)


# --------------------------------------------------------------------------
# Small helpers producers share (kept here so the contract owns them).
# --------------------------------------------------------------------------

def preflop_hand_names() -> list[str]:
    """The 169 canonical starting-hand names in the engine's ``preflop_index``
    order, e.g. ``["AA", "KK", ..., "AKs", ..., "72o"]``. Requires
    ``parciver_engine``; producers put this in ``run_start.config.hand_names`` so
    the viewer can map each range-grid row to the 13x13 layout without knowing the
    engine's internal index order."""
    import parciver_engine as _engine
    return [_engine.preflop_name(i) for i in range(NUM_PREFLOP_HANDS)]


def normalize_mix(row: Iterable[float]) -> list[float]:
    """Normalize an action-mix row to sum 1 (an all-zero row stays all-zero, the
    contract's encoding of an unvisited cell)."""
    row = [float(x) for x in row]
    s = sum(row)
    if s <= 0.0:
        return [0.0] * len(row)
    return [x / s for x in row]
