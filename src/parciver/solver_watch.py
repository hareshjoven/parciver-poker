"""Bookkeeping for runs that play with real-time search (scripts/search_lbr.py,
search_h2h.py, flop_eval.py latency).

The engine treats an exception from a `with_search(solver=...)` solver, or an
answer it cannot use, as a failed search (subgame.rs `solve`): the built-in
solver redoes a turn or river subgame (unless `rescue=False`), the blueprint
decides elsewhere. A match counts the decisions a failed search reached
(`a_search_failed`), not the solver's calls, and a solver that fails every
time reads as "the GPU search is the CPU search" or "search changed
nothing". `Watched` counts the calls and failures a run can check before it
saves a chunk.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time
from pathlib import Path

import numpy as np

# The C functions the binding's conversion calls (PyO3 0.27 `Vec<Vec<f64>>`;
# fresh function objects, so `ctypes.pythonapi`'s own are left as they are).
_is_sequence = ctypes.pythonapi["PySequence_Check"]
_is_sequence.argtypes, _is_sequence.restype = [ctypes.py_object], ctypes.c_int
_as_double = ctypes.pythonapi["PyFloat_AsDouble"]  # raises what it sets
_as_double.argtypes, _as_double.restype = [ctypes.py_object], ctypes.c_double


def _sequence(x) -> bool:
    """What PyO3 extracts into a `Vec`: not a str (`PyUnicode_Check`, which
    reads the type, not `__class__`), and `PySequence_Check`."""
    return not issubclass(type(x), str) and _is_sequence(x) == 1


def _floats(row) -> np.ndarray:
    """The row as the binding reads it: iterated, each entry through
    `PyFloat_AsDouble` (a float, or anything with `__float__` or `__index__`;
    a str is refused). Read at once only where that gives the same numbers: a
    1-D ndarray of booleans, integers or floats (not a subclass: a masked
    array iterates its masked entries as `masked`, which is nan), or a list or
    tuple of floats."""
    t = type(row)
    if t is np.ndarray and row.ndim == 1 and row.dtype.kind in "biuf":
        return row.astype(np.float64, copy=False)
    if (t is list or t is tuple) and all(type(x) is float for x in row):
        return np.array(row, dtype=np.float64)
    return np.array([_as_double(x) for x in row], dtype=np.float64)


def malformed(job: dict, out) -> str | None:
    """Why the engine would refuse `out` as the answer to `job`, or None: the
    binding's conversion to rows of floats (a sequence of sequences, neither a
    str, each entry what `PyFloat_AsDouble` takes), then subgame.rs `solve`'s
    check (one row per wanted slot, the slot's actions times the holdings
    long, every entry finite and >= 0). Never raises for the answer: whatever
    the conversion cannot take is a refusal."""
    try:
        if not _sequence(out):
            return f"malformed answer: a {type(out).__name__}, not a sequence of rows"
        rows = [row for row in out]  # iterated, as the binding does (`list` raises an error of `__len__` it ignores)
        for i, row in enumerate(rows):
            if not _sequence(row):
                return f"malformed answer: row {i} is a {type(row).__name__}, not a sequence"
        rows = [_floats(row) for row in rows]
    except Exception as e:
        return f"malformed answer: not rows of floats ({type(e).__name__}: {e})"
    want, start, n = job["want"], job["child_start"], len(job["ranges"][0])
    if len(rows) != len(want):
        return f"malformed answer: {len(rows)} rows for {len(want)} wanted slots"
    node = {s: i for i, s in enumerate(job["slot"]) if s >= 0}
    for s, vals in zip(want, rows):
        k = start[node[s] + 1] - start[node[s]]
        if len(vals) != k * n:
            return f"malformed answer: slot {s}'s row has {len(vals)} entries, not {k} actions x {n} holdings"
        if not (np.isfinite(vals).all() and (vals >= 0.0).all()):
            return f"malformed answer: slot {s}'s row has a non-finite or negative entry"
    return None


class Watched:
    """A solver as `with_search(solver=...)` takes it, counting its calls and
    failures and timing each call whole, from the engine's side: host
    preparation, the wait for the GPU and the solve. A failure is an exception
    (re-raised: the engine still sees it) or an answer the engine refuses
    (`malformed`; returned as it is: the engine refuses it). The match threads
    call in concurrently, so the counts take a lock. Calls by street: the
    job's board length (3 flop, 4 turn, 5 river)."""

    def __init__(self, solver):
        self.solver = solver
        self.calls = 0
        self.failures = 0
        self.last_error: str | None = None
        self.seconds: dict[int, list[float]] = {}
        self._lock = threading.Lock()

    def __call__(self, job: dict):
        t0 = time.perf_counter()
        try:
            out = self.solver(job)
        except BaseException as e:
            self._count(job, time.perf_counter() - t0, f"{type(e).__name__}: {e}")
            raise
        dt = time.perf_counter() - t0  # the check below is this wrapper's time, not the solver's
        self._count(job, dt, malformed(job, out))
        return out

    def _count(self, job: dict, dt: float, error: str | None) -> None:
        with self._lock:
            self.calls += 1
            if error is not None:
                self.failures += 1
                self.last_error = error
            self.seconds.setdefault(len(job["board"]), []).append(dt)

    def take(self) -> tuple[int, int]:
        """(calls, failures) since the last `take`, for a chunk's check."""
        with self._lock:
            got = self.calls, self.failures
            self.calls = self.failures = 0
        return got

    def check(self, calls: int, failures: int, what: str) -> None:
        """Stop the run (before its chunk is saved) if a search failed."""
        if failures:
            raise SystemExit(f"{what}: {failures} of {calls} search solves failed (the built-in solver or the blueprint decided those); "
                             f"last: {self.last_error}. Nothing of this chunk was saved.")


def persist(tmp: Path, path: Path) -> None:
    """Move the finished file `tmp` over `path` in one step, once its bytes are
    on the disk: neither a stop nor a power cut leaves a torn `path` (Windows
    cannot sync the directory entry: a power cut may leave the old file). Windows
    refuses the replace while another process (an indexer, a virus scan) has
    `path` open: retried for about 3 seconds."""
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    for wait in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, None):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if wait is None:
                raise
            time.sleep(wait)


def write_atomic(path: Path, text: str) -> None:
    """Replace `path` in one step (`persist`): never a torn file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    persist(tmp, path)
