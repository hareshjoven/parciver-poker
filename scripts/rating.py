#!/usr/bin/env python
"""Parciver rating (RATING.md): one number for the bot's strength, from receipts only.

    # once per pool version and engine: anchor round robin, Nash weights, A0's reference rows, acceptance
    .venv/Scripts/python.exe scripts/rating.py calibrate --dir runs/rating/pool-v1 [--threads 8]
    # rate a bot configuration (a blueprint, optionally with real-time search)
    .venv/Scripts/python.exe scripts/rating.py rate --dir runs/rating/pool-v1 --candidate <blueprint.bin> --label v2-2B
    # the live runner's configuration on the GPU path (Amendment 3: weighted anchors, AIVAT, resumable)
    .venv/Scripts/python.exe scripts/rating.py rate --dir runs/rating/pool-v2 --candidate <its blueprint> --label live --search live
    # replay every receipt on the current engine (bit-identical or INVALID); --stale-only: only the STALE ones
    .venv/Scripts/python.exe scripts/rating.py verify --dir runs/rating/pool-v1 [--stale-only]
    # the configuration the author plays (the headline)
    .venv/Scripts/python.exe scripts/rating.py current --dir runs/rating/pool-v1 --label v2-2B
    # recompute every rating from the receipts; table + dashboard (audit: no dashboard, exit 1 on any problem)
    .venv/Scripts/python.exe scripts/rating.py show --dir runs/rating/pool-v1
    .venv/Scripts/python.exe scripts/rating.py audit --dir runs/rating/pool-v1
    # pool-v2's C6 (gating) and C7 (reported) against pool-v1; no pool-v2 rating is VERIFIED without it
    .venv/Scripts/python.exe scripts/rating.py crosscheck --dir runs/rating/pool-v2 --other runs/rating/pool-v1

Nothing is ever typed in: every number shown is recomputed from per-deal arrays whose sha256 is
checked first, and the Nash weights from the calibration's own arrays. A receipt is VERIFIED only once
replayed bit-identically under the current engine and rating code (every new receipt is replayed right
after it is written; RATING.md Amendment 2); STALE until then; INVALID on any mismatch.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parciver import rating as R  # noqa: E402

MAIN = ROOT
POOL_V1 = {
    "version": "pool-v1",
    "stakes": {"sb": 50, "bb": 100, "stack": 20_000},
    "hands_per_deal": 2,
    "abstraction": str(MAIN / "data" / "abstraction"),
    "anchors": [
        {"id": "A0", "kind": "blueprint", "path": str(MAIN / "runs/blueprint-20260914-144924/blueprint.bin"),
         "pin": "9AB151368DA3"},
        {"id": "A1", "kind": "blueprint", "path": str(MAIN / "runs/blueprint-20260927-060745/blueprint.bin"),
         "pin": "DF761C90F9D0"},
        {"id": "A2", "kind": "blueprint", "path": str(MAIN / "runs/blueprint-20260927-141645/blueprint.bin"),
         "pin": "1976A53CF253"},
        {"id": "Ln", "kind": "leak", "of": "A0", "family": "nit", "strength": 0.20},
        {"id": "Ls", "kind": "leak", "of": "A0", "family": "station", "strength": 0.20},
        {"id": "Lm", "kind": "leak", "of": "A0", "family": "maniac", "strength": 0.20},
    ],
    "calibration": {"deals": 1_000_000, "seed_base": 600_000},
    "rating": {"deals": 1_000_000, "seed_base": 700_000},
    "lbr": {"deals": 20_000, "seed": 410_101},
    "chunks": 10,
    "replay": {"chunk": 10_000, "count": 4},
}
# Amendment 2: the same anchors, files and seeds; A0 and blueprint candidates play 20M deals against every
# anchor with Nash weight > 0 (the reference's noise dominated pool-v1's intervals).
POOL_V2 = json.loads(json.dumps(POOL_V1))
POOL_V2["version"] = "pool-v2"
POOL_V2["rating"]["deals_weighted"] = 20_000_000
# The code identity: the files actually imported, read once (CRLF normalized: a checkout's line endings
# must not make every receipt STALE).
CODE = (Path(__file__).resolve(), Path(R.__file__).resolve())
CODE_SHA = hashlib.sha256(b"".join(f.read_bytes().replace(b"\r\n", b"\n") for f in CODE)).hexdigest().upper()
LBR_COUNTS = ("search_failed", "model_search_failed")
ABSTRACTION_FILES = ("flop_buckets.npy", "turn_buckets.npy", "river_buckets.npy")
REFERENCE = "A0-reference"
# Amendment 2's pre-registered C6 (gating): pool-v2 against pool-v1, its sibling directory.
C6_AGAINST = {"pool-v2": "pool-v1"}
C6_DEALS = 1_000_000  # C6 compares the first 1,000,000 deals of every rating row (Amendment 2, pre-registered)
# Amendment 3: the live runner's search, on the GPU path, against the weighted anchors only.
SEARCH_DEALS = {"pool-v2": 20_000}  # deals per weighted anchor, per pool version (no other pool rates search)
SEARCH_CHUNK = 100  # deals per engine call, per partial save and per replay chunk
RUNNER = ROOT / "scripts" / "slumbot_run.py"
DEFAULT_PORT = 8775
PAGE = "/static/rating.html"


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


_SHA = {}


def sha256_file(path) -> str:
    """sha256 (upper hex), memoized per (path, size, mtime) within this process."""
    st = os.stat(path)
    key = (str(Path(path).resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA:
        _SHA[key] = _sha256_file(path)
    return _SHA[key]


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _fsync(path: Path) -> None:
    with open(path, "rb+") as f:
        os.fsync(f.fileno())


def _commit(tmp: Path, path: Path) -> None:
    """Move a finished temp file into place; never overwrites (os.rename refuses an existing target on
    Windows; elsewhere a hard link does)."""
    try:
        if os.name == "nt":
            os.rename(tmp, path)
        else:
            os.link(tmp, path)
            os.unlink(tmp)
    except FileExistsError:
        tmp.unlink(missing_ok=True)
        raise R.RatingError(f"refusing to overwrite {path}") from None


def write_new(path: Path, obj) -> None:
    """Atomic create; never overwrites."""
    if path.exists():
        raise R.RatingError(f"refusing to overwrite {path}")
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    _fsync(tmp)
    _commit(tmp, path)


def write_replace(path: Path, obj) -> None:
    """Atomic replace (for the one mutable pointer, current.json)."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    _fsync(tmp)
    os.replace(tmp, path)


def save_arrays(path: Path, arrays: dict) -> dict:
    """Write per-deal arrays (float64) and return each one's sha256."""
    if path.exists():
        raise R.RatingError(f"refusing to overwrite {path}")
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **{k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()})
    _fsync(tmp)
    _commit(tmp, path)
    return {k: R.array_sha(v) for k, v in arrays.items()}


def load_arrays(path: Path, shas: dict) -> dict:
    """Per-deal arrays, each checked against its recorded sha256."""
    if not path.exists():
        raise R.RatingError(f"missing {path.name}")
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    if set(out) != set(shas):
        raise R.RatingError(f"{path.name}: arrays {sorted(out)} != recorded {sorted(shas)}")
    for k, v in out.items():
        if R.array_sha(v) != shas[k]:
            raise R.RatingError(f"{path.name}: array {k} does not match its recorded sha256")
    return out


def git() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return None
    head, dirty = run("rev-parse", "HEAD"), run("status", "--porcelain")
    return {"commit": head, "dirty": None if dirty is None else bool(dirty)}


def engine():
    import parciver_engine as pe
    return pe


def engine_identity() -> dict:
    """The loaded engine binary and the rating code: a receipt is only trusted under the identity it was
    replayed under."""
    pe = engine()
    f = getattr(getattr(pe, "parciver_engine", None), "__file__", None)
    if not f or Path(f).suffix not in (".pyd", ".so"):
        raise R.RatingError(f"cannot locate the loaded engine binary of {pe.__file__}")
    eng = sha256_file(f)
    return {"path": str(f), "sha256": eng, "code_sha256": CODE_SHA, "id": f"{eng[:16]}+{CODE_SHA[:16]}"}


PLAY_PACKAGES = ("torch", "triton-windows", "triton")  # the GPU stack the search's kernels come from


def play_files() -> list[Path]:
    """Amendment 3 (5): the solver code the live runner imports (src/parciver/**/*.py but the dashboard), the
    runner itself and the net its search reads."""
    base = ROOT / "src" / "parciver"
    code = sorted(f for f in base.rglob("*.py") if "dashboard" not in f.relative_to(base).parts)
    return code + [RUNNER, Path(live_runner().NET)]


def play_env() -> str:
    """The installed versions of PLAY_PACKAGES (Amendment 3 clarification: a GPU stack upgrade makes a search
    receipt STALE, so a replay decides it, never the old verdict)."""
    import importlib.metadata as md
    out = []
    for n in PLAY_PACKAGES:
        try:
            out.append(f"{n}={md.version(n)}")
        except md.PackageNotFoundError:
            out.append(f"{n}=-")
    return ";".join(out)


def play_code_sha() -> str:
    """sha256 of the play code's names and contents (CRLF normalized; the net by its sha256) and of
    play_env(), read now."""
    h = hashlib.sha256()
    for f in play_files():
        name = Path(f).relative_to(ROOT).as_posix().encode()
        body = f.read_bytes().replace(b"\r\n", b"\n") if f.suffix == ".py" else _sha256_file(f).encode()
        h.update(name + b"\0" + body + b"\0")
    h.update(b"env" + b"\0" + play_env().encode() + b"\0")
    return h.hexdigest().upper()


_PLAY = {}


def play_identity() -> str:
    """The play code's sha256 as it is on disk now: re-hashed whenever a play file's size or mtime, or the GPU
    stack, changed (a long-running view sees an edit). A rating and a replay hash fresh at their start and end."""
    files = play_files()
    sig = (tuple((str(f), f.stat().st_size, f.stat().st_mtime_ns) for f in files), play_env())
    if _PLAY.get("sig") != sig:
        _PLAY.update(sig=sig, sha=play_code_sha())
    return _PLAY["sha"]


def is_live(cand: dict) -> bool:
    s = cand.get("search")
    return isinstance(s, dict) and s.get("solver") == "live"


def receipt_identity(r: dict, cur: dict, play: str | None = None) -> str:
    """The identity a receipt's replays count under: the engine + rating code, and for a search receipt the
    play code too (Amendment 3 (5); `play`: a sha256 already read, else play_identity())."""
    return f"{cur['id']}+{(play or play_identity())[:16]}" if is_live(r["candidate"]) else cur["id"]


_RUNNER = {}


def live_runner():
    """scripts/slumbot_run.py as a module (its configuration and `build_agent`), loaded once."""
    if "m" not in _RUNNER:
        import importlib.util
        spec = importlib.util.spec_from_file_location("slumbot_run", RUNNER)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _RUNNER["m"] = m
    return _RUNNER["m"]


def live_search_record(sr, fresh: bool = False) -> dict:
    """Amendment 3 (1): what the live runner's search is (part of the configuration's fingerprint)."""
    net = (_sha256_file if fresh else sha256_file)(sr.NET)
    rec = {"solver": "live", "search": dict(sr.SEARCH), "leaves": sr.LEAVES, "terminals64": sr.TERMINALS64,
           "flop_gamma": sr.FLOP_GAMMA, "net_sha256": net}
    # LIVESEARCH B1 (its Amendment 3): the turn's schedule, only when it is not the default (DCFR gamma 2), so a
    # record of the runner before B1 is unchanged.
    schedule = getattr(sr, "TURN_SCHEDULE", None)
    if schedule and dict(schedule) != {"gamma": 2}:
        rec["turn_schedule"] = dict(schedule)
    return rec


# ---------------------------------------------------------------- pool


def pool_hash(defn: dict) -> str:
    """The pool definition's hash: everything but the hash itself and the creation time."""
    body = {k: v for k, v in defn.items() if k not in ("pool_hash", "created_utc")}
    return hashlib.sha256(canon(body).encode()).hexdigest()


class Pool:
    def __init__(self, d: Path):
        self.dir = d
        self.def_path = d / "pool.json"
        if not self.def_path.exists():
            raise R.RatingError(f"no pool at {d}: run `rating.py calibrate --dir {d}` first")
        self.defn = json.loads(self.def_path.read_text(encoding="utf-8"))
        if self.defn.get("pool_hash") != pool_hash(self.defn):
            raise R.RatingError(f"{self.def_path} was edited after the pool was made (pool_hash mismatch)")
        self.ids = [a["id"] for a in self.defn["anchors"]]
        if len(set(self.ids)) != len(self.ids):
            raise R.RatingError("duplicate anchor ids")
        if len(self.ids) > 10:
            raise R.RatingError("more than 10 anchors would reuse calibration seeds (seed_base + 10 min + max)")
        self.st = self.defn["stakes"]
        self.hpd = self.defn["hands_per_deal"]
        self._bp = {}

    @staticmethod
    def create(d: Path, smoke: bool = False) -> "Pool":
        d.mkdir(parents=True, exist_ok=True)
        if (d / "pool.json").exists():
            pool = Pool(d)
            pool.check_files()
            return pool
        defn = json.loads(json.dumps(POOL_V2))
        if smoke:
            # A tiny pool to test the machinery end to end; never a rating of record.
            defn.update(version="pool-SMOKE", calibration={"deals": 20_000, "seed_base": 600_000},
                        rating={"deals": 20_000, "deals_weighted": 40_000, "seed_base": 700_000},
                        lbr={"deals": 400, "seed": 410_101}, replay={"chunk": 1_000, "count": 4})
        for a in defn["anchors"]:
            if a["kind"] == "blueprint":
                sha = sha256_file(a["path"])
                if not sha.startswith(a["pin"]):
                    raise R.RatingError(f"{a['id']}: {a['path']} sha256 {sha[:12]} != pinned {a['pin']}")
                a["sha256"] = sha
        defn["abstraction_sha256"] = {n: sha256_file(Path(defn["abstraction"]) / n) for n in ABSTRACTION_FILES}
        defn["pool_hash"] = pool_hash(defn)
        defn["created_utc"] = utc()
        write_new(d / "pool.json", defn)
        return Pool(d)

    def check_files(self, fresh: bool = False) -> None:
        """The pool's frozen files are unchanged (`fresh`: hashed now, never a memo of an earlier hash)."""
        h = _sha256_file if fresh else sha256_file
        for a in self.defn["anchors"]:
            if a["kind"] == "blueprint" and h(a["path"]) != a["sha256"]:
                raise R.RatingError(f"anchor {a['id']} file changed since the pool was made: {a['path']}")
        for n, s in self.defn["abstraction_sha256"].items():
            if h(Path(self.defn["abstraction"]) / n) != s:
                raise R.RatingError(f"abstraction file {n} changed since the pool was made")

    def anchor(self, aid: str) -> dict:
        return next(x for x in self.defn["anchors"] if x["id"] == aid)

    def seed(self, j: str) -> int:
        """The rating seed against anchor j (the same for every candidate and the reference)."""
        return self.defn["rating"]["seed_base"] + self.ids.index(j)

    def cal_seed(self, i: int, j: int) -> int:
        return self.defn["calibration"]["seed_base"] + 10 * min(i, j) + max(i, j)

    def rating_deals(self, j: str, weights: dict) -> int:
        """Deals against anchor j: Amendment 2 (6), more against the anchors that carry weight."""
        r = self.defn["rating"]
        return r.get("deals_weighted", r["deals"]) if weights.get(j, 0.0) > 0 else r["deals"]

    def blueprint(self, path: str):
        if path not in self._bp:
            self._bp[path] = engine().SavedBlueprint(self.defn["abstraction"], path)
        return self._bp[path]

    def agent(self, aid: str):
        a = self.anchor(aid)
        if a["kind"] == "blueprint":
            return self.blueprint(a["path"]).play_agent()
        base = self.anchor(a["of"])
        return self.blueprint(base["path"]).s32_policy().leak(a["family"], a["strength"]).play_agent()

    def stakes(self):
        return self.st["sb"], self.st["bb"], self.st["stack"]


def n_chunks(pool: Pool, deals: int) -> int:
    """Engine calls per match (at most 1M deals each; results do not depend on it)."""
    n = max(pool.defn["chunks"], -(-deals // 1_000_000))
    return n if deals % n == 0 else 1


def play(pool: Pool, me, other, seed: int, deals: int, threads: int, first_deal: int = 0, chunks: int | None = None,
         label: str = "", progress=None) -> dict:
    """`me` against `other` on deals first_deal.. of `seed`, in chunks (identical to one call)."""
    n = chunks or n_chunks(pool, deals)
    if deals % n:
        n = 1
    size = deals // n
    per, failed_a, failed_b, sources = [], None, None, None
    t0 = time.time()
    for k in range(n):
        r = me.match_against("agent", size, *pool.stakes(), other=other, seed=seed, threads=threads,
                             first_deal=first_deal + k * size)
        per.append(np.asarray(r["per_deal_bb"], dtype=np.float64))
        for key in ("a_search_failed", "b_search_failed"):
            if key not in r:
                raise R.RatingError(f"the engine did not report {key}: a rating needs failed-search counts")
        failed_a = [x + y for x, y in zip(failed_a, r["a_search_failed"])] if failed_a is not None else list(r["a_search_failed"])
        failed_b = [x + y for x, y in zip(failed_b, r["b_search_failed"])] if failed_b is not None else list(r["b_search_failed"])
        sources = [x + y for x, y in zip(sources, r["a_sources"])] if sources is not None else list(r["a_sources"])
        if progress:
            progress((k + 1) * size)
    per = R.per_hand_mbb(np.concatenate(per), pool.hpd)
    if label:
        print(f"    {label}: {deals:,} deals  {per.mean():+9.2f} mbb/hand  ({time.time() - t0:.1f}s)", flush=True)
    return dict(per=per, a_search_failed=failed_a, b_search_failed=failed_b, a_sources=sources,
                seconds=round(time.time() - t0, 1))


def lbr(pool: Pool, agent, deals: int, seed: int, threads: int, chunks: int, first_deal: int = 0) -> dict:
    """Local best response on deals first_deal.. of `seed`. The preflop split is kept only when the engine
    reports it for every chunk (never filled in); failed-search counts are summed when reported."""
    n = chunks if chunks > 1 and deals % chunks == 0 else 1
    size = deals // n
    per, pre, split, failed = [], [], True, {}
    t0 = time.time()
    for k in range(n):
        r = agent.lbr(size, *pool.stakes(), seed=seed, threads=threads, first_deal=first_deal + k * size)
        per.extend(r["per_deal_bb"])
        if "per_deal_preflop_bb" in r:
            pre.extend(r["per_deal_preflop_bb"])
        else:
            split = False
        for key in LBR_COUNTS:
            if key in r:
                failed[key] = [x + y for x, y in zip(failed[key], r[key])] if key in failed else list(r[key])
    out = dict(per=R.per_hand_mbb(per, pool.hpd), failed=failed, seconds=round(time.time() - t0, 1))
    if split:
        out["pre"] = R.per_hand_mbb(pre, pool.hpd)
    return out


# ---------------------------------------------------------------- receipts


class Receipt(dict):
    """A receipt as parsed. `sha` is the sha256 of the exact bytes it was parsed from: a replay's verdict is
    recorded against them, never against the file read again later (it may have changed during the replay)."""
    sha: str


def read_receipts(pool: Pool) -> tuple[list[tuple[Path, Receipt]], list[tuple[Path, str]]]:
    """(this pool's receipts, oldest first; [(path, why)] for files that cannot be read as one of this pool's
    receipts: unreadable, or made for another pool definition)."""
    good, bad = [], []
    d = pool.dir / "receipts"
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            raw = p.read_bytes()
            r = json.loads(raw.decode("utf-8"))
            if not isinstance(r, dict) or not isinstance(r.get("candidate"), dict):
                raise ValueError("not a receipt")
        except (OSError, ValueError) as e:
            bad.append((p, f"unreadable receipt: {e}"))
            continue
        r = Receipt(r)
        r.sha = hashlib.sha256(raw).hexdigest()
        if r.get("pool_hash") == pool.defn["pool_hash"]:
            good.append((p, r))
        else:
            bad.append((p, f"made for another pool definition (pool_hash {str(r.get('pool_hash'))[:12]})"))
    good.sort(key=lambda pr: (str(pr[1].get("started_utc", "")), pr[0].name))
    return good, bad


def receipts(pool: Pool) -> list[tuple[Path, Receipt]]:
    return read_receipts(pool)[0]


def receipt_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verifications(pool: Pool) -> tuple[list[dict], list[int]]:
    """verify.jsonl's entries and the numbers of unreadable lines (any one makes nothing VERIFIED: the lost
    line could have recorded a failed replay)."""
    f = pool.dir / "verify.jsonl"
    if not f.exists():
        return [], []
    entries, bad = [], []
    for n, raw in enumerate(f.read_bytes().split(b"\n"), 1):
        try:
            x = raw.decode("utf-8")
            if not x.strip():
                continue
            e = json.loads(x)
            # ok: True (identical), False (a mismatch), None (the replay was refused: nothing was compared).
            if (not isinstance(e, dict) or not isinstance(e.get("receipt_sha"), str)
                    or not isinstance(e.get("identity"), str) or "ok" not in e
                    or not (e["ok"] is None or type(e["ok"]) is bool)):
                raise ValueError
            entries.append(e)
        except ValueError:  # includes UnicodeDecodeError
            bad.append(n)
    return entries, bad


@contextlib.contextmanager
def locked(path: Path, wait: float = 600):
    """An exclusive lock on `path`.lock across processes (a `rate`'s replay process and a `verify` may append
    to verify.jsonl at once); the OS releases it if the process dies. Waits up to `wait` seconds (0: refuses at
    once if another process holds it)."""
    with open(path.with_name(path.name + ".lock"), "a+b") as f:
        if os.name == "nt":
            import msvcrt
            deadline = time.time() + wait
            while True:
                f.seek(0)
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.time() >= deadline:
                        raise R.RatingError(f"{path.name} is locked by another process (waited {wait:.0f} s)") from None
                    time.sleep(0.05)
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | (fcntl.LOCK_NB if wait == 0 else 0))
            except BlockingIOError:
                raise R.RatingError(f"{path.name} is locked by another process") from None
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_verification(pool: Pool, entry: dict) -> None:
    f = pool.dir / "verify.jsonl"
    line = (json.dumps(entry) + "\n").encode("utf-8")
    with locked(f), open(f, "ab") as out:
        out.write(line)
        out.flush()
        os.fsync(out.fileno())


RANK = {"VERIFIED": 0, "STALE": 1, "INVALID": 2}
SCHEMAS = ("parciver-rating-receipt-1", "parciver-rating-receipt-2")


def _counts(v) -> bool:
    return isinstance(v, list) and all(type(x) is int for x in v)


def check_receipt(pool: Pool, r: dict, arrays: dict) -> None:
    """Amendment 2 (3): the receipt agrees with the pool it claims and with its own arrays, and has the shape
    every later step reads (a malformed receipt is an INVALID row, never a crash)."""
    if r.get("schema") not in SCHEMAS:
        raise R.RatingError(f"unknown receipt schema {r.get('schema')}")
    rows, lb = r.get("anchors"), r.get("lbr")
    if not isinstance(rows, dict) or not rows:
        raise R.RatingError("no anchor rows")
    if lb is not None and not isinstance(lb, dict):
        raise R.RatingError("malformed LBR record")
    if not isinstance(r.get("arrays"), dict):
        raise R.RatingError("malformed array list")
    eng = r.get("engine")
    if not isinstance(eng, dict) or not isinstance(eng.get("id"), str):
        raise R.RatingError("no engine identity")
    label = r["candidate"].get("label")
    if not isinstance(label, str) or not label:
        raise R.RatingError(f"label {label!r} is not a name")
    want = set(rows)
    if lb:
        want |= {"LBR"} | ({"LBR_preflop"} if "LBR_preflop" in r["arrays"] else set())
    if is_live(r["candidate"]):  # Amendment 3 (3): the raw row beside each AIVAT row
        if lb:
            raise R.RatingError("a search receipt has no LBR guard (Amendment 3 (7))")
        if r.get("threads") != 1 or r.get("chunk") != SEARCH_CHUNK or r.get("aivat") is not True:
            raise R.RatingError(f"a search receipt plays 1 thread, chunks of {SEARCH_CHUNK}, AIVAT (Amendment 3 (2))")
        for j, row in rows.items():
            fails = [row.get("a_search_failed"), row.get("b_search_failed")] if isinstance(row, dict) else [None]
            if not all(isinstance(f, list) and f and not any(f) for f in fails):
                raise R.RatingError(f"row {j}: a search receipt's failed-search counts are all 0 (Amendment 3 (4))")
        want |= {f"{j}.raw" for j in rows}
    if set(r["arrays"]) != want:
        raise R.RatingError(f"arrays {sorted(r['arrays'])} != the receipt's rows {sorted(want)}")
    for k, v in arrays.items():
        if v.ndim != 1:
            raise R.RatingError(f"array {k} is not one value per deal")
    for j, row in rows.items():
        if j not in pool.ids:
            raise R.RatingError(f"a row against {j}, which is not a pool anchor")
        if not isinstance(row, dict):
            raise R.RatingError(f"row {j} is not a record")
        if row.get("seed") != pool.seed(j):
            raise R.RatingError(f"row {j}: seed {row.get('seed')} != the pool's {pool.seed(j)}")
        if row.get("deals") != len(arrays[j]):
            raise R.RatingError(f"row {j}: {row.get('deals')} deals recorded, {len(arrays[j])} in the array")
        if f"{j}.raw" in arrays and len(arrays[f"{j}.raw"]) != len(arrays[j]):
            raise R.RatingError(f"row {j}: the raw row's length differs from the AIVAT row's")
        if not _counts(row.get("a_search_failed")) or not _counts(row.get("b_search_failed")):
            raise R.RatingError(f"no failed-search counts against {j}")
    if lb:
        if lb.get("seed") != pool.defn["lbr"]["seed"]:
            raise R.RatingError(f"LBR seed {lb.get('seed')} != the pool's {pool.defn['lbr']['seed']}")
        if lb.get("deals") != len(arrays["LBR"]):
            raise R.RatingError(f"LBR: {lb.get('deals')} deals recorded, {len(arrays['LBR'])} in the array")


def check_candidate(pool: Pool, c: dict) -> None:
    """The candidate's file is unchanged; reserved labels (the reference, the anchor ids) only name the pool's
    own anchors (Amendment 2 (4))."""
    if c.get("path"):
        if not Path(c["path"]).exists():
            raise R.RatingError(f"candidate file {c['path']} is gone")
        if sha256_file(c["path"]) != c.get("sha256"):
            raise R.RatingError(f"candidate file {c['path']} changed since the receipt")
    label, aid = c.get("label"), c.get("anchor")
    if aid is None and (label == REFERENCE or label in pool.ids):
        raise R.RatingError(f"label {label} is reserved for the pool's own anchors")
    if aid is not None:
        if aid not in pool.ids or label not in (aid, REFERENCE) or (label == REFERENCE and aid != "A0"):
            raise R.RatingError(f"label {label} with anchor {aid}: not a pool anchor receipt")
        a = pool.anchor(aid)
        if (c.get("path") != a.get("path") or c.get("sha256") != a.get("sha256") or c.get("search")
                or c.get("translation", "snap") != "snap"):
            raise R.RatingError(f"{label}: not the pool's anchor {aid} as defined in pool.json")


def state_of(pool: Pool, path: Path, r: Receipt, ver: tuple[list[dict], list[int]], cur: dict) -> tuple[str, str, dict | None]:
    """(VERIFIED | STALE | INVALID, reason, per-deal arrays or None). VERIFIED needs a passing replay under the
    current identity and no failing one (Amendment 2 (1))."""
    try:
        arrays = load_arrays(path.with_suffix(".npz"), r["arrays"])
        check_receipt(pool, r, arrays)
        check_candidate(pool, r["candidate"])
        sha = r.sha
    except MemoryError:
        raise
    except Exception as e:
        return "INVALID", (str(e) if isinstance(e, R.RatingError) else f"malformed receipt ({type(e).__name__}: {e})"), None
    entries, bad_lines = ver
    try:
        rid = receipt_identity(r, cur)
    except MemoryError:
        raise
    except Exception as e:  # e.g. the runner or a module it imports does not load right now
        return "STALE", f"the play code cannot be read: {type(e).__name__}: {e}", arrays
    mine = [v for v in entries if v["receipt_sha"] == sha and v["identity"] == rid]
    failed = [v for v in mine if v["ok"] is False]
    passed = [v for v in mine if v["ok"] is True]  # a refused replay (ok None) compared nothing
    if failed:
        return "INVALID", f"replay mismatch under the current engine+code ({failed[-1].get('utc')}): {failed[-1].get('detail')}", arrays
    if bad_lines:
        return "STALE", f"verify.jsonl has unreadable line(s) {bad_lines}: no replay can be trusted", arrays
    if passed:
        return "VERIFIED", f"replayed bit-identically under the current engine+code ({passed[-1].get('utc')})", arrays
    if mine:
        return "STALE", f"its replay was refused under the current engine+code: {mine[-1].get('detail')}", arrays
    return "STALE", f"not replayed under the current engine+code {rid}: run `rating.py verify`", arrays


def calibration(pool: Pool) -> dict:
    f = pool.dir / "calibration.json"
    if not f.exists():
        raise R.RatingError("no calibration: run `rating.py calibrate` first")
    cal = json.loads(f.read_text(encoding="utf-8"))
    if not isinstance(cal, dict):
        raise R.RatingError("calibration.json is not a calibration")
    return cal


def load_pair(pool: Pool, i: int, j: int) -> tuple[np.ndarray, dict]:
    """A calibration match's per-deal array (sha-checked) and its metadata, checked against the pool."""
    stem = f"pair-{pool.ids[i]}-{pool.ids[j]}"
    meta = json.loads((pool.dir / "cal" / f"{stem}.json").read_text(encoding="utf-8"))
    per = load_arrays(pool.dir / "cal" / f"{stem}.npz", meta["arrays"])["per"]
    if meta["pair"] != [pool.ids[i], pool.ids[j]] or meta["seed"] != pool.cal_seed(i, j):
        raise R.RatingError(f"{stem}: pair or seed differs from the pool's")
    if meta["deals"] != len(per) or len(per) != pool.defn["calibration"]["deals"]:
        raise R.RatingError(f"{stem}: {len(per)} deals, the pool's calibration plays {pool.defn['calibration']['deals']}")
    return per, meta


def check_calibration(pool: Pool) -> tuple[dict, dict, dict]:
    """Amendment 2 (2): calibration.json re-derived from its own arrays. Returns (calibration, weights, the
    re-derived numbers and checks) or raises: A and its standard errors rebuilt from the sha-checked arrays
    equal the stored ones, nash(A) equals the stored p and passes C3, and every swapped pair negates its
    original deal by deal (C4, exact and gating). Nothing the page shows is taken from the file unchecked."""
    cal = calibration(pool)
    if cal.get("pool_hash") != pool.defn["pool_hash"] or cal.get("anchors") != pool.ids:
        raise R.RatingError("calibration.json belongs to another pool definition")
    if cal.get("state") != "accepted":
        raise R.RatingError(f"calibration state is {cal.get('state')}")
    k = len(pool.ids)
    A, se, c4 = np.zeros((k, k)), np.zeros((k, k)), {}
    for i in range(k):
        for j in range(i + 1, k):
            per = load_pair(pool, i, j)[0]
            m = R.mean_ci(per)
            A[i, j], A[j, i] = m["mean"], -m["mean"]
            se[i, j] = se[j, i] = m["se"]
            c4[f"{pool.ids[i]}-{pool.ids[j]}"] = bool(np.array_equal(load_pair(pool, j, i)[0], -per))
            del per
    if not np.array_equal(A, np.asarray(cal["A_mbb_per_hand"], dtype=np.float64)):
        raise R.RatingError("calibration.json's payoff matrix differs from its arrays")
    if not np.array_equal(se, np.asarray(cal["se"], dtype=np.float64)):
        raise R.RatingError("calibration.json's standard errors differ from its arrays")
    if not all(c4.values()):
        raise R.RatingError(f"C4: swapped pair(s) {[x for x, ok in c4.items() if not ok]} do not negate their "
                            "originals deal by deal")
    p = R.nash(A)
    stored = np.asarray(cal["p"], dtype=np.float64)
    if stored.shape != p.shape or np.abs(p - stored).max() > 1e-12:
        raise R.RatingError("calibration.json's Nash weights differ from nash(A) recomputed")
    chk = R.check_nash(A, stored)
    if not chk <= 1e-9:
        raise R.RatingError(f"C3: the stored weights are not an equilibrium of A (max(Ap)/max|A| = {chk:.3g})")
    der = {"A": A, "se": se, "nash_check": chk,
           "C3_nash": {"ok": True, "max_Ap_over_maxA": chk},
           "C4_role_symmetry": {"ok": True, "exact_pairs": len(c4), "pairs": len(c4), "detail": c4}}
    return cal, {j: float(x) for j, x in zip(pool.ids, stored)}, der


def seed_consistency(pool: Pool, ref_arr: dict, weights: dict, A: np.ndarray, se: np.ndarray) -> dict:
    """C2: A0's mean against each weighted anchor on the rating seeds agrees with the calibration's (within 3
    combined SE; against A0 itself, 0 within 3 SE)."""
    out = {}
    for jj, j in enumerate(pool.ids):
        if weights[j] <= 0:
            continue
        mr = R.mean_ci(ref_arr[j])
        if j == "A0":
            out[j] = {"ok": bool(abs(mr["mean"]) <= 3 * mr["se"]), "rating_seeds": mr["mean"], "expected": 0.0,
                      "se": mr["se"]}
        else:
            d = mr["mean"] - A[0, jj]
            s = (mr["se"] ** 2 + se[0, jj] ** 2) ** 0.5
            out[j] = {"ok": bool(abs(d) <= 3 * s), "rating_seeds": mr["mean"], "calibration_seeds": float(A[0, jj]),
                      "combined_se": s}
    return {"ok": all(v["ok"] for v in out.values()), "anchors": out}


def protocol_problem(pool: Pool, r: dict, weights: dict) -> str | None:
    """Why this receipt's deal counts are not the pool's protocol (Amendment 2 (6); a search receipt:
    Amendment 3 (2)), or None."""
    if is_live(r["candidate"]):
        use = [j for j in pool.ids if weights.get(j, 0.0) > 0]
        d = SEARCH_DEALS.get(pool.defn["version"])
        if d is None:
            return f"{pool.defn['version']} rates no search configuration"
        if set(r["anchors"]) != set(use):
            return f"a search receipt plays exactly the weighted anchors {use}"
        for j in use:
            if r["anchors"][j].get("deals") != d:
                return f"{r['anchors'][j].get('deals')} deals against {j}, the protocol is {d}"
        return None
    if set(r["anchors"]) != set(pool.ids):
        return "it does not play every anchor"
    for j in pool.ids:
        if r["anchors"][j].get("deals") != pool.rating_deals(j, weights):
            return f"{r['anchors'][j].get('deals')} deals against {j}, the pool's protocol is {pool.rating_deals(j, weights)}"
    return None


def reference_problem(pool: Pool, r: dict, weights: dict) -> str | None:
    """Why this receipt cannot be the 1500 reference (Amendment 2 (4)), or None."""
    c, a0 = r["candidate"], pool.anchor("A0")
    if (c.get("label") != REFERENCE or c.get("anchor") != "A0" or c.get("path") != a0.get("path")
            or c.get("sha256") != a0.get("sha256") or c.get("search") or c.get("translation", "snap") != "snap"):
        return "not the pool's A0 under the reserved label"
    return protocol_problem(pool, r, weights)


# ---------------------------------------------------------------- candidates


def config_of(cand: dict) -> dict:
    """What a configuration is (Amendment 2 (5)): the blueprint's content, its search and its translation; not
    its label or file path. Leak anchors have no file of their own and are named by their pool id."""
    if not cand.get("sha256"):
        return {"anchor": cand.get("anchor")}
    return {"blueprint_sha256": cand["sha256"], "search": cand.get("search"),
            "translation": cand.get("translation", "snap")}


def fingerprint(cand: dict) -> str:
    return hashlib.sha256(canon(config_of(cand)).encode()).hexdigest()[:16]


def candidate_agent(pool: Pool, cand: dict):
    """The agent a receipt's candidate describes (also used to replay it)."""
    if cand.get("anchor"):
        return pool.agent(cand["anchor"])
    if is_live(cand):
        return live_agent(pool, cand)[0]
    if cand.get("search") is not None:
        raise R.RatingError("search candidates other than the live runner's are refused (RATING.md Amendment 2 "
                            "(8), Amendment 3)")
    agent = pool.blueprint(cand["path"]).play_agent()
    if cand.get("translation", "snap") != "snap":
        agent = agent.with_translation(cand["translation"])
    return agent


def live_agent(pool: Pool, cand: dict):
    """(agent, solver) of a live candidate: the runner's `build_agent`, refused unless the runner plays
    exactly the candidate's blueprint (its sha256), search record and the pool's abstraction and stakes."""
    sr = live_runner()
    if Path(sr.BLUEPRINT).resolve() != Path(cand["path"]).resolve():
        raise R.RatingError(f"the live runner plays {sr.BLUEPRINT}, not the candidate's {cand['path']}")
    if Path(sr.ABSTRACTION).resolve() != Path(pool.defn["abstraction"]).resolve():
        raise R.RatingError("the live runner's abstraction is not the pool's")
    if (sr.S.SB, sr.S.BB, sr.S.STACK) != pool.stakes():
        raise R.RatingError("the live runner's stakes are not the pool's")
    if live_search_record(sr, fresh=True) != cand["search"]:
        raise R.RatingError("the live runner's search (settings or net) is not the candidate's")
    if _sha256_file(cand["path"]) != cand["sha256"]:
        raise R.RatingError(f"candidate file {cand['path']} is not the receipt's")
    try:
        agent, solver, _ = sr.build_agent()
    except SystemExit as e:
        raise R.RatingError(f"the live runner refused to build its agent: {e}") from None
    return agent, solver


def describe_live(cand: dict) -> str:
    s = cand["search"]
    return (f"blueprint + live search ({s['leaves']} flop leaves, flop/turn/river {s['search'].get('flop_iters')}/"
            f"{s['search'].get('turn_iters')}/{s['search'].get('river_iters')}, GPU) + AIVAT")


def play_live(pool: Pool, agent, solver, other, seed: int, first: int, deals: int, note: dict) -> dict:
    """Amendment 3 (2)-(4): deals first..first+deals-1 in one engine call (1 thread, AIVAT). A failed search,
    or a chunk without a flop search, refuses the chunk (logged to refused.jsonl with `note`: whose rating or
    replay): nothing of it is kept."""
    solver.take()
    for v in solver.seconds.values():
        v.clear()
    r = agent.match_against("agent", deals, *pool.stakes(), other=other, seed=seed, threads=1, first_deal=first,
                            aivat=True)
    calls, failures = solver.take()
    for key in ("a_search_failed", "b_search_failed", "per_deal_bb_aivat", "per_deal_bb"):
        if key not in r:
            raise R.RatingError(f"the engine did not report {key}")
    if len(r["per_deal_bb_aivat"]) != deals or len(r["per_deal_bb"]) != deals:
        raise R.RatingError(f"the engine returned {len(r['per_deal_bb_aivat'])} values for {deals} deals")
    why = None
    if failures:
        why = f"{failures} of {calls} search solves failed (last: {solver.last_error})"
    elif any(r["a_search_failed"]) or any(r["b_search_failed"]):
        why = f"decisions reached a failed search ({r['a_search_failed']}, {r['b_search_failed']})"
    elif not solver.seconds.get(3):
        why = "no flop search reached the solver"
    if why:
        f = pool.dir / "refused.jsonl"
        with locked(f), open(f, "a", encoding="utf-8") as out:
            out.write(json.dumps({**note, "seed": seed, "first": first, "deals": deals, "why": why,
                                  "utc": utc()}) + "\n")
        raise R.RatingError(f"deals {first}..{first + deals - 1} of seed {seed}: {why}; the chunk is not kept "
                            "(Amendment 3 (4))")
    return dict(x=R.per_hand_mbb(r["per_deal_bb_aivat"], pool.hpd), raw=R.per_hand_mbb(r["per_deal_bb"], pool.hpd),
                a_search_failed=list(r["a_search_failed"]), b_search_failed=list(r["b_search_failed"]),
                a_sources=list(r["a_sources"]), flop=len(solver.seconds[3]), mass=float(r["aivat_mass_error"]))


def describe(cand: dict) -> str:
    if cand.get("anchor"):
        return f"pool anchor {cand['anchor']}"
    if is_live(cand):
        return describe_live(cand)
    if cand.get("translation", "snap") != "snap":
        return f"blueprint + translation ({cand['translation']}) + fallback (play.rs)"
    return "blueprint + translation + fallback (play.rs)"


def run_rating(pool: Pool, cand: dict, threads: int, weights: dict, lbr_deals: int, live=None,
               git_info: dict | None = None) -> Path:
    """Play the candidate against every anchor on the pool's seeds (protocol deal counts), write the receipt,
    then replay it at once (Amendment 2 (1))."""
    label = cand["label"]
    started = utc()
    ident = engine_identity()
    agent = candidate_agent(pool, cand)
    arrays, rows = {}, {}
    for j in pool.ids:
        deals, seed = pool.rating_deals(j, weights), pool.seed(j)
        what = f"{label} vs {j}"
        prog = (lambda done, what=what, deals=deals: live.progress(what, done, deals)) if live else None
        if prog:
            prog(0)
        res = play(pool, agent, pool.agent(j), seed, deals, threads, label=what, progress=prog)
        arrays[j] = res["per"]
        rows[j] = {k: res[k] for k in ("a_search_failed", "b_search_failed", "a_sources", "seconds")}
        rows[j].update(seed=seed, deals=deals, mean=float(res["per"].mean()))
    lb = None
    if lbr_deals:
        lc = pool.defn["lbr"]
        if live:
            live.progress(f"{label} LBR", 0, lbr_deals)
        res = lbr(pool, agent, lbr_deals, lc["seed"], threads, pool.defn["chunks"])
        arrays["LBR"] = res["per"]
        if "pre" in res:
            arrays["LBR_preflop"] = res["pre"]
        lb = {"deals": lbr_deals, "seed": lc["seed"], "seconds": res["seconds"], "preflop_split": "pre" in res,
              **res["failed"]}
        if live:
            live.progress(f"{label} LBR", lbr_deals, lbr_deals)
    if engine_identity()["id"] != ident["id"]:
        raise R.RatingError("the engine or rating code changed during the rating; result discarded")
    if cand.get("path") and sha256_file(cand["path"]) != cand["sha256"]:
        raise R.RatingError("the candidate file changed during the rating; result discarded")
    (pool.dir / "receipts").mkdir(exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
    stem = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S%fZ}-{safe}"
    shas = save_arrays(pool.dir / "receipts" / f"{stem}.npz", arrays)
    receipt = {
        "schema": "parciver-rating-receipt-2", "pool": pool.defn["version"], "pool_hash": pool.defn["pool_hash"],
        "candidate": cand, "fingerprint": fingerprint(cand), "agent": describe(cand), "anchors": rows, "lbr": lb,
        "arrays": shas, "weights": weights, "engine": ident, "git": git_info or git(), "threads": threads,
        "started_utc": started, "finished_utc": utc(),
    }
    path = pool.dir / "receipts" / f"{stem}.json"
    write_new(path, receipt)
    del arrays, agent, res
    if live:
        live.progress(f"{label}: replaying the receipt", 0, 1)
    replay_in_new_process(pool, path, threads)
    return path


def run_live_rating(pool: Pool, cand: dict, weights: dict, live=None, git_info: dict | None = None,
                    play0: str | None = None) -> Path:
    """Amendment 3: the live configuration against every weighted anchor, SEARCH_DEALS each in chunks of
    SEARCH_CHUNK. After every chunk the rating so far is saved to partial/<fingerprint>-<label>.npz, and a
    later run resumes from it only under the identical identity and protocol (never overwrites another). Then
    the receipt, replayed at once in a fresh process. `play0`: the play code's sha256 read before the runner
    was imported (else read now). One run per partial file at a time."""
    label = cand["label"]
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
    part = pool.dir / "partial" / f"{fingerprint(cand)}-{safe}.npz"
    part.parent.mkdir(exist_ok=True)
    with locked(part, wait=0):
        return _run_live_rating(pool, cand, weights, live, git_info, play0 or play_code_sha(), label, safe, part)


def _run_live_rating(pool, cand, weights, live, git_info, play0, label, safe, part) -> Path:
    use = [j for j in pool.ids if weights.get(j, 0.0) > 0]
    d = SEARCH_DEALS.get(pool.defn["version"])
    if d is None:
        raise R.RatingError(f"{pool.defn['version']} rates no search configuration (Amendment 3 (2))")
    ident = engine_identity()
    rid = f"{ident['id']}+{play0[:16]}"
    key = canon({"candidate": cand, "identity": rid, "pool_hash": pool.defn["pool_hash"], "anchors": use,
                 "seeds": {j: pool.seed(j) for j in use}, "deals": d, "chunk": SEARCH_CHUNK})
    arrays = {}
    if part.exists():
        with np.load(part) as z:
            if str(z["key"]) != key:
                raise R.RatingError(f"{part} holds a rating under another identity or protocol; it is never "
                                    "overwritten: move it aside to start again")
            meta = json.loads(str(z["meta"]))
            arrays = {k: z[k] for k in z.files if k not in ("key", "meta")}
        print(f"  resuming {part.name}: " + ", ".join(f"{j} {len(arrays.get(j, [])):,}" for j in use), flush=True)
        shas = {k: R.array_sha(v) for k, v in arrays.items()}
        for q, r in receipts(pool):  # complete, and its receipt written before the partial was moved aside
            if fingerprint(r["candidate"]) == fingerprint(cand) and r.get("arrays") == shas:
                os.replace(part, part.with_name(f"{part.stem}.done-{q.stem}.npz"))
                print(f"  {part.name} is {q.name}'s rating: not written again; replaying it", flush=True)
                replay_in_new_process(pool, q, 1)
                return q
    else:
        meta = {"started_utc": utc(), "git": git_info or git(),
                "rows": {j: {"a_search_failed": None, "b_search_failed": None, "a_sources": None, "seconds": 0.0,
                             "flop_solves": 0, "aivat_mass_error": 0.0} for j in use}}
    agent, solver = live_agent(pool, cand)
    note = {"label": label, "fingerprint": fingerprint(cand), "during": "rating"}
    for j in use:
        other, seed, row = pool.agent(j), pool.seed(j), meta["rows"][j]
        x, raw = arrays.get(j, np.zeros(0)), arrays.get(f"{j}.raw", np.zeros(0))
        while (done := len(x)) < d:
            n = min(SEARCH_CHUNK, d - done)
            t = time.time()
            res = play_live(pool, agent, solver, other, seed, done, n, note)
            x, raw = np.concatenate([x, res["x"]]), np.concatenate([raw, res["raw"]])
            for k in ("a_search_failed", "b_search_failed", "a_sources"):
                row[k] = res[k] if row[k] is None else [u + v for u, v in zip(row[k], res[k])]
            row["seconds"] += time.time() - t
            row["flop_solves"] += res["flop"]
            row["aivat_mass_error"] = max(row["aivat_mass_error"], res["mass"])
            arrays[j], arrays[f"{j}.raw"] = x, raw
            tmp = part.with_name(f"{part.stem}.{os.getpid()}.tmp.npz")
            np.savez(tmp, key=key, meta=json.dumps(meta), **arrays)
            _fsync(tmp)
            os.replace(tmp, part)
            done = len(x)
            print(f"    {label} vs {j}: {done:,}/{d:,} deals  {x.mean():+9.1f} mbb/hand AIVAT ({raw.mean():+.1f} raw)  "
                  f"{row['seconds'] / done:.2f} s/deal", flush=True)
            if live:
                live.progress(f"{label} vs {j}", done, d)
    if engine_identity()["id"] != ident["id"] or play_code_sha() != play0:
        raise R.RatingError(f"the engine, rating code or play code changed during the rating; {part.name} is kept, "
                            "no receipt is written")
    if _sha256_file(cand["path"]) != cand["sha256"]:
        raise R.RatingError("the candidate file changed during the rating; no receipt is written")
    rows = {j: {**meta["rows"][j], "seed": pool.seed(j), "deals": d, "mean": float(arrays[j].mean()),
                "mean_raw": float(arrays[f"{j}.raw"].mean())} for j in use}
    (pool.dir / "receipts").mkdir(exist_ok=True)
    stem = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S%fZ}-{safe}"
    shas = save_arrays(pool.dir / "receipts" / f"{stem}.npz", arrays)
    receipt = {
        "schema": "parciver-rating-receipt-2", "pool": pool.defn["version"], "pool_hash": pool.defn["pool_hash"],
        "candidate": cand, "fingerprint": fingerprint(cand), "agent": describe(cand), "anchors": rows, "lbr": None,
        "arrays": shas, "weights": weights, "engine": ident, "play_code_sha256": play0, "aivat": True,
        "git": meta["git"], "threads": 1, "chunk": SEARCH_CHUNK, "started_utc": meta["started_utc"],
        "finished_utc": utc(),
    }
    path = pool.dir / "receipts" / f"{stem}.json"
    write_new(path, receipt)
    os.replace(part, part.with_name(f"{part.stem}.done-{stem}.npz"))  # kept; a new rating starts afresh
    del arrays, agent, solver
    gc.collect()
    with contextlib.suppress(Exception):
        import torch
        torch.cuda.empty_cache()  # the replay process needs the GPU
    if live:
        live.progress(f"{label}: replaying the receipt", 0, 1)
    replay_in_new_process(pool, path, 1)
    return path


def replay_in_new_process(pool: Pool, path: Path, threads: int) -> int:
    """Replay one receipt in a fresh interpreter (Amendment 2 (1)): the engine, the rating code and every
    blueprint are loaded again from disk, so nothing this process played with is reused. This process's
    blueprints are released first (each can be gigabytes). Returns the replay's exit code (0: identical)."""
    pool._bp.clear()
    gc.collect()
    rc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "verify", "--dir", str(pool.dir),
                         "--receipt", path.name, "--threads", str(threads)]).returncode
    if rc:
        print(f"  the replay of {path.name} did not pass (exit {rc}): the receipt is not VERIFIED", flush=True)
    return rc


def replayed_files(pool: Pool, cand: dict, when: str) -> None:
    """The files a replay plays are the ones its receipt and its pool name: the candidate's file and the pool's
    frozen files, hashed now (a memo keyed on size and mtime would miss a file put back with its old mtime).
    A RatingError otherwise: the replay is refused (STALE), it never records a verdict on other bytes."""
    p = cand.get("path")
    if p:
        try:
            same = _sha256_file(p) == cand.get("sha256")
        except OSError as e:
            raise R.RatingError(f"candidate file {p} cannot be read {when} the replay: {e}") from None
        if not same:
            raise R.RatingError(f"candidate file {p} is not the receipt's (sha256 {str(cand.get('sha256'))[:12]}) "
                                f"{when} the replay")
    pool.check_files(fresh=True)


def replay(pool: Pool, path: Path, r: Receipt, threads: int, cur: dict) -> dict:
    """Amendment 2 (1): a blueprint receipt replays in FULL (every anchor row, bit-identical, with its
    failed-search counts) plus one LBR chunk chosen by sha256(receipt sha + identity). The candidate's and the
    anchors' files are checked before they are loaded, once loaded and after the last game (the engine reads the
    file itself, so the bytes it loaded cannot be hashed here: a file swapped and put back while the engine is
    reading it is beyond these checks)."""
    arr = load_arrays(path.with_suffix(".npz"), r["arrays"])
    check_receipt(pool, r, arr)
    cand = r["candidate"]
    if is_live(cand):
        return replay_live(pool, r, arr, cur)
    replayed_files(pool, cand, "before")
    pool._bp.clear()  # the files checked above, never a blueprint an earlier replay loaded (candidate or anchor)
    agent = candidate_agent(pool, cand)
    opp = {j: pool.agent(j) for j in r["anchors"]}  # loaded here, so the check below covers what the games play
    replayed_files(pool, cand, "once loaded for")
    ok, detail = True, []
    for j, row in r["anchors"].items():
        res = play(pool, agent, opp[j], row["seed"], row["deals"], threads)
        same = bool(np.array_equal(res["per"], arr[j])) and res["a_search_failed"] == row["a_search_failed"] \
            and res["b_search_failed"] == row["b_search_failed"]
        ok &= same
        detail.append(f"{j} all {row['deals']:,} {'identical' if same else 'DIFFERENT'}")
        if not same:
            break
    lb = r.get("lbr")
    if ok and lb:
        size = min(pool.defn["replay"]["chunk"], lb["deals"])
        h = int(hashlib.sha256((r.sha + cur["id"]).encode()).hexdigest()[:16], 16)
        lo = (h % (lb["deals"] // size)) * size
        res = lbr(pool, agent, size, lb["seed"], threads, 1, first_deal=lo)
        same = bool(np.array_equal(res["per"], arr["LBR"][lo:lo + size]))
        if "LBR_preflop" in arr:
            same &= "pre" in res and bool(np.array_equal(res["pre"], arr["LBR_preflop"][lo:lo + size]))
        ok &= same
        detail.append(f"LBR deals {lo}..{lo + size - 1} {'identical' if same else 'DIFFERENT'}")
    replayed_files(pool, cand, "after")
    return {"ok": ok, "detail": "; ".join(detail), "kind": "full+lbr-chunk"}


def replay_live(pool: Pool, r: Receipt, arr: dict, cur: dict) -> dict:
    """Amendment 3 (6): one chunk of SEARCH_CHUNK deals per weighted anchor, chosen by sha256(receipt sha +
    identity), the same offset against each; its AIVAT and raw rows must both equal the receipt's bit for bit. The
    files are checked as for every replay, and the play code must be unchanged at the end from the identity the
    verdict is recorded under (returned as "identity")."""
    play0 = play_code_sha()
    cand, rid = r["candidate"], receipt_identity(r, cur, play0)
    replayed_files(pool, cand, "before")
    pool._bp.clear()
    agent, solver = live_agent(pool, cand)
    opp = {j: pool.agent(j) for j in r["anchors"]}
    replayed_files(pool, cand, "once loaded for")
    ok, detail = True, []
    h = int(hashlib.sha256((r.sha + rid).encode()).hexdigest()[:16], 16)  # Amendment 3 (6)
    note = {"label": cand.get("label"), "receipt_sha": r.sha, "during": "replay"}
    for j, row in r["anchors"].items():
        size = min(SEARCH_CHUNK, row["deals"])
        lo = (h % (row["deals"] // size)) * size
        res = play_live(pool, agent, solver, opp[j], row["seed"], lo, size, note)
        same = (bool(np.array_equal(res["x"], arr[j][lo:lo + size]))
                and bool(np.array_equal(res["raw"], arr[f"{j}.raw"][lo:lo + size])))
        ok &= same
        detail.append(f"{j} deals {lo}..{lo + size - 1} {'identical' if same else 'DIFFERENT'} (AIVAT and raw)")
        if not same:
            break
    replayed_files(pool, cand, "after")
    if play_code_sha() != play0:
        raise R.RatingError("the play code changed during the replay")
    return {"ok": ok, "detail": "; ".join(detail), "kind": "search-chunk", "identity": rid}


def verify_one(pool: Pool, path: Path, r: Receipt, threads: int, cur: dict) -> dict:
    """Replay a receipt and record the outcome: ok True (identical), False (a mismatch: INVALID), None (the
    replay was refused before anything was compared: the receipt stays STALE, never VERIFIED). The outcome is
    recorded against the bytes that were replayed (r.sha): a file changed meanwhile is not covered by it."""
    try:
        res = replay(pool, path, r, threads, cur)
    except MemoryError:
        raise
    except Exception as e:
        res = {"ok": None, "detail": f"cannot replay: {type(e).__name__}: {e}", "kind": "refused"}
    ident = res.pop("identity", None)  # a search replay's: the play code it played and checked unchanged
    if ident is None:
        try:
            ident = receipt_identity(r, cur)
        except Exception as e:  # only a refused replay gets here for a search receipt
            ident, res = cur["id"] + "+unreadable", {**res, "ok": None, "detail": f"{res['detail']}; {e}"}
    entry = {"receipt": path.name, "receipt_sha": r.sha, "identity": ident, **res, "utc": utc()}
    append_verification(pool, entry)
    tag = {True: "replay ok ", False: "MISMATCH  ", None: "REFUSED   "}[res["ok"]]
    print(f"  {tag} {path.name}: {res['detail']}", flush=True)
    return entry


def verify_all(pool: Pool, threads: int, only_stale: bool, name: str | None = None) -> int:
    """Replays (all, only the STALE ones, or the one named); the number that did not pass, counting every file
    in receipts/ that cannot be read as this pool's receipt."""
    cur = engine_identity()
    ver = verifications(pool)
    good, broken = read_receipts(pool)
    bad = 0
    if name and name not in {p.name for p, _ in good + broken}:
        print(f"  NOT FOUND  {name}", flush=True)
        bad += 1
    for path, why in broken:
        if not name or path.name == name:
            print(f"  UNREADABLE {path.name}: {why}", flush=True)
            bad += 1
    for path, r in good:
        if name and path.name != name:
            continue
        if only_stale and state_of(pool, path, r, ver, cur)[0] != "STALE":
            continue
        bad += verify_one(pool, path, r, threads, cur)["ok"] is not True
    return bad


# ---------------------------------------------------------------- live dashboard


class Live:
    """events.jsonl for the rating page (progress during a run, then the full view)."""

    def __init__(self, d: Path, open_browser: bool, port: int):
        import socket
        from parciver.dashboard import server as dash
        d.mkdir(parents=True, exist_ok=True)
        self.f = open(d / "events.jsonl", "a", encoding="utf-8")
        for p in range(port, port + 40):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", p))
                except OSError:
                    continue
            port = p
            break
        self.httpd = dash.serve(d, port=port, open_browser=False, block=False)
        self.url = f"http://127.0.0.1:{port}{PAGE}"
        print(f"[dashboard] {self.url}", flush=True)
        if open_browser:
            try:
                webbrowser.open(self.url)
            except Exception:
                pass

    def emit(self, e: dict) -> None:
        e.setdefault("t", time.time())
        self.f.write(json.dumps(e) + "\n")
        self.f.flush()

    def progress(self, what: str, done: int, total: int) -> None:
        self.emit({"type": "rating_progress", "what": what, "done": done, "total": total})

    def close(self, linger: float = 4.0) -> None:
        self.emit({"type": "run_end"})
        self.f.close()
        time.sleep(linger)


# ---------------------------------------------------------------- C6: cross-pool determinism (Amendment 2)


def c6_other(pool: Pool) -> Path | None:
    """The earlier pool this pool's C6 is pre-registered against (its sibling directory), or None."""
    o = C6_AGAINST.get(pool.defn["version"])
    # abspath, not resolve(): a pool directory that is a symlink or a junction keeps its sibling next to the link
    return None if o is None else Path(os.path.abspath(pool.dir)).parent / o


def newest(pool: Pool, label: str, rec: list | None = None):
    """The newest receipt under `label` (receipts are oldest first), or None."""
    return ([(p, r) for p, r in (receipts(pool) if rec is None else rec) if r["candidate"].get("label") == label]
            or [None])[-1]


def c6_inputs(pool: Pool, other: Pool) -> dict:
    """Every file C6's verdict depends on, by sha256: each pool's calibration pair records and the newest
    receipt of the reference and of each anchor (each file records its arrays' sha256)."""
    labels = [REFERENCE] + [j for j in pool.ids if j != "A0"]
    out = {}
    for pl in (pool, other):
        k, rec = len(pl.ids), receipts(pl)
        cal = {}
        for i in range(k):
            for j in range(k):
                if i != j:
                    f = pl.dir / "cal" / f"pair-{pl.ids[i]}-{pl.ids[j]}.json"
                    cal[f.stem] = receipt_sha(f) if f.exists() else None
        rows = {}
        for lab in labels:
            n = newest(pl, lab, rec)
            rows[lab] = None if n is None else [n[0].name, n[1].sha]
        out[pl.defn["version"]] = {"pool_hash": pl.defn["pool_hash"], "calibration": cal, "receipts": rows}
    return out


def c6_compare(pool: Pool, other: Pool) -> dict:
    """C6 (gating): `pool`'s calibration arrays equal `other`'s bit for bit, and the first deals of the newest
    reference and anchor rows equal `other`'s rows of the same agent, made on the same engine binary."""
    if pool.ids != other.ids or [x.get("sha256") for x in pool.defn["anchors"]] != [x.get("sha256") for x in other.defn["anchors"]]:
        raise R.RatingError("the pools do not have the same anchors")
    inputs = c6_inputs(pool, other)
    c6, problems = {"calibration": {}, "rows": {}}, []
    k = len(pool.ids)
    for i in range(k):
        for j in range(k):
            if i != j:
                same = bool(np.array_equal(load_pair(pool, i, j)[0], load_pair(other, i, j)[0]))
                c6["calibration"][f"{pool.ids[i]}-{pool.ids[j]}"] = same
                if not same:
                    problems.append(f"calibration pair {pool.ids[i]}-{pool.ids[j]} differs")
    eng = lambda r: r["engine"].get("sha256") if isinstance(r.get("engine"), dict) else None
    for lab in [REFERENCE] + [j for j in pool.ids if j != "A0"]:
        mine, theirs = newest(pool, lab), newest(other, lab)
        if mine is None or theirs is None:
            problems.append(f"{lab}: no receipt in {'this' if mine is None else 'the other'} pool")
            continue
        if eng(mine[1]) is None or eng(mine[1]) != eng(theirs[1]):
            problems.append(f"{lab}: made on different engines; C6 needs the same .pyd")
            continue
        x = load_arrays(mine[0].with_suffix(".npz"), mine[1]["arrays"])
        y = load_arrays(theirs[0].with_suffix(".npz"), theirs[1]["arrays"])
        res = {}
        for j in pool.ids:
            if j not in x or j not in y:
                problems.append(f"{lab}: no row against {j} in {'this' if j not in x else 'the other'} pool")
                continue
            n = C6_DEALS
            if min(len(x[j]), len(y[j])) < n:
                res[j] = {"deals": min(len(x[j]), len(y[j])), "identical": None}
                problems.append(f"{lab} vs {j}: {len(x[j]):,} deals here and {len(y[j]):,} in the other pool; "
                                f"C6 compares the first {n:,}")
                continue
            res[j] = {"deals": n, "identical": bool(np.array_equal(x[j][:n], y[j][:n]))}
            if not res[j]["identical"]:
                problems.append(f"{lab} vs {j}: the first {n:,} deals differ")
        if "LBR" in x and "LBR" in y and len(x["LBR"]) == len(y["LBR"]):
            res["LBR"] = {"deals": len(x["LBR"]), "identical": bool(np.array_equal(x["LBR"], y["LBR"]))}
            if not res["LBR"]["identical"]:
                problems.append(f"{lab}: LBR differs")
        c6["rows"][lab] = res
        del x, y
    if c6_inputs(pool, other) != inputs:
        raise R.RatingError("a pool's files changed during the comparison: run it again")
    return {"C6_ok": not problems, "C6": c6, "C6_problems": problems, "inputs": inputs}


C6_FILE = re.compile(r"\d{8}T\d{12}Z\.json")  # crosscheck-<other>-<UTC to the microsecond>.json, as cmd_crosscheck names it


def c6_state(pool: Pool) -> dict | None:
    """C6 as the view sees it: VERIFIED only when the newest crosscheck file passed on exactly the files there
    now, under the current engine and rating code and C6's deal count; STALE when it was not run on them (or
    by them), passed or failed; INVALID when it failed under them or cannot be read. None: no C6 for this
    pool. (`audit` also re-runs the comparison itself.) Only files named as cmd_crosscheck names them are read:
    their UTC names sort in time order, and an older code's (local-time names, no record of its inputs) could
    never be VERIFIED."""
    other = c6_other(pool)
    if other is None:
        return None
    o = C6_AGAINST[pool.defn["version"]]
    run = f"run `rating.py crosscheck --dir {pool.dir} --other {other}`"
    files = sorted(p for p in pool.dir.glob(f"crosscheck-{o}-*.json") if C6_FILE.fullmatch(p.name[len(o) + 12:]))
    if not files:
        return {"against": o, "state": "STALE", "why": f"C6 against {o} has not been run: {run}"}
    f = files[-1]
    try:
        c = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(c, dict):
            raise ValueError("not a crosscheck result")
    except (OSError, ValueError) as e:
        return {"against": o, "state": "INVALID", "why": f"{f.name} cannot be read: {e}"}
    # A pass or a failure counts only under the engine and code that ran it, at C6's deal count (RATING.md 00:55).
    if c.get("identity") != engine_identity()["id"] or c.get("c6_deals") != C6_DEALS:
        return {"against": o, "state": "STALE",
                "why": f"C6 ({f.name}) was run by another engine or rating code (or another C6 deal count): {run}"}
    if c.get("C6_ok") is not True:
        probs = c.get("C6_problems")
        probs = "; ".join(map(str, probs)) if isinstance(probs, list) else "?"
        return {"against": o, "state": "INVALID", "why": f"C6 against {o} FAILED ({f.name}): {probs[:400]}"}
    try:
        now = c6_inputs(pool, Pool(other))
    except MemoryError:
        raise
    except Exception as e:
        return {"against": o, "state": "STALE", "why": f"C6's files cannot be read ({type(e).__name__}: {e}): {run}"}
    if c.get("inputs") != now:
        return {"against": o, "state": "STALE", "why": f"C6 ({f.name}) was run on other files than these: {run}"}
    return {"against": o, "state": "VERIFIED", "why": f"C6 against {o} passed on these files ({f.name})"}


def c6_recheck(pool: Pool) -> list[str]:
    """`audit`: C6 re-run in memory now (never only read from a file); its problems, if any."""
    other = c6_other(pool)
    if other is None:
        return []
    try:
        res = c6_compare(pool, Pool(other))
    except MemoryError:
        raise
    except Exception as e:
        return [f"C6 against {other.name} cannot be re-run: {type(e).__name__}: {e}"]
    return [f"C6 against {other.name}: {p}" for p in res["C6_problems"]]


# ---------------------------------------------------------------- the view (every number recomputed)


def read_pointer(f: Path) -> dict:
    """current.json as written (a RatingError when it cannot be read or is malformed): a fingerprint (absent
    only in Amendment 1's label-only file), a label and a history of {fingerprint, label, set_utc}."""
    try:
        c = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise R.RatingError(f"current.json cannot be read: {e}") from None
    hist = c.get("history", []) if isinstance(c, dict) else None
    if (not isinstance(hist, list) or not isinstance(c.get("fingerprint", ""), str)
            or not isinstance(c.get("label", ""), str)
            or not all(isinstance(h, dict) and isinstance(h.get("fingerprint"), str)
                       and isinstance(h.get("label", ""), str) for h in hist)):
        raise R.RatingError("current.json is malformed")
    return c


def read_current(pool: Pool, rec: list[tuple[Path, dict]]) -> dict:
    """The current configuration from current.json: {fingerprint, previous, label} (Amendment 1's label-only
    file is resolved through the receipts, and only when the label names exactly one configuration)."""
    f = pool.dir / "current.json"
    if not f.exists():
        return {}
    c = read_pointer(f)
    fp = c.get("fingerprint")
    if fp is None:
        fps = {fingerprint(r["candidate"]) for _, r in rec if r["candidate"].get("label") == c.get("label")}
        if len(fps) != 1:
            raise R.RatingError(f"current.json names label {c.get('label')}, which names {len(fps)} configurations")
        fp = fps.pop()
    prev = next((h for h in reversed(c.get("history", [])) if h["fingerprint"] != fp), None)
    return {"fingerprint": fp, "label": c.get("label"), "previous": prev}


def anchor_diff(x: np.ndarray, y: np.ndarray) -> dict:
    d, v = R.diff(x, y)
    se = v ** 0.5
    return dict(mean=d, se=se, lo=d - R.Z95 * se, hi=d + R.Z95 * se, n=len(x))


def view_row(pool: Pool, path: Path, r: dict, state: tuple, gates: list, ref_arr: dict | None,
             w: dict | None) -> dict:
    """One receipt's row: its own state, then the worst pool-wide gate it inherits, then the recomputed numbers
    (none for an INVALID row)."""
    s, why, arr = state
    for gs, gw in gates:
        if RANK[gs] > RANK[s]:
            s, why = gs, gw
    c = r["candidate"]
    row = {"label": str(c.get("label")), "receipt": path.name, "state": s, "why": why, "fingerprint": fingerprint(c),
           "started_utc": r.get("started_utc"), "agent": str(r.get("agent", "")),
           "identity": r["engine"].get("id") if isinstance(r.get("engine"), dict) else None,
           "git": r.get("git"), "anchor": c.get("anchor")}
    if arr is not None and s != "INVALID":
        x = {j: arr[j] for j in pool.ids if j in arr}
        row["anchors"] = {j: R.mean_ci(v) for j, v in x.items()}
        row["deals"] = {j: len(v) for j, v in x.items()}
        raw = {j: arr[f"{j}.raw"] for j in pool.ids if f"{j}.raw" in arr}  # Amendment 3 (3): beside, never folded in
        if raw:
            row["raw_anchors"] = {j: R.mean_ci(v) for j, v in raw.items()}
        if ref_arr is not None and w is not None:
            row["paired"] = {j: anchor_diff(x[j], ref_arr[j]) for j in x if j in ref_arr}
            try:
                row["rating"] = R.paired(x, ref_arr, w)
                if raw:
                    row["rating_raw"] = R.paired(raw, ref_arr, w)
            except R.RatingError as e:
                row["state"], row["why"] = "INVALID", str(e)
                row.pop("paired", None)
                row.pop("rating_raw", None)
        if "LBR" in arr:
            row["lbr"] = R.mean_ci(arr["LBR"])
            if "LBR_preflop" in arr:
                row["lbr_postflop"] = R.mean_ci(arr["LBR"] - arr["LBR_preflop"])
            lb = r.get("lbr") or {}
            row["lbr_search_failed"] = {k: lb[k] for k in LBR_COUNTS if k in lb}
        row["search_failed"] = int(sum(sum(a["a_search_failed"]) for a in r["anchors"].values()))
    return row


def build_view(pool: Pool, c6: bool = True) -> dict:
    """Every number recomputed from the receipts and the calibration's arrays. A rating inherits the worst of the
    pool-wide gates: the calibration re-derivation (INVALID), the newest reference's state, C6 (pool-v2), and
    any file in receipts/ that cannot be read (STALE: it could be the newest receipt of a configuration).
    `c6=False` leaves C6 out (crosscheck's own C7 view)."""
    cur = engine_identity()
    ver = verifications(pool)
    rec, broken = read_receipts(pool)
    try:
        _, w, der = check_calibration(pool)
        cal_error = None
    except MemoryError:
        raise
    except Exception as e:
        w, der, cal_error = None, None, (str(e) if isinstance(e, R.RatingError) else f"{type(e).__name__}: {e}")
    refs = [(p, r) for p, r in rec if r["candidate"].get("label") == REFERENCE]
    ref, ref_state, ref_why, ref_arr, c1, c2 = None, "INVALID", "no A0 reference receipt", None, None, None
    if refs:
        ref = refs[-1]  # the newest; never a fallback to an older one
        ref_state, ref_why, ref_arr = state_of(pool, ref[0], ref[1], ver, cur)
        c1 = {"ok": {"VERIFIED": True, "INVALID": False}.get(ref_state), "detail": f"the reference is {ref_state}: {ref_why}"}
        if ref_state != "INVALID" and w is not None:
            try:
                bad = reference_problem(pool, ref[1], w)
                if bad is None:
                    c2 = seed_consistency(pool, ref_arr, w, der["A"], der["se"])
                    if not c2["ok"]:
                        bad = "C2 (seed consistency) fails against " + ", ".join(j for j, x in c2["anchors"].items() if not x["ok"])
            except MemoryError:
                raise
            except Exception as e:
                bad = f"{type(e).__name__}: {e}"
            if bad:
                ref_state, ref_why, ref_arr = "INVALID", f"not a valid reference: {bad}", None
    c6v = c6_state(pool) if c6 else None
    gates = [("INVALID", f"the calibration does not check: {cal_error}")] if cal_error else []
    gates.append((ref_state, f"the A0 reference is {ref_state}: {ref_why}"))
    if c6v:
        gates.append((c6v["state"], f"pool-v2's gating check C6 is {c6v['state']}: {c6v['why']}"))
    if broken:
        gates.append(("STALE", f"{len(broken)} file(s) in receipts/ cannot be read as this pool's receipts "
                               f"({', '.join(p.name for p, _ in broken)}): one could be the newest receipt of a "
                               "configuration or the reference; repair or remove them"))
    try:
        current = read_current(pool, rec)
        current_error = None
    except MemoryError:
        raise
    except Exception as e:
        current, current_error = {}, (str(e) if isinstance(e, R.RatingError) else f"{type(e).__name__}: {e}")
    want = {current.get("fingerprint"), (current.get("previous") or {}).get("fingerprint")} - {None}
    keep, rows = {}, []
    for path, r in rec:
        c = r["candidate"]
        fp = fingerprint(c)
        arr = None
        try:
            if ref and path == ref[0]:
                state = (ref_state, ref_why, ref_arr)
            else:
                state = state_of(pool, path, r, ver, cur)
                if state[0] != "INVALID" and w is not None:
                    bad = protocol_problem(pool, r, w)
                    if bad:
                        state = ("INVALID", f"not the pool's protocol: {bad}", None)
            arr = state[2]
            row = view_row(pool, path, r, state, gates, ref_arr, w)
        except MemoryError:
            raise
        except Exception as e:
            row = {"label": str(c.get("label")), "receipt": path.name, "state": "INVALID", "fingerprint": fp,
                   "why": f"cannot be shown: {type(e).__name__}: {e}"}
        if fp in want:
            keep[fp] = (row, arr if row["state"] != "INVALID" else None)  # newest wins: receipts are oldest first
        rows.append(row)
    for path, why in broken:
        rows.append({"label": path.stem, "receipt": path.name, "state": "INVALID", "why": why})
    head, head_why, change = None, None, None
    fp = current.get("fingerprint")
    if current_error:
        head_why = f"current.json: {current_error}"
    elif fp is None:
        head_why = "no current bot set: rating.py current --label ..."
    elif fp not in keep:
        head_why = f"no receipt of the current configuration {fp}"
    elif keep[fp][0]["state"] != "VERIFIED" or "rating" not in keep[fp][0]:
        head_why = f"the newest receipt of the current configuration is {keep[fp][0]['state']}: {keep[fp][0]['why']}"
    else:
        head = keep[fp][0]
        head["headline"] = True
        prev = current.get("previous")
        if prev:
            pfp = prev["fingerprint"]
            if pfp not in keep or keep[pfp][1] is None or keep[pfp][0]["state"] != "VERIFIED":
                change = {"from": prev.get("label"), "error": "the previous configuration has no VERIFIED receipt"}
            else:
                hx, px = keep[fp][1], keep[pfp][1]
                try:
                    change = R.change({j: hx[j] for j in pool.ids if j in hx}, {j: px[j] for j in pool.ids if j in px}, w)
                    change.update({"from": keep[pfp][0]["label"], "from_receipt": keep[pfp][0]["receipt"]})
                except R.RatingError as e:
                    change = {"from": prev.get("label"), "error": str(e)}
    acc = None
    if der is not None:  # only re-derived checks reach the page
        acc = {"C1_replay": c1 or {"ok": False, "detail": ref_why}}
        if c2 is not None:
            acc["C2_seed_consistency"] = c2
        acc["C3_nash"], acc["C4_role_symmetry"] = der["C3_nash"], der["C4_role_symmetry"]
    return {"type": "rating_view", "pool": pool.defn["version"], "pool_hash": pool.defn["pool_hash"][:12],
            "anchors": pool.ids, "anchor_defs": pool.defn["anchors"], "weights": w or {},
            "A": der["A"].tolist() if der else None, "A_se": der["se"].tolist() if der else None,
            "calibration_state": "accepted" if der else None, "calibration_error": cal_error,
            "nash_check": der["nash_check"] if der else None, "acceptance": acc, "c6": c6v,
            "identity": cur["id"], "current_label": current.get("label"), "current_fingerprint": fp,
            "headline": head, "headline_why": head_why, "change": change, "rows": rows, "built_utc": utc(),
            "rating_deals": pool.defn["rating"]["deals"],
            "rating_deals_weighted": pool.defn["rating"].get("deals_weighted", pool.defn["rating"]["deals"]),
            "lbr_deals": pool.defn["lbr"]["deals"]}


def render(view: dict) -> str:
    w = view["weights"]
    out = [f"parciver rating: {view['pool']} ({view['pool_hash']}), identity {view['identity']}",
           ("Nash weights: " + ", ".join(f"{j} {p:.3f}" for j, p in w.items() if p > 0)) if w
           else f"CALIBRATION DOES NOT CHECK: {view['calibration_error']}"]
    if view.get("c6"):
        c6 = view["c6"]
        out.append(f"C6 vs {c6['against']} (gating): {'PASS' if c6['state'] == 'VERIFIED' else c6['state']}: {c6['why']}")
    h = view["headline"]
    if h:
        r = h["rating"]
        out.append(f"HEADLINE  {h['label']} [{h['fingerprint']}]: PR {r['pr']:.1f}  (95% {r['lo']:.1f} .. {r['hi']:.1f}; "
                   f"{r['n']:,} deals)  {h['state']}")
        c = view["change"]
        if c:
            out.append(f"  change vs previous {c['from']}: " + (f"{c['mean']:+.1f} (95% {c['lo']:+.1f} .. {c['hi']:+.1f})"
                                                              if "mean" in c else c["error"]))
    else:
        out.append(f"HEADLINE  none ({view['headline_why']})")
    out.append(f"{'label':<26} {'state':<9} {'PR':>7} {'95% CI':>17} {'LBR mbb/h':>15}  receipt")
    for row in view["rows"]:
        r = row.get("rating")
        pr = f"{r['pr']:7.1f} {r['lo']:8.1f}..{r['hi']:<7.1f}" if r else f"{'-':>7} {'-':>17}"
        lb = row.get("lbr")
        lbs = f"{lb['mean']:+7.0f} +/-{R.Z95 * lb['se']:<4.0f}" if lb else f"{'-':>15}"
        out.append(f"{str(row['label']):<26} {row['state']:<9} {pr} {lbs}  {row['receipt']}")
        rr = row.get("rating_raw")
        if rr:
            out.append(f"    AIVAT per-deal values (Amendment 3); raw: PR {rr['pr']:.1f} (95% {rr['lo']:.1f} .. "
                       f"{rr['hi']:.1f}); no LBR guard for search")
        if row["state"] != "VERIFIED":
            out.append(f"    {row['why']}")
    return "\n".join(out)


# ---------------------------------------------------------------- commands


def open_live(a, pool: Pool, kind: str):
    if getattr(a, "no_dashboard", False):
        return None
    return Live(pool.dir / "views" / f"{kind}-{dt.datetime.now():%Y%m%d-%H%M%S}", not a.no_browser, a.port)


def cmd_calibrate(a) -> int:
    pool = Pool.create(Path(a.dir), smoke=a.smoke)
    pool.check_files()
    cfg = pool.defn["calibration"]
    live = open_live(a, pool, "calibrate")
    (pool.dir / "cal").mkdir(exist_ok=True)
    ident, g = engine_identity(), git()
    k = len(pool.ids)
    A, se = np.zeros((k, k)), np.zeros((k, k))
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    per_of = {}
    for i, j in pairs + [(j, i) for i, j in pairs]:
        stem = f"pair-{pool.ids[i]}-{pool.ids[j]}"
        meta_p = pool.dir / "cal" / f"{stem}.json"
        if meta_p.exists():
            per, meta = load_pair(pool, i, j)
            if meta["engine"]["sha256"] != ident["sha256"]:
                raise R.RatingError(f"{stem} was played on another engine: calibrate a new pool directory")
            print(f"  {stem}: kept ({meta['mean']:+.2f} mbb/hand)")
        else:
            prog = (lambda done, stem=stem: live.progress(stem, done, cfg["deals"])) if live else None
            res = play(pool, pool.agent(pool.ids[i]), pool.agent(pool.ids[j]), pool.cal_seed(i, j), cfg["deals"],
                       a.threads, label=stem, progress=prog)
            per = res["per"]
            shas = save_arrays(pool.dir / "cal" / f"{stem}.npz", {"per": per})
            meta = {"pair": [pool.ids[i], pool.ids[j]], "seed": pool.cal_seed(i, j), "deals": cfg["deals"],
                    "arrays": shas, "mean": float(per.mean()), "engine": ident, "git": g, "utc": utc(),
                    **{key: res[key] for key in ("a_search_failed", "b_search_failed", "seconds")}}
            write_new(meta_p, meta)
        per_of[(i, j)] = per
        if i < j:
            m = R.mean_ci(per)
            A[i, j], A[j, i] = m["mean"], -m["mean"]
            se[i, j] = se[j, i] = m["se"]
    # C4 (exact, gating): the same seed with the seats swapped is the same cards, so every deal negates.
    c4 = {f"{pool.ids[i]}-{pool.ids[j]}": bool(np.array_equal(per_of[(j, i)], -per_of[(i, j)])) for i, j in pairs}
    del per_of
    p = R.nash(A)
    chk = R.check_nash(A, p)
    w = {j: float(x) for j, x in zip(pool.ids, p)}
    print("Nash weights: " + ", ".join(f"{j} {x:.4f}" for j, x in w.items()), flush=True)
    # The reference (A0 vs every anchor on the rating seeds, at the protocol's deal counts).
    a0 = pool.anchor("A0")
    refs = [(pth, r) for pth, r in receipts(pool) if r["candidate"].get("label") == REFERENCE
            and r["engine"]["sha256"] == ident["sha256"] and reference_problem(pool, r, w) is None]
    if not refs:
        run_rating(pool, {"label": REFERENCE, "anchor": "A0", "path": a0["path"], "sha256": a0["sha256"]},
                   a.threads, w, pool.defn["lbr"]["deals"], live, g)
        refs = [(pth, r) for pth, r in receipts(pool) if r["candidate"].get("label") == REFERENCE
                and r["engine"]["sha256"] == ident["sha256"] and reference_problem(pool, r, w) is None]
    ref_path, ref_r = refs[-1]
    ref_sha = ref_r.sha
    mine = lambda: [v for v in verifications(pool)[0] if v["receipt_sha"] == ref_sha and v["identity"] == ident["id"]]
    c1 = mine()
    if not c1:
        replay_in_new_process(pool, ref_path, a.threads)
        c1 = mine()
    # Passed: a replay compared identical and none differed (a refused replay compared nothing).
    acc = {"C1_replay": {"ok": any(v["ok"] is True for v in c1) and not any(v["ok"] is False for v in c1),
                         "detail": c1[-1]["detail"] if c1 else "no replay was recorded"}}
    ref_arr = load_arrays(ref_path.with_suffix(".npz"), ref_r["arrays"])
    acc["C2_seed_consistency"] = seed_consistency(pool, ref_arr, w, A, se)
    del ref_arr
    acc["C3_nash"] = {"ok": bool(chk <= 1e-9), "max_Ap_over_maxA": chk}
    acc["C4_role_symmetry"] = {"ok": all(c4.values()), "exact_pairs": sum(c4.values()), "pairs": len(c4), "detail": c4}
    ok = all(acc[c]["ok"] for c in ("C1_replay", "C2_seed_consistency", "C3_nash", "C4_role_symmetry"))
    out = {"pool_hash": pool.defn["pool_hash"], "state": "accepted" if ok else "REJECTED", "anchors": pool.ids,
           "A_mbb_per_hand": A.tolist(), "se": se.tolist(), "p": p.tolist(), "nash_check": chk,
           "acceptance": acc, "engine": ident, "git": g, "utc": utc()}
    f = pool.dir / "calibration.json"
    if f.exists():
        f.replace(pool.dir / f"calibration-{dt.datetime.now():%Y%m%d-%H%M%S}.json.old")
    write_new(f, out)
    print(f"acceptance: {out['state']}  C1 {'ok' if acc['C1_replay']['ok'] else 'FAIL'}  "
          f"C2 {'ok' if acc['C2_seed_consistency']['ok'] else 'FAIL'}  C3 {'ok' if acc['C3_nash']['ok'] else 'FAIL'}  "
          f"C4 {acc['C4_role_symmetry']['exact_pairs']}/{acc['C4_role_symmetry']['pairs']} exact", flush=True)
    if ok:
        have = {r["candidate"].get("label") for _, r in receipts(pool) if r["engine"]["sha256"] == ident["sha256"]}
        for x in pool.defn["anchors"]:
            if x["id"] == "A0" or x["id"] in have:
                continue
            cand = {"label": x["id"], "anchor": x["id"], "path": x.get("path"), "sha256": x.get("sha256")}
            run_rating(pool, cand, a.threads, w, pool.defn["lbr"]["deals"], live, g)
        verify_all(pool, a.threads, only_stale=True)
        view = build_view(pool)
        print(render(view))
        if live:
            live.emit(view)
    if live:
        live.close()
    return 0 if ok else 1


def cmd_rate(a) -> int:
    pool = Pool(Path(a.dir))
    pool.check_files()
    _, w, _ = check_calibration(pool)
    if a.search == "live":
        return rate_live(a, pool, w)
    if a.search != "none":
        raise R.RatingError("search candidates other than the live runner's are refused (RATING.md Amendment 2 "
                            "(8), Amendment 3)")
    if a.label == REFERENCE or a.label in pool.ids:
        raise R.RatingError(f"label {a.label} is reserved for the pool's own anchors")
    path = Path(a.candidate).resolve()
    cand = {"label": a.label, "path": str(path), "sha256": sha256_file(path), "search": None,
            "translation": a.translation}
    fp = fingerprint(cand)
    clash = {fingerprint(r["candidate"]) for _, r in receipts(pool) if r["candidate"].get("label") == a.label} - {fp}
    if clash:
        raise R.RatingError(f"label {a.label} already names another configuration ({', '.join(sorted(clash))}); "
                            "a label names one configuration: pick a new one")
    lbr_deals = pool.defn["lbr"]["deals"] if a.lbr_deals is None else a.lbr_deals
    if lbr_deals < 0 or lbr_deals == 1:
        raise R.RatingError(f"--lbr-deals {lbr_deals}: 0 (no LBR guard) or at least 2 (an interval needs 2 deals)")
    live = open_live(a, pool, "rate")
    verify_all(pool, a.threads, only_stale=True)
    p = run_rating(pool, cand, a.threads, w, lbr_deals, live)
    view = build_view(pool)
    print(render(view))
    print(f"receipt: {p}  (configuration {fp})")
    if live:
        live.emit(view)
        live.close()
    return 0


def rate_live(a, pool: Pool, w: dict) -> int:
    """Amendment 3: `rate --search live`. The other receipts are not replayed first (CPU hours before a GPU
    run that does not need them); `verify` does that."""
    if a.label == REFERENCE or a.label in pool.ids:
        raise R.RatingError(f"label {a.label} is reserved for the pool's own anchors")
    if a.lbr_deals:
        raise R.RatingError("a search configuration has no LBR guard (Amendment 3 (7)): drop --lbr-deals")
    if a.translation != "snap":
        raise R.RatingError("the live runner plays snap translation")
    try:  # before build_agent imports the solver (play_files() loads only the runner, slumbot and solver_watch)
        play0 = play_code_sha()
    except OSError as e:
        raise R.RatingError(f"the play code cannot be read: {e}") from None
    sr = live_runner()
    path = Path(a.candidate).resolve()
    if path != Path(sr.BLUEPRINT).resolve():
        raise R.RatingError(f"the live runner plays {sr.BLUEPRINT}, not {path}")
    cand = {"label": a.label, "path": str(path), "sha256": sha256_file(path), "search": live_search_record(sr),
            "translation": "snap"}
    fp = fingerprint(cand)
    clash = {fingerprint(r["candidate"]) for _, r in receipts(pool) if r["candidate"].get("label") == a.label} - {fp}
    if clash:
        raise R.RatingError(f"label {a.label} already names another configuration ({', '.join(sorted(clash))}); "
                            "a label names one configuration: pick a new one")
    live = open_live(a, pool, "rate")
    p = run_live_rating(pool, cand, w, live, play0=play0)
    view = build_view(pool)
    print(render(view))
    print(f"receipt: {p}  (configuration {fp})")
    if live:
        live.emit(view)
        live.close()
    return 0


def cmd_verify(a) -> int:
    pool = Pool(Path(a.dir))
    pool.check_files()
    return 1 if verify_all(pool, a.threads, only_stale=a.stale_only, name=a.receipt) else 0


def cmd_current(a) -> int:
    pool = Pool(Path(a.dir))
    rec = receipts(pool)
    fps = {fingerprint(r["candidate"]) for _, r in rec if r["candidate"].get("label") == a.label}
    if not fps:
        raise R.RatingError(f"no receipt labelled {a.label}")
    if len(fps) > 1:
        raise R.RatingError(f"label {a.label} names {len(fps)} configurations; it cannot name the current bot")
    fp = fps.pop()
    f = pool.dir / "current.json"
    hist = []
    if f.exists():
        try:
            old = read_pointer(f)
        except R.RatingError as e:  # never overwrite the record of which bot was current
            raise R.RatingError(f"{e}; `current` will not overwrite it (it holds the history of the current bot): "
                                "repair it or move it aside, then run `current` again") from None
        hist = list(old.get("history", []))
        if not hist and (old.get("label") or old.get("fingerprint")):
            try:
                hist = [{"fingerprint": read_current(pool, rec)["fingerprint"],
                         **({"label": old["label"]} if old.get("label") else {}), "set_utc": old.get("set_utc")}]
            except R.RatingError as e:
                raise R.RatingError(f"{e}; `current` will not overwrite it (it holds the record of the current "
                                    "bot): repair it or move it aside, then run `current` again") from None
    now = utc()
    if not hist or hist[-1]["fingerprint"] != fp or hist[-1].get("label") != a.label:
        hist.append({"fingerprint": fp, "label": a.label, "set_utc": now})
    write_replace(f, {"fingerprint": fp, "label": a.label, "set_utc": now, "history": hist})
    print(f"current bot: {a.label} (configuration {fp})")
    return 0


def watched(pool: Pool) -> tuple:
    """What a view depends on: a change to any of it rebuilds the dashboard's view."""
    files = [pool.def_path, pool.dir / "calibration.json", pool.dir / "current.json", pool.dir / "verify.jsonl"]
    files += sorted(pool.dir.glob("crosscheck-*.json"))
    with contextlib.suppress(Exception):  # a search receipt's identity (Amendment 3 (5))
        files += play_files()
    other = c6_other(pool)  # C6's state also depends on the other pool's calibration and receipts
    for d in [pool.dir] + ([other] if other is not None else []):
        for sub in ("receipts", "cal"):
            if (d / sub).exists():
                files += sorted((d / sub).iterdir())
    out = []
    for p in files:
        try:
            st = p.stat()
            out.append((str(p), st.st_size, st.st_mtime_ns))
        except OSError:
            out.append((str(p), None, None))
    with contextlib.suppress(Exception):
        out.append(("play_env", play_env(), None))
    return tuple(out)


def audit_problems(pool: Pool, view: dict) -> list[str]:
    problems = []
    if view["calibration_error"]:
        problems.append(f"calibration does not check: {view['calibration_error']}")
    else:  # every acceptance check, re-derived in this view
        for name, c in view["acceptance"].items():
            if c.get("ok") is not True:
                problems.append(f"{name} did not pass: {c.get('detail', '')}".rstrip(": "))
    if view.get("c6") and view["c6"]["state"] != "VERIFIED":
        problems.append(f"C6 against {view['c6']['against']} is {view['c6']['state']}: {view['c6']['why']}")
    bad = [r["receipt"] for r in view["rows"] if r["state"] == "INVALID"]
    if bad:
        problems.append(f"{len(bad)} INVALID receipt(s): {bad}")
    stale = [r["receipt"] for r in view["rows"] if r["state"] == "STALE"]
    if stale:
        problems.append(f"{len(stale)} STALE receipt(s) (run verify): {stale}")
    if view["headline"] is None:  # the rating of record is the current bot's VERIFIED rating; none, no pass
        problems.append(f"no VERIFIED headline: {view['headline_why']}")
    ref = [r for r in view["rows"] if r["label"] == REFERENCE]
    if not ref or ref[-1]["state"] != "VERIFIED" or ref[-1].get("rating", {}).get("pr") != R.BASE:
        problems.append("the newest reference is not VERIFIED at exactly 1500")
    return problems


def cmd_show(a) -> int:
    pool = Pool(Path(a.dir))
    pool.check_files()
    view = build_view(pool)
    print(render(view))
    if a.audit:
        problems = audit_problems(pool, view) + c6_recheck(pool)
        print("AUDIT " + ("FAIL: " + "; ".join(problems) if problems else "OK"))
        return 1 if problems else 0
    live = open_live(a, pool, "show")
    if live:
        live.emit(view)
        live.emit({"type": "run_end"})
        print("serving the dashboard (rebuilt whenever a receipt, replay or the current bot changes); Ctrl+C to stop",
              flush=True)
        last = watched(pool)
        try:
            while True:
                time.sleep(2)
                now = watched(pool)
                if now == last:
                    continue
                last = now
                try:
                    view = build_view(pool)
                    live.emit(view)
                    print(f"[{time.strftime('%H:%M:%S')}] view rebuilt", flush=True)
                except Exception as e:  # keep serving; the page shows the error
                    live.emit({"type": "rating_error", "error": f"{type(e).__name__}: {e}"})
        except KeyboardInterrupt:
            pass
    return 0


def cmd_crosscheck(a) -> int:
    """Amendment 2's C6 (gating) and C7 (reported) against an earlier pool version. The result file records
    the files C6 was run on: the view trusts it only while they are unchanged."""
    pool, other = Pool(Path(a.dir)), Pool(Path(a.other))
    res = c6_compare(pool, other)
    problems = res["C6_problems"]
    vm, vo = build_view(pool, c6=False), build_view(other, c6=False)
    c7 = {}
    for lab in ("A1", "A2"):
        rm = [r for r in vm["rows"] if r["label"] == lab and "rating" in r]
        ro = [r for r in vo["rows"] if r["label"] == lab and "rating" in r]
        if rm and ro:
            x, y = rm[-1]["rating"], ro[-1]["rating"]
            c7[lab] = {"pr": x["pr"], "other_lo": y["lo"], "other_hi": y["hi"], "inside": bool(y["lo"] <= x["pr"] <= y["hi"])}
        else:
            c7[lab] = {"inside": None, "why": "no rating in one of the pools"}
    out = {"pool": pool.defn["version"], "other": other.defn["version"], **res, "C7_reported": c7,
           "identity": engine_identity()["id"], "c6_deals": C6_DEALS, "utc": utc()}
    # UTC to the microsecond, like receipts: the newest file sorts last (c6_state reads the last one).
    write_new(pool.dir / f"crosscheck-{other.defn['version']}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S%fZ}.json",
              out)
    print(f"C6 cross-pool determinism vs {other.defn['version']}: " + ("PASS" if not problems else "FAIL: " + "; ".join(problems)))
    for lab, v in c7.items():
        print(f"C7 {lab}: " + (f"PR {v['pr']:.1f}, {other.defn['version']} 95% {v['other_lo']:.1f}..{v['other_hi']:.1f} -> "
                               f"{'inside' if v['inside'] else 'OUTSIDE (investigate)'}" if v["inside"] is not None else v["why"]))
    return 0 if not problems else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, threads=True, dash=True):
        p.add_argument("--dir", required=True)
        if threads:
            p.add_argument("--threads", type=int, default=8, help="results do not depend on it")
        if dash:
            p.add_argument("--port", type=int, default=DEFAULT_PORT)
            p.add_argument("--no-browser", action="store_true")
            p.add_argument("--no-dashboard", action="store_true")
        return p

    p = common(sub.add_parser("calibrate"))
    p.add_argument("--smoke", action="store_true", help="tiny pool (machinery test only, never of record)")
    p.set_defaults(fn=cmd_calibrate)
    p = common(sub.add_parser("rate"))
    p.add_argument("--candidate", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--search", choices=["none", "river", "turn", "live"], default="none",
                   help="live: the live runner's configuration on the GPU path (Amendment 3); others are refused")
    p.add_argument("--translation", choices=["snap", "standin"], default="snap")
    p.add_argument("--lbr-deals", type=int, default=None, help="default: the pool's; 0 = no LBR guard")
    p.set_defaults(fn=cmd_rate)
    p = common(sub.add_parser("verify"), dash=False)
    p.add_argument("--receipt")
    p.add_argument("--stale-only", action="store_true", help="replay only the receipts that are STALE")
    p.set_defaults(fn=cmd_verify)
    p = common(sub.add_parser("current"), threads=False, dash=False)
    p.add_argument("--label", required=True)
    p.set_defaults(fn=cmd_current)
    p = common(sub.add_parser("show"), threads=False)
    p.set_defaults(fn=cmd_show, audit=False)
    p = common(sub.add_parser("audit"), threads=False, dash=False)
    p.set_defaults(fn=cmd_show, audit=True)
    p = common(sub.add_parser("crosscheck"), threads=False, dash=False)
    p.add_argument("--other", required=True, help="the earlier pool directory (C6/C7)")
    p.set_defaults(fn=cmd_crosscheck)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except R.RatingError as e:
        print(f"rating: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
