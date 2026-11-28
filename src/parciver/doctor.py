# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""``parciver doctor``: what works on this machine, what is missing, and the one command that fixes each thing.

It reads nothing it does not have to (the tables are checked by size against the manifest; ``--verify`` hashes them)
and changes nothing. Three tiers, matching the README's "Play" section:

1. Kuhn and Leduc: Python only.
2. No-limit hold'em: the Rust engine, the blueprint tables.
3. The full GPU configuration: PyTorch with CUDA, Triton, the flop value nets, Slumbot's ``sample_api.py``.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import assets
from .play_nlhe import engine_present, home, locate_assets

OK, MISSING, SKIP = "ok", "missing", "skip"
_MARK = {OK: "[ok]".ljust(12), MISSING: "[missing]".ljust(12), SKIP: "[not set up]"}


@dataclass
class Check:
    tier: int
    name: str
    status: str
    detail: str = ""
    fix: str = ""


def _find(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _table_state(path: Path, entry: dict, verify: bool) -> tuple[str, str]:
    """('ok'|'missing', detail) for one installed table; a wrong size means a damaged or partial install."""
    if not path.is_file():
        return MISSING, "not found"
    if path.stat().st_size != entry["bytes"]:
        return MISSING, f"found, but {path.stat().st_size:,} bytes where {entry['bytes']:,} are expected (damaged or partial)"
    if verify and not assets.verified(path, entry):
        return MISSING, "found, but its SHA-256 does not match the published one (damaged)"
    return OK, ("verified" if verify else "present, size matches")


def gpu_state() -> tuple[str, str]:
    """('ok'|'missing'|'skip', detail) for PyTorch with a usable CUDA GPU."""
    if not _find("torch"):
        return SKIP, "PyTorch is not installed"
    try:
        import torch
    except Exception as e:  # a broken install must not crash the doctor
        return MISSING, f"PyTorch fails to import ({type(e).__name__}: {e})"
    if not torch.cuda.is_available():
        build = "a CPU-only build" if torch.version.cuda is None else f"CUDA {torch.version.cuda} but no usable GPU"
        return MISSING, f"PyTorch {torch.__version__} is {build}"
    p = torch.cuda.get_device_properties(0)
    return OK, f"PyTorch {torch.__version__}, {p.name}, {p.total_memory / 2**30:.0f} GB"


def sample_api_path() -> Path:
    return Path(os.environ.get("PARCIVER_SLUMBOT_SAMPLE", "C:/tmp/slumbot/sample_api.py"))


def run_checks(verify: bool = False) -> list[Check]:
    out: list[Check] = []
    v = sys.version_info
    py_ok = v >= (3, 12)
    out.append(Check(1, "Python 3.12 or newer", OK if py_ok else MISSING, f"{v.major}.{v.minor}.{v.micro}",
                     "" if py_ok else "install Python 3.12+ from python.org and make a new virtual environment"))
    out.append(Check(1, "numpy", OK if _find("numpy") else MISSING, "",
                     "" if _find("numpy") else 'pip install -e .   (from the repository folder)'))

    eng = engine_present()
    out.append(Check(2, "Rust engine (parciver_engine)", OK if eng else MISSING, "",
                     "" if eng else "pip install maturin; maturin develop --release -m engine/Cargo.toml   "
                                    "(needs Rust: https://rustup.rs)"))
    m = assets.load_manifest()
    root = home()
    found = locate_assets()  # the same lookup `play nlhe` uses: the install folder, or a source checkout's data/
    if found is not None and not (root / "blueprint.bin").is_file():
        out.append(Check(2, "blueprint tables", OK, f"found at {found[1].parent}"))
    else:
        for e in m["files"]:
            st, detail = _table_state(root / e["dest"], e, verify)
            out.append(Check(2, f"table {e['dest']}", st, detail, "" if st == OK else "parciver fetch-blueprint"))

    gst, gdetail = gpu_state()
    out.append(Check(3, "PyTorch with a CUDA GPU", gst, gdetail,
                     "" if gst == OK else "install PyTorch from pytorch.org for your driver, then: pip install -e \".[gpu]\""))
    if sys.platform == "win32":
        tr = _find("triton")
        out.append(Check(3, "Triton (GPU kernels)", OK if tr else SKIP, "", "" if tr else 'pip install -e ".[gpu]"'))
    for e in m.get("nets", {}).get("files", []):
        st, detail = _table_state(root / e["dest"], e, verify)
        out.append(Check(3, f"flop net {e['dest']}", st if st == OK else SKIP, detail,
                         "" if st == OK else "parciver fetch-nets"))
    sp = sample_api_path()
    out.append(Check(3, "Slumbot sample_api.py (mock runs)", OK if sp.is_file() else SKIP, str(sp),
                     "" if sp.is_file() else "optional: only for scripts/slumbot_run.py --mock; see the README (tier 3)"))
    return out


def summary(checks: list[Check]) -> tuple[list[str], list[str]]:
    """(games you can play now, next steps in order)."""
    def ready(tier: int) -> bool:
        return all(c.status == OK for c in checks if c.tier == tier)

    can = []
    if ready(1):
        can += ["kuhn", "leduc"]
    if ready(1) and ready(2):
        can.append("nlhe")
    steps: list[str] = []
    for c in checks:
        if c.tier <= 2 and c.status != OK and c.fix and c.fix not in steps:
            steps.append(c.fix)
    return can, steps


def run(verify: bool = False, say: Callable[[str], None] = print) -> int:
    checks = run_checks(verify)
    say(f"parciver doctor: checking this machine (tables in {home()})")
    titles = {1: "Kuhn and Leduc poker  (parciver play kuhn | leduc)",
              2: "No-limit hold'em  (parciver play nlhe)",
              3: "Full GPU configuration  (advanced, optional; see README tier 3)"}
    for tier in (1, 2, 3):
        say("")
        say(titles[tier])
        for c in (c for c in checks if c.tier == tier):
            line = f"  {_MARK[c.status]} {c.name}" + (f": {c.detail}" if c.detail else "")
            say(line)
            if c.status != OK and c.fix:
                say(f"               fix: {c.fix}")
    can, steps = summary(checks)
    say("")
    say("You can play now: " + (", ".join(can) if can else "nothing yet"))
    if steps:
        say("Next, in order:")
        for i, s in enumerate(steps, 1):
            say(f"  {i}. {s}")
    else:
        say("Tiers 1 and 2 are ready. Try:  parciver play nlhe")
    return 0
