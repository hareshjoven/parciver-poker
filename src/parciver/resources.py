"""Live machine-resource sampling for the training dashboard — stdlib only.

The dashboard's resource panel wants CPU utilization, RAM, GPU utilization + GPU
memory, and a thread count, sampled cheaply alongside the training metrics. This
box has neither ``psutil`` nor ``pynvml`` installed and the project's rule is to
avoid heavy new deps, so this reads the numbers straight from the OS:

  * **CPU / RAM** via ``ctypes`` calls into ``kernel32`` (``GetSystemTimes`` for a
    system-wide busy fraction between two samples, ``GlobalMemoryStatusEx`` for
    physical memory). Windows-only, which is this box; on any other platform the
    two fields come back ``None`` rather than raising.
  * **GPU** via a short ``nvidia-smi --query-gpu=...`` call, parsed from CSV. That
    subprocess costs ~30-50 ms, far too much to pay on every metrics tick, so the
    result is **cached** and only refreshed at most once per ``gpu_min_interval``
    seconds; callers get the last reading in between.

Everything is best-effort and defensive: a missing tool, a driver hiccup, or a
non-Windows host degrades one field to ``None`` and never throws, because a
resource read must never be able to interrupt or slow a training run. The returned
dict matches the ``resources`` shape the telemetry contract validates.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from typing import Optional

_GIB = float(1 << 30)


def _now() -> float:
    return time.perf_counter()


class ResourceSampler:
    """Samples CPU%, RAM, GPU%, GPU memory, and a thread count on demand.

    CPU utilization is the busy fraction of wall time *between consecutive
    samples*, so the first :meth:`sample` after construction reports ``cpu_pct``
    from construction time to now. Call it on your metrics cadence (e.g. once a
    second); sampling faster is fine but a sub-millisecond gap makes the CPU number
    noisy.

    ``threads`` is not something the OS exposes per-process without extra
    machinery, so the producer passes the count it actually launched (rayon workers
    for the Rust MCCFR, say) as ``threads=`` and it is echoed back; when omitted it
    falls back to ``torch.get_num_threads()`` if torch is importable, else the
    logical CPU count.
    """

    def __init__(self, *, threads: Optional[int] = None, gpu_min_interval: float = 1.0) -> None:
        self._is_windows = sys.platform == "win32"
        self._gpu_min_interval = gpu_min_interval
        self._threads = threads if threads is not None else _default_threads()

        # CPU: remember the previous GetSystemTimes reading to form a delta.
        self._prev_cpu: Optional[tuple[int, int]] = None  # (idle, total) in 100ns ticks
        # GPU: cache the last successful nvidia-smi reading.
        self._gpu_cache: Optional[dict] = None
        self._gpu_cache_t: float = -1e18
        self._nvidia_smi = shutil.which("nvidia-smi")

        if self._is_windows:
            import ctypes  # local import so non-Windows never touches it

            self._ctypes = ctypes
            self._k32 = ctypes.windll.kernel32
            # Prime the CPU delta so the first real sample has a baseline.
            self._prev_cpu = self._read_cpu_times()

    # -- public -----------------------------------------------------------
    def sample(self) -> dict:
        """Return one ``resources`` reading. Never raises; unavailable fields are
        ``None``."""
        cpu = self._cpu_pct()
        ram_used, ram_total = self._ram_gb()
        gpu = self._gpu()
        return {
            "cpu_pct": cpu,
            "ram_used_gb": ram_used,
            "ram_total_gb": ram_total,
            "gpu_pct": gpu.get("gpu_pct"),
            "gpu_mem_used_gb": gpu.get("gpu_mem_used_gb"),
            "gpu_mem_total_gb": gpu.get("gpu_mem_total_gb"),
            "threads": self._threads,
        }

    def set_threads(self, threads: int) -> None:
        self._threads = int(threads)

    # -- CPU --------------------------------------------------------------
    def _read_cpu_times(self) -> Optional[tuple[int, int]]:
        """(idle, total) system CPU time in 100ns ticks, or None if unavailable.

        ``GetSystemTimes`` reports kernel time *including* idle, so total busy+idle
        is ``kernel + user`` and idle is a component of it."""
        if not self._is_windows:
            return None
        try:
            c = self._ctypes
            idle = c.c_ulonglong(0)
            kernel = c.c_ulonglong(0)
            user = c.c_ulonglong(0)
            ok = self._k32.GetSystemTimes(c.byref(idle), c.byref(kernel), c.byref(user))
            if not ok:
                return None
            total = kernel.value + user.value  # kernel already contains idle
            return idle.value, total
        except Exception:
            return None

    def _cpu_pct(self) -> Optional[float]:
        cur = self._read_cpu_times()
        if cur is None or self._prev_cpu is None:
            self._prev_cpu = cur
            return None
        d_idle = cur[0] - self._prev_cpu[0]
        d_total = cur[1] - self._prev_cpu[1]
        self._prev_cpu = cur
        if d_total <= 0:
            return None
        busy = 1.0 - (d_idle / d_total)
        return max(0.0, min(100.0, busy * 100.0))

    # -- RAM --------------------------------------------------------------
    def _ram_gb(self) -> tuple[Optional[float], Optional[float]]:
        if not self._is_windows:
            return None, None
        try:
            c = self._ctypes

            class MEMORYSTATUSEX(c.Structure):
                _fields_ = [
                    ("dwLength", c.c_ulong),
                    ("dwMemoryLoad", c.c_ulong),
                    ("ullTotalPhys", c.c_ulonglong),
                    ("ullAvailPhys", c.c_ulonglong),
                    ("ullTotalPageFile", c.c_ulonglong),
                    ("ullAvailPageFile", c.c_ulonglong),
                    ("ullTotalVirtual", c.c_ulonglong),
                    ("ullAvailVirtual", c.c_ulonglong),
                    ("ullAvailExtendedVirtual", c.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = c.sizeof(MEMORYSTATUSEX)
            if not self._k32.GlobalMemoryStatusEx(c.byref(stat)):
                return None, None
            total = stat.ullTotalPhys / _GIB
            used = (stat.ullTotalPhys - stat.ullAvailPhys) / _GIB
            return used, total
        except Exception:
            return None, None

    # -- GPU --------------------------------------------------------------
    def _gpu(self) -> dict:
        if self._nvidia_smi is None:
            return {}
        now = _now()
        if self._gpu_cache is not None and (now - self._gpu_cache_t) < self._gpu_min_interval:
            return self._gpu_cache
        reading = self._query_nvidia_smi()
        if reading is not None:
            self._gpu_cache = reading
            self._gpu_cache_t = now
            return reading
        # On a transient failure, keep serving the last good reading if we have one.
        return self._gpu_cache or {}

    def _query_nvidia_smi(self) -> Optional[dict]:
        try:
            out = subprocess.run(
                [self._nvidia_smi,
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2.0,
            )
            if out.returncode != 0 or not out.stdout.strip():
                return None
            # First GPU (this box has one). Fields are MiB for memory.
            line = out.stdout.strip().splitlines()[0]
            util_s, used_s, total_s = (x.strip() for x in line.split(","))
            return {
                "gpu_pct": float(util_s),
                "gpu_mem_used_gb": float(used_s) / 1024.0,
                "gpu_mem_total_gb": float(total_s) / 1024.0,
            }
        except Exception:
            return None


def _default_threads() -> int:
    try:
        import torch  # optional; only present with the gpu extra
        return int(torch.get_num_threads())
    except Exception:
        import os
        return os.cpu_count() or 1
