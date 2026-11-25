"""Standard-library SSE server for the training dashboard.

The viewer is a separate process from the trainer. It tails the append-only
``events.jsonl`` and pushes new events to the browser over Server-Sent Events; the
page renders. There is no dependency the trainer can block on, so the trainer is
never slowed and the viewer can never lag it — the whole point of the decoupled
telemetry design.

Design choices that keep it robust and lag-free:

  * **Stdlib only.** ``ThreadingHTTPServer`` + ``BaseHTTPRequestHandler``; no
    framework, no build step. The SPA is static files served from ``static/``.
  * **Replay then tail.** A newly connected browser first receives the full run
    history (so a late viewer sees the whole convergence curve and the latest range
    grid), then live events as they are appended. Reconnects resume cleanly because
    each connection tails from its own offset.
  * **Server-side coalescing.** New events are drained and sent as a single batched
    SSE message per flush tick, so a fast run (``replay_demo --fast`` emits
    thousands) arrives as a few big frames rather than a flood — the client then
    decimates for rendering. The trainer's own emit rate is unaffected.
  * **Heartbeats after completion.** Once ``run_end`` is seen the stream stays open
    with SSE comment heartbeats, so the browser keeps the final state and its
    completion banner instead of auto-reconnecting and replaying history.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import telemetry as T

STATIC_DIR = Path(__file__).resolve().parent / "static"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
}


class DashboardServer(ThreadingHTTPServer):
    """HTTP server bound to a single run directory (the ``events.jsonl`` to tail)."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, run_dir: Path, *, flush_interval: float = 0.05,
                 poll: float = 0.05) -> None:
        self.run_dir = Path(run_dir)
        self.flush_interval = flush_interval
        self.poll = poll
        super().__init__(addr, _Handler)


class _Handler(BaseHTTPRequestHandler):
    server_version = "parciverDash/1.0"
    protocol_version = "HTTP/1.1"

    # -- routing ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (stdlib name)
        path = self.path.split("?", 1)[0]
        try:
            if path == "/" or path == "/index.html":
                self._serve_static("index.html")
            elif path == "/events":
                self._serve_sse()
            elif path == "/healthz":
                self._serve_bytes(b'{"ok":true}', "application/json")
            elif path == "/api/meta":
                self._serve_meta()
            elif path.startswith("/static/"):
                self._serve_static(path[len("/static/"):])
            else:
                self.send_error(404, "not found")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the browser navigated away mid-response; nothing to do

    # -- handlers ---------------------------------------------------------
    def _serve_static(self, rel: str) -> None:
        # Prevent path traversal; only files inside static/ are served.
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            self.send_error(404, "not found")
            return
        body = target.read_bytes()
        self._serve_bytes(body, _CONTENT_TYPES.get(target.suffix, "application/octet-stream"))

    def _serve_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_meta(self) -> None:
        """A tiny status endpoint (run id + whether the log exists yet)."""
        run_dir = self.server.run_dir
        log = run_dir / "events.jsonl"
        meta = {"run_id": run_dir.name, "log_exists": log.exists(),
                "run_dir": str(run_dir)}
        self._serve_bytes(json.dumps(meta).encode(), "application/json")

    def _serve_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        reader = T.RunReader(self.server.run_dir)
        # The sidecar (scripts/blueprint_sidecar.py) fills the action-frequency,
        # bet-sizing, and head-to-head panels by appending to a SEPARATE panels.jsonl
        # — separate so two processes never write one append-only log. Tail it too
        # and merge its events into the same SSE stream; the client applies both by
        # field, so a panels checkpoint (action/bet mixes) and a trainer checkpoint
        # (range grid) accumulate rather than clobber. Absent panels.jsonl is fine
        # (RunReader yields nothing until it appears).
        panels = T.RunReader(self.server.run_dir / "panels.jsonl")
        flush_interval = self.server.flush_interval
        poll = self.server.poll
        last_beat = time.perf_counter()
        ended = False

        def send_batch(events: list[dict]) -> None:
            if not events:
                return
            payload = json.dumps(events, separators=(",", ":"))
            self.wfile.write(b"data: ")
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.write(b"\n\n")
            self.wfile.flush()

        try:
            while True:
                # Drain everything appended since last read from BOTH logs (trainer's
                # events.jsonl and the sidecar's panels.jsonl); tolerate a malformed or
                # partial line without killing the stream.
                try:
                    batch = reader.read_new(validate=False)
                except (json.JSONDecodeError, ValueError):
                    batch = []
                try:
                    batch = batch + panels.read_new(validate=False)
                except (json.JSONDecodeError, ValueError):
                    pass
                if batch:
                    send_batch(batch)
                    if any(e.get("type") == "run_end" for e in batch):
                        ended = True
                now = time.perf_counter()
                # Heartbeat comment keeps the connection (and, post-completion, the
                # final state) alive without triggering an EventSource reconnect.
                if now - last_beat > 10.0:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    last_beat = now
                time.sleep(flush_interval if batch else poll)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return  # client closed the tab
        finally:
            _ = ended

    # -- quieter logging --------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:
        # The default handler logs every request to stderr, which spams the console
        # the trainer is also printing to. Stay silent; errors still raise.
        pass


def serve(run_dir, *, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True, block: bool = True) -> DashboardServer:
    """Start the dashboard server for ``run_dir``. Returns the server (already
    serving in a background thread); blocks until interrupted when ``block``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    httpd = DashboardServer((host, port), run_dir)
    thread = threading.Thread(target=httpd.serve_forever, name="dashboard", daemon=True)
    thread.start()
    url = f"http://{host}:{port}/"
    print(f"[dashboard] serving {run_dir}  ->  {url}")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    if block:
        try:
            while thread.is_alive():
                thread.join(0.5)
        except KeyboardInterrupt:
            print("\n[dashboard] shutting down")
            httpd.shutdown()
    return httpd
