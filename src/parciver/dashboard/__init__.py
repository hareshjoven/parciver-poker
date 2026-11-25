"""The live training dashboard: a local, no-build-step viewer for the telemetry log.

``server`` is a standard-library HTTP server that serves a single-page app and
streams new telemetry events over Server-Sent Events, tailing
``runs/<id>/events.jsonl``. It is deliberately decoupled from training: it only ever
*reads* the append-only log (see :mod:`parciver.telemetry`), so it can never slow or
lag a solve. This is a LOCAL app, not a hosted page — Artifacts are sandboxed
and cannot read local live telemetry.
"""
