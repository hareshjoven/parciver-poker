"""Tests for the training-telemetry contract (Phase 3c, pre-B4).

The telemetry log is the sole channel B4 has for reporting a multi-day solve, so
these lock the contract hard: every event is versioned and schema-valid, the
writer round-trips through the reader byte-for-byte, malformed events are rejected
at the source, the tailing reader resumes correctly across appends (including a
half-written final line), and an abnormal exit still records a terminal event. The
resource sampler is checked to return the right shape and never raise.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from parciver import telemetry as T
from parciver.resources import ResourceSampler


# Synthetic 169 hand names so the core contract tests need no Rust extension.
FAKE_NAMES = [f"H{i}" for i in range(T.NUM_PREFLOP_HANDS)]


def base_config(**over):
    cfg = {
        "hand_names": list(FAKE_NAMES),
        "abstraction": {"preflop": 169, "flop": 10000, "turn": 10000, "river": 1950},
        "stack_bb": 100,
    }
    cfg.update(over)
    return cfg


def make_grid(actions=("fold", "call", "raise")):
    return {
        "actions": list(actions),
        "cells": [T.normalize_mix([1.0] * len(actions)) for _ in range(T.NUM_PREFLOP_HANDS)],
    }


# --- envelope + versioning -------------------------------------------------

def test_every_event_is_versioned_and_seq_monotonic(tmp_path):
    with T.RunWriter(tmp_path) as w:
        w.start(variant="linear-mccfr", game="nlhe-small", config=base_config())
        for i in range(5):
            w.metrics(iteration=i, it_per_s=10.0 * i)
        w.end(iteration=5)
    events = T.read_events(tmp_path)
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert all(e["v"] == T.SCHEMA_VERSION for e in events)
    assert all(isinstance(e["ts"], (int, float)) for e in events)


def test_all_event_types_round_trip(tmp_path):
    with T.RunWriter(tmp_path) as w:
        w.start(variant="v", game="g", config=base_config(), target_iters=1000)
        w.metrics(iteration=1, it_per_s=1.0, exploitability=0.5, mean_abs_regret=0.1,
                  strategy_l1_delta=0.2, infosets=10, hh_bb_per_100=3.3,
                  resources={"cpu_pct": 50.0, "ram_used_gb": 8.0, "threads": 16})
        w.checkpoint(iteration=1, range_grid=make_grid(),
                     action_freq={"P": {"f": 0.1, "c": 0.5, "r": 0.4}},
                     bet_size_freq={"F": {"0.5pot": 0.6, "1.0pot": 0.4}},
                     infosets_by_street={"P": 5, "F": 5})
        w.milestone("checkpoint_saved", "snapshot @1", iteration=1)
        w.end(iteration=1000, reason="completed", summary={"final_exploitability": 0.01})

    events = T.read_events(tmp_path)
    assert [e["type"] for e in events] == \
        ["run_start", "metrics", "checkpoint", "milestone", "run_end"]
    # Structured payloads survive verbatim.
    assert events[1]["resources"]["threads"] == 16
    assert len(events[2]["range_grid"]["cells"]) == T.NUM_PREFLOP_HANDS
    assert events[2]["action_freq"]["P"]["c"] == 0.5
    assert events[4]["summary"]["final_exploitability"] == 0.01


def test_writer_reader_bytewise_equivalent(tmp_path):
    """Reading back yields exactly the dicts the writer serialized."""
    emitted = []
    with T.RunWriter(tmp_path) as w:
        emitted.append(w.start(variant="v", game="g", config=base_config()))
        emitted.append(w.metrics(iteration=1, it_per_s=2.0))
        emitted.append(w.end(iteration=1))
    read = T.read_events(tmp_path)
    assert read == emitted


# --- validation rejects malformed events -----------------------------------

def test_validate_rejects_wrong_version():
    e = {"v": 999, "type": "metrics", "seq": 0, "ts": 1.0, "iteration": 1}
    with pytest.raises(T.SchemaError):
        T.validate_event(e)


def test_validate_rejects_unknown_type():
    e = {"v": T.SCHEMA_VERSION, "type": "bogus", "seq": 0, "ts": 1.0}
    with pytest.raises(T.SchemaError):
        T.validate_event(e)


def test_validate_rejects_missing_required_field():
    # metrics without iteration.
    e = {"v": T.SCHEMA_VERSION, "type": "metrics", "seq": 0, "ts": 1.0, "it_per_s": 1.0}
    with pytest.raises(T.SchemaError):
        T.validate_event(e)


def test_validate_rejects_bool_where_number_expected():
    e = {"v": T.SCHEMA_VERSION, "type": "metrics", "seq": 0, "ts": 1.0,
         "iteration": 1, "exploitability": True}
    with pytest.raises(T.SchemaError):
        T.validate_event(e)


def test_validate_rejects_bad_run_end_reason():
    e = {"v": T.SCHEMA_VERSION, "type": "run_end", "seq": 0, "ts": 1.0,
         "iteration": 1, "reason": "exploded"}
    with pytest.raises(T.SchemaError):
        T.validate_event(e)


def test_validate_rejects_bad_hand_names_length():
    cfg = base_config(hand_names=["only", "three", "names"])
    e = {"v": T.SCHEMA_VERSION, "type": "run_start", "seq": 0, "ts": 1.0,
         "run_id": "r", "variant": "v", "game": "g", "config": cfg, "target_iters": None}
    with pytest.raises(T.SchemaError):
        T.validate_event(e)


def test_validate_rejects_range_grid_wrong_row_count():
    grid = {"actions": ["f", "c", "r"], "cells": [[0.2, 0.3, 0.5]] * 10}  # not 169
    e = {"v": T.SCHEMA_VERSION, "type": "checkpoint", "seq": 0, "ts": 1.0,
         "iteration": 1, "range_grid": grid}
    with pytest.raises(T.SchemaError):
        T.validate_event(e)


def test_validate_rejects_range_grid_row_not_summing_to_one():
    cells = [[0.5, 0.3, 0.1] for _ in range(T.NUM_PREFLOP_HANDS)]  # sums to 0.9
    grid = {"actions": ["f", "c", "r"], "cells": cells}
    e = {"v": T.SCHEMA_VERSION, "type": "checkpoint", "seq": 0, "ts": 1.0,
         "iteration": 1, "range_grid": grid}
    with pytest.raises(T.SchemaError):
        T.validate_event(e)


def test_range_grid_allows_all_zero_unvisited_row():
    cells = [T.normalize_mix([1, 1, 1]) for _ in range(T.NUM_PREFLOP_HANDS)]
    cells[0] = [0.0, 0.0, 0.0]  # an unvisited hand
    grid = {"actions": ["f", "c", "r"], "cells": cells}
    e = {"v": T.SCHEMA_VERSION, "type": "checkpoint", "seq": 0, "ts": 1.0,
         "iteration": 1, "range_grid": grid}
    T.validate_event(e)  # must not raise


def test_validate_rejects_unknown_street_key():
    e = {"v": T.SCHEMA_VERSION, "type": "checkpoint", "seq": 0, "ts": 1.0,
         "iteration": 1, "infosets_by_street": {"Z": 5}}
    with pytest.raises(T.SchemaError):
        T.validate_event(e)


def test_writer_validates_before_writing(tmp_path):
    """A bad emit raises and does not leave a partial line on disk."""
    w = T.RunWriter(tmp_path)
    w.start(variant="v", game="g", config=base_config())
    with pytest.raises(T.SchemaError):
        # infosets must be an int, not a float.
        w.metrics(iteration=1, infosets=3.5)
    w.close()
    events = T.read_events(tmp_path)
    assert [e["type"] for e in events] == ["run_start"]  # the bad metrics never landed


# --- writer lifecycle ------------------------------------------------------

def test_start_must_be_first_and_once(tmp_path):
    w = T.RunWriter(tmp_path)
    w.start(variant="v", game="g", config=base_config())
    with pytest.raises(RuntimeError):
        w.start(variant="v", game="g", config=base_config())
    w.close()


def test_no_events_after_end(tmp_path):
    w = T.RunWriter(tmp_path)
    w.start(variant="v", game="g", config=base_config())
    w.end(iteration=1)
    with pytest.raises(RuntimeError):
        w.metrics(iteration=2)
    w.close()


def test_context_manager_records_error_on_exception(tmp_path):
    with pytest.raises(ValueError):
        with T.RunWriter(tmp_path) as w:
            w.start(variant="v", game="g", config=base_config())
            w.metrics(iteration=1)
            raise ValueError("boom")
    events = T.read_events(tmp_path)
    assert events[-1]["type"] == "run_end"
    assert events[-1]["reason"] == "error"
    assert "boom" in events[-1]["summary"]["note"]


def test_context_manager_records_interrupted_when_no_end(tmp_path):
    with T.RunWriter(tmp_path) as w:
        w.start(variant="v", game="g", config=base_config())
        w.metrics(iteration=1)
        # exit the block without calling end()
    events = T.read_events(tmp_path)
    assert events[-1]["type"] == "run_end"
    assert events[-1]["reason"] == "interrupted"


# --- reader tailing --------------------------------------------------------

def test_reader_read_new_resumes_across_appends(tmp_path):
    w = T.RunWriter(tmp_path)
    w.start(variant="v", game="g", config=base_config())
    r = T.RunReader(tmp_path)
    first = r.read_new()
    assert [e["type"] for e in first] == ["run_start"]
    assert r.read_new() == []  # nothing new yet

    w.metrics(iteration=1)
    w.metrics(iteration=2)
    second = r.read_new()
    assert [e["iteration"] for e in second] == [1, 2]
    w.end(iteration=2)
    assert [e["type"] for e in r.read_new()] == ["run_end"]
    w.close()


def test_reader_ignores_partial_final_line(tmp_path):
    """A line without a trailing newline (writer mid-append) is not yielded until
    completed."""
    path = tmp_path / "events.jsonl"
    good = json.dumps({"v": T.SCHEMA_VERSION, "type": "metrics", "seq": 0,
                       "ts": 1.0, "iteration": 1})
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(good + "\n")
        fh.write('{"partial": ')  # half a line, no newline
    r = T.RunReader(tmp_path)
    got = r.read_new(validate=False)
    assert len(got) == 1 and got[0]["iteration"] == 1
    # Complete the line; the reader should now pick it up.
    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write('"x", "v": %d, "type": "metrics", "seq": 1, "ts": 2.0, "iteration": 2}\n'
                 % T.SCHEMA_VERSION)
    got2 = r.read_new(validate=False)
    assert len(got2) == 1 and got2[0]["iteration"] == 2


def test_read_events_empty_when_no_log(tmp_path):
    assert T.read_events(tmp_path) == []


def test_tail_stops_on_run_end(tmp_path):
    w = T.RunWriter(tmp_path)
    w.start(variant="v", game="g", config=base_config())

    seen = []

    def consume():
        for e in T.RunReader(tmp_path).tail(poll=0.01):
            seen.append(e)

    th = threading.Thread(target=consume, daemon=True)
    th.start()
    time.sleep(0.05)
    w.metrics(iteration=1)
    time.sleep(0.05)
    w.end(iteration=1)
    w.close()
    th.join(timeout=2.0)
    assert not th.is_alive(), "tail did not stop on run_end"
    assert seen[-1]["type"] == "run_end"


# --- resource sampler ------------------------------------------------------

def test_resource_sampler_shape_and_never_raises():
    s = ResourceSampler(threads=16)
    r = s.sample()
    assert set(r) == {"cpu_pct", "ram_used_gb", "ram_total_gb", "gpu_pct",
                      "gpu_mem_used_gb", "gpu_mem_total_gb", "threads"}
    assert r["threads"] == 16
    # A second sample after a tiny delay should produce a CPU reading on Windows.
    time.sleep(0.05)
    r2 = s.sample()
    for k in ("cpu_pct", "ram_used_gb", "ram_total_gb"):
        assert r2[k] is None or isinstance(r2[k], (int, float))


def test_resource_sample_validates_as_telemetry_resources():
    """Whatever the sampler returns must satisfy the contract's resources shape."""
    r = ResourceSampler(threads=8).sample()
    e = {"v": T.SCHEMA_VERSION, "type": "metrics", "seq": 0, "ts": 1.0,
         "iteration": 1, "resources": r}
    T.validate_event(e)  # must not raise


# --- helpers ---------------------------------------------------------------

def test_normalize_mix():
    assert T.normalize_mix([1, 1, 2]) == [0.25, 0.25, 0.5]
    assert T.normalize_mix([0, 0, 0]) == [0.0, 0.0, 0.0]  # unvisited stays zero


def test_preflop_hand_names_matches_engine():
    """The real 169 names come from the engine and align to preflop_index order."""
    engine = pytest.importorskip("parciver_engine")
    names = T.preflop_hand_names()
    assert len(names) == T.NUM_PREFLOP_HANDS
    assert names[0] == engine.preflop_name(0)
    assert "AA" in names and "AKs" in names and "72o" in names
