# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""`solver_watch.Watched`: the calls and failures of a search solver, which the
engine otherwise hides (it counts only the decisions a failed search reached:
redone by the built-in solver, or left to the blueprint)."""
import json
import threading
from fractions import Fraction

import numpy as np
import pytest

from parciver.solver_watch import Watched, malformed, write_atomic

# Two decision slots: slot 0 at node 0 (2 actions), slot 1 at node 2 (3 actions); 2 holdings.
JOB = {"board": [1, 2, 3, 4], "slot": [0, -1, 1, -1, -1, -1, -1], "child_start": [0, 2, 2, 5, 5, 5, 5, 5],
       "ranges": ([1.0, 1.0], [1.0, 1.0]), "want": [1, 0]}


def test_malformed_is_the_engines_check():
    good = [[0.2] * 6, [0.5] * 4]
    assert malformed(JOB, good) is None
    assert malformed(JOB, good[::-1]) is not None  # the rows follow `want`
    assert malformed(JOB, [np.full(6, 0.2, np.float32), np.zeros(4)]) is None  # arrays of any float dtype
    bad = {"one row too few": good[:1], "a short row": [[0.2] * 5, [0.5] * 4], "NaN": [[0.2] * 5 + [np.nan], [0.5] * 4],
           "inf": [[0.2] * 6, [np.inf, 0, 0, 0]], "negative": [[0.2] * 6, [0.5, -1e-300, 0.5, 0.5]],
           "not floats": [[0.2] * 6, ["a"] * 4], "nested": [[0.2] * 6, [[0.5, 0.5]] * 2], "None": None}
    for what, out in bad.items():
        assert malformed(JOB, out) is not None, what


# Answers in other forms, and whether the binding takes them (PyO3: a sequence,
# not a str, at both levels; each entry through `PyFloat_AsDouble`).
FORMS = {
    "tuples": (True, lambda out: tuple(tuple(r) for r in out)),
    "float32 arrays": (True, lambda out: [np.asarray(r, np.float32) for r in out]),
    "Fractions": (True, lambda out: [[Fraction(x) for x in r] for r in out]),
    "a bytes row (ints)": (True, lambda out: [bytes(len(out[0]))] + out[1:]),
    "a generator of rows": (False, lambda out: (r for r in out)),
    "a dict of rows": (False, lambda out: dict(enumerate(out))),
    "numeric strings": (False, lambda out: [[str(x) for x in out[0]]] + out[1:]),
    "an int too large for a float": (False, lambda out: [[2 ** 1024] + out[0][1:]] + out[1:]),
    "a masked row": (False, lambda out: [np.ma.array(out[0], mask=[1] + [0] * (len(out[0]) - 1))] + out[1:]),
}


@pytest.mark.filterwarnings("ignore:Warning. converting a masked element to nan")
@pytest.mark.parametrize("form", FORMS)
def test_malformed_takes_what_the_binding_takes(form):
    want, change = FORMS[form]
    out = change([[0.2] * 6, [0.5] * 4])
    assert (malformed(JOB, out) is None) == want


@pytest.mark.filterwarnings("ignore:Warning. converting a masked element to nan")
def test_the_engine_agrees_with_malformed():
    # One refused form per rule (the dict and the large int fail the same C
    # calls). rescue=False: a refused answer leaves the decision to the blueprint.
    pe = pytest.importorskip("parciver_engine")
    tv = pytest.importorskip("parciver.turn_values")
    agent = pe.PlayAgent.uniform("200bb")

    def play(solver):
        w = Watched(solver)
        res = dict(agent.with_search(from_turn=True, solver=w, rescue=False)
                   .match_against("agent", 12, 50, 100, 20000, seed=5, threads=1))
        # whether the engine played a searched strategy; the decisions the blueprint made in its place
        return w, res["a_sources"][3] > 0, res["a_search_failed"]

    for form in ("tuples", "float32 arrays", "Fractions", "a bytes row (ints)", "a generator of rows", "numeric strings",
                 "a masked row"):
        want, change = FORMS[form]
        w, used, failed = play(lambda job, change=change: change(tv._uniform_answer(job)))
        if want:  # taken: as many calls as the same numbers in lists of floats
            same, _, _ = play(lambda job, change=change: [[float(x) for x in r] for r in change(tv._uniform_answer(job))])
            ok = used and failed == [0, 0] and same.failures == 0 and (w.calls, w.failures) == (same.calls, 0)
        else:  # refused: every call
            ok = not used and failed[0] == 0 and failed[1] > 0 and w.failures == w.calls
        assert w.calls > 0 and ok, (form, used, failed, w.calls, w.last_error)


def test_counts_calls_and_failures_and_re_raises():
    def solver(job):
        if job["fail"]:
            raise RuntimeError("out of memory")
        return [[0.2] * 6, [0.5] * 4]

    w = Watched(solver)
    assert w(dict(JOB, fail=False)) == [[0.2] * 6, [0.5] * 4]
    with pytest.raises(RuntimeError, match="out of memory"):
        w(dict(JOB, board=[1, 2, 3, 4, 5], fail=True))
    assert (w.calls, w.failures) == (2, 1) and w.last_error == "RuntimeError: out of memory"
    assert sorted(w.seconds) == [4, 5] and all(len(v) == 1 for v in w.seconds.values())
    assert w.take() == (2, 1) and w.take() == (0, 0)
    w.check(5, 0, "chunk")
    with pytest.raises(SystemExit, match="1 of 2 search solves failed"):
        w.check(2, 1, "chunk")


def test_counts_concurrent_calls_exactly():
    w = Watched(lambda job: [[0.2] * 6, [0.5] * 4])
    threads = [threading.Thread(target=lambda: [w(dict(JOB, board=[0, 1, 2])) for _ in range(500)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert w.calls == 4000 and len(w.seconds[3]) == 4000 and w.failures == 0


def test_write_atomic_replaces_whole(tmp_path):
    p = tmp_path / "state.json"
    write_atomic(p, json.dumps({"a": 1}))
    write_atomic(p, json.dumps({"a": 2}))
    assert json.loads(p.read_text()) == {"a": 2} and [x.name for x in tmp_path.iterdir()] == ["state.json"]


def test_the_engine_hides_a_failing_solver_and_watched_shows_it():
    pe = pytest.importorskip("parciver_engine")
    tv = pytest.importorskip("parciver.turn_values")

    def broken(job):
        raise RuntimeError("no GPU")

    agent = pe.PlayAgent.uniform("200bb")  # 24 deals: a few reach the turn (8 reach none)

    def match(solver, **kw):
        w = Watched(solver) if solver else None
        res = dict(agent.with_search(from_turn=True, solver=w, **kw)
                   .match_against("agent", 24, 50, 100, 20000, seed=5, threads=1))
        return w, res

    w, res = match(broken, rescue=False)
    assert len(res["per_deal_bb"]) == 24  # the match ran: every failed search fell back to the blueprint
    assert w.calls > 0 and w.failures == w.calls and res["a_sources"][3] == res["b_sources"][3] == 0
    for side in "ab":  # counted by the engine, for both seats (B is the same agent)
        assert res[f"{side}_search_failed"][0] == 0 and res[f"{side}_search_failed"][1] > 0, side
    # (Each public state is tried once; before 1 Oct the engine retried a failed search per holding:
    # 2266 failed calls where a working solver gets 5.)

    # By default the built-in solver redoes a failed turn or river search: the match is the built-in search's.
    w, res = match(broken)
    _, built_in = match(None)
    assert w.calls > 0 and w.failures == w.calls and built_in["a_sources"][3] > 0
    assert res["per_deal_bb"] == built_in["per_deal_bb"]
    for side in "ab":
        assert res[f"{side}_sources"] == built_in[f"{side}_sources"], side
        assert res[f"{side}_search_failed"] == [built_in[f"{side}_sources"][3], 0], side
        assert built_in[f"{side}_search_failed"] == [0, 0], side

    ok, res = match(tv._uniform_answer)
    assert ok.calls > 0 and ok.failures == 0 and res["a_sources"][3] > 0
    assert res["a_search_failed"] == res["b_search_failed"] == [0, 0]

    def nan(job):  # an answer the engine refuses without an exception: counted all the same
        out = tv._uniform_answer(job)
        out[0][0] = float("nan")
        return out

    w, res = match(nan, rescue=False)
    assert w.calls > 0 and w.failures == w.calls and res["a_sources"][3] == 0 and "non-finite" in w.last_error
    assert res["a_search_failed"][1] > 0
