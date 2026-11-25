"""The rating's receipts (RATING.md, Amendment 2): states, tamper detection, recomputation, the headline.
Engine-free except the last test (PARCIVER_RATING_SMOKE=1: the full smoke pipeline on the real engine)."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from parciver import rating as R

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rating_cli", ROOT / "scripts" / "rating.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

IDS = ["A0", "A1", "A2"]
N = 12_000  # candidate deals per anchor
NR = 2 * N  # the reference's (Amendment 2: the reference plays more deals than a candidate)
# Calibration payoffs: a cycle (0 loses to 1, 1 to 2, 2 to 0), so every anchor carries weight.
CAL = {(0, 1): -10.0, (1, 2): -20.0, (0, 2): 30.0}
A0_ROW = {"A0": 0.0, "A1": CAL[(0, 1)], "A2": CAL[(0, 2)]}  # A0's payoff against each anchor
ID1 = {"id": "E1+C1", "sha256": "E1"}
ID2 = {"id": "E2+C1", "sha256": "E2"}


def make_pool(d: Path, monkeypatch, version: str = "pool-test"):
    d.mkdir(parents=True, exist_ok=True)
    defn = {"version": version, "stakes": {"sb": 50, "bb": 100, "stack": 20_000}, "hands_per_deal": 2,
            "abstraction": str(d), "anchors": [{"id": j, "kind": "leak", "of": "A0", "family": "nit", "strength": 0}
                                               for j in IDS],
            "abstraction_sha256": {}, "calibration": {"deals": N, "seed_base": 600_000},
            "rating": {"deals": N, "deals_weighted": NR, "seed_base": 700_000}, "lbr": {"deals": 0, "seed": 1},
            "replay": {"chunk": 1_000, "count": 2}, "chunks": 1}
    defn["pool_hash"] = cli.pool_hash(defn)
    defn["created_utc"] = "2026-10-03T00:00:00"
    (d / "pool.json").write_text(json.dumps(defn))
    monkeypatch.setattr(cli, "engine_identity", lambda: ID1)
    pool = cli.Pool(d)
    (d / "cal").mkdir()
    k = len(IDS)
    A = np.zeros((k, k))
    for (i, j), m in CAL.items():
        for a, b, v in ((i, j, m), (j, i, -m)):
            per = np.full(N, v)
            shas = cli.save_arrays(d / "cal" / f"pair-{IDS[a]}-{IDS[b]}.npz", {"per": per})
            cli.write_new(d / "cal" / f"pair-{IDS[a]}-{IDS[b]}.json",
                          {"pair": [IDS[a], IDS[b]], "seed": pool.cal_seed(a, b), "deals": N, "arrays": shas,
                           "mean": v, "engine": ID1})
        A[i, j], A[j, i] = m, -m
    p = R.nash(A)
    (d / "calibration.json").write_text(json.dumps({
        "pool_hash": defn["pool_hash"], "state": "accepted", "anchors": IDS, "p": p.tolist(), "A_mbb_per_hand": A.tolist(),
        "se": np.zeros((k, k)).tolist(), "nash_check": 0.0, "acceptance": {"C4_role_symmetry": {"ok": True}}}))
    return pool, {j: float(x) for j, x in zip(IDS, p)}


def blueprint_file(pool, name: str, content: bytes) -> dict:
    f = pool.dir / f"{name}.bin"
    f.write_bytes(content)
    return {"path": str(f), "sha256": cli.sha256_file(f), "search": None, "translation": "snap"}


def receipt(pool, cand, arrays, started, ident=ID1, **extra):
    (pool.dir / "receipts").mkdir(exist_ok=True)
    stem = f"{started.replace(':', '')}-{cand['label']}"
    shas = cli.save_arrays(pool.dir / "receipts" / f"{stem}.npz", arrays)
    rows = {j: {"a_search_failed": [0, 0], "b_search_failed": [0, 0], "a_sources": [1, 0, 0, 0], "seconds": 0,
                "seed": pool.seed(j), "deals": len(arrays[j])} for j in IDS}
    r = {"schema": "parciver-rating-receipt-2", "pool_hash": pool.defn["pool_hash"], "candidate": cand,
         "fingerprint": cli.fingerprint(cand), "anchors": rows, "lbr": None, "arrays": shas, "engine": ident,
         "git": {}, "started_utc": started, **extra}
    path = pool.dir / "receipts" / f"{stem}.json"
    cli.write_new(path, r)
    return path


def replayed(pool, *paths, ident=ID1, ok=True):
    for p in paths:
        cli.append_verification(pool, {"receipt": p.name, "receipt_sha": cli.receipt_sha(p), "identity": ident["id"],
                                       "ok": ok, "detail": "" if ok else "A1 DIFFERENT", "utc": "t"})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    pool, w = make_pool(tmp_path, monkeypatch)
    rng = np.random.default_rng(3)
    luck = {j: rng.normal(0, 400, NR) for j in IDS}
    # A0's means on the rating seeds agree with the calibration's (C2 is re-derived); every anchor carries
    # weight, so everyone plays NR deals against each (the protocol is re-checked).
    ref = {j: luck[j] + A0_ROW[j] + rng.normal(0, 30, NR) for j in IDS}
    x = {j: luck[j] + A0_ROW[j] + mu + rng.normal(0, 30, NR) for j, mu in zip(IDS, (20.0, 8.0, -4.0))}
    pr = receipt(pool, {"label": cli.REFERENCE, "anchor": "A0", "path": None, "sha256": None}, ref, "2026-10-03T00:00:00")
    cand = {"label": "cand", **blueprint_file(pool, "cand", b"blueprint one")}
    px = receipt(pool, cand, x, "2026-10-03T00:00:01")
    replayed(pool, pr, px)
    return dict(pool=pool, w=w, pr=pr, px=px, ref=ref, x=x, cand=cand, mp=monkeypatch, rng=rng)


def rows(pool):
    return {r["receipt"]: r for r in cli.build_view(pool)["rows"]}


def set_current(pool, label):
    assert cli.cmd_current(type("A", (), {"dir": str(pool.dir), "label": label})()) == 0


def test_recomputed_rating_matches_the_formula(setup):
    s = setup
    r = rows(s["pool"])
    ref = r[s["pr"].name]
    assert ref["rating"]["pr"] == 1500.0 and ref["state"] == "VERIFIED"
    want = R.paired(s["x"], s["ref"], s["w"])
    got = r[s["px"].name]
    assert got["rating"] == want and got["state"] == "VERIFIED" and min(s["w"].values()) > 0.1
    true = 1500 + sum(s["w"][j] * mu for j, mu in zip(IDS, (20.0, 8.0, -4.0)))
    assert abs(got["rating"]["pr"] - true) < 4 * want["se"]


def test_a_receipt_is_stale_until_replayed(setup):
    s = setup
    x2 = {j: v + 1.0 for j, v in s["x"].items()}
    p2 = receipt(s["pool"], {**s["cand"], "label": "cand2"}, x2, "2026-10-03T00:00:02")
    r = rows(s["pool"])[p2.name]
    assert r["state"] == "STALE" and "rating" in r  # shown, never the headline
    replayed(s["pool"], p2)
    assert rows(s["pool"])[p2.name]["state"] == "VERIFIED"


def test_engine_change_makes_receipts_stale_until_replayed(setup):
    s = setup
    s["mp"].setattr(cli, "engine_identity", lambda: ID2)
    r = rows(s["pool"])
    assert r[s["px"].name]["state"] == "STALE" and r[s["pr"].name]["state"] == "STALE"
    replayed(s["pool"], s["pr"], s["px"], ident=ID2)
    assert rows(s["pool"])[s["px"].name]["state"] == "VERIFIED"


def test_a_failed_replay_is_invalid_and_poisons_dependent_ratings(setup):
    s = setup
    replayed(s["pool"], s["pr"], ok=False)
    r = rows(s["pool"])
    assert r[s["pr"].name]["state"] == "INVALID"
    assert r[s["px"].name]["state"] == "INVALID" and "reference" in r[s["px"].name]["why"]
    assert "rating" not in r[s["px"].name]


def test_an_unreadable_verify_line_trusts_no_replay(setup):
    s = setup
    with open(s["pool"].dir / "verify.jsonl", "a") as f:
        f.write('{"receipt_sha": "trunc\n')
    assert rows(s["pool"])[s["px"].name]["state"] == "STALE"


def test_one_edited_per_deal_value_is_caught(setup):
    s = setup
    npz = s["px"].with_suffix(".npz")
    with np.load(npz) as z:
        arrs = {k: z[k].copy() for k in z.files}
    arrs["A1"][123] = np.nextafter(arrs["A1"][123], np.inf)
    npz.unlink()
    np.savez_compressed(npz.with_name(npz.stem + ".tmp.npz"), **arrs)
    os.replace(npz.with_name(npz.stem + ".tmp.npz"), npz)
    r = rows(s["pool"])[s["px"].name]
    assert r["state"] == "INVALID" and "sha256" in r["why"] and "rating" not in r


def test_missing_failed_search_counts_are_invalid(setup):
    s = setup
    r = json.loads(s["px"].read_text())
    del r["anchors"]["A0"]["a_search_failed"]
    s["px"].write_text(json.dumps(r))
    assert rows(s["pool"])[s["px"].name]["state"] == "INVALID"


def test_a_wrong_seed_or_deal_count_is_invalid(setup):
    s = setup
    r = json.loads(s["px"].read_text())
    r["anchors"]["A2"]["seed"] += 1
    s["px"].write_text(json.dumps(r))
    assert "seed" in rows(s["pool"])[s["px"].name]["why"]


def test_a_corrupt_receipt_is_an_invalid_row_and_trusts_no_rating(setup):
    s = setup
    (s["pool"].dir / "receipts" / "zz-broken.json").write_text('{"pool_hash": ')
    r = rows(s["pool"])
    # It could have been the newest reference or the newest receipt of a configuration: nothing is VERIFIED.
    assert r["zz-broken.json"]["state"] == "INVALID"
    assert r[s["px"].name]["state"] == "STALE" and "cannot be read" in r[s["px"].name]["why"]
    assert "rating" in r[s["px"].name]  # still shown, never the headline


def test_a_changed_candidate_file_is_invalid(setup):
    s = setup
    Path(s["cand"]["path"]).write_bytes(b"blueprint two")
    assert "changed" in rows(s["pool"])[s["px"].name]["why"]


def test_reserved_labels_only_name_the_pools_anchors(setup):
    s = setup
    fake = receipt(s["pool"], {**s["cand"], "label": "A1"}, s["x"], "2026-10-03T00:00:05")
    replayed(s["pool"], fake)
    r = rows(s["pool"])[fake.name]
    assert r["state"] == "INVALID" and "reserved" in r["why"]


def test_the_newest_reference_is_used_and_never_an_older_one(setup):
    s = setup
    # A newer reference that does not follow the protocol (candidate deal counts): no fallback to the older one.
    short = {j: v[:N] for j, v in s["ref"].items()}
    p = receipt(s["pool"], {"label": cli.REFERENCE, "anchor": "A0", "path": None, "sha256": None}, short,
                "2026-10-03T00:00:09")
    replayed(s["pool"], p)
    r = rows(s["pool"])
    assert r[p.name]["state"] == "INVALID" and "protocol" in r[p.name]["why"]
    assert r[s["px"].name]["state"] == "INVALID" and "reference" in r[s["px"].name]["why"]


def test_a_reference_that_is_not_A0_is_refused(setup):
    s = setup
    p = receipt(s["pool"], {"label": cli.REFERENCE, **blueprint_file(s["pool"], "other", b"x")}, s["ref"],
                "2026-10-03T00:00:09")
    replayed(s["pool"], p)
    assert rows(s["pool"])[s["px"].name]["state"] == "INVALID"


def test_an_edited_calibration_rates_nothing(setup):
    s = setup
    f = s["pool"].dir / "calibration.json"
    c = json.loads(f.read_text())
    c["p"] = [0.5, 0.25, 0.25]
    f.write_text(json.dumps(c))
    set_current(s["pool"], "cand")
    v = cli.build_view(s["pool"])
    assert v["calibration_error"] and v["headline"] is None
    assert all(r["state"] == "INVALID" and "rating" not in r for r in v["rows"])


def test_an_edited_pool_definition_is_refused(setup):
    s = setup
    f = s["pool"].dir / "pool.json"
    d = json.loads(f.read_text())
    d["rating"]["deals"] = 5
    f.write_text(json.dumps(d))
    with pytest.raises(R.RatingError):
        cli.Pool(s["pool"].dir)


def test_headline_is_only_the_current_bots_verified_rating(setup):
    s = setup
    pool = s["pool"]
    assert cli.build_view(pool)["headline"] is None  # no current bot
    set_current(pool, "cand")
    v = cli.build_view(pool)
    assert v["headline"]["receipt"] == s["px"].name and v["headline"]["rating"] == R.paired(s["x"], s["ref"], s["w"])
    assert sum(bool(r.get("headline")) for r in v["rows"]) == 1
    s["mp"].setattr(cli, "engine_identity", lambda: ID2)
    v = cli.build_view(pool)
    assert v["headline"] is None and "STALE" in v["headline_why"]  # stale: never shown as the headline


def test_the_headline_is_the_configuration_not_the_label(setup):
    s = setup
    pool = s["pool"]
    set_current(pool, "cand")
    # The same blueprint rated again under another label is the same configuration: the newest receipt wins.
    x2 = {j: v + 0.5 for j, v in s["x"].items()}
    p2 = receipt(pool, {**s["cand"], "label": "cand-again"}, x2, "2026-10-03T00:00:03")
    v = cli.build_view(pool)
    assert v["headline"] is None and "STALE" in v["headline_why"]  # newest is not replayed yet
    replayed(pool, p2)
    assert cli.build_view(pool)["headline"]["receipt"] == p2.name


def test_a_label_only_current_file_still_resolves(setup):
    s = setup
    (s["pool"].dir / "current.json").write_text(json.dumps({"label": "cand"}))
    assert cli.build_view(s["pool"])["headline"]["receipt"] == s["px"].name


def test_change_against_the_previous_bot_is_paired(setup):
    s = setup
    pool = s["pool"]
    set_current(pool, "cand")
    x2 = {j: v + 3.0 + s["rng"].normal(0, 5, NR) for j, v in s["x"].items()}
    nxt = {"label": "next", **blueprint_file(pool, "next", b"blueprint three")}
    p2 = receipt(pool, nxt, x2, "2026-10-03T00:00:04")
    replayed(pool, p2)
    set_current(pool, "next")
    v = cli.build_view(pool)
    assert v["headline"]["receipt"] == p2.name
    want = R.change(x2, s["x"], s["w"])
    assert v["change"]["mean"] == want["mean"] and v["change"]["from"] == "cand"
    assert want["lo"] < 3.0 < want["hi"] and want["se"] < 0.1


def test_receipts_are_never_overwritten(setup):
    with pytest.raises(R.RatingError):
        cli.write_new(setup["px"], {})


def edit_receipt(path, fn):
    r = json.loads(path.read_text())
    fn(r)
    path.write_text(json.dumps(r))


def rewrite_pair(pool, a, b, per):
    for ext in (".json", ".npz"):
        (pool.dir / "cal" / f"pair-{a}-{b}{ext}").unlink()
    i, j = IDS.index(a), IDS.index(b)
    shas = cli.save_arrays(pool.dir / "cal" / f"pair-{a}-{b}.npz", {"per": per})
    cli.write_new(pool.dir / "cal" / f"pair-{a}-{b}.json", {"pair": [a, b], "seed": pool.cal_seed(i, j), "deals": N,
                                                             "arrays": shas, "mean": float(per.mean()), "engine": ID1})


def test_verify_stale_only_reaches_verify_all(setup, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "verify_all", lambda pool, threads, only_stale, name=None: seen.update(
        only_stale=only_stale, name=name) or 0)
    assert cli.main(["verify", "--dir", str(setup["pool"].dir), "--stale-only"]) == 0
    assert seen == {"only_stale": True, "name": None}
    assert cli.main(["verify", "--dir", str(setup["pool"].dir), "--receipt", "x.json"]) == 0
    assert seen == {"only_stale": False, "name": "x.json"}


def test_a_receipt_of_another_pool_definition_trusts_no_rating(setup):
    s = setup
    f = s["pool"].dir / "receipts" / "zz-foreign.json"
    f.write_text(s["px"].read_text().replace(s["pool"].defn["pool_hash"], "f" * 64))
    r = rows(s["pool"])
    assert r[f.name]["state"] == "INVALID" and "another pool" in r[f.name]["why"]
    assert r[s["px"].name]["state"] == "STALE" and r[s["pr"].name]["state"] == "STALE"


@pytest.mark.parametrize("edit", [
    lambda r: r.update(anchors=list(r["anchors"])),
    lambda r: r.update(engine="E1+C1"),
    lambda r: r["candidate"].update(label=None),
    lambda r: r.update(lbr=[1, 2]),
    lambda r: r["anchors"]["A1"].update(a_search_failed=[0.5, 0]),
    lambda r: r["anchors"].update(A2=7),
])
def test_a_malformed_receipt_is_an_invalid_row_not_a_crash(setup, edit):
    s = setup
    set_current(s["pool"], "cand")
    edit_receipt(s["px"], edit)
    v = cli.build_view(s["pool"])
    r = {x["receipt"]: x for x in v["rows"]}
    assert r[s["px"].name]["state"] == "INVALID" and "rating" not in r[s["px"].name]
    assert r[s["pr"].name]["state"] == "VERIFIED" and v["headline"] is None
    cli.render(v)
    json.dumps(v)


def test_an_off_protocol_candidate_is_invalid(setup):
    s = setup
    short = {j: (v[:N] if j == "A1" else v) for j, v in s["x"].items()}
    p = receipt(s["pool"], {**s["cand"], "label": "short"}, short, "2026-10-03T00:00:06")
    replayed(s["pool"], p)
    r = rows(s["pool"])[p.name]
    assert r["state"] == "INVALID" and "protocol" in r["why"] and "rating" not in r


def test_an_undecodable_verify_line_trusts_no_replay(setup):
    s = setup
    with open(s["pool"].dir / "verify.jsonl", "ab") as f:
        f.write(b'\xff\xfe{"ok": true}\n')
    r = rows(s["pool"])[s["px"].name]
    assert r["state"] == "STALE" and "unreadable" in r["why"]


@pytest.mark.parametrize("ok", ["yes", 1, 0])
def test_a_verify_line_without_a_boolean_verdict_trusts_no_replay(setup, ok):
    s = setup
    cli.append_verification(s["pool"], {"receipt": "x", "receipt_sha": "x", "identity": ID1["id"], "ok": ok})
    assert rows(s["pool"])[s["px"].name]["state"] == "STALE"


def test_a_refused_replay_is_stale_not_invalid(setup):
    s = setup
    x2 = {j: v + 1.0 for j, v in s["x"].items()}
    p2 = receipt(s["pool"], {**s["cand"], "label": "cand2"}, x2, "2026-10-03T00:00:02")
    cli.append_verification(s["pool"], {"receipt": p2.name, "receipt_sha": cli.receipt_sha(p2), "identity": ID1["id"],
                                        "ok": None, "detail": "cannot replay: OSError: no file", "utc": "t"})
    r = rows(s["pool"])[p2.name]
    assert r["state"] == "STALE" and "refused" in r["why"]
    replayed(s["pool"], p2)
    assert rows(s["pool"])[p2.name]["state"] == "VERIFIED"


def test_verify_counts_refusals_and_unreadable_files(setup, monkeypatch):
    s = setup
    x2 = {j: v + 1.0 for j, v in s["x"].items()}
    p2 = receipt(s["pool"], {**s["cand"], "label": "cand2"}, x2, "2026-10-03T00:00:02")

    def refuse(pool, path, r, threads, cur):
        raise R.RatingError("the blueprint is gone")
    monkeypatch.setattr(cli, "replay", refuse)
    assert cli.verify_all(s["pool"], 1, only_stale=True) == 1  # only p2 is STALE; refused
    assert "refused" in rows(s["pool"])[p2.name]["why"]
    assert cli.verify_all(s["pool"], 1, only_stale=False, name="missing.json") == 1
    monkeypatch.setattr(cli, "replay", lambda pool, path, r, threads, cur: {"ok": True, "detail": "t", "kind": "t"})
    (s["pool"].dir / "receipts" / "zz-broken.json").write_text("{")
    assert cli.verify_all(s["pool"], 1, only_stale=False) == 1  # the broken file; three replays pass
    monkeypatch.setattr(cli, "replay", lambda pool, path, r, threads, cur: {"ok": False, "detail": "t", "kind": "t"})
    assert cli.verify_all(s["pool"], 1, only_stale=False, name=p2.name) == 1
    assert rows(s["pool"])[p2.name]["state"] == "INVALID"


def test_a_swapped_pair_that_does_not_negate_rates_nothing(setup):
    s = setup
    per = np.full(N, 10.0)
    per[5] = 9.0  # one deal of A1 vs A0 is not A0 vs A1's negation
    rewrite_pair(s["pool"], "A1", "A0", per)
    v = cli.build_view(s["pool"])
    assert "C4" in v["calibration_error"] and v["acceptance"] is None
    assert all(r["state"] == "INVALID" and "rating" not in r for r in v["rows"])


def test_edited_standard_errors_rate_nothing(setup):
    s = setup
    f = s["pool"].dir / "calibration.json"
    c = json.loads(f.read_text())
    c["se"][0][1] = c["se"][1][0] = 1.0
    f.write_text(json.dumps(c))
    v = cli.build_view(s["pool"])
    assert "standard errors" in v["calibration_error"]


def test_the_acceptance_shown_is_re_derived(setup):
    s = setup
    f = s["pool"].dir / "calibration.json"
    c = json.loads(f.read_text())
    c["acceptance"] = {"C4_role_symmetry": {"ok": False}}  # the file's own claims are never shown
    f.write_text(json.dumps(c))
    acc = cli.build_view(s["pool"])["acceptance"]
    assert all(acc[k]["ok"] is True for k in ("C1_replay", "C2_seed_consistency", "C3_nash", "C4_role_symmetry"))
    assert acc["C4_role_symmetry"]["exact_pairs"] == 3


def test_a_reference_that_fails_seed_consistency_is_invalid(setup):
    s = setup
    off = {j: v + (50.0 if j == "A1" else 0.0) for j, v in s["ref"].items()}
    p = receipt(s["pool"], {"label": cli.REFERENCE, "anchor": "A0", "path": None, "sha256": None}, off,
                "2026-10-03T00:00:09")
    replayed(s["pool"], p)
    v = cli.build_view(s["pool"])
    r = {x["receipt"]: x for x in v["rows"]}
    assert r[p.name]["state"] == "INVALID" and "C2" in r[p.name]["why"]
    assert r[s["px"].name]["state"] == "INVALID" and "reference" in r[s["px"].name]["why"]
    assert v["acceptance"]["C2_seed_consistency"]["anchors"]["A1"]["ok"] is False
    assert any("C2" in x for x in cli.audit_problems(s["pool"], v))


def test_rate_refuses_a_one_deal_lbr_before_playing(setup):
    s = setup
    a = type("A", (), {"dir": str(s["pool"].dir), "search": "none", "label": "new", "candidate": s["cand"]["path"],
                       "translation": "snap", "lbr_deals": 1})()
    with pytest.raises(R.RatingError, match="lbr-deals"):
        cli.cmd_rate(a)


def test_a_new_receipt_is_replayed_in_a_new_process(setup, monkeypatch):
    s = setup
    s["pool"]._bp["x"] = object()
    seen = []
    monkeypatch.setattr(cli.subprocess, "run", lambda argv, **kw: seen.append(argv) or type("P", (), {"returncode": 0}))
    assert cli.replay_in_new_process(s["pool"], s["px"], 3) == 0
    assert s["pool"]._bp == {}
    argv = seen[0]
    assert argv[0] == sys.executable and Path(argv[1]) == ROOT / "scripts" / "rating.py"
    assert argv[2:] == ["verify", "--dir", str(s["pool"].dir), "--receipt", s["px"].name, "--threads", "3"]


def test_appends_to_the_replay_log_wait_for_the_lock(setup):
    import threading
    import time
    s = setup
    f = s["pool"].dir / "verify.jsonl"
    before = len(cli.verifications(s["pool"])[0])
    entry = {"receipt": "x", "receipt_sha": "y" * 64, "identity": "z", "ok": True, "detail": "d" * 20_000}
    with cli.locked(f):
        t = threading.Thread(target=cli.append_verification, args=(s["pool"], entry))
        t.start()
        time.sleep(0.5)
        assert t.is_alive() and len(cli.verifications(s["pool"])[0]) == before
    t.join(10)
    assert not t.is_alive()
    threads = [threading.Thread(target=lambda: [cli.append_verification(s["pool"], entry) for _ in range(25)])
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    ent, bad = cli.verifications(s["pool"])
    assert bad == [] and len(ent) == before + 101


def c6_pools(tmp_path, monkeypatch):
    """pool-v2 and pool-v1 side by side, with the same calibration and the same reference and anchor rows."""
    monkeypatch.setattr(cli, "C6_DEALS", N)  # the real floor is 1,000,000 deals; these rows have NR
    rng = np.random.default_rng(5)
    luck = {j: rng.normal(0, 400, NR) for j in IDS}
    ref = {j: luck[j] + A0_ROW[j] for j in IDS}
    out = {}
    for v in ("pool-v1", "pool-v2"):
        pool, w = make_pool(tmp_path / v, monkeypatch, version=v)
        ps = [receipt(pool, {"label": cli.REFERENCE, "anchor": "A0", "path": None, "sha256": None}, ref,
                      "2026-10-03T00:00:00")]
        for k, j in enumerate(IDS[1:], 1):
            ps.append(receipt(pool, {"label": j, "anchor": j, "path": None, "sha256": None},
                              {i: x + 5.0 * k for i, x in ref.items()}, f"2026-10-03T00:00:0{k}"))
        replayed(pool, *ps)
        out[v] = (pool, ps)
    return out


def crosscheck(v2, v1):
    return cli.cmd_crosscheck(type("A", (), {"dir": str(v2.dir), "other": str(v1.dir)})())


def audit(pool):
    return cli.cmd_show(type("A", (), {"dir": str(pool.dir), "audit": True})())


def test_pool_v2_needs_c6_before_anything_is_verified(tmp_path, monkeypatch, capsys):
    pools = c6_pools(tmp_path, monkeypatch)
    v2, ps = pools["pool-v2"]
    v1 = pools["pool-v1"][0]
    set_current(v1, "A2")
    set_current(v2, "A2")
    v = cli.build_view(v2)
    assert v["c6"]["state"] == "STALE" and all(r["state"] == "STALE" and "C6" in r["why"] for r in v["rows"])
    assert cli.build_view(v1)["c6"] is None and audit(v1) == 0
    assert audit(v2) == 1 and "C6" in capsys.readouterr().out
    assert crosscheck(v2, v1) == 0
    v = cli.build_view(v2)
    assert v["c6"]["state"] == "VERIFIED" and all(r["state"] == "VERIFIED" for r in v["rows"])
    json.dumps(v)
    capsys.readouterr()
    assert audit(v2) == 0 and "AUDIT OK" in capsys.readouterr().out
    # A newer pool-v1 receipt of an anchor: C6 was run on other files.
    x = {i: np.full(NR, 1.0) for i in IDS}
    p = receipt(v1, {"label": "A1", "anchor": "A1", "path": None, "sha256": None}, x, "2026-10-03T00:00:09")
    replayed(v1, p)
    v = cli.build_view(v2)
    assert v["c6"]["state"] == "STALE" and "other files" in v["c6"]["why"]
    assert all(r["state"] == "STALE" for r in v["rows"])
    # ... and C6 re-run on them fails: pool-v2's A1 row differs from pool-v1's newest.
    assert crosscheck(v2, v1) == 1
    v = cli.build_view(v2)
    assert v["c6"]["state"] == "INVALID" and "A1" in v["c6"]["why"]
    assert all(r["state"] == "INVALID" and "rating" not in r for r in v["rows"])
    assert any("A1" in x for x in cli.c6_recheck(v2))


def test_the_dashboard_rebuilds_when_the_other_pool_changes(tmp_path, monkeypatch):
    pools = c6_pools(tmp_path, monkeypatch)
    v2, v1 = pools["pool-v2"][0], pools["pool-v1"][0]
    before = cli.watched(v2)
    receipt(v1, {"label": "A1", "anchor": "A1", "path": None, "sha256": None}, {i: np.full(NR, 1.0) for i in IDS},
            "2026-10-03T00:00:09")
    assert cli.watched(v2) != before


def test_an_unreadable_c6_result_is_invalid(tmp_path, monkeypatch):
    pools = c6_pools(tmp_path, monkeypatch)
    v2, v1 = pools["pool-v2"][0], pools["pool-v1"][0]
    assert crosscheck(v2, v1) == 0
    (v2.dir / "crosscheck-pool-v1-99991231T235959999999Z.json").write_text("[")
    assert cli.build_view(v2)["c6"]["state"] == "INVALID"


def test_audit_reruns_c6_instead_of_trusting_its_file(tmp_path, monkeypatch):
    pools = c6_pools(tmp_path, monkeypatch)
    v2, v1 = pools["pool-v2"][0], pools["pool-v1"][0]
    assert crosscheck(v2, v1) == 0
    f = sorted(v2.dir.glob("crosscheck-*.json"))[-1]
    c = json.loads(f.read_text())
    # A pool-v1 calibration array swapped for another one with a consistent record: the inputs snapshot is
    # rewritten to match, so only a re-run can tell.
    rewrite_pair(v1, "A2", "A0", np.full(N, -30.0) + np.r_[1.0, np.zeros(N - 1)])
    c["inputs"] = cli.c6_inputs(v2, v1)
    f.unlink()
    cli.write_new(f, c)
    assert cli.build_view(v2)["c6"]["state"] == "VERIFIED"
    assert any("A2-A0" in x for x in cli.c6_recheck(v2))


def replay_engine(s, monkeypatch, who, during):
    """A new engine (everything STALE) whose replays reproduce the receipts' rows; `during()` runs once, at the
    first game `who` plays (its receipt and arrays are read by then)."""
    monkeypatch.setattr(cli, "engine_identity", lambda: ID2)
    real = {(cli.REFERENCE, j): s["ref"][j] for j in IDS} | {("cand", j): s["x"][j] for j in IDS}
    monkeypatch.setattr(cli, "candidate_agent", lambda pool, c: c["label"])
    monkeypatch.setattr(cli.Pool, "agent", lambda self, j: j)
    done = []

    def play(pool, me, other, seed, deals, threads, **kw):
        if me == who and not done:
            done.append(1)
            during()
        return {"per": real[(me, other)], "a_search_failed": [0, 0], "b_search_failed": [0, 0]}
    monkeypatch.setattr(cli, "play", play)


def test_a_receipt_rewritten_during_its_replay_is_not_verified(setup, monkeypatch):
    s = setup
    forged = {j: v + 50.0 for j, v in s["x"].items()}

    def rewrite():  # the JSON and its arrays, consistently, while the replay runs
        npz = s["px"].with_suffix(".npz")
        npz.unlink()
        edit_receipt(s["px"], lambda r: r.update(arrays=cli.save_arrays(npz, forged)))
    replay_engine(s, monkeypatch, "cand", rewrite)
    assert cli.verify_all(s["pool"], 1, only_stale=True) == 0  # both replays matched the bytes they read
    r = rows(s["pool"])[s["px"].name]
    assert r["state"] == "STALE" and "not replayed" in r["why"]
    set_current(s["pool"], "cand")
    assert audit(s["pool"]) == 1
    assert cli.verify_all(s["pool"], 1, only_stale=True) == 1  # a replay of what is there now catches it
    assert rows(s["pool"])[s["px"].name]["state"] == "INVALID"


def test_a_receipt_reattributed_during_verify_is_not_verified(setup, monkeypatch):
    s = setup
    other = blueprint_file(s["pool"], "weaker", b"another blueprint")
    replay_engine(s, monkeypatch, cli.REFERENCE,
                  lambda: edit_receipt(s["px"], lambda r: r.update(candidate={"label": "weaker", **other})))
    assert cli.verify_all(s["pool"], 1, only_stale=True) == 0
    set_current(s["pool"], "weaker")
    v = cli.build_view(s["pool"])
    assert v["headline"] is None and "STALE" in v["headline_why"]


def test_audit_needs_a_verified_headline(setup, capsys):
    s = setup
    assert audit(s["pool"]) == 1 and "no VERIFIED headline: no current bot set" in capsys.readouterr().out
    set_current(s["pool"], "cand")
    assert audit(s["pool"]) == 0
    for content in ("{", "[]", '{"fingerprint": 5}', '{"label": "nobody"}', '{"history": [{"fingerprint": 5}]}'):
        (s["pool"].dir / "current.json").write_text(content)
        capsys.readouterr()
        assert audit(s["pool"]) == 1 and "no VERIFIED headline: current.json" in capsys.readouterr().out, content


@pytest.mark.parametrize("content", ["{", "[]", '{"history": [{"label": "x"}]}', '{"history": 5}',
                                     '{"fingerprint": "x", "history": [{"fingerprint": 5, "label": "y"}]}',
                                     '{"label": "nobody", "set_utc": "2026-10-01T00:00:00"}'])
def test_current_never_overwrites_a_broken_pointer(setup, content, capsys):
    s = setup
    f = s["pool"].dir / "current.json"
    f.write_text(content)
    assert cli.main(["current", "--dir", str(s["pool"].dir), "--label", "cand"]) == 2
    assert "will not overwrite" in capsys.readouterr().err and f.read_text() == content


@pytest.mark.parametrize("old", ['{"label": "cand"}', '{"fingerprint": "0123456789abcdef"}'])
def test_a_pointer_without_a_history_is_kept_in_the_new_one(setup, old):
    s = setup
    fp = cli.fingerprint(s["cand"])
    f = s["pool"].dir / "current.json"
    f.write_text(old)  # Amendment 1's file (a label), or one naming only a configuration
    set_current(s["pool"], "cand")
    c = cli.read_pointer(f)
    assert c["fingerprint"] == fp and c["history"][-1]["label"] == "cand"
    assert [h["fingerprint"] for h in c["history"]] == ([fp] if "label" in old else ["0123456789abcdef", fp])


def test_current_never_drops_a_label_only_pointer_it_cannot_resolve(setup, capsys):
    s = setup
    for k, content in enumerate((b"bp A", b"bp B")):  # one label on two files: Amendment 1 had no clash check
        receipt(s["pool"], {"label": "v2-2B", **blueprint_file(s["pool"], f"v{k}", content)}, s["x"],
                f"2026-10-03T00:00:2{k}")
    f = s["pool"].dir / "current.json"
    f.write_text('{"label": "v2-2B"}')
    assert cli.main(["current", "--dir", str(s["pool"].dir), "--label", "cand"]) == 2
    err = capsys.readouterr().err
    assert "names 2 configurations" in err and "will not overwrite" in err and f.read_text() == '{"label": "v2-2B"}'


def test_only_crosscheck_files_this_code_names_are_read(tmp_path, monkeypatch):
    pools = c6_pools(tmp_path, monkeypatch)
    v2, v1 = pools["pool-v2"][0], pools["pool-v1"][0]
    assert crosscheck(v2, v1) == 0
    # 2c72995 named its files in local time (they can sort after a newer UTC name) and recorded no inputs.
    for ok in (True, False):
        (v2.dir / "crosscheck-pool-v1-99991231-235959.json").write_text(json.dumps({"C6_ok": ok, "C6_problems": ["x"]}))
        assert cli.build_view(v2)["c6"]["state"] == "VERIFIED"


def test_a_relative_pool_dir_finds_its_sibling(tmp_path, monkeypatch, capsys):
    pools = c6_pools(tmp_path, monkeypatch)
    v2, v1 = pools["pool-v2"][0], pools["pool-v1"][0]
    set_current(v2, "A2")
    monkeypatch.chdir(v2.dir)
    assert cli.c6_other(cli.Pool(Path("."))).resolve() == v1.dir.resolve()
    assert cli.main(["crosscheck", "--dir", ".", "--other", str(v1.dir)]) == 0
    assert cli.build_view(cli.Pool(Path(".")))["c6"]["state"] == "VERIFIED"
    capsys.readouterr()
    assert cli.main(["audit", "--dir", "."]) == 0 and "AUDIT OK" in capsys.readouterr().out


def test_c6_compares_the_pre_registered_number_of_deals(tmp_path, monkeypatch):
    pools = c6_pools(tmp_path, monkeypatch)
    (v2, ps2), (v1, _) = pools["pool-v2"], pools["pool-v1"]
    assert crosscheck(v2, v1) == 0
    c = json.loads(sorted(v2.dir.glob("crosscheck-*.json"))[-1].read_text())
    assert all(x["deals"] == N for x in c["C6"]["rows"]["A1"].values())
    a1 = cli.load_arrays(ps2[1].with_suffix(".npz"), json.loads(ps2[1].read_text())["arrays"])
    p = receipt(v1, {"label": "A1", "anchor": "A1", "path": None, "sha256": None}, {j: a[:10] for j, a in a1.items()},
                "2026-10-03T00:00:09")  # the same first deals, but only 10 of them
    replayed(v1, p)
    assert crosscheck(v2, v1) == 1
    assert cli.build_view(v2)["c6"]["state"] == "INVALID"
    assert any("C6 compares the first" in x for x in cli.c6_recheck(v2))


def test_c6_needs_the_same_engine(tmp_path, monkeypatch):
    pools = c6_pools(tmp_path, monkeypatch)
    (v2, _), (v1, ps1) = pools["pool-v2"], pools["pool-v1"]
    edit_receipt(ps1[2], lambda r: r.update(engine=ID2))
    assert crosscheck(v2, v1) == 1
    assert any("A2: made on different engines" in x for x in cli.c6_recheck(v2))


def test_a_linked_pool_dir_finds_its_sibling_next_to_the_link(tmp_path, monkeypatch, capsys):
    pools = c6_pools(tmp_path / "rating", monkeypatch)
    v1 = pools["pool-v1"][0]
    real, link = tmp_path / "bigdisk" / "pool-v2", tmp_path / "rating" / "pool-v2"
    real.parent.mkdir()
    link.rename(real)
    try:
        link.symlink_to(real, target_is_directory=True)  # on Windows a junction behaves the same
    except OSError as e:
        pytest.skip(f"cannot make a symlink here: {e}")
    v2 = cli.Pool(link)
    set_current(v2, "A2")
    assert cli.c6_other(v2) == v1.dir
    assert cli.main(["crosscheck", "--dir", str(link), "--other", str(v1.dir)]) == 0
    assert cli.build_view(v2)["c6"]["state"] == "VERIFIED"
    capsys.readouterr()
    assert cli.main(["audit", "--dir", str(link)]) == 0 and "AUDIT OK" in capsys.readouterr().out


def test_a_c6_pass_counts_only_under_the_code_that_made_it(tmp_path, monkeypatch):
    """A crosscheck file is the verdict of the engine and rating code that wrote it, at its C6 deal count (0713a34
    wrote the same file name and compared fewer deals): any other is STALE until C6 is run again."""
    pools = c6_pools(tmp_path, monkeypatch)
    v2, v1 = pools["pool-v2"][0], pools["pool-v1"][0]
    assert crosscheck(v2, v1) == 0 and cli.build_view(v2)["c6"]["state"] == "VERIFIED"
    f = sorted(v2.dir.glob("crosscheck-*.json"))[-1]
    c = json.loads(f.read_text())
    c.pop("c6_deals", None)
    f.write_text(json.dumps(c))
    v = cli.build_view(v2)
    assert v["c6"]["state"] == "STALE" and "another engine or rating code" in v["c6"]["why"]
    assert all(r["state"] == "STALE" for r in v["rows"])
    assert crosscheck(v2, v1) == 0 and cli.build_view(v2)["c6"]["state"] == "VERIFIED"
    monkeypatch.setattr(cli, "engine_identity", lambda: ID2)
    assert cli.build_view(v2)["c6"]["state"] == "STALE"


def test_a_c6_failure_counts_only_under_the_code_that_made_it(tmp_path, monkeypatch):
    """A failed crosscheck file is a verdict of the engine and code that wrote it too: INVALID under them, STALE under
    any other (or without C6's deal count) until C6 is run again."""
    pools = c6_pools(tmp_path, monkeypatch)
    v2, v1 = pools["pool-v2"][0], pools["pool-v1"][0]
    p = receipt(v1, {"label": "A1", "anchor": "A1", "path": None, "sha256": None}, {j: np.full(NR, 1.0) for j in IDS},
                "2026-10-03T00:00:09")
    replayed(v1, p)  # pool-v1's newest A1 row differs from pool-v2's: C6 fails
    assert crosscheck(v2, v1) == 1 and cli.c6_state(v2)["state"] == "INVALID"
    monkeypatch.setattr(cli, "engine_identity", lambda: ID2)
    s = cli.c6_state(v2)
    assert s["state"] == "STALE" and "another engine or rating code" in s["why"]
    monkeypatch.setattr(cli, "engine_identity", lambda: ID1)
    f = sorted(v2.dir.glob("crosscheck-*.json"))[-1]
    c = json.loads(f.read_text())
    c.pop("c6_deals")
    f.write_text(json.dumps(c))
    assert cli.c6_state(v2)["state"] == "STALE"


def blueprint_engine(s, monkeypatch, produces, load=None, during=None):
    """A new engine (everything STALE) that loads a candidate's file from disk as the real one does (once per path
    per Pool, pool._bp) and plays what the bytes it loaded produce: `produces[bytes]`. `load()` runs as a file is
    loaded, `during(k)` before the k-th candidate game."""
    monkeypatch.setattr(cli, "engine_identity", lambda: ID2)
    monkeypatch.setattr(cli.Pool, "agent", lambda self, j: ("anchor", j))

    class BP:
        def __init__(self, abstraction, path):
            if load:
                load()
            self.content = Path(path).read_bytes()

        def play_agent(self):
            return ("bp", self.content)
    monkeypatch.setattr(cli, "engine", lambda: type("E", (), {"SavedBlueprint": BP}))
    games = []

    def play(pool, me, other, seed, deals, threads, **kw):
        if me[0] == "bp":
            games.append(1)
            if during:
                during(len(games))
            per = produces[me[1]][other[1]]
        else:
            per = s["ref"][other[1]]
        return {"per": per, "a_search_failed": [0, 0], "b_search_failed": [0, 0]}
    monkeypatch.setattr(cli, "play", play)


def test_a_candidate_file_swapped_in_for_its_replay_is_refused(setup, monkeypatch):
    """A receipt whose arrays were played by another file Q than the one it names (P): Q put at P's path for the
    replay (before it, or as the engine loads it) and P put back is refused, never VERIFIED."""
    s = setup
    f = Path(s["cand"]["path"])
    P, Q = f.read_bytes(), b"a weaker blueprint"
    blueprint_engine(s, monkeypatch, {Q: s["x"], P: {j: v + 7.0 for j, v in s["x"].items()}})
    f.write_bytes(Q)
    assert cli.verify_all(s["pool"], 1, only_stale=False) == 1
    f.write_bytes(P)
    r = rows(s["pool"])[s["px"].name]
    assert r["state"] == "STALE" and "not the receipt's" in r["why"] and "before the replay" in r["why"]
    swapped = []
    blueprint_engine(s, monkeypatch, {Q: s["x"], P: {j: v + 7.0 for j, v in s["x"].items()}},
                     load=lambda: swapped or (swapped.append(1), f.write_bytes(Q)),
                     during=lambda k: f.write_bytes(P))
    assert cli.verify_all(s["pool"], 1, only_stale=False, name=s["px"].name) == 1
    f.write_bytes(P)  # put back (by `during` had the games been played on Q)
    r = rows(s["pool"])[s["px"].name]
    assert r["state"] == "STALE" and "once loaded" in r["why"]
    assert cli.verify_all(s["pool"], 1, only_stale=False, name=s["px"].name) == 1  # P itself: a mismatch
    assert rows(s["pool"])[s["px"].name]["state"] == "INVALID"


def test_a_swapped_file_with_the_old_size_and_mtime_is_still_hashed(setup, monkeypatch):
    s = setup
    f = Path(s["cand"]["path"])
    P = f.read_bytes()
    Q, st = P[::-1], f.stat()  # the same size
    blueprint_engine(s, monkeypatch, {Q: s["x"], P: {j: v + 7.0 for j, v in s["x"].items()}})
    assert cli.sha256_file(f) == s["cand"]["sha256"]  # memoized for P's size and mtime
    f.write_bytes(Q)
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert cli.verify_all(s["pool"], 1, only_stale=False, name=s["px"].name) == 1
    last = cli.verifications(s["pool"])[0][-1]
    assert last["ok"] is None and "before the replay" in last["detail"]  # refused, not played on Q


def test_a_candidate_file_rewritten_during_its_replay_is_refused_not_invalid(setup, monkeypatch):
    s = setup
    f = Path(s["cand"]["path"])
    P = f.read_bytes()
    blueprint_engine(s, monkeypatch, {P: s["x"]}, during=lambda k: k == 1 and f.write_bytes(b"500M more iterations"))
    assert cli.verify_all(s["pool"], 1, only_stale=False) == 1
    f.write_bytes(P)  # put back: the receipt was never wrong
    r = rows(s["pool"])[s["px"].name]
    assert r["state"] == "STALE" and "after the replay" in r["why"]
    assert cli.verify_all(s["pool"], 1, only_stale=True) == 0
    assert rows(s["pool"])[s["px"].name]["state"] == "VERIFIED"


def test_each_replay_loads_the_candidate_file_it_checked(setup, monkeypatch):
    """Receipts of two versions of one file (trained further and rated again), replayed in one process: the second
    replay plays the file it checked, never the blueprint the first one loaded."""
    s = setup
    f = Path(s["cand"]["path"])
    P, P2 = f.read_bytes(), b"blueprint one, trained further"
    x2 = {j: v + 3.0 for j, v in s["x"].items()}
    blueprint_engine(s, monkeypatch, {P: s["x"], P2: x2})
    assert cli.verify_all(s["pool"], 1, only_stale=False) == 0  # the reference and P's receipt
    f.write_bytes(P2)
    p2 = receipt(s["pool"], {"label": "cand2", "path": str(f), "sha256": cli._sha256_file(f), "search": None,
                             "translation": "snap"}, x2, "2026-10-03T00:00:02")
    assert cli.verify_all(s["pool"], 1, only_stale=False, name=p2.name) == 0
    assert rows(s["pool"])[p2.name]["state"] == "VERIFIED"


def anchor_pool(d: Path, monkeypatch):
    """make_pool with blueprint anchors: each anchor is a file of its own bytes. Returns the pool and those bytes."""
    make_pool(d, monkeypatch)
    defn = json.loads((d / "pool.json").read_text())
    content = {}
    for a in defn["anchors"]:
        f = d / f"anchor-{a['id']}.bin"
        content[a["id"]] = f"anchor {a['id']}".encode()
        f.write_bytes(content[a["id"]])
        for k in ("of", "family", "strength"):
            a.pop(k)
        a.update(kind="blueprint", path=str(f), sha256=cli._sha256_file(f))
    defn["pool_hash"] = cli.pool_hash(defn)
    (d / "pool.json").write_text(json.dumps(defn))
    cal = json.loads((d / "calibration.json").read_text())
    cal["pool_hash"] = defn["pool_hash"]
    (d / "calibration.json").write_text(json.dumps(cal))
    return cli.Pool(d), content


def loaded_per(me: bytes, other: bytes):
    """What the fake engine of `anchor_engine` plays for these loaded bytes (deterministic in them)."""
    seed = int.from_bytes(cli.hashlib.sha256(me + b"|" + other).digest()[:8], "little")
    return np.random.default_rng(seed).normal(0, 100, NR)


def anchor_receipt(pool, content, cand_bytes, label, started, anchors=None):
    """A receipt of a candidate file with these bytes, played against `content` (or `anchors` where given)."""
    f = pool.dir / f"{label}.bin"
    f.write_bytes(cand_bytes)
    played = dict(content, **(anchors or {}))
    cand = {"label": label, "path": str(f), "sha256": cli._sha256_file(f), "search": None, "translation": "snap"}
    return receipt(pool, cand, {j: loaded_per(cand_bytes, played[j]) for j in IDS}, started)


def replay_state(pool, p):
    """The receipt's own state from its replays (these pools have no reference receipt to rate it against)."""
    return cli.state_of(pool, p, dict(cli.read_receipts(pool)[0])[p], cli.verifications(pool), cli.engine_identity())[0]


def anchor_engine(monkeypatch, load=None, during=None):
    """A new engine (everything STALE) that loads every blueprint file, the anchors' too, from disk as the real one
    does (once per path per Pool, pool._bp) and plays what the loaded bytes produce. `load(path)` runs as a file is
    loaded, before it is read; `during(k)` before the k-th game."""
    monkeypatch.setattr(cli, "engine_identity", lambda: ID2)

    class BP:
        def __init__(self, abstraction, path):
            if load:
                load(Path(path))
            self.content = Path(path).read_bytes()

        def play_agent(self):
            return self.content
    monkeypatch.setattr(cli, "engine", lambda: type("E", (), {"SavedBlueprint": BP}))
    games = []

    def play(pool, me, other, seed, deals, threads, **kw):
        games.append(1)
        if during:
            during(len(games))
        return {"per": loaded_per(me, other), "a_search_failed": [0, 0], "b_search_failed": [0, 0]}
    monkeypatch.setattr(cli, "play", play)


def test_an_anchor_file_swapped_in_for_its_load_is_refused(tmp_path, monkeypatch):
    """Another file Q at an anchor's path as the engine loads it, the anchor's file put back during the games: the
    replay is refused (the anchors are loaded before the check 'once loaded'), never a verdict on Q's games, whether
    the receipt was played against the anchor (honest) or against Q."""
    Q = b"another A1"
    for anchors in (None, {"A1": Q}):
        pool, content = anchor_pool(tmp_path / ("honest" if anchors is None else "forged"), monkeypatch)
        p = anchor_receipt(pool, content, b"candidate", "cand", "2026-10-03T00:00:01", anchors)
        a1 = Path(pool.anchor("A1")["path"])
        anchor_engine(monkeypatch, load=lambda f: f == a1 and a1.write_bytes(Q),
                      during=lambda k: a1.write_bytes(content["A1"]))
        assert cli.verify_all(pool, 1, only_stale=False) == 1
        a1.write_bytes(content["A1"])  # put back (by `during` had the games been played)
        last = cli.verifications(pool)[0][-1]
        assert last["ok"] is None and "anchor A1 file changed" in last["detail"]
        assert replay_state(pool, p) == "STALE"


def test_each_replay_loads_the_anchor_files_it_checked(tmp_path, monkeypatch):
    """Two replays in one process: the first is refused (an anchor file changed as it was loaded) and the file is put
    back; the second plays the anchor files it checked, never a blueprint the first one loaded."""
    pool, content = anchor_pool(tmp_path, monkeypatch)
    p1 = anchor_receipt(pool, content, b"candidate one", "one", "2026-10-03T00:00:01")
    p2 = anchor_receipt(pool, content, b"candidate two", "two", "2026-10-03T00:00:02")
    a1 = Path(pool.anchor("A1")["path"])
    swapped = []
    anchor_engine(monkeypatch, load=lambda f: f == a1 and not swapped and (swapped.append(1), a1.write_bytes(b"Q")))
    real = cli.append_verification

    def append(pl, entry):  # the file put back once the first replay is refused
        real(pl, entry)
        a1.write_bytes(content["A1"])
    monkeypatch.setattr(cli, "append_verification", append)
    assert cli.verify_all(pool, 1, only_stale=False) == 1
    ent = cli.verifications(pool)[0]
    assert [e["receipt"] for e in ent[-2:]] == [p1.name, p2.name]
    assert ent[-2]["ok"] is None and ent[-1]["ok"] is True
    assert replay_state(pool, p2) == "VERIFIED"


PLAY = "P" * 64  # a play-code sha256 (Amendment 3 (5))
LIVE_SEARCH = {"solver": "live", "search": {"from_flop": True, "flop_iters": 500, "from_turn": True, "turn_iters": 100,
                                            "river_iters": 200, "rescue": False},
               "leaves": "sym", "terminals64": True, "flop_gamma": 5.0, "net_sha256": "N" * 64}


@pytest.fixture
def live_setup(setup):
    """The setup pool with Amendment 3's protocol: every anchor carries weight here, so a search receipt plays
    all three, N deals each, with a play-code identity."""
    s = setup
    s["mp"].setattr(cli, "SEARCH_DEALS", {"pool-test": N})
    s["mp"].setattr(cli, "SEARCH_CHUNK", 3_000)
    s["play"] = s["pool"].dir / "play.py"  # stands in for the play files (their stat decides a re-hash)
    s["play"].write_text("one")
    s["mp"].setattr(cli, "play_files", lambda: [s["play"]])
    s["mp"].setattr(cli, "play_code_sha", lambda: PLAY)
    s["mp"].setattr(cli, "_PLAY", {})
    s["live"] = {"label": "live", **blueprint_file(s["pool"], "live", b"the live blueprint"), "search": LIVE_SEARCH}
    s["rid"] = {"id": f"{ID1['id']}+{PLAY[:16]}"}
    return s


def change_play_code(s, sha="Q" * 64):
    s["mp"].setattr(cli, "play_code_sha", lambda: sha)
    s["play"].write_text("two, longer")


def live_extra():
    return {"threads": 1, "chunk": cli.SEARCH_CHUNK, "aivat": True}


def live_arrays(s, shift=5.0):
    x = {j: v[:N] + shift for j, v in s["x"].items()}
    return {**x, **{f"{j}.raw": v + s["rng"].normal(0, 50, N) for j, v in x.items()}}


def test_a_search_receipt_counts_only_under_its_play_code(live_setup):
    s = live_setup
    arrs = live_arrays(s)
    p = receipt(s["pool"], s["live"], arrs, "2026-10-03T00:00:07", **live_extra())
    replayed(s["pool"], p)  # under the engine + rating identity alone: not the search receipt's
    assert rows(s["pool"])[p.name]["state"] == "STALE"
    replayed(s["pool"], p, ident=s["rid"])
    r = rows(s["pool"])[p.name]
    assert r["state"] == "VERIFIED"
    assert r["rating"] == R.paired({j: arrs[j] for j in IDS}, s["ref"], s["w"])  # AIVAT rows, never the raw
    assert r["rating_raw"] == R.paired({j: arrs[f"{j}.raw"] for j in IDS}, s["ref"], s["w"])
    assert "raw: PR" in cli.render(cli.build_view(s["pool"]))
    change_play_code(s)  # an edit, seen by the same process (a long-running view)
    assert rows(s["pool"])[p.name]["state"] == "STALE"
    assert rows(s["pool"])[s["px"].name]["state"] == "VERIFIED"  # a blueprint receipt does not depend on it


@pytest.mark.parametrize("edit", ["no raw", "short raw", "lbr", "deals", "anchors", "failed", "threads", "chunk",
                                  "aivat"])
def test_a_malformed_search_receipt_is_invalid(live_setup, edit):
    s = live_setup
    arrs = live_arrays(s)
    if edit == "no raw":
        del arrs["A1.raw"]
    if edit == "short raw":
        arrs["A1.raw"] = arrs["A1.raw"][:-1]
    if edit == "deals":
        arrs = {k: v[:N - 1] for k, v in arrs.items()}
    p = receipt(s["pool"], s["live"], arrs, "2026-10-03T00:00:08", **live_extra())
    if edit == "failed":
        edit_receipt(p, lambda r: r["anchors"]["A2"].update(b_search_failed=[0, 1]))
    if edit in ("threads", "chunk", "aivat"):
        edit_receipt(p, lambda r: r.update({edit: {"threads": 4, "chunk": 100, "aivat": False}[edit]}))
    if edit == "lbr":
        edit_receipt(p, lambda r: r.update(lbr={"deals": 0, "seed": 1}))
    if edit == "anchors":
        edit_receipt(p, lambda r: r["anchors"].pop("A0"))
    replayed(s["pool"], p, ident=s["rid"])
    r = rows(s["pool"])[p.name]
    assert r["state"] == "INVALID" and "rating" not in r, r["why"]


def test_raw_rows_on_a_blueprint_receipt_are_invalid(live_setup):
    s = live_setup
    arrs = {**s["x"], **{f"{j}.raw": v for j, v in s["x"].items()}}  # on protocol but for the raw rows
    p = receipt(s["pool"], {**s["cand"], "label": "cand-raw"}, arrs, "2026-10-03T00:00:09")
    replayed(s["pool"], p)
    r = rows(s["pool"])[p.name]
    assert r["state"] == "INVALID" and "protocol" not in r["why"], r["why"]


class FakeSolver:
    def __init__(self, fail_at=None):
        self.calls = self.failures = 0
        self.seconds, self.last_error, self.fail_at, self.n = {}, None, fail_at, 0

    def take(self):
        got = self.calls, self.failures
        self.calls = self.failures = 0
        return got


def test_a_search_receipt_plays_exactly_the_weighted_anchors(live_setup):
    s = live_setup
    r = {"candidate": s["live"], "anchors": {j: {"deals": N} for j in IDS}}
    assert cli.protocol_problem(s["pool"], r, s["w"]) is None
    w = {**s["w"], "A0": 0.0}
    assert "exactly the weighted anchors" in cli.protocol_problem(s["pool"], r, w)
    r["anchors"].pop("A0")
    assert cli.protocol_problem(s["pool"], r, w) is None


def test_the_play_code_sha_covers_the_code_the_net_and_the_gpu_stack(tmp_path, monkeypatch):
    code, net = tmp_path / "a.py", tmp_path / "net.pt"
    code.write_bytes(b"x = 1\n")
    net.write_bytes(b"weights")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "play_files", lambda: [code, net])
    monkeypatch.setattr(cli, "play_env", lambda: "torch=1")
    one = cli.play_code_sha()
    code.write_bytes(b"x = 1\r\n")
    assert cli.play_code_sha() == one  # CRLF normalized
    net.write_bytes(b"weightz")
    assert cli.play_code_sha() != one  # a retrained net (Amendment 3 (5))
    net.write_bytes(b"weights")
    monkeypatch.setattr(cli, "play_env", lambda: "torch=2")
    assert cli.play_code_sha() != one  # an upgraded GPU stack


def test_the_play_files_are_the_solver_code_the_runner_and_its_net():
    files = cli.play_files()
    assert files[-2:] == [cli.RUNNER, Path(cli.live_runner().NET)]
    assert ROOT / "src" / "parciver" / "flop_search.py" in files
    assert not any("dashboard" in f.parts for f in files)


class FakeAgent:
    """Deterministic per (seed, first deal, size): what a replay of the same chunk plays again."""
    def __init__(self, solver, offset=0.0, raw_offset=0.0):
        self.solver, self.offset, self.raw_offset = solver, offset, raw_offset

    def match_against(self, kind, deals, sb, bb, stack, other, seed, threads, first_deal, aivat):
        assert kind == "agent" and threads == 1 and aivat is True
        sv = self.solver
        sv.n += 1
        sv.calls += 4
        sv.seconds.setdefault(3, []).extend([0.1, 0.1])
        if sv.fail_at == sv.n:
            sv.failures += 1
            sv.last_error = "CUDA error"
        g = np.random.default_rng([seed, first_deal, deals])
        u = g.normal(0, 30, deals)
        return {"per_deal_bb": list(u + self.raw_offset), "per_deal_bb_aivat": list(0.5 * u + 0.2 + self.offset),
                "per_deal_bb_cv": list(u), "a_search_failed": [0, 0], "b_search_failed": [0, 0],
                "a_sources": [1, 2, 0, 0], "aivat_mass_error": 1e-7}


def rate_live(s, solver, offset=0.0):
    s["mp"].setattr(cli, "live_agent", lambda pool, cand: (FakeAgent(solver, offset), solver))
    s["mp"].setattr(cli.Pool, "agent", lambda self, j: None)
    return cli.run_live_rating(s["pool"], s["live"], s["w"])


def test_a_live_rating_resumes_after_a_refused_chunk_and_replays(live_setup, monkeypatch):
    s = live_setup
    replays = []
    monkeypatch.setattr(cli, "replay_in_new_process",
                        lambda pool, path, threads: replays.append(cli.verify_all(pool, 1, False, path.name)))
    with pytest.raises(R.RatingError, match="not kept"):
        rate_live(s, FakeSolver(fail_at=6))  # chunks 1-5 kept (A0 4 chunks, A1 1), the 6th refused
    part = list((s["pool"].dir / "partial").glob("*.npz"))
    assert len(part) == 1 and len((s["pool"].dir / "refused.jsonl").read_text().splitlines()) == 1
    with np.load(part[0]) as z:
        assert len(z["A0"]) == N and len(z["A1"]) == 3_000 and "A2" not in z.files
    p = rate_live(s, FakeSolver())
    assert replays == [0]  # replayed (one chunk per anchor) and identical
    r = rows(s["pool"])[p.name]
    assert r["state"] == "VERIFIED", r["why"]
    arr = cli.load_arrays(p.with_suffix(".npz"), json.loads(p.read_text())["arrays"])
    for j in IDS:  # the resumed rating equals one uninterrupted run, chunk by chunk
        want = [np.random.default_rng([s["pool"].seed(j), f, 3_000]).normal(0, 30, 3_000) for f in range(0, N, 3_000)]
        raw = np.concatenate(want)
        assert np.array_equal(arr[f"{j}.raw"], R.per_hand_mbb(raw, s["pool"].hpd))
        assert np.array_equal(arr[j], R.per_hand_mbb(0.5 * raw + 0.2, s["pool"].hpd))
    left = [q.name for q in (s["pool"].dir / "partial").glob("*.npz")]  # moved aside as done, never deleted
    assert left == [f"{cli.fingerprint(s['live'])}-live.done-{p.stem}.npz"]


def test_a_live_rating_refuses_a_partial_file_of_another_identity(live_setup):
    s = live_setup
    with pytest.raises(R.RatingError):
        rate_live(s, FakeSolver(fail_at=2))
    change_play_code(s)
    with pytest.raises(R.RatingError, match="another identity"):
        rate_live(s, FakeSolver())


def test_a_live_rating_runs_once_at_a_time(live_setup):
    s = live_setup
    part = s["pool"].dir / "partial" / f"{cli.fingerprint(s['live'])}-live.npz"
    part.parent.mkdir()
    with cli.locked(part):
        with pytest.raises(R.RatingError, match="locked"):
            rate_live(s, FakeSolver())
    assert not part.exists()


def test_a_complete_partial_whose_receipt_exists_is_not_written_twice(live_setup, monkeypatch):
    s = live_setup
    replays = []
    monkeypatch.setattr(cli, "replay_in_new_process", lambda pool, path, threads: replays.append(path.name) or 0)
    p = rate_live(s, FakeSolver())
    part = s["pool"].dir / "partial" / f"{cli.fingerprint(s['live'])}-live.npz"
    done = part.with_name(f"{part.stem}.done-{p.stem}.npz")
    part.write_bytes(done.read_bytes())  # as if the run died between the receipt and moving the partial aside
    assert rate_live(s, FakeSolver(fail_at=1)) == p  # nothing played (the first chunk would be refused)
    assert sorted(q.name for q in (s["pool"].dir / "receipts").glob("*-live.json")) == [p.name]
    assert replays == [p.name, p.name] and not part.exists() and done.exists()


@pytest.mark.parametrize("which", ["aivat", "raw"])
def test_a_search_replay_that_differs_is_invalid(live_setup, monkeypatch, which):
    s = live_setup
    monkeypatch.setattr(cli, "replay_in_new_process", lambda pool, path, threads: 0)
    p = rate_live(s, FakeSolver())
    sv = FakeSolver()
    off = {"offset": 1e-9} if which == "aivat" else {"raw_offset": 1e-9}
    monkeypatch.setattr(cli, "live_agent", lambda pool, cand: (FakeAgent(sv, **off), sv))
    assert cli.verify_all(s["pool"], 1, False, p.name) == 1
    r = rows(s["pool"])[p.name]
    assert r["state"] == "INVALID" and "DIFFERENT" in r["why"]


def test_a_replay_verdict_counts_under_the_play_code_it_played(live_setup, monkeypatch):
    s = live_setup
    p = receipt(s["pool"], s["live"], live_arrays(s), "2026-10-03T00:00:10", **live_extra())
    rid = s["rid"]["id"]

    def replay_then_edit(pool, path, r, threads, cur):
        change_play_code(s)  # an edit lands after the replay's last check, before its verdict is written
        return {"ok": True, "detail": "t", "kind": "search-chunk", "identity": rid}
    monkeypatch.setattr(cli, "replay", replay_then_edit)
    assert cli.verify_all(s["pool"], 1, False, p.name) == 0
    last = cli.verifications(s["pool"])[0][-1]
    assert last["identity"] == rid and last["ok"] is True
    assert rows(s["pool"])[p.name]["state"] == "STALE"  # under the edited code, never VERIFIED


def test_a_runner_that_does_not_load_is_stale_not_invalid(live_setup):
    s = live_setup
    p = receipt(s["pool"], s["live"], live_arrays(s), "2026-10-03T00:00:11", **live_extra())
    replayed(s["pool"], p, ident=s["rid"])

    def broken():
        raise SyntaxError("half-saved runner")
    s["mp"].setattr(cli, "play_files", broken)
    r = rows(s["pool"])[p.name]
    assert r["state"] == "STALE" and "SyntaxError" in r["why"]
    assert rows(s["pool"])[s["px"].name]["state"] == "VERIFIED"


def test_rate_live_refuses_lbr_and_another_blueprint(live_setup, monkeypatch):
    s = live_setup
    sr = type("SR", (), {"BLUEPRINT": s["live"]["path"]})
    monkeypatch.setattr(cli, "live_runner", lambda: sr)
    a = type("A", (), {"dir": str(s["pool"].dir), "label": "live", "lbr_deals": 5, "translation": "snap",
                       "candidate": s["live"]["path"], "search": "live", "no_dashboard": True})()
    with pytest.raises(R.RatingError, match="LBR"):
        cli.rate_live(a, s["pool"], s["w"])
    a.lbr_deals, a.candidate = None, s["cand"]["path"]
    with pytest.raises(R.RatingError, match="live runner plays"):
        cli.rate_live(a, s["pool"], s["w"])


@pytest.mark.skipif(os.environ.get("PARCIVER_RATING_SMOKE") != "1", reason="set PARCIVER_RATING_SMOKE=1 (real engine)")
def test_smoke_pipeline_on_the_real_engine(tmp_path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / "scripts" / "rating.py"), *a, "--dir", str(tmp_path)],
                                    capture_output=True, text=True, env=env)
    c = run("calibrate", "--smoke", "--no-dashboard")
    assert c.returncode == 0, c.stdout + c.stderr
    assert "4/4" not in c.stdout and "15/15 exact" in c.stdout, c.stdout
    assert run("current", "--label", "A2").returncode == 0
    a = run("audit")
    assert a.returncode == 0 and "AUDIT OK" in a.stdout and "HEADLINE  A2" in a.stdout, a.stdout + a.stderr
    v = run("verify")
    assert v.returncode == 0 and "MISMATCH" not in v.stdout, v.stdout + v.stderr
    npz = sorted((tmp_path / "receipts").glob("*-A2.npz"))[0]
    with np.load(npz) as z:
        arrs = {k: z[k].copy() for k in z.files}
    arrs["A0"][7] += 1.0
    npz.unlink()
    np.savez_compressed(npz, **arrs)
    a = run("audit")
    assert a.returncode == 1 and "INVALID" in a.stdout and "HEADLINE  none" in a.stdout
