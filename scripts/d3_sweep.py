# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""D3-SWEEP's driver (records/d3-sweep/PREREG.md): which training recipe N2 uses, decided on D1 + D1b alone.

    d3_sweep.py check               N1's recipe through the new code for 3 epochs of the 200-epoch schedule against N1's history, the two
                                    saved controls rescored by source against records/d2b/selection.json, and the host peak (check.json)
    d3_sweep.py train [--arm TAG]   round 1's five arms (those not finished), or `--arm d3s-combo` when the rule calls it; refused unless
                                    `check` passed on this code and the peak commit was told (--told GIB)
    d3_sweep.py select              every net scored by source on the selection half (selection.json, then combo.json for round 2)
    d3_sweep.py confirm             `d3s-confirm`: the overall winner's recipe at seed 1, scored (confirm.json)
    d3_sweep.py verdict             verdict.json and RESULT.md: the pin is N1's recipe or the one confirmed winner

The rules are pure functions below (`wins`, `round1`, `combo_decision`, `confirmed`) and are tested without a GPU. Only the selection half
(23,728 rows) of D1 + D1b is ever scored; the gate halves are not read. A NaN arm is FAILED (recorded, cannot win, not retried), never VOID."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import d3  # noqa: E402
import d3_load as dl  # noqa: E402
import turn_net as tn  # noqa: E402

ROOT = tn.ROOT
RECORDS = ROOT / "records" / "d3-sweep"
CHECK_TAG, COMBO_TAG, CONFIRM_TAG = "d3s-check", "d3s-combo", "d3s-confirm"
N1_TAG, SEED1_TAG = dl.N1_TAG, "d2b-seed1"  # the controls: N1 and its seed-1 twin
CHECK_EPOCHS = 3
BAR, COMBO_BAR, CONFIRM_BAR = 0.98, 0.99, 0.99
FAMILIES = ("dropout", "ema", "loss")  # the option each family moves
LABELS = ("all", "real", "random")
ARMS = (dict(tag="d3s-drop05", family="dropout", dropout=0.05),
        dict(tag="d3s-drop10", family="dropout", dropout=0.10),
        dict(tag="d3s-ema999", family="ema", ema=0.999),
        dict(tag="d3s-ema9999", family="ema", ema=0.9999),
        dict(tag="d3s-rw", family="loss", loss="rw"))
ARM_TAGS = tuple(x["tag"] for x in ARMS)
ALL_TAGS = (*ARM_TAGS, COMBO_TAG, CONFIRM_TAG)
OFF = dict(dropout=0.0, ema=0.0, loss="huber")


# ------------------------------------------------------------------ the rules (pure)


def arm_of(tag: str, **opts) -> dict:
    """An arm for `d2b_train_arm`: N1's recipe (w2048, wd 0.1, the union) with the three options."""
    return dict(tag=tag, width=2048, data="union", wd=0.1, **dict(OFF, **opts))


def table_arm(tag: str) -> dict:
    row = next(x for x in ARMS if x["tag"] == tag)
    return arm_of(tag, **{k: v for k, v in row.items() if k in OFF})


def options_of(row: dict) -> dict:
    return {k: row.get(k, tn.CFG_DEFAULTS[k]) for k in tn.OPTION_KEYS}


def wins(score: dict, control: dict, bar: float) -> bool:
    """PREREG rule 2: the plain selection loss at most `bar` x the control's AND the real-row loss at most the control's."""
    return bool(score["all"] <= bar * control["all"] and score["real"] <= control["real"])


def confirmed(score: dict, control: dict) -> bool:
    """PREREG rule 5: at most 0.99 x the seed-1 control's loss AND its real-row loss at most the control's."""
    return wins(score, control, CONFIRM_BAR)


def round1(scores: dict, control: dict) -> dict:
    """Rule 2 and 3 on round 1. `scores`: arm tag -> {all, real, random} losses, or None for a failed arm. Exact ties go to the arm lower in
    the table (the later row). Returns the winners, the best of each winning family, the best single winner and the combo's options if two families won."""
    order = {t: i for i, t in enumerate(ARM_TAGS)}
    won = {t: s is not None and wins(s, control, BAR) for t, s in scores.items()}
    best = lambda tags: min(tags, key=lambda t: (scores[t]["all"], -order[t])) if tags else None  # noqa: E731
    fam = {f: best([x["tag"] for x in ARMS if x["family"] == f and won.get(x["tag"])]) for f in FAMILIES}
    fam = {f: t for f, t in fam.items() if t}
    single = best([t for t in ARM_TAGS if won.get(t)])
    combo = None
    if len(fam) >= 2:
        opts = {}
        for f, t in fam.items():
            row = next(x for x in ARMS if x["tag"] == t)
            opts[f] = row[f]
        combo = dict(tag=COMBO_TAG, options=dict(OFF, **opts), parents=fam)
    return dict(wins=won, family_best=fam, single=single, combo=combo)


def combo_decision(single: str, single_score: dict, combo_score: dict | None, control: dict) -> dict:
    """Rule 3: the combo goes to the confirmation in place of the best single winner only if it passes rule 2 itself and its selection loss is
    at most 0.99 x the single winner's. `combo_score` None: the combo failed (NaN)."""
    passes = combo_score is not None and wins(combo_score, control, BAR)
    beats = bool(passes and combo_score["all"] <= COMBO_BAR * single_score["all"])
    return dict(combo_passes_rule2=bool(passes), combo_beats_single=beats, candidate=COMBO_TAG if beats else single)


def candidate_of(sel: dict, combo: dict | None) -> tuple[str | None, str]:
    """(the tag whose recipe goes to the confirmation or None, why) from selection.json and combo.json. Refuses while the combo is due and
    not yet decided."""
    if sel["single"] is None:
        return None, "no arm won round 1: the pin is N1's recipe"
    if sel["combo"] is None:
        return sel["single"], "the best single winner (fewer than two families won)"
    if combo is None:
        raise SystemExit("the combo is called for (two families won) and has no combo.json yet: `train --arm d3s-combo`, then `select`")
    return combo["candidate"], "the combo" if combo["candidate"] == COMBO_TAG else "the best single winner (the combo did not displace it)"


def candidate_arm(sel: dict, tag: str, seed_tag: str | None = None) -> dict:
    """The arm (options) of a candidate tag: a table arm, or the combo."""
    if tag == COMBO_TAG:
        return arm_of(seed_tag or COMBO_TAG, **sel["combo"]["options"])
    return table_arm(tag) if seed_tag is None else arm_of(seed_tag, **options_of(table_arm(tag)))


def ema_horizon_epochs(d: float, steps_per_epoch: int) -> float:
    return 1.0 / ((1.0 - d) * steps_per_epoch)


def ema_d_for(horizon_epochs: float, steps_per_epoch: int) -> float:
    """The per-step d that gives `horizon_epochs` at `steps_per_epoch` (PREREG: an EMA pin is carried as its epoch horizon)."""
    return 1.0 - 1.0 / (horizon_epochs * steps_per_epoch)


def compare_history(rec: dict, n1: dict, k: int) -> dict:
    """`check` item 1: the first k epochs of the check net against N1's, every field to the bit, and the settings (the full schedule)."""
    keys = ("epoch", "train_loss", "val_loss", "val_mae")
    cut = lambda r: [{x: h[x] for x in keys} for h in r["history"][:k]]  # noqa: E731
    cfg = ("width", "depth", "wd", "seed", "data", "train_samples", "val_samples", "pools", "epochs", "batch", "lr", "canonical")
    same = dict(rows=len(rec["history"]) == k and len(n1["history"]) >= k, history=cut(rec) == cut(n1),
                config=all(rec.get(x, tn.CFG_DEFAULTS.get(x)) == n1.get(x, tn.CFG_DEFAULTS.get(x)) for x in cfg),
                options_off=options_of(rec) == OFF)
    return dict(**same, equal=all(same.values()))


def compare_scores(got: dict, want: dict) -> dict:
    """`check` item 2: a rescore against records/d2b/selection.json's, rows, loss and MAE to the bit, per source."""
    return {lab: all(got[lab][x] == want[lab][x] for x in ("rows", "loss", "mae")) for lab in LABELS}


def check_refused(chk: dict | None, trees: dict, union: str) -> str | None:
    """Why no arm may train: `check` has not passed, or the code or data is not what it passed on."""
    if not chk:
        return f"{RECORDS / 'check.json'} does not exist: run `d3_sweep check` first"
    if not chk.get("passed"):
        return "check.json did not pass: nothing else is used until the cause is found and an amendment records it"
    if chk.get("trees") != trees:
        return f"src or scripts changed since `check` passed (trees {chk.get('trees')} then, {trees} now): run `check --redo`"
    if chk.get("union") != union:
        return f"`check` ran on data {chk.get('union')}, not {union}"
    return None


def train_plan(requested: str | None, finished: set, failed: set, sel: dict | None, combo: dict | None, redo: bool) -> list[str]:
    """Which arms `train` may train now, or SystemExit with the rule that forbids it (the table, a finished arm, the stage's result)."""
    if requested is not None and requested not in (*ARM_TAGS, COMBO_TAG):
        raise SystemExit(f"{requested} is not in the table {ARM_TAGS} (+ {COMBO_TAG} when the rule calls it; {CONFIRM_TAG} is trained by `confirm`)")
    if redo and requested is None:
        raise SystemExit("--redo needs --arm: it moves one finished arm aside, never every finished arm")
    if requested == COMBO_TAG:
        if sel is None or sel["combo"] is None:
            raise SystemExit("the rule does not call for d3s-combo (it needs selection.json with two winning families)")
        if combo is not None:
            raise SystemExit(f"{RECORDS / 'combo.json'} exists: the combo's result is written once")
    elif sel is not None:
        raise SystemExit(f"{RECORDS / 'selection.json'} exists: round 1 is decided, an arm is not trained again (a bug needs an amendment first)")
    todo = [requested] if requested else [t for t in ARM_TAGS if t not in finished and t not in failed]
    for t in todo:
        if t in failed:
            raise SystemExit(f"{t} FAILED (a non-finite loss): recorded, cannot win, not retried")
        if t in finished and not redo:
            raise SystemExit(f"{t} is finished: `--redo` moves it aside (kept) and trains it again, before its stage's result exists")
    if not todo:
        raise SystemExit("every round-1 arm is finished or failed: `select`")
    return todo


# ------------------------------------------------------------------ small disk pieces


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_trees() -> dict:
    """The git trees of src and scripts at HEAD: what `check` passed on. A docs-only commit leaves them as they were."""
    try:
        run = subprocess.run(["git", "rev-parse", "HEAD:src", "HEAD:scripts"], cwd=ROOT, capture_output=True, text=True)
        out = run.stdout.split() if run.returncode == 0 else []
    except OSError:
        out = []
    if len(out) != 2:
        raise SystemExit("git could not say the trees of src and scripts at HEAD: nothing is checked or trained on a code state that cannot be named")
    return dict(zip(("src", "scripts"), out))


def failed_list() -> list[dict]:
    return read_json(RECORDS / "failed.json") or []


def voided() -> list[dict]:
    """The sweep's nets moved aside as VOID so far: their cause, and how far each got."""
    return [json.loads(p.read_text()) for p in sorted(tn.NETS.glob("d3s-*-void-*/void.json"))]


def result_file(tag: str) -> Path:
    """The file whose existence ends a net's right to be voided (PREREG: the sweep's own "result exists")."""
    return RECORDS / {COMBO_TAG: "combo.json", CONFIRM_TAG: "confirm.json"}.get(tag, "selection.json")


def finished_nets() -> set:
    """Tags with a complete net on disk (200 epochs, net.pt, no GPU peer seen), not failed."""
    return {t for t in ALL_TAGS if t not in {f["tag"] for f in failed_list()} and tn.d2b_net_refused(t) is None
            and (tn.NETS / t / "record.json").exists()}


def nets_record(fp: str) -> dict:
    """nets.json from the disk: every finished net's row, the failed arms, the void list."""
    rows = []
    for t in ALL_TAGS:
        if t in finished_nets():
            rows.append(d3.net_row(json.loads((tn.NETS / t / "record.json").read_text()), json.loads((tn.NETS / t / "gpu.json").read_text())))
    return dict(nets=rows, failed=failed_list(), voided=voided(), union=fp)


def nets_refused(nets_json: dict, fp: str, tags: tuple) -> str | None:
    """Why nets.json may not be selected from: the nets expected, the data, the void list, the disk, the options of each row."""
    got, bad = [r["tag"] for r in nets_json["nets"]], [f["tag"] for f in nets_json["failed"]]
    if sorted(got + bad) != sorted(tags) or set(got) & set(bad):
        return f"nets.json holds {got} and failed {bad}, not the registered {list(tags)}"
    if nets_json.get("union") != fp:
        return f"nets.json was trained on data {nets_json.get('union')}, not {fp}"
    if nets_json["voided"] != voided():
        return f"nets.json lists {len(nets_json['voided'])} voided nets, {len(voided())} are on disk: rerun `train` (it rewrites nets.json)"
    for r in nets_json["nets"]:
        why = tn.d2b_net_refused(r["tag"])
        if why:
            return why
        want = options_of(table_arm(r["tag"])) if r["tag"] in ARM_TAGS else None
        if want is not None and options_of(r) != want:
            return f"{r['tag']}: trained with {options_of(r)}, the table says {want}"
    return tn.d2b_nets_refused(nets_json["nets"]) or tn.d2b_records_differ(nets_json["nets"])


def import_ls():
    return d3.import_ls()


def gpu_refused(ls) -> str | None:
    g = ls.gpu_state()
    if ls.busy(g) or tn.d2b_blind(g):
        return f"another process of this Python is on the GPU ({g['peers']}), or nvidia-smi could not say: stop it first"
    return None


# ------------------------------------------------------------------ scoring


def union_batches(views: dict, parts: dict, dev) -> dict:
    """The union's selection half in all and by source (the only rows the sweep scores)."""
    v, p = views["union"], parts["union"]
    return {lab: tn.Batches(v, np.flatnonzero((p == tn.SELECT) & keep), dev)
            for lab, keep in (("all", np.ones(len(v["source"]), bool)), ("real", v["source"] == 0), ("random", v["source"] == 1))}


def rw_loss(net, b) -> float:
    """The range-weighted Huber on a Batches, float32, eval: reported only (PREREG)."""
    import torch

    net.eval()
    tot = 0.0
    with torch.no_grad():
        for s in range(0, b.n, 4096):
            r, v, onehot, scal, live = b.get(torch.arange(s, min(s + 4096, b.n), device=b.dev), False)
            tot += float(tn._loss_rw(net(r, onehot, scal, live).float(), v, r)) * len(r)
    net.train()
    return tot / b.n


def score_net(net, batches: dict) -> dict:
    out = {}
    for lab, b in batches.items():
        e = tn.evaluate(net, b)
        out[lab] = dict(rows=b.n, loss=e["loss"], mae=e["mae"], rw_loss=rw_loss(net, b))
    return out


def free_gpu() -> None:
    """The training pools and samples are not needed once a net is trained: release them before the selection batches go up."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def load_net(tag: str):
    from parciver import value_net as vn

    return vn.load(tn.NETS / tag / "net.pt")[0]


def losses(score: dict | None) -> dict | None:
    return None if score is None else {lab: score[lab]["loss"] for lab in LABELS}


# ------------------------------------------------------------------ training


def void_net(tag: str, cause: str, ls) -> int:
    """PREREG, VOID and bugs: a sweep net's folder moved aside with its cause, kept and listed in nets.json; the next `train` repeats it."""
    if tag not in ALL_TAGS or not cause.strip():
        raise SystemExit(f"--void needs a cause, and --tag one of {list(ALL_TAGS)}")
    if tag in {f["tag"] for f in failed_list()}:
        raise SystemExit(f"{tag} FAILED (a non-finite loss): that is not VOID, and a failed arm is not retried")
    folder = tn.NETS / tag
    if not folder.exists():
        raise SystemExit(f"{folder}: no net to void")
    done = result_file(tag)
    if done.exists():
        raise SystemExit(f"{done} exists: voiding a net after its result needs an amendment first")
    why = gpu_refused(ls)
    if why:
        raise SystemExit(why + " (it may be this net's training)")
    rec = read_json(folder / "record.json") or {}
    n = 1 + len(list(tn.NETS.glob(f"{tag}-void-*")))
    while (tn.NETS / f"{tag}-void-{n}").exists():
        n += 1
    d3._write(folder / "void.json", json.dumps(dict(tag=tag, cause=cause, at=ls.utc(), folder=f"{tag}-void-{n}", commit=rec.get("commit"),
                                                    epochs=len(rec.get("history", [])), best=rec.get("best")), indent=1))
    folder.rename(tn.NETS / f"{tag}-void-{n}")
    print(f"VOID: {tag} moved aside to {tag}-void-{n}: {cause}")
    return 0


def run_arm(views: dict, parts: dict, arm: dict, seed: int, commit: str, ls, clients: set, held: dict) -> dict | None:
    """One net with GPU looks before it, every D2B_GPU_EVERY epochs and after its last. None: it failed (a non-finite loss), recorded."""
    tag, folder, seen = arm["tag"], tn.NETS / arm["tag"], []
    path = folder / "gpu.json"
    why = tn.d2b_net_refused(tag) or tn.d2b_net_differs(arm, views, seed)
    if why:
        raise SystemExit(f"{why}: this run is VOID (PREREG). `d3_sweep train --void \"<cause>\" --tag {tag}` keeps it with its cause, then this repeats it")
    fresh = not path.exists()

    def look():
        g = ls.gpu_state()
        seen.append(g)
        new = sorted(set(g["apps"]) - clients)
        if new:
            print(f"  {tag}: GPU clients new at this look (a compute job voids the net, read them): {new}", flush=True)
            clients.update(new)
        if len(seen) == 1 and g["peers"]:
            raise SystemExit(f"{tag}: another process of this Python on the GPU ({g['peers']}): nothing trained, nothing changed; rerun when it has gone")
        if fresh or len(seen) > 1:
            folder.mkdir(parents=True, exist_ok=True)
            d3._write(path, json.dumps(seen, indent=1))
        if g["peers"]:
            raise SystemExit(f"{tag}: another process of this Python on the GPU ({g['peers']}): this run is VOID. "
                             f"`d3_sweep train --void \"<cause>\" --tag {tag}`, then rerun")

    try:
        return tn.d2b_train_arm(views, parts, arm, held, look, seed, commit=commit)
    except FloatingPointError as e:
        rec = read_json(folder / "record.json") or {}
        res = failed_list() + [dict(tag=tag, error=str(e), epochs=len(rec.get("history", [])), commit=commit, options=options_of(arm))]
        d3._write(RECORDS / "failed.json", json.dumps(res, indent=1))
        print(f"  {tag}: FAILED, {e}: recorded; it cannot win and is not retried")
        return None


def setup(a):
    tn.NETS = Path(a.data).parent / "turn-net"
    return import_ls()


def guard(a, ls, need_told: bool = True) -> tuple[str, dict, dict, dict, str]:
    """Everything an arm needs before it trains: a clean commit, the pinned data, `check` passed on this code, the peak commit told and below
    the free commit, no other GPU job. Returns (commit, views, parts, check.json, the union's fingerprint)."""
    from parciver import value_net  # noqa: F401  # loaded before the commit is read

    commit = d3.need_commit()
    free_before = d3.free_commit_gib()  # before the data is loaded: the measured peak includes the data the process is about to hold
    views, parts = tn.d2b_data(a)
    union = tn.d2b_refuse_other_data(views)
    chk = read_json(RECORDS / "check.json")
    why = check_refused(chk, code_trees(), union)
    if why:
        raise SystemExit(why)
    if need_told:
        why = d3.memory_refused(chk, int((parts["union"] == tn.TRAIN).sum()), free_before, a.told)
        if why:
            raise SystemExit(why)
    why = gpu_refused(ls)
    if why:
        raise SystemExit(why)
    return commit, views, parts, chk, union


def cmd_check(a) -> int:
    import torch

    from parciver import value_net as vn  # noqa: F401

    ls = setup(a)
    commit = d3.need_commit()
    out = RECORDS / "check.json"
    if getattr(a, "redo", False):
        for old in (tn.NETS / CHECK_TAG, out):
            if dl.aside(old):
                print(f"moved aside: {old}")
    elif out.exists() or (tn.NETS / CHECK_TAG).exists():
        raise SystemExit(f"{out} or {tn.NETS / CHECK_TAG} exists: `check --redo` moves both aside (kept) and runs again")
    why = gpu_refused(ls)
    if why:
        raise SystemExit(why)
    sel = json.loads((tn.D2B_RECORDS / "selection.json").read_text())
    if sel["n1"] != N1_TAG or d3.sha_of(N1_TAG) != dl.N1_SHA:
        raise SystemExit(f"records/d2b/selection.json names {sel['n1']}, or {N1_TAG}'s net.pt is not N1 ({dl.N1_SHA})")
    n1 = json.loads((tn.NETS / N1_TAG / "record.json").read_text())
    views, parts = tn.d2b_data(a)
    union = tn.d2b_refuse_other_data(views)
    t0, free0, gpu = time.time(), d3.free_commit_gib(), []

    def look():
        g = ls.gpu_state()
        gpu.append(g)
        if g["peers"]:
            raise SystemExit(f"another process of this Python on the GPU ({g['peers']}): the check is void; rerun when it has gone")

    held = {}
    # 1. N1's recipe through the new code, options off, the first epochs of its own 200-epoch schedule.
    rec = tn.d2b_train_arm(views, parts, dict(arm_of(CHECK_TAG)), held, look, None, commit=commit, stop_after=CHECK_EPOCHS)
    held.clear()
    free_gpu()
    history = compare_history(rec, n1, CHECK_EPOCHS)
    (tn.NETS / CHECK_TAG / "net.pt").unlink(missing_ok=True)  # the record is what is kept; the net is not used
    # 2. The two saved controls, rescored by source on the selection half, against records/d2b/selection.json.
    batches = union_batches(views, parts, torch.device("cuda"))
    rescored = {}
    for tag in (N1_TAG, SEED1_TAG):
        net = load_net(tag)
        got = score_net(net, batches)
        del net
        rescored[tag] = dict(scores=got, equal=compare_scores(got, sel["scores"][tag]["union"]))
    # 3. The host peak at this size, for the memory guard.
    mem = dl.memory()
    control = {t: {lab: sel["scores"][t]["union"][lab]["loss"] for lab in LABELS} for t in (N1_TAG, SEED1_TAG)}
    unknown = sum(tn.d2b_blind(g) for g in gpu)
    passed = bool(history["equal"] and all(all(r["equal"].values()) for r in rescored.values()) and unknown < len(gpu))
    res = dict(passed=passed, history=history, rescored=rescored, control=control, commit=commit, trees=code_trees(), union=union,
               train_rows=int((parts["union"] == tn.TRAIN).sum()), pools=tn.D2B_POOLS, epochs_compared=CHECK_EPOCHS,
               check_history=rec["history"], n1_history=n1["history"][:CHECK_EPOCHS], d2b_selection_sha256=sha_file(tn.D2B_RECORDS / "selection.json"),
               seconds=round(time.time() - t0, 1), free_commit_before=free0, free_commit_after=d3.free_commit_gib(),
               gpu_looks=len(gpu), gpu_unknown=unknown, **{k: round(x, 2) for k, x in mem.items()})
    d3._write(out, json.dumps(res, indent=1))
    print(f"check: history {'equal' if history['equal'] else 'DIFFERS ' + str(history)}; N1 rescore {rescored[N1_TAG]['equal']}; "
          f"seed-1 rescore {rescored[SEED1_TAG]['equal']}; peak commit {res['peak_commit']:.1f} GiB, working set {res['peak_working_set']:.1f}")
    if not passed:
        print("nothing else is used until the cause is found and an amendment records it (PREREG)")
        return 1
    print(f"PASSED. Tell the author the measured peak commit ({res['peak_commit']:.1f} GiB, free now {d3.free_commit_gib():.1f}), then "
          f"`d3_sweep train --told {res['peak_commit']:.1f}`")
    return 0


def cmd_train(a) -> int:
    ls = setup(a)
    if a.void is not None:
        return void_net(a.tag, a.void, ls)
    commit, views, parts, chk, union = guard(a, ls)
    sel, combo = read_json(RECORDS / "selection.json"), read_json(RECORDS / "combo.json")
    todo = train_plan(a.arm, finished_nets(), {f["tag"] for f in failed_list()}, sel, combo, a.redo)
    if a.redo:
        for t in todo:
            if dl.aside(tn.NETS / t):
                print(f"moved aside: {tn.NETS / t}")
    clients, held = set(), {}
    for t in todo:
        arm = arm_of(COMBO_TAG, **sel["combo"]["options"]) if t == COMBO_TAG else table_arm(t)
        rec = run_arm(views, parts, arm, tn.LEAFNET_SETTINGS["seed"], commit, ls, clients, held)
        res = nets_record(union)
        d3._write(RECORDS / "nets.json", json.dumps(res, indent=1))
        if rec is not None:
            print(f"  {t}: selection loss {rec['best']['val_loss']:.6f} (epoch {rec['best']['epoch']})")
    print(f"GPU clients seen (desktop apps do not void a net, a compute job does): "
          f"{sorted({x for r in res['nets'] for x in r['gpu_apps']})}; failed arms: {[f['tag'] for f in res['failed']]}; "
          f"voided nets: {len(res['voided'])}: read them before `d3_sweep select`")
    return 0


def print_scores(scores: dict, control: dict, won: dict | None = None) -> None:
    for t, s in scores.items():
        if s is None:
            print(f"  {t:<14s} FAILED")
            continue
        l = losses(s)
        print(f"  {t:<14s} loss {l['all']:.6f} ({100 * (l['all'] / control[N1_TAG]['all'] - 1):+.2f}% vs N1)  real {l['real']:.6f}  random {l['random']:.6f}  "
              f"MAE {s['all']['mae']:.3f}%  rw {s['all']['rw_loss']:.6f}" + ("" if won is None else f"  {'WINS' if won[t] else 'no'}"))


def cmd_select(a) -> int:
    import torch

    setup(a)
    sel, combo = read_json(RECORDS / "selection.json"), read_json(RECORDS / "combo.json")
    if sel is None:
        stage, tags = "round1", ARM_TAGS
    elif sel["combo"] is not None and combo is None:
        stage, tags = "combo", (*ARM_TAGS, COMBO_TAG)
    else:
        raise SystemExit("selection.json" + (" and combo.json exist" if combo else " exists and no combo is called for") + ": nothing left to select (`confirm`)")
    views, parts = tn.d2b_data(a)
    union = tn.d2b_refuse_other_data(views)
    chk = read_json(RECORDS / "check.json")
    why = check_refused(chk, code_trees(), union)
    if why:
        raise SystemExit(why)
    if chk["d2b_selection_sha256"] != sha_file(tn.D2B_RECORDS / "selection.json"):
        raise SystemExit("records/d2b/selection.json is not the file `check` rescored against")
    nets_json = read_json(RECORDS / "nets.json")
    why = nets_refused(nets_json, union, tags) if nets_json else "nets.json does not exist"
    if why:
        raise SystemExit(f"nets.json: {why}")
    failed = {f["tag"] for f in nets_json["failed"]}
    batches = union_batches(views, parts, torch.device("cuda"))
    scores, match = {}, {}
    for t in tags:
        if t in failed:
            scores[t] = None
            continue
        net = load_net(t)
        scores[t] = score_net(net, batches)
        del net
        match[t] = scores[t]["all"]["loss"] == json.loads((tn.NETS / t / "record.json").read_text())["best"]["val_loss"]
    control = chk["control"]
    print(f"controls: N1 {control[N1_TAG]['all']:.6f}  real {control[N1_TAG]['real']:.6f}; seed 1 {control[SEED1_TAG]['all']:.6f}")
    if not all(match.values()):
        print_scores(scores, control)
        print(f"a saved net does not score what its record says ({[t for t, m in match.items() if not m]}): no decision until the cause is found")
        return 1
    common = dict(control=control, bars=dict(win=BAR, combo=COMBO_BAR, confirm=CONFIRM_BAR), scores=scores, matches_record=match,
                  failed=sorted(failed), sha256={t: d3.sha_of(t) for t in tags if scores[t] is not None}, data=union, check_commit=chk["commit"])
    if stage == "round1":
        r1 = round1({t: losses(scores[t]) for t in ARM_TAGS}, control[N1_TAG])
        res = dict(common, stage="round1", **r1)
        print_scores(scores, control, r1["wins"])
        print(f"best of each winning family: {r1['family_best']}; best single winner: {r1['single']}; combo: "
              f"{'called ' + json.dumps(r1['combo']['options']) if r1['combo'] else 'not called'}")
        d3._write(RECORDS / "selection.json", json.dumps(res, indent=1))
        return 0
    one = sel["single"]
    dec = combo_decision(one, losses(scores[one]), losses(scores[COMBO_TAG]), control[N1_TAG])
    res = dict(common, stage="combo", single=one, **dec, single_score=losses(scores[one]), combo_score=losses(scores[COMBO_TAG]))
    print_scores({t: scores[t] for t in (one, COMBO_TAG)}, control)
    print(f"combo passes rule 2: {dec['combo_passes_rule2']}; beats the single winner by the 0.99 bar: {dec['combo_beats_single']}; "
          f"candidate for the confirmation: {dec['candidate']}")
    d3._write(RECORDS / "combo.json", json.dumps(res, indent=1))
    return 0


def cmd_confirm(a) -> int:
    import torch

    ls = setup(a)
    sel, combo = read_json(RECORDS / "selection.json"), read_json(RECORDS / "combo.json")
    out = RECORDS / "confirm.json"
    if sel is None:
        raise SystemExit("selection.json does not exist: `select` first")
    if out.exists():
        raise SystemExit(f"{out} exists: the confirmation is written once (a bug needs an amendment first)")
    cand, why = candidate_of(sel, combo)
    if cand is None:
        raise SystemExit(f"the rule does not call for a confirmation: {why}")
    commit, views, parts, chk, union = guard(a, ls, need_told=(CONFIRM_TAG not in finished_nets()))
    arm = candidate_arm(sel, cand, CONFIRM_TAG)
    failed = {f["tag"] for f in failed_list()}
    if CONFIRM_TAG not in failed and CONFIRM_TAG not in finished_nets():
        run_arm(views, parts, arm, 1, commit, ls, set(), {})
        free_gpu()
        d3._write(RECORDS / "nets.json", json.dumps(nets_record(union), indent=1))
        failed = {f["tag"] for f in failed_list()}
    control = chk["control"][SEED1_TAG]
    if CONFIRM_TAG in failed:
        res = dict(candidate=cand, options=options_of(arm), failed=True, confirmed=False, control=control, data=union, check_commit=chk["commit"])
        print(f"{CONFIRM_TAG} FAILED (a non-finite loss): not confirmed")
    else:
        nets_json = read_json(RECORDS / "nets.json")
        tags = (*ARM_TAGS, *([COMBO_TAG] if sel["combo"] else []), CONFIRM_TAG)
        why = nets_refused(nets_json, union, tags) if nets_json else "nets.json does not exist"
        if why:
            raise SystemExit(f"nets.json: {why}")
        net = load_net(CONFIRM_TAG)
        score = score_net(net, union_batches(views, parts, torch.device("cuda")))
        del net
        rec = json.loads((tn.NETS / CONFIRM_TAG / "record.json").read_text())
        match = score["all"]["loss"] == rec["best"]["val_loss"]
        if not match:
            raise SystemExit(f"{CONFIRM_TAG} does not score what its record says: no decision until the cause is found")
        ok = confirmed(losses(score), control)
        res = dict(candidate=cand, options=options_of(arm), failed=False, confirmed=ok, control=control, score=score, bar=CONFIRM_BAR,
                   matches_record=match, sha256=d3.sha_of(CONFIRM_TAG), data=union, check_commit=chk["commit"], best=rec["best"])
        print(f"{CONFIRM_TAG} (seed 1, {cand}'s recipe): loss {score['all']['loss']:.6f} vs bar {CONFIRM_BAR * control['all']:.6f}; real "
              f"{score['real']['loss']:.6f} vs {control['real']:.6f}: {'CONFIRMED' if ok else 'NOT CONFIRMED'}")
    d3._write(out, json.dumps(res, indent=1))
    return 0


def verdict_of(sel: dict, combo: dict | None, conf: dict | None) -> dict:
    """The pin: N1's recipe, or the one confirmed winner."""
    cand, why = candidate_of(sel, combo)
    if cand is None:
        return dict(pin="N1", recipe=dict(OFF), candidate=None, why=why)
    if conf is None:
        raise SystemExit("a candidate is called for and confirm.json does not exist: `confirm` first")
    if not conf["confirmed"]:
        return dict(pin="N1", recipe=dict(OFF), candidate=cand, why=f"{cand} was not confirmed at seed 1: N1's recipe, no second place and no other seed")
    return dict(pin=cand, recipe=conf["options"], candidate=cand, why=f"{cand} was confirmed at seed 1")


def result_md(v: dict, sel: dict, combo: dict | None, conf: dict | None) -> str:
    c = sel["control"][N1_TAG]
    rows = ["| arm | loss | vs N1 | real | random | MAE % pot | wins |", "|---|---|---|---|---|---|---|"]
    for t in ARM_TAGS:
        s = sel["scores"][t]
        if s is None:
            rows.append(f"| `{t}` | FAILED (non-finite loss) | | | | | no |")
            continue
        rows.append(f"| `{t}` | {s['all']['loss']:.6f} | {100 * (s['all']['loss'] / c['all'] - 1):+.2f}% | {s['real']['loss']:.6f} | "
                    f"{s['random']['loss']:.6f} | {s['all']['mae']:.3f} | {'yes' if sel['wins'][t] else 'no'} |")
    lines = ["# D3-SWEEP result", "", f"**Pin: {v['pin']}.** {v['why']}.", "",
             f"Control (N1, `{N1_TAG}`): loss {c['all']:.6f}, real {c['real']:.6f}, random {c['random']:.6f}. Win bar: loss <= {BAR * c['all']:.6f} "
             f"and real <= {c['real']:.6f}.", "", *rows, ""]
    lines.append(f"Best of each winning family: {sel['family_best'] or 'none'}. Best single winner: {sel['single']}. Combo: "
                 f"{json.dumps(sel['combo']['options']) if sel['combo'] else 'not called'}.")
    if combo:
        lines.append(f"Combo: passes rule 2 {combo['combo_passes_rule2']}; beats the single winner by 0.99 {combo['combo_beats_single']}; "
                     f"candidate {combo['candidate']}.")
    if conf:
        c1 = conf["control"]
        lines.append(f"Confirmation (`{CONFIRM_TAG}`, seed 1, recipe {json.dumps(conf['options'])}): "
                     + ("FAILED (non-finite loss)." if conf["failed"] else
                        f"loss {conf['score']['all']['loss']:.6f} (bar {CONFIRM_BAR * c1['all']:.6f}), real {conf['score']['real']['loss']:.6f} "
                        f"(control {c1['real']:.6f}): {'CONFIRMED' if conf['confirmed'] else 'NOT CONFIRMED'}."))
    opts = v["recipe"]
    if opts.get("ema", 0) > 0:
        d = opts["ema"]
        lines += ["", f"EMA pin: d = {d} at 464 steps an epoch is a horizon of {ema_horizon_epochs(d, 464):.2f} epochs; N2 (about 928 steps an epoch) "
                      f"uses the per-step d that gives the same horizon there ({ema_d_for(ema_horizon_epochs(d, 464), 928):.6f} at 928 steps; the amendment states it "
                      f"for the exact step count once D3's union is pinned)."]
    lines += ["", "Limits (accepted in the registration): winner's curse on the shared selection half; a weak confirmation (one seed pair); the real-row clause "
              "is close to a coin flip for a small effect; the range-weighted arm is trained toward MAE and judged on the plain Huber, and stands in for D3's "
              "\"EV-space loss\" (it is not DEVN). A recipe tuned at 951k rows is carried, not proven, at ~1.9M: Gate A on D1c's gate half tests it.", "",
              "Next: an amendment to records/d3/PREREG.md naming only this pin, committed before D1c's first label is read; `N2_ARM` in scripts/d3.py updated to match."]
    return "\n".join(lines) + "\n"


def cmd_verdict(a) -> int:
    setup(a)
    sel, combo, conf = (read_json(RECORDS / f) for f in ("selection.json", "combo.json", "confirm.json"))
    if sel is None:
        raise SystemExit("selection.json does not exist: `select` first")
    v = verdict_of(sel, combo, conf)
    res = dict(v, selection=sel, combo=combo, confirm=conf, voided=voided(), failed=failed_list())
    d3._write(RECORDS / "verdict.json", json.dumps(res, indent=1))
    d3._write(RECORDS / "RESULT.md", result_md(v, sel, combo, conf))
    print(f"pin: {v['pin']} ({v['why']}); recipe {json.dumps(v['recipe'])}: records/d3-sweep/verdict.json and RESULT.md written; the author sees it before the amendment")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "runs" / "turn-data"), help="D1's folder; D1b's is beside it")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").add_argument("--redo", action="store_true", help="move an earlier check (net folder and json) aside, then run again")
    t = sub.add_parser("train")
    t.add_argument("--arm", default=None, help="one arm of the table (or d3s-combo when the rule calls it); default: the round-1 arms not finished")
    t.add_argument("--redo", action="store_true", help="move a finished arm aside (kept), then train it again, before its stage's result exists")
    t.add_argument("--told", type=float, default=None, help="the measured peak commit (GiB) as told to the author")
    t.add_argument("--void", metavar="CAUSE", default=None, help="this net's run is VOID, for this cause: move it aside")
    t.add_argument("--tag", default=None, help="--void: the net")
    sub.add_parser("select")
    sub.add_parser("confirm").add_argument("--told", type=float, default=None, help="the measured peak commit (GiB) as told to the author")
    sub.add_parser("verdict")
    a = ap.parse_args()
    if a.cmd == "train" and (a.void is not None) != (a.tag is not None):
        raise SystemExit("--void and --tag go together")
    return {"check": cmd_check, "train": cmd_train, "select": cmd_select, "confirm": cmd_confirm, "verdict": cmd_verdict}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
