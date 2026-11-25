#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Scout: do VAD-CFR / WOP-CFR components or HS-DCFR schedules beat the live DCFR
(alpha 1.5, beta 0, gamma 2, alternating) at the real-time search budget
(100 and 400 iterations) on exact exploitability? Offline only; the live bot and
src/ are untouched. Plan and pass rule: records/vad-scout/PREREG.md.

    .venv/Scripts/python.exe scripts/vad_scout.py --out records/vad-scout/raw.json

The `live` arm must reproduce `gpu_search.GpuSolver` bit for bit (checked first).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from parciver import gpu_search as gs  # noqa: E402


class VarSolver(gs.GpuSolver):
    """GpuSolver with pluggable discounts. Defaults == live DCFR exactly."""

    def __init__(self, *a, cfg: dict | None = None, **k):
        super().__init__(*a, **k)
        c = dict(pos=lambda t: t ** 1.5 / (t ** 1.5 + 1.0), neg=lambda t: 0.5, gam=lambda t: 2.0,
                 warm=0, power=1.0, opt=None, boost=1.0)
        c.update(cfg or {})
        self.c = c
        self.inst = torch.zeros_like(self.regret) if c["opt"] is not None else None

    def begin(self):
        self.iter += 1
        t = float(self.iter)
        c = self.c
        r = self.regret
        r.copy_(torch.where(r > 0, r * c["pos"](t), r * c["neg"](t)))
        self.sum_w = (t / (t + 1.0)) ** c["gam"](t)
        if c["warm"] and self.iter == c["warm"] + 1:
            self.sum.zero_()  # hard warm start: nothing accumulated before iteration warm+1
        self.t = t

    def _current(self, r):
        c = self.c
        if c["power"] == 1.0 and c["opt"] is None:
            return super()._current(r)
        x = r
        if c["opt"] is not None:
            x = r + c["opt"](self.t) * self.inst
        pos = x.clamp(min=0) * self.mask
        if c["power"] != 1.0:
            pos = pos ** c["power"]
        tot = pos.sum(1, keepdim=True)
        return torch.where(tot > 0, pos / tot.clamp(min=1e-30), self.mask / self.kcount)

    def traverse(self, p: int):
        c = self.c
        if c["boost"] == 1.0 and c["opt"] is None:
            return super().traverse(p)
        sig = self._current(self.regret)
        rq, _ = self._down(p, sig, both=False, update_sum=self.sum_w)
        val = self._terminals(p, rq)
        for L in range(self.max_depth, -1, -1):
            self._chance_up(L, val)
            d = self.dec_levels[L]
            if len(d):
                vc = val[self.children[d]]
                s = sig[d]
                isp = (self.dec_player[d] == p)[:, None, None]
                v = torch.where(isp[:, :, 0], (s * vc).sum(1), vc.sum(1))
                val[self.dec_node[d]] = v
                inst = torch.where(isp, (vc - v[:, None, :]) * self.mask[d], 0.0)
                if c["boost"] != 1.0:
                    inst = torch.where(inst > 0, inst * c["boost"], inst)
                self.regret[d] += inst
                if self.inst is not None:
                    self.inst[d] = torch.where(isp, inst, self.inst[d])


def arms(T: int) -> dict[str, dict]:
    """Every arm as a cfg for VarSolver; T = the budget (needed by schedules and the scaled warm start)."""
    vneg = lambda t: t ** -0.1 / (t ** -0.1 + 1.0)  # noqa: E731
    copt = lambda t: 1.0 / (1.0 + t / 100.0)  # noqa: E731
    W = int(0.25 * T)
    a = {"live": {}}
    # WOP-CFR / VAD-CFR components, one at a time, then together.
    a["vneg"] = dict(neg=vneg)
    a["pow15"] = dict(power=1.5)
    a["opt"] = dict(opt=copt)
    a["boost"] = dict(boost=1.1)
    a["warm25"] = dict(warm=W)
    a["wop"] = dict(neg=vneg, power=1.5, opt=copt, warm=W)  # the distilled core (no boost)
    a["vad"] = dict(neg=vneg, power=1.5, opt=copt, boost=1.1, warm=W)
    a["vad_nowarm"] = dict(neg=vneg, power=1.5, opt=copt, boost=1.1)
    # HS-DCFR (arXiv 2404.09097): linear schedules over the known budget n = T.
    hs = dict(pos=lambda t: (t ** (1 + 3 * t / T)) / (t ** (1 + 3 * t / T) + 1.0),
              neg=lambda t: (t ** (-1 - 2 * t / T)) / (t ** (-1 - 2 * t / T) + 1.0))
    a["hs30"] = dict(hs, gam=lambda t: 30 - 5 * t / T)
    a["hs15"] = dict(hs, gam=lambda t: 15 - 5 * t / T)
    return a


def spots():
    """(name, board cards, spec, range seed pair, peaky). River and a small turn, several boards."""
    out = []
    for i, seed in enumerate((7, 11, 23, 42, 77, 101)):
        b = gs.board(seed)
        peaky = i % 2 == 1
        out.append((f"river-b{seed}{'-peaky' if peaky else ''}", b, gs.spec(), (31 + i, 61 + i), peaky))
    for i, seed in enumerate((7, 11, 23, 42, 77, 101)):
        b = gs.board(seed)[:4]
        peaky = i % 2 == 1
        out.append((f"turn-b{seed}{'-peaky' if peaky else ''}", b, gs.small_turn_spec(), (31 + i, 61 + i), peaky))
    return out


def build(board, sp, rs, peaky, cfg, dev):
    hands = gs.Hands(board)
    tree = gs.Tree(sp, hands)
    rng = [gs.range_(hands.n, rs[0]), gs.range_(hands.n, rs[1])]
    if peaky:
        rng = [r ** 3 for r in rng]
    return VarSolver(hands, tree, rng, device=dev, dtype=torch.float32, cfg=cfg), hands, tree, rng


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="records/vad-scout/raw.json")
    ap.add_argument("--budgets", default="100,400")
    ap.add_argument("--only", default="", help="substring filter on spot names (smoke runs)")
    ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    budgets = [int(x) for x in args.budgets.split(",")]
    res = {}
    t0 = time.time()
    for name, board, sp, rs, peaky in spots():
        if args.only and args.only not in name:
            continue
        # Equivalence gate: live arm == GpuSolver, bit for bit.
        s, hands, tree, rng = build(board, sp, rs, peaky, {}, args.dev)
        g = gs.GpuSolver(hands, tree, rng, device=args.dev, dtype=torch.float32)
        for _ in range(50):
            s.step()
            g.step()
        if name.startswith("river"):
            assert torch.equal(s.regret, g.regret) and torch.equal(s.sum, g.sum), f"live arm differs on {name}"
        else:  # turn: GpuSolver's index_add_ is nondeterministic (two stock runs differ), so compare exploitability
            ea, eb = s.exploitability(), g.exploitability()
            assert abs(ea - eb) <= 0.03 * eb, f"live arm differs on {name}: {ea} vs {eb}"
        res[name] = {}
        for T in budgets:
            for arm, cfg in arms(T).items():
                s, *_ = build(board, sp, rs, peaky, cfg, args.dev)
                for _ in range(T):
                    s.step()
                ex = s.exploitability()
                res[name].setdefault(arm, {})[str(T)] = ex
            row = {a: res[name][a][str(T)] for a in res[name]}
            print(f"{name} T={T} " + " ".join(f"{a}={v:.4f}" for a, v in row.items()), flush=True)
        print(f"  [{time.time() - t0:.0f}s]", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
