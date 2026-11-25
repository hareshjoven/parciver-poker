# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""The flop search's net leaves through the value net's linear ends
(KERNELS.md, N1): `flop_search.NetLeaves(board, net_values(net))` without the
49 dense first and last layers per leaf.

A leaf's 49 turn boards share its two reaches. Per turn card t only the
card-removal mask, the mass off t and the board one-hot change, and the
solver uses one thing: the opponent-mass-weighted sum over t of the values.

* First layer. `A_q (r * mask_t) / m_t` for player q's columns `A_q`: the
  product over the holdings off t is the product over all holdings minus the
  48 that hold t. A difference of two rounded sums loses digits when nearly
  all of the mass sits on those 48, so the 48 heaviest holdings of each range
  are kept out of both sums and masked exactly: the rest's mass is then at
  most the mass off t (the 48 heaviest outweigh any 48), and the difference is
  taken between numbers no larger than its result's scale.
* Hidden layers: as the net's, on the 49 boards of every leaf.
* Last layer. With `g_t = M_t [h_t, 1]` (`M_t` the opponent's mass off t,
  `h_t` the last hidden activation) and `W_p` = player p's rows with the bias
  as a last column: `sum_t M_t mask_t[h] (W_p[h] . [h_t, 1])
  = W_p[h] . G - W_p[h] . g_a - W_p[h] . g_b`, a and b the holding's cards,
  `G = sum_t g_t`. The zero-sum term `eps_t = sum_q u_q,t . (W_q [h_t, 1])`
  is `(W_q^T u_q,t) . [h_t, 1]`, its range side computed as the first layer's.

Same values as the dense path up to rounding (another summation order):
tests/test_flop_leaves.py, float64 against the dense forward.

`last=torch.float64` (KERNELS.md, N1b) takes the last stage alone in float64
under a float32 net: the opponent's masses, `g`, `G`, the `W_p` products,
the per-card corrections and `e = M_t eps_t`; the result is rounded to the
reach's dtype once. Most of either path's float32 error is that layer's dot
products. `eps_t`'s range side stays float32 there; `last_eps=True` (N1c)
takes it in `last` too.
"""
from __future__ import annotations

import numpy as np
import torch

from . import value_net
from .flop_search import TURN_SHARE, TURNS, combos

DEAD = TURNS - 1  # holdings of a flop that hold a given turn card


class FactoredLeaves:
    """`NetLeaves` for a `value_net.ValueNet`: `__call__(reach [L, 2, n],
    half [L], stack [L])` -> [L, 2, n] chips, each player's counterfactual
    values at each leaf; `players=(p,)` computes only that player's. `last`:
    the last stage's dtype (the net's if None); `last_eps`: the zero-sum
    term's range side in it as well (nothing changes when `last` is the net's
    dtype)."""

    def __init__(self, board, net, device="cuda", dtype=torch.float32, last=None, last_eps=False):
        flop = [int(c) for c in board]
        hold = combos(flop)
        n = len(hold)
        self.turns = [c for c in range(52) if c not in flop]
        turn_of = {c: t for t, c in enumerate(self.turns)}
        card = np.array([[turn_of[int(a)], turn_of[int(b)]] for a, b in hold])  # [n, 2]: a holding's cards as turns
        dead: list[list[int]] = [[] for _ in self.turns]
        at = np.zeros((2, n), dtype=np.int64)  # where a holding sits in its cards' dead lists, flattened [T * 48]
        for h in range(n):
            for k in (0, 1):
                t = card[h, k]
                at[k, h] = t * DEAD + len(dead[t])
                dead[t].append(h)
        dead = np.array(dead)
        assert dead.shape == (TURNS, DEAD)
        if getattr(net, "canonical", False):  # it reads the MLP directly and would skip the frame
            raise ValueError("FactoredLeaves cannot run a canonical net (records/leaf-net/PREREG.md)")
        first, out = net.mlp[0], net.mlp[-1]
        self.hidden = net.mlp[1:-1]
        self.wd = wd = first.weight.dtype
        self.ld = ld = last or wd
        self.last_eps = last_eps and ld != wd
        H = value_net.HOLDINGS
        idx = torch.as_tensor(value_net.PAIR_INDEX[hold[:, 0], hold[:, 1]], device=device)
        t = torch.as_tensor(self.turns, device=device)
        h = torch.as_tensor(hold, device=device)
        self.dead = torch.as_tensor(dead, device=device)
        self.at = torch.as_tensor(at, device=device)
        self.card = torch.as_tensor(card, device=device)
        self.mask = ((h[None, :, 0] != t[:, None]) & (h[None, :, 1] != t[:, None])).to(wd)  # [T, n]
        uniform = self.mask / self.mask.sum(1, keepdim=True)
        onehot = torch.zeros(TURNS, 52, dtype=wd, device=device)
        onehot[:, flop] = 1.0
        onehot[torch.arange(TURNS, device=device), t] = 1.0
        with torch.no_grad():
            w1, wo = first.weight, out.weight
            # Input side, per player: rows of [n, w] (first layer) and [n, w + 1] (last layer and its bias).
            self.A = [w1[:, q * H + idx].T.contiguous() for q in (0, 1)]
            self.W = [torch.cat([wo[q * H + idx], out.bias[q * H + idx, None]], 1) for q in (0, 1)]
            self.A_dead = [a[self.dead] for a in self.A]  # [T, 48, w]
            self.W_dead = [w[self.dead] for w in self.W]
            self.W_last = [w.to(ld) for w in self.W]  # the same tensors when `last` is the net's dtype
            self.W_last_dead = [w.to(ld) for w in self.W_dead]
            self.mask_last = self.mask.to(ld)
            self.A_uniform = [uniform @ a for a in self.A]  # [T, w]: a zero-mass range enters as uniform
            self.W_uniform = [uniform @ w for w in self.W]
            if self.last_eps:
                uniform_last = self.mask_last / self.mask_last.sum(1, keepdim=True)
                self.W_last_uniform = [uniform_last @ w for w in self.W_last]
            self.board = onehot @ w1[:, 2 * H:2 * H + 52].T + first.bias  # [T, w]
            self.scal = w1[:, 2 * H + 52:].T.contiguous()  # [2, w]
            self.scal_mean, self.scal_std = net.scal_mean.to(wd), net.scal_std.to(wd)

    def _ranges(self, r, top, live, safe, rows, rows_dead, rows_uniform):
        """[L, T, width]: `(r * mask_t / m_t) @ rows` for every turn t (the
        uniform range's where m_t is 0), one player. `top`: the (values,
        indices) of the 48 heaviest holdings of each row of `r`."""
        heavy = torch.bmm(self.mask[:, top[1]].permute(1, 0, 2) * top[0][:, None, :], rows[top[1]])  # masked exactly
        light = r.scatter(1, top[1], 0.0)
        rest = (light @ rows)[:, None] - torch.bmm(light[:, self.dead].permute(1, 0, 2), rows_dead).permute(1, 0, 2)
        return torch.where(live[..., None], (heavy + rest) / safe[..., None], rows_uniform[None])

    def __call__(self, reach, half, stack, players=(0, 1)):
        L = reach.shape[0]
        with torch.no_grad():
            r = reach.to(self.wd)
            mass = (r[:, None] * self.mask[None, :, None, :]).sum(-1)  # [L, T, 2], as the dense path's
            live = mass > 0
            safe = torch.where(live, mass, 1.0)
            top = [torch.topk(r[:, q], DEAD, dim=1) for q in (0, 1)]
            hh, s = half.double(), stack.double()
            scal = torch.stack([hh / (hh + s), s / (2 * hh)], -1).to(self.wd)  # value_net.scalars
            z = self.board[None] + (((scal - self.scal_mean) / self.scal_std) @ self.scal)[:, None]
            for q in (0, 1):
                z = z + self._ranges(r[:, q], top[q], live[..., q], safe[..., q],
                                     self.A[q], self.A_dead[q], self.A_uniform[q])
            h = self.hidden(z.reshape(L * TURNS, -1)).view(L, TURNS, -1)
            h1 = torch.cat([h, h.new_ones(L, TURNS, 1)], -1)  # [L, T, w + 1]
            if self.last_eps:  # N1c: eps's ranges, masses and rows in the last stage's dtype (the same 48 kept out)
                h1 = h1.to(self.ld)
                r_eps = reach.to(self.ld)
                mass = (r_eps @ self.mask_last.T).transpose(1, 2)  # the last stage's masses (zero where `live` is not)
                safe_eps = torch.where(live, mass, 1.0)
                top_eps = [(r_eps[:, q].gather(1, top[q][1]), top[q][1]) for q in (0, 1)]
                rows = [(self.W_last[q], self.W_last_dead[q], self.W_last_uniform[q]) for q in (0, 1)]
            else:
                r_eps, safe_eps, top_eps = r, safe, top
                rows = [(self.W[q], self.W_dead[q], self.W_uniform[q]) for q in (0, 1)]
            eps = h1.new_zeros(L, TURNS)
            for q in (0, 1):
                eps = eps + (self._ranges(r_eps[:, q], top_eps[q], live[..., q], safe_eps[..., q], *rows[q]) * h1).sum(-1)
            pot = (2 * half).to(self.ld) / TURN_SHARE
            h1, eps = h1.to(self.ld), eps.to(self.ld)
            if self.ld != self.wd and not self.last_eps:  # the last stage's masses in its dtype (the first layer keeps the dense path's)
                mass = (reach.to(self.ld) @ self.mask_last.T).transpose(1, 2)
            out = []
            for p in players:
                m = mass[..., 1 - p]  # the opponent's mass off each turn card
                g = m[..., None] * h1
                both = g.sum(1) @ self.W_last[p].T  # [L, n]: every turn card, the holding's own two included
                own = torch.bmm(self.W_last_dead[p], g.permute(1, 2, 0)).permute(2, 0, 1).reshape(L, -1)  # [L, T * 48]
                e = m * eps
                v = (both - own[:, self.at[0]] - own[:, self.at[1]]
                     - 0.5 * (e.sum(1, keepdim=True) - e[:, self.card[:, 0]] - e[:, self.card[:, 1]]))
                out.append(v * pot[:, None])
        return torch.stack(out, 1).to(reach.dtype)
