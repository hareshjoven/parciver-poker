"""The turn-start value net of the depth-limited flop search (FLOPSEARCH.md, D2).

Input: both ranges over the 1,326 two-card holdings (fixed deck order, dead
holdings 0, each scaled to mass 1), the board as a 52-card one-hot, and
pot / (pot + stacks) and stack / pot. Output: both players' values per
holding (pots per unit of opponent mass, `turn_values`' units), dead holdings
0. DeepStack's zero-sum layer makes the range-weighted outputs satisfy the
game's identity r0 . v0 + r1 . v1 = 0 exactly.

Data comes in the solver's form, over the 1,128 holdings off a 4-card board
(`turn_values.combos` order: the 1,326 pairs in order, minus those that touch
the board), so `expand` / `contract` are a scatter / gather.
Suit relabelling maps a situation's values to the relabelled situation's
(`SUIT_PERMS`, `permute`); the train/validation split is by suit-canonical
board (`canonical_board`), so no validation board is a relabelling of a
training board.

A canonical net (`ValueNet(..., canonical=True)`, records/leaf-net/PREREG.md)
relabels each input to its board's canonical suit frame (`frame`), runs the
inner network there and relabels the values back: relabelled situations whose
board has no suit symmetry give the inner network the same input."""
from __future__ import annotations

import itertools

import numpy as np

HOLDINGS = 1326
PAIRS = np.array([(a, b) for a in range(52) for b in range(a + 1, 52)], dtype=np.int64)  # [1326, 2]
PAIR_INDEX = np.full((52, 52), -1, dtype=np.int64)
PAIR_INDEX[PAIRS[:, 0], PAIRS[:, 1]] = np.arange(HOLDINGS)
PAIR_INDEX[PAIRS[:, 1], PAIRS[:, 0]] = np.arange(HOLDINGS)

SUIT_PERMS = np.array(list(itertools.permutations(range(4))), dtype=np.int64)  # [24, 4], [0] = identity
CARD_PERM = (np.arange(52)[None, :] & ~3) | SUIT_PERMS[:, np.arange(52) & 3]  # [24, 52]: card c -> CARD_PERM[k, c]
PAIR_PERM = PAIR_INDEX[CARD_PERM[:, PAIRS[:, 0]], CARD_PERM[:, PAIRS[:, 1]]]  # [24, 1326]: holding i -> PAIR_PERM[k, i]
PAIR_PERM_INV = np.argsort(PAIR_PERM, axis=1)
CARD_PERM_INV = np.argsort(CARD_PERM, axis=1)
INPUTS = 2 * HOLDINGS + 52 + 2


def live_mask(board) -> np.ndarray:
    """[..., 1326] bool: the holdings that do not touch the board ([..., 4])."""
    b = np.asarray(board, dtype=np.int64)
    return ~(PAIRS[..., :, :, None] == b[..., None, None, :]).any(-1).any(-1)


def canonical_board(board) -> tuple:
    """The suit-relabelling class of a board: its smallest relabelled sorted form."""
    b = np.asarray(board, dtype=np.int64)
    return min(tuple(sorted(CARD_PERM[k, b].tolist())) for k in range(len(SUIT_PERMS)))


def frame_index(board) -> np.ndarray:
    """[...] the canonical frame of boards ([..., 4] card indices): the
    SUIT_PERMS row whose relabelled sorted board is `canonical_board`'s, the
    lowest such row on a tie (NumPy reference of `ValueNet.frame`)."""
    b = np.asarray(board, dtype=np.int64)
    rel = np.sort(np.moveaxis(CARD_PERM[:, b], 0, -2), -1)  # [..., 24, 4]
    key = ((rel[..., 0] * 52 + rel[..., 1]) * 52 + rel[..., 2]) * 52 + rel[..., 3]
    return (key * len(SUIT_PERMS) + np.arange(len(SUIT_PERMS))).argmin(-1)


def scalars(half, stack) -> np.ndarray:
    """[..., 2]: pot / (pot + both stacks) and stack / pot."""
    half = np.asarray(half, dtype=np.float64)
    stack = np.asarray(stack, dtype=np.float64)
    return np.stack([half / (half + stack), stack / (2 * half)], -1)


LIVE = 1128  # holdings off a 4-card board


def _live_index(live, p: int):
    """[B, p, 1128]: the live holdings' 1,326 indices, in order (a stable
    sort puts them first). Gathers and scatters with it need no device sync,
    unlike boolean indexing."""
    if isinstance(live, np.ndarray):
        idx = np.argsort(~live, axis=1, kind="stable")[:, :LIVE]
        return np.broadcast_to(idx[:, None, :], (len(idx), p, LIVE))
    import torch

    idx = torch.argsort((~live).to(torch.int8), dim=1, stable=True)[:, :LIVE]
    return idx[:, None, :].expand(-1, p, -1)


def expand(x, live):
    """[B, P, 1128] per-board-holding data -> [B, P, 1326] (dead holdings 0).
    `live`: [B, 1326] bool (`live_mask`). NumPy or torch."""
    idx = _live_index(live, x.shape[1])
    if isinstance(x, np.ndarray):
        out = np.zeros(x.shape[:2] + (HOLDINGS,), dtype=x.dtype)
        np.put_along_axis(out, idx, x, axis=2)
        return out
    return x.new_zeros(x.shape[:2] + (HOLDINGS,)).scatter_(2, idx, x)


def contract(y, live):
    """Inverse of `expand`: [B, P, 1326] -> [B, P, 1128]."""
    idx = _live_index(live, y.shape[1])
    if isinstance(y, np.ndarray):
        return np.take_along_axis(y, idx, axis=2)
    return y.gather(2, idx)


def range_mae(pred, true, ranges):
    """Range-weighted mean absolute error per sample ([B]), in the values'
    units: sum_h r_p[h] |pred - true| / sum_h r_p[h], averaged over players.
    NumPy or torch."""
    w = ranges / ranges.sum(-1)[..., None]
    return (w * abs(pred - true)).sum(-1).mean(-1)


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ValueNet(nn.Module):
        """MLP (PReLU) with DeepStack's zero-sum output layer; with
        `canonical`, run in each board's canonical suit frame. `p`: dropout
        (D3-SWEEP), a plain attribute set by training and never saved: it is
        applied by `inner` with `F.dropout` after each hidden PReLU while
        training, so `mlp` holds no dropout module, the state dict's keys are
        a plain net's, and `load` builds a net with p = 0."""

        def __init__(self, width: int, depth: int, canonical: bool = False):
            super().__init__()
            layers, d = [], INPUTS
            for _ in range(depth):
                layers += [nn.Linear(d, width), nn.PReLU()]
                d = width
            layers.append(nn.Linear(d, 2 * HOLDINGS))
            self.mlp = nn.Sequential(*layers)
            self.width, self.depth = width, depth
            self.p = 0.0
            self.register_buffer("scal_mean", torch.zeros(2))
            self.register_buffer("scal_std", torch.ones(2))
            self.canonical = bool(canonical)
            if self.canonical:  # not in the state dict: a canonical net's keys are a plain net's
                for name, t in (("card_perm", CARD_PERM), ("card_perm_inv", CARD_PERM_INV), ("pair_perm", PAIR_PERM),
                                ("pair_perm_inv", PAIR_PERM_INV)):
                    self.register_buffer(name, torch.tensor(t), persistent=False)  # a copy: never a view of the tables
                self.register_buffer("perm_rank", torch.arange(len(SUIT_PERMS)), persistent=False)

        def frame(self, board):
            """[B] the canonical frame (`frame_index`) of a [B, 52] one-hot
            board of 4 cards. Exact index operations only, none of which
            reads the device from the host (CUDA-graph capturable)."""
            cards = board.topk(4, dim=1).indices  # [B, 4]: the 4 ones, any order
            rel = self.card_perm[:, cards].permute(1, 0, 2).sort(-1).values  # [B, 24, 4]
            key = ((rel[..., 0] * 52 + rel[..., 1]) * 52 + rel[..., 2]) * 52 + rel[..., 3]
            return (key * len(SUIT_PERMS) + self.perm_rank).argmin(1)

        def forward(self, ranges, board, scal, live):
            """As `inner`; a canonical net relabels the inputs by `frame`
            (as `permute`), runs `inner` and relabels the values back."""
            if not self.canonical:
                return self.inner(ranges, board, scal, live)
            k = self.frame(board)
            inv = self.pair_perm_inv[k]  # [B, 1326]: frame holding i <- holding inv[i]
            v = self.inner(ranges.gather(2, inv[:, None, :].expand(-1, 2, -1)), board.gather(1, self.card_perm_inv[k]),
                           scal, live.gather(1, inv))
            return v.gather(2, self.pair_perm[k][:, None, :].expand(-1, 2, -1))  # holding i <- frame PAIR_PERM[k, i]

        def inner(self, ranges, board, scal, live):
            """`ranges` [B, 2, 1326], each row of mass 1; `board` [B, 52]
            one-hot; `scal` [B, 2] (`scalars`); `live` [B, 1326] bool.
            Returns values [B, 2, 1326], dead holdings 0, and
            sum_p r_p . v_p = 0 exactly (up to rounding). The output layer
            runs in float32 even under autocast (FLOPSEARCH.md D2 amendment):
            values reach tens of pots, where bfloat16 rounds by up to 1/8 pot."""
            x = torch.cat([ranges.flatten(1), board, (scal - self.scal_mean) / self.scal_std], 1)
            lv = live[:, None, :].to(ranges.dtype)
            if self.p > 0 and self.training:  # D3-SWEEP: after each hidden PReLU, none on the input or the output layer
                h = x
                for lin, act in zip(self.mlp[:-1:2], self.mlp[1:-1:2]):
                    h = F.dropout(act(lin(h)), self.p, True)
            else:
                h = self.mlp[:-1](x)
            out = self.mlp[-1]
            with torch.autocast(h.device.type, enabled=False):
                v = out(h.to(out.weight.dtype))
            v = v.view(-1, 2, HOLDINGS).to(ranges.dtype) * lv
            eps = (ranges * v).sum((1, 2))
            return v - 0.5 * eps[:, None, None] * lv

    return ValueNet


def ValueNet(width: int, depth: int, canonical: bool = False):
    return _torch()(width, depth, canonical)


def load(path, device="cuda"):
    """A net saved by `save`."""
    import torch

    blob = torch.load(path, map_location=device, weights_only=False)
    net = ValueNet(blob["width"], blob["depth"], blob.get("canonical", False)).to(device)
    net.load_state_dict(blob["state"])
    net.eval()
    return net, blob.get("meta", {})


def save(net, path, meta: dict | None = None):
    import torch

    blob = dict(width=net.width, depth=net.depth, state=net.state_dict(), meta=meta or {})
    if net.canonical:  # a plain net's blob stays as before
        blob["canonical"] = True
    torch.save(blob, path)


def predict(net, board, ranges, half, stack):
    """Values for situations in the solver's form: `board` [B, 4] card
    indices, `ranges` [B, 2, 1128] (any positive mass), `half`, `stack` [B].
    Returns [B, 2, 1128] float64 NumPy, pots per unit of opponent mass."""
    import torch

    dev = next(net.parameters()).device
    board = np.asarray(board, dtype=np.int64)
    live = torch.as_tensor(live_mask(board), device=dev)
    r = torch.as_tensor(np.asarray(ranges, dtype=np.float32), device=dev)
    r = expand(r / r.sum(-1, keepdim=True), live)
    onehot = torch.zeros(len(board), 52, device=dev)
    onehot.scatter_(1, torch.as_tensor(board, device=dev), 1.0)
    scal = torch.as_tensor(scalars(half, stack), dtype=torch.float32, device=dev)
    with torch.no_grad():
        v = net(r, onehot, scal, live)
    return contract(v, live).double().cpu().numpy()


def permute(k, ranges, values, board, live):
    """Relabel suits by SUIT_PERMS[k] ([B] indices into the 24) on batched
    1326-form tensors: holding i moves to PAIR_PERM[k, i], card c to
    CARD_PERM[k, c]. `board` is the [B, 52] one-hot. torch only."""
    import torch

    inv = torch.as_tensor(PAIR_PERM_INV, device=ranges.device)[k]  # [B, 1326]: new i <- old inv[i]
    cinv = torch.as_tensor(CARD_PERM_INV, device=ranges.device)[k]
    g = inv[:, None, :].expand(-1, 2, -1)
    return (ranges.gather(2, g), values.gather(2, g), board.gather(1, cinv), live.gather(1, inv))
