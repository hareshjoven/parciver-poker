#!/usr/bin/env python
# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Size of a full flop subgame (flop round, 49 turn cards, turn round, 48
river cards, river round) for several bet menus, from the same betting rules
as the search's trees (gpu_search.Tree / search.rs Tree::build), without
building it: nodes, decisions, action slots, and the memory of float32 /
16-bit regrets + strategy sums over 1176 holdings. The last row checks the
counter against the turn tree the search uses today (7305 nodes).

    .venv/Scripts/python.exe scripts/flop_tree_size.py
"""
import functools
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parciver.gpu_search import Tree, Menu, _round

def legal(menu, b, min_bet):
    p = b["to_act"]; q = 1 - p
    c = b["stake"][q] - b["stake"][p]; stack = b["stacks"][p]
    acts = []
    if c > 0: acts.append(("f", 0))
    acts.append(("c", 0))
    if stack > c and b["stacks"][q] > 0:
        pot = float(sum(b["stake"])); floor = max(b["last_raise"], min_bet); adds = []
        if b["raises"] < menu.max_raises:
            for f in (menu.bet_fracs if c == 0 else menu.raise_fracs):
                add = max(_round(f * pot), floor) if c == 0 else c + max(_round(f * (pot + c)), floor)
                if add < stack: adds.append(add)
        adds.append(stack)
        acts.extend(("b", a) for a in sorted(set(adds)))
    return acts

def key(b):
    return (tuple(b["stake"]), tuple(b["stacks"]), b["to_act"], b["last_raise"], b["raises"], tuple(b["acted"]))

def count(menus, street, b, min_bet, cards):
    """(nodes, decisions, action slots) of a round from b, then later streets."""
    @functools.lru_cache(None)
    def rnd(street, k):
        b = dict(stake=list(k[0]), stacks=list(k[1]), to_act=k[2], last_raise=k[3], raises=k[4], acted=list(k[5]))
        acts = legal(menus[street], b, min_bet)
        N, D, A = 1, 1, len(acts)
        for a in acts:
            nb, end = Tree._apply(b, a)
            if end is not None and end[0] == "fold":
                N += 1
            elif end is not None:
                x = nxt(street, tuple(nb["stake"]), tuple(nb["stacks"]))
                N += x[0]; D += x[1]; A += x[2]
            else:
                x = rnd(street, key(nb)); N += x[0]; D += x[1]; A += x[2]
        return N, D, A
    @functools.lru_cache(None)
    def nxt(street, stake, stacks):
        if street == len(menus) - 1:
            return 1, 0, 0  # showdown
        allin = min(stacks) == 0
        kids = cards[street]  # cards dealt next
        if allin:
            # remaining chance levels straight to showdown
            m = 1
            for s in range(street, len(menus) - 1): m *= cards[s]
            return 1 + m, 0, 0
        start = dict(stake=list(stake), stacks=list(stacks), to_act=1, last_raise=0, raises=0, acted=[False, False])
        x = rnd(street + 1, key(start))
        return 1 + kids * x[0], kids * x[1], kids * x[2]
    return rnd(street, key(b))

M = Menu
configs = {
    "tiny   (flop .5 | turn .75 | river .75, no raises)": [M([0.5], [], 1), M([0.75], [], 1), M([0.75], [], 1)],
    "small  (flop .33/.75 r1 x2 | turn .75 | river .75)": [M([0.33, 0.75], [1.0], 2), M([0.75], [], 1), M([0.75], [], 1)],
    "medium (flop .33/.75 r1 x2 | turn .5/1 r1 x2 | river .75)": [M([0.33, 0.75], [1.0], 2), M([0.5, 1.0], [1.0], 2), M([0.75], [], 1)],
    "rich   (flop .33/.75/1.5 r1 x3 | turn .5/1 r1 x2 | river .33/.75 r1 x2)": [M([0.33, 0.75, 1.5], [1.0], 3), M([0.5, 1.0], [1.0], 2), M([0.33, 0.75], [1.0], 2)],
}
pots = {"SRP 100bb (pot 5bb)": ((250, 250), (9750, 9750)), "3bet 100bb (pot 20bb)": ((1000, 1000), (9000, 9000))}
n = 1176
print(f"{'config':66s} {'pot':22s} {'nodes':>11s} {'decisions':>10s} {'act slots':>10s} | regret+sum f32 / 16-bit | rq+val f32")
for pname, (stake, stacks) in pots.items():
    for cname, menus in configs.items():
        b = dict(stake=list(stake), stacks=list(stacks), to_act=1, last_raise=0, raises=0, acted=[False, False])
        N, D, A = count(menus, 0, b, 100, (49, 48))
        rs = 2 * A * n
        print(f"{cname:66s} {pname:22s} {N:11,d} {D:10,d} {A:10,d} | {rs*4/2**30:6.1f} / {rs*2/2**30:5.1f} GB | {2*N*n*4/2**30:6.1f} GB")
# turn trees for reference (our current turn search)
for cname, menus in {"turn agent (turn .5/1 r1 x2 | river .75)": [M([0.5, 1.0], [1.0], 2), M([0.75], [], 1)]}.items():
    b = dict(stake=[1000, 1000], stacks=[9000, 9000], to_act=1, last_raise=0, raises=0, acted=[False, False])
    N, D, A = count(menus, 0, b, 100, (48,))
    print(f"{cname:66s} {'(check: 7305 nodes)':22s} {N:11,d} {D:10,d} {A:10,d}")
