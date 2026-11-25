# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""The real-chip rules (nlhe.rs `apply_real` and `payoffs`, play.rs's replay of
a hand) and the showdown evaluator against pokerkit, an independent
implementation of no-limit hold'em: LBR's traced hands replay there with the
same actor, pot, amount owed, legal actions and result, the engine accepts
exactly the actions pokerkit accepts, and it ranks every five-card hand and
decides showdowns as pokerkit does. pokerkit is not a project dependency: run these in a venv that
has it (e.g. `uv venv C:/tmp/pk/.venv; uv pip install pokerkit pytest`, with
the engine on PYTHONPATH)."""
import itertools
import random

import pytest

pk = pytest.importorskip("pokerkit")
pe = pytest.importorskip("parciver_engine")

RANKS, SUITS = "23456789TJQKA", "cdhs"  # the engine's card index is rank * 4 + suit
A = pk.Automation
AUTO = (A.ANTE_POSTING, A.BET_COLLECTION, A.BLIND_OR_STRADDLE_POSTING, A.HOLE_CARDS_SHOWING_OR_MUCKING,
        A.HAND_KILLING, A.CHIPS_PUSHING, A.CHIPS_PULLING)  # cards are burned face down by hand: no deck draws
# Blinds and (equal) stacks: deep, shallow, a stack an odd number of chips, and one
# that only covers a raise all-in.
TABLES = [(50, 100, 10_000), (50, 100, 20_000), (50, 100, 1_000), (25, 50, 333), (1, 2, 37), (50, 100, 150)]


def cards(ix) -> str:
    return "".join(RANKS[c // 4] + SUITS[c % 4] for c in ix)


def player(seat: int) -> int:
    """pokerkit's player for an engine seat: heads-up its player 1 is the button
    (small blind), the engine's seat 0."""
    return 1 - seat


def table(sb: int, bb: int, stack: int, holes) -> "pk.State":
    st = pk.NoLimitTexasHoldem.create_state(AUTO, True, 0, (sb, bb), bb, (stack, stack), 2)
    for p in range(2):  # pokerkit deals player 0 (the engine's seat 1) first
        st.deal_hole(cards(holes[1 - p]))
    return st


def deal(st, board, dealt: int) -> int:
    """Deal the board cards the hand needs next; the count dealt so far."""
    while True:
        if st.can_burn_card("??"):
            st.burn_card("??")
        if not st.can_deal_board():
            return dealt
        n = st.board_dealing_count
        st.deal_board(cards(board[dealt:dealt + n]))
        dealt += n


def owed(st) -> int:
    return max(st.bets) - st.bets[st.actor_index]


def legal(st, kind: str, add: int) -> bool:
    """Whether pokerkit allows the engine's action (`add`: the chips a bet adds;
    adding no more than the call is a call, legal if it is the call or all-in)."""
    left = st.stacks[st.actor_index]
    if kind == "fold":
        return st.can_fold()
    if kind == "check":
        return owed(st) == 0
    if kind == "call":
        return owed(st) > 0
    if add > left or add < 1:
        return False
    if add <= owed(st):
        return add == owed(st) or add == left
    return st.can_complete_bet_or_raise_to(st.bets[st.actor_index] + add)


def act(st, kind: str, add: int) -> None:
    assert legal(st, kind, add), (kind, add)
    if kind == "fold":
        st.fold()
    elif kind in ("check", "call") or add <= owed(st):
        st.check_or_call()
    else:
        st.complete_bet_or_raise_to(st.bets[st.actor_index] + add)


@pytest.mark.parametrize("sb,bb,stack", TABLES)
def test_lbr_hands_replay_in_pokerkit_with_the_same_pot_owed_and_result(sb, bb, stack):
    hands = pe.PlayAgent.uniform().lbr_trace(list(range(40)), sb, bb, stack, seed=7)
    showdowns = 0
    for h in hands:
        c = h["cards"]
        st, dealt = table(sb, bb, stack, [c[0:2], c[2:4]]), 0
        for s in h["steps"]:
            dealt = deal(st, c[4:], dealt)
            seat = h["lbr_seat"] if s["actor"] == "lbr" else 1 - h["lbr_seat"]
            where = f"deal {h['deal']} LBR seat {h['lbr_seat']} {s}"
            assert (dealt, st.actor_index, st.total_pot_amount, owed(st)) == (
                s["street"], player(seat), s["pot"], s["owed"]), where
            for kind, add, _ in s.get("options", []):  # the agent's whole menu is legal
                assert legal(st, kind, add), (where, kind, add)
            act(st, s["action"], s["add"])
        dealt = deal(st, c[4:], dealt)
        assert not st.status, f"deal {h['deal']}: the engine ended a hand pokerkit has not"
        showdowns += dealt == 5 and not any(s["action"] == "fold" for s in h["steps"])
        assert abs(st.payoffs[player(h["lbr_seat"])] - h["net_bb"] * bb) < 1e-6, h
    assert len(hands) == 80 and showdowns > 0


def test_the_engine_accepts_exactly_the_actions_pokerkit_accepts():
    """Random hands (some actions the agent's own, the rest random legal ones);
    at every turn the engine finds the same player to act, the agent's choice
    and menu are legal, and a fold, check, call or random bet is refused by the
    engine exactly where pokerkit refuses it."""
    agent, rng = pe.PlayAgent.uniform(), random.Random(5)
    probes, edges = {True: 0, False: 0}, {False: 0, True: 0}  # edges: preflop, postflop
    for hand in range(250):
        sb, bb, stack = rng.choice(TABLES)
        dk = rng.sample(range(52), 9)
        holes, board = [dk[0:2], dk[2:4]], dk[4:]
        st, dealt, hist = table(sb, bb, stack, holes), 0, []

        def engine_legal(kind, add):
            try:
                agent.decide(1 - seat, tuple(holes[1 - seat]), board, hist + [(kind, add)], sb, bb, (stack, stack))
            except ValueError as e:  # refused when replayed, not a later error of the probed line
                return f"illegal action {len(hist)} (" not in str(e)
            return True

        while True:
            dealt = deal(st, board, dealt)
            if not st.status:
                break
            seat = 1 - st.actor_index
            left, o = st.stacks[st.actor_index], owed(st)
            d = agent.decide(seat, tuple(holes[seat]), board[:dealt], hist, sb, bb, (stack, stack), seed=hand)
            assert legal(st, d["action"], d["add"]), (hist, d)
            # The chips it puts in: pokerkit's amount for a call, all-in exactly when they are its stack.
            add = {"fold": 0, "check": 0, "call": st.checking_or_calling_amount}.get(d["action"], d["add"])
            assert (d["add"], d["all_in"]) == (add, add == left), (hist, d)
            for kind, add, _ in agent.options(seat, board[:dealt], hist, sb, bb, (stack, stack), [tuple(holes[seat])])[0]:
                assert legal(st, kind, add), (hist, kind, add)
            floor = (st.min_completion_betting_or_raising_to_amount or 0) - st.bets[st.actor_index]
            adds = {1, o, o + 1, floor - 1, floor, floor + 1, left - 1, left, left + 1, rng.randint(1, left)}
            for kind, add in [("f", 0), ("x", 0), ("c", 0)] + [("b", a) for a in sorted(adds) if a > 0]:
                full = {"f": "fold", "x": "check", "c": "call", "b": "bet"}[kind]
                want = legal(st, full, add)
                assert engine_legal(kind, add) == want, (hist, kind, add, sb, bb, stack)
                probes[want] += 1
                # The min-raise edge facing a bet or raise, below the stack (a short raise, the least full one).
                edges[dealt > 0] += kind == "b" and 0 < o < floor - 1 and floor < left and add in (floor - 1, floor)
            # Play on: the agent's choice, or a random legal action (all-ins as "a" at times).
            if rng.random() < 0.5:
                kind, add = d["action"], d["add"]
            else:
                choices = [(k, a) for k, a in [("fold", 0), ("check", 0), ("call", 0)] + [("bet", a) for a in adds]
                           if a <= left and legal(st, k, a)]
                kind, add = rng.choice(choices)
            if kind == "bet" and add == left and rng.random() < 0.5:
                hist.append(("a", 0))
            else:
                hist.append(({"fold": "f", "check": "x", "call": "c", "bet": "b"}[kind], add))
            act(st, kind, add)
    assert probes[True] > 1000 and probes[False] > 1000 and edges[False] > 300 and edges[True] > 30, (probes, edges)


def test_hands_rank_and_showdowns_go_as_in_pokerkit():
    """Showdowns: over all C(52,5) hands, eval5's order, ties included, is
    pokerkit's (a strictly increasing one-to-one map between their 7,462
    values; the engine's own gates count categories, not the order inside one).
    eval7 is the best of its 21 fives by construction, and eval7_fast, which
    every showdown and equity uses, equals it on all seven-card hands (the
    exhaustive ignored tests in hand_eval.rs); here eval7_fast decides river
    showdowns (`equity_vs_hand` with five board cards) as pokerkit's best fives
    do, on random deals and on deals from small decks rich in the rare classes
    (straight flushes inside six- and seven-card flushes, wheels, two trips,
    three pairs)."""
    deck = list(pk.Card.parse(cards(range(52))))
    lookup, pairs = pk.StandardLookup(), set()
    for h in itertools.combinations(range(52), 5):
        pairs.add((pe.eval5(list(h)), lookup.get_entry(tuple(deck[i] for i in h)).index))
    assert len(pairs) == len({v for v, _ in pairs}) == 7462
    assert [ix for _, ix in sorted(pairs)] == list(range(7462))

    def best(seven):
        return max((lookup.get_entry(tuple(deck[i] for i in five)) for five in itertools.combinations(seven, 5)),
                   key=lambda e: e.index)

    rng = random.Random(9)

    def deals():
        for case in ("5sKdKsQsAs2s3s4s9s",  # a wheel straight flush against the six-card ace-high flush
                     "AdKcAs2dAhAcKhKs2c",  # two trips against trips and two pairs: aces full of kings, a tie
                     "QdQc7c2hAhAcKhKc7d"):  # three pairs: the best two and the best kicker
            yield [RANKS.index(r) * 4 + SUITS.index(s) for r, s in zip(case[::2], case[1::2])]
        for i in range(20_000):
            if i % 2 == 0:
                small = range(52)
            elif i % 6 == 1:  # four ranks: trips, quads, two trips, full houses
                small = [r * 4 + s for r in rng.sample(range(13), 4) for s in range(4)]
            elif i % 6 == 3:  # a suit and four other cards: long flushes, straight flushes
                s = rng.randrange(4)
                small = [r * 4 + s for r in range(13)] + rng.sample([c for c in range(52) if c % 4 != s], 4)
            else:  # seven ranks in a row (A-7 too): straights, wheels, three pairs
                lo = rng.randrange(-1, 7)
                small = [r % 13 * 4 + s for r in range(lo, lo + 7) for s in range(4)]
            yield rng.sample(list(small), 9)

    labels, ties = {}, 0
    for h in deals():
        a, b = best(h[:2] + h[4:]), best(h[2:4] + h[4:])
        want = 1.0 if a.index > b.index else 0.0 if a.index < b.index else 0.5
        assert pe.equity_vs_hand(h[:2], h[2:4], h[4:]) == want, cards(h)
        labels[a.label] = labels.get(a.label, 0) + 1
        ties += want == 0.5
    assert len(labels) == 9 and min(labels.values()) > 100 and ties > 500, (labels, ties)
