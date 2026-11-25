# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Card abstraction — the Python-ergonomic face of the Rust engine.

The heavy lifting (suit isomorphism, exact/Monte-Carlo equity) lives in
``parciver_engine`` where it is fast and exhaustively gated; this module adds the
niceties: card strings *or* 0..52 indices as input, a street-aware ``equity``
dispatcher (exact postflop, sampled preflop), and the canonical 169-entry
preflop table. Phase 3b (equity-distribution histograms + GPU/EMD clustering)
builds on this surface.

Cards are 0..52 indices (``rank * 4 + suit``; ranks 2..A = 0..12, suits c,d,h,s
= 0..3) or two-char strings like ``"As"``. Use :func:`card` / :func:`cards` to
convert.
"""
from __future__ import annotations

from typing import Iterable, Sequence

try:
    import parciver_engine as _engine
except ImportError as exc:  # pragma: no cover - exercised only without the build
    raise ImportError(
        "parciver_engine is not built for this interpreter. Build it with:\n"
        "  .venv/Scripts/python.exe -m maturin develop --release -m engine/Cargo.toml"
    ) from exc

__all__ = [
    "card",
    "cards",
    "card_name",
    "preflop_class",
    "preflop_class_name",
    "PREFLOP_CLASS_NAMES",
    "canonical",
    "equity",
    "equity_vs_hand",
]

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"

CardLike = int | str


def card(c: CardLike) -> int:
    """Coerce a card string (``"As"``) or an already-valid 0..52 index to index."""
    if isinstance(c, str):
        return _engine.parse_card(c)
    if isinstance(c, int):
        if not 0 <= c < 52:
            raise ValueError(f"card index {c} out of 0..52")
        return c
    raise TypeError(f"card must be str or int, got {type(c).__name__}")


def cards(cs: Iterable[CardLike]) -> list[int]:
    """Coerce an iterable of card strings/indices to a list of 0..52 indices."""
    return [card(c) for c in cs]


def card_name(c: CardLike) -> str:
    """Two-char name of a card, e.g. ``"As"``."""
    i = card(c)
    return f"{_RANKS[i >> 2]}{_SUITS[i & 3]}"


# --- preflop 169 classes ---------------------------------------------------

#: Names of the 169 suit-isomorphic preflop classes, indexed by class id.
PREFLOP_CLASS_NAMES: list[str] = [_engine.preflop_name(i) for i in range(169)]


def preflop_class(a: CardLike, b: CardLike) -> int:
    """Suit-isomorphic preflop class (0..169) of a two-card holding."""
    return _engine.preflop_index(card(a), card(b))


def preflop_class_name(a: CardLike, b: CardLike) -> str:
    """Name of the preflop class of a two-card holding, e.g. ``"AKs"``."""
    return _engine.preflop_name(preflop_class(a, b))


# --- suit isomorphism ------------------------------------------------------

def canonical(cs: Iterable[CardLike]) -> tuple[int, ...]:
    """Canonical suit-isomorphic representative of a card set (sorted indices).

    Suit-isomorphic sets share a canonical form, so this deduplicates boards.
    """
    return tuple(_engine.canonical_cards(cards(cs)))


# --- equity ----------------------------------------------------------------

def equity(
    hole: Sequence[CardLike],
    board: Sequence[CardLike] = (),
    *,
    exact: bool | None = None,
    samples: int = 200_000,
    seed: int = 0,
) -> float:
    """All-in equity of ``hole`` vs a uniform-random opponent on ``board``.

    Ties count as half. By default this is exact postflop (3/4/5 board cards,
    where enumeration is cheap) and Monte-Carlo preflop (empty board, where the
    exact count is ~2.1e9 per hand). Force one path with ``exact=True/False``;
    ``samples``/``seed`` control the sampler.
    """
    h = cards(hole)
    b = cards(board)
    if exact is None:
        exact = len(b) >= 3
    if exact:
        return _engine.equity_vs_random(h, b)
    return _engine.equity_vs_random_mc(h, b, samples, seed)


def equity_vs_hand(
    hole: Sequence[CardLike],
    villain: Sequence[CardLike],
    board: Sequence[CardLike] = (),
) -> float:
    """Exact all-in equity of ``hole`` vs one specific ``villain`` on ``board``."""
    return _engine.equity_vs_hand(cards(hole), cards(villain), cards(board))
