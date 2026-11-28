# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""The ``parciver`` command line: play the solved bots.

    parciver play kuhn              # nothing to download, instant
    parciver play leduc --peek      # see the opponent's mixed strategy after each hand
    parciver fetch-blueprint        # once: ~0.7 GB of tables for the No-Limit game
    parciver play nlhe              # heads-up No-Limit hold'em vs the blueprint (CPU)
    parciver fetch-nets             # optional: the flop value nets of the full GPU configuration (~0.2 GB)
    parciver doctor                 # what works on this machine, what is missing, and the fix for each
"""
from __future__ import annotations

import argparse
import sys
from importlib import metadata

from .play_small import GAMES, run as run_small


def _version() -> str:
    try:
        return metadata.version("parciver")
    except metadata.PackageNotFoundError:
        return "dev"


START_HERE = """\
parciver: play poker against a bot solved with Counterfactual Regret Minimization (CFR).

Start here:
  parciver play kuhn       the smallest poker game: 3 cards, instant, nothing to install
  parciver play leduc      a slightly bigger one; add --peek to see the bot's exact mixed strategy
  parciver doctor          check what works on this machine and what to run next
  parciver play nlhe       real heads-up no-limit hold'em (needs the Rust engine and a one-time download;
                           `parciver doctor` says what is missing)

Type ? at any No-Limit prompt for the commands and hand rankings.   More: parciver --help, docs/BEGINNERS.md
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="parciver", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"parciver {_version()}")
    sub = ap.add_subparsers(dest="command")

    p = sub.add_parser("play", help="play a solved bot in the terminal")
    p.add_argument("game", choices=sorted(GAMES) + ["nlhe"], help="which game")
    p.add_argument("--hands", type=int, default=None, help="stop after this many hands (default: until you quit)")
    p.add_argument("--iterations", type=int, default=None, help="kuhn/leduc: CFR iterations used to solve the opponent")
    p.add_argument("--seed", type=int, default=None, help="seed the deals for a reproducible session")
    p.add_argument("--peek", action="store_true", help="after each hand, show the opponent's strategy at its decisions")
    p.add_argument("--search", action="store_true", help="nlhe: re-solve the turn and river on the CPU (slower)")
    p.add_argument("--blueprint", help="nlhe: path to blueprint.bin (default: where fetch-blueprint put it)")
    p.add_argument("--abstraction", help="nlhe: directory with {flop,turn,river}_buckets.npy")

    f = sub.add_parser("fetch-blueprint", help="download the blueprint tables the nlhe game needs")
    f.add_argument("--url", help="directory URL holding the release's .gz files (default: the project's release)")

    n = sub.add_parser("fetch-nets", help="download the flop value nets (N0, N1) of the full GPU configuration")
    n.add_argument("--url", help="directory URL holding the release's net files (default: the project's release)")

    d = sub.add_parser("doctor", help="check what works on this machine and what to run next")
    d.add_argument("--verify", action="store_true", help="also hash the installed tables (about a minute)")

    a = ap.parse_args(argv)
    try:
        if a.command is None:
            print(START_HERE)
            return 0
        if a.command == "doctor":
            from .doctor import run as run_doctor
            return run_doctor(verify=a.verify)
        if a.command == "fetch-blueprint":
            from .assets import fetch
            return fetch(a.url)
        if a.command == "fetch-nets":
            from .assets import fetch
            return fetch(a.url, group="nets")
        if a.game == "nlhe":
            from .play_nlhe import run as run_nlhe
            return run_nlhe(hands=a.hands, seed=a.seed, peek=a.peek, search=a.search,
                            blueprint=a.blueprint, abstraction=a.abstraction)
        return run_small(a.game, iterations=a.iterations, hands=a.hands, seed=a.seed, peek=a.peek)
    except (KeyboardInterrupt, EOFError):
        print("\nbye")
        return 0


if __name__ == "__main__":
    sys.exit(main())
