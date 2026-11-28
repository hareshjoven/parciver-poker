# Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
"""Test-suite configuration: the fast tier.

`pytest -m "not slow"` skips the modules below, which together take most of the full suite's ~22 minutes (CFR
convergence runs, exact best-response checks, kernel and search comparisons). CI and the release gates still run
everything with a plain `pytest`. The list comes from `pytest --durations` on a clean run; add a module here when it
becomes slow rather than marking tests one by one.
"""
import pytest

SLOW_MODULES = {
    "test_rmapss_step2b", "test_rmapss_step2a", "test_rmapss_step2", "test_rmapss_step1", "test_rmapss_stepQ",
    "test_arena", "test_lbr", "test_kernel_designs", "test_flop_sym", "test_scorecard", "test_flop_m1",
    "test_solver_watch", "test_leduc_multi", "test_solver", "test_search_ab", "test_nlhe", "test_gpu_search",
    "test_d2b", "test_mccfr",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.module.__name__.rsplit(".", 1)[-1] in SLOW_MODULES:
            item.add_marker(pytest.mark.slow)
