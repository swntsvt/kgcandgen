"""Aggregate unittest discovery entry point."""

from __future__ import annotations

import unittest


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    """Load all tests from the tests/ directory for default unittest runs."""
    return loader.discover(
        start_dir="tests",
        pattern="test_*.py",
        top_level_dir="tests",
    )
