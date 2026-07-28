"""Hypothesis configuration for the property suite.

Derandomised on purpose. The rest of the suite is deterministic, and a property test that
explores a different input set on every run turns a red build into a coin flip -- a failure you
cannot reproduce is a failure you cannot fix. With ``derandomize=True`` the same examples are
generated every run, so a counterexample found in CI reproduces exactly on a laptop.
"""
from __future__ import annotations

from hypothesis import HealthCheck, settings

settings.register_profile(
    "translator",
    derandomize=True,
    max_examples=300,
    deadline=None,  # coverage instrumentation makes per-example timing meaningless
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("translator")
