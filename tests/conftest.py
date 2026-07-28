"""Shared test wiring.

Puts ``tests/fixtures`` on ``sys.path`` so the fake carrier adapter is importable by name,
the way a real adapter package would be on a user's path. Nothing else here is global; per-
area fixtures live with their tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
if str(_FIXTURES) not in sys.path:
    sys.path.insert(0, str(_FIXTURES))
