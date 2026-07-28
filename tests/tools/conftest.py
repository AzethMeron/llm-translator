"""Shared wiring for the ``tools/lib`` evaluation-harness tests.

The harnesses are scripts, not an installed package: they live in ``tools/lib`` and are run by
path. Putting that directory on ``sys.path`` here is what makes them importable by name, the way
``tools/*.sh`` makes them runnable. Nothing else is global.

These harnesses decide what gets published about the engine's quality, so a silent defect in one
of them corrupts a finding rather than crashing a run -- which is why they are tested at all. Like
the rest of the suite, nothing here needs a server, a GPU, or a network: HTTP goes through
``httpx.MockTransport`` and every file is under ``tmp_path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parents[2] / "tools" / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
