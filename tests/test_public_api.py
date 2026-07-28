"""The public package surface: both packages expose a version marker, and it is part of the
exported contract (``__all__``) so it is discoverable, not just present."""
from __future__ import annotations

import re

import transunit
import translator


def _is_semver(value: str) -> bool:
    return bool(re.fullmatch(r"\d+\.\d+\.\d+", value))


def test_both_packages_expose_a_semver_version() -> None:
    assert isinstance(translator.__version__, str) and _is_semver(translator.__version__)
    assert isinstance(transunit.__version__, str) and _is_semver(transunit.__version__)


def test_the_two_versions_are_kept_in_step() -> None:
    assert translator.__version__ == transunit.__version__


def test_version_is_part_of_the_exported_contract() -> None:
    assert "__version__" in translator.__all__
    assert "__version__" in transunit.__all__
