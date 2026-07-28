"""Built-in per-model backend profiles.

Each module here registers one backend on import. Importing this package therefore
registers all of them, which is what :func:`translator.backend.core._ensure_profiles_loaded`
does on first use of the registry.

Adding support for a new model is adding a module here that calls ``register_backend`` --
a few lines over :class:`~translator.backend.core.SchemaBackend` or
:class:`~translator.backend.core.JsonObjectBackend` -- and nothing else in the codebase
changes. See ``docs/writing-a-backend.md``.
"""
from __future__ import annotations

from . import bielik, eurollm, generic, qwen

__all__ = ["generic", "qwen", "bielik", "eurollm"]
