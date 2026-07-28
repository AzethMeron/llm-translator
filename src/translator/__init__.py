"""A local-model translation engine: an agentic translate/review harness over the
:mod:`transunit` intermediate format.

The engine knows nothing about any particular carrier -- it reads units and writes them
back -- and nothing about any particular model, which it reaches through a
:mod:`translator.backend`. Everything a caller needs to wire it into a project is exported
here.

Typical use::

    from translator import (
        TranslationAgents, RuleSet, AgentSet, TranslationMemory,
        LlmClient, ServerConfig, resolve_backend, run_batch, pending_units,
    )
    from transunit import read_glossary, lexical_detector, load_profiles

    profiles = load_profiles(Path("config/languages.toml"))
    client = LlmClient(ServerConfig(model="qwen3-14b"),
                       backend=resolve_backend("auto", "qwen3-14b"))
    agents = TranslationAgents(
        client, RuleSet.load(rules_path), read_glossary(glossary_path),
        agent_set=AgentSet.load(agents_path, {"source_language": "English",
                                              "target_language": "Polish"}),
        is_untranslated=lexical_detector(profiles, "en"),
        source_label="EN", target_label="PL")
    run_batch(agents, pending_units(catalog, journal), journal)

The two boundaries are named to stay unambiguous: the model side is a **backend**
(:mod:`translator.backend`); the carrier side is an **adapter** (:mod:`transunit.adapter`).
"""
from __future__ import annotations

from .agents import (
    Attempt,
    Outcome,
    Review,
    TranslationAgents,
)
from .backend import (
    Backend,
    BackendError,
    LlmClient,
    LlmContentError,
    LlmError,
    LlmRefusalError,
    LlmTruncationError,
    Message,
    ServerConfig,
    UsageStats,
    available_backends,
    get_backend,
    register_backend,
    resolve_backend,
)
from .memory import TranslationMemory
from .roles import (
    AgentConfigError,
    AgentSet,
    Leniency,
    Limits,
    Reviewer,
)
from .rules import (
    RuleConfigError,
    RuleSet,
    Severity,
    Violation,
    check_mechanical,
    check_names,
)
from .runner import (
    Progress,
    RunnerError,
    catalog_order,
    completed_ids,
    group_duplicates,
    pending_units,
    run_batch,
)

__version__ = "0.1.0"
"""Package version. The public API listed in ``__all__`` is the stable surface; anything not
exported here is an implementation detail and may change without a version bump."""

__all__ = [
    "__version__",
    # harness
    "TranslationAgents", "Outcome", "Review", "Attempt",
    # policy
    "RuleSet", "RuleConfigError", "Severity", "Violation", "check_mechanical", "check_names",
    # roles
    "AgentSet", "Reviewer", "Limits", "Leniency", "AgentConfigError",
    # memory
    "TranslationMemory",
    # batch execution
    "run_batch", "Progress", "RunnerError", "pending_units", "completed_ids",
    "group_duplicates", "catalog_order",
    # model backend
    "LlmClient", "ServerConfig", "UsageStats", "Message", "Backend", "BackendError",
    "LlmError", "LlmContentError", "LlmRefusalError", "LlmTruncationError",
    "resolve_backend", "get_backend", "available_backends", "register_backend",
]
