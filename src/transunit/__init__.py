"""The contract between a carrier adapter and a translator.

Everything here is carrier-agnostic and model-agnostic: the translation unit and its
catalogue, the placeholder convention, terminology, display width, and language
detection.

Two independent sides depend on this package and neither depends on the other:

* a **carrier adapter** (a subtitle track, a game's files, a document) turns a source
  into units and writes translated units back;
* a **translator** fills those units in.

That makes both swappable. A new carrier needs a new adapter and nothing else; a
different translator needs to read and write this format and nothing else.

The names of the two sides are chosen to stay unambiguous: the downstream side is a
carrier **adapter** (:mod:`transunit.adapter`), and the model-facing side, upstream in
the translator, is a **backend** (:mod:`translator.backend`) -- never an "adapter".
"""
from __future__ import annotations

from .adapter import Adapter, AdapterError, load_adapter, passthrough
from .glossary import (
    GlossaryError,
    Term,
    read_glossary,
    relevant_terms,
    write_glossary,
)
from .language import (
    Detection,
    LanguageError,
    LanguageProfile,
    ScriptDetector,
    UntranslatedDetector,
    available_scripts,
    detect,
    find_profile,
    is_untranslated,
    japanese_detector,
    lexical_detector,
    load_profiles,
    script_detector,
    strip_accents,
    tokenize,
)
from .placeholders import PLACEHOLDER, placeholder_indices
from .reference import (
    GrowableLexicalRetriever,
    LexicalRetriever,
    ReferenceEntry,
    ReferenceError,
    Retriever,
    Retrieved,
    WritableRetriever,
    read_reference,
    reference_from_units,
)
from .units import (
    CatalogError,
    Status,
    Unit,
    make_unit_id,
    merge_journal,
    read_catalog,
    read_journal,
    write_catalog,
)
from .width import display_columns

__version__ = "0.1.0"
"""Package version. The names in ``__all__`` are the stable contract; everything else is an
implementation detail. Kept in step with :data:`translator.__version__`."""

__all__ = [
    "__version__",
    # units
    "Status", "Unit", "CatalogError", "make_unit_id",
    "read_catalog", "write_catalog", "read_journal", "merge_journal",
    # glossary
    "Term", "GlossaryError", "read_glossary", "write_glossary", "relevant_terms",
    # placeholders / width
    "PLACEHOLDER", "placeholder_indices", "display_columns",
    # reference translations / retrieval
    "ReferenceEntry", "ReferenceError", "Retrieved", "Retriever", "WritableRetriever",
    "LexicalRetriever", "GrowableLexicalRetriever", "read_reference", "reference_from_units",
    # language detection
    "LanguageProfile", "LanguageError", "Detection", "UntranslatedDetector",
    "load_profiles", "find_profile", "detect", "is_untranslated", "lexical_detector",
    "tokenize", "strip_accents", "ScriptDetector", "japanese_detector",
    "script_detector", "available_scripts",
    # carrier adapter contract
    "Adapter", "AdapterError", "load_adapter", "passthrough",
]
