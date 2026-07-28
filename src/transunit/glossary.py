"""Established terminology, as a source-to-target mapping.

A Term is carrier-agnostic: how an adapter *derives* one differs completely between
sources -- a subtitle adapter might take names from a supplied cast list, a game adapter
from an entity database -- but what a translator needs from it never does.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


class GlossaryError(Exception):
    """A glossary source table could not be read or parsed."""

    def __init__(self, reason: str, *, path: Path | None = None) -> None:
        super().__init__(f"{path}: {reason}" if path else reason)
        self.reason = reason
        self.path = path


@dataclass(frozen=True, slots=True)
class Term:
    """One source-language term and its established target rendering.

    ``category`` is free-form and adapter-defined, but two values are read by the
    translator and worth reserving:

    * ``"character"`` -- a speaker whose name resolves a diarisation/speaker label;
    * ``"name"`` -- a proper noun the translator holds to its source spelling
      mechanically, inflecting only the ending.

    ``entity_id`` is an adapter-side handle back to whatever the term came from; the
    translator never interprets it.
    """

    source: str
    target: str
    category: str = ""
    entity_id: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def write_glossary(terms: list[Term], path: Path) -> int:
    """Write terms as JSON Lines, longest source first. Returns the count written.

    Longest-first is a convenience for a human reading the file; :func:`relevant_terms`
    re-sorts on read, so callers do not depend on the order here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for term in sorted(terms, key=lambda t: (-len(t.source), t.source)):
            handle.write(term.to_json())
            handle.write("\n")
    return len(terms)


def read_glossary(path: Path) -> list[Term]:
    """Load established terminology, or none if the file does not exist.

    A missing glossary means "no established terms", which is a legitimate and common
    state: terminology is something a project accumulates, not something every source
    arrives with. Raising here would force every caller without one to fabricate an
    empty file, which is a workaround standing in for a supported case.

    A glossary that exists and is malformed is still an error: that is a corrupt file,
    not an absent one, and silently treating it as empty would drop terminology the
    caller believes is in force.
    """
    if not path.exists():
        return []
    if not path.is_file():
        raise GlossaryError("glossary path is not a regular file", path=path)
    terms: list[Term] = []
    for line_no, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GlossaryError(f"line {line_no}: invalid JSON: {exc}", path=path) from exc
        if not isinstance(record, dict) or "source" not in record or "target" not in record:
            raise GlossaryError(
                f"line {line_no}: a glossary entry needs at least 'source' and 'target'",
                path=path)
        try:
            terms.append(Term(**record))
        except TypeError as exc:
            raise GlossaryError(f"line {line_no}: {exc}", path=path) from exc
    return terms


def relevant_terms(source_text: str, terms: list[Term], limit: int = 24) -> list[Term]:
    """Glossary entries whose source term literally occurs in ``source_text``.

    Longest-first so that a compound term wins over its constituents.

    Substring matching is a deliberate over-match. In an inflected language a lemma does
    not equal its declined form, so requiring a whole-word hit would miss most real
    occurrences. The asymmetry is what settles it: over-matching shows the model a term
    it did not need, while under-matching silently drops one it did.
    """
    hits = [term for term in terms if term.source and term.source in source_text]
    hits.sort(key=lambda t: -len(t.source))
    return hits[:limit]
