"""The intermediate translation format: the translation unit (TU) and its catalogue.

A TU is the pipeline's single source of truth for one translatable payload. It is
deliberately *provenance-carrying*: it stores the exact span it came from, so the
translation can be put back where it belongs rather than re-derived.

What a span measures is the adapter's business, not this module's. For a subtitle track
it is a half-open interval in milliseconds along the timeline; for a text file it would
be a byte offset. Only the ordering matters here, which is why the invariant is
``span_end >= span_start`` and nothing more.

Serialised as JSON Lines: streamable, diffable, appendable, and resumable after an
interrupted run without holding the whole catalogue in memory.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path


class Status(str, Enum):
    """Lifecycle of a translation unit."""

    PENDING = "pending"

    TRANSLATED = "translated"
    """Mechanically sound, but the reviewers still objected when the budget ran out.

    Rendered like a verified line, and kept under this status so the units a human might
    want to revisit stay findable in the catalogue.
    """

    VERIFIED = "verified"
    """Passed the mechanical checks and every reviewer."""

    REJECTED = "rejected"
    """Failed the mechanical checks, or the model produced nothing usable.

    Never rendered: the target may be empty or still in the source language, so showing
    it would put a defect in front of the reader.
    """

    SKIPPED = "skipped"
    """Payload needs no translation (already in the target language, or pure
    punctuation)."""

    @property
    def is_injectable(self) -> bool:
        """Whether a unit in this state may have its target rendered back into a carrier."""
        return self in (Status.VERIFIED, Status.TRANSLATED)


class CatalogError(Exception):
    """A translation catalogue is malformed or internally inconsistent."""

    def __init__(self, reason: str, *, path: Path | None = None,
                 line_no: int | None = None) -> None:
        location = f"{path}:{line_no}" if path and line_no else str(path or "<memory>")
        super().__init__(f"{location}: {reason}")
        self.reason = reason
        self.path = path
        self.line_no = line_no


@dataclass(frozen=True, slots=True)
class Unit:
    """One translatable payload, with everything needed to translate and reinject it."""

    unit_id: str
    rel_path: str
    line_no: int
    span_start: int
    """Where the payload begins in its source, in whatever unit the adapter uses --
    milliseconds along a timeline for a subtitle track, a byte offset for a text file."""
    span_end: int
    """Where it ends. Half-open, so a payload's end is its successor's start."""
    command: str
    kind: str
    source: str
    """The text to translate. A whole sentence where the source allows it, since a
    fragment cannot be translated well in isolation."""
    placeholders: tuple[str, ...] = ()
    speaker: str | None = None
    """Who is speaking, when the source identifies them -- a diarisation label for a
    subtitle track, an entity id for a game."""
    context_before: tuple[str, ...] = ()
    context_after: tuple[str, ...] = ()
    max_columns: int | None = None
    """Hard character budget for this payload.

    Set by adapters whose carrier has a fixed amount of room. For a subtitle that is the
    reading-speed limit; for a fixed message box it is the box width. The translator is
    told the budget up front so it can aim for a shorter rendering rather than
    discovering by rejection that it wrote too much.

    Distinct from the rule set's ``max_line_columns``, which is a project-wide guideline
    and only warns. A limit here comes from the carrier itself, so exceeding it is a
    blocking error.

    ``None`` means the carrier imposes nothing.
    """
    status: Status = Status.PENDING
    target: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.span_end < self.span_start:
            raise CatalogError(
                f"unit {self.unit_id}: span_end {self.span_end} precedes "
                f"span_start {self.span_start}")
        if self.max_columns is not None and self.max_columns < 1:
            raise CatalogError(
                f"unit {self.unit_id}: max_columns must be positive, got "
                f"{self.max_columns}")
        if self.status.is_injectable and self.target is None:
            # Both VERIFIED and TRANSLATED are injectable, so both must carry the text that
            # would be rendered -- otherwise a reinjector would write None into the carrier.
            raise CatalogError(
                f"unit {self.unit_id}: status {self.status.value!r} is injectable but "
                f"target is None")

    @property
    def is_done(self) -> bool:
        return self.status in (Status.VERIFIED, Status.SKIPPED)

    def with_target(self, target: str, status: Status) -> Unit:
        return replace(self, target=target, status=status)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["placeholders"] = list(self.placeholders)
        payload["context_before"] = list(self.context_before)
        payload["context_after"] = list(self.context_after)
        payload["notes"] = list(self.notes)
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str, *, path: Path | None = None,
                  line_no: int | None = None) -> Unit:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CatalogError(f"invalid JSON: {exc}", path=path, line_no=line_no) from exc
        missing = {"unit_id", "rel_path", "line_no", "span_start", "span_end", "command",
                   "kind", "source"} - raw.keys()
        if missing:
            raise CatalogError(f"missing fields {sorted(missing)}", path=path,
                               line_no=line_no)
        try:
            return cls(
                unit_id=raw["unit_id"],
                rel_path=raw["rel_path"],
                line_no=raw["line_no"],
                span_start=raw["span_start"],
                span_end=raw["span_end"],
                command=raw["command"],
                kind=raw["kind"],
                source=raw["source"],
                placeholders=tuple(raw.get("placeholders", ())),
                speaker=raw.get("speaker"),
                context_before=tuple(raw.get("context_before", ())),
                context_after=tuple(raw.get("context_after", ())),
                max_columns=raw.get("max_columns"),
                status=Status(raw.get("status", Status.PENDING.value)),
                target=raw.get("target"),
                notes=tuple(raw.get("notes", ())),
            )
        except ValueError as exc:
            raise CatalogError(f"bad field value: {exc}", path=path, line_no=line_no) from exc


def make_unit_id(rel_path: str, source: str, ordinal: int) -> str:
    """Deterministic id that survives the unit moving within its source.

    Keyed on (source name, text, occurrence index) rather than position, so re-running
    recognition with different settings keeps the ids of every payload that came out the
    same, and their translations survive with them. The occurrence index is what keeps a
    phrase repeated verbatim from collapsing into one id.
    """
    digest = hashlib.sha1(f"{rel_path}\0{source}\0{ordinal}".encode("utf-8")).hexdigest()
    return digest[:16]


def fsync_directory(path: Path) -> None:
    """Flush a directory entry, so a rename into it survives power loss.

    Renaming is atomic, but atomicity is not durability: without this the new name can be
    lost while the old one is already gone. Not every filesystem supports opening a
    directory, and where it is unsupported the rename is durable anyway, so a refusal is
    not an error.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_catalog(units: Iterable[Unit], path: Path) -> int:
    """Write units as JSONL, atomically. Returns the number written.

    Writes a temporary sibling and renames it over the target, so a failure part-way
    through cannot leave a half-written catalogue where a good one used to be. The data
    is fsynced before the rename, because a rename that lands before its contents do
    would publish a truncated file under the real name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for unit in units:
                handle.write(unit.to_json())
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        fsync_directory(path.parent)
    except BaseException:
        # The source iterable can raise mid-write; leaving the temporary behind would
        # accumulate stale ``.partial`` files that look like interrupted runs.
        temporary.unlink(missing_ok=True)
        raise
    return count


def read_catalog(path: Path) -> Iterator[Unit]:
    """Stream units from a JSONL catalogue, validating each line."""
    if not path.is_file():
        raise CatalogError("catalogue not found", path=path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if stripped:
                yield Unit.from_json(stripped, path=path, line_no=line_no)


def read_journal(journal: Path) -> Iterator[Unit]:
    """Yield every result recorded in ``journal``, in write order.

    A malformed *final* line without a trailing newline is tolerated and dropped: that is
    the expected shape of a journal left behind by a process killed mid-write, and the
    unit it describes simply stays pending. A malformed line anywhere else means the
    journal is corrupt, and skipping it would discard a completed translation while still
    reporting success, so it raises instead.

    Holds the journal in memory, which is bounded by the catalogue size.
    """
    if not journal.is_file():
        return
    with journal.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        try:
            yield Unit.from_json(line)
        except Exception as exc:
            if index == len(lines) - 1 and not raw.endswith("\n"):
                return
            # CatalogError already renders its own location, so take the bare reason
            # rather than the formatted message, or the path prints twice.
            reason = exc.reason if isinstance(exc, CatalogError) else str(exc)
            raise CatalogError(f"corrupt journal record: {reason}",
                               path=journal, line_no=index + 1) from exc


def merge_journal(catalog: Path, journal: Path, output: Path) -> tuple[int, int]:
    """Fold journalled results into the catalogue.

    Later journal entries win, so re-running a unit supersedes its earlier result.
    Returns ``(units_written, units_updated)``.

    A journalled result whose unit id is not in the catalogue is refused rather than dropped:
    silently discarding a completed translation while reporting success is exactly the failure
    :func:`read_journal` guards against, and it usually means the journal and the catalogue
    have drifted apart (a unit was renamed or removed) -- which the operator needs to know.
    """
    if not journal.is_file():
        raise CatalogError("journal not found", path=journal)
    results: dict[str, Unit] = {unit.unit_id: unit for unit in read_journal(journal)}

    updated = 0
    used: set[str] = set()
    merged: list[Unit] = []
    for unit in read_catalog(catalog):
        replacement = results.get(unit.unit_id)
        if replacement is not None:
            merged.append(replacement)
            used.add(unit.unit_id)
            updated += 1
        else:
            merged.append(unit)

    orphans = results.keys() - used
    if orphans:
        sample = ", ".join(sorted(orphans)[:5])
        raise CatalogError(
            f"journal holds {len(orphans)} result(s) with no matching unit in the catalogue "
            f"(e.g. {sample}); merging would silently discard them", path=journal)
    return write_catalog(merged, output), updated
