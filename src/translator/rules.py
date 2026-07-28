"""Configurable translation rules and their mechanical enforcement.

Rules come in two kinds, and the split is deliberate:

* **Mechanical** rules are decided by code -- placeholder integrity, untranslated text,
  line width, glossary term presence, forbidden patterns, respelled names. These are
  cheap, deterministic and not subject to a model's opinion, so they never consume GPU
  time and never produce a false "looks fine to me".
* **Advisory** rules are prose criteria handed to the reviewing agents (register, tense,
  voice). They are expressed as text because judging them is exactly what a language model
  is for.

Both are declared in one TOML file so the policy is data, not code.
"""
from __future__ import annotations

import re
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path

from transunit.placeholders import placeholder_indices
from transunit.width import display_columns

# Anything a single-line carrier cannot hold. Line breaks are excluded from a translation
# on purpose: where a payload wraps is decided by the carrier -- a subtitle layout, an
# engine's fixed box -- against a budget the model cannot measure, so a break arriving from
# the model would be placed before that budget is known.
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")


def _describe_control(character: str) -> str:
    """Readable name for a control character; unicodedata has none for these."""
    return {"\n": "LINE FEED", "\r": "CARRIAGE RETURN", "\x00": "NUL",
            "\x0b": "VERTICAL TAB", "\x0c": "FORM FEED", "\x1b": "ESCAPE",
            "\x7f": "DELETE"}.get(character, f"U+{ord(character):04X}")


class Severity(str, Enum):
    ERROR = "error"
    """Translation must be rejected and retried."""
    WARNING = "warning"
    """Recorded for review; does not block acceptance."""


@dataclass(frozen=True, slots=True)
class Violation:
    rule_id: str
    severity: Severity
    message: str

    @property
    def blocking(self) -> bool:
        return self.severity is Severity.ERROR


class RuleConfigError(Exception):
    """The rule file is missing, malformed, or internally inconsistent."""

    def __init__(self, reason: str, *, path: Path | None = None) -> None:
        super().__init__(f"{path}: {reason}" if path else reason)
        self.reason = reason
        self.path = path


def _boolean(section: dict, key: str, default: bool, *, label: str, path: Path) -> bool:
    """Read a TOML boolean, rejecting a mistyped value rather than coercing it.

    bool("false") is True, so coercion here would silently *invert* a safety gate: quoting
    allow_untranslated would switch off the check that keeps untranslated text out of the
    finished output.
    """
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise RuleConfigError(
            f"{label}.{key} must be a boolean, got {type(value).__name__} {value!r}",
            path=path)
    return value


def _number(section: dict, key: str, default: float, *, label: str, path: Path) -> float:
    """A TOML number, rejected rather than coerced when mistyped.

    Range is left to :class:`RuleSet`, so the bound holds for a directly-constructed rule set
    too rather than only for one that came through a file.
    """
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuleConfigError(
            f"{label}.{key} must be a number, got {type(value).__name__} {value!r}",
            path=path)
    return float(value)


def _string_list(section: dict, key: str, *, label: str, path: Path) -> tuple[str, ...]:
    """Read a TOML array of strings.

    A bare string is rejected because tuple() would silently split it into one-character
    directives rather than failing.
    """
    value = section.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuleConfigError(
            f"{label}.{key} must be an array of strings, got {value!r}", path=path)
    return tuple(value)


def _reject_unknown(section: object, allowed: set[str], *, label: str, path: Path) -> dict:
    """The section as a table, with any unrecognised key refused.

    Every section is read through here, so the "is this even a table?" question is answered
    here once: writing ``limits = 5`` instead of ``[limits]`` used to reach ``set(section)``
    and surface as a bare ``TypeError`` traceback rather than a diagnosis. An unrecognised key
    is refused rather than dropped, so a typo cannot silently leave a default in force.
    """
    if not isinstance(section, dict):
        raise RuleConfigError(
            f"{label} must be a table, got {type(section).__name__} {section!r}", path=path)
    unknown = set(section) - allowed
    if unknown:
        raise RuleConfigError(
            f"{label} has unknown key(s) {sorted(unknown)}; allowed keys are {sorted(allowed)}",
            path=path)
    return section


def _tables(data: dict, key: str, *, path: Path) -> list[dict]:
    """The entries of an array of tables, e.g. ``[[forbidden]]``.

    Writing ``[forbidden]`` for ``[[forbidden]]`` is named explicitly because iterating a
    table yields its *keys*: each "entry" became a bare string, and the resulting complaint
    about unknown keys ``['a', 'e', 'n', ...]`` pointed nowhere near the real mistake.
    """
    entries = data.get(key, [])
    if not isinstance(entries, list):
        raise RuleConfigError(
            f"[[{key}]] must be an array of tables -- write [[{key}]], not [{key}] -- got "
            f"{type(entries).__name__}", path=path)
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuleConfigError(
                f"[[{key}]] entry {position} must be a table, got {type(entry).__name__} "
                f"{entry!r}", path=path)
    return entries


MIN_LINE_COLUMNS = 20
"""Below this a "line" cannot hold even a short clause, so a smaller budget is a mistyped
value rather than a tight one -- and would reject every translation offered against it."""


@dataclass(frozen=True, slots=True)
class RuleSet:
    """Policy governing an acceptable translation."""

    max_line_columns: int = 110
    max_columns_tolerance: float = 0.12
    """How far a translation may exceed a unit's hard budget and still be accepted.

    The budget is reading time or box space, not an exact cliff: exceeding it slightly is a
    marginal defect, while rejecting the unit means the untranslated source is shown in its
    place. Zero tolerance therefore trades a real defect for a much worse one, which is what
    0 would mean here.
    """
    allow_untranslated: bool = False
    require_nonempty: bool = True
    forbidden_patterns: tuple[tuple[str, str], ...] = ()
    """``(regex, explanation)`` pairs rejected outright."""
    style_directives: tuple[str, ...] = ()
    advisory_rules: tuple[tuple[str, str], ...] = ()
    """``(id, description)`` criteria evaluated by the reviewing agents."""
    glossary_severity: Severity = Severity.WARNING

    def __post_init__(self) -> None:
        if self.max_line_columns < MIN_LINE_COLUMNS:
            raise ValueError(
                f"max_line_columns={self.max_line_columns} is implausibly small "
                f"(minimum {MIN_LINE_COLUMNS})")
        if not 0.0 <= self.max_columns_tolerance <= 1.0:
            raise ValueError(
                f"max_columns_tolerance must be a fraction in [0, 1], got "
                f"{self.max_columns_tolerance}")

    @classmethod
    def load(cls, path: Path) -> RuleSet:
        if not path.is_file():
            raise RuleConfigError("rule file not found", path=path)
        try:
            data = tomllib.loads(path.read_text("utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise RuleConfigError(f"invalid TOML: {exc}", path=path) from exc

        _reject_unknown(data, {"limits", "glossary", "style", "forbidden", "advisory"},
                        label="the rule file", path=path)
        limits = _reject_unknown(data.get("limits", {}),
                                 {"max_line_columns", "max_columns_tolerance",
                                  "allow_untranslated", "require_nonempty"},
                                 label="[limits]", path=path)
        style = _reject_unknown(data.get("style", {}), {"directives"},
                                label="[style]", path=path)
        glossary = _reject_unknown(data.get("glossary", {}), {"severity"},
                                   label="[glossary]", path=path)
        forbidden_entries = _tables(data, "forbidden", path=path)
        advisory_entries = _tables(data, "advisory", path=path)
        for entry in forbidden_entries:
            _reject_unknown(entry, {"pattern", "reason"}, label="[[forbidden]]", path=path)
        for entry in advisory_entries:
            _reject_unknown(entry, {"id", "description"}, label="[[advisory]]", path=path)
        max_columns = limits.get("max_line_columns", 110)
        # bool is a subclass of int, so `= true` would otherwise read as 1 column.
        if not isinstance(max_columns, int) or isinstance(max_columns, bool):
            raise RuleConfigError(
                f"limits.max_line_columns must be an integer, got "
                f"{type(max_columns).__name__} {max_columns!r}", path=path)

        forbidden: list[tuple[str, str]] = []
        for entry in forbidden_entries:
            pattern, why = entry.get("pattern"), entry.get("reason", "")
            if not pattern:
                raise RuleConfigError("a [[forbidden]] entry has no 'pattern'", path=path)
            try:
                re.compile(pattern)
            except re.error as exc:
                raise RuleConfigError(
                    f"forbidden pattern {pattern!r} is not a valid regex: {exc}",
                    path=path) from exc
            forbidden.append((pattern, why))

        advisory: list[tuple[str, str]] = []
        for entry in advisory_entries:
            rule_id, description = entry.get("id"), entry.get("description")
            if not rule_id or not description:
                raise RuleConfigError("an [[advisory]] entry needs 'id' and 'description'",
                                      path=path)
            advisory.append((rule_id, description))

        severity_name = glossary.get("severity", "warning")
        try:
            glossary_severity = Severity(severity_name)
        except ValueError as exc:
            raise RuleConfigError(
                f"glossary.severity must be 'error' or 'warning', got {severity_name!r}",
                path=path) from exc

        try:
            return cls(
                max_line_columns=max_columns,
                max_columns_tolerance=_number(
                    limits, "max_columns_tolerance", 0.12, label="limits", path=path),
                allow_untranslated=_boolean(
                    limits, "allow_untranslated", False, label="limits", path=path),
                require_nonempty=_boolean(
                    limits, "require_nonempty", True, label="limits", path=path),
                forbidden_patterns=tuple(forbidden),
                style_directives=_string_list(style, "directives", label="style", path=path),
                advisory_rules=tuple(advisory),
                glossary_severity=glossary_severity,
            )
        except ValueError as exc:
            raise RuleConfigError(f"[limits]: {exc}", path=path) from exc


def check_mechanical(source: str, target: str, ruleset: RuleSet,
                     *, expected_placeholders: int,
                     required_terms: tuple[str, ...] = (),
                     max_columns: int | None = None,
                     is_untranslated: Callable[[str, str], bool] | None = None,
                     names: Sequence[str] = ()) -> list[Violation]:
    """Every code-decidable rule violation in ``target``.

    ``source`` is the masked source payload; ``target`` the proposed translation.

    ``is_untranslated`` decides whether the model handed back the source instead of
    translating it. It is injected rather than imported because the answer depends on the
    language pair, and a rule module that imported one pair's detector would be usable for
    only that pair. Omitting it disables the check, which is correct for a caller that has
    no way to tell the languages apart -- a check that cannot be evaluated must not be
    reported as passed.
    """
    violations: list[Violation] = []

    if ruleset.require_nonempty and source.strip() and not target.strip():
        violations.append(Violation("nonempty", Severity.ERROR,
                                    "translation is empty but the source is not"))

    expected = set(range(expected_placeholders))
    actual = placeholder_indices(target)
    if set(actual) != expected:
        violations.append(Violation(
            "placeholders", Severity.ERROR,
            f"placeholder set changed: expected {sorted(expected)}, found "
            f"{sorted(set(actual))}"))
    elif len(actual) != len(set(actual)):
        violations.append(Violation("placeholders", Severity.ERROR,
                                    f"a placeholder is repeated: {actual}"))

    if (not ruleset.allow_untranslated and is_untranslated is not None
            and is_untranslated(source, target)):
        violations.append(Violation(
            "untranslated", Severity.ERROR,
            "translation is still in the source language, or is a verbatim copy of the "
            "source"))

    # The translator returns one continuous sentence; where it breaks across displayed
    # lines is decided afterwards by the carrier, against a budget the model cannot measure
    # reliably. A line break arriving from the model would be applied before that budget is
    # known and would survive into the rendered output in the wrong place.
    if control := _CONTROL_CHARACTERS.search(target):
        violations.append(Violation(
            "control_character", Severity.ERROR,
            f"translation contains a raw {_describe_control(control.group(0))} character; "
            f"return one unbroken line and let the carrier decide the breaks"))

    if max_columns is not None:
        # A hard limit comes from the carrier itself -- reading time for a subtitle, box
        # width for a fixed cell. The whole payload is measured rather than each
        # break-separated part, because whether a break is permitted or rendered is carrier
        # knowledge the translator does not have, and for reading time a break costs the
        # reader nothing anyway.
        width = display_columns(target)
        if width > max_columns:
            # Tolerated overshoot is a warning, not a rejection: a few per cent over costs a
            # fraction of a second or a hair of overflow, while a rejected unit is not
            # rendered at all and the untranslated source is shown instead. Past the
            # tolerance the budget has plainly been ignored rather than narrowly missed, and
            # repairing is worth a round.
            allowed = int(max_columns * (1.0 + ruleset.max_columns_tolerance))
            over = width - max_columns
            if width > allowed:
                violations.append(Violation(
                    "line_width", Severity.ERROR,
                    f"translation is {width} characters, {over} over the {max_columns} this "
                    f"payload has room for; say it more briefly"))
            else:
                violations.append(Violation(
                    "line_width", Severity.WARNING,
                    f"translation is {width} characters, {over} over the {max_columns} "
                    f"budget but within tolerance"))
    else:
        # The project guideline is about readability in a carrier that wraps, so it measures
        # each line the translator chose to break and only warns.
        for index, segment in enumerate(target.split("\\n")):
            width = display_columns(segment)
            if width > ruleset.max_line_columns:
                violations.append(Violation(
                    "line_width", Severity.WARNING,
                    f"line {index + 1} is {width} columns, over the "
                    f"{ruleset.max_line_columns}-column limit; break it with \\n"))

    violations.extend(check_names(source, target, names))

    for pattern, why in ruleset.forbidden_patterns:
        if re.search(pattern, target):
            violations.append(Violation("forbidden", Severity.ERROR,
                                        f"matches forbidden pattern {pattern!r}"
                                        + (f": {why}" if why else "")))

    for term in required_terms:
        if term and term.lower() not in target.lower():
            violations.append(Violation(
                "glossary", ruleset.glossary_severity,
                f"established term {term!r} does not appear in the translation"))

    return violations


def blocking(violations: list[Violation]) -> list[Violation]:
    return [violation for violation in violations if violation.blocking]


NAME_SIMILARITY = 0.5
NAME_PREFIX = 2
NAME_STEM_NUMERATOR = 2
NAME_STEM_DENOMINATOR = 3
"""What marks a word as a botched attempt at a name rather than an unrelated word.

Similarity alone cannot separate the two: measured against a real run, genuine corruptions
scored 0.50 to 0.92 and coincidental resemblances 0.36 to 0.73, which overlap across most
of their range. "Bazyla" for Basil scores 0.55 and is a corruption; "drzewa" for Dream
scores 0.73 and is simply a Polish word.

A shared opening separates them cleanly. A model reaching for a name and mangling it keeps
the beginning -- every corrupted form of Basil observed began "Ba" -- while a coincidental
resemblance usually shares nothing at the front. Requiring both a common prefix and a
modest similarity admitted every real corruption in the sample and rejected every false
one."""


def _common_prefix(first: str, second: str) -> int:
    length = 0
    # strict=False on purpose: a common prefix ends at the shorter string, so unequal lengths
    # are expected, not an error.
    for left, right in zip(first, second, strict=False):
        if left != right:
            break
        length += 1
    return length


def check_names(source: str, target: str, names: Sequence[str]) -> list[Violation]:
    """Names whose stem the translation respelled rather than inflected.

    Advisory instruction does not hold a model to a spelling. Told to keep a name as it is,
    a 14B model still substituted its target-language equivalent in a third of the mentions,
    and across a single film produced eleven spellings of one character. Consistency of a
    name is decidable in code, so it is decided here.

    Inflection is not corruption, and the difference is *where* the two diverge. A declined
    form keeps the name's stem and changes only the tail: Polish "Praga" -> "Pragi" (genitive),
    "Anna" -> "Anny", "Warszawa" -> "Warszawy", "Kraków" -> "Krakowie". A respelling diverges
    earlier, inside the stem: "Basil" -> "Bazyl", "Adam" -> "Adem". So a token is treated as a
    legitimate inflection -- and left alone -- when it shares at least two thirds of the name as a
    prefix; only a token that diverges before that, and is still similar enough to be reaching
    for the name, is flagged.

    The threshold is a *fraction* rather than an absolute margin on purpose: an absolute
    "all but the last N characters" collapses to an empty detection window for short names
    (with N=2 and the two-character prefix floor, nothing of length <= 4 could ever be flagged),
    which silently disabled the check for the most common given names. Two thirds admits the
    genitive of a four-letter name (3/4) and a mid-stem change of a six-letter one (Krakowie,
    4/6) while still catching a respelling of a short one (Adem, 2/4).

    An earlier version instead demanded the *whole* name survive as a substring, which held only
    for suffixing inflections and flagged every stem-changing genitive -- the dominant pattern
    for the project's flagship pairs. Worse, the only way the model could clear the false flag
    was to revert to the uninflected base form, which is ungrammatical target text: the check was
    pushing toward the very error its docstring warns against. One case is inherently
    undecidable -- a corruption that changes only the final character ("John" -> "Johh") is
    structurally identical to an inflection ("Anna" -> "Anny") -- and is deliberately allowed
    through, because in the inflected languages this serves the inflection reading is right far
    more often.

    Absence is not reported. Subtitling drops a repeated name for a pronoun as a matter of
    course, and a check that insisted on the name appearing would reject good concision. Only a
    word that is clearly *reaching* for the name and missing counts.
    """
    violations: list[Violation] = []
    lowered = target.lower()
    for name in names:
        name_lower = name.lower()
        if not re.search(rf"\b{re.escape(name)}", source, re.IGNORECASE):
            continue
        if name_lower in lowered:
            continue
        for token in re.findall(r"\b\w{3,}\b", target, re.UNICODE):
            # Capitalised only: an ordinary lower-case word that happens to resemble a name
            # is not an attempt at it, and flagging one costs a repair round chasing a defect
            # that is not there.
            if not token[:1].isupper():
                continue
            lowered_token = token.lower()
            prefix = _common_prefix(lowered_token, name_lower)
            if prefix < NAME_PREFIX:
                continue
            # Shares at least two thirds of the name as a prefix -> the stem is preserved and
            # only the tail differs -> inflection, not corruption. Integer form of
            # prefix / len(name) >= 2/3, to avoid float rounding at the boundary.
            if prefix * NAME_STEM_DENOMINATOR >= len(name_lower) * NAME_STEM_NUMERATOR:
                continue
            ratio = SequenceMatcher(None, lowered_token, name_lower).ratio()
            if ratio >= NAME_SIMILARITY:
                violations.append(Violation(
                    "name_respelled", Severity.ERROR,
                    f"the name {name!r} appears in the target as {token!r}; keep the "
                    f"spelling {name!r} and inflect only the ending"))
                break
    return violations
