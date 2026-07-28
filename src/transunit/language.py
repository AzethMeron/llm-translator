"""Deciding whether a translation actually happened, well enough to catch a failure.

The pipeline needs this for one blocking check: did the translator translate, or did it
hand back the source? A model asked to translate will occasionally echo its input, and
without a check that lands in the finished output.

There is no single right way to decide this, because it depends on the language pair,
so this module offers two mechanisms and both produce the same injectable shape --
``Callable[[str, str], bool]`` taking ``(source, target)``:

* **Lexical** (:class:`LanguageProfile`, :func:`lexical_detector`) for pairs that share a
  script. Polish and English are both Latin, so "does the target use a different
  alphabet" is always false and catches nothing; the evidence that separates them is
  lexical -- the closed-class words each language uses constantly and the other does not,
  plus each language's distinctive characters. Profiles are TOML data, so a new pair is a
  configuration change, not a code change.

* **Script** (:class:`ScriptDetector`) for pairs whose scripts differ. When the source is
  Japanese and the target English, the strongest evidence a translation did not happen is
  the source's own letters surviving into the target, which is decidable with no profiles
  or thresholds at all.

Both share two properties that matter more here than raw accuracy:

*They abstain / answer only on positive evidence.* A blocking check must never fire
without evidence -- an abstention used as a negative would turn every short line into a
reported failure.

*They are deterministic.* Same text, same verdict, always: a blocking check should not
depend on a model's weights.
"""
from __future__ import annotations

import re
import tomllib
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

MINIMUM_TOKENS = 3
"""Below this, the lexical detector offers no verdict. Two tokens is not enough to
outvote a single word that happens to exist in both languages."""

MINIMUM_MARGIN = 0.08
"""How far ahead the winner must be before a lexical result is reported rather than
abstained on. Set from the observation that a genuine match scores several times its
rival, so a near-tie means the evidence is absent, not balanced."""

# The injected untranslated-detector shape. A caller that cannot tell the two languages
# apart passes ``None`` instead, and the check is skipped rather than reported as passed.
UntranslatedDetector = Callable[[str, str], bool]


class LanguageError(ValueError):
    """A language profile is missing, malformed, or unknown."""

    def __init__(self, reason: str, *, code: str | None = None,
                 path: Path | None = None) -> None:
        location = f"{path}: " if path else ""
        target = f"[{code}] " if code else ""
        super().__init__(f"{location}{target}{reason}")
        self.reason = reason
        self.code = code
        self.path = path


# --- lexical detection --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    """The lexical fingerprint of one language."""

    code: str
    """ISO 639-1 code, matching what the rest of the pipeline uses."""
    name: str
    stopwords: frozenset[str] = field(default_factory=frozenset)
    """Function words that are frequent in this language and rare in its counterparts.
    Frequency is what makes them useful: a single occurrence is weak evidence, but their
    rate across a sentence is strong."""
    distinctive_characters: frozenset[str] = field(default_factory=frozenset)
    """Characters that occur in this language and effectively not in the others it is
    compared against -- Polish's accented letters against English."""

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise LanguageError("language code is empty")
        if not self.stopwords and not self.distinctive_characters:
            raise LanguageError(
                "profile has neither stopwords nor distinctive characters, so it can "
                "never match any text", code=self.code)

    def score(self, text: str) -> float:
        """How strongly this text looks like this language, from 0 upward.

        The two kinds of evidence are added rather than multiplied so that either alone
        can carry a verdict: a Polish sentence typed without diacritics is still
        recognisable from its function words, and a fragment too short for stopwords may
        still contain a distinctive letter.
        """
        tokens = tokenize(text)
        if not tokens:
            return 0.0
        hits = sum(1 for token in tokens if token in self.stopwords)
        lexical = hits / len(tokens)
        distinctive = sum(1 for character in text.lower()
                          if character in self.distinctive_characters)
        # Capped so a single heavily-accented word cannot outweigh a whole sentence of
        # the other language's function words.
        orthographic = min(distinctive / len(tokens), 1.0)
        return lexical + orthographic


@dataclass(frozen=True, slots=True)
class Detection:
    """A verdict, with the evidence behind it."""

    code: str
    confidence: float
    """The winner's margin over the runner-up, from 0.0 to 1.0."""
    scores: tuple[tuple[str, float], ...]

    def __str__(self) -> str:
        detail = ", ".join(f"{code}={score:.2f}" for code, score in self.scores)
        return f"{self.code} (confidence {self.confidence:.2f}; {detail})"


def tokenize(text: str) -> list[str]:
    """Split into casefolded word tokens, ignoring digits and punctuation.

    Casefolds rather than lowercases, because casefolding is the operation defined for
    comparison and handles the cases lowercasing does not.
    """
    return [match.group().casefold() for match in WORD.finditer(text)]


def detect(text: str, profiles: tuple[LanguageProfile, ...]) -> Detection | None:
    """Identify the language of ``text``, or return ``None`` if unsure.

    ``None`` means the evidence is insufficient, never that the text matched nothing.
    Callers must treat it as "no information" -- an abstention used as a negative would
    turn every short line into a reported failure.
    """
    if not profiles:
        raise LanguageError("no language profiles supplied, so nothing can be detected")
    if len(tokenize(text)) < MINIMUM_TOKENS:
        return None

    ranked = sorted(((profile.code, profile.score(text)) for profile in profiles),
                    key=lambda entry: entry[1], reverse=True)
    best_code, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if best_score <= 0.0 or best_score - runner_up < MINIMUM_MARGIN:
        return None
    return Detection(code=best_code, confidence=min(best_score - runner_up, 1.0),
                     scores=tuple(ranked))


def is_untranslated(source: str, target: str,
                    profiles: tuple[LanguageProfile, ...],
                    source_code: str) -> bool:
    """Whether ``target`` appears to still be in the source language, lexically.

    Answers ``True`` only on positive evidence, so the blocking check it backs cannot
    fire on an abstention. An identical copy of the source is caught outright, before any
    detection runs, because that case needs no evidence and is the most common way the
    failure actually appears.
    """
    stripped = target.strip()
    if not stripped:
        return False
    if stripped == source.strip():
        return True
    verdict = detect(stripped, profiles)
    return verdict is not None and verdict.code == source_code


def lexical_detector(profiles: tuple[LanguageProfile, ...],
                     source_code: str) -> UntranslatedDetector:
    """Bind :func:`is_untranslated` to a profile set and source language.

    Returns the ``(source, target) -> bool`` shape the translator harness injects, so a
    caller wires it in without threading ``profiles`` and ``source_code`` through every
    layer. ``source_code`` is validated against the profiles up front, turning a typo
    into an error here rather than a check that silently never fires.
    """
    find_profile(profiles, source_code)
    return lambda source, target: is_untranslated(source, target, profiles, source_code)


def load_profiles(path: Path) -> tuple[LanguageProfile, ...]:
    """Load every profile from the ``[languages]`` table of a TOML file."""
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except FileNotFoundError:
        raise LanguageError("language profiles not found", path=path) from None
    except tomllib.TOMLDecodeError as exc:
        raise LanguageError(f"invalid TOML: {exc}", path=path) from exc

    table = document.get("languages")
    if not isinstance(table, dict) or not table:
        raise LanguageError(
            "no [languages] table, or it is empty; at least one profile is needed to "
            "tell a translation from its source", path=path)

    return tuple(_profile(code, body, path=path) for code, body in table.items())


def _profile(code: str, body: Any, *, path: Path) -> LanguageProfile:
    if not isinstance(body, dict):
        raise LanguageError(f"must be a table, got {type(body).__name__}",
                            code=code, path=path)
    unknown = set(body) - {"name", "stopwords", "distinctive_characters"}
    if unknown:
        raise LanguageError(
            f"unknown key(s) {sorted(unknown)}; expected name, stopwords, "
            f"distinctive_characters", code=code, path=path)
    return LanguageProfile(
        code=code,
        name=_text(body.get("name", code), "name", code=code, path=path),
        stopwords=frozenset(
            word.casefold()
            for word in _words(body.get("stopwords", []), "stopwords", code=code, path=path)),
        distinctive_characters=frozenset(
            character.lower()
            for character in _text(body.get("distinctive_characters", ""),
                                   "distinctive_characters", code=code, path=path)
            if not character.isspace()),
    )


def _text(value: Any, key: str, *, code: str, path: Path) -> str:
    if not isinstance(value, str):
        raise LanguageError(f"{key} must be a string, got {type(value).__name__}",
                            code=code, path=path)
    return value


def _words(value: Any, key: str, *, code: str, path: Path) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LanguageError(f"{key} must be a list of strings", code=code, path=path)
    return value


def find_profile(profiles: tuple[LanguageProfile, ...], code: str) -> LanguageProfile:
    """Look up a profile by code, listing what is available when it is missing."""
    for profile in profiles:
        if profile.code == code:
            return profile
    known = ", ".join(sorted(profile.code for profile in profiles)) or "none"
    raise LanguageError(
        f"no profile for language {code!r}; configured profiles are {known}", code=code)


def strip_accents(text: str) -> str:
    """Remove diacritics, for comparing text that may have lost them in transit.

    Useful when checking whether a target is the source with its diacritics stripped,
    which is a distinct failure from an untranslated echo: it means the text survived but
    the encoding did not.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch)))


# --- script detection ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScriptDetector:
    """An untranslated-detector for language pairs whose scripts differ.

    Callable as ``(source, target) -> bool``, so it drops straight into the translator
    harness in place of the lexical detector. It reports ``True`` when the target is a
    verbatim copy of the source, or still contains letters of the source's script.

    ``ignore`` names characters that fall inside ``letters`` but carry no word of their
    own -- Japanese small kana, which modify a preceding syllable, or punctuation living
    inside a letter block. A payload made only of those is not evidence of an
    untranslated line.
    """

    letters: re.Pattern[str]
    """Matches a single letter of the source script."""
    ignore: frozenset[str] = frozenset()

    @classmethod
    def from_ranges(cls, ranges: tuple[tuple[str, str], ...], *,
                    ignore: str = "") -> ScriptDetector:
        """Build from ``(first, last)`` inclusive character-range pairs."""
        if not ranges:
            raise LanguageError("a ScriptDetector needs at least one character range")
        body = "".join(f"{re.escape(low)}-{re.escape(high)}" for low, high in ranges)
        return cls(letters=re.compile(f"[{body}]"), ignore=frozenset(ignore))

    def has_source_script(self, text: str) -> bool:
        """True when ``text`` contains a source-script character that stands as a word."""
        return any(self.letters.match(char) and char not in self.ignore for char in text)

    def __call__(self, source: str, target: str) -> bool:
        stripped = target.strip()
        if not stripped:
            return False
        if stripped == source.strip():
            return True
        return self.has_source_script(stripped)


# Japanese letters: hiragana, katakana, kanji and halfwidth katakana. Excludes U+30FB
# (・), U+30FC (ー) and U+FF65 (･), which are punctuation living inside those blocks.
JAPANESE_RANGES: tuple[tuple[str, str], ...] = (
    ("ぁ", "ゖ"),  # hiragana letters
    ("ゝ", "ゟ"),  # hiragana iteration marks
    ("ァ", "ヺ"),  # katakana letters (stops before ・ and ー)
    ("ヽ", "ヿ"),  # katakana iteration marks
    ("一", "鿿"),  # CJK unified ideographs
    ("ｦ", "ﾝ"),  # halfwidth katakana letters
)

# Small kana modify the syllable before them and have no reading of their own, so a
# payload made only of these has no word in it to translate.
JAPANESE_SMALL_KANA = "ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶｧｨｩｪｫｬｭｮｯ"


def japanese_detector() -> ScriptDetector:
    """A ready-made :class:`ScriptDetector` for a Japanese source.

    A preset, not a special case: it is the general script mechanism with Japanese ranges.
    Any different-script source builds its own with :meth:`ScriptDetector.from_ranges`.
    """
    return ScriptDetector.from_ranges(JAPANESE_RANGES, ignore=JAPANESE_SMALL_KANA)


# Letter ranges (as Unicode code points) for the world's major scripts, so a script-based
# untranslated check for any different-script pair is a name, not hand-built ranges. These
# cover letters; punctuation and digit sub-blocks are left out where practical, because the
# check looks for a source *word* surviving into the target, not a stray sign. Han covers
# Chinese; Japanese has its own preset because of the small-kana exclusion.
_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "han": ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x323AF)),
    # Cyrillic letters, excluding the combining-mark sub-block (U+0483-U+0489) and the
    # thousands sign (U+0482), which are not word-forming.
    "cyrillic": ((0x0400, 0x0481), (0x048A, 0x052F)),
    # Greek letters including Ά (U+0386) but not the ano teleia (U+0387, punctuation).
    "greek": ((0x0386, 0x0386), (0x0388, 0x03A1), (0x03A3, 0x03FF), (0x1F00, 0x1FFF)),
    "arabic": ((0x0620, 0x064A), (0x066E, 0x06D3), (0x06FA, 0x06FF),
               (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "hebrew": ((0x05D0, 0x05EA), (0xFB1D, 0xFB4F)),
    "hangul": ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7A3)),
    "devanagari": ((0x0904, 0x0939), (0x0958, 0x0961)),
    "thai": ((0x0E01, 0x0E3A), (0x0E40, 0x0E4E)),
}


def available_scripts() -> list[str]:
    """Every script name :func:`script_detector` accepts, sorted."""
    return sorted({"japanese", *_SCRIPT_RANGES})


def script_detector(script: str) -> ScriptDetector:
    """A :class:`ScriptDetector` for a named source script.

    Turns a same-alphabet-blind but script-obvious pair -- Chinese/English, Russian/English,
    Greek/English, Arabic/English -- into a one-line configuration choice: name the *source*
    script and the detector reports a translation that still carries its letters. For a pair
    that shares a script (English/German), use :func:`lexical_detector` instead; for a script
    not listed here, build one directly with :meth:`ScriptDetector.from_ranges`.
    """
    if script == "japanese":
        return japanese_detector()
    ranges = _SCRIPT_RANGES.get(script)
    if ranges is None:
        raise LanguageError(
            f"unknown script {script!r}; known scripts are "
            f"{', '.join(available_scripts())}. Build a custom one with "
            f"ScriptDetector.from_ranges((first_char, last_char), ...)")
    return ScriptDetector.from_ranges(tuple((chr(low), chr(high)) for low, high in ranges))
