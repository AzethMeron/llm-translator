"""Who the agents are, and how much room they get, loaded from configuration.

Separate from :mod:`translator.rules` because the two answer different questions. The rule
set says what an acceptable translation *is*; this says who does the judging, what each
one's remit is, how much output each may produce, how much surrounding context each prompt
carries, and how hard the harness tries before giving up. Editing a criterion and editing a
role are different jobs, and conflating them meant editing Python to reword a prompt or
retune a budget.

Every role shares one loaded model and differs only by its instructions, so these strings
are the entire difference between a translator and a copy-editor.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from transunit.units import Status

_TOKEN = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
"""A substitution placeholder. Deliberately narrow -- lowercase identifiers only -- so
prose containing braces is not mistaken for one."""

DEFAULT_SUBSTITUTIONS = {
    "source_language": "the source language",
    "target_language": "the target language",
}
"""Filled in when a caller names no language pair.

The prompts still read correctly with these -- "translating from the source language into
the target language" is vague but not wrong -- so the agent file stays loadable without a
caller that knows the direction. A caller that does know passes the real names and gets a
better prompt; one that does not gets a working one rather than an exception."""


class AgentConfigError(Exception):
    """The agent file is missing, malformed, or internally inconsistent."""

    def __init__(self, reason: str, *, path: Path | None = None) -> None:
        super().__init__(f"{path}: {reason}" if path else reason)
        self.reason = reason
        self.path = path


def _at_least(value: int, minimum: int, *, what: str) -> None:
    """Enforce a lower bound, raising the plain :class:`ValueError` a dataclass invariant uses.

    Lives beside the types rather than the parser because the bound belongs to the type: a
    directly-constructed :class:`Limits` must be as impossible to make invalid as a loaded one.
    The loader turns this into an :class:`AgentConfigError` carrying the file and section, which
    are the only things it knows that the type does not.
    """
    if value < minimum:
        raise ValueError(f"{what} must be >= {minimum}, got {value}")


def _within_unit_interval(value: float, *, what: str) -> None:
    """Enforce a fraction in ``[0, 1]``, the shape every similarity floor and blend weight takes."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{what} must be in [0, 1], got {value}")


def _int(section: dict, key: str, default: int, *, label: str, path: Path) -> int:
    """A TOML integer, rejected rather than coerced when mistyped.

    bool is a subclass of int, so ``= true`` would otherwise read as 1 -- which silently
    turns a budget of "one token" or a context window of "one unit" into a plausible-looking
    but wrong value.

    Range is deliberately *not* checked here: every value this produces is handed to a
    dataclass that enforces its own bounds, so the bound has one home rather than two.
    """
    value = section.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise AgentConfigError(
            f"{label}.{key} must be an integer, got {type(value).__name__} {value!r}", path=path)
    return value


def _bool(section: dict, key: str, default: bool, *, label: str, path: Path) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise AgentConfigError(
            f"{label}.{key} must be a boolean, got {type(value).__name__} {value!r}",
            path=path)
    return value


def _string(section: dict, key: str, default: str, *, label: str, path: Path) -> str:
    value = section.get(key, default)
    if not isinstance(value, str):
        raise AgentConfigError(
            f"{label}.{key} must be a string, got {type(value).__name__}", path=path)
    return value


def _float(section: dict, key: str, default: float, *, label: str, path: Path) -> float:
    """A TOML number, rejected rather than coerced when mistyped.

    An int is accepted and widened (``0`` and ``1`` are natural values for a fraction), but bool
    is not: ``= true`` reading as ``1.0`` would silently turn a similarity floor into "accept
    only exact matches".

    Range, like :func:`_int`, is left to the dataclass that receives the value.
    """
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentConfigError(
            f"{label}.{key} must be a number, got {type(value).__name__} {value!r}", path=path)
    return float(value)


_LEARNABLE_STATUSES = (Status.VERIFIED, Status.TRANSLATED)
"""The only statuses that may seed the reference store: the injectable ones, which carry a
usable target. A rejected/skipped/pending unit does not, so offering it as reference would
spread a known-bad or absent rendering."""


def _statuses(section: dict, key: str, *, label: str, path: Path) -> tuple[Status, ...]:
    """An array of status names restricted to the learnable set, order-preserving and de-duped.

    Empty (or absent) means self-population is off. Any value outside verified/translated is
    rejected with a reason, rather than silently dropped, because a caller listing "rejected"
    almost certainly misunderstands what may seed the store.
    """
    raw = section.get(key, [])
    if not isinstance(raw, list):
        raise AgentConfigError(
            f"{label}.{key} must be an array of status names, got {type(raw).__name__}", path=path)
    allowed = {status.value for status in _LEARNABLE_STATUSES}
    result: list[Status] = []
    for item in raw:
        if not isinstance(item, str):
            raise AgentConfigError(
                f"{label}.{key} entries must be strings, got {type(item).__name__} {item!r}",
                path=path)
        if item not in allowed:
            raise AgentConfigError(
                f"{label}.{key}: {item!r} cannot seed the reference store; only "
                f"{sorted(allowed)} carry a usable target", path=path)
        status = Status(item)
        if status not in result:
            result.append(status)
    return tuple(result)


def _reject_unknown(section: object, allowed: set[str], *, label: str, path: Path) -> dict:
    """The section as a table, with any unrecognised key refused.

    Every section is read through here, so the "is this even a table?" question is answered
    here too, once. Writing ``limits = 5`` instead of ``[limits]`` is an ordinary TOML slip,
    and without this check it reached ``set(section)`` and surfaced as a bare ``TypeError``
    traceback rather than a diagnosis.

    An unrecognised key is refused rather than dropped: a mistyped ``max_reivsions`` would
    otherwise leave its default in force, leaving the operator to wonder why their setting
    had no effect -- the exact silent failure the project's rules forbid.
    """
    if not isinstance(section, dict):
        raise AgentConfigError(
            f"{label} must be a table, got {type(section).__name__} {section!r}", path=path)
    unknown = set(section) - allowed
    if unknown:
        raise AgentConfigError(
            f"{label} has unknown key(s) {sorted(unknown)}; allowed keys are {sorted(allowed)}",
            path=path)
    return section


def _tables(data: dict, key: str, *, path: Path) -> list[dict]:
    """The entries of an array of tables, e.g. ``[[reviewer]]``.

    Writing ``[reviewer]`` for ``[[reviewer]]`` is the likeliest mistake in the file, and it
    is named explicitly because the failure it used to produce -- iterating a table yields its
    *keys*, so each "entry" was a bare string -- pointed nowhere near the actual error.
    """
    entries = data.get(key, [])
    if not isinstance(entries, list):
        raise AgentConfigError(
            f"[[{key}]] must be an array of tables -- write [[{key}]], not [{key}] -- got "
            f"{type(entries).__name__}", path=path)
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise AgentConfigError(
                f"[[{key}]] entry {position} must be a table, got {type(entry).__name__} "
                f"{entry!r}", path=path)
    return entries


@dataclass(frozen=True, slots=True)
class Leniency:
    """How tolerant the *console* is of a role's unusable replies before it warns.

    A reviewer occasionally returns a reply the harness cannot evaluate (unparseable JSON, a
    field of the wrong shape). Each such reply is always recorded -- in the unit's journal note
    and in the full log -- but printing a warning for every one buries a systemic problem in
    noise. This bounds only the *printed* warning: within a rolling window of the last
    ``window`` replies from a role, up to ``max_bad`` unusable ones pass without a console
    warning; the one that pushes the count past ``max_bad`` is surfaced, because at that rate
    the role is failing rather than merely flaky. The log always gets every one regardless.
    """

    window: int = 20
    max_bad: int = 2

    def __post_init__(self) -> None:
        _at_least(self.window, 1, what="leniency window")
        _at_least(self.max_bad, 0, what="leniency max_bad")

    def surfaces(self, bad_in_window: int) -> bool:
        """Whether a bad reply, given the count of bad replies now in the window (itself
        included), should reach the console rather than only the log."""
        return bad_in_window > self.max_bad


@dataclass(frozen=True, slots=True)
class Reviewer:
    """One member of the review panel."""

    id: str
    instructions: str = ""
    from_rules: bool = False
    """Build the instructions from the rule set's advisory criteria instead.

    Keeps project policy in one file: the compliance reviewer reads the same ``[[advisory]]``
    entries that document what the project enforces, rather than restating them here where the
    two could drift apart.
    """
    max_tokens: int | None = None
    """Per-reviewer output ceiling, or ``None`` to use :attr:`Limits.review_tokens`.

    A reviewer returns its verdict *and*, on objection, a full corrected translation, so a
    reviewer that rewrites long units needs more room than one that only checks punctuation.
    Set it per role rather than forcing one number to cover every reviewer.
    """
    leniency: Leniency = Leniency()
    """This reviewer's console-warning tolerance. Resolved at load time to the reviewer's own
    ``leniency`` override, or the file-wide ``[leniency]`` default when it sets none, so the
    harness reads one value here without a fallback lookup."""

    def __post_init__(self) -> None:
        if self.max_tokens is not None:
            _at_least(self.max_tokens, 1, what="max_tokens")


@dataclass(frozen=True, slots=True)
class Limits:
    """Output token budgets."""

    translate_tokens_per_source_char: int = 8
    translate_tokens_floor: int = 256
    translate_tokens_ceiling: int = 1024
    review_tokens: int = 1024
    """Default per-reviewer output ceiling, used by any reviewer that sets no ``max_tokens``."""

    def __post_init__(self) -> None:
        _at_least(self.translate_tokens_per_source_char, 1,
                  what="translate_tokens_per_source_char")
        _at_least(self.translate_tokens_floor, 1, what="translate_tokens_floor")
        _at_least(self.translate_tokens_ceiling, 1, what="translate_tokens_ceiling")
        _at_least(self.review_tokens, 1, what="review_tokens")
        if self.translate_tokens_floor > self.translate_tokens_ceiling:
            raise ValueError(
                f"translate_tokens_floor ({self.translate_tokens_floor}) exceeds "
                f"translate_tokens_ceiling ({self.translate_tokens_ceiling}), so every budget "
                f"would be the ceiling")

    def translate_budget(self, source_length: int) -> int:
        """Token ceiling for translating a source of ``source_length`` characters.

        Bounded at both ends: the floor keeps a very short line able to produce a full
        sentence, the ceiling stops an unusually long one asking for an unbounded response.
        """
        scaled = source_length * self.translate_tokens_per_source_char
        return max(self.translate_tokens_floor,
                   min(self.translate_tokens_ceiling, scaled))

    def review_budget(self, reviewer: Reviewer) -> int:
        """The output ceiling for ``reviewer`` -- its own if set, else the shared default."""
        return reviewer.max_tokens if reviewer.max_tokens is not None else self.review_tokens


@dataclass(frozen=True, slots=True)
class Context:
    """How much surrounding material each prompt carries.

    These are the knobs that keep the compounded prompt bounded. Context and glossary grow
    the input, and the input shares the model's window with the output budget, so caps here
    are what stop a long run from silently overflowing on units with a lot of neighbours or a
    large glossary. Set them from your unit size: short subtitle or dialogue lines tolerate a
    wider window than paragraph-sized document units.
    """

    before_units: int = 3
    """How many preceding units to show as context (the nearest ``before_units``)."""
    after_units: int = 2
    """How many following units to show (the nearest ``after_units``)."""
    include_translations: bool = True
    """Also list a neighbour's established translation when memory has one.

    The surrounding-context passage always shows neighbours in the *source* language; this adds
    a separate list of the ones already translated, paired with their targets. That pairing is
    what keeps terminology, register and pronouns continuous, because the model sees how a
    neighbouring line was actually rendered rather than only what it said. Turn it off to debug,
    or to keep the prompt to the source passage alone.
    """
    context_char_budget: int = 0
    """Soft character budget the continuity block (surrounding context plus, if enabled,
    retrieved reference examples) is trimmed to, beyond the innermost guaranteed neighbour on
    each side. ``0`` -- the default -- disables the budget: every neighbour ``before_units`` /
    ``after_units`` selects, and every reference example ``reference_examples`` selects, is
    shown, exactly like before this budget existed. Set it only if a unit type has enough
    neighbours or reference examples that the assembled prompt risks the model's context window;
    the immediate neighbour on each side is never trimmed regardless of the budget."""
    glossary_terms: int = 12
    """Maximum established terms shown for a unit (the longest-matching ``glossary_terms``)."""
    reference_examples: int = 0
    """How many external reference translations to retrieve for a unit *before* translating,
    keyed on the unit's source (the nearest ``reference_examples`` by similarity). ``0`` -- the
    default -- turns retrieval off entirely: with no reference corpus, or this at zero, the
    prompt is exactly as it is without the feature. Needs a corpus supplied to the run (see
    ``--reference``). Shares the prompt window with context and the glossary, so size it the
    same way."""
    reference_min_score: float = 0.30
    """Similarity floor in ``[0, 1]`` for a retrieved reference to be shown. A match below it is
    dropped rather than shown as weak guidance, because a wrong exemplar steers worse than none.
    Only consulted when ``reference_examples`` or ``reference_revision_examples`` is above zero.
    With the hybrid retriever (``--hybrid``) this is the *final* relevance floor -- applied after
    fusion and, if configured, reranking -- not a per-arm floor; see ``reference_lexical_min_score``
    / ``reference_embedding_min_score`` for those. On that path the scale is fixed rather than
    per-query, so the floor doubles as an agreement dial: a candidate both arms rank first scores
    ``1.0`` and one only a single arm found tops out at ``0.5``, so a floor above ``0.5`` keeps
    only candidates both arms agree on.

    Default ``0.30``: a relevance audit found precision only becomes high (~three quarters clearly
    relevant, negligible noise) at ~0.30, so the default favours clean matches over recall. Lower
    it (0.15-0.20) to retrieve on more units, accepting more marginal matches; see
    ``docs/reference-translations.md``."""
    reference_candidate_pool: int = 40
    """How many candidates each arm of the hybrid retriever (``--hybrid``) retrieves before
    fusion, reranking and MMR narrow them down to ``reference_examples``. Unused by the lexical
    or plain-embedding retrievers. Default ``40``: ample material for Reciprocal Rank Fusion and
    MMR to work with even for a handful of final examples, while staying cheap -- a rerank call
    scores at most this many candidates, not the whole corpus."""
    reference_mmr_lambda: float = 0.7
    """The relevance/diversity balance for the hybrid retriever's Maximal Marginal Relevance
    step, in ``[0, 1]``: ``1.0`` is pure relevance ranking (no diversity term), ``0.0`` maximises
    diversity and ignores relevance past the first pick. Default ``0.7``, the conventional
    relevance-leaning MMR value: examples should mostly be the most relevant ones, with just
    enough diversity pressure to stop a top-k list being dominated by near-duplicate lines from
    the same passage."""
    reference_lexical_min_score: float = 0.30
    """The hybrid retriever's *per-arm* recall floor for its lexical (character n-gram) side, in
    ``[0, 1]``, applied before fusion -- not the final floor a shown example must clear (see
    ``reference_min_score``). Defaults to the same evidence-based value as the plain lexical
    retriever's own floor, since it is the same algorithm underneath."""
    reference_embedding_min_score: float = 0.55
    """The hybrid retriever's *per-arm* recall floor for its dense (embedding) side, in ``[0, 1]``,
    applied before fusion. Embedding cosines cluster higher than lexical scores, so this floor is
    calibrated separately -- the same reasoning, and the same default, as the plain embedding
    retriever's own floor."""
    reference_revision_examples: int = 0
    """How many reference translations to retrieve *during revision*, keyed on the current draft
    (the nearest ``reference_revision_examples`` by target-side similarity). ``0`` -- the default
    -- turns this off. This is the path that uses reference material we hold *without* its source:
    it cannot seed a translation, but matched against a draft it can pull phrasing and register
    toward an established target voice."""
    reference_learn_statuses: tuple[Status, ...] = ()
    """Which of this run's own results to add to the reference store as it goes, so a later unit
    can retrieve an earlier one by similarity (not only the exact matches dedup shares).

    Empty -- the default -- keeps the reference store read-only. A subset of ``["verified",
    "translated"]`` turns self-population on for those statuses: ``["verified"]`` learns only from
    fully-verified lines, ``["verified", "translated"]`` also from the ones kept for review
    ("needs verification"). Only those two are allowed -- a rejected, skipped or pending unit has
    no usable target to offer. Turning this on makes retrieval *adaptive rather than
    reproducible*: what a unit sees depends on what finished before it. Only useful alongside a
    non-zero ``reference_examples`` or ``reference_revision_examples``, so the learned entries
    have a direction to surface through."""
    anonymous_subject: str = "they"
    """Stand-in for a placeholder in a *context* line whose speaker is unknown.

    A placeholder in a neighbour's line stands for a name lookup; shown raw it teaches the
    model to emit one on a line that has none. It is replaced with the speaker's established
    name when known, and with this word otherwise. Set it to a natural third-person word in
    the *source* language, since context lines are shown in the source until translated.
    """

    def __post_init__(self) -> None:
        _at_least(self.before_units, 0, what="before_units")
        _at_least(self.after_units, 0, what="after_units")
        _at_least(self.context_char_budget, 0, what="context_char_budget")
        _at_least(self.glossary_terms, 0, what="glossary_terms")
        _at_least(self.reference_examples, 0, what="reference_examples")
        _at_least(self.reference_candidate_pool, 1, what="reference_candidate_pool")
        _at_least(self.reference_revision_examples, 0, what="reference_revision_examples")
        _within_unit_interval(self.reference_min_score, what="reference_min_score")
        _within_unit_interval(self.reference_mmr_lambda, what="reference_mmr_lambda")
        _within_unit_interval(self.reference_lexical_min_score,
                              what="reference_lexical_min_score")
        _within_unit_interval(self.reference_embedding_min_score,
                              what="reference_embedding_min_score")


@dataclass(frozen=True, slots=True)
class AgentSet:
    """The translator, the panel that reviews it, and the budgets they operate under."""

    translator_instructions: str
    reviewers: tuple[Reviewer, ...]
    limits: Limits = Limits()
    context: Context = Context()
    leniency: Leniency = Leniency()
    """The file-wide default console-warning tolerance, from ``[leniency]``. Each reviewer
    carries its own resolved :attr:`Reviewer.leniency`; this is kept for reference/reporting."""
    max_revisions: int = 2
    """How many times a unit may be sent back to the review panel before its dispute is
    recorded rather than retried. 0 means translate-and-mechanically-check with no review."""
    max_repairs: int = 2
    """How many times one generation may be mechanically repaired before the unit is rejected."""
    repair_truncated_json: bool = True
    """Whether a translate reply whose JSON envelope was cut short (the model stopped right
    after closing the ``"translation"`` string, before the object's closing brace) may be
    mechanically recovered and, if the repair budget runs out still holding one, KEPT as
    ``TRANSLATED`` rather than discarded (see
    ``docs/feature-requests/json-envelope-truncation-repair.md``). The recovered text always
    goes through the same mechanical checks and review panel as any other candidate -- this
    never bypasses them, and the loop always spends at least one more repair round trying for
    an ordinary reply first. Set ``False`` for a project that would rather lose the unit
    (``REJECTED``, matching the pre-recovery behaviour) than ever ship a translation whose
    provenance is a repaired envelope."""

    def __post_init__(self) -> None:
        _at_least(self.max_revisions, 0, what="max_revisions")
        _at_least(self.max_repairs, 0, what="max_repairs")

    @classmethod
    def load(cls, path: Path,
             substitutions: dict[str, str] | None = None) -> AgentSet:
        """Load the roles and budgets, filling in any ``{placeholder}`` tokens.

        Substitution is what keeps the prompts free of a hard-coded language pair: the same
        file serves any direction, with the pair supplied by the caller.

        Applied by literal replacement rather than :meth:`str.format`, because instructions
        are prose that legitimately contains braces, and ``format`` would raise on them or
        silently consume them as fields.
        """
        if not path.is_file():
            raise AgentConfigError("agent file not found", path=path)
        try:
            data = tomllib.loads(path.read_text("utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise AgentConfigError(f"invalid TOML: {exc}", path=path) from exc

        _reject_unknown(data,
                        {"translate", "reviewer", "limits", "revision", "context", "leniency"},
                        label="the agent file", path=path)
        translate = _reject_unknown(data.get("translate", {}), {"instructions"},
                                    label="[translate]", path=path)
        revision = _reject_unknown(data.get("revision", {}),
                                   {"max_revisions", "max_repairs", "repair_truncated_json"},
                                   label="[revision]", path=path)

        resolved = {**DEFAULT_SUBSTITUTIONS, **(substitutions or {})}

        def fill(text: str, *, where: str) -> str:
            filled = text
            for key, value in resolved.items():
                filled = filled.replace(f"{{{key}}}", value)
            if remaining := _TOKEN.findall(filled):
                raise AgentConfigError(
                    f"{where} refers to unknown placeholder(s) {sorted(set(remaining))}; "
                    f"known placeholders are {sorted(resolved)}", path=path)
            return filled

        instructions = translate.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise AgentConfigError(
                "[translate] needs a non-empty 'instructions' string; without it the model "
                "is asked to translate with no role at all", path=path)
        instructions = fill(instructions, where="[translate] instructions")

        default_leniency = _leniency(data.get("leniency", {}), Leniency(),
                                     label="[leniency]", path=path)
        reviewers = _reviewers(data, fill, default_leniency=default_leniency, path=path)
        if not reviewers:
            raise AgentConfigError(
                "no [[reviewer]] entries: every translation would be accepted unread. Set "
                "revision.max_revisions to 0 if that is genuinely what you want", path=path)

        limits = _limits(data.get("limits", {}), path=path)
        context = _context(data.get("context", {}), path=path)
        max_revisions = _int(revision, "max_revisions", 2, label="revision", path=path)
        max_repairs = _int(revision, "max_repairs", 2, label="revision", path=path)
        repair_truncated_json = _bool(revision, "repair_truncated_json", True,
                                      label="revision", path=path)
        try:
            return cls(translator_instructions=instructions.strip(), reviewers=reviewers,
                       limits=limits, context=context, leniency=default_leniency,
                       max_revisions=max_revisions, max_repairs=max_repairs,
                       repair_truncated_json=repair_truncated_json)
        except ValueError as exc:
            raise AgentConfigError(f"[revision]: {exc}", path=path) from exc


def _leniency(section: object, default: Leniency, *, label: str, path: Path) -> Leniency:
    """Parse a leniency table, falling back to ``default`` for any key it omits."""
    table = _reject_unknown(section, {"window", "max_bad"}, label=label, path=path)
    try:
        return Leniency(
            window=_int(table, "window", default.window, label=label, path=path),
            max_bad=_int(table, "max_bad", default.max_bad, label=label, path=path))
    except ValueError as exc:
        raise AgentConfigError(f"{label}: {exc}", path=path) from exc


def _reviewers(data: dict, fill, *, default_leniency: Leniency,
               path: Path) -> tuple[Reviewer, ...]:
    reviewers: list[Reviewer] = []
    seen: set[str] = set()
    for entry in _tables(data, "reviewer", path=path):
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise AgentConfigError("a [[reviewer]] entry has no 'id'", path=path)
        if identifier in seen:
            raise AgentConfigError(f"duplicate reviewer id {identifier!r}", path=path)
        seen.add(identifier)
        _reject_unknown(entry, {"id", "instructions", "from_rules", "max_tokens", "leniency"},
                        label=f"reviewer {identifier!r}", path=path)
        leniency = (_leniency(entry["leniency"], default_leniency,
                              label=f"reviewer {identifier!r} leniency", path=path)
                    if "leniency" in entry else default_leniency)

        from_rules = entry.get("from_rules", False)
        if not isinstance(from_rules, bool):
            raise AgentConfigError(
                f"reviewer {identifier!r}: from_rules must be a boolean, got "
                f"{type(from_rules).__name__}", path=path)
        text = entry.get("instructions", "")
        if not isinstance(text, str):
            raise AgentConfigError(
                f"reviewer {identifier!r}: instructions must be a string", path=path)
        if from_rules and text.strip():
            raise AgentConfigError(
                f"reviewer {identifier!r} sets both from_rules and instructions; one of them "
                f"would be silently ignored", path=path)
        text = fill(text, where=f"reviewer {identifier!r} instructions")
        if not from_rules and not text.strip():
            raise AgentConfigError(
                f"reviewer {identifier!r} has no instructions and does not set from_rules, so "
                f"it would judge against nothing", path=path)

        max_tokens: int | None = None
        if "max_tokens" in entry:
            max_tokens = _int(entry, "max_tokens", 1, label=f"reviewer {identifier!r}", path=path)
        try:
            reviewers.append(Reviewer(id=identifier, instructions=text.strip(),
                                      from_rules=from_rules, max_tokens=max_tokens,
                                      leniency=leniency))
        except ValueError as exc:
            raise AgentConfigError(f"reviewer {identifier!r}: {exc}", path=path) from exc
    return tuple(reviewers)


def _limits(section: object, *, path: Path) -> Limits:
    table = _reject_unknown(section, {"translate_tokens_per_source_char",
                                      "translate_tokens_floor", "translate_tokens_ceiling",
                                      "review_tokens"},
                            label="[limits]", path=path)
    try:
        return Limits(
            translate_tokens_per_source_char=_int(
                table, "translate_tokens_per_source_char", 8, label="limits", path=path),
            translate_tokens_floor=_int(
                table, "translate_tokens_floor", 256, label="limits", path=path),
            translate_tokens_ceiling=_int(
                table, "translate_tokens_ceiling", 1024, label="limits", path=path),
            review_tokens=_int(table, "review_tokens", 1024, label="limits", path=path),
        )
    except ValueError as exc:
        raise AgentConfigError(f"[limits]: {exc}", path=path) from exc


def _context(section: object, *, path: Path) -> Context:
    table = _reject_unknown(section,
                            {"before_units", "after_units", "include_translations",
                             "context_char_budget", "glossary_terms", "reference_examples",
                             "reference_min_score", "reference_candidate_pool",
                             "reference_mmr_lambda", "reference_lexical_min_score",
                             "reference_embedding_min_score", "reference_revision_examples",
                             "reference_learn_statuses", "anonymous_subject"},
                            label="[context]", path=path)
    try:
        return Context(
            before_units=_int(table, "before_units", 3, label="context", path=path),
            after_units=_int(table, "after_units", 2, label="context", path=path),
            include_translations=_bool(table, "include_translations", True,
                                       label="context", path=path),
            context_char_budget=_int(table, "context_char_budget", 0,
                                     label="context", path=path),
            glossary_terms=_int(table, "glossary_terms", 12, label="context", path=path),
            reference_examples=_int(table, "reference_examples", 0, label="context", path=path),
            reference_min_score=_float(table, "reference_min_score", 0.30,
                                       label="context", path=path),
            reference_candidate_pool=_int(table, "reference_candidate_pool", 40,
                                          label="context", path=path),
            reference_mmr_lambda=_float(table, "reference_mmr_lambda", 0.7,
                                        label="context", path=path),
            reference_lexical_min_score=_float(table, "reference_lexical_min_score", 0.30,
                                               label="context", path=path),
            reference_embedding_min_score=_float(table, "reference_embedding_min_score", 0.55,
                                                 label="context", path=path),
            reference_revision_examples=_int(table, "reference_revision_examples", 0,
                                             label="context", path=path),
            reference_learn_statuses=_statuses(table, "reference_learn_statuses",
                                               label="context", path=path),
            anonymous_subject=_string(table, "anonymous_subject", "they",
                                      label="context", path=path),
        )
    except ValueError as exc:
        raise AgentConfigError(f"[context]: {exc}", path=path) from exc
