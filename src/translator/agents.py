"""Sequential agentic translation and review.

Every role shares one loaded model and differs only by system prompt, because the target
GPU cannot hold more than one model at a time. Roles therefore run strictly in sequence:

    translate -> mechanical check (code) -> review panel -> revise

The mechanical check sits between the model steps on purpose. Placeholder integrity,
untranslated text and display width are decidable in code, so they are settled before any
GPU time is spent asking a model for an opinion, and a translation that fails them is sent
straight back for repair with a precise reason.

Revision is bounded. Each unit gets at most ``max_revisions`` review rounds and each
generation at most ``max_repairs`` mechanical repairs, after which the unit is recorded as
rejected rather than retried forever.

Prompt assembly follows one rule throughout: an optional section is included only when it
has content. An empty glossary, absent context, no matching reference translation, or a
project with no style directives contributes nothing to the prompt -- never a dangling header
with nothing under it, and never a field reporting its own absence.

Surrounding context and retrieved reference examples are selected and rendered by
:mod:`translator.context_packing`, which shows the previous, current and next lines as one
continuous passage and then repeats the line to translate verbatim; see that module for why.
"""
from __future__ import annotations

import logging
import re
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from transunit.adapter import passthrough
from transunit.glossary import Term, relevant_terms
from transunit.reference import ReferenceEntry, Retriever, WritableRetriever
from transunit.units import Status, Unit

from .backend import (
    LlmClient,
    LlmContentError,
    LlmIncompleteJsonError,
    LlmRefusalError,
    LlmTruncationError,
    Message,
)
from .context_packing import ContextPacker, render_context, trigram_similarity
from .memory import TranslationMemory
from .roles import AgentConfigError, AgentSet, Reviewer
from .rules import RuleSet, Violation, blocking, check_mechanical

logger = logging.getLogger(__name__)


def _clip(text: str, limit: int = 120) -> str:
    """A single-line, length-bounded rendering of text for one log line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


DEFAULT_AGENT_FILE = Path(__file__).resolve().parents[2] / "config" / "agents.toml"

SPEAKER = re.compile(r"^(\d+):(.*)$")

_LENGTH_RULES = frozenset({"line_width", "name_respelled"})
"""Blocking rules that mean "imperfect" rather than "unusable".

Kept separate because the right response differs. An unusable translation must not be
shown; one that runs long, or spells a name wrongly, must -- because the only alternative
on offer is the untranslated source, which is worse for the reader by every measure. Both
still block long enough to spend the repair budget on them, which is where they get fixed;
what changes is only what happens when that budget runs out."""

# Placeholders appearing in *surrounding* lines. They belong to those lines' engine
# expressions, never to the one being translated, so they are replaced with a readable
# stand-in before the context is shown. Left in, they are a worked example: a unit whose
# own line has no placeholder is handed neighbours that all use [[0]] as the subject, and
# the model copies the pattern.
_PLACEHOLDER = re.compile(r"\[\[\d+\]\]")

TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["translation"],
    "properties": {
        "translation": {"type": "string"},
    },
}

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["acceptable", "issues"],
    "properties": {
        "acceptable": {"type": "boolean"},
        # Nullable for the same reason as improved_translation: a reviewer that accepts has no
        # issues and, in json_object mode, encodes that empty list as null. The consumer coerces
        # null/absent to an empty list, so forbidding it here only turned an accepting review
        # into a malformed-response error that (via the panel) skipped the other reviewers.
        "issues": {"type": ["array", "null"], "items": {"type": "string"}},
        # Nullable: a reviewer that accepts naturally has nothing to improve and returns null,
        # especially in json_object mode where the shape is prompt-guided. The consumer already
        # treats null/empty/absent identically (``... or None``), so forbidding null here only
        # turned a valid "accepted, no change" verdict into a malformed-response error that
        # aborted the review -- silently bypassing the rest of the panel for that unit.
        "improved_translation": {"type": ["string", "null"]},
    },
}

def ensure_reviewers_have_rules(agent_set: AgentSet, ruleset: RuleSet) -> None:
    """Refuse a ``from_rules`` reviewer with no advisory criteria to judge against.

    The agent file and the rule set are loaded independently, so a compliance reviewer built
    from the rule set's ``[[advisory]]`` entries, with none present, would review against
    nothing (and emit a dangling "these project rules:" header). Raising here keeps the
    inconsistency a configuration error rather than a silently useless reviewer. Exposed as a
    free function so the CLI can run it before contacting the server, without constructing the
    whole harness.
    """
    if (any(reviewer.from_rules for reviewer in agent_set.reviewers)
            and not ruleset.advisory_rules):
        raise AgentConfigError(
            "a from_rules reviewer is configured but the rule set has no [[advisory]] "
            "criteria, so it would review against nothing; add advisory rules or drop the "
            "from_rules reviewer")


PLACEHOLDER_CONTRACT = """
The text contains placeholders written as [[0]], [[1]], and so on. Each stands for a piece
of code the surrounding program will substitute -- a name lookup, a number, a branch.

Rules for placeholders, which override every other instruction:
- Reproduce every placeholder exactly, including the double brackets.
- Use each placeholder exactly once. Never add, drop, merge, or renumber them.
- You may move a placeholder to wherever target-language word order requires.
- Never translate, expand, or guess what a placeholder contains.
""".strip()


@dataclass(frozen=True, slots=True)
class Review:
    role: str
    acceptable: bool
    issues: tuple[str, ...] = ()
    improved: str | None = None
    error: str | None = None
    """Set when this reviewer's reply could not be evaluated. Such a review is an abstention:
    it does not object (so it cannot veto the reviewers that parsed), but it also means the
    panel did not fully run, so the unit is kept for human review rather than verified."""


@dataclass(frozen=True, slots=True)
class Attempt:
    """A rejected translation and everything known about why.

    Carried into the next try so the model revises rather than re-translates. Without the
    text it produced, a model told only "your previous attempt was rejected for these
    reasons" has to guess what it wrote, and frequently reproduces the same defect or breaks
    something that was already right.
    """

    target: str
    issues: tuple[str, ...]
    suggestions: tuple[str, ...] = ()
    """Corrections the reviewers proposed. Advisory: the translator still owns the line,
    because a reviewer's rewrite is not bound by the placeholder contract."""


@dataclass(frozen=True, slots=True)
class Outcome:
    """Result of running one unit through the full harness."""

    unit: Unit
    status: Status
    target: str | None
    violations: tuple[Violation, ...] = ()
    reviews: tuple[Review, ...] = ()
    rounds: int = 0
    error: str | None = None

    def applied(self) -> Unit:
        """The unit updated with this outcome, ready to be written back."""
        return self.applied_to(self.unit)

    def applied_to(self, unit: Unit) -> Unit:
        """This outcome written onto ``unit``.

        Used to share one translation across duplicate units. Only the verdict travels:
        ``unit`` keeps its own id, path, span and placeholder values, so each occurrence
        still injects into its own location with its own engine expressions.
        """
        notes = tuple(f"{v.rule_id}: {v.message}" for v in self.violations)
        notes += tuple(f"{r.role}: {issue}" for r in self.reviews for issue in r.issues)
        notes += tuple(f"{r.role}: could not be evaluated ({r.error})"
                       for r in self.reviews if r.error)
        if self.error:
            notes += (f"error: {self.error}",)
        return replace(unit, status=self.status, target=self.target, notes=notes)


class TranslationAgents:
    """The role definitions and the loop that sequences them.

    Thread-safety: one instance is meant to be shared by all of :func:`run_batch`'s concurrent
    workers -- that is the supported concurrency path. :meth:`process` is otherwise free of
    shared mutable state, and the three places that do share state across workers are each
    guarded: the client's :class:`~translator.backend.UsageStats`, the injected
    :class:`~translator.memory.TranslationMemory` (its ``get``/``record`` lock), and this
    object's per-reviewer leniency window (``_history_lock``). The journal and the progress
    counters are never touched here -- :func:`run_batch` owns them on its own thread. The
    injected reference :class:`~transunit.reference.Retriever` needs no lock: it is immutable
    after construction and only ever read.
    """

    def __init__(self, client: LlmClient, ruleset: RuleSet, glossary: list[Term],
                 *, max_revisions: int | None = None, max_repairs: int | None = None,
                 repair_truncated_json: bool | None = None,
                 memory: TranslationMemory | None = None,
                 reference: Retriever | None = None,
                 sanitize: Callable[[str], str] = passthrough,
                 agent_set: AgentSet | None = None,
                 is_untranslated: Callable[[str, str], bool] | None = None,
                 source_label: str = "SOURCE",
                 target_label: str = "TARGET") -> None:
        self.client = client
        self.ruleset = ruleset
        self.glossary = glossary
        # Optional so the harness stays usable without one. Without it the surrounding-context
        # passage is unchanged -- it is always in the source language -- and the prompt simply
        # carries no list of established neighbour renderings.
        self.memory = memory
        # External reference translations, kept strictly separate from ``memory`` (our own
        # output): this store is read-only guidance retrieved into the prompt. Off unless a
        # caller both supplies it and sets context.reference_examples (or
        # reference_revision_examples) above zero. Immutable after construction, so sharing one
        # across the concurrent workers is safe.
        self.reference = reference
        # Supplied by the carrier adapter, because only it knows what text the carrier can
        # hold. Applied here purely to save a repair round: injection normalises and refuses
        # independently, so a translator run without one is correct, just wasteful.
        self.sanitize = sanitize
        # Decides whether the model handed back its input instead of translating it. Injected
        # because the answer depends on the language pair, and left None by a caller that
        # cannot tell the two apart -- in which case the check is not evaluated rather than
        # reported as passed.
        self.is_untranslated = is_untranslated
        # Language names rather than "source"/"target" where the caller knows them: a pair
        # labelled EN and PL is read by the model as an alignment, which is what the
        # preceding context is for.
        self.source_label = source_label
        self.target_label = target_label
        # Who the agents are, how much room they get, and how much context each prompt
        # carries is configuration, not code: see config/agents.toml.
        self.agent_set = agent_set or AgentSet.load(DEFAULT_AGENT_FILE)
        self.context = self.agent_set.context
        # An explicit argument overrides the config; None defers to the agent file, so the
        # revision budget can live in config with the rest of the tuning.
        self.max_revisions = (max_revisions if max_revisions is not None
                              else self.agent_set.max_revisions)
        self.max_repairs = (max_repairs if max_repairs is not None
                            else self.agent_set.max_repairs)
        self.repair_truncated_json = (repair_truncated_json if repair_truncated_json is not None
                                      else self.agent_set.repair_truncated_json)
        if self.max_revisions < 0 or self.max_repairs < 0:
            raise ValueError("max_revisions and max_repairs must be non-negative")
        # A from_rules reviewer judges against the rule set's advisory criteria; with none, it
        # would judge against nothing. The two files are loaded independently, so the
        # inconsistency is caught here (and, for a clearer error, before the server is
        # contacted -- see cli.cmd_translate, which shares this check).
        ensure_reviewers_have_rules(self.agent_set, self.ruleset)
        self._character_names = {
            term.source: term.target for term in glossary if term.category == "character"
        }
        # Held to their source spelling mechanically, because instruction alone does not do
        # it: told to keep a name as it is, the model still substituted the target language's
        # own version of one in a third of its mentions.
        self._names = tuple(term.source for term in glossary if term.category == "name")
        # Per-reviewer rolling window of recent replies (True == unusable), for the console
        # leniency. One client and one TranslationAgents are shared by every worker in a
        # concurrent run, so this shared state is guarded by a lock.
        self._reply_history: dict[str, deque[bool]] = {}
        self._history_lock = threading.Lock()
        # Selects and budgets positional neighbours + retrieved reference examples into one
        # continuous passage (see context_packing.py). Built once: its parameters are fixed for
        # this instance's lifetime, and it holds no resources of its own to guard. The ±1
        # guarantee is an architectural constant, not a config knob -- discourse continuity
        # requires at least the immediate neighbour on each side, so it is never configured away.
        self._context_packer = ContextPacker(char_budget=self.context.context_char_budget,
                                             mmr_lambda=self.context.reference_mmr_lambda,
                                             guaranteed_adjacent=1)

    def _register_reply(self, reviewer: Reviewer, *, bad: bool) -> bool:
        """Record one reply from ``reviewer`` in its rolling window and, for a bad reply,
        report whether it should reach the console (its window's bad count now exceeds the
        allowance). A good reply always returns ``False``.
        """
        with self._history_lock:
            history = self._reply_history.get(reviewer.id)
            if history is None or history.maxlen != reviewer.leniency.window:
                # Rebuilt if the window size changed (only possible across configs in tests),
                # so the deque's bound always matches the reviewer's current policy.
                history = deque(history or (), maxlen=reviewer.leniency.window)
                self._reply_history[reviewer.id] = history
            history.append(bad)
            return bad and reviewer.leniency.surfaces(sum(history))

    # -- prompt construction -------------------------------------------------

    def _resolve_character(self, raw: str) -> str | None:
        """The established target name whose source is the *longest* match in ``raw``.

        Longest wins so a shorter name that is a prefix of another -- "Ann" against "Anna" --
        does not shadow the exact one when both are in the glossary.
        """
        best_source = ""
        best_target: str | None = None
        for source, target_name in self._character_names.items():
            if source and source in raw and len(source) > len(best_source):
                best_source, best_target = source, target_name
        return best_target

    def _speaker_label(self, unit: Unit) -> str | None:
        """A label for who is speaking, or ``None`` when the source names no one.

        Returning ``None`` rather than a filler like "unknown" keeps the speaker line out of
        the prompt entirely for narration and plain corpora, instead of emitting a header
        that says nothing.
        """
        if not unit.speaker:
            return None
        match = SPEAKER.match(unit.speaker)
        if not match:
            return unit.speaker
        identifier, raw = match.group(1), match.group(2)
        resolved = self._resolve_character(raw)
        if resolved is not None:
            return f"{resolved} (character {identifier})"
        return f"character {identifier}"

    def _system_prompt(self, role_instructions: str, *, placeholders: bool) -> str:
        """Per-role prefix, one variant for lines with placeholders and one without.

        The placeholder contract is omitted entirely when the line has none: explaining that
        ``[[0]]`` stands for a code substitution, then asking the model not to use one, is a
        demonstration it reaches for, and most lines have no placeholders. The style policy
        is likewise omitted when the project declares no directives, rather than left as an
        empty header. Both variants are constant across units, so the server's prefix cache
        holds each.
        """
        sections = [role_instructions]
        if placeholders:
            sections.append(PLACEHOLDER_CONTRACT)
        if self.ruleset.style_directives:
            directives = "\n".join(f"- {d}" for d in self.ruleset.style_directives)
            sections.append(f"Style policy:\n{directives}")
        sections.append(
            f"Keep each output line at or under {self.ruleset.max_line_columns} display "
            f"columns; insert a literal \\n where a break is needed.")
        sections.append("Respond only with the requested JSON object.")
        return "\n\n".join(sections)

    def _context_name(self, unit: Unit) -> str:
        """What to call the character in surrounding lines, for continuity only."""
        match = SPEAKER.match(unit.speaker or "")
        if match:
            resolved = self._resolve_character(match.group(2))
            if resolved is not None:
                return resolved
        return self.context.anonymous_subject

    def learn_reference(self, source: str, target: str | None, status: Status) -> None:
        """Add an accepted translation to a self-populating reference store, if one is in use.

        A no-op unless the caller supplied a *writable* reference and the unit's ``status`` is one
        of ``context.reference_learn_statuses``; otherwise the reference stays read-only. The
        status gate lives here (one place), so the runner can hand every result over without
        deciding what qualifies. Called by the runner from its own thread after a result is
        durable -- the same single-writer path it uses to record to memory -- so the writable
        retriever's lock is all the synchronisation needed.
        """
        if (target and status in self.context.reference_learn_statuses
                and isinstance(self.reference, WritableRetriever)):
            self.reference.add(ReferenceEntry(source=source, target=target))

    def _reference_revision(self, draft: str, stand_in: str) -> str | None:
        """Retrieved reference *targets* similar to a draft, or ``None`` when there are none.

        Keyed on the draft rather than the source, this is the path that uses reference material
        held without its original: matched against what the model just wrote, established target
        renderings can pull phrasing and register into line. ``None`` (so no section) when the
        path is off or nothing clears the floor.
        """
        if self.reference is None or not self.context.reference_revision_examples:
            return None
        hits = self.reference.by_target(
            draft, k=self.context.reference_revision_examples,
            min_score=self.context.reference_min_score)
        if not hits:
            return None
        rendered = "\n".join(f"  {_PLACEHOLDER.sub(stand_in, hit.entry.target)}" for hit in hits)
        return (
            "Established target-language renderings similar to your draft. Where they fit, align "
            "your phrasing, terminology and register with them (they are references, not the "
            "required answer):\n" + rendered)

    def _unit_block(self, unit: Unit) -> str:
        terms = relevant_terms(unit.source, self.glossary, limit=self.context.glossary_terms)
        stand_in = self._context_name(unit)
        # Windowed to the nearest N on each side. before is the *last* N of the preceding
        # lines (the closest ones); after is the *first* N of the following lines. The guard
        # matters: lines[-0:] would return everything, not nothing.
        before = (unit.context_before[-self.context.before_units:]
                  if self.context.before_units else ())
        after = unit.context_after[:self.context.after_units]
        parts: list[str] = []
        speaker = self._speaker_label(unit)
        if speaker:
            parts.append(f"Speaker: {speaker}")
        if terms:
            rendered = "\n".join(f"  {t.source} -> {t.target}  [{t.category}]" for t in terms)
            parts.append(f"Established terminology (use these renderings):\n{rendered}")
        # Retrieval stays exactly where it was (keyed on the unit's source, gated on reference
        # and reference_examples); everything downstream of the hits -- selection, dedup, the
        # continuous surrounding-context passage, the character budget -- is the packer's job.
        # See context_packing.py for why positional context is one continuous blob rather than
        # separate "preceding"/"following" sections.
        hits = (self.reference.by_source(unit.source, k=self.context.reference_examples,
                                         min_score=self.context.reference_min_score)
               if self.reference is not None and self.context.reference_examples else ())
        packed = self._context_packer.pack(
            before=before, current=unit.source, after=after, references=hits, stand_in=stand_in,
            translate_of=(self.memory.get if self.memory and self.context.include_translations
                         else lambda _source: None),
            similarity=trigram_similarity)
        rendered_context = render_context(packed, stand_in=stand_in,
                                          source_label=self.source_label,
                                          target_label=self.target_label)
        if rendered_context:
            parts.append(rendered_context)
        parts.append(self._placeholder_expectation(unit))
        if unit.max_columns is not None:
            # Stated rather than left to be discovered by rejection: the model can choose a
            # shorter phrasing while translating, but cannot compress one it has already
            # committed to without being told to start again.
            #
            # Deliberately suggests no way to break the line. How -- or whether -- text can
            # wrap is the carrier's business, and this side does not know which carrier it is
            # serving. Advising an escape here would either be ignored or written into the
            # output as literal characters.
            parts.append(
                f"Length limit: {unit.max_columns} characters. Past it the text will not fit "
                f"the space it has -- clipped, spilled, or read too fast. Say it more briefly "
                f"-- a shorter, plainer rendering that keeps the meaning beats a fuller one "
                f"that does not fit. Drop filler first.")
        parts.append(f"Line to translate:\n{unit.source}")
        return "\n\n".join(parts)

    @staticmethod
    def _placeholder_expectation(unit: Unit) -> str:
        """State this line's placeholder budget, which the mechanical check enforces.

        A language that omits the subject still needs one in the target, so a line with no
        placeholders can tempt the model to supply the missing subject as ``[[0]]`` --
        reasonable, but the placeholder refers to nothing, and the unit is rejected for
        changing the placeholder set. Naming the expectation per unit costs nothing: it lives
        in the user message, so the cached system prefix is unaffected.
        """
        count = len(unit.placeholders)
        if not count:
            # Stated as an instruction, not a report of an absent field ("this line has none"):
            # the guidance that follows is load-bearing (an anti-hallucination guard for
            # subject-dropping languages), so the whole line stays even though the count is zero
            # -- but it must not read as filling a "field: none" template.
            return ("This line has no placeholders. Do not introduce any. Where the source "
                    "omits a subject, supply a natural pronoun or the speaker's name in the "
                    "target language -- never invent a placeholder for it.")
        expected = ", ".join(f"[[{index}]]" for index in range(count))
        return (f"Placeholders: this line has exactly {count} ({expected}). Reproduce each "
                f"one exactly once, and add no others.")

    # -- roles ---------------------------------------------------------------

    def translate(self, unit: Unit, previous: Attempt | None = None) -> str:
        instructions = self.agent_set.translator_instructions
        user = self._unit_block(unit)
        if previous is not None:
            user += f"\n\nYour previous attempt was:\n{previous.target}"
            # Guard the reasons block on there being reasons: an objection can carry none,
            # and a header with nothing under it is the churn this module avoids everywhere
            # else. (review() also gives an empty objection a synthetic reason, so this is
            # belt and braces.)
            if previous.issues:
                user += (
                    "\n\nIt was sent back for these reasons. Fix exactly these and change "
                    "nothing else that already works:\n"
                    + "\n".join(f"- {issue}" for issue in previous.issues))
            if previous.suggestions:
                user += (
                    "\n\nA reviewer proposed this rewrite. Adopt whatever is right about it, "
                    "but you own the final line: it must keep every placeholder and obey the "
                    "rules above, which the reviewer's version may not.\n"
                    + "\n".join(f"- {s}" for s in previous.suggestions))
            revision_reference = self._reference_revision(previous.target,
                                                          self._context_name(unit))
            if revision_reference:
                user += "\n\n" + revision_reference
        response = self.client.complete_json(
            [Message("system", self._system_prompt(instructions,
                                                     placeholders=bool(unit.placeholders))),
             Message("user", user)],
            TRANSLATION_SCHEMA, role="translate", temperature=0.3,
            # Scaled to this line: the ceiling exists to stop a repetition loop, and a flat
            # one let a loop run before being caught.
            max_tokens=self.agent_set.limits.translate_budget(len(unit.source)))
        # Normalise before the mechanical checks see it: a raw newline has exactly one correct
        # repair, and spending a translate-and-review round to obtain a string we can compute
        # is waste. Checks still run afterwards, so a defect this misses is rejected rather
        # than shipped.
        return self.sanitize(str(response["translation"]))

    def review(self, unit: Unit, target: str, reviewer: Reviewer) -> Review:
        # The bar is stated explicitly because a reviewer asked to find problems will find
        # them: left to its own judgement this panel objected on ~80% of units, almost
        # entirely over word choices it would merely have made differently, which burns the
        # whole revision budget and lands a sound translation in "needs review".
        instructions = (self._rule_instructions() if reviewer.from_rules
                        else reviewer.instructions)
        user = (
            f"{self._unit_block(unit)}\n\n"
            f"Proposed translation:\n{target}\n\n"
            f"Judge only your assigned criteria. Acceptable is the expected answer: object "
            f"only when you can name a definite error a competent editor would have to fix. A "
            f"wording you would merely have chosen differently is not an error, and anything "
            f"outside your assigned criteria is not yours to judge. If you do object, list "
            f"each concrete problem and supply a corrected translation that keeps every "
            f"placeholder intact.")
        response = self.client.complete_json(
            [Message("system", self._system_prompt(instructions,
                                                     placeholders=bool(unit.placeholders))),
             Message("user", user)],
            REVIEW_SCHEMA, role=reviewer.id, temperature=0.0,
            # Each reviewer's own budget: a role that rewrites long units needs more room than
            # one that only checks punctuation.
            max_tokens=self.agent_set.limits.review_budget(reviewer))
        improved = response.get("improved_translation") or None
        acceptable = bool(response["acceptable"])
        # ``or []`` covers a present ``null`` as well as an absent key: the schema now permits
        # null issues (a json_object model's encoding of "no issues"), and ``get(..., [])``
        # would still return that null and crash the comprehension.
        issues = tuple(str(i) for i in (response.get("issues") or []))
        # An objection must carry a reason. If the model rejected the line but named no issue,
        # synthesise one: it keeps the revision prompt from showing a "reasons:" header with
        # nothing under it, and gives the next attempt something to act on.
        if not acceptable and not issues:
            issues = ("marked unacceptable without a specific reason; reconsider the "
                      "translation against the assigned criteria",)
        return Review(
            role=reviewer.id,
            acceptable=acceptable,
            issues=issues,
            improved=improved if improved and improved != target else None,
        )

    def review_panel(self, unit: Unit, target: str) -> list[Review]:
        """Run reviewers in turn, stopping at the first objection.

        Running the whole panel before revising was measured and rejected. The theory was
        that a translator hearing one complaint per round would need three rounds to satisfy
        three reviewers; in practice, once the revision prompt included the attempt being
        criticised, completeness stopped mattering. Over 48 units the two arrangements
        verified exactly the same 15, while the full panel spent 81% more tokens -- a reviewer
        that has seen a concrete previous attempt asks for the right change first time, so the
        later reviewers rarely add a new demand.
        """
        reviews: list[Review] = []
        for reviewer in self.agent_set.reviewers:
            try:
                review = self.review(unit, target, reviewer)
            except LlmContentError as exc:
                # Isolate one reviewer's unusable reply so it cannot abort the panel: the
                # reviewers after it -- including compliance/no-drop -- still run. It is
                # recorded as an abstention (acceptable, so it cannot veto) carrying the error;
                # process() then keeps the unit for human review rather than verifying it on a
                # panel that did not fully run. Not retried: reviewers run at temperature 0, so
                # a retry is deterministic. A bare LlmError (transport) is deliberately NOT
                # caught here -- it means every unit fails alike and must abort the run.
                #
                # Always log it (with context and the model's actual reply, so the failure can
                # be diagnosed without reproducing it); leniency governs only whether it also
                # reaches the console, via the flag the console filter reads. Below the
                # allowance the warning is suppressed on screen but still recorded in full.
                surfaces = self._register_reply(reviewer, bad=True)
                logger.warning(
                    "reviewer %r could not evaluate unit %s (%s -> %s); it abstains and the "
                    "unit is kept for human review. source: %s | model reply: %s",
                    reviewer.id, unit.unit_id, self.source_label, self.target_label,
                    _clip(unit.source), _clip(exc.body or "<empty>", 1000),
                    extra={"leniency_suppress_console": not surfaces})
                reviews.append(Review(role=reviewer.id, acceptable=True,
                                      error=f"reply could not be evaluated: {exc}"))
                continue
            self._register_reply(reviewer, bad=False)
            reviews.append(review)
            if not review.acceptable:
                break
        return reviews

    def _rule_instructions(self) -> str:
        criteria = "\n".join(f"- {rule_id}: {description}"
                             for rule_id, description in self.ruleset.advisory_rules)
        return (
            "You are a compliance checker. Judge the translation against exactly these project "
            f"rules and nothing else:\n{criteria}")

    # -- orchestration -------------------------------------------------------

    def _generate(self, unit: Unit,
                  feedback: Attempt | None) -> tuple[str, list[Violation], bool]:
        """Translate and mechanically repair until the hard rules pass or budget runs out.

        ``feedback`` carries the review objections from the previous round, if any. Each
        mechanical repair then rebuilds it around the attempt that just failed, so the model
        always sees the exact text it is being asked to fix rather than the text from a round
        ago.

        The third return value is ``True`` when the returned translation came from a recovered
        JSON envelope (:class:`LlmIncompleteJsonError`) rather than an ordinary reply -- see
        ``docs/feature-requests/json-envelope-truncation-repair.md``. The recovered value is
        never trusted at face value: it is mechanically checked exactly like any other candidate,
        and this loop spends at least one more repair round trying to get an ordinary reply
        before falling back to it, only on the round that exhausts the budget. ``process()`` uses
        the flag to withhold ``VERIFIED`` from a candidate this loop was never able to confirm
        independently, even if the review panel accepts it.
        """
        # Enforce presence of the same terms the model is shown (context.glossary_terms), so
        # "shown" and "required" are one number, not two that can silently disagree.
        required = tuple(t.target for t in relevant_terms(
            unit.source, self.glossary, limit=self.context.glossary_terms))
        # Held constant across repair rounds within this call -- only the outer revision round's
        # objections belong in every attempt's issues; a repair round's own violations do not
        # accumulate into the next repair round, only into that round's own attempt.
        carried = feedback.issues if feedback else ()
        suggestions = feedback.suggestions if feedback else ()
        attempt = feedback
        target = ""
        violations: list[Violation] = []
        for round_index in range(self.max_repairs + 1):
            last_round = round_index == self.max_repairs
            try:
                target = self.translate(unit, attempt)
            except (LlmRefusalError, LlmTruncationError):
                # Both stay terminal: a refusal re-asked is wasted budget and reads as
                # jailbreaking, and a ceiling hit is a property of this prompt, so another
                # attempt loops the same way. Explicit re-raise so the broader
                # LlmIncompleteJsonError/LlmContentError branches below never see them.
                raise
            except LlmIncompleteJsonError as exc:
                if not self.repair_truncated_json:
                    # A project that wants strictness: never accept a recovered envelope, no
                    # matter what -- treated exactly like an ordinary unusable reply below.
                    if last_round:
                        raise
                    attempt = Attempt(
                        target=target,
                        issues=(*carried,
                               "your previous reply could not be read as valid JSON -- reply "
                               "with the translation as one valid JSON object and nothing "
                               "else."),
                        suggestions=suggestions)
                    continue
                # Envelope recovered, content unverified: check it like any other candidate,
                # but never accept it as-is while budget remains -- only the round that
                # exhausts the budget returns it.
                target = self.sanitize(str(exc.recovered["translation"]))
                violations = check_mechanical(
                    unit.source, target, self.ruleset,
                    expected_placeholders=len(unit.placeholders), required_terms=required,
                    max_columns=unit.max_columns, is_untranslated=self.is_untranslated,
                    names=self._names)
                if last_round:
                    return target, violations, True
                attempt = Attempt(
                    target=target,
                    issues=(*carried,
                           "your previous reply's JSON envelope was cut off before the "
                           "closing brace -- repeat the FULL translation as one complete, "
                           "valid JSON object, with nothing after it."),
                    suggestions=suggestions)
                continue
            except LlmContentError:
                # Unparseable/garbled: nothing was produced to check, only a pointed re-ask.
                # If the budget runs out having never produced a checkable candidate, this
                # re-raises exactly as an uncaught LlmContentError does today, so process()'s
                # outer handler still rejects the unit -- unchanged terminal status, just
                # after the repair budget was actually spent trying first.
                if last_round:
                    raise
                attempt = Attempt(
                    target=target,
                    issues=(*carried,
                           "your previous reply could not be read as valid JSON -- reply "
                           "with the translation as one valid JSON object and nothing "
                           "else."),
                    suggestions=suggestions)
                continue
            violations = check_mechanical(
                unit.source, target, self.ruleset,
                expected_placeholders=len(unit.placeholders), required_terms=required,
                max_columns=unit.max_columns, is_untranslated=self.is_untranslated,
                names=self._names)
            hard = blocking(violations)
            if not hard:
                return target, violations, False
            # Review objections stay in view: a mechanical repair must not undo the wording
            # change the reviewers asked for on the previous round.
            attempt = Attempt(target=target, issues=carried + tuple(v.message for v in hard),
                              suggestions=suggestions)
        return target, violations, False

    def process(self, unit: Unit) -> Outcome:
        """Run one unit through translation, mechanical checks, review and revision.

        Returns an Outcome only for verdicts about *this unit*. Infrastructure failures
        (transport loss, server errors, exhausted retries) propagate as LlmError instead,
        because the caller journals every returned Outcome as a completed result and skips
        journalled ids on resume: turning "the server was down" into a terminal REJECTED would
        burn the unit permanently on a run that still exits 0. The client has already
        exhausted its own retries by the time such an error escapes, so the correct response
        is to stop and let the operator fix the server and resume.
        """
        if unit.status is Status.SKIPPED:
            return Outcome(unit=unit, status=Status.SKIPPED, target=None)

        if not unit.source.strip():
            # A blank payload has nothing to translate, and asking the model to translate an
            # empty line is not harmless: with no source to work from it renders the prompt's
            # own scaffolding into the target, an injectable non-translation the mechanical
            # checks pass (it is non-empty and carries no placeholder) and only the review
            # panel might catch. A blank source also lets an empty target through, since the
            # non-empty rule cannot fire when the source is itself blank. Skipping it is both
            # correct -- there is no text to carry over -- and the safe default.
            return Outcome(unit=unit, status=Status.SKIPPED, target=None)

        feedback: Attempt | None = None
        all_reviews: list[Review] = []
        try:
            for round_index in range(self.max_revisions + 1):
                target, violations, from_repair = self._generate(unit, feedback)
                hard = blocking(violations)
                if hard:
                    # Too long is not the same kind of failure as wrong. Every other blocking
                    # rule means the text cannot be shown at all; this one means it reads a
                    # little fast or spells a name off. Rejecting it discards the translation
                    # and the caller falls back to displaying the untranslated source -- so
                    # for length alone, keep it and flag it.
                    if all(v.rule_id in _LENGTH_RULES for v in hard):
                        return Outcome(unit=unit, status=Status.TRANSLATED, target=target,
                                       violations=tuple(violations),
                                       reviews=tuple(all_reviews), rounds=round_index + 1,
                                       error="over the length budget after repairs; kept "
                                             "because the alternative is showing the "
                                             "untranslated source")
                    return Outcome(unit=unit, status=Status.REJECTED, target=target,
                                   violations=tuple(violations), reviews=tuple(all_reviews),
                                   rounds=round_index + 1,
                                   error="mechanical rules still violated after repairs")
                reviews = self.review_panel(unit, target)
                all_reviews.extend(reviews)
                objections = [r for r in reviews if not r.acceptable]
                if not objections:
                    unevaluated = [r for r in reviews if r.error]
                    if unevaluated:
                        # Every reviewer that could be read accepted, but at least one could
                        # not be evaluated, so the panel did not fully run. Keep the
                        # mechanically-sound text but mark it for human review rather than
                        # verifying it on a partial panel.
                        return Outcome(unit=unit, status=Status.TRANSLATED, target=target,
                                       violations=tuple(violations),
                                       reviews=tuple(all_reviews), rounds=round_index + 1,
                                       error="review incomplete: "
                                             + "; ".join(f"{r.role}: {r.error}"
                                                         for r in unevaluated))
                    if from_repair:
                        # The panel accepted it, but it was never independently confirmed --
                        # its JSON envelope had to be mechanically recovered, and the repair
                        # budget ran out before an ordinary reply could replace it. Keep the
                        # text (the alternative is losing it) but withhold VERIFIED, the same
                        # precedent as "review incomplete" and "over the length budget" above.
                        return Outcome(unit=unit, status=Status.TRANSLATED, target=target,
                                       violations=tuple(violations),
                                       reviews=tuple(all_reviews), rounds=round_index + 1,
                                       error="translation recovered from a truncated JSON "
                                             "envelope; kept but not independently confirmed, "
                                             "flagged for review")
                    return Outcome(unit=unit, status=Status.VERIFIED, target=target,
                                   violations=tuple(violations), reviews=tuple(all_reviews),
                                   rounds=round_index + 1)

                if round_index == self.max_revisions:
                    # Budget exhausted: keep the text but mark it for human attention rather
                    # than silently shipping a disputed line as verified.
                    return Outcome(unit=unit, status=Status.TRANSLATED, target=target,
                                   violations=tuple(violations), reviews=tuple(all_reviews),
                                   rounds=round_index + 1,
                                   error="review objections unresolved within budget")

                # Every objecting reviewer is heard in the next attempt, tagged by role so the
                # model can tell a spelling complaint from a fidelity one, along with the text
                # they were objecting to and any rewrite they proposed.
                feedback = Attempt(
                    target=target,
                    issues=tuple(f"{review.role}: {issue}"
                                 for review in objections for issue in review.issues),
                    suggestions=tuple(review.improved for review in objections
                                      if review.improved),
                )
        except LlmRefusalError as exc:
            return Outcome(unit=unit, status=Status.REJECTED, target=None,
                           reviews=tuple(all_reviews), error=f"model refused: {exc}")
        except LlmContentError as exc:
            # Unusable output on the *translate* path for this input only -- a repetition loop,
            # unparseable JSON, a missing field -- so no sound translation was produced this
            # round. Terminal for this unit, but the next is unaffected, so record it and let
            # the run continue. A reviewer's unusable reply never reaches here: review_panel
            # isolates it as an abstention. Bare LlmError (transport, server) is deliberately
            # not caught -- it means every remaining unit would fail identically.
            return Outcome(unit=unit, status=Status.REJECTED, target=None,
                           reviews=tuple(all_reviews), error=str(exc))

        raise AssertionError("unreachable: revision loop always returns")
