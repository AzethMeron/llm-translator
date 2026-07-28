"""Command-line entry point for the translator.

Reads the transunit intermediate format and writes it back with translations filled in. It
knows nothing about where the text came from: subtitles, game files, or documents belong to
a *carrier adapter*, and this side never touches them. The adapter is named on the command
line and resolved by import, which is the only place a specific carrier is mentioned -- and
it is optional, so the translator also runs against a plain corpus.

Which model is served is likewise not this side's concern: ``--backend`` selects how a
structured request is shaped for the model in use, and ``--backend auto`` picks that from the
model id. Switching models never edits code here.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from transunit.adapter import AdapterError, load_adapter, passthrough
from transunit.glossary import GlossaryError, read_glossary
from transunit.language import (
    LanguageError,
    LanguageProfile,
    UntranslatedDetector,
    find_profile,
    lexical_detector,
    load_profiles,
    script_detector,
)
from transunit.reference import (
    GrowableLexicalRetriever,
    LexicalRetriever,
    ReferenceEntry,
    ReferenceError,
    Retriever,
    read_reference,
    reference_from_units,
)
from transunit.units import CatalogError, read_journal

from .agents import TranslationAgents, ensure_reviewers_have_rules
# Importing this module does not import numpy (it is lazy inside the retrievers), so the CLI's
# import path stays numpy-free; numpy loads only if an EmbeddingRetriever/HybridRetriever is
# actually built.
from .retrieval import (
    DEFAULT_EMBEDDING_MIN_SCORE,
    DEFAULT_RERANK_MIN_SCORE,
    EmbeddingError,
    EmbeddingRetriever,
    HybridRetriever,
    RerankError,
)
from .backend import (
    BackendError,
    LlmClient,
    LlmError,
    ServerConfig,
    available_backends,
    resolve_backend,
)
from .memory import TranslationMemory
from .roles import AgentConfigError, AgentSet
from .rules import RuleConfigError, RuleSet
from .runner import RunnerError, completed_ids, pending_units, run_batch

logger = logging.getLogger("translator.cli")


class UsageError(Exception):
    """The command cannot run as invoked."""


# Enough common ISO 639-1 names that the prompts read as a real language pair even with no
# profiles file. A code not here falls back to itself, which still works, just less legibly.
COMMON_LANGUAGE_NAMES = {
    "en": "English", "pl": "Polish", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "sv": "Swedish", "cs": "Czech",
    "ru": "Russian", "uk": "Ukrainian", "el": "Greek", "tr": "Turkish", "ar": "Arabic",
    "he": "Hebrew", "fa": "Persian", "hi": "Hindi", "th": "Thai", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean",
}


def _language_name(code: str, profiles) -> str:
    """The display name for a code: the profile's name if present, else a known name, else
    the code itself."""
    if profiles:
        try:
            return find_profile(profiles, code).name
        except LanguageError:
            pass
    return COMMON_LANGUAGE_NAMES.get(code, code)


def _language_wiring(args: argparse.Namespace):
    """Resolve language names and the untranslated detector for the run.

    Returns ``(source_name, target_name, source_label, target_label, is_untranslated)``.

    Three ways to get the untranslated-echo check, in order of precedence:

    * ``--source-script NAME`` -- script-based, for a different-script pair (Japanese, Chinese,
      Russian, Greek, Arabic, ...); needs no profiles file.
    * a ``--languages`` profiles file -- lexical, for a same-script pair (English/Polish/German).
    * neither -- the check is disabled, because a check that cannot be evaluated must not be
      reported as passed.

    The language *names* reach the prompts regardless, from the profiles, a built-in table, or
    the code itself, so the model is asked for the right language even when detection is off.
    """
    source_code, target_code = args.source_language, args.target_language
    if source_code == target_code:
        raise UsageError(
            f"source and target are both {source_code!r}; there is nothing to translate")

    profiles: tuple[LanguageProfile, ...] = ()
    if args.languages is not None and Path(args.languages).is_file():
        profiles = load_profiles(args.languages)

    source_name = _language_name(source_code, profiles)
    target_name = _language_name(target_code, profiles)
    labels = (source_code.upper(), target_code.upper())

    detector: UntranslatedDetector | None
    if args.source_script:
        detector = script_detector(args.source_script)
    elif profiles:
        # Validates the source code against the profiles: a lexical run whose source has no
        # profile is a misconfiguration, caught here rather than silently never firing.
        detector = lexical_detector(profiles, source_code)
    else:
        detector = None
    return (source_name, target_name, labels[0], labels[1], detector)


def _load_reference(paths: list[Path] | None, journal: Path, context, *,
                    embedding_url: str = "", embedding_model: str = "local",
                    hybrid: bool = False, rerank_url: str = "",
                    rerank_model: str = "local") -> Retriever | None:
    """Load the reference corpora and build a retriever, or ``None`` when retrieval is off.

    Retrieval needs two things together: material to retrieve -- a ``--reference`` corpus, or
    ``reference_learn`` accumulating this run's own output -- *and* a retrieval direction, a
    non-zero ``reference_examples`` (source-keyed) or ``reference_revision_examples``
    (target-keyed). Either half without the other is almost certainly a mistake (a corpus never
    read, a knob that never fires, or learning with nowhere to surface), so each is warned about
    rather than passing silently. Only the directions the config asks for are indexed.

    When ``reference_learn`` is on the retriever is *growable* and is seeded from the resumed
    journal as well as any ``--reference`` files, so a restart keeps what earlier runs learned.
    """
    # These two are pure flag-shape errors, independent of whether retrieval is even configured
    # on -- catching them here, unconditionally, means a nonsensical combination is reported for
    # what it is rather than silently doing nothing (if hybrid is off) or silently ignoring
    # --rerank-url (if hybrid is on but not checked).
    if hybrid and not embedding_url:
        raise UsageError(
            "--hybrid needs --embedding-url: it fuses the lexical retriever with the embedding "
            "retriever, so an embedding endpoint is required.")
    if rerank_url and not hybrid:
        raise UsageError("--rerank-url needs --hybrid: reranking is a stage over the hybrid "
                         "retriever's fused candidate pool.")

    wants_source = context.reference_examples > 0
    wants_target = context.reference_revision_examples > 0
    learn = bool(context.reference_learn_statuses)

    # With no retrieval direction, nothing would ever be shown, so do not even read the corpora
    # -- a large one is wasted work, and a missing one is moot when it cannot be used. Warn if
    # material or learning was configured, since that is the misconfiguration.
    if not (wants_source or wants_target):
        if paths or learn:
            logger.warning(
                "reference material or learning is configured but retrieval is off "
                "(context.reference_examples and reference_revision_examples are both 0), so "
                "nothing will be retrieved; set one above zero")
        return None

    entries: list[ReferenceEntry] = []
    for path in paths or ():  # read_reference raises ReferenceError on a malformed file
        entries.extend(read_reference(path))
    if learn:  # seed from earlier runs' results, filtered to the same statuses learning uses
        entries.extend(reference_from_units(read_journal(journal),
                                            context.reference_learn_statuses))

    if not entries and not learn:
        logger.warning(
            "reference retrieval is enabled but no --reference corpus was supplied and "
            "reference_learn_statuses is empty, so nothing will be retrieved")
        return None

    retriever: Retriever
    if embedding_url:
        # Both the plain semantic retriever and the hybrid one match by meaning via an embedding
        # endpoint, and both are read-only -- learning is the lexical retriever's job -- so
        # combining either with self-population is a misconfiguration, refused rather than
        # half-honoured.
        if learn:
            what = "--hybrid" if hybrid else "--embedding-url"
            raise UsageError(
                f"{what} cannot be combined with reference_learn_statuses: the "
                f"{'hybrid' if hybrid else 'embedding'} retriever is read-only (self-population "
                f"is the lexical retriever's). Drop one.")
        # reference_min_score is applied to a different scale in each mode, so the advice has to
        # name the right one -- and only one. Telling a *reranked* run to raise its floor to the
        # embedding scale would be actively wrong: a reranker's sigmoid runs lower than an
        # embedding cosine (a correct cross-script match measures ~0.44), so following that
        # advice would discard good matches.
        if rerank_url:
            # Final score = sigmoid of the cross-encoder's logit: a real, absolute relevance
            # judgement, and the only stage here that produces one.
            if context.reference_min_score < DEFAULT_RERANK_MIN_SCORE:
                logger.warning(
                    "reference_min_score=%.2f is below the reranker-calibrated floor; with "
                    "--rerank-url the final score is the reranker's own relevance judgement, so "
                    "this floor admits almost everything. Consider ~%.2f.",
                    context.reference_min_score, DEFAULT_RERANK_MIN_SCORE)
        elif hybrid:
            # Final score = fused rank agreement (1.0 = both arms rank it first, 0.5 = one arm's
            # top hit only), NOT an embedding cosine -- the per-arm floors did the absolute
            # gating already, so a low final floor here is sound rather than a misconfiguration.
            # Nothing to warn about; the scale is documented in config/README.md.
            pass
        elif context.reference_min_score < 0.4:
            logger.warning(
                "reference_min_score=%.2f is calibrated for the lexical retriever; embedding "
                "cosines cluster higher, so this floor admits almost everything. Consider ~%.2f "
                "for the embedding retriever.",
                context.reference_min_score, DEFAULT_EMBEDDING_MIN_SCORE)
        if hybrid and context.reference_candidate_pool < max(context.reference_examples,
                                                              context.reference_revision_examples):
            logger.warning(
                "reference_candidate_pool=%d is smaller than the largest requested example "
                "count (%d); the hybrid retriever cannot return more examples than it retrieves "
                "as candidates per arm. Raise reference_candidate_pool.",
                context.reference_candidate_pool,
                max(context.reference_examples, context.reference_revision_examples))
        # Constructing either embeds the whole corpus, so a down endpoint or missing numpy fails
        # here, before the server is contacted, with a clear EmbeddingError.
        if hybrid:
            retriever = HybridRetriever(
                entries, embedding_base_url=embedding_url, embedding_model=embedding_model,
                rerank_base_url=rerank_url, rerank_model=rerank_model,
                index_source=wants_source, index_target=wants_target,
                candidate_pool=context.reference_candidate_pool,
                mmr_lambda=context.reference_mmr_lambda,
                lexical_min_score=context.reference_lexical_min_score,
                embedding_min_score=context.reference_embedding_min_score)
            kind = (f"hybrid (lexical + embedding via {embedding_url}"
                   + (f" + rerank via {rerank_url}" if rerank_url else "") + ")")
        else:
            retriever = EmbeddingRetriever(entries, base_url=embedding_url, model=embedding_model,
                                           index_source=wants_source, index_target=wants_target)
            kind = f"embedding via {embedding_url}"
    elif learn:
        retriever = GrowableLexicalRetriever(entries, index_source=wants_source,
                                             index_target=wants_target)
        kind = "lexical, self-populating"
    else:
        retriever = LexicalRetriever(entries, index_source=wants_source,
                                     index_target=wants_target)
        kind = "lexical"

    with_source = retriever.source_entries
    detail = (f"{len(retriever):,} seed entries ({with_source:,} with source, "
              f"{len(retriever) - with_source:,} target-only)")
    if learn:
        print(f"reference [{kind}]: {detail}; learning from accepted output")
    else:
        print(f"reference [{kind}]: {detail} from {len(paths or ())} file(s)")
    if wants_source and with_source == 0 and not learn:
        logger.warning(
            "context.reference_examples=%d but no reference entry has a source, so source-keyed "
            "retrieval will never match; only reference_revision_examples (target-keyed) can use "
            "this corpus", context.reference_examples)
    return retriever


def cmd_translate(args: argparse.Namespace) -> int:
    for flag, value in (("--max-revisions", args.max_revisions),
                        ("--max-repairs", args.max_repairs)):
        if value is not None and value < 0:
            raise UsageError(f"{flag} must be >= 0, got {value}")
    # Every check and file read below needs no server. Doing them first means a bad argument,
    # a malformed catalogue, or an inconsistent config is reported for what it is, instead of
    # being masked by a "no inference server" error when none happens to be running.
    if args.limit is not None and args.limit < 1:
        raise UsageError(f"--limit must be >= 1, got {args.limit}")
    if args.context_window < 0:
        raise UsageError(f"--context-window must be >= 0, got {args.context_window}")
    ruleset = RuleSet.load(args.rules)
    source_name, target_name, source_label, target_label, is_untranslated = \
        _language_wiring(args)
    # The named languages reach the prompts here. Without them the agent file's placeholders
    # fall back to "the source language", and a run never actually asks for the language it is
    # supposed to produce.
    agent_set = AgentSet.load(args.agents, {"source_language": source_name,
                                            "target_language": target_name})
    # A from_rules reviewer with no advisory criteria would review against nothing; catch that
    # here rather than after the server is up. Shared with TranslationAgents so the rule lives
    # in one place.
    ensure_reviewers_have_rules(agent_set, ruleset)
    glossary = read_glossary(args.glossary)
    config = ServerConfig(base_url=args.base_url, model=args.model,
                          enable_reasoning=args.enable_reasoning,
                          context_window=args.context_window)
    backend = resolve_backend(args.backend, args.model)
    # Only the adapter knows what text this carrier can hold, so its normaliser is passed in
    # rather than imported. Resolved here, before the server is contacted, so a mistyped
    # adapter name does not cost the wait for a model to load before it is reported. Without
    # one the run is still correct -- rendering normalises and refuses independently -- just
    # wasteful, since the model spends repair rounds on defects code could fix.
    sanitize = load_adapter(args.adapter).sanitize_payload if args.adapter else passthrough
    # External reference translations, read before the server is contacted so a missing or
    # malformed corpus is reported precisely. Kept separate from the run's own memory; None when
    # no corpus is supplied or retrieval is left off, in which case the prompt is unchanged.
    reference = _load_reference(args.reference, args.journal, agent_set.context,
                                embedding_url=args.embedding_url,
                                embedding_model=args.embedding_model,
                                hybrid=args.hybrid, rerank_url=args.rerank_url,
                                rerank_model=args.rerank_model)

    # The retriever may own a resource (the embedding retriever holds an HTTP client), so the CLI
    # that built it closes it on every exit path -- normal, "nothing to do", or an error.
    try:
        # The catalogue and journal are local files. Read and validate them before contacting the
        # server, so a malformed catalogue or an already-finished job is reported precisely, and a
        # completed run resumes to "nothing to do" without needing a server up at all.
        units = pending_units(args.catalog, args.journal)
        # What earlier runs finished, so a resumed run reports its position in the whole job
        # instead of appearing to start over.
        already_done = len(completed_ids(args.journal))
        if args.limit is not None:
            units = units[:args.limit]
        if not units:
            print("nothing to do: no pending units outside the journal")
            return 0

        print(f"model {args.model!r} via backend {backend.name!r}")
        with LlmClient(config, backend=backend) as client:
            if not client.health():
                raise UsageError(
                    f"no inference server responding at {args.base_url}; start one with "
                    f"tools/serve_model.sh")
            # Seeded from the journal so a resumed run keeps the wording its earlier half
            # established, rather than starting the scene over from the source.
            memory = TranslationMemory.from_units(read_journal(args.journal))
            agents = TranslationAgents(client, ruleset, glossary,
                                       max_revisions=args.max_revisions,
                                       max_repairs=args.max_repairs,
                                       memory=memory, reference=reference, sanitize=sanitize,
                                       agent_set=agent_set,
                                       is_untranslated=is_untranslated,
                                       source_label=source_label,
                                       target_label=target_label)
            print(f"{len(units):,} units queued -> {args.journal} "
                  f"({args.concurrency} at a time)")
            progress = run_batch(agents, units, args.journal,
                                 concurrency=args.concurrency,
                                 already_done=already_done,
                                 on_progress=lambda p: print(f"  {p.summary()}", flush=True))

        print(f"\n{progress.summary()}")
        print(f"tokens: {client.stats.completion_tokens:,} completion "
              f"({client.stats.completion_tokens_per_second:.0f}/s), "
              f"{client.stats.prompt_tokens:,} prompt "
              f"(peak {client.stats.peak_prompt_tokens:,}/call), "
              f"{client.stats.retries:,} retries, {client.stats.refusals:,} refusals")
        return 0
    finally:
        if reference is not None and hasattr(reference, "close"):
            reference.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="translator",
        description="Translate pending units from a catalogue")
    parser.set_defaults(handler=cmd_translate)
    parser.add_argument("-c", "--catalog", type=Path, default=Path("work/units.jsonl"))
    parser.add_argument("-j", "--journal", type=Path, default=Path("work/journal.jsonl"))
    parser.add_argument("-g", "--glossary", type=Path, default=Path("work/glossary.jsonl"))
    parser.add_argument("-r", "--rules", type=Path,
                        default=Path("config/translation_rules.toml"))
    parser.add_argument("-a", "--agents", type=Path, default=Path("config/agents.toml"),
                        help="who the agents are and what each one reviews")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="local",
                        help="model id the server exposes (default: %(default)s)")
    parser.add_argument("--backend", default="auto",
                        help="how to shape structured requests for the served model: "
                             f"{', '.join(available_backends())}, or 'auto' to pick from the "
                             f"model id (default: %(default)s)")
    parser.add_argument("--context-window", type=int, default=0,
                        help="the server's context size in tokens (llama.cpp -c / --ctx-size, "
                             "per slot). Enables a one-time warning when an assembled prompt "
                             "plus its output budget nears the window, before the server "
                             "rejects it. 0 (default) disables the warning")
    parser.add_argument("--limit", type=int, help="stop after N units (for trials)")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="units translated at once against the same server")
    parser.add_argument("--max-revisions", type=int, default=None,
                        help="review rounds per unit; overrides revision.max_revisions in "
                             "the agent config")
    parser.add_argument("--max-repairs", type=int, default=None,
                        help="mechanical repair attempts per generation; overrides "
                             "revision.max_repairs in the agent config")
    parser.add_argument("--source-language", default="en",
                        help="language code of the source (default: %(default)s)")
    parser.add_argument("--target-language", default="pl",
                        help="language code to translate into (default: %(default)s)")
    parser.add_argument("--languages", type=Path, default=Path("config/languages.toml"),
                        help="lexical language profiles for a same-script pair; enables the "
                             "untranslated-echo check (default: %(default)s). Pass a missing "
                             "path to disable it")
    parser.add_argument("--source-script",
                        help="for a different-script pair, the source script for the "
                             "untranslated check (e.g. japanese, han, cyrillic, greek, "
                             "arabic); overrides --languages for detection")
    parser.add_argument("--adapter", default="",
                        help="carrier adapter supplying payload normalisation; empty runs "
                             "against a plain corpus (default: empty)")
    parser.add_argument("--reference", type=Path, action="append",
                        help="reference-translation corpus as JSONL (a prior journal, or "
                             "{\"source\",\"target\"} lines; omit source for target-only). May "
                             "be repeated. Retrieved into prompts only when the agent config "
                             "sets context.reference_examples (or reference_revision_examples) "
                             "above 0; off by default")
    parser.add_argument("--embedding-url", default="",
                        help="use the OPTIONAL semantic (embedding) reference retriever instead "
                             "of the lexical default: an OpenAI-compatible /v1 base URL exposing "
                             "/v1/embeddings (e.g. http://127.0.0.1:8081/v1 -- see "
                             "tools/serve_embeddings.sh). Needs numpy. Matches by meaning, not "
                             "characters; read-only (no self-population). Empty = lexical default")
    parser.add_argument("--embedding-model", default="local",
                        help="model id the embedding endpoint exposes (default: %(default)s)")
    parser.add_argument("--hybrid", action="store_true",
                        help="fuse the lexical and embedding retrievers with Reciprocal Rank "
                             "Fusion instead of using the embedding retriever alone -- combines "
                             "exact-term matching with semantic matching. Needs --embedding-url. "
                             "Read-only, like the plain embedding retriever. Off by default. "
                             "MEASURE BEFORE ADOPTING (tools/eval_retrieval.sh): equal-weight "
                             "fusion blends its arms rather than taking the better one, so on "
                             "two real corpora it scored BELOW the embedding arm alone unless "
                             "--rerank-url was also given")
    parser.add_argument("--rerank-url", default="",
                        help="add a cross-encoder reranking stage over the hybrid retriever's "
                             "fused candidate pool: an OpenAI-compatible /v1 base URL exposing "
                             "/v1/rerank (e.g. http://127.0.0.1:8082/v1 -- see "
                             "tools/serve_reranker.sh). Needs --hybrid. Empty = no reranking "
                             "(default: empty)")
    parser.add_argument("--rerank-model", default="local",
                        help="model id the rerank endpoint exposes (default: %(default)s)")
    parser.add_argument("--enable-reasoning", action="store_true",
                        help="let a reasoning model (e.g. Qwen3) emit its chain-of-thought. "
                             "DISCOURAGED: the hidden reasoning tokens are billed against the "
                             "per-call token budget and truncate the answer, which is then "
                             "rejected. Off by default; if you enable it, raise the "
                             "translate/review token budgets in the agent config to suit")
    parser.add_argument("--log-file", type=Path, default=Path("work/translator.log"),
                        help="full, unfiltered log: every engine warning (each reviewer reply "
                             "that could not be evaluated, with its context and the model's "
                             "response) and any fatal error that aborts the run. Leniency "
                             "(config) only filters the stderr copy, never this file "
                             "(default: %(default)s)")
    parser.add_argument("--no-log-file", action="store_true",
                        help="do not write the log file; keep only the leniency-filtered stderr "
                             "warnings")
    return parser


class _LeniencyConsoleFilter(logging.Filter):
    """Keeps a record off the console while leaving it in every other sink (the log file).

    Attached only to the stderr handler, it drops a record flagged either ``console_shown``
    (already printed to the console another way -- a fatal error the CLI prints itself) or
    ``leniency_suppress_console`` (a reviewer's leniency asked to keep it off-screen). Both
    still reach the unfiltered log file; any other record passes untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not (getattr(record, "leniency_suppress_console", False)
                    or getattr(record, "console_shown", False))


def _configure_logging(log_file: Path | None) -> None:
    """Wire the engine's warnings to two sinks: a full log file that keeps *every* record, and
    a leniency-filtered stderr that shows only what crosses the warning threshold.

    Configured on the ``translator`` logger (not root) with propagation off, so repeated calls
    in one process -- and a host application's own logging -- do not collide or duplicate. Level
    is WARNING: the HTTP client logs a line per request at INFO, and a long run makes thousands.
    """
    log = logging.getLogger("translator")
    log.setLevel(logging.WARNING)
    log.propagate = False
    for handler in list(log.handlers):  # idempotent across repeated calls; release old files
        log.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    console.addFilter(_LeniencyConsoleFilter())
    log.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # delay=True: the file is opened only when the first warning is written, so a run (or a
        # test) that emits none leaves no empty log behind.
        full = logging.FileHandler(log_file, encoding="utf-8", delay=True)
        full.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        log.addHandler(full)  # unfiltered on purpose: the log keeps every warning


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(None if args.no_log_file else args.log_file)
    try:
        return args.handler(args)
    except (UsageError, CatalogError, GlossaryError, RuleConfigError, LanguageError,
            RunnerError, LlmError, AdapterError, AgentConfigError, BackendError,
            ReferenceError, EmbeddingError, RerankError, FileNotFoundError) as exc:
        # Printed for the operator, and logged so the failure is in the --log-file too: a run
        # that dies on a bad config, corpus, or server otherwise left no trace in the log. The
        # console copy is suppressed (the print already showed it); the file copy is not.
        print(f"translator: {exc}", file=sys.stderr)
        logger.error("run aborted: %s", exc, extra={"console_shown": True})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
