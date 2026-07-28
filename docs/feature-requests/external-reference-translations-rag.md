# External reference translations, retrieved per unit (translation memory / style exemplars / RAG)

- **Status:** proposed
- **Date:** 2026-07-23
- **Engine commit this was written against:** `a8dc518` (general-translator, `__version__` 0.1.0)
- **Intended home:** `docs/upstream-feature-requests/00N-external-reference-translations-rag.md`
  in the host project; filed to the general-translator maintainer.

## What we need

A supported way to give the engine a body of **already-translated reference material that did
not come from the current run**, and have it surface the **most relevant** entries into each
unit's prompt as guidance — worked examples for style, register, terminology, and phrasing.

Generalize past any one use case. The single class of usage is: *"here is prior/reference
translation knowledge; when translating a unit, show the model the parts of it that are relevant
to this unit."* Concrete members of that class:

- **Translation memory (TM).** Source→target pairs from outside this run: a previous run, a
  sister project, a vendor/human reference, or the already-translated portion of the same work
  (e.g. a game half-translated by an upstream fork, subtitles with an existing partial track).
  The new lines should match what already exists.
- **Style / voice exemplars.** A curated set of example renderings that demonstrate the target
  voice and register, surfaced to steer style even when no exact source match exists.
- **Domain phrasing beyond the glossary.** The glossary is exact term→term (names, fixed
  terminology). A reference corpus additionally carries idiom, sentence shape, and register that
  a term list cannot express.

The retrieval is the crux, and is why this reads as **RAG**: a reference corpus is far too large
to place in every prompt, and dumping unrelated pairs is noise that hurts more than it helps. The
engine should retrieve the top-k entries *relevant to the current unit's source* and show only
those, within a bounded budget.

## Why the existing seams don't cover it

Walked against `a8dc518`:

- **`TranslationMemory` is run-local and exact-match only.** It is seeded solely from the run's
  own journal (`cli.py`: `TranslationMemory.from_units(read_journal(args.journal))`), and
  `get(source)` is an exact dictionary lookup keyed on the masked source. So it (a) never sees
  external material, and (b) cannot surface a *similar* (non-identical) prior translation — which
  is exactly what a TM/exemplar bank is for. At the start of a run it is empty.
- **Neighbour context is positional, not relevant.** `[context].before_units/after_units` show
  the carrier-adjacent lines (`context_before/after`), chosen by document order, not by content
  similarity to the unit. Useful for local continuity, useless for "find the three most similar
  things we've translated before."
- **The glossary is exact term pairs.** `relevant_terms(source, glossary, limit)` matches
  glossary *terms that appear in the source*; it is term-granular, not sentence/phrase exemplars,
  and it is not a retrieval over arbitrary translated text.
- **Style directives are prose, not examples.** `[style].directives` are policy lines added to
  every prompt; they cannot carry concrete source→target demonstrations, and they are static
  rather than per-unit.
- **No surface to point at an external corpus.** There is no CLI flag or config section for a
  reference file, no retriever abstraction, and no prompt section for retrieved examples.

None of these is closable by configuration or by the carrier adapter — the adapter produces the
`transunit` stream and `sanitize_payload`; it has no hook into prompt assembly or memory. This
genuinely needs an engine change, which is why it is a feature request rather than a change on
our side.

## Proposed shape (offered, not prescribed)

Mirror the engine's existing separations — a small, injectable interface with a
stdlib-friendly default and an optional stronger implementation behind it:

1. **A reference store**, separate from the run journal: one or more collections of masked
   source→target pairs (natural fit: the existing `transunit` unit JSONL, or a minimal
   `{"source","target"}` JSONL). Loaded read-only; never written to; never treated as ground
   truth (guidance only).
2. **A `Retriever` interface** — `retrieve(source, k) -> list[(source, target, score)]` — shared
   across worker threads and **deterministic** (same inputs → same top-k, to preserve
   reproducibility). Selection is pluggable:
   - **Lexical default** (BM25 / n-gram / token-overlap): pure Python, **keeps the
     stdlib-plus-httpx footprint** — no new dependency. Good enough for near-duplicate and
     shared-terminology retrieval.
   - **Embedding / vector (true RAG): optional, behind the interface.** This is the real design
     tension: embeddings need either a model dependency or an embedding endpoint. The cleanest fit
     is to reuse the *existing* local OpenAI-compatible server via its `/v1/embeddings` route
     (llama.cpp/ollama/vLLM expose it), injected the same way a `backend` is — so no new hard
     dependency and no second process. A vector index can stay optional.
3. **Prompt surface + budget.** Show retrieved pairs as a clearly-labelled, bounded "reference
   translations" block — parallel to the neighbour-context "worked examples" block — with its own
   knob (e.g. `[context].reference_examples = k`, `reference_min_score = …`) so it competes for
   the slot window under an explicit cap and priority against neighbour context and the glossary.
   Off by default (`k = 0`), so existing runs are unchanged.
4. **Config / CLI**, matching the `--languages` / `--adapter` / `--backend` style: e.g.
   `--reference <file...>` plus a `[reference]` section (paths, retriever type, k, threshold,
   optional embedding endpoint). Optional throughout.

Design constraints worth stating up front (each is a place a naive RAG bolt-on would break the
engine's guarantees):

- **Masking/placeholders.** Reference pairs must pass through the same placeholder masking so
  `[[n]]` tokens are consistent with the unit; a raw external TM will not be pre-masked.
- **Footprint.** `transunit` is stdlib-only and `translator` adds only `httpx`; the default
  retriever must preserve that, and anything heavier must be optional and injected.
- **Determinism & thread-safety.** Retrieval shared across the `ThreadPoolExecutor` workers and
  stable across runs.
- **Guidance, not answers.** Retrieved targets are shown to steer, never spliced in; the review
  panel and mechanical checks still decide the output.

## Impact if unaddressed

Any project that must **match an existing body of translation** has no supported path. Our
motivating case is a game already half-translated by an upstream English fork: to make the new
half read like the old, the existing translations should inform the new ones — but the engine's
memory is empty at run start and never ingests them. The class is broader: resuming a franchise,
aligning with a human reference, or carrying a house style across projects all want the same
capability.

The available workarounds are poor and we would rather not ship them:

- **Pre-seed the run journal** with fabricated "done" entries so `TranslationMemory` picks them
  up — hacky, pollutes the output journal, and is still **exact-match only** (no retrieval of
  similar lines), so it helps only verbatim repeats.
- **Bake exemplars into the static prompt** (`[style]` / translator instructions) — unbounded,
  not per-unit-relevant, and blows the prompt budget as the exemplar set grows.

Both are strictly worse than a bounded, per-unit retrieval, and neither generalizes. In the
meantime the host project documents the gap as a known limitation and relies on the glossary plus
within-run neighbour context alone.
