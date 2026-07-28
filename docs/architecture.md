# Architecture

## What this is

A translation engine that fills in an intermediate format using a local language model, with
a review harness that makes the output trustworthy. It is the shared core lifted out of three
projects that had each forked it — a game-text translator, a Flash-game translator, and a
video-subtitle translator — and generalised so one copy serves all of them and more.

It is deliberately **only the translator**. The things that turn a specific source into units
and write translated units back — subtitle tracks, game script files, documents — are *carrier
adapters*, and none of them live here. This repository is the part every one of those projects
had in common.

## The two boundaries, and why they are named differently

```
      carrier adapter                     translator                     model
  (subtitles / game / docs)        (this repo: src/translator)     (qwen / bielik / ...)
           │                                  │                            │
           │  transunit units                 │  backend                   │
           └────────────► transunit ◄─────────┤                            │
                       (src/transunit)        └──── translator.backend ────┘
```

Two boundaries meet in the translator, and calling them both "adapter" is the ambiguity this
design refuses:

- The **downstream** boundary, toward the text, is a carrier **adapter**
  (`transunit.adapter`). The translator depends on it for one thing only —
  `sanitize_payload`, "what text can this carrier hold" — and that is *passed in*, never
  imported, so the translator never learns which carrier it is serving.
- The **upstream** boundary, toward the model, is a **backend** (`translator.backend`). It
  shapes a request for the model in use. It is emphatically not an "adapter".

`transunit` sits under both and depends on neither. It is standard-library only, so rendering
a catalogue can never break because a machine-learning dependency was upgraded. The boundaries
are enforced by `tests/test_separation.py`.

## The harness: why a review loop at all

A single-shot local model produces plausible translations with occasional silent defects — a
dropped clause, an echoed source line, a placeholder quietly renumbered. The harness turns that
into something you can ship unattended:

```
translate ──> mechanical check (code) ──> review panel ──> revise
```

The mechanical check sits between the model steps on purpose. Placeholder integrity, empty
output, untranslated text, control characters and display width are **decidable in code**, so
they are settled before any GPU time is spent asking a model for an opinion, and a translation
that fails them is sent straight back with a precise reason. Only what genuinely needs
judgement — register, fluency, fidelity — reaches the review panel, which is several roles
sharing the one loaded model and differing only by system prompt.

A revision sees the exact attempt it is being asked to fix, the objections tagged by role, and
any rewrite a reviewer proposed. Without the previous attempt, a model told only "that was
rejected" re-translates blind and reproduces its own mistake.

Everything the harness does is bounded: `max_repairs` mechanical repairs per generation,
`max_revisions` review rounds per unit, after which the unit is recorded rather than retried
forever.

## Errors by blast radius

The single most important distinction in the engine, because it decides whether one bad unit
stops a run of tens of thousands:

- `LlmContentError` — *this* request is unusable (unparseable JSON, a repetition loop, a
  refusal). It says nothing about the next unit, so the unit is recorded `REJECTED` and the run
  continues.
  - `LlmIncompleteJsonError` — a narrower subtype, for a `json_object` backend (Bielik,
    EuroLLM) reply that stopped right after closing the `"translation"` string, before the
    object's own closing brace. `complete_json` recovers it mechanically and raises this
    instead of the base type, carrying the recovered object; `translate()`'s repair loop
    catches it specifically and spends the repair budget re-checking it like any other
    candidate, rather than losing the unit for free. Never trusted outright: the recovered
    text goes through every mechanical check and the review panel, and can only ever be kept
    as `TRANSLATED`, never silently `VERIFIED`. `[revision].repair_truncated_json = false`
    turns this off for a project that wants strictness. See
    `docs/feature-requests/json-envelope-truncation-repair.md`.
- `LlmError` — the transport or server is broken, so every remaining unit would fail the same
  way. It propagates, the run **stops** with the journal intact, and a resume loses nothing.

Turning the second into the first once burned every remaining unit on a run that still exited 0.

## The model layer: one object per model family

Local models served over an OpenAI-compatible endpoint do not all honour the same request. The
sharpest difference is structured output: Qwen accepts a strict `json_schema` grammar, while
Bielik's build rejects it ("Failed to initialize samplers") but accepts `json_object` with the
shape described in the prompt. Before this was factored out, that difference lived as
conditionals inside the client keyed on a config string, and switching from Qwen to Bielik
"broke something that should not have".

Now each model family is one small `Backend` in `translator/backend/profiles/`, selected by
name or by `--backend auto` from the model id. The translator itself never changes when the
model does; adding a model is a new profile and nothing else. When a server still rejects a
structured request, the client **names the fix** rather than degrading silently. See
`docs/writing-a-backend.md`.

## Deciding "is this still untranslated"

The one check whose right answer depends on the language pair. `transunit.language` offers two
mechanisms with the same injectable `(source, target) -> bool` shape:

- **Lexical** — for pairs that share a script (English/Polish). Script detection is useless
  there — both are Latin — so the evidence is function words and distinctive characters, loaded
  from TOML profiles.
- **Script** — for pairs whose scripts differ (Japanese/English, Chinese/English,
  Russian/English, …). The evidence is the source's own letters surviving into the target,
  which needs no profiles at all — just a named preset (`han`, `cyrillic`, `greek`, `arabic`,
  `japanese`, …) or a custom range.

Both **abstain rather than guess**: they answer only on positive evidence, so the blocking
check they back never fires on a line too short to judge. The detector is injected, not
imported, so a rule module never depends on one language pair. That injection is what lets one
copy of the engine serve **any pair in the world with changes only in configuration** — the
detector, the language names, register, transliteration, punctuation, and pro-drop are all data
or injected, never code. `docs/languages.md` is the guide.

## Everything tunable is configuration, not code

The behaviour an operator wants to adjust does not live in Python. `config/agents.toml` carries
the whole review panel (as prompts), the per-role token budgets — the translator's scales with
the source, each reviewer has its own ceiling — the revision budget as plain counts
(`max_revisions`, `max_repairs`), the context window, and the console-warning leniency
(`[leniency]`). `config/translation_rules.toml` carries the mechanical limits and the advisory
criteria the compliance reviewer judges.

A reviewer whose reply cannot be read does not take the panel down with it: it abstains, the
reviewers after it still run, and the unit is kept for review rather than verified on a partial
panel. Every such reply is logged in full (with the model's response); `[leniency]` bounds only
how many pass without a *console* warning, so a flaky reviewer neither hides a systemic failure
nor buries it in noise.

The context window is where prompt size is kept sound: `before_units` and `after_units` bound
how many neighbours a prompt carries, `glossary_terms` bounds the terminology, and those caps
are what stop the compounded request — system prompt, glossary, context, the unit, and on
revision the previous attempt and the reviewers' notes — from overflowing the model's window on
a long run. A neighbour is shown with its translation whenever memory has one, on **both** sides:
a following unit that a duplicate, a resume, or a caller-chosen order already translated is real
context, not to be hidden.

Alongside positional context sits an optional third retrieval input: an external **reference
corpus** (`transunit.reference`), separate from the run's own memory and read-only by default,
from which the most *similar* prior translations are retrieved per unit and shown as worked
examples. It is a `Retriever` behind an interface — a deterministic, stdlib-only lexical default
(TF-IDF over character n-grams, an inverted index so a large corpus stays cheap per query), with an
optional semantic `EmbeddingRetriever` (`--embedding-url`, `translator.retrieval.embedding`)
dropping in behind the same seam for meaning-based matching, and an optional
`HybridRetriever` (`--hybrid`, `translator.retrieval.hybrid`) fusing the two via Reciprocal Rank
Fusion, with an optional cross-encoder reranking stage (`--rerank-url`) and Maximal Marginal
Relevance diversity over the result. Only the lexical retriever can be told to learn from the
run's own accepted output, within a configured status filter (a lock-guarded, order-dependent,
deliberately opt-in variant); the embedding and hybrid retrievers are always read-only, refused
outright if combined with learning rather than silently ignoring it. Off by default, it
changes no prompt. Its purpose is *consistency* with an existing body of translation, not per-line
quality: because it faithfully steers toward the corpus, the corpus's quality is the corpus's, and
a similarity floor (the relevance dial) keeps off-target matches out. See `reference-translations.md`.

Positional context and retrieved reference examples are assembled into the prompt by one
`ContextPacker` (`translator.context_packing`): previous, current, and next lines are rendered as
a single continuous passage rather than separately-labelled sections, because a carrier line is
frequently a fragment — a sentence, clause, or idiom split across unit boundaries — and a model
reads continuous prose the way it would read the source, resolving a cross-boundary idiom
correctly where individually-labelled snippets read as unrelated thoughts. The immediate neighbour
on each side is always shown regardless of any budget; an optional character budget
(`context_char_budget`, off by default) trims further neighbours from the outside in, preserving
contiguity, while retrieved reference examples — independent lines with no adjacency to preserve —
are the ones Maximal Marginal Relevance diversifies. The line being translated is repeated
verbatim, unchanged, immediately before generation, so however continuous the passage reads,
the translation target is never ambiguous.

## Durability

A full run spans hours and will be interrupted, so durability is a correctness requirement:

- Catalogues and journals are JSON Lines — streamable, appendable, resumable.
- `write_catalog` writes a temporary sibling, fsyncs, renames, then fsyncs the directory. A
  crash cannot leave a half-written catalogue under the real name.
- The journal is append-only and only the main thread writes it, so a run killed mid-way
  resumes without re-translating what it already finished.
- `read_journal` tolerates a torn *final* record — the expected shape of a process killed
  mid-write — and refuses a malformed one anywhere else, because skipping that would discard a
  completed translation while reporting success.

Concurrency does not weaken any of this: workers only call the model; the single main thread
owns the journal and the progress counters.

## Work order is a caller's choice

Which units to translate first — a game's menus before its dialogue, so a partial run is still
usable — depends on the carrier, so the runner takes an injectable sort key and defaults to
catalogue order. Baking one project's answer into the engine is exactly what made this module
project-specific before.

## Provenance

The translation harness, intermediate format and durability machinery were carried over from an
earlier game-translation project and generalised here. What was carried over is the part that
had nothing to do with any one project's subject: the review loop, the journal, the atomic
catalogue writes, the placeholder contract. What was generalised for this repository:

- the "LLM adapter" became `translator.backend`, a named layer with one profile per model,
  because a config-string conditional did not survive a model switch;
- `transunit.language` gained the lexical detector alongside the script detector, so a
  same-script pair is served as well as a different-script one;
- the runner's work order became injectable, dropping one game's hard-coded area priorities.
