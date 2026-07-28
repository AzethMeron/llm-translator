# translator — the translation engine

Fills in translation units using a local language model. Knows nothing about any
particular carrier (it reads the `transunit` format and writes it back) or any particular
model (it reaches one through a `backend`).

| Module | Responsibility |
|---|---|
| `backend/` | the model-facing layer: the client, and one small profile per model family |
| `rules.py` | translation policy, and the mechanical checks that need no GPU |
| `roles.py` | who translates and who reviews, loaded from `config/agents.toml` |
| `agents.py` | the harness: translate → check → review → revise |
| `memory.py` | *our own* output this run: duplicate reuse and target-language context (distinct from the external reference store) |
| `context_packing.py` | selects and renders positional context + retrieved reference examples as one continuous passage |
| `retrieval/` | *optional* reference retrievers behind the `Retriever` seam: `embedding.py` (cosine over `/v1/embeddings`, needs numpy, lazy), `rerank.py` (cross-encoder over `/v1/rerank`), `hybrid.py` (fuses lexical + embedding via RRF, optional rerank + MMR), `fusion.py` (pure RRF/MMR math) |
| `embedding.py` | back-compat re-export shim for `retrieval/embedding.py` |
| `runner.py` | resumable, concurrent batch execution over a catalogue |
| `cli.py` | the entry point |

## The harness

```
translate ──> mechanical check (code) ──> review panel ──> revise
```

The mechanical check sits between the model steps deliberately. Placeholder integrity,
untranslated text and display width are decidable in code, so they are settled before any
GPU time is spent asking a model for an opinion.

Reviewers are roles sharing one loaded model, differing only by system prompt. A revision
sees the attempt it is being asked to fix, the objections tagged by role, and any rewrite
the reviewers proposed — without the previous attempt a model re-translates blind and
reproduces its own mistake.

## Errors, by blast radius

- `LlmContentError` — this one request is unusable (unparseable JSON, a repetition loop, a
  refusal). The unit is recorded `REJECTED` and the run continues.
  - `LlmIncompleteJsonError` — a `json_object` reply (Bielik, EuroLLM) cut short right after
    the `"translation"` value closed; `translate()`'s repair loop spends the repair budget
    re-checking the mechanically-recovered candidate instead of losing the unit for free, and
    can only ever keep it as `TRANSLATED`, never silently `VERIFIED`. Toggle:
    `[revision].repair_truncated_json`.
- `LlmError` — the transport or server is broken, so every remaining unit would fail the
  same way. The run **stops**, journal intact, and a resume loses nothing.

## Reference translations (optional)

Beyond the glossary and neighbour context, the harness can retrieve **similar prior
translations** from an external reference corpus — a previous run, a sister project, an
existing partial track — and show them per unit as worked examples, to match an established
rendering. It is a `transunit.reference.Retriever` injected as `reference=`, off unless the
agent config's `reference_examples` / `reference_revision_examples` is above zero. The store
is read-only by default; a `GrowableLexicalRetriever` with `reference_learn_statuses` set
also learns from this run's own accepted output. Two optional retrievers drop in behind the
same seam: `EmbeddingRetriever` (`--embedding-url`, semantic matching) and `HybridRetriever`
(`--hybrid`, fuses lexical + embedding via Reciprocal Rank Fusion with an optional cross-encoder
reranking stage, `--rerank-url`); both are always read-only. Full design and how-to:
[`docs/reference-translations.md`](../../docs/reference-translations.md).

## Prompt hygiene

Prompt assembly includes an optional section **only when it has content**. No glossary
means no terminology header; no style directives means no "Style policy" header; a
speakerless line has no "Speaker" line; no matching reference means no reference section.
Nothing is emitted just to say it is empty.

## Two boundaries, two names

- **backend** (`translator.backend`) — the model side. Selects how a structured request is
  shaped for the model in use; `--backend auto` picks it from the model id.
- **adapter** (`transunit.adapter`) — the carrier side. Passed in (never imported) so the
  engine stays carrier-agnostic.

They are never both called "adapter".

## Usage

```bash
PYTHONPATH=src python -m translator.cli \
    -c work/units.jsonl -j work/journal.jsonl -g work/glossary.jsonl \
    --model qwen3-14b --backend auto \
    --source-language en --target-language pl
```

`--adapter NAME` names a carrier adapter supplying payload normalisation; omit it for a
plain corpus. `--source-language`/`--target-language` name the pair in the prompts and
select the profile the untranslated check uses. `--reference FILE` (repeatable) supplies a
reference corpus for retrieval, active only when the agent config turns it on. `--context-window
N` gives the server's token window so the engine warns once, before the wall, when a prompt plus
its output budget nears 80% of it (the server's own rejection is the hard stop either way).
Resumable and interrupt-safe: results append to the journal as each unit completes, and a resume
skips what is already recorded.

## Changing policy

Prefer `config/translation_rules.toml` and `config/agents.toml` — they are data, and the
reviewers read them directly. Reach for a code change only when the config cannot express
it.
