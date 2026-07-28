# general-translator

A carrier-agnostic, model-agnostic translation engine for local language models, with a review
harness that makes unattended translation trustworthy.

This is the shared core lifted out of three projects that had each forked it — a game-text
translator, a Flash-game translator, and a video-subtitle translator — generalised so one copy
serves all of them and anything else. It is **only the translator**: the code that turns a
specific source into units and writes translations back (subtitle tracks, game files, documents)
is a *carrier adapter*, and none of those live here. This repository is the part they had in
common.

## What it does

Given translatable text in an intermediate format (`transunit`), it drives a local model
through a bounded loop:

```
translate ──> mechanical check (code) ──> review panel ──> revise
```

Cheap, code-decidable rules — placeholder integrity, empty output, untranslated echoes, control
characters, display width — are settled before any GPU time is spent. Only what needs judgement
— fidelity, fluency, register — reaches a panel of reviewer roles sharing the one loaded model.
Runs are resumable and interrupt-safe: a run of tens of thousands of units survives a crash and
resumes without repeating finished work.

## Two boundaries, two names

- **carrier adapter** (`transunit.adapter`) — the downstream boundary, toward the text. Passed
  in, never imported, so the engine never learns which carrier it serves.
- **backend** (`translator.backend`) — the upstream boundary, toward the model. One small
  profile per model family (Qwen, Bielik, EuroLLM, …); switching models is a `--backend` choice,
  not a code change.

They are never both called "adapter" — that ambiguity is exactly what the naming avoids.

## Layout

```
src/transunit/     the contract: units, catalogue, glossary, reference retrieval, placeholders, width, detection
src/translator/    the engine: harness, rules, roles, memory, context packing, runner, cli
src/translator/backend/     the model layer: client + one profile per model family
src/translator/retrieval/   optional retrievers: embedding, reranking, hybrid fusion
config/            general, high-quality sample prompts and rules — copy and tune
docs/              architecture, and the two wiring guides
tools/             hardened setup / check / test / serve scripts, plus a retrieval-quality gate
tests/             the translator's own suite (no server, GPU, or network needed)
```

## Quick start

```bash
tools/setup_python_env.sh     # build the pinned environment (.venv/)
tools/check_environment.sh    # confirm it is ready
tools/run_tests.sh            # 861 tests, no server or GPU required
```

To translate for real you point it at a local OpenAI-compatible server:

```bash
tools/serve_model.sh qwen3:14b     # ollama; prints the --base-url / --model to use
PYTHONPATH=src python -m translator.cli \
    --base-url http://127.0.0.1:11434/v1 --model qwen3:14b --backend auto \
    -c work/units.jsonl -j work/journal.jsonl \
    --source-language en --target-language pl
```

## Any language pair

It works for any pair in the world — English, Polish, German, Arabic, Chinese, Japanese,
Russian, Greek, … — with changes only in configuration. Same-script pairs add a lexical profile;
different-script pairs pass `--source-script` (a named preset) and need none. The review panel,
token budgets, revision counts, and context window are all `config/` data too, so tuning
behaviour never means editing code. See `docs/languages.md` and `config/README.md`.

## Wiring your project onto it

Read `docs/writing-a-carrier-adapter.md` (connect a source of text), `docs/writing-a-backend.md`
(add a model), `docs/languages.md` (set up a language pair), and
`docs/reference-translations.md` (match an existing body of translation via per-unit
retrieval). The engine's public API is documented in `src/translator/__init__.py` and
`src/transunit/__init__.py`.

## License

Source-available and **dual-licensed**: free for any noncommercial use (personal,
educational, research) under the **PolyForm Noncommercial License 1.0.0**, and **commercial
use requires a separate paid license** from the author. The full terms are in
[`license/`](./license/) — start with [`license/README.md`](./license/README.md). Copyright
© 2026 Jakub Grzana.

## Design

`docs/architecture.md` covers the whole design. The short version: the contract is
standard-library only so rendering can never break on a dependency upgrade; errors are typed by
blast radius so one bad unit never aborts a long run; the untranslated check works for both
same-script and different-script language pairs; an optional external **reference corpus** is
retrieved per unit to match an existing translation (off by default, stdlib-only lexical matching
with optional semantic and hybrid-fused-with-reranking variants); positional context and retrieved
examples are packed into one continuous passage rather than separate labelled sections; and every
optional prompt section is omitted when it has no content, so nothing leaks into the model but
signal.
