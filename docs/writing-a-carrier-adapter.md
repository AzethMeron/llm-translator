# Wiring in a carrier adapter

A **carrier adapter** turns one source — a subtitle track, a game's script files, a document —
into translation units and writes translated units back. The translator never learns which
carrier it is serving, so supporting a new one means writing an adapter and nothing else.

This library ships **no** adapters on purpose: it is the translator alone. This guide is how a
project (a video pipeline, a game translator, ...) wires itself onto it. It is the distilled
experience of the three projects this engine was lifted from; most of the mistakes below were
made at least once, and each cost either a corrupted output file or wasted GPU hours.

## The minimum the translator requires

Just one thing: a package exposing an `adapter` module with a `name` and a `sanitize_payload`.

```python
# mycarrier/adapter.py
name = "mycarrier"

def sanitize_payload(text: str) -> str:
    """Normalise a translation into text this carrier can hold."""
    return text.replace("\n", "\\n")   # e.g. a carrier whose record ends at a newline
```

```bash
PYTHONPATH=src:. python -m translator.cli --adapter mycarrier -c units.jsonl -j journal.jsonl
```

Everything else here is about producing units the translator can work with, and writing the
results back without breaking the carrier.

## Calling the translator as a library

Most real projects drive the engine directly rather than through its CLI, because they own the
extraction and the write-back:

```python
from pathlib import Path
from translator import (TranslationAgents, RuleSet, AgentSet, TranslationMemory,
                        LlmClient, ServerConfig, resolve_backend, run_batch, pending_units)
from transunit import (read_glossary, load_profiles, lexical_detector,
                       read_reference, LexicalRetriever)

model = "qwen3-14b"
client = LlmClient(ServerConfig(model=model), backend=resolve_backend("auto", model))
profiles = load_profiles(Path("config/languages.toml"))
agents = TranslationAgents(
    client,
    RuleSet.load(Path("config/translation_rules.toml")),
    read_glossary(Path("work/glossary.jsonl")),
    # AgentSet carries the panel, the token budgets, the revision counts, and the context
    # window -- all of it configuration, so you tune behaviour without touching this code.
    agent_set=AgentSet.load(Path("config/agents.toml"),
                            {"source_language": "English", "target_language": "Polish"}),
    memory=TranslationMemory.from_units(...),      # optional, for context + dedup reuse
    # optional: retrieve similar prior translations (an existing partial track, a sister
    # project) into each prompt, to match an established rendering. Off unless the agent
    # config's reference_examples/reference_revision_examples are > 0. See
    # docs/reference-translations.md. Use GrowableLexicalRetriever to also learn from this run.
    reference=LexicalRetriever(read_reference(Path("reference/prior.jsonl"))),
    sanitize=mycarrier.adapter.sanitize_payload,   # your normaliser, passed in
    is_untranslated=lexical_detector(profiles, "en"),  # see "The untranslated check" below
    source_label="EN", target_label="PL")
run_batch(agents, pending_units(catalog, journal), journal)
```

Note what is *injected*: the sanitiser and the untranslated detector. Neither is imported by
the translator, which is what keeps it carrier- and language-agnostic. Note also what is *not*
here: the reviewer panel, how hard the harness tries (`max_revisions`/`max_repairs`), how many
neighbouring units are shown for context (`before_units`/`after_units`), and every token budget
all live in `config/agents.toml`, so they are the operator's to tune, not the adapter's to
hard-code. See `docs/languages.md` and `config/README.md`.

## The four jobs of an adapter

| Job | What it must guarantee |
|---|---|
| **extract** | source → units, each carrying enough provenance to be written back |
| **sanitize** | what this carrier can hold, applied to translated text |
| **inject** | units → source, changing nothing but the payloads |
| **verify** | prove that, by reparsing the result |

Verification is not optional if you write back: it is the only thing between a plausible
translation and an output that will not load, and it has caught real corruption that earlier
layers missed.

## Extraction

**Store a span, do not re-render.** Give each unit the exact range its payload occupied
(`span_start`/`span_end` — bytes, milliseconds, whatever your carrier measures) and splice the
translation back into that range, copying every other byte through untouched. That makes
"injecting untranslated units reproduces the input exactly" true *by construction* rather than
by re-implementing the file's every quirk (BOMs, line endings, trailing-newline quirks). If a
format genuinely cannot be spliced (JSON that must be re-serialised), re-render — but write the
identity test first and expect to spend real time on whitespace, key order and escaping.

**Make `unit_id` stable against unrelated edits.** `transunit.make_unit_id` keys on
`(rel_path, masked source, occurrence index)`, not line number, so an edit elsewhere in the
file does not invalidate existing translations. If your ids move when unrelated lines move,
every upstream update discards the work done so far. Test it.

**Mask everything the translator must not touch.** Engine syntax, inline markup, escapes become
`[[0]]`, `[[1]]`, … (`transunit.placeholders`) and are restored on injection. The translator
only preserves the *set*. Masking must round-trip exactly — run it over your **whole** corpus,
not a sample.

**Withhold what you cannot mask confidently.** Mark a unit `SKIPPED` rather than send a payload
whose engine syntax you could not fully mask. Leaving a line untranslated is a cosmetic loss;
letting the model rewrite an engine expression writes broken syntax into the output.

**Skip what needs no translation.** Text already in the target language, and punctuation that
merely looks like language, should be `SKIPPED` — queueing them wastes GPU and invites the model
to "improve" finished work or to fail a payload it can only return unchanged.

**Populate the optional fields when you can:** `speaker` (identity in the prompt, and the dedup
key), `context_before` / `context_after` (neighbouring lines, shown to the model as one
continuous passage around the line being translated — and, once a neighbour has been translated,
listed again with its established rendering — which keeps a scene consistent), and `max_columns`
(below).

## Sanitization — the adapter's job, and only the adapter's

"What text can this carrier hold" is knowledge only you have. It runs in two places: the
translator calls it early, purely to save a repair round on something code can fix; injection
applies it again and *refuses* what it cannot normalise, because a catalogue can be hand-edited.

- **In it:** defects with exactly one correct repair (a raw newline a carrier cannot hold; a
  full-width form that doubles a column).
- **Not in it:** anything requiring judgement (rephrasing an overlong line, softening a
  euphemism) — that belongs to the model and its reviewers.
- **Rules:** idempotent (it runs on every repair), information-preserving (fixes form, never
  meaning), narrow (one plainly-correct result), and it must never touch the masking sentinel.

## Injection and verification (if you write back)

- **Ship only what is safe.** Inject `VERIFIED` and `TRANSLATED` (mechanically sound; reviewers
  merely still objected). Never inject `REJECTED` — it may have lost a placeholder or kept
  source text. Write `PENDING`/`SKIPPED` through as the original.
- **Never sanitise the passthrough.** Normalise `target`; copy `source` through byte for byte,
  or you rewrite the original's punctuation and break the identity guarantee.
- **Refuse rather than write something broken.** Injection is the last code before bytes hit an
  output file. Check what would be catastrophic and raise, naming the file, line and unit.
- **Verify independently of the injector**, so an injector bug cannot mask itself: reparse the
  result and prove it differs from the original only inside text payloads.

## Width limits

If your carrier draws into a fixed space and clips rather than wrapping, set `max_columns` on
the unit; the translator is then told the budget up front and exceeding it (past the configured
tolerance) blocks. Two things matter: it measures the **whole payload**, not each break-
separated part (whether a break is permitted is carrier knowledge the translator lacks); and
one number cannot express a box of N columns by M lines — leave it unset for a multi-line box.
The translator states the budget but never suggests *how* to break a line; break syntax is
yours.

## The untranslated check

Pass the translator an `is_untranslated` detector suited to your pair (full guide in
`docs/languages.md`):

- **same script** (English↔Polish↔German, all Latin): `transunit.lexical_detector(profiles,
  source_code)`, from TOML language profiles;
- **different scripts** (Japanese, Chinese, Russian, Greek, Arabic, … → Latin, or the reverse):
  `transunit.script_detector("japanese" | "han" | "cyrillic" | "greek" | "arabic" | …)` — no
  profile needed. `transunit.available_scripts()` lists the presets;
  `ScriptDetector.from_ranges(...)` builds one for any script not in the list;
- **cannot tell the languages apart**: pass `None`, and the check is skipped rather than
  reported as passed.

## Checklist

- [ ] Identity: injecting unmodified units reproduces the source exactly
- [ ] Masking round-trips exactly, over the **whole** corpus
- [ ] Anything unmaskable is withheld (`SKIPPED`), not sent
- [ ] `unit_id` survives unrelated edits to the same file
- [ ] Already-translated and untranslatable payloads are `SKIPPED`
- [ ] `sanitize_payload` is idempotent and touches no masking sentinel
- [ ] Injection refuses what it cannot make safe, and never sanitises the passthrough
- [ ] Verification is independent of the injector and gates the build
- [ ] `max_columns` set only where the carrier genuinely clips
- [ ] An `is_untranslated` detector suited to the language pair is wired in
