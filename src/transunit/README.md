# transunit — the contract

The intermediate format two sides agree on, and nothing else. Standard library only, so
neither side can be broken by a third-party upgrade.

| Module | Responsibility |
|---|---|
| `units.py` | the translation unit, its lifecycle `Status`, and the atomic/resumable JSONL catalogue and journal |
| `placeholders.py` | the `[[0]]` convention for text the translator must not touch |
| `glossary.py` | established terminology, looked up by literal occurrence |
| `reference.py` | external reference translations retrieved per unit by similarity (translation memory / RAG); a read-only lexical retriever and a self-populating one |
| `width.py` | display width in half-width columns, so both sides measure a length budget the same way |
| `language.py` | deciding whether a translation actually happened — lexical (same-script pairs) and script-based (different-script pairs) |
| `adapter.py` | the **carrier adapter** contract: what a translator needs from whatever supplied the units |

## The two sides, and the two "adapters" problem

```
        carrier adapter                     translator
   (subtitles / game / docs)         (this repo's src/translator)
              │                                │
              └──────────► transunit ◄─────────┘
                        (this package)
```

A translator reads units and writes them back; it never learns what a placeholder
contained or where the text came from. It has exactly one downstream dependency on the
carrier — `sanitize_payload`, "what text can this carrier hold" — and that is *passed in*,
not imported, so the translator stays carrier-agnostic.

To keep the vocabulary unambiguous:

* the **downstream** boundary, toward the carrier, is a carrier **adapter**
  (`transunit.adapter`);
* the **upstream** boundary, toward the model, is a **backend** (`translator.backend`).

They are never both called "adapter".

## Spans are opaque

`Unit.span_start` / `span_end` mark where a payload came from, in whatever unit the
adapter chose — milliseconds along a timeline, a byte offset in a file. This module only
requires `span_end >= span_start`; what a span *means* is the adapter's business.

## Deciding "is this still untranslated"

There is no single right answer, because it depends on the language pair, so `language.py`
offers two mechanisms that produce the same injectable `(source, target) -> bool` shape:

* **Lexical** — `lexical_detector(profiles, source_code)` for pairs that share a script
  (English/Polish). Evidence is function words and distinctive characters, loaded from
  TOML profiles.
* **Script** — `ScriptDetector.from_ranges(...)` (or `japanese_detector()`) for pairs
  whose scripts differ (Japanese/English). Evidence is the source's own letters surviving
  into the target.

Both **abstain rather than guess**: they report `True` only on positive evidence, so the
blocking check they back never fires on a line too short to judge.

## Durability

`write_catalog` writes a temporary sibling, fsyncs, renames, then fsyncs the directory —
a crash cannot leave a half-written catalogue under the real name. `read_journal` tolerates
a torn *final* record (a process killed mid-write) and refuses a malformed one anywhere
else, because skipping that would discard a completed translation while reporting success.
