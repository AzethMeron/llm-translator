# Languages and generalization

The translator is built to work for **any language pair in the world**, with changes only in
configuration — never in code. This guide is how: what a new pair needs, where each concern
lives, and worked setups for the common cases (English, Polish, German, Arabic, Chinese,
Japanese, Russian, Greek, and others).

The engine hard-codes nothing about any language. Direction, names, register conventions,
punctuation, transliteration, and the "did it actually translate?" check are all either data
(the config files) or injected (the detector), so a new pair is a config change.

## The three things a pair needs

### 1. The language names (for the prompts)

`agents.toml` uses `{source_language}` and `{target_language}`, filled from
`--source-language` / `--target-language`. Codes map to names via a built-in table (so `ar` →
"Arabic", `zh` → "Chinese", `ja` → "Japanese", `ru` → "Russian", `el` → "Greek", …) or via a
profile's `name` in `languages.toml`. A code the table doesn't know falls back to itself, which
still runs — just add it to the table or a profile for a legible prompt.

### 2. The untranslated-echo check (the one language-dependent check)

A model asked to translate occasionally hands back its input. Catching that is the only check
whose method depends on the pair, and there are two, both producing the same injected
`(source, target) -> bool`:

**Same script** — the pair shares an alphabet (English/Polish/German, all Latin; or two
Cyrillic languages). Script tells you nothing, so the evidence is *lexical*: function words and
distinctive letters, from a profile in `languages.toml`.

```bash
--languages config/languages.toml --source-language en --target-language pl
```

Add a profile per language (see `config/languages.toml` for the format and how to pick
stopwords). Keep the stopword lists disjoint across your languages.

**Different scripts** — the source and target use different writing systems (Japanese, Chinese,
Russian, Greek, Arabic, Hebrew, Korean, … against Latin, or against each other). The evidence is
the source's own letters surviving into the target, which needs *no profile at all*:

```bash
--source-script japanese --source-language ja --target-language en
```

Presets: `japanese`, `han` (Chinese), `cyrillic`, `greek`, `arabic`, `hebrew`, `hangul`,
`devanagari`, `thai`. `python -m translator.cli --help` lists them; `transunit.available_scripts()`
returns them. For a script not listed, build one in a few characters:

```python
from transunit import ScriptDetector
armenian = ScriptDetector.from_ranges((("Ա", "֏"),))   # (first, last) code points
```

If you genuinely cannot tell the two languages apart, pass neither: the check is **skipped**
rather than reported as passed (an unevaluable check must never read as a pass).

### 3. Whatever the pair makes salient (in the prompts)

Everything else is translation quality, and it lives in `agents.toml` (the panel) and
`translation_rules.toml` (the advisory criteria). The shipped defaults already handle, in a
language-neutral way:

- **Formality / register** — the informal vs. formal "you" (tu/vous, du/Sie, ты/вы), honorifics
  and politeness levels (Japanese, Korean). The `register` advisory and the `grammar` reviewer
  cover it; name your pair's specifics in the `[translate]` prompt if they are unusual.
- **Proper nouns / transliteration** — same-script keeps the spelling and inflects only the
  ending; cross-script transliterates once and reuses that spelling. The `proper_nouns` advisory
  covers it.
- **Pronouns and pro-drop** — languages that drop the subject (Japanese, Arabic, Polish,
  Spanish) vs. ones that require it (English). The `pronouns_and_reference` advisory and the
  `[translate]` prompt cover supplying/omitting subjects correctly.
- **Punctuation and orthography** — the target's own marks and quotation style (CJK full-width
  `。，「」`, Spanish `¡¿`, French `« »`, Arabic `؟،`, Greek `;` as a question mark). The
  `grammar` reviewer and the `localization` advisory cover it.
- **Numbers, dates, measurements** — target-locale formats without changing values
  (`localization` advisory).

You rarely need to touch these for a new pair; add to them when your pair or domain has a
specific hazard.

## Two things worth knowing about the mechanics

- **Display width is measured in half-width columns**, so a full-width CJK glyph counts as two.
  This matters when a carrier sets a `max_columns` budget for Chinese, Japanese, or Korean
  output — the budget is compared against the true drawn width, not the character count.
- **The mechanical name-respelling check is script-aware by omission.** It catches a respelled
  proper noun by capitalisation and stem similarity, which only applies to cased scripts (Latin,
  Cyrillic, Greek). For Arabic, Chinese, Japanese, and other uncased scripts it simply does not
  fire — it fails safe, and cross-script transliteration consistency is left to the
  `proper_nouns` advisory reviewer instead.
- **Right-to-left scripts** (Arabic, Hebrew) work as source or target; the translator operates on
  text, not layout. Bidirectional *display* is the carrier's concern.

## Worked setups

| Pair | Detection | Command tail |
|---|---|---|
| English → Polish | lexical | `--source-language en --target-language pl` |
| German → English | lexical | `--source-language de --target-language en` |
| Japanese → English | script | `--source-language ja --target-language en --source-script japanese` |
| Chinese → English | script | `--source-language zh --target-language en --source-script han` |
| Russian → English | script | `--source-language ru --target-language en --source-script cyrillic` |
| Greek → English | script | `--source-language el --target-language en --source-script greek` |
| English → Arabic | script | `--source-language en --target-language ar --source-script arabic`* |

\* For an English **source** into Arabic, the echo you are guarding against is English text left
in the Arabic output — a *Latin* residue, not Arabic. Use a lexical `en` profile (`--languages`)
rather than `--source-script arabic`, or skip the check. Choose the detector by which script the
*untranslated leftover* would be in: it is always the **source** script. (Latin→Latin uses
lexical; source-in-another-script uses that script's preset.)

## Adding a language: checklist

- [ ] Its name resolves — it is in the built-in table or you added a `languages.toml` profile.
- [ ] Detection is chosen by the source script: lexical profile for same-script, `--source-script`
      for different-script, or `None` if the pair is genuinely indistinguishable.
- [ ] Lexical profiles (if used) have disjoint stopword lists and the language's distinctive
      letters.
- [ ] The `[translate]` prompt and advisory criteria name any pair-specific hazard (an unusual
      honorific system, a script-specific punctuation rule) — the general defaults cover the rest.
- [ ] If the carrier clips text, `max_columns` is set and width is understood in half-width
      columns for CJK.
