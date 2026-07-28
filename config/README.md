# config/ — policy, prompts, and tuning as data

Everything that decides *how* the translator behaves lives here, not in code. These files are
a **strong, general, high-quality default** that everyone is expected to adopt and adapt.
Nothing here is tied to a language, a game, or a medium beyond the example language profiles.

| File | What it decides |
| --- | --- |
| `agents.toml` | who translates, who reviews, the token budgets, the revision budget, and the context window |
| `translation_rules.toml` | what counts as an acceptable translation (mechanical + advisory) |
| `languages.toml` | lexical fingerprints for the untranslated check, for same-script pairs |

## What lives where in `agents.toml`

```toml
[limits]      # output token budgets (translate scales with source; review_tokens is the default)
[revision]    # max_revisions and max_repairs -- how hard the harness tries, as plain counts
[context]     # how much surrounding material each prompt carries (see below)
[leniency]    # how tolerant the console is of a reviewer's unusable replies before it warns
[translate]   # the translator's instructions
[[reviewer]]  # one per reviewer, in consultation order; each may set its own max_tokens/leniency
```

### The reviewer panel

Each `[[reviewer]]` shares the one loaded model and differs only by `instructions`. They run in
order and **the loop stops at the first objection**, so put the cheapest, most decisive check
first. The shipped panel is:

`accuracy` (fidelity + force) → `structure` (who-did-what-to-whom) → `grammar` (agreement,
orthography) → `fluency` (no calque) → `compliance` (reads all the `[[advisory]]` criteria).

Add, remove, or reorder freely. One reviewer may set `from_rules = true` instead of
`instructions`, which builds its prompt from the `[[advisory]]` criteria so policy has one home.
Any reviewer may set `max_tokens` to override `limits.review_tokens` for itself (a punctuation
checker needs less room than one that rewrites long sentences). Keep at least one reviewer, or
set `revision.max_revisions = 0` — an empty panel would accept every translation unread, and the
loader refuses one.

### The context window (`[context]`)

| key | meaning |
|---|---|
| `before_units` / `after_units` | how many neighbouring units to show on each side |
| `include_translations` | show a neighbour's translation alongside its source when memory has one (both sides) |
| `context_char_budget` | soft character budget for the surrounding-context passage plus reference examples, beyond the guaranteed innermost neighbour on each side (`0` = off, the default: nothing further is trimmed) |
| `glossary_terms` | most established terms shown per unit |
| `reference_examples` | external reference translations retrieved by source similarity and shown before translating (`0` = off; needs `--reference`) |
| `reference_min_score` | relevance floor in `[0, 1]` a reference must clear (default `0.30`, high-precision; lower to `0.15`–`0.20` for more recall and more noise). With `--hybrid` this is the *final* floor, applied after fusion and any reranking, on a fixed (not per-query) scale — above `0.5` it keeps only candidates both arms agree on |
| `reference_candidate_pool` | with `--hybrid`, how many candidates each arm retrieves before fusion/rerank/MMR narrow them down (default `40`) |
| `reference_mmr_lambda` | with `--hybrid`, the relevance/diversity balance for Maximal Marginal Relevance, in `[0, 1]` (default `0.7`, relevance-leaning); also used to diversify reference examples in the unified context packer. **Swept end-to-end across the full `0.0`–`1.0` range (corpus A, n=300, six values): no effect on consistency** — mean overlap flat at 0.098–0.106, non-monotone, and `1.0` (diversity off) no better than `0.0`. This table previously advised raising it toward `1.0` for consistency; that advice was an inference, the sweep falsified it, and it is withdrawn. Leave it at the default (see `docs/reference-translations.md`, "The λ sweep — a null result") |
| `reference_lexical_min_score` / `reference_embedding_min_score` | with `--hybrid`, the *per-arm* recall floors before fusion (defaults `0.30`/`0.55`, matching the plain lexical/embedding retrievers) |
| `reference_revision_examples` | reference translations retrieved by *target* similarity to the draft, shown during revision (`0` = off) — the path that uses reference held without its source |
| `reference_learn_statuses` | which of this run's own results to add to the reference store for later similarity retrieval (`[]` = read-only) |
| `anonymous_subject` | stand-in for a placeholder in a context line whose speaker is unknown — set it to a natural third-person word in the *source* language |

Reference retrieval is off by default and fully documented in
[`../docs/reference-translations.md`](../docs/reference-translations.md), including how to supply
a corpus (`--reference`), the optional hybrid retriever (`--hybrid`, `--rerank-url`) that fuses
lexical and embedding matching with Reciprocal Rank Fusion and an optional cross-encoder reranker,
and how self-population (`reference_learn_statuses`) trades reproducibility for a store that
learns as it runs (lexical retriever only — the embedding and hybrid retrievers are always
read-only).

**Whether to turn it on at all.** Measured end-to-end on 1000 held-out Japanese→English lines:
reference retrieval does **not** improve per-line quality (165 no-reference wins to 140, 523
ties — not significant), and it **nearly triples the rejection rate** (3.0% → 8.8%) because a
corrupt corpus row shown as an authoritative example destabilises generation. It does steer output
toward the corpus — that is the whole point, and it is close to a tautology — but the "**~3×** more
often" multiplier this section used to quote came from a biased metric and has been **withdrawn**;
see `docs/reference-translations.md`. Turn it on to match an existing body of translation, not to
get better prose, and only with a corpus you trust.

**Which retriever, if you turn it on — pick on robustness, measured on your corpus.** End-to-end
on 1000 held-out Japanese→English lines, embedding alone vs `--hybrid --rerank-url`: per-line
quality is a wash (161 hybrid+rerank wins to 147 embedding, 551 ties), and **consistency shows no
measurable difference** — against real held-out ground truth the two arms are a dead heat (223 vs
222 of 593, z = 0.05). **The only measured differentiator is robustness, and it is
corpus-dependent**: rejections 8.5% → 3.0% on the dirty corpus, because the cross-encoder filters
out corrupt rows that derail the dense arm on short lines — but 0.5% → 0.8%, i.e. nothing, on a
clean one. So measure your own rejection rate before paying for a second server; bare `--hybrid`
without a reranker is not recommended.

> **Retracted.** This section previously said the **embedding retriever alone is better at
> consistency** (222 units to the hybrid's 132), replicated on a second corpus. That number came
> from a metric that scored each arm against what an embedding retriever fetches — i.e. against the
> embedding arm's own top pick — so it was circular. The replication reproduced the bias, not the
> finding. Full account: `docs/reference-translations.md`, "The consistency finding was an
> artifact".

**Migration note.** Positional context (`before_units`/`after_units`) used to render as two
separately-labelled "Preceding sentences" / "What is said next" sections, each a bulleted list.
It now renders as one continuous passage (a "Surrounding context" section) with the current line
embedded at its natural position and repeated verbatim afterward — a carrier line is frequently a
fragment of a sentence or idiom split across unit boundaries, and continuous prose reads a
cross-boundary fragment correctly where separately-labelled snippets do not. This applies to every
run using `before_units`/`after_units` > 0, the existing default; which neighbours are shown,
established translations, and placeholder masking are all unchanged — only the rendering shape is.

### When a reviewer's reply can't be read (`[leniency]`)

A reviewer occasionally returns a reply the harness cannot evaluate — unparseable JSON, or a
field of the wrong shape (json_object-mode models do this most). Such a reply is **isolated**:
that reviewer abstains, the reviewers after it still run, and the unit is kept for human review
(`TRANSLATED`) rather than verified on a partial panel. Every such reply is **always** recorded —
in the unit's journal note and in the full log file (`--log-file`, default `work/translator.log`),
with the reviewer, unit, source, and the model's actual reply, so it can be diagnosed without
reproducing it. The log file also captures any **fatal error** that aborts a run (a bad config, a
malformed corpus, an unreachable server), so a failed run leaves a trace there and not only on
stderr.

`[leniency]` bounds only the **console** copy, so a flaky reviewer does not bury the signal in
noise: within a rolling window of the last `window` replies from a reviewer, up to `max_bad`
unusable ones pass without a stderr warning; the one past the allowance is surfaced, because at
that rate the reviewer is failing, not merely flaky. The log keeps everything regardless. Tracked
per reviewer; set `max_bad = 0` to warn on every one. A reviewer may override either value with
its own `leniency = { window = N, max_bad = M }`.

### Token soundness — why the prompt won't overflow

The `[context]` caps are what keep the compounded request bounded. The prompt is
`system + glossary (≤ glossary_terms) + reference (≤ reference_examples) + context (≤ before+after
neighbours) + the unit + on revision, the previous attempt and the reviewers' notes`. With the
defaults that stays a few thousand tokens even in the worst case — comfortably inside an 8k
window — so a long run does not overflow while compounding. **If your units are paragraph-sized
rather than a line, lower `before_units`/`after_units`** (and any `reference_examples`) so the sum
stays within your model's context window minus the output ceiling.

Two guards catch a prompt that grows too large anyway. Pass **`--context-window N`** (the
server's token window, e.g. llama.cpp `-c`) and the engine warns **once**, before the wall, when
an assembled prompt plus its output budget crosses **80%** of it — naming the knobs above to
shrink it. Set nothing and the server's own rejection still catches an oversized prompt, reported
as a clear "the prompt is larger than the server's context window" rather than a mystery failure.
The run summary also prints total and peak prompt tokens, so you can see the real headroom.

## `translation_rules.toml`

Two kinds of rule:

- **mechanical** (`[limits]`, `[[forbidden]]`) — enforced in code, no GPU. `[[forbidden]]`
  catches model failure modes (preamble, refusals, labels, and collapsed placeholders that
  signal dropped text).
- **advisory** (`[[advisory]]`) — prose criteria the `compliance` reviewer judges together. The
  defaults cover register/formality, voice, continuity, transliteration, pronouns/pro-drop,
  figurative language, and localization.

This file is **not** substituted, so write "the target language", never `{target_language}`.

## Choosing the language pair

The pair is a command-line choice, not a file edit — the same `agents.toml` serves every
direction. `{source_language}` / `{target_language}` are filled from `--source-language` /
`--target-language`; an unfilled `{placeholder}` is an error, not a literal brace.

For the untranslated-echo check and everything about supporting a specific pair (same-script vs
different-script detection, adding a language, per-language hazards like honorifics, RTL, or
CJK width), see **`docs/languages.md`**. In short: same-script pairs add a profile to
`languages.toml`; different-script pairs pass `--source-script NAME` and need no profile.

## Full option reference

Every key each file accepts. Wrong types and unknown keys are errors, not silent defaults.

### `agents.toml`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `[translate].instructions` | string | *required* | the translator's system prompt |
| `[[reviewer]].id` | string | *required* | reviewer name; appears in a unit's notes |
| `[[reviewer]].instructions` | string | *required unless `from_rules`* | the reviewer's prompt |
| `[[reviewer]].from_rules` | bool | `false` | build this reviewer's prompt from the `[[advisory]]` criteria instead of `instructions` |
| `[[reviewer]].max_tokens` | int ≥ 1 | `limits.review_tokens` | this reviewer's output ceiling |
| `[[reviewer]].leniency` | table `{window, max_bad}` | `[leniency]` | per-reviewer override of the console-warning tolerance |
| `[limits].translate_tokens_per_source_char` | int ≥ 1 | `8` | translate output ceiling = source length × this |
| `[limits].translate_tokens_floor` | int ≥ 1 | `256` | lower bound on the translate ceiling |
| `[limits].translate_tokens_ceiling` | int ≥ 1 | `1024` | upper bound on the translate ceiling |
| `[limits].review_tokens` | int ≥ 1 | `1024` | default reviewer output ceiling |
| `[revision].max_revisions` | int ≥ 0 | `2` | review rounds per unit before its dispute is recorded (`0` = no review) |
| `[revision].max_repairs` | int ≥ 0 | `2` | mechanical repair attempts per generation before rejection |
| `[revision].repair_truncated_json` | bool | `true` | on a `json_object` backend (Bielik, EuroLLM), recover a reply whose JSON envelope was cut short right after the `"translation"` value closed, spending the repair budget on it like any other defect instead of rejecting the unit outright; the recovered text still goes through every mechanical check and the review panel, and can only ever be kept as `TRANSLATED` (never silently `VERIFIED`) — see `docs/feature-requests/json-envelope-truncation-repair.md`. Set `false` to lose the unit (`REJECTED`) instead of ever shipping a translation recovered this way |
| `[context].before_units` | int ≥ 0 | `3` | preceding units shown as context |
| `[context].after_units` | int ≥ 0 | `2` | following units shown as context |
| `[context].include_translations` | bool | `true` | show a neighbour's translation beside its source when memory has one |
| `[context].glossary_terms` | int ≥ 0 | `12` | most established terms shown per unit |
| `[context].context_char_budget` | int ≥ 0 | `0` | soft character budget for the continuity block beyond the guaranteed innermost neighbour on each side (`0` = off) |
| `[context].reference_examples` | int ≥ 0 | `0` | reference translations retrieved by source similarity, shown before translating (`0` = off; needs `--reference`) |
| `[context].reference_min_score` | float in `[0, 1]` | `0.30` | relevance floor a retrieved reference must clear; the default favours precision, lower to `0.15`–`0.20` for more recall (see `docs/reference-translations.md`). The *final* floor when `--hybrid` is used |
| `[context].reference_candidate_pool` | int ≥ 1 | `40` | with `--hybrid`, candidates each arm retrieves before fusion/rerank/MMR |
| `[context].reference_mmr_lambda` | float in `[0, 1]` | `0.7` | with `--hybrid`, the MMR relevance/diversity balance (`1.0` = pure relevance); swept end-to-end across `0.0`–`1.0` with **no measurable effect on consistency**, so leave it at the default — the former "raise toward `1.0`" advice is withdrawn (see `docs/reference-translations.md`, "The λ sweep — a null result") |
| `[context].reference_lexical_min_score` | float in `[0, 1]` | `0.30` | with `--hybrid`, the lexical arm's per-arm recall floor before fusion |
| `[context].reference_embedding_min_score` | float in `[0, 1]` | `0.55` | with `--hybrid`, the embedding arm's per-arm recall floor before fusion |
| `[context].reference_revision_examples` | int ≥ 0 | `0` | reference translations retrieved by target similarity to the draft, shown during revision (`0` = off) |
| `[context].reference_learn_statuses` | array ⊆ `["verified", "translated"]` | `[]` | which of this run's results seed the reference store for later retrieval (`[]` = read-only) |
| `[context].anonymous_subject` | string | `"they"` | stand-in for a context placeholder whose speaker is unknown (source language) |
| `[leniency].window` | int ≥ 1 | `20` | rolling window of recent replies per reviewer over which `max_bad` is counted |
| `[leniency].max_bad` | int ≥ 0 | `2` | unusable replies tolerated within the window before a **console** warning (the log always keeps all) |

`--max-revisions` / `--max-repairs` on the command line override `[revision]`; the language pair
comes from `--source-language` / `--target-language`, filling `{source_language}` /
`{target_language}`. `--log-file` (default `work/translator.log`) is the full, unfiltered log —
every warning and any fatal error that aborts the run; `--no-log-file` keeps only the
leniency-filtered stderr. `--reference FILE` (repeatable) supplies a reference corpus; by default
it is matched with the lexical retriever, or pass `--embedding-url http://host/v1` to use the
optional semantic retriever instead (raise `reference_min_score` to ~`0.55` for it — its cosine
scale is higher). Add `--hybrid` to fuse the lexical and embedding retrievers via Reciprocal Rank
Fusion (needs `--embedding-url` too), and `--rerank-url http://host/v1` to add a cross-encoder
reranking stage over the fused pool (needs `--hybrid`; see `tools/serve_reranker.sh`). See
`docs/reference-translations.md`.

### `translation_rules.toml`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `[limits].max_line_columns` | int ≥ 20 | `110` | soft per-line width guideline (warns; half-width columns) |
| `[limits].max_columns_tolerance` | float ≥ 0 | `0.12` | how far a unit's hard `max_columns` may be exceeded before it blocks |
| `[limits].allow_untranslated` | bool | `false` | disable the source-echo check (leave `false`) |
| `[limits].require_nonempty` | bool | `true` | a non-empty source must produce a non-empty translation |
| `[glossary].severity` | `"error"` \| `"warning"` | `"warning"` | severity when a required glossary term is absent |
| `[style].directives` | array of strings | `[]` | short policy lines added to every role's prompt (no `{}` substitution) |
| `[[forbidden]].pattern` | string (regex) | *required* | a regex rejected outright |
| `[[forbidden]].reason` | string | `""` | why, shown in the violation |
| `[[advisory]].id` | string | *required* | criterion name |
| `[[advisory]].description` | string | *required* | prose the compliance reviewer judges |

### `languages.toml`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `[languages.<code>].name` | string | the code | display name used in prompts |
| `[languages.<code>].stopwords` | array of strings | `[]` | frequent function words distinctive to this language |
| `[languages.<code>].distinctive_characters` | string | `""` | letters that occur in this language and not its counterparts |

A profile needs at least one of `stopwords` / `distinctive_characters`, or it can match nothing.

## Conventions these files share

- **Unknown keys / wrong types are errors**, not silently-ignored typos. A boolean where a
  number is expected is rejected (Python makes `bool` an `int` subclass, so `bold = 1` for a
  size would be a baffling way to fail).
- **An unfilled `{placeholder}`** in `agents.toml` is an error, never a literal brace reaching
  the model.

## Choosing a model / backend

Which model you serve is a command-line choice too. `--backend` selects how a structured request
is shaped: `generic`/`qwen` (strict `json_schema`, the default), `bielik`/`eurollm`
(`json_object`), or `auto` (pick from the model id). A reasoning model (e.g. Qwen3) is run with
its chain-of-thought **suppressed by default**, because those hidden tokens are billed against the
per-call budget and truncate the answer; `--enable-reasoning` opts back in (discouraged — raise
the token budgets if you do). See `docs/writing-a-backend.md`.
