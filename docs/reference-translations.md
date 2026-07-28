# Reference translations (external translation memory / retrieval)

This is both the design record for the feature requested in
`docs/feature-requests/external-reference-translations-rag.md` and the how-to for using it.

## The need, restated

Give the engine a body of **already-translated reference material that did not come from the
current run** — a previous run, a sister project, a vendor or human reference, or the
already-translated half of the same work — and surface the parts **relevant to each unit** into
that unit's prompt, as guidance for terminology, phrasing, register and voice. Because a
reference corpus is far larger than any prompt, the engine retrieves the most relevant entries
per unit rather than dumping the whole thing: that retrieval step is what makes this *RAG* and
not just "a bigger glossary".

## Is the requested approach right? (critical read)

Mostly yes; the request is well reasoned and maps cleanly onto the engine's existing seams. It
is a sibling of two mechanisms already here — `transunit.glossary.relevant_terms` (retrieve the
terms occurring in a unit) and `translator.memory.TranslationMemory` (reuse our own output) —
and it is injected the way a `backend` is. Where this implementation refines or diverges:

- **Keep it lexical first, and make the interface the extension point — not a vector store.**
  For translation *memory*, most of the signal is surface overlap (same words → same rendering),
  which a deterministic, stdlib-only lexical retriever captures well while preserving the
  `transunit` zero-dependency guarantee and run-to-run reproducibility. An embedding retriever
  behind the same interface (reusing the local server's `/v1/embeddings`) is also provided, as an
  *optional* translator-side module that needs an endpoint + numpy — see "The optional embedding
  (semantic) retriever" below — and the two can be fused via Reciprocal Rank Fusion, with an
  optional cross-encoder reranking stage, in "Hybrid retrieval" further down.

- **Two stores, never merged.** External reference is a **separate**, read-only store; the run's
  own `TranslationMemory` is untouched. Our own output keeps doing dedup/duplicate-sharing and
  resume-seeding exactly as before. Retrieving *external* material into the prompt is a new,
  **config-gated, off-by-default** path — turning it on never happens by accident, and turning it
  off leaves the prompt byte-for-byte as it is today.

- **The request only models `source → target` pairs. That leaves the user's real question open:
  reference text we have *without* its original.** Answered below — it needs a second retrieval
  direction, keyed on the draft rather than the source.

- **No empty boilerplate, enforced.** Like every optional section in `agents.py`, the reference
  block is added to the prompt only when retrieval actually returns something above threshold.
  With the feature off, or with no hit, nothing — no header, no "reference translations: none".

## Two stores, three uses

| Store | What it holds | Used for | Config |
|---|---|---|---|
| `TranslationMemory` (`translator.memory`) | our **own** output, this run + resumed journal | dedup / duplicate-sharing, resume seeding | always on |
| `TranslationMemory`, as context | same | translated **neighbour** lines (positional) | `[context].include_translations` (on) |
| reference retriever (`transunit.reference`) | **external** already-translated material | per-unit **retrieval** into the prompt | `[context].reference_examples` (**off**, `0`) |

The first two are our translations; the third is external and separate. Retrieval is the only
new prompt input, and it is off until asked for. Dedup never consults the reference store, and
the reference store never becomes output.

### Self-population (a writable reference)

The reference store is read-only by default, but it can be told to **learn from this run's own
accepted output**, so a later unit retrieves an earlier one by *similarity* -- not only the
exact matches dedup shares, nor only the positional neighbours context shows. It is off until
`[context].reference_learn_statuses` names which results may seed it:

```toml
reference_learn_statuses = ["verified"]               # only fully-verified lines
reference_learn_statuses = ["verified", "translated"] # also the "needs verification" ones
```

Only the two **injectable** statuses are allowed -- a rejected, skipped or pending unit has no
usable target, and offering one as reference would spread a known-bad or absent rendering
(config rejects anything else). On a resumed run the store is re-seeded from the journal, using
the same status filter, so nothing learned earlier is lost. Two consequences to know:

- **It makes retrieval adaptive, not reproducible.** What a unit sees depends on what finished
  before it, which under concurrency is completion order. That is the nature of a learning
  memory; the read-only path stays fully deterministic. This is why it is off by default.
- It needs a retrieval direction to matter (`reference_examples` or
  `reference_revision_examples` above zero) -- learning with nowhere to surface is warned about.

## Retrieval, two directions

A reference entry is `(source, target)` where **`target` is required** and **`source` is
optional**. That single shape supports both the bilingual case and the user's target-only case,
because retrieval can be keyed on either field:

### By source — before translating (the primary use)

Keyed on the **unit's source**, over entries that *have* a source. Surfaces prior renderings of
similar source lines as worked examples before the model writes anything. This is the classic
"match the existing half-translation" case, and it is what `reference_examples` turns on.

### By target — while revising (answering "text with no original")

Text translated **without its source** cannot be matched by source similarity — so it is matched
by **target** similarity instead, against the model's own **draft**. After a first attempt
exists, the engine retrieves target-side entries similar to that draft and shows them during
revision as established renderings to align phrasing, terminology and register with. This is how
monolingual target reference earns its keep: it cannot seed a translation, but it can pull a
draft toward the house voice. It is a distinct knob, `reference_revision_examples`, also off by
default. (Bilingual entries participate here too — indexed on their target — so a corpus need not
be split by whether it has sources.)

**The limitation to know:** because it keys on the draft, target-side retrieval only fires on a
unit that is **revised at least once** — a unit the panel accepts on the first pass has no draft
to match, so a *purely* target-only corpus never reaches it. In practice that is often the right
units (the ones a reviewer objected to are exactly where house-style alignment matters), but it
is a genuine gap versus bilingual reference, whose by-source path fires on every unit's first
pass. If you have the originals, supply them — you get both directions; if you only have the
translated half, target-only still helps, just concentrated on the revised units.

Source-only entries (a source with no target) are *not* translation examples and are dropped on
load with a warning; they can steer nothing.

## Why not just reuse the existing seams

Walked against `a8dc518`: `TranslationMemory` is run-local and **exact-match** (it never sees
external material and cannot surface a *similar* line); neighbour context is **positional, not
relevant**; the glossary is **exact term pairs**, not sentence/phrase exemplars; and style
directives are static prose, not per-unit examples. None is closable by config or by the carrier
adapter, which has no hook into prompt assembly. Hence an engine change.

## The lexical retriever

`transunit.reference.LexicalRetriever`. Deterministic, thread-safe (immutable after
construction), stdlib-only.

- **Similarity:** TF-IDF cosine over **character n-grams** (default `n = 3`). Character n-grams
  are deliberately script-agnostic — they work for space-delimited and non-space-delimited
  scripts alike (so Chinese and Japanese are handled like English), and they reward the
  morphological overlap that word matching misses in inflected languages (Polish *Praga/Pragi*).
- **Placeholders** (`[[0]]`) are stripped before scoring, so shared engine tokens do not inflate
  similarity, and are neutralised to the context stand-in when an example is rendered — the same
  handling neighbour context gets, so an exemplar never teaches the model to emit a placeholder a
  line does not have.
- **Bounded and floored:** every query returns at most `k` results, each at or above a similarity
  floor (`reference_min_score`), so an off-target corpus contributes nothing rather than noise —
  the failure mode the request warns about. A wrong exemplar is worse than none, so the floor
  errs toward under-retrieval.
- **Determinism:** shingles are summed in sorted order and ties break on corpus position, so the
  same corpus and query always yield the same top-k, across runs and across the concurrent
  workers that share one retriever. (Self-population is the documented exception: a learning
  store's results depend on insertion order.)
- **Efficient on a large corpus:** an inverted index makes a query proportional to the postings
  of its shingles, not to corpus size, and norms are precomputed so scoring a match is one
  division — ~5 ms/query against 50k entries with realistic vocabulary. The self-populating
  variant keeps this by *freezing* idf at the seed: `add` is O(the entry's shingles) and never
  triggers a full re-index, the trap a naive growable index falls into.
- **Guidance, not answers:** retrieved targets are shown to steer only. The mechanical checks and
  the review panel still decide the output, exactly as without a corpus.

## When it helps, and when it hurts (measured)

The feature's job is **consistency** — make new lines match an existing body of translation.
That is not the same as raising per-line quality, and a blind A/B of the two (a Japanese→English
game, ~60 fresh lines, an external LLM judging each pair for faithfulness and naturalness)
separates the two effects clearly:

- **Quality tracks the corpus, not the switch.** Reference faithfully pulls the output toward the
  corpus's terminology and phrasing. So the sign of the quality change is roughly *(corpus quality
  − what the model would write unaided)*: with a reference corpus **worse** than fresh output
  (here, an older machine translation), the judge preferred **no-reference ~2:1**; with a corpus
  at **fresh-model quality**, reference was **neutral-to-slightly-better**. Garbage in, garbage
  out — a good corpus is the whole game. This is the flip side of "guidance, not answers".
- **Consistency is delivered regardless.** On **duplicate** lines — lines whose exact match is in
  the corpus — the reference run reproduced the established rendering **~97%** of the time versus
  **~17%** without it. That is exactly the point of the feature (and exactly what a per-line
  quality judge does *not* reward), so judge it on consistency with your corpus, not on an isolated
  quality score. (Scope: this is the *duplicate-line* case, where the corpus contains the answer.
  It says nothing about which *retriever* is better at consistency — that comparison was measured,
  got a wrong answer from a biased metric, and is retracted; see "The consistency finding was an
  artifact" below.)
- **The floor is the relevance dial, and the score is a reliable relevance signal.** An
  independent audit (1000 unique, non-duplicate query lines; each top match rated for relevance
  by a separate LLM) found precision climbs steadily with the score, so the score genuinely
  predicts relevance:

  | similarity score | share the judge called clearly relevant | share it called unrelated (noise) |
  |---|---|---|
  | below 0.15 | ~17% | ~31% |
  | 0.15–0.20 | ~40% | ~17% |
  | 0.20–0.25 | ~49% | ~11% |
  | 0.25–0.30 | ~49% | ~6% |
  | **0.30–0.40 (default floor)** | ~74% | ~0% |
  | 0.40+ | ~93–100% | ~0% |

  **The default `reference_min_score = 0.30` is set from this: precision only becomes high
  (~three quarters clearly relevant, negligible noise) at ~0.30, so the default shows only
  strong matches.** A lower floor admits *marginal* ones — at 0.15, roughly two in five
  retrievals are clearly relevant and about one in six is outright noise (lines sharing a word or
  particle, not meaning). That is a precision/recall trade: the price of the clean default is
  recall — on this data ~7% of lines retrieve something at 0.30 versus ~47% at 0.15. **Lower the
  floor to 0.15–0.20 when you want the feature to fire on more units and can tolerate weaker
  exemplars; raise nothing above 0.30 unless you want near-duplicates only.** A wrong exemplar is
  worse than none, so tune it to the point where what you see retrieved is material you would be
  happy for the model to imitate.
- **It never pads to fill the slots.** `reference_examples = 3` is a *ceiling*: a query returns
  only the entries above the floor — three, one, or none. A unit with nothing relevant gets no
  reference section at all, not three weak ones.

The by-target **revision** path (`reference_revision_examples`) is the more aggressive of the two:
it rewrites an already-acceptable draft toward corpus phrasing, and in the same test it added a
little more harm than by-source alone. Turn it on when aligning a draft to a house voice is worth
that risk; leave it off (the default) otherwise.

## Using it

```bash
PYTHONPATH=src python -m translator.cli \
    --base-url http://127.0.0.1:11434/v1 --model qwen3:14b --backend auto \
    -c work/units.jsonl -j work/journal.jsonl \
    --source-language en --target-language pl \
    --reference reference/prior-translation.jsonl \
    --reference reference/vendor-tm.jsonl
```

`--reference` may be repeated; corpora are concatenated. Each file is JSON Lines. Two shapes are
accepted, and may be mixed:

- a minimal reference entry — `{"source": "...", "target": "..."}` (omit `source` for
  target-only);
- a `transunit` unit/journal line — the same records the engine writes; a prior run's
  `journal.jsonl` is a reference corpus as-is. Pending/rejected lines (no injectable target) are
  skipped automatically.

Then turn retrieval on in the agent config (it is off until you do):

```toml
[context]
reference_examples = 3               # by-source exemplars before translating (0 = off); a ceiling, not a quota
reference_min_score = 0.30           # relevance floor in [0, 1]; default 0.30 = high precision; lower to 0.15-0.20 for more recall
reference_revision_examples = 0      # by-target exemplars during revision (0 = off); the more aggressive path
reference_learn_statuses = []        # e.g. ["verified"] to self-populate; [] = read-only

# Only consulted with --hybrid (see "Hybrid retrieval" below); harmless to leave at these
# defaults otherwise. context_char_budget applies to every run.
context_char_budget = 0              # 0 = off: no trimming beyond the counts above
reference_candidate_pool = 40        # candidates each arm retrieves before fusion/rerank/MMR
reference_mmr_lambda = 0.7           # MMR relevance/diversity balance; 1.0 = pure relevance.
                                     # Swept end-to-end across 0.0-1.0: no measurable effect on
                                     # consistency (see "End-to-end (Phase B)")
reference_lexical_min_score = 0.30   # the lexical arm's own per-arm recall floor
reference_embedding_min_score = 0.55 # the embedding arm's own per-arm recall floor
```

A corpus is optional when `reference_learn_statuses` is set: with no `--reference`, the store
starts empty and learns entirely from this run's own accepted output.

Retrieval shares the prompt window with neighbour context and the glossary, so size
`reference_examples` the way you size `before_units`/`after_units`: a few short exemplars, not a
flood. Pass `--context-window N` and the engine warns once, before the wall, if the compounded
prompt (reference included) plus its output budget nears 80% of the window. See
`config/README.md` for the full option table.

## The optional embedding (semantic) retriever

The lexical retriever matches on the *surface* of a line — shared character n-grams. That misses
paraphrases and cross-script equivalents: Japanese 猫 (kanji) and ネコ (katakana) both mean "cat"
but share no characters, so lexical similarity is ~0.
`translator.retrieval.embedding.EmbeddingRetriever` (a back-compat shim keeps the old
`translator.embedding` import path working) is an **optional** drop-in behind the same
`Retriever` seam that matches on *meaning* instead — cosine over sentence embeddings from a local
OpenAI-compatible `/v1/embeddings` endpoint.

It is optional because it needs two things the lexical default avoids: an embedding server, and
`numpy` (a pinned but lazily-imported dependency, so the core engine never loads it — only building
an embedding retriever does). Turn it on:

```bash
tools/serve_embeddings.sh                                  # EmbeddingGemma-300M on :8081, alongside your translation server
PYTHONPATH=src python -m translator.cli ... \
    --reference reference/prior-translation.jsonl \
    --embedding-url http://127.0.0.1:8081/v1               # <- selects the semantic retriever
# and, in the agent config, raise the floor -- embedding cosines cluster higher than the lexical scale:
#   [context]
#   reference_examples = 3
#   reference_min_score = 0.55        # ~0.55 for embeddings, not the lexical 0.30
```

**Measured** on the same Japanese game corpus (11k lines — "corpus A" in the end-to-end results
below; 150 unique queries, an LLM judging each retrieval's relevance): the embedding retriever
lifted top-1 relevance from **~37% to ~49%**, won
the head-to-head **47 to 20**, and — most usefully — found a **clearly-relevant** match on **~43%
of queries where the lexical floor returned nothing** (the cross-script/paraphrase cases). It is
**read-only** (self-population is the lexical retriever's job; combining `--embedding-url` with
`reference_learn_statuses` is refused), and its cosine scale differs, so tune its floor separately.

**End-to-end, on two independent real corpora** — call them **corpus A** (a ~11k-line
Japanese→English game corpus, the *noisier* of the two) and **corpus B** (a ~2.8k-line
Japanese→English game corpus, the *cleaner* one) — translating held-out lines with vs without the
embedding reference, a blinded order-randomized Qwen A/B judge: per-line *quality* was a wash —
near-tie on the cleaner corpus B (embedding 12, noref 11, tie 1) and slightly *behind* on the
noisier corpus A (noref 28, embedding 24, tie 5), because that corpus is not higher quality than
fresh generation. *Consistency*, the actual job, moved in the reference's favour on both — the
reference-steered output matched what the harness called the established rendering far more often
than the reference-free output did.

> **Correction (2026-07-27).** The multipliers this paragraph used to quote (**~4×**; 33% vs 9%
> and 54% vs 17%; overlap up ~1.5×) came from the *biased* consistency instrument described under
> "The consistency finding was an artifact" below: it defined "the established rendering" as what
> an embedding retriever fetches, which is the very thing the reference arm was shown. **The
> direction survives and is near-tautological** — show a model corpus text and it reuses some of
> it — **but the magnitude is not a measurement and has been withdrawn.** Everything else in this
> paragraph (the quality comparison, judged blind with no retriever involved) stands.

#### Re-measured at n=1000: what turning the feature ON actually costs and buys

The paragraph above rests on 57 and 24 judged pairs. The same comparison was later run through
`tools/eval_translation.sh` at **1000 held-out units** on corpus A (10,952 entries, ja→en),
baseline = **no reference at all**, candidate = **embedding retrieval**:

| metric | no reference | embedding reference |
|---|---|---|
| verified | **718** | 611 |
| needs review (translated) | 252 | 301 |
| **rejected** | **30 (3.0%)** | **88 (8.8%)** |
| blinded A/B wins | 165 | 140 |
| A/B ties | 523 | 523 |
| ~~closer to the established rendering~~ (of 894 having one) | ~~90~~ | ~~272~~ |
| ~~mean trigram overlap with the established rendering~~ | ~~0.092~~ | ~~0.122~~ |

(904 units compared, 76 identical under both arms, 828 differing, 0 judge errors.)

**The two struck rows are withdrawn.** They were produced by the biased instrument documented in
"The consistency finding was an artifact" below — one of the two arms here *is* the embedding
retriever, which is the exact configuration in which the metric scores an arm against its own top
pick. The rows are left visible, struck, so a reader who saw the earlier numbers can find out what
happened to them. The rejection, verified and A/B rows are unaffected: they come from plain journal
status counts and a blinded judge that no retriever touches.

**Quality: still a wash, and if anything it leans the other way.** 165 to 140 among 305 decisive
verdicts is **z ≈ 1.4 — not significant**; the honest reading is "no measurable per-line quality
difference", with a point estimate mildly favouring *no* reference. This reproduces the small-n
result at 17× the sample. **Reference retrieval does not make translations better.**

**Consistency: direction only — the "~3×" is withdrawn.** This paragraph used to report 272 units
to 90 and "mean overlap up ~1.3×" as the feature's one clear win. Both figures came from the biased
instrument and are not a measurement of consistency with the corpus. What remains defensible is the
*direction*, and it is close to a tautology: a model shown corpus text in its prompt reuses some of
that text. **Do not quote a multiplier.** If you need to know how much consistency retrieval buys
on your corpus, measure it against held-out ground truth (`eval_translation.sh --ground-truth`);
that comparison has not been re-run for reference-on versus reference-off.

**Robustness: a real, previously unquantified cost.** Turning the feature on **nearly tripled the
rejection rate**, 3.0% → 8.8%, and cost 107 verified units. The small-n study saw a hint of this
("four lines … were rejected with it, zero the other way") but could not size it. It is the same
corpus-defect mechanism described above — a corrupt row presented as an authoritative example
destabilises generation — and at scale it is the feature's dominant cost. A rejected unit means
the untranslated source is shown, so this is not a rounding error. (The old summary here — "for
every ~3 units the reference makes consistent, it loses roughly 1 to rejection" — is withdrawn with
the multiplier it was built on. The 5.8-percentage-point rejection cost is what was measured; the
consistency side of that ratio was not.)

Read together with the Phase B comparison below, the picture is: reference retrieval costs
robustness on a dirty corpus, and a reranker buys some of that back — but only on a corpus dirty
enough to need it. It does **not** cost consistency in exchange: measured against real held-out
ground truth, the two retrievers are indistinguishable on consistency.

The same runs sharpen the **corpus-quality caveat**, and the mechanism is worth spelling out.
Four lines that translated cleanly without the reference were **rejected** with it (zero went the
other way), from two distinct causes:

- *A single corrupt row.* One corpus A entry has a leaked `"line_break": false}` JSON fragment
  baked into its target (18 of 11,215 targets carry such garbage). Its source is short, so it
  embeds near many short interjections — a "short-line magnet" retrieved for three of the four
  failures. On one, the model **copied the JSON tail into its own output** (then derailed into a
  `</think>` trace leak); on the others it destabilised generation (a stray newline; a repetition
  loop). One bad row, presented as an authoritative example, poisons every short line near it.
- *House style colliding with a hard invariant.* On the fourth (a substantive line), the retrieved
  examples were clean and on-topic, and they taught a real house style: name the character with its
  `[[0]]` placeholder where the source uses a bare pronoun. The model faithfully applied it — and
  thereby emitted a **second `[[0]]` the source unit does not contain**, tripping placeholder
  integrity. The reference did its job; its job conflicted with a mechanical rule.

The harness caught every one (they never shipped), but the lesson stands: a reference propagates
its corpus's defects as faithfully as its virtues, and consistency-steering can even fight the
mechanical checks. Clean the corpus first.

When to use which: **lexical** (the default) for near-duplicate/terminology reuse with zero setup
and full reproducibility; **embedding** when your corpus says the same thing in different words, or
across scripts, and you can run an embedding endpoint; **hybrid + reranker** (below) when your
corpus is known to contain corrupt rows, or you measure a high rejection rate. Between embedding
and hybrid+rerank, **robustness on your own corpus is the only measured differentiator** — quality
is a wash and consistency shows no measurable difference. See "End-to-end (Phase B)" for the
numbers, and for the consistency claim this document used to make here and has since retracted.

## Hybrid retrieval: fusing lexical + embedding, with optional reranking

Lexical and embedding retrieval are complementary, not competing: lexical is strong on exact
terms, names, and IDs (things an embedding can blur past); embedding is strong on paraphrase and
cross-script equivalence (things lexical is blind to). `--hybrid` fuses both into one retriever
(`translator.retrieval.hybrid.HybridRetriever`) rather than choosing one:

```bash
tools/serve_embeddings.sh                                  # dense arm, :8081
tools/serve_reranker.sh                                    # optional reranking stage, :8082
PYTHONPATH=src python -m translator.cli ... \
    --reference reference/prior-translation.jsonl \
    --embedding-url http://127.0.0.1:8081/v1 --hybrid \    # fuse lexical + embedding
    --rerank-url http://127.0.0.1:8082/v1                  # optional: add reranking
```

**The algorithm**, per query: each arm (lexical, embedding) retrieves its own top
`reference_candidate_pool` candidates, gated by its own calibrated recall floor
(`reference_lexical_min_score` / `reference_embedding_min_score` — the two are on different score
scales, so each keeps its own). The two rankings are fused with **Reciprocal Rank Fusion**: a
candidate both arms agree on outranks one only one arm found, without needing to calibrate the
arms' otherwise-incomparable raw scores against each other. If a reranker is configured
(`--rerank-url`), the fused candidate pool — never the whole corpus, only the short list fusion
already narrowed to — is re-scored by a cross-encoder (a much more accurate but far more expensive
judge than either first-stage arm, affordable only because it never sees more than a short list),
and its raw logit is squashed to `[0, 1]` by a sigmoid. Without a reranker the fused score is
divided by the **largest score fusion can produce** (both arms ranking a candidate first) — a
*fixed* denominator, so the resulting `[0, 1]` scale means the same thing for every query, and
`reference_min_score` stays the absolute floor it is everywhere else. (It is deliberately *not*
min-max normalised over the query's own candidates: that would score the best of three terrible
candidates 1.0 and always drop the worst of three excellent ones.) Either way the result is one
final relevance score, gated by `reference_min_score` exactly as the single retrievers are.

That fixed scale makes the floor a genuinely useful dial without a reranker: a candidate **both**
arms rank first scores `1.0`, one that only a single arm found tops out at `0.5`, and one deep in
a single arm's ranking scores lower still — so **raising `reference_min_score` above `0.5` keeps
only candidates both arms agree on**, and a query where they disagree returns nothing rather than
one arm's best guess.
The survivors are then diversified with **Maximal Marginal Relevance** (`reference_mmr_lambda`,
`1.0` = pure relevance, the conventional relevance-leaning default `0.7`), so a shortlist is not
dominated by several near-identical examples. This document used to warn that the diversification
costs consistency, on the argument that near-identical examples are what reproducing an established
rendering needs. **That was an inference, and a sweep across the whole `0.0`–`1.0` range falsified
it** — see "The λ sweep" under "End-to-end (Phase B)". Leave `reference_mmr_lambda` at its default
unless you have your own measurement saying otherwise.

**Fails loud, like the plain embedding retriever.** A down embedding endpoint fails at
construction; a down rerank endpoint fails out of the retrieval call mid-run — there is no silent
fallback to the fused-but-unreranked order. Read-only, like the embedding retriever: combining
`--hybrid` with `reference_learn_statuses` is refused.

**The reranker** (`translator.retrieval.rerank.RerankClient`) talks to any Jina/TEI-compatible
`/v1/rerank` endpoint; `tools/serve_reranker.sh` serves one via the same llama.cpp build already
in use, downloading **bge-reranker-v2-m3** (**Apache-2.0** licensed, multilingual) on first use —
verify the exact repo/quant in the script if you want a different one.

### Measured — and the result is a caution, not a win

Run with `tools/eval_retrieval.sh` on the same two independent real Japanese game corpora —
**corpus A**, the noisier one (11,215 lines, 150 held-out queries), and **corpus B**, the cleaner
one (2,843 lines, 120 held-out queries) — using unique, non-corpus-duplicate queries with each
variant's top-1 rated for relevance by an independent LLM:

| variant | corpus A | corpus B |
|---|---|---|
| lexical alone | 58% | 94% |
| **embedding alone** | **78%** | **98%** |
| hybrid (RRF fusion, no reranker) | 57% | 88% |
| hybrid + reranker | 77% | 96% |

(The two corpora sit at very different absolute difficulty — corpus B's held-out lines are closer
to its corpus — but the *ordering* is identical in both, which is the point.)

**Bare hybrid did not beat its own best arm — it landed near its *worst* one.** This is not a bug;
it is what equal-weight rank fusion does when the arms differ sharply in quality. The per-query
breakdown shows the mechanism exactly: the arms disagreed on 50 of 145 queries, and on those the
embedding arm was right 41 times to the lexical arm's 9 — a 4.6:1 quality gap. RRF weights both
rankings equally, so bare hybrid followed the better arm on only 24 of those 50 (a coin flip, as
equal weighting must be). It therefore **gave up 23 of the embedding arm's 41 wins to gain at most
9** — a bad trade whenever one arm is much stronger, and the reason fusion *blends* its arms
rather than taking their maximum.

The reranker is what makes fusion safe: it recovered 23 of those 41 embedding wins (bare hybrid
recovered 18) *and* kept 7 of the 9 the lexical arm uniquely found — landing at 77%, level with
the embedding arm alone.

On corpus B the same ordering appears with a second mechanism visible: bare hybrid (88%) fell below
**both** arms (94% and 98%), not merely between them. RRF rewards *agreement*, and a candidate
both arms rank second outscores one that a single arm ranks first — so where both arms are
already strong, fusion can promote a mutual compromise over either arm's own best pick. Agreement
is a good signal when arms are independent and comparable; it is not a substitute for being right.

**So, concretely, on this kind of corpus — for *retrieval relevance*, which is what this phase
measured:**

- **Use the embedding retriever alone**, or **`--hybrid` with `--rerank-url`**. They measured the
  same *here*; the single arm is cheaper (two servers, not three). This is a statement about
  retrieval relevance only — end-to-end the two differ on **robustness**, in a corpus-dependent
  way; see "End-to-end (Phase B)" immediately below before choosing.
- **Do not use bare `--hybrid`** unless you have checked, on *your* corpus, that the two arms are
  comparably strong. Run `tools/eval_retrieval.sh` and look at the per-arm numbers first — that is
  exactly what it is for.
- Expect a different balance where the lexical arm is genuinely strong — exact terminology, part
  numbers, identifiers, proper nouns. Japanese game dialogue, full of short interjections whose
  character trigrams overlap by coincidence, is close to the worst case for it. **We have not
  measured such a corpus, so that is a hypothesis, not a claim.**

The obvious next step, now that there is evidence motivating it rather than speculation, is
**weighted fusion** — letting a known-stronger arm carry more weight than a known-weaker one. It
is deliberately *not* built yet: picking a default weight needs its own evaluation across more
than one corpus, and an untuned knob would be worse than none.

### End-to-end (Phase B) — robustness is the only measured differentiator

The table above ranks *retrieval relevance*: does the retriever fetch material a judge calls
relevant? Phase B ranks what actually ships: translate a held-out set under two retrievers and
compare the **output**. It was run for the first time with `tools/eval_translation.sh`, on real
corpus A Japanese→English text — 10,952 entries, **1000 held-out units**, baseline =
**embedding alone**, candidate = **hybrid + reranker** (the other Phase A co-winner):

| metric | embedding alone | hybrid + rerank |
|---|---|---|
| verified | 606 | 674 |
| needs review (translated) | 309 | 296 |
| **rejected** | **85 (8.5%)** | **30 (3.0%)** |
| blinded A/B wins | 147 | 161 |
| A/B ties | 551 | 551 |
| ~~closer to the established rendering~~ (of 958 units having one) | ~~222~~ | ~~132~~ |
| ~~mean trigram overlap with the established rendering~~ | ~~0.120~~ | ~~0.111~~ |

(Of the 1000 units, 120 produced *identical* renderings under both arms and 859 differed; the
judge returned 0 errors. The 551 ties are among the differing units.)

> ### The consistency finding was an artifact — retracted 2026-07-27
>
> **What this document used to say.** That the embedding arm is meaningfully *better at
> consistency* — the metric the whole feature exists to serve — 222 units to 132 on corpus A and
> 193 to 122 on corpus B; that the cause was MMR diversification in the hybrid path; and that a
> consistency-focused project should therefore prefer the embedding arm and raise
> `reference_mmr_lambda` toward `1.0`. **All of that is withdrawn.** The struck rows above and in
> the corpus B table below are left visible on purpose, so a reader who saw them can find out why
> they are gone.
>
> **The bias.** `consistency()` defined "the established rendering" as the most similar corpus
> entry found with an `EmbeddingRetriever` at floor `0.55` — which is *exactly* what the
> `embedding` arm is configured to retrieve and place in the model's prompt. Measured on 297 real
> queries: the embedding arm had been shown that very entry as its own top example on **297 of 297
> queries**; the hybrid arm on **93 of 297**. The similarity between the embedding arm's top
> example and the text it was then scored against was **1.000, by construction**. The metric was
> not measuring agreement with the corpus. It was measuring agreement with one arm's own top pick,
> and that arm wins by definition.
>
> **The corrected measurement.** Ground truth is each held-out unit's **own original translation**
> — available because the queries were held out of an already-translated journal, so neither arm
> could have seen them. On corpus B, n = 593:
>
> | | mean overlap with the real original |
> |---|---|
> | embedding alone | **0.5928** |
> | hybrid + rerank | **0.5874** |
>
> Closer to the original: **embedding 223, hybrid 222, tie 148** — 445 decisive, **z = 0.05, not
> significant**. A dead heat. **There is no measurable consistency difference between the two
> retrievers.** (This correction was run on corpus B, where held-out ground truth exists in the
> shape the split produces; the corpus A consistency numbers are withdrawn without a replacement,
> not re-measured.)
>
> **Why the replication did not save it.** The original finding replicated across two independent
> corpora at almost the same ratio, and that replication was cited as the strongest evidence in the
> whole evaluation. It was not corroboration: **a structural bias reproduces perfectly.** Two runs
> that share an instrument share its errors, so replication only corroborates when the instrument
> is independent of what it measures. A biased number that reproduces cleanly is *more* dangerous
> than one that does not, because reproducing is what makes it persuasive.
>
> **What the harness does now.** `consistency()` takes `--ground-truth` and **refuses** to run the
> retriever-derived proxy while either arm is `embedding`, rather than warning about it. Refusing
> beats warning here because the failure is silent and the output looks good: the proxy still
> produces a plausible, well-formed, reproducible number that a reader has no way to distinguish
> from a sound one. A warning is a note beside a number; this needed the number not to exist. (With
> ground truth supplied, no retriever is built at all — the measurement is arithmetic over two
> journals and a file, and needs no server.)

**What Phase B actually established. Two things, not three.**

1. **Per-line quality is a wash** — 161 to 147 with 551 ties out of 859 differing units. That
   *confirms* the standing expectation (stated since the very first A/B above), now at n=1000
   rather than n=57. This one is untouched by the retraction: it was judged blind, by an
   independent judge, with no retriever involved. Nobody should choose a retriever on this number.

2. **Hybrid + rerank is markedly more robust — but only on a corpus that needs it.** On the
   noisier corpus A, rejections fell from 85 (8.5%) to 30 (3.0%) and verified rose 606 → 674. This
   is consistent with the "short-line magnet" failure documented above: the embedding arm retrieves
   a corrupt corpus row for a short input and destabilises generation; the cross-encoder filters
   that row out. **The corpus B replication below shows this benefit does not generalise** — it is
   a function of how dirty the corpus is, not a property of the reranker. These are plain journal
   status counts, so the retraction does not touch them either.

**The recommendation, therefore, is simpler than it used to be — and smaller:**

- **Quality is a wash. Consistency shows no measurable difference. Robustness is the only
  measured differentiator, and it is corpus-dependent** (rejections 8.5% → 3.0% on the noisier
  corpus A; 0.5% → 0.8%, i.e. nothing, on the cleaner corpus B).
- So: **pick a retriever on robustness measured on *your* corpus.** Run `eval_translation.sh` and
  look at your own rejection rate. If it is high — or you know your corpus contains corrupt rows —
  **`--hybrid --rerank-url`** is worth its third server. If it is already low, **embedding alone**
  is cheaper and measured no worse on anything.
- **Do not choose between them on consistency.** That was the old advice and it rested on a broken
  metric.
- **Bare `--hybrid`** (no reranker) remains **not recommended**, on the Phase A evidence above.

#### The λ sweep — a null result

`tools/sweep_mmr.sh` swept `reference_mmr_lambda` end-to-end on corpus A (**n = 300**, six values
spanning `0.0` to `1.0`, plus an embedding-arm reference point). It was run *before* the bias above
was found, to test the standing claim that MMR diversification costs consistency.

**Mean overlap was flat: 0.098 to 0.106 across the whole range — a spread of 0.0086, non-monotone,
no trend.** `λ = 1.0`, with the diversity penalty fully off, was no better than `λ = 0.0`. So MMR
was never the cause of the apparent consistency gap — which is consistent with there being no real
gap at all, as the ground-truth measurement then showed.

**The practical consequence:** `reference_mmr_lambda` has now been measured across its full range
and **no effect on consistency was observed**. The advice to raise it toward `1.0` is withdrawn
from this document, `config/README.md` and `config/agents.toml`. Leave it at the default.

(Honest caveat on the sweep's own instrument: it scored each λ against the same retriever-derived
proxy, so its *absolute* overlap values sit on the biased yardstick too. The proxy is identical for
every λ, though, so it cannot manufacture flatness across them — the null result stands as a
comparison between λ values, which is what it was run to answer.)

#### Replicated on a second corpus — what that did and did not show

The same comparison on corpus B (2,422 entries, **600 held-out units**, ja→en), the cleaner of the
two corpora:

| metric | embedding alone | hybrid + rerank |
|---|---|---|
| verified | 467 | 462 |
| needs review (translated) | 130 | 133 |
| rejected | 3 (0.5%) | 5 (0.8%) |
| blinded A/B wins | 120 | 124 |
| A/B ties | 253 | 253 |
| ~~closer to the established rendering~~ (of 593 units having one) | ~~193~~ | ~~122~~ |
| ~~mean trigram overlap with the established rendering~~ | ~~0.394~~ | ~~0.378~~ |

(593 units compared, 96 identical under both arms, 497 differing, 0 judge errors.)

- **Finding 1 (quality is a wash) replicates.** 124 to 120 with 253 ties. Independent instrument,
  unaffected by the retraction.
- **Finding 2 (hybrid+rerank loses consistency) is RETRACTED, not replicated.** The struck rows
  came from the biased metric; on the same 593 units, measured against each unit's own original
  translation, the arms are a dead heat (223 / 222 / 148 tie, z = 0.05). The apparent replication
  was the artifact reproducing, not the finding confirming.
- **Finding 3 (robustness) does NOT replicate.** Corpus B's rejections were 0.5% against 3.0%–8.5%
  on corpus A, and hybrid+rerank was *marginally worse* (0.8% vs 0.5%). That is the honest reading:
  the reranker's job is filtering corrupt corpus rows, so **a corpus without corrupt rows has
  nothing for it to filter**, and the second server buys nothing. The win on corpus A was a
  property of that corpus, not of the retriever — which is exactly why the plan called for two
  corpora, and a single-corpus result would have been published as a general robustness claim.

**Scope of what survives, stated plainly:** two corpora, one direction (ja→en), one comparison
(embedding vs hybrid+rerank). Quality is a wash on both. The consistency comparison, re-run against
real held-out ground truth on corpus B, is a dead heat. The robustness difference is
corpus-dependent and must be measured per project, not assumed. `reference_mmr_lambda` has been
swept across `0.0`–`1.0` with no effect observed.

## What was deliberately *not* built

- **A vector index / ANN.** The embedding retriever scores the whole corpus per query (fast enough
  for tens of thousands of entries); a large-scale ANN index is a further drop-in behind the seam,
  not built here.
- **A default-on change to *retrieval*.** All of `--embedding-url`, `--hybrid` and `--rerank-url`
  stay off unless asked for, exactly as the lexical-only baseline always was. (Positional context's
  *rendering*, unlike retrieval, did become a default-on change — one continuous passage instead
  of two labelled sections — because it is not RAG; see `docs/architecture.md` and the migration
  note in `config/README.md`.)
