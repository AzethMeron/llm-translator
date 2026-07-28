# Audit — Phase B end-to-end evaluation, branch coverage, and property testing (2026-07-26)

Follows `2026-07-26-hybrid-retrieval-and-context-packing.md`. That session measured which
retriever *fetches* the best material; this one measured which retriever produces the best
**output**, and hardened the code the measurement runs through. Both results are uncomfortable
and are recorded as they came out.

> **CORRECTION, 2026-07-27 — the headline below is wrong, and the way it was wrong is the most
> useful thing in this file.** The consistency finding ("the embedding arm is better at
> consistency") was an artifact of a circular metric. It is retracted in full, along with the MMR
> mechanism offered to explain it and the `reference_mmr_lambda` advice derived from it. The
> quality results, the rejection/verified counts and the Phase A relevance table are unaffected.
> Read "The correction" at the end of this section before using any consistency number here.

## The headline: the retrieval winner is not the end-to-end winner

`tools/eval_translation.sh` (built this session, the "Phase B" the plan specified but never ran)
translated 1000 held-out ja→en units of **corpus A** (a ~11k-line Japanese→English game corpus, the
noisier of the two used here) against a 10,952-entry corpus, baseline **embedding alone** versus the
Phase A winner **hybrid + reranker**.

| metric | embedding | hybrid+rerank |
|---|---|---|
| verified | 606 | 674 |
| needs review | 309 | 296 |
| rejected | 85 (8.5%) | 30 (3.0%) |
| blinded A/B wins | 147 | 161 |
| ~~closer to the established rendering~~ (of 958) | ~~222~~ | ~~132~~ |
| ~~mean trigram overlap with the established~~ | ~~0.120~~ | ~~0.111~~ |

It measured a **trade, not a win**.

- **Per-line quality is a wash** — 161 to 147 with 551 ties out of 859 differing units (120 units
  rendered identically under both arms; 0 judge errors). This confirms the plan's own stated
  expectation, now at n=1000 rather than the earlier n=57. *(Stands. Blinded independent judge, no
  retriever in the loop.)*
- ~~**The Phase A winner loses on consistency**~~ — **RETRACTED, see below.** As written: 132
  versus 222, with lower overlap, attributed to MMR diversification
  (`reference_mmr_lambda = 0.7`) suppressing the redundancy that reproducing an established
  rendering requires, plus per-arm floors differing by design (embedding cosine 0.55 versus the
  fused/reranker 0.30). Both the finding and its mechanism turned out to be wrong.
- **Hybrid+rerank is markedly more robust** — rejections 8.5% → 3.0%, verified 606 → 674.
  Consistent with the already-documented "short-line magnet" failure: the dense arm retrieves a
  corrupt corpus row for a short input and destabilises generation; the cross-encoder filters it
  out. *(Stands. Plain journal status counts.)*

**Both corpora ran.** Corpus A took 10,952 corpus / 1000 held-out; **corpus B** (a ~2.4k-line
Japanese→English game corpus, the cleaner one) took 2,422 / 600 (its own sizes, because it repeats
about a third of its lines and only 897 rows were ever eligible as held-out queries). The
robustness win did **not** replicate: corpus B's rejections were 0.5% against 0.8%, where corpus A
saw 8.5% → 3.0%. The reranker filters corrupt corpus rows, so a clean corpus has nothing for it to
filter. Had only the noisier corpus A run, "hybrid+rerank is markedly more robust" would have gone
into the docs as a property of the retriever rather than of that corpus. That is the whole argument
for the plan's two-corpus rule — and it worked, here.

The consistency loss *did* "replicate" — 222 vs 132 on corpus A, 193 vs 122 on corpus B — and that
replication was written up as the strongest evidence in the whole evaluation. It was the weakest.

## The correction: a metric that measured itself

The headline consistency finding is **withdrawn**. It was never a property of the retrievers.

**The bias.** `consistency()` defined "the established rendering" as *the most similar corpus entry
found with an `EmbeddingRetriever` at floor 0.55*. That is, to the character, what the `embedding`
arm is configured to retrieve and paste into the model's prompt. Measured on 297 real queries:

| | shown the "established" entry as its own top example | similarity of its top example to the thing it was scored against |
|---|---|---|
| embedding arm | **297 / 297 (100%)** | **1.000, by construction** |
| hybrid + rerank arm | **93 / 297 (31%)** | 0.314 |

So the metric did not measure agreement with the corpus. It measured **agreement with one arm's own
top pick**, and that arm wins by definition. The MMR explanation was invented to account for a gap
that the instrument had manufactured.

**The corrected measurement.** Ground truth is each held-out unit's **own original translation** —
available because the queries were held out of an already-translated journal, so neither arm could
have seen them. Corpus B, n = 593:

| | mean overlap with the real original |
|---|---|
| embedding | 0.5928 |
| hybrid + rerank | 0.5874 |

Closer to the original: **embedding 223, hybrid 222, tie 148** → 445 decisive, **z = 0.05, not
significant**. A dead heat. There is **no measurable consistency difference between the two
retrievers.** (Corpus A's consistency numbers are withdrawn without replacement rather than
re-measured.)

**The MMR mechanism was independently falsified, before the bias was even found.** `sweep_mmr.sh`
swept `reference_mmr_lambda` end-to-end on corpus A — n = 300, six values spanning 0.0 to 1.0, plus
an embedding reference point — precisely because the docs' "raise it toward 1.0 for consistency"
was a named mechanism and never a measurement. **Mean overlap was flat: 0.098 to 0.106, spread
0.0086, non-monotone, no trend.** λ = 1.0, diversity penalty fully off, was no better than λ = 0.0.
MMR was never the cause of the apparent gap — entirely consistent with there being no gap. (The
sweep scored every λ against the same biased proxy, so its absolute values are on the compromised
yardstick; the proxy is constant across λ, though, so it cannot manufacture flatness between them.)

**The lesson, which is the point of recording this at all.** The finding replicated cleanly across
two independent corpora, and that replication was cited here as its strongest evidence. It was not
corroboration. **A structural bias reproduces perfectly** — better than a real effect does, because
it carries no noise. Replication only corroborates when the runs do not share an instrument; two
corpora through one biased metric is one experiment run twice, not two experiments. And the
two-corpus rule that correctly caught the over-general robustness claim was powerless here, because
that rule varies the *data* and this defect lived in the *instrument*. **A number that reproduces
cleanly is more dangerous than one that does not, precisely because reproducing is what makes it
persuasive.** The question to ask of any evaluation metric is not "does it replicate?" but "is it
independent of the thing it is ranking?" — and here it was not: the metric's yardstick was one of
the arms under test.

**The harness now refuses, rather than warns.** `consistency()` takes `--ground-truth` and **raises
rather than running the proxy while either arm is `embedding`**. Refusing beats warning here for a
specific reason: the failure is silent and the output looks *good*. The proxy still emits a
plausible, well-formed, reproducible number that a reader cannot distinguish from a sound one, and
the write-up around it will be persuasive in proportion to how clean it looks. A warning is a note
printed beside a number that will outlive it — this is exactly the situation where the number must
not exist. (Secondary benefit: with ground truth supplied, no retriever is constructed at all, so
the measurement is arithmetic over two journals and a file and needs no server.)

**Docs corrected:** `docs/reference-translations.md` (Phase B section, the corpus B replication, the
no-reference n=1000 comparison, the MMR paragraph, the "when to use which" advice),
`docs/feature-requests/hybrid-retrieval-reranking-rag.md` (Phase 8B outcome), `config/README.md`,
`config/agents.toml` comments, `tools/README.md`. Retracted claims are struck in place, not
deleted, so a reader who saw the earlier numbers can find out what happened to them.

The resulting advice, now carried by `docs/reference-translations.md`, `config/agents.toml` and
`config/README.md`: quality is a wash, consistency shows no measurable difference, and
**robustness is the only measured differentiator — and it is corpus-dependent**, so pick a
retriever by measuring your own rejection rate. Bare RRF hybrid still not recommended. Nothing was
retuned to make any arm look better.

**And then the question the harness could not ask.** As first written, `eval_translation.sh`
compared retrievers against each other but not against **not retrieving** — so "does this feature
help at all?" was the one question it could not answer, because the Phase B spec had been
implemented literally. With a `none` arm added and run at n=1000 on corpus A: quality is a wash
leaning *against* retrieval (165 to 140, z ≈ 1.4, not significant) and the rejection rate **nearly
triples** (3.0% → 8.8%, 107 verified units lost). The prior evidence for that cost was "four lines".
Its consistency figure — "~3× better", and the "one unit lost to rejection for every three made
consistent" summary built on it — came from the **same biased metric** (one arm here *is* the
embedding retriever) and is **withdrawn**. The direction survives and is close to a tautology: show
a model corpus text and it reuses some of it. The magnitude is not a measurement. The rejection
cost, 5.8 percentage points, is.

**Deviations to keep honest.** The split sizes are not the plan's literal 8000/2000; the
blob-rendering sanity check was later recorded as dropped rather than deferred; `reference_mmr_lambda`
has since been swept across `0.0`–`1.0` with no effect found (above); and the corrected consistency
comparison was run on corpus B only — corpus A's consistency numbers are withdrawn, not replaced,
and reference-on vs reference-off consistency has not been re-measured against ground truth at all.

## Defects found and fixed

**Branch coverage was never enabled, and hid four untaken branches behind "100% coverage".**
The suite reported 100% *statement* coverage with four branches never taken. Two were real
defects, and each was fixed at the root rather than by a test that merely walks the arc:

- *A silent counter miscount.* `runner.py::Progress.record` incremented `done` for an unhandled
  status while incrementing no category, so the counters stopped summing to `done` and every
  derived figure (position, rate, ETA) drifted — silently, since nothing ever compared them. Now
  raises a structured `RunnerError` naming the status and unit id, and counts `done` last so a
  rejected outcome leaves every counter untouched.
- *Dead code asserting a non-invariant.* `client.py::_close_truncated_json_object`'s
  `isinstance(dict)` check could never fail — both repair suffixes end in `}`, and the only JSON
  document ending in `}` is an object. Removed, with the invariant recorded.

The other two (`rules.py::_common_prefix`'s exhausted-loop path, `agents.py::translate()` with an
issue-free attempt) were untested rather than wrong, and now have direct tests. `src` is at 100%
statement *and* branch coverage, and `run_tests.sh` measures branches from now on — the lesson
being that a 100% figure means nothing until you know which 100%.

**Five defects in the retrieval path the long GPU evaluation hammers**, each with a regression
test that fails on the previous code:

1. `_sigmoid` raised `OverflowError` below about −710, so a single outlier reranker logit could
   kill an entire run. Now branches on the sign so `exp()` only ever sees a non-positive argument;
   NaN is refused rather than silently poisoning the ranking.
2. `RerankClient` accepted duplicate result indices, which surfaced far away as a **bare
   `KeyError` inside MMR** — the textbook case of a boundary violation being detected somewhere
   useless. Now a `RerankError` naming the endpoint and the duplicated indices.
3. `mmr_order` returned a wrong-but-plausible order for a non-finite relevance: nothing beats
   `-inf`, so the sentinel index stayed `-1` and `pop(-1)` picked the last candidate — a silent
   mis-ordering, not a crash. Non-finite relevance/similarity now raise `ValueError` naming the
   offender, and the sentinel is gone entirely.
4. `EmbeddingRetriever` clamped its score at the top only, so a negative cosine escaped into
   `Retrieved.score`, documented as `[0, 1]`, whenever `min_score < 0`. Clamped both ends.
5. `close()` ignored `_owns_client` and tore down a caller-**injected** client, while the
   construction-failure path honoured it. One rule now across `EmbeddingRetriever`,
   `RerankClient` and `HybridRetriever`: an object closes only what it opened.

**Three display-width over-counts, found by property testing.** Width feeds a *blocking* rule, so
an over-count rejects a good line outright and ships the untranslated source in its place. The
property "the same text spelled NFC and NFD measures the same" found the worst one: conjoining
Hangul jamo are letters with canonical combining class 0, so no mark test saw them and
**decomposed Korean measured double** its rendered width (NFD `한` = 4 columns against NFC's 2) —
an NFD corpus would have been rejected wholesale. Also fixed: format characters (`Cf` — the
joiners, bidi marks, soft hyphen, BOM) and the combining-class-0 marks a `unicodedata.combining`
test misses (variation selectors above all) each cost a column they never draw. One over-count
**remains, deliberately**: a ZWJ emoji sequence is measured per code point, and reconciling it
needs grapheme-cluster segmentation this per-code-point model cannot express. It is an `xfail`
with that reason, not a silent gap.

## Method notes worth keeping

- `build_eval_split.sh` exists because its ad-hoc predecessor asked for 40 queries, silently
  produced 24, and reported neither — an evaluation ran at 60% of its intended sample unnoticed.
  A shortfall is now an error naming the shortfall.
- `eval_translation.sh` **reports** per-line quality but **gates** only consistency and rejection
  rate: the plan itself predicted quality would be a wash, and gating on a wash is gating on noise.
  (Sharpened by the correction above: gating on a *biased* metric is worse than gating on noise —
  it gates in a preordained direction. The consistency gate now demands `--ground-truth` wherever
  the proxy would be circular, and refuses to run otherwise.)
- Quality and consistency are measured over **injectable** units only. A rejected unit often still
  carries the text that failed its checks, and scoring that text measured renderings the engine
  had already refused to ship.
- Two properties were initially too strong and were corrected rather than papered over: a
  whitespace-only source normalises to `""`, yields no shingles, and therefore cannot be indexed
  or retrieved — asserting otherwise tests something the data cannot carry.
