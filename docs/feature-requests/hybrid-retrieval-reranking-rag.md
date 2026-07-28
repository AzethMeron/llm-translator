# Plan: Hybrid retrieval + reranking + unified context packing

## Context

The translator's reference retriever steers **consistency** with an existing body of translation
(per-line quality is a wash, dominated by corpus quality. This line originally cited "~4× more
house-style reproduction" from the previous session; that multiplier came from a metric later found
to be circular and is **withdrawn** — see "Phase 8B outcome" below. The steering direction stands;
the number does not). Two retrievers exist behind one `Retriever` seam but are **mutually exclusive**: the
stdlib `LexicalRetriever` (TF-IDF char-n-gram = exact-term/"sparse" signal) and the optional
`EmbeddingRetriever` (dense/semantic). A semantic-only retriever once surfaced a corrupt corpus line
as a false match and induced malformed output.

The two attached RAG documents describe production RAG for large document **blobs** (chunking, vector
DBs, Postgres/S3, ANN, versioning). **Rejected as inapplicable**: our units are already atomic and
tiny, runs are ephemeral, corpora are ≤~1M short lines where brute-force numpy cosine already does
~200 q/s. What transfers is the retrieval-quality core: **hybrid (sparse+dense) retrieval, RRF
fusion, cross-encoder reranking, MMR/dedup packing**. Enabling fact (verified in source): the local
llama.cpp build has reranking compiled in (`/v1/rerank`, `--reranking --pooling rank`).

**Goal:** promote embedding to a first-class arm; add a `HybridRetriever` (RRF + optional rerank +
MMR) behind the **unchanged** `Retriever` seam; unify positional context + retrieved examples into one
budgeted, MMR-diversified packing layer with the **±1 neighbor always guaranteed**. Lose no existing
functionality; keep the stdlib lexical retriever the zero-dependency default. Prove it with a staged
A/B on two independent real corpora: **corpus A** (a ~11k-line Japanese→English game corpus, the
noisier of the two) and **corpus B** (a ~2.4k-line Japanese→English game corpus, the cleaner one).

**Prompt-format requirement (applies to every run, not just hybrid retrieval):** positional context
(previous, current, next) shall be rendered as **one continuous text blob** — not the current
per-neighbour bulleted `SOURCE:`/`TARGET:` lines — because a game line is frequently a fragment: a
sentence, clause, or idiom split across unit boundaries. A model shown fragmented, individually
labelled snippets treats each as a self-contained thought; a model shown continuous prose reads the
break the way a human translator would and resolves the idiom correctly. The line actually being
translated is embedded in the blob in its natural position **and then repeated verbatim afterward**
under its own unambiguous heading, so the model is never in doubt about exactly what to output a
translation for, however continuous the surrounding prose reads. This is a deliberate change to the
*default* prompt shape (positional context is on by default at `before_units=3`/`after_units=2`), not
an opt-in knob — see the revised `context_packing.py` spec below.

## Design principles (CLAUDE.md)

- **Seam-compatible:** everything implements `transunit.reference.Retriever` (`by_source`/`by_target`
  → `tuple[Retrieved, ...]`). Engine retrieval call sites unchanged.
- **SSOT / composition:** `HybridRetriever` *composes* the existing `LexicalRetriever` +
  `EmbeddingRetriever`; never forks them.
- **Decoupled, isolation-testable:** pure fusion math (`fusion.py`, stdlib-only) separate from the
  rerank HTTP client (`rerank.py`) separate from composition (`hybrid.py`) separate from prompt
  packing (`context_packing.py`, pure). Each 100% unit-testable with no network.
- **No silent fallback:** a hybrid whose embedding/rerank endpoint is down **fails loud** (mirrors
  cli.py refusing `--embedding-url` + learning). Never degrade to lexical silently.
- **RAII + structured errors + injectable `httpx.Client`:** rerank client mirrors
  `EmbeddingRetriever`'s ownership/`close()` and a `RerankError(reason, *, url=...)` shaped like
  `EmbeddingError`, with a `client=` test seam.
- **stdlib default preserved:** hybrid/rerank live on the `translator` side (numpy allowed);
  `transunit` stays stdlib-only.

---

## Module specs (concrete signatures)

### `src/translator/retrieval/` (new package)

`__init__.py` re-exports `HybridRetriever`, `EmbeddingRetriever`, `EmbeddingError`, `RerankClient`,
`RerankError`, `DEFAULT_EMBEDDING_MIN_SCORE`, `DEFAULT_RERANK_MIN_SCORE`.

`retrieval/embedding.py` — the current `src/translator/embedding.py`, **moved verbatim**, plus one
internal accessor for MMR redundancy:
```python
def vectors_for(self, entries: Sequence[ReferenceEntry], *, target: bool) -> "np.ndarray":
    """Stored L2-normalised rows for these already-indexed entries (rows aligned to `entries`),
    for the hybrid's MMR redundancy term. Raises EmbeddingError if an entry isn't indexed."""
```
Build `self._row_source: dict[ReferenceEntry, int]` / `self._row_target` at construction (entry→row;
frozen entries are hashable, duplicates collapse — harmless). Keep a thin
`src/translator/embedding.py` shim re-exporting from `retrieval.embedding` (back-compat; no
functionality lost).

`retrieval/fusion.py` — **pure, stdlib-only, no numpy, no I/O:**
```python
def rrf(rankings: Sequence[Sequence[H]], *, k: int = 60) -> dict[H, float]:
    """Reciprocal Rank Fusion: for each best-first ranking, add 1/(k + rank_1based) per id.
    k >= 1 (else ValueError). Deterministic."""

def mmr_order(candidates: Sequence[H], relevance: Mapping[H, float],
              similarity: Callable[[H, H], float], *, lambda_: float, k: int) -> list[H]:
    """Greedy Maximal Marginal Relevance: pick argmax(lambda_*rel - (1-lambda_)*max_sim_to_selected).
    0 <= lambda_ <= 1 (else ValueError); k >= 0; ties broken by input order (stable); returns up to k."""
```

`retrieval/rerank.py` — cross-encoder client behind the injectable seam:
```python
DEFAULT_RERANK_MIN_SCORE = 0.30  # on the sigmoid of the raw rerank score

class RerankError(Exception):
    def __init__(self, reason: str, *, url: str | None = None) -> None: ...  # .reason, .url

class RerankClient:
    def __init__(self, *, base_url: str, model: str = "local", timeout_seconds: float = 120.0,
                 client: httpx.Client | None = None) -> None:
        # self._url = base_url.rstrip("/") + "/rerank"; owns client unless injected (_owns_client)
    def rerank(self, query: str, documents: Sequence[str]) -> list[tuple[int, float]]:
        # empty documents -> [] (no HTTP call).
        # POST {"model", "query", "documents": list(documents)}; parse resp["results"] ->
        #   [(int index in range, float relevance_score), ...]; re-sort desc defensively.
        # Requires one result per document (indices cover range(len(documents))).
    def close(self) -> None: ...
```
Request/response pinned to the verified llama.cpp Jina schema (request `{model,query,documents}`,
response `{"results":[{"index","relevance_score"},...]}`). Error taxonomy → `RerankError`:
HTTP error (`httpx.HTTPError`) → "rerank request failed"; non-JSON / missing `results` →
"malformed rerank response"; item missing `index`/`relevance_score`, non-numeric, or index out of
range → "malformed…"; fewer results than documents → "endpoint returned N results for M documents".

`retrieval/hybrid.py` — the composition, implements `Retriever`:
```python
class HybridRetriever:
    def __init__(self, entries, *, embedding_base_url: str, embedding_model: str = "local",
                 rerank_base_url: str = "", rerank_model: str = "local",
                 index_source: bool = True, index_target: bool = False,
                 candidate_pool: int = 40, mmr_lambda: float = 0.7,
                 lexical_min_score: float = 0.30, embedding_min_score: float = 0.55,
                 ngram: int = 3, batch_size: int = 64, timeout_seconds: float = 120.0,
                 embedding_client: httpx.Client | None = None,
                 rerank_client: httpx.Client | None = None) -> None: ...
    def by_source(self, query, *, k, min_score=0.0) -> tuple[Retrieved, ...]: ...
    def by_target(self, query, *, k, min_score=0.0) -> tuple[Retrieved, ...]: ...
    def close(self) -> None: ...                      # closes dense arm + rerank client (cascade)
    @property
    def source_entries(self) -> int: ...              # delegate to lexical arm
    def __len__(self) -> int: ...
```
Construction: validate `candidate_pool >= 1`, `0 <= mmr_lambda <= 1`; build `LexicalRetriever` then
`EmbeddingRetriever` (dense) then optional `RerankClient`. **RAII:** wrap arm/rerank construction so
if a later step raises, already-opened resources are closed before re-raising (mirror
`EmbeddingRetriever.__init__`'s `except BaseException: ... raise`).

`by_{source,target}(query, k, min_score)` algorithm:
1. `lex = lexical.by_X(query, k=candidate_pool, min_score=lexical_min_score)`;
   `dense = dense.by_X(query, k=candidate_pool, min_score=embedding_min_score)` — each arm gates
   recall with **its own calibrated floor** (preserves today's 0.30/0.55 meaning).
2. `fused = rrf([[h.entry for h in lex], [h.entry for h in dense]])`; take top `candidate_pool`.
3. **Relevance:** if reranker configured → `sigmoid(rerank(query, [entry text]))` per candidate;
   else min-max-normalise the fused RRF scores to `[0,1]` (single candidate → 1.0). Text = source
   for `by_source`, target for `by_target`.
4. Drop candidates with relevance < `min_score` (the **final** floor).
5. `mmr_order(survivors, relevance, similarity=dense-cosine via vectors_for, lambda_=mmr_lambda, k=k)`.
6. Return `tuple(Retrieved(entry, relevance) …)` in MMR order.

Fail-loud: dense endpoint down at construction → `EmbeddingError`; rerank endpoint down at query →
`RerankError` propagates out of `by_source`/`by_target` (no silent fall-back to fused order).

### `src/translator/context_packing.py` (new, pure)

Two responsibilities, kept as separate pure functions per SSOT — **selection** (which neighbours and
reference examples survive budget/MMR; unchanged reasoning from the original design) and **rendering**
(how the survivors become prompt text; this is where the blob requirement lives):

```python
@dataclass(frozen=True, slots=True)
class PackedContext:
    blob: str
    # Continuous text: masked(before[0]) ... masked(before[n]) unit_source masked(after[0]) ...
    # masked(after[m]), joined in reading order with a single separator (newline). The CURRENT
    # unit's OWN placeholders are left intact (never masked -- they must survive into the
    # translation); only NEIGHBOUR placeholders are masked to `stand_in`, exactly like today's
    # `_render_context` rule, so a neighbour's [[0]] is never mistaken for this unit's own.
    established: tuple[tuple[str, str], ...]
    # (neighbour_source, neighbour_target) pairs for surviving neighbours that already have a
    # translation (include_translations=True and TranslationMemory has a hit). Kept as its OWN
    # labelled section, separate from the blob -- these are a distinct fact ("this exact
    # neighbouring line was already rendered as X"), not narrative flow, and folding them into the
    # blob would defeat the point of a *continuous* passage.
    references: tuple[Retrieved, ...]
    # Retrieved reference examples that survived selection. Unchanged semantics/section: these are
    # similarity-matched lines from elsewhere in the corpus, NOT narratively adjacent, so they stay
    # their own clearly-labelled section and are never merged into the continuity blob (merging
    # them in would be actively misleading -- they are not "what comes next").

class ContextPacker:
    def __init__(self, *, char_budget: int = 0, mmr_lambda: float = 0.7,
                 guaranteed_adjacent: int = 1) -> None: ...
    def pack(self, *, before: Sequence[str], current: str, after: Sequence[str],
             references: Sequence[Retrieved], stand_in: str,
             translate_of: Callable[[str], str | None],
             similarity: Callable[[str, str], float]) -> PackedContext: ...

def render_context(packed: PackedContext, *, unit_source: str,
                   source_label: str, target_label: str) -> str:
    """Pure formatter: PackedContext -> the prompt text agents.py appends to `parts`."""
```

**`pack()` behaviour** (selection; unchanged reasoning from the original design, retained in full):
(1) **always** include the innermost `guaranteed_adjacent` before/after neighbours (discourse floor),
even at `char_budget == 1`; (2) **dedup** a neighbour whose masked source equals a reference source
(keep the neighbour, drop the duplicate reference); (3) MMR-diversify the remaining neighbours and
references using the injected `similarity` (a **stdlib char-trigram Jaccard**, so packing needs no
embedding round-trip per unit); (4) enforce `char_budget` (0 = disabled → today's count-cap
behaviour) but never evict a guaranteed neighbour. Pure: `translate_of` (memory lookup) and
`similarity` are injected — no I/O, no globals.

**`render_context()` behaviour** (the blob requirement): builds, in this order, only the sections
that have content —
1. *Reference translations of similar lines* — unchanged existing label/format, one bullet per
   `Retrieved`, kept separate from the blob (see rationale above).
2. *Surrounding context* — one paragraph: `"This is surrounding context (previous and next lines).
   It may contain partial sentences, clauses, or idioms that continue across lines -- use it only to
   understand meaning and continuity. Do not translate this context; the exact line to translate is
   given again below."` followed by `packed.blob` as continuous prose (no per-line bullets, no
   `SOURCE:`/`TARGET:` labels). This single paragraph replaces today's two separately-worded
   "Preceding sentences" / "What is said next" sections — one unified instruction covering both
   directions, because the model needs to read them as one passage, not two.
3. *Already-translated neighbouring lines* (only if `packed.established` non-empty) — the existing
   established-rendering behaviour, preserved verbatim as its own compact list so no functionality is
   lost.
4. *(unchanged)* placeholder expectation, length limit.
5. *Text to translate:* `\n{unit_source}` — **always last**, **always present**, byte-identical to
   how the current line appears embedded in the blob (same unmasked placeholders). This is the
   explicit repetition the user asked for: whatever continuity reading the model just did, the exact
   translation target is restated with zero ambiguity immediately before generation.

`agents.py`'s `_unit_block` calls `pack()` then `render_context()` and appends the result in place of
today's `_reference_examples` + `_render_context` calls. This is an intentional, uniform change to
prompt wording for **every** run using `before_units`/`after_units` > 0 — i.e. the existing default —
not merely an additive knob; see the corrected compatibility note below (this replaces the earlier,
now-inaccurate claim that defaults reproduce today's output byte-for-byte).

**No null-value fields (strict — every optional section, no exceptions).** Every optional
field/section in `render_context()`'s output is included **only when it has real content**; there is
no "field: none" / "this line has none" filler anywhere. Concretely: the *Surrounding context*
section (including the blob) is omitted **entirely** when `before` and `after` are both empty —
never rendered as an empty or placeholder heading; `PackedContext.blob == ""` is a legal value and is
treated exactly like every other empty optional field (no heading at all); *Already-translated
neighbouring lines* is omitted entirely when `packed.established == ()`, never rendered as
"Already-translated neighbouring lines: none"; *Reference translations…* is omitted entirely when
`packed.references == ()`. This matches the module's own existing style rule (`_system_prompt`'s
docstring already states it for the style-policy section: "omitted... rather than left as an empty
header") — this plan simply makes it explicit and non-negotiable for every section
`context_packing.py` renders.

**Pre-existing violation to fix in the same phase (Phase 5, since it rewrites `_unit_block`
anyway).** `agents.py::_placeholder_expectation` currently violates exactly this rule for the
zero-placeholder case — its literal current text is `"Placeholders: this line has none. Do not
introduce any. Where the source omits a subject, supply a natural pronoun or the speaker's name in
the target language -- never invent a placeholder for it."`, which opens by stating a null field
value. **This section is not simply deletable, though** — unlike the other optional sections, the
zero-placeholder case carries real, load-bearing, already-tested guidance: the "do not invent a
placeholder for a dropped subject" instruction is an existing anti-hallucination guard (see the
function's own docstring: a language that omits its subject can otherwise tempt the model to supply
one as a bogus `[[N]]`). Silently omitting the whole section to satisfy the "no null field" rule
would violate "no functionality shall be lost." The correct fix is to **reword away from the
null-value/report framing while preserving the guidance as a direct instruction**, e.g.: `"This line
has no placeholders. Do not introduce any -- where the source omits a subject, supply a natural
pronoun or the speaker's name in the target language."` The non-empty branch ("has exactly N (...)")
is real, always-present content and is unaffected — it stays as-is. Add a test asserting the literal
string `"has none"` (or any `": none"` framing) never appears in a rendered prompt, covering both the
new `context_packing.py` sections and this reworded `_placeholder_expectation` case.

---

## Config (`roles.Context` + `_context` parsing) and CLI

New `Context` fields (validate like existing `_int`/`_float`): `reference_candidate_pool: int = 40`
(min 1), `reference_mmr_lambda: float = 0.7` (0–1), `reference_lexical_min_score: float = 0.30`
(0–1), `reference_embedding_min_score: float = 0.55` (0–1), `context_char_budget: int = 0` (min 0;
0 = disabled). Keep `reference_examples`, `reference_min_score` (now the **final-relevance** floor),
`reference_revision_examples`, `reference_learn_statuses`.

**Sane defaults, feature OFF unless opted in.** The whole reference subsystem stays **off by
default** (`reference_examples = 0`, `reference_revision_examples = 0`) exactly as today, so no
existing run changes. When a user turns it on, every new knob has a justified default shipped in
`config/agents.toml` with an inline rationale: `reference_candidate_pool = 40` (ample pool for a
final `k` of 2–5 so RRF/rerank/MMR have material, still cheap), `reference_mmr_lambda = 0.7`
(relevance-leaning, the standard MMR value), `reference_lexical_min_score = 0.30` and
`reference_embedding_min_score = 0.55` (this session's evidence-based per-arm floors),
`reference_min_score` final floor defaulting to `DEFAULT_RERANK_MIN_SCORE = 0.30` when reranking
else the existing `0.30`, `context_char_budget = 0` (disabled → today's count-cap behaviour),
`before_units = 3` / `after_units = 2` (unchanged). Validation ranges are enforced in `_context`
parsing (min/max), so an out-of-range value fails fast with a clear message rather than a silent
bad default.

**Positional context knobs preserved (`before_units`, `after_units`, `include_translations`,
`anonymous_subject`).** These keep their exact current meaning and config-file wiring — they are the
**neighbour window**. `agents.py` still slices `before = unit.context_before[-before_units:]` and
`after = unit.context_after[:after_units]` (today's code) and hands those windows to
`ContextPacker.pack`. The packer's ±1 "guarantee" is scoped to the window: `guaranteed_adjacent`
covers the innermost neighbour **that the window already includes**, so `before_units = 0` /
`after_units = 0` still fully disables that side (config wins — nothing is force-added that the
operator excluded). `context_char_budget` and MMR only ever trim/reorder neighbours *beyond* the
guaranteed innermost one.

**Compatibility note (corrected — this is NOT byte-for-byte identical to today).** At
`context_char_budget = 0` and no reference retrieval, the *selection* logic is unchanged (the same
neighbours that appear today still appear), but the **rendering is intentionally different**: today's
two labelled sections ("Preceding sentences…" / "What is said next…", each a bulleted `SOURCE:`/
`TARGET:` list) become the single continuous-prose blob + repeated "Text to translate:" line
described above, for every run with `before_units`/`after_units` > 0 — i.e. the existing default.
This is a deliberate behaviour change to prompt wording, made because fragmented per-line sections
were shown to make cross-line idioms and split sentences harder for the model to read correctly;
functionality (which lines are shown, established translations, placeholder masking, reference
examples) is fully preserved, only the *shape* of the text changes. `config/README.md` documents this
rendering change explicitly as a migration note, and the A/B in this plan includes a sanity check
(old fragmented rendering vs new blob rendering, reference off) so the change is verified, not just
asserted, before it becomes the default for every existing config.

CLI (`cli.py`): flags `--hybrid` (store_true, requires `--embedding-url`), `--rerank-url` (default
`""`), `--rerank-model` (default `"local"`). `_load_reference` selection ladder (back-compatible):
lexical (default) · growable (learn) · embedding (`--embedding-url`) · hybrid (`--embedding-url
--hybrid`) · hybrid+rerank (`+ --rerank-url`). Refuse learn+hybrid (`UsageError`, read-only); refuse
`--rerank-url` without hybrid (`UsageError`). Close cascade on every exit path via existing
`hasattr(reference, "close")`. Add `RerankError` to the `main()` except tuple (clean exit-1 + logged).

---

## Serving / setup (hardened `tools/`, per script-threshold rule)

- `tools/serve_reranker.sh` — mirror `serve_embeddings.sh` via `lib/common.sh` (`die`/`note`/
  `require_command`, port validation, `--help`): `exec llama-server --reranking --embedding
  --pooling rank <model_args> --host --port -ngl 99`, default port **8082**, `--` passthrough.
  **Revised from the original two-script plan**: llama-server has a native `-hf <repo>[:quant]`
  auto-download flag (verified against `--help` and the real `tools/server/README.md`) — the
  exact mechanism `serve_embeddings.sh` already uses via `--embd-gemma-default`. A separate
  `tools/fetch_reranker.sh` would duplicate that hardened, checksum-verified downloader for no
  benefit, so fetch+serve are one script, matching the established sibling-script pattern
  exactly. Default model: `gpustack/bge-reranker-v2-m3-GGUF:Q8_0` -- verified via the HF API to
  be real, **Apache-2.0**-licensed, and multilingual (a GGUF conversion of BAAI's
  bge-reranker-v2-m3, the model bge-m3's own authors recommend for hybrid-retrieval reranking);
  `--model` overrides with a local GGUF, `--hf-repo` overrides the HF repo/quant.
- `requirements.txt`: no new pin (reuse numpy/httpx); update OPTIONAL-section comment.
- `tools/check_environment.sh`: reranker note (optional, like numpy).

---

## Test matrix (current regime: pytest, `httpx.MockTransport`, injected `client=`, structured
errors, RAII spies, `test_separation` allowlist, maintain 100% coverage — every error branch tested)

`tests/translator/retrieval/test_fusion.py` (pure): rrf — single/multiple lists, overlapping ids
accumulate, empty input, default & custom `k`, determinism, `k<1` → ValueError. mmr_order —
`lambda_=1` == relevance order, `lambda_=0` maximises diversity, `k=0` → [], `k>len` → all, stable
ties, redundancy suppresses near-duplicates, `lambda_` out of range → ValueError.

`tests/translator/retrieval/test_rerank.py` (MockTransport + injected client): happy path maps
`results`→`[(index,score)]` with correct index mapping even when server pre-sorts; empty `documents`
→ `[]` with **no HTTP call** (assert transport untouched); `.url` on error. **Error branches →
RerankError:** 503/`HTTPError`, non-JSON body, missing `results`, item missing `index`/
`relevance_score`, non-numeric score, index out of range, fewer results than documents, bad
base_url. RAII: `close()` closes the client (injected-client variant, `SpyClient`).

`tests/translator/retrieval/test_hybrid.py` (real `LexicalRetriever` + dense/rerank via MockTransport
injected clients): `isinstance(_, Retriever)`; fusion surfaces an entry ranked by both arms above one
ranked by one; rerank reorders vs no-rerank; per-arm floors gate recall; final `min_score` gates
output; MMR suppresses a near-duplicate example (low `lambda_`); `by_target` uses target text and
finds target-only entries; `k=0`/empty corpus → `()`. **Error/robustness:** dense endpoint 503 at
construction → `EmbeddingError` **and** an already-built rerank client is closed (RAII, spy); rerank
endpoint 503 at query → `RerankError` out of `by_source` (assert **no** silent fused-order fallback);
`close()` cascades to dense + rerank clients (spies); `candidate_pool<1`/`mmr_lambda` out of range →
ValueError; `source_entries`/`__len__` delegate.

`tests/translator/test_context_packing.py` (pure, split by function):

- **`pack()` (selection):** ±1 neighbour guaranteed at `char_budget=1`; neighbour∩reference dedup
  (neighbour kept, duplicate reference dropped); budget trims low-priority but never evicts a
  guaranteed neighbour; MMR diversity among references; empty references, empty neighbours,
  all-empty; `before_units=0`/`after_units=0` (empty `before`/`after` input) fully disables that
  side even with `guaranteed_adjacent=1` (config wins, nothing force-added).
- **`render_context()` (rendering — the blob requirement, tested directly and exhaustively):**
  `packed.blob` is a **single string** (not a list of bulleted lines) containing, in order,
  before-neighbour source(s), the **current unit's source**, then after-neighbour source(s); a
  fixture with a sentence deliberately split across a neighbour and the current unit (e.g. before
  ends mid-clause, current completes it) is asserted to appear as continuous, readable text with no
  bullet/label interruption between them; **neighbour placeholders are masked** to `stand_in` inside
  the blob while **the current unit's own placeholders are left unmasked** (regression-style
  assertion: construct a case where a neighbour and the current unit each carry a `[[0]]` and assert
  only the neighbour's is replaced); the *"Text to translate:"* section is present, **last**, and its
  text is **byte-identical** to the unmasked current-unit source that appears embedded in the blob
  (i.e. the exact repetition the design requires); the *"Already-translated neighbouring lines"*
  section appears iff `packed.established` is non-empty and lists exactly those pairs; the
  *"Reference translations…"* section stays separate from the blob (never concatenated into it) and
  appears iff `packed.references` is non-empty; section order is reference → blob → established →
  placeholder/length → text-to-translate; an all-empty `PackedContext` still renders a text-to-translate
  section and nothing else (no dangling empty headers — mirrors the existing "no empty section"
  invariant `_unit_block` already guarantees elsewhere). **No-null-field assertions (dedicated
  cases):** empty `before`/`after` → no "Surrounding context" heading at all (not an empty one, not
  a blob saying "no context"); `established == ()` → no "Already-translated neighbouring lines"
  heading (not "...: none"); `references == ()` → no reference heading; grep the full rendered
  output of every case in this file for the literal substrings `"none"`/`": none"` and assert none
  are present outside of legitimate prose the fixtures don't otherwise contain.

`tests/translator/test_cli.py` (extend): `--hybrid`/`--rerank-url`/`--rerank-model` parse;
`--embedding-url --hybrid` builds a `HybridRetriever` (monkeypatched stub, assert params);
learn+hybrid → `UsageError`; `--rerank-url` without hybrid → `UsageError`; hybrid stub closed on
every exit path (reuse `ClosableRetriever`); `RerankError` in `main()` → clean exit 1 + logged
(mirror the existing embedding fatal-error test).

`tests/translator/test_agents.py` (extend): `_unit_block` wired end-to-end through the real
`ContextPacker`/`render_context` with a fake retriever + `TranslationMemory` — assert the rendered
prompt contains one continuous surrounding-context paragraph (not the old two-section
Preceding/Following format — a regression test asserting the OLD labels/format are gone), the
"Text to translate:" line is present and matches `unit.source` verbatim, a translated neighbour's
established rendering still surfaces (no functionality lost vs. today's `include_translations`
behaviour), the reference-examples section still appears distinctly when a retriever is wired, and
`before_units=after_units=0` yields no surrounding-context section at all (not an empty one). Also
extend the existing `_placeholder_expectation` tests: the zero-placeholder case's rendered text no
longer contains `"has none"`, while the "do not introduce any" / "supply a natural pronoun" guidance
is still present verbatim (functionality preserved, only the null-value framing removed); the
non-zero branch (`"has exactly N (...)"`) is unchanged and still covered by its existing test.

**Diagnostics → log file (every error and warning, exhaustively).** Logging invariant to hold and
test: the `--log-file` handler is **unfiltered**, so *every* `WARNING`/`ERROR` record reaches the
file; `_LeniencyConsoleFilter` only trims the console. `_configure_logging(log_file)` + the existing
`test_a_fatal_error_lands_in_the_log_file_and_prints_once` pattern are reused. Enumerated and each
given a test that asserts it (a) surfaces to the operator and (b) lands in the `--log-file`:

- **Errors that reach the operator via `main()` → printed once to stderr AND written to the log
  ("run aborted: …"), exit 1:** `EmbeddingError` at hybrid construction (dense endpoint down /
  malformed); `RerankError` mid-run (rerank endpoint dies at query time — driven through the runner,
  asserting the journal is intact *and* the log captured it, mirroring the embedding mid-run test);
  `UsageError` for learn+hybrid; `UsageError` for `--rerank-url` without `--hybrid`. Each added to the
  `main()` except tuple (`RerankError` newly) and covered by a `--log-file` test.
- **Warnings (`logger.warning` → console + log file), each with a log-file/`caplog` assertion:**
  the four existing `_load_reference` warnings (retrieval-off-with-material; no-corpus-and-no-learning;
  source-wanted-but-no-entry-has-a-source; embedding floor < 0.4) **plus two new ones** — *W1:* hybrid
  **and** rerank enabled but `reference_min_score` is below the reranker-sigmoid range (advise
  `DEFAULT_RERANK_MIN_SCORE`); *W2:* `reference_candidate_pool < reference_examples` (pool smaller
  than the requested final `k`). At least one test proves a warning reaches the real `--log-file`
  (not just `caplog`), pinning the unfiltered-file invariant.
- **Unit-level failure modes (structured exception raised, tested per module above):** every
  `RerankError` branch (HTTP error, non-JSON, missing `results`, missing/`non-numeric`
  `index`/`relevance_score`, index out of range, count mismatch, bad URL); every `EmbeddingError`
  branch (unchanged, retained); `fusion` `ValueError`s (`k<1`, `lambda_` out of range); `hybrid`
  `ValueError`s (`candidate_pool<1`, `mmr_lambda` out of range); hybrid **fail-loud** (dense/rerank
  down → exception out of `by_source`/`by_target`, never a silent fused-order fallback). These are the
  raise-sites; the CLI/mid-run tests above are what carry the representative ones into the log file.

`tests/test_separation.py`: extend the numpy allowlist from `translator/embedding.py` to
`translator/retrieval/*.py`; `transunit` stays stdlib-only. Update the moved-module path assertions.

**Real-data retrieval-relevance test (required, GPU/server-gated).** Beyond the hermetic
hand-picked-vector ranking tests, a reproducible harness proves the RAG *actually fetches relevant
samples on real language* — `tools/eval_retrieval.sh` (hardened per `lib/common.sh`; requires the
embedding server on 8081, Qwen judge on 8080, optional reranker on 8082; **fails with a clear,
actionable message if any needed server is down**, never a silent skip) driving a Python evaluator.
Methodology follows the established rule (unique, **non-corpus-duplicate** queries + an **independent
Qwen judge**, precision by score band): build a real reference corpus from corpus A / corpus B, sample
~1–2k unique held-out queries (deterministic seed), and for each of {lexical, embedding, hybrid,
hybrid+rerank} retrieve top-k and have the Qwen judge rate each retrieved example 0/1/2. It prints
clearly-relevant precision, per-band precision, and head-to-head, and **asserts** (non-zero exit on
failure): (a) hybrid's clearly-relevant precision ≥ an absolute floor, and (b) hybrid ≥ its best
single arm within noise, and (c) hybrid+rerank ≥ hybrid. This same harness is Phase A of the A/B
(one implementation, two uses). It is **not** part of the hermetic `tools/run_tests.sh` (needs
servers) but is a first-class, documented gate in `tools/README.md` and the Verification section.

`.coveragerc`/`tools/run_tests.sh`: unchanged; keep the 100% convention (report `--skip-covered`).

---

## Docs (updated alongside code, not after)

`docs/reference-translations.md` (flip the "deliberately not built: ANN" note; add hybrid/rerank
sections, new knobs, the A/B results table, and the blob-rendering rationale/example), `docs/architecture.md`
(Retriever paragraph + the "how positional context is shown" description, updated for the blob
format), `config/README.md` + `config/agents.toml` (`[context]` knobs + the rendering-change
migration note), `tools/README.md` (serve_reranker row), `src/translator/README.md` (retrieval
package + `context_packing.py`), top-level `README.md` (one-liner).

---

## Staged A/B (corpus A + corpus B; ~8000 reference + 2000 held-out; per rag-relevance-testing rule)

Sample 10 000 units; 8000 → reference corpus, 2000 → held-out queries, **unique and not
verbatim in the corpus**, length-stratified. **Phase A (cheap, no translation):** for {lexical,
embedding, hybrid, hybrid+rerank} retrieve top-k over ~1–2k queries; a Qwen relevance judge rates
each retrieved example (0/1/2); report precision + head-to-head; pick the winner. **Phase B
(confirm):** translate the 2000 held-out under baseline (best single retriever) vs the winner;
blinded order-randomised Qwen A/B judge (the established `ab_compare` protocol) **plus** consistency
(trigram overlap with established rendering — **see "Phase 8B outcome": specifying this metric
without specifying where "the established rendering" comes from is what let a circular version of
it ship; it must come from held-out ground truth, never from a retriever an arm also uses**)
**plus** rejection-rate. Honest framing: expected wins
are relevance/consistency/robustness; per-line quality may be a wash. Servers on 8080/8081/8082;
scratch in `/tmp/gt-*`; GPU freed after.

**Blob-rendering sanity check (reference OFF, isolates the prompt-format change from retrieval).**
Because the continuous-blob rendering is a default-on change independent of hybrid retrieval, run one
additional small A/B with reference retrieval disabled entirely: old fragmented `SOURCE:`/`TARGET:`
positional-context rendering vs. the new continuous-blob rendering, on a sample containing units
known to be sentence/idiom fragments continuing from a neighbour (filter the held-out set for units
whose `context_before`/`context_after` end/start mid-clause, or use corpus A's known split-sentence
lines from earlier sessions). Blinded Qwen A/B judge. This isolates whether the blob format itself
helps (the specific, named motivation for the change) rather than conflating it with the retrieval
work. Report honestly even if the effect is small on this corpus — the mechanism (continuous prose
reads better across a boundary than bulleted fragments) is the justification, not a guaranteed
win on any specific held-out sample.

---

## File change list

**New:** `src/translator/retrieval/{__init__,embedding,fusion,rerank,hybrid}.py`,
`src/translator/context_packing.py`, `src/translator/embedding.py` (shim),
`tools/serve_reranker.sh`, `tools/eval_retrieval.sh` (+ its Python
evaluator, e.g. `tools/lib/eval_retrieval.py`),
`tests/translator/retrieval/{test_fusion,test_rerank,test_hybrid}.py`,
`tests/translator/test_context_packing.py`.
**Modified:** `src/translator/roles.py` (Context fields + parsing), `src/translator/agents.py`
(`_unit_block` → `ContextPacker`), `src/translator/cli.py` (flags, `_load_reference`, `main` except
tuple), `tests/test_separation.py` (allowlist + paths), `tests/translator/{test_cli,test_agents,
test_embedding}.py` (import path), `config/agents.toml`, `config/README.md`, `requirements.txt`
(comment), `tools/check_environment.sh`, the six doc files above.

## Implementation phasing (each phase: code + its tests + lint green before next)

1. `retrieval/` package; move `embedding.py` (+ shim, `vectors_for`, row maps); `fusion.py` + tests.
2. `rerank.py` + tests; `serve_reranker.sh`.
3. `hybrid.py` + tests.
4. `cli.py` flags + `_load_reference` + config knobs (`roles.py`) + tests.
5. `context_packing.py` + `agents.py` integration + tests.
6. `test_separation.py` allowlist; docs; env/scripts.
7. Full `tools/run_tests.sh --coverage` (100%) + `tools/lint.sh` (ruff+mypy) green.
8. Staged A/B on GPU (Phase A → Phase B); record numbers in the doc. **8A done** (see
   `docs/reference-translations.md`, "Measured — and the result is a caution, not a win").
   **8B done** on corpus A **and** corpus B — see "Phase 8B outcome" below.
9. Commit/push; open PR (stack on `claude-branch` / PR #8, or fresh branch after it merges).

## Verification

- `tools/run_tests.sh --coverage` → 100%; `tools/lint.sh` clean; `tests/test_separation.py` green
  (transunit stdlib-only; numpy only under `translator/retrieval`).
- Live smoke: `serve_embeddings.sh` + `serve_reranker.sh` up; a real `--embedding-url --hybrid
  --rerank-url` run translates a small real-corpus sample end-to-end; kill each endpoint mid-run and
  confirm a clean, **logged, fail-loud** stop (no silent lexical fallback), journal intact.
- Existing *retrieval* behaviour unchanged when no new flags are passed (lexical default;
  single-retriever modes); config defaults keep the reference subsystem off unless opted in. The
  positional-context **rendering** changes for every run by design (blob + repeated target line) —
  verified by the `test_agents.py` regression assertion that the old fragmented format is gone, and
  by the blob-rendering sanity check in the A/B above.
- **Real-data relevance gate:** `tools/eval_retrieval.sh` on the GPU box passes its assertions —
  the hybrid (and hybrid+rerank) actually fetch relevant samples on the real corpus A / corpus B text, ≥ the
  single-arm baselines (this is also A/B Phase A).
- Staged A/B numbers (Phase A relevance + Phase B end-to-end) recorded in `docs/reference-translations.md`.

---

## Phase 8B outcome (run; the plan's own premise did not hold)

Phase B has now been run, once, with `tools/eval_translation.sh`. Full numbers, mechanism and the
revised recommendation live in `docs/reference-translations.md` ("End-to-end (Phase B)"); recorded
here is what it means **for this plan**.

**What ran, and how it deviated from the literal plan.** The plan specified sampling 10 000 units
into an 8000-unit reference corpus and a 2000-unit held-out set, across corpus A **and** corpus B.
What actually ran, on **both** corpora, ja→en, comparing the baseline (**embedding alone**) against
the Phase A winner (**hybrid + reranker**): corpus A at a **10,952-entry** corpus with **1000
held-out** units, and corpus B at a **2,422-entry** corpus with **600 held-out** units. The split
sizes differ from the literal 8000/2000 because they are what the real corpora can supply — corpus B
repeats roughly a third of its own lines, so a held-out line is frequently still present verbatim in
what remains, and only 897 of its rows were ever eligible. Running both corpora is what kept a
corpus-specific result from being published as a general one (see below). The blob-rendering
sanity check (reference off, old fragmented vs new continuous rendering) was **dropped, not
deferred**: the continuous rendering has since been in use long enough to be judged good on
its own terms, and resurrecting the deleted fragmented renderer purely to A/B against it
would have meant carrying a legacy prompt format for one experiment.

**The premise that failed — and the retraction that followed.** This plan's staging assumes Phase A
picks a winner and Phase B merely *confirms* it ("Phase B (confirm)"). It did not confirm it: Phase
B reported that on the metric the reference feature exists to serve — consistency with an
established rendering — the Phase A winner **lost**, 132 units to the embedding arm's 222, blamed
on step 5 of this plan's own hybrid algorithm (MMR at `reference_mmr_lambda = 0.7` suppressing the
redundancy that reproducing a rendering needs).

**That result is retracted (2026-07-27). It was an artifact of the measuring instrument.** The
consistency metric defined "the established rendering" as the most similar corpus entry found with
an `EmbeddingRetriever` at floor `0.55` — precisely what the `embedding` arm is configured to
retrieve and show the model. On 297 real queries the embedding arm had been shown that entry as its
own top example **297/297** times against the hybrid arm's **93/297**, at similarity **1.000 by
construction**. Re-measured against real held-out ground truth (each unit's own original
translation, which neither arm could see), the arms are a **dead heat**: 223 vs 222 of 593, tie
148, **z = 0.05**. Mean overlap with the real original: embedding **0.5928**, hybrid **0.5874**.
The MMR mechanism was also falsified independently — `tools/sweep_mmr.sh` swept
`reference_mmr_lambda` across `0.0`–`1.0` (corpus A, n=300, six values) and mean overlap was
**flat**, 0.098–0.106, spread 0.0086, non-monotone, `1.0` no better than `0.0`.

**The lesson for the staging.** The plan's stated worry was that a relevance-only Phase A cannot
pick an end-to-end winner. The real failure was one layer down: **Phase B's own headline metric was
not independent of the arms it compared.** A staged evaluation needs its Phase B instrument audited
for independence before its verdict is trusted — being end-to-end is not the same as being
unbiased. `tools/eval_translation.sh` now takes `--ground-truth` and **refuses** to run the proxy
while an arm is `embedding`, rather than warning.

**What held.** The plan's honest framing — "per-line quality may be a wash" — was correct on
both corpora: 161 to 147 with 551 ties on corpus A, 124 to 120 with 253 ties on corpus B. That
comparison used a blinded independent judge with no retriever in the loop, so it is untouched.

**What the second corpus changed — and what it could not.** Running corpus B as the plan required
is what stopped a corpus-specific *robustness* result being published as a general one: corpus B's
rejections were 0.5% (embedding) against 0.8% (hybrid+rerank), where corpus A saw 8.5% → 3.0%. The
reranker's job is filtering corrupt corpus rows, so a corpus without them has nothing for it to
filter and the second server buys nothing. Had only the noisier corpus A run, "hybrid+rerank is
markedly more robust" would have gone into the docs as a property of the retriever rather than of
that corpus.

But the two-corpus rule did **not** catch the consistency artifact — it made it *more* convincing.
The loss "replicated" at almost the same ratio (193 vs 122 on corpus B, 222 vs 132 on corpus A) and
that replication was cited here as the finding's strongest evidence. **A structural bias reproduces
perfectly.** Replication corroborates only when the runs do not share the instrument; two corpora
through one biased metric is one experiment, not two.

**Consequence for the shipped defaults:** none. The subsystem stays off by default; the change is
to the *advice*, which is now simply: quality is a wash, consistency shows no measurable
difference, and **robustness — corpus-dependent — is the only measured differentiator**, so pick a
retriever by measuring your own rejection rate. Bare hybrid still not recommended.
`config/agents.toml` and `config/README.md` carry that at the point of decision. The note telling a
consistency-focused project to raise `reference_mmr_lambda` toward `1.0` has been **removed** from
both, as the sweep found no effect across the range.

**Also measured after the fact (not in the original plan):** no-reference vs embedding at n=1000
on corpus A, using the `none` arm added to `tools/eval_translation.sh` because the harness as
first written could compare retrievers against each other but not against not retrieving at all
— the single most important question about the feature. Result: quality a wash (165 to 140, not
significant), rejection rate nearly tripled (3.0% → 8.8%). Its consistency figure ("~3× better")
used the same biased metric and is **withdrawn** — the direction is plausible and near-tautological,
the magnitude is not a measurement. See `docs/reference-translations.md`.

**Open, unmeasured:**
weighted fusion (still deliberately unbuilt); reference-on vs reference-off consistency against
real ground truth (only the retriever-vs-retriever comparison was re-run).
