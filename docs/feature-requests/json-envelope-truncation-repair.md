# Plan — recovering a truncated JSON *envelope*, and giving the translate path its repair budget

Filed by a downstream subtitle carrier using this library, as
`llm-translator-translate-path-json-truncation-unrepaired.md` (Bielik-11B via the
`json_object` backend, EN→PL, 327 units across three real runs). This is the implementation plan
for it.

## The report, verified against current `HEAD`

Every structural claim in the request checks out against the code as it stands today (the request
was filed against pin `5b62ee3`):

| Claim | Verified |
|---|---|
| `complete_json` raises on any `JSONDecodeError`, with no recovery | `backend/client.py` — one `try/except json.JSONDecodeError` → `LlmContentError` |
| `translate()` calls it uncaught | `agents.py::translate` |
| `_generate`'s repair loop has no `try/except` around `self.translate(...)` | `agents.py::_generate` |
| the error lands in `process()`'s outer handler → immediate `REJECTED` | `agents.py::process` |
| **`max_repairs` is therefore never spent on it** | confirmed — the loop only repairs *rule violations on parsed text* |

So the asymmetry the request describes is real: a mechanically-defective translation gets
`max_repairs` extra attempts; a reply whose JSON envelope is one character short gets **zero**,
and costs the whole unit.

**It is also not the truncation case we already handle.** `LlmTruncationError` fires on
`finish_reason == "length"` (the token ceiling). Here the model emits EOS early, so
`finish_reason` is `"stop"`, the truncation check never fires, and it falls through to the
generic parse failure. Worth stating explicitly so the fix is not mistaken for a duplicate.

## The thing the request under-weights, and why it changes the design

The request's primary proposal is to append the missing `}` and carry on, "no extra model call".
That is attractive, and for the *envelope* it is correct. But it quietly converts a **loud**
failure into a **potentially silent** one, which is the one trade this project's rules forbid.

Look at the request's own evidence. The recovered value ends:

> `... Nie ma sposobu, aby to przewidzieć. Więcej na ten temat później. Po wejściu na`

`Po wejściu na` is *"After entering the"* — the string is **syntactically closed but
semantically cut off mid-clause**. The request characterises this as "a well-formed value with one
character missing from the envelope"; its own example shows the value is truncated too. Both can
be true (the model closed the quote and stopped early), and that is exactly the danger: repairing
the brace yields a *parseable, plausible-looking, incomplete* translation.

What would catch that downstream? I enumerated the mechanical checks —
`nonempty`, `placeholders`, `untranslated`, `control_character`, `line_width`, `forbidden`,
`glossary`, `name_respelled`. **There is no length-ratio or truncation guard among them.** A
truncated translation with no placeholders (or whose placeholders all appear early) passes every
one of them, leaving only the review panel between it and a `VERIFIED` unit. The panel is good,
but "a reviewer will probably notice" is not a guarantee, and today's behaviour — `REJECTED`,
never shipped — is strictly safer than a silent maybe.

**So: repair the envelope, but never *trust* the result.** The repaired text is a *candidate* to
be re-checked and re-reviewed, not an answer. That single decision shapes everything below.

## Design

Two concerns, kept in the layers that own them:

- **Can the envelope be recovered?** A parsing question → `translator/backend/client.py`.
- **May the recovered content be trusted?** A policy question → `translator/agents.py`.

### 1. `client.py` — classify precisely, recover narrowly

Add a helper and a distinct error type:

```python
def _close_truncated_json_object(text: str) -> dict[str, Any] | None:
    """Parse a flat JSON object whose closing brace (and at most its final closing quote) the
    model never emitted. Returns None for anything else -- this recovers ONE verified failure
    shape, it is not a general "make this JSON parse" routine.

    Recovers the *envelope only*: appending the closing quote closes the value string wherever
    the model stopped, so a value severed mid-clause yields an object as valid as a finished
    one, and the two are indistinguishable here. Hence the caller obligation below."""

class LlmIncompleteJsonError(LlmContentError):
    """The reply's JSON envelope was cut short but its content parsed once closed.

    Carries `.recovered` (the parsed object). Distinct from a garbled reply because the
    response is *usable but unverified* -- the caller decides whether to trust it, and the
    engine deliberately does not.
    """
```

`complete_json` tries the strict parse first, exactly as now; only on `JSONDecodeError` does it
attempt the narrow close, and it raises `LlmIncompleteJsonError` (carrying the recovered object)
rather than returning it. **Raising, not returning, is the point**: every existing caller keeps
today's safe behaviour by default, and only a caller that explicitly opts in can use the payload.

Deliberate constraints, so this cannot become a silent garbage-accepter:
- only a **flat** object (the schemas in use are flat — `client.py`'s docstring already says so);
- append at most `"` then `}`, then re-parse **strictly**, and require the result to satisfy the
  same schema validation the normal path applies;
- reject anything with unbalanced brackets, trailing commas, or nested containers;
- an already-parseable reply never touches this path.

### 2. `agents.py` — spend the repair budget that already exists

`_generate`'s loop gains a `try/except` around `self.translate(...)`, converting a content
failure into a repair round with a specific instruction, instead of letting it unwind to
`process()`:

```python
try:
    target = self.translate(unit, attempt)
except LlmIncompleteJsonError as exc:
    # Envelope recovered, content unverified: re-check it like any other candidate, and force
    # at least one more round rather than accepting it as-is.
    ...
except LlmContentError as exc:
    # Unparseable/garbled: nothing to check, so re-ask with a pointed instruction.
    ...
```

Precision matters in what is caught. `LlmRefusalError` and `LlmTruncationError` are both
`LlmContentError` subclasses and must **keep** their current terminal behaviour — a refusal
re-asked is wasted budget (and reads as jailbreaking), and a ceiling hit "is a property of this
prompt, so another attempt loops the same way" (`client.py`'s own words). They are re-raised
explicitly, not swept in by the base class.

### 3. Never let a recovered translation reach `VERIFIED` silently

When a unit's final answer came from a repaired envelope and the repair budget ran out, the
outcome is `TRANSLATED` (kept, flagged for review) with an explanatory `error`, **not**
`VERIFIED`. This reuses the exact precedent already in `process()` for "review incomplete" and
"over the length budget after repairs": keep the useful text, refuse to certify it. The caller
gets the translation *and* the signal, which is strictly better than both today's `REJECTED`
(loses the text) and a naive repair (loses the signal).

### 4. Optional, and worth measuring first: a truncation guard

A `length_ratio` mechanical check would be the general safety net for this whole class. It is
**deliberately out of scope here** and listed as a follow-up, because expected length ratios vary
enormously by pair (JA→EN expands, EN→JA contracts) and a naive threshold would produce false
positives on exactly the short lines this engine handles most. If added, it should flag only
extreme cases and be configurable per project.

## Also worth checking first (it could delete the whole problem)

The `json_object` backend exists because Bielik's build rejected `json_schema`
("Failed to initialize samplers"). That diagnosis is old. **If Bielik now accepts a
grammar-constrained request, this failure mode disappears entirely** — a grammar cannot omit the
closing brace, which is precisely why `SchemaBackend` models never hit this. Re-testing that is
an hour of work and could make everything above unnecessary for this reporter, so it belongs
first.

## Phases

1. **Reproduce and characterise** (no production code). Serve Bielik, drive the `json_object`
   path, capture raw bodies. Answer: how often is the brace genuinely the only thing missing;
   is the recovered value semantically complete or cut mid-clause; what `finish_reason` comes
   back. **Also re-test whether `json_schema` works on this build.** This phase decides whether
   to proceed, and its findings are recorded in the doc either way.
2. **`client.py`**: `_close_truncated_json_object` + `LlmIncompleteJsonError` + wiring. Tests:
   both `JSONDecodeError` shapes from the request, nested/garbled/unbalanced bodies refused,
   schema validation still applied to the recovered object, an ordinary reply unaffected.
3. **`agents.py`**: repair-loop integration; refusal/ceiling stay terminal. Tests: budget is
   actually spent, the repaired candidate goes through `check_mechanical`, exhaustion yields
   `TRANSLATED`-with-reason (never a silent `VERIFIED`), and current behaviour for every other
   `LlmContentError` is unchanged.
4. **Config + docs**: a knob to disable envelope repair if a project wants strictness; update
   `docs/` and `config/README.md`.
5. **Validate on the reporter's data**: hand the branch to the reporting carrier and re-run the
   corpus that produced the three unresolved units. The success criterion is theirs, not ours —
   those three units resolving without a caller-side batch retry.

## Phase 1 findings (reproduced live; decision: proceed)

**`json_schema` re-test: still broken, not a way out.** Sent a real `response_format:
{"type":"json_schema", ...}` request directly to a live Bielik-11B-v2.3 server (the same GGUF/build
the reporting carrier serves). It fails identically to the original diagnosis:
`{"error":{"code":400,"message":"Failed to initialize samplers: std::exception", ...}}`. The
`json_object` path, and therefore this whole repair mechanism, remains necessary for Bielik/EuroLLM.

**Reproduction.** A naive attempt with a simplified prompt and short/medium synthetic sentences (80
calls) produced zero failures of any kind. The failure only showed up once the reproduction used full
production fidelity: the real `TranslationAgents.translate()` call path, the reporting carrier's actual
`[translate]` instructions (the full subtitle-specific system prompt, not a stand-in), and real source
lines from the subtitle corpus that produced the original report — specifically its **longest,
most punctuation-sparse, run-on ASR lines** (250–390 characters, no internal commas/periods). Driving
the 25 longest units in that corpus, 8 repetitions each (200 calls, temperature 0.3 as
`translate()` hard-codes, 2-way concurrent matching `--parallel 2`) reproduced it directly:

- **3/200 trials failed (1.5%)**, both failing units among the two longest in the corpus (391 and
  351 source characters) — consistent with the report's own framing ("a low but non-trivial
  per-generation probability", concentrated on the long run-on blocks it explicitly called out
  elsewhere).
- **`finish_reason` is confirmed `"stop"`, never `"length"`**, in all 3 cases: none were caught by
  the existing `LlmTruncationError` path (which fires only on `"length"`), and every failing body
  (631–689 chars) is far short of the unit's actual `max_tokens` budget
  (`translate_budget`: `391 * 8 = 3128`, capped at the `1024`-token ceiling — several thousand
  characters of headroom). This is genuinely the spontaneous-EOS failure mode, not a disguised
  ceiling hit.
- **Two distinct sub-shapes, both already anticipated by this doc's proposed fix, and now both
  confirmed to occur:**
  - **Shape A** (2 of 3, matches the original report exactly): the `"translation"` string closes
    correctly; only the object's `}` is missing (`Expecting ',' delimiter` at EOF). Repairable by
    appending `}` alone.
  - **Shape B** (1 of 3, a real nuance the original report's own evidence didn't show, though its
    proposed fix already accounted for it: *"append the missing `}` (and `\"` if the string itself
    didn't close)"*): the model's own nested single-quoted dialogue (`'...'` inside the translated
    text) apparently confuses it, and it emits a bare `}` without ever closing the JSON string's
    `"` (`Unterminated string starting at` the value's opening quote). Repairable only by appending
    `"}` (quote *then* brace) — appending `}` alone is not sufficient here. **The repair function
    must attempt both, not just the brace-only case the report's own examples showed.**
- **Semantic completeness of the repaired value is genuinely inconsistent — this is the design
  tension in this doc's "Never let a recovered translation reach `VERIFIED` silently" section,
  confirmed rather than theoretical:**
  - One repaired case is fully complete: ends on a closed quote and terminal `?` (proper sentence).
  - Another (same narrative unit, different sampling draw) is genuinely cut mid-clause: ends on
    `"...żebyś mógł"` ("...that you could") with no completing verb and no terminal punctuation at
    all — a real defect a downstream check must be able to catch.
  - A third is a trap for any cheap completeness heuristic: it ends in a literal `"..."` the model
    itself generated as trailing-off dialogue, which *reads* finished but is grammatically the same
    unfinished-clause shape as the second case. A naive "ends in punctuation" check would accept it;
    a naive "ends in an ellipsis, therefore intentional" check would also accept it. **This is
    concrete evidence against adding any mechanical completeness heuristic** — the repaired
    candidate must go back through the full existing check/review pipeline (as already designed
    above), never be trusted on its syntactic shape alone.

**Conclusion: proceed to Phase 2** as designed, with one refinement carried into the `client.py`
spec: `_close_truncated_json_object` must try the brace-only close first and the quote+brace close
second (or otherwise determine whether the string itself closed), since both shapes are now
confirmed live rather than assumed from the report's single example.

**Reproduction artifacts** (not part of this repo; kept for reference): `/tmp/gt-truncation/`,
scripts `reproduce2.py`/`reproduce3.py`, results in `results3.json`.

## Phase 5: live validation against the fixed engine

Implemented (Phases 2-4), committed to `claude-branch` (`a63b319`), 100% coverage, lint clean.
Validated live against real Bielik-11B-v2.3, the reporting carrier's real `agents.toml`, and the
real subtitle corpus that produced the report -- this time driving the full, fixed
`TranslationAgents.process()` pipeline (translate → mechanical check → review panel → revise), not
just the raw `complete_json` path Phase 1 used.

- **224 real end-to-end `process()` calls, zero regressions.** 200 calls across the 25 longest
  units (8 reps each) plus 24 further calls on the single highest-risk unit from Phase 1
  (`21a1d2781bae6224`, which alone produced 2 of Phase 1's 3 raw failures). **Zero `REJECTED`
  outcomes.** Every unit resolved to `TRANSLATED` or `VERIFIED`, exactly the guarantee this fix
  exists to provide.
- **No live envelope-truncation event landed in this window.** Neither run happened to trigger
  the underlying stochastic failure -- consistent with Phase 1's own estimate (3/200, a 95% CI of
  roughly 0.3%-4.3%): 224 further trials landing zero is not improbable at that rate, and the
  same unit's own Phase 1 sub-sample (2/8) is too small to trust as a stable per-unit rate. This
  is an honest gap: Phase 5 does **not** contain a directly-observed live full-pipeline recovery
  (translate → `LlmIncompleteJsonError` → repair round → `TRANSLATED`/`VERIFIED` with the
  envelope-recovery reason). What it does contain, and what the conclusion below relies on
  instead:
  - **The exact failure is reproduced live** (Phase 1, same corpus/config/model) and **the exact
    recovery mechanism is exercised live** at the `client.py` level (Phase 2's
    `_close_truncated_json_object` is the literal function that already recovered both shapes
    found in Phase 1's raw bodies).
  - **The full `agents.py` wiring is proven deterministically**, not just by construction:
    Phase 3's `TestEnvelopeTruncationRepair` suite scripts a fake client to raise
    `LlmIncompleteJsonError` through the identical `_generate`/`process()` code path this live run
    exercised, and asserts every claim this doc makes (budget spent, mechanical re-check applied,
    exhaustion yields `TRANSLATED`-with-reason, never silent `VERIFIED`, the strict-mode config
    flag). The only thing live validation adds beyond that is confidence that the *plumbing*
    (real config, real prompt, real server) is wired correctly end-to-end -- which this run
    confirms (zero crashes, zero unexpected exceptions, zero regressions) even without directly
    observing the rare branch fire.
- **This unit's dominant real defect is a different, pre-existing mechanism.** Most of the 24
  targeted calls on `21a1d2781bae6224` (a very long, dense narrative unit) hit
  `"over the length budget after repairs"` -- an unrelated, already-existing path (this unit's
  Polish rendering legitimately exceeds its character budget), which this fix does not touch.

**Conclusion:** ship it. The mechanism is proven both live (Phase 1 + this section) and
deterministically (Phase 2 + Phase 3), and this run adds a clean end-to-end regression check on
real production wiring with zero unit losses. Re-running a larger live sweep to directly catch the
recovery branch firing end-to-end would need substantially more GPU time for a low-probability
event already characterised in Phase 1; not done here as the marginal evidence would be small
given what Phases 1-3 already establish. If the reporting carrier's own future runs hit
`"translated recovered from a truncated JSON envelope"` in a unit's notes, that is this mechanism
firing for real in production, and worth a quick sanity read of the kept text.

## Post-merge audit fixes

- **A fenced *and* truncated reply never reached the repair at all.** `_strip_code_fence`
  required a closing fence and otherwise returned the text with the fence still attached — which
  is precisely what a truncated fenced reply looks like. For any model that wraps JSON in a fence
  "despite instructions", the payload therefore stayed unparseable for a reason unrelated to the
  envelope and the unit was lost as before this feature existed. Fixed: an opening ` ``` ` is
  unambiguous on its own, so the opening line is always dropped and the closing fence only when
  present. A properly-closed fence and an unfenced reply behave exactly as before.
- **`_close_truncated_json_object`'s docstring claimed more than the function delivers.** It said
  the model "emits EOS immediately after finishing the requested content" — true of the `}` branch,
  false of the `"}` branch, which closes the string at an arbitrary cut point and so recovers a
  mid-clause truncation just as cleanly (this doc's own Phase 1 findings record exactly that, the
  `"...żebyś mógł"` case). The docstring now states what each suffix recovers, that the two are
  indistinguishable at that layer, and why the caller's re-check/re-review obligation therefore
  cannot be removed as redundant. Pinned by tests in `tests/backend/test_client.py`.

## What this does not do

- It does **not** retry a refusal or a ceiling-hit truncation (both stay terminal, by design).
- It does **not** add a schema-less `complete()` fallback. The request itself warns that one
  over-generates past the requested unit, hallucinating sentences the source never contained;
  without a stop condition that is a worse failure than the one being fixed.
- It does **not** attempt general JSON repair. One verified shape, or raise.
