# tests

The translator's own test suite. It tests **this library alone** — the engine and the
contract — and never any overlying carrier adapter, because those are deliberately not part
of this repository. Where the adapter *protocol* needs exercising, a minimal fake carrier
(`tests/fixtures/fakecarrier/`) stands in for a real one.

Every model call is served by a scripted fake or an in-memory HTTP transport, so **the suite
needs no inference server, no GPU, and no network**. Run it with:

```bash
tools/run_tests.sh              # or: PYTHONPATH=src python -m pytest
tools/run_tests.sh --coverage   # statement AND branch coverage
```

Needs `pytest` and `hypothesis` (both pinned in `requirements.txt`; `tools/check_environment.sh`
verifies them).

## Layout

| Path | What it covers |
|---|---|
| `transunit/` | the contract: units & catalogue durability, glossary, reference retrieval (loaders, lexical + growable retrievers, determinism, thread-safety, plus adversarial scale/unicode/floor-boundary/heavy-concurrency probing), placeholders, width, both language detectors, the adapter protocol |
| `translator/` | the engine: the agent harness, rules, roles, memory, the runner, the CLI, prompt hygiene, context packing (the continuous surrounding-context passage, the ±1 neighbour guarantee, dedup, budget), and reference-retrieval wiring (config, prompt block, self-population). Also the failure surfaces: `test_error_modes.py` (every one of the fourteen fatal error types exits 1, prints once and reaches the `--log-file`, with a guard that reads the except tuple off the live function so the set cannot drift), `test_concurrency.py` (the thread-safety the harness documents, under real contention), and `test_end_to_end_stress.py` (payloads at both length extremes, 500 units at concurrency 8, duplicate grouping, resume) |
| `translator/retrieval/` | the optional retrievers, each against an in-memory transport so no server is needed: the pure fusion math (RRF, MMR), the rerank client and its whole error taxonomy, and the hybrid retriever (fusion, relevance scale, fail-loud, RAII) |
| `backend/` | the model layer: the backend registry & request shaping, and the client against a mock transport |
| `property/` | property-based and fuzz tests (hypothesis): display width, the token estimate, the fusion math, the lexical/growable retrievers, the JSON-envelope trio and placeholders. Each test names the *invariant* it defends rather than the implementation. Derandomised in `property/conftest.py`, so a counterexample found once reproduces exactly |
| `tools/` | the evaluation harnesses in `tools/lib/` — the A/B judge's blinding and de-blinding (asserted per unit against the slot the transport actually saw), the consistency and rejection metrics, the gates, the corpus/query split builder, and the retrieval-relevance sweep. Judges run against an in-memory transport and retrievers are stubbed, so these need no server either |
| `test_separation.py` | the module boundaries — the contract is stdlib-only, the translator imports no carrier adapter, and each half runs without the other |
| `test_durability.py` | the atomic-write, torn-record and resume guarantees, exercised by injecting the failures they defend against |
| `fixtures/fakecarrier/` | a fake carrier adapter, the smallest thing that satisfies `transunit.adapter.Adapter` |
| `conftest.py` | puts `fixtures/` on `sys.path` so the fake adapter is importable by name (`tools/conftest.py` does the same for `tools/lib`) |

## Conventions

- Tests are grouped into classes by behaviour, with descriptive method names.
- A docstring explains *why* a non-obvious case matters — several are regression tests for
  bugs that reached output, and the docstring records the failure.
- Filesystem tests use `tmp_path`; nothing writes outside it.
- Coverage is measured with **branches**, not statements alone: the suite once reported 100%
  statement coverage while four branches were never taken, and two of those hid real defects.
  `src/` and `tools/lib/` are both at 100% statement and branch.
- An `xfail` is always `strict=True` and its reason names a specific, diagnosed defect or a
  deliberate limitation. It is never a way to quiet a failing test.
