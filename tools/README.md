# tools/

Hardened scripts for the routines that take more than one command. Each validates its own
preconditions and fails with an actionable message rather than crashing opaquely.

| Script | What it does |
|---|---|
| `setup_python_env.sh` | create `.venv/` and install the exact pinned `requirements.txt` |
| `check_environment.sh` | report, read-only, whether the environment can run the code and tests |
| `run_tests.sh` | run the suite (needs no server/GPU/network); `--coverage` for a report |
| `lint.sh` | static analysis: ruff (ruleset pinned in `ruff.toml`) + mypy; `--fix` to autofix |
| `serve_model.sh` | start a local OpenAI-compatible server (ollama or llama.cpp) |
| `serve_embeddings.sh` | start a local `/v1/embeddings` server for the optional `--embedding-url` semantic reference retriever |
| `serve_reranker.sh` | start a local `/v1/rerank` server for the optional `--hybrid --rerank-url` cross-encoder reranking stage; auto-downloads a GGUF on first use |
| `eval_retrieval.sh` | measure, on real text, whether reference retrieval fetches *relevant* material — a gate, not just a report (needs servers; not part of `run_tests.sh`) |
| `build_eval_split.sh` | build a reproducible reference corpus + held-out query set from a real journal, for either evaluation below (a shortfall is an error, never a quietly smaller sample) |
| `eval_translation.sh` | translate a held-out set under two retrieval arms and compare the OUTPUT: blinded A/B quality (reported), consistency with the held-out lines' own established translations (gated; pass `--ground-truth`), and rejection rate (gated). Needs servers; not part of `run_tests.sh` |
| `sweep_mmr.sh` | sweep `reference_mmr_lambda` end-to-end and measure mean overlap, a "reproduced" rate and rejection rate at each value — an absolute metric, so each λ is translated once instead of A/B'd pairwise. Needs servers; not part of `run_tests.sh` |
| `lib/common.sh` | shared helpers, sourced by the others |
| `lib/eval_retrieval.py` | the evaluator `eval_retrieval.sh` drives (run it through the script, which checks the servers first) |
| `lib/build_eval_split.py` | the splitter `build_eval_split.sh` drives |
| `lib/eval_translation.py` | the evaluator `eval_translation.sh` drives (run it through the script, which checks the servers first) |
| `lib/sweep_mmr.py` | the sweeper `sweep_mmr.sh` drives |

## From a fresh checkout

```bash
tools/setup_python_env.sh     # one-time: build the pinned environment
tools/check_environment.sh    # confirm it is ready
tools/run_tests.sh            # run everything
```

The scripts find `.venv/` on their own, so you do not need to activate it first.

## For a real translation run

The tests never touch a model, but translating for real needs a server:

```bash
tools/serve_model.sh qwen3:14b                 # ollama, an OpenAI-compatible endpoint
# then, against it:
PYTHONPATH=src python -m translator.cli \
    --base-url http://127.0.0.1:11434/v1 --model qwen3:14b --backend auto \
    -c work/units.jsonl -j work/journal.jsonl
```

`--backend auto` picks the request shaping from the model id (Qwen → grammar-constrained
JSON, Bielik/EuroLLM → JSON-object). See `docs/writing-a-backend.md` to add a model.

`serve_model.sh` serves `--parallel 2` slots by default, matching the translator's default
`--concurrency 2` so the two in-flight requests do not serialise. Hardware-specific flags the
design leaves out — GPU offload, context size — go after a literal `--`, e.g.
`tools/serve_model.sh model.gguf --engine llama -- -ngl 999 -c 8192`.

For the optional **semantic reference retriever**, start a second, tiny embedding server
alongside the translation one and point the CLI at it:

```bash
tools/serve_embeddings.sh                 # EmbeddingGemma-300M on :8081 (auto-downloaded)
# then add to the translate command:  --embedding-url http://127.0.0.1:8081/v1
```

To fuse the lexical and embedding retrievers, optionally with a reranking stage, add `--hybrid`
and start a third, tiny server:

```bash
tools/serve_reranker.sh                   # bge-reranker-v2-m3 on :8082 (auto-downloaded)
# then add:  --embedding-url http://127.0.0.1:8081/v1 --hybrid --rerank-url http://127.0.0.1:8082/v1
```

See `docs/reference-translations.md` ("The optional embedding retriever", "Hybrid retrieval").

## Checking that retrieval actually retrieves something useful

`run_tests.sh` pins retrieval *mechanics* with hand-picked vectors and needs no server. Whether a
retriever fetches genuinely relevant examples from a real corpus in a real language is a different
question, and needs real servers plus an independent judge — so it is a separate gate:

```bash
tools/eval_retrieval.sh \
    --corpus reference/prior-translation.jsonl \
    --queries-from work/units.jsonl \
    --rerank-url http://127.0.0.1:8082/v1     # optional; adds the hybrid+rerank variant
```

It samples **held-out, unique, non-corpus-duplicate** queries (a query present verbatim in the
corpus retrieves itself at similarity 1.0 and would inflate every score), retrieves with each
variant — lexical, embedding, hybrid, hybrid+rerank — and has an independent LLM rate each
retrieved example `0`/`1`/`2` for relevance. It **exits non-zero** if the hybrid retriever falls
below an absolute precision floor or below its own best single arm, so it can gate a change rather
than merely describe one. It refuses to run, naming the URL and the script that starts it, if any
server it needs is down — never a silent skip.

## Checking what the retrieval actually *ships*

Relevant retrieval and better output are different questions, and the second needs real
translation. `build_eval_split.sh` builds a reproducible corpus/held-out split from a real
journal; `eval_translation.sh` then translates the held-out set under two retrieval arms and
compares the output — blinded A/B quality, consistency, and rejection rate.

**Pass `--ground-truth` (the journal the held-out queries came from).** Without it the consistency
measure falls back to a proxy — the most similar corpus entry, found with the embedding retriever —
and that proxy is *circular* whenever an arm uses the same retriever: it scores that arm against its
own top pick, which the arm matches at similarity 1.000 by construction. That is not hypothetical;
it produced, and then cleanly "replicated", a false headline finding. The evaluator now **refuses**
to run the proxy while an arm is `embedding` rather than warning about it. Full account:
`docs/reference-translations.md`, "The consistency finding was an artifact".

Its first run (1000 held-out ja→en units from a ~11k-line game corpus, embedding alone vs
`--hybrid --rerank-url`) measured: per-line quality a wash, and the hybrid+rerank arm markedly more
robust (rejections 8.5% → 3.0%) — but only on that corpus; on a cleaner one the difference vanished.
Its consistency verdict was the retracted one; re-measured against real ground truth the arms are a
dead heat. Read the numbers in `docs/reference-translations.md` ("End-to-end (Phase B)") before
treating either arm as the default choice — and note that a variant winning `eval_retrieval.sh` is
not thereby the end-to-end winner.
