# translator.backend — the model-facing layer

The intermediate layer between the translator and a local language model. Named a
**backend**, never an "adapter": "adapter" is the *carrier* side (`transunit.adapter`),
and reusing the word for both is the ambiguity this name avoids.

| Module | Responsibility |
|---|---|
| `client.py` | `LlmClient` — transport only: HTTP, retry/backoff, error taxonomy, usage stats. Model-agnostic. |
| `core.py` | `Backend` + `SchemaBackend`/`JsonObjectBackend`, `StructuredRequest`, and the name→backend registry |
| `profiles/` | one small module per model family, each registering a backend |

## Why this layer exists

"The same request" is not "the same bytes on the wire" across models. The sharpest
difference — the one that motivated the split — is how a model honours a request for
**structured (JSON) output**:

- **Qwen / most builds** honour strict `json_schema` (grammar-constrained decoding). This
  is `SchemaBackend`, the default.
- **Bielik / EuroLLM** reject that grammar ("Failed to initialize samplers") but honour
  `json_object` with the shape described in the prompt. This is `JsonObjectBackend`.

Before this layer, that difference lived as conditionals inside the client keyed on a
config string, and switching from Qwen to Bielik "broke something that shouldn't have".
Now each model family is one object; the translator does not change when the model does.

## Selecting a backend

```python
from translator.backend import LlmClient, ServerConfig, resolve_backend

cfg = ServerConfig(model="bielik-11b")
client = LlmClient(cfg, backend=resolve_backend("auto", cfg.model))   # -> bielik
```

- By name: `get_backend("bielik")`.
- `resolve_backend("auto", model)` picks by the served model id (hint match), falling back
  to `generic`. The caller reports the choice, so it is never silent.
- If a server still rejects a schema request, the client's error **names the fix**
  (`--backend bielik`) rather than degrading silently.

## Prompt-budget guards

Truncation is caught from both ends. Output cut off at the ceiling (`finish_reason == "length"`)
becomes a typed `LlmTruncationError`, recorded against the unit, never shipped. A distinct
failure — the model choosing to stop (`finish_reason == "stop"`) right after closing the
`"translation"` value but before the object's own closing brace, well under the ceiling — is
caught by `complete_json` on a `json_object` backend: it tries closing the envelope and re-parsing
strictly, and on success raises `LlmIncompleteJsonError` (carrying the recovered object) instead
of the base `LlmContentError`, so a caller that wants to (`translator.agents`, on its repair
budget) can treat it as an unverified candidate rather than losing the unit for free. Input
overflow is
caught twice: the server's rejection is rewritten into a clear "prompt is larger than the context
window" with the knobs that shrink it; and, when `ServerConfig.context_window` is set, the client
warns **once** *before* sending, when the estimated prompt (a tokenizer-free, CJK-aware char
estimate) plus the output budget crosses 80% of the window. `UsageStats` also carries total and
peak prompt tokens for the run summary.

## Adding a model

Drop a module in `profiles/`:

```python
# profiles/mistral.py
from ..core import SchemaBackend, register_backend
register_backend(SchemaBackend("mistral", model_hints=("mistral",)))
```

and import it from `profiles/__init__.py`. Nothing else in the codebase changes. See
`docs/writing-a-backend.md`.
