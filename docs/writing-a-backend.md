# Wiring in a new LLM (writing a backend)

A **backend** is the model-facing side of the translator: it shapes a request for one family
of local model. It is the layer that made "switch from Qwen to Bielik" a one-word change
instead of a code edit. This guide is everything you need to serve a new model, from "it
already works" to "it needs a request shape nothing else uses".

> Vocabulary: the model side is a **backend** (`translator.backend`). The *text* side — the
> thing that turns subtitles or game files into units — is a carrier **adapter**
> (`transunit.adapter`), covered in `writing-a-carrier-adapter.md`. The two are never both
> called "adapter".

## 0. First: does your model already work?

The translator talks to any server exposing an OpenAI-compatible `POST /v1/chat/completions`
with a `response_format`. If your model's server accepts a **strict `json_schema`** response
format (grammar-constrained decoding), the built-in `generic` backend already handles it —
you write nothing. This covers Qwen and most llama.cpp, vLLM, and ollama builds.

```bash
python -m translator.cli --model my-model --backend generic ...   # or --backend auto
```

You only write or select a backend when the model wants its structured requests shaped
differently. In practice that is one question:

## 1. The one question: how does the model do structured output?

Every role asks the model for a JSON object matching a schema (the translation, or a review
verdict). Local servers honour that request in one of two ways, and this is the entire reason
backends exist:

| The server… | Strategy | Backend class |
|---|---|---|
| compiles a strict `json_schema` grammar and constrains decoding to it | grammar-constrained | `SchemaBackend` |
| rejects that grammar but accepts `json_object` (valid JSON, shape unconstrained) | shape-in-prompt | `JsonObjectBackend` |

`SchemaBackend` is the strongest guarantee and the default. `JsonObjectBackend` is for builds
that fail the grammar — the classic symptom is **Bielik**, whose llama.cpp build answers a
strict-schema request with *"Failed to initialize samplers"*. It honours `json_object`
perfectly once the shape is described in the prompt, which is what `JsonObjectBackend` adds.

Either way, the client validates the returned JSON against the schema, so `json_object` is a
weaker *constraint*, never a weaker *check*: a malformed shape is caught and retried, not
shipped.

If you are unsure which your model needs, try `--backend generic` first; if the server rejects
the request, the client's error tells you to switch (see §5).

## 2. Add a profile

A backend is registered by a small module in `src/translator/backend/profiles/`. To add a
model that uses grammar-constrained JSON:

```python
# src/translator/backend/profiles/mistral.py
from ..core import SchemaBackend, register_backend

register_backend(SchemaBackend("mistral", model_hints=("mistral",)))
```

For a model whose build needs the `json_object` strategy:

```python
# src/translator/backend/profiles/gemma.py
from ..core import JsonObjectBackend, register_backend

register_backend(JsonObjectBackend("gemma", model_hints=("gemma",)))
```

Then import it so it self-registers, in `src/translator/backend/profiles/__init__.py`:

```python
from . import bielik, eurollm, gemma, generic, mistral, qwen  # noqa: F401
```

That is the whole change. Nothing in the translator, the client, the runner, or the CLI is
touched. The constructor arguments:

- **`name`** — how the backend is selected: `--backend mistral`.
- **`aliases=(...)`** — extra names it also answers to (e.g. `generic` also answers to
  `openai` and `json_schema`).
- **`model_hints=(...)`** — lower-cased substrings of a served model id that point to this
  backend. They drive `--backend auto` and the diagnostic hint in §5. Give the hints your
  model ids actually contain (`"qwen"`, `"bielik-11b"`, …).

## 3. Selecting a backend at run time

```bash
# by name
python -m translator.cli --model bielik-11b --backend bielik ...

# by the model id — auto picks the backend whose hint matches, else 'generic'
python -m translator.cli --model bielik-11b-v2.3 --backend auto ...
```

From the library:

```python
from translator.backend import LlmClient, ServerConfig, resolve_backend, get_backend

cfg = ServerConfig(model="bielik-11b")
client = LlmClient(cfg, backend=resolve_backend("auto", cfg.model))   # -> the bielik backend
# or explicitly:
client = LlmClient(cfg, backend=get_backend("bielik"))
```

`auto` is what hardens a model switch: change the model, and the request shape follows,
because the caller reports the chosen backend (the CLI prints `model 'x' via backend 'y'`) —
the selection is never silent.

## 4. Serving the model

The translator needs the server *running*; how you start it is separate. `tools/serve_model.sh`
drives the two common options and prints the exact `--base-url` / `--model` / `--backend`:

```bash
tools/serve_model.sh qwen3:14b                          # ollama (registry models)
tools/serve_model.sh ./models/Bielik-11B-Q5_K_M.gguf --engine llama   # any local GGUF
```

- **ollama** is easiest for models in its registry; `ollama pull` is the fetch.
- **llama.cpp** (`llama-server`) serves any GGUF you supply — this is the path for community
  GGUFs like Bielik and EuroLLM. `--jinja` activates the model's own chat template (and is
  required for the reasoning suppression below to take effect); context size and GPU layers are
  the server's concern, not the translator's.

The design target is one model resident on a single ~16 GB GPU, every role differing only by
prompt. Do not try to serve several models at once.

**Slots must match `--concurrency`.** The translator sends up to `--concurrency` requests at
once (default 2). Give the server at least that many parallel slots or they serialise:
`llama-server --parallel 2`, or for ollama `OLLAMA_NUM_PARALLEL=2` on the daemon.
`tools/serve_model.sh` sets both to 2 by default.

**Reasoning ("thinking") models.** A model like Qwen3 emits a hidden chain-of-thought before its
answer. Those reasoning tokens are billed against the per-call token budget and routinely
truncate the JSON answer, which the harness then rejects as unusable — a flaky failure that
looks like a bad translation but is not. The translator therefore **suppresses reasoning by
default**, in-band, via the `enable_thinking=false` chat-template kwarg the reasoning families
honour (a template that does not read it ignores the kwarg, so non-reasoning models are
unaffected). Translation does not benefit from the reasoning, so this is pure upside here. To
opt back in, pass `--enable-reasoning` — but then raise the `translate`/`review` token budgets in
`config/agents.toml` to leave room for the trace, or the truncation returns.

## 5. When a request is rejected

If you point the default `generic` backend at a model whose build cannot compile the grammar,
the server returns a 4xx and the client **raises with the fix named**, rather than degrading
silently:

```
role=translate: request rejected: the server rejected schema-constrained decoding, which
some model builds cannot compile. Retry with a json_object backend (--backend bielik) (HTTP 400)
```

This is deliberate: a silent fallback to another strategy would hide a real capability
mismatch. You switch backends explicitly and the run is reproducible.

## 6. A genuinely new strategy

If a model needs request shaping that neither built-in strategy covers, subclass `Backend` and
implement `structured_request`, returning the (possibly rewritten) messages and the
`response_format` body fragment:

```python
from ..core import Backend, StructuredRequest, register_backend

class VendorBackend(Backend):
    def structured_request(self, messages, schema):
        # e.g. a server that wants the schema under a vendor-specific key, or the shape
        # described differently. Rewrite messages if the model must be told the shape in prose;
        # leave them untouched if the response_format constrains decoding.
        return StructuredRequest(
            messages=tuple(messages),
            response_format={"type": "vendor_json", "spec": schema})

register_backend(VendorBackend("vendor", model_hints=("vendor",)))
```

The client always validates the reply against the schema regardless of strategy, so you cannot
weaken the check by choosing a looser constraint.

## 7. Test it

Add a test under `tests/backend/`. `tests/backend/test_backend.py` shows the shape:

```python
def test_my_backend_shapes_the_request():
    req = get_backend("mistral").structured_request([Message("user", "u")], SCHEMA)
    assert req.response_format["type"] == "json_schema"        # or your strategy
    assert resolve_backend("auto", "mistral-7b-instruct").name == "mistral"
```

`tests/backend/test_client.py` shows how to exercise the whole client against an in-memory
`httpx.MockTransport` — no server needed — if you want to test how your strategy behaves over
the wire (retries, the rejection hint, structured routing).

## The full contract, in one place

- `Backend.structured_request(messages, schema) -> StructuredRequest` — the only method a
  backend must implement. `StructuredRequest(messages, response_format)`.
- `SchemaBackend` / `JsonObjectBackend` — the two ready strategies.
- `register_backend(backend)` — register under name + aliases; a name collision with a
  *different* backend is an error, re-registering the same instance is a no-op.
- `get_backend(name)` / `resolve_backend(name, model)` / `available_backends()` /
  `suggest_backend(model)` — selection and introspection.
- `LlmClient(config, backend=...)` — the transport; `ServerConfig` is connection settings only
  (no structured-output field — that is the backend's job).
