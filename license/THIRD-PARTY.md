# Third-party dependencies

This project depends on the packages below. **None of them is bundled or vendored into
this repository** — they are installed from PyPI by `tools/setup_python_env.sh` — so their
licenses impose no obligations on this project's own source, and none constrains the
licensing in [`README.md`](./README.md). Every one is permissively licensed.

Verified against the installed package metadata in the pinned environment.

## Runtime (needed to talk to an inference server)

| Package | License | Notes |
|---|---|---|
| `httpx` | BSD-3-Clause | the one direct runtime dependency |
| `httpcore` | BSD-3-Clause | transitive |
| `h11` | MIT | transitive |
| `anyio` | MIT | transitive |
| `sniffio` | MIT / Apache-2.0 | transitive |
| `certifi` | MPL-2.0 | transitive; a CA-certificate bundle |
| `idna` | BSD-3-Clause | transitive |
| `typing_extensions` | PSF-2.0 | transitive |

## Development / tests only (not shipped, not imported by `src/`)

| Package | License |
|---|---|
| `pytest` | MIT |
| `coverage` | Apache-2.0 |
| `iniconfig` | MIT |
| `packaging` | Apache-2.0 / BSD-2-Clause |
| `pluggy` | MIT |

## On `certifi` (MPL-2.0)

MPL-2.0 is *file-level* weak copyleft: it obliges you to share the source of MPL-licensed
files **only if you modify and distribute those files**. `certifi` is an unmodified
third-party dependency installed from PyPI, not copied into this repository, so MPL-2.0
places no requirement on this project's code.

## On the language models

The models this translator talks to (Qwen, Bielik, EuroLLM, …) are **not** part of this
repository — no weights are bundled. Each model's own license governs its weights and, in
some cases, restricts commercial use or redistribution independently of this code. If you
serve or redistribute a model commercially, check that model's license separately; it is
orthogonal to the license on this software.

## On the source itself

The source in `src/` is original to this project (consolidated from the author's own
earlier translator projects). It embeds no third-party code, so it is the author's to
license as above.
