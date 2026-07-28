# CLAUDE.md

## Repository Structure

- **`src/`** — source code.
- **`docs/`** — documentation, references, design notes.
- **`.audit/`** — hidden but tracked directory holding results of past audits.
- **`tools/`** — scripts for setup, build, run, and other routines.
- **`tests/`** — everything related to testing.
- **`license/`** — project license and third-party licenses. Never add, remove, or modify a license here without explicit command and permission.

- **Script threshold.** If performing a routine (setup, starting the program, compilation, running tests, etc.) takes more than one command, or a single command exceeds 60 characters, it shall be implemented as a script in `tools/` instead of run ad hoc. Every such script must be hardened: it must validate its own preconditions, handle bad arguments, detect a misconfigured environment or missing packages, and fail with a clear, actionable error rather than crashing opaquely or proceeding on bad assumptions.

- **Project must be self-contained.** Any required configuration, data download, or installation step must be covered by a hardened script in `tools/` (same error-handling standard as above) — never a manual, undocumented step a person or agent has to remember to perform.

- **Pin and document the environment.** Keep presets for anything configurable, and record the exact versions of libraries, packages, and tools in use so the environment can be reproduced later. Where possible, provide a script in `tools/` that (re)creates the environment from these pinned versions.

## Workflow

- **Start clean.** Before starting any work, confirm you're on the correct branch and have pulled all changes from remote.

- **Audit, then plan, then implement.** Read the relevant code (or run it / test it) before making any claim about existing behavior — never assert or assume how something works. Only after understanding the current state, plan the change, then implement it.

- **Never assume a request is correct as stated.** Take the user's intent seriously and implement what they're asking for, but do not treat their framing, approach, or assumptions as automatically right. Analyze the consequences of the requested change against the actual codebase, take as long as needed to think it through, and proactively flag and explain any issue, risk, or better alternative before or while implementing — correct the user's understanding when it's wrong, rather than silently complying or silently working around it.

- **Use the designated branch (`claude-branch`).** Push to it any time a meaningful, logically-complete change has been implemented (e.g. after a step is done and its scoped tests pass) — don't leave completed work unpushed, to prevent data loss.

- **Expect the branch to be deleted after merge.** Once `claude-branch` is merged via pull request it will be removed. This is expected, not an error — recreate it freely when starting new work. Rebasing is usually unnecessary since `main` only changes via merged pull requests from this branch; rebase only if `main` has advanced with changes that conflict with the current work.

- **Never merge or push directly to `main`.** Only open a pull request when explicitly asked. Never force-push to `main`.

- **Test before pushing.** Before each push, run scoped tests covering only the changes being pushed. Before opening a pull request, run the full test suite as a final, wider check.

- **Keep documentation, references, and the test suite up to date** alongside any code change — not as a follow-up task.

## General Programming Rules

- **Single Source of Truth (SSOT).** Every behavior, business rule, or algorithm shall have exactly one authoritative implementation, testable and probable in isolation. Reuse through parameterization, composition, or generic abstractions rather than duplicating logic. If a variation is needed, extend the existing implementation (e.g. via parameters) instead of creating a parallel one.

- **Self-explaining code.** Prefer expressive, terse names over comments. Comments shall explain *why* something is implemented a certain way — assumptions, invariants, complexity guarantees, non-obvious tradeoffs — never *what* the code already expresses. If a comment is needed to explain what the code does, improve the code instead.

- **Never allow failures to be silent.** Functions that can fail shall expose failure explicitly using the language's idiomatic mechanism (exceptions, `Result`/`Expected` types, error codes, etc.) so that ignoring a failure is difficult or impossible — e.g. a bad value should be of a type that throws on any operation attempted on it, not silently propagate. Fail fast when invariants are violated.

- **Keep it simple (KISS).** Everything can be done many ways; prefer the simplest one, as long as it doesn't sacrifice the algorithm's or data structure's required complexity bounds. Avoid clever solutions unless they provide a measurable benefit.

- **Scope generality to the module, not the feature.** When implementing something, build the full-featured, complete version of *that thing* (that class, algorithm, or data structure) rather than a partial slice — this pays off in reusability and easier testing. But do not add speculative *features* nobody asked for (YAGNI). In short: don't half-implement what you build, but don't build things you weren't asked to build. Design each module so it's naturally extensible later, without pre-building extensions it doesn't yet need.

- **No silent fallbacks or workarounds.** Do not paper over a problem in our own codebase with a fallback path, retry-and-hope, or silent degradation — a single well-behaved algorithm that fails clearly is preferable to a "smarter" one that occasionally degrades unnoticed. Fix the actual cause instead. This does not apply to genuinely external failure modes outside our codebase (e.g. a flaky third-party service, transient network loss) where a documented, explicit fallback is a legitimate design choice — the distinction is whether the fallback exists to hide a problem in our own code.

- **Audit system-wide impact before and during implementation.** Before making a change, check against the actual sources (not memory or assumption) how it interacts with the rest of the codebase — callers, shared state, invariants relied upon elsewhere. Keep this in mind throughout planning and implementation, not just at the start; revisit it if the change's scope grows.

- **Deterministic resource ownership (RAII / CADRE).** Every resource has one clear owner responsible for acquiring and automatically releasing it — via destructor, `using`/`with`/`defer`, or the language's equivalent. Resource leaks are unacceptable.

- **Preserve invariants.** Every type shall maintain valid state throughout its lifetime. Invalid objects should be impossible or difficult to construct.

- **No implicit or unsafe conversions.** Prefer explicit constructors, casts, or conversion functions over implicit/automatic type conversion. Forbid silent narrowing, truncation, or coercion that can lose data or change meaning; if a conversion can fail or lose information, it must be explicit and checked.

- **No undefined behavior allowed.** Code shall never rely on or trigger undefined behavior (e.g. out-of-bounds access, use-after-free, signed overflow, data races, aliasing violations). Where the language permits UB, use tooling (sanitizers, static analysis, safe abstractions) to guarantee it cannot occur, rather than merely hoping it doesn't in practice.

- **No global mutable state.** Avoid global/static mutable variables and singletons that hold mutable state. State shall be owned explicitly (by an object, module, or passed as a dependency) and its mutation shall be traceable to a clear owner and scope.

- **Error notifications shall be structured and diagnostic.** Errors (exceptions, error codes, `Result`/`Expected` values) shall be well-structured and easy to catch, inspect, and handle programmatically — not bare strings. Each error shall carry enough information to debug it without reproducing it: **what** went wrong (a specific, unambiguous condition, not a generic failure), **where** it went wrong (the operation, module, or call site involved), and relevant diagnostic context such as expected vs. actual values, offending inputs, or relevant state at the time of failure.

- **Separate concerns.** Keep business logic independent from I/O, networking, databases, UI, OS APIs, devices, and frameworks. Put raw access to these behind well-designed interfaces; depend on interfaces, not concrete implementations, so components remain independently testable, swappable, and replaceable.

- **Prefer pure functions.** Functions should transform inputs into outputs without mutating them or external state. Returning values is preferred over mutable output parameters. Modifying state not passed in as an argument and not owned by the object is only acceptable when forced by a framework.

- **Single-purpose functions.** No hard limit on length — each function should do exactly one conceptual thing. If that thing is complex, extract smaller single-purpose helper functions rather than letting one function do several things.

- **Minimize coupling, maximize cohesion.** Each module should have a clearly defined responsibility and minimal knowledge of other modules.

- **Small, stable, hard-to-misuse public APIs.** Hide implementation details wherever possible. Every piece of code should make invalid usage difficult and correct usage natural.

- **Eliminate duplicated knowledge, not just duplicated code.** Similar behavior should share a common implementation whenever practical.

- **Prefer immutability and const-correctness** wherever practical. Mutable state should be minimized and clearly justified.

- **Validate inputs and enforce contracts at module boundaries.** Internal code may assume documented preconditions and invariants hold.

- **Prefer deterministic behavior.** Given identical inputs and state, code should produce identical outputs, unless randomness, time, or external systems are explicitly part of the contract.

- **Prefer standard library facilities** over third-party dependencies unless an external dependency provides substantial, justified value.

- **Consider performance during design**, but never sacrifice correctness or maintainability for premature optimization. Optimize only after identifying actual bottlenecks — and never at the cost of an algorithm's or data structure's big-O guarantees.

- **Document non-trivial complexity.** Algorithmic complexity, performance assumptions, synchronization requirements, or invariants shall be documented when important for correct usage or maintenance.

## Testing Rules

- Every non-trivial piece of code MUST be thoroughly tested.
- Write unit tests for individual components in isolation.
- Test edge cases, boundary conditions, invalid inputs, and empty inputs — including adversarial attempts to break the code.
- Test failure paths: verify exceptions, error codes, cleanup, rollback, and resource release all behave correctly under failure.
- Use property-based or fuzz testing whenever applicable.
- Use randomized testing where appropriate, with deterministic seeds when reproducibility matters; sanity-check outputs of random cases.
- For critical algorithms, verify results using independent implementations, mathematical properties, or sanity checks whenever practical.
- Every discovered bug shall be accompanied by a regression test, before or alongside the fix.
- Design code to be independently testable: databases, filesystems, clocks, RNGs, devices, and networks shall be abstracted behind interfaces or injected as dependencies.

## General Philosophy

- Correctness comes before performance.
- Simplicity comes before cleverness.
- Readability comes before brevity.
- Reusability comes through good abstractions, not speculative features.
- Every abstraction should reduce overall complexity. If it does not, remove it.
- Every module should be understandable, testable, and replaceable in isolation.
- Every piece of code should make invalid usage difficult and correct usage natural.
