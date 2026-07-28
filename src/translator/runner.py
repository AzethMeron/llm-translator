"""Resumable batch execution over a catalogue of translation units.

A full run can be tens of thousands of units, each costing several sequential model calls,
so it will span many hours and will be interrupted. Durability is therefore a design
requirement rather than a nicety:

* results are appended to a JSONL journal immediately after each unit completes, so an
  interrupted run loses at most the unit in flight;
* restarting reads the journal, skips everything already recorded, and continues;
* the journal is separate from the catalogue, so a crashed run can never leave the
  catalogue itself truncated.

Work order is injectable. By default units are translated in catalogue order, but a caller
can supply any sort key -- to translate a game's interface text before its dialogue, say, so
a partial run still yields usable menus. Ordering is a caller concern precisely because what
is "most valuable first" depends on the carrier, and baking one carrier's answer into the
engine is what made this module carrier-specific before.
"""
from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transunit.units import Status, Unit, read_catalog, read_journal

from .agents import Outcome, TranslationAgents

# The default work order: catalogue order, which is stable and needs no carrier knowledge.
PriorityKey = Callable[[Unit], Any]


def catalog_order(unit: Unit) -> tuple[str, int]:
    """Stable default sort key: the unit's position in its source."""
    return (unit.rel_path, unit.line_no)


class RunnerError(Exception):
    """The run cannot start or continue."""


@dataclass
class Progress:
    """Live counters for a job that may span many runs.

    Two quantities are deliberately separate, because conflating them is what made a resumed
    run appear to start from zero:

    * ``done`` counts what *this* run translated, and is what the rate is measured from --
      earlier runs happened at a different time, possibly a different speed, and averaging
      them in would make the estimate lie.
    * ``already_done`` is what previous runs finished, so the reported position and the
      remaining count describe the whole job rather than this session.
    """

    total: int
    """Size of the whole job, including work finished by earlier runs."""
    already_done: int = 0
    done: int = 0
    verified: int = 0
    translated: int = 0
    rejected: int = 0
    skipped: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def record(self, outcome: Outcome) -> None:
        # Counted last so a rejected status leaves every counter untouched rather than a
        # ``done`` that outruns its categories.
        match outcome.status:
            case Status.VERIFIED:
                self.verified += 1
            case Status.TRANSLATED:
                self.translated += 1
            case Status.REJECTED:
                self.rejected += 1
            case Status.SKIPPED:
                self.skipped += 1
            case unhandled:
                # The category counters must always sum to ``done``; a status with no bucket
                # would break that silently and make every reported figure wrong for the rest
                # of a multi-hour run. Only terminal statuses may reach here.
                raise RunnerError(
                    f"Progress.record: outcome for unit {outcome.unit.unit_id!r} carries "
                    f"non-terminal status {unhandled!r}, which has no counter; expected one "
                    f"of {Status.VERIFIED!r}, {Status.TRANSLATED!r}, {Status.REJECTED!r}, "
                    f"{Status.SKIPPED!r}")
        self.done += 1

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def units_per_second(self) -> float:
        return self.done / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def completed(self) -> int:
        """Position in the whole job, not in this run."""
        return self.already_done + self.done

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed)

    @property
    def eta_seconds(self) -> float:
        rate = self.units_per_second
        return self.remaining / rate if rate > 0 else float("inf")

    @staticmethod
    def _duration(seconds: float) -> str:
        """Minutes for a short job, hours for a shift, days beyond that."""
        if seconds == float("inf"):
            return "unknown"
        if seconds < 90 * 60:
            return f"{seconds / 60:.0f}m"
        if seconds < 48 * 3600:
            return f"{seconds / 3600:.1f}h"
        return f"{seconds / 86400:.1f}d"

    def summary(self) -> str:
        return (
            f"{self.completed:,}/{self.total:,} done, {self.remaining:,} left "
            f"(verified {self.verified:,} | needs review {self.translated:,} | "
            f"rejected {self.rejected:,}) "
            f"{self.units_per_second:.2f} units/s  "
            f"elapsed {self._duration(self.elapsed)}  ETA {self._duration(self.eta_seconds)}")


def pending_units(catalog: Path, journal: Path, *,
                  key: PriorityKey = catalog_order) -> list[Unit]:
    """Units still needing work, in ``key`` order, excluding journalled results."""
    completed = completed_ids(journal)
    units = [unit for unit in read_catalog(catalog)
             if unit.status is Status.PENDING and unit.unit_id not in completed]
    units.sort(key=key)
    return units


def group_duplicates(units: Iterable[Unit]) -> list[tuple[Unit, tuple[Unit, ...]]]:
    """Group units that should receive one translation, keeping input order.

    A large fraction of a typical queue is text the catalogue already contains somewhere
    else. Translating each occurrence separately pays for the same work repeatedly and,
    worse, lets one string come back many different ways, which is precisely the
    inconsistency the glossary and style rules exist to prevent.

    Grouping is keyed on ``(source, speaker)``, not source alone: two speakers saying the
    same line may well want different renderings, and keeping them apart costs little.

    Members share the translation produced for the group's first unit, so it is chosen
    against that unit's surrounding lines. A line whose translation genuinely depends on its
    neighbours will get the first occurrence's reading.

    Placeholders need no special handling: members share masked *text*, and each unit keeps
    its own placeholder values, which injection substitutes back per unit.
    """
    groups: dict[tuple[str, str | None], list[Unit]] = {}
    for unit in units:
        groups.setdefault((unit.source, unit.speaker), []).append(unit)
    return [(members[0], tuple(members)) for members in groups.values()]


def completed_ids(journal: Path) -> set[str]:
    """Unit ids already recorded in the journal."""
    return {unit.unit_id for unit in read_journal(journal)}


class _Interruptible:
    """Turn SIGINT/SIGTERM into a cooperative stop flag.

    A run holds an open journal and a GPU-backed HTTP session; tearing those down mid-write
    on the first Ctrl-C risks a corrupt trailing record. Instead the first signal asks the
    loop to finish the current unit and exit cleanly.
    """

    def __init__(self) -> None:
        self.stop = False
        self._previous: dict[int, object] = {}

    def __enter__(self) -> _Interruptible:
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            self._previous[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, self._handle)
        return self

    def _handle(self, *_: object) -> None:
        self.stop = True

    def __exit__(self, *_: object) -> None:
        for signal_number, handler in self._previous.items():
            signal.signal(signal_number, handler)  # type: ignore[arg-type]


def run_batch(
    agents: TranslationAgents,
    units: Iterable[Unit],
    journal: Path,
    *,
    total: int | None = None,
    already_done: int = 0,
    on_progress: Callable[[Progress], None] | None = None,
    report_every: int = 25,
    concurrency: int = 1,
) -> Progress:
    """Translate ``units``, appending each result to ``journal`` as it completes.

    ``concurrency`` units are translated at once against the same server. The agents within a
    single unit still run strictly in sequence, and there is still exactly one set of weights
    resident, so this costs no additional VRAM: the server pre-allocates its slots regardless,
    and leaving them idle simply wastes them.

    Only this thread ever touches the journal or the progress counters -- workers just call
    the model -- so the append-per-result durability guarantee is unchanged and needs no
    locking. Journal order becomes completion order rather than input order, which resume does
    not depend on since it is keyed by unit id.

    Raises whatever a worker raised, once results already in flight have been recorded, so an
    infrastructure failure never discards completed work.
    """
    if concurrency < 1:
        raise RunnerError(f"concurrency must be >= 1, got {concurrency}")
    if report_every < 1:
        # Used as a modulus below; 0 would be a bare ZeroDivisionError at this boundary.
        raise RunnerError(f"report_every must be >= 1, got {report_every}")

    units = list(units)
    progress = Progress(total=total if total is not None else len(units) + already_done,
                        already_done=already_done)
    journal.parent.mkdir(parents=True, exist_ok=True)
    # One model pass per group of identical text; every member is still journalled
    # individually, so resume and injection stay per-unit.
    queued = iter(group_duplicates(units))
    failure: BaseException | None = None

    def translate_group(
        representative: Unit, members: tuple[Unit, ...]
    ) -> tuple[Outcome, tuple[Unit, ...]]:
        return agents.process(representative), members

    with _Interruptible() as interrupt, \
            journal.open("a", encoding="utf-8") as handle, \
            ThreadPoolExecutor(max_workers=concurrency) as pool:

        def submit_next() -> bool:
            group = next(queued, None)
            if group is None:
                return False
            in_flight.add(pool.submit(translate_group, *group))
            return True

        in_flight: set[Future[tuple[Outcome, tuple[Unit, ...]]]] = set()
        for _ in range(concurrency):
            if not submit_next():
                break

        while in_flight:
            finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                try:
                    outcome, members = future.result()
                except BaseException as exc:  # noqa: BLE001  -- re-raised below
                    failure = failure or exc
                    continue
                for member in members:
                    handle.write(outcome.applied_to(member).to_json())
                    handle.write("\n")
                    progress.record(outcome)
                # flush() reaches the OS, which survives this process dying; fsync reaches the
                # disk, which survives the machine dying. A run measured in days is worth a few
                # minutes of fsync in total, and the alternative is silently re-translating
                # whatever the page cache lost.
                handle.flush()
                os.fsync(handle.fileno())
                # Make the result available to later units in the same scene, recorded from this
                # thread only, after it is durable: as exact-match target context in memory, and
                # -- when self-population is enabled -- as a similarity-retrievable reference too.
                # learn_reference applies its own configured status filter, so it is handed every
                # result rather than only the injectable ones memory keeps.
                if agents.memory is not None and outcome.target and \
                        outcome.status in (Status.VERIFIED, Status.TRANSLATED):
                    agents.memory.record(outcome.unit.source, outcome.target)
                agents.learn_reference(outcome.unit.source, outcome.target, outcome.status)
                if on_progress and progress.done % report_every == 0:
                    on_progress(progress)

            # Stop feeding the pool once anything has failed or the operator interrupted, but
            # let work already in flight finish and be journalled.
            if failure is None and not interrupt.stop:
                for _ in range(len(finished)):
                    if not submit_next():
                        break

    if failure is not None:
        raise failure

    if on_progress:
        on_progress(progress)
    return progress
