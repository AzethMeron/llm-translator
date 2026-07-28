"""Every fatal error the CLI recognises is reported the same way, exhaustively.

`main()` catches fourteen error types and promises three things for each: exit 1, one line on
stderr, and a "run aborted" record in the `--log-file`. Only two of the fourteen were ever
verified. That asymmetry matters because the log file is what a caller inspects after an
unattended run: an error type quietly missing from the except tuple exits with a traceback and
a non-1 status, and one missing from the log leaves an aborted run with no trace at all.

Parametrised over the whole tuple rather than a chosen sample, so ADDING an error type to
`main()` without adding it here fails the completeness test below -- the gap cannot reopen
silently.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from translator import cli
from translator.backend import BackendError, LlmError
from translator.cli import UsageError, main
from translator.retrieval import EmbeddingError, RerankError
from translator.roles import AgentConfigError
from translator.rules import RuleConfigError
from translator.runner import RunnerError
from transunit.adapter import AdapterError
from transunit.glossary import GlossaryError
from transunit.language import LanguageError
from transunit.reference import ReferenceError
from transunit.units import CatalogError

# One instance per type in main()'s except tuple, each carrying a distinctive message so the
# test can prove THAT error -- not merely some error -- reached stderr and the log.
FATAL_ERRORS = [
    UsageError("bad flag combination"),
    CatalogError("catalogue row 7 is torn"),
    GlossaryError("glossary term has no target"),
    RuleConfigError("rule file section is wrong"),
    LanguageError("no profile for 'xx'"),
    RunnerError("worker died mid-batch"),
    AdapterError("carrier adapter refused the payload"),
    AgentConfigError("agent file has no reviewers"),
    BackendError("no backend named 'nope'"),
    ReferenceError("reference corpus line 3 is malformed"),
    EmbeddingError("embedding request failed"),
    RerankError("rerank endpoint returned nonsense"),
    LlmError("server exploded", role="translate"),
    FileNotFoundError("config/missing.toml"),
]


def _ids(errors):
    return [type(error).__name__ for error in errors]


class TestEveryFatalErrorIsReportedAndLogged:
    @pytest.mark.parametrize("error", FATAL_ERRORS, ids=_ids(FATAL_ERRORS))
    def test_exits_one_prints_once_and_lands_in_the_log_file(
            self, error: Exception, tmp_path: Path, monkeypatch, capsys) -> None:
        log_file = tmp_path / "translator.log"
        monkeypatch.setattr(cli, "cmd_translate", lambda args: (_ for _ in ()).throw(error))

        try:
            code = main(["--log-file", str(log_file)])
        finally:
            # The file handler holds the log open; drop it so the file is readable and the next
            # parametrised case starts clean.
            for handler in list(logging.getLogger().handlers):
                handler.close()
                logging.getLogger().removeHandler(handler)

        assert code == 1, f"{type(error).__name__} must exit 1, not raise or exit 0"
        stderr = capsys.readouterr().err
        assert str(error) in stderr
        assert stderr.count("translator: ") == 1, "the operator gets exactly one line"

        assert log_file.is_file(), f"{type(error).__name__} left no log file at all"
        logged = log_file.read_text(encoding="utf-8")
        assert "run aborted" in logged
        assert str(error) in logged, "the log must carry the cause, not just that it aborted"

    def test_the_parametrisation_covers_every_type_main_catches(self) -> None:
        """Guards the list above against drift.

        Reads the except tuple off the live function rather than restating it, so adding an
        error type to `main()` without adding it here fails HERE, loudly, instead of leaving a
        fatal path silently unverified -- which is how thirteen of these went untested.
        """
        import dis

        caught = {
            instruction.argval
            for instruction in dis.get_instructions(main)
            if instruction.opname == "LOAD_GLOBAL" and isinstance(instruction.argval, str)
        }
        expected = {type(error).__name__ for error in FATAL_ERRORS}
        missing = {name for name in caught if name.endswith("Error")} - expected
        assert not missing, (
            f"main() references error type(s) {sorted(missing)} that FATAL_ERRORS does not "
            f"cover; add an instance so its reporting is verified")


class TestNoLogFileStillReportsToTheOperator:
    def test_a_fatal_error_with_no_log_file_still_exits_one_and_prints(
            self, monkeypatch, capsys) -> None:
        """`--no-log-file` suppresses the file, never the operator's copy: a run that dies must
        say so on stderr even when nothing is being recorded."""
        monkeypatch.setattr(
            cli, "cmd_translate",
            lambda args: (_ for _ in ()).throw(UsageError("no server configured")))
        try:
            code = main(["--no-log-file"])
        finally:
            for handler in list(logging.getLogger().handlers):
                handler.close()
                logging.getLogger().removeHandler(handler)
        assert code == 1
        assert "no server configured" in capsys.readouterr().err
