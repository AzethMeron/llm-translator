"""Tests for the command-line entry point.

Focus on argument wiring and error handling, which are decidable without a server; a full
run is exercised by the harness and runner tests. The one live behaviour checked here is
that errors surface as a clean non-zero exit with a message, never a traceback.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from translator.cli import _language_wiring, _load_reference, build_parser, main
from translator.roles import Context
from transunit.reference import GrowableLexicalRetriever, LexicalRetriever, ReferenceError
from transunit.units import Status

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
RULES = REPO / "config" / "translation_rules.toml"
LANGUAGES = REPO / "config" / "languages.toml"


def args(**overrides):
    parser = build_parser()
    namespace = parser.parse_args([])
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


class TestParserDefaults:
    def test_backend_defaults_to_auto(self) -> None:
        assert build_parser().parse_args([]).backend == "auto"

    def test_adapter_defaults_to_empty_for_a_plain_corpus(self) -> None:
        assert build_parser().parse_args([]).adapter == ""

    def test_language_pair_defaults(self) -> None:
        parsed = build_parser().parse_args([])
        assert parsed.source_language == "en" and parsed.target_language == "pl"

    def test_context_window_defaults_to_zero_off(self) -> None:
        assert build_parser().parse_args([]).context_window == 0
        assert build_parser().parse_args(["--context-window", "8192"]).context_window == 8192


class TestLanguageWiring:
    def test_profiles_enable_the_lexical_detector(self) -> None:
        s_name, t_name, s_label, t_label, detector = _language_wiring(
            args(languages=LANGUAGES, source_language="en", target_language="pl"))
        assert (s_name, t_name, s_label, t_label) == ("English", "Polish", "EN", "PL")
        assert detector is not None
        assert detector("hello world friend", "hello world friend") is True  # echo caught

    def test_missing_profiles_disable_the_check(self, tmp_path: Path) -> None:
        # A check that cannot be evaluated must not be reported as passed -- so with no
        # profiles and no script, the detector is None rather than a permissive stub. The
        # language names still resolve, from the built-in table, so the prompt reads right.
        s_name, t_name, s_label, t_label, detector = _language_wiring(
            args(languages=tmp_path / "nope.toml", source_language="de", target_language="fr"))
        assert detector is None
        assert (s_name, t_name, s_label, t_label) == ("German", "French", "DE", "FR")

    def test_an_unknown_code_falls_back_to_itself_as_a_name(self, tmp_path: Path) -> None:
        s_name, *_ = _language_wiring(
            args(languages=tmp_path / "nope.toml", source_language="xx", target_language="en"))
        assert s_name == "xx"

    def test_source_script_enables_script_detection_without_profiles(self, tmp_path: Path) -> None:
        # A different-script pair needs no lexical profile: naming the source script is enough.
        s_name, t_name, s_label, t_label, detector = _language_wiring(
            args(languages=tmp_path / "nope.toml", source_language="ja", target_language="en",
                 source_script="japanese"))
        assert (s_name, t_name) == ("Japanese", "English")
        assert detector is not None
        assert detector("日本語", "日本語") is True          # echo caught
        assert detector("日本語", "Hello there") is False   # clean translation

    def test_source_script_overrides_lexical_profiles_for_detection(self) -> None:
        # --source-script wins over --languages for *detection*; names still come from either.
        _, _, _, _, detector = _language_wiring(
            args(languages=LANGUAGES, source_language="ja", target_language="en",
                 source_script="han"))
        assert detector("这是中文", "This is English") is False
        assert detector("这是中文", "还有中文") is True

    def test_an_unknown_script_is_reported(self) -> None:
        from transunit.language import LanguageError
        with pytest.raises(LanguageError, match="unknown script"):
            _language_wiring(args(source_language="ja", target_language="en",
                                  source_script="klingon"))

    def test_same_source_and_target_is_refused_with_profiles(self) -> None:
        from translator.cli import UsageError
        with pytest.raises(UsageError, match="nothing to translate"):
            _language_wiring(args(languages=LANGUAGES, source_language="en",
                                  target_language="en"))

    def test_same_source_and_target_is_refused_without_profiles(self, tmp_path: Path) -> None:
        from translator.cli import UsageError
        with pytest.raises(UsageError, match="nothing to translate"):
            _language_wiring(args(languages=tmp_path / "nope.toml", source_language="en",
                                  target_language="en"))


class TestErrorHandling:
    def test_a_missing_rule_file_exits_nonzero_without_a_traceback(
            self, tmp_path: Path, capsys) -> None:
        code = main(["--rules", str(tmp_path / "absent.toml")])
        assert code == 1
        assert "translator:" in capsys.readouterr().err

    def test_same_language_pair_exits_nonzero(self, capsys) -> None:
        code = main(["--rules", str(RULES), "--languages", str(LANGUAGES),
                     "--source-language", "en", "--target-language", "en"])
        assert code == 1
        assert "nothing to translate" in capsys.readouterr().err

    def test_negative_context_window_exits_nonzero_without_a_traceback(self, capsys) -> None:
        code = main(["--rules", str(RULES), "--context-window", "-1"])
        assert code == 1
        assert "--context-window must be >= 0" in capsys.readouterr().err

    def test_negative_revision_budget_exits_nonzero_without_a_traceback(self, capsys) -> None:
        # Regression: a negative --max-revisions raised a bare ValueError past main()'s except
        # tuple, printing a stacktrace instead of the clean exit-1 path.
        code = main(["--rules", str(RULES), "--max-revisions", "-1"])
        assert code == 1
        assert "translator: " in capsys.readouterr().err

    # The next three pin an ordering guarantee: local validation happens before the server is
    # contacted, so a bad argument / catalogue / config is reported for what it is rather than
    # masked by "no inference server" when none is running. No LlmClient is mocked -- the point
    # is precisely that none is reached.

    def test_limit_below_one_is_reported_before_contacting_the_server(self, capsys) -> None:
        code = main(["--rules", str(RULES), "--languages", str(LANGUAGES), "--limit", "0"])
        assert code == 1
        err = capsys.readouterr().err
        assert "--limit must be >= 1" in err
        assert "no inference server" not in err

    def test_a_malformed_catalogue_is_reported_before_contacting_the_server(
            self, tmp_path: Path, capsys) -> None:
        catalog = tmp_path / "units.jsonl"
        catalog.write_text('{"unit_id": "x"}\n', encoding="utf-8")  # missing required fields
        code = main(["--rules", str(RULES), "--languages", str(LANGUAGES),
                     "--catalog", str(catalog), "--journal", str(tmp_path / "j.jsonl")])
        assert code == 1
        err = capsys.readouterr().err
        assert "missing fields" in err
        assert "no inference server" not in err

    def test_from_rules_reviewer_without_advisory_is_reported_before_the_server(
            self, tmp_path: Path, capsys) -> None:
        # The default panel has a from_rules compliance reviewer; a rule set with no
        # [[advisory]] leaves it nothing to judge. That must surface as itself, not as a
        # server error.
        rules = tmp_path / "rules.toml"
        rules.write_text("[limits]\nmax_line_columns = 110\n", encoding="utf-8")
        code = main(["--rules", str(rules), "--languages", str(LANGUAGES)])
        assert code == 1
        err = capsys.readouterr().err
        assert "advisory" in err
        assert "no inference server" not in err


class TestFullRun:
    """cmd_translate wires backend + agents + runner together and journals results.

    The server is replaced wholesale so no network or GPU is touched; the point is the
    wiring, not the model.
    """

    def _fake_client_class(self):
        from translator.backend import UsageStats

        class FakeClient:
            def __init__(self, config=None, *, backend=None):
                self.config = config
                self.backend = backend
                self.stats = UsageStats()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def health(self):
                return True

            def complete_json(self, messages, schema, *, role="<none>", **_):
                if role == "translate":
                    return {"translation": "Dzień dobry."}
                return {"acceptable": True, "issues": []}

            def close(self):
                pass

        return FakeClient

    def test_translates_a_catalogue_and_journals_verified_results(
            self, tmp_path: Path, monkeypatch, capsys) -> None:
        from transunit.units import Status, Unit, read_journal, write_catalog

        catalog = tmp_path / "units.jsonl"
        journal = tmp_path / "journal.jsonl"
        write_catalog([
            Unit(unit_id="a", rel_path="f", line_no=1, span_start=0, span_end=1,
                 command="", kind="T", source="Good morning."),
            Unit(unit_id="b", rel_path="f", line_no=2, span_start=0, span_end=1,
                 command="", kind="T", source="Good morning."),  # duplicate, deduplicated
        ], catalog)

        monkeypatch.setattr("translator.cli.LlmClient", self._fake_client_class())
        code = main(["--rules", str(RULES), "--languages", str(LANGUAGES),
                     "--source-language", "en", "--target-language", "pl",
                     "--catalog", str(catalog), "--journal", str(journal),
                     "--concurrency", "1"])
        assert code == 0
        results = {u.unit_id: u for u in read_journal(journal)}
        assert set(results) == {"a", "b"}
        assert all(u.status is Status.VERIFIED for u in results.values())
        assert results["a"].target == "Dzień dobry."
        out = capsys.readouterr().out
        assert "via backend" in out
        assert "prompt" in out and "peak" in out  # the token summary reports prompt usage

    def test_limit_stops_after_n_units(self, tmp_path, monkeypatch) -> None:
        from transunit.units import Unit, read_journal, write_catalog
        catalog = tmp_path / "units.jsonl"
        journal = tmp_path / "journal.jsonl"
        write_catalog([
            Unit(unit_id="a", rel_path="f", line_no=1, span_start=0, span_end=1,
                 command="", kind="T", source="One."),
            Unit(unit_id="b", rel_path="f", line_no=2, span_start=0, span_end=1,
                 command="", kind="T", source="Two."),
        ], catalog)
        monkeypatch.setattr("translator.cli.LlmClient", self._fake_client_class())
        code = main(["--rules", str(RULES), "--languages", str(LANGUAGES),
                     "--catalog", str(catalog), "--journal", str(journal), "--limit", "1"])
        assert code == 0
        assert len(list(read_journal(journal))) == 1  # only the first of two units

    def test_no_server_responding_exits_with_a_clean_usage_error(
            self, tmp_path, monkeypatch, capsys) -> None:
        from transunit.units import Unit, write_catalog
        catalog = tmp_path / "units.jsonl"
        write_catalog([Unit(unit_id="a", rel_path="f", line_no=1, span_start=0, span_end=1,
                            command="", kind="T", source="One.")], catalog)

        class DeadServer:
            def __init__(self, config=None, *, backend=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def health(self):
                return False

        monkeypatch.setattr("translator.cli.LlmClient", DeadServer)
        code = main(["--rules", str(RULES), "--languages", str(LANGUAGES),
                     "--catalog", str(catalog), "--journal", str(tmp_path / "j.jsonl")])
        assert code == 1
        assert "no inference server responding" in capsys.readouterr().err

    def test_a_full_run_wires_a_reference_corpus_end_to_end(
            self, tmp_path: Path, monkeypatch, capsys) -> None:
        # Exercises the CLI glue specifically: --reference -> _load_reference -> the retriever
        # reaching TranslationAgents, with retrieval and self-population both enabled.
        from transunit.units import Unit, read_journal, write_catalog

        catalog = tmp_path / "units.jsonl"
        journal = tmp_path / "journal.jsonl"
        write_catalog([Unit(unit_id="a", rel_path="f", line_no=1, span_start=0, span_end=1,
                            command="", kind="T", source="Good morning.")], catalog)
        reference = tmp_path / "ref.jsonl"
        reference.write_text('{"source": "Good morning.", "target": "Dzień dobry."}\n',
                             encoding="utf-8")
        agents_file = tmp_path / "agents.toml"
        agents_file.write_text(
            '[translate]\ninstructions = "Translate."\n'
            '[[reviewer]]\nid = "accuracy"\ninstructions = "Check."\n'
            '[context]\nreference_examples = 2\nreference_min_score = 0.1\n'
            'reference_learn_statuses = ["verified"]\n', encoding="utf-8")

        monkeypatch.setattr("translator.cli.LlmClient", self._fake_client_class())
        code = main(["--rules", str(RULES), "--languages", str(LANGUAGES),
                     "--source-language", "en", "--target-language", "pl",
                     "--catalog", str(catalog), "--journal", str(journal),
                     "--agents", str(agents_file), "--reference", str(reference),
                     "--concurrency", "1"])
        assert code == 0
        out = capsys.readouterr().out
        assert "reference [lexical, self-populating]: 1 seed entries (1 with source" in out
        assert "learning from accepted output" in out
        assert {u.unit_id for u in read_journal(journal)} == {"a"}

    def test_the_reference_retriever_is_closed_after_a_run(
            self, tmp_path: Path, monkeypatch) -> None:
        # A retriever that owns a resource (the embedding retriever holds an HTTP client) must be
        # closed by the CLI that built it. A lexical retriever has no close() and is left alone.
        from transunit.units import Unit, write_catalog

        catalog = tmp_path / "units.jsonl"
        write_catalog([Unit(unit_id="a", rel_path="f", line_no=1, span_start=0, span_end=1,
                            command="", kind="T", source="Good morning.")], catalog)
        closed = {"value": False}

        class ClosableRetriever:
            def by_source(self, query, *, k, min_score=0.0):
                return ()

            def by_target(self, query, *, k, min_score=0.0):
                return ()

            def close(self):
                closed["value"] = True

        monkeypatch.setattr("translator.cli._load_reference", lambda *a, **k: ClosableRetriever())
        monkeypatch.setattr("translator.cli.LlmClient", self._fake_client_class())
        code = main(["--rules", str(RULES), "--languages", str(LANGUAGES),
                     "--source-language", "en", "--target-language", "pl",
                     "--catalog", str(catalog), "--journal", str(tmp_path / "j.jsonl"),
                     "--concurrency", "1"])
        assert code == 0
        assert closed["value"] is True

    def test_a_reference_failing_mid_run_stops_cleanly_and_closes(
            self, tmp_path: Path, monkeypatch, capsys) -> None:
        # The embedding endpoint going down at query time is a systemic failure -- every unit
        # would hit it -- so the run stops with a clean exit 1 and the cause reported, not a
        # traceback and not one unit silently rejected. The retriever is still closed on the way
        # out (the finally runs on the exception path too).
        from transunit.units import Unit, write_catalog
        from translator.embedding import EmbeddingError

        catalog = tmp_path / "units.jsonl"
        write_catalog([Unit(unit_id="a", rel_path="f", line_no=1, span_start=0, span_end=1,
                            command="", kind="T", source="Good morning.")], catalog)
        agents_file = tmp_path / "agents.toml"
        agents_file.write_text(
            '[translate]\ninstructions = "Translate."\n'
            '[[reviewer]]\nid = "accuracy"\ninstructions = "Check."\n'
            '[context]\nreference_examples = 1\nreference_min_score = 0.1\n', encoding="utf-8")
        closed = {"value": False}

        class FailingRetriever:
            def by_source(self, query, *, k, min_score=0.0):
                raise EmbeddingError("endpoint down", url="http://x/v1/embeddings")

            def by_target(self, query, *, k, min_score=0.0):
                return ()

            def close(self):
                closed["value"] = True

        monkeypatch.setattr("translator.cli._load_reference", lambda *a, **k: FailingRetriever())
        monkeypatch.setattr("translator.cli.LlmClient", self._fake_client_class())
        code = main(["--rules", str(RULES), "--languages", str(LANGUAGES),
                     "--source-language", "en", "--target-language", "pl",
                     "--catalog", str(catalog), "--journal", str(tmp_path / "j.jsonl"),
                     "--agents", str(agents_file), "--concurrency", "1"])
        assert code == 1
        assert "endpoint down" in capsys.readouterr().err
        assert closed["value"] is True  # closed even though a worker raised

    def test_a_rerank_failure_mid_run_stops_cleanly_closes_journals_and_logs(
            self, tmp_path: Path, monkeypatch, capsys) -> None:
        # The mirror of the embedding mid-run test, for the reranker: a systemic query-time
        # failure stops the run (exit 1), closes the retriever, and -- the part this test adds --
        # is captured in the --log-file, and the unit already journalled before the failure
        # survives it (durability: a run that dies partway does not lose completed work).
        from transunit.units import Unit, read_journal, write_catalog
        from translator.cli import _configure_logging
        from translator.retrieval.rerank import RerankError

        catalog = tmp_path / "units.jsonl"
        journal = tmp_path / "j.jsonl"
        log_file = tmp_path / "translator.log"
        write_catalog([
            Unit(unit_id="a", rel_path="f", line_no=1, span_start=0, span_end=1,
                command="", kind="T", source="Good morning."),
            Unit(unit_id="b", rel_path="f", line_no=2, span_start=0, span_end=1,
                command="", kind="T", source="Good evening."),
        ], catalog)
        agents_file = tmp_path / "agents.toml"
        agents_file.write_text(
            '[translate]\ninstructions = "Translate."\n'
            '[[reviewer]]\nid = "accuracy"\ninstructions = "Check."\n'
            '[context]\nreference_examples = 1\nreference_min_score = 0.1\n', encoding="utf-8")
        closed = {"value": False}

        class FailingRerankRetriever:
            # _unit_block queries by_source with unit.source for BOTH the translate step and
            # every review step, so a unit that succeeds is queried more than once; keying the
            # failure on unit "b"'s own source text (rather than a raw call count) is what makes
            # this robust to exactly how many times a successful unit is queried.
            def by_source(self, query, *, k, min_score=0.0):
                if "morning" not in query:  # unit "a" is "Good morning."; unit "b" is not
                    raise RerankError("rerank endpoint down", url="http://y/v1/rerank")
                return ()

            def by_target(self, query, *, k, min_score=0.0):
                return ()

            def close(self):
                closed["value"] = True

        monkeypatch.setattr("translator.cli._load_reference",
                            lambda *a, **k: FailingRerankRetriever())
        monkeypatch.setattr("translator.cli.LlmClient", self._fake_client_class())
        try:
            code = main(["--rules", str(RULES), "--languages", str(LANGUAGES),
                        "--source-language", "en", "--target-language", "pl",
                        "--catalog", str(catalog), "--journal", str(journal),
                        "--agents", str(agents_file), "--concurrency", "1",
                        "--log-file", str(log_file)])
            assert code == 1
            assert "rerank endpoint down" in capsys.readouterr().err
            assert closed["value"] is True
            logged = log_file.read_text(encoding="utf-8")
            assert "run aborted" in logged and "rerank endpoint down" in logged
            # Unit "a" finished and was journalled before unit "b" hit the failure -- durable.
            assert {u.unit_id for u in read_journal(journal)} == {"a"}
        finally:
            _configure_logging(None)

    def test_a_dead_embedding_endpoint_at_hybrid_construction_stops_cleanly_and_logs(
            self, tmp_path: Path, monkeypatch, capsys) -> None:
        # The other half of the plan's diagnostics requirement: a REAL HybridRetriever (not a
        # stand-in) built against a genuinely unreachable endpoint. Its construction-time
        # EmbeddingError is raised by _load_reference BEFORE cmd_translate's try/finally even
        # starts (see cli.py), so this specifically proves that failure still reaches main()'s
        # except tuple, is printed once, and lands in the --log-file -- not just the mid-run
        # rerank case, which is a different code path (inside the try/finally, after a worker
        # already started).
        from transunit.units import Unit, write_catalog
        from translator.cli import _configure_logging

        catalog = tmp_path / "units.jsonl"
        write_catalog([Unit(unit_id="a", rel_path="f", line_no=1, span_start=0, span_end=1,
                            command="", kind="T", source="Good morning.")], catalog)
        reference = tmp_path / "ref.jsonl"
        reference.write_text('{"source": "Good morning.", "target": "Dzien dobry."}\n',
                             encoding="utf-8")
        agents_file = tmp_path / "agents.toml"
        agents_file.write_text(
            '[translate]\ninstructions = "Translate."\n'
            '[[reviewer]]\nid = "accuracy"\ninstructions = "Check."\n'
            '[context]\nreference_examples = 1\nreference_min_score = 0.1\n', encoding="utf-8")
        log_file = tmp_path / "translator.log"
        monkeypatch.setattr("translator.cli.LlmClient", self._fake_client_class())
        try:
            code = main(["--rules", str(RULES), "--languages", str(LANGUAGES),
                        "--source-language", "en", "--target-language", "pl",
                        "--catalog", str(catalog), "--journal", str(tmp_path / "j.jsonl"),
                        "--agents", str(agents_file), "--reference", str(reference),
                        "--embedding-url", "http://127.0.0.1:1/v1", "--hybrid",
                        "--concurrency", "1", "--log-file", str(log_file)])
            assert code == 1
            err = capsys.readouterr().err
            assert "translator:" in err and "embedding request failed" in err
            logged = log_file.read_text(encoding="utf-8")
            assert "run aborted" in logged and "embedding request failed" in logged
        finally:
            _configure_logging(None)

    def test_a_hybrid_warning_reaches_the_real_log_file_end_to_end(
            self, tmp_path: Path, monkeypatch) -> None:
        # Not caplog: a genuine --log-file, through a full main() run, for one of the new
        # warnings specifically -- pinning that these reach the unfiltered log file exactly like
        # the pre-existing reference warnings do, not just that caplog can observe them.
        from transunit.units import Unit, write_catalog
        from translator.cli import _configure_logging

        class FakeHybrid:
            def __init__(self, entries, **_kwargs):
                self._n = len(list(entries))
            source_entries = 1
            def close(self):
                pass
            def by_source(self, query, *, k, min_score=0.0):
                return ()
            def by_target(self, query, *, k, min_score=0.0):
                return ()
            def __len__(self):
                return self._n

        catalog = tmp_path / "units.jsonl"
        write_catalog([Unit(unit_id="a", rel_path="f", line_no=1, span_start=0, span_end=1,
                            command="", kind="T", source="Good morning.")], catalog)
        reference = tmp_path / "ref.jsonl"
        reference.write_text('{"source": "Good morning.", "target": "Dzien dobry."}\n',
                             encoding="utf-8")
        agents_file = tmp_path / "agents.toml"
        agents_file.write_text(
            '[translate]\ninstructions = "Translate."\n'
            '[[reviewer]]\nid = "accuracy"\ninstructions = "Check."\n'
            '[context]\nreference_examples = 1\nreference_min_score = 0.15\n', encoding="utf-8")
        log_file = tmp_path / "translator.log"
        monkeypatch.setattr("translator.cli.HybridRetriever", FakeHybrid)
        monkeypatch.setattr("translator.cli.LlmClient", self._fake_client_class())
        try:
            code = main(["--rules", str(RULES), "--languages", str(LANGUAGES),
                        "--source-language", "en", "--target-language", "pl",
                        "--catalog", str(catalog), "--journal", str(tmp_path / "j.jsonl"),
                        "--agents", str(agents_file), "--reference", str(reference),
                        "--embedding-url", "http://x/v1", "--hybrid",
                        "--rerank-url", "http://y/v1", "--concurrency", "1",
                        "--log-file", str(log_file)])
            assert code == 0
            logged = log_file.read_text(encoding="utf-8")
            assert "below the reranker-calibrated floor" in logged
        finally:
            _configure_logging(None)

    def test_nothing_to_do_when_the_catalogue_is_all_journalled(
            self, tmp_path: Path, monkeypatch, capsys) -> None:
        from transunit.units import Status, Unit, write_catalog

        catalog = tmp_path / "units.jsonl"
        journal = tmp_path / "journal.jsonl"
        write_catalog([Unit(unit_id="a", rel_path="f", line_no=1, span_start=0, span_end=1,
                            command="", kind="T", source="Hi", status=Status.VERIFIED,
                            target="Cześć")], catalog)
        monkeypatch.setattr("translator.cli.LlmClient", self._fake_client_class())
        code = main(["--rules", str(RULES), "--languages", str(LANGUAGES),
                     "--catalog", str(catalog), "--journal", str(journal)])
        assert code == 0
        assert "nothing to do" in capsys.readouterr().out


class TestLoggingSinks:
    """The full log keeps every warning; the console honours leniency."""

    def test_console_filter_drops_only_leniency_suppressed_records(self) -> None:
        import logging

        from translator.cli import _LeniencyConsoleFilter

        f = _LeniencyConsoleFilter()

        def record():
            return logging.LogRecord("translator.agents", logging.WARNING, "f", 1, "m",
                                     None, None)

        plain = record()
        assert f.filter(plain) is True  # no flag -> always shown
        suppressed = record()
        suppressed.leniency_suppress_console = True
        assert f.filter(suppressed) is False  # suppressed -> hidden from console
        surfaced = record()
        surfaced.leniency_suppress_console = False
        assert f.filter(surfaced) is True

    def test_the_log_file_keeps_a_warning_the_console_would_suppress(self, tmp_path) -> None:
        import logging

        from translator.cli import _configure_logging

        log_file = tmp_path / "translator.log"
        _configure_logging(log_file)
        try:
            logging.getLogger("translator.agents").warning(
                "reviewer 'accuracy' could not evaluate unit u1; model reply: {...}",
                extra={"leniency_suppress_console": True})
            for handler in logging.getLogger("translator").handlers:
                handler.flush()
            assert "could not evaluate unit u1" in log_file.read_text(encoding="utf-8")
        finally:
            _configure_logging(None)  # detach the file handler so other tests are unaffected

    def test_a_fatal_error_lands_in_the_log_file_and_prints_once(self, tmp_path, capsys) -> None:
        # A run that dies (here: source == target) must leave the failure in the --log-file, not
        # only on stderr -- and the console copy must not be doubled by the added log record.
        from translator.cli import _configure_logging, main

        log_file = tmp_path / "translator.log"
        try:
            code = main(["--rules", str(RULES), "--languages", str(LANGUAGES),
                         "--source-language", "en", "--target-language", "en",
                         "--log-file", str(log_file)])
            assert code == 1
            err = capsys.readouterr().err
            assert err.count("nothing to translate") == 1        # printed once, not doubled
            assert log_file.is_file()
            logged = log_file.read_text(encoding="utf-8")
            assert "run aborted" in logged and "nothing to translate" in logged
        finally:
            _configure_logging(None)


class TestEntryPoint:
    def _run(self, *cli_args):
        return subprocess.run([sys.executable, "-m", "translator.cli", *cli_args],
                              capture_output=True, text=True,
                              env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"})

    def test_help_lists_the_key_flags(self) -> None:
        result = self._run("--help")
        assert result.returncode == 0
        for flag in ("--catalog", "--journal", "--backend", "--adapter"):
            assert flag in result.stdout, flag

    def test_help_names_the_available_backends(self) -> None:
        out = self._run("--help").stdout
        for backend in ("generic", "qwen", "bielik", "eurollm"):
            assert backend in out, backend


class TestReferenceLoading:
    """--reference wiring: what gets built, and the warnings that guard misconfiguration."""

    def _corpus(self, tmp_path: Path, *lines: str) -> Path:
        path = tmp_path / "ref.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_reference_flag_is_repeatable(self) -> None:
        parsed = build_parser().parse_args(["--reference", "a.jsonl", "--reference", "b.jsonl"])
        assert parsed.reference == [Path("a.jsonl"), Path("b.jsonl")]

    def test_reference_defaults_to_none(self) -> None:
        assert build_parser().parse_args([]).reference is None

    def test_no_corpus_and_no_learning_builds_nothing(self, tmp_path: Path) -> None:
        result = _load_reference(None, tmp_path / "j.jsonl", Context())
        assert result is None

    def test_a_corpus_with_retrieval_on_builds_a_read_only_retriever(self, tmp_path: Path) -> None:
        corpus = self._corpus(tmp_path, '{"source": "the door", "target": "drzwi"}')
        result = _load_reference([corpus], tmp_path / "j.jsonl",
                                 Context(reference_examples=2))
        assert isinstance(result, LexicalRetriever) and len(result) == 1

    def test_learning_builds_a_growable_retriever(self, tmp_path: Path) -> None:
        result = _load_reference(None, tmp_path / "j.jsonl",
                                 Context(reference_examples=2,
                                         reference_learn_statuses=(Status.VERIFIED,)))
        assert isinstance(result, GrowableLexicalRetriever)

    def test_a_missing_corpus_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ReferenceError):
            _load_reference([tmp_path / "nope.jsonl"], tmp_path / "j.jsonl",
                            Context(reference_examples=2))

    def test_a_corpus_is_not_even_read_when_retrieval_is_off(self, tmp_path: Path) -> None:
        # Retrieval off -> the corpus is unused, so it is not read: a missing (or huge) file is
        # never touched, only the retrieval-off misconfiguration is reported.
        result = _load_reference([tmp_path / "missing.jsonl"], tmp_path / "j.jsonl", Context())
        assert result is None  # no ReferenceError despite the missing path

    def test_retrieval_on_without_material_warns_and_builds_nothing(
            self, tmp_path: Path, caplog) -> None:
        logging.getLogger("translator").propagate = True
        with caplog.at_level(logging.WARNING, logger="translator.cli"):
            result = _load_reference(None, tmp_path / "j.jsonl", Context(reference_examples=2))
        assert result is None
        assert any("no --reference corpus" in record.message for record in caplog.records)

    def test_a_corpus_with_retrieval_off_warns(self, tmp_path: Path, caplog) -> None:
        corpus = self._corpus(tmp_path, '{"source": "a", "target": "b"}')
        logging.getLogger("translator").propagate = True
        with caplog.at_level(logging.WARNING, logger="translator.cli"):
            result = _load_reference([corpus], tmp_path / "j.jsonl", Context())  # examples all 0
        assert result is None
        assert any("retrieval is off" in record.message for record in caplog.records)

    # --- the optional embedding retriever (selected by --embedding-url) --------

    def _fake_embedding(self, monkeypatch):
        """Swap EmbeddingRetriever for a stub so the endpoint (and numpy) are not needed."""
        seen = {}

        class Fake:
            def __init__(self, entries, *, base_url, model="local", **_):
                seen["base_url"] = base_url
                seen["model"] = model
                self._n = len(list(entries))
            source_entries = 0
            def __len__(self):
                return self._n

        monkeypatch.setattr("translator.cli.EmbeddingRetriever", Fake)
        return seen

    def test_embedding_url_flags_parse(self) -> None:
        parsed = build_parser().parse_args(["--embedding-url", "http://x/v1",
                                            "--embedding-model", "emb"])
        assert parsed.embedding_url == "http://x/v1" and parsed.embedding_model == "emb"

    def test_embedding_url_selects_the_embedding_retriever(self, tmp_path, monkeypatch) -> None:
        seen = self._fake_embedding(monkeypatch)
        corpus = self._corpus(tmp_path, '{"source": "the door", "target": "drzwi"}')
        result = _load_reference([corpus], tmp_path / "j.jsonl",
                                 Context(reference_examples=2, reference_min_score=0.55),
                                 embedding_url="http://x/v1", embedding_model="emb")
        assert result is not None and seen["base_url"] == "http://x/v1" and seen["model"] == "emb"

    def test_embedding_with_learning_is_refused(self, tmp_path, monkeypatch) -> None:
        self._fake_embedding(monkeypatch)
        from translator.cli import UsageError
        with pytest.raises(UsageError, match="read-only"):
            _load_reference(None, tmp_path / "j.jsonl",
                            Context(reference_examples=2,
                                    reference_learn_statuses=(Status.VERIFIED,)),
                            embedding_url="http://x/v1")

    def test_a_low_floor_warns_it_is_lexical_calibrated_for_embeddings(
            self, tmp_path, monkeypatch, caplog) -> None:
        self._fake_embedding(monkeypatch)
        corpus = self._corpus(tmp_path, '{"source": "s", "target": "t"}')
        logging.getLogger("translator").propagate = True
        with caplog.at_level(logging.WARNING, logger="translator.cli"):
            _load_reference([corpus], tmp_path / "j.jsonl",
                            Context(reference_examples=2, reference_min_score=0.15),
                            embedding_url="http://x/v1")
        assert any("calibrated for the lexical" in r.message for r in caplog.records)

    def test_source_retrieval_with_only_target_only_entries_warns(
            self, tmp_path: Path, caplog) -> None:
        corpus = self._corpus(tmp_path, '{"target": "only a target"}')
        logging.getLogger("translator").propagate = True
        with caplog.at_level(logging.WARNING, logger="translator.cli"):
            _load_reference([corpus], tmp_path / "j.jsonl", Context(reference_examples=2))
        assert any("no reference entry has a source" in record.message
                   for record in caplog.records)

    # --- the optional hybrid retriever (selected by --embedding-url --hybrid) --

    def _fake_hybrid(self, monkeypatch):
        """Swap HybridRetriever for a stub so no endpoint (embedding or rerank) is needed."""
        seen = {}

        class Fake:
            def __init__(self, entries, *, embedding_base_url, rerank_base_url="", **kwargs):
                seen["embedding_base_url"] = embedding_base_url
                seen["rerank_base_url"] = rerank_base_url
                seen.update(kwargs)
                self._n = len(list(entries))
            source_entries = 0
            def close(self):
                seen["closed"] = True
            def __len__(self):
                return self._n

        monkeypatch.setattr("translator.cli.HybridRetriever", Fake)
        return seen

    def test_hybrid_and_rerank_flags_parse(self) -> None:
        parsed = build_parser().parse_args(["--hybrid", "--rerank-url", "http://y/v1",
                                            "--rerank-model", "rr"])
        assert parsed.hybrid is True
        assert parsed.rerank_url == "http://y/v1" and parsed.rerank_model == "rr"

    def test_hybrid_defaults_to_off(self) -> None:
        assert build_parser().parse_args([]).hybrid is False

    def test_embedding_url_and_hybrid_selects_the_hybrid_retriever(
            self, tmp_path, monkeypatch) -> None:
        seen = self._fake_hybrid(monkeypatch)
        corpus = self._corpus(tmp_path, '{"source": "the door", "target": "drzwi"}')
        result = _load_reference([corpus], tmp_path / "j.jsonl",
                                 Context(reference_examples=2, reference_min_score=0.55),
                                 embedding_url="http://x/v1", hybrid=True)
        assert result is not None and seen["embedding_base_url"] == "http://x/v1"
        assert seen["rerank_base_url"] == ""

    def test_hybrid_passes_the_candidate_pool_and_mmr_knobs_through(
            self, tmp_path, monkeypatch) -> None:
        seen = self._fake_hybrid(monkeypatch)
        corpus = self._corpus(tmp_path, '{"source": "the door", "target": "drzwi"}')
        _load_reference([corpus], tmp_path / "j.jsonl",
                        Context(reference_examples=2, reference_candidate_pool=17,
                                reference_mmr_lambda=0.4, reference_lexical_min_score=0.2,
                                reference_embedding_min_score=0.6),
                        embedding_url="http://x/v1", hybrid=True,
                        rerank_url="http://y/v1", rerank_model="rr")
        assert seen["rerank_base_url"] == "http://y/v1" and seen["rerank_model"] == "rr"
        assert seen["candidate_pool"] == 17 and seen["mmr_lambda"] == pytest.approx(0.4)
        assert seen["lexical_min_score"] == pytest.approx(0.2)
        assert seen["embedding_min_score"] == pytest.approx(0.6)

    def test_hybrid_without_embedding_url_is_refused(self, tmp_path) -> None:
        from translator.cli import UsageError
        with pytest.raises(UsageError, match="--hybrid needs --embedding-url"):
            _load_reference(None, tmp_path / "j.jsonl", Context(reference_examples=2),
                            hybrid=True)

    def test_rerank_url_without_hybrid_is_refused(self, tmp_path) -> None:
        from translator.cli import UsageError
        with pytest.raises(UsageError, match="--rerank-url needs --hybrid"):
            _load_reference(None, tmp_path / "j.jsonl", Context(reference_examples=2),
                            embedding_url="http://x/v1", rerank_url="http://y/v1")

    def test_hybrid_with_learning_is_refused(self, tmp_path, monkeypatch) -> None:
        self._fake_hybrid(monkeypatch)
        from translator.cli import UsageError
        with pytest.raises(UsageError, match="read-only"):
            _load_reference(None, tmp_path / "j.jsonl",
                            Context(reference_examples=2,
                                    reference_learn_statuses=(Status.VERIFIED,)),
                            embedding_url="http://x/v1", hybrid=True)

    def test_hybrid_without_a_reranker_does_not_get_embedding_scale_advice(
            self, tmp_path, monkeypatch, caplog) -> None:
        # reference_min_score is applied to a DIFFERENT scale per mode, and the advice must name
        # the right one. On the hybrid-without-rerank path the final score is fused rank
        # agreement, not an embedding cosine, and the per-arm floors have already done the
        # absolute gating -- so the "embedding cosines cluster higher, raise to 0.55" advice
        # would be factually wrong here, and is not given.
        self._fake_hybrid(monkeypatch)
        corpus = self._corpus(tmp_path, '{"source": "s", "target": "t"}')
        logging.getLogger("translator").propagate = True
        with caplog.at_level(logging.WARNING, logger="translator.cli"):
            _load_reference([corpus], tmp_path / "j.jsonl",
                            Context(reference_examples=2, reference_min_score=0.15),
                            embedding_url="http://x/v1", hybrid=True)
        assert not any("calibrated for the lexical" in r.message for r in caplog.records)

    def test_a_reranked_run_is_not_told_to_raise_its_floor_to_the_embedding_scale(
            self, tmp_path, monkeypatch, caplog) -> None:
        # The sharp edge of the same defect: 0.30 is the CORRECT reranker floor (a correct
        # cross-script match reranks to a sigmoid of ~0.44, measured live), so advising a raise
        # to the embedding scale's 0.55 would make the operator discard good matches. Neither
        # warning may fire at the correctly-calibrated value.
        self._fake_hybrid(monkeypatch)
        corpus = self._corpus(tmp_path, '{"source": "s", "target": "t"}')
        logging.getLogger("translator").propagate = True
        with caplog.at_level(logging.WARNING, logger="translator.cli"):
            _load_reference([corpus], tmp_path / "j.jsonl",
                            Context(reference_examples=2, reference_min_score=0.30),
                            embedding_url="http://x/v1", hybrid=True,
                            rerank_url="http://y/v1")
        assert not any("calibrated for the lexical" in r.message for r in caplog.records)
        assert not any("reranker-calibrated floor" in r.message for r in caplog.records)

    def test_a_floor_below_the_rerank_default_warns_when_reranking(
            self, tmp_path, monkeypatch, caplog) -> None:
        self._fake_hybrid(monkeypatch)
        corpus = self._corpus(tmp_path, '{"source": "s", "target": "t"}')
        logging.getLogger("translator").propagate = True
        with caplog.at_level(logging.WARNING, logger="translator.cli"):
            _load_reference([corpus], tmp_path / "j.jsonl",
                            Context(reference_examples=2, reference_min_score=0.15),
                            embedding_url="http://x/v1", hybrid=True,
                            rerank_url="http://y/v1")
        assert any("below the reranker-calibrated floor" in r.message for r in caplog.records)

    def test_a_floor_at_or_above_the_rerank_default_does_not_warn(
            self, tmp_path, monkeypatch, caplog) -> None:
        self._fake_hybrid(monkeypatch)
        corpus = self._corpus(tmp_path, '{"source": "s", "target": "t"}')
        logging.getLogger("translator").propagate = True
        with caplog.at_level(logging.WARNING, logger="translator.cli"):
            _load_reference([corpus], tmp_path / "j.jsonl",
                            Context(reference_examples=2, reference_min_score=0.30),
                            embedding_url="http://x/v1", hybrid=True,
                            rerank_url="http://y/v1")
        assert not any("below the reranker-calibrated floor" in r.message
                       for r in caplog.records)

    def test_no_rerank_warning_without_a_reranker(self, tmp_path, monkeypatch, caplog) -> None:
        # A low floor with hybrid but NO reranker must trigger only the existing embedding-scale
        # warning, not the reranker-specific one (there is no reranker to be miscalibrated for).
        self._fake_hybrid(monkeypatch)
        corpus = self._corpus(tmp_path, '{"source": "s", "target": "t"}')
        logging.getLogger("translator").propagate = True
        with caplog.at_level(logging.WARNING, logger="translator.cli"):
            _load_reference([corpus], tmp_path / "j.jsonl",
                            Context(reference_examples=2, reference_min_score=0.15),
                            embedding_url="http://x/v1", hybrid=True)
        assert not any("below the reranker-calibrated floor" in r.message
                       for r in caplog.records)

    def test_a_small_candidate_pool_warns(self, tmp_path, monkeypatch, caplog) -> None:
        self._fake_hybrid(monkeypatch)
        corpus = self._corpus(tmp_path, '{"source": "s", "target": "t"}')
        logging.getLogger("translator").propagate = True
        with caplog.at_level(logging.WARNING, logger="translator.cli"):
            _load_reference([corpus], tmp_path / "j.jsonl",
                            Context(reference_examples=5, reference_candidate_pool=2),
                            embedding_url="http://x/v1", hybrid=True)
        assert any("smaller than the largest requested example count" in r.message
                   for r in caplog.records)

    def test_an_ample_candidate_pool_does_not_warn(self, tmp_path, monkeypatch, caplog) -> None:
        self._fake_hybrid(monkeypatch)
        corpus = self._corpus(tmp_path, '{"source": "s", "target": "t"}')
        logging.getLogger("translator").propagate = True
        with caplog.at_level(logging.WARNING, logger="translator.cli"):
            _load_reference([corpus], tmp_path / "j.jsonl",
                            Context(reference_examples=2, reference_candidate_pool=40),
                            embedding_url="http://x/v1", hybrid=True)
        assert not any("smaller than the largest requested example count" in r.message
                       for r in caplog.records)

    def test_learning_seeds_from_the_journal(self, tmp_path: Path) -> None:
        # A resumed run must keep what earlier runs learned: journalled injectable results seed
        # the growable store.
        from transunit.units import Unit
        journal = tmp_path / "j.jsonl"
        done = Unit(unit_id="u", rel_path="f", line_no=1, span_start=0, span_end=1, command="",
                    kind="T", source="the dragon breathes fire", target="smok zieje ogniem",
                    status=Status.VERIFIED)
        journal.write_text(done.to_json() + "\n", encoding="utf-8")
        result = _load_reference(None, journal,
                                 Context(reference_examples=2,
                                         reference_learn_statuses=(Status.VERIFIED,)))
        assert isinstance(result, GrowableLexicalRetriever)
        hit = result.by_source("the dragon breathes fire", k=1, min_score=0.1)[0]
        assert hit.entry.target == "smok zieje ogniem"
