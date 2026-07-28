"""The boundaries between the two packages.

    transunit    the contract: the translation unit, placeholders, terminology, detection
    translator   one translator: a local language model behind a backend

The contract depends on nobody. The translator depends only on the contract and an HTTP
client. Crucially, the translator never imports a *carrier adapter* -- the concrete adapters
(subtitles, game engines, documents) are not even in this repository, and the translator
must run without any of them, learning what a carrier needs only through a value passed in.
An import in the wrong direction is easy to add and invisible until someone tries to use one
half alone, so it is checked here.

Two boundaries, two names: the model side is a *backend* (translator.backend); the carrier
side is an *adapter* (transunit.adapter). Neither is ever called the other.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGES = ("transunit", "translator")


def imported_packages(module: Path) -> set[str]:
    """Top-level packages a module imports, ignoring relative imports."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
    return found


def package_imports(package: str) -> dict[str, set[str]]:
    """Every module in ``package`` (recursively) mapped to what it imports."""
    return {str(module.relative_to(SRC)): imported_packages(module)
            for module in sorted((SRC / package).rglob("*.py"))}


class TestTheContractDependsOnNobody:
    """transunit is the shared bottom of the stack and must stay there."""

    def test_contract_imports_neither_a_translator_nor_a_backend(self) -> None:
        for name, imports in package_imports("transunit").items():
            assert "translator" not in imports, f"{name} imports the translator"

    def test_contract_is_stdlib_only(self) -> None:
        stdlib = set(sys.stdlib_module_names)
        for name, imports in package_imports("transunit").items():
            outside = {i for i in imports if i not in stdlib and i not in PACKAGES}
            assert not outside, f"{name} imports {outside}"


class TestTheTranslatorDependsOnlyOnTheContract:
    def test_translator_imports_only_stdlib_and_httpx(self) -> None:
        # The core translator needs only httpx. The one exception is the OPTIONAL embedding
        # retriever, which adds numpy -- and it alone may, imported lazily so the core import path
        # never pulls it. Confining the allowance to that module keeps the dependency honest.
        # (translator/embedding.py is a back-compat shim: a relative import, invisible to this
        # absolute-import-only walker, so it needs no allowance of its own.)
        stdlib = set(sys.stdlib_module_names)
        for name, imports in package_imports("translator").items():
            allowed = {"httpx"} | ({"numpy"} if name == "translator/retrieval/embedding.py"
                                   else set())
            outside = {i for i in imports
                       if i not in stdlib and i not in PACKAGES and i not in allowed}
            assert not outside, f"{name} imports {outside}"

    def test_translator_imports_no_concrete_carrier_adapter(self) -> None:
        # The property that lets any carrier be swapped in without the translator noticing.
        # The one thing it needs from an adapter -- what text the carrier can hold -- is
        # passed in, not imported.
        forbidden = {"subtitles", "gamescript", "documents", "media", "fakecarrier"}
        for name, imports in package_imports("translator").items():
            assert not (imports & forbidden), f"{name} imports a carrier adapter"


class TestEachSideRunsWithoutTheOther:
    """The strongest form: exercise one half with the other's dependency unimportable."""

    def _run(self, tmp_path: Path, body: str) -> subprocess.CompletedProcess:
        script = tmp_path / "check.py"
        script.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(SRC)!r})\n"
            "sys.modules['httpx'] = None\n"  # prove the contract needs no HTTP client
            + body + "print('ok')\n", encoding="utf-8")
        return subprocess.run([sys.executable, str(script)], capture_output=True, text=True)

    def test_the_contract_works_with_no_http_client_installed(self, tmp_path: Path) -> None:
        body = (
            "from transunit.units import Unit, Status, write_catalog, read_catalog\n"
            "from transunit.language import japanese_detector, LanguageProfile\n"
            "from transunit.placeholders import placeholder_indices\n"
            "import tempfile, pathlib\n"
            "assert placeholder_indices('[[0]] and [[1]]') == [0, 1]\n"
            "assert japanese_detector()('日本語', 'Still 日本語') is True\n"
            "with tempfile.TemporaryDirectory() as d:\n"
            "    p = pathlib.Path(d) / 'c.jsonl'\n"
            "    u = Unit(unit_id='x', rel_path='f', line_no=1, span_start=0, span_end=1,\n"
            "             command='', kind='T', source='hi')\n"
            "    write_catalog([u], p)\n"
            "    assert next(read_catalog(p)).unit_id == 'x'\n")
        result = self._run(tmp_path, body)
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout


class TestAdapterProtocol:
    """How a translator finds a carrier adapter without knowing which one it is."""

    def test_a_carrier_adapter_satisfies_the_protocol(self) -> None:
        from transunit.adapter import Adapter, load_adapter
        adapter = load_adapter("fakecarrier")
        assert isinstance(adapter, Adapter)
        assert adapter.name == "fakecarrier"

    def test_the_sanitiser_normalises_for_the_carrier(self) -> None:
        from transunit.adapter import load_adapter
        assert load_adapter("fakecarrier").sanitize_payload("a\nb") == "a\\nb"

    def test_unknown_adapter_fails_with_an_actionable_message(self) -> None:
        from transunit.adapter import AdapterError, load_adapter
        with pytest.raises(AdapterError, match="no adapter named 'rpgmaker'"):
            load_adapter("rpgmaker")

    def test_passthrough_is_available_for_a_plain_corpus(self) -> None:
        from transunit.adapter import passthrough
        assert passthrough("a\nb") == "a\nb"


class TestEntryPoints:
    def test_translator_cli_is_its_own_entry_point(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "translator.cli", "--help"],
            capture_output=True, text=True,
            env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"})
        assert result.returncode == 0, result.stderr
        for flag in ("--catalog", "--journal", "--adapter", "--backend"):
            assert flag in result.stdout, flag
