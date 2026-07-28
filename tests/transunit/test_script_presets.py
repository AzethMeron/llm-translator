"""Named script detectors, so a different-script pair is a configuration choice.

These back the "any language pair in the world" claim for pairs whose scripts differ:
naming the source script gives an untranslated-echo detector with no lexical profile at all.
"""
from __future__ import annotations

import pytest

from transunit.language import (
    LanguageError,
    ScriptDetector,
    available_scripts,
    script_detector,
)

# One echo, one clean translation into Latin, and one target with residual source letters,
# per script the user named plus a couple of common others.
CASES = {
    "han": ("这是中文", "This is English", "还有 中文 here"),
    "japanese": ("日本語のテキスト", "English text only", "some 日本語 left"),
    "cyrillic": ("Это русский текст", "This is English", "mixed русский in"),
    "greek": ("Αυτό είναι ελληνικά", "This is English", "some ελληνικά here"),
    "arabic": ("هذا نص عربي", "This is English", "with عربي inside"),
    "hebrew": ("זה עברית", "This is English", "some עברית here"),
    "hangul": ("이것은 한국어", "This is English", "some 한국어 left"),
}


class TestRegistry:
    def test_the_user_named_scripts_are_all_available(self) -> None:
        available = set(available_scripts())
        for script in ("arabic", "han", "japanese", "cyrillic", "greek"):
            assert script in available, script

    def test_an_unknown_script_is_reported_with_the_known_ones(self) -> None:
        with pytest.raises(LanguageError, match="unknown script 'tengwar'"):
            script_detector("tengwar")


class TestDetectors:
    @pytest.mark.parametrize("script", list(CASES))
    def test_echo_is_flagged(self, script: str) -> None:
        source, _, _ = CASES[script]
        assert script_detector(script)(source, source) is True

    @pytest.mark.parametrize("script", list(CASES))
    def test_a_clean_latin_translation_is_not_flagged(self, script: str) -> None:
        source, clean, _ = CASES[script]
        assert script_detector(script)(source, clean) is False

    @pytest.mark.parametrize("script", list(CASES))
    def test_residual_source_letters_are_flagged(self, script: str) -> None:
        source, _, residual = CASES[script]
        assert script_detector(script)(source, residual) is True

    @pytest.mark.parametrize("script", list(CASES))
    def test_an_empty_target_is_never_flagged(self, script: str) -> None:
        source, _, _ = CASES[script]
        assert script_detector(script)(source, "") is False

    def test_han_and_cyrillic_do_not_cross_detect(self) -> None:
        # A Chinese-source detector must not fire on Russian residue and vice versa, or a
        # mixed corpus would misreport.
        assert script_detector("han")("这是中文", "это русский") is False
        assert script_detector("cyrillic")("это русский", "这是中文") is False

    def test_a_returned_detector_is_a_scriptdetector(self) -> None:
        assert isinstance(script_detector("arabic"), ScriptDetector)


class TestCustomRanges:
    def test_a_script_not_in_the_registry_can_be_built_by_hand(self) -> None:
        # Runic, say: the escape hatch for a script the registry does not name.
        runic = ScriptDetector.from_ranges((("ᚠ", "᛿"),))
        assert runic("ᚠᚢᚦ", "plain latin") is False
        assert runic("ᚠᚢᚦ", "has ᚠ rune") is True
