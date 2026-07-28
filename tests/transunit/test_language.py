"""Tests for :mod:`transunit.language`.

Untranslated detection backs one blocking check -- did the model translate, or echo its
input? Two mechanisms produce the same ``(source, target) -> bool`` shape: a *lexical*
detector for same-script pairs and a *script* detector for differing-script pairs. The
properties that matter more than raw accuracy are that both **abstain rather than
guess** (an abstention used as a negative would fail every short line) and both are
deterministic. These tests exercise both paths and the profile-loading around them.
"""
from __future__ import annotations

import pytest

from transunit.language import (
    Detection,
    LanguageError,
    LanguageProfile,
    ScriptDetector,
    detect,
    find_profile,
    is_untranslated,
    japanese_detector,
    lexical_detector,
    load_profiles,
    strip_accents,
    tokenize,
)

EN = LanguageProfile(
    code="en", name="English",
    stopwords=frozenset({"the", "a", "and", "is", "of", "to", "in", "on", "it"}))
PL = LanguageProfile(
    code="pl", name="Polish",
    stopwords=frozenset({"nie", "jest", "się", "że", "jak", "to", "co"}),
    distinctive_characters=frozenset("ąćęłńóśźż"))
PROFILES = (EN, PL)


class TestTokenize:
    def test_splits_casefolds_and_drops_digits_and_punctuation(self) -> None:
        assert tokenize("Hello, World! 123") == ["hello", "world"]

    def test_empty_text_yields_no_tokens(self) -> None:
        assert tokenize("") == []

    def test_casefold_not_lowercase(self) -> None:
        """Casefolding is the comparison-defined operation: ``ß`` folds to ``ss`` where a
        plain lowercase would leave it, and comparison against a stopword list depends on
        that normalisation."""
        assert tokenize("Straße") == ["strasse"]


class TestStripAccents:
    def test_removes_combining_diacritics(self) -> None:
        assert strip_accents("café") == "cafe"
        assert strip_accents("naïve") == "naive"

    def test_ascii_is_unchanged(self) -> None:
        assert strip_accents("hello") == "hello"

    def test_non_combining_letters_survive(self) -> None:
        """A stroked ``ł`` is one code point, not a base plus combining mark, so accent
        stripping cannot reach it -- only the decomposable ó and ź go."""
        assert strip_accents("łódź") == "łodz"


class TestLanguageProfileInvariants:
    def test_empty_code_is_rejected(self) -> None:
        with pytest.raises(LanguageError):
            LanguageProfile(code="  ", name="x", stopwords=frozenset({"a"}))

    def test_profile_with_no_evidence_is_rejected(self) -> None:
        """A profile with neither stopwords nor distinctive characters can never match any
        text, so it is a useless configuration and refused at construction."""
        with pytest.raises(LanguageError) as excinfo:
            LanguageProfile(code="en", name="English")
        assert excinfo.value.code == "en"

    def test_score_of_empty_text_is_zero(self) -> None:
        assert EN.score("") == 0.0

    def test_distinctive_characters_alone_can_carry_a_score(self) -> None:
        """Lexical and orthographic evidence are added, not multiplied, so a fragment too
        short for stopwords still scores on a distinctive letter."""
        assert PL.score("słowo") > 0.0


class TestDetect:
    def test_no_profiles_raises(self) -> None:
        with pytest.raises(LanguageError):
            detect("the cat is here", ())

    def test_abstains_below_minimum_tokens(self) -> None:
        """Two tokens cannot outvote a single word shared by both languages, so anything
        under the minimum yields no verdict rather than a guess."""
        assert detect("the a", PROFILES) is None

    def test_abstains_when_nothing_matches(self) -> None:
        """A best score of zero is absence of evidence, distinct from a tie; it too must
        abstain rather than pick an arbitrary winner."""
        assert detect("xxx yyy zzz", PROFILES) is None

    def test_abstains_on_a_near_tie(self) -> None:
        """One English and one Polish function word in equal measure leaves the margin
        below threshold -- balanced evidence is treated as no evidence."""
        assert detect("the nie xxx", PROFILES) is None

    def test_reports_a_clear_winner(self) -> None:
        result = detect("the cat is on the mat", PROFILES)
        assert isinstance(result, Detection)
        assert result.code == "en"
        assert result.confidence > 0.0

    def test_confidence_is_the_margin_over_the_runner_up(self) -> None:
        result = detect("nie wiem jak to co", PROFILES)
        assert result is not None and result.code == "pl"
        top, second = result.scores[0][1], result.scores[1][1]
        assert result.confidence == pytest.approx(min(top - second, 1.0))

    def test_determinism(self) -> None:
        assert detect("the cat is on the mat", PROFILES) == detect(
            "the cat is on the mat", PROFILES)


class TestIsUntranslated:
    def test_verbatim_echo_is_caught_without_any_detection(self) -> None:
        """The commonest failure -- the model returning its input -- needs no evidence and
        is caught outright, even for text too short for the detector."""
        assert is_untranslated("Ala ma kota", "Ala ma kota", PROFILES, "pl") is True

    def test_echo_ignoring_surrounding_whitespace(self) -> None:
        assert is_untranslated("Ala ma kota", "  Ala ma kota\n", PROFILES, "pl") is True

    def test_positive_same_language_detection_is_untranslated(self) -> None:
        """A different-but-still-Polish target is positive evidence the translation did not
        happen."""
        assert is_untranslated("Ala ma kota", "nie wiem jak to", PROFILES, "pl") is True

    def test_a_real_translation_is_not_flagged(self) -> None:
        assert is_untranslated("Ala ma kota", "the cat is on the mat", PROFILES, "pl") is False

    def test_empty_target_is_not_flagged(self) -> None:
        assert is_untranslated("Ala ma kota", "", PROFILES, "pl") is False
        assert is_untranslated("Ala ma kota", "   ", PROFILES, "pl") is False

    def test_abstention_is_not_treated_as_untranslated(self) -> None:
        """A target too short to detect and not an echo yields no verdict; the check must
        read that as 'no information', not as a failure."""
        assert is_untranslated("Ala ma kota", "cat", PROFILES, "pl") is False


class TestLexicalDetector:
    def test_returns_a_callable_matching_is_untranslated(self) -> None:
        detector = lexical_detector(PROFILES, "pl")
        assert callable(detector)
        assert detector("Ala ma kota", "Ala ma kota") is True
        assert detector("Ala ma kota", "the cat is on the mat") is False

    def test_unknown_source_code_fails_fast(self) -> None:
        """Validating the source code at wiring time turns a typo into an error here,
        rather than a check that silently never fires."""
        with pytest.raises(LanguageError) as excinfo:
            lexical_detector(PROFILES, "xx")
        assert excinfo.value.code == "xx"


class TestFindProfile:
    def test_returns_matching_profile(self) -> None:
        assert find_profile(PROFILES, "en") is EN

    def test_missing_code_lists_what_is_available(self) -> None:
        with pytest.raises(LanguageError) as excinfo:
            find_profile(PROFILES, "de")
        assert excinfo.value.code == "de"
        assert "en" in str(excinfo.value) and "pl" in str(excinfo.value)


class TestLoadProfiles:
    def _write(self, tmp_path, text: str):
        path = tmp_path / "languages.toml"
        path.write_text(text)
        return path

    def test_parses_a_valid_table(self, tmp_path) -> None:
        path = self._write(tmp_path, """
[languages.en]
name = "English"
stopwords = ["the", "and"]

[languages.pl]
name = "Polish"
distinctive_characters = "ąćę"
""")
        profiles = load_profiles(path)
        by_code = {p.code: p for p in profiles}
        assert set(by_code) == {"en", "pl"}
        assert by_code["en"].name == "English"
        assert "the" in by_code["en"].stopwords
        assert "ą" in by_code["pl"].distinctive_characters

    def test_stopwords_are_casefolded_on_load(self, tmp_path) -> None:
        path = self._write(tmp_path, '[languages.en]\nname="E"\nstopwords=["The","AND"]\n')
        (profile,) = load_profiles(path)
        assert "the" in profile.stopwords and "and" in profile.stopwords

    def test_missing_file_raises_with_path(self, tmp_path) -> None:
        with pytest.raises(LanguageError) as excinfo:
            load_profiles(tmp_path / "nope.toml")
        assert excinfo.value.path == tmp_path / "nope.toml"
        assert "not found" in excinfo.value.reason

    def test_invalid_toml_raises(self, tmp_path) -> None:
        path = self._write(tmp_path, "this is = = not toml")
        with pytest.raises(LanguageError) as excinfo:
            load_profiles(path)
        assert "invalid TOML" in excinfo.value.reason

    def test_absent_languages_table_raises(self, tmp_path) -> None:
        path = self._write(tmp_path, '[other]\nkey = "value"\n')
        with pytest.raises(LanguageError):
            load_profiles(path)

    def test_empty_languages_table_raises(self, tmp_path) -> None:
        """At least one profile is needed to tell a translation from its source, so an
        empty table is a misconfiguration."""
        path = self._write(tmp_path, "[languages]\n")
        with pytest.raises(LanguageError):
            load_profiles(path)

    def test_unknown_key_in_profile_raises(self, tmp_path) -> None:
        path = self._write(tmp_path, '[languages.en]\nname="E"\nstopwords=["a"]\nbogus=1\n')
        with pytest.raises(LanguageError) as excinfo:
            load_profiles(path)
        assert excinfo.value.code == "en"
        assert "unknown key" in excinfo.value.reason

    def test_profile_without_evidence_raises(self, tmp_path) -> None:
        """A profile giving neither stopwords nor distinctive characters propagates the
        LanguageProfile invariant up through the loader."""
        path = self._write(tmp_path, '[languages.en]\nname = "English"\n')
        with pytest.raises(LanguageError):
            load_profiles(path)

    def test_profile_body_must_be_a_table(self, tmp_path) -> None:
        path = self._write(tmp_path, '[languages]\nen = "not a table"\n')
        with pytest.raises(LanguageError) as excinfo:
            load_profiles(path)
        assert excinfo.value.code == "en"

    def test_stopwords_must_be_a_list_of_strings(self, tmp_path) -> None:
        path = self._write(tmp_path, '[languages.en]\nname="E"\nstopwords="the"\n')
        with pytest.raises(LanguageError):
            load_profiles(path)


class TestScriptDetector:
    def test_japanese_detector_flags_residual_source_letters(self) -> None:
        """When source and target scripts differ, source letters surviving into the target
        is the strongest evidence the translation did not happen -- decidable with no
        profiles or thresholds."""
        detect_ja = japanese_detector()
        assert detect_ja("ソースの文", "translated テキスト") is True

    def test_verbatim_echo_is_flagged(self) -> None:
        assert japanese_detector()("日本語", "日本語") is True

    def test_clean_target_is_not_flagged(self) -> None:
        assert japanese_detector()("日本語", "Hello world") is False

    def test_empty_target_is_not_flagged(self) -> None:
        assert japanese_detector()("日本語", "") is False
        assert japanese_detector()("日本語", "   ") is False

    def test_small_kana_only_is_not_source_script(self) -> None:
        """Small kana modify a preceding syllable and carry no word of their own, so a
        payload made only of them is not evidence of an untranslated line."""
        assert japanese_detector()("source", "ぁぃぅ") is False

    def test_from_ranges_builds_a_working_detector(self) -> None:
        detector = ScriptDetector.from_ranges((("あ", "ん"),))
        assert detector.has_source_script("あ") is True
        assert detector.has_source_script("A") is False

    def test_from_ranges_honours_the_ignore_set(self) -> None:
        detector = ScriptDetector.from_ranges((("ぁ", "ん"),), ignore="ぁ")
        assert detector.has_source_script("ぁ") is False
        assert detector.has_source_script("あ") is True

    def test_from_ranges_requires_at_least_one_range(self) -> None:
        with pytest.raises(LanguageError):
            ScriptDetector.from_ranges(())

    def test_determinism(self) -> None:
        det = japanese_detector()
        assert det("ソース", "テキスト") == det("ソース", "テキスト")


class TestUncoveredPaths:
    def test_detection_str_shows_the_code_and_confidence(self) -> None:
        detection = Detection(code="en", confidence=0.87, scores=(("en", 0.9), ("pl", 0.03)))
        text = str(detection)
        assert "en" in text and "confidence" in text and "0.87" in text

    def test_a_non_string_profile_name_is_rejected(self, tmp_path) -> None:
        path = tmp_path / "langs.toml"
        path.write_text('[languages.en]\nname = 5\nstopwords = ["the"]\n', encoding="utf-8")
        with pytest.raises(LanguageError):
            load_profiles(path)
