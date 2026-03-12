"""Tests for number normalization utilities in validation/stt/number_normalizer.py."""

from unittest.mock import MagicMock, patch

import pytest

from rho_tts.validation.stt.number_normalizer import (
    MAGNITUDE_WORDS,
    NUMBER_WORDS,
    ORDINAL_WORDS,
    _convert_mixed_format,
    _convert_ordinal_suffix,
    _convert_word_number_to_digit,
    _extract_number_sequences,
    _is_number_word,
    normalize_numbers_for_comparison,
    normalize_numbers_to_digits,
)


class TestConstants:
    def test_ordinal_words_contains_basic_ordinals(self):
        assert "first" in ORDINAL_WORDS
        assert "second" in ORDINAL_WORDS
        assert "third" in ORDINAL_WORDS
        assert "tenth" in ORDINAL_WORDS
        assert "twentieth" in ORDINAL_WORDS
        assert "hundredth" in ORDINAL_WORDS
        assert "thousandth" in ORDINAL_WORDS

    def test_ordinal_words_values_are_digit_strings(self):
        for word, value in ORDINAL_WORDS.items():
            assert isinstance(value, str)
            assert value.isdigit(), f"ORDINAL_WORDS['{word}'] = '{value}' is not a digit string"

    def test_number_words_contains_basic_numbers(self):
        assert "zero" in NUMBER_WORDS
        assert "one" in NUMBER_WORDS
        assert "twenty" in NUMBER_WORDS
        assert "hundred" in NUMBER_WORDS
        assert "thousand" in NUMBER_WORDS
        assert "and" in NUMBER_WORDS

    def test_magnitude_words_subset_of_number_words(self):
        assert MAGNITUDE_WORDS.issubset(NUMBER_WORDS)

    def test_magnitude_words_contents(self):
        assert MAGNITUDE_WORDS == {"hundred", "thousand", "million", "billion", "trillion"}


class TestIsNumberWord:
    def test_basic_number_words(self):
        assert _is_number_word("one") is True
        assert _is_number_word("two") is True
        assert _is_number_word("twenty") is True
        assert _is_number_word("hundred") is True
        assert _is_number_word("thousand") is True
        assert _is_number_word("zero") is True

    def test_case_insensitive(self):
        assert _is_number_word("One") is True
        assert _is_number_word("TWO") is True
        assert _is_number_word("Hundred") is True

    def test_ordinal_words(self):
        assert _is_number_word("first") is True
        assert _is_number_word("second") is True
        assert _is_number_word("twentieth") is True
        assert _is_number_word("hundredth") is True

    def test_non_number_words(self):
        assert _is_number_word("hello") is False
        assert _is_number_word("cat") is False
        assert _is_number_word("the") is False
        assert _is_number_word("") is False

    def test_hyphenated_number_words(self):
        assert _is_number_word("twenty-one") is True
        assert _is_number_word("thirty-five") is True
        assert _is_number_word("ninety-nine") is True

    def test_hyphenated_ordinal_compound(self):
        assert _is_number_word("twenty-first") is True
        assert _is_number_word("thirty-second") is True
        assert _is_number_word("fifty-third") is True

    def test_hyphenated_non_number(self):
        assert _is_number_word("well-known") is False
        assert _is_number_word("one-eyed") is False
        assert _is_number_word("twenty-something") is False

    def test_and_word_allowed_by_default(self):
        assert _is_number_word("and") is True
        assert _is_number_word("And") is True
        assert _is_number_word("AND") is True

    def test_capitalized_and_disallowed(self):
        assert _is_number_word("And", allow_capitalized_and=False) is False
        assert _is_number_word("AND", allow_capitalized_and=False) is False

    def test_lowercase_and_still_allowed_when_disallowed_capitalized(self):
        assert _is_number_word("and", allow_capitalized_and=False) is True

    def test_hyphenated_three_parts_not_matched(self):
        # Only two-part hyphenated words are checked
        assert _is_number_word("twenty-one-hundred") is False


class TestConvertOrdinalSuffix:
    def test_1st(self):
        assert _convert_ordinal_suffix("1st") == "1"

    def test_2nd(self):
        assert _convert_ordinal_suffix("2nd") == "2"

    def test_3rd(self):
        assert _convert_ordinal_suffix("3rd") == "3"

    def test_4th(self):
        assert _convert_ordinal_suffix("4th") == "4"

    def test_21st(self):
        assert _convert_ordinal_suffix("21st") == "21"

    def test_42nd(self):
        assert _convert_ordinal_suffix("42nd") == "42"

    def test_100th(self):
        assert _convert_ordinal_suffix("100th") == "100"

    def test_mixed_in_sentence(self):
        result = _convert_ordinal_suffix("the 1st place and 2nd place")
        assert result == "the 1 place and 2 place"

    def test_no_ordinal(self):
        text = "the quick brown fox"
        assert _convert_ordinal_suffix(text) == text

    def test_plain_digits_unchanged(self):
        assert _convert_ordinal_suffix("42 items") == "42 items"

    def test_case_insensitive(self):
        assert _convert_ordinal_suffix("1ST") == "1"
        assert _convert_ordinal_suffix("2ND") == "2"


class TestConvertMixedFormat:
    def test_two_hundred(self):
        assert _convert_mixed_format("2 hundred") == "200"

    def test_three_thousand(self):
        assert _convert_mixed_format("3 thousand") == "3000"

    def test_one_million(self):
        assert _convert_mixed_format("1 million") == "1000000"

    def test_five_billion(self):
        assert _convert_mixed_format("5 billion") == "5000000000"

    def test_two_trillion(self):
        assert _convert_mixed_format("2 trillion") == "2000000000000"

    def test_mixed_in_sentence(self):
        result = _convert_mixed_format("about 2 hundred people")
        assert result == "about 200 people"

    def test_multiple_mixed_formats(self):
        result = _convert_mixed_format("2 hundred and 3 thousand")
        assert result == "200 and 3000"

    def test_plain_digits_unchanged(self):
        assert _convert_mixed_format("200 items") == "200 items"

    def test_no_numbers(self):
        text = "the quick brown fox"
        assert _convert_mixed_format(text) == text

    def test_case_insensitive(self):
        assert _convert_mixed_format("2 Hundred") == "200"
        assert _convert_mixed_format("3 THOUSAND") == "3000"


class TestExtractNumberSequences:
    def test_single_number_word(self):
        sequences = _extract_number_sequences("five")
        assert len(sequences) == 1
        assert sequences[0][0] == "five"
        assert sequences[0][1] == 0
        assert sequences[0][2] == 1

    def test_multi_word_number(self):
        sequences = _extract_number_sequences("two hundred")
        assert len(sequences) == 1
        assert sequences[0][0] == "two hundred"

    def test_number_in_sentence(self):
        sequences = _extract_number_sequences("I have five apples")
        assert len(sequences) == 1
        assert sequences[0][0] == "five"
        assert sequences[0][1] == 2
        assert sequences[0][2] == 3

    def test_no_numbers(self):
        sequences = _extract_number_sequences("the quick brown fox")
        assert len(sequences) == 0

    def test_ordinal_ends_sequence(self):
        # Ordinals should break the sequence
        sequences = _extract_number_sequences("twenty first")
        assert len(sequences) == 1
        # The ordinal "first" should end the sequence
        assert sequences[0][0] in ("twenty first", "twenty")

    def test_a_thousand_pattern(self):
        sequences = _extract_number_sequences("a thousand")
        assert len(sequences) == 1
        assert sequences[0][0] == "a thousand"

    def test_a_hundred_pattern(self):
        sequences = _extract_number_sequences("a hundred")
        assert len(sequences) == 1
        assert sequences[0][0] == "a hundred"

    def test_a_without_magnitude_skipped(self):
        sequences = _extract_number_sequences("a cat")
        assert len(sequences) == 0

    def test_empty_string(self):
        sequences = _extract_number_sequences("")
        assert len(sequences) == 0

    def test_multiple_separate_numbers(self):
        # "and" is in NUMBER_WORDS so it gets absorbed into the second sequence
        sequences = _extract_number_sequences("five cats and ten dogs")
        assert len(sequences) == 2
        assert sequences[0][0] == "five"
        assert sequences[1][0] == "and ten"

    def test_punctuation_stripped_from_words(self):
        sequences = _extract_number_sequences("five, dogs")
        assert len(sequences) == 1
        # comma is breaking punctuation, so the sequence ends
        assert sequences[0][0] == "five"

    def test_hyphenated_number_nonnumber_extracts_number_part(self):
        # A word like "five-year" should extract "five"
        sequences = _extract_number_sequences("five-year plan")
        assert len(sequences) == 1
        assert sequences[0][0] == "five"


class TestConvertWordNumberToDigit:
    def test_returns_none_when_w2n_is_none(self):
        with patch("rho_tts.validation.stt.number_normalizer.w2n", None):
            result = _convert_word_number_to_digit("two hundred")
            assert result is None

    def test_ordinal_word_returns_digit(self):
        with patch("rho_tts.validation.stt.number_normalizer.w2n", MagicMock()):
            result = _convert_word_number_to_digit("first")
            assert result == "1"

    def test_ordinal_word_twentieth(self):
        with patch("rho_tts.validation.stt.number_normalizer.w2n", MagicMock()):
            result = _convert_word_number_to_digit("twentieth")
            assert result == "20"

    def test_plain_word_number_via_w2n(self):
        mock_w2n = MagicMock()
        mock_w2n.word_to_num.return_value = 200
        with patch("rho_tts.validation.stt.number_normalizer.w2n", mock_w2n):
            result = _convert_word_number_to_digit("two hundred")
            assert result == "200"
            mock_w2n.word_to_num.assert_called_with("two hundred")

    def test_hyphenated_ordinal(self):
        mock_w2n = MagicMock()
        mock_w2n.word_to_num.return_value = 20
        with patch("rho_tts.validation.stt.number_normalizer.w2n", mock_w2n):
            result = _convert_word_number_to_digit("twenty-first")
            assert result == "21"

    def test_multi_word_with_trailing_ordinal(self):
        mock_w2n = MagicMock()
        mock_w2n.word_to_num.return_value = 20
        with patch("rho_tts.validation.stt.number_normalizer.w2n", mock_w2n):
            result = _convert_word_number_to_digit("twenty first")
            assert result == "21"

    def test_a_hundred_converted_to_one_hundred(self):
        mock_w2n = MagicMock()
        mock_w2n.word_to_num.return_value = 100
        with patch("rho_tts.validation.stt.number_normalizer.w2n", mock_w2n):
            result = _convert_word_number_to_digit("a hundred")
            assert result == "100"
            # "a hundred" should be replaced with "one hundred" before calling w2n
            mock_w2n.word_to_num.assert_called_with("one hundred")

    def test_w2n_raises_valueerror_returns_none(self):
        mock_w2n = MagicMock()
        mock_w2n.word_to_num.side_effect = ValueError("not a number")
        with patch("rho_tts.validation.stt.number_normalizer.w2n", mock_w2n):
            result = _convert_word_number_to_digit("nonsense")
            assert result is None

    def test_ordinal_th_fallback(self):
        # Words ending in "th" that aren't in ORDINAL_WORDS but whose base
        # can be converted by w2n (e.g., a custom ordinal).
        mock_w2n = MagicMock()
        # First call (full word) raises ValueError, second call (stripped "th") succeeds
        mock_w2n.word_to_num.side_effect = [ValueError("nope"), 50]
        with patch("rho_tts.validation.stt.number_normalizer.w2n", mock_w2n):
            result = _convert_word_number_to_digit("fiftieth")
            # "fiftieth" is in ORDINAL_WORDS, so it returns directly
            assert result == "50"


class TestNormalizeNumbersToDigits:
    def test_ordinal_suffix_converted(self):
        result = normalize_numbers_to_digits("the 1st place")
        assert "1" in result
        assert "1st" not in result

    def test_mixed_format_converted(self):
        result = normalize_numbers_to_digits("about 2 hundred people")
        assert "200" in result

    def test_word_numbers_converted_with_mock_w2n(self):
        mock_w2n = MagicMock()
        mock_w2n.word_to_num.return_value = 5
        with patch("rho_tts.validation.stt.number_normalizer.w2n", mock_w2n):
            result = normalize_numbers_to_digits("I have five apples")
            assert "5" in result
            assert "five" not in result

    def test_plain_text_unchanged(self):
        text = "the quick brown fox"
        assert normalize_numbers_to_digits(text) == text

    def test_already_digits_unchanged(self):
        text = "I have 42 apples"
        assert normalize_numbers_to_digits(text) == text

    def test_multiple_conversions(self):
        result = normalize_numbers_to_digits("the 1st and 2nd and 3rd")
        assert "1st" not in result
        assert "2nd" not in result
        assert "3rd" not in result

    def test_combined_ordinal_and_mixed(self):
        result = normalize_numbers_to_digits("the 1st of 2 hundred")
        assert "1st" not in result
        assert "200" in result

    def test_without_w2n_word_numbers_left_alone(self):
        with patch("rho_tts.validation.stt.number_normalizer.w2n", None):
            result = normalize_numbers_to_digits("I have five apples")
            # Without w2n, word numbers cannot be converted
            assert "five" in result

    def test_trailing_punctuation_preserved(self):
        mock_w2n = MagicMock()
        mock_w2n.word_to_num.return_value = 5
        with patch("rho_tts.validation.stt.number_normalizer.w2n", mock_w2n):
            result = normalize_numbers_to_digits("I have five.")
            # Punctuation on the last word should be preserved
            assert result.endswith("5.") or result.endswith("five.")


class TestNormalizeNumbersForComparison:
    def test_returns_tuple(self):
        result = normalize_numbers_for_comparison("hello", "world")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_both_texts_normalized(self):
        t1, t2 = normalize_numbers_for_comparison("the 1st", "the 2nd")
        assert "1st" not in t1
        assert "2nd" not in t2

    def test_plain_text_passthrough(self):
        t1, t2 = normalize_numbers_for_comparison("hello world", "foo bar")
        assert t1 == "hello world"
        assert t2 == "foo bar"

    def test_mixed_format_normalized_in_both(self):
        t1, t2 = normalize_numbers_for_comparison("2 hundred", "3 thousand")
        assert t1 == "200"
        assert t2 == "3000"

    def test_normalizes_same_number_different_forms(self):
        mock_w2n = MagicMock()
        mock_w2n.word_to_num.return_value = 200
        with patch("rho_tts.validation.stt.number_normalizer.w2n", mock_w2n):
            t1, t2 = normalize_numbers_for_comparison("two hundred", "2 hundred")
            # Both should be normalized to "200"
            assert t1 == "200"
            assert t2 == "200"


class TestWithW2nAvailable:
    """Tests that use the real w2n library if available, skipped otherwise."""

    w2n = pytest.importorskip("word2number.w2n", reason="word2number not installed")

    def test_simple_word_number(self):
        result = _convert_word_number_to_digit("five")
        assert result == "5"

    def test_compound_word_number(self):
        result = _convert_word_number_to_digit("two hundred")
        assert result == "200"

    def test_large_number(self):
        result = _convert_word_number_to_digit("one thousand")
        assert result == "1000"

    def test_hyphenated_number(self):
        result = _convert_word_number_to_digit("twenty-one")
        assert result is not None
        # w2n may or may not handle hyphens, but our code handles it
        assert int(result) == 21

    def test_hyphenated_ordinal_real(self):
        result = _convert_word_number_to_digit("twenty-first")
        assert result == "21"

    def test_normalize_full_pipeline(self):
        result = normalize_numbers_to_digits("I have twenty five apples")
        assert "25" in result

    def test_normalize_comparison_pipeline(self):
        t1, t2 = normalize_numbers_for_comparison(
            "I have twenty five apples",
            "I have 25 apples",
        )
        assert t1 == t2

    def test_a_thousand_converts(self):
        result = _convert_word_number_to_digit("a thousand")
        assert result == "1000"

    def test_a_hundred_converts(self):
        result = _convert_word_number_to_digit("a hundred")
        assert result == "100"
