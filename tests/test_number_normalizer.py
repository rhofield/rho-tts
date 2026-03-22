"""Tests for number normalization utilities in validation/stt/number_normalizer.py."""

import pytest

from rho_tts.validation.stt.number_normalizer import (
    normalize_numbers_for_comparison,
    normalize_numbers_to_digits,
)


class TestNormalizeNumbersToDigits:
    def test_ordinal_suffix_converted(self):
        result = normalize_numbers_to_digits("the 1st place")
        assert "1st" not in result
        assert "1" in result

    def test_mixed_format_converted(self):
        assert normalize_numbers_to_digits("2 hundred") == "200"
        assert normalize_numbers_to_digits("3 thousand") == "3000"
        assert normalize_numbers_to_digits("about 2 hundred people") == "about 200 people"

    def test_word_numbers_converted(self):
        assert normalize_numbers_to_digits("I have five apples") == "I have 5 apples"
        assert normalize_numbers_to_digits("twenty five") == "25"
        assert normalize_numbers_to_digits("two hundred and three") == "203"

    def test_ordinal_words_converted(self):
        assert normalize_numbers_to_digits("first place") == "1 place"
        assert normalize_numbers_to_digits("twenty first") == "21"

    def test_plain_text_unchanged(self):
        text = "the quick brown fox"
        assert normalize_numbers_to_digits(text) == text

    def test_already_digits_unchanged(self):
        assert normalize_numbers_to_digits("I have 42 apples") == "I have 42 apples"

    def test_multiple_ordinal_suffixes(self):
        result = normalize_numbers_to_digits("the 1st and 2nd and 3rd")
        assert "1st" not in result
        assert "2nd" not in result
        assert "3rd" not in result

    def test_combined_ordinal_and_mixed(self):
        result = normalize_numbers_to_digits("the 1st of 2 hundred")
        assert "1st" not in result
        assert "200" in result

    def test_trailing_punctuation_preserved(self):
        result = normalize_numbers_to_digits("I have five.")
        assert result == "I have 5."

    def test_a_hundred_pattern(self):
        assert normalize_numbers_to_digits("a hundred apples") == "100 apples"
        assert normalize_numbers_to_digits("a thousand miles") == "1000 miles"

    def test_date_spoken_form(self):
        result = normalize_numbers_to_digits("march twenty second twenty twenty six")
        assert result == "march 22 2026"

    def test_currency_spoken_form(self):
        result = normalize_numbers_to_digits("five dollars and ninety nine cents")
        assert result == "$5.99"

    def test_time_spoken_form(self):
        result = normalize_numbers_to_digits("three thirty pm")
        assert result == "03:30 p.m."


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

    def test_same_number_different_forms(self):
        t1, t2 = normalize_numbers_for_comparison("two hundred", "2 hundred")
        assert t1 == "200"
        assert t2 == "200"

    def test_word_vs_digit_match(self):
        t1, t2 = normalize_numbers_for_comparison(
            "I have twenty five apples",
            "I have 25 apples",
        )
        assert t1 == t2


class TestWithLibrariesAvailable:
    """Integration tests using the real NeMo and text-to-num libraries."""

    nemo = pytest.importorskip(
        "nemo_text_processing.inverse_text_normalization.inverse_normalize",
        reason="nemo_text_processing not installed",
    )
    text2num = pytest.importorskip("text_to_num", reason="text2num not installed")

    def test_simple_word_number(self):
        assert normalize_numbers_to_digits("five") == "5"

    def test_compound_word_number(self):
        assert normalize_numbers_to_digits("two hundred") == "200"

    def test_large_number(self):
        assert normalize_numbers_to_digits("one thousand") == "1000"

    def test_hyphenated_ordinal(self):
        assert normalize_numbers_to_digits("twenty-first") == "21"

    def test_full_pipeline(self):
        assert normalize_numbers_to_digits("I have twenty five apples") == "I have 25 apples"

    def test_comparison_pipeline(self):
        t1, t2 = normalize_numbers_for_comparison(
            "I have twenty five apples",
            "I have 25 apples",
        )
        assert t1 == t2

    def test_date_normalization(self):
        result = normalize_numbers_to_digits("march twenty second twenty twenty six")
        assert result == "march 22 2026"

    def test_currency_normalization(self):
        result = normalize_numbers_to_digits("five dollars and ninety nine cents")
        assert result == "$5.99"

    def test_time_normalization(self):
        result = normalize_numbers_to_digits("three thirty pm")
        assert result == "03:30 p.m."

    def test_a_hundred_converts(self):
        assert normalize_numbers_to_digits("a hundred") == "100"

    def test_a_thousand_converts(self):
        assert normalize_numbers_to_digits("a thousand") == "1000"

    def test_decimal_number(self):
        assert normalize_numbers_to_digits("three point five") == "3.5"

    def test_large_compound_number(self):
        assert normalize_numbers_to_digits("one million two hundred thousand") == "1200000"

    def test_currency_in_sentence(self):
        result = normalize_numbers_to_digits("the meeting costs five hundred dollars")
        assert result == "the meeting costs $500"

    def test_standalone_year(self):
        assert normalize_numbers_to_digits("twenty twenty six") == "2026"

    def test_comma_separated_numbers(self):
        assert normalize_numbers_to_digits("five, six") == "5, 6"
