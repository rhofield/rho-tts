"""
Number normalization utilities for comparing text with different number representations.

Handles converting spoken-form numbers, dates, currency, and times to canonical
written form to enable accurate comparison between TTS output and STT transcriptions.
"""
import re
from typing import Tuple

from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
from text_to_num import alpha2digit
from num2words import num2words  # noqa: F401 (available for callers)

_itn = InverseNormalizer(lang="en")

# Converts "2 hundred" → "200", "3 thousand" → "3000", etc. (digit + magnitude word)
_MIXED_FORMAT = re.compile(
    r'\b(\d+)\s+(hundred|thousand|million|billion|trillion)\b',
    re.IGNORECASE,
)
_MIXED_MULTIPLIERS = {
    'hundred': 100, 'thousand': 1_000, 'million': 1_000_000,
    'billion': 1_000_000_000, 'trillion': 1_000_000_000_000,
}

# Strips ordinal suffixes: "22nd" → "22", "3rd" → "3"
_ORDINAL_SUFFIX = re.compile(r'\b(\d+)(st|nd|rd|th)\b', re.IGNORECASE)

# Cleans up residual "a 100" → "100" when "a hundred" wasn't fully resolved
_A_BEFORE_NUMBER = re.compile(r'\ba\s+(\d{2,})\b')

# Strip commas from digit groups: "1,500" → "1500", "1,000,000" → "1000000"
_DIGIT_COMMAS = re.compile(r'(\d),(\d{3})\b')

# Currency symbols to strip: "$500" → "500"
_CURRENCY_SYMBOL = re.compile(r'[\$\£\€\¥](\d)')


def _strip_digit_commas(text: str) -> str:
    """Remove commas from digit groups: '1,500' → '1500'."""
    while _DIGIT_COMMAS.search(text):
        text = _DIGIT_COMMAS.sub(r'\1\2', text)
    return text


def _strip_currency_symbols(text: str) -> str:
    """Remove currency symbols preceding digits: '$500' → '500'."""
    return _CURRENCY_SYMBOL.sub(r'\1', text)


def normalize_numbers_to_digits(text: str) -> str:
    """
    Normalize spoken-form expressions in text to digit/written form.

    Handles numbers, ordinals, dates, currency, times, and measurements.
    Pipeline:
      1. Strip commas from digit groups ("1,500" → "1500")
      2. Strip currency symbols ("$500" → "500")
      3. Mixed digit-word formats ("2 hundred" → "200")
      4. NeMo inverse text normalization (dates, currency, times, compound numbers)
      5. text-to-num for remaining word numbers ("five" → "5")
      6. Ordinal suffix stripping ("22nd" → "22")
      7. Residual "a N" cleanup ("a 100" → "100")

    Args:
        text: Input text with mixed number representations

    Returns:
        Text with all numbers in canonical written form
    """
    text = _strip_digit_commas(text)
    text = _strip_currency_symbols(text)
    text = _MIXED_FORMAT.sub(
        lambda m: str(int(m.group(1)) * _MIXED_MULTIPLIERS[m.group(2).lower()]),
        text,
    )
    text = _itn.normalize(text, verbose=False)
    text = alpha2digit(text, "en", threshold=0)
    text = _ORDINAL_SUFFIX.sub(r'\1', text)
    text = _A_BEFORE_NUMBER.sub(r'\1', text)
    return text


def normalize_numbers_for_comparison(text1: str, text2: str) -> Tuple[str, str]:
    """Normalize numbers in both texts to enable accurate comparison."""
    return normalize_numbers_to_digits(text1), normalize_numbers_to_digits(text2)
