"""
Number normalization utilities for comparing text with different number representations.

Handles converting spoken-form numbers, dates, currency, and times to canonical
written form to enable accurate comparison between TTS output and STT transcriptions.
"""
import re
from typing import Optional, Tuple

try:
    from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
    _itn: Optional[InverseNormalizer] = InverseNormalizer(lang="en")
except ImportError:
    _itn = None

try:
    from text_to_num import alpha2digit
except ImportError:
    alpha2digit = None

try:
    from num2words import num2words  # noqa: F401 (available for callers)
except ImportError:
    num2words = None

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


def normalize_numbers_to_digits(text: str) -> str:
    """
    Normalize spoken-form expressions in text to digit/written form.

    Handles numbers, ordinals, dates, currency, times, and measurements.
    Pipeline:
      1. Mixed digit-word formats ("2 hundred" → "200")
      2. NeMo inverse text normalization (dates, currency, times, compound numbers)
      3. text-to-num for remaining word numbers ("five" → "5")
      4. Ordinal suffix stripping ("22nd" → "22")
      5. Residual "a N" cleanup ("a 100" → "100")

    Args:
        text: Input text with mixed number representations

    Returns:
        Text with all numbers in canonical written form
    """
    text = _MIXED_FORMAT.sub(
        lambda m: str(int(m.group(1)) * _MIXED_MULTIPLIERS[m.group(2).lower()]),
        text,
    )
    if _itn is not None:
        text = _itn.normalize(text, verbose=False)
    if alpha2digit is not None:
        text = alpha2digit(text, "en", threshold=0)
    text = _ORDINAL_SUFFIX.sub(r'\1', text)
    text = _A_BEFORE_NUMBER.sub(r'\1', text)
    return text


def normalize_numbers_for_comparison(text1: str, text2: str) -> Tuple[str, str]:
    """Normalize numbers in both texts to enable accurate comparison."""
    return normalize_numbers_to_digits(text1), normalize_numbers_to_digits(text2)
