"""
Number normalization utilities for comparing text with different number representations.

Handles converting between word numbers (e.g., "two hundred") and
digit numbers (e.g., "200") to enable accurate comparison between
TTS output and STT transcriptions.
"""
import re
from typing import List, Optional, Tuple

try:
    from num2words import num2words  # noqa: F401 (available for callers)
    from word2number import w2n
except ImportError:
    num2words = None
    w2n = None

# Ordinal word mappings
ORDINAL_WORDS = {
    'first': '1', 'second': '2', 'third': '3', 'fourth': '4', 'fifth': '5',
    'sixth': '6', 'seventh': '7', 'eighth': '8', 'ninth': '9', 'tenth': '10',
    'eleventh': '11', 'twelfth': '12', 'thirteenth': '13', 'fourteenth': '14',
    'fifteenth': '15', 'sixteenth': '16', 'seventeenth': '17', 'eighteenth': '18',
    'nineteenth': '19', 'twentieth': '20', 'thirtieth': '30', 'fortieth': '40',
    'fiftieth': '50', 'sixtieth': '60', 'seventieth': '70', 'eightieth': '80',
    'ninetieth': '90', 'hundredth': '100', 'thousandth': '1000', 'millionth': '1000000',
}

NUMBER_WORDS = {
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
    'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million',
    'billion', 'trillion', 'and',
}

MAGNITUDE_WORDS = {'hundred', 'thousand', 'million', 'billion', 'trillion'}

ORDINAL_SUFFIX_PATTERN = re.compile(r'\b(\d+)(st|nd|rd|th)\b', re.IGNORECASE)

MIXED_NUMBER_PATTERN = re.compile(
    r'\b(\d+)\s+(hundred|thousand|million|billion|trillion)\b',
    re.IGNORECASE,
)


def _is_number_word(word: str, allow_capitalized_and: bool = True) -> bool:
    """Check if a word is a number word."""
    word_lower = word.lower()

    if not allow_capitalized_and and word != word_lower and word_lower == 'and':
        return False

    if word_lower in NUMBER_WORDS:
        return True
    if word_lower in ORDINAL_WORDS:
        return True

    if '-' in word_lower:
        parts = word_lower.split('-')
        if len(parts) == 2:
            if parts[0] in NUMBER_WORDS and parts[1] in ORDINAL_WORDS:
                return True
            if parts[0] in NUMBER_WORDS and parts[1] in NUMBER_WORDS:
                return True
    return False


def _convert_ordinal_suffix(text: str) -> str:
    """Convert ordinal suffixes (1st, 2nd, 3rd, 21st, etc.) to plain numbers."""
    return ORDINAL_SUFFIX_PATTERN.sub(r'\1', text)


def _convert_mixed_format(text: str) -> str:
    """Convert mixed digit-word formats like '2 hundred' to '200'."""
    def replace_mixed(match):
        digit = int(match.group(1))
        magnitude = match.group(2).lower()
        multipliers = {
            'hundred': 100, 'thousand': 1000, 'million': 1000000,
            'billion': 1000000000, 'trillion': 1000000000000,
        }
        result = digit * multipliers.get(magnitude, 1)
        return str(result)

    return MIXED_NUMBER_PATTERN.sub(replace_mixed, text)


def _extract_number_sequences(text: str) -> List[Tuple[str, int, int]]:
    """Extract sequences of number words from text."""
    words = text.split()
    sequences = []
    i = 0

    while i < len(words):
        word = words[i]
        clean_word = re.sub(r'[^\w\s-]', '', word)

        if '-' in clean_word and not _is_number_word(clean_word):
            parts = clean_word.split('-')
            if len(parts) >= 2 and _is_number_word(parts[0]) and not _is_number_word(parts[1]):
                sequences.append((parts[0], i, i + 1))
                i += 1
                continue

        is_num_word = _is_number_word(clean_word, allow_capitalized_and=False)

        if not is_num_word:
            if clean_word.lower() == 'a' and i + 1 < len(words):
                next_word = re.sub(r'[^\w\s-]', '', words[i + 1])
                if next_word.lower() in MAGNITUDE_WORDS:
                    pass
                else:
                    i += 1
                    continue
            else:
                i += 1
                continue

        sequence_words = []
        start_idx = i

        while i < len(words):
            word = words[i]
            clean = re.sub(r'[^\w\s-]', '', word)
            has_breaking_punct = bool(re.search(r'[,;]', word))

            if not sequence_words and clean.lower() == 'a':
                if i + 1 < len(words):
                    next_clean = re.sub(r'[^\w\s-]', '', words[i + 1])
                    if next_clean.lower() in MAGNITUDE_WORDS:
                        sequence_words.append(clean)
                        i += 1
                        continue
                break

            is_ordinal = clean.lower() in ORDINAL_WORDS

            if _is_number_word(clean) or is_ordinal:
                sequence_words.append(clean)
                i += 1
                if has_breaking_punct or is_ordinal:
                    break
            else:
                break

        if sequence_words:
            sequences.append((
                ' '.join(sequence_words),
                start_idx,
                i,
            ))

    return sequences


def _convert_word_number_to_digit(word_number: str) -> Optional[str]:
    """Convert a word number to its digit representation."""
    if w2n is None:
        return None

    clean = word_number.strip().lower()

    clean = re.sub(r'\ba\s+(hundred|thousand|million|billion)', r'one \1', clean)

    if clean in ORDINAL_WORDS:
        return ORDINAL_WORDS[clean]

    parts = clean.split()
    if len(parts) >= 2 and parts[-1] in ORDINAL_WORDS:
        base_words = ' '.join(parts[:-1])
        ordinal_word = parts[-1]
        try:
            base = w2n.word_to_num(base_words) if base_words else 0
            ordinal_val = int(ORDINAL_WORDS[ordinal_word])
            if base % 10 == 0 and base < 100:
                return str(base + ordinal_val)
            else:
                return str(ordinal_val)
        except (ValueError, KeyError):
            pass

    if '-' in clean:
        parts = clean.split('-')
        if len(parts) == 2:
            if parts[1] in ORDINAL_WORDS:
                try:
                    base = w2n.word_to_num(parts[0])
                    ordinal = int(ORDINAL_WORDS[parts[1]])
                    return str(base + ordinal)
                except (ValueError, KeyError):
                    pass

    try:
        number = w2n.word_to_num(clean)
        return str(number)
    except ValueError:
        pass

    ordinal_match = re.match(r'^(.+)(th)$', clean)
    if ordinal_match:
        base_word = ordinal_match.group(1)
        try:
            number = w2n.word_to_num(base_word)
            return str(number)
        except ValueError:
            pass

    return None


def normalize_numbers_to_digits(text: str) -> str:
    """
    Normalize all number representations in text to digit form.

    Converts:
    1. Ordinal suffixes (1st -> 1, 2nd -> 2)
    2. Mixed formats (2 hundred -> 200)
    3. Word numbers (two hundred -> 200)

    Args:
        text: Input text with mixed number representations

    Returns:
        Text with all numbers in digit form
    """
    text = _convert_ordinal_suffix(text)
    text = _convert_mixed_format(text)

    sequences = _extract_number_sequences(text)

    words = text.split()
    for word_seq, start_idx, end_idx in reversed(sequences):
        digit_form = _convert_word_number_to_digit(word_seq)
        if digit_form:
            if start_idx == end_idx - 1:
                original_word = words[start_idx]
                if '-' in original_word:
                    clean = re.sub(r'[^\w\s-]', '', original_word)
                    parts = clean.split('-')
                    if len(parts) >= 2 and parts[0].lower() == word_seq.lower():
                        trailing_punct = ''
                        punct_match = re.search(r'[^\w\s-]+$', original_word)
                        if punct_match:
                            trailing_punct = punct_match.group(0)

                        new_word = digit_form + '-' + '-'.join(parts[1:]) + trailing_punct
                        words[start_idx] = new_word
                        continue

            last_word = words[end_idx - 1]
            trailing_punct = ''
            punct_match = re.search(r'[^\w\s-]+$', last_word)
            if punct_match:
                trailing_punct = punct_match.group(0)

            words[start_idx:end_idx] = [digit_form + trailing_punct]

    return ' '.join(words)


def normalize_numbers_for_comparison(text1: str, text2: str) -> Tuple[str, str]:
    """Normalize numbers in both texts to enable accurate comparison."""
    return (
        normalize_numbers_to_digits(text1),
        normalize_numbers_to_digits(text2),
    )
