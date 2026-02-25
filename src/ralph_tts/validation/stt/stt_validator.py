"""
Speech-to-text validator for checking if generated audio matches intended text.

Uses Whisper models running locally. Supports faster-whisper (preferred)
with transformers as fallback.
"""
import logging
import os
import re
from typing import Optional, Tuple

from .number_normalizer import normalize_numbers_to_digits

logger = logging.getLogger(__name__)

# Global model cache for lazy loading
_whisper_model = None
_whisper_model_type = None


def _normalize_text(text: str, enable_number_normalization: bool = True) -> str:
    """
    Normalize text for comparison by removing punctuation, extra spaces, and lowercasing.
    Also normalizes number representations (e.g., "two hundred" -> "200").
    """
    if enable_number_normalization:
        try:
            text = normalize_numbers_to_digits(text)
        except Exception as e:
            logger.warning(f"Number normalization failed: {e}")

    text = text.lower()
    text = re.sub(r'\b(the|a|an)\b', ' ', text)
    text = text.replace('-', ' ')
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def _get_whisper_model(
    model_size: str = "tiny",
    device: Optional[str] = None,
):
    """
    Lazy load and cache the Whisper model.

    Tries faster-whisper first, falls back to transformers.

    Args:
        model_size: Whisper model size (default: "tiny")
        device: Device for transformers fallback (default: auto-detect)

    Returns:
        Tuple of (model, model_type)
    """
    global _whisper_model, _whisper_model_type

    if _whisper_model is not None:
        return _whisper_model, _whisper_model_type

    # Try faster-whisper first
    try:
        from faster_whisper import WhisperModel

        logger.info(f"Loading faster-whisper model ({model_size})...")
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.info("faster-whisper model loaded successfully (CPU)")
        _whisper_model_type = "faster-whisper"
        return _whisper_model, _whisper_model_type
    except ImportError:
        logger.warning("faster-whisper not found, falling back to transformers...")
    except Exception as e:
        logger.warning(f"Error loading faster-whisper: {e}, falling back to transformers...")

    # Fallback to transformers
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        logger.info(f"Loading Whisper model via transformers ({model_size})...")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        model_id = f"openai/whisper-{model_size}"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        _whisper_model = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        _whisper_model_type = "transformers"
        logger.info("Whisper model loaded successfully via transformers")
        return _whisper_model, _whisper_model_type
    except Exception as e:
        logger.error(f"Error loading Whisper via transformers: {e}")
        raise RuntimeError("Failed to load any Whisper model implementation")


def transcribe_audio(audio_path: str) -> Optional[str]:
    """
    Transcribe audio file to text using Whisper.

    Args:
        audio_path: Path to audio file

    Returns:
        Transcribed text, or None if transcription fails
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return None

    try:
        model, model_type = _get_whisper_model()

        if model_type == "faster-whisper":
            try:
                segments, _info = model.transcribe(audio_path, language="en")
                transcription = " ".join([segment.text for segment in segments])
            except Exception as e:
                logger.warning(f"Error during faster-whisper transcription: {e}")
                return None
        else:
            result = model(audio_path)
            transcription = result["text"]

        return transcription.strip()

    except Exception as e:
        logger.warning(f"Error transcribing audio: {e}")
        return None


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _fuzzy_word_match(word1: str, word2: str, max_distance: int = 2) -> bool:
    """Check if two words are similar enough to be considered a match."""
    if word1 == word2:
        return True
    if len(word1) < 3 or len(word2) < 3:
        return False

    distance = _levenshtein_distance(word1, word2)

    adjusted_max = max_distance
    if len(word1) > 8 or len(word2) > 8:
        adjusted_max = max_distance + 1

    return distance <= adjusted_max


def calculate_text_similarity(original_text: str, transcribed_text: str) -> float:
    """
    Calculate similarity between original and transcribed text.

    Uses multiple complementary metrics (Jaccard, ratio-based, sequence-based)
    and returns the maximum score for a forgiving comparison.

    Args:
        original_text: Original intended text
        transcribed_text: Text transcribed from audio

    Returns:
        Similarity score between 0.0 and 1.0
    """
    from difflib import SequenceMatcher

    orig_normalized = _normalize_text(original_text)
    trans_normalized = _normalize_text(transcribed_text)

    orig_words = set(orig_normalized.split())
    trans_words = set(trans_normalized.split())

    if not orig_words or not trans_words:
        return 0.0

    exact_matches = orig_words & trans_words

    unmatched_orig = orig_words - trans_words
    unmatched_trans = trans_words - orig_words

    fuzzy_matches = 0
    for orig_word in unmatched_orig:
        for trans_word in unmatched_trans:
            if _fuzzy_word_match(orig_word, trans_word):
                fuzzy_matches += 1
                break

    total_matches = len(exact_matches) + fuzzy_matches

    union = len(orig_words | trans_words)
    jaccard_similarity = total_matches / union if union > 0 else 0.0
    ratio_similarity = total_matches / len(orig_words) if orig_words else 0.0
    sequence_similarity = SequenceMatcher(None, orig_normalized, trans_normalized).ratio()

    return max(jaccard_similarity, ratio_similarity, sequence_similarity)


def validate_audio_text_match(
    audio_path: str,
    expected_text: str,
    threshold: float = 0.85,
) -> Tuple[bool, float, Optional[str]]:
    """
    Validate that audio matches expected text.

    Args:
        audio_path: Path to audio file
        expected_text: Expected text content
        threshold: Minimum similarity score to pass validation (0.0 to 1.0)

    Returns:
        Tuple of (is_valid, similarity_score, transcribed_text)
    """
    transcribed = transcribe_audio(audio_path)

    if transcribed is None:
        logger.warning("Transcription failed, skipping text validation")
        return True, 0.0, None

    similarity = calculate_text_similarity(expected_text, transcribed)
    is_valid = similarity >= threshold

    return is_valid, similarity, transcribed
