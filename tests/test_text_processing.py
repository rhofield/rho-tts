"""Tests for text processing: phonetic mapping, text splitting, number normalization."""
import pytest

from ralph_tts.base_tts import BaseTTS


class ConcreteTTS(BaseTTS):
    """Minimal concrete implementation for testing base class methods."""

    def __init__(self, **kwargs):
        # Skip voice encoder and CUDA setup for testing
        self.device = "cpu"
        self.seed = 42
        self.deterministic = False
        self.phonetic_mapping = kwargs.get("phonetic_mapping", {})
        self.silence_threshold_db = -50.0
        self.crossfade_duration_sec = 0.05
        self.trim_silence = True
        self.fade_duration_sec = 0.02
        self.force_sentence_split = True
        self.inter_sentence_pause_sec = 0.1
        self._voice_encoder = None
        self.reference_embedding = None
        self._sample_rate = 24000

    def _generate_audio(self, text, **kwargs):
        pass

    def generate(self, texts, output_base_path, cancellation_token=None):
        pass

    def generate_single(self, text, output_path, cancellation_token=None):
        pass

    @property
    def sample_rate(self):
        return self._sample_rate


class TestPhoneticMapping:
    def test_empty_mapping(self):
        tts = ConcreteTTS()
        assert tts._apply_phonetic_mapping("hello world") == "hello world"

    def test_single_mapping(self):
        tts = ConcreteTTS(phonetic_mapping={"exocrine": "exo-crene"})
        assert tts._apply_phonetic_mapping("the exocrine fires") == "the exo-crene fires"

    def test_multiple_mappings(self):
        mapping = {"AI": "A.I.", "GPU": "G.P.U."}
        tts = ConcreteTTS(phonetic_mapping=mapping)
        result = tts._apply_phonetic_mapping("AI runs on GPU")
        assert result == "A.I. runs on G.P.U."

    def test_no_match(self):
        tts = ConcreteTTS(phonetic_mapping={"xyz": "abc"})
        assert tts._apply_phonetic_mapping("hello world") == "hello world"

    def test_case_sensitive(self):
        tts = ConcreteTTS(phonetic_mapping={"Hello": "Heh-lo"})
        assert tts._apply_phonetic_mapping("Hello") == "Heh-lo"
        assert tts._apply_phonetic_mapping("hello") == "hello"


class TestTextSplitting:
    def test_single_short_sentence(self):
        tts = ConcreteTTS()
        tts.force_sentence_split = False
        result = tts._split_text_into_segments("Hello world.", 100)
        assert result == ["Hello world."]

    def test_force_sentence_split(self):
        tts = ConcreteTTS()
        tts.force_sentence_split = True
        result = tts._split_text_into_segments("First sentence. Second sentence.", 1000)
        assert len(result) == 2
        assert "First" in result[0]
        assert "Second" in result[1]

    def test_long_text_splits_at_word_boundary(self):
        tts = ConcreteTTS()
        tts.force_sentence_split = False
        text = "word " * 100  # 500 chars
        result = tts._split_text_into_segments(text.strip(), 50)
        assert len(result) > 1
        for seg in result:
            assert len(seg) <= 55  # Some tolerance

    def test_empty_text(self):
        tts = ConcreteTTS()
        result = tts._split_text_into_segments("", 100)
        assert result == []

    def test_single_sentence_no_split(self):
        tts = ConcreteTTS()
        tts.force_sentence_split = True
        result = tts._split_text_into_segments("Just one sentence here", 1000)
        assert len(result) == 1


class TestNumberNormalization:
    """Test the number normalizer used in STT validation."""

    def test_ordinal_suffix(self):
        from ralph_tts.validation.stt.number_normalizer import normalize_numbers_to_digits

        assert "1" in normalize_numbers_to_digits("1st")
        assert "2" in normalize_numbers_to_digits("2nd")
        assert "3" in normalize_numbers_to_digits("3rd")

    def test_word_to_digit(self):
        from ralph_tts.validation.stt.number_normalizer import normalize_numbers_to_digits, w2n

        if w2n is None:
            pytest.skip("word2number not installed")

        result = normalize_numbers_to_digits("two hundred")
        assert "200" in result

    def test_mixed_format(self):
        from ralph_tts.validation.stt.number_normalizer import normalize_numbers_to_digits

        result = normalize_numbers_to_digits("3 thousand")
        assert "3000" in result

    def test_no_numbers(self):
        from ralph_tts.validation.stt.number_normalizer import normalize_numbers_to_digits

        result = normalize_numbers_to_digits("hello world")
        assert result == "hello world"
