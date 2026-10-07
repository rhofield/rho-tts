"""Tests for text processing: phonetic mapping, text splitting, number normalization."""
import pytest

from rho_tts.base_tts import BaseTTS


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
        self.voice_id = None
        self.drift_model_path = None
        self._max_chars_explicit = True
        self._max_model_chars = 3000

    def _generate_audio(self, text, **kwargs):
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


class TestSplitOn:
    """``split_on`` picks the delimiter; None hands the model each text whole."""

    TEXT = "Here's a puzzle. Take three boxes: dog, bites, man. So how does a model notice?"

    def test_default_splits_on_full_stop(self):
        assert len(ConcreteTTS()._split_text_into_segments(self.TEXT, 1000)) == 3

    def test_none_keeps_text_whole(self):
        tts = ConcreteTTS()
        tts.split_on = None
        assert tts._split_text_into_segments(f"  {self.TEXT} ", 1000) == [self.TEXT]

    def test_none_ignores_max_chars(self):
        tts = ConcreteTTS()
        tts.split_on = None
        assert tts._split_text_into_segments(self.TEXT, 20) == [self.TEXT]

    def test_none_empty_text(self):
        tts = ConcreteTTS()
        tts.split_on = None
        assert tts._split_text_into_segments("   ", 100) == []

    def test_custom_delimiter(self):
        tts = ConcreteTTS()
        tts.split_on = "\n"
        assert tts._split_text_into_segments("One. Two.\nThree.", 1000) == ["One. Two.", "Three."]


class TestShortSentenceMerging:
    """A one-line opener spoken alone has too little context to settle the voice
    and too little voiced audio for continuity to judge, so it merges onward."""

    TEXT = "Here's a puzzle. Take three boxes: dog, bites, man. Arrange them one way and you get a headline."

    def _split(self, text, min_chars, max_chars=1000):
        tts = ConcreteTTS()
        tts.min_segment_chars = min_chars
        return tts._split_text_into_segments(text, max_chars)

    def test_off_by_default(self):
        assert ConcreteTTS()._split_text_into_segments(self.TEXT, 1000)[0] == "Here's a puzzle."

    def test_short_opener_merges_into_next_sentence(self):
        assert self._split(self.TEXT, 30) == [
            "Here's a puzzle. Take three boxes: dog, bites, man.",
            "Arrange them one way and you get a headline.",
        ]

    def test_consecutive_short_sentences_merge_until_long_enough(self):
        assert self._split("Yes. No. Maybe so. This sentence is long enough alone.", 15) == [
            "Yes. No. Maybe so.",
            "This sentence is long enough alone.",
        ]

    def test_short_final_sentence_merges_backward(self):
        assert self._split("This first sentence is plenty long. Done.", 30) == [
            "This first sentence is plenty long. Done.",
        ]

    def test_merge_never_exceeds_max_chars(self):
        assert self._split("Hi. This next sentence is quite long here.", 30, max_chars=40) == [
            "Hi.", "This next sentence is quite long here.",
        ]


class TestNumberNormalization:
    """Test the number normalizer used in STT validation."""

    def test_ordinal_suffix(self):
        from rho_tts.validation.stt.number_normalizer import normalize_numbers_to_digits

        assert "1" in normalize_numbers_to_digits("1st")
        assert "2" in normalize_numbers_to_digits("2nd")
        assert "3" in normalize_numbers_to_digits("3rd")

    def test_word_to_digit(self):
        from rho_tts.validation.stt.number_normalizer import normalize_numbers_to_digits

        result = normalize_numbers_to_digits("two hundred")
        assert "200" in result

    def test_mixed_format(self):
        from rho_tts.validation.stt.number_normalizer import normalize_numbers_to_digits

        result = normalize_numbers_to_digits("3 thousand")
        assert "3000" in result

    def test_no_numbers(self):
        from rho_tts.validation.stt.number_normalizer import normalize_numbers_to_digits

        result = normalize_numbers_to_digits("hello world")
        assert result == "hello world"
