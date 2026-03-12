"""Tests for GenerationResult dataclass."""

import torch

from rho_tts.result import GenerationResult


class TestGenerationResult:
    def test_defaults(self):
        r = GenerationResult()
        assert r.path is None
        assert r.audio is None
        assert r.sample_rate == 0
        assert r.duration_sec == 0.0
        assert r.segments_count == 0
        assert r.format == "wav"

    def test_with_audio(self):
        audio = torch.zeros(16000)
        r = GenerationResult(
            audio=audio,
            sample_rate=16000,
            duration_sec=1.0,
            segments_count=1,
            format="wav",
        )
        assert r.audio is not None
        assert r.audio.shape == (16000,)
        assert r.sample_rate == 16000
        assert r.duration_sec == 1.0
        assert r.segments_count == 1

    def test_with_path(self):
        r = GenerationResult(
            path="/tmp/test.wav",
            sample_rate=24000,
            format="wav",
        )
        assert r.path == "/tmp/test.wav"

    def test_format_field(self):
        r = GenerationResult(format="mp3")
        assert r.format == "mp3"

    def test_equality(self):
        """Dataclasses support equality by default."""
        r1 = GenerationResult(sample_rate=16000, duration_sec=1.0)
        r2 = GenerationResult(sample_rate=16000, duration_sec=1.0)
        assert r1 == r2
