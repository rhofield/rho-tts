"""Optional adjacent-speech validation; thresholds are perceptual heuristics."""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from .exceptions import ValidationError


@dataclass(frozen=True)
class ContinuityConfig:
    """Per-call context, never stored on a provider.

    With no predecessor, the first segment establishes the voice. Subsequent
    segments (including list items) are compared to the last accepted segment.
    Retries share the provider's max_iterations budget with other validators.
    strict_validation overrides allow_fallback.
    """

    previous_audio: Optional[str] = None
    min_similarity: float = 0.75
    max_loudness_db: float = 6.0
    allow_fallback: bool = True

    def __post_init__(self):
        if not math.isfinite(self.min_similarity) or not 0 < self.min_similarity <= 1:
            raise ValueError("min_similarity must be finite and in (0, 1]")
        if not math.isfinite(self.max_loudness_db) or not 0 < self.max_loudness_db <= 30:
            raise ValueError("max_loudness_db must be finite and in (0, 30]")
        if self.previous_audio is not None:
            object.__setattr__(self, "previous_audio", str(self.previous_audio))


@lru_cache(maxsize=1)
def _encoder():
    from resemblyzer import VoiceEncoder

    return VoiceEncoder(device="cpu")


def speech_level(samples, rate):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all() or rate <= 0:
        raise ValueError("Audio continuity requires finite speech samples and a positive sample rate")
    frame = max(1, round(rate * 0.02))
    samples = np.pad(samples, (0, (-len(samples)) % frame))
    rms = np.sqrt(np.mean(samples.reshape(-1, frame) ** 2, axis=1))
    active = rms[rms > max(float(rms.max()) * 0.1, 1e-5)]
    if len(active) * frame / rate < 0.1:
        raise ValueError("Audio continuity requires detectable speech")
    return float(20 * np.log10(np.sqrt(np.mean(active**2))))


def _features(samples, rate):
    from resemblyzer import preprocess_wav

    level = speech_level(samples, rate)
    wav = preprocess_wav(np.asarray(samples, dtype=np.float32), source_sr=rate)
    if len(wav) < 1600:
        raise ValueError("Audio continuity requires detectable speech")
    embedding = _encoder().embed_utterance(wav)
    norm = np.linalg.norm(embedding)
    if not np.isfinite(embedding).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError("Audio continuity returned invalid speaker features")
    return embedding / norm, level


class ContinuityValidator:
    """Call-local validator with explicit advancement after candidate selection."""

    def __init__(self, config):
        self.config = config
        self.previous = None
        if config.previous_audio is not None:
            try:
                import soundfile as sf

                samples, rate = sf.read(config.previous_audio, always_2d=True)
                self.previous = _features(samples.mean(axis=1), rate)
            except Exception as exc:
                raise ValidationError(f"Audio continuity reference unavailable: {exc}") from exc

    def evaluate(self, audio, rate):
        try:
            samples = audio.detach().cpu().numpy()
            if samples.ndim == 2:
                samples = samples.mean(axis=0)
            features = _features(samples, rate)
            if self.previous is None:
                return dict(passed=True, reason="First segment; no predecessor", score=0.0), features
            similarity = float(np.clip(np.dot(self.previous[0], features[0]), -1, 1))
            difference = abs(features[1] - self.previous[1])
            cfg = self.config
            score = (
                max(0, cfg.min_similarity - similarity) / cfg.min_similarity
                + max(0, difference - cfg.max_loudness_db) / cfg.max_loudness_db
            )
            return dict(
                passed=score == 0,
                speaker_similarity=similarity,
                loudness_difference_db=difference,
                score=score,
                minimum_speaker_similarity=cfg.min_similarity,
                maximum_loudness_difference_db=cfg.max_loudness_db,
            ), features
        except Exception as exc:
            raise ValidationError(f"Audio continuity validation unavailable: {exc}") from exc
