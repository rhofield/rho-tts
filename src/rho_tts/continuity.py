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

    Speaker identity is compared with the provider's reference voice plus the
    most recent accepted speech (at least ``WINDOW_SECONDS`` of it); loudness is
    compared with that accepted speech alone. With neither a reference voice nor
    a predecessor, the first segment establishes the voice.
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


_RATE = 16000
# Speaker embeddings are unreliable on short clips: a 0.8 s opening sentence
# scored 0.59-0.64 against the voice it was cloned from while its 2-4 s
# successors scored 0.84-0.89, so a lone short predecessor is never the anchor.
WINDOW_SECONDS = 3.0


def _speech(samples, rate):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if rate != _RATE:
        import librosa

        samples = librosa.resample(samples, orig_sr=rate, target_sr=_RATE)
    return samples


def _read(path):
    import soundfile as sf

    samples, rate = sf.read(path, always_2d=True)
    return _speech(samples, rate)


class ContinuityValidator:
    """Call-local validator with explicit advancement after candidate selection.

    ``previous`` is the accepted speech window, oldest first. It is an immutable
    tuple so callers can snapshot and restore it between decay rounds.
    """

    def __init__(self, config, reference_audio=None):
        self.config = config
        self.previous = ()
        self._anchor = None
        try:
            self.reference = _read(reference_audio) if reference_audio is not None else None
            if config.previous_audio is not None:
                self.previous = (_read(config.previous_audio),)
            self._anchors()  # Reject unusable reference audio before synthesis.
        except Exception as exc:
            raise ValidationError(f"Audio continuity reference unavailable: {exc}") from exc

    def advance(self, speech):
        """Accept a selected candidate, keeping the newest clips that cover the window."""
        window = self.previous + (speech,)
        while len(window) > 1 and sum(map(len, window[1:])) >= WINDOW_SECONDS * _RATE:
            window = window[1:]
        self.previous = window

    def _anchors(self):
        if self._anchor is None or self._anchor[0] is not self.previous:
            speaker = ([self.reference] if self.reference is not None else []) + list(self.previous)
            embedding = _features(np.concatenate(speaker), _RATE)[0] if speaker else None
            level = speech_level(np.concatenate(self.previous), _RATE) if self.previous else None
            seconds = sum(map(len, speaker)) / _RATE
            self._anchor = (self.previous, embedding, level, seconds)
        return self._anchor[1:]

    def evaluate(self, audio, rate):
        try:
            samples = audio.detach().cpu().numpy()
            if samples.ndim == 2:
                samples = samples.mean(axis=0)
            speech = _speech(samples, rate)
            embedding, level = _features(speech, _RATE)
            anchor, anchor_level, anchor_seconds = self._anchors()
            if anchor is None:
                return dict(passed=True, reason="First segment; no predecessor", score=0.0), speech
            similarity = float(np.clip(np.dot(anchor, embedding), -1, 1))
            difference = abs(level - anchor_level) if anchor_level is not None else None
            cfg = self.config
            score = (
                max(0, cfg.min_similarity - similarity) / cfg.min_similarity
                + max(0, (difference or 0.0) - cfg.max_loudness_db) / cfg.max_loudness_db
            )
            return dict(
                passed=score == 0,
                speaker_similarity=similarity,
                loudness_difference_db=difference,
                score=score,
                minimum_speaker_similarity=cfg.min_similarity,
                maximum_loudness_difference_db=cfg.max_loudness_db,
                reference_voice=self.reference is not None,
                anchor_seconds=anchor_seconds,
            ), speech
        except Exception as exc:
            raise ValidationError(f"Audio continuity validation unavailable: {exc}") from exc
