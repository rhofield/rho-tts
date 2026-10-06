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
    compared with that accepted speech alone. Speaker identity is not judged on
    clips with under ``MIN_SPEAKER_SECONDS`` of speech. Pitch (median F0) and
    brightness (spectral tilt of voiced frames) are compared with the accepted
    speech, topped up with the reference voice until that speech covers
    ``WINDOW_SECONDS``. With
    neither a reference voice nor a predecessor, the first segment establishes
    the voice.
    Retries share the provider's max_iterations budget with other validators.
    strict_validation overrides allow_fallback.
    """

    previous_audio: Optional[str] = None
    min_similarity: float = 0.75
    max_loudness_db: float = 6.0
    max_pitch_semitones: float = 3.0
    max_tilt_db: float = 5.0
    allow_fallback: bool = True

    def __post_init__(self):
        if not math.isfinite(self.min_similarity) or not 0 < self.min_similarity <= 1:
            raise ValueError("min_similarity must be finite and in (0, 1]")
        if not math.isfinite(self.max_loudness_db) or not 0 < self.max_loudness_db <= 30:
            raise ValueError("max_loudness_db must be finite and in (0, 30]")
        if not math.isfinite(self.max_pitch_semitones) or not 0 < self.max_pitch_semitones <= 24:
            raise ValueError("max_pitch_semitones must be finite and in (0, 24]")
        if not math.isfinite(self.max_tilt_db) or not 0 < self.max_tilt_db <= 30:
            raise ValueError("max_tilt_db must be finite and in (0, 30]")
        if self.previous_audio is not None:
            object.__setattr__(self, "previous_audio", str(self.previous_audio))


@lru_cache(maxsize=1)
def _encoder():
    from resemblyzer import VoiceEncoder

    return VoiceEncoder(device="cpu")


def _active_frames(samples, rate):
    """RMS of 20 ms frames above a tenth of the loudest frame, and the frame length."""
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
    return active, frame


def speech_level(samples, rate):
    active, _ = _active_frames(samples, rate)
    return float(20 * np.log10(np.sqrt(np.mean(active**2))))


def speech_seconds(samples, rate):
    active, frame = _active_frames(samples, rate)
    return len(active) * frame / rate


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
# The same 0.8 s sentence scored 0.56-0.64 against its own sibling sentences
# too, so its similarity says nothing about the voice; it is reported, not judged.
MIN_SPEAKER_SECONDS = 1.5
# Pitch and tilt need this much voiced speech to be measured at all.
MIN_VOICED_SECONDS = 0.25
_HOP = 256


def voice_features(samples, rate=_RATE):
    """Median F0 and spectral tilt over voiced frames, plus voiced duration.

    Resemblyzer is built to ignore pitch and recording colour, which is exactly
    what makes adjacent sentences sound like different takes. Tilt is measured on
    voiced frames only: whole-clip brightness mostly tracks how many sibilants a
    sentence contains rather than how the voice sounds.
    """
    import librosa

    samples = np.asarray(samples, dtype=np.float32)
    f0, voiced, _ = librosa.pyin(samples, fmin=50, fmax=500, sr=rate, frame_length=1024, hop_length=_HOP)
    power = np.abs(librosa.stft(samples, n_fft=1024, hop_length=_HOP)) ** 2
    count = min(power.shape[1], len(voiced))
    power, voiced, f0 = power[:, :count], voiced[:count], f0[:count]
    voiced = voiced & np.isfinite(f0)
    seconds = float(voiced.sum() * _HOP / rate)
    if seconds < MIN_VOICED_SECONDS:
        return dict(pitch_hz=None, tilt_db=None, voiced_seconds=seconds)
    freqs = librosa.fft_frequencies(sr=rate, n_fft=1024)
    spectrum = power[:, voiced].mean(axis=1)
    low = spectrum[(freqs >= 100) & (freqs < 1000)].sum()
    high = spectrum[(freqs >= 2000) & (freqs < 6000)].sum()
    tilt = float(10 * np.log10(high / low)) if low > 0 and high > 0 else None
    return dict(pitch_hz=float(np.median(f0[voiced])), tilt_db=tilt, voiced_seconds=seconds)


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
            # Pitch and tilt follow the neighbouring speech once it fills the window;
            # until then the reference voice tops up a short predecessor.
            voice = list(self.previous)
            if sum(map(len, voice)) < WINDOW_SECONDS * _RATE and self.reference is not None:
                voice.insert(0, self.reference)
            prosody = voice_features(np.concatenate(voice)) if voice else None
            self._anchor = (self.previous, embedding, level, seconds, prosody)
        return self._anchor[1:]

    def evaluate(self, audio, rate):
        try:
            samples = audio.detach().cpu().numpy()
            if samples.ndim == 2:
                samples = samples.mean(axis=0)
            speech = _speech(samples, rate)
            embedding, level = _features(speech, _RATE)
            anchor, anchor_level, anchor_seconds, anchor_voice = self._anchors()
            if anchor is None:
                return dict(passed=True, reason="First segment; no predecessor", score=0.0), speech
            cfg = self.config
            similarity = float(np.clip(np.dot(anchor, embedding), -1, 1))
            judged = speech_seconds(speech, _RATE) >= MIN_SPEAKER_SECONDS
            difference = abs(level - anchor_level) if anchor_level is not None else None
            voice = voice_features(speech)
            pitch = tilt = None
            if voice["pitch_hz"] is not None and anchor_voice and anchor_voice["pitch_hz"] is not None:
                pitch = float(12 * np.log2(voice["pitch_hz"] / anchor_voice["pitch_hz"]))
            if voice["tilt_db"] is not None and anchor_voice and anchor_voice["tilt_db"] is not None:
                tilt = voice["tilt_db"] - anchor_voice["tilt_db"]
            # Half a second of speech with no voiced frame at all is whispered or broken.
            unvoiced = voice["voiced_seconds"] == 0 and speech_seconds(speech, _RATE) >= 0.5
            score = (
                (max(0, cfg.min_similarity - similarity) / cfg.min_similarity if judged else 0.0)
                + max(0, (difference or 0.0) - cfg.max_loudness_db) / cfg.max_loudness_db
                + max(0, abs(pitch or 0.0) - cfg.max_pitch_semitones) / cfg.max_pitch_semitones
                + max(0, abs(tilt or 0.0) - cfg.max_tilt_db) / cfg.max_tilt_db
                + float(unvoiced)
            )
            return dict(
                passed=score == 0,
                speaker_similarity=similarity,
                speaker_judged=judged,
                loudness_difference_db=difference,
                pitch_difference_semitones=pitch,
                tilt_difference_db=tilt,
                pitch_hz=voice["pitch_hz"],
                tilt_db=voice["tilt_db"],
                voiced_seconds=voice["voiced_seconds"],
                unvoiced=unvoiced,
                score=score,
                minimum_speaker_similarity=cfg.min_similarity,
                maximum_loudness_difference_db=cfg.max_loudness_db,
                maximum_pitch_difference_semitones=cfg.max_pitch_semitones,
                maximum_tilt_difference_db=cfg.max_tilt_db,
                reference_voice=self.reference is not None,
                anchor_seconds=anchor_seconds,
            ), speech
        except Exception as exc:
            raise ValidationError(f"Audio continuity validation unavailable: {exc}") from exc
