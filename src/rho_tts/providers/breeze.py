"""
Breeze-TTS-2 provider implementation.

Voice cloning from reference audio plus its exact transcript, and — unique
among the providers here — natural-language voice *design* and *direction*
via an ``instruction`` string.

Always runs in an isolated venv: upstream pins torch/transformers exactly, and
its source is cloned rather than pip-installed (see isolation.venv_manager).

Licensing: the upstream code is Apache 2.0, but the model weights are under the
BreezeBlue Research and Non-Commercial License. Commercial use requires written
authorization from RESONIA, INC. This provider is excluded from the ``all``
extra for that reason.
"""
import contextlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from ..base_tts import BaseTTS
from ..provider_info import ProviderInfo, VoiceInfo

logger = logging.getLogger(__name__)

MODEL_REPO = "BreezeBlue/Breeze-TTS-2"
# Pinned deliberately. The weights repo has no tags, and its main branch moved
# during development (c1c8ca18 -> ee9fc7cf, "Update model license to v1.1"),
# so an unpinned snapshot_download can silently swap the weights — and the
# licence — underneath a working install.
MODEL_REVISION = "ee9fc7cf3ea481e1b1cb3a0aef13038efbb68fb5"

# Upstream's own defaults, from infer.py.
NEUTRAL_INSTRUCTION = "Speak clearly and naturally."
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1

# Mimi codec frame rate; upstream emits 24 kHz mono.
SAMPLE_RATE = 24000

# Template names understood by breeze_infer.templates.
TEMPLATE_INSTRUCTION = "tts_instruction"  # instruction only, no reference audio
TEMPLATE_REFERENCE = "ref_edit_tata"      # reference audio (+ optional instruction)


class BreezeTTS(BaseTTS):
    """
    Breeze-TTS-2 implementation.

    Supports three modes, selected automatically from the arguments given:

    - **Voice cloning** — ``reference_audio`` + ``reference_text``.
    - **Voice design** — ``instruction`` only, no reference. Generates a voice
      from a text description. Note the voice is not stable across seeds.
    - **Voice direction** — reference *and* instruction, to steer tone or pace
      while keeping the cloned timbre.

    Args:
        device: Device to run the model on ('cuda' or 'cpu'). CPU is unusably slow.
        seed: Random seed for consistent voice generation
        deterministic: If True, use deterministic CUDA operations
        reference_audio: Path to audio file for voice cloning (optional)
        reference_text: Exact transcript of reference_audio. Required whenever
            reference_audio is set — Breeze aligns the two to factor timbre
            from content, and a wrong transcript degrades the clone.
        instruction: Natural-language voice/style description. Used alone for
            voice design, or alongside a reference for voice direction.
        cfg_scale: Classifier-free guidance strength for ``instruction``.
            1.0 disables steering; upstream recommends ~4.0 when instructing.
        attn_implementation: Attention backend. Defaults to "eager" because the
            shipped config requests flash_attention_2 while upstream's
            requirements.txt does not install flash-attn. Set to
            "flash_attention_2" on a GPU where it is available — the result is
            numerically identical, only faster.
        fast: Enable upstream's compiled fast path. Off by default: it needs a
            torch.compile/CUDA-graph warmup that does not complete on smaller
            GPUs (observed to hang on an RTX 3060).
        strict_validation: Require working validators and reject audio after failed retries
        max_chars_per_segment: Max characters per text segment
        max_iterations: Maximum validation retry iterations
        accent_drift_threshold: Threshold for accent drift
        text_similarity_threshold: Min similarity for STT validation
        drift_model_path: Explicit path to a .pkl drift classifier model
        phonetic_mapping: Custom word-to-pronunciation mapping
    """

    MAX_MODEL_CHARS = 3000

    def __init__(
        self,
        device: str = "cuda",
        seed: int = 789,
        deterministic: bool = False,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        instruction: Optional[str] = None,
        cfg_scale: float = 1.0,
        attn_implementation: str = "eager",
        fast: bool = False,
        strict_validation: bool = False,
        max_chars_per_segment: Optional[int] = None,
        max_iterations: int = 10,
        accent_drift_threshold: float = 0.17,
        text_similarity_threshold: float = 0.85,
        drift_model_path: Optional[str] = None,
        phonetic_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__(device, seed, deterministic, phonetic_mapping=phonetic_mapping)

        if reference_audio is not None and reference_text is None:
            raise ValueError(
                "reference_text (exact transcript of reference_audio) is required "
                "when reference_audio is set"
            )
        if reference_audio is None and instruction is None:
            raise ValueError(
                "BreezeTTS requires either reference_audio (with reference_text) "
                "for voice cloning, or instruction for voice design."
            )
        if not cfg_scale > 0:
            raise ValueError(f"cfg_scale must be greater than 0, got {cfg_scale}")

        self.reference_audio_path = reference_audio
        self.reference_text = reference_text
        self.voice_cloning = reference_audio is not None
        self.instruction = instruction if instruction is not None else NEUTRAL_INSTRUCTION
        self.cfg_scale = cfg_scale
        self.attn_implementation = attn_implementation
        self.fast = fast
        self.strict_validation = strict_validation
        self.drift_model_path = drift_model_path

        self._max_chars_explicit = max_chars_per_segment is not None
        self.max_chars_per_segment = (
            max_chars_per_segment if max_chars_per_segment is not None else 800
        )
        self.max_iterations = max_iterations
        self.accent_drift_threshold = accent_drift_threshold
        self.text_similarity_threshold = text_similarity_threshold

        self._ref_audio_24k: Optional[str] = None
        self._load_model()

    # -- Model loading --------------------------------------------------------

    def _load_model(self) -> None:
        """Load the Breeze model, tokenizers and streaming runtime.

        Deliberately does not call upstream's ``breeze_infer.runtime.load_runtime``:
        that helper cannot override the text encoder's attention backend, which
        the shipped config.json pins to flash_attention_2 — a dependency
        upstream does not declare in requirements.txt.
        """
        try:
            from models.breeze import BreezeForConditionalGeneration
            from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
            from qwen_tts import Qwen3TTSTokenizer
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Breeze-TTS dependencies are unavailable. This provider is "
                "designed to run in an isolated venv — use "
                "TTSFactory.get_tts_instance('breeze'), which provisions it "
                "automatically."
            ) from exc

        from huggingface_hub import snapshot_download

        ckpt = Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION))

        self._tokenizer = AutoTokenizer.from_pretrained(ckpt)

        config = AutoConfig.from_pretrained(ckpt)
        config.text_encoder_config.preferred_attn_implementation = self.attn_implementation

        self.model = BreezeForConditionalGeneration.from_pretrained(
            ckpt,
            config=config,
            dtype=torch.bfloat16,
            attn_implementation=self.attn_implementation,
        )
        self.model.to(self.device).eval()
        self._apply_generation_config()

        self._audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(ckpt / "audio_tokenizer"), device_map=self.device
        )

        self._runtime = FastBreezeStreamingRuntime(
            self.model,
            self._audio_tokenizer,
            FastStreamingConfig(
                max_new_tokens=MAX_NEW_TOKENS,
                max_seq_len=MAX_SEQ_LEN,
                fast_all=True if self.fast else None,
                repetition_penalty=REPETITION_PENALTY,
            ),
            tokenizer=self._tokenizer,
        )
        if self.fast and self._runtime.fast_enabled:
            self._warmup()

        if self.voice_cloning:
            self._ref_audio_24k = self._prepare_reference(self.reference_audio_path)

    def _apply_generation_config(self) -> None:
        """Apply upstream's generation defaults to the model and depth decoder."""
        from breeze_infer.runtime import update_generation_config_for_breeze

        update_generation_config_for_breeze(self.model)

    def _warmup(self) -> None:
        """Run the compiled fast path's warmup profile."""
        from dataclasses import replace

        import models
        from models.warmup_profile import load_warmup_profile

        config_path = Path(models.__file__).resolve().parent.parent / "configs" / "fast.json"
        profile = load_warmup_profile(config_path)
        profile = replace(profile, codec_chunk_frames=self._runtime.codec_chunk_frames)
        self._runtime.warmup_from_profile(profile)

    def _prepare_reference(self, src: str) -> str:
        """Downmix and resample the reference to mono 24 kHz.

        Done explicitly rather than left to the audio tokenizer: a silently
        resampled reference degrades clone fidelity in ways that are hard to
        trace back. soundfile is used for I/O because torchaudio >= 2.9
        delegates all loading and saving to torchcodec.
        """
        import soundfile as sf
        import torchaudio

        data, sr = sf.read(src, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data).T
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

        fd, path = tempfile.mkstemp(suffix="_ref24k.wav", prefix="rho_breeze_")
        os.close(fd)
        sf.write(path, wav.squeeze(0).numpy(), SAMPLE_RATE, subtype="PCM_16")
        return path

    # -- Generation -----------------------------------------------------------

    def _build_request(self, text: str) -> tuple:
        """Build the upstream request dict and pick the matching template."""
        request = {
            "id": "rho-tts",
            "text": text,
            "instruction": self.instruction,
            "speaker": "S0",
        }
        if self.voice_cloning:
            request["ref_audio_path"] = self._ref_audio_24k
            request["ref_text"] = self.reference_text
            return request, TEMPLATE_REFERENCE
        return request, TEMPLATE_INSTRUCTION

    def _generate_audio(
        self,
        text: Union[str, List[str]],
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate audio, concatenating the runtime's streaming chunks."""
        if isinstance(text, list):
            return [self._generate_audio(t, **kwargs) for t in text]

        import numpy as np
        from breeze_infer.templates import get_template, prepare_inputs

        self._set_seeds()
        request, template_name = self._build_request(text)

        inputs = prepare_inputs(
            self._tokenizer,
            self._audio_tokenizer,
            self.model,
            [request],
            get_template(template_name),
            guidance_scale=kwargs.get("cfg_scale", self.cfg_scale),
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )

        chunks = [
            chunk.audio
            for chunk in self._runtime.iter_audio_chunks(inputs, request_id=request["id"])
        ]
        if not chunks:
            raise RuntimeError(f"Breeze produced no audio for text: {text[:80]!r}")

        audio = np.concatenate(chunks).astype(np.float32)
        return torch.from_numpy(audio).unsqueeze(0)

    def close(self) -> None:
        """Release model and free GPU memory."""
        if getattr(self, "model", None) is not None:
            del self.model
            self.model = None
        self._runtime = None
        self._audio_tokenizer = None

        if self._ref_audio_24k:
            with contextlib.suppress(OSError):
                os.unlink(self._ref_audio_24k)
            self._ref_audio_24k = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def provider_info(cls) -> ProviderInfo:
        """Return Breeze provider metadata."""
        return ProviderInfo(
            name="breeze",
            supports_voice_cloning=True,
            supported_languages=["English", "Chinese"],
            builtin_voices=[
                VoiceInfo(id="design", name="Voice Design (instruction)", language="English"),
            ],
        )

    @property
    def sample_rate(self) -> int:
        """Get the sample rate for Breeze TTS."""
        runtime = getattr(self, "_runtime", None)
        return runtime.sample_rate if runtime is not None else SAMPLE_RATE
