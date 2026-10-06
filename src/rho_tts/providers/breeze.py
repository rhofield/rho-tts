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
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from ..base_tts import BaseTTS
from ..cancellation import CancellationToken
from ..provider_info import ProviderInfo, VoiceInfo
from ..result import GenerationResult

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
        attn_implementation: Attention backend. Defaults to PyTorch "sdpa",
            which uses fused attention kernels on supported GPUs without the
            optional flash-attn package. Set to "eager" for comparison or if
            the device does not support SDPA.
        fast: Enable upstream's compiled fast path. Off by default: it needs a
            torch.compile/CUDA-graph warmup that does not complete on smaller
            GPUs (observed to hang on an RTX 3060).
        compile_depth: Compile depth-decoder layers with torch.compile while
            retaining the eager runtime. First use pays a compilation cost;
            useful for long-lived workers serving repeated requests.
        cuda_graph_depth: Compile and capture only the depth-decoder loop as a
            CUDA Graph. First use pays compilation and graph-capture costs;
            useful for long-lived workers with repeated decoder frames.
        triton_depth_sampling: Use a fused Triton top-50 sampler in the eager
            depth loop. Requires CUDA and Triton; changes seeded token output.
        codec_chunk_frames: Codec frames decoded per streamed chunk (1–16).
            The default is 2; smaller values reduce first-audio latency but
            may increase total generation time.
        temperature: Sampling temperature for the backbone and depth decoder.
            None keeps upstream's 0.9. Lower values make separately generated
            sentences vary less in pitch and delivery, at some cost in liveliness.
        segment_level_db: Active-speech level (dBFS) each sentence is scaled to
            before validation and joining, so separately generated sentences
            match in loudness. None leaves output as generated.
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
        attn_implementation: str = "sdpa",
        fast: bool = False,
        strict_validation: bool = False,
        max_chars_per_segment: Optional[int] = None,
        max_iterations: int = 10,
        accent_drift_threshold: float = 0.17,
        text_similarity_threshold: float = 0.85,
        drift_model_path: Optional[str] = None,
        phonetic_mapping: Optional[Dict[str, str]] = None,
        compile_depth: bool = False,
        cuda_graph_depth: bool = False,
        triton_depth_sampling: bool = False,
        codec_chunk_frames: int = 2,
        temperature: Optional[float] = None,
        segment_level_db: Optional[float] = -23.0,
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
        if sum((fast, compile_depth, cuda_graph_depth)) > 1:
            raise ValueError("fast, compile_depth, and cuda_graph_depth are mutually exclusive")
        if triton_depth_sampling and (fast or cuda_graph_depth or not str(device).startswith("cuda")):
            raise ValueError("triton_depth_sampling requires eager CUDA depth decoding")
        if (
            isinstance(codec_chunk_frames, bool)
            or not isinstance(codec_chunk_frames, int)
            or not 1 <= codec_chunk_frames <= 16
        ):
            raise ValueError("codec_chunk_frames must be an integer from 1 to 16")
        if fast and codec_chunk_frames not in (1, 2):
            raise ValueError("codec_chunk_frames cannot override the upstream fast codec")
        if temperature is not None and not (math.isfinite(temperature) and 0 < temperature <= 2):
            raise ValueError(f"temperature must be in (0, 2], got {temperature}")
        if segment_level_db is not None and not (math.isfinite(segment_level_db) and -60 <= segment_level_db < 0):
            raise ValueError(f"segment_level_db must be in [-60, 0), got {segment_level_db}")

        self.reference_audio_path = reference_audio
        self.reference_text = reference_text
        self.voice_cloning = reference_audio is not None
        self.instruction = instruction if instruction is not None else NEUTRAL_INSTRUCTION
        self.cfg_scale = cfg_scale
        self.temperature = temperature
        self.segment_level_db = segment_level_db
        self.attn_implementation = attn_implementation
        self.fast = fast
        self.compile_depth = compile_depth
        self.cuda_graph_depth = cuda_graph_depth
        self.triton_depth_sampling = triton_depth_sampling
        self.codec_chunk_frames = 1 if fast else codec_chunk_frames
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
        self.reference_resample_device = "none"
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

        from .breeze_sampling import install_eager_topk_sampler

        class ProviderRuntime(FastBreezeStreamingRuntime):
            def _ensure_graphs(self, *args, **kwargs):
                super()._ensure_graphs(*args, **kwargs)
                if not self._fast_depth_decoder:
                    install_eager_topk_sampler(
                        self._depth_decoder_graph,
                        use_triton=provider.triton_depth_sampling,
                    )
                    if provider.compile_depth:
                        provider._compile_depth_decoder(self._depth_decoder_graph)
                # Graph capture/warmup may sample. Start the actual request at
                # its recorded attempt seed whether graphs are fresh or reused.
                provider._set_seeds()

        provider = self
        self._runtime = ProviderRuntime(
            self.model,
            self._audio_tokenizer,
            FastStreamingConfig(
                max_new_tokens=MAX_NEW_TOKENS,
                max_seq_len=MAX_SEQ_LEN,
                fast_all=True if self.fast else None,
                fast_depth_decoder=self.cuda_graph_depth,
                repetition_penalty=REPETITION_PENALTY,
            ),
            tokenizer=self._tokenizer,
        )
        if not self.fast:
            self._runtime._codec_chunk_frames = self.codec_chunk_frames
        if self.fast and self._runtime.fast_enabled:
            self._warmup()

        if self.voice_cloning:
            self._ref_audio_24k = self._prepare_reference(self.reference_audio_path)

    @staticmethod
    def _compile_depth_decoder(graph) -> None:
        """Compile the repeated depth layers once, without CUDA Graph capture."""
        if getattr(graph, "_rho_depth_compiled", False):
            return
        limit = graph.num_layers * 4 + 16  # separate layer-index guards
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, limit)
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, limit)
        for index, layer in enumerate(graph.depth_model.layers):
            graph.depth_model.layers[index] = torch.compile(layer, mode="default", fullgraph=True)
        graph.depth_model.norm = torch.compile(graph.depth_model.norm, mode="default", fullgraph=True)
        graph.codebooks_head = torch.compile(graph.codebooks_head, mode="default", fullgraph=True)
        graph._rho_depth_compiled = True

    def _apply_generation_config(self) -> None:
        """Apply upstream's generation defaults to the model and depth decoder."""
        from breeze_infer.runtime import update_generation_config_for_breeze

        update_generation_config_for_breeze(self.model)
        if self.temperature is not None:
            # Both configs, so the replay manifest records what was sampled with.
            self.model.generation_config.temperature = self.temperature
            self.model.depth_decoder.generation_config.temperature = self.temperature

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

        from .breeze_schedule import choose_device, reference_resample_costs

        data, sr = sf.read(src, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data).T
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            gpu_name = None
            gpu_free_bytes = 0
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(self.device)
                gpu_free_bytes, _ = torch.cuda.mem_get_info(self.device)
            costs = reference_resample_costs(
                source_rate=sr,
                target_rate=SAMPLE_RATE,
                samples=wav.numel(),
                gpu_name=gpu_name,
                gpu_free_bytes=gpu_free_bytes,
            )
            selected = choose_device(costs)
            if selected == "cuda":
                try:
                    wav = torchaudio.functional.resample(
                        wav.to(self.device), sr, SAMPLE_RATE
                    ).cpu()
                except torch.cuda.OutOfMemoryError:
                    logger.warning("Breeze GPU reference resampling ran out of memory; using CPU")
                    selected = "cpu"
                    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            else:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            self.reference_resample_device = selected
            logger.info(
                "Breeze reference resample on %s (predicted %.2f ms)",
                selected,
                costs[selected].predicted_ms,
            )

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

        chunks = [chunk.audio for chunk in self._iter_codec_chunks(text, **kwargs)]
        if not chunks:
            raise RuntimeError(f"Breeze produced no audio for text: {text[:80]!r}")

        audio = np.concatenate(chunks).astype(np.float32)
        return torch.from_numpy(audio).unsqueeze(0)

    def _iter_codec_chunks(self, text: str, **kwargs):
        """Prepare one request and yield each decoded codec chunk immediately."""
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
        yield from self._runtime.iter_audio_chunks(inputs, request_id=request["id"])

    def stream(
        self,
        text: str,
        cancellation_token: Optional[CancellationToken] = None,
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
    ):
        """Yield decoded audio chunks within each text segment as they arrive.

        Boundary trimming and fades require the complete utterance, so this
        low-latency API leaves the codec waveform intact. Speed/pitch adjustment
        is applied per emitted chunk when requested.
        """
        token = cancellation_token or CancellationToken()
        mapped_text = self._apply_phonetic_mapping(text)
        segments = self._split_text_into_segments(mapped_text, self._compute_max_chars())
        for segment in segments:
            if token.is_cancelled():
                return
            iterator = self._iter_codec_chunks(segment)
            try:
                for chunk in iterator:
                    if token.is_cancelled():
                        return
                    audio = torch.from_numpy(chunk.audio)
                    if speed != 1.0 or pitch_semitones != 0.0:
                        audio = self._apply_speed_pitch(audio, speed, pitch_semitones)
                    yield GenerationResult(
                        audio=audio,
                        sample_rate=self.sample_rate,
                        duration_sec=audio.numel() / self.sample_rate,
                        segments_count=1,
                        format="wav",
                    )
            finally:
                iterator.close()

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
