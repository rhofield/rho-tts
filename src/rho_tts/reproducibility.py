"""Content-addressed replay manifests and scoped synthesis random streams."""
from contextvars import ContextVar
from typing import Any
from dataclasses import asdict
from functools import wraps
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import threading
import platform
import inspect
import importlib.util

import numpy as np
import torch

_active_seed: ContextVar[int | None] = ContextVar('rho_tts_attempt_seed', default=None)
_active_job: ContextVar[dict[str, Any] | None] = ContextVar('rho_tts_job', default=None)
_rng_lock = threading.RLock()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_identity(path):
    if path is None:
        return None
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None)


def package_identity():
    from . import __version__
    root = Path(__file__).parent
    return dict(version=__version__, source_hash=fingerprint({
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*.py'))}))


def generation_manifest(provider, texts, *, job_id=None, job_seed=None, continuity=None,
                        format='wav', speed=1.0, pitch_semitones=0.0):
    texts = [texts] if isinstance(texts, str) else list(texts)
    # Only resolved, serializable generation settings; runtime objects never enter identity.
    names = ('seed', 'device', 'deterministic', 'phonetic_mapping', 'max_iterations', 'max_decay_retries',
             'strict_validation', 'accent_drift_threshold', 'text_similarity_threshold', 'sound_decay_threshold',
             'voice_id', 'reference_text', 'attn_implementation', 'fast', 'compile_depth', 'cuda_graph_depth',
             'triton_depth_sampling', 'codec_chunk_frames', 'cfg_scale', 'instruction', 'temperature', 'top_p',
             'top_k', 'max_chars_per_segment', '_max_chars_explicit', '_max_model_chars', 'force_sentence_split',
             'silence_threshold_db', 'trim_silence', 'crossfade_duration_sec', 'fade_duration_sec',
             'inter_sentence_pause_sec', 'reference_resample_device', 'segment_level_db')
    settings = {name: getattr(provider, name, None) for name in names}
    settings['sample_rate'] = provider.sample_rate
    settings['resolved_max_chars'] = provider._compute_max_chars()
    model = getattr(provider, 'model', None)
    config = getattr(model, 'config', None)
    generation = getattr(model, 'generation_config', None)
    provider_module = __import__(type(provider).__module__, fromlist=['MODEL_REPO'])
    model_identity = dict(repo=getattr(provider_module, 'MODEL_REPO', None),
                          revision=getattr(provider_module, 'MODEL_REVISION', None),
                          config=config.to_dict() if config is not None else None,
                          decoding=generation.to_dict() if generation is not None else None,
                          depth_decoding=(model.depth_decoder.generation_config.to_dict()
                              if model is not None and hasattr(model, 'depth_decoder') else None))
    constructor: dict[str, Any] = {}
    for name in inspect.signature(type(provider).__init__).parameters:
        if name == 'max_chars_per_segment' and not getattr(provider, '_max_chars_explicit', False):
            constructor[name] = None
        elif name == 'reference_audio':
            constructor[name] = getattr(provider, 'reference_audio_path', None)
        elif name in settings:
            constructor[name] = settings[name]
        elif name == 'drift_model_path':
            constructor[name] = getattr(provider, name, None)
    dependencies: dict[str, str | None] = {}
    for name in ('torch', 'torchaudio', 'numpy', 'transformers', 'breeze', 'qwen-tts',
                 'resemblyzer', 'faster-whisper', 'scikit-learn', 'librosa', 'soundfile'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    upstream = {}
    for name in ('breeze_infer', 'models', 'resemblyzer'):
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.submodule_search_locations:
            directory = Path(next(iter(spec.submodule_search_locations)))
            upstream[name] = fingerprint({str(p.relative_to(directory)): file_identity(p)['sha256']
                for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix in ('.py', '.pt')})
    context = asdict(continuity) if continuity else None
    if context:
        context['previous_audio'] = file_identity(context['previous_audio'])
    classifier_path = getattr(provider, 'drift_model_path', None)
    if not classifier_path:
        from .validation import classifier
        voice_id = getattr(provider, 'voice_id', None)
        classifier_path = classifier.get_model_path(voice_id) if voice_id else os.environ.get('RHO_TTS_CLASSIFIER_MODEL', str(Path(classifier.__file__).with_name('voice_quality_model.pkl')))
    mapped = [provider._apply_phonetic_mapping(text) for text in texts]
    return dict(schema_version=1, provider=type(provider).__name__, initialization=constructor, package=package_identity(),
                profile=dict(voice_id=getattr(provider, "voice_id", None), transcript=settings["reference_text"],
                             encoder=dependencies["resemblyzer"], source_hashes=upstream),
                runtime=dict(python=platform.python_version(), platform=platform.platform(), packages=dependencies, torch_version=torch.__version__, cuda=torch.version.cuda,
                             device=str(provider.device), gpu=(torch.cuda.get_device_name() if torch.cuda.is_available() else None)),
                job=dict(id=job_id or fingerprint(texts), seed=provider.seed if job_seed is None else job_seed, texts=texts),
                settings=settings, model=model_identity,
                reference=file_identity(getattr(provider, 'reference_audio_path', None)),
                classifier=file_identity(classifier_path), continuity=context,
                segmentation=[provider._split_text_into_segments(t, settings['resolved_max_chars']) for t in mapped],
                delivery=dict(format=format, speed=speed, pitch_semitones=pitch_semitones),
                seed_scheme='sha256-job-item-segment-round-attempt-v1',
                equivalence='Matching seeds do not guarantee cross-runtime bitwise equivalence.')


def attempt_seed(root, job_id, item, segment, round_index, attempt):
    return int(fingerprint([root, job_id, item, segment, round_index, attempt])[:8], 16)


def scoped_generation(method):
    @wraps(method)
    def generate(self, texts, *args, job_id=None, job_seed=None, **kwargs):
        if job_seed is not None and (isinstance(job_seed, bool) or not isinstance(job_seed, int) or not 0 <= job_seed < 2**32):
            raise ValueError('job_seed must be an integer in [0, 2**32)')
        bound = inspect.signature(method).bind(self, texts, *args, **kwargs)
        bound.apply_defaults()
        job = dict(id=job_id or fingerprint([texts] if isinstance(texts, str) else texts),
                   seed=self.seed if job_seed is None else job_seed,
                   delivery={key: bound.arguments[key] for key in ('format', 'speed', 'pitch_semitones')})
        # Backends use process-global generators. Serialize and restore those generators;
        # the seed itself is context-local and never assigned to the shared provider.
        with _rng_lock, torch.random.fork_rng():
            python_state, numpy_state = random.getstate(), np.random.get_state()
            flags = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
            job_token = _active_job.set(job)
            seed_token = _active_seed.set(job['seed'])
            try:
                return method(self, texts, *args, **kwargs)
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags
                _active_seed.reset(seed_token)
                _active_job.reset(job_token)
    return generate
