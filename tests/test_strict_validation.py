from unittest.mock import Mock

import pytest

from tests.test_pipeline import FakeTTS
from rho_tts.exceptions import ValidationError
from rho_tts.validation import classifier
from rho_tts.validation.stt import stt_validator


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    monkeypatch.setattr(stt_validator, 'transcribe_audio', lambda _: 'Hello world')


def strict_tts():
    tts = FakeTTS()
    tts.strict_validation = True
    tts.voice_cloning = True
    tts.max_iterations = 2
    return tts


def test_failed_transcription_cannot_pass(monkeypatch):
    monkeypatch.setattr(stt_validator, 'transcribe_audio', lambda _: None)
    assert stt_validator.validate_audio_text_match('audio.wav', 'Hello')[0] is False


def test_missing_classifier_stops_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(classifier, 'predict_accent_drift_probability', lambda *a, **k: None)
    with pytest.raises(ValidationError, match='Accent.*unavailable'):
        strict_tts().generate('Hello world', output_path=str(tmp_path / 'out.wav'))
    assert not (tmp_path / 'out.wav').exists()


def test_failed_transcription_stops_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(classifier, 'predict_accent_drift_probability', lambda *a, **k: 0.01)
    monkeypatch.setattr(stt_validator, 'transcribe_audio', lambda _: None)
    with pytest.raises(ValidationError, match='[Tt]ranscription.*unavailable'):
        strict_tts().generate('Hello world', output_path=str(tmp_path / 'out.wav'))
    assert not (tmp_path / 'out.wav').exists()


def test_exhausted_retries_never_publish_best_failed_audio(tmp_path):
    tts = strict_tts()
    tts._validate_accent_drift = Mock(return_value=(0.01, True))
    tts._validate_text_match = Mock(return_value=(False, 0.2, 'wrong words'))
    with pytest.raises(ValidationError, match='validation.*2 attempts'):
        tts.generate('Hello world', output_path=str(tmp_path / 'out.wav'))
    assert tts._validate_text_match.call_count == 2
    assert not (tmp_path / 'out.wav').exists()


def test_single_attempt_still_validates(tmp_path):
    tts = strict_tts()
    tts.max_iterations = 1
    tts._validate_accent_drift = Mock(return_value=(0.01, True))
    tts._validate_text_match = Mock(return_value=(True, 1.0, 'Hello world'))
    assert tts.generate('Hello world', output_path=str(tmp_path / 'out.wav'))
    tts._validate_text_match.assert_called_once()


def test_isolated_breeze_installs_validation_extras(monkeypatch, tmp_path):
    from rho_tts.isolation import venv_manager
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(venv_manager.subprocess, 'run', run)
    venv_manager.VenvManager('breeze', venvs_root=tmp_path)._install_package()
    assert any('[breeze,validation]' in str(arg) for arg in run.call_args.args[0])


def test_validation_error_survives_worker_protocol(tmp_path):
    import io
    import json
    from rho_tts.isolation.worker import Worker
    output = io.StringIO()
    worker = Worker(protocol_out=output)
    worker._tts = Mock()
    worker._tts.generate.side_effect = ValidationError('Transcription validation unavailable')
    worker._handle_generate({'text': 'Hello', 'output_path': str(tmp_path / 'out.wav')})
    response = json.loads(output.getvalue())
    assert response['type'] == 'error'
    assert 'validation unavailable' in response['message']


def test_failed_sound_decay_never_publishes(tmp_path):
    tts = strict_tts()
    tts.max_decay_retries = 1
    tts._validate_accent_drift = Mock(return_value=(0.01, True))
    tts._validate_text_match = Mock(return_value=(True, 1.0, 'Hello world'))
    tts._validate_sound_decay = Mock(return_value=(0.0, False))
    with pytest.raises(ValidationError, match='Sound decay validation failed'):
        tts.generate('Hello world', output_path=str(tmp_path / 'out.wav'))
    assert not (tmp_path / 'out.wav').exists()
