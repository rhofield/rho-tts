from tests.test_continuity import TestContinuity as ContinuityHarness, candidates
from rho_tts import ContinuityConfig


class TestAcceptance(ContinuityHarness):
    def test_text_failure_cannot_be_accepted_by_continuity(self, tmp_path):
        tts = self.tts
        candidates(tts, [.1, .1, .1])
        tts._validate_text_match.return_value = (False, .2, 'wrong')
        result = tts.generate('Hello', str(tmp_path / 'out.wav'), continuity=ContinuityConfig())
        assert result.continuity['passed']
        assert result.acceptance['accepted'] is False
        selected = result.acceptance['segments'][0]
        assert selected['validators']['text']['status'] == 'fail'
        assert selected['selected_attempt'] == 1
        assert selected['fallback_reasons'] == ['text failed', 'retries exhausted']
        assert len(selected['attempts']) == 3
        assert all(__import__('pathlib').Path(a['raw_audio']).is_file() for a in selected['attempts'])

    def test_missing_classifier_is_unavailable(self):
        candidates(self.tts, [.1])
        self.tts._validate_accent_drift.return_value = (None, True)
        result = self.tts.generate('Hello', continuity=ContinuityConfig())
        assert not result.acceptance['accepted']
        assert result.acceptance['segments'][0]['validators']['accent']['status'] == 'unavailable'

    def test_actual_missing_classifier_is_reported(self, monkeypatch):
        from rho_tts.base_tts import BaseTTS
        from rho_tts.validation import classifier
        self.tts.voice_cloning = True
        self.tts._validate_accent_drift = BaseTTS._validate_accent_drift.__get__(self.tts)
        monkeypatch.setattr(classifier, 'predict_accent_drift_probability', lambda *a, **kw: None)
        candidates(self.tts, [.1])
        result = self.tts.generate('Hello', continuity=ContinuityConfig())
        assert result.acceptance['segments'][0]['validators']['accent']['status'] == 'unavailable'
        assert result.drift_prob is None

    def test_general_fallback_metadata_matches_selected_audio(self):
        candidates(self.tts, [.1, .2, .3])
        self.tts._validate_accent_drift.side_effect = [(.1, True), (.02, True), (.08, True)]
        self.tts._validate_text_match.side_effect = [(False, .3, 'wrong'), (False, .2, 'wrong'), (False, .4, 'wrong')]
        result = self.tts.generate('Hello')
        assert result.acceptance['segments'][0]['selected_attempt'] == 2
        assert result.text_similarity == .2
        assert result.audio.abs().max().item() == __import__('pytest').approx(.2, abs=.001)

    def test_strict_exhaustion_retains_rejections(self, tmp_path):
        from rho_tts import ValidationError
        from pathlib import Path
        import pytest
        candidates(self.tts, [.1, .1, .1])
        self.tts.strict_validation = True
        self.tts._validate_text_match.return_value = (False, .2, 'wrong')
        with pytest.raises(ValidationError) as error:
            self.tts.generate('Hello', str(tmp_path / 'out.wav'), continuity=ContinuityConfig())
        evidence = error.value.acceptance
        assert evidence['accepted'] is False
        assert len(evidence['segments'][0]['attempts']) == 3
        assert all(Path(a['raw_audio']).is_file() for a in evidence['segments'][0]['attempts'])
        assert not (tmp_path / 'out.wav').exists()

    def test_decay_retries_keep_all_rounds(self):
        candidates(self.tts, [.1, .1])
        self.tts._validate_sound_decay.side_effect = [(.1, False), (1., True)]
        result = self.tts.generate('Hello', continuity=ContinuityConfig())
        assert result.acceptance['selected_round'] == 2
        assert [r['sound_decay']['status'] for r in result.acceptance['rounds']] == ['fail', 'pass']
        assert result.acceptance['accepted']

    def test_vcs_caller_provisions_worker_from_same_commit(self, monkeypatch, tmp_path):
        from rho_tts.isolation import venv_manager
        from unittest.mock import Mock
        from types import SimpleNamespace
        import json
        monkeypatch.setattr(venv_manager, '_find_project_root', lambda: None)
        monkeypatch.setattr(venv_manager, 'distribution', lambda name: SimpleNamespace(read_text=lambda name: json.dumps(
            dict(url='ssh://git@github.com/rhofield/rho-tts.git', vcs_info=dict(vcs='git', commit_id='abc123')))))
        run = Mock(return_value=Mock(returncode=0))
        monkeypatch.setattr(venv_manager.subprocess, 'run', run)
        venv_manager.VenvManager('breeze', venvs_root=tmp_path)._install_package()
        assert run.call_args.args[0][-1] == 'rho-tts[breeze,validation] @ git+ssh://git@github.com/rhofield/rho-tts.git@abc123'
