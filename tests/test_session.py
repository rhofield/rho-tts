"""Tests for rho_tts.ui.session.SessionContext per-session isolation."""

import copy
import os
import pytest

from rho_tts.ui.config import AppConfig, GenerationRecord, ModelConfig, VoiceProfile
from rho_tts.ui.session import SessionContext


def _make_record(record_id="rec-1"):
    return GenerationRecord(
        id=record_id,
        timestamp=1000.0,
        audio_path="/tmp/audio.wav",
        text="Hello",
        model_id="m1",
        model_name="Model",
        voice_id="v1",
        voice_name="Voice",
        provider="chatterbox",
    )


def _make_config():
    """Build a config with one voice and one model for isolation tests."""
    cfg = AppConfig()
    cfg.voices["v1"] = VoiceProfile(id="v1", name="Voice One")
    cfg.models["m1"] = ModelConfig(id="m1", name="Model One", provider="qwen", params={"max_iterations": 10})
    cfg.phonetic_mappings["v1::m1"] = {"hello": "HH EH L OW"}
    cfg.model_voice_params["v1::m1"] = {"seed": 42}
    return cfg


class TestSessionContextInit:
    def test_initial_state(self):
        ctx = SessionContext(session_id="abc123")
        assert ctx.session_id == "abc123"
        assert ctx.config is None
        assert ctx.cancellation_token is None
        assert ctx.history == []
        assert ctx._output_dir is None

    def test_initial_state_with_config(self):
        cfg = AppConfig()
        ctx = SessionContext(session_id="abc123", config=cfg)
        assert ctx.config is cfg


class TestCancellation:
    def test_new_cancellation_token(self):
        ctx = SessionContext(session_id="abc")
        token = ctx.new_cancellation_token()
        assert token is ctx.cancellation_token
        assert not token.is_cancelled()

    def test_new_token_replaces_old(self):
        ctx = SessionContext(session_id="abc")
        old = ctx.new_cancellation_token()
        new = ctx.new_cancellation_token()
        assert new is not old
        assert ctx.cancellation_token is new

    def test_cancel_generation(self):
        ctx = SessionContext(session_id="abc")
        token = ctx.new_cancellation_token()
        ctx.cancel_generation()
        assert token.is_cancelled()

    def test_cancel_generation_noop_when_no_token(self):
        ctx = SessionContext(session_id="abc")
        ctx.cancel_generation()  # should not raise


class TestHistory:
    def test_add_generation_record(self):
        ctx = SessionContext(session_id="abc")
        record = _make_record("r1")
        ctx.add_generation_record(record)
        assert record in ctx.history
        assert len(ctx.history) == 1

    def test_delete_generation_record_found(self):
        ctx = SessionContext(session_id="abc")
        record = _make_record("r1")
        ctx.add_generation_record(record)
        assert ctx.delete_generation_record("r1") is True
        assert len(ctx.history) == 0

    def test_delete_generation_record_not_found(self):
        ctx = SessionContext(session_id="abc")
        ctx.add_generation_record(_make_record("r1"))
        assert ctx.delete_generation_record("nonexistent") is False
        assert len(ctx.history) == 1

    def test_clear_history(self):
        ctx = SessionContext(session_id="abc")
        ctx.add_generation_record(_make_record("r1"))
        ctx.add_generation_record(_make_record("r2"))
        count = ctx.clear_history()
        assert count == 2
        assert ctx.history == []

    def test_clear_empty_history(self):
        ctx = SessionContext(session_id="abc")
        count = ctx.clear_history()
        assert count == 0


class TestOutputDir:
    def test_output_dir_lazy_creates(self):
        ctx = SessionContext(session_id="abcdef1234567890")
        assert ctx._output_dir is None
        out = ctx.output_dir
        assert out is not None
        assert os.path.isdir(out)
        assert "rho_abcdef12_" in out
        # Cleanup
        ctx.cleanup()

    def test_output_dir_stable(self):
        ctx = SessionContext(session_id="abc")
        first = ctx.output_dir
        second = ctx.output_dir
        assert first == second
        ctx.cleanup()

    def test_cleanup_removes_dir(self):
        ctx = SessionContext(session_id="abc")
        out = ctx.output_dir
        assert os.path.isdir(out)
        ctx.cleanup()
        assert not os.path.exists(out)
        assert ctx._output_dir is None

    def test_cleanup_noop_when_no_dir(self):
        ctx = SessionContext(session_id="abc")
        ctx.cleanup()  # should not raise

    def test_cleanup_idempotent(self):
        ctx = SessionContext(session_id="abc")
        _ = ctx.output_dir
        ctx.cleanup()
        ctx.cleanup()  # second call should not raise


class TestMultipleSessionsIsolated:
    def test_separate_histories(self):
        a = SessionContext(session_id="aaa")
        b = SessionContext(session_id="bbb")
        a.add_generation_record(_make_record("r1"))
        b.add_generation_record(_make_record("r2"))
        assert len(a.history) == 1
        assert len(b.history) == 1
        assert a.history[0].id == "r1"
        assert b.history[0].id == "r2"

    def test_separate_cancellation(self):
        a = SessionContext(session_id="aaa")
        b = SessionContext(session_id="bbb")
        token_a = a.new_cancellation_token()
        token_b = b.new_cancellation_token()
        a.cancel_generation()
        assert token_a.is_cancelled()
        assert not token_b.is_cancelled()

    def test_separate_output_dirs(self):
        a = SessionContext(session_id="aaa")
        b = SessionContext(session_id="bbb")
        assert a.output_dir != b.output_dir
        a.cleanup()
        b.cleanup()


class TestConfigIsolation:
    """Test that session configs are independent deep copies."""

    def test_session_gets_independent_config(self):
        original = _make_config()
        session = SessionContext(session_id="abc", config=copy.deepcopy(original))

        # Mutate session config
        session.config.voices["v2"] = VoiceProfile(id="v2", name="Voice Two")

        # Original unaffected
        assert "v2" not in original.voices
        assert "v2" in session.config.voices

    def test_voice_add_on_session_does_not_affect_shared(self):
        shared = _make_config()
        session = SessionContext(session_id="abc", config=copy.deepcopy(shared))

        session.config.voices["new"] = VoiceProfile(id="new", name="New Voice")
        assert "new" not in shared.voices
        assert len(shared.voices) == 1
        assert len(session.config.voices) == 2

    def test_voice_delete_on_session_does_not_affect_shared(self):
        shared = _make_config()
        session = SessionContext(session_id="abc", config=copy.deepcopy(shared))

        del session.config.voices["v1"]
        assert "v1" in shared.voices
        assert "v1" not in session.config.voices

    def test_model_add_on_session_does_not_affect_shared(self):
        shared = _make_config()
        session = SessionContext(session_id="abc", config=copy.deepcopy(shared))

        session.config.models["m2"] = ModelConfig(
            id="m2", name="Model Two", provider="chatterbox",
        )
        assert "m2" not in shared.models
        assert "m2" in session.config.models

    def test_model_delete_on_session_does_not_affect_shared(self):
        shared = _make_config()
        session = SessionContext(session_id="abc", config=copy.deepcopy(shared))

        del session.config.models["m1"]
        assert "m1" in shared.models
        assert "m1" not in session.config.models

    def test_phonetic_mapping_isolated(self):
        shared = _make_config()
        session = SessionContext(session_id="abc", config=copy.deepcopy(shared))

        session.config.phonetic_mappings["v1::m1"]["world"] = "W ER L D"
        assert "world" not in shared.phonetic_mappings["v1::m1"]

    def test_model_voice_params_isolated(self):
        shared = _make_config()
        session = SessionContext(session_id="abc", config=copy.deepcopy(shared))

        session.config.model_voice_params["v1::m1"]["seed"] = 999
        assert shared.model_voice_params["v1::m1"]["seed"] == 42

    def test_save_is_noop(self):
        session = SessionContext(session_id="abc", config=AppConfig())
        session.save()  # should not raise, should not write to disk

    def test_two_sessions_fully_independent(self):
        shared = _make_config()
        a = SessionContext(session_id="aaa", config=copy.deepcopy(shared))
        b = SessionContext(session_id="bbb", config=copy.deepcopy(shared))

        a.config.voices["va"] = VoiceProfile(id="va", name="Voice A")
        b.config.models["mb"] = ModelConfig(id="mb", name="Model B", provider="qwen")

        # Each session only sees its own changes
        assert "va" in a.config.voices
        assert "va" not in b.config.voices
        assert "mb" in b.config.models
        assert "mb" not in a.config.models
        # Shared config unchanged
        assert len(shared.voices) == 1
        assert len(shared.models) == 1
