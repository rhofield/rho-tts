"""Manual integration tests for drift detection and smart segmentation.

These tests require a GPU, real models, and training data.
Gate behind RHO_TTS_INTEGRATION=1 environment variable.
"""

import os

import pytest

requires_integration = pytest.mark.skipif(
    os.environ.get("RHO_TTS_INTEGRATION") != "1",
    reason="Set RHO_TTS_INTEGRATION=1 to run integration tests",
)


@requires_integration
def test_train_and_use_drift_classifier(tmp_path):
    """Train a classifier on sample data and verify it returns predictions."""
    from rho_tts import train_drift_classifier
    from rho_tts.validation.classifier import predict_accent_drift_probability

    good_dir = tmp_path / "dataset" / "good"
    bad_dir = tmp_path / "dataset" / "bad"
    good_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)

    # Generate synthetic wav files for training
    import torch
    import torchaudio

    sr = 16000
    for i in range(5):
        # "good" = clean sine wave
        t = torch.linspace(0, 1, sr)
        good_wav = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
        torchaudio.save(str(good_dir / f"good_{i}.wav"), good_wav, sr)

        # "bad" = noisy
        bad_wav = torch.randn(1, sr) * 0.5
        torchaudio.save(str(bad_dir / f"bad_{i}.wav"), bad_wav, sr)

    output_path = str(tmp_path / "test_model.pkl")
    train_drift_classifier(
        dataset_dir=str(tmp_path / "dataset"),
        output_path=output_path,
    )

    assert os.path.exists(output_path)

    # Use the trained model
    test_wav = good_dir / "good_0.wav"
    prob = predict_accent_drift_probability(str(test_wav), model_path=output_path)
    assert prob is not None
    assert 0.0 <= prob <= 1.0


@requires_integration
def test_custom_pkl_path_end_to_end(tmp_path):
    """Verify drift_model_path parameter flows through to validation."""
    from rho_tts import TTSFactory

    tts = TTSFactory.get_tts_instance(
        provider="qwen",
        drift_model_path="/nonexistent/model.pkl",
    )
    assert tts.drift_model_path == "/nonexistent/model.pkl"


@requires_integration
def test_validation_during_generation(tmp_path):
    """End-to-end: train classifier, then verify generate() populates drift_prob.

    Exercises the full path:
      generate() → _run_pipeline() → _validate_accent_drift() → predict_accent_drift_probability()
    """
    import torch
    import torchaudio

    from rho_tts import TTSFactory, train_drift_classifier

    # --- 1. Generate synthetic training data ---
    good_dir = tmp_path / "dataset" / "good"
    bad_dir = tmp_path / "dataset" / "bad"
    good_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)

    sr = 16000
    for i in range(5):
        t = torch.linspace(0, 1, sr)
        good_wav = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
        torchaudio.save(str(good_dir / f"good_{i}.wav"), good_wav, sr)

        bad_wav = torch.randn(1, sr) * 0.5
        torchaudio.save(str(bad_dir / f"bad_{i}.wav"), bad_wav, sr)

    # --- 2. Train drift classifier ---
    model_path = str(tmp_path / "model.pkl")
    train_drift_classifier(
        dataset_dir=str(tmp_path / "dataset"),
        output_path=model_path,
    )
    assert os.path.exists(model_path)

    # --- 3. Create reference audio for voice cloning ---
    ref_audio_path = str(tmp_path / "reference.wav")
    ref_wav = torch.sin(2 * 3.14159 * 440 * torch.linspace(0, 3, sr * 3)).unsqueeze(0)
    torchaudio.save(ref_audio_path, ref_wav, sr)

    # --- 4. Create TTS with drift_model_path and voice cloning enabled ---
    tts = TTSFactory.get_tts_instance(
        provider="qwen",
        reference_audio=ref_audio_path,
        reference_text="Test reference audio.",
        drift_model_path=model_path,
        max_iterations=3,
    )

    # --- 5. Generate and verify drift_prob is populated ---
    result = tts.generate("Hello, this is a test.")
    assert result is not None
    assert result.drift_prob is not None, (
        "drift_prob should be populated when validation runs during generation"
    )
    assert 0.0 <= result.drift_prob <= 1.0


@requires_integration
def test_smart_segmentation_with_real_provider():
    """Verify smart segmentation computes a reasonable value with real provider."""
    from rho_tts import TTSFactory

    tts = TTSFactory.get_tts_instance(provider="qwen")
    max_chars = tts._compute_max_chars()

    # Should be between floor (200) and MAX_MODEL_CHARS (4000)
    assert 200 <= max_chars <= 4000
