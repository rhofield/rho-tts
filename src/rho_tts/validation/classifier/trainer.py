"""
Voice quality classifier trainer.

Extracts acoustic features from audio samples and trains a classifier
to detect accent drift in TTS output. The trained model is used by
the classifier module for runtime validation.

Usage:
    python -m rho_tts.validation.classifier.trainer --dataset-dir /path/to/dataset

The dataset directory should contain 'good/' and 'bad/' subdirectories
with .wav files.
"""
import logging
import os
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def extract_features(path: str) -> Optional[np.ndarray]:
    """
    Extract speaker, acoustic, and prosodic features from one audio file.

    Features include:
    - Speaker embedding (256-dim) via resemblyzer
    - MFCC mean and std (13 each)
    - Pitch statistics (F0 mean, F0 std)
    - Formants (F1, F2)

    Args:
        path: Path to audio file

    Returns:
        Feature vector as numpy array, or None if extraction fails
    """
    try:
        import librosa
        from resemblyzer import preprocess_wav

        # Use module-level encoder to avoid reloading
        encoder = _get_encoder()

        wav = preprocess_wav(path)
        embed = encoder.embed_utterance(wav)

        y, sr = librosa.load(path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_voiced = f0[~np.isnan(f0)]
        f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        f0_std = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0

        f1, f2 = _estimate_formants(y, sr)

        return np.concatenate([
            embed,
            mfcc_mean, mfcc_std,
            [f0_mean, f0_std, f1, f2]
        ])
    except Exception as e:
        logger.error(f"Error processing {path}: {e}")
        return None


def _estimate_formants(y: np.ndarray, sr: int) -> tuple:
    """Estimate F1 and F2 using LPC analysis on a mid-file frame."""
    import librosa

    # Pre-emphasis to flatten spectral tilt before LPC
    y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # 25 ms Hann-windowed frame from the midpoint
    frame_len = int(0.025 * sr)
    center = len(y_pre) // 2
    frame = y_pre[max(0, center - frame_len // 2):center + frame_len // 2]
    frame = frame * np.hanning(len(frame))

    # LPC order: 1 pole-pair per kHz + 2 for glottal/lip radiation
    order = max(12, sr // 1000 + 2)
    A = librosa.lpc(frame, order=order)

    # Formants are roots of A with positive imaginary part (upper half-plane)
    roots = np.roots(A)
    roots = roots[roots.imag > 0]
    freqs = np.sort(np.angle(roots) * (sr / (2 * np.pi)))
    freqs = freqs[(freqs > 90) & (freqs < sr / 4)]

    f1 = float(freqs[0]) if len(freqs) > 0 else 0.0
    f2 = float(freqs[1]) if len(freqs) > 1 else 0.0
    return f1, f2


# Lazy-loaded voice encoder singleton
_encoder = None


def _get_encoder():
    """Get or create the voice encoder singleton."""
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
    return _encoder


def train(
    dataset_dir: str,
    voice_id: Optional[str] = None,
    output_path: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
):
    """
    Train the voice quality classifier.

    Args:
        dataset_dir: Directory containing 'good/' and 'bad/' subdirectories with .wav files
        voice_id: Voice ID to associate with this model. When given and output_path is None,
            the model is saved to ~/.rho_tts/models/{voice_id}_classifier.pkl.
        output_path: Explicit path to save the trained model. When both voice_id and
            output_path are None, defaults to voice_quality_model.pkl in the classifier
            package directory (legacy global model).
        progress_callback: Optional callable receiving progress messages as training proceeds.
    """
    from datetime import datetime

    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import train_test_split

    if output_path is None:
        if voice_id is not None:
            models_dir = os.path.join(os.path.expanduser("~"), ".rho_tts", "models")
            os.makedirs(models_dir, exist_ok=True)
            output_path = os.path.join(models_dir, f"{voice_id}_classifier.pkl")
        else:
            output_path = os.path.join(os.path.dirname(__file__), "voice_quality_model.pkl")

    logger.info("Voice Quality Classifier Training")

    # Pre-scan both folders to get total file count for progress reporting
    PROGRESS_INTERVAL = 10
    all_wav: list[tuple[str, str]] = []
    for folder in ["good", "bad"]:
        fp = os.path.join(dataset_dir, folder)
        if os.path.exists(fp):
            all_wav.extend((folder, f) for f in sorted(os.listdir(fp)) if f.endswith(".wav"))
    total_files = len(all_wav)

    # Load data
    X, y = [], []
    file_index = 0
    for label, folder in enumerate(["good", "bad"]):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Dataset folder not found: {folder_path}")

        logger.info(f"Loading from: {folder_path}")
        wav_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".wav")]
        count = 0
        for i, f in enumerate(wav_files):
            file_index += 1
            path = os.path.join(folder_path, f)
            feat = extract_features(path)
            if feat is not None:
                X.append(feat)
                y.append(label)
                count += 1
            is_last = (i == len(wav_files) - 1)
            pct = file_index * 100 // total_files if total_files else 0
            if file_index % PROGRESS_INTERVAL == 0 or is_last:
                msg = f"Extracting: {file_index}/{total_files} ({pct}%) — {folder}/{f}"
                logger.info(msg)
                if progress_callback:
                    progress_callback(msg)

        summary = f"Loaded {count} files from {folder}/"
        logger.info(summary)
        if progress_callback:
            progress_callback(summary)

    X, y = np.array(X), np.array(y)
    n_good, n_bad = int(np.sum(y == 0)), int(np.sum(y == 1))
    logger.info(f"Loaded {len(X)} samples ({n_good} good, {n_bad} bad)")
    if progress_callback:
        progress_callback(f"Loaded {len(X)} samples ({n_good} good, {n_bad} bad)")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train/test split: {len(X_train)} train, {len(X_test)} test")

    # Train with cost-sensitive class weights
    n_good, n_bad = np.sum(y_train == 0), np.sum(y_train == 1)
    total = len(y_train)
    fn_cost = 5.0
    fp_cost = 1.0

    weight_good = (total / (2 * n_good)) * fn_cost
    weight_bad = (total / (2 * n_bad)) * fp_cost
    class_weights = {0: weight_good, 1: weight_bad}

    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features='sqrt',
        random_state=42,
        class_weight=class_weights,
    )

    model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    if progress_callback:
        progress_callback("Training model (this may take a moment)...")
    model.fit(X_train, y_train)
    logger.info("Training completed!")
    if progress_callback:
        progress_callback("Training complete! Optimizing threshold...")

    # Find optimal threshold
    probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.01, 1.0, 0.01)

    best_cost = float('inf')
    optimal_threshold = 0.18

    for thresh in thresholds:
        y_pred = (probs >= thresh).astype(int)
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        cost = (fn * fn_cost + fp * fp_cost) / len(y_test)
        if cost < best_cost:
            best_cost = cost
            optimal_threshold = thresh

    brier = brier_score_loss(y_test, probs)
    if progress_callback:
        progress_callback(f"Optimal threshold: {optimal_threshold:.3f}")

    # Save model
    model_metadata = {
        'model': model,
        'model_name': 'RandomForest',
        'optimal_threshold': optimal_threshold,
        'fn_cost': fn_cost,
        'fp_cost': fp_cost,
        'training_date': datetime.now().isoformat(),
        'class_distribution': {'good': int(np.sum(y == 0)), 'bad': int(np.sum(y == 1))},
        'expected_cost': best_cost,
        'brier_score': brier,
    }
    joblib.dump(model_metadata, output_path)
    logger.info(f"Model saved to {output_path} (threshold: {optimal_threshold:.3f}, brier: {brier:.4f})")
    if progress_callback:
        progress_callback(f"Model saved to {output_path} (threshold: {optimal_threshold:.3f}, brier: {brier:.4f})")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train voice quality classifier")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory with good/ and bad/ folders")
    parser.add_argument("--voice-id", default=None, help="Voice ID for per-voice model (saved to ~/.rho_tts/models/)")
    parser.add_argument("--output", default=None, help="Explicit output path for trained model")
    args = parser.parse_args()

    train(args.dataset_dir, voice_id=args.voice_id, output_path=args.output)
