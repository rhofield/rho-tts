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
from typing import Optional

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
        import parselmouth
        from resemblyzer import preprocess_wav

        # Use module-level encoder to avoid reloading
        encoder = _get_encoder()

        wav = preprocess_wav(path)
        embed = encoder.embed_utterance(wav)

        y, sr = librosa.load(path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        snd = parselmouth.Sound(path)
        pitch = snd.to_pitch()
        f0 = pitch.selected_array["frequency"]
        f0_mean = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0
        f0_std = np.std(f0[f0 > 0]) if np.any(f0 > 0) else 0

        formants = snd.to_formant_burg()
        f1 = formants.get_value_at_time(1, 0.5)
        f2 = formants.get_value_at_time(2, 0.5)

        return np.concatenate([
            embed,
            mfcc_mean, mfcc_std,
            [f0_mean, f0_std, f1 or 0, f2 or 0]
        ])
    except Exception as e:
        logger.error(f"Error processing {path}: {e}")
        return None


# Lazy-loaded voice encoder singleton
_encoder = None


def _get_encoder():
    """Get or create the voice encoder singleton."""
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        _encoder = VoiceEncoder()
    return _encoder


def train(dataset_dir: str, output_path: Optional[str] = None):
    """
    Train the voice quality classifier.

    Args:
        dataset_dir: Directory containing 'good/' and 'bad/' subdirectories with .wav files
        output_path: Path to save the trained model. Defaults to voice_quality_model.pkl
            in the classifier package directory.
    """
    from datetime import datetime

    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import train_test_split

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "voice_quality_model.pkl")

    logger.info("Voice Quality Classifier Training")

    # Load data
    X, y = [], []
    for label, folder in enumerate(["good", "bad"]):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Dataset folder not found: {folder_path}")

        logger.info(f"Loading from: {folder_path}")
        for f in sorted(os.listdir(folder_path)):
            if not f.endswith(".wav"):
                continue
            path = os.path.join(folder_path, f)
            feat = extract_features(path)
            if feat is not None:
                X.append(feat)
                y.append(label)

    X, y = np.array(X), np.array(y)
    logger.info(f"Loaded {len(X)} samples ({np.sum(y == 0)} good, {np.sum(y == 1)} bad)")

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
    model.fit(X_train, y_train)
    logger.info("Training completed!")

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


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train voice quality classifier")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory with good/ and bad/ folders")
    parser.add_argument("--output", default=None, help="Output path for trained model")
    args = parser.parse_args()

    train(args.dataset_dir, args.output)
