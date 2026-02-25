"""
Voice cloning example using Chatterbox TTS.

Chatterbox is optimized for single-segment regeneration with
comprehensive validation (accent drift, speaker similarity, STT).

Voice cloning requires a reference audio file. Without it, Chatterbox
will use its default voice (see basic_usage.py for that pattern).

Requirements:
    pip install rho-tts[chatterbox,validation]
"""
from rho_tts import TTSFactory

# Create a Chatterbox TTS instance with voice cloning
tts = TTSFactory.get_tts_instance(
    provider="chatterbox",
    reference_audio="my_voice_sample.wav",  # Required for voice cloning
    # Customize validation thresholds
    accent_drift_threshold=0.2,     # Higher = more lenient (default: 0.17)
    text_similarity_threshold=0.7,  # Lower = more lenient (default: 0.75)
    max_iterations=30,              # Fewer retries (default: 50)
    # Use the faster implementation with rsxdalv optimizations
    implementation="faster",
)

# Generate with full validation loop
result = tts.generate_single(
    text="The quick brown fox jumps over the lazy dog.",
    output_path="cloned_voice_output.wav",
)

if result is not None:
    print(f"Voice cloned audio saved successfully!")
    print(f"Audio tensor shape: {result.shape}")
