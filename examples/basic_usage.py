"""
Basic usage example: Generate speech from text using Qwen3-TTS.

Requirements:
    pip install ralph-tts[qwen,validation]

You also need:
    - A reference audio file (.wav) for voice cloning
    - A transcript of that reference audio
    - A GPU with at least 8GB VRAM (for the 0.6B model) or 16GB (for 1.7B)
"""
from ralph_tts import TTSFactory

# Create a Qwen TTS instance
# You MUST provide your own reference audio and its transcript
tts = TTSFactory.get_tts_instance(
    provider="qwen",
    reference_audio="my_voice_sample.wav",
    reference_text="This is the transcript of my voice sample.",
    # Optional: specify a local model path or HuggingFace model ID
    # model_path="Qwen/Qwen3-TTS-12Hz-0.6B-Base",  # smaller model
    # model_path="/path/to/local/model",  # local model
)

# Generate a single audio file
result = tts.generate_single(
    text="Hello! This is a test of the text to speech system.",
    output_path="output_single.wav",
)

if result is not None:
    print(f"Generated audio saved to output_single.wav")
else:
    print("Audio generation failed")

# Generate multiple audio files in batch (more efficient)
texts = [
    "This is the first sentence to generate.",
    "Here is the second sentence with different content.",
    "And finally, a third sentence to complete the set.",
]

output_files = tts.generate(
    texts=texts,
    output_base_path="output_batch",
)

if output_files:
    for path in output_files:
        if path:
            print(f"Generated: {path}")
