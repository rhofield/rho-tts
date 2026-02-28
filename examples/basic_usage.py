"""
Basic usage example: Generate speech from text using Qwen3-TTS.

Requirements:
    pip install rho-tts[qwen]

Optional (for voice cloning + validation):
    pip install rho-tts[qwen,validation]

You need a GPU with at least 8GB VRAM (for the 0.6B model) or 16GB (for 1.7B).
"""
from rho_tts import TTSFactory

# --- Option 1: Default voice (no reference audio needed) ---
tts = TTSFactory.get_tts_instance(
    provider="qwen",
    # Optional: specify a local model path or HuggingFace model ID
    # model_path="Qwen/Qwen3-TTS-12Hz-0.6B-Base",  # smaller model
    # model_path="/path/to/local/model",  # local model
)

result = tts.generate(
    "Hello! This is a test of the text to speech system.",
    "output_default_voice.wav",
)

if result:
    print(f"Generated audio with default voice: {result}")
else:
    print("Audio generation failed")


# --- Option 2: Voice cloning (provide reference audio + transcript) ---
tts_cloned = TTSFactory.get_tts_instance(
    provider="qwen",
    reference_audio="my_voice_sample.wav",
    reference_text="This is the transcript of my voice sample.",
)

result = tts_cloned.generate(
    "Hello! This is a test of the text to speech system.",
    "output_cloned_voice.wav",
)

if result:
    print(f"Generated audio with cloned voice: {result}")
else:
    print("Audio generation failed")


# --- Batch generation (works with either mode) ---
texts = [
    "This is the first sentence to generate.",
    "Here is the second sentence with different content.",
    "And finally, a third sentence to complete the set.",
]

output_files = tts.generate(texts, "output_batch")

if output_files:
    for path in output_files:
        if path:
            print(f"Generated: {path}")
