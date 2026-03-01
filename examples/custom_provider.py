"""
Custom TTS provider registration example.

Shows how to create and register your own TTS provider that
integrates with the rho-tts factory pattern.

Only two abstract methods need implementing:
  - _generate_audio(text) -> audio tensor
  - sample_rate (property)

The base class provides generate() with text splitting, segment
joining, phonetic mapping, and validation/retry out of the box.
Override _post_process_audio() to add custom audio post-processing.
"""
from typing import Dict, Optional

import torch

from rho_tts import BaseTTS, TTSFactory


class MyCustomTTS(BaseTTS):
    """Example custom TTS provider."""

    def __init__(
        self,
        device: str = "cuda",
        seed: int = 42,
        deterministic: bool = False,
        phonetic_mapping: Optional[Dict[str, str]] = None,
        my_custom_param: str = "default",
    ):
        super().__init__(device, seed, deterministic, phonetic_mapping=phonetic_mapping)
        self.my_custom_param = my_custom_param
        self._sample_rate = 24000
        print(f"MyCustomTTS initialized with param: {my_custom_param}")

    def _generate_audio(self, text: str, **kwargs) -> torch.Tensor:
        """Generate audio using your custom model."""
        # Replace this with your actual model inference.
        # BaseTTS.generate() always calls this with a single string.
        # Placeholder: generate 2 seconds of silence
        return torch.zeros(self._sample_rate * 2)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate


# Register the custom provider
TTSFactory.register_provider("my_custom", MyCustomTTS)

# Now use it through the factory
tts = TTSFactory.get_tts_instance(
    provider="my_custom",
    my_custom_param="hello",
    phonetic_mapping={"AI": "A.I."},
)

# List all available providers
print(f"Available providers: {TTSFactory.list_providers()}")

# Generate audio — generate() is the primary API
result = tts.generate("Hello from my custom TTS provider!", "custom_output.wav")
if result:
    print(f"Custom provider audio generated: {result}")
