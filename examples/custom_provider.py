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
from typing import Dict, List, Optional, Union

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

    def _generate_audio(self, text: Union[str, List[str]], **kwargs) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate audio using your custom model."""
        # Replace this with your actual model inference
        is_single = isinstance(text, str)
        text_list = [text] if is_single else text

        results = []
        for t in text_list:
            # Placeholder: generate 2 seconds of silence
            duration_samples = self._sample_rate * 2
            audio = torch.zeros(duration_samples)
            results.append(audio)

        return results[0] if is_single else results

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
result = tts.generate(["Hello from my custom TTS provider!"], "custom_output")
if result and result[0]:
    print(f"Custom provider audio generated: {result[0]}")
