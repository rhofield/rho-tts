"""
Custom TTS provider registration example.

Shows how to create and register your own TTS provider that
integrates with the rho-tts factory pattern.
"""
from typing import Dict, List, Optional, Union

import torch

from rho_tts import BaseTTS, CancellationToken, TTSFactory


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

    def generate(
        self,
        texts: List[str],
        output_base_path: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Optional[List[str]]:
        """Batch generation."""
        import torchaudio as ta

        token = cancellation_token or CancellationToken()
        output_files = []

        for idx, text in enumerate(texts):
            token.raise_if_cancelled()
            mapped_text = self._apply_phonetic_mapping(text)
            audio = self._generate_audio(mapped_text)

            output_path = f"{output_base_path}_{idx}.wav"
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            ta.save(output_path, audio, self._sample_rate)
            output_files.append(output_path)

        return output_files

    def generate_single(
        self,
        text: str,
        output_path: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Optional[torch.Tensor]:
        """Single text generation."""
        import torchaudio as ta

        mapped_text = self._apply_phonetic_mapping(text)
        audio = self._generate_audio(mapped_text)

        if audio.dim() == 1:
            save_audio = audio.unsqueeze(0)
        else:
            save_audio = audio

        ta.save(output_path, save_audio, self._sample_rate)
        return audio

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

# Generate audio
tts.generate_single("Hello from my custom TTS provider!", "custom_output.wav")
print("Custom provider audio generated!")
