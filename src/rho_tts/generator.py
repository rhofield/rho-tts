"""
High-level audio generation orchestrator.

Manages TTS instance lifecycle, batch generation, segment regeneration,
and audio file joining.
"""
import asyncio
import logging
import os
import time
from typing import Callable, List, Optional

from pydub import AudioSegment

from .base_tts import BaseTTS
from .cancellation import CancellationToken, CancelledException
from .factory import TTSFactory

logger = logging.getLogger(__name__)


class GenerateAudio:
    """
    High-level audio generator that orchestrates TTS providers.

    Manages singleton TTS instances per provider for voice consistency,
    supports async generation with progress callbacks, and handles
    audio file joining.

    Args:
        text: List of text strings to generate audio for
        file_name: Base filename for output files
        output_dir: Directory for output files (default: current directory)
        progress_callback: Optional callback for progress reporting (current, total, message)
        cancellation_token: Optional token for cooperative cancellation
        sample_rate: Output sample rate (default: 44100)
        bit_rate: Output bit rate (default: "320k")
    """

    # Global TTS instances shared across all GenerateAudio instances for voice consistency
    _global_tts_instances = {}

    def __init__(
        self,
        text: List[str],
        file_name: str,
        output_dir: str = ".",
        progress_callback: Optional[Callable] = None,
        cancellation_token: Optional[CancellationToken] = None,
        sample_rate: int = 44100,
        bit_rate: str = "320k",
    ):
        self.text = text
        self.file_name = file_name
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        self.cancellation_token = cancellation_token or CancellationToken()
        self.sample_rate = sample_rate
        self.bit_rate = bit_rate
        self._generated_files = []

        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def _get_tts_instance(cls, provider: str = "qwen", **kwargs) -> BaseTTS:
        """Get or create provider-specific global TTS instance for consistent voice cloning."""
        if provider not in cls._global_tts_instances:
            logger.info(f"Creating new TTS instance (provider: {provider})")
            cls._global_tts_instances[provider] = TTSFactory.get_tts_instance(
                provider=provider,
                deterministic=True,
                **kwargs,
            )
            logger.info(f"TTS instance created for {provider}")
        return cls._global_tts_instances[provider]

    @classmethod
    def reset_instances(cls):
        """Clear all cached TTS instances. Useful for freeing GPU memory."""
        cls._global_tts_instances.clear()

    async def _report_progress(self, current: int, total: int, message: str = ""):
        """Report progress if callback is available."""
        if self.progress_callback:
            try:
                if asyncio.iscoroutinefunction(self.progress_callback):
                    await self.progress_callback(current, total, message)
                else:
                    self.progress_callback(current, total, message)
            except Exception as e:
                logger.error(f"Error reporting progress: {e}")

    async def run_async(self, provider: str = "qwen", **provider_kwargs):
        """
        Generate audio for all texts asynchronously with progress reporting.

        Args:
            provider: TTS provider to use (default: "qwen")
            **provider_kwargs: Additional kwargs passed to TTS constructor

        Returns:
            List of output file paths for successfully generated segments
        """
        local_tts = self._get_tts_instance(provider=provider, **provider_kwargs)

        total_files = len(self.text)
        total_steps = (total_files * 100) + 10
        await self._report_progress(0, total_steps, "Starting audio generation...")

        output_files = []
        await self._report_progress(10, total_steps, "Initializing audio generation...")

        loop = asyncio.get_event_loop()
        tts_start_time = time.time()

        for idx, text in enumerate(self.text):
            if self.cancellation_token.is_cancelled():
                await self._report_progress(0, total_steps, "Cancelling audio generation...")
                raise CancelledException("Audio generation was cancelled")

            try:
                start_progress = 10 + (idx * 100)
                await self._report_progress(start_progress, total_steps, f"Starting segment {idx + 1}/{total_files}")

                output_path = os.path.join(self.output_dir, f"{self.file_name}_{idx}.wav")

                result = await loop.run_in_executor(
                    None, local_tts.generate, [text], output_path.rsplit('_', 1)[0], self.cancellation_token
                )

                if result is None or len(result) == 0 or result[0] is None:
                    raise RuntimeError("Failed to generate audio")

                output_files.append(output_path)
                self._generated_files.append(output_path)
                logger.info(f"Generated: {self.file_name}_{idx}.wav")

                end_progress = 10 + ((idx + 1) * 100)
                await self._report_progress(end_progress, total_steps, f"Completed segment {idx + 1}/{total_files}")
            except CancelledException:
                raise
            except Exception as inner_e:
                logger.error(f"Failed to generate {self.file_name}_{idx}.wav: {inner_e}")
                continue

        tts_duration = time.time() - tts_start_time
        logger.info(f"Total TTS time: {tts_duration:.2f}s")

        final_output_path = await self._join_audio_files(output_files, total_steps)

        await self._report_progress(
            total_steps, total_steps,
            f"Audio generation completed! Generated {len(output_files)}/{total_files} files."
        )

        if final_output_path:
            logger.info(f"Combined audio: {final_output_path}")

        logger.info(f"Audio generation completed: {len(output_files)}/{len(self.text)} files")
        return output_files

    async def run_async_batch(self, provider: str = "qwen", **provider_kwargs):
        """
        Generate all audio in a single batch call (more efficient for providers that support it).

        Args:
            provider: TTS provider to use (default: "qwen")
            **provider_kwargs: Additional kwargs passed to TTS constructor

        Returns:
            List of output file paths for successfully generated segments
        """
        local_tts = self._get_tts_instance(provider=provider, **provider_kwargs)

        total_files = len(self.text)
        total_steps = 100
        await self._report_progress(0, total_steps, "Starting audio generation...")
        await self._report_progress(10, total_steps, "Initializing audio generation...")

        loop = asyncio.get_event_loop()
        tts_start_time = time.time()

        if self.cancellation_token.is_cancelled():
            await self._report_progress(0, total_steps, "Cancelling audio generation...")
            raise CancelledException("Audio generation was cancelled")

        try:
            output_base_path = os.path.join(self.output_dir, self.file_name)
            await self._report_progress(20, total_steps, f"Generating {total_files} audio file(s)...")

            output_files = await loop.run_in_executor(
                None, local_tts.generate, self.text, output_base_path, self.cancellation_token
            )

            await self._report_progress(80, total_steps, "Audio generation completed")

        except CancelledException:
            raise
        except Exception as e:
            logger.error(f"Failed to generate audio: {e}")
            return []

        tts_duration = time.time() - tts_start_time
        logger.info(f"Total TTS time: {tts_duration:.2f}s")

        if output_files is None:
            logger.error("Audio generation completely failed")
            await self._report_progress(100, total_steps, "Audio generation failed")
            return []

        successful_files = [f for f in output_files if f is not None]
        failed_count = len(output_files) - len(successful_files)

        self._generated_files.extend(successful_files)

        if failed_count > 0:
            logger.warning(f"{failed_count}/{total_files} file(s) failed to generate")

        if not successful_files:
            logger.error("All audio files failed to generate")
            await self._report_progress(100, total_steps, "Audio generation failed")
            return []

        await self._join_audio_files(successful_files, total_steps, progress_start=85)

        await self._report_progress(
            100, total_steps,
            f"Audio generation completed! Generated {len(successful_files)}/{total_files} files."
        )

        logger.info(f"Audio generation completed: {len(successful_files)}/{total_files} files")
        return successful_files

    async def _join_audio_files(
        self, output_files: List[str], total_steps: int, progress_start: int = None
    ) -> Optional[str]:
        """Join multiple audio files into a single combined file."""
        if not output_files:
            return None

        progress = progress_start or (total_steps - 5)
        await self._report_progress(progress, total_steps, "Joining audio files...")
        logger.info("Joining audio files together...")

        try:
            if self.cancellation_token.is_cancelled():
                raise CancelledException("Audio generation was cancelled during file joining")

            loop = asyncio.get_event_loop()

            def join_and_export():
                combined_audio = AudioSegment.empty()
                for file_path in output_files:
                    audio_segment = AudioSegment.from_wav(file_path)
                    combined_audio += audio_segment

                final_output_path = os.path.join(self.output_dir, f"{self.file_name}.wav")
                combined_audio.export(
                    final_output_path,
                    format="wav",
                    parameters=[
                        "-q:a", "0",
                        "-b:a", self.bit_rate,
                        "-ar", str(self.sample_rate),
                        "-acodec", "pcm_s24le",
                    ]
                )
                return final_output_path

            final_output_path = await loop.run_in_executor(None, join_and_export)
            logger.info(f"Combined audio saved to: {final_output_path}")
            return final_output_path

        except CancelledException:
            raise
        except Exception as e:
            logger.error(f"Failed to join audio files: {e}")
            return None

    def regenerate_segment(
        self,
        segment_index: int,
        segment_text: Optional[str] = None,
        provider: str = "chatterbox",
        **provider_kwargs,
    ) -> str:
        """
        Regenerate a specific audio segment by index.

        Args:
            segment_index: Index of the segment to regenerate
            segment_text: Text to generate. If None, uses self.text[segment_index]
            provider: TTS provider (default: "chatterbox" for single segment quality)
            **provider_kwargs: Additional kwargs passed to TTS constructor

        Returns:
            Path to the regenerated audio file
        """
        local_tts = self._get_tts_instance(provider=provider, **provider_kwargs)

        if segment_text is None:
            if segment_index < len(self.text):
                segment_text = self.text[segment_index]
            else:
                segment_text = self.text[0]

        output_path = os.path.join(self.output_dir, f"{self.file_name}_{segment_index}.wav")
        local_tts.generate_single(segment_text, output_path)
        logger.info(f"Regenerated segment {segment_index}: {self.file_name}_{segment_index}.wav")
        return output_path
