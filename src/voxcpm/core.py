import os
import sys
import re
import json
import tempfile
import numpy as np
from typing import Generator, Optional
from .model.voxcpm2 import VoxCPM2Model
from .model.utils import next_and_close


class VoxCPM:
    def __init__(
        self,
        voxcpm_model_path: str,
        zipenhancer_model_path: str | None = "iic/speech_zipenhancer_ans_multiloss_16k_base",
        enable_denoiser: bool = True,
        optimize: bool = True,
        device: str | None = None,
    ):
        """Initialize VoxCPM TTS pipeline.

        Args:
            voxcpm_model_path: Local filesystem path to the VoxCPM model assets
                (weights, configs, etc.). Typically the directory returned by
                a prior download step.
            zipenhancer_model_path: ModelScope acoustic noise suppression model
                id or local path. If None, denoiser will not be initialized.
            enable_denoiser: Whether to initialize the denoiser pipeline.
            optimize: Whether to optimize the model with torch.compile. True by default, but can be disabled for debugging.
            device: Runtime device. If set to ``None`` or ``"auto"``, VoxCPM
                will choose automatically (preferring CUDA).
                If set explicitly, that device is used or a clear error is raised.
        """
        print(
            f"voxcpm_model_path: {voxcpm_model_path}, zipenhancer_model_path: {zipenhancer_model_path}, enable_denoiser: {enable_denoiser}",
            file=sys.stderr,
        )

        self.tts_model = VoxCPM2Model.from_local(
            voxcpm_model_path,
            optimize=optimize,
            device=device,
        )
        print("Loaded VoxCPM2Model", file=sys.stderr)


        self.text_normalizer = None
        self.denoiser = None
        if enable_denoiser and zipenhancer_model_path is not None:
            from .zipenhancer import ZipEnhancer

            self.denoiser = ZipEnhancer(zipenhancer_model_path)
        else:
            self.denoiser = None
        if optimize:
            print("Warm up VoxCPMModel...", file=sys.stderr)
            self.tts_model.generate(
                target_text="Hello, this is the first test sentence.",
                max_len=10,
            )

    @classmethod
    def from_pretrained(
        cls,
        hf_model_id: str = "OpenBMB/VoxCPM2",
        load_denoiser: bool = True,
        zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
        cache_dir: str = None,
        local_files_only: bool = False,
        optimize: bool = True,
        device: str | None = None,
        **kwargs,
    ):
        """Instantiate ``VoxCPM`` from a Hugging Face Hub snapshot.

        Args:
            hf_model_id: Explicit Hugging Face repository id (e.g. "org/repo") or local path.
            load_denoiser: Whether to initialize the denoiser pipeline.
            optimize: Whether to optimize the model with torch.compile. True by default, but can be disabled for debugging.
            zipenhancer_model_id: Denoiser model id or path for ModelScope
                acoustic noise suppression.
            cache_dir: Custom cache directory for the snapshot.
            local_files_only: If True, only use local files and do not attempt
                to download.
            device: Runtime device. Use ``None``/``"auto"`` for automatic
                fallback, or an explicit value such as ``"cuda"``, or ``"cuda:0"``.
        Kwargs:
            Additional keyword arguments passed to the ``VoxCPM`` constructor.

        Returns:
            VoxCPM: Initialized instance whose ``voxcpm_model_path`` points to
            the downloaded snapshot directory.

        Raises:
            ValueError: If neither a valid ``hf_model_id`` nor a resolvable
                ``hf_model_id`` is provided.
        """
        repo_id = hf_model_id
        if not repo_id:
            raise ValueError("You must provide hf_model_id")

        if not os.path.isdir(repo_id):
            raise FileNotFoundError(f"Offline model directory not found: {repo_id}")

        return cls(
            voxcpm_model_path=repo_id,
            zipenhancer_model_path=zipenhancer_model_id if load_denoiser else None,
            enable_denoiser=load_denoiser,
            optimize=optimize,
            device=device,
            **kwargs,
        )

    def generate(self, *args, **kwargs) -> np.ndarray:
        return next_and_close(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[np.ndarray, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    def _generate(
        self,
        text: str,
        prompt_wav_path: str = None,
        prompt_text: str = None,
        reference_wav_path: str = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """Synthesize speech for the given text and return a single waveform.

        Args:
            text: Input text to synthesize.
            prompt_wav_path: Path to prompt audio for continuation mode.
                Must be paired with ``prompt_text``.
            prompt_text: Text content corresponding to the prompt audio.
            reference_wav_path: Path to reference audio for voice cloning
                (structurally isolated via ref_audio tokens). Can be used
                alone or combined with ``prompt_wav_path`` + ``prompt_text``.
            cfg_value: Guidance scale for the generation model.
            inference_timesteps: Number of inference steps.
            min_len: Minimum audio length.
            max_len: Maximum token length during generation.
            normalize: Whether to run text normalization before generation.
            denoise: Whether to denoise the prompt/reference audio if a
                denoiser is available.
            retry_badcase: Whether to retry badcase.
            retry_badcase_max_times: Maximum number of times to retry badcase.
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio.
            streaming: Whether to return a generator of audio chunks.
        Returns:
            Generator of numpy.ndarray: 1D waveform array (float32) on CPU.
            Yields audio chunks for each generation step if ``streaming=True``,
            otherwise yields a single array containing the final audio.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("target text must be a non-empty string")

        if prompt_wav_path is not None:
            if not os.path.exists(prompt_wav_path):
                raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")

        if reference_wav_path is not None:
            if not os.path.exists(reference_wav_path):
                raise FileNotFoundError(f"reference_wav_path does not exist: {reference_wav_path}")

        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")

        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        temp_files = []

        try:
            actual_prompt_path = prompt_wav_path
            actual_ref_path = reference_wav_path

            if denoise and self.denoiser is not None:
                if prompt_wav_path is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        temp_files.append(tmp.name)
                    self.denoiser.enhance(prompt_wav_path, output_path=temp_files[-1])
                    actual_prompt_path = temp_files[-1]
                if reference_wav_path is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        temp_files.append(tmp.name)
                    self.denoiser.enhance(reference_wav_path, output_path=temp_files[-1])
                    actual_ref_path = temp_files[-1]

            if actual_prompt_path is not None or actual_ref_path is not None:
                    fixed_prompt_cache = self.tts_model.build_prompt_cache(
                        prompt_text=prompt_text,
                        prompt_wav_path=actual_prompt_path,
                        reference_wav_path=actual_ref_path,
                    )

            else:
                fixed_prompt_cache = None

            if normalize:
                if self.text_normalizer is None:
                    from .utils.text_normalize import TextNormalizer

                    self.text_normalizer = TextNormalizer()
                text = self.text_normalizer.normalize(text)

            generate_result = self.tts_model._generate_with_prompt_cache(
                target_text=text,
                prompt_cache=fixed_prompt_cache,
                min_len=min_len,
                max_len=max_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                retry_badcase=retry_badcase,
                retry_badcase_max_times=retry_badcase_max_times,
                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                streaming=streaming,
            )

            if streaming:
                try:
                    for wav, _, _ in generate_result:
                        yield wav.squeeze(0).cpu().numpy()
                finally:
                    generate_result.close()
            else:
                wav, _, _ = next_and_close(generate_result)
                yield wav.squeeze(0).cpu().numpy()

        finally:
            for tmp_path in temp_files:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
