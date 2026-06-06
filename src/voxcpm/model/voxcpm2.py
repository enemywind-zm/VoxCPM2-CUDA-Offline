"""
VoxCPM: A Tokenizer-free speech generation model

This module contains the main VoxCPM model implementation, including configuration classes
and the core VoxCPMModel for text-to-speech generation.

Copyright 2026 OpenBMB
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
from typing import Tuple, Union, Generator, List, Optional

import torch
import torch.nn as nn
import warnings
import librosa
import numpy as np
from einops import rearrange
from pydantic import BaseModel

try:
    from safetensors.torch import load_file

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
from tqdm import tqdm
from transformers import LlamaTokenizerFast

from ..modules.audiovae import AudioVAEV2, AudioVAEConfigV2
from ..modules.layers import ScalarQuantizationLayer
from ..modules.locdit import CfmConfig, UnifiedCFM, VoxCPMLocDiTV2
from ..modules.locenc import VoxCPMLocEnc
from ..modules.minicpm4 import MiniCPM4Config, MiniCPMModel
from .utils import (
    get_dtype,
    mask_multichar_chinese_tokens,
    next_and_close,
    pick_runtime_dtype,
    resolve_runtime_device,
)


# A simple function to trim audio silence using VAD, not used default
def _trim_audio_silence_vad(audio: torch.Tensor, sample_rate: int, max_silence_ms: float = 200.0, top_db: float = 35.0) -> torch.Tensor:
    if audio.numel() == 0:
        return audio
    y = audio.squeeze(0).numpy()
    n = len(y)
    frame_length = 2048
    hop_length = 512
    ref = np.max(np.abs(y))
    if ref <= 0:
        return audio
    threshold = ref * (10.0 ** (-top_db / 20.0))

    try:
        _, (start, end) = librosa.effects.trim(
            y, top_db=top_db, ref=np.max, frame_length=frame_length, hop_length=hop_length
        )
    except Exception:
        start, end = 0, n

    # Find the last frame with continuous energy, trim the long pseudo-silence at the end (low energy background noise, etc.)
    n_frames = max(0, (n - frame_length) // hop_length + 1)
    last_voice_frame = -1
    for j in range(n_frames):
        idx = j * hop_length
        if idx + frame_length > n:
            break
        rms = np.sqrt(np.mean(y[idx : idx + frame_length] ** 2))
        if rms >= threshold:
            last_voice_frame = j
    if last_voice_frame >= 0:
        end_by_vad = min(n, (last_voice_frame + 1) * hop_length + (frame_length - hop_length))
        end = min(end, end_by_vad)

    max_silence_samples = int(max_silence_ms * sample_rate / 1000.0)
    new_start = max(0, start - max_silence_samples)
    new_end = min(n, end + max_silence_samples)
    return audio[:, new_start:new_end]


class VoxCPMEncoderConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None


class VoxCPMDitConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None
    dit_mean_mode: bool = False

    cfm_config: CfmConfig


class VoxCPMConfig(BaseModel):
    lm_config: MiniCPM4Config
    patch_size: int = 4
    feat_dim: int = 64
    residual_lm_num_layers: int = 8
    residual_lm_no_rope: bool = False
    scalar_quantization_latent_dim: int = 512
    scalar_quantization_scale: int = 9

    encoder_config: VoxCPMEncoderConfig
    dit_config: VoxCPMDitConfig
    audio_vae_config: Optional[AudioVAEConfigV2] = None

    max_length: int = 8192
    device: str = "cuda"
    dtype: str = "bfloat16"


VoxCPMConfig.model_rebuild()


class VoxCPM2Model(nn.Module):
    def __init__(
        self,
        config: VoxCPMConfig,
        tokenizer: LlamaTokenizerFast,
        audio_vae: AudioVAEV2,
        device: str | None = None,
    ):
        super().__init__()
        self.config = config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size
        self.device = resolve_runtime_device(device, config.device)
        self.config.device = self.device
        resolved_dtype = pick_runtime_dtype(self.device, self.config.dtype)
        if resolved_dtype != self.config.dtype:
            print(
                f"[voxcpm2] adjusted dtype {self.config.dtype} -> {resolved_dtype} for device {self.device}",
                file=sys.stderr,
            )
            self.config.dtype = resolved_dtype
        print(f"Running on device: {self.device}, dtype: {self.config.dtype}", file=sys.stderr)

        # Text-Semantic LM
        self.base_lm = MiniCPMModel(config.lm_config)
        self.base_lm.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        self.text_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        self.audio_start_token = 101
        self.audio_end_token = 102
        self.ref_audio_start_token = 103
        self.ref_audio_end_token = 104

        # Residual Acoustic LM
        residual_lm_config = config.lm_config.model_copy(deep=True)
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers
        residual_lm_config.vocab_size = 0
        residual_lm_config.no_rope = config.residual_lm_no_rope
        self.residual_lm = MiniCPMModel(residual_lm_config)
        self.residual_lm.setup_cache(1, config.max_length, self.device, get_dtype(self.config.dtype))

        # Local Encoder
        encoder_config = config.lm_config.model_copy(deep=True)
        encoder_config.hidden_size = config.encoder_config.hidden_dim
        encoder_config.intermediate_size = config.encoder_config.ffn_dim
        encoder_config.num_attention_heads = config.encoder_config.num_heads
        encoder_config.num_hidden_layers = config.encoder_config.num_layers
        encoder_config.kv_channels = config.encoder_config.kv_channels
        encoder_config.vocab_size = 0
        self.feat_encoder = VoxCPMLocEnc(encoder_config, input_dim=config.feat_dim)

        # Local DiT
        decoder_config = config.lm_config.model_copy(deep=True)
        decoder_config.hidden_size = config.dit_config.hidden_dim
        decoder_config.intermediate_size = config.dit_config.ffn_dim
        decoder_config.num_attention_heads = config.dit_config.num_heads
        decoder_config.num_hidden_layers = config.dit_config.num_layers
        decoder_config.kv_channels = config.dit_config.kv_channels
        decoder_config.vocab_size = 0
        self.feat_decoder = UnifiedCFM(
            in_channels=config.feat_dim,
            cfm_params=config.dit_config.cfm_config,
            estimator=VoxCPMLocDiTV2(decoder_config, in_channels=config.feat_dim),
            mean_mode=config.dit_config.dit_mean_mode,
        )

        # Projection layers
        self.fsq_layer = ScalarQuantizationLayer(
            config.lm_config.hidden_size,
            config.lm_config.hidden_size,
            config.scalar_quantization_latent_dim,
            config.scalar_quantization_scale,
        )
        self.enc_to_lm_proj = nn.Linear(config.encoder_config.hidden_dim, config.lm_config.hidden_size)
        self.lm_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        self.res_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        self.fusion_concat_proj = nn.Linear(config.lm_config.hidden_size * 2, config.lm_config.hidden_size)

        # Stop Predictor
        self.stop_proj = nn.Linear(config.lm_config.hidden_size, config.lm_config.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(config.lm_config.hidden_size, 2, bias=False)

        # Audio VAE
        self.audio_vae = audio_vae
        self.chunk_size = audio_vae.chunk_size
        self._decode_chunk_size = getattr(audio_vae, "decode_chunk_size", audio_vae.chunk_size)
        self._encode_sample_rate = audio_vae.sample_rate
        self.sample_rate = getattr(audio_vae, "out_sample_rate", audio_vae.sample_rate)

    def optimize(self, disable: bool = False):
        if disable:
            return self
        try:
            if self.device != "cuda":
                raise ValueError("VoxCPMModel can only be optimized on CUDA device")
            try:
                import triton  # noqa: F401
            except ImportError:
                raise ValueError("triton is not installed")
            self.base_lm.forward_step = torch.compile(self.base_lm.forward_step, mode="reduce-overhead", fullgraph=True)
            self.residual_lm.forward_step = torch.compile(
                self.residual_lm.forward_step, mode="reduce-overhead", fullgraph=True
            )
            self._feat_encoder_raw = self.feat_encoder
            self.feat_encoder = torch.compile(self.feat_encoder, mode="reduce-overhead", fullgraph=True)
            self.feat_decoder.estimator = torch.compile(
                self.feat_decoder.estimator, mode="reduce-overhead", fullgraph=True
            )
        except Exception as e:
            print(f"Warning: torch.compile disabled - {e}", file=sys.stderr)
        return self

    def _dtype(self):
        return get_dtype(self.config.dtype)

    def _encode_wav(
        self,
        wav_path: str,
        padding_mode: str = "right",
        trim_silence_vad: bool = False,
    ) -> torch.Tensor:
        """Load, trim, pad and VAE-encode an audio file.

        Args:
            wav_path: path to the audio file.
            padding_mode: "right" (default) or "left" padding for alignment.
            trim_silence_vad: whether to apply VAD-based silence trimming.

        Returns:
            audio_feat: (T, P, D) tensor of latent patches.
        """
        audio, _ = librosa.load(wav_path, sr=self._encode_sample_rate, mono=True)
        audio = torch.from_numpy(audio).unsqueeze(0)
        if trim_silence_vad:
            audio = _trim_audio_silence_vad(audio, self._encode_sample_rate, max_silence_ms=200.0)
        patch_len = self.patch_size * self.chunk_size
        if audio.size(1) % patch_len != 0:
            padding_size = patch_len - audio.size(1) % patch_len
            pad = (padding_size, 0) if padding_mode == "left" else (0, padding_size)
            audio = torch.nn.functional.pad(audio, pad)
        feat = self.audio_vae.encode(audio.to(self.device), self._encode_sample_rate).cpu()
        return feat.view(self.audio_vae.latent_dim, -1, self.patch_size).permute(1, 2, 0)

    def _make_ref_prefix(self, ref_feat: torch.Tensor, device: torch.device):
        """Build the [ref_start ref_audio ref_end] prefix segments.

        Returns:
            tokens, feats, text_mask, audio_mask  (all 1-D / 2-D tensors)
        """
        ref_len = ref_feat.size(0)
        z1 = torch.zeros((1, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32, device=device)
        tokens = torch.cat(
            [
                torch.tensor([self.ref_audio_start_token], dtype=torch.int32, device=device),
                torch.zeros(ref_len, dtype=torch.int32, device=device),
                torch.tensor([self.ref_audio_end_token], dtype=torch.int32, device=device),
            ]
        )
        feats = torch.cat([z1, ref_feat, z1], dim=0)
        t_mask = torch.cat(
            [
                torch.tensor([1], dtype=torch.int32),
                torch.zeros(ref_len, dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
            ]
        ).to(device)
        a_mask = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                torch.ones(ref_len, dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
            ]
        ).to(device)
        return tokens, feats, t_mask, a_mask

    def generate(self, *args, **kwargs) -> torch.Tensor:
        return next_and_close(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[torch.Tensor, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        reference_wav_path: str = "",
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        trim_silence_vad: bool = False,
        streaming: bool = False,
        streaming_prefix_len: int = 4,
    ) -> Generator[torch.Tensor, None, None]:
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False

        if reference_wav_path and prompt_wav_path:
            # Combined mode: reference isolation prefix + continuation suffix
            text = prompt_text + target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            ref_feat = self._encode_wav(
                reference_wav_path,
                padding_mode="right",
                trim_silence_vad=trim_silence_vad,
            )
            prompt_feat = self._encode_wav(prompt_wav_path, padding_mode="left", trim_silence_vad=trim_silence_vad)
            prompt_audio_length = prompt_feat.size(0)

            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_feat, text_token.device)

            prompt_pad_token = torch.zeros(prompt_audio_length, dtype=torch.int32, device=text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )

            text_token = torch.cat([ref_tokens, text_token, prompt_pad_token])
            audio_feat = torch.cat([ref_feats, text_pad_feat, prompt_feat], dim=0)
            text_mask = torch.cat(
                [
                    ref_t_mask,
                    torch.ones(text_length, dtype=torch.int32).to(text_token.device),
                    torch.zeros(prompt_audio_length, dtype=torch.int32).to(text_token.device),
                ]
            )
            audio_mask = torch.cat(
                [
                    ref_a_mask,
                    torch.zeros(text_length, dtype=torch.int32).to(text_token.device),
                    torch.ones(prompt_audio_length, dtype=torch.int32).to(text_token.device),
                ]
            )

        elif reference_wav_path:
            # Reference-only mode (prompt isolation)
            text = target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            ref_feat = self._encode_wav(
                reference_wav_path,
                padding_mode="right",
                trim_silence_vad=trim_silence_vad,
            )
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_feat, text_token.device)

            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([ref_tokens, text_token])
            audio_feat = torch.cat([ref_feats, text_pad_feat], dim=0)
            text_mask = torch.cat(
                [
                    ref_t_mask,
                    torch.ones(text_length, dtype=torch.int32).to(text_token.device),
                ]
            )
            audio_mask = torch.cat(
                [
                    ref_a_mask,
                    torch.zeros(text_length, dtype=torch.int32).to(text_token.device),
                ]
            )

        elif len(prompt_wav_path) == 0:
            # Zero-shot mode
            text = target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            audio_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_mask = torch.ones(text_length, dtype=torch.int32).to(text_token.device)
            audio_mask = torch.zeros(text_length, dtype=torch.int32).to(text_token.device)

        else:
            # Continuation-only mode
            text = prompt_text + target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
                ],
                dim=-1,
            )
            text_length = text_token.shape[0]

            prompt_feat = self._encode_wav(prompt_wav_path, padding_mode="left", trim_silence_vad=trim_silence_vad)
            prompt_audio_length = prompt_feat.size(0)
            prompt_pad_token = torch.zeros(prompt_audio_length, dtype=torch.int32, device=text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([text_token, prompt_pad_token])
            audio_feat = torch.cat([text_pad_feat, prompt_feat], dim=0)
            text_mask = torch.cat(
                [
                    torch.ones(text_length, dtype=torch.int32),
                    torch.zeros(prompt_audio_length, dtype=torch.int32),
                ]
            ).to(text_token.device)
            audio_mask = torch.cat(
                [
                    torch.zeros(text_length, dtype=torch.int32),
                    torch.ones(prompt_audio_length, dtype=torch.int32),
                ]
            ).to(text_token.device)

        text_token = text_token.unsqueeze(0).to(self.device)
        text_mask = text_mask.unsqueeze(0).to(self.device)
        audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
        audio_mask = audio_mask.unsqueeze(0).to(self.device)

        target_text_length = len(self.text_tokenizer(target_text))

        retry_badcase_times = 0
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len),
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
                streaming_prefix_len=streaming_prefix_len,
            )
            if streaming:
                with self.audio_vae.streaming_decode() as vae_dec:
                    for latent_pred, _, _ctx in inference_result:
                        decode_audio = vae_dec.decode_chunk(latent_pred.to(torch.float32))
                        decode_audio = decode_audio.squeeze(1).cpu()
                        yield decode_audio
                break
            else:
                latent_pred, pred_audio_feat, context_len = next_and_close(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(
                            f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...",
                            file=sys.stderr,
                        )
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break

        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
            decode_patch_len = self.patch_size * self._decode_chunk_size
            if context_len > 0:
                decode_audio = decode_audio[..., decode_patch_len * context_len:].squeeze(1).cpu()
            else:
                decode_audio = decode_audio.squeeze(1).cpu()
            yield decode_audio

    @torch.inference_mode()
    def build_prompt_cache(
        self,
        prompt_text: str = None,
        prompt_wav_path: str = None,
        reference_wav_path: str = None,
        trim_silence_vad: bool = False,
    ):
        """
        Build prompt cache for subsequent generation.

        Supports the same parameter combinations as ``generate()``:
        - ``reference_wav_path`` only -> reference mode (voice cloning, isolated)
        - ``prompt_text`` + ``prompt_wav_path`` -> continuation mode
        - all three -> combined ref + continuation mode

        Args:
            prompt_text: prompt text for continuation mode.
                Must be paired with ``prompt_wav_path``.
            prompt_wav_path: prompt audio path for continuation mode.
                Must be paired with ``prompt_text``.
            reference_wav_path: reference audio path for voice cloning
                (structurally isolated via ref_audio tokens).
            trim_silence_vad: whether to apply VAD-based silence trimming
                before encoding prompt/reference audio.

        Returns:
            prompt_cache: dict used by ``_generate_with_prompt_cache``.
        """
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")
        if prompt_wav_path is None and reference_wav_path is None:
            raise ValueError("At least one of prompt_wav_path or reference_wav_path must be provided")

        cache = {}

        if reference_wav_path:
            cache["ref_audio_feat"] = self._encode_wav(
                reference_wav_path,
                padding_mode="right",
                trim_silence_vad=trim_silence_vad,
            )

        if prompt_wav_path and prompt_text is not None:
            cache["prompt_text"] = prompt_text
            cache["audio_feat"] = self._encode_wav(
                prompt_wav_path,
                padding_mode="left",
                trim_silence_vad=trim_silence_vad,
            )

        has_ref = "ref_audio_feat" in cache
        has_prompt = "audio_feat" in cache
        if has_ref and has_prompt:
            cache["mode"] = "ref_continuation"
        elif has_ref:
            cache["mode"] = "reference"
        else:
            cache["mode"] = "continuation"

        return cache

    def merge_prompt_cache(
        self,
        original_cache: dict,
        new_text: str,
        new_audio_feat: torch.Tensor,
    ):
        """
        Merge original prompt cache with newly generated content to stabilize voice.

        Args:
            original_cache: original prompt cache (any mode)
            new_text: newly generated text
            new_audio_feat: newly generated audio features

        Returns:
            merged_cache: merged cache with prompt_text and audio_feat
        """
        if original_cache is None:
            return {
                "prompt_text": new_text,
                "audio_feat": new_audio_feat,
                "mode": "continuation",
            }
        merged = {}
        if "ref_audio_feat" in original_cache:
            merged["ref_audio_feat"] = original_cache["ref_audio_feat"]
        merged["prompt_text"] = original_cache.get("prompt_text", "") + new_text
        old_feat = original_cache.get("audio_feat", new_audio_feat.new_empty(0, *new_audio_feat.shape[1:]))
        merged["audio_feat"] = torch.cat([old_feat, new_audio_feat], dim=0)
        merged["mode"] = "ref_continuation" if "ref_audio_feat" in merged else "continuation"
        return merged

    def generate_with_prompt_cache(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return next_and_close(self._generate_with_prompt_cache(*args, streaming=False, **kwargs))

    def generate_with_prompt_cache_streaming(
        self, *args, **kwargs
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]], None, None]:
        return self._generate_with_prompt_cache(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate_with_prompt_cache(
        self,
        target_text: str,
        prompt_cache: dict,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
        streaming_prefix_len: int = 4,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """
        Generate audio using pre-built prompt cache.

        Args:
            target_text: Text to convert to speech
            prompt_cache: Cache built by ``build_prompt_cache()``. Can be None
                for zero-shot generation.
            min_len: Minimum audio length to avoid very short audio
            max_len: Maximum audio length
            inference_timesteps: Number of diffusion sampling steps
            cfg_value: Classifier-free guidance value
            retry_badcase: Whether to retry on bad cases
            retry_badcase_max_times: Maximum retry attempts
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio
            streaming: Whether to return a generator of audio chunks
            streaming_prefix_len: Number of prefix audio patches to use for streaming mode

        Returns:
            Generator of Tuple containing:
                - Decoded audio tensor for the current step if ``streaming=True``, else final decoded audio tensor
                - Tensor of new text tokens
                - New audio features up to the current step as a List if ``streaming=True``, else as a concatenated Tensor
        """
        if retry_badcase and streaming:
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False

        # Determine mode from cache
        if prompt_cache is None:
            mode = "zero_shot"
            text = target_text
        else:
            mode = prompt_cache.get("mode", "continuation")
            if mode in ("continuation", "ref_continuation"):
                prompt_text = prompt_cache.get("prompt_text", "")
                text = prompt_text + target_text
            else:
                text = target_text

        text_token = torch.LongTensor(self.text_tokenizer(text))
        text_token = torch.cat(
            [
                text_token,
                torch.tensor([self.audio_start_token], dtype=torch.int32, device=text_token.device),
            ],
            dim=-1,
        )

        target_text_token = torch.LongTensor(self.text_tokenizer(target_text))
        text_length = text_token.shape[0]

        if mode in ("zero_shot", "continuation"):
            prompt_audio_feat = (
                prompt_cache["audio_feat"]
                if prompt_cache
                else torch.empty((0, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32)
            )
            audio_length = prompt_audio_feat.size(0)
            text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([text_token, text_pad_token])
            audio_feat = torch.cat([text_pad_feat, prompt_audio_feat], dim=0)
            text_mask = torch.cat(
                [torch.ones(text_length, dtype=torch.int32), torch.zeros(audio_length, dtype=torch.int32)]
            ).to(text_token.device)
            audio_mask = torch.cat(
                [torch.zeros(text_length, dtype=torch.int32), torch.ones(audio_length, dtype=torch.int32)]
            ).to(text_token.device)

        elif mode == "reference":
            ref_audio_feat = prompt_cache["ref_audio_feat"]
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_audio_feat, text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([ref_tokens, text_token])
            audio_feat = torch.cat([ref_feats, text_pad_feat], dim=0)
            text_mask = torch.cat([ref_t_mask, torch.ones(text_length, dtype=torch.int32).to(text_token.device)])
            audio_mask = torch.cat([ref_a_mask, torch.zeros(text_length, dtype=torch.int32).to(text_token.device)])

        else:
            # ref_continuation mode
            ref_audio_feat = prompt_cache["ref_audio_feat"]
            prompt_audio_feat = prompt_cache["audio_feat"]
            prompt_audio_length = prompt_audio_feat.size(0)

            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_audio_feat, text_token.device)

            prompt_pad_token = torch.zeros(prompt_audio_length, dtype=torch.int32, device=text_token.device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )

            text_token = torch.cat([ref_tokens, text_token, prompt_pad_token])
            audio_feat = torch.cat([ref_feats, text_pad_feat, prompt_audio_feat], dim=0)
            text_mask = torch.cat(
                [
                    ref_t_mask,
                    torch.ones(text_length, dtype=torch.int32).to(text_token.device),
                    torch.zeros(prompt_audio_length, dtype=torch.int32).to(text_token.device),
                ]
            )
            audio_mask = torch.cat(
                [
                    ref_a_mask,
                    torch.zeros(text_length, dtype=torch.int32).to(text_token.device),
                    torch.ones(prompt_audio_length, dtype=torch.int32).to(text_token.device),
                ]
            )

        text_token = text_token.unsqueeze(0).to(self.device)
        text_mask = text_mask.unsqueeze(0).to(self.device)
        audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
        audio_mask = audio_mask.unsqueeze(0).to(self.device)

        # run inference
        target_text_length = len(self.text_tokenizer(target_text))
        retry_badcase_times = 0
        while retry_badcase_times < retry_badcase_max_times:
            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                max_len=min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len),
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
                streaming_prefix_len=streaming_prefix_len,
            )
            if streaming:
                with self.audio_vae.streaming_decode() as vae_dec:
                    for latent_pred, pred_audio_feat, _ctx in inference_result:
                        decode_audio = vae_dec.decode_chunk(latent_pred.to(torch.float32))
                        decode_audio = decode_audio.squeeze(1).cpu()
                        yield (decode_audio, target_text_token, pred_audio_feat)
                break
            else:
                latent_pred, pred_audio_feat, context_len = next_and_close(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(
                            f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...",
                            file=sys.stderr,
                        )
                        retry_badcase_times += 1
                        continue
                    else:
                        break
                else:
                    break
        if not streaming:
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
            decode_patch_len = self.patch_size * self._decode_chunk_size
            if context_len > 0:
                decode_audio = decode_audio[..., decode_patch_len * context_len:].squeeze(1).cpu()
            else:
                decode_audio = decode_audio.squeeze(1).cpu()
            yield (decode_audio, target_text_token, pred_audio_feat)

    def inference(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_pred, generated_feat, _ = next_and_close(self._inference(*args, streaming=False, **kwargs))
        return feat_pred, generated_feat

    def inference_streaming(self, *args, **kwargs) -> Generator[Tuple[torch.Tensor, List[torch.Tensor]], None, None]:
        for feat_pred, pred_feat_seq, _ in self._inference(*args, streaming=True, **kwargs):
            yield feat_pred, pred_feat_seq

    @torch.inference_mode()
    def _inference(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        streaming: bool = False,
        streaming_prefix_len: int = 4,
    ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """Core inference method for audio generation.

        This is the main inference loop that generates audio features
        using the language model and diffusion transformer.

        Args:
            text: Input text tokens
            text_mask: Mask for text tokens
            feat: Input audio features
            feat_mask: Mask for audio features
            min_len: Minimum generation length
            max_len: Maximum generation length
            inference_timesteps: Number of diffusion steps
            cfg_value: Classifier-free guidance value
            streaming: Whether to yield each step latent feature or just the final result

        Returns:
            Generator of Tuple containing:
                - Predicted latent feature at the current step if ``streaming=True``, else final latent features
                - Predicted audio feature sequence so far as a List if ``streaming=True``, else as a concatenated Tensor
        """
        B, T, P, D = feat.shape

        prefill_encoder = getattr(self, "_feat_encoder_raw", self.feat_encoder)
        feat_embed = prefill_encoder(feat)  # [b, t, h_feat]
        feat_embed = self.enc_to_lm_proj(feat_embed)

        if self.config.lm_config.use_mup:
            scale_emb = self.config.lm_config.scale_emb
        else:
            scale_emb = 1.0

        text_embed = self.base_lm.embed_tokens(text) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

        prefix_feat_cond = feat[:, -1, ...]  # b, p, d
        pred_feat_seq = []  # b, t, p, d
        curr_embed = None

        # Prepare prompt context patches for streaming mode
        # - Continuation modes (feat_mask ends with 1): use the last (streaming_prefix_len - 1)
        #   trailing audio patches as initial context so the VAE can decode smoothly.
        # - Reference-only / zero-shot (feat_mask ends with 0): start from scratch.
        has_continuation_audio = feat_mask[0, -1].item() == 1
        context_len = 0
        if has_continuation_audio:
            audio_indices = feat_mask.squeeze(0).nonzero(as_tuple=True)[0]
            context_len = min(streaming_prefix_len - 1, len(audio_indices))
            last_audio_indices = audio_indices[-context_len:]
            pred_feat_seq = list(feat[:, last_audio_indices, :, :].split(1, dim=1))
        else:
            pred_feat_seq = []

        enc_outputs, kv_cache_tuple = self.base_lm(
            inputs_embeds=combined_embed,
            is_causal=True,
        )
        self.base_lm.kv_cache.fill_caches(kv_cache_tuple)

        enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]

        residual_enc_inputs = self.fusion_concat_proj(
            torch.cat((enc_outputs, feat_mask.unsqueeze(-1) * feat_embed), dim=-1)
        )
        residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
            inputs_embeds=residual_enc_inputs,
            is_causal=True,
        )
        self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
        residual_hidden = residual_enc_outputs[:, -1, :]

        for i in tqdm(range(max_len)):
            dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)  # [b, h_dit]
            dit_hidden_2 = self.res_to_dit_proj(residual_hidden)  # [b, h_dit]
            dit_hidden = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)

            pred_feat = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                n_timesteps=inference_timesteps,
                cfg_value=cfg_value,
            ).transpose(
                1, 2
            )  # [b, p, d]

            curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))  # b, 1, c
            curr_embed = self.enc_to_lm_proj(curr_embed)

            pred_feat_seq.append(pred_feat.unsqueeze(1))  # b, 1, p, d
            prefix_feat_cond = pred_feat

            if streaming:
                # Yield only the newest patch latent for stateful VAE decode
                feat_pred = rearrange(pred_feat.unsqueeze(1), "b t p d -> b d (t p)", b=B, p=self.patch_size)

                yield feat_pred, pred_feat_seq, context_len

                if len(pred_feat_seq) > streaming_prefix_len:
                    pred_feat_seq = pred_feat_seq[-streaming_prefix_len:]

            stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
            if i > min_len and stop_flag == 1:
                break

            lm_hidden = self.base_lm.forward_step(
                curr_embed[:, 0, :], torch.tensor([self.base_lm.kv_cache.step()], device=curr_embed.device)
            ).clone()

            lm_hidden = self.fsq_layer(lm_hidden)
            curr_residual_input = self.fusion_concat_proj(torch.cat((lm_hidden, curr_embed[:, 0, :]), dim=-1))
            residual_hidden = self.residual_lm.forward_step(
                curr_residual_input, torch.tensor([self.residual_lm.kv_cache.step()], device=curr_embed.device)
            ).clone()

        if not streaming:
            pred_feat_seq = torch.cat(pred_feat_seq, dim=1)  # b, t, p, d
            feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size)
            generated_feat = pred_feat_seq[:, context_len:, :, :].squeeze(0).cpu()
            yield feat_pred, generated_feat, context_len

    @classmethod
    def from_local(
        cls,
        path: str,
        optimize: bool = True,
        device: str | None = None,
    ):
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as _cfg_f:
            config = VoxCPMConfig.model_validate_json(_cfg_f.read())
        tokenizer = LlamaTokenizerFast.from_pretrained(path)
        audio_vae_config = getattr(config, "audio_vae_config", None)
        audio_vae = AudioVAEV2(config=audio_vae_config) if audio_vae_config else AudioVAEV2()
        # Try to load AudioVAE from safetensors first, fallback to pytorch
        audiovae_safetensors_path = os.path.join(path, "audiovae.safetensors")
        audiovae_pth_path = os.path.join(path, "audiovae.pth")
        if os.path.exists(audiovae_safetensors_path) and SAFETENSORS_AVAILABLE:
            print(f"Loading AudioVAE from safetensors: {audiovae_safetensors_path}", file=sys.stderr)
            vae_state_dict = load_file(audiovae_safetensors_path, device="cpu")
        elif os.path.exists(audiovae_pth_path):
            print(f"Loading AudioVAE from pytorch: {audiovae_pth_path}", file=sys.stderr)
            checkpoint = torch.load(
                audiovae_pth_path,
                map_location="cpu",
                weights_only=True,
            )
            vae_state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            raise FileNotFoundError(
                f"AudioVAE checkpoint not found. Expected either {audiovae_safetensors_path} or {audiovae_pth_path}"
            )
        model = cls(config, tokenizer, audio_vae, device=device)
        lm_dtype = get_dtype(model.config.dtype)
        model = model.to(lm_dtype)
        model.audio_vae = model.audio_vae.to(torch.float32)

        # Try to load from safetensors first, fallback to pytorch_model.bin
        safetensors_path = os.path.join(path, "model.safetensors")
        pytorch_model_path = os.path.join(path, "pytorch_model.bin")

        if os.path.exists(safetensors_path) and SAFETENSORS_AVAILABLE:
            print(f"Loading model from safetensors: {safetensors_path}", file=sys.stderr)
            model_state_dict = load_file(safetensors_path)
        elif os.path.exists(pytorch_model_path):
            print(f"Loading model from pytorch_model.bin: {pytorch_model_path}", file=sys.stderr)
            checkpoint = torch.load(
                pytorch_model_path,
                map_location="cpu",
                weights_only=True,
            )
            model_state_dict = checkpoint.get("state_dict", checkpoint)
        else:
            raise FileNotFoundError(f"Model file not found. Expected either {safetensors_path} or {pytorch_model_path}")

        for kw, val in vae_state_dict.items():
            model_state_dict[f"audio_vae.{kw}"] = val

        model.load_state_dict(model_state_dict, strict=False)
        return model.to(model.device).eval().optimize(disable=not optimize)
