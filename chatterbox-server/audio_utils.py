import io
import os
import sys
import threading
from typing import Optional

import srt
import torch
import torchaudio as ta

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from config import AUDIO_PROMPT_PATH


# HF_TOKEN must be set via environment variable for HuggingFace authentication
# Example: export HF_TOKEN="your_token_here"


class NingAudio:
    _instance = None
    _lock = threading.Lock()
    _model = None
    _default_audio_prompt_path = AUDIO_PROMPT_PATH

    def __new__(cls, audio_prompt: Optional[str] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, audio_prompt: Optional[str] = None):
        if self._initialized:
            return
        self._initialized = True
        self.audio_prompt_path = audio_prompt if audio_prompt else NingAudio._default_audio_prompt_path
        self.model = None
        self.sample_rate = None

    def get_model(self, device: str = "cpu") -> ChatterboxMultilingualTTS:
        if NingAudio._model is None:
            import warnings
            warnings.filterwarnings("ignore")
            NingAudio._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            NingAudio._model.prepare_conditionals(self.audio_prompt_path)
        self.model = NingAudio._model
        self.sample_rate = self.model.sr
        return self.model

    def setup(self, device: str = "cpu"):
        self.get_model(device)

    def wav_to_bytes(self, wav: torch.Tensor, sample_rate: int) -> io.BytesIO:
        buffer = io.BytesIO()
        ta.save(buffer, wav, sample_rate, format="wav")
        buffer.seek(0)
        return buffer

    def text_to_wave(
        self,
        text: str,
        prompt_file: Optional[str] = None,
        temperature: float = 0.6,
        target_language: str = "en",
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
    ) -> io.BytesIO:
        self.setup()
        if prompt_file:
            self.model.prepare_conditionals(prompt_file)
        wav = self.model.generate(
            text,
            language_id=target_language,
            temperature=temperature,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
        )
        return self.wav_to_bytes(wav, self.model.sr)

    def generate_silence(self, duration_sec, sample_rate):
        num_frames = int(duration_sec * sample_rate)
        return torch.zeros(1, num_frames)

    def load_subs(self, srt_path):
        with open(srt_path, "r", encoding="utf-8") as f:
            return list(srt.parse(f.read()))

    def generate_audio(self, text, output_path, sample_rate, temperature=None, prompt_file=None, target_language="en", cfg_weight=0.5, exaggeration=0.5):
        if temperature is None:
            temperature = self.model.temperature
        if prompt_file:
            self.model.prepare_conditionals(prompt_file)
        wav = self.model.generate(text, language_id=target_language, temperature=temperature, cfg_weight=cfg_weight, exaggeration=exaggeration)
        # Ensure wav is 2D tensor [1, samples]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        ta.save(output_path, wav, sample_rate)
        wav_duration = wav.shape[1] / sample_rate
        return wav, wav_duration

    def combine_audio_segments(self, segments_info, total_duration, sample_rate):
        # Calculate actual needed duration from segments
        max_end = 0
        for seg in segments_info:
            end_sample = int((seg["new_start"] + seg["wav_duration"]) * sample_rate)
            if end_sample > max_end:
                max_end = end_sample

        # Use max of original duration and actual needed
        needed_duration = max_end / sample_rate
        actual_duration = max(total_duration, needed_duration)

        if actual_duration <= 0:
            actual_duration = 1

        combined = self.generate_silence(actual_duration, sample_rate)
        for seg in segments_info:
            wav_data = seg["wav_data"]
            if wav_data.dim() == 1:
                wav_data = wav_data.unsqueeze(0)
            start_sample = int(seg["new_start"] * sample_rate)
            end_sample = start_sample + wav_data.shape[1]
            if end_sample > combined.shape[1]:
                new_combined = torch.zeros(1, end_sample)
                new_combined[:, :combined.shape[1]] = combined
                combined = new_combined
            combined[:, start_sample:end_sample] = wav_data
        return combined

    def save_audio(self, output_path, wav_tensor, sample_rate):
        ta.save(output_path, wav_tensor, sample_rate)
