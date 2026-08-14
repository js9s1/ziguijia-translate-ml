import io

# ── GPU / model management lives in gpu_manage.py ──
import gpu_manage as _gm
import soundfile as sf
import srt
import torch
from config import AUDIO_PROMPT_PATH
from singleton import singleton
# read_srt_text imported lazily in load_subs() to avoid pulling in
# the full server dependency chain (video_util → jobqueue → flask)
# when gen_audio runs as a standalone Python 3.11 subprocess.

_default_audio_prompt_path = AUDIO_PROMPT_PATH


@singleton
class NingAudio:
    def __init__(self, audio_prompt: str | None = None):
        global _default_audio_prompt_path
        self.audio_prompt_path = audio_prompt if audio_prompt else _default_audio_prompt_path
        self.model = None
        self.sample_rate = None

    def _ensure_model(self, target_language: str = "en") -> None:
        """Lazy GPU swap: only reloads if *target_language* differs from
        what is currently on GPU.  No-op if the correct model is already
        loaded."""
        _gm._acquire_gpu_for(target_language)
        if target_language == "id":
            self.model = None  # break reference so old model can be GC'd
            self.sample_rate = _gm._indonesian_model.sr if _gm._indonesian_model is not None else None
            return
        # Multilingual path: sync instance fields to the global
        self.model = _gm._model
        self.sample_rate = _gm._model.sr if _gm._model is not None else None

    def setup(self, device: str = "cuda"):
        """Load the model for English (backward-compat; prefer _ensure_model)."""
        self._ensure_model("en")

    def get_model(self, device: str = "cuda"):
        """Return the multilingual model (backward-compat; prefer _ensure_model)."""
        self._ensure_model("en")
        return self.model

    def wav_to_bytes(self, wav: torch.Tensor, sample_rate: int) -> io.BytesIO:
        buffer = io.BytesIO()
        sf.write(buffer, wav.squeeze(0).cpu().numpy(), sample_rate, format="wav")
        buffer.seek(0)
        return buffer

    def text_to_wave(
        self,
        text: str,
        prompt_file: str | None = None,
        temperature: float = 0.6,
        target_language: str = "en",
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5,
    ) -> io.BytesIO:
        if target_language == "id":
            self.model = None  # break ref to old model
            _gm._acquire_gpu_for("id")
            wav = _gm._generate_indonesian(text, prompt_file=prompt_file, temperature=temperature)
            return self.wav_to_bytes(wav, _gm._indonesian_model.sr)

        self._ensure_model(target_language)
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
        from video_util import read_srt_text

        return list(srt.parse(read_srt_text(srt_path)))

    def _generate_with_retry(
        self,
        text,
        target_language,
        temperature,
        cfg_weight,
        exaggeration,
        attempts=3,
    ):
        for attempt in range(attempts):
            try:
                return self.model.generate(
                    text,
                    language_id=target_language,
                    temperature=temperature + attempt * 0.02,
                    cfg_weight=cfg_weight,
                    exaggeration=exaggeration,
                )
            except Exception as e:
                if "not finite" not in str(e):
                    raise
                print(
                    f"Non-finite audio buffer (attempt {attempt + 1}/{attempts}), "
                    f"retrying with temperature={temperature + (attempt + 1) * 0.02}"
                )
        raise RuntimeError(f"model produced non-finite audio {attempts} times")

    def generate_audio(
        self,
        text,
        output_path,
        sample_rate,
        temperature=None,
        prompt_file=None,
        target_language="en",
        cfg_weight=0.5,
        exaggeration=0.5,
    ):
        # ── Indonesian fine-tuned model path ──
        if target_language == "id":
            self.model = None  # break ref to old model
            _gm._acquire_gpu_for("id")
            temp = temperature if temperature is not None else 0.6
            wav = _gm._generate_indonesian(text, prompt_file=prompt_file, temperature=temp)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            sf.write(output_path, wav.squeeze(0).cpu().numpy(), _gm._indonesian_model.sr)
            wav_duration = wav.shape[1] / _gm._indonesian_model.sr
            return wav, wav_duration

        if temperature is None:
            temperature = 0.6
        self._ensure_model(target_language)
        if prompt_file:
            self.model.prepare_conditionals(prompt_file)
        wav = self._generate_with_retry(
            text,
            target_language,
            temperature,
            cfg_weight,
            exaggeration,
        )
        # Ensure wav is 2D tensor [1, samples]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        sf.write(output_path, wav.squeeze(0).cpu().numpy(), sample_rate)
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
                new_combined[:, : combined.shape[1]] = combined
                combined = new_combined
            combined[:, start_sample:end_sample] = wav_data
        return combined

    def save_audio(self, output_path, wav_tensor, sample_rate):
        sf.write(output_path, wav_tensor.squeeze(0).cpu().numpy(), sample_rate)
