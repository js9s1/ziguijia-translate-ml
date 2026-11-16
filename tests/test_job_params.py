"""Tests for engine-specific audio parameter schemas and normalization."""

from job_params import (
    get_audio_param_schema,
    get_job_params,
    get_output_params,
    get_video_params,
    normalize_audio_params,
    normalize_job_params,
    resolve_audio_engine,
)


class TestResolveAudioEngine:
    def test_cantonese_uses_voxcpm(self):
        assert resolve_audio_engine("yue") == "voxcpm"
        assert resolve_audio_engine("zh-HK") == "voxcpm"

    def test_indonesian_uses_chatterbox_id(self):
        assert resolve_audio_engine("id") == "chatterbox_id"

    def test_other_languages_use_chatterbox(self):
        assert resolve_audio_engine("en") == "chatterbox"
        assert resolve_audio_engine("zh") == "chatterbox"


class TestAudioParamSchema:
    def test_cantonese_schema(self):
        schema = get_audio_param_schema("tts", "yue")
        assert schema["engine"] == "voxcpm"
        keys = {field["key"] for field in schema["fields"]}
        assert keys == {"cfg_value", "inference_timesteps", "seed", "retry_badcase"}

    def test_english_schema(self):
        schema = get_audio_param_schema("srt", "en")
        assert schema["engine"] == "chatterbox"
        keys = {field["key"] for field in schema["fields"]}
        assert keys == {"temperature", "cfg_weight", "exaggeration"}

    def test_unknown_pipeline_rejected(self):
        try:
            get_audio_param_schema("not-a-pipeline", "en")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown pipeline should be rejected")


class TestNormalizeAudioParams:
    def test_engine_is_derived_from_language(self):
        params = normalize_audio_params(
            '{"engine": "chatterbox", "cfg_value": 2.5, "inference_timesteps": 12}',
            "yue",
        )
        assert params["engine"] == "voxcpm"
        assert params["cfg_value"] == 2.5
        assert params["inference_timesteps"] == 12
        assert params["seed"] == 42

    def test_values_are_clamped(self):
        params = normalize_audio_params(
            {"cfg_value": 99, "inference_timesteps": 1},
            "yue",
        )
        assert params["cfg_value"] == 3.0
        assert params["inference_timesteps"] == 5

    def test_legacy_flat_fallback(self):
        params = normalize_audio_params(
            None,
            "en",
            fallback={"temperature": 0.7, "cfg_weight": 0.3, "exaggeration": 0.4},
        )
        assert params["engine"] == "chatterbox"
        assert params["temperature"] == 0.7
        assert params["cfg_weight"] == 0.3
        assert params["exaggeration"] == 0.4

    def test_invalid_json_rejected(self):
        try:
            normalize_audio_params("{broken", "en")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid JSON should be rejected")


class TestJobEnvelope:
    def test_nested_audio_video_output(self):
        params = normalize_job_params(
            {
                "audio": {"cfg_value": 2.5, "inference_timesteps": 12},
                "video": {"mode": "srt", "blur": "no", "ocr_only": "yes"},
                "output": {"filename": "cantonese.wav"},
            },
            "yue",
        )
        assert params["target_language"] == "yue"
        assert params["audio"]["engine"] == "voxcpm"
        assert params["audio"]["cfg_value"] == 2.5
        assert params["video"]["blur"] == "no"
        assert params["output"]["filename"] == "cantonese.wav"

    def test_flat_legacy_fallback(self):
        params = normalize_job_params(
            None,
            "en",
            fallback={
                "temperature": 0.7,
                "cfg_weight": 0.3,
                "exaggeration": 0.4,
                "blur": "no",
                "ocr_only": "yes",
                "filename": "legacy.wav",
            },
        )
        assert params["audio"]["temperature"] == 0.7
        assert params["video"]["blur"] == "no"
        assert params["output"]["filename"] == "legacy.wav"

    def test_runtime_helpers_prefer_nested(self):
        job_data = {
            "target_language": "yue",
            "params": {
                "audio": {"engine": "voxcpm", "cfg_value": 2.2},
                "video": {"blur": "no"},
                "output": {"filename": "nested.wav"},
            },
            "blur": "yes",
            "filename": "flat.wav",
        }
        assert get_job_params(job_data)["audio"]["cfg_value"] == 2.2
        assert get_video_params(job_data)["blur"] == "no"
        assert get_output_params(job_data)["filename"] == "nested.wav"
