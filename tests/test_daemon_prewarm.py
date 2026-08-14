"""Tests for daemon_prewarm — job-type → daemon mapping (no I/O)."""

from unittest import mock

import pytest
from daemon_prewarm import _run_prewarm, prewarm_choices, prewarm_for_job


class TestPrewarmChoices:
    @pytest.mark.parametrize(
        "run_func",
        [
            "_run_gen_audio",
            "_run_tts_job",
            "_run_audio_segmentation_job",
        ],
    )
    def test_audio_only_jobs_need_tts(self, run_func):
        needs_tts, needs_translate = prewarm_choices(run_func, {})
        assert needs_tts is True
        assert needs_translate is False

    @pytest.mark.parametrize(
        "run_func",
        [
            "_run_video_job",
            "_run_video_custom_job",
        ],
    )
    def test_video_jobs_without_translate_need_tts_only(self, run_func):
        needs_tts, needs_translate = prewarm_choices(run_func, {})
        assert (needs_tts, needs_translate) == (True, False)

    @pytest.mark.parametrize(
        "run_func",
        [
            "_run_video_auto_job",
            "_run_video_ocr_job",
            "_run_video_ning_ocr_job",
            "_run_video_ning_auto_job",
        ],
    )
    def test_full_pipeline_jobs_need_both(self, run_func):
        needs_tts, needs_translate = prewarm_choices(run_func, {})
        assert (needs_tts, needs_translate) == (True, True)

    @pytest.mark.parametrize(
        "run_func",
        [
            "_run_video_ocr_translate_only_job",
            "_run_video_ning_ocr_translate_only_job",
        ],
    )
    def test_translate_only_jobs_need_translate(self, run_func):
        needs_tts, needs_translate = prewarm_choices(run_func, {"ocr_only": "no"})
        assert (needs_tts, needs_translate) == (False, True)

    @pytest.mark.parametrize(
        "run_func",
        [
            "_run_video_ocr_translate_only_job",
            "_run_video_ning_ocr_translate_only_job",
        ],
    )
    def test_ocr_only_jobs_need_nothing(self, run_func):
        needs_tts, needs_translate = prewarm_choices(run_func, {"ocr_only": "yes"})
        assert (needs_tts, needs_translate) == (False, False)

    def test_ocr_extract_job_needs_nothing(self):
        assert prewarm_choices("_run_ocr_only_job", {}) == (False, False)

    def test_unknown_job_needs_nothing(self):
        assert prewarm_choices("_run_unknown_job", {}) == (False, False)


class TestRunPrewarm:
    def test_worker_calls_only_tts_for_audio_job(self, monkeypatch):
        tts_calls = []
        translate_calls = []
        monkeypatch.setattr("daemon_prewarm.prewarm_tts", lambda lang: tts_calls.append(lang))
        monkeypatch.setattr("daemon_prewarm.prewarm_translate", lambda: translate_calls.append(1))
        _run_prewarm("_run_gen_audio", {"target_language": "zh"})
        assert tts_calls == ["zh"]
        assert translate_calls == []

    def test_worker_calls_both_for_full_pipeline(self, monkeypatch):
        tts_calls = []
        translate_calls = []
        monkeypatch.setattr("daemon_prewarm.prewarm_tts", lambda lang: tts_calls.append(lang))
        monkeypatch.setattr("daemon_prewarm.prewarm_translate", lambda: translate_calls.append(1))
        _run_prewarm("_run_video_auto_job", {"target_language": "en"})
        assert tts_calls == ["en"]
        assert translate_calls == [1]

    def test_worker_defaults_language_to_en(self, monkeypatch):
        tts_calls = []
        monkeypatch.setattr("daemon_prewarm.prewarm_tts", lambda lang: tts_calls.append(lang))
        monkeypatch.setattr("daemon_prewarm.prewarm_translate", lambda: None)
        _run_prewarm("_run_tts_job", {})
        assert tts_calls == ["en"]

    def test_worker_skips_translate_when_ocr_only(self, monkeypatch):
        tts_calls = []
        translate_calls = []
        monkeypatch.setattr("daemon_prewarm.prewarm_tts", lambda lang: tts_calls.append(lang))
        monkeypatch.setattr("daemon_prewarm.prewarm_translate", lambda: translate_calls.append(1))
        _run_prewarm("_run_video_ocr_translate_only_job", {"ocr_only": "yes"})
        assert tts_calls == []
        assert translate_calls == []


class TestPrewarmForJob:
    def test_no_thread_for_jobs_that_need_nothing(self, monkeypatch):
        patched = mock.patch("daemon_prewarm.threading.Thread")
        thread_cls = patched.start()
        monkeypatch.setattr(thread_cls, "_test_patch", True)
        prewarm_for_job("_run_ocr_only_job", {})
        assert thread_cls.call_count == 0
        patched.stop()

    def test_spawns_thread_for_tts_job(self, monkeypatch):
        patched = mock.patch("daemon_prewarm.threading.Thread")
        thread_cls = patched.start()
        prewarm_for_job("_run_gen_audio", {"target_language": "zh"})
        assert thread_cls.call_count == 1
        kwargs = thread_cls.call_args.kwargs
        assert kwargs.get("daemon") is True
        assert kwargs.get("target").__name__ == "_worker"
        patched.stop()
