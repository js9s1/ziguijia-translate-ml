"""Tests for JobQueue: add, status, cancel, resubmit, delete, checkpoints."""

import pytest
from job_types import _get_run_func
from jobqueue import (
    JobStatus,
    _get_job_type_label,
    _safe_close_proc,
    _safe_close_psutil_procs,
    get_job_queue,
)


@pytest.fixture
def jq():
    return get_job_queue()


@pytest.fixture
def user_id():
    return 1


class TestJobQueueAdd:
    def test_add_job_pending(self, jq, user_id):
        access_code = jq.add_job(
            {
                "temperature": 0.5,
                "target_language": "en",
            },
            _run_gen_audio_stub,
            user_id,
        )
        assert access_code is not None
        assert len(access_code) == 8
        assert access_code.isupper()

    def test_add_preserves_checkpoint(self, jq, user_id):
        code = jq.add_job(
            {
                "checkpoint": "",
            },
            _run_gen_audio_stub,
            user_id,
        )
        jq.set_checkpoint(code, "ocr")
        # Re-add same job (simulating resubmit)
        code2 = jq.add_job(
            {
                "access_code": code,
            },
            _run_gen_audio_stub,
            user_id,
        )
        assert code2 == code
        ckpt = jq.get_checkpoint(code)
        assert "ocr" in ckpt


class TestJobQueueStatus:
    def test_get_status_not_found(self, jq):
        assert jq.get_status("DEADBEEF") is None

    def test_get_status_after_add(self, jq, user_id):
        code = jq.add_job(
            {
                "temperature": 0.5,
            },
            _run_gen_audio_stub,
            user_id,
        )
        status = jq.get_status(code)
        assert status is not None
        assert status["access_code"] == code
        assert status["status"] == JobStatus.PENDING.value
        assert status["temperature"] == 0.5


class TestJobQueueCancel:
    def test_cancel_pending_job(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        result = jq.cancel_job(code)
        assert result["success"] is True
        status = jq.get_status(code)
        assert status["status"] == JobStatus.CANCELLED.value

    def test_cancel_nonexistent(self, jq):
        result = jq.cancel_job("NOPE1234")
        assert result["success"] is False

    def test_cancel_already_failed(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        conn = jq._get_conn()
        conn.execute(
            "UPDATE jobs SET status = ? WHERE access_code = ?",
            (JobStatus.FAILED.value, code),
        )
        conn.commit()
        result = jq.cancel_job(code)
        assert result["success"] is False


class TestJobQueueResubmit:
    def test_resubmit_failed(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        conn = jq._get_conn()
        conn.execute(
            "UPDATE jobs SET status = ? WHERE access_code = ?",
            (JobStatus.FAILED.value, code),
        )
        conn.commit()
        result = jq.resubmit_job(code)
        assert result["success"] is True
        status = jq.get_status(code)
        assert status["status"] == JobStatus.PENDING.value

    def test_resubmit_cancelled(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        conn = jq._get_conn()
        conn.execute(
            "UPDATE jobs SET status = ? WHERE access_code = ?",
            (JobStatus.CANCELLED.value, code),
        )
        conn.commit()
        result = jq.resubmit_job(code)
        assert result["success"] is True

    def test_resubmit_completed_without_edit(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        conn = jq._get_conn()
        conn.execute(
            "UPDATE jobs SET status = ? WHERE access_code = ?",
            (JobStatus.COMPLETED.value, code),
        )
        conn.commit()
        result = jq.resubmit_job(code)
        assert result["success"] is False

    def test_resubmit_nonexistent(self, jq):
        result = jq.resubmit_job("DEADBEEF")
        assert result["success"] is False

    def test_resubmit_deleted(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        conn = jq._get_conn()
        conn.execute(
            "UPDATE jobs SET status = ? WHERE access_code = ?",
            (JobStatus.DELETED.value, code),
        )
        conn.commit()
        result = jq.resubmit_job(code)
        assert result["success"] is False


class TestJobQueueDelete:
    def test_delete_sets_status(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        result = jq.delete_job(code)
        assert result["success"] is True
        status = jq.get_status(code)
        assert status["status"] == JobStatus.DELETED.value

    def test_delete_nonexistent(self, jq):
        result = jq.delete_job("DEADBEEF")
        assert result["success"] is False


class TestJobQueueCheckpoints:
    def test_set_and_get_checkpoint(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        jq.set_checkpoint(code, "ocr,translate")
        assert jq.get_checkpoint(code) == "ocr,translate"

    def test_empty_checkpoint(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        assert jq.get_checkpoint(code) == ""

    def test_invalidate_after(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        jq.set_checkpoint(code, "download,decompress,ocr,translate,audio")
        jq.invalidate_checkpoints_after(code, "ocr")
        ckpt = jq.get_checkpoint(code)
        # Should keep download,decompress,ocr, remove translate,audio
        parts = [s for s in ckpt.split(",") if s]
        assert "ocr" in parts
        assert "translate" not in parts
        assert "audio" not in parts

    def test_checkpoint_edited(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        assert jq.get_checkpoint_edited(code) is False
        jq.set_checkpoint_edited(code, True)
        assert jq.get_checkpoint_edited(code) is True

    def test_edited_srt_files(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        assert jq.get_edited_srt_files(code) == []
        jq.set_edited_srt_file(code, "ocr_screen.srt")
        assert jq.get_edited_srt_files(code) == ["ocr_screen.srt"]
        jq.clear_edited_srt_files(code)
        assert jq.get_edited_srt_files(code) == []


class TestJobQueueGetUserJobs:
    def test_returns_user_jobs(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        jobs = jq.get_user_jobs(user_id)
        assert len(jobs) >= 1
        assert any(j["access_code"] == code for j in jobs)

    def test_excludes_deleted(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        jq.delete_job(code)
        jobs = jq.get_user_jobs(user_id)
        assert not any(j["access_code"] == code for j in jobs)


class TestJobQueueProgress:
    def test_update_progress(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        jq.update_job_progress(code, "Step 1: downloading")
        status = jq.get_status(code)
        assert status["progress"] == "Step 1: downloading"


class TestJobQueueClear:
    def test_clear_deleted_jobs_dry_run(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        jq.delete_job(code)
        result = jq.clear_job_queue(dry_run=True)
        assert result["success"] is True
        assert result["jobs_removed"] >= 1

    def test_clear_deleted_jobs_actually_removes(self, jq, user_id):
        code = jq.add_job({}, _run_gen_audio_stub, user_id)
        jq.delete_job(code)
        result = jq.clear_job_queue(dry_run=False)
        assert result["success"] is True
        assert jq.get_status(code) is None


def _run_gen_audio_stub(job_data: dict):
    """Stub handler that does nothing — used as a mock run_func in tests."""
    pass


class TestHelperFunctions:
    def test_get_run_func_valid(self):
        func = _get_run_func("_run_gen_audio")
        assert callable(func)

    def test_get_run_func_invalid(self):
        assert _get_run_func("nonexistent_handler") is None

    def test_get_job_type_label(self):
        assert "音频" in _get_job_type_label("_run_gen_audio")

    def test_safe_close_proc_none(self):
        _safe_close_proc(None)

    def test_safe_close_psutil_empty(self):
        _safe_close_psutil_procs([])


class TestJobQueueAddJobParams:
    def test_add_job_with_all_params(self, jq, user_id):
        code = jq.add_job(
            {
                "srt_path": "/tmp/test.srt",
                "output_dir": "/tmp/out",
                "temperature": 0.5,
                "video_number": "123",
                "video_file": "video.mp4",
                "text": "test text",
                "blur": "yes",
                "target_language": "fr",
                "cfg_weight": 0.5,
                "exaggeration": 1.0,
                "start_trim": 1.0,
                "end_trim": 10.0,
                "cached_path": "/cache/v.mp4",
                "filename": "input.srt",
            },
            _run_gen_audio_stub,
            user_id,
        )
        status = jq.get_status(code)
        assert status["temperature"] == 0.5
        assert status["target_language"] == "fr"
        assert status["cfg_weight"] == 0.5
        assert status["exaggeration"] == 1.0
