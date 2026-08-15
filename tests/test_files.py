"""Tests for file management routes: list, read, download, delete, SRT save."""

import pytest


@pytest.fixture(autouse=True)
def _bypass_rate_limit():
    from middleware import _ip_limiter

    old = _ip_limiter.limit
    _ip_limiter.limit = 10000
    yield
    _ip_limiter.limit = old


class TestFilesList:
    def test_no_directory(self, auth_client):
        client, _ = auth_client
        resp = client.get("/files/list")
        assert resp.status_code == 400

    def test_unauthorized(self, client):
        resp = client.get("/files/list?dir=/tmp")
        assert resp.status_code == 401

    def test_nonexistent_directory(self, auth_client):
        client, _ = auth_client
        resp = client.get("/files/list?dir=/nonexistent/dir/12345")
        assert resp.status_code in (403, 404)

    def test_disallowed_directory(self, auth_client):
        client, _ = auth_client
        resp = client.get("/files/list?dir=/etc")
        assert resp.status_code == 403


class TestFilesRead:
    def test_no_path(self, auth_client):
        client, _ = auth_client
        resp = client.get("/files/read")
        assert resp.status_code == 400

    def test_unauthorized(self, client):
        resp = client.get("/files/read?path=/tmp/x.txt")
        assert resp.status_code == 401

    def test_gbk_encoded_srt_reads_ok(self, auth_client):
        """Users upload SRTs in GBK/legacy encodings; the reader must not
        blow up with a UnicodeDecodeError (regression for access code
        841A7C64)."""
        import os

        from config import VIDEO_DIR
        from middleware import ALLOWED_FILE_DIRS

        client, _ = auth_client
        video_dir = str(VIDEO_DIR)
        os.makedirs(video_dir, exist_ok=True)
        real = os.path.realpath(video_dir)
        if real not in ALLOWED_FILE_DIRS:
            ALLOWED_FILE_DIRS.append(real)

        path = os.path.join(video_dir, "gbk.srt")
        with open(path, "wb") as fh:
            fh.write("1\n00:00:01,000 --> 00:00:02,000\n你好，世界\n".encode("gbk"))

        resp = client.get("/files/read?path=" + path)
        assert resp.status_code == 200
        assert "你好" in resp.get_data(as_text=True)


class TestFilesDelete:
    def test_unauthorized(self, client):
        resp = client.post(
            "/files/delete",
            json={"path": "/tmp/x.txt"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 401

    def test_no_path(self, auth_client, csrf_headers):
        client, _ = auth_client
        resp = client.post("/files/delete", json={}, headers=csrf_headers)
        assert resp.status_code == 400

    def test_csrf_required(self, auth_client):
        client, _ = auth_client
        resp = client.post("/files/delete", json={"path": "/tmp/x.txt"})
        assert resp.status_code == 403  # CSRF missing


class TestFilesDownload:
    def test_no_path(self, auth_client):
        client, _ = auth_client
        resp = client.get("/files/download")
        assert resp.status_code == 400


class TestSRTSave:
    def test_unauthorized(self, client):
        resp = client.post(
            "/files/save-srt",
            json={
                "path": "/tmp/x.srt",
                "content": "00:00:01,000 --> 00:00:03,000\ntest",
                "access_code": "ABC",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 401

    def test_missing_content(self, auth_client, csrf_headers):
        client, _ = auth_client
        resp = client.post(
            "/files/save-srt",
            json={
                "path": "/tmp/x.srt",
                "access_code": "ABC",
            },
            headers=csrf_headers,
        )
        assert resp.status_code == 400

    def test_non_srt_file(self, auth_client, csrf_headers):
        client, _ = auth_client
        resp = client.post(
            "/files/save-srt",
            json={
                "path": "/tmp/x.txt",
                "content": "not srt",
                "access_code": "ABC",
            },
            headers=csrf_headers,
        )
        assert resp.status_code in (400, 404)

    def test_invalid_srt_content(self, auth_client, csrf_headers, tmp_path):
        client, _ = auth_client
        f = tmp_path / "test.srt"
        f.write_text("dummy")
        resp = client.post(
            "/files/save-srt",
            json={
                "path": str(f),
                "content": "no timing line here",
                "access_code": "ABC",
            },
            headers=csrf_headers,
        )
        assert resp.status_code in (400, 404)  # path not allowed, or content invalid


class TestSRTResubmit:
    def test_unauthorized(self, client):
        resp = client.post(
            "/files/srt-resubmit/ABC",
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 401

    def test_nonexistent_job(self, auth_client, csrf_headers):
        client, _ = auth_client
        resp = client.post("/files/srt-resubmit/DEADBEEF", headers=csrf_headers)
        assert resp.status_code == 400
