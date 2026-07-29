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


class TestFilesDelete:
    def test_unauthorized(self, client):
        resp = client.post("/files/delete", json={"path": "/tmp/x.txt"})
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
        resp = client.post("/files/srt-resubmit/ABC")
        assert resp.status_code == 401

    def test_nonexistent_job(self, auth_client, csrf_headers):
        client, _ = auth_client
        resp = client.post("/files/srt-resubmit/DEADBEEF", headers=csrf_headers)
        assert resp.status_code == 400
