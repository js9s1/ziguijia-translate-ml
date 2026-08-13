"""Tests for API endpoints: health, languages, job status, static pages."""

import pytest


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert "message" in data


class TestLanguagesEndpoint:
    def test_returns_language_list(self, client):
        resp = client.get("/api/languages")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "languages" in data
        langs = data["languages"]
        assert len(langs) > 5
        codes = {l["code"] for l in langs}
        assert "en" in codes
        assert "zh" in codes


class TestStaticPages:
    def test_index(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_tts_page(self, client):
        resp = client.get("/tts")
        assert resp.status_code == 200

    def test_result_page(self, client):
        resp = client.get("/result")
        assert resp.status_code == 200

    def test_srt_page(self, client):
        resp = client.get("/srt")
        assert resp.status_code == 200

    def test_my_jobs_page(self, client):
        resp = client.get("/my-jobs")
        assert resp.status_code == 200


class TestJobStatus:
    def test_nonexistent_job(self, client):
        resp = client.get("/api/jobs/DEADBEEF/status")
        assert resp.status_code == 404


@pytest.fixture
def _stub_tts(monkeypatch):
    import lazy_imports

    import tts_job

    calls = {}

    def fake_process_tts(text, filename, user_id, **params):
        calls["text"] = text
        calls["params"] = params
        return {"access_code": "TESTCODE", "message": "Job queued successfully"}

    monkeypatch.setattr(tts_job, "process_tts", fake_process_tts)
    lazy_imports._MODULES.pop("tts_job.process_tts", None)
    return calls


class TestTtsProcess:
    def test_form_data_submission(self, auth_client, _stub_tts):
        client, _ = auth_client
        resp = client.post(
            "/tts/process",
            data={
                "text": "hello world",
                "filename": "out.wav",
                "temperature": "0.6",
                "target_language": "en",
                "cfg_weight": "0.25",
                "exaggeration": "0.3",
                "csrf_token": "test-csrf-token",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["access_code"] == "TESTCODE"
        assert _stub_tts["text"] == "hello world"
        assert _stub_tts["params"]["temperature"] == 0.6

    def test_form_data_submission_redirects_for_browser(self, auth_client, _stub_tts):
        client, _ = auth_client
        resp = client.post(
            "/tts/process",
            data={"text": "hello world", "csrf_token": "test-csrf-token"},
        )
        assert resp.status_code == 302
        assert resp.headers["Location"] == "/result?code=TESTCODE"

    def test_json_submission(self, auth_client, _stub_tts, csrf_headers):
        client, _ = auth_client
        headers = {**csrf_headers, "X-Requested-With": "XMLHttpRequest"}
        resp = client.post(
            "/tts/process",
            json={"text": "hello json", "filename": "out.wav"},
            headers=headers,
        )
        assert resp.status_code == 200
        assert _stub_tts["text"] == "hello json"

    def test_missing_text(self, auth_client, csrf_headers):
        client, _ = auth_client
        headers = {**csrf_headers, "X-Requested-With": "XMLHttpRequest"}
        resp = client.post("/tts/process", json={}, headers=headers)
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "Missing text"


class TestJobManagementUnauthorized:
    def test_my_jobs_unauthorized(self, client):
        resp = client.get("/api/my-jobs")
        assert resp.status_code == 401

    def test_cancel_unauthorized(self, client):
        resp = client.post(
            "/api/jobs/X/cancel",
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 401

    def test_resubmit_unauthorized(self, client):
        resp = client.post(
            "/api/jobs/X/resubmit",
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 401


class TestCatchAllStatic:
    def test_serves_css(self, client):
        resp = client.get("/ning.css")
        assert resp.status_code == 200
        assert resp.headers["Cache-Control"] == "no-cache, no-store, must-revalidate"

    def test_serves_js(self, client):
        resp = client.get("/utils.js")
        assert resp.status_code == 200

    def test_404_on_missing_file(self, client):
        resp = client.get("/nonexistent_file_12345.xyz")
        assert resp.status_code == 404
