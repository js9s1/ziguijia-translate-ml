"""Tests for API endpoints: health, languages, job status, static pages."""

import json

import pytest


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert "cuda" in data


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


class TestJobManagementUnauthorized:
    def test_my_jobs_unauthorized(self, client):
        resp = client.get("/api/my-jobs")
        assert resp.status_code == 401

    def test_cancel_unauthorized(self, client):
        resp = client.post("/api/jobs/X/cancel")
        assert resp.status_code == 401

    def test_resubmit_unauthorized(self, client):
        resp = client.post("/api/jobs/X/resubmit")
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
