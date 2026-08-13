"""Simple test: submit a TTS job with English text 'love to code'."""

import pytest


@pytest.fixture
def _stub_tts(monkeypatch):
    import lazy_imports
    import tts_job

    calls = {}

    def fake_process_tts(text, filename, user_id, **params):
        calls["text"] = text
        calls["filename"] = filename
        calls["params"] = params
        return {"access_code": "LOVETOCODE", "message": "Job queued successfully"}

    monkeypatch.setattr(tts_job, "process_tts", fake_process_tts)
    lazy_imports._MODULES.pop("tts_job.process_tts", None)
    return calls


class TestTtsSubmitLoveToCode:
    def test_submit_form_data(self, auth_client, _stub_tts):
        client, _ = auth_client
        resp = client.post(
            "/tts/process",
            data={
                "text": "love to code",
                "filename": "love_to_code.wav",
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
        assert data["access_code"] == "LOVETOCODE"
        assert _stub_tts["text"] == "love to code"
        assert _stub_tts["filename"] == "love_to_code.wav"
        assert _stub_tts["params"]["target_language"] == "en"

    def test_submit_json(self, auth_client, _stub_tts, csrf_headers):
        client, _ = auth_client
        headers = {**csrf_headers, "X-Requested-With": "XMLHttpRequest"}
        resp = client.post(
            "/tts/process",
            json={"text": "love to code", "filename": "love_to_code.wav"},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.get_json()["access_code"] == "LOVETOCODE"
        assert _stub_tts["text"] == "love to code"
