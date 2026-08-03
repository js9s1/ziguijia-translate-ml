"""TTS + Audio processing routes."""

import json
import logging

from flask import Blueprint, Response, jsonify, request, session
from jobqueue import get_job_queue
from lazy_imports import _lazy
from middleware import login_required, parse_job_params, validate_file_upload

tts_bp = Blueprint("tts", __name__)
logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 500


# ── TTS routes ────────────────────────────────────────────


@tts_bp.route("/tts/process", methods=["POST"])
@login_required
def tts_process():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Missing text"}), 400
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({"error": f"文字长度超过限制（最多{MAX_TEXT_LENGTH}字符）"}), 400
        process_tts = _lazy("tts_job", "process_tts")
        params = parse_job_params(data)
        result = process_tts(
            text,
            data.get("filename", "output.wav"),
            session["user_id"],
            temperature=params["temperature"],
            target_language=params["target_language"],
            cfg_weight=params["cfg_weight"],
            exaggeration=params["exaggeration"],
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"TTS process error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@tts_bp.route("/tts/status/<access_code>", methods=["GET"])
def tts_status(access_code):
    status = get_job_queue().get_status(access_code)
    if not status:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(status)


@tts_bp.route("/tts/status-stream/<access_code>", methods=["GET"])
def tts_status_stream(access_code):
    """SSE endpoint: pushes job status updates until the job finishes."""

    def generate():
        from valkey_util import get_valkey

        r = get_valkey()
        if r is None:
            status = get_job_queue().get_status(access_code)
            yield f"data: {json.dumps(status or {'error': 'Job not found'})}\n\n"
            return

        pubsub = r.pubsub()
        pubsub.subscribe(f"job:{access_code}")
        terminal = {"completed", "failed", "cancelled", "deleted"}

        status = get_job_queue().get_status(access_code)
        if status:
            yield f"data: {json.dumps(status)}\n\n"
            if status.get("status") in terminal:
                pubsub.unsubscribe()
                return

        for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                data = json.loads(message["data"])
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("status") in terminal:
                    break
            except (json.JSONDecodeError, TypeError):
                pass
        pubsub.unsubscribe()

    return Response(
        generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── Audio routes ──────────────────────────────────────────


@tts_bp.route("/audio/process", methods=["POST"])
@login_required
def audio_process():
    try:
        if "file" in request.files:
            file = request.files["file"]
            validate_file_upload(file, "audio")
            file.seek(0)
            content = file.read().decode("utf-8")
            params = parse_job_params(request.form)
            process_audio_file = _lazy("audio_job", "process_audio_file")
            result = process_audio_file(
                content,
                file.filename,
                params["temperature"],
                session["user_id"],
                target_language=params["target_language"],
                cfg_weight=params["cfg_weight"],
                exaggeration=params["exaggeration"],
            )
            return jsonify(result)
        else:
            data = request.get_json(silent=True) or {}
            text = data.get("text", "")
            if not text:
                return jsonify({"error": "Missing text"}), 400
            if len(text) > MAX_TEXT_LENGTH:
                return jsonify({"error": f"文字长度超过限制（最多{MAX_TEXT_LENGTH}字符）"}), 400
            process_tts = _lazy("tts_job", "process_tts")
            params = parse_job_params(data)
            result = process_tts(
                text,
                data.get("filename", "output.wav"),
                session["user_id"],
                temperature=params["temperature"],
                target_language=params["target_language"],
                cfg_weight=params["cfg_weight"],
                exaggeration=params["exaggeration"],
            )
            return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Audio process error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
