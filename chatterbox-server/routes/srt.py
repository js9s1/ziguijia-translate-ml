"""SRT processing and status routes."""

from flask import Blueprint, jsonify, request, session
from jobqueue import get_job_queue
from lazy_imports import _lazy
from middleware import api_endpoint, csrf_required, login_required, parse_job_params, validate_file_upload

srt_bp = Blueprint("srt", __name__)


@srt_bp.route("/srt/process", methods=["POST"])
@login_required
@csrf_required
@api_endpoint
def srt_process():
    if "srt_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    srt_file = request.files["srt_file"]
    validate_file_upload(srt_file, "SRT")
    params = parse_job_params(request.form)

    process_srt_file = _lazy("audio_job", "process_srt_file")
    result = process_srt_file(
        srt_file,
        params["temperature"],
        session["user_id"],
        target_language=params["target_language"],
        cfg_weight=params["cfg_weight"],
        exaggeration=params["exaggeration"],
    )
    return jsonify(result)


@srt_bp.route("/srt/status/<access_code>", methods=["GET"])
def srt_status(access_code):
    status = get_job_queue().get_status(access_code)
    if not status:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(status)


@srt_bp.route("/srt/resubmit/<access_code>", methods=["POST"])
@csrf_required
@api_endpoint
def srt_resubmit(access_code):
    checkpoint = None
    body = request.get_json(silent=True)
    if isinstance(body, dict) and isinstance(body.get("checkpoint"), str):
        checkpoint = body["checkpoint"]
    result = get_job_queue().resubmit_job(access_code, checkpoint=checkpoint)
    if result["success"]:
        return jsonify(result)
    return jsonify(result), 400
