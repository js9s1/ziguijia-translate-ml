"""SRT processing and status routes."""

import logging

from flask import Blueprint, jsonify, request, session
from middleware import login_required, parse_job_params, validate_file_upload

srt_bp = Blueprint("srt", __name__)
logger = logging.getLogger(__name__)


def _lazy(module_name: str, attr: str):
    from lazy_imports import _lazy as _l
    return _l(module_name, attr)


@srt_bp.route("/srt/process", methods=["POST"])
@login_required
def srt_process():
    try:
        if "srt_file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        srt_file = request.files["srt_file"]
        validate_file_upload(srt_file, "SRT")
        params = parse_job_params(request.form)

        process_srt_file = _lazy("audio_job", "process_srt_file")
        result = process_srt_file(srt_file, params["temperature"], session["user_id"],
                                  target_language=params["target_language"], cfg_weight=params["cfg_weight"],
                                  exaggeration=params["exaggeration"])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing SRT: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@srt_bp.route("/srt/status/<access_code>", methods=["GET"])
def srt_status(access_code):
    from jobqueue import get_job_queue
    status = get_job_queue().get_status(access_code)
    if not status:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(status)


@srt_bp.route("/srt/resubmit/<access_code>", methods=["POST"])
def srt_resubmit(access_code):
    try:
        from jobqueue import get_job_queue
        result = get_job_queue().resubmit_job(access_code)
        if result["success"]:
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.error(f"Error resubmitting job {access_code}: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
