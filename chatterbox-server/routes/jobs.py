"""Job management routes — my-jobs, cancel, delete."""

import logging

from flask import Blueprint, jsonify, session
from jobqueue import get_job_queue
from middleware import csrf_required, login_required

jobs_bp = Blueprint("jobs", __name__)
logger = logging.getLogger(__name__)


@jobs_bp.route("/api/my-jobs", methods=["GET"])
@login_required
def api_my_jobs():
    user_id = session["user_id"]
    jobs = get_job_queue().get_user_jobs(user_id)
    return jsonify({"jobs": jobs})


@jobs_bp.route("/api/jobs/<access_code>/cancel", methods=["POST"])
@login_required
@csrf_required
def api_job_cancel(access_code):
    result = get_job_queue().cancel_job(access_code)
    return jsonify(result)


@jobs_bp.route("/api/jobs/<access_code>/delete", methods=["POST"])
@login_required
@csrf_required
def api_job_delete(access_code):
    result = get_job_queue().delete_job(access_code)
    return jsonify(result)


@jobs_bp.route("/api/jobs/<access_code>/resubmit", methods=["POST"])
@login_required
@csrf_required
def api_job_resubmit(access_code):
    try:
        result = get_job_queue().resubmit_job(access_code)
        if result["success"]:
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.error(f"Error resubmitting job {access_code}: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
