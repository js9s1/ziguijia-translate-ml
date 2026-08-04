"""File management routes — list, read, download, delete, SRT save."""

import logging
import os

from config import FILENAME_TO_CHECKPOINT_STEP
from flask import Blueprint, jsonify, request, send_file
from jobqueue import get_job_queue
from middleware import csrf_required, login_required, safe_file_path, validate_srt_content

files_bp = Blueprint("files", __name__)
logger = logging.getLogger(__name__)


def _allowed_dirs():
    from middleware import ALLOWED_FILE_DIRS

    return ALLOWED_FILE_DIRS


@files_bp.route("/files/list", methods=["GET"])
@login_required
def files_list():
    try:
        dir_path = request.args.get("dir")
        if not dir_path:
            return jsonify({"error": "No directory specified"}), 400

        resolved = os.path.realpath(dir_path)
        allowed = False
        for d in _allowed_dirs():
            if resolved.startswith(d + "/") or resolved == d:
                allowed = True
                break
        if not allowed:
            return jsonify({"error": "Directory not allowed"}), 403

        if not os.path.exists(resolved) or not os.path.isdir(resolved):
            return jsonify({"error": "Directory not found"}), 404

        files = []
        for name in os.listdir(resolved):
            if name.startswith("."):
                continue
            file_path = os.path.join(resolved, name)
            if not os.path.isfile(file_path):
                continue
            files.append({"name": name, "size": os.path.getsize(file_path)})

        return jsonify({"files": files})
    except Exception as e:
        logger.error(f"Error listing files: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@files_bp.route("/files/read", methods=["GET"])
@login_required
def files_read():
    try:
        file_path = request.args.get("path")
        if not file_path:
            return jsonify({"error": "No file specified"}), 400

        safe = safe_file_path(file_path)
        if not safe:
            return jsonify({"error": "File not allowed or not found"}), 404

        with open(safe, encoding="utf-8") as f:
            content = f.read()
        return content, 200, {"Content-Type": "text/plain; charset=utf-8"}
    except Exception as e:
        logger.error(f"Error reading file: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@files_bp.route("/files/download", methods=["GET"])
@login_required
def files_download():
    try:
        file_path = request.args.get("path")
        if not file_path:
            return jsonify({"error": "No file specified"}), 400

        safe = safe_file_path(file_path)
        if not safe:
            return jsonify({"error": "File not allowed or not found"}), 404

        return send_file(safe, as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@files_bp.route("/files/delete", methods=["POST"])
@login_required
@csrf_required
def files_delete():
    try:
        data = request.get_json(silent=True) or {}
        file_path = data.get("path")
        if not file_path:
            return jsonify({"success": False, "error": "No file specified"}), 400

        safe = safe_file_path(file_path)
        if not safe:
            return jsonify({"success": False, "error": "File not allowed or not found"}), 404

        os.remove(safe)

        access_code = data.get("access_code")
        if access_code:
            get_job_queue().clear_checkpoint_for_file(access_code, file_path)

        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting file: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ── SRT save/edit endpoint ────────────────────────────────


@files_bp.route("/files/save-srt", methods=["POST"])
@login_required
@csrf_required
def files_save_srt():
    try:
        data = request.get_json(silent=True) or {}
        file_path = data.get("path")
        content = data.get("content")
        access_code = data.get("access_code")

        if not file_path or content is None:
            return jsonify({"success": False, "error": "Missing path or content"}), 400

        safe = safe_file_path(file_path)
        if not safe:
            return jsonify({"success": False, "error": "File not allowed or not found"}), 404

        if not safe.lower().endswith(".srt"):
            return jsonify({"success": False, "error": "Only .srt files can be saved via this endpoint"}), 400

        try:
            validate_srt_content(content, "SRT")
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 400

        with open(safe, "w", encoding="utf-8") as f:
            f.write(content)

        jq = get_job_queue()
        basename = os.path.basename(safe)
        step = FILENAME_TO_CHECKPOINT_STEP.get(basename)
        if step:
            jq.invalidate_checkpoints_after(access_code, step)

        if access_code:
            jq.set_checkpoint_edited(access_code, True)
            jq.set_edited_srt_file(access_code, basename)

        return jsonify({"success": True, "message": "File saved, downstream steps marked for re-run"})
    except Exception as e:
        logger.error(f"Error saving SRT: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@files_bp.route("/files/srt-resubmit/<access_code>", methods=["POST"])
@login_required
@csrf_required
def files_srt_resubmit(access_code):
    try:
        jq = get_job_queue()
        result = jq.resubmit_job(access_code)
        if result["success"]:
            jq.set_checkpoint_edited(access_code, False)
            jq.clear_edited_srt_files(access_code)
            return jsonify(result)
        return jsonify(result), 400
    except Exception as e:
        logger.error(f"Error resubmitting SRT-edited job {access_code}: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
