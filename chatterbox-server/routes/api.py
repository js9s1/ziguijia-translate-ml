"""API and static file routes — health, languages, oldrun SRT, static files.

Must be registered last due to the catch-all ``/<path:filename>`` route.
"""

import io
import logging
import os
import zipfile

from flask import Blueprint, jsonify, request, send_file, send_from_directory
from jobqueue import get_job_queue

api_bp = Blueprint("api", __name__)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_DIR = os.path.join(BASE_DIR, "html")


# ── Health ────────────────────────────────────────────────


@api_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "message": "Server is running",
        }
    )


# ── Languages ─────────────────────────────────────────────


@api_bp.route("/api/languages", methods=["GET"])
def api_languages():
    from config import LANG_MAP

    return jsonify(
        {
            "languages": [{"code": code, "name": name} for code, name in sorted(LANG_MAP.items())],
        }
    )


# ── Generic job status ───────────────────────────────────


@api_bp.route("/api/jobs/<access_code>/status", methods=["GET"])
def api_job_status(access_code):
    status = get_job_queue().get_status(access_code)
    if not status:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(status)


# ── Oldrun SRT list API ──────────────────────────────────


@api_bp.route("/api/oldrun-srt", methods=["GET"])
def api_oldrun_srt():
    lang = request.args.get("lang", "zh")
    try:
        from oldrun import _collect_incremental

        zh, en, zh_en, _changed = _collect_incremental()
        if lang == "en":
            files = en
        elif lang == "zh+en":
            files = zh_en
        else:
            files = zh
        return jsonify({"files": files})
    except Exception as e:
        logger.error(f"Error listing oldrun srt: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/oldrun-srt/download", methods=["POST"])
def api_oldrun_srt_download():
    try:
        from oldrun import OLDRUN_SRT_DIR

        # Accept JSON body (fetch) or form-encoded (fallback)
        body = request.get_json(silent=True) or {}
        file_list = body.get("files", [])

        if not file_list:
            # Try parsing form-encoded fallback: files[0][path], files[0][name], etc.
            form_files = {}
            for key in request.form:
                # key looks like "files[0][path]" or "files[0][name]"
                if key.startswith("files[") and "][" in key:
                    try:
                        idx_str = key[key.index("[") + 1 : key.index("]")]
                        idx = int(idx_str)
                        field = key[key.index("][") + 2 : -1]  # path, name, etc.
                        if idx not in form_files:
                            form_files[idx] = {}
                        form_files[idx][field] = request.form[key]
                    except (ValueError, IndexError):
                        pass
            # Sort by index to preserve order
            file_list = [form_files[i] for i in sorted(form_files.keys())]

        if not file_list:
            return jsonify({"error": "No files specified"}), 400

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for entry in file_list:
                path = entry.get("path", "")
                name = entry.get("name", os.path.basename(path))
                real = os.path.realpath(path)
                if not real.startswith(os.path.realpath(OLDRUN_SRT_DIR) + "/"):
                    logger.warning("Blocked download path outside oldrun: %s", real)
                    continue
                if not os.path.isfile(real):
                    logger.warning("Skipping missing SRT: %s", real)
                    continue
                zf.write(real, name)

        buf.seek(0)
        return send_file(
            buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name="srts.zip",
        )
    except Exception as e:
        logger.error(f"Error downloading SRTs: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ── Static pages (GET) ───────────────────────────────────


@api_bp.route("/", methods=["GET"])
@api_bp.route("/index.html", methods=["GET"])
def index():
    return send_from_directory(HTML_DIR, "index.html")


@api_bp.route("/tts", methods=["GET"])
def tts_page():
    return send_from_directory(HTML_DIR, "ningSound.html")


@api_bp.route("/result", methods=["GET"])
def result_page():
    return send_from_directory(HTML_DIR, "result.html")


@api_bp.route("/srt", methods=["GET"])
def srt_page():
    return send_from_directory(HTML_DIR, "srt.html")


@api_bp.route("/srt-view", methods=["GET"])
def srt_view_page():
    return send_from_directory(HTML_DIR, "srt-view.html")


@api_bp.route("/video/ning", methods=["GET"])
def video_ning_page():
    return send_from_directory(HTML_DIR, "ningVideo.html")


@api_bp.route("/video/custom", methods=["GET"])
def video_custom_page():
    return send_from_directory(HTML_DIR, "userVideo.html")


@api_bp.route("/video/ocr", methods=["GET"])
def video_ocr_page():
    return send_from_directory(HTML_DIR, "videoOCR.html")


@api_bp.route("/my-jobs", methods=["GET"])
def my_jobs_page():
    return send_from_directory(HTML_DIR, "my-jobs.html")


# ── Catch-all static file serving (must be last) ─────────


@api_bp.route("/<path:filename>", methods=["GET"])
def serve_static(filename):
    resp = send_from_directory(HTML_DIR, filename)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp
