"""Video processing routes — ning, custom, OCR."""

import logging
import os

from flask import Blueprint, jsonify, request, session
from lazy_imports import _lazy
from middleware import api_endpoint, csrf_required, get_video_metadata, login_required, parse_job_params, validate_file_upload

video_bp = Blueprint("video", __name__)
logger = logging.getLogger(__name__)


# ── Cache helpers ─────────────────────────────────────────


def _build_cached_response(find_cached_func, number) -> dict | None:
    """Check for a cached video file and build a response dict."""
    cached = find_cached_func(number)
    if not cached:
        return None
    meta = get_video_metadata(cached)
    meta_parts = []
    if meta.get("duration_str"):
        meta_parts.append(f"时长 {meta['duration_str']}")
    if meta.get("resolution"):
        meta_parts.append(f"分辨率 {meta['resolution']}")
    meta_str = "，".join(meta_parts) if meta_parts else ""
    basename = os.path.basename(cached)
    msg = f"已找到缓存的视频文件 ({basename})"
    if meta_str:
        msg += f"\n{meta_str}"
    msg += "\n是否使用该已下载的版本？"
    return {
        "cached_found": True,
        "paths": [cached],
        "number": number,
        "message": msg,
        "metadata": meta,
    }


def _video_process_with_cache(
    number: str,
    find_cached_func,
    process_func,
    blur: str,
    extra_process_kwargs: dict | None = None,
):
    """Handle the cached-video check → process flow."""
    cached_path = request.form.get("cached_path", "")
    if cached_path:
        kwargs = {"cached_path": cached_path}
        if extra_process_kwargs:
            kwargs.update(extra_process_kwargs)
        return process_func(**kwargs)

    no_cache_check = request.form.get("no_cache_check", "false") == "true"
    if no_cache_check:
        kwargs = {}
        if extra_process_kwargs:
            kwargs.update(extra_process_kwargs)
        return process_func(**kwargs)

    cached_resp = _build_cached_response(find_cached_func, number)
    if cached_resp is not None:
        return cached_resp

    kwargs = {}
    if extra_process_kwargs:
        kwargs.update(extra_process_kwargs)
    return process_func(**kwargs)


# ── Ning video routes ─────────────────────────────────────


@video_bp.route("/video/ning/process", methods=["POST"])
@login_required
@csrf_required
@api_endpoint
def video_ning_process():
    number = request.form.get("number")
    if not number:
        return jsonify({"error": "No video number provided"}), 400

    if "srt_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    srt_file = request.files["srt_file"]
    validate_file_upload(srt_file, "SRT")
    params = parse_job_params(request.form)

    process_video_ning = _lazy("video_ning_job", "process_video_ning")
    find_cached = _lazy("video_ning_job", "_find_cached_video")
    blur = request.form.get("blur", "yes")

    def _process(**kwargs):
        return jsonify(
            process_video_ning(
                number,
                srt_file,
                params["temperature"],
                session["user_id"],
                blur,
                target_language=params["target_language"],
                cfg_weight=params["cfg_weight"],
                exaggeration=params["exaggeration"],
                **kwargs,
            )
        )

    return _video_process_with_cache(number, find_cached, _process, blur)


@video_bp.route("/video/ning/ocr-process", methods=["POST"])
@login_required
@csrf_required
@api_endpoint
def video_ning_ocr_process():
    number = request.form.get("number")
    if not number:
        return jsonify({"error": "No video number provided"}), 400

    params = parse_job_params(request.form)
    mode = request.form.get("ocr_mode", "full")
    ocr_only = request.form.get("ocr_only", "yes")
    process_video_ning_ocr = _lazy(
        "video_ning_job", "process_video_ning_ocr" if mode == "full" else "process_video_ning_ocr_translate_only"
    )
    find_cached = _lazy("video_ning_job", "_find_cached_video")
    blur = request.form.get("blur", "yes")

    def _process(**kwargs):
        kwargs_dict = dict(
            target_language=params["target_language"],
            cfg_weight=params["cfg_weight"],
            exaggeration=params["exaggeration"],
        )
        if mode != "full":
            kwargs_dict["ocr_only"] = ocr_only
        return jsonify(
            process_video_ning_ocr(
                number,
                params["temperature"],
                session["user_id"],
                blur,
                **kwargs_dict,
                **kwargs,
            )
        )

    return _video_process_with_cache(number, find_cached, _process, blur)


@video_bp.route("/video/ning/auto-process", methods=["POST"])
@login_required
@csrf_required
@api_endpoint
def video_ning_auto_process():
    number = request.form.get("number")
    if not number:
        return jsonify({"error": "No video number provided"}), 400

    params = parse_job_params(request.form)
    process_video_ning_auto = _lazy("video_ning_job", "process_video_ning_auto")
    find_cached = _lazy("video_ning_job", "_find_cached_video")
    blur = request.form.get("blur", "yes")

    def _process(**kwargs):
        return jsonify(
            process_video_ning_auto(
                number,
                params["temperature"],
                session["user_id"],
                blur,
                target_language=params["target_language"],
                cfg_weight=params["cfg_weight"],
                exaggeration=params["exaggeration"],
                **kwargs,
            )
        )

    return _video_process_with_cache(number, find_cached, _process, blur)


# ── Custom video routes ───────────────────────────────────


@video_bp.route("/video/custom/process", methods=["POST"])
@login_required
@csrf_required
@api_endpoint
def video_custom_process():
    if "video_file" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    if "srt_file" not in request.files:
        return jsonify({"error": "No SRT file uploaded"}), 400

    video_file = request.files["video_file"]
    validate_file_upload(video_file, "video")
    srt_file = request.files["srt_file"]
    validate_file_upload(srt_file, "SRT")
    params = parse_job_params(request.form)

    process_video_custom = _lazy("video_custom_job", "process_video_custom")
    blur = request.form.get("blur", "yes")
    result = process_video_custom(
        video_file,
        srt_file,
        params["temperature"],
        session["user_id"],
        blur,
        target_language=params["target_language"],
        cfg_weight=params["cfg_weight"],
        exaggeration=params["exaggeration"],
    )
    return jsonify(result)


@video_bp.route("/video/custom/auto-process", methods=["POST"])
@login_required
@csrf_required
@api_endpoint
def video_custom_auto_process():
    if "video_file" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video_file"]
    validate_file_upload(video_file, "video")
    params = parse_job_params(request.form)

    process_video_auto = _lazy("video_custom_job", "process_video_auto")
    blur = request.form.get("blur", "yes")
    result = process_video_auto(
        video_file,
        params["temperature"],
        session["user_id"],
        blur,
        target_language=params["target_language"],
        cfg_weight=params["cfg_weight"],
        exaggeration=params["exaggeration"],
    )
    return jsonify(result)


# ── OCR routes ────────────────────────────────────────────


@video_bp.route("/video/ocr/process", methods=["POST"])
@login_required
@csrf_required
@api_endpoint
def video_ocr_process():
    if "video_file" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video_file"]
    validate_file_upload(video_file, "video")
    process_ocr_only = _lazy("video_ocr_job", "process_ocr_only")
    result = process_ocr_only(video_file, session["user_id"])
    return jsonify(result)


@video_bp.route("/video/custom/ocr-process", methods=["POST"])
@login_required
@csrf_required
@api_endpoint
def video_custom_ocr_process():
    if "video_file" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video_file"]
    validate_file_upload(video_file, "video")
    params = parse_job_params(request.form)
    mode = request.form.get("ocr_mode", "full")
    ocr_only = request.form.get("ocr_only", "yes")

    if mode == "translate-only":
        process_video_ocr = _lazy("video_custom_job", "process_video_ocr_translate_only")
    else:
        process_video_ocr = _lazy("video_custom_job", "process_video_ocr")

    kwargs = dict(
        target_language=params["target_language"],
        cfg_weight=params["cfg_weight"],
        exaggeration=params["exaggeration"],
    )
    if mode == "translate-only":
        kwargs["ocr_only"] = ocr_only

    blur = request.form.get("blur", "yes")
    result = process_video_ocr(
        video_file,
        params["temperature"],
        session["user_id"],
        blur,
        **kwargs,
    )
    return jsonify(result)
