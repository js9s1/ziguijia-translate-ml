"""Shared segment-stretch logic used by both pipeline.py (audio atempo) and
gen_video.py (video setpts).  Single source of truth for the leading-gap,
subtitle, inter-subtitle-gap, trailing-gap segmentation algorithm.
"""

_SLOP = 0.04


def build_segment_defs(
    orig_starts: list[float],
    orig_ends: list[float],
    adj_starts: list[float],
    adj_ends: list[float],
    video_duration: float,
    offset: float = 0.0,
) -> list[dict]:
    """Build a list of {start, end, stretch} segment definitions.

    Returns segments that describe how to stretch each portion of the
    original timeline so that the result matches the adjusted timeline.
    Mirrors the logic originally duplicated between pipeline.py audio
    adjustment and gen_video.py video adjustment.

    Parameters
    ----------
    orig_starts, orig_ends:
        Start/end times (seconds) of each subtitle in the original SRT.
    adj_starts, adj_ends:
        Start/end times (seconds) of each subtitle in the adjusted SRT.
    video_duration:
        Total video duration (seconds).
    offset:
        Seconds to subtract from original timestamps (e.g. audio trim offset).
    """
    n = len(orig_starts)
    seg_defs = []
    video_cumul = 0.0

    # Leading gap
    first_adj_start = adj_starts[0]
    first_orig_start = orig_starts[0] - offset
    if first_adj_start > 0 and first_orig_start > 0:
        stretch = first_adj_start / first_orig_start
        seg_defs.append({"start": 0, "end": min(first_orig_start, video_duration), "stretch": stretch})
        video_cumul += first_orig_start * stretch

    for i in range(n):
        orig_start = orig_starts[i] - offset
        orig_end = orig_ends[i] - offset
        adj_start = adj_starts[i]
        adj_end = adj_ends[i]
        orig_dur = orig_end - orig_start
        adj_dur = adj_end - adj_start

        if orig_start >= video_duration:
            break
        if orig_end > video_duration:
            orig_end = video_duration
            orig_dur = orig_end - orig_start

        # Gap before this subtitle
        if i > 0:
            prev_orig_end = orig_ends[i - 1] - offset
            orig_gap = orig_start - prev_orig_end
            needed_gap = adj_start - video_cumul
            if orig_gap > 0.001:
                gap_start, gap_end = prev_orig_end, min(orig_start, video_duration)
            elif needed_gap > 0.001:
                gap_start = max(0, prev_orig_end - _SLOP)
                gap_end = min(prev_orig_end, video_duration)
                orig_gap = gap_end - gap_start
            else:
                orig_gap = 0.0
            if orig_gap > 0:
                stretch = max(0.0, min(100.0, needed_gap / orig_gap))
                seg_defs.append({"start": gap_start, "end": gap_end, "stretch": stretch})
                video_cumul += orig_gap * stretch

        # Subtitle segment (never squeeze, only stretch)
        stretch = adj_dur / orig_dur if orig_dur > 0 else 1.0
        stretch = max(1.0, min(10.0, stretch))
        seg_defs.append({"start": orig_start, "end": orig_end, "stretch": stretch})
        video_cumul += orig_dur * stretch

    # Trailing gap
    last_orig_end = orig_ends[-1] - offset
    if last_orig_end < video_duration:
        seg_defs.append({"start": last_orig_end, "end": video_duration, "stretch": 1.0})

    return seg_defs
