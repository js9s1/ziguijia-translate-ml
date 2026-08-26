"""File management routes — list, read, download, delete, SRT save."""

import logging
import os
import re
from datetime import timedelta

from config import FILENAME_TO_CHECKPOINT_STEP, validate_upload_filename
from flask import Blueprint, jsonify, request, send_file
from jobqueue import get_job_queue
from middleware import _SRT_TIMING_RE, api_endpoint, csrf_required, login_required, safe_file_path, validate_srt_content

logger = logging.getLogger("chatterbox_server")

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")

files_bp = Blueprint("files", __name__)


def _allowed_dirs():
    from middleware import ALLOWED_FILE_DIRS

    return ALLOWED_FILE_DIRS


def _decode_bytes(raw: bytes) -> str:
    """Decode raw bytes, tolerating common encodings (UTF-8, GBK/GB_*, Big5).

    ``gb18030`` is a superset of GB2312/GBK, so it covers the whole GB_*
    family; we try it before plain ``gbk`` to survive four-byte extensions.
    UTF-8 is preferred first (SRT text is most commonly written in it).
    """
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "big5"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _decode_text_file_bytes(raw: bytes) -> str:
    """Decode bytes for SRT/plain-text uploads, normalizing line endings."""
    text = _decode_bytes(raw)
    return _normalize_line_endings(text)


def _normalize_line_endings(text: str) -> str:
    """Normalize CRLF/CR to LF and strip a UTF-8 BOM if present."""
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")


def _extract_docx_text(raw: bytes) -> str:
    """Extract paragraph text from a .docx (OOXML) archive.

    .docx files are ZIP archives containing ``word/document.xml``. We pull
    the ``w:t`` runs and join them per paragraph, keeping the document's
    wording (Chinese or bilingual) for calibration matching.
    """
    import io
    import zipfile
    from xml.etree import ElementTree as ET

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            if "word/document.xml" not in zf.namelist():
                raise ValueError("不是有效的 .docx 文件：缺少 word/document.xml")
            xml_bytes = zf.read("word/document.xml")
    except zipfile.BadZipFile as e:
        raise ValueError("不是有效的 .docx 文件") from e

    NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    root = ET.fromstring(xml_bytes)
    paragraphs = []
    for p in root.iter(NS + "p"):
        parts = []
        # Document order matters: w:t is the visible text, w:br is a line
        # break inside the paragraph, w:tab is a tab stop.
        for el in p.iter():
            if el.tag == NS + "t":
                parts.append(el.text or "")
            elif el.tag == NS + "br":
                parts.append("\n")
            elif el.tag == NS + "tab":
                parts.append("\t")
        line = "".join(parts)
        if line.strip():
            paragraphs.append(line)
    text = "\n".join(paragraphs)
    if not text.strip():
        raise ValueError("无法从 .docx 文件中提取文字，请确认文档包含文本内容")
    return _normalize_line_endings(text)


def _extract_doc_text(raw: bytes) -> str:
    """Extract text from a legacy .doc (OLE compound) file.

    Legacy Word .doc files wrap the document in an OLE container.  We read
    the ``WordDocument`` stream and walk the FIB/piece table to collect the
    visible text runs (which may be UTF-16 or the document codepage, e.g.
    GBK).  A best-effort fallback scans for printable runs if the piece
    table is unparseable.
    """
    import io

    try:
        import olefile
    except ImportError:
        raise ValueError("服务器缺少 olefile 依赖，无法解析 .doc 文件") from None

    try:
        ole = olefile.OleFileIO(io.BytesIO(raw))
    except Exception as e:
        raise ValueError("不是有效的 .doc 文件") from e

    try:
        return _extract_doc_text_from_ole(ole)
    finally:
        ole.close()


def _extract_doc_text_from_ole(ole) -> str:
    """Extract text from an already-open OLE .doc container."""
    try:
        if not ole.exists("WordDocument"):
            raise ValueError("不是有效的 .doc 文件：缺少 WordDocument 流")
        word_stream = ole.openstream("WordDocument").read()
        text = _doc_piece_text(ole, word_stream)
        # If the piece table produced no meaningful CJK/printable text, fall
        # back to a raw scan — a mismatched FIB offset must not yield an empty
        # "standard" that silently calibrates everything to silence.
        if len(_CJK_RE.findall(text)) < 2 and not re.search(r"[A-Za-z]{3,}", text):
            text = _doc_scan_text(word_stream)
    except ValueError:
        raise
    if not text.strip():
        raise ValueError("无法从 .doc 文件中提取文字，请另存为 .srt/.txt 后上传")
    return _normalize_line_endings(text)


def _doc_decode_run(run: bytes, single_byte: bool = False) -> str:
    """Decode one .doc text run, preferring UTF-16 then GB_*/Big5.

    ``single_byte=True`` marks compressed pieces (codepage chars), where
    UTF-16 would silently produce garbage from any byte pair.
    """
    if not run:
        return ""
    if not single_byte:
        try:
            return run.decode("utf-16-le")
        except UnicodeDecodeError:
            pass
    return _decode_bytes(run)


def _doc_scan_text(word_stream: bytes) -> str:
    """Fallback: scan the WordDocument stream for printable text runs.

    Word keeps the visible body text somewhere in the stream even when the
    piece-table walk fails; collect the longest plausible CJK/printable runs
    decoded as UTF-16LE or the GB_*/codepage family.
    """
    import re

    candidates = []
    # UTF-16LE interpretation: runs of plausible chars, then CJK-heavy blocks.
    try:
        as_utf16 = word_stream.decode("utf-16-le", errors="ignore")
        utf16_runs = re.findall(
            r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaffA-Za-z0-9，。！？、；：“”‘’（）《》\n]{4,}",
            as_utf16,
        )
        if utf16_runs:
            candidates.append("\n".join(utf16_runs))
    except (UnicodeDecodeError, ValueError):
        pass

    # Single-byte (GB_*/Big5/codepage): keep runs that decode cleanly.
    runs = re.findall(rb"[\x20-\x7e\x80-\xff]{4,}", word_stream)
    best_single = ""
    for run in runs:
        try:
            piece = run.decode("gb18030")
        except UnicodeDecodeError:
            continue
        printable = "".join(ch for ch in piece if ch == "\n" or ch.isprintable())
        if _CJK_RE.search(printable) and len(printable) > len(best_single):
            best_single = printable
    if best_single:
        candidates.append(best_single)

    return max(candidates, key=lambda c: len(_CJK_RE.findall(c)), default="")


def _doc_piece_text(ole, word_stream: bytes) -> str:
    """Walk the .doc FIB + piece table to gather visible text runs.

    Returns the concatenated text of all pieces.  Empty if the structure
    can't be interpreted (caller falls back to a raw scan).
    """
    import struct

    try:
        if len(word_stream) < 0x200:
            return ""
        # FibBase: 32 bytes; flags at 0x0A. fComplex = bit 2 (0x0004),
        # fWhichTblStm = bit 8 (0x0100).
        flags = struct.unpack_from("<H", word_stream, 0x0A)[0]
        # FibRgW starts after FibBase; csw counts its 16-bit fields.
        csw = struct.unpack_from("<H", word_stream, 0x20)[0]
        fib_rglw = 0x22 + csw * 2  # after FibBase + csw + FibRgW
        cslw = struct.unpack_from("<H", word_stream, fib_rglw - 2)[0]

        if not (flags & 0x0004):
            # Simple document: body text runs from fcMin..fcMac, located in
            # FibRgLw97 at offsets 0x0044/0x0048 (spec) — older FIBs put them
            # at 0x0018/0x001C, so try both and keep the plausible one.
            candidates = [
                (
                    struct.unpack_from("<I", word_stream, fib_rglw + 0x44)[0],
                    struct.unpack_from("<I", word_stream, fib_rglw + 0x48)[0],
                ),
                (
                    struct.unpack_from("<I", word_stream, fib_rglw + 0x18)[0],
                    struct.unpack_from("<I", word_stream, fib_rglw + 0x1C)[0],
                ),
            ]
            for fc_min, fc_mac in candidates:
                if 0 <= fc_min < fc_mac <= len(word_stream) and fc_mac - fc_min < len(word_stream):
                    return _doc_decode_run(word_stream[fc_min:fc_mac])
            return ""

        table_name = "1Table" if (flags & 0x0100) else "0Table"
        if not ole.exists(table_name):
            return ""
        table = ole.openstream(table_name).read()

        # fcClx/lcbClx live in FibRgFcLcb97. For standard Word 97-2003 docs
        # (csw=14, cslw=22, cbRgFcLcb=0x5D) they sit at absolute offsets
        # 0x01A2/0x01A6 in the WordDocument stream.
        if csw != 14 or cslw != 22:
            return ""
        fc_clx = struct.unpack_from("<I", word_stream, 0x01A2)[0]
        lcb_clx = struct.unpack_from("<I", word_stream, 0x01A6)[0]
        if fc_clx < 0 or fc_clx + lcb_clx > len(table) or lcb_clx < 4:
            return ""
        clx = table[fc_clx : fc_clx + lcb_clx]

        # CLX: skip rgPrc entries (clxt==0x02), find Pcdt (clxt==0x01).
        pos = 0
        while pos < len(clx) and clx[pos] == 0x02:
            cb = struct.unpack_from("<H", clx, pos + 1)[0]
            pos += 3 + cb
        if pos >= len(clx) or clx[pos] != 0x01:
            return ""
        pos += 1
        if pos + 4 > len(clx):
            return ""
        lcb = struct.unpack_from("<I", clx, pos)[0]
        pos += 4
        if pos + lcb > len(clx):
            return ""
        plc_pcd = clx[pos : pos + lcb]

        n = (lcb - 4) // 12
        if n < 1 or len(plc_pcd) < 4 * (n + 1) + 8 * n:
            return ""
        acp = [struct.unpack_from("<I", plc_pcd, 4 * i)[0] for i in range(n + 1)]
        pcds = [plc_pcd[4 * (n + 1) + 8 * i : 4 * (n + 1) + 8 * (i + 1)] for i in range(n)]

        parts = []
        for i, pcd in enumerate(pcds):
            # Pcd layout: prm (2 bytes) + fc (4 bytes); bit 30 of fc is
            # fCompressed, bits 0-29 are the file character position.
            fc = struct.unpack_from("<I", pcd, 2)[0]
            compressed = bool(fc & 0x40000000)
            fc &= 0x3FFFFFFF
            if compressed:
                fc //= 2  # single-byte chars, so position is byte offset
                length = acp[i + 1] - acp[i]
                run = word_stream[fc : fc + length]
                parts.append(_doc_decode_run(run, single_byte=True))
            else:
                length = (acp[i + 1] - acp[i]) * 2  # UTF-16LE, 2 bytes per char
                run = word_stream[fc : fc + length]
                try:
                    parts.append(run.decode("utf-16-le"))
                except (UnicodeDecodeError, IndexError):
                    parts.append("")
        return "".join(parts)
    except (struct.error, IndexError, ValueError, KeyError):
        return ""


def _read_text_file(path: str) -> str:
    """Read a text file, tolerating common non-UTF-8 encodings (GBK, Big5).

    Users frequently upload SRTs saved with the Windows/legacy Chinese
    encodings; decoding strictly as UTF-8 raised UnicodeDecodeError which
    surfaced as a confusing 400 error in the SRT editor.
    """
    with open(path, "rb") as f:
        raw = f.read()
    return _decode_bytes(raw)


# ── SRT calibration (校准) ───────────────────────────────

_OK_TOKEN_RE = re.compile(r"ok", re.IGNORECASE)


def _chinese_key(text: str) -> str:
    """Normalize a segment's text into a matching key by its CJK content.

    Only CJK ideographs are kept, so segments like ``行\\nOkay`` and ``行``
    share the same key regardless of translations, punctuation or whitespace.
    ``OK``/``ok`` tokens are ignored explicitly (a stray "OK" must never
    prevent a match).
    """
    cleaned = _OK_TOKEN_RE.sub("", text or "")
    return "".join(_CJK_RE.findall(cleaned))


_SRT_DOWNLOAD_MARKER_RE = re.compile(r"#\s*\d+\s*\[\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*\]")


def _marker_timing(marker_text: str) -> tuple[timedelta | None, timedelta | None]:
    """Parse ``#001 [00:11 - 00:14]`` into (start, end) timedeltas."""
    m = _SRT_DOWNLOAD_MARKER_RE.search(marker_text)
    if not m:
        return None, None
    start = timedelta(minutes=int(m.group(1)), seconds=int(m.group(2)))
    end = timedelta(minutes=int(m.group(3)), seconds=int(m.group(4)))
    return start, end


def _parse_segments(text: str) -> list[dict]:
    """Parse SRT (or bare-text) content into a list of segment dicts.

    Each dict has ``content`` (full text), ``key`` (CJK-only match key) and
    optional ``start``/``end`` (timedelta, when the source carries timing).
    Three formats are recognized:

    1. SRT-download documents that number segments like ``#001 [00:11 - 00:14]``
       followed by the subtitle text — split on those markers wherever they
       appear (they may sit on their own line or inline); the marker times
       become the segment timing.
    2. Plain SRT (index + timing + text).
    3. Anything else: each non-empty line becomes a segment (no timing).
    """
    markers = list(_SRT_DOWNLOAD_MARKER_RE.finditer(text))
    if len(markers) >= 2:
        segments = []
        # Text before the first marker (e.g. a title).
        head = text[: markers[0].start()]
        content = "\n".join(line for line in head.splitlines() if line.strip())
        if content:
            segments.append({"content": content, "key": _chinese_key(content), "start": None, "end": None})
        for idx, m in enumerate(markers):
            seg_end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
            seg_text = text[m.end() : seg_end]
            content = "\n".join(line for line in seg_text.splitlines() if line.strip())
            start, end = _marker_timing(m.group(0))
            segments.append({"content": content, "key": _chinese_key(content), "start": start, "end": end})
        return segments
    try:
        import srt

        subs = list(srt.parse(text))
        if subs:
            return [
                {"content": s.content, "key": _chinese_key(s.content), "start": s.start, "end": s.end} for s in subs
            ]
    except Exception:
        pass
    return [
        {"content": line, "key": _chinese_key(line), "start": None, "end": None}
        for line in text.splitlines()
        if line.strip()
    ]


def _parse_displayed_segments(text: str) -> list[dict]:
    """Parse the displayed SRT into segments preserving start/end timing."""
    import srt

    subs = list(srt.parse(text))
    return [
        {
            "start": s.start,
            "end": s.end,
            "content": s.content,
            "key": _chinese_key(s.content),
        }
        for s in subs
    ]


def _format_timestamp(td) -> str:
    import srt

    return srt.timedelta_to_srt_timestamp(td)


_MATCH_THRESHOLD = 0.8  # minimum Chinese-content similarity to establish a match
# (tolerates common OCR errors like 自→白, 叉→又, 毕竞→毕竟)
_WALK_WINDOW = 10  # look-ahead distance when re-anchoring after a mismatch


def _chinese_similarity(a: str, b: str) -> float:
    """Similarity of two CJK-only keys in [0, 1].

    A length prefilter avoids paying for difflib on pairs that cannot reach
    the match threshold (SequenceMatcher ratio is bounded by 2·min/(la+lb)).
    """
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    if 2 * min(la, lb) / (la + lb) < _MATCH_THRESHOLD:
        return 0.0
    import difflib

    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _pair_matches(dseg: dict, sseg: dict) -> bool:
    """True when a displayed segment and a standard segment correspond.

    Prefers the CJK-content key; when neither side has Chinese (e.g. noise
    segments like ``OK``) the normalized plain text must match exactly, so
    such segments pair up instead of being duplicated.
    """
    dkey = dseg["key"]
    skey = sseg["key"]
    if dkey and skey:
        return _chinese_similarity(dkey, skey) >= _MATCH_THRESHOLD
    if not dkey and not skey:
        d = re.sub(r"\s+", "", dseg["content"] or "").lower()
        s = re.sub(r"\s+", "", sseg["content"] or "").lower()
        return bool(d) and d == s
    return False


def _align_segments(displayed: list[dict], standard: list[dict]) -> list[dict]:
    """Merge displayed & standard segment lists, two-pointer walk.

    Matched pairs take the displayed timing with the standard content.
    Unmatched segments are copied from either side — but only *between the
    first and the last matched standard segment*.  Before the first match the
    displayed segments are silenced (emptied) and the leading standard
    segments are dropped; after the last match the trailing displayed
    segments are silenced and the trailing standard segments are dropped.

    Standard-only segments keep their own timing when available (otherwise
    they are slotted right after the previous entry).  After copying a
    segment the walk moves to the next segment and re-anchors matching at
    the new pair.

    Returns a list of entries ``{start, end, content, kind}`` in walk order
    with monotonic non-decreasing timing (``kind`` ∈ matched/displayed/
    standard).
    """
    n_disp = len(displayed)
    n_std = len(standard)
    entries: list[dict] = []
    di = 0
    si = 0

    def add_displayed(idx):
        d = displayed[idx]
        entries.append({"start": d["start"], "end": d["end"], "content": d["content"], "kind": "displayed"})

    def add_standard(idx):
        s = standard[idx]
        entries.append({"start": s.get("start"), "end": s.get("end"), "content": s["content"], "kind": "standard"})

    while di < n_disp and si < n_std:
        if _pair_matches(displayed[di], standard[si]):
            d = displayed[di]
            s = standard[si]
            entries.append({"start": d["start"], "end": d["end"], "content": s["content"], "kind": "matched"})
            di += 1
            si += 1
            continue

        # Re-anchor: find the nearest future matching pair within the window.
        found = None
        for a in range(0, _WALK_WINDOW + 1):
            for b in range(0, _WALK_WINDOW + 1):
                if a == 0 and b == 0:
                    continue
                if di + a >= n_disp or si + b >= n_std:
                    continue
                if _pair_matches(displayed[di + a], standard[si + b]) and (
                    found is None or a + b < found[0] + found[1]
                ):
                    found = (a, b)
        if found is None:
            # No future match nearby: copy everything that is left.
            for x in range(di, n_disp):
                add_displayed(x)
            for y in range(si, n_std):
                add_standard(y)
            di, si = n_disp, n_std
            break
        a, b = found
        for x in range(di, di + a):
            add_displayed(x)
        for y in range(si, si + b):
            add_standard(y)
        di += a
        si += b

    while di < n_disp:
        add_displayed(di)
        di += 1
    while si < n_std:
        add_standard(si)
        si += 1

    # Copy-only window: unmatched segments are copied only between the first
    # and the last matched standard segment.  Outside that window displayed
    # entries are silenced and standard entries are dropped.
    first_match = next((i for i, e in enumerate(entries) if e["kind"] == "matched"), None)
    last_match = max((i for i, e in enumerate(entries) if e["kind"] == "matched"), default=None)
    if first_match is None:
        for e in entries:
            if e["kind"] == "displayed":
                e["content"] = ""
            elif e["kind"] == "standard":
                e["kind"] = "drop"
    else:
        for i, e in enumerate(entries):
            if i < first_match or i > last_match:
                if e["kind"] == "displayed":
                    e["content"] = ""
                elif e["kind"] == "standard":
                    e["kind"] = "drop"
    entries = [e for e in entries if e["kind"] != "drop"]

    # Timing: displayed-based entries (matched/displayed) are the gold
    # standard and are never touched.  Standard-only entries are inserted
    # into the gaps between them: their timing is taken from the missed
    # segment's own timing when it overlaps the gap (clamped to fit), else a
    # 2-second heuristic slot at the start of the gap.
    _slot_standard_entries(entries)
    return entries


def _slot_standard_entries(entries: list[dict]) -> None:
    """Assign timing to standard-only entries inside the gaps between
    displayed-based entries, without ever modifying displayed timing."""
    i = 0
    n = len(entries)
    while i < n:
        if entries[i]["kind"] != "standard":
            i += 1
            continue
        j = i
        while j < n and entries[j]["kind"] == "standard":
            j += 1
        gap_start = entries[i - 1]["end"] if i > 0 else timedelta(0)
        gap_end = entries[j]["start"] if j < n else None

        cursor = gap_start
        for k in range(i, j):
            e = entries[k]
            own_start, own_end = e.get("start"), e.get("end")
            placed = False
            if own_start is not None and own_end is not None and own_end > own_start:
                # Reference the missed segment's own timing: use the overlap
                # with the gap when it exists.
                hi = gap_end if gap_end is not None else own_end
                s = max(own_start, cursor)
                t = min(own_end, hi)
                if t > s:
                    e["start"], e["end"] = s, t
                    cursor = t
                    placed = True
                elif own_start >= cursor:
                    e["start"], e["end"] = own_start, own_end
                    cursor = own_end
                    placed = True
            if not placed:
                dur = timedelta(seconds=2)
                s = cursor
                t = s + dur
                if gap_end is not None and t > gap_end:
                    t = gap_end
                if t <= s:
                    t = s + dur
                e["start"], e["end"] = s, t
                cursor = t
        i = j


def _build_corrected_srt(entries: list[dict]) -> str:
    """Compose the corrected SRT from merged entries (walk order)."""
    lines = []
    for i, e in enumerate(entries, start=1):
        lines.append(str(i))
        lines.append(f"{_format_timestamp(e['start'])} --> {_format_timestamp(e['end'])}")
        if e["content"]:
            lines.append(e["content"])
        lines.append("")
    return "\n".join(lines)


def build_corrected_srt(displayed_text: str, standard_text: str) -> str:
    """Rebuild a SRT merging the displayed and standard files.

    Matched pairs keep the displayed timing with the standard content
    (matched by fuzzy CJK content only, ignoring stray ``OK`` tokens).
    Unmatched segments — from either the displayed or the standard file —
    are copied into the output between the first and the last matched
    standard segment, so nothing in that range is lost.
    """
    displayed = _parse_displayed_segments(displayed_text)
    standard = _parse_segments(standard_text)

    entries = _align_segments(displayed, standard)
    return _build_corrected_srt(entries)


def corrected_srt_name(safe_path: str) -> str:
    """Return the calibrated output filename for a displayed SRT (e.g.
    ``00001.zh+en.corrected.srt``)."""
    base = os.path.basename(safe_path)
    return base.rsplit(".", 1)[0] + ".corrected.srt"


def corrected_srt_path(safe_path: str) -> str:
    """Return the full output path for the corrected SRT — same directory as
    the displayed SRT, name suffixed with ``corrected``."""
    return os.path.join(os.path.dirname(safe_path), corrected_srt_name(safe_path))


def validate_corrected_srt_content(text: str) -> None:
    """Validate corrected SRT structure (timing lines + parseable) without
    enforcing a single language.

    Corrected files may be bilingual (e.g. ``*.zh+en.corrected.srt``) where
    the strict ``validate_srt_content`` language check would reject them.
    """
    text = text.lstrip("\ufeff")
    if not _SRT_TIMING_RE.search(text):
        raise ValueError("校准后的 SRT 文件格式无效：缺少时间轴行（如 '00:00:01,000 --> 00:00:03,000'）。")
    import srt

    try:
        list(srt.parse(text))
    except Exception as e:
        raise ValueError(f"校准后的 SRT 文件格式有误：{e}") from e


@files_bp.route("/files/list", methods=["GET"])
@login_required
@api_endpoint
def files_list():
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


@files_bp.route("/files/read", methods=["GET"])
@login_required
@api_endpoint
def files_read():
    file_path = request.args.get("path")
    if not file_path:
        return jsonify({"error": "No file specified"}), 400

    safe = safe_file_path(file_path)
    if not safe:
        return jsonify({"error": "File not allowed or not found"}), 404

    content = _read_text_file(safe)
    return content, 200, {"Content-Type": "text/plain; charset=utf-8"}


@files_bp.route("/files/download", methods=["GET"])
@login_required
@api_endpoint
def files_download():
    file_path = request.args.get("path")
    if not file_path:
        return jsonify({"error": "No file specified"}), 400

    safe = safe_file_path(file_path)
    if not safe:
        return jsonify({"error": "File not allowed or not found"}), 404

    return send_file(safe, as_attachment=True)


@files_bp.route("/files/delete", methods=["POST"])
@login_required
@csrf_required
@api_endpoint
def files_delete():
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


# ── SRT save/edit endpoint ────────────────────────────────


@files_bp.route("/files/save-srt", methods=["POST"])
@login_required
@csrf_required
@api_endpoint
def files_save_srt():
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

    validate_srt_content(content, "SRT")

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


def _extract_standard_text(raw: bytes, filename: str) -> str:
    """Extract readable text from an uploaded standard file.

    Supports plain SRT/TXT (decoded with GB_*/Big5 tolerance) and Word
    documents (.docx via OOXML XML, .doc via OLE piece table).  Word files
    are a common way users supply the "standard" transcript, and they are
    frequently saved with legacy GBK/GB_* Chinese encodings.
    """
    name = (filename or "").lower()
    if name.endswith(".docx"):
        return _extract_docx_text(raw)
    if name.endswith(".doc"):
        return _extract_doc_text(raw)
    return _decode_text_file_bytes(raw)


@files_bp.route("/files/correct-srt", methods=["POST"])
@login_required
@csrf_required
@api_endpoint
def files_correct_srt():
    """Calibrate a displayed SRT against an uploaded standard SRT/text file.

    Timing is taken from the displayed file; content is taken from the
    uploaded standard, matched by CJK content. Unmatched segments are copied
    from either side between the first and the last matched standard segment.
    """
    file_path = request.form.get("path")
    if not file_path:
        return jsonify({"error": "No file specified"}), 400

    safe = safe_file_path(file_path)
    if not safe:
        return jsonify({"error": "File not allowed or not found"}), 404

    if "standard_file" not in request.files:
        return jsonify({"error": "No standard file uploaded"}), 400

    standard_file = request.files["standard_file"]
    validate_upload_filename(standard_file.filename)
    raw = standard_file.read()
    if not raw:
        return jsonify({"error": "Uploaded standard file is empty"}), 400

    standard_text = _extract_standard_text(raw, standard_file.filename)
    displayed_text = _read_text_file(safe)

    displayed = _parse_displayed_segments(displayed_text)
    standard = _parse_segments(standard_text)
    if not any(seg["key"] for seg in standard):
        return jsonify({"error": "上传的标准文件中没有检测到中文内容，请确认文件内容正确（需包含中文字幕文字）"}), 400

    entries = _align_segments(displayed, standard)
    corrected = _build_corrected_srt(entries)

    matched_count = sum(1 for e in entries if e["kind"] == "matched")
    displayed_only = [e for e in entries if e["kind"] == "displayed" and e["content"].strip()]
    standard_only = [e for e in entries if e["kind"] == "standard" and e["content"].strip()]
    logger.info(
        "correct-srt: standard=%s bytes=%d text_len=%d cjk_chars=%d std_segments=%d matched=%d displayed_only=%d standard_only=%d",
        standard_file.filename,
        len(raw),
        len(standard_text),
        len(_CJK_RE.findall(standard_text)),
        len(standard),
        matched_count,
        len(displayed_only),
        len(standard_only),
    )
    name = corrected_srt_name(safe)
    return jsonify(
        {
            "success": True,
            "content": corrected,
            "filename": name,
            "matched": matched_count,
            "total": len(entries),
            "displayed_only": [e["content"] for e in displayed_only],
            "standard_only": [e["content"] for e in standard_only],
        }
    )


@files_bp.route("/files/save-corrected-srt", methods=["POST"])
@login_required
@csrf_required
@api_endpoint
def files_save_corrected_srt():
    """Save a corrected SRT next to the displayed SRT with a ``+corrected`` name.

    The corrected file is written into the same directory as the displayed
    SRT so it sits beside the original (e.g. ``00001.zh+en.corrected.srt``).
    """
    data = request.get_json(silent=True) or {}
    file_path = data.get("path")
    content = data.get("content")

    if not file_path or content is None:
        return jsonify({"success": False, "error": "Missing path or content"}), 400

    safe = safe_file_path(file_path)
    if not safe:
        return jsonify({"success": False, "error": "File not allowed or not found"}), 404

    if not safe.lower().endswith(".srt"):
        return jsonify({"success": False, "error": "Only .srt files can be corrected"}), 400

    # Corrected SRTs may legitimately be bilingual (e.g. *.zh+en files), so
    # validate the SRT structure only — not single-language enforcement.
    validate_corrected_srt_content(content)

    out_path = corrected_srt_path(safe)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    return jsonify({"success": True, "path": out_path, "filename": os.path.basename(out_path)})


@files_bp.route("/files/srt-resubmit/<access_code>", methods=["POST"])
@login_required
@csrf_required
@api_endpoint
def files_srt_resubmit(access_code):
    jq = get_job_queue()
    result = jq.resubmit_job(access_code)
    if result["success"]:
        jq.set_checkpoint_edited(access_code, False)
        jq.clear_edited_srt_files(access_code)
        return jsonify(result)
    return jsonify(result), 400
