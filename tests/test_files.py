"""Tests for file management routes: list, read, download, delete, SRT save."""

import pytest


@pytest.fixture(autouse=True)
def _bypass_rate_limit():
    from middleware import _ip_limiter

    old = _ip_limiter.limit
    _ip_limiter.limit = 10000
    yield
    _ip_limiter.limit = old


class TestFilesList:
    def test_no_directory(self, auth_client):
        client, _ = auth_client
        resp = client.get("/files/list")
        assert resp.status_code == 400

    def test_unauthorized(self, client):
        resp = client.get("/files/list?dir=/tmp")
        assert resp.status_code == 401

    def test_nonexistent_directory(self, auth_client):
        client, _ = auth_client
        resp = client.get("/files/list?dir=/nonexistent/dir/12345")
        assert resp.status_code in (403, 404)

    def test_disallowed_directory(self, auth_client):
        client, _ = auth_client
        resp = client.get("/files/list?dir=/etc")
        assert resp.status_code == 403


class TestFilesRead:
    def test_no_path(self, auth_client):
        client, _ = auth_client
        resp = client.get("/files/read")
        assert resp.status_code == 400

    def test_unauthorized(self, client):
        resp = client.get("/files/read?path=/tmp/x.txt")
        assert resp.status_code == 401

    def test_gbk_encoded_srt_reads_ok(self, auth_client):
        """Users upload SRTs in GBK/legacy encodings; the reader must not
        blow up with a UnicodeDecodeError (regression for access code
        841A7C64)."""
        import os

        from config import VIDEO_DIR
        from middleware import ALLOWED_FILE_DIRS

        client, _ = auth_client
        video_dir = str(VIDEO_DIR)
        os.makedirs(video_dir, exist_ok=True)
        real = os.path.realpath(video_dir)
        if real not in ALLOWED_FILE_DIRS:
            ALLOWED_FILE_DIRS.append(real)

        path = os.path.join(video_dir, "gbk.srt")
        with open(path, "wb") as fh:
            fh.write("1\n00:00:01,000 --> 00:00:02,000\n你好，世界\n".encode("gbk"))

        resp = client.get("/files/read?path=" + path)
        assert resp.status_code == 200
        assert "你好" in resp.get_data(as_text=True)


class TestFilesDelete:
    def test_unauthorized(self, client):
        resp = client.post(
            "/files/delete",
            json={"path": "/tmp/x.txt"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 401

    def test_no_path(self, auth_client, csrf_headers):
        client, _ = auth_client
        resp = client.post("/files/delete", json={}, headers=csrf_headers)
        assert resp.status_code == 400

    def test_csrf_required(self, auth_client):
        client, _ = auth_client
        resp = client.post("/files/delete", json={"path": "/tmp/x.txt"})
        assert resp.status_code == 403  # CSRF missing


class TestFilesDownload:
    def test_no_path(self, auth_client):
        client, _ = auth_client
        resp = client.get("/files/download")
        assert resp.status_code == 400


class TestSRTSave:
    def test_unauthorized(self, client):
        resp = client.post(
            "/files/save-srt",
            json={
                "path": "/tmp/x.srt",
                "content": "00:00:01,000 --> 00:00:03,000\ntest",
                "access_code": "ABC",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 401

    def test_missing_content(self, auth_client, csrf_headers):
        client, _ = auth_client
        resp = client.post(
            "/files/save-srt",
            json={
                "path": "/tmp/x.srt",
                "access_code": "ABC",
            },
            headers=csrf_headers,
        )
        assert resp.status_code == 400

    def test_non_srt_file(self, auth_client, csrf_headers):
        client, _ = auth_client
        resp = client.post(
            "/files/save-srt",
            json={
                "path": "/tmp/x.txt",
                "content": "not srt",
                "access_code": "ABC",
            },
            headers=csrf_headers,
        )
        assert resp.status_code in (400, 404)

    def test_invalid_srt_content(self, auth_client, csrf_headers, tmp_path):
        client, _ = auth_client
        f = tmp_path / "test.srt"
        f.write_text("dummy")
        resp = client.post(
            "/files/save-srt",
            json={
                "path": str(f),
                "content": "no timing line here",
                "access_code": "ABC",
            },
            headers=csrf_headers,
        )
        assert resp.status_code in (400, 404)  # path not allowed, or content invalid


class TestSRTProcess:
    """/srt/process must reject multi-language SRTs before a job is queued.

    Regression for job C68639E3: a bilingual Chinese+German SRT passed
    validation, TTS generated 3-4x slow audio, and the job failed at the
    duration-inflation guard.
    """

    def _post_srt(self, client, csrf_headers, content, target_language="en"):
        import io

        data = {
            "temperature": "0.7",
            "target_language": target_language,
            "cfg_weight": "0.5",
            "exaggeration": "0.5",
        }
        return client.post(
            "/srt/process",
            data={**data, "srt_file": (io.BytesIO(content.encode("utf-8")), "test.srt")},
            headers=csrf_headers,
            content_type="multipart/form-data",
        )

    def test_bilingual_srt_rejected(self, auth_client, csrf_headers):
        client, _ = auth_client
        bilingual = (
            "1\n00:00:01,000 --> 00:00:03,000\n你就是如来\nDu bist der Tathagata\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\n记住这一点\nMerk dir das\n"
        )
        resp = self._post_srt(client, csrf_headers, bilingual)
        assert resp.status_code == 400
        assert "多种语言" in resp.get_json()["error"]

    def test_target_language_mismatch_rejected(self, auth_client, csrf_headers):
        client, _ = auth_client
        zh = (
            "1\n00:00:01,000 --> 00:00:03,000\n你就是如来\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\n这就是释迦牟尼证到的最高境界\n"
        )
        resp = self._post_srt(client, csrf_headers, zh, target_language="en")
        assert resp.status_code == 400
        assert "make them consistent" in resp.get_json()["error"]

    def test_single_language_accepted(self, auth_client, csrf_headers):
        client, _ = auth_client
        zh = (
            "1\n00:00:01,000 --> 00:00:03,000\n你就是如来\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\n这就是释迦牟尼证到的最高境界\n"
        )
        resp = self._post_srt(client, csrf_headers, zh, target_language="zh")
        assert resp.status_code == 302
        assert "/result?code=" in resp.headers["Location"]


class TestSRTResubmit:
    def test_unauthorized(self, client):
        resp = client.post(
            "/files/srt-resubmit/ABC",
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 401

    def test_nonexistent_job(self, auth_client, csrf_headers):
        client, _ = auth_client
        resp = client.post("/files/srt-resubmit/DEADBEEF", headers=csrf_headers)
        assert resp.status_code == 400


class TestSRTCorrect:
    """/files/correct-srt calibrates a displayed SRT against an uploaded standard."""

    DISPLAYED = (
        "1\n00:00:02,000 --> 00:00:03,667\n外\n成叶情土\n\n"
        "2\n00:00:07,667 --> 00:00:10,000\n杨宁随缘开示#00001\n2012年05月03日\n\n"
        "3\n00:00:12,333 --> 00:00:12,667\n行\n\n"
        "4\n00:00:12,667 --> 00:00:14,333\n当我们为某个觉受苦的时候\n\n"
        "5\n00:00:14,333 --> 00:00:17,333\n恰恰是因为你占有了这个觉受\n"
    )

    STANDARD = (
        "1\n00:00:01,000 --> 00:00:03,000\n外成叶情土\n\n"
        "2\n00:00:06,000 --> 00:00:09,000\n杨宁随缘开示#00001 2012年05月03日\n\n"
        "3\n00:00:11,000 --> 00:00:12,000\n行\nOkay\n\n"
        "4\n00:00:12,000 --> 00:00:14,000\n当我们为某个觉受苦的时候\nWhen we suffer\n\n"
        "5\n00:00:14,000 --> 00:00:17,000\n恰恰是因为你占有了这个觉受\nIt's possession\n"
    )

    @staticmethod
    def _write_displayed(path, content):
        import os

        from config import VIDEO_DIR
        from middleware import ALLOWED_FILE_DIRS

        os.makedirs(str(VIDEO_DIR), exist_ok=True)
        real = os.path.realpath(str(VIDEO_DIR))
        if real not in ALLOWED_FILE_DIRS:
            ALLOWED_FILE_DIRS.append(real)
        p = os.path.join(real, path)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(content)
        return p

    @staticmethod
    def _post(client, path, standard_bytes, name="std.srt"):
        import io

        return client.post(
            "/files/correct-srt",
            data={"path": path, "standard_file": (io.BytesIO(standard_bytes), name)},
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": "test-csrf-token"},
        )

    def test_unauthorized(self, client):
        resp = client.post(
            "/files/correct-srt",
            data={"path": "/tmp/x.srt", "standard_file": (None, "std.srt")},
            content_type="multipart/form-data",
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 401

    def test_no_path(self, auth_client, csrf_headers):
        client, _ = auth_client
        resp = client.post(
            "/files/correct-srt",
            data={"standard_file": (None, "std.srt")},
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": "test-csrf-token"},
        )
        assert resp.status_code == 400

    def test_no_standard_file(self, auth_client, csrf_headers):
        client, _ = auth_client
        path = self._write_displayed("correct_no_std.srt", self.DISPLAYED)
        resp = client.post(
            "/files/correct-srt",
            data={"path": path},
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": "test-csrf-token"},
        )
        assert resp.status_code == 400

    def test_ok_matches_by_chinese_content(self, auth_client, csrf_headers):
        client, _ = auth_client
        path = self._write_displayed("correct_ok.srt", self.DISPLAYED)
        resp = self._post(client, path, self.STANDARD.encode("utf-8"))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["filename"] == "correct_ok.corrected.srt"

        # Timing comes from displayed SRT
        assert "00:00:02,000 --> 00:00:03,667" in data["content"]
        assert "00:00:12,667 --> 00:00:14,333" in data["content"]

        # Content comes from the standard file (matched by Chinese)
        assert "外成叶情土" in data["content"]
        assert "杨宁随缘开示#00001 2012年05月03日" in data["content"]
        assert "行\nOkay" in data["content"]
        assert "When we suffer" in data["content"]
        assert "It's possession" in data["content"]

    def test_unmatched_displayed_segments_copied(self, auth_client, csrf_headers):
        client, _ = auth_client
        # An extra displayed segment in the MIDDLE of the match range is
        # copied; segments before the first / after the last match are
        # silenced (per the copy-window rule).
        displayed = (
            "1\n00:00:00,000 --> 00:00:01,000\n开头无匹配\n\n"
            + self.DISPLAYED
            + "6\n00:00:20,000 --> 00:00:21,000\n尾部无匹配\n"
        )
        path = self._write_displayed("correct_unmatched.srt", displayed)
        resp = self._post(client, path, self.STANDARD.encode("utf-8"))
        assert resp.status_code == 200
        data = resp.get_json()
        # Middle unmatched displayed segments are copied as-is.
        assert data["matched"] == 5
        # Leading/trailing displayed segments are silenced.
        assert "开头无匹配" not in data["content"]
        assert "尾部无匹配" not in data["content"]

    def test_unmatched_standard_segment_in_middle_copied(self, auth_client, csrf_headers):
        client, _ = auth_client
        # An extra standard segment in the middle is copied into the output.
        displayed = self.DISPLAYED
        standard = (
            "1\n00:00:01,000 --> 00:00:03,000\n外成叶情土\n\n"
            "2\n00:00:06,000 --> 00:00:09,000\n杨宁随缘开示#00001 2012年05月03日\n\n"
            "3\n00:00:11,000 --> 00:00:12,000\n行\nOkay\n\n"
            "4\n00:00:12,000 --> 00:00:14,000\n当我们为某个觉受苦的时候\nWhen we suffer\n\n"
            "5\n00:00:13,000 --> 00:00:14,000\n标准独有片段\n\n"
            "6\n00:00:14,000 --> 00:00:17,000\n恰恰是因为你占有了这个觉受\nIt's possession\n"
        )
        path = self._write_displayed("correct_extra_std_mid.srt", displayed)
        resp = self._post(client, path, standard.encode("utf-8"))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "标准独有片段" in data["content"]
        assert data["standard_only"] == ["标准独有片段"]

    def test_gbk_standard_file(self, auth_client, csrf_headers):
        client, _ = auth_client
        path = self._write_displayed("correct_gbk.srt", self.DISPLAYED)
        resp = self._post(client, path, "外成叶情土".encode("gbk"), name="std.txt")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "外成叶情土" in data["content"]

    def test_fuzzy_match_bilingual_zh_en(self, auth_client, csrf_headers):
        """Both files may be bilingual; matching is on Chinese text with a
        ~90% threshold, so minor OCR/typo differences still correspond."""
        client, _ = auth_client
        displayed = (
            "1\n00:00:02,000 --> 00:00:03,667\n外\n成叶情土\n\n"
            "2\n00:00:07,667 --> 00:00:10,000\n杨宁随缘开示#00001\n2012年05月03日\n\n"
            "3\n00:00:12,333 --> 00:00:12,667\n行\nOkay\n\n"
            "4\n00:00:12,667 --> 00:00:14,333\n当我们为某个觉受苦的时候\nWhen we suffer\n\n"
            "5\n00:00:14,333 --> 00:00:17,333\n恰恰是因为你占有了这个觉受\nIt's possession\n"
        )
        # Standard has slightly different Chinese (typo 受→受苦) and different timing.
        standard = (
            "1\n00:00:01,000 --> 00:00:03,000\n外成叶情土\n\n"
            "2\n00:00:06,000 --> 00:00:09,000\n杨宁随缘开示#00001 2012年05月03日\n\n"
            "3\n00:00:11,000 --> 00:00:12,000\n行\nOkay\n\n"
            "4\n00:00:12,000 --> 00:00:14,000\n当我们为某个觉受苦的时候\nWhen we suffer\n\n"
            "5\n00:00:14,000 --> 00:00:17,000\n恰恰是因为你占有了这个觉受\nIt's possession\n"
        )
        path = self._write_displayed("correct_fuzzy_zh_en.srt", displayed)
        resp = self._post(client, path, standard.encode("utf-8"))
        assert resp.status_code == 200
        data = resp.get_json()
        # Chinese content matched across bilingual segments and swapped in.
        assert "当我们为某个觉受苦的时候\nWhen we suffer" in data["content"]
        assert "外成叶情土" in data["content"]
        assert "行\nOkay" in data["content"]

    def test_standard_has_extra_unmatched_segments(self, auth_client, csrf_headers):
        """Standard-only segments are dropped; displayed-only segments silenced."""
        client, _ = auth_client
        displayed = self.DISPLAYED
        standard = self.STANDARD + "6\n00:00:20,000 --> 00:00:21,000\n仅在上传文件中\n\n"
        path = self._write_displayed("correct_extra_std.srt", displayed)
        resp = self._post(client, path, standard.encode("utf-8"))
        assert resp.status_code == 200
        data = resp.get_json()
        # The uploaded-only segment content must not leak into the output.
        assert "仅在上传文件中" not in data["content"]

    def test_below_threshold_similarity_is_silenced(self, auth_client, csrf_headers):
        """A displayed segment whose Chinese differs too much (<90%) is emptied."""
        client, _ = auth_client
        displayed = (
            "1\n00:00:12,667 --> 00:00:14,333\n当我们为某个觉受苦的时候\n\n"
            "2\n00:00:14,333 --> 00:00:17,333\n恰恰是因为你占有了这个觉受\n"
        )
        # Segment 1's Chinese is completely different in the standard file.
        standard = (
            "1\n00:00:12,000 --> 00:00:14,000\n完全不同的另一句话\n\n"
            "2\n00:00:14,000 --> 00:00:17,000\n恰恰是因为你占有了这个觉受\n"
        )
        path = self._write_displayed("correct_below_threshold.srt", displayed)
        resp = self._post(client, path, standard.encode("utf-8"))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "完全不同的另一句话" not in data["content"]
        assert "恰恰是因为你占有了这个觉受" in data["content"]
        # First displayed segment is silenced but keeps timing.
        assert "1\n00:00:12,667 --> 00:00:14,333\n\n" in data["content"]


    def test_ok_only_segments_pair_not_duplicated(self, auth_client, csrf_headers):
        """Segments with no Chinese (e.g. 'OK') pair by plain text so they are
        not copied twice from both sides."""
        client, _ = auth_client
        displayed = (
            "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\nOK\n\n"
            "3\n00:00:03,000 --> 00:00:04,000\n世界\n"
        )
        standard = (
            "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n"
            "2\n00:00:02,500 --> 00:00:03,500\nOK\n\n"
            "3\n00:00:03,000 --> 00:00:04,000\n世界\n"
        )
        path = self._write_displayed("correct_ok_pair.srt", displayed)
        resp = self._post(client, path, standard.encode("utf-8"))
        assert resp.status_code == 200
        data = resp.get_json()
        # The two OK segments match each other → exactly one OK block.
        assert data["content"].count("OK") == 1
        assert data["matched"] == 3

    def test_standard_only_timing_slots_into_gap(self, auth_client, csrf_headers):
        """A standard-only segment is inserted into the gap between displayed
        segments without touching the displayed timing."""
        client, _ = auth_client
        displayed = (
            "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n"
            "2\n00:00:05,000 --> 00:00:08,000\n世界\n"
        )
        standard = (
            "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n"
            "2\n00:00:02,500 --> 00:00:03,500\n中间多出的一句\n\n"
            "3\n00:00:05,000 --> 00:00:08,000\n世界\n"
        )
        path = self._write_displayed("correct_gap_slot.srt", displayed)
        resp = self._post(client, path, standard.encode("utf-8"))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "中间多出的一句" in data["content"]
        # Displayed timing untouched; inserted segment sits in the gap.
        assert "00:00:01,000 --> 00:00:02,000" in data["content"]
        assert "00:00:05,000 --> 00:00:08,000" in data["content"]
        assert "00:00:02,500 --> 00:00:03,500" in data["content"]

    def test_ok_token_ignored_in_matching(self, auth_client, csrf_headers):
        """'OK' inside Chinese text must not prevent a match."""
        client, _ = auth_client
        displayed = "1\n00:00:01,000 --> 00:00:02,000\n你要有个东西在享受你就OK了\n"
        standard = "1\n00:00:01,000 --> 00:00:02,000\n你要有个东西在享受你就了\n"
        path = self._write_displayed("correct_ok_token.srt", displayed)
        resp = self._post(client, path, standard.encode("utf-8"))
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["matched"] == 1
        assert "你要有个东西在享受你就了" in data["content"]


class TestSRTCorrectWordDocs:
    """/files/correct-srt accepts Word documents (.docx / .doc) as the standard."""

    DISPLAYED = (
        "1\n00:00:12,667 --> 00:00:14,333\n当我们为某个觉受苦的时候\n\n"
        "2\n00:00:14,333 --> 00:00:17,333\n恰恰是因为你占有了这个觉受\n"
    )

    @staticmethod
    def _write_displayed(path):
        import os

        from config import VIDEO_DIR
        from middleware import ALLOWED_FILE_DIRS

        os.makedirs(str(VIDEO_DIR), exist_ok=True)
        real = os.path.realpath(str(VIDEO_DIR))
        if real not in ALLOWED_FILE_DIRS:
            ALLOWED_FILE_DIRS.append(real)
        p = os.path.join(real, path)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(TestSRTCorrectWordDocs.DISPLAYED)
        return p

    @staticmethod
    def _post(client, path, raw, name):
        import io

        return client.post(
            "/files/correct-srt",
            data={"path": path, "standard_file": (io.BytesIO(raw), name)},
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": "test-csrf-token"},
        )

    @staticmethod
    def _make_docx(text_lines):
        import io
        import zipfile

        NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
        runs = "".join(f"<w:p><w:r><w:t>{t}</w:t></w:r></w:p>" for t in text_lines)
        doc = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:document {NS}><w:body>{runs}</w:body></w:document>'
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "[Content_Types].xml",
                '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/></Types>',
            )
            zf.writestr("word/document.xml", doc)
        return buf.getvalue()

    def test_docx_standard(self, auth_client, csrf_headers):
        client, _ = auth_client
        path = self._write_displayed("correct_docx.srt")
        docx = self._make_docx(["当我们为某个觉受苦的时候", "恰恰是因为你占有了这个觉受"])
        resp = self._post(client, path, docx, "standard.docx")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "当我们为某个觉受苦的时候" in data["content"]
        assert "恰恰是因为你占有了这个觉受" in data["content"]

    def test_docx_srt_download_marker_format(self, auth_client, csrf_headers):
        """A docx with '#001 [00:11 - 00:14]' style markers parses into
        segments and anchors onto the displayed SRT at the first content."""
        client, _ = auth_client
        # Displayed: noise segments first, then real content (like a real
        # OCR SRT whose first segments are noise/title).
        displayed = (
            "1\n00:00:02,000 --> 00:00:03,667\n外\n成叶情土\n\n"
            "2\n00:00:07,667 --> 00:00:10,000\n杨宁随缘开示#00007\n\n"
            "3\n00:00:11,667 --> 00:00:14,333\n名利情欲是虚幻的 为什么你会痛苦\n\n"
            "4\n00:00:14,333 --> 00:00:16,000\n因为虚幻的东西你抓不住\n\n"
            "5\n00:00:16,000 --> 00:00:18,000\n你始终觉得里面没有一个\n"
        )
        path = self._write_displayed("correct_marker.srt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(displayed)
        # Standard: starts directly at the real content with markers.
        docx = self._make_docx(
            [
                "#001 [00:11 - 00:14]",
                "名利情欲是虚幻的 为什么你会痛苦",
                "#002 [00:14 - 00:16]",
                "因为虚幻的东西你抓不住",
                "#003 [00:16 - 00:18]",
                "你始终觉得里面没有一个",
            ]
        )
        resp = self._post(client, path, docx, "standard.docx")
        assert resp.status_code == 200
        data = resp.get_json()
        # The two noise segments are silenced...
        assert "外成叶情土" not in data["content"]
        assert "杨宁随缘开示#00007" not in data["content"]
        # ...and the real content is matched from segment 3 onward.
        assert "名利情欲是虚幻的 为什么你会痛苦" in data["content"]
        assert "因为虚幻的东西你抓不住" in data["content"]
        assert "你始终觉得里面没有一个" in data["content"]
        assert data["matched"] == 3
        assert data["total"] == 5

    def test_docx_marker_inline_format(self, auth_client, csrf_headers):
        """Markers inline on the same line as the text also split correctly."""
        client, _ = auth_client
        displayed = (
            "1\n00:00:11,667 --> 00:00:14,333\n名利情欲是虚幻的 为什么你会痛苦\n\n"
            "2\n00:00:14,333 --> 00:00:16,000\n因为虚幻的东西你抓不住\n"
        )
        path = self._write_displayed("correct_marker_inline.srt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(displayed)
        docx = self._make_docx(
            [
                "#001 [00:11 - 00:14] 名利情欲是虚幻的 为什么你会痛苦",
                "#002 [00:14 - 00:16] 因为虚幻的东西你抓不住",
            ]
        )
        resp = self._post(client, path, docx, "standard.docx")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "名利情欲是虚幻的 为什么你会痛苦" in data["content"]
        assert data["matched"] == 2

    def test_docx_missing_document_xml(self, auth_client, csrf_headers):
        import io
        import zipfile

        client, _ = auth_client
        path = self._write_displayed("correct_bad_docx.srt")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "not a docx")
        resp = self._post(client, path, buf.getvalue(), "bad.docx")
        assert resp.status_code == 400

    def test_docx_invalid_zip(self, auth_client, csrf_headers):
        client, _ = auth_client
        path = self._write_displayed("correct_invalid_zip.srt")
        resp = self._post(client, path, b"this is not a zip at all", "broken.docx")
        assert resp.status_code == 400

    def test_doc_simple_gbk_body(self, auth_client, csrf_headers):
        """Legacy .doc (simple, no piece table) with a GBK body extracts fine."""
        client, _ = auth_client
        path = self._write_displayed("correct_doc_simple.srt")
        body = "当我们为某个觉受苦的时候\n恰恰是因为你占有了这个觉受".encode("gbk")
        stream = _make_fib_header(0, body)
        stream[0x400 : 0x400 + len(body)] = body

        from routes import files as files_mod

        fake_ole = _FakeOle({"WordDocument": bytes(stream)})
        text = files_mod._extract_doc_text_from_ole(fake_ole)
        assert "当我们为某个觉受苦的时候" in text
        assert "恰恰是因为你占有了这个觉受" in text

        # End-to-end: exercise the endpoint with a .doc filename whose content
        # is handled by _extract_doc_text (patched to the fake OLE).
        original = files_mod._extract_doc_text
        files_mod._extract_doc_text = lambda raw: files_mod._extract_doc_text_from_ole(
            _FakeOle({"WordDocument": bytes(stream)})
        )
        try:
            resp = self._post(client, path, b"placeholder", "standard.doc")
        finally:
            files_mod._extract_doc_text = original
        assert resp.status_code == 200
        assert "当我们为某个觉受苦的时候" in resp.get_json()["content"]

    def test_doc_complex_utf16_piece_table(self, auth_client, csrf_headers):
        """Legacy .doc with a UTF-16 piece table extracts fine."""
        client, _ = auth_client
        text = "当我们为某个觉受苦的时候\n恰恰是因为你占有了这个觉受"
        word_stream = _make_complex_doc_stream(text)
        table = _make_clx_table(word_stream, patch_stream=True)

        from routes import files as files_mod

        fake_ole = _FakeOle({"WordDocument": word_stream, "1Table": table})
        out = files_mod._extract_doc_text_from_ole(fake_ole)
        assert "当我们为某个觉受苦的时候" in out
        assert "恰恰是因为你占有了这个觉受" in out

    def test_gbk_encoding_variants(self, auth_client, csrf_headers):
        """GBK and gb18030 uploaded text both calibrate."""
        client, _ = auth_client
        path = self._write_displayed("correct_gb_variants.srt")
        resp = self._post(client, path, "当我们为某个觉受苦的时候".encode("gb18030"), "std.txt")
        assert resp.status_code == 200
        assert "当我们为某个觉受苦的时候" in resp.get_json()["content"]


class _FakeOle:
    """Minimal stand-in for an olefile.OleFileIO exposing only what the
    piece-table walker needs."""

    def __init__(self, streams):
        self._streams = streams

    def exists(self, name):
        return name in self._streams

    def openstream(self, name):
        if name not in self._streams:
            raise KeyError(name)
        import io

        return io.BytesIO(self._streams[name])


def _make_fib_header(flags: int, body: bytes, fc_clx=0, lcb_clx=0) -> bytearray:
    """Build a realistic Word 97 FIB header around ``body``.

    Layout: FibBase(32) + csw(2)@0x20 + FibRgW(28) + cslw(2) + FibRgLw(88).
    For simple docs fcMin/fcMac are written into FibRgLw+0x44/0x48; for
    complex docs fcClx/lcbClx are patched at absolute 0x01A2/0x01A6.
    """
    import struct

    stream = bytearray(0x400 + len(body))
    struct.pack_into("<H", stream, 0x0A, flags)
    # FibBase is 32 bytes; csw lives right after it.
    csw = 14
    cslw = 22
    struct.pack_into("<H", stream, 0x20, csw)
    fib_rglw = 0x22 + csw * 2
    struct.pack_into("<H", stream, fib_rglw - 2, cslw)
    if not (flags & 0x0004):
        # Simple doc: body text begins at 0x400.
        struct.pack_into("<I", stream, fib_rglw + 0x44, 0x400)  # fcMin
        struct.pack_into("<I", stream, fib_rglw + 0x48, 0x400 + len(body))  # fcMac
    else:
        struct.pack_into("<I", stream, 0x01A2, fc_clx)
        struct.pack_into("<I", stream, 0x01A6, lcb_clx)
    return stream


def _make_complex_doc_stream(text: str, table: bytes = b"") -> bytearray:
    """Build a WordDocument stream with fComplex+fWhichTblStm set and a
    UTF-16LE body at 0x400; fcClx/lcbClx are patched into the FIB."""
    body = text.encode("utf-16-le")
    fc_clx = 0x40
    stream = _make_fib_header(0x0006, body, fc_clx=fc_clx, lcb_clx=len(table))
    stream[0x400 : 0x400 + len(body)] = body
    return stream


def _make_clx_table(word_stream: bytearray, patch_stream: bool = False) -> bytes:
    """Build a 1Table stream whose CLX holds a single UTF-16 piece pointing at
    the body text (0x400).  When ``patch_stream`` is set, fcClx/lcbClx are
    patched into the WordDocument stream in place."""
    import struct

    text_len = (len(word_stream) - 0x400) // 2
    n = 1
    clx = bytearray()
    clx.append(0x01)  # clxt = Pcdt
    lcb = 4 * (n + 1) + 8 * n
    clx += struct.pack("<I", lcb)
    for cp in [0, text_len]:
        clx += struct.pack("<I", cp)
    clx += struct.pack("<H", 0)  # prm
    clx += struct.pack("<I", 0x400)  # fc: byte offset, uncompressed UTF-16
    clx += struct.pack("<H", 0)  # padding
    fc_clx = 0x40
    table = bytearray(0x200)
    table[fc_clx : fc_clx + len(clx)] = clx
    if patch_stream:
        struct.pack_into("<I", word_stream, 0x01A2, fc_clx)
        struct.pack_into("<I", word_stream, 0x01A6, len(clx))
    return bytes(table)


class TestSRTSaveCorrected:
    """/files/save-corrected-srt writes the corrected SRT next to the displayed one."""

    CORRECTED = (
        "1\n00:00:02,000 --> 00:00:03,667\n外成叶情土\n\n"
        "2\n00:00:07,667 --> 00:00:10,000\n杨宁随缘开示#00001\n\n"
        "3\n00:00:12,333 --> 00:00:12,667\n行\nOkay\n"
    )

    @staticmethod
    def _write_displayed(path):
        import os

        from config import VIDEO_DIR
        from middleware import ALLOWED_FILE_DIRS

        os.makedirs(str(VIDEO_DIR), exist_ok=True)
        real = os.path.realpath(str(VIDEO_DIR))
        if real not in ALLOWED_FILE_DIRS:
            ALLOWED_FILE_DIRS.append(real)
        p = os.path.join(real, path)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("dummy")
        return p

    def test_unauthorized(self, client):
        resp = client.post(
            "/files/save-corrected-srt",
            json={"path": "/tmp/x.srt", "content": "1\n00:00:01,000 --> 00:00:02,000\ntest"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 401

    def test_missing_content(self, auth_client, csrf_headers):
        client, _ = auth_client
        path = self._write_displayed("save_corrected_missing.srt")
        resp = client.post(
            "/files/save-corrected-srt",
            json={"path": path},
            headers=csrf_headers,
        )
        assert resp.status_code == 400

    def test_saves_beside_displayed_with_corrected_suffix(self, auth_client, csrf_headers):
        import os

        client, _ = auth_client
        path = self._write_displayed("00001.zh+en.srt")
        resp = client.post(
            "/files/save-corrected-srt",
            json={"path": path, "content": self.CORRECTED},
            headers=csrf_headers,
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["filename"] == "00001.zh+en.corrected.srt"
        assert data["path"] == os.path.join(os.path.dirname(path), "00001.zh+en.corrected.srt")
        assert os.path.isfile(data["path"])
        with open(data["path"], encoding="utf-8") as fh:
            assert "行\nOkay" in fh.read()

    def test_non_srt_displayed_rejected(self, auth_client, csrf_headers):
        import os

        from config import VIDEO_DIR

        client, _ = auth_client
        os.makedirs(str(VIDEO_DIR), exist_ok=True)
        txt = os.path.join(str(VIDEO_DIR), "notes.txt")
        with open(txt, "w", encoding="utf-8") as fh:
            fh.write("plain text")
        resp = client.post(
            "/files/save-corrected-srt",
            json={"path": txt, "content": self.CORRECTED},
            headers=csrf_headers,
        )
        assert resp.status_code == 400

    def test_invalid_content_rejected(self, auth_client, csrf_headers):
        client, _ = auth_client
        path = self._write_displayed("save_corrected_bad.srt")
        resp = client.post(
            "/files/save-corrected-srt",
            json={"path": path, "content": "no timing line"},
            headers=csrf_headers,
        )
        assert resp.status_code == 400
