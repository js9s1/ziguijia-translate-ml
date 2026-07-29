"""Tests for oldrun SRT scanning, incremental indexing, and compressed output."""

import gzip
import json
import os

import pytest


@pytest.fixture
def oldrun_dirs(tmp_path, monkeypatch):
    """Set up temporary oldrun directory structure and redirect module paths."""
    srt_dir = tmp_path / "oldrun"
    srt_dir.mkdir()
    ts_file = tmp_path / "list_updated.timestamp"
    html_dir = tmp_path / "html"
    html_dir.mkdir()

    monkeypatch.setattr("oldrun.OLDRUN_SRT_DIR", str(srt_dir))
    monkeypatch.setattr("oldrun.OLDRUN_SRT_TIMESTAMP", str(ts_file))
    monkeypatch.setattr("oldrun.HTML_DIR", str(html_dir))

    return {"srt_dir": srt_dir, "ts_file": ts_file, "html_dir": html_dir}


def _make_srt(dirpath, *filenames):
    """Create empty .srt files and return a dict of {name: fullpath}."""
    paths = {}
    for name in filenames:
        p = os.path.join(dirpath, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        p = str(p)  # Ensure string path for cross-Python compat
        with open(p, "a"):
            pass
        paths[name] = p
    return paths


# ── _scan_dir ──────────────────────────────────────────────────


class TestScanDir:
    def test_empty_dir(self, tmp_path):
        from oldrun import _scan_dir

        zh, en, zh_en = _scan_dir(str(tmp_path))
        assert zh == []
        assert en == []
        assert zh_en == []

    def test_categorizes_by_suffix(self, tmp_path):
        from oldrun import _scan_dir

        _make_srt(str(tmp_path), "00001.srt", "00002.en.srt", "00003.zh+en.srt", "notes.txt")
        zh, en, zh_en = _scan_dir(str(tmp_path))
        assert len(zh) == 1 and zh[0]["name"] == "00001.srt"
        assert len(en) == 1 and en[0]["name"] == "00002.en.srt"
        assert len(zh_en) == 1 and zh_en[0]["name"] == "00003.zh+en.srt"

    def test_path_is_absolute(self, tmp_path):
        from oldrun import _scan_dir

        _make_srt(str(tmp_path), "00001.srt")
        zh, _, _ = _scan_dir(str(tmp_path))
        assert os.path.isabs(zh[0]["path"])

    def test_recursive_walk(self, tmp_path):
        from oldrun import _scan_dir

        sub = tmp_path / "sub"
        sub.mkdir()
        _make_srt(str(tmp_path), "a.srt")
        _make_srt(str(sub), "b.srt")
        zh, _, _ = _scan_dir(str(tmp_path))
        assert len(zh) == 2

    def test_ignores_non_srt(self, tmp_path):
        from oldrun import _scan_dir

        _make_srt(str(tmp_path), "a.srt", "b.txt", "c.py", "d.SRT")  # .SRT is uppercase — should match
        zh, en, zh_en = _scan_dir(str(tmp_path))
        assert len(zh) == 2  # a.srt and d.SRT
        assert en == []
        assert zh_en == []

    def test_en_suffix_has_priority_over_zh(self, tmp_path):
        from oldrun import _scan_dir

        _make_srt(str(tmp_path), "foo.en.srt")
        zh, en, zh_en = _scan_dir(str(tmp_path))
        assert zh == []
        assert len(en) == 1
        assert zh_en == []

    def test_zh_en_suffix_has_priority_over_en(self, tmp_path):
        from oldrun import _scan_dir

        _make_srt(str(tmp_path), "foo.zh+en.srt")
        zh, en, zh_en = _scan_dir(str(tmp_path))
        assert zh == []
        assert en == []
        assert len(zh_en) == 1


# ── _collect_incremental ────────────────────────────────────────


class TestCollectIncremental:
    def test_empty_srt_dir(self, oldrun_dirs):
        from oldrun import _collect_incremental

        zh, en, zh_en, changed = _collect_incremental(force=True)
        assert (zh, en, zh_en) == ([], [], [])
        assert changed is True

    def test_scans_multiple_runs(self, oldrun_dirs):
        from oldrun import _collect_incremental

        srt_dir = oldrun_dirs["srt_dir"]
        (srt_dir / "run1" / "srts").mkdir(parents=True)
        (srt_dir / "run2" / "srts").mkdir(parents=True)
        _make_srt(str(srt_dir / "run1" / "srts"), "a.srt")
        _make_srt(str(srt_dir / "run2" / "srts"), "b.en.srt", "c.zh+en.srt")
        zh, en, zh_en, changed = _collect_incremental(force=True)
        assert len(zh) == 1 and zh[0]["name"] == "a.srt"
        assert len(en) == 1 and en[0]["name"] == "b.en.srt"
        assert len(zh_en) == 1 and zh_en[0]["name"] == "c.zh+en.srt"
        assert changed is True

    def test_incremental_skips_when_no_new_dirs(self, oldrun_dirs):
        from oldrun import _collect_incremental

        srt_dir = oldrun_dirs["srt_dir"]
        (srt_dir / "run1" / "srts").mkdir(parents=True)
        _make_srt(str(srt_dir / "run1" / "srts"), "a.srt")
        _collect_incremental(force=True)
        zh, en, zh_en, changed = _collect_incremental()
        assert len(zh) == 1
        assert changed is False

    def test_incremental_picks_up_new_dirs(self, oldrun_dirs):
        from oldrun import _collect_incremental

        srt_dir = oldrun_dirs["srt_dir"]
        (srt_dir / "run1" / "srts").mkdir(parents=True)
        _make_srt(str(srt_dir / "run1" / "srts"), "a.srt")
        _collect_incremental(force=True)
        (srt_dir / "run2" / "srts").mkdir(parents=True)
        _make_srt(str(srt_dir / "run2" / "srts"), "b.srt")
        zh, en, zh_en, changed = _collect_incremental()
        assert len(zh) == 2
        assert changed is True

    def test_sorts_when_merging_new_dirs(self, oldrun_dirs):
        from oldrun import _collect_incremental

        srt_dir = oldrun_dirs["srt_dir"]
        (srt_dir / "run1" / "srts").mkdir(parents=True)
        _make_srt(str(srt_dir / "run1" / "srts"), "c.srt", "a.srt")
        _collect_incremental(force=True)
        (srt_dir / "run2" / "srts").mkdir(parents=True)
        _make_srt(str(srt_dir / "run2" / "srts"), "b.srt")
        zh, _, _, _ = _collect_incremental()
        names = [e["name"] for e in zh]
        assert names == ["a.srt", "b.srt", "c.srt"]

    def test_missing_srt_dir_returns_empty(self, tmp_path, monkeypatch):
        from oldrun import _collect_incremental

        monkeypatch.setattr("oldrun.OLDRUN_SRT_DIR", str(tmp_path / "no_such_dir"))
        monkeypatch.setattr("oldrun.OLDRUN_SRT_TIMESTAMP", str(tmp_path / "ts"))
        zh, en, zh_en, changed = _collect_incremental()
        assert (zh, en, zh_en) == ([], [], [])
        assert changed is False


# ── _write_static_html ──────────────────────────────────────────


class TestWriteStaticHtml:
    def test_creates_html_and_gz(self, oldrun_dirs):
        from oldrun import _write_static_html

        html_dir = oldrun_dirs["html_dir"]
        files = [{"name": "test.srt", "path": "/tmp/test.srt"}]
        _write_static_html("zh", files)
        assert (html_dir / "srt-zh.html").is_file()
        assert (html_dir / "srt-zh.json.gz").is_file()

    def test_html_no_embedded_data_json(self, oldrun_dirs):
        from oldrun import _write_static_html

        html_dir = oldrun_dirs["html_dir"]
        files = [{"name": "test.srt", "path": "/tmp/test.srt"}]
        _write_static_html("zh", files)
        content = (html_dir / "srt-zh.html").read_text()
        assert "var DATA = []" in content
        assert 'var DATA = [{"name":"test.srt"' not in content

    def test_html_contains_decompress_fetch(self, oldrun_dirs):
        from oldrun import _write_static_html

        html_dir = oldrun_dirs["html_dir"]
        _write_static_html("en", [])
        content = (html_dir / "srt-en.html").read_text()
        assert "async" in content
        assert "DecompressionStream" in content
        assert "srt-en.json.gz" in content

    def test_gz_is_valid_compressed_json(self, oldrun_dirs):
        from oldrun import _write_static_html

        html_dir = oldrun_dirs["html_dir"]
        files = [
            {"name": "a.srt", "path": "/tmp/a.srt"},
            {"name": "b.srt", "path": "/tmp/b.srt"},
        ]
        _write_static_html("zh+en", files)
        with gzip.open(html_dir / "srt-zh+en.json.gz", "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
        assert data == files

    def test_gz_smaller_than_raw_json(self, oldrun_dirs):
        from oldrun import _write_static_html

        html_dir = oldrun_dirs["html_dir"]
        many = [{"name": f"{i:05d}.srt", "path": f"/x/y/{i:05d}.srt"} for i in range(100)]
        _write_static_html("zh", many)
        raw = json.dumps(many, ensure_ascii=False).encode("utf-8")
        compressed = (html_dir / "srt-zh.json.gz").read_bytes()
        assert len(compressed) < len(raw)

    def test_all_three_lang_flavors(self, oldrun_dirs):
        from oldrun import _write_static_html

        html_dir = oldrun_dirs["html_dir"]
        _write_static_html("zh", [{"name": "z.srt", "path": "/z.srt"}])
        _write_static_html("en", [{"name": "e.srt", "path": "/e.srt"}])
        _write_static_html("zh+en", [{"name": "ze.srt", "path": "/ze.srt"}])
        assert (html_dir / "srt-zh.html").is_file()
        assert (html_dir / "srt-zh.json.gz").is_file()
        assert (html_dir / "srt-en.html").is_file()
        assert (html_dir / "srt-en.json.gz").is_file()
        assert (html_dir / "srt-zh+en.html").is_file()
        assert (html_dir / "srt-zh+en.json.gz").is_file()


# ── build_all_static_srt ────────────────────────────────────────


class TestBuildAllStaticSrt:
    def test_full_workflow_generates_everything(self, oldrun_dirs):
        from oldrun import _collect_incremental, build_all_static_srt

        srt_dir = oldrun_dirs["srt_dir"]
        html_dir = oldrun_dirs["html_dir"]

        (srt_dir / "run1" / "srts").mkdir(parents=True)
        _make_srt(str(srt_dir / "run1" / "srts"), "a.srt", "b.en.srt", "c.zh+en.srt")
        _collect_incremental(force=True)  # update index / timestamp
        build_all_static_srt()

        for lang in ("zh", "en", "zh+en"):
            assert (html_dir / f"srt-{lang}.html").is_file(), f"missing {lang}.html"
            assert (html_dir / f"srt-{lang}.json.gz").is_file(), f"missing {lang}.json.gz"

    def test_no_rebuild_when_html_newer_than_timestamp(self, oldrun_dirs):
        from oldrun import _collect_incremental, build_all_static_srt

        srt_dir = oldrun_dirs["srt_dir"]
        html_dir = oldrun_dirs["html_dir"]

        (srt_dir / "run1" / "srts").mkdir(parents=True)
        _make_srt(str(srt_dir / "run1" / "srts"), "a.srt")
        _collect_incremental(force=True)
        build_all_static_srt()

        mtimes_before = {lang: os.path.getmtime(str(html_dir / f"srt-{lang}.html")) for lang in ("zh", "en", "zh+en")}

        build_all_static_srt()  # second call — should be a no-op

        for lang in ("zh", "en", "zh+en"):
            assert os.path.getmtime(str(html_dir / f"srt-{lang}.html")) == mtimes_before[lang], (
                f"{lang}.html was unexpectedly rewritten"
            )

    def test_rebuilds_when_html_missing(self, oldrun_dirs):
        from oldrun import _collect_incremental, build_all_static_srt

        srt_dir = oldrun_dirs["srt_dir"]
        html_dir = oldrun_dirs["html_dir"]

        (srt_dir / "run1" / "srts").mkdir(parents=True)
        _make_srt(str(srt_dir / "run1" / "srts"), "a.srt")
        _collect_incremental(force=True)

        # Delete zh.html to force rebuild
        build_all_static_srt()  # first build
        (html_dir / "srt-zh.html").unlink()
        (html_dir / "srt-zh.json.gz").unlink()

        build_all_static_srt()
        assert (html_dir / "srt-zh.html").is_file()


# ── Index persistence ───────────────────────────────────────────


class TestIndexPersistence:
    def test_save_and_load_roundtrip(self, oldrun_dirs):
        from oldrun import _load_index, _save_index

        idx = {"zh": [{"name": "a.srt"}], "en": [], "scanned_dirs": ["/tmp/d1"]}
        _save_index(idx)
        loaded = _load_index()
        assert loaded is not None
        assert loaded["zh"] == idx["zh"]
        assert loaded["scanned_dirs"] == idx["scanned_dirs"]

    def test_load_returns_none_on_missing_file(self, oldrun_dirs):
        from oldrun import _load_index

        # ts_file doesn't exist yet — should return None
        result = _load_index()
        assert result is None

    def test_load_returns_none_on_corrupt_file(self, oldrun_dirs):
        from oldrun import _load_index

        oldrun_dirs["ts_file"].write_text("not valid pickle or json")
        result = _load_index()
        assert result is None
