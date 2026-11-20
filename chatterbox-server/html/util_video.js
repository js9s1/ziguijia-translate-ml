/* ── Video page shared init (ningVideo + userVideo) ────── */

var VIDEO_MODE_OPTIONS = {
    ning: [
        { v: 'srt', l: '上传SRT字幕' },
        { v: 'extract-srt', l: '提取SRT字幕（仅识别，不翻译）' },
        { v: 'translate-only', l: '识别并翻译字幕（不生成音频/视频）' },
        { v: 'full', l: '识别翻译并生成配音视频' },
    ],
    custom: [
        { v: 'srt', l: '上传SRT字幕' },
        { v: 'extract-srt', l: '提取SRT字幕（仅识别，不翻译）' },
        { v: 'translate-only', l: '识别并翻译字幕（不生成音频/视频）' },
        { v: 'full', l: '识别翻译并生成配音视频' },
    ],
};

function _isCantonese(langCode) {
    var code = String(langCode || '').toLowerCase();
    return code === 'yue' || code === 'zh-hk' || code === 'zh-yue' || code === 'cantonese';
}

function _buildVideoModeSelect(pageType, langCode) {
    var options = VIDEO_MODE_OPTIONS[pageType] || VIDEO_MODE_OPTIONS.ning;
    var sel = document.getElementById('mode');
    if (!sel) return;
    var cantonese = _isCantonese(langCode);
    var current = sel.value;
    sel.innerHTML = '';
    options.forEach(function(o) {
        // Cantonese has no translation pipeline: only "upload SRT" and
        // "extract SRT only" (no translation involved) are valid.
        if (cantonese && o.v !== 'srt' && o.v !== 'extract-srt') return;
        var opt = document.createElement('option');
        opt.value = o.v;
        opt.textContent = o.l;
        sel.appendChild(opt);
    });
    if (cantonese) {
        sel.value = 'srt';
    } else if (current && options.some(function(o) { return o.v === current; })) {
        sel.value = current;
    }
}

function _showCantoneseNotice(cantonese) {
    var mode = document.getElementById('mode');
    if (!mode) return;
    var row = mode.closest('.form-row');
    if (!row) return;
    var notice = document.getElementById('cantoneseNotice');
    if (!notice) {
        notice = document.createElement('div');
        notice.id = 'cantoneseNotice';
        notice.className = 'info-text';
        notice.textContent = '粤语目标语言仅支持「上传SRT字幕」模式；自动翻译与OCR翻译流程已隐藏。';
        row.insertAdjacentElement('afterend', notice);
    }
    notice.classList.toggle('hidden', !cantonese);
}

function _syncVideoModeVisibility() {
    var mode = document.getElementById('mode');
    if (!mode) return;
    _toggleSrtSection(mode.value);
    _toggleExtractEngineRow(mode.value);
    _toggleAudioParams(mode.value);
    _toggleBlurRow(mode.value);
}

function _toggleSrtSection(mode) {
    var srtSection = document.getElementById('srtSection');
    var srtFile = document.getElementById('srt_file');
    if (!srtSection || !srtFile) return;
    if (mode === 'srt') {
        srtSection.classList.remove('hidden');
        srtFile.setAttribute('required', '');
    } else {
        srtSection.classList.add('hidden');
        srtFile.removeAttribute('required');
    }
}

function _toggleExtractEngineRow(mode) {
    var row = document.getElementById('extractEngineRow');
    if (!row) return;
    row.classList.toggle('hidden', !(mode === 'extract-srt' || mode === 'translate-only' || mode === 'full'));
}

function _toggleAudioParams(mode) {
    var el = document.getElementById('floatSliders');
    if (!el) return;
    el.classList.toggle('hidden', mode === 'extract-srt' || mode === 'translate-only');
}

function _toggleBlurRow(mode) {
    var el = document.getElementById('blurRow');
    if (!el) return;
    el.classList.toggle('hidden', mode === 'extract-srt' || mode === 'translate-only');
}

function getExtractEngineValue() {
    var el = document.querySelector('input[name="extract_engine"]:checked');
    return el ? el.value : 'ocr';
}

function collectCommonVideoFields(formData) {
    formData = formData || new FormData();
    formData.append('target_language', document.getElementById('target_language').value);
    var audioParams = collectAudioParams();
    var video = {};
    var modeEl = document.getElementById('mode');
    if (modeEl) video.mode = modeEl.value;
    var blurEl = document.querySelector('input[name="blur"]:checked');
    if (blurEl) video.blur = blurEl.value;
    var extractEngineEl = document.querySelector('input[name="extract_engine"]:checked');
    if (extractEngineEl) video.extract_engine = extractEngineEl.value;
    formData.append('job_params', JSON.stringify({ audio: audioParams, video: video }));
    return formData;
}

function initVideoForm(config) {
    config = config || {};
    var pageType = config.pageType || 'ning';

    _buildVideoModeSelect(pageType);

    document.getElementById('mode').addEventListener('change', function() {
        _syncVideoModeVisibility();
    });

    _syncVideoModeVisibility();
    _showCantoneseNotice(false);

    var langSelect = initLanguageSelect('langSelectContainer');
    if (langSelect) {
        langSelect.addEventListener('change', function() {
            _buildVideoModeSelect(pageType, this.value);
            _showCantoneseNotice(_isCantonese(this.value));
            _syncVideoModeVisibility();
        });
    }
    initAudioParams(
        'floatSliders',
        pageType === 'ning' ? 'video_ning' : 'video_custom',
        langSelect
    );

    initResultSection('resultSection', 'checkProgressLink');

    if (config.onSubmit) {
        document.getElementById('videoForm').addEventListener('submit', config.onSubmit);
    }
}
