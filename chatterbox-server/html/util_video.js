/* ── Video page shared init (ningVideo + userVideo) ────── */

var VIDEO_MODE_OPTIONS = {
    ning: [
        { v: 'srt', l: '上传SRT字幕' },
        { v: 'auto', l: '语音识别翻译（Whisper + HY-MT）' },
        { v: 'ocr', l: 'OCR识别翻译（RapidVideOCR + HY-MT）' },
        { v: 'ocr-translate-only', l: 'OCR识别并生成翻译字幕（不生成音频/视频）' },
    ],
    custom: [
        { v: 'srt', l: '上传SRT字幕' },
        { v: 'auto', l: '语音识别翻译（Whisper + HY-MT）' },
        { v: 'ocr', l: 'OCR识别翻译（RapidVideOCR + HY-MT）' },
        { v: 'ocr-translate-only', l: 'OCR识别并生成翻译字幕（不生成音频/视频）' },
    ],
};

function _buildVideoModeSelect(pageType) {
    var options = VIDEO_MODE_OPTIONS[pageType] || VIDEO_MODE_OPTIONS.ning;
    var sel = document.getElementById('mode');
    if (!sel) return;
    options.forEach(function(o) {
        var opt = document.createElement('option');
        opt.value = o.v;
        opt.textContent = o.l;
        sel.appendChild(opt);
    });
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

function _toggleOcrOnlyRadio(mode) {
    var row = document.getElementById('ocrOnlyRow');
    if (!row) return;
    row.classList.toggle('hidden', mode !== 'ocr-translate-only');
}

function _toggleAudioParams(mode) {
    var el = document.getElementById('floatSliders');
    if (!el) return;
    el.classList.toggle('hidden', mode === 'ocr-translate-only');
}

function getOcrOnlyValue() {
    var el = document.querySelector('input[name="ocr_only"]:checked');
    return el ? el.value : 'yes';
}

function collectCommonVideoFields(formData) {
    formData = formData || new FormData();
    formData.append('temperature', document.getElementById('temperature').value);
    formData.append('target_language', document.getElementById('target_language').value);
    formData.append('cfg_weight', document.getElementById('cfg_weight').value);
    formData.append('exaggeration', document.getElementById('exaggeration').value);
    return formData;
}

function initVideoForm(config) {
    config = config || {};
    var pageType = config.pageType || 'ning';

    _buildVideoModeSelect(pageType);

    document.getElementById('mode').addEventListener('change', function() {
        _toggleSrtSection(this.value);
        _toggleOcrOnlyRadio(this.value);
        _toggleAudioParams(this.value);
    });

    _toggleSrtSection(document.getElementById('mode').value);
    _toggleOcrOnlyRadio(document.getElementById('mode').value);
    _toggleAudioParams(document.getElementById('mode').value);

    initFloatSliders('floatSliders', '音频模型参数');
    initLanguageSelect('langSelectContainer');
    initResultSection('resultSection', 'checkProgressLink');

    if (config.onSubmit) {
        document.getElementById('videoForm').addEventListener('submit', config.onSubmit);
    }
}
