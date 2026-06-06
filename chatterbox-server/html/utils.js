function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function unescapeHtml(html) {
    const div = document.createElement('div');
    div.innerHTML = html;
    return div.textContent || div.innerText || '';
}

/* ── CSRF token management ────────────────────────────── */

var _csrfToken = null;

function getCsrfToken() {
    if (_csrfToken) return Promise.resolve(_csrfToken);
    return fetch('/auth/csrf-token')
        .then(function (r) { return r.json(); })
        .then(function (data) {
            _csrfToken = data.csrf_token;
            return _csrfToken;
        });
}

function fetchWithCsrf(url, options) {
    options = options || {};
    return getCsrfToken().then(function (token) {
        options.headers = options.headers || {};
        if (typeof options.headers === 'object' && !(options.headers instanceof Headers)) {
            options.headers['X-CSRF-Token'] = token;
        }
        return fetch(url, options);
    });
}

/* ── Language select (shared) ──────────────────────────── */

var _langOptions = [
    { v: 'ar', l: '🇸🇦 阿拉伯语 (ar)' },
    { v: 'da', l: '🇩🇰 丹麦语 (da)' },
    { v: 'de', l: '🇩🇪 德语 (de)' },
    { v: 'el', l: '🇬🇷 希腊语 (el)' },
    { v: 'en', l: '🇬🇧 英文 (en)' },
    { v: 'es', l: '🇪🇸 西班牙语 (es)' },
    { v: 'fi', l: '🇫🇮 芬兰语 (fi)' },
    { v: 'fr', l: '🇫🇷 法语 (fr)' },
    { v: 'he', l: '🇮🇱 希伯来语 (he)' },
    { v: 'hi', l: '🇮🇳 印地语 (hi)' },
    { v: 'it', l: '🇮🇹 意大利语 (it)' },
    { v: 'ja', l: '🇯🇵 日语 (ja)' },
    { v: 'ko', l: '🇰🇷 韩语 (ko)' },
    { v: 'ms', l: '🇲🇾 马来语 (ms)' },
    { v: 'nl', l: '🇳🇱 荷兰语 (nl)' },
    { v: 'no', l: '🇳🇴 挪威语 (no)' },
    { v: 'pl', l: '🇵🇱 波兰语 (pl)' },
    { v: 'pt', l: '🇵🇹 葡萄牙语 (pt)' },
    { v: 'ru', l: '🇷🇺 俄语 (ru)' },
    { v: 'sv', l: '🇸🇪 瑞典语 (sv)' },
    { v: 'sw', l: '🇹🇿 斯瓦希里语 (sw)' },
    { v: 'tr', l: '🇹🇷 土耳其语 (tr)' },
    { v: 'zh', l: '🇨🇳 中文 (zh)' },
];

function initLanguageSelect(containerId, defaultVal) {
    var container = document.getElementById(containerId);
    if (!container) return;
    var sel = document.createElement('select');
    sel.id = 'target_language';
    sel.name = 'target_language';
    _langOptions.forEach(function(o) {
        var opt = document.createElement('option');
        opt.value = o.v;
        opt.textContent = o.l;
        if (o.v === (defaultVal || 'en')) opt.selected = true;
        sel.appendChild(opt);
    });
    container.appendChild(sel);
    return sel;
}

/* ── Float slider replacements (mobile-friendly) ───────── */

function validateFloatInput(input, min, max) {
    var val = parseFloat(input.value);
    if (isNaN(val)) {
        input.value = input.defaultValue;
        return;
    }
    if (val < min) { input.value = min; }
    else if (val > max) { input.value = max; }
}

var _floatSlidersConfig = [
    { id: 'temperature',   label: '温度',       min: 0.5, max: 1.5, step: 0.01, val: 0.6 },
    { id: 'cfg_weight',    label: 'CFG 权重',    min: 0,   max: 1.5, step: 0.01, val: 0.26 },
    { id: 'exaggeration',  label: '口音夸张度',  min: 0,   max: 1.0, step: 0.01, val: 0.3 },
];

function initFloatSliders(containerId) {
    var container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    _floatSlidersConfig.forEach(function(cfg) {
        var row = document.createElement('div');
        row.className = 'form-row';
        row.innerHTML =
            '<label for="' + cfg.id + '">' + cfg.label + '</label>' +
            '<input type="number" class="float-input" id="' + cfg.id + '" name="' + cfg.id + '" ' +
            'min="' + cfg.min + '" max="' + cfg.max + '" step="' + cfg.step + '" value="' + cfg.val + '">' +
            '<span class="slider-hint">(' + cfg.min + ' – ' + cfg.max + ')</span>';
        container.appendChild(row);
        var input = row.querySelector('input');
        input.addEventListener('blur', function() {
            validateFloatInput(this, cfg.min, cfg.max);
        });
        input.addEventListener('input', function() {
            var v = parseFloat(this.value);
            if (!isNaN(v) && v >= cfg.min && v <= cfg.max) {
                this.setCustomValidity('');
            } else {
                this.setCustomValidity('请输入 ' + cfg.min + ' 到 ' + cfg.max + ' 之间的值');
            }
        });
    });
}

/* ── Result section (shared) ──────────────────────────── */

function initResultSection(containerId, linkId) {
    var container = document.getElementById(containerId);
    if (!container) return;
    container.className = 'hidden';
    container.innerHTML =
        '<label>访问码：</label>' +
        '<span id="accessCode" class="access-code"></span>' +
        '<a href="#" id="' + (linkId || 'checkLink') + '" class="btn-download" style="margin-left:10px;vertical-align:middle;">查看进度 →</a>';
    return container;
}

function showResult(accessCode, resultUrl, linkId) {
    document.getElementById('accessCode').textContent = accessCode;
    var link = document.getElementById(linkId || 'checkLink');
    link.href = resultUrl || '/result?code=' + accessCode;
    var section = link.closest('.hidden') || document.getElementById('resultSection');
    section.classList.remove('hidden');
    fetch('/srt/status/' + accessCode).then(function(r) { return r.json(); }).then(function(s) {
        var labels = {pending:'等待中', processing:'处理中', completed:'已完成', failed:'已失败'};
        link.textContent = (labels[s.status] || '查看进度') + ' →';
    }).catch(function(err) {
        console.warn('Failed to fetch job status for ' + accessCode + ':', err);
    });
}
