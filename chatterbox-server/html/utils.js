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

var _langOptions = null;
var _langNames = null;  // code → name map, e.g. {en: 'English', zh: 'Chinese'}

// Country flag emojis for language codes
var _langFlags = {
    ar: '🇸🇦', da: '🇩🇰', de: '🇩🇪', el: '🇬🇷', en: '🇬🇧',
    es: '🇪🇸', fi: '🇫🇮', fr: '🇫🇷', he: '🇮🇱', hi: '🇮🇳',
    it: '🇮🇹', ja: '🇯🇵', ko: '🇰🇷', ms: '🇲🇾', nl: '🇳🇱',
    no: '🇳🇴', pl: '🇵🇱', pt: '🇵🇹', ru: '🇷🇺', sv: '🇸🇪',
    sw: '🇹🇿', tr: '🇹🇷', vi: '🇻🇳', th: '🇹🇭', id: '🇮🇩', zh: '🇨🇳',
};

function loadLanguages() {
    if (_langOptions) return Promise.resolve(_langOptions);
    return fetch('/api/languages')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            _langNames = {};
            _langOptions = (data.languages || []).map(function(item) {
                _langNames[item.code] = item.name;
                var flag = _langFlags[item.code] || '';
                return { v: item.code, l: flag + ' ' + item.name + ' (' + item.code + ')' };
            });
            return _langOptions;
        })
        .catch(function() {
            // Fallback to hardcoded list if API unavailable
            _langNames = {
                ar:'Arabic',da:'Danish',de:'German',el:'Greek',
                en:'English',es:'Spanish',fi:'Finnish',fr:'French',
                he:'Hebrew',hi:'Hindi',it:'Italian',ja:'Japanese',
                ko:'Korean',ms:'Malay',nl:'Dutch',no:'Norwegian',
                pl:'Polish',pt:'Portuguese',ru:'Russian',sv:'Swedish',
                sw:'Swahili',tr:'Turkish',zh:'Chinese',vi:'Vietnamese',th:'Thai',id:'Indonesian'
            };
            _langOptions = [];
            for (var k in _langNames) {
                _langOptions.push({v: k, l: (_langFlags[k]||'') + ' ' + _langNames[k] + ' (' + k + ')'});
            }
            return _langOptions;
        });
}

function getLangName(code) {
    if (_langNames && _langNames[code]) return _langNames[code];
    return code;
}

function initLanguageSelect(containerId, defaultVal) {
    var container = document.getElementById(containerId);
    if (!container) return;
    var sel = document.createElement('select');
    sel.id = 'target_language';
    sel.name = 'target_language';
    loadLanguages().then(function(options) {
        options.forEach(function(o) {
            var opt = document.createElement('option');
            opt.value = o.v;
            opt.textContent = o.l;
            if (o.v === (defaultVal || 'en')) opt.selected = true;
            sel.appendChild(opt);
        });
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
    { id: 'temperature',   label: '温度',       min: 0.5, max: 1.5, step: 0.01, val: 0.6,  desc: '越高输出越随机，越低越稳定' },
    { id: 'cfg_weight',    label: 'CFG 权重',    min: 0,   max: 1.5, step: 0.01, val: 0.25, desc: '控制生成音频与提示的契合度' },
    { id: 'exaggeration',  label: '口音夸张度',  min: 0,   max: 1.0, step: 0.01, val: 0.5,  desc: '口音/语气的夸张程度' },
];

function initFloatSliders(containerId, label) {
    var container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    var wrapper = container;
    if (label) {
        var fieldset = document.createElement('fieldset');
        fieldset.style.cssText = 'border:1px solid var(--color-border,#555);border-radius:8px;padding:12px 16px 4px 16px;margin:0;width:100%;box-sizing:border-box;';
        var legend = document.createElement('legend');
        legend.style.cssText = 'color:var(--color-text-medium,#aaa);font-weight:bold;font-size:0.9em;padding:0 6px;';
        legend.textContent = label;
        fieldset.appendChild(legend);
        container.appendChild(fieldset);
        wrapper = fieldset;
    }
    _floatSlidersConfig.forEach(function(cfg) {
        var row = document.createElement('div');
        row.className = 'form-row';
        row.innerHTML =
            '<label for="' + cfg.id + '">' + cfg.label + '</label>' +
            '<input type="number" class="float-input" id="' + cfg.id + '" name="' + cfg.id + '" ' +
            'min="' + cfg.min + '" max="' + cfg.max + '" step="' + cfg.step + '" value="' + cfg.val + '">' +
            '<span class="slider-hint">(' + cfg.min + ' – ' + cfg.max + ') ' + (cfg.desc || '') + '</span>';
        wrapper.appendChild(row);
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

/* ── Auto-init audio params on all pages with #floatSliders ── */
if (document.getElementById('floatSliders')) {
    initFloatSliders('floatSliders', '音频模型参数');
}

/* ── Download file (shared) ────────────────────────────── */

function downloadFile(url, filename, fetchOptions) {
    fetchOptions = fetchOptions || {};
    var isGet = !fetchOptions.method || fetchOptions.method === 'GET';
    var isIos = /iPhone|iPad|iPod/.test(navigator.userAgent);

    if (isGet && isIos) {
        _checkFileSize(url, function(contentLength) {
            if (contentLength && contentLength > 1048576) {
                window.location.href = url;
            } else {
                _tryDirectLink(url, filename, function(succeeded) {
                    if (succeeded) return;
                    _tryFetchDownload(url, filename, fetchOptions, function(succeeded2) {
                        if (succeeded2) return;
                        window.location.href = url;
                    });
                });
            }
        });
    } else if (isGet) {
        _tryDirectLink(url, filename, function(succeeded) {
            if (succeeded) return;
            // Strategy 2: fetch + blob + createObjectURL
            _tryFetchDownload(url, filename, fetchOptions, function(succeeded2) {
                if (succeeded2) return;
                // Strategy 3: navigation fallback
                window.location.href = url;
            });
        });
    } else {
        _tryFormPost(url, filename, fetchOptions, function(succeeded) {
            if (succeeded) return;
            // Strategy 2: fetch + blob + createObjectURL
            _tryFetchDownload(url, filename, fetchOptions, function(succeeded2) {
                if (succeeded2) return;
                alert('下载失败: 无法获取文件');
            });
        });
    }
}

function _tryFetchDownload(url, filename, fetchOptions, callback) {
    var opts = {};
    for (var k in fetchOptions) {
        if (k !== 'body' && k !== 'headers') opts[k] = fetchOptions[k];
    }
    if (opts.credentials === undefined) {
        opts.credentials = 'same-origin';
    }
    // Clone headers for the fetch (avoid mutating caller's object)
    if (fetchOptions.headers) {
        opts.headers = {};
        var h = fetchOptions.headers;
        if (h instanceof Headers) {
            h.forEach(function(v, k) { opts.headers[k] = v; });
        } else {
            for (var kh in h) { opts.headers[kh] = h[kh]; }
        }
    }

    fetch(url, opts)
        .then(function(r) {
            if (!r.ok) throw new Error('下载失败 (' + r.status + ')');
            return r.blob();
        })
        .then(function(blob) {
            var a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            setTimeout(function() { URL.revokeObjectURL(a.href); }, 100);
            callback(true);
        })
        .catch(function(e) {
            console.warn('Strategy 2 (fetch+blob) failed: ' + e.message + ', trying next...');
            callback(false);
        });
}

function _tryDirectLink(url, filename, callback) {
    try {
        var a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.target = '_blank';
        a.rel = 'noopener';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        // Give the browser a moment to start the download before reporting success
        setTimeout(function() { callback(true); }, 200);
    } catch(e) {
        console.warn('Strategy 1 (direct link) failed: ' + e.message);
        callback(false);
    }
}

function _tryFormPost(url, filename, fetchOptions, callback) {
    try {
        var form = document.createElement('form');
        form.method = 'POST';
        form.action = url;
        form.style.display = 'none';
        form.target = '_blank';

        var body = fetchOptions.body;
        if (body && typeof body === 'string') {
            try {
                var json = JSON.parse(body);
                if (json && typeof json === 'object') {
                    if (Array.isArray(json.files)) {
                        // Flatten files array into form fields
                        json.files.forEach(function(f, i) {
                            for (var key in f) {
                                var input = document.createElement('input');
                                input.type = 'hidden';
                                input.name = 'files[' + i + '][' + key + ']';
                                input.value = f[key];
                                form.appendChild(input);
                            }
                        });
                    } else {
                        for (var key in json) {
                            var input = document.createElement('input');
                            input.type = 'hidden';
                            input.name = key;
                            input.value = typeof json[key] === 'string' ? json[key] : JSON.stringify(json[key]);
                            form.appendChild(input);
                        }
                    }
                }
            } catch(_) {
                var input = document.createElement('input');
                input.type = 'hidden';
                input.name = 'body';
                input.value = body;
                form.appendChild(input);
            }
        }

        // Carry over CSRF token if present in headers
        if (fetchOptions.headers) {
            var csrf = fetchOptions.headers['X-CSRF-Token'];
            if (!csrf && fetchOptions.headers instanceof Headers) {
                csrf = fetchOptions.headers.get('X-CSRF-Token');
            }
            if (csrf) {
                var csrfInput = document.createElement('input');
                csrfInput.type = 'hidden';
                csrfInput.name = 'csrf_token';
                csrfInput.value = csrf;
                form.appendChild(csrfInput);
            }
        }

        document.body.appendChild(form);
        form.submit();
        document.body.removeChild(form);
        setTimeout(function() { callback(true); }, 200);
    } catch(e) {
        console.warn('Strategy 1 (form POST) failed: ' + e.message);
        callback(false);
    }
}

function _checkFileSize(url, callback) {
    var xhr = new XMLHttpRequest();
    xhr.open('HEAD', url, true);
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            var len = parseInt(xhr.getResponseHeader('Content-Length'), 10);
            callback(len || 0);
        }
    };
    xhr.onerror = function() { callback(0); };
    xhr.send();
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
    fetch('/api/jobs/' + accessCode + '/status').then(function(r) { return r.json(); }).then(function(s) {
        var labels = {pending:'等待中', processing:'处理中', completed:'已完成', failed:'已失败'};
        link.textContent = (labels[s.status] || '查看进度') + ' →';
    }).catch(function(err) {
        console.warn('Failed to fetch job status for ' + accessCode + ':', err);
    });
}

/* ── Form submission helper (shared) ────────────────────── */

function submitJob(url, bodyOrFormData, resultLinkId, redirectPath) {
    var overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.classList.add('active');

    var options = { method: 'POST' };
    if (bodyOrFormData instanceof FormData) {
        options.body = bodyOrFormData;
    } else {
        options.headers = { 'Content-Type': 'application/json' };
        options.body = JSON.stringify(bodyOrFormData);
    }

    return fetch(url, options)
        .then(function(r) {
            if (r.status === 401) {
                if (overlay) overlay.classList.remove('active');
                var loginUrl = '/auth/login?next=' + encodeURIComponent(redirectPath || window.location.pathname);
                if (confirm('请先登录')) { window.location.href = loginUrl; }
                return Promise.reject(new Error('auth'));
            }
            return r.json();
        })
        .then(function(data) {
            if (overlay) overlay.classList.remove('active');
            if (!data) return;
            if (data.error) { alert('错误: ' + data.error); return; }
            if (data.access_code) {
                showResult(data.access_code, '/result?code=' + data.access_code, resultLinkId);
            }
        })
        .catch(function(err) {
            if (overlay) overlay.classList.remove('active');
            if (err.message !== 'auth') alert('错误: ' + err.message);
        });
}


