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

/* ── Shared API error handling ─────────────────────────── */

function handleApiResponse(response, redirectPath) {
    // Centralize 401 handling + JSON parsing for API POSTs.  Resolves to
    // the parsed body, or null when the user is redirected to login or the
    // response is not JSON.
    if (response.status === 401) {
        alert('请先登录');
        window.location.href = '/auth/login?next=' + encodeURIComponent(redirectPath || window.location.pathname);
        return Promise.resolve(null);
    }
    return response.json().catch(function () { return null; });
}

function showApiError(data, prefix) {
    // Alert the server error message if *data* carries one.
    // Returns true when an error was shown (caller should stop).
    if (data && data.error) {
        alert((prefix || '错误') + ': ' + data.error);
        return true;
    }
    return false;
}

/* ── Language select (shared) ──────────────────────────── */

var _langOptions = null;
var _langNames = null;  // code → native display name, e.g. {en: 'English', zh: '中文'}

// Country flag emojis for language codes
var _langFlags = {
    ar: '🇸🇦', da: '🇩🇰', de: '🇩🇪', el: '🇬🇷', en: '🇬🇧',
    es: '🇪🇸', fi: '🇫🇮', fr: '🇫🇷', he: '🇮🇱', hi: '🇮🇳',
    it: '🇮🇹', ja: '🇯🇵', ko: '🇰🇷', ms: '🇲🇾', nl: '🇳🇱',
    no: '🇳🇴', pl: '🇵🇱', pt: '🇵🇹', ru: '🇷🇺', sv: '🇸🇪',
    sw: '🇹🇿', tr: '🇹🇷', vi: '🇻🇳', th: '🇹🇭', id: '🇮🇩', zh: '🇨🇳',
    yue: '🇭🇰',
};

function loadLanguages() {
    if (_langOptions) return Promise.resolve(_langOptions);
    return fetch('/api/languages')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            _langNames = {};
            _langOptions = (data.languages || []).map(function(item) {
                var displayName = item.native_name || item.name;
                _langNames[item.code] = displayName;
                var flag = _langFlags[item.code] || '';
                return { v: item.code, l: flag + ' ' + displayName + ' (' + item.code + ')' };
            });
            return _langOptions;
        })
        .catch(function() {
            // Fallback to hardcoded list if API unavailable
            _langNames = {
                ar:'العربية',da:'Dansk',de:'Deutsch',el:'Ελληνικά',
                en:'English',es:'Español',fi:'Suomi',fr:'Français',
                he:'עברית',hi:'हिन्दी',it:'Italiano',ja:'日本語',
                ko:'한국어',ms:'Bahasa Melayu',nl:'Nederlands',no:'Norsk',
                pl:'Polski',pt:'Português',ru:'Русский',sv:'Svenska',
                sw:'Kiswahili',tr:'Türkçe',zh:'中文',yue:'粤语',vi:'Tiếng Việt',th:'ไทย',id:'Bahasa Indonesia'
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

/* ── Audio parameter display (result page) ────────────── */

var _AUDIO_PARAM_LABELS = {
    cfg_value: 'CFG',
    inference_timesteps: '推理步数',
    seed: '种子',
    retry_badcase: 'Badcase重试',
    temperature: '温度',
    cfg_weight: 'CFG权重',
    exaggeration: '夸张度',
};
var _AUDIO_ENGINE_LABELS = {
    chatterbox: 'Chatterbox',
    chatterbox_id: 'Chatterbox (Indonesian)',
    voxcpm: 'VoxCPM2',
};

function formatAudioParams(ap) {
    var parts = [];
    if (!ap || typeof ap !== 'object') return parts;
    if (ap.engine) parts.push('模型: ' + (_AUDIO_ENGINE_LABELS[ap.engine] || ap.engine));
    for (var key in ap) {
        if (key === 'engine') continue;
        var value = ap[key];
        if (typeof value === 'boolean') value = value ? '是' : '否';
        parts.push((_AUDIO_PARAM_LABELS[key] || key) + ': ' + value);
    }
    return parts;
}

function audioParamParts(data) {
    var parts = formatAudioParams(data && data.audio_params);
    if (parts.length) return parts;
    // Legacy jobs that predate the per-engine schema.
    if (data && data.temperature != null) parts.push('温度: ' + data.temperature);
    if (data && data.cfg_weight != null) parts.push('CFG: ' + data.cfg_weight);
    if (data && data.exaggeration != null) parts.push('夸张: ' + data.exaggeration);
    return parts;
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

/* ── Engine-specific audio params (schema-driven) ─────── */

var _audioSchema = null;
var _audioPipeline = 'tts';
var _audioContainerId = 'floatSliders';
var _audioValuesByEngine = {};
var _audioRequestSeq = 0;

function _audioFieldId(key) {
    return 'audio_param_' + key;
}

function _currentAudioValues() {
    var values = {};
    if (!_audioSchema) return values;
    _audioSchema.fields.forEach(function(field) {
        var el = document.getElementById(_audioFieldId(field.key));
        if (!el) return;
        if (field.type === 'bool') {
            values[field.key] = el.tagName === 'SELECT' ? el.value === '1' : !!el.checked;
        } else {
            values[field.key] = parseFloat(el.value);
        }
    });
    return values;
}

function syncAudioParams() {
    if (!_audioSchema) return null;
    var values = _currentAudioValues();
    _audioValuesByEngine[_audioSchema.engine] = values;
    var payload = { engine: _audioSchema.engine };
    for (var key in values) payload[key] = values[key];

    var container = document.getElementById(_audioContainerId);
    if (!container) return payload;
    var hidden = document.getElementById('audio_params');
    if (!hidden) {
        hidden = document.createElement('input');
        hidden.type = 'hidden';
        hidden.id = 'audio_params';
        hidden.name = 'audio_params';
        container.appendChild(hidden);
    }
    hidden.value = JSON.stringify(payload);

    // Canonical nested envelope for pages that submit the form directly.
    var jobParams = document.getElementById('job_params');
    if (!jobParams) {
        jobParams = document.createElement('input');
        jobParams.type = 'hidden';
        jobParams.id = 'job_params';
        jobParams.name = 'job_params';
        container.appendChild(jobParams);
    }
    jobParams.value = JSON.stringify({ audio: payload });
    return payload;
}

function collectAudioParams() {
    return syncAudioParams();
}

function _fallbackAudioSchema(language) {
    var code = String(language || '').toLowerCase();
    var cantonese = code === 'yue' || code === 'zh-hk' || code === 'zh-yue' || code === 'cantonese';
    if (cantonese) {
        return {
            engine: 'voxcpm',
            engine_label: 'VoxCPM2',
            fields: [
                { key: 'cfg_value', label: 'CFG / 引导强度', type: 'float', min: 1.0, max: 3.0, step: 0.05, default: 2.0, help: 'VoxCPM2 CFG guidance；默认 2.0' },
                { key: 'inference_timesteps', label: '推理步数', type: 'int', min: 5, max: 30, step: 1, default: 10, help: '越高越慢，通常 10 足够' },
                { key: 'seed', label: '随机种子', type: 'int', min: -1, max: 2147483647, step: 1, default: 42, help: '-1 表示每次随机' },
                { key: 'retry_badcase', label: 'Badcase 重试', type: 'bool', default: false, help: '音频明显过长/异常时自动重试' },
            ],
        };
    }
    return {
        engine: 'chatterbox',
        engine_label: 'Chatterbox',
        fields: [
            { key: 'temperature', label: '温度', type: 'float', min: 0.2, max: 1.2, step: 0.01, default: 0.6, help: '越高输出越随机，越低越稳定' },
            { key: 'cfg_weight', label: 'CFG 权重', type: 'float', min: 0.0, max: 1.5, step: 0.01, default: 0.25, help: '控制生成音频与提示的契合度' },
            { key: 'exaggeration', label: '口音夸张度', type: 'float', min: 0.0, max: 1.0, step: 0.01, default: 0.3, help: '口音/语气的夸张程度' },
        ],
    };
}

function _renderAudioParams(schema) {
    var container = document.getElementById(_audioContainerId);
    if (!container) return;
    if (_audioSchema && _audioSchema.engine) {
        _audioValuesByEngine[_audioSchema.engine] = _currentAudioValues();
    }
    _audioSchema = schema;
    var saved = _audioValuesByEngine[schema.engine] || {};

    container.innerHTML = '';
    var fieldset = document.createElement('fieldset');
    fieldset.style.cssText = 'border:1px solid var(--color-border,#555);border-radius:8px;padding:12px 16px 4px 16px;margin:0;width:100%;box-sizing:border-box;';
    var legend = document.createElement('legend');
    legend.style.cssText = 'color:var(--color-text-medium,#aaa);font-weight:bold;font-size:0.9em;padding:0 6px;';
    legend.textContent = '音频模型参数 · ' + (schema.engine_label || schema.engine || '');
    fieldset.appendChild(legend);
    container.appendChild(fieldset);

    schema.fields.forEach(function(field) {
        var value = saved[field.key];
        if (value == null) value = field.default;
        var id = _audioFieldId(field.key);
        var row = document.createElement('div');
        row.className = 'form-row';
        if (field.type === 'bool') {
            row.innerHTML =
                '<label for="' + id + '">' + field.label + '</label>' +
                '<select id="' + id + '" class="float-input">' +
                '<option value="1">是</option><option value="0">否</option></select>' +
                '<span class="slider-hint">' + (field.help || '') + '</span>';
            fieldset.appendChild(row);
            var select = row.querySelector('select');
            select.value = value ? '1' : '0';
            select.addEventListener('change', syncAudioParams);
        } else {
            row.innerHTML =
                '<label for="' + id + '">' + field.label + '</label>' +
                '<input type="number" class="float-input" id="' + id + '" ' +
                'min="' + field.min + '" max="' + field.max + '" step="' + field.step + '" value="' + value + '">' +
                '<span class="slider-hint">(' + field.min + ' – ' + field.max + ') ' + (field.help || '') + '</span>';
            fieldset.appendChild(row);
            var input = row.querySelector('input');
            input.addEventListener('blur', function() {
                validateFloatInput(this, field.min, field.max);
                syncAudioParams();
            });
            input.addEventListener('input', function() {
                var v = parseFloat(this.value);
                if (!isNaN(v) && v >= field.min && v <= field.max) {
                    this.setCustomValidity('');
                } else {
                    this.setCustomValidity('请输入 ' + field.min + ' 到 ' + field.max + ' 之间的值');
                }
                syncAudioParams();
            });
            input.addEventListener('change', syncAudioParams);
        }
    });
    syncAudioParams();
}

function _loadAudioSchema(pipeline, language) {
    var url = '/api/param-schema?pipeline=' + encodeURIComponent(pipeline || 'tts') +
        '&language=' + encodeURIComponent(language || 'en');
    return fetch(url)
        .then(function(response) {
            if (!response.ok) throw new Error('param schema unavailable');
            return response.json();
        })
        .catch(function() {
            return _fallbackAudioSchema(language);
        });
}

function initAudioParams(containerId, pipeline, langSelect) {
    var container = document.getElementById(containerId);
    if (!container) return;
    _audioContainerId = containerId;
    _audioPipeline = pipeline || 'tts';

    function refresh() {
        var language = langSelect && langSelect.value ? langSelect.value : 'en';
        var seq = ++_audioRequestSeq;
        _loadAudioSchema(_audioPipeline, language).then(function(schema) {
            if (seq !== _audioRequestSeq) return;
            _renderAudioParams(schema);
        });
    }

    refresh();
    if (langSelect) langSelect.addEventListener('change', refresh);
}

/* ── Download file (shared) ────────────────────────────── */

function downloadFile(url, filename, fetchOptions) {
    fetchOptions = fetchOptions || {};
    var isGet = !fetchOptions.method || fetchOptions.method === 'GET';
    var isIos = /iPhone|iPad|iPod/.test(navigator.userAgent);
    var isAndroid = /Android/i.test(navigator.userAgent);

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
    } else if (isGet && isAndroid) {
        // Android browsers (Huawei/HarmonyOS etc.) frequently ignore the
        // download attribute on direct links, so fetch + blob first.
        _tryFetchDownload(url, filename, fetchOptions, function(succeeded) {
            if (succeeded) return;
            _tryDirectLink(url, filename, function(succeeded2) {
                if (succeeded2) return;
                window.location.href = url;
            });
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
    fetch('/auth/me').then(function(r) { return r.json(); }).then(function(data) {
        if (!data.authenticated) {
            window.location.href = '/auth/login?next=' + encodeURIComponent(redirectPath || window.location.pathname);
            return;
        }

        var overlay = document.getElementById('loadingOverlay');
        if (overlay) overlay.classList.add('active');

        var body = bodyOrFormData;
        if (!(body instanceof FormData)) {
            body = new FormData();
            for (var key in bodyOrFormData) {
                var value = bodyOrFormData[key];
                if (value === null || value === undefined) continue;
                if (typeof value === 'object') value = JSON.stringify(value);
                body.append(key, value);
            }
        }

        // AJAX headers keep the server from converting the JSON response
        // into a redirect, so errors can be shown in-page like other errors
        // instead of rendering raw JSON (black screen).
        fetchWithCsrf(url, {
            method: 'POST',
            body: body,
            headers: { 'Accept': 'application/json', 'X-Requested-With': 'XMLHttpRequest' }
        }).then(function(response) {
            return handleApiResponse(response, redirectPath);
        }).then(function(data) {
            if (overlay) overlay.classList.remove('active');
            if (!data) return;
            if (showApiError(data)) return;
            if (data.cached_found && typeof window.showCachedPrompt === 'function') {
                window.showCachedPrompt(data);
                return;
            }
            if (data.access_code) {
                window.location.href = '/result?code=' + data.access_code;
            }
        }).catch(function(err) {
            if (overlay) overlay.classList.remove('active');
            alert('错误: ' + (err && err.message ? err.message : err));
        });
    });
}


