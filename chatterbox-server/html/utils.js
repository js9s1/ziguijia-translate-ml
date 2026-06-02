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
