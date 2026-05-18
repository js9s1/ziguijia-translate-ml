document.addEventListener('click', function(e) {
    var label = e.target.closest('label');
    if (label && label.htmlFor) {
        var input = document.getElementById(label.htmlFor);
        if (input && input.type === 'file') {
            e.preventDefault();
            input.click();
        }
    }
    var row = e.target.closest('.form-row');
    if (row && !label) {
        var fileInput = row.querySelector('input[type="file"]');
        if (fileInput && !fileInput.contains(e.target)) {
            fileInput.click();
        }
    }
});

fetch('/auth/me').then(r => r.json()).then(data => {
    const area = document.getElementById('userArea');
    if (!area) return;
    if (data.authenticated) {
        area.innerHTML = '<span id="userEmail" style="color:#4CAF50;font-weight:bold;">' + escapeHtml(data.user.email) + '</span>' +
            '<span style="color:#ddd;margin:0 6px;">|</span>' +
            '<a href="/my-jobs" style="color:#2196F3;text-decoration:none;">我的任务</a>' +
            '<span style="color:#ddd;margin:0 6px;">|</span>' +
            '<a href="#" id="logoutLink" style="color:#f44336;text-decoration:none;">退出</a>';
        document.getElementById('logoutLink').addEventListener('click', function(e) {
            e.preventDefault();
            fetch('/auth/logout', { method: 'POST' }).then(() => location.href = '/');
        });
    } else {
        area.innerHTML = '<a href="/auth/login" style="color:#4CAF50;text-decoration:none;">登录</a>' +
            '<span style="color:#ddd;margin:0 6px;">|</span>' +
            '<a href="/auth/register" style="color:#4CAF50;text-decoration:none;">注册</a>';
    }
});

