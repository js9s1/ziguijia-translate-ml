document.getElementById('temperature').addEventListener('input', function(e) {
    document.getElementById('tempValue').textContent = e.target.value;
});

const ttsForm = document.getElementById('ttsForm');
if (!ttsForm) return;
ttsForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.add('active');

    const formData = new FormData(this);
    const text = formData.get('text');

    try {
        const response = await fetch('/tts/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (response.status === 401) {
            alert('请先登录后再使用语音合成功能');
            window.location.href = '/auth/login?next=' + encodeURIComponent('/tts');
            overlay.classList.remove('active');
            return;
        }

        if (!response.ok || data.error) {
            alert('Error: ' + (data.error || 'Unknown error'));
            overlay.classList.remove('active');
            return;
        }

        overlay.classList.remove('active');
        document.getElementById('accessCode').textContent = data.access_code;
        document.getElementById('checkLink').href = '/result?code=' + data.access_code;
        document.getElementById('resultSection').classList.remove('hidden');

    } catch (err) {
        alert('Error: ' + err.message);
        overlay.classList.remove('active');
    }
});

