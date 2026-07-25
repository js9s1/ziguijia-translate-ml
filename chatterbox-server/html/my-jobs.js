/* ── my-jobs page logic ─────────────────────────────── */

(function () {
  const container = document.getElementById('jobsContainer');
  if (!container) return;

  fetch('/api/my-jobs')
    .then(function (r) {
      if (r.status === 401) {
        window.location.href = '/auth/login';
        return;
      }
      return r.json();
    })
    .then(function (data) {
      if (!data || !data.jobs || data.jobs.length === 0) {
        container.innerHTML =
          '<div class="empty-state"><h3>暂无任务</h3><p><a href="/">去创建新任务</a></p></div>';
        return;
      }

      var labels = {
        pending: '等待中',
        processing: '处理中',
        completed: '已完成',
        failed: '已失败',
        cancelled: '已取消',
        deleted: '已删除',
      };

      var html =
        '<table class="jobs-table"><thead><tr><th>类型</th><th>访问码</th><th>状态</th><th>更新时间(UTC)</th><th>操作</th></tr></thead><tbody>';

      data.jobs.forEach(function (job) {
        var statusLabel = labels[job.status] || job.status;
        var actions = '';
        actions +=
          '<a href="/result?code=' +
          job.access_code +
          '" class="btn-view">查看</a> ';

        if (job.status === 'failed' || job.status === 'completed' || job.status === 'cancelled' || job.status === 'deleted') {
          if (job.status === 'failed' || job.status === 'cancelled') {
            actions +=
              '<button class="btn-resubmit" data-code="' +
              job.access_code +
              '">重新提交</button> ';
          }
          actions +=
            '<button class="btn-delete" data-code="' +
            job.access_code +
            '">删除</button>';
        } else {
          actions +=
            '<button class="btn-cancel" data-code="' +
            job.access_code +
            '">取消</button>';
        }

        html +=
          '<tr><td>' +
          escapeHtml(job.type) +
          '</td><td><code>' +
          escapeHtml(job.access_code) +
          '</code></td><td><span class="status-badge status-' +
          job.status +
          '">' +
          statusLabel +
          '</span></td><td style="font-size:0.9em;color:#666;">' +
          (job.status_changed_at || job.created_at || '') +
          ' <span style="color:#999;">(' + Intl.DateTimeFormat().resolvedOptions().timeZone + ')</span>' +
          '</td><td style="white-space:nowrap;">' +
          actions +
          '</td></tr>';
      });

      html += '</tbody></table>';
      container.innerHTML = html;

      // Cancel buttons
      container.querySelectorAll('.btn-cancel').forEach(function (btn) {
        btn.addEventListener('click', function () {
          var code = this.getAttribute('data-code');
          if (!confirm('确定要取消任务 ' + code + ' 吗？')) return;
          var self = this;
          self.disabled = true;
          self.textContent = '取消中...';
          fetchWithCsrf('/api/jobs/' + code + '/cancel', { method: 'POST' })
            .then(function (r) {
              return r.json();
            })
            .then(function (result) {
              if (result.success) {
                location.reload();
              } else {
                alert('取消失败: ' + (result.error || '未知错误'));
                self.disabled = false;
                self.textContent = '取消';
              }
            })
            .catch(function (err) {
              alert('取消失败: ' + err.message);
              self.disabled = false;
              self.textContent = '取消';
            });
        });
      });

      // Resubmit buttons
      container.querySelectorAll('.btn-resubmit').forEach(function (btn) {
        btn.addEventListener('click', function () {
          var code = this.getAttribute('data-code');
          if (!confirm('确定要重新提交任务 ' + code + ' 吗？')) return;
          var self = this;
          self.disabled = true;
          self.textContent = '提交中...';
          fetchWithCsrf('/api/jobs/' + code + '/resubmit', { method: 'POST' })
            .then(function (r) { return r.json(); })
            .then(function (result) {
              if (result.success) {
                location.reload();
              } else {
                alert('重新提交失败: ' + (result.error || '未知错误'));
                self.disabled = false;
                self.textContent = '重新提交';
              }
            })
            .catch(function (err) {
              alert('重新提交失败: ' + err.message);
              self.disabled = false;
              self.textContent = '重新提交';
            });
        });
      });

      // Delete buttons
      container.querySelectorAll('.btn-delete').forEach(function (btn) {
        btn.addEventListener('click', function () {
          var code = this.getAttribute('data-code');
          if (!confirm('确定要删除任务 ' + code + ' 吗？')) return;
          var self = this;
          self.disabled = true;
          self.textContent = '删除中...';
          fetchWithCsrf('/api/jobs/' + code + '/delete', { method: 'POST' })
            .then(function (r) {
              return r.json();
            })
            .then(function (result) {
              if (result.success) {
                location.reload();
              } else {
                alert('删除失败: ' + (result.error || '未知错误'));
                self.disabled = false;
                self.textContent = '删除';
              }
            })
            .catch(function (err) {
              alert('删除失败: ' + err.message);
              self.disabled = false;
              self.textContent = '删除';
            });
        });
      });
    })
    .catch(function (err) {
      container.innerHTML =
        '<div class="empty-state"><h3>加载失败</h3><p>' +
        err.message +
        '</p></div>';
    });
})();
