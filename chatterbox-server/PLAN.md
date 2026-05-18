# User Management Implementation Plan

## 1. Database — `jobqueue.py`
- Add `users` table (`id`, `wechat_openid`, `nickname`, `avatar_url`, `created_at`)
- Add `user_id` column to `jobs` table
- New methods: `get_user_jobs(user_id)`, `get_or_create_user()`, `get_user_by_id()`

## 2. WeChat Auth Module — `wechat_auth.py` (new)
- Config via env vars: `WECHAT_APP_ID`, `WECHAT_APP_SECRET`, `WECHAT_REDIRECT_URI`
- `get_oauth_url()` — generate WeChat OAuth authorization URL
- `exchange_code_for_userinfo(code)` — exchange code for OpenID + nickname + avatar

## 3. Server Routes — `chatterbox_server.py`
- Flask session config with `secret_key`
- `GET /auth/wechat/login` — redirect to WeChat OAuth
- `GET /auth/wechat/callback` — handle OAuth callback, create/find user, set session
- `GET /auth/me` — return current user info or `{logged_in: false}`
- `POST /auth/logout` — clear session
- `GET /my-jobs` — serve `my-jobs.html`
- `GET /api/my-jobs` — return JSON of user's jobs
- Modify all 5 job submission endpoints to check `session['user_id']`, reject with 401 if not logged in, pass `user_id` to process functions

## 4. Job Processing — `srt_action.py`
- Add `user_id` parameter to all 5 `process_*` functions
- Pass `user_id` into `job_data`

## 5. Frontend — `auth.js` (new)
- On page load, fetch `/auth/me`
- If logged in: show nickname + avatar, "我的任务" link, logout button
- If not: show "微信登录" button

## 6. Frontend — `my-jobs.html` (new)
- Table of user's jobs: type, access_code, status badge, created_at, action
- Completed jobs link to `/result?code=<code>`
- Empty state when no jobs

## 7. Frontend — All 8 HTML files
- Update `.page-header` to flex layout with `header-left` / `header-right`
- Add login button (`#userArea`) in header-right
- Include `<script src="/auth.js"></script>`

## 8. CSS — `ning.css`
- `.header-left`, `.header-right` — flex layout
- `.btn-login` — WeChat green button
- `.btn-myjobs` — blue button
- `.btn-logout` — subtle outline button
- `.user-info`, `.user-avatar`, `.user-name`
