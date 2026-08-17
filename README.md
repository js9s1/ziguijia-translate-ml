# 子归家视频翻译系统 (Ziguijia Video Translation)

Web-based pipeline that translates 杨宁's dharma talk videos (ziguijia.com) into other
languages, re-speaks them with a voice-clone TTS (Chatterbox, trained on the speaker's
voice), retimes the video track to stay lip/gesture-synced with the new audio, and burns
in translated subtitles.

Runs on an AMD Strix Halo APU (gfx1151) with ROCm for ML inference and VAAPI for video
encoding.

## Architecture

```
┌───────────────────────────── gunicorn (1 worker × 8 threads) ──────────────────────────┐
│ Flask app (chatterbox-server/)                                                         │
│   routes/   auth · files · jobs · srt · tts · video · api                              │
│   JobQueue (jobqueue.py)  ── 1 worker thread, strictly sequential                      │
│        │  SQLite jobs.db (state) · Valkey (shared pending queue, BLPOP)                │
│        ▼                                                                               │
│   job handlers → pipeline.py steps (checkpoint-aware, subprocess-based)                │
└──────┬──────────────────────┬───────────────────────┬──────────────────────────────────┘
       │ spawns subprocesses  │                       │
       ▼                      ▼                       ▼
 gen_audio.py            gen_video.py           translate_srt.py          rapid_videocr
 (Python 3.11, ROCm)     (system Python,       (Python 3.11, ROCm)        daemon (ROCm)
       │                   VAAPI ffmpeg)               │                        ▲
       ▼                                                ▼                        │
 TTS daemon (unix sock)                        translate daemon               │
 gen_audio_daemon.py     ── max 2 concurrent jobs, idle-exit when GPU hot ──        │
```

### Long-lived processes

| Process | Purpose | Lifecycle |
|---|---|---|
| `gunicorn` | Web UI + job queue | `start_server.sh` / `stop_server.sh` |
| `valkey-server` | Shared job queue + pub/sub | started/stopped by the scripts |
| TTS daemon (`gen_audio_daemon.py`) | Chatterbox voice-clone TTS | detached; auto-started by `gen_audio.py` clients or prewarmed when the worker picks up a job; auto-exits when the GPU is hot while idle, after a short post-job grace (`GEN_AUDIO_IDLE_TEMPERATURE` / `GEN_AUDIO_IDLE_GRACE_SECS`) |
| translate daemon (`translate_daemon.py`) | HY-MT machine translation | detached; auto-started by `translate_srt.py` clients or prewarmed when the worker picks up a job; auto-exits when the GPU is hot while idle (`TRANSLATE_IDLE_TEMPERATURE`) |
| OCR daemon (`rapid_videocr_daemon.py`) | RapidOCR (burned-in zh subtitles) | started/stopped by the scripts; also shared with the batch pipeline |
| `jobs_tui.py` | Terminal job monitor | manual |

**Server restarts do not restart the TTS/translate daemons** — they are detached
(`start_new_session=True`). They keep running (or idle-exit) across gunicorn restarts;
code changes to them take effect the next time they (re)start. Restarting the server only
reloads the Flask app, job queue, and job handlers.

## Job queue

- One worker thread processes **one job at a time** (heavy steps run as subprocesses).
- Jobs are durable: SQLite `jobs.db` + Valkey list `jobqueue:pending`. On boot, jobs
  left in `processing` are reset to `pending` and orphan subprocesses are killed.
- **No heartbeat watchdog** — the worker thread is restarted only if it actually died.
  (A previous heartbeat-based watchdog mis-detected busy workers during long TTS steps
  and spawned a second consumer thread, running two jobs concurrently.)
- Jobs and resubmits can be enqueued externally: set `status='pending'` in `jobs.db`
  and `RPUSH` the access code onto `jobqueue:pending`.

## Job types

| Handler | UI label | Flow |
|---|---|---|
| `_run_gen_audio` | 音频生成 | SRT → TTS audio only |
| `_run_audio_segmentation_job` | 音频分段合成 | segmented audio synthesis |
| `_run_tts_job` | 语音合成 | TTS from text |
| `_run_video_job` | 宁视频翻译 | download video by number → user SRT → audio → video |
| `_run_video_custom_job` | 自定义视频 | upload video + SRT → audio → video |
| `_run_video_auto_job` | 自动翻译视频 | upload video → whisper → translate → audio → video |
| `_run_video_ocr_job` | OCR翻译视频 | upload video → OCR → translate → audio → video |
| `_run_video_ning_ocr_job` | 宁视频OCR翻译 | download → OCR → translate → audio → video |
| `_run_video_ning_ocr_translate_only_job` | 宁视频OCR仅翻译 | download → OCR → translate (stops before audio/video) |
| `_run_video_ning_auto_job` | 宁视频语音识别翻译 | download → whisper → translate → audio → video |
| `_run_ocr_only_job` | 视频OCR提取字幕 | OCR only |

## Pipeline (custom video example)

Checkpoint order: `download, decompress, trim, extract_audio, whisper, ocr, translate, audio, video`.
Completed steps are skipped on resubmit, so failed/resubmitted jobs resume instead of
redoing work.

1. **gen_audio** (`gen_audio.py`, subprocess on Python 3.11 + ROCm)
   - Splits the SRT into segments; per segment, asks the TTS daemon to generate speech
     in the target language using the cloned speaker voice.
   - Writes `output.wav`, `output_adjusted.srt` (segment timings stretched so each
     cue's audio fits), `changed_segments.json` (which cues changed duration).
   - Per-segment WAV cache (`tmp/combined_segment_N.wav` + `cache_meta.json`) makes
     resubmits fast: only changed/silent segments regenerate.
2. **gen_video** (`gen_video.py`, system Python + VAAPI)
   - Rebuilds the video track segment-by-segment (`-ss/-to` + `setpts` stretch per
     subtitle cue, batched into 42-segment ffmpeg filter graphs), placing each cue's
     video exactly at its adjusted start time.
   - Corrects accumulated frame-boundary drift with one setpts pass keyed to the
     designed timeline, trims trailing content to the audio length, then muxes audio
     (AAC) + subtitles (mov_text, `language=eng`) → `output_final.mp4`.
   - Intermediates: `output_modified.mp4` (retimed), `output_corrected.mp4` (drift-corrected).
3. **Zh audio adjustment** (non-fatal) — stretches the original Chinese audio track
   with per-segment `atempo` to match the adjusted timing → `orig_zh_adjusted.wav`
   (a whole-file atempo pass corrects accumulated drift).

## TTS (gen_audio) key behaviours

- **Silence markers** — `<1.5>` inline in cue text inserts intentional silence.
- **Non-verbal cues** — only text with *no letters at all* (pure symbols) is replaced
  by silence; any letter-based text is spoken, however short (short Japanese/English
  sentences were previously misclassified).
- **Silent-output detection** — generated audio whose voiced-sample fraction is
  below threshold is treated as a failure and retried up to 3× with temperature
  jitter; the final fallback silence is flagged and never reused from cache.
- **Daemon crash recovery** — if the TTS daemon dies mid-request (e.g. a ROCm GPU
  fault from an out-of-range flow token), the client restarts it, reloads the model,
  and retries the chunk instead of failing the whole job.
- **Model-load coordination** — daemon serializes model loads against in-flight
  generations (concurrent load+generate produced permanently NaN audio on ROCm).
- **Timeouts** (env-configurable, see `.env.example`):
  - `GEN_AUDIO_STALL_TIMEOUT` — no new log output while alive → kill (default 900 s).
  - `GEN_AUDIO_SEGMENT_BUDGET` × segment count (floor `GEN_AUDIO_MIN_TOTAL_TIMEOUT`)
    — absolute cap that scales with job size.
  - On timeout: SIGTERM first (gen_audio stops after the current segment and keeps
    its cache), then SIGKILL after a grace period.

## Configuration

- `.env` (gitignored, sourced by `start_server.sh`) — secrets and tunables:
  `GEN_AUDIO_MAX_JOBS`, `TRANSLATE_MAX_JOBS`, `GEN_AUDIO_IDLE_TEMPERATURE`,
  `TRANSLATE_IDLE_TEMPERATURE`, the three
  `GEN_AUDIO_*_TIMEOUT` values, Valkey password, SMTP credentials.
- `chatterbox-server/config.py` — paths, interpreters, sockets; everything overridable
  via environment variables.
- Python interpreters follow a strict matrix (see [PYTHON_ENV.md](PYTHON_ENV.md)):
  server/tests/`gen_video.py` on system Python 3.14; TTS + translation subprocesses on
  pyenv Python 3.11.14 (ROCm).

## Operations

```sh
./start_server.sh     # gunicorn + valkey + OCR daemon
./stop_server.sh      # gunicorn + orphan gen_audio clients + valkey + OCR daemon
python jobs_tui.py --watch        # monitor the queue
```

- Job output lives in `../video/<ACCESS_CODE>/` (and `../audio_tracks/<ACCESS_CODE>/`
  for audio jobs); browse/download via the web UI file manager.
- Resubmit a failed/completed job through the UI, or manually:
  `UPDATE jobs SET status='pending' WHERE access_code=...; RPUSH jobqueue:pending <code>`.
- Logs: `~/logs/chatterbox-server/{access,error}.log`, per-job `job.log` (also
  `audio_tracks/job.log` for the gen_audio subprocess), `~/logs/*_daemon.log`.

## Testing

```sh
python3 -m pytest tests/ -q -o addopts=""
```

268 tests cover the queue, auth, pipeline, files routes, schemas, and utilities.
