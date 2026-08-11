# Python environment matrix

This repo runs across **multiple Python interpreters**. The version declared in
`pyproject.toml` (`requires-python = ">=3.13"`) applies only to the server and
the dev toolchain (tests / lint / type-check); the ML/GPU subprocesses use a
separate, pyenv-managed Python that sits **below** that floor on purpose.

| Role | Python | Interpreter | Configured via |
|---|---|---|---|
| Flask server (gunicorn) | 3.14 | `/usr/bin/python3` (`start_server.sh` launches `~/.local/bin/gunicorn`, shebang `#!/usr/bin/python3`) | — |
| Tests + lint + type-check | 3.14 | `python` / `/usr/bin/python3` | `pyproject.toml` `requires-python = ">=3.13"` |
| TTS / `gen_audio.py` subprocess | 3.11 | `~/.pyenv/versions/3.11.14/bin/python3.11` | `GEN_AUDIO_PYTHON` in `chatterbox-server/config.py` |
| HY-MT translation subprocess | 3.11 | `~/.pyenv/versions/3.11.14/bin/python3.11` | `TRANSLATE_PYTHON` in `chatterbox-server/config.py` |
| `gen_video.py` / download / misc | 3.14 | `/usr/bin/python3` | `PYTHON_BIN` in `chatterbox-server/config.py` |

## Notes

- The **server** and the **dev toolchain** run on Python 3.14 and honor the
  `>=3.13` floor in `pyproject.toml`.
- The **GPU/ML subprocesses** (`gen_audio`, HY-MT translation) run on the
  separate pyenv **Python 3.11.14** install so they can hold ROCm/PyTorch
  handles without blocking the web worker. They are invoked by absolute path
  through `GEN_AUDIO_PYTHON` / `TRANSLATE_PYTHON`, so they are exempt from the
  `requires-python` constraint.
- Every interpreter path is overridable via the matching environment variable
  (e.g. `GEN_AUDIO_PYTHON`, `TRANSLATE_PYTHON`, `PYTHON_BIN`).
- Keep in-flight comments in code consistent with this matrix — gen_audio runs
  on Python **3.11**, not 3.13.
