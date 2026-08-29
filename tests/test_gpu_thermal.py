"""Tests for gpu_thermal.HangGuard — shared daemon hang self-kill."""

import signal
import threading
import time

from gpu_thermal import HangGuard


def _guard(hang=100.0, grace=10.0, check=0.01):
    return HangGuard(hang_timeout_secs=hang, grace_secs=grace, check_secs=check)


def test_arm_then_disarm_not_stuck():
    g = _guard()
    assert g.active_jobs == 0
    g.arm()
    assert g.active_jobs == 1
    assert g.stuck() == []
    g.disarm()
    assert g.active_jobs == 0
    assert g.stuck() == []


def test_stuck_after_timeout():
    g = _guard(hang=0.05)
    g.arm()
    time.sleep(0.2)
    stuck = g.stuck()
    assert len(stuck) == 1
    tid, elapsed = stuck[0]
    assert elapsed >= 0.1
    g.disarm()
    assert g.stuck() == []


def test_watch_sigterm_then_sigkill_after_grace(monkeypatch):
    g = HangGuard(hang_timeout_secs=0.05, grace_secs=0.1, check_secs=0.01)

    quit_called = threading.Event()
    kills = []

    monkeypatch.setattr("os.kill", lambda _pid, sig: kills.append(sig))

    def request_quit():
        quit_called.set()

    t = threading.Thread(
        target=g.watch,
        kwargs={"request_quit": request_quit, "is_quit": quit_called.is_set},
        daemon=True,
    )
    g.arm()
    t.start()
    try:
        assert quit_called.wait(5)
        deadline = time.time() + 5
        while len(kills) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert kills[:2] == [signal.SIGTERM, signal.SIGKILL]
        t.join(timeout=5)
        assert not t.is_alive()
    finally:
        g.disarm()


def test_watch_exits_early_when_daemon_quitting():
    g = _guard()
    t = threading.Thread(
        target=g.watch,
        kwargs={"is_quit": lambda: True},
        daemon=True,
    )
    t.start()
    t.join(timeout=5)
    assert not t.is_alive()
