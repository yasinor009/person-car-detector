"""Cross-platform best-effort audible alarm with cooldown and clean shutdown."""
from __future__ import annotations
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
import config


class AlarmSystem:
    def __init__(self, enabled=None, cooldown_seconds=None, duration_seconds=None, sound_file=None):
        self.enabled = config.ALARM_ENABLED if enabled is None else bool(enabled)
        self.cooldown_seconds = float(config.ALARM_COOLDOWN_SECONDS if cooldown_seconds is None else cooldown_seconds)
        self.duration_seconds = float(config.ALARM_DURATION_SECONDS if duration_seconds is None else duration_seconds)
        self.sound_file = str(sound_file or config.ALARM_SOUND_FILE)
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = None
        self._last_trigger = 0.0
        self._audio_backend = self._detect_backend()

    def _detect_backend(self):
        if sys.platform == "win32":
            return "winsound"
        for command in ("aplay", "paplay", "ffplay"):
            if shutil.which(command):
                return command
        return "terminal_bell"

    def trigger_alarm(self, info=None):
        with self._lock:
            now = time.monotonic()
            if not self.enabled or (now - self._last_trigger) < self.cooldown_seconds:
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            self._last_trigger = now
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._play, name="drone-alarm", daemon=True)
            self._thread.start()
            self.logger.warning("Drone detection alarm triggered: %s", info or {})
            return True

    def _play(self):
        deadline = time.monotonic() + max(0.1, self.duration_seconds)
        try:
            if self._audio_backend == "winsound":
                import winsound
                while not self._stop_event.is_set() and time.monotonic() < deadline:
                    if Path(self.sound_file).is_file():
                        winsound.PlaySound(self.sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
                    else:
                        winsound.Beep(1000, 250)
                    self._stop_event.wait(0.25)
            elif self._audio_backend in ("aplay", "paplay", "ffplay") and Path(self.sound_file).is_file():
                command = [self._audio_backend, self.sound_file]
                if self._audio_backend == "ffplay":
                    command = [self._audio_backend, "-nodisp", "-autoexit", "-loglevel", "error", self.sound_file]
                process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                while not self._stop_event.is_set() and time.monotonic() < deadline and process.poll() is None:
                    self._stop_event.wait(0.1)
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
            else:
                while not self._stop_event.is_set() and time.monotonic() < deadline:
                    print("\a", end="", flush=True)
                    self._stop_event.wait(0.5)
        except Exception:
            self.logger.exception("Alarm playback failed")
        finally:
            self._stop_event.set()

    def stop_alarm(self):
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.5)

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        if not self.enabled:
            self.stop_alarm()
        return self.enabled

    def test_alarm(self):
        with self._lock:
            self._last_trigger = 0.0
        return self.trigger_alarm({"test": True})

    def get_status(self):
        remaining = max(0.0, self.cooldown_seconds - (time.monotonic() - self._last_trigger)) if self._last_trigger else 0.0
        return {
            "enabled": self.enabled,
            "playing": bool(self._thread and self._thread.is_alive() and not self._stop_event.is_set()),
            "backend": self._audio_backend,
            "cooldown_remaining_s": round(remaining, 1),
        }
