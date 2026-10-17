"""The monitor as a light source (the second instrument of the bench): a window with a uniform grey patch that the camera looks at.

The control is called `screen_level`: 0 = black ... 255 = white. The window is a separate process (pattern_window.py) that
keeps the window alive and drawn; this module talks to it through two small files (a command file and an answer file), and
`set()` returns only after the window has confirmed that the new level is on the screen.

Backends: "window" (the real thing, default) and "fake" (tests: the level is written to a file that the synthetic camera
reads). HW3_SCREEN_BACKEND chooses; with the synthetic camera the fake is used.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
DEFAULT_LEVEL = 128
LEVEL_RANGE = (0, 255)


class ScreenError(RuntimeError):
    pass


def _write(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class WindowScreen:
    """A real window on the monitor, in its own process."""
    kind = "window"

    def __init__(self, workdir, geom: Optional[str] = None, initial: int = DEFAULT_LEVEL, timeout: float = 15.0):
        self.dir = Path(workdir) / "screen"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state, self.ack, self.log = self.dir / "state.json", self.dir / "ack.json", self.dir / "viewer.log"
        for p in (self.state, self.ack):
            p.unlink(missing_ok=True)
        self.seq, self.level, self.timeout = 0, None, timeout
        cmd = [sys.executable, str(HERE / "pattern_window.py"), "--state", str(self.state), "--ack", str(self.ack)]
        g = geom or os.environ.get("HW3_SCREEN_GEOM")
        if g:
            cmd += ["--geom", g]
        self._log = self.log.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=self._log, env=dict(os.environ))
        self._wait(lambda a: a.get("ready"), "the pattern window did not start")
        if not self.set(initial):
            raise ScreenError("the pattern window did not show the first level")

    def _wait(self, ok, what: str):
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            a = _read(self.ack)
            if a and ok(a):
                return a
            if self.proc.poll() is not None:
                self._log.flush()
                tail = self.log.read_text(encoding="utf-8", errors="replace")[-600:].strip()
                raise ScreenError(f"{what}: the window process ended (exit code {self.proc.returncode}). {tail}")
            time.sleep(0.02)
        raise ScreenError(f"{what}: no answer from the window within {self.timeout:g} s")

    def set(self, level: float) -> bool:
        level = int(round(level))
        self.seq += 1
        _write(self.state, dict(seq=self.seq, level=level))
        a = self._wait(lambda a: a.get("seq") == self.seq or a.get("closed"), f"showing level {level}")
        if a.get("closed"):
            raise ScreenError("the test patch window was closed (do not close it during a run)")
        self.level = a["level"]
        return a["level"] == level

    def get(self) -> Optional[float]:
        return None if self.level is None else float(self.level)

    def close(self) -> None:
        try:
            _write(self.state, dict(quit=True))
            self.proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            self.proc.terminate()
        finally:
            try:
                self._log.close()
            except Exception:  # noqa: BLE001
                pass


class FakeScreen:
    """Tests: no window; the level is written to a file that camera_connector.FakeBackend draws into its picture."""
    kind = "fake"

    def __init__(self, workdir, geom=None, initial: int = DEFAULT_LEVEL):
        path = os.environ.get("HW3_FAKE_SCREEN")
        if not path:
            d = Path(workdir) / "screen"
            d.mkdir(parents=True, exist_ok=True)
            path = str(d / "fake_screen.json")
            os.environ["HW3_FAKE_SCREEN"] = path
        self.path = Path(path)
        self.level = None
        self.set(initial)

    def set(self, level: float) -> bool:
        self.level = int(round(level))
        _write(self.path, dict(level=self.level))
        return True

    def get(self) -> Optional[float]:
        return None if self.level is None else float(self.level)

    def close(self) -> None:
        self.path.unlink(missing_ok=True)


def open_screen(workdir, initial: int = DEFAULT_LEVEL):
    kind = os.environ.get("HW3_SCREEN_BACKEND") or ("fake" if os.environ.get("HW3_CAMERA_BACKEND") == "fake" else "window")
    if kind == "fake":
        return FakeScreen(workdir, initial=initial)
    return WindowScreen(workdir, initial=initial)
