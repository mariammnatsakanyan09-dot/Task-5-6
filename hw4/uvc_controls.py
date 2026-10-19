"""UVC controls on macOS through the command-line tool uvc-util (https://github.com/jtfrey/uvc-util).

Why: OpenCV cannot read or change the UVC controls on macOS (every property answers -1). uvc-util sends the USB video class
control requests itself. The Pano-360 implements exactly four of them: brightness, contrast, saturation and hue.

    ./uvc-util -V 0x1bcf:0x2c99 --list-controls          which controls the camera has
    ./uvc-util -V 0x1bcf:0x2c99 -S brightness            minimum / maximum / step-size / default-value
    ./uvc-util -V 0x1bcf:0x2c99 -o brightness            the current value, ONLY the number
    ./uvc-util -V 0x1bcf:0x2c99 -s brightness=10         set a value

Where the program looks for the binary: HW3_UVC_UTIL (path to the file), then PATH, then the usual folders. If it is not
found, the camera simply has no controls on this computer (resolution and readings still work).
"""
from __future__ import annotations

import functools
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

VENDOR_ID, PRODUCT_ID = 0x1BCF, 0x2C99
UVC_NAMES = ("brightness", "contrast", "saturation", "hue", "gain")       # control names that uvc-util and this project share
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
HERE = Path(__file__).resolve().parent


def first_number(text: str) -> Optional[float]:
    m = _NUM.search(text or "")
    return float(m.group(0)) if m else None


def parse_show_control(text: str) -> dict:
    """minimum / maximum / step-size / default-value from the answer of `-S name`. Accepts `minimum: 0`, `minimum: {x=0}`
    and a value with units; the first number after the colon is taken."""
    out: dict = {}
    for key, label in (("min", "minimum"), ("max", "maximum"), ("step", "step-size"), ("default", "default-value")):
        m = re.search(rf"{label}\s*[:=]\s*([^\n]*)", text or "", flags=re.I)
        if m:
            v = first_number(m.group(1))
            if v is not None:
                out[key] = v
    return out


def _candidates() -> list[Path]:
    home = Path.home()
    names = []
    for base in (Path.cwd(), HERE, HERE.parent, HERE.parent.parent, home, home / "Desktop", home / "Documents",
                 home / "Downloads", home / "Desktop" / "Task5-6", home / "Desktop" / "Task5-6" / "hw3"):
        names += [base / "uvc-util", base / "uvc-util" / "uvc-util", base / "uvc-util" / "src" / "uvc-util"]
    return names


def find_exe() -> Optional[str]:
    env = os.environ.get("HW3_UVC_UTIL")
    if env:
        p = Path(env).expanduser()
        p = p / "uvc-util" if p.is_dir() else p
        return str(p) if p.is_file() and os.access(p, os.X_OK) else None
    w = shutil.which("uvc-util")
    if w:
        return w
    for p in _candidates():
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


class UvcUtil:
    def __init__(self, exe: str, target: list[str], timeout: float = 15.0):
        self.exe, self.target, self.timeout = exe, list(target), timeout
        self.names: list[str] = []
        self.raw: dict = {}                 # raw answers kept for diagnosis (printed by probe_camera.py)
        self._ranges: dict = {}

    # ---- one call
    def run(self, *args: str) -> tuple[int, str, str]:
        try:
            r = subprocess.run([self.exe, *self.target, *args], capture_output=True, text=True, timeout=self.timeout)
            return r.returncode, r.stdout or "", r.stderr or ""
        except (OSError, subprocess.TimeoutExpired) as e:
            return 127, "", f"{type(e).__name__}: {e}"

    # ---- what the camera has
    def refresh(self) -> list[str]:
        rc, out, err = self.run("--list-controls")
        self.raw["list-controls"] = (out + err).strip()
        self.names = [n for n in UVC_NAMES if rc == 0 and re.search(rf"(?<![\w-]){n}(?![\w-])", out + err)]
        return self.names

    def control_range(self, name: str) -> Optional[dict]:
        if name in self._ranges:
            return self._ranges[name]
        rc, out, err = self.run("-S", name)
        self.raw[f"-S {name}"] = (out + err).strip()
        r = parse_show_control(out) if rc == 0 else {}
        self._ranges[name] = r if ("min" in r and "max" in r) else None
        return self._ranges[name]

    def get(self, name: str) -> Optional[float]:
        rc, out, _ = self.run("-o", name)
        return first_number(out) if rc == 0 else None

    def set(self, name: str, value: float) -> bool:
        v = int(value) if float(value).is_integer() else round(float(value), 4)
        rc, _, _ = self.run("-s", f"{name}={v}")
        return rc == 0

    # ---- finding the camera
    @classmethod
    def detect(cls) -> Optional["UvcUtil"]:
        exe = find_exe()
        if not exe:
            return None
        env_target = os.environ.get("HW3_UVC_TARGET")
        targets = [shlex.split(env_target)] if env_target else [
            ["-V", f"0x{VENDOR_ID:04x}:0x{PRODUCT_ID:04x}"], ["-V", f"{VENDOR_ID}:{PRODUCT_ID}"], ["-I", "0"]]
        for t in targets:
            u = cls(exe, t)
            if u.refresh():
                return u
        return None


@functools.lru_cache(maxsize=1)
def _shared(key: str) -> Optional[UvcUtil]:
    return UvcUtil.detect()


def shared() -> Optional[UvcUtil]:
    """One detection per process (the answer depends only on the environment)."""
    return _shared(os.environ.get("HW3_UVC_UTIL", "") + "|" + os.environ.get("HW3_UVC_TARGET", ""))


def wanted() -> bool:
    """uvc-util is tried on macOS, and anywhere HW3_UVC_UTIL names a binary (tests use a fake one). HW3_UVC=off disables."""
    if os.environ.get("HW3_UVC", "").lower() == "off":
        return False
    import sys
    return sys.platform == "darwin" or bool(os.environ.get("HW3_UVC_UTIL"))
