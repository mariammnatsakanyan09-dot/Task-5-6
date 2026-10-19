"""Everything that depends on the operating system, in one place: which OpenCV camera backend to use, how to list the
cameras by name, how to open a picture file for the person to look at.

  Windows : DirectShow (cv2.CAP_DSHOW). It lets OpenCV read and change the UVC controls of most USB cameras
            (brightness, exposure, gain ...). Media Foundation (CAP_MSMF) is the alternative: HW3_CV_API=msmf.
  macOS   : AVFoundation. OpenCV cannot read or change the UVC controls there (it answers -1).
  Linux   : V4L2.
"""
from __future__ import annotations

import os
import subprocess
import sys

import cv2
import numpy as np

def rotate_log(path, max_bytes: int) -> bool:
    """A log file that grows for ever fills the disk of a guard that runs for weeks. When the file is larger than max_bytes it becomes
    <name>.1 (the older .1 is dropped) and a new one starts: at most about 2 x max_bytes stay on the disk. True if it rotated."""
    from pathlib import Path
    p = Path(path)
    try:
        if max_bytes > 0 and p.exists() and p.stat().st_size > max_bytes:
            os.replace(p, p.with_name(p.name + ".1"))
            return True
    except OSError:
        pass
    return False


def prune_old(folder, days: float, pattern: str = "*.png", now: float | None = None) -> int:
    """Delete the files of `pattern` in `folder` that are older than `days` days (pictures of people are not kept for ever). days <= 0: keep all.
    Returns how many were deleted."""
    import time
    from pathlib import Path
    d = Path(folder)
    if days <= 0 or not d.is_dir():
        return 0
    limit = (time.time() if now is None else now) - days * 86400.0
    n = 0
    for f in d.glob(pattern):
        try:
            if f.is_file() and f.stat().st_mtime < limit:
                f.unlink()
                n += 1
        except OSError:
            pass
    return n


IS_WIN = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"

_API = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "avfoundation": cv2.CAP_AVFOUNDATION,
        "v4l2": cv2.CAP_V4L2, "any": cv2.CAP_ANY}


def capture_api(preferred: str | None = None) -> int:
    p = (preferred or os.environ.get("HW3_CV_API") or "").lower()
    if p in _API:
        return _API[p]
    if IS_WIN:
        return cv2.CAP_DSHOW
    if IS_MAC:
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def api_label(api: int | None = None) -> str:
    api = capture_api() if api is None else api
    for k, v in _API.items():
        if v == api:
            return {"dshow": "Windows DirectShow", "msmf": "Windows Media Foundation", "avfoundation": "macOS AVFoundation",
                    "v4l2": "Linux V4L2", "any": "OpenCV default backend"}[k]
    return f"OpenCV backend {api}"


def want_fourcc() -> str | None:
    """Pixel format to ask for when the camera is opened. MJPG gives full-size pictures at a normal frame rate on
    Windows and Linux; macOS chooses itself. HW3_FOURCC=none switches this off."""
    v = os.environ.get("HW3_FOURCC")
    if v:
        return None if v.lower() == "none" else v[:4]
    return None if IS_MAC else "MJPG"


def open_capture(index: int, api: int | None = None, size: tuple[int, int] | None = (1920, 1080)):
    """Open a camera and ask for the format and size (the Pano-360 delivers its two-part picture at 1920 x 1080; some
    Windows drivers start at 640 x 480 until they are asked)."""
    cap = cv2.VideoCapture(index, capture_api() if api is None else api)
    if cap.isOpened():
        fc = want_fourcc()
        if fc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fc))
        if size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    return cap


def camera_names() -> str:
    try:
        if IS_MAC:
            r = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True, timeout=20)
        elif IS_WIN:
            ps = ("Get-PnpDevice -Class Camera,Image -PresentOnly | Select-Object Status,FriendlyName,InstanceId | "
                  "Format-Table -AutoSize | Out-String -Width 220")
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30)
        else:
            r = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr).strip() or "(no camera names reported)"
    except Exception as e:  # noqa: BLE001
        return f"(could not list names: {e})"


def permission_hint() -> str:
    if IS_WIN:
        return ("Windows Settings > Privacy & security > Camera: 'Camera access' and 'Let desktop apps access your camera' "
                "must be On; close the Camera app and any program that shows the camera (Teams, Zoom, OBS, browser tabs)")
    if IS_MAC:
        return "System Settings > Privacy & Security > Camera: allow Terminal (or VS Code); close QuickTime / FaceTime / Zoom"
    return "check the permissions of /dev/video* and that no other program uses the camera"


def show_file(path) -> None:
    """Open a picture file in the default viewer (for the video)."""
    try:
        if IS_WIN:
            os.startfile(str(path))                     # type: ignore[attr-defined]
        elif IS_MAC:
            subprocess.Popen(["open", str(path)])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(path)], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    except OSError:
        pass


def utf8_console() -> None:
    """Windows consoles often use a legacy code page; model replies may contain characters it cannot print."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")      # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass


def imwrite(path, img) -> bool:
    """cv2.imwrite fails silently on Windows when the path has non-ASCII characters (a user name such as 'Мариам'):
    encode in memory and write the bytes with numpy instead."""
    from pathlib import Path
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def imread(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags) if data.size else None
