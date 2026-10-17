"""(a) THE CONNECTOR - the code that sends a command to the real instrument and reads a measurement back.

Instrument : the camera called "Pano-360" (case label THAT DRAGONFLY): a USB video class (UVC) camera, USB vendor id
             0x1BCF, product id 0x2C99, 360-degree optics. It delivers ONE 1920 x 1080 picture that is made of two parts:
             the main view (rows 0..815) and a panorama strip (rows 816..1079).
Interface  : the standard USB video class driver of the operating system, reached through OpenCV
             (cv2.VideoCapture): DirectShow on Windows, AVFoundation on macOS, V4L2 on Linux (see cam_platform.py).

What can be commanded through this interface depends on the platform and the camera, and is CHECKED, not assumed:
  * the frame size (resolution) - always offered (640x480, 1280x720, 1920x1080), with a check of what the camera delivers;
  * the UVC controls (the Pano-360 implements brightness, contrast, saturation and hue) - only those that probe_camera.py
    verified on this camera (the picture really changes when the value changes); they are listed in controls.json.
    Windows: OpenCV/DirectShow reaches them. macOS: OpenCV cannot, so the command-line tool uvc-util is used (uvc_controls.py).
  * taking pictures and measuring regions of them;

  * screen_level - the second instrument of the bench: a window on the monitor in front of the camera shows a uniform grey patch
    whose level (0 black .. 255 white) is commanded; the camera measures it (screen_patch.py, pattern_window.py).

Only crops of the picture are ever written to disk by this module (the region that the project measures); a whole picture can contain people, so it is kept in memory only.

A backend "fake" (a synthetic picture whose brightness reacts to the commands) is used by the offline tests. It is
selected with the environment variable HW3_CAMERA_BACKEND=fake; it is never used silently.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import cam_platform as plat
import screen_patch as scr
import uvc_controls as uvc

ALLOWED_RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]
SAT_LEVEL = 250          # grey level (0..255) at or above which a pixel counts as saturated
SHARP_WIDTH = 400        # regions are resized to this width before the "normalised" sharpness is computed
SEAM_FRACTION = 0.7556   # fallback position of the seam between main view and panorama (816 of 1080 rows)
PANORAMA_BOTTOM_SKIP = 4 # the last rows of the panorama strip contain an artefact of the camera
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class CameraError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- backends
PROPS = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS, "contrast": cv2.CAP_PROP_CONTRAST, "saturation": cv2.CAP_PROP_SATURATION,
    "hue": cv2.CAP_PROP_HUE, "gain": cv2.CAP_PROP_GAIN, "exposure": cv2.CAP_PROP_EXPOSURE, "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "focus": cv2.CAP_PROP_FOCUS, "autofocus": cv2.CAP_PROP_AUTOFOCUS,
}


class CvBackend:
    """The real camera: OpenCV on top of the operating system's camera driver."""
    kind = "opencv"

    def __init__(self, index: int):
        self.index = index
        self.api = plat.capture_api()
        self.cap = plat.open_capture(index, self.api)
        self.ok = bool(self.cap.isOpened())
        self.uvc = uvc.shared() if (self.ok and uvc.wanted()) else None       # macOS: the controls go through uvc-util

    def read(self):
        return self.cap.read()

    def set_size(self, w: int, h: int) -> None:
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def reopen(self, w: int, h: int) -> None:
        """Some drivers only take a new size when the device is opened again: close it, open it, ask for the size."""
        self.cap.release()
        self.cap = plat.open_capture(self.index, self.api, size=(w, h))
        self.ok = bool(self.cap.isOpened())

    def via(self, name: str) -> str:
        return "uvc-util" if (self.uvc is not None and name in self.uvc.names) else "opencv"

    def get_prop(self, name: str) -> Optional[float]:
        if self.via(name) == "uvc-util":
            return self.uvc.get(name)
        v = self.cap.get(PROPS[name])
        return None if v is None or v != v or v == -1.0 else float(v)      # -1 / NaN: OpenCV says "not supported"

    def set_prop(self, name: str, value: float) -> bool:
        if self.via(name) == "uvc-util":
            return self.uvc.set(name, value)
        return bool(self.cap.set(PROPS[name], float(value)))

    def release(self) -> None:
        self.cap.release()


class FakeBackend:
    """A synthetic stand-in for tests: a two-part picture (main view + panorama strip). Its look is controlled by a
    JSON file (HW3_FAKE_STATE): {"covered": bool, "light": float, "shift": int}; the frame size follows set_size.
    It also has controls that act on the picture like a real driver would and that clamp the values they are given:
    exposure (-10..-2, log2 s), gain (0..255) and auto_exposure (0.25 manual / 0.75 auto), and the four controls of the
    Pano-360: brightness (-64..64), contrast (0..100, 50 = unchanged), saturation (0..100, 50 = normal colour) and hue.
    If HW3_UVC_UTIL names a (fake) uvc-util, brightness / contrast / saturation / hue are held by that tool, in the file
    HW3_FAKE_UVC_STATE, exactly as on a Mac."""
    kind = "fake"
    ok = True
    _cache: dict = {}
    TINT_B, TINT_R = -39.3, 15.0         # colour cast per unit of saturation; 0.114*B + 0.299*R = 0, so the grey level is kept
    PATCH = (400, 830, 800, 200)         # where the (imitated) monitor patch is in the 1920 x 1080 picture: x, y, w, h

    def __init__(self, index: int = 0, state_file: Optional[str] = None):
        self.index = index
        self.state_file = state_file or os.environ.get("HW3_FAKE_STATE")
        self.size = (1920, 1080)
        self.n_reads = 0
        self.props = {"auto_exposure": 0.75, "exposure": -5.0, "gain": 0.0, "brightness": 0.0,
                      "contrast": 50.0, "saturation": 50.0, "hue": 0.0}
        self.limits = {"exposure": (-10.0, -2.0), "gain": (0.0, 255.0), "brightness": (-64.0, 64.0),
                       "contrast": (0.0, 100.0), "saturation": (0.0, 100.0), "hue": (-40.0, 40.0)}
        self.uvc = uvc.shared() if uvc.wanted() else None
        self.uvc_state = os.environ.get("HW3_FAKE_UVC_STATE")

    def via(self, name: str) -> str:
        return "uvc-util" if (self.uvc is not None and name in self.uvc.names) else "opencv"

    def get_prop(self, name: str) -> Optional[float]:
        if self.via(name) == "uvc-util":
            return self.uvc.get(name)
        return self.props.get(name)

    def set_prop(self, name: str, value: float) -> bool:
        if self.via(name) == "uvc-util":
            return self.uvc.set(name, value)
        if name not in self.props:
            return False
        lo, hi = self.limits.get(name, (-1e9, 1e9))
        self.props[name] = float(min(max(value, lo), hi))
        return True

    def _prop(self, name: str) -> float:
        """The value that the picture is drawn with (the uvc-util state file, when there is one)."""
        if self.via(name) == "uvc-util" and self.uvc_state and Path(self.uvc_state).exists():
            try:
                return float(json.loads(Path(self.uvc_state).read_text(encoding="utf-8"))[name])
            except Exception:  # noqa: BLE001
                pass
        return self.props[name]

    def _screen_level(self) -> Optional[float]:
        """The level of the imitated monitor patch (the file written by screen_patch.FakeScreen), None if there is none."""
        p = os.environ.get("HW3_FAKE_SCREEN")
        if p and Path(p).exists():
            try:
                return float(json.loads(Path(p).read_text(encoding="utf-8"))["level"])
            except Exception:  # noqa: BLE001
                return None
        return None

    def _gain_factor(self) -> float:
        if self.props["auto_exposure"] > 0.5:                       # automatic exposure keeps the picture at one level
            return 1.0
        return (2.0 ** (self.props["exposure"] + 5.0)) * (1.0 + self.props["gain"] / 64.0)

    def _state(self) -> dict:
        st = dict(covered=False, light=1.0, shift=0)
        if self.state_file and Path(self.state_file).exists():
            try:
                st.update(json.loads(Path(self.state_file).read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass
        return st

    @classmethod
    def _base(cls, shift: int) -> np.ndarray:
        if shift in cls._cache:
            return cls._cache[shift]
        H, W, seam = 1080, 1920, 816
        yy, xx = np.mgrid[0:seam, 0:W]
        main = 150 + 40 * np.sin(xx / 300.0) - 0.05 * yy
        x0 = 150 + int(shift)
        main[500:800, x0:x0 + 800] = 60 + 25 * ((np.arange(800) // 20) % 2)[None, :]
        main[300:310, :] = 230
        img = np.zeros((H, W), np.float32)
        img[:seam] = main
        pano = cv2.resize(main.astype(np.float32), (W, H - seam), interpolation=cv2.INTER_AREA) + 50
        img[seam:] = pano
        cls._cache[shift] = np.clip(img, 0, 255)
        return cls._cache[shift]

    def read(self):
        st = self._state()
        self.n_reads += 1
        img = self._base(int(st["shift"])) * float(st["light"]) * self._gain_factor()
        img = (img - 128.0) * (self._prop("contrast") / 50.0) + 128.0 + self._prop("brightness")
        lvl = self._screen_level()
        if lvl is not None:                                    # the monitor patch: the camera sees it a little dimmer
            x, y, w, h = self.PATCH
            img[y:y + h, x:x + w] = lvl * 0.85 + 10.0
        if st["covered"]:
            img = img * 0.03
        rng = np.random.default_rng(self.n_reads)
        img = img + rng.normal(0, 1.5, img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.float32)
        w, h = self.size
        if (w, h) != (1920, 1080):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        # colour: a cast whose strength follows the saturation control (zero near black and near white)
        s = self._prop("saturation") / 50.0
        taper = np.clip(img / 40.0, 0, 1) * np.clip((250.0 - img) / 60.0, 0, 1)
        bgr = np.stack([img + self.TINT_B * s * taper, img, img + self.TINT_R * s * taper], axis=-1)
        return True, np.clip(bgr, 0, 255).astype(np.uint8)

    def set_size(self, w: int, h: int) -> None:
        self.size = (int(w), int(h))

    def release(self) -> None:
        pass


# ----------------------------------------------------------------------------- picture analysis
def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def find_seam(gray: np.ndarray) -> Optional[int]:
    """Row where the main view ends and the panorama strip begins (the biggest jump between neighbouring rows in
    the lower part of the picture). None if there is no clear jump (a dark or uniform picture, or another camera)."""
    H = gray.shape[0]
    d = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(axis=1)
    lo, hi = int(H * 0.55), int(H * 0.95)
    r = lo + int(np.argmax(d[lo:hi]))
    median = float(np.median(d)) + 1e-6
    return r + 1 if (d[r] > 25 and d[r] > 4 * median) else None


def region_stats(crop_gray: np.ndarray, crop_bgr: Optional[np.ndarray] = None) -> dict:
    g = crop_gray
    h, w = g.shape
    if w != SHARP_WIDTH:
        norm = cv2.resize(g, (SHARP_WIDTH, max(8, int(round(h * SHARP_WIDTH / w)))),
                          interpolation=cv2.INTER_AREA if w > SHARP_WIDTH else cv2.INTER_LINEAR)
    else:
        norm = g
    out = dict(
        mean_brightness=round(float(g.mean()), 2),
        max=int(g.max()),
        saturated_fraction=round(float((g >= SAT_LEVEL).mean()), 4),
        rms_contrast=round(float(g.std()), 2),                       # spread of the grey levels (what the contrast control changes)
        sharpness=round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1),
        sharpness_norm=round(float(cv2.Laplacian(norm, cv2.CV_64F).var()), 1),
        size=[int(w), int(h)],
    )
    if crop_bgr is not None and crop_bgr.ndim == 3 and crop_bgr.shape[2] == 3:
        out["colour_saturation"] = round(float(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[..., 1].mean()), 2)   # 0..255
    return out


# ----------------------------------------------------------------------------- verified controls
SETTABLE = ("screen_level", "exposure", "gain", "brightness", "contrast", "saturation", "focus")    # what set_control may change


def controls_path() -> Path:
    return Path(os.environ.get("HW3_CONTROLS_FILE", "controls.json"))


def load_controls(path=None) -> dict:
    """The controls that probe_camera.py verified on this camera (the picture really changed when the value changed):
    {name: {"min", "max", "unit", "needs": {prop: value}, ...}}. Empty when the file does not exist: then the camera has no
    verified controls and the tool set_control is not offered at all. A file made on another operating system is ignored
    (the value ranges of Windows and macOS drivers differ): run probe_camera.py on this computer."""
    p = Path(path) if path else controls_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    made_on = (data.get("camera") or {}).get("platform")
    if made_on and os.environ.get("HW3_CAMERA_BACKEND") != "fake" and made_on != sys.platform:
        print(f"[connector] {p} was made on '{made_on}', this computer is '{sys.platform}': its controls are ignored. "
              f"Run probe_camera.py here.", file=sys.stderr)
        return {}
    out = {}
    for name, c in (data.get("controls") or {}).items():
        if name in SETTABLE and isinstance(c, dict) and isinstance(c.get("min"), (int, float)) \
                and isinstance(c.get("max"), (int, float)) and c["min"] < c["max"]:
            out[name] = c
    return out


def controls_summary(controls: dict) -> str:
    return ", ".join(f"{n} {c['min']:g}..{c['max']:g} ({c.get('unit', 'driver units')})" for n, c in controls.items())


# ----------------------------------------------------------------------------- the camera
class PanoCamera:
    def __init__(self, backend, index: int, workdir, layout_verified: bool = True):
        self.b = backend
        self.index = index
        self.workdir = Path(workdir)
        self.layout_verified = layout_verified
        self.roi_file = Path(os.environ.get("HW3_ROI_FILE", "roi.json"))
        self.resolution: Optional[tuple[int, int]] = None
        self.extra_discard = 0          # frames to throw away at the next grab (auto exposure settles after a size change)
        self.controls = load_controls()  # verified controls (empty: none)
        self._initial: dict = {}        # values of the controls before the first change, to put them back
        self.screen = None              # the monitor patch, opened when screen_level is among the verified controls
        if "screen_level" in self.controls:
            try:
                self.screen = scr.open_screen(self.workdir)
            except scr.ScreenError as e:
                raise CameraError(f"the test patch window could not be opened: {e}") from e

    # ---- commands and readings
    def _read(self, tries: int = 5):
        for _ in range(tries):
            ok, f = self.b.read()
            if ok and f is not None:
                return f
        return None

    def grab(self, n_frames: int = 1, n_discard: int = 3) -> list[np.ndarray]:
        n_discard += self.extra_discard
        self.extra_discard = 0
        for _ in range(n_discard):
            self._read(1)
        frames = []
        for _ in range(n_frames):
            f = self._read()
            if f is None:
                raise CameraError("no picture received from the camera (switched off? unplugged? "
                                  "another program using it?)")
            frames.append(f)
        h, w = frames[-1].shape[:2]
        self.resolution = (w, h)
        return frames

    def _try_grab(self):
        try:
            return self.grab(1, n_discard=8)[0]
        except CameraError:
            return None

    def set_resolution(self, width: int, height: int) -> dict:
        if (width, height) not in ALLOWED_RESOLUTIONS:
            allowed = ", ".join(f"{w}x{h}" for w, h in ALLOWED_RESOLUTIONS)
            return dict(ok=False, error=f"rejected: {width}x{height} is not offered; choose one of {allowed}",
                        requested=[width, height])
        self.b.set_size(width, height)
        f = self._try_grab()
        reopened = False
        if (f is None or f.shape[1::-1] != (width, height)) and hasattr(self.b, "reopen"):
            self.b.reopen(width, height)                      # the driver may need the device to be opened again
            reopened = True
            f = self._try_grab()
        if f is None:
            if hasattr(self.b, "reopen"):                     # leave a working camera behind
                self.b.reopen(*ALLOWED_RESOLUTIONS[-1])
            raise CameraError(f"the camera gave no picture at {width}x{height}; it was reopened at "
                              f"{ALLOWED_RESOLUTIONS[-1][0]}x{ALLOWED_RESOLUTIONS[-1][1]}")
        h, w = f.shape[:2]
        self.resolution = (w, h)
        self.extra_discard = 12
        ok = (w, h) == (width, height)
        return dict(ok=ok, requested=[width, height], accepted=[w, h],
                    note=("the camera delivers this size now" + (" (the device had to be reopened)" if reopened else ""))
                    if ok else f"the camera kept {w}x{h}: it did not accept {width}x{height}")

    # ---- controls (only those verified by probe_camera.py)
    def _get(self, name: str) -> Optional[float]:
        if name == "screen_level":
            return self.screen.get() if self.screen is not None else None
        return self.b.get_prop(name)

    def _set(self, name: str, value: float) -> bool:
        if name == "screen_level":
            if self.screen is None:
                return False
            try:
                return bool(self.screen.set(value))
            except scr.ScreenError as e:
                raise CameraError(str(e)) from e
        return bool(self.b.set_prop(name, value))

    def _remember_initial(self) -> None:
        if self._initial:
            return
        names = set(self.controls) | {n for c in self.controls.values() for n in (c.get("needs") or {})}
        for n in names:
            v = self._get(n)
            if v is not None:
                self._initial[n] = v

    def control_values(self) -> dict:
        return {n: self._get(n) for n in self.controls}

    def set_control(self, name: str, value) -> dict:
        if not self.controls:
            return dict(ok=False, error="rejected: this camera has no verified controls (only the resolution can be set)")
        cfg = self.controls.get(name)
        if cfg is None:
            return dict(ok=False, error=f"rejected: unknown control {name!r}; choose one of {controls_summary(self.controls)}")
        try:
            value = float(value)
        except (TypeError, ValueError):
            return dict(ok=False, error=f"rejected: value {value!r} is not a number")
        if not (cfg["min"] <= value <= cfg["max"]):
            return dict(ok=False, error=f"rejected: {name} must be between {cfg['min']:g} and {cfg['max']:g} "
                                        f"({cfg.get('unit', 'driver units')}), not {value:g}")
        asked = value
        step = cfg.get("step")
        if isinstance(step, (int, float)) and step >= 1 and float(step).is_integer():
            value = float(round(value))                              # the camera takes whole numbers
        self._remember_initial()
        before = self._get(name)
        for prop, v in (cfg.get("needs") or {}).items():          # e.g. exposure only acts when auto exposure is off
            self._set(prop, v)
        self._set(name, value)
        got = self._get(name)
        self._settle(cfg)
        tol = max(1e-6, 0.02 * (cfg["max"] - cfg["min"]) / 10)
        ok = got is not None and abs(got - value) <= tol
        return dict(ok=ok, control=name, requested=value, read_back=None if got is None else round(got, 4),
                    before=None if before is None else round(before, 4), range=[cfg["min"], cfg["max"]],
                    unit=cfg.get("unit", "driver units"),
                    note=(("the camera took this value" + (f" (rounded from {asked:g} to a whole number)" if asked != value else "")
                           + "; measure to see its effect") if ok else f"the driver gave {got} instead of {value:g}"))

    def _settle(self, cfg: dict) -> None:
        """After a command the picture needs a moment to follow (the camera's automatic exposure, the monitor): wait, and
        throw away a few frames at the next reading."""
        wait = float(os.environ.get("HW3_SETTLE_S", cfg.get("settle_s", 0.0)))
        if wait > 0:
            time.sleep(wait)
        self.extra_discard = max(self.extra_discard, 10 if wait > 0 else 8)

    def reset_controls(self) -> dict:
        """Put every control back to the value it had before the first set_control (auto modes last)."""
        if not self._initial:
            return dict(ok=True, restored={}, note="nothing was changed")
        order = sorted(self._initial, key=lambda n: n.startswith("auto"))
        restored = {}
        for n in order:
            self._set(n, self._initial[n])
            restored[n] = self._get(n)
        self.extra_discard = max(self.extra_discard, 8)
        return dict(ok=True, restored={k: (None if v is None else round(v, 4)) for k, v in restored.items()})

    # ---- regions
    def layout(self, frame: np.ndarray) -> dict:
        H = frame.shape[0]
        seam = find_seam(to_gray(frame))
        found = seam is not None
        if not found:
            seam = int(round(SEAM_FRACTION * H))
        return dict(seam_row=int(seam), seam_found=found)

    def load_rois(self, frame_shape) -> tuple[dict, str]:
        """Named rectangles (x, y, w, h) of the current frame size, from roi.json (made by pick_roi.py)."""
        H, W = frame_shape[:2]
        rois: dict = {}
        source = "roi.json"
        if self.roi_file.exists():
            try:
                data = json.loads(self.roi_file.read_text(encoding="utf-8"))
                for name, r in data.items():
                    sx = W / float(r.get("frame_width", W))
                    sy = H / float(r.get("frame_height", H))
                    x, y = int(round(r["x"] * sx)), int(round(r["y"] * sy))
                    w, h = int(round(r["w"] * sx)), int(round(r["h"] * sy))
                    x, y = max(0, min(x, W - 2)), max(0, min(y, H - 2))
                    w, h = max(2, min(w, W - x)), max(2, min(h, H - y))
                    rois[name] = (x, y, w, h)
            except Exception as e:  # noqa: BLE001
                raise CameraError(f"cannot read {self.roi_file}: {e}") from e
        if not rois:
            source = "default (no roi.json found)"
            seam = self.layout(np.zeros((H, W), np.uint8))["seam_row"]
            rois["scene"] = (W // 4, seam // 4, W // 2, seam // 2)
        return rois, source

    def regions(self, frame: np.ndarray) -> dict:
        H, W = frame.shape[:2]
        seam = self.layout(frame)["seam_row"]
        regs = {"main_view": (0, 0, W, seam),
                "panorama": (0, seam, W, max(2, H - seam - PANORAMA_BOTTOM_SKIP))}
        rois, _ = self.load_rois(frame.shape)
        regs.update(rois)
        return regs

    def region_names(self) -> list[str]:
        f = self.grab(1, n_discard=2)[0]
        return list(self.regions(f).keys())

    @staticmethod
    def crop(gray: np.ndarray, rect) -> np.ndarray:
        x, y, w, h = rect
        return gray[y:y + h, x:x + w]

    # ---- tools
    def info(self) -> dict:
        f = self.grab(1, n_discard=2)[0]
        lay = self.layout(f)
        rois, source = self.load_rois(f.shape)
        regs = self.regions(f)
        return dict(
            ok=True,                                  # without it camera_info was shown as "REJECTED: ?" although the camera answered
            instrument="Pano-360 (THAT DRAGONFLY), USB video class camera, 360-degree optics",
            interface=(f"{plat.api_label(getattr(self.b, 'api', None)) if self.b.kind == 'opencv' else 'synthetic'} through OpenCV "
                       f"({self.b.kind} backend), camera index {self.index}"
                       + ("; camera controls through uvc-util (USB video class control requests)"
                          if getattr(self.b, "uvc", None) is not None else "")
                       + ("; a window on the monitor in front of the camera shows a grey patch whose level is commanded (screen_level)"
                          if self.screen is not None else "")),
            layout_verified=bool(self.layout_verified),
            frame_size=[int(f.shape[1]), int(f.shape[0])],
            layout=lay, regions={k: list(v) for k, v in regs.items()},
            measured_region_source=source,
            allowed_resolutions=[f"{w}x{h}" for w, h in ALLOWED_RESOLUTIONS],
            controls=({"verified": {n: dict(min=c["min"], max=c["max"], unit=c.get("unit", "driver units"),
                                            current=self._get(n)) for n, c in self.controls.items()},
                       "other": "resolution (allowed_resolutions)"} if self.controls else
                      "resolution only; no camera control (brightness, contrast, saturation ...) is available on this computer"),
            colour=bool(f.ndim == 3 and (np.abs(f[..., 0].astype(int) - f[..., 2].astype(int)).mean() > 2.0)),
        )

    def capture(self, label: str = "", n_frames: int = 1) -> dict:
        n_frames = max(1, min(int(n_frames), 30))
        frames = self.grab(n_frames)
        f = frames[-1]
        gray = to_gray(f)
        lay = self.layout(f)
        regs = self.regions(f)
        stats = {name: region_stats(self.crop(gray, r), self.crop(f, r) if f.ndim == 3 else None) for name, r in regs.items()}
        crops_dir = self.workdir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        k = len(list(crops_dir.glob("capture_*_*.png"))) // max(1, len(self._roi_names(f))) + 1
        files = []
        for name in self._roi_names(f):
            p = crops_dir / f"capture_{k:03d}_{name}.png"
            plat.imwrite(p, self.crop(f if f.ndim == 3 else gray, regs[name]))
            files.append(p.relative_to(self.workdir).as_posix())
        return dict(ok=True, capture=k, label=label, resolution=[int(f.shape[1]), int(f.shape[0])],
                    seam_found=lay["seam_found"], regions=stats, saved_crops=files)

    def _roi_names(self, frame) -> list[str]:
        return list(self.load_rois(frame.shape)[0].keys())

    def measure_region(self, region: str = "scene", n_frames: int = 5) -> dict:
        n_frames = max(1, min(int(n_frames), 30))
        frames = self.grab(n_frames)
        regs = self.regions(frames[0])
        if region not in regs:
            return dict(ok=False, error=f"rejected: unknown region {region!r}; choose one of {', '.join(regs)}")
        per = []
        for fr in frames:
            rect = self.regions(fr)[region]
            per.append(region_stats(self.crop(to_gray(fr), rect), self.crop(fr, rect) if fr.ndim == 3 else None))
        means = np.array([p["mean_brightness"] for p in per])
        extra = {}
        if "colour_saturation" in per[0]:
            extra["colour_saturation"] = round(float(np.mean([p["colour_saturation"] for p in per])), 2)
        return dict(
            ok=True, region=region, n_frames=n_frames, resolution=[int(frames[0].shape[1]), int(frames[0].shape[0])],
            mean_brightness=round(float(means.mean()), 2),
            brightness_noise_std=round(float(means.std()), 3),
            max=int(max(p["max"] for p in per)),
            saturated_fraction=round(float(np.mean([p["saturated_fraction"] for p in per])), 4),
            rms_contrast=round(float(np.mean([p["rms_contrast"] for p in per])), 2),
            sharpness=round(float(np.mean([p["sharpness"] for p in per])), 1),
            sharpness_norm=round(float(np.mean([p["sharpness_norm"] for p in per])), 1),
            region_size=per[0]["size"], **extra,
        )

    def _ref_paths(self, name: str):
        d = self.workdir / "references"
        return d / f"{name}.png", d / f"{name}.json"

    def save_reference(self, name: str, overwrite: bool = False) -> dict:
        if not NAME_RE.match(name or ""):
            return dict(ok=False, error="rejected: name must be 1-40 characters: letters, digits, _ or -")
        png, js = self._ref_paths(name)
        if png.exists() and not overwrite:
            return dict(ok=False, error=f"rejected: reference {name!r} already exists; saving would overwrite it "
                                        f"(call again with overwrite=true only if that is wanted)")
        frames = self.grab(3)
        regs = self.regions(frames[-1])
        if "scene" not in regs:
            return dict(ok=False, error="rejected: no region named 'scene' (run pick_roi.py)")
        crop = np.mean([self.crop(to_gray(fr), regs["scene"]).astype(np.float32) for fr in frames], axis=0)
        crop = np.clip(crop, 0, 255).astype(np.uint8)
        png.parent.mkdir(parents=True, exist_ok=True)
        existed = png.exists()
        plat.imwrite(png, crop)
        stats = region_stats(crop)
        js.write_text(json.dumps(dict(name=name, stats=stats, resolution=list(self.resolution or [])), indent=2), encoding="utf-8")
        return dict(ok=True, name=name, overwritten=existed, file=png.relative_to(self.workdir).as_posix(), region="scene",
                    stats=stats)

    def compare_to_reference(self, name: str, region: str = "scene") -> dict:
        if not NAME_RE.match(name or ""):
            return dict(ok=False, error="rejected: name must be 1-40 characters: letters, digits, _ or -")
        png, _ = self._ref_paths(name)
        if not png.exists():
            return dict(ok=False, error=f"rejected: no reference named {name!r} (save_reference first)")
        ref = plat.imread(png, cv2.IMREAD_GRAYSCALE)
        frames = self.grab(3)
        regs = self.regions(frames[-1])
        if region not in regs:
            return dict(ok=False, error=f"rejected: unknown region {region!r}; choose one of {', '.join(regs)}")
        cur = np.mean([self.crop(to_gray(fr), regs[region]).astype(np.float32) for fr in frames], axis=0)
        cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
        a, b = cur, ref.astype(np.float32)
        diff = float(np.abs(a - b).mean())
        denom = float(a.std() * b.std())
        corr = float(((a - a.mean()) * (b - b.mean())).mean() / denom) if denom > 1e-6 else 0.0
        return dict(ok=True, reference=name, region=region,
                    mean_abs_difference=round(diff, 2),
                    brightness_change=round(float(a.mean() - b.mean()), 2),
                    correlation=round(corr, 4),
                    note="mean_abs_difference is in grey levels (0..255); correlation 1.0 = identical pattern")

    def close(self) -> None:
        try:
            self.reset_controls()
        except Exception:  # noqa: BLE001
            pass
        if self.screen is not None:
            self.screen.close()
        self.b.release()


# ----------------------------------------------------------------------------- opening
def open_camera(workdir, index: Optional[int] = None) -> PanoCamera:
    """Open the Pano-360. Without an index, the cameras 0, 1, 2 ... are tried in order and the first one whose picture
    has the two-part layout (main view above a panorama strip) is taken, so that the camera of the laptop is never
    mistaken for it. HW3_CAMERA_INDEX or `index` forces a number."""
    kind = os.environ.get("HW3_CAMERA_BACKEND", "opencv")
    if kind == "fake":
        return PanoCamera(FakeBackend(0), 0, workdir)
    forced = index if index is not None else (int(os.environ["HW3_CAMERA_INDEX"])
                                              if os.environ.get("HW3_CAMERA_INDEX") else None)
    candidates = [forced] if forced is not None else list(range(0, 6))
    kept: Optional[PanoCamera] = None
    for i in candidates:
        b = CvBackend(i)
        if not b.ok:
            b.release()
            if forced is None:
                break                                   # indexes are consecutive: no more cameras
            continue
        cam = PanoCamera(b, i, workdir, layout_verified=True)
        try:
            f = cam.grab(1, n_discard=15)[0]
        except CameraError:
            b.release()
            continue
        if find_seam(to_gray(f)) is not None:
            return cam
        if forced is not None:                          # the user named the camera: trust, but say it was not verified
            cam.layout_verified = False
            kept = cam
        else:
            b.release()
    if kept is not None:
        return kept
    raise CameraError(
        "the Pano-360 was not found: no camera shows the two-part picture (main view + panorama strip). "
        "Check that it is switched on and plugged in. " + plat.permission_hint() + ". A dark picture (lens covered) also hides "
        "the layout: uncover the lens, or force the number with HW3_CAMERA_INDEX=0 (Windows: HW3_CV_API=msmf tries the "
        "other driver).")
