"""Find the camera and check which of its controls REALLY work (Windows, macOS, Linux).

    python probe_camera.py                 # all cameras, then writes controls.json for the Pano-360
    python probe_camera.py --index 0       # only camera 0
    python probe_camera.py --api msmf      # Windows: try the other driver (default: dshow)
    python probe_camera.py --quick         # no control tests, only the camera, the layout and the resolutions
    python probe_camera.py --uvc-util /full/path/to/uvc-util     # macOS: where the uvc-util binary is
    python probe_camera.py --screen        # also test the monitor patch (screen_level): the camera must look at the monitor

The Pano-360 implements four UVC controls: brightness, contrast, saturation, hue. Windows: OpenCV (DirectShow) reaches them.
macOS: OpenCV cannot (it answers -1), so the tool uvc-util is used (https://github.com/jtfrey/uvc-util: git clone, cd uvc-util,
make). Exposure, gain and focus are tried too, in case another camera or driver has them.

Before you run it: close the Camera app and every program that shows the camera, switch the camera on, point it at a STILL
scene with some colour in it (steady light, nobody moving in front of it). The test changes the camera settings one by one,
measures the picture and puts every setting back.

What it decides: a control is VERIFIED only if the picture really changes when the value changes. What is measured depends on
the control: brightness / exposure / gain -> the mean grey level moves by at least 6; contrast -> the spread of the grey levels
(rms contrast) changes by at least 15 %; saturation -> the colour saturation changes by at least 30 %; focus -> the sharpness
changes by 30 %. Only verified controls are written to controls.json, and the MCP server offers set_control only for them.
Hue is reported but not used (it shifts colours; the project measures brightness, contrast and colour saturation).

--screen: a window with a uniform grey patch opens on the monitor; the camera must look at it and the region 'scene' of roi.json
must lie inside the patch (python pattern_window.py --level 128; python snap.py; python pick_roi.py probe_out/snap.png scene).
The probe shows the patch at nine levels from black to white and measures the mean grey of the region 'scene' (it must move by
at least 10); then screen_level is a verified control like the others.
Send me the printed text, not the pictures. Pictures stay in probe_out/ on your computer (do not put them on GitHub).

If the script is stopped half way, it puts the settings back before it ends; if the camera was unplugged meanwhile, plug it in again.
"""
import argparse
import datetime
import json
import os
import pathlib
import sys
import time

import cv2
import numpy as np

import cam_platform as plat
import screen_patch as scr
import uvc_controls as uvc

OUT = pathlib.Path("probe_out")
RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]
PROPS = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS, "contrast": cv2.CAP_PROP_CONTRAST, "saturation": cv2.CAP_PROP_SATURATION,
    "hue": cv2.CAP_PROP_HUE, "gain": cv2.CAP_PROP_GAIN, "exposure": cv2.CAP_PROP_EXPOSURE,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE, "focus": cv2.CAP_PROP_FOCUS, "autofocus": cv2.CAP_PROP_AUTOFOCUS,
}
# values tried when the range of a control is not known (OpenCV does not tell it); uvc-util tells it, then it is used
SWEEPS = {
    "gain": [0, 16, 32, 64, 96, 128, 192, 255],
    "brightness": [-64, -32, 0, 32, 64, 128, 192, 255],
    "contrast": [0, 16, 32, 64, 96, 128, 192, 255],
    "saturation": [0, 16, 32, 64, 96, 128, 192, 255],
    "focus": [0, 20, 40, 60, 80, 100, 150, 200, 255],
}
# control -> (what is measured, key of read_all(), mode, threshold, smallest absolute change)
#   mode "abs": verified if (largest - smallest value) >= threshold;  mode "ratio": if largest / smallest >= threshold
METRICS = {
    "screen_level": ("mean grey of the region 'scene' (on the monitor patch)", "mean", "abs", 10.0, 0.0),
    "exposure":   ("picture mean grey", "mean", "abs", 6.0, 0.0),
    "gain":       ("picture mean grey", "mean", "abs", 6.0, 0.0),
    "brightness": ("picture mean grey", "mean", "abs", 6.0, 0.0),
    "contrast":   ("spread of the grey levels (rms contrast)", "contrast", "ratio", 1.15, 2.0),
    "saturation": ("colour saturation (HSV S, 0-255)", "colour", "ratio", 1.3, 4.0),
    "focus":      ("sharpness (Laplacian variance)", "sharp", "ratio", 1.3, 0.0),
}
MANUAL_EXPOSURE_CANDIDATES = [0.25, 1.0, 0.0]     # values of auto_exposure that mean "manual": DirectShow 0.25, V4L2 1, some 0
UNITS = {"exposure": "driver units (DirectShow: log2 of seconds, -5 = 1/32 s)", "gain": "driver units",
         "brightness": "driver units", "contrast": "driver units", "saturation": "driver units", "focus": "driver units"}
SCREEN_SETTLE_S = 1.0     # seconds to wait after a new monitor level (the camera's automatic exposure follows)
LOW_COLOUR = 12.0         # colour saturation (0-255) below which the scene has hardly any colour to measure


class Controls:
    """get / set a control by name: the monitor patch (screen_level), uvc-util when it implements the control (macOS), else
    OpenCV. `roi` is the rectangle (x, y, w, h) of the region 'scene' that screen_level is judged by."""

    def __init__(self, cap, tool, screen=None, roi=None):
        self.cap, self.tool, self.screen, self.roi = cap, tool, screen, roi

    def via(self, name):
        if name == "screen_level":
            return "screen"
        return "uvc-util" if (self.tool is not None and name in self.tool.names) else "opencv"

    def get(self, name):
        if name == "screen_level":
            return self.screen.get() if self.screen is not None else None
        if self.via(name) == "uvc-util":
            return self.tool.get(name)
        v = self.cap.get(PROPS[name])
        return None if v is None or v != v or v == -1.0 else float(v)

    def set(self, name, value):
        if name == "screen_level":
            ok = bool(self.screen.set(value)) if self.screen is not None else False
            time.sleep(float(os.environ.get("HW3_SETTLE_S", SCREEN_SETTLE_S)))     # the camera's automatic exposure follows
            return ok
        if self.via(name) == "uvc-util":
            return self.tool.set(name, value)
        return bool(self.cap.set(PROPS[name], float(value)))

    def range(self, name):
        if name == "screen_level":
            return dict(min=0.0, max=255.0, step=1.0, default=float(scr.DEFAULT_LEVEL))
        return self.tool.control_range(name) if self.via(name) == "uvc-util" else None


def frames(cap, n_discard, n):
    for _ in range(n_discard):
        cap.read()
    out = []
    for _ in range(n):
        ok, f = cap.read()
        if ok and f is not None:
            out.append(f)
    return out


def read_all(cap, n_discard=8, n=3, roi=None) -> dict:
    """The four pictures statistics that the controls are judged by (mean of n frames); of the rectangle `roi` if given."""
    fs = frames(cap, n_discard, n)
    if roi is not None:
        x, y, w, h = roi
        fs = [f[y:y + h, x:x + w] for f in fs]
    if not fs:
        return dict(mean=float("nan"), contrast=float("nan"), colour=float("nan"), sharp=float("nan"))
    m, c, k, sh = [], [], [], []
    for f in fs:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        m.append(float(g.mean()))
        c.append(float(g.std()))
        k.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 1].mean()))
        sh.append(float(cv2.Laplacian(g[h // 4:3 * h // 4, w // 4:3 * w // 4], cv2.CV_64F).var()))
    return dict(mean=float(np.mean(m)), contrast=float(np.mean(c)), colour=float(np.mean(k)), sharp=float(np.mean(sh)))


def mean_grey(cap, n_discard=8, n=3) -> float:
    return read_all(cap, n_discard, n)["mean"]


def seam_found(frame) -> bool:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = np.abs(np.diff(g, axis=0)).mean(axis=1)
    H = g.shape[0]
    lo, hi = int(H * 0.55), int(H * 0.95)
    r = lo + int(np.argmax(d[lo:hi]))
    return bool(d[r] > 25 and d[r] > 4 * (float(np.median(d)) + 1e-6))


def describe(frame) -> dict:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    b, gr, r = (frame[..., i].astype(float) for i in range(3))
    colour_diff = float((np.abs(b - gr).mean() + np.abs(gr - r).mean()) / 2)
    return dict(height=int(frame.shape[0]), width=int(frame.shape[1]), mean=round(float(g.mean()), 1),
                saturated_fraction=round(float((g >= 250).mean()), 4), grey_picture=colour_diff < 2.0,
                two_part_layout=seam_found(frame))


def trial(ctl, name, value, key):
    """Set a control, read back what the camera took, measure the picture. Returns (read back, measurement dict)."""
    ctl.set(name, value)
    got = ctl.get(name)
    if name == "screen_level":
        return got, read_all(ctl.cap, 10, 3, roi=ctl.roi)
    return got, read_all(ctl.cap)


def sweep(ctl, name, values, key):
    rows = []
    for v in values:
        got, m = trial(ctl, name, v, key)
        rows.append(dict(asked=v, got=got, value=m[key], mean=round(m["mean"], 2)))
    return rows


def verdict(rows, name):
    """(verified, spread, lo, hi) from the sweep rows: at least 3 distinct read-back values and a real change of the picture."""
    _, key, mode, thr, min_abs = METRICS[name]
    ok = [r for r in rows if r["got"] is not None and r["value"] == r["value"]]
    distinct = {round(r["got"], 3) for r in ok}
    if len(distinct) < 3:
        return False, 0.0, None, None
    vals = [r["value"] for r in ok]
    hi, lo = max(vals), min(vals)
    if mode == "ratio":
        spread = hi / max(lo, 1e-6 if key == "sharp" else 1.0)
        verified = spread >= thr and (hi - lo) >= min_abs
    else:
        spread = hi - lo
        verified = spread >= thr
    gots = [r["got"] for r in ok]
    return verified, round(spread, 2), min(gots), max(gots)


def show_rows(rows, fmt="{:.1f}"):
    print("      asked   " + " ".join(f"{r['asked']:>7g}" for r in rows))
    print("      read    " + " ".join(f"{'--' if r['got'] is None else format(r['got'], 'g'):>7}" for r in rows))
    print("      value   " + " ".join(f"{fmt.format(r['value']):>7}" for r in rows))


def sweep_values(ctl, name):
    """Values to try: the whole range when uvc-util tells it (9 points), else the standard list for OpenCV."""
    r = ctl.range(name)
    if r:
        lo, hi, step = r["min"], r["max"], r.get("step", 1.0)
        vals = [lo + (hi - lo) * i / 8 for i in range(9)]
        if float(step).is_integer() and step >= 1:
            vals = [float(round(v)) for v in vals]
        if "default" in r:
            vals.append(r["default"])
        return sorted(set(vals))
    return SWEEPS[name]


def probe_controls(ctl) -> tuple[dict, dict]:
    """Returns (verified controls for controls.json, full report). Every setting is put back, also after an error."""
    verified, report = {}, {}
    start = {n: ctl.get(n) for n in PROPS}
    if ctl.screen is not None:
        start["screen_level"] = ctl.get("screen_level")
    print("    values read at the start: " + ", ".join(f"{n}={'n/a' if v is None else format(v, 'g')}" for n, v in start.items()))
    try:
        _probe(ctl, start, verified, report)
    finally:
        order = sorted((n for n in start if start[n] is not None), key=lambda n: n.startswith("auto"))
        for n in order:                                  # exposure / gain first, the automatic modes last
            ctl.set(n, start[n])
        read_all(ctl.cap, 12, 1)
    report["start"] = start
    return verified, report


def _probe(ctl, start, verified, report):
    manual_m, e_mid = None, None            # the auto_exposure value that means "manual", and an exposure that gives a mid-grey picture
    base = mean_grey(ctl.cap)
    report["mean_at_start"] = round(base, 1)
    cap = ctl.cap

    # ---- exposure: first find out which auto_exposure value makes it manual
    if start["exposure"] is None:
        print("    exposure       not supported by this driver / camera")
        report["exposure"] = dict(verdict="not supported")
    else:
        needs, chosen = {}, None
        cands = MANUAL_EXPOSURE_CANDIDATES if start["auto_exposure"] is not None else [None]
        for m in cands:
            if m is not None:
                ctl.set("auto_exposure", m)
            a, b = trial(ctl, "exposure", -2, "mean")[1]["mean"], trial(ctl, "exposure", -9, "mean")[1]["mean"]
            if abs(a - b) >= METRICS["exposure"][3]:
                chosen = m
                break
        if chosen is None and cands != [None]:
            print(f"    exposure       no value of auto_exposure {MANUAL_EXPOSURE_CANDIDATES} lets exposure change the picture")
            report["exposure"] = dict(verdict="no effect")
        else:
            if chosen is not None:
                needs = {"auto_exposure": chosen}
                manual_m = chosen
            values = list(range(-1, -14, -1)) if start["exposure"] <= 0 else \
                sorted({round(start["exposure"] * k, 3) for k in (0.1, 0.25, 0.5, 1, 2, 4)} | {1, 5, 10, 50, 100, 200, 500})
            rows = sweep(ctl, "exposure", values, "mean")
            ok_, spread, lo, hi = verdict(rows, "exposure")
            print(f"    exposure       {'VERIFIED' if ok_ else 'no usable effect'} (picture mean moves by {spread}; manual mode: {needs or 'n/a'})")
            show_rows(rows)
            report["exposure"] = dict(rows=rows, needs=needs, verified=ok_, spread=spread)
            if ok_:
                # keep the range in which the picture is neither black nor white (the useful part of the sweep)
                useful = [r for r in rows if r["got"] is not None and 5 <= r["value"] <= 250]
                if useful:
                    e_mid = min(useful, key=lambda r: abs(r["value"] - 128))["got"]
                lo_u = min((r["got"] for r in useful), default=lo)
                hi_u = max((r["got"] for r in useful), default=hi)
                verified["exposure"] = dict(min=lo_u, max=hi_u, unit=UNITS["exposure"], needs=needs, via=ctl.via("exposure"),
                                            default=start["exposure"], spread=spread)
        # put exposure and the auto mode back
        ctl.set("exposure", start["exposure"])
        if start["auto_exposure"] is not None:
            ctl.set("auto_exposure", start["auto_exposure"])
        mean_grey(cap, 12, 1)

    # ---- the other controls
    for name in SWEEPS:
        v0 = start[name]
        if v0 is None:
            print(f"    {name:14s} not supported by this driver / camera")
            report[name] = dict(verdict="not supported")
            continue
        label, key, mode, thr, _ = METRICS[name]
        is_focus = name == "focus"
        if is_focus and start["autofocus"] is not None:
            ctl.set("autofocus", 0)
        values = sweep_values(ctl, name)
        rows = sweep(ctl, name, values, key)
        ok_, spread, lo, hi = verdict(rows, name)
        needs = {"autofocus": 0} if is_focus and start["autofocus"] is not None else {}
        note = ""
        ctl.set(name, v0)
        if not ok_ and manual_m is not None and key == "mean":
            # many drivers act on gain / brightness only when the exposure is manual: try again in that mode
            ctl.set("auto_exposure", manual_m)
            if e_mid is not None:
                ctl.set("exposure", e_mid)
            mean_grey(cap, 10, 1)
            rows2 = sweep(ctl, name, values, key)
            ok2, spread2, lo2, hi2 = verdict(rows2, name)
            if ok2:
                rows, ok_, spread, lo, hi = rows2, ok2, spread2, lo2, hi2
                needs = {"auto_exposure": manual_m}
                note = " [only with manual exposure]"
            ctl.set(name, v0)
            ctl.set("exposure", start["exposure"])
            ctl.set("auto_exposure", start["auto_exposure"])
        via = ctl.via(name)
        print(f"    {name:14s} {'VERIFIED' if ok_ else 'no usable effect'}{note} via {via}: {label} "
              f"{'x' if mode == 'ratio' else 'moves by '}{spread}")
        show_rows(rows, "{:.1f}")
        report[name] = dict(rows=rows, verified=ok_, spread=spread, needs=needs, via=via, measured=label)
        if not ok_ and key == "colour" and rows and max(r["value"] for r in rows) < LOW_COLOUR:
            print("                   the scene has almost no colour (colour saturation stays below "
                  f"{LOW_COLOUR:g} of 255): point the camera at something coloured and run again")
        if ok_:
            rng = ctl.range(name) or {}
            entry = dict(min=rng.get("min", lo), max=rng.get("max", hi), unit=("camera units (UVC)" if via == "uvc-util" else UNITS[name]),
                         needs=needs, via=via, default=rng.get("default", v0), spread=spread)
            if "step" in rng:
                entry["step"] = rng["step"]
            verified[name] = entry
        if is_focus and start["autofocus"] is not None:
            ctl.set("autofocus", start["autofocus"])
        mean_grey(cap, 8, 1)

    if ctl.screen is not None:
        name, v0 = "screen_level", start["screen_level"]
        label, key, mode, thr, _ = METRICS[name]
        rows = sweep(ctl, name, [0, 32, 64, 96, 128, 160, 192, 224, 255], key)
        ok_, spread, lo, hi = verdict(rows, name)
        ctl.set(name, v0)
        print(f"    {name:14s} {'VERIFIED' if ok_ else 'no usable effect'} via screen: {label} moves by {spread}")
        show_rows(rows, "{:.1f}")
        report[name] = dict(rows=rows, verified=ok_, spread=spread, needs={}, via="screen", measured=label)
        if ok_:
            verified[name] = dict(min=0, max=255, step=1, unit="grey level of the patch on the monitor (0 black, 255 white)",
                                  needs={}, via="screen", default=v0, spread=spread, settle_s=SCREEN_SETTLE_S)
        else:
            print("                   the camera reading did not follow the patch: is the camera pointed at the grey window, "
                  "and does the region 'scene' of roi.json lie inside it? (python pattern_window.py --level 128, then python snap.py, "
                  "then python pick_roi.py probe_out/snap.png scene)")

    if start["hue"] is not None:
        print("    hue            present on this camera; not tested and not offered (it shifts colours, the project "
              "measures brightness, contrast and colour saturation)")
        report["hue"] = dict(verdict="present, not used")


def scene_rect(frame_shape):
    """The region 'scene' of roi.json as (x, y, w, h) of a frame with this shape (None if there is no roi.json)."""
    p = pathlib.Path(os.environ.get("HW3_ROI_FILE", "roi.json"))
    if not p.exists():
        return None
    try:
        r = json.loads(p.read_text(encoding="utf-8"))["scene"]
    except (ValueError, KeyError):
        return None
    H, W = frame_shape[:2]
    sx, sy = W / float(r.get("frame_width", W)), H / float(r.get("frame_height", H))
    x, y = int(round(r["x"] * sx)), int(round(r["y"] * sy))
    w, h = int(round(r["w"] * sx)), int(round(r["h"] * sy))
    x, y = max(0, min(x, W - 2)), max(0, min(y, H - 2))
    return x, y, max(2, min(w, W - x)), max(2, min(h, H - y))


def probe_resolutions(idx, api, size0):
    out = []
    cap = None
    for w, h in RESOLUTIONS:
        if cap is not None:
            cap.release()
        cap = plat.open_capture(idx, api, size=(w, h))
        fs = frames(cap, 12, 1) if cap.isOpened() else []
        if fs:
            f = fs[0]
            out.append(dict(asked=[w, h], got=[int(f.shape[1]), int(f.shape[0])], two_part_layout=seam_found(f)))
        else:
            out.append(dict(asked=[w, h], got=None, two_part_layout=False))
    if cap is not None:
        cap.release()
    return out


def main() -> int:
    plat.utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--api", default=None, help="dshow | msmf | avfoundation | v4l2 | any")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--screen", action="store_true", help="also test the monitor patch (screen_level)")
    ap.add_argument("--uvc-util", default=None, help="macOS: full path of the uvc-util binary (or set HW3_UVC_UTIL)")
    args = ap.parse_args()
    if args.uvc_util:
        os.environ["HW3_UVC_UTIL"] = args.uvc_util
    api = plat.capture_api(args.api)
    OUT.mkdir(exist_ok=True)
    print(f"Platform backend: {plat.api_label(api)}\n")
    print("Cameras known to the system:\n" + plat.camera_names() + "\n")
    tool = uvc.shared() if uvc.wanted() else None
    if uvc.wanted():
        if tool:
            print(f"uvc-util: {tool.exe} (camera selected with: {' '.join(tool.target)}); controls it found: {', '.join(tool.names)}\n")
        else:
            print("uvc-util: NOT found, or it does not list the Pano-360 (0x1bcf:0x2c99). On macOS OpenCV cannot change the camera "
                  "controls, so without uvc-util there are none. Build it: git clone https://github.com/jtfrey/uvc-util ; cd uvc-util ; make ; "
                  "then run this script again with  --uvc-util /full/path/to/uvc-util\n")

    report, chosen, controls_doc = {}, None, None
    found = False
    indexes = [args.index] if args.index is not None else list(range(0, 6))
    for idx in indexes:
        cap = plat.open_capture(idx, api)
        if not cap.isOpened():
            cap.release()
            if args.index is None:
                break                                   # camera indexes are consecutive: no more cameras
            print(f"index {idx}: could not be opened\n{plat.permission_hint()}")
            continue
        fs = frames(cap, 20, 1)
        if not fs:
            print(f"index {idx}: opened, but no frame ({plat.permission_hint()})")
            cap.release()
            continue
        found = True
        info = describe(fs[0])
        plat.imwrite(OUT / f"camera_{idx}.png", fs[0])
        print(f"=== index {idx}: {info['width']} x {info['height']}, mean brightness {info['mean']} of 255, "
              f"{'GREY picture' if info['grey_picture'] else 'colour picture'}, "
              f"two-part layout (main view + panorama strip): {'YES -> this is the Pano-360' if info['two_part_layout'] else 'no'}"
              f"  (saved probe_out/camera_{idx}.png)")
        entry = dict(frame=info)
        is_pano = info["two_part_layout"] or args.index == idx
        if is_pano and not args.quick:
            print("    testing the controls one by one, keep the scene still (about 1-2 minutes) ...")
            screen, roi = None, None
            if args.screen:
                roi = scene_rect(fs[0].shape)
                if roi is None:
                    print("    --screen needs the region 'scene' (roi.json): show the patch with  python pattern_window.py --level 128 ,"
                          " aim the camera at it, run python snap.py and python pick_roi.py probe_out/snap.png scene , then run this again.")
                else:
                    print("    a grey window opens on your monitor: do not touch it and keep the camera pointed at it ...")
                    screen = scr.open_screen(OUT)
            try:
                verified, rep = probe_controls(Controls(cap, tool, screen, roi))
            finally:
                if screen is not None:
                    screen.close()
            entry["controls"] = rep
            if tool is not None:
                rep["uvc_util"] = dict(exe=tool.exe, target=tool.target, names=tool.names, raw=tool.raw)
            if chosen is None:
                chosen = idx
                vias = sorted({c.get("via", "opencv") for c in verified.values()})
                controls_doc = dict(camera=dict(index=idx, api=plat.api_label(api), frame=[info["width"], info["height"]],
                                                platform=sys.platform, via="+".join(vias) or "none",
                                                probed=datetime.datetime.now().isoformat(timespec="seconds")),
                                    controls=verified)
        cap.release()
        entry["resolutions"] = probe_resolutions(idx, api, (info["width"], info["height"])) if is_pano else []
        if entry["resolutions"]:
            print("    resolutions: " + ", ".join(
                f"asked {r['asked'][0]}x{r['asked'][1]} -> got {'x'.join(map(str, r['got'])) if r['got'] else 'no frame'}"
                f"{' (layout ok)' if r['two_part_layout'] else ' (layout NOT found)' if r['got'] else ''}"
                for r in entry["resolutions"]))
        report[idx] = entry
    if not found:
        print("No camera could be opened. Check: cable, camera is on, " + plat.permission_hint())
        return 1
    (OUT / "probe_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if controls_doc is not None:
        pathlib.Path("controls.json").write_text(json.dumps(controls_doc, indent=2), encoding="utf-8")
        names = list(controls_doc["controls"]) or ["none"]
        print(f"\nWrote controls.json for camera {chosen}: verified controls = {', '.join(names)}.")
        if names == ["none"]:
            print("No control changed the picture: the agent will have only the resolution and the readings.")
    elif not args.quick:
        print("\nNo camera with the two-part layout of the Pano-360 was found, so controls.json was not written "
              "(try --index N for the camera number).")
    print("Saved probe_out/probe_report.json. Send me the text printed above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
