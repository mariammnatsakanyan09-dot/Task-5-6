"""Offline check of probe_camera.py with fake drivers (no camera needed). Three cameras are imitated:

  1. a DirectShow-like driver: exposure (only after auto_exposure is set to 0.25), gain and brightness work; the rest reads -1
  2. a DirectShow-like driver of a camera like the Pano-360: brightness, contrast, saturation, hue work; no exposure / gain / focus
  3. macOS: OpenCV reads -1 for everything; the four controls are reached through a (fake) uvc-util command-line tool
  4. a camera whose own controls do nothing (like the real Pano-360), looking at a monitor patch: --screen verifies screen_level

    python3 test_probe_offline.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["HW3_CAMERA_BACKEND"] = "fake"
os.environ["HW3_SETTLE_S"] = "0"
os.environ.pop("HW3_UVC_UTIL", None)
import cv2  # noqa: E402

import camera_connector as cc  # noqa: E402
import cam_platform as plat  # noqa: E402
import fake_uvc_util  # noqa: E402
import probe_camera as P  # noqa: E402
import uvc_controls as uvc  # noqa: E402

INV = {v: k for k, v in P.PROPS.items()}


class FakeCap:
    """Looks like cv2.VideoCapture; the picture and the controls come from the synthetic camera. `supported` are the
    properties the imitated driver answers; all others read -1 and ignore set()."""

    def __init__(self, index, size, supported):
        self.b = cc.FakeBackend(index)
        self.b.size = size
        self.opened = index == 0
        self.supported = set(supported)
        self.b.props["auto_exposure"] = 0.75

    def isOpened(self):
        return self.opened

    def release(self):
        pass

    def read(self):
        return self.b.read()

    def set(self, prop, v):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            self.b.size = (int(v), self.b.size[1])
            return True
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self.b.size = (self.b.size[0], int(v))
            return True
        name = INV.get(prop)
        if name not in self.supported:
            return False
        if name == "auto_exposure":
            if v in (0.25, 0.75):                      # DirectShow semantics; other values are ignored
                self.b.props["auto_exposure"] = float(v)
                return True
            return False
        return self.b.set_prop(name, v)

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.b.size[0])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.b.size[1])
        name = INV.get(prop)
        v = self.b.props.get(name) if name in self.supported else None
        return -1.0 if v is None else v


def run_probe(supported, uvc_tool=False, argv=None, roi=False):
    tmp = Path(tempfile.mkdtemp())
    os.chdir(tmp)
    if roi:
        (tmp / "roi.json").write_text(json.dumps({"scene": dict(x=400, y=830, w=800, h=200, frame_width=1920, frame_height=1080)}))
    os.environ.pop("HW3_ROI_FILE", None)
    os.environ.pop("HW3_FAKE_SCREEN", None)
    if uvc_tool:
        os.environ["HW3_UVC_UTIL"] = fake_uvc_util.make_launcher(tmp / "uvc-util")
        os.environ["HW3_FAKE_UVC_STATE"] = str(tmp / "uvc_state.json")
        cc.FakeBackend._cache.clear()
    else:
        os.environ.pop("HW3_UVC_UTIL", None)
        os.environ.pop("HW3_FAKE_UVC_STATE", None)
    plat.open_capture = lambda index, api=None, size=(1920, 1080): FakeCap(index, size or (1920, 1080), supported)
    plat.camera_names = lambda: "(fake camera list)"
    P.OUT = Path("probe_out")
    sys.argv = ["probe_camera.py"] + list(argv or [])
    rc = P.main()
    assert rc == 0
    doc = json.loads(Path("controls.json").read_text(encoding="utf-8"))
    rep = json.loads(Path("probe_out/probe_report.json").read_text(encoding="utf-8"))
    return doc, rep, tmp


def main():
    # ---- 1. DirectShow-like: exposure, gain, brightness
    doc, rep, _ = run_probe({"exposure", "gain", "brightness", "auto_exposure"})
    ctl = doc["controls"]
    assert set(ctl) == {"exposure", "gain", "brightness"}, set(ctl)           # contrast / saturation / focus read -1: not offered
    assert ctl["exposure"]["needs"] == {"auto_exposure": 0.25}, ctl["exposure"]
    assert ctl["exposure"]["min"] < ctl["exposure"]["max"] <= -2 and doc["camera"]["index"] == 0
    assert ctl["gain"]["min"] == 0 and ctl["gain"]["max"] == 255
    assert doc["camera"]["platform"] == sys.platform and doc["camera"]["via"] == "opencv"
    res = rep["0"]["resolutions"]
    assert [r["got"] for r in res] == [[640, 480], [1280, 720], [1920, 1080]] and all(r["two_part_layout"] for r in res)
    assert rep["0"]["frame"]["two_part_layout"] is True
    loaded = cc.load_controls("controls.json")
    assert set(loaded) == {"exposure", "gain", "brightness"}
    print("ok  1 probe, DirectShow-like: verifies exposure (manual mode 0.25), gain, brightness; skips contrast/saturation/focus (-1); "
          "resolutions; writes controls.json that the connector loads")

    # ---- 2. DirectShow-like camera of the Pano-360 kind: brightness, contrast, saturation, hue
    doc, rep, _ = run_probe({"brightness", "contrast", "saturation", "hue"})
    ctl = doc["controls"]
    assert set(ctl) == {"brightness", "contrast", "saturation"}, set(ctl)    # hue is present but not offered
    assert rep["0"]["controls"]["hue"]["verdict"] == "present, not used"
    assert rep["0"]["controls"]["exposure"]["verdict"] == "not supported"
    for n in ctl:
        assert ctl[n]["via"] == "opencv" and ctl[n]["min"] < ctl[n]["max"] and ctl[n]["needs"] == {}, (n, ctl[n])
    assert ctl["brightness"]["min"] == -64 and ctl["brightness"]["max"] == 64
    assert ctl["contrast"]["max"] == 100 and ctl["saturation"]["max"] == 100
    assert set(cc.load_controls("controls.json")) == {"brightness", "contrast", "saturation"}
    print("ok  2 probe, camera with brightness / contrast / saturation / hue through OpenCV: the three that change the picture are verified "
          "(each by its own reading), hue is reported and not used")

    # ---- 3. macOS: OpenCV reads -1, the controls come from uvc-util
    doc, rep, tmp = run_probe(set(), uvc_tool=True)
    ctl = doc["controls"]
    assert set(ctl) == {"brightness", "contrast", "saturation"}, set(ctl)
    assert doc["camera"]["via"] == "uvc-util" and doc["camera"]["platform"] == sys.platform
    for n in ctl:
        assert ctl[n]["via"] == "uvc-util" and ctl[n]["unit"].startswith("camera units") and ctl[n]["step"] == 1, (n, ctl[n])
    assert (ctl["brightness"]["min"], ctl["brightness"]["max"], ctl["brightness"]["default"]) == (-64, 64, 0)    # taken from `-S`
    assert (ctl["contrast"]["min"], ctl["contrast"]["max"], ctl["contrast"]["default"]) == (0, 100, 50)
    u = rep["0"]["controls"]["uvc_util"]
    assert u["names"] == ["brightness", "contrast", "saturation", "hue"] and u["target"] == ["-V", "0x1bcf:0x2c99"]
    assert "minimum: -64" in u["raw"]["-S brightness"]
    st = json.loads((tmp / "uvc_state.json").read_text(encoding="utf-8"))
    assert st["brightness"] == 0 and st["contrast"] == 50 and st["saturation"] == 50, st               # every setting put back
    assert rep["0"]["controls"]["hue"]["verdict"] == "present, not used"
    print("ok  3 probe, macOS-like (OpenCV reads -1, uvc-util fake): ranges come from `-S`, brightness / contrast / saturation verified, "
          "settings put back, controls.json says via uvc-util")

    # ---- 4. the camera's own controls do nothing; the monitor patch is the control
    doc, rep, tmp = run_probe(set(), argv=["--screen"], roi=True)
    ctl = doc["controls"]
    assert set(ctl) == {"screen_level"}, set(ctl)
    c = ctl["screen_level"]
    assert (c["min"], c["max"], c["step"], c["default"], c["via"], c["settle_s"]) == (0, 255, 1, 128, "screen", 1.0), c
    assert c["spread"] > 100 and c["unit"].startswith("grey level"), c
    rows = rep["0"]["controls"]["screen_level"]["rows"]
    means = [r["value"] for r in rows]
    assert [r["asked"] for r in rows] == [0, 32, 64, 96, 128, 160, 192, 224, 255] and means == sorted(means), means
    assert os.environ.get("HW3_FAKE_SCREEN") and not Path(os.environ["HW3_FAKE_SCREEN"]).exists()          # the patch was closed
    assert set(cc.load_controls("controls.json")) == {"screen_level"}
    print("ok  4 probe --screen: the patch is shown at nine levels, the reading of the region 'scene' follows, screen_level is verified "
          "(0..255, settle 1 s); without roi.json the probe asks for it")
    doc, rep, tmp = run_probe(set(), argv=["--screen"], roi=False)
    assert doc["controls"] == {}, doc["controls"]

    # ---- parser details
    assert uvc.parse_show_control("x {\n minimum: {pan=-648000,tilt=-648000}\n maximum: 7\n step-size: 0.5\n default-value: 3 (x)\n}") == \
        dict(min=-648000, max=7, step=0.5, default=3)
    real = ("brightness {\n  type-description: {\n    single value, signed 16-bit integer\n  },\n  minimum: 0\n  maximum: 255\n"
            "  step-size: 1\n  default-value: 128\n  current-value: 128\n}\n")          # the answer of the real Pano-360
    assert uvc.parse_show_control(real) == dict(min=0, max=255, step=1, default=128)
    assert uvc.first_number("128\n") == 128
    assert uvc.first_number("value = -12.5") == -12.5 and uvc.first_number("none") is None
    print("ok  uvc-util parser: the real Pano-360 answer of `-S brightness`, {a=..} and decimal forms")


if __name__ == "__main__":
    main()
