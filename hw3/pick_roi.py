"""Mark the area of the picture that the project will measure (the sheet of paper).

Run (after probe_camera.py, from the same folder, with the same environment):

    python pick_roi.py probe_out/camera_1.png            # use the number of YOUR camera
    python pick_roi.py probe_out/camera_1.png paper      # optional: a name for the area (default: paper)

A window opens with the saved picture. Drag a rectangle around the sheet of paper with the mouse and press
ENTER (or SPACE). ESC cancels. The rectangle is written to roi.json; a crop of it is saved to
probe_out/roi_<name>.png and its brightness is printed. Nothing outside the rectangle is used later.
Send me the printed text (or the content of roi.json), not the pictures.
"""
import json
import pathlib
import sys

import cv2

import cam_platform as plat


def main(argv) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = pathlib.Path(argv[1])
    name = argv[2] if len(argv) > 2 else "paper"
    img = plat.imread(path)
    if img is None:
        print(f"cannot read {path}")
        return 1
    h, w = img.shape[:2]
    scale = min(1.0, 1400 / w, 800 / h)                 # fit the screen; coordinates are converted back
    shown = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img
    x, y, rw, rh = cv2.selectROI("Drag a box around the paper, then press ENTER (ESC = cancel)", shown,
                                 showCrosshair=False)
    cv2.destroyAllWindows()
    if rw == 0 or rh == 0:
        print("cancelled: no rectangle chosen")
        return 1
    x, y, rw, rh = (int(round(v / scale)) for v in (x, y, rw, rh))
    x, y = max(0, x), max(0, y)
    rw, rh = min(rw, w - x), min(rh, h - y)
    crop = img[y:y + rh, x:x + rw]
    out = pathlib.Path("probe_out")
    out.mkdir(exist_ok=True)
    plat.imwrite(out / f"roi_{name}.png", crop)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    store = pathlib.Path("roi.json")
    data = json.loads(store.read_text(encoding="utf-8")) if store.exists() else {}
    data[name] = dict(x=x, y=y, w=rw, h=rh, frame_width=w, frame_height=h, source=str(path))
    store.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"area '{name}': x={x} y={y} width={rw} height={rh} (frame {w} x {h})")
    print(f"mean brightness inside: {g.mean():.1f} of 255, saturated pixels: {(g >= 250).mean() * 100:.2f} %")
    print("saved roi.json and probe_out/roi_%s.png" % name)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
