"""Take ONE picture from the Pano-360 and save it as probe_out/snap.png (to check the scene before marking the region).

    python snap.py          # finds the Pano-360 itself: the camera whose picture has the two-part layout
    python snap.py 1        # or force a camera number (macOS may number the cameras differently from one run to the next)

Close every program that shows the camera first. Then open probe_out/snap.png (a double click is enough).
(macOS / Linux: python3 snap.py)
"""
import pathlib
import sys

import cv2

import cam_platform as plat
import camera_connector as cc


def main(argv) -> int:
    out = pathlib.Path("probe_out")
    out.mkdir(exist_ok=True)
    frame, idx, how = None, None, ""
    if len(argv) > 1:
        idx = int(argv[1])
        cap = plat.open_capture(idx)
        if not cap.isOpened():
            print(f"camera {idx} could not be opened: switch the camera on, close other programs that use it.\n"
                  + plat.permission_hint())
            return 1
        for _ in range(30):                      # the first frames are dark: let the automatic exposure settle
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        how = "number given by you"
        if frame is not None and cc.find_seam(cc.to_gray(frame)) is None:
            print(f"WARNING: camera {idx} does not show the two-part picture of the Pano-360 (main view + panorama strip). "
                  f"Run without a number to let the program find the Pano-360.")
    else:
        try:
            cam = cc.open_camera(out)
        except cc.CameraError as e:
            print(e)
            return 1
        try:
            frame = cam.grab(1, n_discard=30)[-1]
            idx = cam.index
        except cc.CameraError as e:
            print(e)
            return 1
        finally:
            cam.close()
        how = "found by its two-part picture"
    if frame is None:
        print("no frame received")
        return 1
    path = out / "snap.png"
    plat.imwrite(path, frame)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"saved {path}: {frame.shape[1]} x {frame.shape[0]}, mean brightness {g.mean():.1f} of 255 (camera {idx}, {how})")
    print("now open probe_out/snap.png and look at it")
    return 0


if __name__ == "__main__":
    plat.utf8_console()
    sys.exit(main(sys.argv))
