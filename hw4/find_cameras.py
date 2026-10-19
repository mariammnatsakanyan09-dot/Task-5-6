"""Find out which camera number is which (the laptop camera, the phone, the Pano-360).

    python3 find_cameras.py

For every camera number 0..5 that opens, one picture is saved as probe_out/cam<N>.png (a folder next to where you run this).
Open the pictures (double click) and see which one is the phone. Then start the server with  --camera-index <N>.
Close every program that shows a camera first (FaceTime, Zoom, Photo Booth). The pictures are only saved here, nothing is sent.
"""
import pathlib
import subprocess
import sys

import cv2

OUT = pathlib.Path("probe_out")
OUT.mkdir(exist_ok=True)

if sys.platform == "darwin":
    print("Cameras that macOS knows (names, in the order of the system):")
    try:
        r = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True, timeout=30)
        names = [ln.strip().rstrip(":") for ln in r.stdout.splitlines() if ln.strip().endswith(":") and not ln.startswith("Camera")]
        print("  " + "\n  ".join(names) if names else "  (none listed)")
    except Exception as e:  # noqa: BLE001
        print("  (could not list names:", e, ")")
    print()

api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
found = 0
for i in range(6):
    cap = cv2.VideoCapture(i, api)
    if not cap.isOpened():
        print(f"camera {i}: does not open")
        cap.release()
        continue
    frame = None
    for _ in range(25):                          # the first pictures are dark: let the exposure settle
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    cap.release()
    if frame is None:
        print(f"camera {i}: opens, but gives no picture (permission? another program uses it?)")
        continue
    h, w = frame.shape[:2]
    path = OUT / f"cam{i}.png"
    cv2.imwrite(str(path), frame)
    found += 1
    print(f"camera {i}: {w} x {h}, mean brightness {frame.mean():.0f}  ->  {path}")

print()
if found == 0:
    print("No picture from any camera. macOS: System Settings > Privacy & Security > Camera: allow Terminal / VS Code.")
else:
    print("Open the pictures (open probe_out/cam0.png ...) and use the number of the one you want:  --camera-index <N>")
