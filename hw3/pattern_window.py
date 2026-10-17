"""The light source of the test bench: a window that shows ONE uniform grey patch (level 0 = black ... 255 = white).
The camera looks at the monitor, so the level of this patch is a control that the agent can command (screen_level).

    python pattern_window.py --level 128          # by hand: put the camera in front of the screen, aim it, look at the picture
                                                  #   keys: + / - change the level by 16, q or Esc closes the window
    python pattern_window.py --state S --ack A    # controlled by the MCP server (screen_patch.py); not for typing by hand

    --geom x,y,w,h   position and size of the window in screen points (default 80,80,1000,650)

Nothing but the grey patch is drawn in the window, so the camera sees a uniform area. The name of the window is its only text.
HW3_VIEWER_HEADLESS=1 runs the same loop without opening a window (used by the tests on a machine with no screen).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

TITLE = "Pano-360 test patch - leave open"


def parse_geom(text: str) -> tuple[int, int, int, int]:
    x, y, w, h = (int(float(v)) for v in text.split(","))
    return x, y, max(200, w), max(150, h)


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write(path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)                       # the reader never sees a half-written file


class Window:
    def __init__(self, geom, headless: bool):
        self.geom, self.headless, self.level, self.opened = geom, headless, None, False
        self.cv2 = None
        if not headless:
            import cv2
            self.cv2 = cv2

    def show(self, level: int) -> None:
        self.level = int(max(0, min(255, level)))
        if self.headless:
            return
        cv2, (x, y, w, h) = self.cv2, self.geom
        import numpy as np
        if not self.opened:
            cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(TITLE, w, h)
            cv2.moveWindow(TITLE, x, y)
            self.opened = True
        cv2.imshow(TITLE, np.full((h, w, 3), self.level, np.uint8))
        for _ in range(2):                      # let the window system draw it before anyone is told that it is shown
            cv2.waitKey(1)

    def key(self, wait_ms: int) -> int:
        if self.headless:
            time.sleep(wait_ms / 1000.0)
            return -1
        k = self.cv2.waitKey(wait_ms)
        if self.opened and self.cv2.getWindowProperty(TITLE, self.cv2.WND_PROP_VISIBLE) < 1:
            return 27                           # the person closed the window
        return k

    def close(self) -> None:
        if self.cv2 is not None and self.opened:
            self.cv2.destroyAllWindows()
            for _ in range(3):
                self.cv2.waitKey(1)


def run_controlled(state, ack, geom, headless: bool) -> int:
    parent = os.getppid()
    win = Window(geom, headless)
    seq_done = -1
    win.show(128)
    _write(ack, dict(seq=-1, level=win.level, ready=True))
    while True:
        st = _read(state)
        if st and st.get("quit"):
            break
        if st and st.get("seq") != seq_done and "level" in st:
            win.show(st["level"])
            seq_done = st["seq"]
            _write(ack, dict(seq=seq_done, level=win.level, time=time.time()))
        if win.key(25) in (27, ord("q")):
            _write(ack, dict(seq=seq_done, level=win.level, closed=True))
            break
        if os.getppid() != parent:              # the server is gone: do not stay on the screen
            break
    win.close()
    return 0


def run_by_hand(level: int, geom) -> int:
    win = Window(geom, headless=os.environ.get("HW3_VIEWER_HEADLESS") == "1")
    win.show(level)
    print(f"showing grey level {win.level}; keys: + / - change it by 16, q or Esc closes the window")
    while True:
        k = win.key(50)
        if k in (27, ord("q")):
            break
        if k in (ord("+"), ord("=")):
            win.show(win.level + 16)
            print("level", win.level)
        elif k in (ord("-"), ord("_")):
            win.show(win.level - 16)
            print("level", win.level)
        if win.headless:
            break
    win.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=128)
    ap.add_argument("--geom", default=os.environ.get("HW3_SCREEN_GEOM", "80,80,1000,650"))
    ap.add_argument("--state")
    ap.add_argument("--ack")
    a = ap.parse_args()
    geom = parse_geom(a.geom)
    if a.state and a.ack:
        return run_controlled(a.state, a.ack, geom, os.environ.get("HW3_VIEWER_HEADLESS") == "1")
    return run_by_hand(a.level, geom)


if __name__ == "__main__":
    sys.exit(main())
