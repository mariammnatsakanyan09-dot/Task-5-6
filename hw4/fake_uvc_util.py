"""A stand-in for the macOS tool uvc-util, for the offline tests only (it is NOT used with the real camera).

It understands the options that uvc_controls.py uses and prints in the format of the uvc-util README:
    -V 0x1bcf:0x2c99 | -V 7119:11417 | -I 0        select the camera (anything else: error, exit code 1)
    -d / --list-devices      -c / --list-controls    -S name    -o name    -g name    -s name=value
The control values live in the JSON file named by HW3_FAKE_UVC_STATE (default fake_uvc_state.json in the current folder),
so that the synthetic camera (camera_connector.FakeBackend) can draw its picture with them.

Tests make it callable with a small launcher script whose first line is  #!<python>  (see make_launcher).
"""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#            min   max  step default
CONTROLS = {"brightness": (-64, 64, 1, 0), "saturation": (0, 100, 1, 50), "contrast": (0, 100, 1, 50), "hue": (-40, 40, 1, 0)}
IDS = {"0x1bcf:0x2c99", "7119:11417", "0"}


def state_path() -> Path:
    return Path(os.environ.get("HW3_FAKE_UVC_STATE", "fake_uvc_state.json"))


def load() -> dict:
    p = state_path()
    st = {k: float(v[3]) for k, v in CONTROLS.items()}
    if p.exists():
        st.update(json.loads(p.read_text(encoding="utf-8")))
    return st


def fail(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    target, actions = None, []
    i = 0
    while i < len(a):
        t = a[i]
        if t in ("-V", "-I", "-L", "-N") and i + 1 < len(a):
            target = (t, a[i + 1].lower())
            i += 2
        elif t in ("-S", "-o", "-g", "-s") and i + 1 < len(a):
            actions.append((t, a[i + 1]))
            i += 2
        elif t in ("-d", "--list-devices", "-c", "--list-controls"):
            actions.append((t, None))
            i += 1
        elif t == "-0":
            target = None
            i += 1
        else:
            fail(f"unknown option {t}")
    for act, arg in actions:
        if act in ("-d", "--list-devices"):
            print("Found 1 UVC device:\n  Index 0\n    Vendor id: 0x1bcf\n    Product id: 0x2c99\n    Name: Pano-360")
            continue
        if target is None or target[1] not in IDS:
            fail("no UVC device selected, or no such device")
        st = load()
        if act in ("-c", "--list-controls"):
            print("Controls implemented by the device:")
            for k in CONTROLS:
                print(f"  {k}")
        elif act == "-S":
            if arg not in CONTROLS:
                fail(f"unknown control {arg}")
            lo, hi, step, df = CONTROLS[arg]
            cur = int(st[arg]) if float(st[arg]).is_integer() else st[arg]
            print(f"{arg} {{\n  type-description: {{\n    single value, signed 16-bit integer\n  }},\n  minimum: {lo}\n"
                  f"  maximum: {hi}\n  step-size: {step}\n  default-value: {df}\n  current-value: {cur}\n}}")
        elif act in ("-o", "-g"):
            if arg not in CONTROLS:
                fail(f"unknown control {arg}")
            v = st[arg]
            v = int(v) if float(v).is_integer() else v
            print(v if act == "-o" else f"{arg} = {v}")
        elif act == "-s":
            name, _, val = arg.partition("=")
            if name not in CONTROLS or not val:
                fail(f"bad setting {arg}")
            try:
                v = float(val)
            except ValueError:
                fail(f"bad value {val}")
            lo, hi, step, _ = CONTROLS[name]
            if not (lo <= v <= hi) or not float(v).is_integer():
                fail(f"value {val} not accepted for {name} (range {lo}..{hi}, whole numbers)")
            st[name] = v
            state_path().write_text(json.dumps(st), encoding="utf-8")
    return 0


def make_launcher(path) -> str:
    """Write an executable file that runs this fake tool; returns its path (for HW3_UVC_UTIL)."""
    p = Path(path)
    p.write_text(f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(HERE)!r})\nimport fake_uvc_util\n"
                 f"sys.exit(fake_uvc_util.main())\n", encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


if __name__ == "__main__":
    sys.exit(main())
