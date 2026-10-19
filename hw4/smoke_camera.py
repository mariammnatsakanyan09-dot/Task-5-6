"""Check the REAL camera through the same MCP tools the agent will use - no model, no API key, no request quota.

    python smoke_camera.py                  # the checks   (macOS / Linux: python3)
    python smoke_camera.py --cover-test     # also asks you to cover the lens once (tests change detection)

It prints a short table; send me the printed text (the pictures stay on your Mac). Only crops of the measured region are saved,
in smoke_work/.
"""
import argparse
import json
import sys
from pathlib import Path

import cam_platform as plat
from mcp_bridge import McpBridge, default_env


def call(b, tool, **args):
    return json.loads(b.call(tool, args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="smoke_work")
    ap.add_argument("--roi", default="roi.json")
    ap.add_argument("--camera-index", type=int, default=None)
    ap.add_argument("--fake", action="store_true", help="synthetic camera (to try this script without the camera)")
    ap.add_argument("--controls", default="controls.json")
    ap.add_argument("--cover-test", action="store_true")
    args = ap.parse_args()
    plat.utf8_console()

    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    roi = Path(args.roi)
    ctl = Path(args.controls)
    env = default_env(str(work), roi_file=str(roi) if roi.exists() else None, camera_index=args.camera_index, fake=args.fake,
                      controls_file=str(ctl) if ctl.exists() else None)
    b = McpBridge(env=env)
    problems = []
    try:
        b.start()
        print("tools:", ", ".join(b.tool_names()))
        info = call(b, "camera_info")
        if not info.get("ok", True) or "error" in info:
            print("CAMERA NOT READY:", info.get("error"))
            return 1
        print(f"\n1. camera_info\n   {info['instrument']}\n   {info['interface']}")
        print(f"   frame {info['frame_size']}, layout {info['layout']}, picture is colour: {info['colour']}")
        print(f"   regions (x, y, w, h): {info['regions']}   [from {info['measured_region_source']}]")
        if not info["layout"]["seam_found"]:
            problems.append("the two-part layout was not found (is the lens covered? is this the right camera?)")
        if not info["layout_verified"]:
            problems.append("camera was chosen by number and its picture has no two-part layout")

        print("\n2. repeatability: the same reading three times, same scene (do not move anything)")
        rows = [call(b, "measure_region", region="scene", n_frames=5) for _ in range(3)]
        for r in rows:
            print(f"   brightness {r['mean_brightness']:7.2f}  noise {r['brightness_noise_std']:6.3f}  "
                  f"sharpness_norm {r['sharpness_norm']:8.1f}  saturated {r['saturated_fraction'] * 100:5.2f} %")
        spread = max(r["mean_brightness"] for r in rows) - min(r["mean_brightness"] for r in rows)
        print(f"   spread of the three brightness values: {spread:.2f} grey levels")
        if rows[0]["saturated_fraction"] > 0.2:
            problems.append("more than 20 % of the pixels of the region are saturated: the readings will not be informative")

        print("\n3. resolutions (set_resolution, then the same reading)")
        print("   asked      got        region size   brightness  sharpness_norm  sharpness   seam")
        for w, h in [(640, 480), (1280, 720), (1920, 1080)]:
            try:
                s = call(b, "set_resolution", width=w, height=h)
            except Exception as e:  # noqa: BLE001
                print(f"   {w}x{h}: {e}")
                continue
            if not s.get("ok"):
                print(f"   {w}x{h:<5}  NOT ACCEPTED: {s.get('error') or s.get('note')}")
                problems.append(f"resolution {w}x{h}: {s.get('error') or s.get('note')}")
                continue
            r = call(b, "measure_region", region="scene", n_frames=5)
            c = call(b, "capture", label=f"{w}x{h}", n_frames=1)
            print(f"   {w}x{h:<5}  {s['accepted'][0]}x{s['accepted'][1]:<5}  {r['region_size'][0]}x{r['region_size'][1]:<8}   "
                  f"{r['mean_brightness']:9.2f}  {r['sharpness_norm']:14.1f}  {r['sharpness']:9.1f}   "
                  f"{'found' if c['seam_found'] else 'NOT found'}")
            if not c["seam_found"]:
                problems.append(f"at {w}x{h} the two-part layout was not found: the measured region may not be the same scene")
        call(b, "set_resolution", width=1920, height=1080)

        if "set_control" in b.tool_names():
            print("\n3b. controls verified by probe_camera.py, through the MCP tool set_control: the same readings at the lowest, "
                  "middle and highest value (scene must stay still)")
            print("   control      asked -> read back   brightness  contrast  colour sat.  saturated")
            key = {"screen_level": "mean_brightness", "brightness": "mean_brightness", "exposure": "mean_brightness",
                   "gain": "mean_brightness",
                   "contrast": "rms_contrast", "saturation": "colour_saturation"}
            need = {"mean_brightness": ("abs", 6.0), "rms_contrast": ("ratio", 1.15), "colour_saturation": ("ratio", 1.3)}
            for name, c in info["controls"]["verified"].items():
                seen = []
                for frac in (0.0, 0.5, 1.0):
                    v = c["min"] + frac * (c["max"] - c["min"])
                    s = call(b, "set_control", name=name, value=v)
                    r = call(b, "measure_region", region="scene", n_frames=5)
                    seen.append(r.get(key.get(name, "mean_brightness")))
                    print(f"   {name:11s} {v:8.2f} -> {str(s.get('read_back')):>8}    {r['mean_brightness']:9.2f}  {r.get('rms_contrast', 0):8.2f}  "
                          f"{r.get('colour_saturation', 0):11.2f}  {r['saturated_fraction'] * 100:7.2f} %"
                          + ("" if s.get("ok") else f"   [{s.get('note') or s.get('error')}]"))
                    if not s.get("ok"):
                        problems.append(f"set_control {name}={v:g}: {s.get('note') or s.get('error')}")
                rs = call(b, "reset_controls")
                print(f"   reset_controls: {rs.get('restored')}")
                back = call(b, "measure_region", region="scene", n_frames=5)
                print(f"   after reset: brightness {back['mean_brightness']:.2f}, contrast {back.get('rms_contrast')}, "
                      f"colour saturation {back.get('colour_saturation')}")
                mode, thr = need.get(key.get(name, "mean_brightness"), ("abs", 6.0))
                vals = [x for x in seen if x is not None]
                if vals:
                    moved = (max(vals) - min(vals)) if mode == "abs" else max(vals) / max(min(vals), 1.0)
                    if moved < thr:
                        problems.append(f"{name}: its reading ({key.get(name)}) moved only {moved:.2f} between the lowest and the highest "
                                        f"value (needed {thr:g}); was the scene still, and does it have colour (for saturation)?")
        else:
            print("\n3b. no verified controls (controls.json missing or empty): set_control is not offered")

        print("\n4. reference and comparison")
        sv = call(b, "save_reference", name="smoke", overwrite=True)
        print("   save_reference:", "ok" if sv.get("ok") else sv.get("error"))
        cmp1 = call(b, "compare_to_reference", name="smoke")
        print(f"   compare, scene unchanged: mean abs difference {cmp1.get('mean_abs_difference')}, brightness change "
              f"{cmp1.get('brightness_change')}, correlation {cmp1.get('correlation')}")
        if args.cover_test:
            input("\n   Cover the camera lens with your hand (or turn the camera towards a wall), then press Enter ... ")
            cmp2 = call(b, "compare_to_reference", name="smoke")
            print(f"   compare, lens covered   : mean abs difference {cmp2.get('mean_abs_difference')}, brightness change "
                  f"{cmp2.get('brightness_change')}, correlation {cmp2.get('correlation')}")
            input("   Uncover the lens and press Enter ... ")
        again = call(b, "save_reference", name="smoke")
        print("   save_reference without overwrite when it exists:", "REFUSED (as designed)" if not again.get("ok") else "NOT refused!")
        if again.get("ok"):
            problems.append("an existing reference was overwritten without overwrite=true")
        print(f"\ncrops written to {work}/crops (only the measured region); log: {work}/instrument_log.jsonl")
    finally:
        b.close()

    print("\nPROBLEMS:" if problems else "\nNo problems found.")
    for p in problems:
        print(" -", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
