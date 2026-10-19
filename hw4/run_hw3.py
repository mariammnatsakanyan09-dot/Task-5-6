"""Run the hardware-in-the-loop system on a request, with the real camera.

    export OPENROUTER_API_KEY=sk-or-...
    python3 run_hw3.py --model "<openrouter model id>" --use-case 1
    python3 run_hw3.py --model "<openrouter model id>" --request "How sharp is the picture?"

The instrument is one USB device, the Pano-360: its camera (use cases 1-11) and its loudspeaker + microphone (15-19; the camera is
not even switched on in 15-18).

What happens: the instrument server (mcp_camera_server.py) is started and checked, then the graph runs. Every decision of the
orchestrator, every tool call of the hardware agent and every answer of the instrument is printed as it happens (this is
what the video shows). When the orchestrator sends the turn to `human` the graph PAUSES (LangGraph interrupt), this script
prints the question and waits for your reply, then resumes the same thread.

Writes runs/<timestamp>/ : transcript.md, report.md, evidence.md, events.json, instrument_log.jsonl (the camera calls of this
run), images/ (the saved crops of the measured region), graph.mmd, run.json.
The working folder (default hw3_work/) keeps the references between runs.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cam_platform as plat
import hw3_agents as A
import llm_utils as U
from mcp_bridge import McpBridge, default_env

NEEDS = {"6": "screen_level", "7": "screen_level", "8": "screen_level",
         "9": "brightness", "10": "contrast", "11": "saturation"}      # the control that a use case works with
AUDIO_CASES = ("15", "16", "17", "18", "19")   # use cases that play tones through the device's loudspeaker (see audio_check.py)
AUDIO_ONLY = ("15", "16", "17", "18")          # ... and never need the camera: the camera is not opened at all
BUTTON_CASES = ("17", "18", "19")              # use cases in which the person presses a button of the device (scene_action)
USE_CASES = {
    "1": ("baseline",
          "Set the camera to 1920x1080 (the size the reference is made at), then take a first look at the scene: how bright "
          "and how sharp is it? Store what you see as the reference called base."),
    "2": ("resolution",
          "Which of the three resolutions (640x480, 1280x720, 1920x1080) gives the sharpest picture of the scene? "
          "Measure each one the same way, tell me which to use, and leave the camera at 1920x1080."),
    "3": ("overwrite",
          "The scene is arranged anew. Set the camera to 1920x1080 (the size of the reference), then replace the reference base "
          "with a picture of the scene as it is now."),
    "4": ("changed",
          "Has the scene changed since the reference base was stored? Set the camera to 1920x1080 (the size of the reference), "
          "compare the scene now with it and tell me how much."),
    "5": ("ambiguous",
          "Is the image good?"),
    "6": ("response curve (monitor patch; needs screen_level in controls.json)",
          "How does the camera respond to light? A grey patch on the monitor in front of the camera is the light source. Set the "
          "patch (screen_level) to black (0), to a middle grey (128) and to white (255), measure the mean brightness of the scene "
          "at each level, and report the three readings with the share of saturated pixels. Say whether the camera reading "
          "grows with the level and whether the camera saturates at white. Then put the screen back to its original level."),
    "7": ("tune the light (monitor patch; needs screen_level)",
          "Find the screen level (screen_level) at which the mean brightness of the scene, as the camera reads it, is as close as "
          "possible to 128 grey levels, and within 8 of it. Try a level, measure, and decide the next level from the reading "
          "(halve the interval). Report the level you found with its reading, and put the screen back to its original level."),
    "8": ("saturation point (monitor patch; needs screen_level)",
          "Find the highest screen level (screen_level) at which no more than 1 % of the pixels of the scene are saturated. "
          "Measure the saturated share at a few levels, narrow down, report the level with its readings, and put the screen "
          "back to its original level."),
    "9": ("camera brightness control (only if the camera's own control works; needs brightness)",
          "Use the brightness control of the camera to make the scene 40 grey levels brighter than it is now: first measure the "
          "mean brightness of the scene, then change the control, check, correct, until the mean brightness is 40 higher "
          "(within 5 grey levels) and no more than 1 % of the pixels of the scene are saturated. Report the setting you "
          "found with the readings before and after, and put the camera back to its original settings."),
    "10": ("camera contrast control (needs contrast)",
           "Use the contrast control of the camera to make the contrast of the scene 1.5 times what it is now: measure the "
           "rms_contrast of the scene first, then change the control, check, correct, until rms_contrast is 1.5 times the first "
           "value (within 10 %). Do not use any other control. Report the setting you found with the readings before and "
           "after, and put the camera back to its original settings."),
    "11": ("camera saturation control (needs saturation)",
           "How does the colour of the scene depend on the saturation control of the camera? Measure the colour saturation of "
           "the scene at the lowest, the middle and the highest allowed setting, report the three readings and say whether "
           "the colour grows with the setting. Then put the camera back to its original settings."),
    "15": ("sound: the quietest tone the microphone still hears (adaptive; the device's loudspeaker and microphone)",
           "The loudspeaker and the microphone of the device are a second instrument. Find the quietest level (in dBFS) at which a "
           "1000 Hz tone is still heard by the microphone: play it at a few levels, decide the next level from each result and narrow "
           "the interval around the threshold. Report the lowest level that was heard and the highest level that was not heard."),
    "16": ("sound: frequency response, looked at more closely where it is strongest (adaptive; the next tones are chosen from the readings)",
           "How does the loudspeaker-to-microphone path of the device respond to the frequency? Use one level at which a 1000 Hz tone is "
           "heard clearly (at least 20 dB above the noise). Play 300, 600, 1000, 2000, 3000, 4000 and 6000 Hz at that level and "
           "report the level that the microphone measured at each. Then look at the strongest response more closely: take the "
           "frequency with the strongest reading and play, at the same level, one frequency between it and each of its two neighbours "
           "in that list (only one if it is the first or the last of the list). Report at which frequency the response is strongest "
           "and how sure the readings are about it."),
    "17": ("sound: does the microphone mute button work (the person presses it; pauses of kind scene_action)",
           "Check that the microphone mute button of the device works. Play a 1000 Hz tone at a level that the microphone clearly hears "
           "(at least 20 dB above the noise; this microphone is far from the loudspeaker that plays, so try -30, then -20, then -12) "
           "and keep that reading under the label before. Then ask the person to press the mute button of the device and say done. "
           "Play the same tone again under the label after, compare the two readings, and report by how many dB the tone fell and "
           "whether the microphone is silent. Then ask the person to press the button again to unmute, play the tone once more "
           "(label again) and report whether the sound is back."),
    "18": ("sound: do the volume buttons change the loudness (the person presses them; pauses of kind scene_action)",
           "Check that the volume buttons of the device change the loudness of its loudspeaker. Play a 1000 Hz tone at -30 dBFS under "
           "the label before. Ask the person to press the volume - button of the device three times and say done. Play the same tone "
           "under the label after and compare the two readings: report the change in dB. Then ask the person to press the volume + "
           "button three times, play the same tone under the label again, compare it with before and report whether the loudness "
           "is back to the first reading."),
    "19": ("acceptance test of the whole device: camera + loudspeaker + microphone + buttons (one agent, two instruments)",
           "Run an acceptance test of the whole device and finish with one table. Camera: measure the sharpness_norm of the scene at "
           "640x480, 1280x720 and 1920x1080 (the same way each time) and leave the camera at 1920x1080. Loudspeaker and microphone: "
           "find the quietest level at which a 1000 Hz tone is heard (play a few levels, then halve the interval), then measure a "
           "tone at 500, 2000 and 4000 Hz at a level that is heard at least 20 dB above the noise. Volume buttons: keep a reading "
           "of the 1000 Hz tone at that level under the label before, ask the person to press the volume - button of the device "
           "three times and say done, play the same tone under the label after, compare them and report the change in dB, then ask "
           "the person to press the volume + button three times to put the volume back."),
}


def audio_wanted(use_case) -> bool:
    """The sound tools are offered to the agent for the sound use cases and for a free request; the camera use cases (1-11) never
    get them (a beep that nobody asked for would be a surprise)."""
    return use_case is None or use_case in AUDIO_CASES


def audio_requirement(use_case, loudspeaker: str, microphone: str):
    """Does the chosen loudspeaker / microphone pair suit the use case? Returns None, or the advice to print before any model
    request is spent. The mute button (17) belongs to the microphone of the Pano-360, the volume buttons (18, 19) to its loudspeaker."""
    hint = (os.environ.get("HW3_AUDIO_NAME") or "Pano").lower()
    if use_case == "17" and hint not in microphone.lower():
        return (f"use case 17 checks the mute button of the Pano-360 MICROPHONE, but the microphone chosen now is {microphone!r}. "
                "The Pano-360 microphone cannot hear the Pano-360 loudspeaker (the device cancels its own sound), so let it listen "
                "to the loudspeaker of the computer:\n"
                f'    HW3_AUDIO_OUT="MacBook Pro Speakers" HW3_AUDIO_IN="{hint.title()}" python run_hw3.py --model ... --use-case 17\n'
                "(python audio_check.py --list shows the names; python audio_check.py --pairs finds out which loudspeaker it hears.)")
    if use_case in ("18", "19") and hint not in loudspeaker.lower():
        return (f"use case {use_case} checks the volume buttons of the Pano-360 LOUDSPEAKER, but the loudspeaker chosen now is "
                f"{loudspeaker!r}. Run  python audio_check.py --pairs  again, or choose it: "
                f'HW3_AUDIO_OUT="{hint.title()}" python run_hw3.py ... --use-case {use_case}')
    return None


def main():
    plat.utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", help="what you ask the system (you talk only to the orchestrator)")
    ap.add_argument("--request-file", help="read the request from a text file instead")
    ap.add_argument("--use-case", choices=sorted(USE_CASES), help="; ".join(f"{k} = {v[0]}" for k, v in USE_CASES.items()))
    ap.add_argument("--controls", default="controls.json", help="controls verified by probe_camera.py (optional)")
    ap.add_argument("--provider", choices=["openrouter", "google", "anthropic"], default=None)
    ap.add_argument("--model", default=None,
                    help="model id (env AGENT_MODEL). OpenRouter: one id, or several separated by commas (fallbacks)")
    ap.add_argument("--hardware-model", default=None, help="a different model for the hardware agent (the one that calls tools)")
    ap.add_argument("--work", default="hw3_work", help="folder for crops, references and the instrument log (kept between runs)")
    ap.add_argument("--roi", default="roi.json", help="measured region made by pick_roi.py")
    ap.add_argument("--camera-index", type=int, default=None, help="force the camera number (default: find the Pano-360)")
    ap.add_argument("--fake", action="store_true", help="use the synthetic camera and the synthetic loudspeaker/microphone (tests only; "
                                                        "NOT a measurement)")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--max-decisions", type=int, default=A.MAX_DECISIONS)
    ap.add_argument("--reply", action="append", default=[],
                    help="a scripted reply for the n-th pause (repeatable). Without it you are asked in the terminal.")
    ap.add_argument("--show-images", action="store_true", help="open each saved crop when the camera makes it (for the video)")
    ap.add_argument("--thread-id", default=None)
    args = ap.parse_args()

    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit("Set OPENROUTER_API_KEY (or GOOGLE_API_KEY / ANTHROPIC_API_KEY) first.")
    request = args.request or (Path(args.request_file).read_text(encoding="utf-8") if args.request_file else None)
    if args.use_case:
        request = request or USE_CASES[args.use_case][1]
    if not request:
        sys.exit('Give --use-case 1..19, --request "..." or --request-file path.')
    roi = Path(args.roi)
    if not roi.exists() and not args.fake:
        print(f"note: {roi} not found; a default rectangle in the middle of the main view will be measured "
              "(run pick_roi.py to choose the region).")

    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = (Path(args.runs) / stamp).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = work / "instrument_log.jsonl"
    log_start = log_path.stat().st_size if log_path.exists() else 0

    controls = Path(args.controls)
    env = default_env(str(work), roi_file=str(roi) if roi.exists() else None, camera_index=args.camera_index, fake=args.fake,
                      fake_state=str(work / "fake_state.json") if args.fake else None,
                      controls_file=str(controls) if controls.exists() else None,
                      fake_audio=args.fake, fake_audio_state=str(work / "fake_audio_state.json") if args.fake else None)
    if not audio_wanted(args.use_case):
        env["HW3_AUDIO"] = "off"
        env.pop("HW3_AUDIO_BACKEND", None)
    print(f"working folder: {work}\nrun folder    : {run_dir}")
    print("starting the instrument server (MCP, stdio) ...", flush=True)
    bridge = McpBridge(env=env)
    try:
        bridge.start()
        print("tools offered by the server:", ", ".join(bridge.tool_names()))
        have_audio = "play_tone" in bridge.tool_names()
        if args.use_case in AUDIO_CASES and not have_audio:
            sys.exit(f"use case {args.use_case} needs the loudspeaker and microphone of the device, but the server offers no sound tools. "
                     "Run: python audio_check.py --list   (and python audio_check.py); also pip install sounddevice.")
        camera_needed = args.use_case not in AUDIO_ONLY                 # the sound-only use cases never switch the camera on
        if camera_needed:
            first = json.loads(bridge.call("camera_info"))          # the program checks the camera before any model request is spent
            if not first.get("ok", True) or "error" in first:
                sys.exit("the camera is not ready: " + str(first.get("error")))
            print(f"instrument: {first['instrument']}\ninterface : {first['interface']}\n"
                  f"frame     : {first['frame_size'][0]}x{first['frame_size'][1]}, regions: {', '.join(first['regions'])}"
                  f" (measured region from {first['measured_region_source']})")
            ctl = first.get("controls")
            print("controls  : " + (", ".join(f"{n} {c['min']:g}..{c['max']:g}" for n, c in ctl["verified"].items())
                                    + " (verified on this camera)" if isinstance(ctl, dict) else str(ctl)))
        else:
            first, ctl = {}, None
            print("camera    : not used in this use case (it is not switched on)")
        if have_audio:
            ai = json.loads(bridge.call("audio_info"))
            if ai.get("ok"):
                ch = ai.get("chosen_by") or {}
                by = lambda k: f" [chosen by {ch[k]}]" if ch.get(k) not in (None, "default") else ""
                print(f"loudspeaker: {ai['loudspeaker']} ({ai['sample_rate_out']} Hz){by('loudspeaker')}\nmicrophone : {ai['microphone']} "
                      f"({ai['sample_rate_in']} Hz){by('microphone')}; levels {ai['level_db_range'][0]:g}..{ai['level_db_range'][1]:g} dBFS, "
                      f"frequencies {ai['freq_hz_range'][0]:g}..{ai['freq_hz_range'][1]:g} Hz")
                advice = audio_requirement(args.use_case, ai["loudspeaker"], ai["microphone"])
                if advice:
                    sys.exit("\n" + advice)
            else:
                print("sound     : " + str(ai.get("error")))
        need = NEEDS.get(args.use_case or "")
        have = ctl["verified"] if isinstance(ctl, dict) else {}
        if need and need not in have:
            print(f"WARNING: use case {args.use_case} needs the control '{need}' in controls.json (run probe_camera.py first); "
                  f"this camera offers: {', '.join(have) or 'no controls'}.")
        if args.use_case in AUDIO_CASES and not args.fake:
            print("Sound use case: keep the room quiet (no talking, no music) and the volume on the device about in the middle; "
                  "you will hear short beeps, never loud. Check first with: python audio_check.py"
                  + ("\nIn this use case the system will ask you to press a button of the device: do it, then answer 'done'."
                     if args.use_case in BUTTON_CASES else ""))
        if not first.get("layout_verified", True):
            print("WARNING: the picture of this camera does not show the two-part layout of the Pano-360.")
        if args.fake:
            print("WARNING: --fake: this is the synthetic camera, not a measurement.")

        if args.show_images:
            def show_image(ev):
                r = ev["result"]
                paths = [work / p for p in r.get("saved_crops", [])]
                if ev["tool"] == "save_reference" and ev["ok"]:
                    paths.append(work / r["file"])
                for p in paths:
                    plat.show_file(p)
            A.HOOKS["after_tool"] = show_image

        models = U.default_models(args.model, args.provider, args.hardware_model)
        graph = A.build_graph(models, bridge, work, max_decisions=args.max_decisions,
                              curve=A.CURVE_PLAN if args.use_case == "16" else None)
        (run_dir / "graph.mmd").write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
        thread_id = args.thread_id or "hw3-" + stamp
        task = A.build_task(request)
        print(f"thread: {thread_id}\n\nREQUEST: {request}\n")

        scripted = list(args.reply)

        def ask(payload: dict) -> str:
            print("\n" + "=" * 78)
            print(f"GRAPH PAUSED ({payload['kind']}) - waiting for the human")
            print("=" * 78)
            print(payload["question"])
            if payload.get("step"):
                print(f"\nstep that would run: {payload['step']['role']}\n  {payload['step']['instruction']}")
            print(f"\n(answer with: {payload['answer_with']})")
            if scripted:
                reply = scripted.pop(0)
                print(f"your reply> {reply}   [scripted]")
            else:
                try:
                    reply = input("your reply> ")
                except EOFError:
                    sys.exit("\nNo terminal to read the reply from. The graph is paused; pass --reply \"...\" for each pause.")
            print("=" * 78 + "\n")
            return reply

        def show(node: str, upd: dict):
            for t in upd.get("dialogue", []):
                text = t["text"].strip()
                if t["role"] == "hardware":
                    text = upd.get("hw_report") or "(no closing words)"
                lines = [ln for ln in text.splitlines() if ln.strip()]
                head = lines[0][:150] if lines else ""
                if t["role"] == "orchestrator":
                    head = " | ".join(ln[:140] for ln in lines[:3])
                print(f"[{t['role']:12s}] {head}")

        final = A.drive(graph, task, ask, thread_id=thread_id, on_update=show)
    finally:
        try:                                    # the camera keeps its settings until they are changed back: do it in any case
            if "reset_controls" in bridge.tool_names():
                back = json.loads(bridge.call("reset_controls", {})).get("restored")
                if back:
                    print(f"controls put back to their original values: {back}")
        except Exception:  # noqa: BLE001
            pass
        bridge.close()

    events = final.get("events", [])
    report = A.final_report_md(final)
    (run_dir / "transcript.md").write_text(A.transcript_md(final), encoding="utf-8")
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    (run_dir / "evidence.md").write_text(A.evidence_md(final["dialogue"]), encoding="utf-8")
    (run_dir / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    (run_dir / "state.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
    if log_path.exists():                                       # the camera calls of this run only
        with log_path.open("rb") as fh:
            fh.seek(log_start)
            (run_dir / "instrument_log.jsonl").write_bytes(fh.read())
    images = run_dir / "images"
    for ev in events:
        files = list(ev["result"].get("saved_crops", [])) + ([ev["result"]["file"]] if ev["tool"] == "save_reference" and ev["ok"] else [])
        for f in files:
            src = work / f
            if src.exists():
                images.mkdir(exist_ok=True)
                shutil.copy2(src, images / f"{ev['n']:02d}_{src.name}")
    (run_dir / "run.json").write_text(json.dumps(dict(
        request=request, model=args.model, hardware_model=args.hardware_model, fake_camera=args.fake, thread=thread_id,
        decisions=final.get("decisions"), camera_calls=final.get("n_calls"), readings=len(A.readings_of(events)),
        refused=final.get("refused"), stop_reason=final.get("stop_reason"), model_requests=U.STATS["model_calls"]), indent=2), encoding="utf-8")

    print("\n" + "=" * 78 + "\n" + report + "\n" + "=" * 78)
    print("\n=== summary ===")
    print("turn order    :", " -> ".join(t["role"] for t in final["dialogue"][1:]))
    print("decisions     :", final.get("decisions"), "  refused hard decisions:", final.get("refused"))
    print("instrument calls:", final.get("n_calls"), "  readings:", len(A.readings_of(events)))
    print("stop reason   :", final.get("stop_reason"))
    print("model requests:", U.STATS["model_calls"], "(OpenRouter free models: 20 per minute, 50 per day)")
    print("files         :", run_dir, "  (report.md, evidence.md, images/ ...)")
    print("Look at images/ before you publish anything: only the measured region is saved, but check that no person is in it.")


if __name__ == "__main__":
    main()
