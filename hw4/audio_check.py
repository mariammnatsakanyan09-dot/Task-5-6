"""Check, without any model, that the loudspeaker of the Pano-360 can be heard by a microphone.

    python audio_check.py --list        # the sound devices of this computer (look for "Pano-360")
    python audio_check.py               # speaker -> microphone of the Pano-360: quiet-to-louder tones, then some frequencies
    python audio_check.py --pairs       # try every speaker / microphone combination that includes the Pano-360, say which hears,
                                        # and save the choice in audio.json (the project reads it; delete the file to undo)
    python audio_check.py --out "MacBook" --in "Pano"      # choose the devices by a part of the name (or by number)

You will hear short beeps (about 1.5 s each, a few of them, starting very quiet and getting louder). Set the volume on the device to
about the middle, keep the room quiet (no talking, no music), and do not put your hand on the speaker. The microphone listens to the
room for a moment, so do not say anything you do not want measured: nothing is recorded to disk, only numbers are computed.
On a Mac the first run asks for permission to use the microphone: click Allow (for Terminal or Visual Studio Code).
"""
import argparse
import json
import os
import sys

import audio_connector as ac
import cam_platform as plat

LEVELS = (-80, -70, -60, -50, -40, -30, -20)
FREQS = (300, 600, 1000, 2000, 3000, 4000, 6000)


def line(r: dict, what: str) -> str:
    if not r.get("ok"):
        return f"ERROR   {what}: {r.get('error')}"
    state = "HEARD    " if r["heard"] else ("SILENT   " if r["silent_input"] else "not heard")
    extra = (f" [NO VERDICT: the room changed while the tone played (separation {r['separation']:.0%}); repeat]"
             if r.get("interference") else "")
    return (f"{state} {what}: microphone {r['tone_db']:7.1f} dB, noise {r['noise_db']:7.1f} dB, "
            f"{r['snr_db']:5.1f} dB above the noise" + (f", delay {r['delay_s']:.2f} s" if r["heard"] else "") + (" [clipped]" if r.get("clipped") else "") + extra)


def sweep_levels(conn: "ac.AudioConnector", freq: float = 1000, levels=LEVELS, verbose: bool = True) -> tuple:
    """Play `freq` at increasing levels until it is clearly heard. Returns (best result, lowest heard level, highest level not heard)."""
    best, lowest, highest_not = None, None, None
    for lv in levels:
        r = conn.play_tone(freq, lv)
        if verbose:
            print(line(r, f"{freq:g} Hz at {lv:g} dBFS"), flush=True)
        if not r.get("ok"):
            return r, None, None
        if r["heard"]:
            lowest = lv if lowest is None else lowest
            best = r
            if r["snr_db"] >= 25:
                break                                            # loud enough to be sure; no need to go louder
        else:
            highest_not = lv
            best = best or r
    return best, lowest, highest_not


def run_default(args) -> int:
    try:
        conn = ac.open_audio(args.out, args.inp)
    except ac.AudioError as e:
        print(f"cannot open the sound devices: {e}")
        return 1
    b = conn.b
    src = lambda k: f" [chosen by {b.source[k]}]" if b.source.get(k) not in (None, "default") else ""
    print(f"loudspeaker: {b.out_name} ({b.sr_out} Hz){src('out')}\nmicrophone : {b.in_name} ({b.sr_in} Hz){src('in')}; "
          f"frequencies up to {conn.max_freq:g} Hz, levels {ac.MIN_DB:g} .. {ac.max_db():g} dBFS\n")
    print("1. the room, nothing played ...", flush=True)
    n = conn.listen_noise(1.5)
    if not n.get("ok"):
        print(n.get("error"))
        return 1
    print(f"   noise {n['noise_rms_db']:.1f} dBFS (strongest component {n['strongest_hz']:g} Hz); {n['note']}")
    print("\n2. a 1000 Hz tone, quiet to louder (you will hear beeps) ...", flush=True)
    best, lowest, not_heard = sweep_levels(conn)
    if best is None or not best.get("ok"):
        return 1
    if best["silent_input"]:
        print("\nRESULT: the microphone delivers pure silence. Is the microphone muted on the device (press its mute button), or "
              "is the permission denied? System Settings > Privacy & Security > Microphone: allow Terminal / Visual Studio Code.")
        return 2
    if lowest is None:
        print(f"\nRESULT: nothing was heard up to {LEVELS[-1]} dBFS. Possible reasons: the volume on the device is at the minimum "
              "(press + on it), the loudspeaker is in Bluetooth mode, or the device cancels its own sound out of its microphone "
              "(echo cancellation). Run:  python audio_check.py --pairs")
        return 2
    print(f"\n3. the frequencies at {best['level_db']:g} dBFS ...", flush=True)
    lv = best["level_db"]
    curve = []
    for f in FREQS:
        if f > conn.max_freq:
            continue
        r = conn.play_tone(f, lv)
        curve.append((f, r))
        print(line(r, f"{f:g} Hz"), flush=True)
    ok_n = sum(1 for _, r in curve if r.get("heard"))
    top = max((r for _, r in curve if r.get("ok")), key=lambda r: r["tone_db"], default=None)
    print(f"\nRESULT: the microphone hears the loudspeaker. At 1000 Hz the quietest level heard here is {lowest:g} dBFS"
          + (f" (not heard at {not_heard:g})" if not_heard is not None else "")
          + f"; {ok_n} of {len(curve)} frequencies were heard at {lv:g} dBFS"
          + (f", the strongest at {top['freq_hz']:g} Hz" if top else "") + ".")
    print("You can now run the whole system with a model: python run_hw3.py --model ... --use-case 15")
    return 0


def run_pairs(args) -> int:
    try:
        import sounddevice as sd
        devs = sd.query_devices()
    except Exception as e:  # noqa: BLE001
        print(f"sounddevice is not usable: {e}")
        return 1
    hint = (os.environ.get("HW3_AUDIO_NAME") or "Pano").lower()
    real = lambda d: not any(v in d["name"].lower() for v in ac.VIRTUAL)       # skip virtual devices (Teams, Zoom ...)
    outs = [i for i, d in enumerate(devs) if d["max_output_channels"] > 0 and real(d)]
    ins = [i for i, d in enumerate(devs) if d["max_input_channels"] > 0 and real(d)]
    pairs = [(o, i) for o in outs for i in ins if hint in devs[o]["name"].lower() or hint in devs[i]["name"].lower()]
    if not pairs:
        print("no pair includes a device whose name contains " + repr(hint) + "; devices:\n" + ac.device_table(devs))
        return 1
    print("Each pair: a 1000 Hz tone at -30 dBFS (and at -20 if it is not heard). You will hear beeps from the loudspeaker named first.\n")
    results = []
    for o, i in pairs:
        what = f"{devs[o]['name']} (speaker {o}) -> {devs[i]['name']} (microphone {i})"
        try:
            conn = ac.open_audio(str(o), str(i))
            r = conn.play_tone(1000, -30)
            if r.get("ok") and not r["heard"] and not r["silent_input"]:
                r = conn.play_tone(1000, -20)
        except ac.AudioError as e:
            print(f"ERROR   {what}: {e}")
            continue
        results.append((o, i, r))
        print(line(r, what), flush=True)
    good = [(o, i, r) for o, i, r in results if r.get("ok") and r["heard"]]
    if not good:
        print("\nRESULT: no pair was heard. Check the volume on the device, that nothing else plays or records, and the microphone "
              "permission (System Settings > Privacy & Security > Microphone).")
        return 2
    both = [x for x in good if hint in devs[x[0]]["name"].lower() and hint in devs[x[1]]["name"].lower()]
    o, i, r = both[0] if both else max(good, key=lambda x: x[2]["snr_db"])
    print(f"\nRESULT: use the speaker {devs[o]['name']!r} and the microphone {devs[i]['name']!r}.")
    path = ac.pair_file()
    if both:
        if path.exists():
            path.unlink()
            print(f"(the old {path} was deleted: the default pair, the Pano-360 for both, is the right one)")
    else:
        path.write_text(json.dumps({"out": devs[o]["name"], "in": devs[i]["name"],
                                    "note": "written by audio_check.py --pairs; delete this file to go back to the default"}, indent=2),
                        encoding="utf-8")
        print("The Pano-360 microphone did not hear the Pano-360 loudspeaker (the device cancels its own sound out of its microphone). "
              "The experiment still works: the Pano-360 loudspeaker is commanded and another microphone measures it.\n"
              f"The choice is saved in {path} and the system uses it automatically (python run_hw3.py ... --use-case 15, 16, 18, 19).")
    pano_mic = [x for x in good if hint in devs[x[1]]["name"].lower() and hint not in devs[x[0]]["name"].lower()]
    if pano_mic:
        po, pi, _ = pano_mic[0]
        print("\nUse case 17 checks the mute button of the Pano-360 MICROPHONE, so the Pano-360 must be the microphone. Its microphone hears "
              f"{devs[po]['name']!r}, so run that one with:\n"
              f'    HW3_AUDIO_OUT="{devs[po]["name"]}" HW3_AUDIO_IN="{devs[pi]["name"]}" python run_hw3.py --model ... --use-case 17')
    return 0


def main() -> int:
    plat.utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list the sound devices and stop")
    ap.add_argument("--pairs", action="store_true", help="try every speaker/microphone pair that includes the Pano-360")
    ap.add_argument("--out", default=None, help="loudspeaker: a part of its name, or its number")
    ap.add_argument("--in", dest="inp", default=None, help="microphone: a part of its name, or its number")
    args = ap.parse_args()
    if ac.backend_kind() != "fake":
        try:
            import sounddevice as sd
        except Exception as e:  # noqa: BLE001
            print(f"the python package sounddevice is not usable ({type(e).__name__}: {e}). Run: pip install sounddevice")
            return 1
        if args.list:
            print(ac.device_table(sd.query_devices()))
            return 0
    elif args.list:
        print("(synthetic sound devices)")
        return 0
    try:
        return run_pairs(args) if args.pairs else run_default(args)
    except ac.AudioError as e:
        print(f"\nERROR: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nstopped")
        return 130


if __name__ == "__main__":
    sys.exit(main())
