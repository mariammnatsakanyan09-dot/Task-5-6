"""Offline checks of the SOUND part of Homework 3 (no API calls, no sound card, no camera, nothing is played).

    python3 test_audio_offline.py

S1  analysis        a tone in a recording is measured at its own frequency; noise, silence, clipping, a wrong frequency, a louder
                    hum at another frequency: the verdict `heard` is the program's
S2  connector       the synthetic loudspeaker-to-microphone path: a threshold, a resonance, limits that are enforced, a muted
                    microphone, a volume change, labelled readings and the comparison computed by the program
S3  real backend    the sounddevice code (device choice, the thread that records while the tone plays, error messages) on a
                    stand-in for the python package sounddevice; audio_check.py: which pair of devices to use
S4  MCP server      the sound tools through the bridge, bad arguments answered, every call logged, no sound written to disk
S5  agent graph     the quietest-tone search (adaptive), a button check with the person in the loop, no camera needed, nothing stored
S6  runner          use cases 15-19
S7  frequency curve the numbers of the real run 16, a scripted model that plays one tone per reply and wants to finish too early,
                    the guard that must not trap a run
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

os.environ["HW3_CAMERA_BACKEND"] = "fake"
os.environ["HW3_AUDIO"] = "off"             # each test switches on what it needs; a real loudspeaker is never used here

import numpy as np  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

import audio_connector as ac  # noqa: E402
import hw3_agents as A  # noqa: E402
from hw3_agents import Decision  # noqa: E402
from mcp_bridge import McpBridge, default_env  # noqa: E402
import test_hw3_offline as T  # noqa: E402     (helpers: Env, TC, D, roles, make_roi ...)

HERE = Path(__file__).resolve().parent
PASSED = []


def ok(name):
    PASSED.append(name)
    print("ok ", name)


def synth(sr, f, level_db, noise_db, delay=0.12, seconds=ac.DEFAULT_SECONDS, coupling_db=-20.0, seed=1, extra=None):
    """A recording as a microphone would deliver it: white noise (RMS noise_db dBFS) and, `delay` seconds late, the tone played at
    level_db in the on/off pattern of the connector after a loss of coupling_db; `extra(t, rng)` adds a room sound.
    Returns (recording, pattern)."""
    play, pattern = ac.build_play(f, level_db + coupling_db, ac.n_pairs_for(seconds), sr)
    n = int((len(play) / sr + ac.TAIL_S) * sr)
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 10 ** (noise_db / 20.0), n)
    t = np.arange(n) / sr
    x += np.interp(t - delay, np.arange(len(play)) / sr, play.astype(np.float64), left=0.0, right=0.0)
    if extra is not None:
        x += extra(t, rng)
    return x, pattern


def an(sr, f, level_db, noise_db, analyse_f=None, **kw):
    x, pat = synth(sr, f, level_db, noise_db, **kw)
    return ac.analyse(x, sr, analyse_f or f, pat)


# ============================================================================ S1: analysis
def test_analysis():
    for sr in (16000, 48000):
        t = np.arange(int(0.4 * sr)) / sr
        assert abs(ac.lockin(0.5 * np.sin(2 * np.pi * 1000 * t + 0.7), sr, 1000) - 0.5) < 0.01     # amplitude in peak units, any phase
        assert ac.lockin(0.5 * np.sin(2 * np.pi * 1000 * t), sr, 1500) < 0.005                     # another frequency does not count
        r = an(sr, 1000, -30, -60)
        assert abs(r["tone_db"] - (-50.0)) < 1.0 and r["heard"] and r["steady"] and r["snr_db"] > 25, r
        assert abs(r["delay_s"] - 0.12) < 0.02 and r["separation"] == 1.0 and r["match"] > r["chance"], r   # the delay is found in the recording itself
        assert abs(r["dominant_hz"] - 1000) < 15 and not r["clipped"] and not r["silent_input"] and not r["interference"]
    # the pattern: balanced, never three equal in a row, not a plain alternation, the same every time
    for n in range(4, 8):
        pat = ac.chop_pattern(n)
        assert len(pat) == 2 * n and sum(pat) == n and pat == ac.chop_pattern(n) and pat != [0, 1] * n and pat != [1, 0] * n
        assert all(not (pat[i] == pat[i + 1] == pat[i + 2]) for i in range(len(pat) - 2)), pat
        assert len(ac._surrogate_patterns(pat)) >= 40 and pat not in ac._surrogate_patterns(pat)
    assert ac.n_pairs_for(1.6) == 4 and ac.n_pairs_for(3.0) == 7 and ac.n_pairs_for(0.1) == 4
    r = an(16000, 1000, -120, -60)                                                                 # nothing played
    assert not r["heard"] and r["steady"] is None and r["snr_db"] < ac.MARGIN_DB, r
    wrong = an(16000, 1500, -30, -60, analyse_f=1000)                                              # the tone is at another frequency
    assert not wrong["heard"], wrong
    # the threshold: heard from about 10 dB above the noise on, in every one of 12 recordings with another noise each
    for lv, want in ((-72, 0), (-48, 12), (-40, 12)):
        got = sum(an(16000, 1000, lv, -60, seed=sd)["heard"] for sd in range(12))
        assert (got == 0) if want == 0 else (got == want), (lv, got)
    assert 0 < sum(an(16000, 1000, -56, -60, seed=sd)["heard"] for sd in range(12)) < 12            # near the threshold it is a matter of chance

    # ------------- ROOM SOUNDS. The room is never quiet: a fan, voices far away, a hum at the tone's own frequency, silence ...
    def hum(hz, db_):
        return lambda t, rng: 10 ** (db_ / 20) * np.sin(2 * np.pi * hz * t)

    def voices(db_, rate=1.5):                 # bursts of broadband noise, 0.2 - 0.8 s long, at random moments, many of them
        def f(t, rng):
            env, k = np.zeros_like(t), 0.0
            while k < t[-1]:
                k += rng.exponential(1.0 / rate)
                env[(t >= k) & (t < k + rng.uniform(0.2, 0.8))] = 1.0
            return env * rng.normal(0.0, 10 ** (db_ / 20), len(t))
        return f

    def slam(t0, t1, amp):
        return lambda t, rng: amp * rng.normal(0.0, 1.0, len(t)) * ((t > t0) & (t < t1))

    def heard_rate(extra, level=-30, n=8, f=1000, sr=16000):
        rs = [an(sr, f, level, -60, seed=sd, extra=extra) for sd in range(n)]
        return sum(r["heard"] for r in rs), sum(r["interference"] for r in rs), rs

    h, i, rs = heard_rate(hum(505, -35))                         # a steady hum, 15 dB louder than the tone, at another frequency
    assert h == 8 and all(abs(r["dominant_hz"] - 505) < 15 for r in rs), (h, i)
    h, i, rs = heard_rate(hum(1000, -70))                        # a hum exactly at the tone's frequency, 20 dB below the tone
    assert h == 8, (h, i)
    h, i, rs = heard_rate(hum(196, -45))                         # the wandering tone that was in the first test room
    assert h == 8, (h, i)
    h, i, rs = heard_rate(voices(-45))                           # voices far away: broadband bursts, noise much louder than the room alone
    assert h == 8 and min(r["snr_db"] for r in rs) > 10, (h, i)
    h, i, rs = heard_rate(slam(1.3, 1.45, 0.3))                  # a door slam in the middle of the measurement
    assert h == 8, (h, i)
    h, i, rs = heard_rate(lambda t, rng: 0.2 * rng.normal(0, 1, len(t)) * (((t > 1.3) & (t < 1.45)) | ((t > 2.0) & (t < 2.1))))
    assert h == 8, (h, i)                                        # two slams

    # nothing is played, the room is NOT quiet: it must never be reported as the tone
    for name, extra in (("no sound", None), ("fan hum", hum(505, -35)), ("hum at the tone's frequency", hum(1000, -45)),
                        ("voices far away", voices(-45)), ("loud voices", voices(-35)), ("a click at 1000 Hz", lambda t, rng:
                         0.2 * np.sin(2 * np.pi * 1000 * t + 0.3) * ((t > 1.1) & (t < 1.2))),
                        ("a notification that beeps at 1000 Hz every 1.2 s", lambda t, rng: 10 ** (-45 / 20) * np.sin(2 * np.pi * 1000 * t) * ((t % 1.2) < 0.3))):
        h, i, rs = heard_rate(extra, level=-120, n=10)
        assert h == 0, (name, h, i)
    # a tone that the room masks: voices louder than the tone are reported as 'not heard' (or no verdict), never as heard when there is none
    h, i, rs = heard_rate(voices(-30), level=-60, n=10)
    assert h == 0, h
    # beeps at the tone's own frequency, louder than the tone, WHILE the tone plays: the tone is there, so 'inaudible' would be wrong; the
    # program says 'heard' when the pattern wins and 'no verdict' when the beeps decide, never 'not heard'
    h, i, rs = heard_rate(lambda t, rng: 10 ** (-45 / 20) * np.sin(2 * np.pi * 1000 * t) * ((t % 1.2) < 0.3), n=6)
    assert h + i == 6 and i >= 1, (h, i)
    r = next(x for x in rs if x["interference"])
    note = ac.AudioConnector._note(dict(r, freq_hz=1000.0, steady=None, clipped=False), [])
    assert "NO VERDICT" in note and "neither 'heard' nor 'inaudible'" in note and "pairs" in note, note

    r = ac.analyse(np.zeros(len(synth(16000, 1000, -30, -60)[0])), 16000, 1000, ac.chop_pattern(4))
    assert r["silent_input"] and not r["heard"] and not r["interference"] and r["tone_db"] == ac.FLOOR_DB, r
    x, pat = synth(16000, 1000, 0, -60, coupling_db=6.0)
    assert ac.analyse(np.clip(x, -1, 1), 16000, 1000, pat)["clipped"]
    try:
        ac.analyse(np.zeros(1000), 16000, 1000, ac.chop_pattern(4))
        raise AssertionError("a short recording must be refused")
    except ac.AudioError:
        pass
    ok("S1 analysis (the tone is switched on and off in a fixed pseudo-random pattern and only what follows the pattern counts): level and "
       "delay are found in the recording, 16 and 48 kHz; the threshold is 10 dB above the room; a fan, a hum at the tone's own frequency, "
       "wandering tones, voices far away, door slams do not matter; with NO tone the same sounds are never reported as the tone; a room "
       "that follows the pattern gives 'no verdict'; silence and clipping are flagged")


def test_sound_summary():
    """The numbers of the first real run (use case 15): the model called -45 'not heard' although the program had given no verdict."""
    def ev(level, heard, interference=False, silent=False, f=1000.0):
        return {"tool": "play_tone", "ok": True, "result": dict(freq_hz=f, level_db=level, heard=heard, interference=interference,
                                                                silent_input=silent)}
    events = [ev(-30, True), ev(-50, False, True), ev(-50, False, True), ev(-40, True), ev(-45, False, True), ev(-50, False, True), ev(-42, True)]
    txt = "\n".join(A.sound_summary_md(events))
    assert "heard at -42, -40, -30" in txt, txt
    assert "no verdict (the room changed while the tone played) at -50, -45" in txt and "not heard at none" in txt, txt
    assert "Lowest level heard: -42" in txt and "highest level not heard below it: none" in txt, txt
    # a clean run: -80 and -65 not heard, -50 and -35 heard -> the bracket is -65 / -50
    clean = [ev(-80, False), ev(-65, False), ev(-50, True), ev(-35, True)]
    txt = "\n".join(A.sound_summary_md(clean))
    assert "Lowest level heard: -50" in txt and "highest level not heard below it: -65" in txt, txt
    # one level per frequency (a frequency curve) and a silent microphone give no threshold line
    assert A.sound_summary_md([ev(-30, True, f=300.0), ev(-30, True, f=600.0)]) == []
    assert "heard at none; not heard at none" in "\n".join(A.sound_summary_md([ev(-30, False, silent=True), ev(-20, False, silent=True)]))
    ok("S1b the report's sound summary is computed by the program: a reading with no verdict is neither 'heard' nor 'not heard', so the "
       "bracket of the threshold cannot be built from it (the first real run, where a model reported -45 as 'not heard', is the test)")


# ============================================================================ S2: the connector with the synthetic path
def fake_conn(state: dict | None = None):
    p = Path(tempfile.mkdtemp()) / "audio_state.json"
    p.write_text(json.dumps(state or {}))
    return ac.AudioConnector(ac.FakeSoundBackend(str(p))), p


def test_connector():
    c, st = fake_conn()
    info = c.info()
    assert info["ok"] and info["level_db_range"] == [-80.0, -12.0] and info["freq_hz_range"] == [100.0, 6400.0], info
    assert info["loudspeaker"] != info["microphone"] and "Pano-360" in info["loudspeaker"]
    heard = [c.play_tone(1000, lv)["heard"] for lv in (-80, -65, -45, -40, -25, -12)]
    assert heard == [False, False, True, True, True, True], heard                                    # a threshold between -60 and -48
    r = c.play_tone(1000, -40)
    assert r["ok"] and -24 < r["coupling_db"] < -18 and r["steady"] and r["loudspeaker"] and not r["silent_input"], r
    assert "heard" in r["note"] and r["seconds"] == 1.6 and r["tone_segments"] == 4 and abs(r["delay_s"] - 0.12) < 0.02, r
    t = {f: c.play_tone(f, -40)["tone_db"] for f in (200, 1000, 2500, 6000)}                         # the resonance and the weak low end
    assert t[2500] > t[1000] + 1 and t[2500] > t[6000] + 3 and t[200] < t[1000] - 10, t
    # limits are enforced, not suggested
    r = c.play_tone(1000, 0)
    assert r["level_db"] == -12.0 and "cut back" in r["note"], r
    r = c.play_tone(1000, -100)
    assert r["level_db"] == -80.0 and "raised" in r["note"] and not r["heard"], r
    assert c.play_tone(1000, -40, 10)["seconds"] == 2.8 and c.play_tone(1000, -40, 0.1)["seconds"] == 1.6      # the pattern: 4 to 7 pairs of 0.2 s
    for bad in ({"freq_hz": 50}, {"freq_hz": 7000}, {"freq_hz": float("nan")}, {"freq_hz": "abc"}, {"freq_hz": 1000, "label": "a b"}):
        r = c.play_tone(**bad)
        assert r["ok"] is False and r["error"], bad
    assert "between 100 and 6400" in c.play_tone(50)["error"]
    os.environ["HW3_AUDIO_MAX_DB"] = "-30"
    try:
        assert c.play_tone(1000, -20)["level_db"] == -30.0 and c.info()["level_db_range"][1] == -30.0
        os.environ["HW3_AUDIO_MAX_DB"] = "5"
        assert ac.max_db() == -3.0                                                                    # the cap itself is capped
    finally:
        del os.environ["HW3_AUDIO_MAX_DB"]
    # a muted microphone delivers digital silence; a volume change moves the level
    base = c.play_tone(1000, -30, label="before")
    st.write_text(json.dumps({"muted": True}))
    muted = c.play_tone(1000, -30, label="after")
    assert muted["silent_input"] and not muted["heard"] and muted["tone_db"] == ac.FLOOR_DB and "muted" in muted["note"], muted
    cmp = c.compare_audio("before", "after")
    assert cmp["ok"] and cmp["same_setting"] and cmp["significant"] and cmp["delta_tone_db"] < -40, cmp
    assert cmp["heard_a"] and not cmp["heard_b"] and cmp["silent_input_b"] and "silent" in cmp["note"], cmp
    st.write_text(json.dumps({"volume_db": -12}))
    quiet = c.play_tone(1000, -30, label="lower")
    d = c.compare_audio("before", "lower")
    assert -16 < d["delta_tone_db"] < -8 and d["significant"] and quiet["heard"] and "quieter" in d["note"], d
    st.write_text(json.dumps({}))
    again = c.play_tone(1000, -30, label="again")
    same = c.compare_audio("before", "again")
    assert abs(same["delta_tone_db"]) < 3 and same["significant"] is False and "no real change" in same["note"], same
    other = c.play_tone(2000, -30, label="other")
    diff = c.compare_audio("before", "other")
    assert diff["same_setting"] is False and diff["significant"] is False and "different frequency or level" in diff["note"], diff
    miss = c.compare_audio("before", "nope")
    assert miss["ok"] is False and "'nope'" in miss["error"] and "again" in miss["error"], miss
    assert set(c.info()["labels"]) == {"before", "after", "lower", "again", "other"}
    n = c.listen_noise(1.0)
    assert n["ok"] and -62 < n["noise_rms_db"] < -58 and not n["silent_input"], n
    assert c.listen_noise("x")["ok"] is False
    st.write_text(json.dumps({"muted": True}))
    assert c.listen_noise(1.0)["silent_input"] is True
    ok("S2 connector (synthetic path): threshold between -60 and -48 dBFS, resonance near 2.5 kHz, level / frequency / duration limits "
       "are enforced, a muted microphone reads as silence, a volume change moves the tone by the same dB, compare_audio is computed by "
       "the program, bad arguments are answered")


# ============================================================================ S3: the real backend on a stand-in for sounddevice
class StubSD:
    """A stand-in for the python package sounddevice: five devices and a loudspeaker-to-microphone loopback. The microphone of
    the Pano-360 can be made deaf to the Pano-360 loudspeaker (aec: echo cancellation) or refuse to open (deny_mic)."""

    def __init__(self, aec=False, deny_mic=False):
        self.devs = [
            dict(name="Pano-360", max_input_channels=0, max_output_channels=2, default_samplerate=48000.0),
            dict(name="Pano-360", max_input_channels=1, max_output_channels=0, default_samplerate=16000.0),
            dict(name="MacBook Pro Microphone", max_input_channels=1, max_output_channels=0, default_samplerate=48000.0),
            dict(name="MacBook Pro Speakers", max_input_channels=0, max_output_channels=2, default_samplerate=48000.0),
            dict(name="Microsoft Teams Audio", max_input_channels=1, max_output_channels=1, default_samplerate=48000.0)]
        self.aec, self.deny_mic = aec, deny_mic
        self.written = threading_event()
        self.out = None
        stub = self

        class InputStream:
            def __init__(self, device=None, channels=1, samplerate=48000, dtype="float32"):
                if stub.deny_mic:
                    raise RuntimeError("PortAudioError: Error opening InputStream: Internal PortAudio error [PaErrorCode -9986]")
                self.dev, self.sr = device, int(samplerate)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, n):
                stub.written.wait(0.6)                                     # a tone is written within this time, or nothing is played
                rng = np.random.default_rng(7)
                x = rng.normal(0.0, 1e-3, n)                                # -60 dBFS room noise
                if stub.out is not None:
                    odev, data, osr = stub.out
                    gain = 0.05                                             # -26 dB
                    if stub.aec and odev == 0 and self.dev == 1:
                        gain = 1e-6                                         # the device removes its own sound from its microphone
                    t = np.arange(n) / self.sr - 0.15
                    x = x + gain * np.interp(t, np.arange(len(data)) / osr, data, left=0.0, right=0.0)
                stub.out = None
                stub.written.clear()
                return x.astype(np.float32)[:, None], False

        class OutputStream:
            def __init__(self, device=None, channels=2, samplerate=48000, dtype="float32"):
                self.dev, self.sr = device, int(samplerate)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def write(self, data):
                assert data.dtype == np.float32 and data.shape[1] == 2
                stub.out = (self.dev, np.asarray(data[:, 0], np.float64), self.sr)
                stub.written.set()

        self.module = types.SimpleNamespace(query_devices=lambda: self.devs, InputStream=InputStream, OutputStream=OutputStream)


def threading_event():
    import threading
    return threading.Event()


class stub_sounddevice:
    def __init__(self, **kw):
        self.stub = StubSD(**kw)

    def __enter__(self):
        sys.modules["sounddevice"] = self.stub.module
        self.saved = {k: os.environ.pop(k, None) for k in ("HW3_AUDIO", "HW3_AUDIO_BACKEND", "HW3_AUDIO_NAME", "HW3_AUDIO_OUT", "HW3_AUDIO_IN",
                                                           "HW3_AUDIO_FILE")}
        self.tmp = tempfile.TemporaryDirectory()
        self.stub.pair_path = Path(self.tmp.name) / "audio.json"                           # audio_check.py --pairs writes here, not into the project
        os.environ["HW3_AUDIO_FILE"] = str(self.stub.pair_path)
        return self.stub

    def __exit__(self, *a):
        self.tmp.cleanup()
        sys.modules.pop("sounddevice", None)
        for k, v in self.saved.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)


def run_check(fn, *args) -> tuple[int, str]:
    import audio_check
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fn(*args)
    return code, buf.getvalue()


def test_real_backend_with_stub():
    import audio_check
    with stub_sounddevice() as stub:
        okay, why = ac.available()
        assert okay and "'Pano-360'" in why, why
        c = ac.open_audio()
        assert (c.b.out_idx, c.b.in_idx) == (0, 1) and (c.b.sr_out, c.b.sr_in) == (48000, 16000) and c.b.kind == "sounddevice"
        assert c.info()["freq_hz_range"] == [100.0, 6400.0]
        r = c.play_tone(1000, -40)
        assert r["ok"] and r["heard"] and r["steady"] and -50 < r["coupling_db"] < -20 and r["microphone"] == "Pano-360", r
        assert -70 < r["tone_db"] < -62, r                                                          # -40 dBFS played, -26 dB path in the stand-in
        assert c.play_tone(1000, -80)["heard"] is False
        n = c.listen_noise(1.0)
        assert n["ok"] and -63 < n["noise_rms_db"] < -57, n
        mac = ac.open_audio(None, "MacBook")                                                        # another microphone by a part of its name
        assert mac.b.in_idx == 2 and mac.b.sr_in == 48000 and mac.play_tone(1000, -40)["heard"]
        by_number = ac.open_audio("3", "2")                                                         # or by number
        assert (by_number.b.out_name, by_number.b.in_name) == ("MacBook Pro Speakers", "MacBook Pro Microphone")
        for sel, direction in (("9", "output"), ("1", "output"), ("Nothing", "input")):
            try:
                (ac.open_audio(sel, None) if direction == "output" else ac.open_audio(None, sel))
                raise AssertionError("must fail")
            except ac.AudioError as e:
                assert "Pano-360" in str(e) or "no channels" in str(e), e
        os.environ["HW3_AUDIO_NAME"] = "no-such-device"
        okay, why = ac.available()
        assert not okay and "no sound output device" in why and "MacBook Pro Speakers" in why, why   # the list of devices helps to choose
        del os.environ["HW3_AUDIO_NAME"]
        os.environ["HW3_AUDIO"] = "off"
        assert ac.available() == (False, "switched off (HW3_AUDIO=off)")
        del os.environ["HW3_AUDIO"]

        # audio_check.py on a good device
        code, out = run_check(audio_check.run_default, types.SimpleNamespace(out=None, inp=None))
        assert code == 0 and "RESULT: the microphone hears the loudspeaker" in out and "quietest level heard here is" in out, out
        code, out = run_check(audio_check.run_pairs, types.SimpleNamespace(out=None, inp=None))
        assert code == 0 and "use the speaker 'Pano-360' and the microphone 'Pano-360'" in out and "export" not in out, out
        assert "Teams" not in out and out.count("->") == 3, out                                    # the virtual device is skipped
        assert not stub.pair_path.exists() and "audio.json" not in out, out                          # the default pair needs no file

        # the device cancels its own sound out of its microphone: the check says so and names the other microphone
        stub.aec = True
        assert not c.play_tone(1000, -20)["heard"] and ac.open_audio(None, "MacBook").play_tone(1000, -40)["heard"]
        code, out = run_check(audio_check.run_default, types.SimpleNamespace(out=None, inp=None))
        assert code == 2 and "--pairs" in out and "echo cancellation" in out, out
        code, out = run_check(audio_check.run_pairs, types.SimpleNamespace(out=None, inp=None))
        assert code == 0 and "use the speaker 'Pano-360' and the microphone 'MacBook Pro Microphone'" in out, out
        assert "export" not in out and "audio.json" in out, out
        assert json.loads(stub.pair_path.read_text())["out"] == "Pano-360" and json.loads(stub.pair_path.read_text())["in"] == "MacBook Pro Microphone"
        assert ('HW3_AUDIO_OUT="MacBook Pro Speakers" HW3_AUDIO_IN="Pano-360" python run_hw3.py --model ... --use-case 17' in out), out   # mute needs the Pano mic

        # the saved pair is used without any environment variable; an explicit choice is stronger; a stale file falls back to the name
        c2 = ac.open_audio()
        assert (c2.b.out_name, c2.b.in_name) == ("Pano-360", "MacBook Pro Microphone") and c2.info()["chosen_by"] == \
            {"loudspeaker": "audio.json", "microphone": "audio.json"}, c2.info()
        assert c2.play_tone(1000, -40)["heard"] and c2.info()["microphone"] == "MacBook Pro Microphone"
        os.environ["HW3_AUDIO_OUT"], os.environ["HW3_AUDIO_IN"] = "MacBook Pro Speakers", "Pano"
        c3 = ac.open_audio()
        assert (c3.b.out_name, c3.b.in_name) == ("MacBook Pro Speakers", "Pano-360") and c3.info()["chosen_by"]["microphone"] == "environment"
        assert c3.play_tone(1000, -20)["heard"], "Mac loudspeaker -> Pano microphone is the pair of use case 17"
        os.environ.pop("HW3_AUDIO_OUT"); os.environ.pop("HW3_AUDIO_IN")
        c4 = ac.open_audio(None, "Pano")                                                         # an argument is stronger than the file
        assert c4.b.in_name == "Pano-360" and c4.info()["chosen_by"] == {"loudspeaker": "audio.json", "microphone": "argument"}
        stub.pair_path.write_text(json.dumps({"out": "No such speaker", "in": "No such microphone"}))
        c5 = ac.open_audio()                                                                     # another computer: the file does not fit
        assert (c5.b.out_name, c5.b.in_name) == ("Pano-360", "Pano-360") and c5.info()["chosen_by"] == {"loudspeaker": "default", "microphone": "default"}
        stub.pair_path.write_text("not json at all")
        assert ac.saved_pair() == {} and ac.open_audio().b.in_name == "Pano-360"
        stub.pair_path.write_text(json.dumps({"out": "Pano-360", "in": "MacBook Pro Microphone"}))
        stub.aec = False
        code, out = run_check(audio_check.run_pairs, types.SimpleNamespace(out=None, inp=None))          # the device works again: the file is removed
        assert code == 0 and not stub.pair_path.exists() and "was deleted" in out, out

        # permission denied: a message that says what to click, not a traceback
        stub.deny_mic = True
        try:
            c.play_tone(1000, -40)
            raise AssertionError("must fail")
        except ac.AudioError as e:
            assert "Privacy & Security" in str(e) and "microphone could not be used" in str(e), e
    with stub_sounddevice(deny_mic=True):
        sys.argv = ["audio_check.py"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = audio_check.main()
        assert code == 1 and "ERROR:" in buf.getvalue() and "Privacy & Security" in buf.getvalue(), buf.getvalue()
    ok("S3 real backend on a stand-in for sounddevice: the Pano-360 is chosen by name (virtual devices skipped), the recording thread "
       "overlaps the tone, other devices by name or number, bad choices list the devices; audio_check.py finds the working pair "
       "(also when the device cancels its own sound), saves it in audio.json (environment variable > file > name; a stale file is "
       "ignored; the file is removed when the default works), prints the command for use case 17 and explains a denied microphone")


def test_audio_check_script_with_fake():
    env = dict(os.environ, HW3_AUDIO_BACKEND="fake")
    env.pop("HW3_AUDIO", None)
    p = subprocess.run([sys.executable, str(HERE / "audio_check.py")], env=env, capture_output=True, text=True, timeout=120, cwd=HERE)
    assert p.returncode == 0 and "RESULT: the microphone hears the loudspeaker" in p.stdout, p.stdout + p.stderr
    assert "--use-case 15" in p.stdout and not list(HERE.glob("*.wav"))
    p = subprocess.run([sys.executable, str(HERE / "audio_check.py"), "--list"], env=env, capture_output=True, text=True, timeout=60, cwd=HERE)
    assert p.returncode == 0 and "synthetic" in p.stdout
    ok("S3b audio_check.py as a program (synthetic devices): the three steps run and the result is printed; nothing is written to disk")


# ============================================================================ S4: MCP server
def audio_bridge(tmp: Path, audio_state: Path | None = None) -> McpBridge:
    env = default_env(str(tmp / "work"), roi_file=str(T.make_roi(tmp)), fake=True, fake_state=str(tmp / "state.json"),
                      controls_file=str(tmp / "no_controls.json"), fake_audio=True,
                      fake_audio_state=str(audio_state) if audio_state else None)
    b = McpBridge(env=env)
    b.start()
    return b


def test_server():
    tmp = Path(tempfile.mkdtemp())
    st = tmp / "audio_state.json"
    st.write_text("{}")
    b = audio_bridge(tmp, st)
    try:
        names = b.tool_names()
        assert names == ["camera_info", "set_resolution", "capture", "measure_region", "save_reference", "compare_to_reference",
                         "audio_info", "play_tone", "listen_noise", "compare_audio"], names
        spec = {s["function"]["name"]: s["function"] for s in b.tool_specs()}
        assert set(spec["play_tone"]["parameters"]["properties"]) == {"freq_hz", "level_db", "seconds", "label"}
        assert spec["play_tone"]["parameters"]["required"] == ["freq_hz"]
        d = spec["play_tone"]["description"]
        assert "do not judge it yourself" in d and "-12" in d and "Nothing is recorded to disk" in d and "snr_db" in d, d
        assert "before / after" in spec["compare_audio"]["description"]
        assert json.loads(b.call("audio_info"))["ok"]
        r = json.loads(b.call("play_tone", {"freq_hz": 1000, "level_db": -40, "label": "a"}))
        assert r["ok"] and r["heard"] and r["label"] == "a", r
        assert json.loads(b.call("play_tone", {"freq_hz": 1000, "level_db": -40, "label": "b"}))["ok"]
        cmp = json.loads(b.call("compare_audio", {"label_a": "a", "label_b": "b"}))         # the labels are kept between calls
        assert cmp["ok"] and cmp["same_setting"] and not cmp["significant"], cmp
        assert json.loads(b.call("play_tone", {"freq_hz": "abc"}))["ok"] is False                 # refused by the argument check
        e = json.loads(b.call("play_tone", {"freq_hz": 30}))
        assert e["ok"] is False and "between 100 and 6400" in e["error"], e
        assert json.loads(b.call("compare_audio", {"label_a": "a", "label_b": "zz"}))["ok"] is False
        assert json.loads(b.call("listen_noise", {"seconds": 0.5}))["ok"]
        log = [json.loads(x) for x in (tmp / "work" / "instrument_log.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [x["tool"] for x in log] == ["audio_info", "play_tone", "play_tone", "compare_audio", "play_tone", "compare_audio",
                                            "listen_noise"], [x["tool"] for x in log]
        assert log[1]["args"]["freq_hz"] == 1000 and log[1]["result"]["tone_db"] < 0
        files = sorted(p.name for p in (tmp / "work").rglob("*") if p.is_file())
        assert files == ["instrument_log.jsonl"], files                                           # no sound, no picture on disk
    finally:
        b.close()
    ok("S4 MCP server: 11 tools (7 camera + 4 sound), play_tone is described with its limits, labels are kept between calls, bad "
       "arguments are answered, every call is logged, nothing but the log is written to disk")


# ============================================================================ S5: agent graph
def audio_env(audio_state: dict | None = None):
    tmp = Path(tempfile.mkdtemp())
    st = tmp / "audio_state.json"
    st.write_text(json.dumps(audio_state or {}))
    e = T.Env(tmp=tmp, bridge=audio_bridge(tmp, st))
    return e, st


def PT(freq, level, label=None, seconds=None):
    a = {"freq_hz": freq, "level_db": level}
    if label:
        a["label"] = label
    return ("play_tone", a)


def test_agent_threshold_search():
    e, _ = audio_env()
    try:
        hw = [T.TC(PT(1000, -80), PT(1000, -65), PT(1000, -50), PT(1000, -35)),                       # a coarse pass in one reply
              T.TC(PT(1000, -57)),                                                                  # then halve the interval: one call per reply
              T.TC(PT(1000, -53)),
              AIMessage(content="The tone was not heard at -65 dBFS and was heard from -50 dBFS up.")]
        final, models, asked = e.run(
            orch=[T.D("hardware", "Find the quietest level at which a 1000 Hz tone is heard by the microphone."),
                  T.D("finish", "The threshold lies between -65 and -50 dBFS.")], hw=hw)
        assert T.roles(final) == ["user", "orchestrator", "hardware", "orchestrator"] and not asked and final["stop_reason"] == "finish"
        ev = final["events"]
        assert [x["tool"] for x in ev] == ["play_tone"] * 6 and all(x["ok"] for x in ev) and final["n_calls"] == 6
        assert [x["result"]["heard"] for x in ev[:4]] == [False, False, True, True], [x["result"] for x in ev[:4]]
        assert len(A.readings_of(ev)) == 6                                    # the finish guard accepts sound readings: no camera needed
        assert len(models["hardware"].calls) == 4                             # three rounds of tools, then the report
        sysmsg = models["hardware"].calls[0][0].content
        assert "play_tone(freq_hz, level_db, seconds, label)" in sysmsg and "Sound readings" in sysmsg and "NEVER judge 'heard' yourself" in sysmsg
        assert "-80, -65, -50 and -35" in sysmsg and "below the quietest level" in sysmsg and "go up by 15 dB" in sysmsg
        assert "QUIETEST level" in sysmsg and "speakerphone" in sysmsg and "No camera control is available" in sysmsg
        assert {"audio_info", "play_tone", "listen_noise", "compare_audio"} <= {s["function"]["name"] for s in models["hardware"].bound}
        orch_sys = models["orchestrator"].calls[0][0].content
        assert "audio_info, play_tone, listen_noise and compare_audio" in orch_sys and "button of the device" in orch_sys
        assert "speakerphone" in A.fill(A.SCIENTIST_SYSTEM)
        for role in ("orchestrator", "scientist", "general"):
            assert not models[role].bound, role                               # only `hardware` holds the tools, the sound tools included
        rep = A.final_report_md(final)
        n_heard = sum(1 for x in ev if x["result"]["heard"])
        assert 2 <= n_heard <= 6 and rep.count("-> HEARD") == n_heard and rep.count("-> NOT heard") == 6 - n_heard, rep
        assert "tone 1000 Hz at -80 dBFS: microphone" in rep, rep
        assert "what the instrument answered" in rep
        assert "## Sound levels summarised by the program" in rep and "Lowest level heard: " in rep and "instrument tool call(s)" in rep, rep
        low = min(x["result"]["level_db"] for x in ev if x["result"]["heard"])
        assert f"Lowest level heard: {low:g}" in rep, rep
        assert not (e.work / "crops").exists() and sorted(p.name for p in e.work.rglob("*") if p.is_file()) == ["instrument_log.jsonl"]
    finally:
        e.close()
    ok("S5a threshold search: the coarse pass in one reply, then one halving per reply (decide -> tone -> reading -> decide); the sound readings "
       "count for the finish guard; only `hardware` holds the sound tools; the camera is never opened and nothing but the log is on disk")


def test_agent_button_check_with_person():
    e, st = audio_env()
    try:
        def press_mute(payload):
            assert payload["kind"] == "scene_action" and "done" in payload["answer_with"]
            st.write_text(json.dumps({"muted": True}))                         # the person presses the button of the device
            return "done, muted"

        def press_unmute(payload):
            st.write_text(json.dumps({"muted": False}))
            return "done, unmuted"

        ASK = lambda q: Decision(next="human", kind="scene_action", question=q, reason="only the person can press the button.")
        hw = [T.TC(PT(1000, -30, "before")), AIMessage(content="Reading 'before' taken: the tone is heard."),
              T.TC(PT(1000, -30, "after"), ("compare_audio", {"label_a": "before", "label_b": "after"})),
              AIMessage(content="After the press the microphone is silent."),
              T.TC(PT(1000, -30, "back"), ("compare_audio", {"label_a": "before", "label_b": "back"})),
              AIMessage(content="The sound is back.")]
        final, models, asked = e.run(
            orch=[T.D("hardware", "Play 1000 Hz at -30 dBFS and keep it under the label before."),
                  ASK("Please press the mute button of the device, then reply 'done'."),
                  T.D("hardware", "The person pressed mute. Play the same tone under the label after and compare it with before."),
                  ASK("Please press the mute button again to unmute, then reply 'done'."),
                  T.D("hardware", "The person unmuted. Play the same tone under the label back and compare it with before."),
                  T.D("finish", "The mute button silences the microphone (-) and unmuting brings the sound back.")],
            hw=hw, replies=[press_mute, press_unmute])
        assert T.roles(final) == ["user", "orchestrator", "hardware", "orchestrator", "human", "orchestrator", "hardware", "orchestrator", "human",
                                  "orchestrator", "hardware", "orchestrator"], T.roles(final)
        assert [a["kind"] for a in asked] == ["scene_action", "scene_action"] and final["refused"] == 0 and final["decisions"] == 6
        ev = final["events"]
        assert [x["tool"] for x in ev] == ["play_tone", "play_tone", "compare_audio", "play_tone", "compare_audio"] and all(x["ok"] for x in ev)
        before, after, c1, back, c2 = (x["result"] for x in ev)
        assert before["heard"] and not after["heard"] and after["silent_input"] and back["heard"], (before, after, back)
        assert c1["significant"] and c1["delta_tone_db"] < -40 and not c2["significant"] and abs(c2["delta_tone_db"]) < 3, (c1, c2)
        assert "[human]\ndone, muted" in models["hardware"].calls[2][-1].content                     # the hardware agent is told what happened
        seen = models["orchestrator"].calls
        assert "answered 'done'): 1" in seen[2][-1].content and "answered 'done'): 2" in seen[4][-1].content
        assert "reading BEFORE the physical change" in seen[0][0].content                            # the orchestrator's rule for before / after
        assert "compare_audio('before', 'after')" in models["hardware"].calls[0][0].content
        rep = A.final_report_md(final)
        assert "before -> after: tone" in rep and "significant" in rep and "the microphone is silent" in rep, rep
        evi = A.evidence_md(final["dialogue"])
        assert "Pause 1: scene_action" in evi and "Pause 2: scene_action" in evi and "done, unmuted" in evi
    finally:
        e.close()
    ok("S5b button check with the person in the loop: reading before, the system asks the person to press mute (scene_action), reading after "
       "and the drop computed by the program, unmute and the sound is back; the report and the evidence show it")


def test_agent_one_agent_two_instruments():
    """One hardware agent holds BOTH instruments: the camera and the sound tools in the same turn and the same run."""
    tmp = Path(tempfile.mkdtemp())
    st = tmp / "audio_state.json"
    st.write_text("{}")
    (tmp / "state.json").write_text("{}")
    e = T.Env(tmp=tmp, bridge=audio_bridge(tmp, st))
    try:
        hw = [T.TC(("set_resolution", {"width": 640, "height": 480}), T.MEASURE, PT(1000, -40, "x"), ("listen_noise", {"seconds": 0.5})),
              AIMessage(content="Measured the scene and played a tone.")]
        final, models, _ = e.run(orch=[T.D("hardware", "measure the scene at 640x480 and play a tone"), T.D("finish", "done")], hw=hw)
        assert [x["tool"] for x in final["events"]] == ["set_resolution", "measure_region", "play_tone", "listen_noise"]
        assert all(x["ok"] for x in final["events"]) and len(A.readings_of(final["events"])) == 3
        hw_tools = {s["function"]["name"] for s in models["hardware"].bound}
        assert {"set_resolution", "measure_region", "play_tone"} <= hw_tools
        assert not any(models[r].bound for r in ("orchestrator", "scientist", "general"))
        log = [json.loads(x)["tool"] for x in (e.work / "instrument_log.jsonl").read_text().splitlines()]
        assert log == ["set_resolution", "measure_region", "play_tone", "listen_noise"]
    finally:
        e.close()
    ok("S5c one hardware agent, two instruments: camera and sound tools in the same turn, one log; no other role holds a tool")


# ============================================================================ S7: the frequency curve (use case 16)
def real_run_16_events():
    """The readings of the real run of use case 16 on 16 October (nemotron, Pano-360 loudspeaker, MacBook microphone), as the graph stores them."""
    rows = [(1000, -53.4, 26.2, True), (300, -45.3, 20.9, True), (600, -39.2, 20.4, True), (2000, -45.8, 23.4, False),
            (3000, -47.4, 14.3, True), (4000, -51.3, 21.7, True), (6000, -49.1, 16.0, True),
            (578, -41.3, 19.5, True), (620, -43.6, 21.7, True), (600, -42.4, 20.2, True)]
    return [dict(n=i + 1, tool="play_tone", args=dict(freq_hz=f, level_db=-32.3), ok=True,
                 result=dict(ok=True, freq_hz=float(f), level_db=-32.3, tone_db=t, snr_db=s, heard=True, steady=st,
                             interference=False, silent_input=False)) for i, (f, t, s, st) in enumerate(rows)]


REAL_FINAL_16 = ("Human set device volume to a middle level. Hardware measured room noise (noise_rms_db = -52.3 dBFS), set playback level to "
                 "-32.3 dBFS (noise + 20 dB), verified a 1000 Hz tone was heard (SNR 26.2 dB, heard=true). At that level, played tones at "
                 "300, 600, 1000, 2000, 3000, 4000 and 6000 Hz and recorded microphone levels (tone_db): 300 Hz -> -45.3 dBFS, 600 Hz -> -39.2 "
                 "dBFS, 1000 Hz -> -53.4 dBFS, 2000 Hz -> -45.8 dBFS, 3000 Hz -> -47.4 dBFS, 4000 Hz -> -51.3 dBFS, 6000 Hz -> -49.1 dBFS. The "
                 "highest level occurs at 600 Hz. A check tone at 600 Hz (label verify600) gave tone_db = -42.4 dBFS, confirming strong response "
                 "in that region.")


def test_curve_summary_on_the_real_run():
    plan = A.CURVE_PLAN
    ev = real_run_16_events()
    st = A.curve_status(ev, plan)
    assert st["level"] == -32.3 and st["missing"] == [] and len(st["played"]) == 7 and st["extra"] == [578.0, 620.0]
    row600 = next(r for r in st["rows"] if r["freq"] == 600.0)
    assert row600["n"] == 2 and abs(row600["spread"] - 3.2) < 1e-9 and abs(row600["mean"] + 40.8) < 1e-9, row600
    assert next(r for r in st["rows"] if r["freq"] == 2000.0)["uneven"] and st["best"]["freq"] == 600.0
    assert abs(st["spread_eff"] - 3.2) < 1e-9 and [r["freq"] for r in st["within"]] == [578.0, 600.0, 620.0], st["within"]
    assert st["best_planned"] == 600.0 and st["refine_missing"] == [] and A.curve_missing(ev, plan) == []
    md = "\n".join(A.curve_summary_md(ev, plan, REAL_FINAL_16))
    assert "Verdict: NOT RESOLVED" in md and "somewhere between 578 and 620 Hz" in md, md          # 'confirmed' is not what the numbers say
    assert "Largest difference between repeated readings of the same tone: 3.2 dB" in md, md
    assert "| 2000 Hz | -45.8 | -45.8 | 23.4 | tone uneven, the level may read low; NOT repeated |" in md, md
    assert "| 600 Hz | -39.2, -42.4 | -40.8 |" in md and "| 578 Hz (extra) |" in md and "| 620 Hz (extra) |" in md, md
    assert "Played, but not named in the final message of the orchestrator: 578, 620 Hz" in md, md   # the two tones the model did not mention
    assert "Not played: none." in md

    # the first hardware turn of that run ended after 7 calls, without 6000 Hz: the program names what is missing
    first = [e for e in ev if e["args"]["freq_hz"] in (1000, 300, 600, 2000, 3000, 4000)][:6]
    assert A.curve_status(first, plan)["missing"] == [6000.0]
    miss = A.curve_missing(first, plan)
    assert miss == ["still to play at the level of the curve (-32.3 dBFS): 6000 Hz"], miss
    assert "play 6000 Hz at -32.3 dBFS" in A.curve_goal(first, plan)
    # all seven are there, nothing in between yet: the closer tones around the strongest one are still missing
    seven = [e for e in ev if e["args"]["freq_hz"] in (1000, 300, 600, 2000, 3000, 4000, 6000)][:7]
    st7 = A.curve_status(seven, plan)
    assert st7["best_planned"] == 600.0 and st7["refine_missing"] == [(300.0, 600.0), (600.0, 1000.0)], st7["refine_missing"]
    m7 = A.curve_missing(seven, plan)
    assert len(m7) == 2 and "one tone between 300 and 600 Hz" in m7[0] and "one tone between 600 and 1000 Hz" in m7[1], m7
    md7 = "\n".join(A.curve_summary_md(seven, plan))
    assert "No tone was repeated" in md7 and "still missing between 300 and 600, 600 and 1000 Hz" in md7, md7
    # nothing played at all; a reading with NO VERDICT does not count as played
    assert A.curve_status([], plan)["level"] is None and "has been played with a usable reading yet" in A.curve_missing([], plan)[0]
    bad = [dict(e) for e in seven]
    bad[2] = dict(bad[2], result=dict(bad[2]["result"], interference=True, heard=False))
    assert A.curve_status(bad, plan)["missing"] == [600.0] and A.curve_status(bad, plan)["no_verdict"] == [600.0]
    # a tone at another level does not belong to the curve
    other = seven + [dict(n=99, tool="play_tone", args={}, ok=True, result=dict(ok=True, freq_hz=1500.0, level_db=-20.0, tone_db=-30.0,
                                                                           snr_db=30.0, heard=True, steady=True, interference=False, silent_input=False))]
    assert 1500.0 not in [r["freq"] for r in A.curve_status(other, plan)["rows"]]
    assert A._named("600 Hz", 600) and not A._named("6000 Hz", 600) and not A._named("1600.5", 600) and A._named("label verify600", 600)
    # a clear maximum is called a maximum
    clear = [dict(e, result=dict(e["result"], tone_db=(-20.0 if e["args"]["freq_hz"] == 2000 else e["result"]["tone_db"]), steady=True))
             for e in seven]
    assert "Verdict: the response is strongest at 2000 Hz" in "\n".join(A.curve_summary_md(clear, plan))
    ok("S7a curve summary on the numbers of the real run 16: 578/600/620 Hz are not separable (spread 3.2 dB), the uneven 2000 Hz tone is flagged, "
       "the two tones the model did not name are listed, the missing 6000 Hz and the missing closer tones are named, NO VERDICT readings do not count")


def test_agent_curve_not_cut_off_and_not_finished_early():
    """What went wrong in the real run, with a scripted model: one tone per reply (the old limit cut the turn after 7 calls) and an
    orchestrator that wants to finish before the closer tones are played (the program refuses and says what is missing)."""
    e, _ = audio_env({"noise_db": -80.0})
    try:
        seq = [("listen_noise", {"seconds": 0.5}), PT(1000, -30, "check1000")] + [PT(f, -30) for f in (300, 600, 2000, 3000, 4000, 6000)]
        hw = [T.TC(c) for c in seq] + [AIMessage(content="Played the level check and the seven frequencies.")]
        assert len(seq) > 7 and A.MAX_ROUNDS >= len(hw)                                       # the old limit of 8 rounds could not hold this
        hw += [T.TC(PT(1500, -30), PT(2500, -30), PT(3500, -30)), AIMessage(content="Played 1500, 2500 and 3500 Hz.")]
        final, models, _ = e.run(
            orch=[T.D("hardware", "Find a level and play the seven frequencies."),
                  T.D("finish", "Done: the strongest is wherever."),                               # too early: nothing between the frequencies yet
                  T.D("hardware", "Play the tones between the strongest frequency and its neighbours."),
                  T.D("finish", "The response is strongest near 2.5 kHz.")], hw=hw, curve=A.CURVE_PLAN)
        ev = final["events"]
        first_turn = [x for x in ev if x["n"] <= 8]
        assert [x["tool"] for x in first_turn] == ["listen_noise"] + ["play_tone"] * 7 and all(x["ok"] for x in first_turn)
        assert 6000 in [x["args"]["freq_hz"] for x in first_turn if x["tool"] == "play_tone"]       # the seventh frequency was not cut off
        refused = [t for t in final["dialogue"] if t["role"] == "orchestrator" and "[orchestrator decision refused]" in t["text"]]
        assert len(refused) == 1 and "finish was refused by the finish guard: the request is not complete yet" in refused[0]["text"], refused
        assert "one tone between" in refused[0]["text"] and "dBFS" in refused[0]["text"], refused[0]["text"]
        assert final["stop_reason"] == "finish" and final["decisions"] == 4, (final["stop_reason"], final["decisions"])
        seen = [m[-1].content for m in models["orchestrator"].calls]
        assert "[checklist counted by the program: the frequency curve of this request]" in seen[0] and "No requested frequency" in seen[0], seen[0][-400:]
        assert "Still missing: none" in seen[1] and "finish is NOT allowed yet" in seen[1] and "Send hardware to: play, at -30 dBFS" in seen[1], seen[1][-500:]
        assert "Nothing is missing: you may finish." in seen[3], seen[3][-300:]
        st = A.curve_status(ev, A.CURVE_PLAN)
        assert st["level"] == -30.0 and st["missing"] == [] and st["refine_missing"] == [] and len(st["extra"]) == 3, st
        rep = A.final_report_md(final)
        assert "## Frequency curve summarised by the program" in rep and "Not played: none." in rep and "| 2500 Hz (extra) |" in rep, rep
        assert "Played, but not named in the final message of the orchestrator" in rep                  # the model's message does not name most frequencies
        hsys = models["hardware"].calls[0][0].content
        assert "send all the frequencies in ONE reply" in hsys and "Do not fit curves" in hsys and "Report EVERY tone you played" in hsys
        osys = models["orchestrator"].calls[0][0].content
        assert "do not compute a tone level for it" in osys and "Do not ask the person to set the volume unless" in osys
        for role in ("orchestrator", "scientist", "general"):
            assert not models[role].bound, role                                                   # the program only counts and refuses; the tools stay with `hardware`
    finally:
        e.close()
    ok("S7b frequency curve with a scripted model: one tone per reply is no longer cut off (the real run lost 6000 Hz), an early finish is refused "
       "with the missing tones named, the orchestrator reads the program's checklist, the report holds the curve summary; only `hardware` holds tools")


def test_curve_guard_does_not_trap_the_run():
    """The guard must never hold a run for ever: with the decision budget used up the finish is accepted and the report says what is missing;
    a run without a curve plan has no checklist at all."""
    e, _ = audio_env({"noise_db": -80.0})
    try:
        hw = [T.TC(PT(1000, -30), PT(600, -30)), AIMessage(content="Played two tones.")]
        final, models, _ = e.run(orch=[T.D("hardware", "play some tones"), T.D("finish", "Only two tones were played.")], hw=hw,
                                 max_decisions=2, curve=A.CURVE_PLAN)
        assert final["stop_reason"] == "finish" and final["decisions"] == 2
        rep = A.final_report_md(final)
        assert "(the request is NOT complete)" in rep and "Not played: 300, 2000, 3000, 4000, 6000" in rep, rep
    finally:
        e.close()
    e, _ = audio_env({"noise_db": -80.0})
    try:
        final, models, _ = e.run(orch=[T.D("hardware", "play a tone"), T.D("finish", "Played.")],
                                 hw=[T.TC(PT(1000, -30)), AIMessage(content="Played.")])
        assert "frequency curve of this request" not in models["orchestrator"].calls[1][-1].content
        assert "Frequency curve summarised" not in A.final_report_md(final) and A.CTX["curve"] is None
    finally:
        e.close()
    ok("S7c the finish guard of the curve does not trap a run (accepted when the decision budget is used up, the report says what is missing); "
       "without a curve plan there is no checklist and no curve summary")


# ============================================================================ S6: runner
def test_use_cases():
    import re
    import run_hw3
    assert run_hw3.AUDIO_CASES == ("15", "16", "17", "18", "19") and set(run_hw3.AUDIO_ONLY) < set(run_hw3.AUDIO_CASES)
    assert set(run_hw3.BUTTON_CASES) == {"17", "18", "19"} and "19" not in run_hw3.AUDIO_ONLY
    assert run_hw3.audio_wanted(None) and all(run_hw3.audio_wanted(k) for k in run_hw3.AUDIO_CASES)
    assert not any(run_hw3.audio_wanted(str(k)) for k in range(1, 15))                    # the camera use cases never get the sound tools
    for k in run_hw3.AUDIO_ONLY:
        txt = run_hw3.USE_CASES[k][1].lower()
        assert "camera" not in txt and "picture" not in txt and k not in run_hw3.NEEDS, k             # these never touch the camera
        assert re.search(r"\d{3,4} hz|loudspeaker|microphone", txt), k
    assert "quietest level" in run_hw3.USE_CASES["15"][1] and "not heard" in run_hw3.USE_CASES["15"][1]
    t16 = run_hw3.USE_CASES["16"][1]
    assert "strongest" in t16 and "between it and each of its two neighbours" in t16 and "how sure the readings are" in t16
    assert "script" not in t16 and "predict" not in t16 and "instrument_log" not in t16                  # no fit in a script: the next tones come from the readings
    assert all(f"{f}" in t16 for f in (300, 600, 1000, 2000, 3000, 4000, 6000)) and set(map(int, A.CURVE_PLAN["freqs"])) == {300, 600, 1000, 2000, 3000, 4000, 6000}
    t17 = run_hw3.USE_CASES["17"][1]
    assert "mute" in t17 and "ask the person" in t17 and "label" in t17
    for k in ("18", "19"):
        t = run_hw3.USE_CASES[k][1]
        assert "volume" in t and "three times" in t and "ask the person" in t and "label" in t and "mute" not in t, k
    t19 = run_hw3.USE_CASES["19"][1]
    assert "sharpness_norm" in t19 and "640x480" in t19 and "1000 Hz" in t19 and "volume - button" in t19    # camera + loudspeaker + microphone + button

    # the pair of devices must suit the use case (checked before any model request is spent)
    req = run_hw3.audio_requirement
    assert req("15", "Pano-360", "MacBook Pro Microphone") is None and req("16", "Pano-360", "Pano-360") is None
    assert req("17", "MacBook Pro Speakers", "Pano-360") is None and req("17", "Pano-360", "Pano-360") is None
    adv = req("17", "Pano-360", "MacBook Pro Microphone")
    assert adv and "MUTE" not in adv and "mute button of the Pano-360 MICROPHONE" in adv and 'HW3_AUDIO_IN="Pano"' in adv and "--use-case 17" in adv, adv
    for k in ("18", "19"):
        assert req(k, "Pano-360", "MacBook Pro Microphone") is None
        adv = req(k, "MacBook Pro Speakers", "Pano-360")
        assert adv and "volume buttons of the Pano-360 LOUDSPEAKER" in adv and f"--use-case {k}" in adv, adv
    os.environ["HW3_AUDIO_NAME"] = "Dragon"
    assert req("18", "Dragon-1", "x") is None and req("18", "Pano-360", "x") is not None
    del os.environ["HW3_AUDIO_NAME"]
    from mcp_bridge import default_env as de
    f = Path(de("w")["HW3_AUDIO_FILE"])                                                    # the server reads the same audio.json as audio_check.py
    assert f.is_absolute() and f.name == "audio.json", f
    ok("S6 use cases 15-19: 15-18 never touch the camera, 16 looks closer where the response is strongest, 17 is the mute button and 18-19 the volume "
       "buttons (the person presses them), 19 is the acceptance test of camera + loudspeaker + microphone; a pair that does not suit the "
       "use case is refused with the exact command, before any model request")


if __name__ == "__main__":
    test_analysis()
    test_sound_summary()
    test_connector()
    test_real_backend_with_stub()
    test_audio_check_script_with_fake()
    test_server()
    test_agent_threshold_search()
    test_agent_button_check_with_person()
    test_agent_one_agent_two_instruments()
    test_curve_summary_on_the_real_run()
    test_agent_curve_not_cut_off_and_not_finished_early()
    test_curve_guard_does_not_trap_the_run()
    test_use_cases()
    print(f"\nall {len(PASSED)} sound checks passed")
