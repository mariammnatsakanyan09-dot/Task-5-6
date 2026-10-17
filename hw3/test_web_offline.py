"""Offline checks of Homework 4 (no API calls, no camera, nothing is played).

    pip install fastapi uvicorn httpx
    python3 test_web_offline.py

W1  watchdog logic     normal samples are no alarm, a changed scene / noise / silence / lost check tone is, one odd sample is not
W2  watch loop         the synthetic instrument is broken while the watchdog runs: it notices by itself, hands an ALERT to the graph,
                       the agent investigates (its own `why` on every call), the person is asked (a pause), the normal values are measured again
W3  the web server     the HTTP interface (state, log, event stream, ask, reply, stop, crop, simulated faults) and the page
W4  `why`              the reason of a call is shown and written to the report, and the instrument never receives it
W5  Stop               a run that is in progress ends after its current step (not at the end), leaves no empty folder, and watching goes idle
W6  request counter    the number of model requests of the final line and of the run folder counts ONE run (not the whole server session)
W7  page settings      a letter, a negative or a too small number is refused with a sentence (HTTP 400); good numbers start the watch;
                       the .gitignore keeps the copied pictures of the runs out of git
"""
import json
import os
import tempfile
import time
from pathlib import Path

os.environ["HW3_CAMERA_BACKEND"] = "fake"
os.environ["HW3_AUDIO"] = "off"

from langchain_core.messages import AIMessage  # noqa: E402

import hw3_agents as A  # noqa: E402
import llm_utils as U  # noqa: E402
import watchdog as W  # noqa: E402
import web_app as web  # noqa: E402
from hw3_agents import Decision  # noqa: E402
from mcp_bridge import McpBridge, default_env  # noqa: E402
from test_hw3_offline import D, Scripted, TC, make_roi  # noqa: E402

U.RETRY_SLEEP[:] = [0, 0]
PASSED = []


def ok(name):
    PASSED.append(name)
    print("ok ", name)


def wait_for(cond, timeout=60.0, what="condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError(f"timeout waiting for {what}")


def types_of(session):
    return [e["type"] for e in session.hub.events]


# ============================================================================ W1
def test_watchdog_logic():
    base = W.Normal()
    for i in range(6):
        base.add(dict(brightness=150 + i % 2, sharpness=30.0 + i % 3, noise_db=-60.0 + 0.3 * (i % 2), tone_snr=28.0, tone_coupling=-40.0 - i % 2))
    assert base.ready()
    normal = dict(brightness=151.0, sharpness=31.0, noise_db=-60.0, tone_snr=27.0, errors=[])
    assert W.judge(base, normal) == []
    covered = dict(normal, brightness=4.0, sharpness=0.3)
    f = W.judge(base, covered)
    assert any("brightness" in x and "-> 4" in x for x in f) and any("sharpness" in x for x in f), f
    assert any("room noise" in x for x in W.judge(base, dict(normal, noise_db=-45.0)))
    assert any("silence" in x for x in W.judge(base, dict(normal, silent=True)))
    assert any("no longer heard" in x for x in W.judge(base, dict(normal, tone_heard=False, tone_snr=2.0)))
    assert any("weaker" in x for x in W.judge(base, dict(normal, tone_heard=True, tone_snr=12.0)))
    assert W.judge(base, dict(normal, tone_heard=False, tone_snr=40.0, tone_interference=True)) == [], "interference is no verdict"
    assert W.judge(base, dict(normal, tone_heard=True, tone_snr=27.0, tone_coupling=-43.0)) == [], "3 dB is the spread of two plays of the same tone"
    f = W.judge(base, dict(normal, tone_heard=True, tone_snr=27.0, tone_coupling=-49.0))
    assert f and "check tone is weaker at the microphone" in f[0] and "-> -49" in f[0], "the loudspeaker 8 dB quieter is caught by the level at the microphone, the margin above the noise did not change"
    assert any("camera" in x for x in W.judge(base, dict(errors=["camera: no picture received"])))
    assert W.judge(base, dict(normal, sharpness=300.0)) == [], "a rise of sharpness is no alarm (real camera: 187 -> 1000 when the picture settled)"
    assert W.judge(base, dict(normal, sharpness=12.0)) == [], "a fall to a half is no alarm either; a collapse to a third and below is"
    assert W.judge(base, dict(normal, sharpness=8.0)), "a collapse of sharpness with the same brightness (covered lens on the real camera) is caught"
    assert "sharpness collapse" in W.alert_request(["scene sharpness 31.0 -> 8.0"], base, None), "the hint about the covered lens is in the request"
    rq = W.alert_request(["room noise -51.0 -> -40.0 dBFS"], base, None)
    assert "Do NOT ask the person to be quiet" in rq and "raise level_db yourself" in rq, "noise: the agent acts itself, the person is not bothered"
    rq = W.alert_request(["the check tone is no longer heard (it was 20 dB above the noise, now 6 dB)"], base, dict(level_db=-30.0))
    assert "COMPENSATE YOURSELF" in rq and "the maximum" in rq, "a quieter loudspeaker is compensated by the agent"
    rq = W.alert_request(["the microphone delivers pure silence (muted?)"], base, None)
    assert "only a person can unmute" in rq and "COMPENSATE" not in rq, "a muted microphone cannot be fixed by a louder tone"
    # the limits of the watchdog are in the request, with numbers (in a real run the agent compared with the old normal value 1027 only,
    # found 395-403 "far below normal", asked for a wipe and ended with "not back to normal" - although the watchdog itself calls >= 342 normal)
    real = W.Normal()
    for i in range(6):
        real.add(dict(brightness=119.6 + (i % 2) * 0.2, sharpness=1027.4, noise_db=-50.6, tone_snr=None))
    lim = W.accepted_text(real, None)
    assert "between 90 and 150 grey levels" in lim and "sharpness_norm at least 342" in lim and "at most -42.6 dBFS" in lim, lim
    rq = W.alert_request(["scene sharpness 1027.4 -> 202.9"], real, None)
    assert "The watchdog itself calls the instrument normal when" in rq and "at least 342" in rq and "inside them is normal" in rq, rq
    assert "BELOW the sharpness limit" in rq, "a wipe is asked only when the reading is still below the limit"
    assert "do not play tones" in rq and "do not call camera_info" in rq and "at most 5 short lines" in rq, "short, economical, camera-only"
    assert "do not play tones" not in W.alert_request(["room noise -51.0 -> -40.0 dBFS"], base, None), "a noise alarm may use the check tone"
    now = dict(brightness=122.45, sharpness=395.5, noise_db=-49.2, errors=[])          # the lens uncovered again (real readings)
    assert W.judge(real, now) == [], "395 is inside the limits: the watchdog calls it normal"
    assert W.judge(real, dict(now, brightness=105.4, sharpness=216.5)), "the covered lens (real readings) is below the limit"
    # warm-up: the normal values wait until the picture has settled (but not longer than KEEP samples)
    wu = W.Normal()
    for v in (187.0, 187.0, 1000.0, 1000.0):
        wu.add(dict(brightness=120.0, sharpness=v))
    assert len(wu.items) == 4 and not wu.ready(), "the first readings still jump: not ready"
    wu.add(dict(brightness=120.0, sharpness=1000.0))
    wu.add(dict(brightness=120.0, sharpness=1000.0))
    assert wu.ready(), "the last four agree: ready"
    flip = W.Normal()
    for i in range(W.KEEP):
        flip.add(dict(brightness=120.0, sharpness=200.0 if i % 2 else 1000.0))
    assert flip.ready(), "a camera that never settles gets its normal values after KEEP samples"
    # the check tone is known from the calibration beep before the first beep sample is in
    tr = W.Normal()
    for i in range(5):
        tr.add(dict(brightness=150.0, sharpness=30.0, noise_db=-60.0))
    assert not W.judge(tr, dict(brightness=150.0, sharpness=30.0, noise_db=-60.0, tone_heard=False, tone_snr=1.0))
    tr.tone_ref = 17.0
    assert any("no longer heard" in x for x in W.judge(tr, dict(brightness=150.0, sharpness=30.0, noise_db=-60.0, tone_heard=False, tone_snr=1.0)))
    assert tr.snapshot().tone_ref == 17.0 and W.Normal().tone_ref is None
    # slow drift is not an alarm: the normal values follow it
    drift = W.Normal()
    for i in range(30):
        s = dict(brightness=150 - i * 1.0, sharpness=30.0, noise_db=-60.0, errors=[])
        if drift.ready():
            assert W.judge(drift, s) == [], (i, W.judge(drift, s))
        drift.add(s)
    # one odd sample (a flash) does not call the agent, two in a row do
    calls, events = [], []

    class Fake:
        def __init__(self):
            self.seq = iter([150, 150, 150, 150, 150, 40, 150, 150, 40, 40, 40])

        def __call__(self, name, args):
            if name == "measure_region":
                self.cur = next(self.seq)
                return json.dumps(dict(ok=True, mean_brightness=self.cur, sharpness_norm=30.0, saturated_fraction=0.0))
            if name == "listen_noise":
                return json.dumps(dict(ok=True, noise_rms_db=-60.0, silent_input=False))
            return json.dumps(dict(ok=False, error="unknown"))

    wd = W.Watchdog(Fake(), ["measure_region", "listen_noise"], events.append, lambda r, f: calls.append((r, f)), interval=0.0)
    got = [wd.step() for _ in range(11)]
    assert got[:5] == [None] * 5 and got[5] is None and got[6] is None and got[7] is None and got[8] is None, got
    assert got[9] and "brightness" in got[9][0], got
    assert [e["status"] for e in events[:4]] == ["warming up"] * 4 and events[5]["status"] == "changed"
    req = W.alert_request(got[9], wd.base, None)
    assert req.startswith("ALERT from the watchdog") and "Normal values until now" in req and "-> 40" in req
    # a burst of noise, a quiet sample, another burst (music, a clap): 2 of the last 3 samples is an alarm; one burst alone is not
    nseq = iter([-60.0] * 5 + [-40.0, -60.0, -40.0])
    wd3 = W.Watchdog(lambda n, a: json.dumps(dict(ok=True, noise_rms_db=next(nseq), silent_input=False)), ["listen_noise"],
                     [].append, lambda r, f: None, interval=0.0)
    res3 = [wd3.step() for _ in range(8)]
    assert res3[:7] == [None] * 7 and res3[7] and "room noise" in res3[7][0], res3
    # the instrument is not plugged in at the start: no alarm, no model call, errors are not "normal values"; then it appears
    ev2, called2 = [], []
    state = dict(plugged=False)

    def maybe(name, args):
        if not state["plugged"]:
            return json.dumps(dict(ok=False, error="the Pano-360 was not found"))
        return json.dumps(dict(ok=True, mean_brightness=150.0, sharpness_norm=30.0, saturated_fraction=0.0))

    wd2 = W.Watchdog(maybe, ["measure_region"], ev2.append, lambda r, f: called2.append(r), interval=0.0)
    assert [wd2.step() for _ in range(10)] == [None] * 10 and not called2
    assert all(e["status"] == "waiting for the instrument" for e in ev2) and len(wd2.base.items) == 0
    state["plugged"] = True
    assert [wd2.step() for _ in range(5)] == [None] * 5 and wd2.base.ready() and ev2[-1]["status"] == "normal"
    state["plugged"] = False                         # unplugged while watching: THIS is an alarm
    got2 = [wd2.step() for _ in range(3)]
    assert got2[:1] == [None] and got2[1] and "camera" in got2[1][0], got2
    # a reflecting room: the margin above the noise stays at 12 dB at every level (real: 11.6 / 12.3 / 11.7 dB at -30 / -20 / -12 dBFS), but the level at
    # the microphone follows the level played: the check is calibrated at the quietest level and a quieter loudspeaker is caught by the coupling
    room = dict(vol=0.0)

    def tone_room(name, args):
        if name == "play_tone":
            lv = float(args["level_db"])
            tone_db = lv - 20.0 + room["vol"]                       # the loudspeaker: -20 dB coupling, a volume change shifts it
            return json.dumps(dict(ok=True, tone_db=tone_db, noise_db=tone_db - 12.0, snr_db=12.0, heard=True, interference=False,
                                   silent_input=False, coupling_db=tone_db - lv))
        return json.dumps(dict(ok=True, mean_brightness=150.0, sharpness_norm=30.0, saturated_fraction=0.0, noise_rms_db=-60.0, silent_input=False))

    nts = []
    wdt = W.Watchdog(tone_room, ["measure_region", "listen_noise", "play_tone"], nts.append, lambda r, f: None, interval=0.0, tone_every=0.001)
    wdt.calibrate_tone()
    assert wdt.tone and wdt.tone["level_db"] == -30.0, wdt.tone
    assert wdt.base.coupling_ref == -20.0 and "does not grow with the level" in nts[-1]["text"] and "coupling -20 dB" in nts[-1]["text"], nts
    ok_s = W.sample(tone_room, ["play_tone"], wdt.tone)
    assert W.judge(wdt.base, ok_s) == [], "the same loudspeaker: no alarm, although the margin is only 12 dB"
    room["vol"] = -9.0                                              # the volume was turned down
    low = W.sample(tone_room, ["play_tone"], wdt.tone)
    assert low["tone_heard"] is True and low["tone_snr"] == 12.0, "the margin above the noise did not change ..."
    fl = W.judge(wdt.base, low)
    assert fl and "weaker at the microphone" in fl[0], "... but the level at the microphone fell by 9 dB: this is the alarm"
    # the normal values of the check tone in the request, and the agent is told to compensate by the loss in coupling
    wdt.base.add(dict(brightness=150.0, sharpness=30.0, noise_db=-60.0))
    rq = W.alert_request(fl, wdt.base, wdt.tone)
    assert "coupling" in rq and "Raise level_db by about that loss" in rq and "-12 dBFS is the maximum" in rq, rq
    assert "at least -26 dB" in W.accepted_text(wdt.base, wdt.tone), W.accepted_text(wdt.base, wdt.tone)
    # the program's own check after an investigation says NOT back when the readings are still outside the limits
    notes = []
    bad = W.Watchdog(lambda n, a: json.dumps(dict(ok=True, mean_brightness=105.0, sharpness_norm=216.5, saturated_fraction=0.0))
                     if n == "measure_region" else json.dumps(dict(ok=True, noise_rms_db=-50.0, silent_input=False)),
                     ["measure_region", "listen_noise"], notes.append, lambda r, f: None, interval=0.0)
    bad.verify(real)
    assert notes and "NOT back inside the limits" in notes[-1]["text"] and "sharpness" in notes[-1]["text"], notes
    ok("W1 watchdog logic: normal is quiet; brightness, noise, silence, lost tone and instrument errors are caught; a flash is not an alarm; slow drift is not an alarm; an instrument that is not plugged in at the start calls no model")


# ============================================================================ helpers for W2 / W3
def make_session(models, tmp, tone_check=False):
    work = tmp / "work"
    cam_state, aud_state = work / "fake_state.json", work / "fake_audio_state.json"
    env = default_env(str(work), roi_file=str(make_roi(tmp)), fake=True, fake_state=str(cam_state),
                      controls_file=str(tmp / "no_controls.json"), fake_audio=True, fake_audio_state=str(aud_state))
    bridge = McpBridge(env=env)
    bridge.start()
    sh = web.SharedBridge(bridge)
    s = web.Session(models, sh, work, tmp / "runs", info=dict(camera="synthetic"), model_name="scripted", fake=True)
    s.SETTLE_AFTER_HUMAN = 0.0                       # the wait for the real camera to readjust is not needed with the synthetic one
    return s, bridge


def investigation_models():
    why1 = "brightness fell from about 150 to 4: I suspect a covered lens, so I take my own reading of the scene"
    why2 = "the person says the lens is uncovered: I measure again to confirm the brightness is back"
    orch = [D("hardware", "Measure the scene brightness yourself to confirm the alert."),
            Decision(next="human", kind="scene_action", question="Please uncover the lens and reply done.", reason="a person must act."),
            D("hardware", "Measure the scene again and compare with the normal brightness (about 150)."),
            D("finish", "The lens was covered (brightness 4 vs about 150); after uncovering it is back to normal.")]
    hw = [TC(("measure_region", {"region": "scene", "n_frames": 3, "why": why1})), AIMessage(content="Brightness is about 4: dark."),
          TC(("measure_region", {"region": "scene", "n_frames": 3, "why": why2})), AIMessage(content="Brightness is back near 150.")]
    return dict(orchestrator=Scripted(orch), scientist=Scripted(), general=Scripted(), hardware=Scripted(hw), executor=Scripted()), why1


# ============================================================================ W2
def test_watch_loop_catches_a_covered_lens():
    tmp = Path(tempfile.mkdtemp())
    models, why1 = investigation_models()
    s, bridge = make_session(models, tmp)
    s.SETTLE_AFTER_HUMAN = 0.2
    try:
        s.start_watch(interval=0.02, tone_every=0)
        wait_for(lambda: sum(1 for e in s.hub.events if e["type"] == "sample" and e["status"] == "normal") >= 3, what="normal samples")
        assert not any(e["type"] == "anomaly" for e in s.hub.events), "nothing wrong yet: no alert, no model call"
        assert all(not m.calls for m in models.values()), "no model is called while everything is normal"
        s.fake_set(covered=True)                                       # the person covers the lens
        pause = wait_for(lambda: next((e for e in s.hub.events if e["type"] == "pause"), None), what="the pause")
        assert s.mode == "paused" and "uncover" in pause["payload"]["question"].lower()
        an = next(e for e in s.hub.events if e["type"] == "anomaly")
        assert any("brightness" in f for f in an["facts"]) and an["request"].startswith("ALERT from the watchdog")
        s.fake_set(covered=False)                                      # the person uncovers it and answers
        s.reply("done, uncovered")
        wait_for(lambda: any(e["type"] == "final" for e in s.hub.events), what="the final report")
        wait_for(lambda: s.mode == "watching", what="back to watching")
        wait_for(lambda: sum(1 for e in s.hub.events[-40:] if e["type"] == "sample" and e["status"] == "normal") >= 2, what="normal again")
        order = types_of(s)
        i_an, i_run, i_pause, i_rep, i_fin = (order.index(x) for x in ("anomaly", "run_start", "pause", "reply", "final"))
        assert i_an < i_run < i_pause < i_rep < i_fin, order
        calls = [e for e in s.hub.events if e["type"] == "tool_call"]
        assert len(calls) == 2 and calls[0]["why"] == why1 and "uncovered" in calls[1]["why"]
        res = [e for e in s.hub.events if e["type"] == "tool_result"]
        assert res[0]["values"]["mean_brightness"] < 20 and res[1]["values"]["mean_brightness"] > 100, [r["values"] for r in res]
        run = sorted((tmp / "runs").iterdir())[-1]
        assert run.name.endswith("-alert") and (run / "report.md").exists() and (run / "transcript.md").exists()
        assert why1 in (run / "transcript.md").read_text(encoding="utf-8"), "the reason of the call is in the transcript"
        log = [json.loads(x) for x in (tmp / "work" / "instrument_log.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all("why" not in e["args"] for e in log), "the instrument never receives `why`"
        assert any(e["type"] == "note" and "checking the instrument once more" in e["text"] for e in s.hub.events)
        chk = wait_for(lambda: next((e for e in s.hub.events if e["type"] == "note" and e["text"].startswith("check after the investigation")), None),
                       what="the program's own check after the investigation")
        assert "made by the program, no model" in chk["text"] and "back inside the limits of the alert" in chk["text"], chk
        order = types_of(s)
        i_wait = next(i for i, e in enumerate(s.hub.events) if e["type"] == "note" and "waiting 0.2 s" in e["text"])
        assert order.index("reply") < i_wait < [i for i, t in enumerate(order) if t == "tool_call"][1], "the camera settles before the next reading"
        s.stop()
        wait_for(lambda: s.mode == "idle", what="idle after stop")
    finally:
        s.stop()
        bridge.close()
    ok("W2 watch loop: covered lens noticed by itself, no model before the alert, ALERT -> agent -> pause -> person -> verification -> watching again; `why` on every call")


def test_mute_and_noise_alerts():
    tmp = Path(tempfile.mkdtemp())
    models = dict(orchestrator=Scripted([D("hardware", "check the microphone"), D("finish", "the microphone is muted")]),
                  scientist=Scripted(), general=Scripted(),
                  hardware=Scripted([TC(("listen_noise", {"seconds": 1.0, "why": "the watchdog saw silence: I listen myself"})),
                                     AIMessage(content="The microphone delivers silence.")]), executor=Scripted())
    s, bridge = make_session(models, tmp)
    try:
        s.start_watch(interval=0.02, tone_every=0)
        wait_for(lambda: sum(1 for e in s.hub.events if e["type"] == "sample" and e["status"] == "normal") >= 3, what="normal")
        s.fake_set(muted=True)
        an = wait_for(lambda: next((e for e in s.hub.events if e["type"] == "anomaly"), None), what="the alert")
        assert any("silence" in f for f in an["facts"]), an
        wait_for(lambda: any(e["type"] == "final" for e in s.hub.events), what="final")
        res = [e for e in s.hub.events if e["type"] == "tool_result"][0]
        assert res["values"]["silent_input"] is True
        s.stop()
        wait_for(lambda: s.mode == "idle")
    finally:
        s.stop()
        bridge.close()
    ok("W2b muted microphone is noticed (silence) and the agent confirms it with its own reading")


def test_tone_check():
    tmp = Path(tempfile.mkdtemp())
    s, bridge = make_session(dict(orchestrator=Scripted([D("hardware", "x"), D("finish", "y")]), scientist=Scripted(),
                                  general=Scripted(), hardware=Scripted([TC(("listen_noise", {"seconds": 1.0})), AIMessage(content="ok")]),
                                  executor=Scripted()), tmp)
    try:
        s.start_watch(interval=0.02, tone_every=0.05)
        note = wait_for(lambda: next((e for e in s.hub.events if e["type"] == "note" and "check tone" in e["text"]), None), what="calibration")
        assert "is heard" in note["text"] and "coupling" in note["text"], note
        wait_for(lambda: sum(1 for e in s.hub.events if e["type"] == "sample" and e["status"] == "normal" and e.get("tone_snr")) >= 3,
                 what="samples with the tone")
        s.fake_set(volume_db=-60.0)                                    # the volume of the loudspeaker is turned down
        an = wait_for(lambda: next((e for e in s.hub.events if e["type"] == "anomaly"), None), what="tone alert")
        assert any("check tone" in f for f in an["facts"]), an
        wait_for(lambda: any(e["type"] == "note" and "investigation is over" in e["text"] for e in s.hub.events), what="investigation over")
        wait_for(lambda: sum(1 for e in s.hub.events if e["type"] == "note" and "check tone" in e["text"]) >= 2,
                 what="the check tone is tuned again after the investigation")
        off = next(e for e in s.hub.events if e["type"] == "note" and "tone check is off" in e["text"])
        assert "-12 dBFS:" in off["text"] and "dB" in off["text"] and "at least 10 dB above the noise" in off["text"], off
        W.RECAL_S = 0.1                                                # the volume is back: the switched-off check comes back by itself
        s.fake_set(volume_db=0.0)
        wait_for(lambda: sum(1 for e in s.hub.events if e["type"] == "note" and "is heard" in e["text"]) >= 2, what="the tone check comes back")
        s.stop()
    finally:
        s.stop()
        wait_for(lambda: s.mode in ("idle", "paused", "running", "investigating", "watching"))
        bridge.close()
    ok("W2c check tone: calibrated, then a lowered loudspeaker volume raises the alert")


# ============================================================================ W3
def test_web_server():
    from fastapi.testclient import TestClient
    tmp = Path(tempfile.mkdtemp())
    why = "I take one reading of the scene to answer the question"
    models = dict(orchestrator=Scripted([D("hardware", "measure the scene"), Decision(next="human", kind="scene_action",
                                                                                      question="Hold your hand in front, reply done.",
                                                                                      reason="a person must act."),
                                         D("finish", "done: brightness measured")]),
                  scientist=Scripted(), general=Scripted(),
                  hardware=Scripted([TC(("capture", {"label": "t", "n_frames": 1, "why": why})), AIMessage(content="Captured.")]),
                  executor=Scripted())
    s, bridge = make_session(models, tmp)
    c = TestClient(web.create_app(s))
    try:
        assert "Hardware-in-the-loop lab" in c.get("/").text and "EventSource" in c.get("/").text
        st = c.get("/api/state").json()
        assert st["mode"] == "idle" and st["fake"] and st["presets"]["15"].startswith("sound:") and st["info"]["camera"] == "synthetic"
        assert c.post("/api/reply", json={"text": "x"}).status_code == 409, "nothing is waiting for a reply"
        assert c.post("/api/ask", json={}).status_code == 400
        r = c.post("/api/ask", json={"request": "How bright is the scene?"})
        assert r.json() == {"ok": True}
        wait_for(lambda: c.get("/api/state").json()["mode"] == "paused", what="the pause")
        assert c.get("/api/state").json()["pending"]["kind"] == "scene_action"
        assert c.post("/api/ask", json={"request": "again"}).status_code == 409, "one activity at a time"
        assert c.post("/api/watch/start", json={}).status_code == 409
        assert c.post("/api/fake", json={"covered": False}).json()["ok"]
        assert c.post("/api/reply", json={"text": "done"}).json() == {"ok": True}
        fin = wait_for(lambda: next((e for e in c.get("/api/log").json()["events"] if e["type"] == "final"), None), what="final")
        assert "brightness" in fin["report"].lower() and fin["calls"] == 1
        wait_for(lambda: c.get("/api/state").json()["mode"] == "idle", what="idle")
        evs = c.get("/api/log").json()["events"]
        assert [e["id"] for e in evs] == list(range(1, len(evs) + 1))
        assert c.get(f"/api/log?since={len(evs) - 1}").json()["events"][0]["id"] == len(evs)
        tr = next(e for e in evs if e["type"] == "tool_result")
        assert tr["crops"] and c.get("/api/crop", params={"path": tr["crops"][0]}).status_code == 200
        assert c.get("/api/crop", params={"path": "../fake_state.json"}).status_code == 404
        assert c.get("/api/crop", params={"path": "instrument_log.jsonl"}).status_code == 404
        assert next(e for e in evs if e["type"] == "tool_call")["why"] == why
        assert c.post("/api/stop").json()["ok"]
        # a use case of run_hw3.py can be started by its number
        s.models = dict(orchestrator=Scripted([D("hardware", "x"), D("finish", "y")]), scientist=Scripted(), general=Scripted(),
                        hardware=Scripted([TC(("measure_region", {"region": "scene", "n_frames": 2})), AIMessage(content="ok")]), executor=Scripted())
        wait_for(lambda: c.get("/api/state").json()["mode"] == "idle")
        assert c.post("/api/ask", json={"use_case": "5"}).json() == {"ok": True}
        wait_for(lambda: sum(1 for e in c.get("/api/log").json()["events"] if e["type"] == "final") == 2, what="second final")
        assert any(e["type"] == "run_start" and e["request"] == "Is the image good?" for e in c.get("/api/log").json()["events"])
    finally:
        s.stop()
        bridge.close()
    ok("W3 web server: state, log, ask (also by use case number), reply (409 when nothing waits), one activity at a time, crops are safe, simulated faults, the page")


def test_fake_only_endpoint():
    from fastapi.testclient import TestClient
    tmp = Path(tempfile.mkdtemp())
    s, bridge = make_session(dict(orchestrator=Scripted(), scientist=Scripted(), general=Scripted(), hardware=Scripted(),
                                  executor=Scripted()), tmp)
    s.fake = False
    try:
        c = TestClient(web.create_app(s))
        assert c.post("/api/fake", json={"covered": True}).status_code == 409, "the simulated faults exist only with --fake"
    finally:
        bridge.close()
    ok("W3b the simulated-fault endpoint refuses to work on a real instrument")


def test_event_stream():
    """The page reads the log as a server-sent-event stream: tested against a real server (the test client does not stream)."""
    import socket
    import threading

    import httpx
    import uvicorn
    tmp = Path(tempfile.mkdtemp())
    models, _ = investigation_models()
    s, bridge = make_session(models, tmp)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = uvicorn.Server(uvicorn.Config(web.create_app(s), host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    try:
        wait_for(lambda: srv.started, what="the server")
        base = f"http://127.0.0.1:{port}"
        saved = dict(web.WATCH_LIMITS)
        web.WATCH_LIMITS["interval"] = (0.01, 600.0)                  # the limits are for the person at the page; the test needs a fast watch
        try:
            assert httpx.post(base + "/api/watch/start", json={"interval": 0.05}).json() == {"ok": True}
        finally:
            web.WATCH_LIMITS.update(saved)
        wait_for(lambda: sum(1 for e in s.hub.events if e["type"] == "sample") >= 3, what="samples")
        got = []
        with httpx.stream("GET", base + "/api/stream?since=0", timeout=10) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            for line in r.iter_lines():
                if line.startswith("data: "):
                    got.append(json.loads(line[6:]))
                if len(got) >= 5:
                    break
        assert [e["id"] for e in got] == [1, 2, 3, 4, 5] and got[0] == dict(got[0], type="mode", mode="watching")
        assert any(e["type"] == "sample" for e in got)
        assert httpx.get(base + "/favicon.ico").status_code == 204
    finally:
        s.stop()
        srv.should_exit = True
        wait_for(lambda: s.mode == "idle", what="idle")
        bridge.close()
    ok("W3c event stream: the page's stream delivers the log from the first event, on a real server")


# ============================================================================ W5 - W7
class SlowScripted(Scripted):
    """A scripted model that takes `delay` seconds per request, like a real one takes about 10."""

    def __init__(self, items=(), delay=0.5):
        super().__init__(items)
        self.delay = delay

    def invoke(self, messages):
        time.sleep(self.delay)
        return super().invoke(messages)


def three_round_models():
    orch = [D("hardware", "measure"), D("hardware", "measure again"), D("hardware", "and again"), D("finish", "done")]
    hw = []
    for i in range(3):
        hw += [TC(("measure_region", {"region": "scene", "n_frames": 2, "why": f"round {i}"})), AIMessage(content=f"round {i} done")]
    return dict(orchestrator=SlowScripted(orch), scientist=SlowScripted(), general=SlowScripted(), hardware=SlowScripted(hw),
                executor=SlowScripted())


def test_stop_ends_the_run():
    tmp = Path(tempfile.mkdtemp())
    s, bridge = make_session(three_round_models(), tmp)
    try:
        s.run_request("measure the scene three times")
        wait_for(lambda: any(e["type"] == "tool_call" for e in s.hub.events), what="the first instrument call")
        t0 = time.time()
        s.stop()
        wait_for(lambda: s.mode == "idle", timeout=30, what="idle after Stop")
        took = time.time() - t0
        n_calls = sum(1 for e in s.hub.events if e["type"] == "tool_call")
        assert n_calls < 3 and not any(e["type"] == "final" for e in s.hub.events), (n_calls, "the run went on to the end after Stop")
        assert any(e["type"] == "note" and e["text"] == "the run was stopped" for e in s.hub.events)
        assert took < 8.0, took
        runs = list((tmp / "runs").iterdir()) if (tmp / "runs").exists() else []
        assert runs == [], f"a stopped run leaves no empty folder: {runs}"
        # a new run can start at once (before the fix the old run still held the session: 'busy')
        s.models = three_round_models()
        s.run_request("measure the scene three times")
        wait_for(lambda: s.mode == "idle", timeout=60, what="the second run")
        assert any(e["type"] == "final" for e in s.hub.events), "the next run is complete"
    finally:
        s.stop()
        bridge.close()
    ok("W5 Stop ends a run after its current step, leaves no empty folder, and the next run can start")


def test_request_counter_is_per_run():
    tmp = Path(tempfile.mkdtemp())
    s, bridge = make_session(three_round_models(), tmp)
    try:
        counts = []
        for _ in range(2):
            s.models = three_round_models()
            n0 = len([e for e in s.hub.events if e["type"] == "final"])
            s.run_request("measure the scene three times")
            wait_for(lambda: len([e for e in s.hub.events if e["type"] == "final"]) > n0 and s.mode == "idle", timeout=60, what="a run")
            counts.append([e for e in s.hub.events if e["type"] == "final"][-1]["model_requests"])
        assert counts[0] == counts[1] == 10, counts                    # 4 orchestrator decisions + 3 x (a tool request + a report) = 10, every time
        runs = sorted((tmp / "runs").iterdir())
        assert len(runs) == 2
        for r in runs:
            assert json.loads((r / "run.json").read_text(encoding="utf-8"))["model_requests"] == 10
            assert "**model requests**: 10" in (r / "transcript.md").read_text(encoding="utf-8")
    finally:
        s.stop()
        bridge.close()
    ok("W6 the number of model requests counts one run: 10 and 10, not 10 and 20 (final line, run.json, transcript)")


def test_watch_settings_and_gitignore():
    import subprocess

    from fastapi.testclient import TestClient
    tmp = Path(tempfile.mkdtemp())
    models, _ = investigation_models()
    s, bridge = make_session(models, tmp)
    c = TestClient(web.create_app(s), raise_server_exceptions=False)
    try:
        for bad in ({"interval": "abc"}, {"interval": None}, {"interval": -1}, {"interval": 0}, {"interval": 1}, {"interval": 100000},
                    {"interval": "nan"}, {"interval": 4, "tone_every": -5}, {"interval": 4, "tone_every": 3}, {"interval": 4, "tone_every": "x"}):
            r = c.post("/api/watch/start", json=bad)
            assert r.status_code == 400 and r.json()["ok"] is False and r.json()["error"], (bad, r.status_code, r.text)
        assert s.mode == "idle" and not s.hub.events, "a refused setting starts nothing"
        r = c.post("/api/watch/start", json={"interval": 2, "tone_every": 0})
        assert r.status_code == 200 and r.json() == {"ok": True}, r.text
        assert c.post("/api/watch/start", json={"interval": 4}).status_code == 409, "already watching"
        s.stop()
        wait_for(lambda: s.mode == "idle", what="idle")
        r = c.post("/api/watch/start", json={"interval": 4, "tone_every": 10})
        assert r.status_code == 200, r.text
        s.stop()
        wait_for(lambda: s.mode == "idle", what="idle")
    finally:
        s.stop()
        bridge.close()
    # the .gitignore of this folder hides the pictures that the page copies into runs/, but not the reports
    repo = Path(tempfile.mkdtemp())
    (repo / "hw3" / "runs" / "20261017-141345-alert" / "images").mkdir(parents=True)
    (repo / "hw3" / ".gitignore").write_text((Path(__file__).resolve().parent / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8")
    for rel in ("images/03_capture_004_scene.png", "report.md"):
        (repo / "hw3" / "runs" / "20261017-141345-alert" / rel).write_text("x", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    ign = lambda rel: subprocess.run(["git", "check-ignore", "-q", rel], cwd=repo).returncode == 0
    assert ign("hw3/runs/20261017-141345-alert/images/03_capture_004_scene.png"), "pictures of the runs stay out of git"
    assert not ign("hw3/runs/20261017-141345-alert/report.md"), "the report can be committed"
    ok("W7 page settings: letters, negative and too small numbers are refused with a sentence, good ones start the watch; the pictures of runs are git-ignored")


# ============================================================================ W4
def test_why_in_specs():
    tmp = Path(tempfile.mkdtemp())
    env = default_env(str(tmp / "work"), roi_file=str(make_roi(tmp)), fake=True, controls_file=str(tmp / "no.json"))
    b = McpBridge(env=env)
    b.start()
    try:
        plain = {s["function"]["name"]: s["function"]["parameters"] for s in b.tool_specs()}
        with_why = {s["function"]["name"]: s["function"]["parameters"] for s in A.with_why(b.tool_specs())}
        assert all("why" not in p.get("properties", {}) for p in plain.values()), "the server does not know `why`"
        assert all("why" in p["properties"] and "why" not in p.get("required", []) for p in with_why.values())
        assert with_why["set_resolution"]["required"] == ["width", "height"]
        assert A.with_why(b.tool_specs()) == A.with_why(b.tool_specs()) and "why" not in b.tool_specs()[0]["function"]["parameters"]["properties"]
    finally:
        b.close()
    ok("W4 `why` is an optional argument of every tool for the model only; the server's own tool list is unchanged")


if __name__ == "__main__":
    test_watchdog_logic()
    test_why_in_specs()
    test_watch_loop_catches_a_covered_lens()
    test_mute_and_noise_alerts()
    test_tone_check()
    test_web_server()
    test_fake_only_endpoint()
    test_event_stream()
    test_stop_ends_the_run()
    test_request_counter_is_per_run()
    test_watch_settings_and_gitignore()
    print(f"\nall {len(PASSED)} checks passed")
