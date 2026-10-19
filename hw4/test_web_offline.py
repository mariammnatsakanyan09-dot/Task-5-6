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
W11 reference file     the reference state is saved to a file, read at the next start; a differing state asks a person (accept / keep);
                       a damaged file is ignored aloud; only a person's "accept" rewrites it
W12 guard mode         a change of the room (somebody enters, the light, the noise, a warm shift) with the instrument tested and in order is a
                       ROOM ALERT without any model; a covered lens or a quieter loudspeaker is the loop of the agent, not a room alert
W13 notification       the alert goes to alerts.log and to a push address; a dead address does not stop the others
W14 the whole path     fake camera -> MCP server -> watchdog -> page: a person enters, the ROOM ALERT comes with a picture, no model is called
W15 guard hours        people are normal by day and suspicious at night: the room is watched only inside the window, the room of the night
                       (not of the day) is the reference, the evening lights-off and the morning lights-on are no alarm, a restart in the
                       same night remembers the reference, a question that nobody answers does not stop the watch for ever
W16 smoke detector     an alarm of an outside sensor is announced at once, without a model and without the camera; a secret is needed; a
                       repeated message is not a new alarm; the synthetic detector goes through the same door
W17 summary, report  one CSV line a minute, the night report with limits from the quiet minutes, rotation of the big log, deletion of old pictures
W18 hints file        the words for the agent are replaceable without touching the code; a mistake changes nothing
W19 steady room       ESCALATION at once when only the light or the noise is wrong and two checks gave the same numbers
W20 pushes, health    a lost camera / ESCALATION / the return of the instrument are pushed; health() says when the heartbeat must stop
W21 night guard       arming push, "I am in the lab" pause, no beep at night, a dead microphone is a fault, a room that is not last night's
W22 pause endpoint    POST /api/guard/postpone: 0-720 minutes, wrong input refused, 409 when the guard is not running
W23 heartbeat, crash  pings only while healthy, address never shown, an unreachable service told once, a crash of the loop pushed
W24 signed sensor     HMAC of timestamp+body; copied, changed, old and future messages refused; X-Token unless --sensor-signed-only
W25 scripts and CLI   sensor_send.py against a real door; web_app.py started as a person starts it (refusals, then everything on at once)
W7  page settings      a letter, a negative or a too small number is refused with a sentence (HTTP 400); good numbers start the watch;
                       the .gitignore keeps the copied pictures of the runs out of git
"""
import json
import os
import sys
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
    fell = W.alert_request(["scene brightness 73 -> 53 grey levels"], base, None)
    rose = W.alert_request(["scene brightness 44 -> 86 grey levels"], base, None)
    assert "in front of the camera lens" in fell and "BEFORE you conclude that the light changed" in fell, "a brightness fall: the person is asked to remove an obstacle first"
    assert "in front of the camera lens" not in rose and "brightness ROSE" in rose, "a brightness rise: the light changed, nobody is asked"
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
def make_session(models, tmp, tone_check=False, **session_kw):
    work = tmp / "work"
    cam_state, aud_state = work / "fake_state.json", work / "fake_audio_state.json"
    env = default_env(str(work), roi_file=str(make_roi(tmp)), fake=True, fake_state=str(cam_state),
                      controls_file=str(tmp / "no_controls.json"), fake_audio=True, fake_audio_state=str(aud_state))
    bridge = McpBridge(env=env)
    bridge.start()
    sh = web.SharedBridge(bridge)
    s = web.Session(models, sh, work, tmp / "runs", info=dict(camera="synthetic"), model_name="scripted", fake=True, **session_kw)
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
    n_inv = W.MAX_RETRIES + 1                                          # the loop may call the agent again while the tone is still weak
    s, bridge = make_session(dict(orchestrator=Scripted([D("hardware", "x"), D("finish", "y")] * n_inv), scientist=Scripted(),
                                  general=Scripted(), hardware=Scripted([TC(("listen_noise", {"seconds": 1.0})), AIMessage(content="ok")] * n_inv),
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
        wait_for(lambda: any(e["type"] == "note" and "ESCALATION" in e["text"] for e in s.hub.events), timeout=120, what="the loop gives up with an ESCALATION: the tone stays weak")
        wait_for(lambda: s.pending is not None, what="after the ESCALATION the program asks whether this state may become the new normal")
        assert "reference" in s.pending["question"] and "accept" in s.pending["question"], s.pending
        s.reply("keep")                                                # not accepted: the reference state stays
        off = wait_for(lambda: next((e for e in s.hub.events if e["type"] == "note" and "tone check is off" in e["text"]), None),
                       what="the check tone is tuned again after the investigation (and found too weak)")
        assert "-12 dBFS:" in off["text"] and "dB" in off["text"] and "at least 10 dB above the noise" in off["text"], off
        W.RECAL_S = 0.1                                                # the volume is back: the switched-off check comes back
        s.fake_set(volume_db=0.0)                                      # the person turned the volume up when the program asked
        wait_for(lambda: s.pending is not None, what="the program asks the person about the lost tone (no model)")
        assert "check tone is not heard" in s.pending["question"], s.pending
        s.reply("done")
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


# ============================================================================ W8
def test_loop_closes_or_escalates():
    """The loop is closed on the PROGRAM's own measurement: fixed at once -> one investigation; not fixed -> the agent is called again
    with the numbers of the program's check; still not fixed -> an ESCALATION note (a person is needed), then the watch goes on."""
    import threading

    def run_case(fix_at, wait_what, drift=0.0):
        room = dict(noise=-52.0)
        notes, reqs = [], []

        def call(name, args):
            if name == "measure_region":
                return json.dumps(dict(ok=True, mean_brightness=150.0, sharpness_norm=900.0, saturated_fraction=0.0))
            if name == "listen_noise":
                return json.dumps(dict(ok=True, noise_rms_db=room["noise"], silent_input=False))
            return json.dumps(dict(ok=False, error="not offered"))

        def investigate(req, facts):
            reqs.append(req)
            if len(reqs) == fix_at:
                room["noise"] = -52.0                                # the room is quiet again
            elif drift:
                room["noise"] += drift                               # the room does not stay the same between two checks

        wd = W.Watchdog(call, ["measure_region", "listen_noise"], notes.append, investigate, interval=0.0)
        stop = threading.Event()
        th = threading.Thread(target=wd.run, args=(stop,), daemon=True)
        th.start()
        try:
            wait_for(lambda: wd.base.ready(), what="the first normal values")
            room["noise"] = -35.0                                    # the room gets louder and stays so
            wait_for(lambda: wait_what(notes), what="the end of the investigation(s)")
            time.sleep(0.4)                                          # nothing more is started: the new state is the new normal
        finally:
            stop.set()
            th.join(10)
        return reqs, [n.get("text", "") for n in notes if n["type"] == "note"], sum(1 for n in notes if n["type"] == "anomaly")

    ends = lambda notes: any(("back inside the limits" in n.get("text", "") and "NOT" not in n.get("text", "")) or "ESCALATION" in n.get("text", "")
                             for n in notes if n["type"] == "note")
    # 1. the first investigation fixes it: one call, no retry
    reqs, texts, anomalies = run_case(1, ends)
    assert len(reqs) == 1 and anomalies == 1 and not any("NOT back" in t or "ESCALATION" in t for t in texts), (len(reqs), texts)
    # 2. not fixed by the first, fixed by the second: the second request carries the numbers of the program's check
    reqs, texts, anomalies = run_case(2, ends)
    assert len(reqs) == 2 and anomalies == 2, (len(reqs), anomalies, texts)
    assert reqs[1].startswith("NEXT INVESTIGATION") and "still NOT back inside the limits: room noise" in reqs[1] and not reqs[0].startswith("NEXT"), reqs[1][:300]
    assert any("NOT back inside the limits" in t for t in texts) and any(f"investigation 2 of {W.MAX_RETRIES + 1}" in t for t in texts), texts
    assert not any("ESCALATION" in t for t in texts), texts
    assert any(t.endswith("limits of the alert") or "back inside the limits of the alert" in t for t in texts), texts
    # 3. never fixed and the numbers keep moving (3 dB between checks): exactly MAX_RETRIES + 1 investigations, then a loud ESCALATION
    reqs, texts, anomalies = run_case(99, ends, drift=3.0)
    assert len(reqs) == W.MAX_RETRIES + 1 and anomalies == W.MAX_RETRIES + 1, (len(reqs), anomalies)
    esc = [t for t in texts if t.startswith("ESCALATION")]
    assert len(esc) == 1 and "a person has to look" in esc[0] and "room noise" in esc[0] and "two checks in a row" not in esc[0], texts
    # 4. never fixed and the room is STEADY (the same noise in two checks in a row): an agent cannot make the room quiet -> ESCALATION after 2
    reqs, texts, anomalies = run_case(99, ends)
    assert len(reqs) == 2 and anomalies == 2, (len(reqs), anomalies)
    esc = [t for t in texts if t.startswith("ESCALATION")]
    assert len(esc) == 1 and "two checks in a row gave the same numbers" in esc[0] and "after 2 investigation(s)" in esc[0], texts
    ok("W8 the loop closes on the program's own check: fixed -> done; not fixed -> the agent is called again with those numbers; still not -> an ESCALATION note (after 2 when the room is steady), no endless calls")


# ============================================================================ W9
def test_person_is_asked_when_the_tone_is_lost():
    """The check tone is not heard at any level: the PROGRAM (no model) asks the person, calibrates after each answer, and says ESCALATION
    when it is still not heard; it asks once per episode, not every RECAL_S seconds."""
    import threading

    def run_case(fix_at):
        room, notes, asked = dict(fixed=False), [], []

        def call(name, args):
            if name == "play_tone":
                if room["fixed"]:
                    return json.dumps(dict(ok=True, tone_db=-40.0, noise_db=-65.0, snr_db=25.0, heard=True, interference=False,
                                           silent_input=False, coupling_db=-20.0))
                return json.dumps(dict(ok=True, tone_db=-60.0, noise_db=-58.0, snr_db=2.0, heard=False, interference=False,
                                       silent_input=False, coupling_db=-40.0))
            return json.dumps(dict(ok=True, mean_brightness=150.0, sharpness_norm=900.0, saturated_fraction=0.0, noise_rms_db=-52.0, silent_input=False))

        def ask(q):
            asked.append(q)
            if len(asked) == fix_at:
                room["fixed"] = True
            return "done"

        wd = W.Watchdog(call, ["measure_region", "listen_noise", "play_tone"], notes.append, lambda r, f: None, interval=0.0, tone_every=0.001, ask=ask)
        wd.calibrate_tone()
        wd.secure_tone(threading.Event())
        wd.secure_tone(threading.Event())                            # a second call in the same episode asks nobody
        return wd, asked, [n.get("text", "") for n in notes if n["type"] == "note"]

    wd, asked, texts = run_case(2)
    assert len(asked) == 2 and "Pano-360" in asked[0] and "Still not heard" in asked[1], asked
    assert wd.tone is not None and not any("ESCALATION" in t for t in texts), texts
    wd, asked, texts = run_case(99)
    assert len(asked) == W.TONE_ASKS and wd.tone is None, (len(asked), wd.tone)
    esc = [t for t in texts if t.startswith("ESCALATION")]
    assert len(esc) == 1 and "NOT watched" in esc[0], texts
    ok("W9 a lost check tone: the program asks the person (no model) up to 3 times, calibrates after each answer, then says ESCALATION; once per episode")


# ============================================================================ W10
def test_reference_state_is_not_forgotten():
    """After an ESCALATION an unfixable problem does NOT silently become the normal: a person is asked. 'accept' -> the reference is measured
    again at the new state; 'keep' -> the normal values go back to the reference state, no new alarm for ESC_QUIET_S, nothing is forgotten."""
    import threading

    def run_case(answer):
        room = dict(noise=-52.0)
        notes, asked = [], []

        def call(name, args):
            if name == "measure_region":
                return json.dumps(dict(ok=True, mean_brightness=150.0, sharpness_norm=900.0, saturated_fraction=0.0))
            if name == "listen_noise":
                return json.dumps(dict(ok=True, noise_rms_db=room["noise"], silent_input=False))
            return json.dumps(dict(ok=False, error="not offered"))

        def ask(q):
            asked.append(q)
            return answer

        wd = W.Watchdog(call, ["measure_region", "listen_noise"], notes.append, lambda r, f: None, interval=0.0, ask=ask)
        stop = threading.Event()
        th = threading.Thread(target=wd.run, args=(stop,), daemon=True)
        th.start()
        try:
            wait_for(lambda: wd.ref is not None, what="the reference state is taken at the start")
            ref_noise = wd.ref.med("noise_db")
            room["noise"] = -35.0                                    # the room gets louder and stays so
            wait_for(lambda: asked, timeout=60, what="after the ESCALATION the person is asked")
            time.sleep(0.6)
        finally:
            stop.set()
            th.join(10)
        return wd, asked, ref_noise, [n.get("text", "") for n in notes if n["type"] == "note"], sum(1 for n in notes if n["type"] == "anomaly")

    # 1. kept: the reference stays, the watch goes on against it, and no new alarm starts during the quiet time
    wd, asked, ref_noise, texts, anomalies = run_case("keep")
    assert len(asked) == 1 and "reference" in asked[0] and "accept" in asked[0], asked
    assert abs(wd.ref.med("noise_db") - ref_noise) < 0.01 and abs(wd.base.med("noise_db") - ref_noise) < 0.01, (wd.ref.med("noise_db"), wd.base.med("noise_db"))
    assert anomalies == 2 and any("did not accept" in t for t in texts), (anomalies, texts)     # the room is steady: ESCALATION after 2 checks, not after 6
    assert any("two checks in a row gave the same numbers" in t for t in texts), texts
    # 2. accepted: the new state is measured as the reference
    wd, asked, ref_noise, texts, anomalies = run_case("accept")
    assert len(asked) == 1 and any("accepted this state" in t for t in texts), texts
    assert wd.ref is not None and abs(wd.ref.med("noise_db") + 35.0) < 0.5, wd.ref.med("noise_db")
    ok("W10 an unfixable problem does not silently become the normal: a person is asked; 'keep' keeps the reference, 'accept' measures a new one")


# ============================================================================ W11-W14: reference file, guard mode
class Room:
    """A synthetic instrument for the watchdog (no MCP server): the room is a dict that the test changes while the watchdog runs."""
    BASE = [90.0 + 6.0 * (i % 16) + 4.0 * (i // 16) for i in range(144)]            # the coarse picture of an empty room, 16 x 9 cells
    PERSON = {r * 16 + c for r in (3, 4, 5, 6) for c in (8, 9)}                       # a figure of 8 cells = 5.6 % of the picture

    def __init__(self, **kw):
        self.v = dict(light=1.0, person=False, covered=False, noise=-52.0, sharp=900.0, warm=1.6, silent=False, coupling=-40.0, jitter=0.0)
        self.v.update(kw)
        self.calls = []
        self.regions = set()
        self.n_noise = 0

    def __call__(self, name, args):
        v = self.v
        self.calls.append(name)
        if name == "measure_region":
            self.regions.add(args.get("region"))
            t = [x * v["light"] for x in self.BASE]
            if v["person"]:
                t = [15.0 if i in self.PERSON else x for i, x in enumerate(t)]
            if v["covered"]:
                t = [x * 0.03 for x in t]
            return json.dumps(dict(ok=True, mean_brightness=sum(t) / len(t), sharpness_norm=3.0 if v["covered"] else v["sharp"],
                                   saturated_fraction=0.0, thumb=t, warm_ratio=v["warm"]))
        if name == "listen_noise":
            self.n_noise += 1
            wobble = v["jitter"] * ((self.n_noise % 7) - 3)                             # a live microphone never gives the same number 8 times
            return json.dumps(dict(ok=True, noise_rms_db=v["noise"] + wobble, silent_input=v["silent"]))
        if name == "play_tone":
            return json.dumps(dict(ok=True, snr_db=25.0, heard=True, interference=False, silent_input=False,
                                   coupling_db=v["coupling"], tone_db=v["coupling"] + float(args["level_db"])))
        if name == "capture":
            return json.dumps(dict(ok=True, saved_crops=["crops/capture_001_scene.png"]))
        return json.dumps(dict(ok=False, error="not offered"))


def run_watchdog(room, names=("measure_region", "listen_noise", "capture"), ask=None, **kw):
    """Start a watchdog on a Room in a thread. Returns (watchdog, events, investigations, stop, thread)."""
    import threading
    events, inv = [], []
    wd = W.Watchdog(room, list(names), events.append, lambda r, f: inv.append(f), interval=0.0, ask=ask, **kw)
    stop = threading.Event()
    th = threading.Thread(target=wd.run, args=(stop,), daemon=True)
    th.start()
    return wd, events, inv, stop, th


def end(stop, th):
    stop.set()
    th.join(10)


def kinds(events, t):
    return [e for e in events if e["type"] == t]


def notes_of(events):
    return [e.get("text", "") for e in events if e["type"] == "note"]


def test_scene_change_and_room_facts():
    base = Room.BASE
    assert W.scene_change(base, list(base)) == 0.0
    assert W.scene_change(base, [x * 1.5 for x in base]) == 0.0, "a lamp that is switched on changes every cell by the same factor: that is the light, not a movement"
    assert W.scene_change(base, [x * 0.5 for x in base]) == 0.0
    person = [15.0 if i in Room.PERSON else x for i, x in enumerate(base)]
    ch = W.scene_change(base, person)
    assert ch is not None and ch >= W.SCENE_CHANGE and abs(ch - 8 / 144) < 1e-9, ch
    assert W.scene_change(None, person) is None and W.scene_change(base, base[:10]) is None and W.scene_change([0.0] * 144, base) is None
    ref = W.Normal()
    for _ in range(W.KEEP):
        ref.add(dict(brightness=150.0, sharpness=900.0, noise_db=-52.0, warm=1.6))
    s0 = dict(brightness=150.0, sharpness=900.0, noise_db=-52.0, warm=1.6, change=0.0)
    assert W.judge_room(ref, s0) == []
    assert any("picture of the room changed" in f for f in W.judge_room(ref, dict(s0, change=0.1)))
    assert W.judge_room(ref, dict(s0, change=0.03)) == [], "3 % is below the limit"
    assert any("warmer" in f for f in W.judge_room(ref, dict(s0, warm=2.2)))
    assert W.judge_room(ref, dict(s0, warm=1.2)) == [], "a picture that gets colder is no flame"
    assert any("room noise" in f for f in W.judge_room(ref, dict(s0, noise_db=-40.0)))
    assert any("light in the room changed" in f for f in W.judge_room(ref, dict(s0, brightness=230.0)))
    assert W.judge_room(ref, dict(s0, noise_db=-99.0, silent=True)) == [], "pure silence is a fault of the microphone, not a room event"
    # the instrument-only view leaves out exactly the two readings that can also be the room
    b = W.Normal()
    for _ in range(W.KEEP):
        b.add(dict(brightness=150.0, sharpness=900.0, noise_db=-52.0))
    loud_dark = dict(brightness=60.0, sharpness=900.0, noise_db=-40.0, errors=[])
    assert len(W.judge_cats(b, loud_dark)) == 2 and W.judge_cats(b, loud_dark, instrument_only=True) == []
    assert W.judge_cats(b, dict(loud_dark, sharpness=5.0), instrument_only=True), "a collapse of sharpness stays an instrument fault"
    assert W.judge_cats(b, dict(loud_dark, silent=True), instrument_only=True), "a muted microphone stays an instrument fault"
    # the reference file keeps what it must and refuses what it must
    r2 = W.Normal.from_dict(json.loads(json.dumps(W.Normal.to_dict(ref))))
    assert r2.ready() and abs(r2.med("brightness") - 150.0) < 0.01 and abs(r2.med("warm") - 1.6) < 0.01
    for bad in ({}, {"values": {}}, {"values": {"brightness": "x"}}, {"values": 5}):
        try:
            W.Normal.from_dict(bad)
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        raise AssertionError(f"a damaged reference must be refused: {bad}")
    ok("W11a coarse picture: a lamp is the light, a figure is a change; room facts have numbers; the reference file keeps its values")


def test_reference_file():
    tmp = Path(tempfile.mkdtemp())
    path = tmp / "reference.json"
    # 1. first run: no file -> the first normal values are the reference and are saved
    room = Room()
    wd, ev, inv, stop, th = run_watchdog(room, ref_path=path)
    try:
        wait_for(lambda: path.exists(), what="the reference file")
    finally:
        end(stop, th)
    d = json.loads(path.read_text(encoding="utf-8"))
    assert abs(d["values"]["noise_db"] + 52.0) < 0.5 and len(d["thumb"]) == 144 and d["saved_at"], d
    assert any("no reference file yet" in t for t in notes_of(ev)) and any("saved to reference.json" in t for t in notes_of(ev))
    first = path.read_text(encoding="utf-8")
    # 2. second run, same room: the stored reference is kept and NOT rewritten
    wd, ev, inv, stop, th = run_watchdog(Room(), ref_path=path)
    try:
        wait_for(lambda: wd.ref is not None, what="the reference")
        time.sleep(0.3)
    finally:
        end(stop, th)
    assert any("agrees with the stored reference" in t for t in notes_of(ev)), notes_of(ev)
    assert path.read_text(encoding="utf-8") == first, "an agreeing state does not rewrite the file"
    # 3. third run, the room is louder than the stored reference, the person says keep: the stored reference stays and counts
    asked = []

    def keep(q):
        asked.append(q)
        return "keep"
    wd, ev, inv, stop, th = run_watchdog(Room(noise=-35.0), ref_path=path, ask=keep)
    try:
        wait_for(lambda: asked, what="the question at the start")
        wait_for(lambda: kinds(ev, "anomaly"), what="the difference to the stored reference is an alarm")
    finally:
        end(stop, th)
    assert "stored on" in asked[0] and "room noise" in asked[0] and "accept" in asked[0], asked[0]
    assert abs(wd.ref.med("noise_db") + 52.0) < 0.5, "the stored reference stays the reference"
    assert path.read_text(encoding="utf-8") == first, "'keep' leaves the file as it was"
    # 4. the same, the person says accept: the present state replaces the file
    wd, ev, inv, stop, th = run_watchdog(Room(noise=-35.0), ref_path=path, ask=lambda q: "accept")
    try:
        wait_for(lambda: json.loads(path.read_text(encoding="utf-8"))["values"]["noise_db"] > -40, what="the new reference in the file")
        time.sleep(0.3)
    finally:
        end(stop, th)
    assert not kinds(ev, "anomaly") and abs(wd.ref.med("noise_db") + 35.0) < 0.5
    # 5. nobody to ask: the stored reference is kept (the file is never overwritten without a person)
    path.write_text(first, encoding="utf-8")
    wd, ev, inv, stop, th = run_watchdog(Room(noise=-35.0), ref_path=path, ask=None)
    try:
        wait_for(lambda: kinds(ev, "anomaly"), what="alarm against the stored reference")
    finally:
        end(stop, th)
    assert path.read_text(encoding="utf-8") == first, "nobody to ask: the file is never overwritten (in memory the run goes on, as before)"
    assert any("noise" in f for f in kinds(ev, "anomaly")[0]["facts"]) and any("nobody to ask" in t for t in notes_of(ev))
    # 6. a damaged file is said aloud and replaced by a new reference; --fresh-reference ignores a good file on purpose
    path.write_text("{not json", encoding="utf-8")
    wd, ev, inv, stop, th = run_watchdog(Room(), ref_path=path)
    try:
        wait_for(lambda: json.loads(path.read_text(encoding="utf-8")).get("values") if path.read_text(encoding="utf-8").startswith("{\n") else None,
                 what="a new reference replaces the damaged file")
    finally:
        end(stop, th)
    assert any("cannot be used" in t for t in notes_of(ev))
    wd, ev, inv, stop, th = run_watchdog(Room(noise=-35.0), ref_path=path, ask=lambda q: 1 / 0, fresh_reference=True)
    try:
        wait_for(lambda: json.loads(path.read_text(encoding="utf-8"))["values"]["noise_db"] > -40, what="a fresh reference replaces the file")
    finally:
        end(stop, th)
    assert any("fresh reference" in t for t in notes_of(ev))
    # 6b. the file knows which part of the picture it is about: another region is another reference
    path.write_text(first, encoding="utf-8")
    assert json.loads(first)["region"] == "scene"
    room = Room()
    wd, ev, inv, stop, th = run_watchdog(room, ref_path=path, region="main_view")
    try:
        wait_for(lambda: json.loads(path.read_text(encoding="utf-8")).get("region") == "main_view", what="a reference of the other region")
    finally:
        end(stop, th)
    assert any("is of the region 'scene'" in t for t in notes_of(ev)) and room.regions == {"main_view"}, (notes_of(ev), room.regions)
    # 7. an unfixable problem after the ESCALATION: "keep" leaves the file, "accept" rewrites it
    for answer in ("keep", "accept"):
        path.unlink()
        room = Room()
        wd, ev, inv, stop, th = run_watchdog(room, ref_path=path, ask=lambda q, a=answer: a)
        try:
            wait_for(lambda: path.exists() and wd.ref is not None, what="reference")
            room.v["noise"] = -35.0
            if answer == "keep":
                wait_for(lambda: any("did not accept" in t for t in notes_of(ev)), timeout=60, what="ESCALATION with keep")
                time.sleep(0.3)
                assert json.loads(path.read_text(encoding="utf-8"))["values"]["noise_db"] < -50, "kept: the file still holds the quiet room"
            else:
                wait_for(lambda: json.loads(path.read_text(encoding="utf-8"))["values"]["noise_db"] > -40, timeout=60,
                         what="ESCALATION with accept rewrites the file")
        finally:
            end(stop, th)
    ok("W11b reference file: saved at the first run, kept when it agrees, a person decides when it differs (keep = file unchanged, accept = rewritten), "
       "no person = never overwritten, damaged file ignored aloud, --fresh-reference on purpose")


def test_guard_mode():
    def guard_run(change, **kw):
        room = Room()
        alerts = []
        wd, ev, inv, stop, th = run_watchdog(room, guard=True, notify=lambda t: (alerts.append(t), ["stub"])[1], **kw)
        try:
            wait_for(lambda: wd.ref is not None and sum(1 for e in ev if e["type"] == "sample" and e["status"] == "normal") >= 3, what="a quiet start")
            assert not kinds(ev, "room_alert") and not kinds(ev, "anomaly"), "an empty, quiet room is no alarm"
            change(room)
            wait_for(lambda: kinds(ev, "room_alert") or kinds(ev, "anomaly"), timeout=60, what="the alarm")
            time.sleep(0.3)
        finally:
            end(stop, th)
        return wd, ev, inv, alerts

    # a person enters: the instrument is in order, so it is a ROOM ALERT, with a picture, a message, and without a model
    wd, ev, inv, alerts = guard_run(lambda r: r.v.update(person=True))
    ra = kinds(ev, "room_alert")
    assert len(ra) == 1 and not kinds(ev, "anomaly") and inv == [], "no model, no agent, exactly one alert (the quiet time holds the next one)"
    assert "picture of the room changed" in ra[0]["text"] and "passed its own test" in ra[0]["text"], ra[0]["text"]
    assert ra[0]["evidence"] == ["crops/capture_001_scene.png"] and ra[0]["notified"] == ["stub"] and alerts == [ra[0]["text"]]
    assert all("thumb" not in e for e in kinds(ev, "sample")), "the coarse grid is not sent to the page"
    assert max(e.get("change") or 0 for e in kinds(ev, "sample")) >= W.SCENE_CHANGE
    # the lamp is switched on: the light changed, nothing moved
    wd, ev, inv, alerts = guard_run(lambda r: r.v.update(light=1.6))
    ra = kinds(ev, "room_alert")
    assert ra and "light in the room changed" in ra[0]["text"] and "picture of the room changed" not in ra[0]["text"] and inv == [], ra
    # a flame colour
    wd, ev, inv, alerts = guard_run(lambda r: r.v.update(warm=2.2))
    ra = kinds(ev, "room_alert")
    assert ra and "warmer" in ra[0]["text"] and inv == [], ra
    # the room gets loud while the instrument is in order: in guard mode this is the room, without guard mode it is the loop of the agent
    wd, ev, inv, alerts = guard_run(lambda r: r.v.update(noise=-35.0))
    ra = kinds(ev, "room_alert")
    assert ra and "room noise" in ra[0]["text"] and inv == [] and not kinds(ev, "anomaly"), ra
    room = Room()
    wd, ev, inv, stop, th = run_watchdog(room)
    try:
        wait_for(lambda: wd.ref is not None, what="reference")
        room.v.update(person=True, noise=-35.0)
        wait_for(lambda: kinds(ev, "anomaly"), what="the old behaviour")
    finally:
        end(stop, th)
    assert any("room noise" in f for f in kinds(ev, "anomaly")[0]["facts"]) and not kinds(ev, "room_alert"), "without guard mode a person in the room is no event"
    # a covered lens is the instrument: the loop of the agent starts, not a room alert (the picture changed as well)
    wd, ev, inv, alerts = guard_run(lambda r: r.v.update(covered=True))
    assert kinds(ev, "anomaly") and not kinds(ev, "room_alert"), "a covered lens is no burglar"
    assert all(a.startswith("WATCH TROUBLE") for a in alerts), alerts          # the only messages out are about the watch itself (the loop could not fix it), not a ROOM ALERT
    assert any("sharpness" in f for f in kinds(ev, "anomaly")[0]["facts"])
    # a muted microphone is the instrument too
    wd, ev, inv, alerts = guard_run(lambda r: r.v.update(silent=True, noise=-99.0))
    assert kinds(ev, "anomaly") and not kinds(ev, "room_alert") and any("silence" in f for f in kinds(ev, "anomaly")[0]["facts"])
    # a person enters AND the loudspeaker is quieter: the program tests the instrument before it says ROOM ALERT, finds the fault, starts the loop
    import threading
    room = Room()                   # the check tone is played at the start and then only by the self-test of a room alert (tone_every is an hour)
    events, inv = [], []
    wd = W.Watchdog(room, ["measure_region", "listen_noise", "capture", "play_tone"], events.append, lambda r, f: inv.append(f), interval=0.0,
                    tone_every=3600.0, guard=True, notify=lambda t: ["stub"])
    stop = threading.Event()
    th = threading.Thread(target=wd.run, args=(stop,), daemon=True)
    th.start()
    try:
        wait_for(lambda: wd.ref is not None and wd.tone is not None, what="reference and check tone")
        time.sleep(0.3)
        room.v.update(person=True, coupling=-52.0)
        wait_for(lambda: kinds(events, "anomaly") or kinds(events, "room_alert"), timeout=60, what="the alarm")
    finally:
        end(stop, th)
    assert kinds(events, "anomaly") and not kinds(events, "room_alert"), "the instrument is not in order: no room alert"
    assert any("not in order itself" in t for t in notes_of(events)) and any("check tone" in f for f in kinds(events, "anomaly")[0]["facts"])
    ok("W12 guard mode: a person, a lamp, a flame colour and a loud room are ROOM ALERTs (tested instrument, picture as evidence, no model); "
       "a covered lens, a muted microphone and a quieter loudspeaker are the loop of the agent; without guard mode a person in the room is no event")


def test_notifier():
    import http.server
    import threading
    got = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            got.append((self.path, self.rfile.read(n).decode("utf-8"), self.headers.get("Title")))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    old = os.environ.get("NO_PROXY")
    os.environ["NO_PROXY"] = "127.0.0.1"
    tmp = Path(tempfile.mkdtemp())
    try:
        notify = W.make_notifier(tmp / "alerts.log", f"http://127.0.0.1:{srv.server_address[1]}/secret-topic")
        done = notify("ROOM ALERT: test\nsecond line")
        assert got and got[0][0] == "/secret-topic" and got[0][1].startswith("ROOM ALERT: test") and got[0][2] == "Lab Watchdog ROOM ALERT", got
        assert any("alerts.log" in d for d in done) and any("push message sent" in d for d in done) and "secret-topic" not in " ".join(done), done
        line = (tmp / "alerts.log").read_text(encoding="utf-8")
        assert "ROOM ALERT: test second line" in line and line.count("\n") == 1, line
        dead = W.make_notifier(tmp / "alerts2.log", "http://127.0.0.1:9/nothing")
        done = dead("ROOM ALERT: x")
        assert any("FAILED" in d for d in done) and (tmp / "alerts2.log").exists(), "a dead address does not stop the log line"
        nothing = W.make_notifier(None, None)
        assert nothing("x") in ([], ["notification of the system shown"]), "no way configured: nothing is sent"
    finally:
        srv.shutdown()
        if old is None:
            os.environ.pop("NO_PROXY", None)
        else:
            os.environ["NO_PROXY"] = old
    ok("W13 notification: one line in alerts.log, a POST to the push address (the address is never printed), a dead address does not stop the rest")


def test_camera_grid():
    import camera_connector as cc
    tmp = Path(tempfile.mkdtemp())
    st = tmp / "state.json"
    roi = make_roi(tmp)
    cam = cc.PanoCamera(cc.FakeBackend(0, state_file=str(st)), 0, tmp, True)
    cam.roi_file = roi
    plain = cam.measure_region("scene", 2)
    assert "thumb" not in plain and "warm_ratio" not in plain, "the agent's measure_region does not carry the grid"
    r0 = cam.measure_region("scene", 3, True)
    assert len(r0["thumb"]) == cc.THUMB_W * cc.THUMB_H == 144 and r0["warm_ratio"] > 0

    def now(state):
        st.write_text(json.dumps(state))
        return cam.measure_region("scene", 3, True)
    same, person, lamp, warm, cover = now({}), now({"person": True}), now({"light": 1.4}), now({"warm": 1.15}), now({"covered": True})
    assert W.scene_change(r0["thumb"], same["thumb"]) == 0.0
    assert W.scene_change(r0["thumb"], person["thumb"]) >= W.SCENE_CHANGE, "a figure in the picture is seen"
    assert abs(person["mean_brightness"] - r0["mean_brightness"]) < max(W.BRIGHT_ABS, W.BRIGHT_REL * r0["mean_brightness"]), "...although the brightness alone says nothing"
    assert W.scene_change(r0["thumb"], lamp["thumb"]) < W.SCENE_CHANGE, "a lamp is not a figure"
    assert warm["warm_ratio"] / r0["warm_ratio"] > 1.0 + W.WARM_RISE and same["warm_ratio"] == r0["warm_ratio"]
    assert cover["sharpness_norm"] < r0["sharpness_norm"] / W.SHARP_RATIO, "a covered lens is an instrument fault (sharpness collapses)"
    ok("W13b camera grid: 16 x 9 cells and the red / blue ratio only on request; a figure is seen, a lamp is not, a warm shift is seen, a covered lens collapses sharpness")


def test_guard_whole_path():
    tmp = Path(tempfile.mkdtemp())
    models, _ = investigation_models()
    notify = W.make_notifier(tmp / "alerts.log", None)
    ref = tmp / "reference.json"
    s, bridge = make_session(models, tmp, guard_room=True, ref_path=ref, notify=notify)
    try:
        assert s.state()["guard_room"] is True and s.state()["reference_file"] == "reference.json"
        s.start_watch(interval=0.05, tone_every=0)
        wait_for(lambda: sum(1 for e in s.hub.events if e["type"] == "sample" and e["status"] == "normal") >= 4, what="quiet watching")
        wait_for(lambda: ref.exists(), what="the reference file")
        s.fake_set(person=True)                                     # somebody walks into the (synthetic) room
        ra = wait_for(lambda: next((e for e in s.hub.events if e["type"] == "room_alert"), None), timeout=60, what="the ROOM ALERT")
        assert "picture of the room changed" in ra["text"] and "passed its own test" in ra["text"], ra["text"]
        assert ra["evidence"] and (tmp / "work" / ra["evidence"][0]).exists(), "a picture of the region is the evidence"
        assert all(not m.calls for m in models.values()), "no model was called for a room alert"
        assert not any(e["type"] == "anomaly" for e in s.hub.events)
        assert "ROOM ALERT" in (tmp / "alerts.log").read_text(encoding="utf-8")
        log = (tmp / "work" / "instrument_log.jsonl").read_text(encoding="utf-8")
        assert '"thumb": "[144 numbers]"' in log and '"thumb": [' not in log, "the grid is not copied into the instrument log"
        s.stop()
        wait_for(lambda: s.mode == "idle", what="idle")
        s.fake_set(person=False)
        s.start_watch(interval=0.05, tone_every=0)                  # a second run: the stored reference is read and agrees
        wait_for(lambda: any(e["type"] == "note" and "agrees with the stored reference" in e["text"] for e in s.hub.events), what="the stored reference")
        s.stop()
        wait_for(lambda: s.mode == "idle", what="idle")
    finally:
        s.stop()
        bridge.close()
    ok("W14 the whole path: fake camera -> MCP server -> watchdog -> page: a person enters, ROOM ALERT with a picture and a line in alerts.log, no model; "
       "after a restart the stored reference is read and agrees")


# ============================================================================ W15: guard hours, W16: outside sensors
class Clock:
    """A clock of the test: it runs, and the test can set it to any time of the day (the watchdog uses it only for the window of the night)."""

    def __init__(self, hour, minute, day=19):
        self.set(hour, minute, day)

    def set(self, hour, minute, day=19):
        self.off = time.mktime((2026, 10, day, hour, minute, 0, 0, 0, -1)) - time.time()

    def __call__(self):
        return time.time() + self.off


def test_guard_hours():
    H = W.parse_hours("22:00-07:00")
    assert H == (1320, 420) and W.parse_hours(" 8:30 - 18:00 ") == (510, 1080)
    for bad in ("", "abc", "25:00-07:00", "22:60-07:00", "22:00-22:00", "22-07", None):
        try:
            W.parse_hours(bad)
        except ValueError as e:
            assert "guard hours" in str(e)
            continue
        raise AssertionError(f"must be refused: {bad!r}")
    at = lambda h, m, day=19: time.mktime((2026, 10, day, h, m, 0, 0, 0, -1))
    assert [W.in_hours(H, at(h, m)) for h, m in ((23, 0), (3, 0), (6, 59), (7, 0), (12, 0), (21, 59), (22, 0))] == [True, True, True, False, False, False, True]
    assert W.in_hours(W.parse_hours("08:00-18:00"), at(12, 0)) and not W.in_hours(W.parse_hours("08:00-18:00"), at(20, 0))
    assert W.window_start(H, at(23, 30)) == at(22, 0) and W.window_start(H, at(3, 0, 20)) == at(22, 0, 19) and W.window_start(H, at(22, 5, 20)) == at(22, 0, 20)

    old_settle = W.GUARD_SETTLE_S
    W.GUARD_SETTLE_S = 0.3
    tmp = Path(tempfile.mkdtemp())
    gref = tmp / "guard_reference.json"
    try:
        clock = Clock(12, 0)
        room = Room(person=True, jitter=0.02)                      # by day somebody is in the room: that is normal
        alerts = []
        notify = lambda t: (alerts.append(t), ["stub"])[1]
        wd, ev, inv, stop, th = run_watchdog(room, guard_hours=H, clock=clock, guard_ref_path=gref, notify=notify)
        try:
            wait_for(lambda: sum(1 for e in ev if e["type"] == "sample" and e["status"] == "normal") >= 4, what="a quiet day")
            time.sleep(0.5)
            assert not kinds(ev, "room_alert") and not kinds(ev, "anomaly"), "a person by day is no event"
            assert any("GUARD HOURS 22:00-07:00" in t for t in notes_of(ev))
            # the evening: the people leave and the lights go off; that is no alarm, whatever the instrument-only rules say
            room.v.update(person=False, light=0.5)
            clock.set(22, 5)
            wait_for(lambda: any("the reference of the night is taken" in t for t in notes_of(ev)), what="the reference of the night")
            time.sleep(0.4)
            assert not kinds(ev, "room_alert") and not kinds(ev, "anomaly"), "lights off and people gone at the start of the night: no alarm"
            d = json.loads(gref.read_text(encoding="utf-8"))
            assert d["window_start"] == at(22, 0) and d["values"]["brightness"] < 100 and len(d["thumb"]) == 144, d
            # the night: somebody enters
            room.v.update(person=True)
            wait_for(lambda: kinds(ev, "room_alert"), timeout=60, what="the room alert at night")
            assert len(kinds(ev, "room_alert")) == 1 and "picture of the room changed" in kinds(ev, "room_alert")[0]["text"] and inv == []
            # the morning: the window ends, the lights come on, the staff arrive: no alarm, no agent
            room.v.update(person=True, light=1.6)
            clock.set(7, 30, 20)
            wait_for(lambda: any("GUARD HOURS ended" in t for t in notes_of(ev)), what="the end of the window")
            wait_for(lambda: sum(1 for e in ev[-60:] if e["type"] == "sample" and e["status"] == "normal") >= 3, what="normal again in the morning")
            assert len(kinds(ev, "room_alert")) == 1 and not kinds(ev, "anomaly"), "the day begins with the lights on: nothing to investigate"
        finally:
            end(stop, th)
        # the same night again (a restart, a power cut): the reference of the night is read back, so a person who is there is NOT taken as normal
        clock.set(23, 30, 19)
        room = Room(person=True, jitter=0.02)
        wd, ev, inv, stop, th = run_watchdog(room, guard_hours=H, clock=clock, guard_ref_path=gref, notify=notify)
        try:
            wait_for(lambda: kinds(ev, "room_alert"), timeout=60, what="the alert after the restart")
            assert any("read back" in t for t in notes_of(ev)), notes_of(ev)
        finally:
            end(stop, th)
        # the next night: the old reference is of another night, the room is measured anew
        clock.set(22, 10, 20)
        room = Room(person=False, light=0.5, jitter=0.02)
        wd, ev, inv, stop, th = run_watchdog(room, guard_hours=H, clock=clock, guard_ref_path=gref, notify=notify)
        try:
            wait_for(lambda: any("the reference of the night is taken" in t for t in notes_of(ev)), what="a new reference")
            assert any("GUARD HOURS began" in t for t in notes_of(ev)) and not any("read back" in t for t in notes_of(ev))
            wait_for(lambda: json.loads(gref.read_text(encoding="utf-8")).get("window_start") == at(22, 0, 20), what="the file of the new night")   # the note comes a moment before the file
        finally:
            end(stop, th)
    finally:
        W.GUARD_SETTLE_S = old_settle
    # a question that nobody answers does not stop the watch for ever (a night guard)
    s = web.Session({}, None, tmp, tmp / "runs")
    old_to, W.ASK_TIMEOUT_S = W.ASK_TIMEOUT_S, 0.5
    try:
        t0 = time.time()
        r = s._ask_person("Reply accept or keep?")
        assert r == "" and 0.4 < time.time() - t0 < 5 and s.pending is None and s.mode == "idle", (r, s.pending, s.mode)
        assert any(e["type"] == "note" and "nobody answered" in e["text"] for e in s.hub.events)
    finally:
        W.ASK_TIMEOUT_S = old_to
    ok("W15 guard hours: a person by day is normal; at night the room of the night is the reference; lights off in the evening and on in the morning are no alarm; "
       "a restart in the same night remembers it; the next night is measured anew; a question that nobody answers counts as 'keep'")


def test_smoke_detector():
    from fastapi.testclient import TestClient
    tmp = Path(tempfile.mkdtemp())
    models, _ = investigation_models()
    sent = []
    s, bridge = make_session(models, tmp, notify=lambda t: (sent.append(t), ["stub"])[1])
    token = "0123456789abcdef"
    c = TestClient(web.create_sensor_app(s, token), raise_server_exceptions=False)
    try:
        body = {"sensor": "smoke-1", "alarm": True}
        r = c.post("/api/sensor", json=body)
        assert r.status_code == 401 and not s.hub.events and not sent, "no secret: nothing happens"
        assert c.post("/api/sensor", json=body, headers={"X-Token": "wrong-secret-xx"}).status_code == 401
        for bad in ({"sensor": "x y", "alarm": True}, {"sensor": "smoke-1", "alarm": "yes"}, {"alarm": True}, {"sensor": "a" * 41, "alarm": True}, {"sensor": 5, "alarm": True}):
            r = c.post("/api/sensor", json=bad, headers={"X-Token": token})
            assert r.status_code == 400 and r.json()["ok"] is False, (bad, r.text)
        assert not s.hub.events, "a refused message announces nothing"
        # the alarm: announced at once, no model, the watch need not be running
        assert s.mode == "idle"
        r = c.post("/api/sensor", json=dict(body, detail="smoke in room 2"), headers={"X-Token": token})
        assert r.status_code == 200 and r.json()["alarm"] is True and r.json()["notified"] == ["stub"], r.text
        ev = [e for e in s.hub.events if e["type"] == "room_alert"]
        assert len(ev) >= 1 and "SENSOR ALERT" in ev[0]["text"] and "smoke in room 2" in ev[0]["text"] and "'smoke-1'" in ev[0]["text"] and sent == [ev[0]["text"]], ev
        assert "no confirmation is needed" in ev[0]["text"] and "do not depend on this program" in ev[0]["text"]
        assert all(not m.calls for m in models.values()), "no model for a smoke alarm"
        # a picture of the region follows by itself
        pic = wait_for(lambda: next((e for e in s.hub.events if e["type"] == "room_alert" and e["evidence"]), None), timeout=30, what="the picture")
        assert (tmp / "work" / pic["evidence"][0]).exists() and pic["sensor"] == "smoke-1"
        # the detector repeats its message: not a new alarm; after 'all clear' and a new alarm it is
        n = len([e for e in s.hub.events if e["type"] == "room_alert"])
        assert c.post("/api/sensor", json=body, headers={"X-Token": token}).json().get("repeated") is True
        assert len([e for e in s.hub.events if e["type"] == "room_alert"]) == n and len(sent) == 1
        r = c.post("/api/sensor", json={"sensor": "smoke-1", "alarm": False}, headers={"X-Token": token})
        assert r.status_code == 200 and any("all clear" in e.get("text", "") for e in s.hub.events if e["type"] == "note")
        assert c.post("/api/sensor", json=body, headers={"X-Token": token}).json().get("alarm") is True and len(sent) == 2
        assert s.sensors == {"smoke-1": True}
        # the synthetic detector (--fake page button) goes through the same door
        s.fake_set(smoke=True)
        assert s.sensors.get("smoke-detector") is True and len(sent) == 3 and "simulated" in sent[-1]
        # the watch runs at the same time: a smoke alarm does not wait for it and does not disturb it
        s.start_watch(interval=0.05, tone_every=0)
        wait_for(lambda: sum(1 for e in s.hub.events if e["type"] == "sample" and e["status"] == "normal") >= 3, what="watching")
        s.sensor_event("smoke-2", True)
        assert len(sent) == 4
        s.stop()
        wait_for(lambda: s.mode == "idle", what="idle")
    finally:
        s.stop()
        bridge.close()
    ok("W16 smoke detector: the door needs a secret and a clean message; an alarm is announced at once (page, notification, a picture), no model; "
       "a repeated message is not a new alarm; 'all clear' is noted; the synthetic detector uses the same door")


# ============================================================================ W17-W25: block 1 of the weak places (v7)
HERE = Path(__file__).resolve().parent


def test_summary_report_rotation():
    import csv
    import cam_platform as plat
    import night_report as NR
    tmp = Path(tempfile.mkdtemp())
    path = tmp / "watch_summary.csv"
    ms = W.MinuteSummary(path)
    t0 = time.mktime((2026, 10, 19, 23, 10, 5, 0, 0, -1))
    mk = lambda dt, **kw: dict(dict(type="sample", t=t0 + dt, status="normal", errors=[]), **kw)
    ms.feed(mk(0, brightness=90.0, sharpness=900.0, noise_db=-52.0, change=0.0))
    ms.feed(mk(4, brightness=95.0, sharpness=910.0, noise_db=-50.0, change=0.01, status="changed"))
    ms.feed(dict(type="anomaly", t=t0 + 8, facts=["x"]))
    ms.feed(dict(type="note", t=t0 + 9, text="not counted"))
    ms.feed(mk(70, brightness=91.0, sharpness=905.0, noise_db=-51.0, change=0.0, errors=["camera: x"]))     # the next minute
    ms.feed(dict(type="room_alert", t=t0 + 75, text="ROOM ALERT"))
    ms.flush()
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["minute"] for r in rows] == ["2026-10-19 23:10", "2026-10-19 23:11"], rows
    a, b = rows
    assert (a["samples"], a["changed"], a["errors"], a["anomalies"], a["room_alerts"]) == ("2", "1", "0", "1", "0"), a
    assert (a["brightness_min"], a["brightness_max"], a["noise_max"], a["change_max"]) == ("90.0", "95.0", "-50.0", "0.01"), a
    assert (b["samples"], b["errors"], b["room_alerts"]) == ("1", "1", "1"), b
    assert path.read_text(encoding="utf-8").count("minute,samples") == 1, "one header only, also after a second flush"
    ms.feed(mk(200, brightness=1.0)), ms.flush()
    assert path.read_text(encoding="utf-8").count("minute,samples") == 1 and len(path.read_text(encoding="utf-8").splitlines()) == 4
    # the night report: counts, rates, and limits from the QUIET minutes
    quiet = [dict(minute=f"2026-10-19 23:{m:02d}", samples="15", changed="0", errors="0", brightness_min="90", brightness_mean="91", brightness_max="92",
                  sharpness_mean="900", noise_mean="-52", noise_max="-51", change_max="0.0%d" % (m % 3), anomalies="0", room_alerts="0") for m in range(10, 40)]
    loud = dict(quiet[0], minute="2026-10-19 23:40", room_alerts="1", changed="3")
    day = dict(quiet[0], minute="2026-10-19 12:00", change_max="0.5")
    text = NR.report(quiet + [loud, day])
    assert "minutes watched: 32" in text and "room alerts: 1" in text and "SCENE_CHANGE must stay above 0.500" in text, text
    text = NR.report(quiet + [loud, day], hours=(22 * 60, 7 * 60))                     # only the night: the noon minute is out
    assert "minutes watched: 31" in text and "must stay above 0.02" in text, text
    assert "no minutes to report" in NR.report([], None) and "no minutes to report" in NR.report([day], hours=(22 * 60, 7 * 60))
    # the log of the instrument does not grow for ever; the evidence pictures do not stay for ever
    log = tmp / "instrument_log.jsonl"
    log.write_bytes(b"x" * 100)
    assert plat.rotate_log(log, 1000) is False and not (tmp / "instrument_log.jsonl.1").exists()
    log.write_bytes(b"y" * 5000)
    assert plat.rotate_log(log, 1000) is True and not log.exists() and (tmp / "instrument_log.jsonl.1").read_bytes() == b"y" * 5000
    log.write_bytes(b"z" * 5000)
    assert plat.rotate_log(log, 1000) is True and (tmp / "instrument_log.jsonl.1").read_bytes() == b"z" * 5000, "the older .1 is replaced: two files at most"
    assert plat.rotate_log(log, 0) is False and plat.rotate_log(tmp / "nothing", 10) is False
    crops = tmp / "crops"
    crops.mkdir()
    now = time.time()
    for name, age_days in (("old1.png", 9), ("old2.png", 8), ("new.png", 1)):
        f = crops / name
        f.write_bytes(b"png")
        os.utime(f, (now - age_days * 86400, now - age_days * 86400))
    (crops / "notes.txt").write_text("keep", encoding="utf-8")
    os.utime(crops / "notes.txt", (now - 99 * 86400, now - 99 * 86400))
    assert plat.prune_old(crops, 0) == 0 and len(list(crops.iterdir())) == 4, "0 days = keep everything"
    assert plat.prune_old(crops, 7) == 2 and sorted(p.name for p in crops.iterdir()) == ["new.png", "notes.txt"]
    assert plat.prune_old(tmp / "nothing", 7) == 0
    ok("W17 the summary (one CSV line a minute), the night report (rates and limits from quiet minutes), log rotation (2 files at most), "
       "deletion of evidence pictures older than the limit (other files stay)")


def test_hints_file():
    import copy
    keep = copy.deepcopy(W.HINTS)
    tmp = Path(tempfile.mkdtemp())
    try:
        facts = ["scene brightness 150 -> 4 (limit 135-165)"]
        before = W.hints(facts)
        assert W.HINTS["fell"] in before
        f = tmp / "hints.json"
        f.write_text(json.dumps({"fell": "MY OWN WORDS ABOUT A DARK PICTURE"}), encoding="utf-8")
        assert W.load_hints(f) == ["fell"]
        after = W.hints(facts)
        assert "MY OWN WORDS ABOUT A DARK PICTURE" in after and keep["fell"] not in after and W.HINTS["cover"] == keep["cover"]
        # a mistake changes nothing and says what is wrong
        for bad, word in (({"fel": "typo"}, "unknown hint 'fel'"), ({"fell": 5}, "non-empty text"), ({"fell": "  "}, "non-empty text"), ([], "JSON object"), ({}, "JSON object")):
            f.write_text(json.dumps(bad), encoding="utf-8")
            try:
                W.load_hints(f)
            except ValueError as e:
                assert word in str(e), (bad, str(e))
            else:
                raise AssertionError(f"must be refused: {bad}")
        assert W.HINTS["fell"].startswith("MY OWN") and W.HINTS["rose"] == keep["rose"], "a refused file changes nothing"
        f.write_text("{not json", encoding="utf-8")
        try:
            W.load_hints(f)
        except ValueError:                                                     # json.JSONDecodeError is a ValueError: the CLI answers with one sentence
            pass
        else:
            raise AssertionError("a broken file must be refused")
    finally:
        W.HINTS.clear()
        W.HINTS.update(keep)
    assert W.hints(facts) == before, "the defaults are the words of the incident, until a file replaces them"
    assert (HERE / "hints.example.json").exists() and set(json.loads((HERE / "hints.example.json").read_text(encoding="utf-8"))) <= set(W.HINT_KEYS)
    ok("W18 hints: the words for the agent can be replaced from a JSON file without touching the code; a wrong key or text changes nothing; the defaults stay")


def test_steady_environment():
    se = W.steady_environment
    p, c = dict(brightness=150.0, noise_db=-35.0), dict(brightness=151.0, noise_db=-34.5)
    left = ["room noise -35.0 dBFS is 17 dB above the normal"]
    assert se(p, c, left) is True
    assert se(None, c, left) is False and se(p, None, left) is False and se(p, c, []) is False, "one check is not 'the same twice'"
    assert se(p, dict(c, noise_db=-31.0), left) is False, "the noise is still moving (4 dB)"
    assert se(p, dict(c, brightness=170.0), ["scene brightness 170 -> 150"]) is False, "the light is still moving"
    assert se(p, c, left + ["sharpness fell 900 -> 100"]) is False and se(p, c, ["camera: no frame"]) is False, "something else is wrong: the agent can still help"
    assert se(dict(noise_db=-35.0), dict(noise_db=-35.2), left) is True, "a missing reading is not a difference"
    ok("W19 the loop says ESCALATION at once when what stays wrong is only the light or the noise and two checks in a row gave the same numbers")


def test_push_on_trouble_and_health():
    import threading
    texts = []
    notify = lambda t: (texts.append(t), ["stub"])[1]
    room = Room()
    dead = dict(on=False)
    def call(name, args):
        if dead["on"] and name in ("measure_region", "capture"):
            return json.dumps(dict(ok=False, error="camera not found (cable?)"))
        return room(name, args)
    notes, inv = [], []
    wd = W.Watchdog(call, ["measure_region", "listen_noise", "capture"], notes.append, lambda r, f: inv.append(f), interval=0.0, notify=notify)
    stop = threading.Event()
    th = threading.Thread(target=wd.run, args=(stop,), daemon=True)
    th.start()
    try:
        wait_for(lambda: wd.base.ready(), what="the normal values")
        assert texts == [] and wd.health() == (True, ""), "a healthy watch pushes nothing"
        dead["on"] = True                                                       # somebody pulls the cable at night
        wait_for(lambda: any("ESCALATION" in t for t in texts), timeout=60, what="the push about the lost camera and the ESCALATION")
        lost = [t for t in texts if "does not answer" in t]
        assert len(lost) == 1 and lost[0].startswith("WATCH TROUBLE") and "camera: camera not found" in lost[0] and "NOT be watched" in lost[0], texts
        assert len([t for t in texts if "ESCALATION" in t]) == 1 and len(inv) >= 2, (texts, len(inv))
        assert any("[stub]" in n.get("text", "") for n in notes if n["type"] == "note" and "WATCH TROUBLE" in n.get("text", "")), "the page says what was done"
        dead["on"] = False                                                      # the cable is back
        wait_for(lambda: any("ended" in t for t in texts), timeout=60, what="the message that the instrument answers again")
        assert texts[-1].startswith("WATCH TROUBLE ended") and len([t for t in texts if "does not answer" in t]) == 1, texts
    finally:
        stop.set()
        th.join(10)
    # the same message is not repeated every four seconds
    wd2 = W.Watchdog(lambda n, a: "{}", [], lambda e: None, lambda r, f: None, notify=notify)
    n0 = len(texts)
    wd2.push("k", "WATCH TROUBLE: x"), wd2.push("k", "WATCH TROUBLE: x"), wd2.push("k", "WATCH TROUBLE: x", quiet=0.0)
    assert len(texts) == n0 + 2, "within the quiet time the same key is sent once (quiet=0 sends again)"
    wd2.notify = lambda t: 1 / 0
    wd2.push("other", "WATCH TROUBLE: y")                                       # a broken notifier does not stop the watch
    # health(): what the heartbeat asks
    now = time.time()
    h = W.Watchdog(lambda n, a: "{}", [], lambda e: None, lambda r, f: None, interval=4.0)
    assert h.health() == (True, "")
    h.trouble_since = now - 10
    assert h.health()[0] is True, "a short trouble is not yet a reason to stop the heartbeat"
    h.trouble_since = now - W.BLIND_S - 5
    ok_, why = h.health()
    assert ok_ is False and "has not answered" in why, why
    h.trouble_since, h.last_step_t = None, now - (20 * 4.0 + 120.0) - 5
    ok_, why = h.health()
    assert ok_ is False and "no step" in why, why
    h.state = "investigating"                                                   # waiting for an agent takes minutes: not a hang
    assert h.health()[0] is True
    h.last_step_t = now - W.ASK_TIMEOUT_S - 900.0 - 5
    assert h.health()[0] is False
    ok("W20 a lost camera, the ESCALATION and the return of the instrument are PUSHED (once each, not every 4 s); a broken notifier does not stop the watch; "
       "health() is false when the instrument is silent for BLIND_S or the loop hangs, not when an agent is working")


def test_night_arming_pause_and_quiet():
    """The night guard without surprises: a push when it arms, 'I am in the lab' pauses it, no beep at night, a dead microphone is not a quiet room,
    and a room that is not last night's is told while there is still time to look."""
    H = W.parse_hours("22:00-07:00")
    old_settle, W.GUARD_SETTLE_S = W.GUARD_SETTLE_S, 0.3
    tmp = Path(tempfile.mkdtemp())
    try:
        # 1. arming push, pause, resume
        gref = tmp / "g1.json"
        clock = Clock(23, 0)
        room = Room(jitter=0.02)
        sent = []
        notify = lambda t: (sent.append(t), ["stub"])[1]
        wd, ev, inv, stop, th = run_watchdog(room, guard_hours=H, clock=clock, guard_ref_path=gref, notify=notify)
        try:
            wait_for(lambda: any("the reference of the night is taken" in t for t in notes_of(ev)), what="the reference of the night")
            assert sent and sent[0].startswith("GUARD ARMING: the room is taken as empty in") and "I am in the lab" in sent[0], sent
            text = wd.postpone(60)                                              # somebody walks in with a key: allowed
            assert "paused until" in text and any(t.startswith("GUARD ARMING: guard hours are paused until") for t in sent)
            wait_for(lambda: any("GUARD paused (somebody is in the lab)" in t for t in notes_of(ev)), what="the pause")
            room.v.update(person=True)
            time.sleep(0.8)
            assert not kinds(ev, "room_alert") and not kinds(ev, "anomaly"), "somebody who is allowed to be there is not an alarm"
            assert wd.postpone(10_000) and wd._postponed_until - wd.clock() <= 720 * 60 + 1, "at most 12 hours"
            wd.postpone(0)                                                      # leaves: the guard comes back and measures the room as it is NOW
            wait_for(lambda: sum(1 for t in notes_of(ev) if "the reference of the night is taken" in t) >= 2, what="a new reference after the pause")
            assert any("the pause is cancelled" in t for t in sent) and sum(1 for t in notes_of(ev) if "GUARD HOURS began" in t) >= 2
            time.sleep(0.5)
            assert not kinds(ev, "room_alert"), "the new reference is the room as it is after the pause"
            room.v.update(person=False)                                         # now the room changes against the new reference: that IS an alert
            wait_for(lambda: kinds(ev, "room_alert"), timeout=60, what="an alert after the pause")
        finally:
            end(stop, th)
        # 2. nothing is played at night, also not at the first step; by day the check tone is played as before
        for hour, tone_allowed in ((23, False), (12, True)):
            room = Room(jitter=0.02)
            wd, ev, inv, stop, th = run_watchdog(room, names=("measure_region", "listen_noise", "capture", "play_tone"), guard_hours=H,
                                                 clock=Clock(hour, 30), guard_ref_path=tmp / f"g2_{hour}.json", tone_every=0.001)
            try:
                wait_for(lambda: sum(1 for e in ev if e["type"] == "sample") >= 8, what="samples")
            finally:
                end(stop, th)
            assert ("play_tone" in room.calls) is tone_allowed, (hour, tone_allowed, sorted(set(room.calls)))
        # 3. a dead microphone is not a quiet room: at night the level of a living microphone moves a little, a dead input gives one number
        room = Room(jitter=0.0)                                                 # the number never changes
        sent = []
        wd, ev, inv, stop, th = run_watchdog(room, guard_hours=H, clock=Clock(23, 0), guard_ref_path=tmp / "g3.json", notify=lambda t: (sent.append(t), ["s"])[1])
        try:
            wait_for(lambda: any("does not move at all" in f for f in sum((a["facts"] for a in kinds(ev, "anomaly")), [])), timeout=60, what="the dead microphone")
            assert any("WATCH TROUBLE" in t and "does not move at all" in t for t in sent), sent
        finally:
            end(stop, th)
        # 4. tonight's empty room is not last night's: a push at arming (nothing is blocked)
        gref = tmp / "g4.json"
        night1 = Clock(22, 10, 19)
        wd, ev, inv, stop, th = run_watchdog(Room(light=0.5, jitter=0.02), guard_hours=H, clock=night1, guard_ref_path=gref, notify=lambda t: ["s"])
        try:
            wait_for(lambda: gref.exists(), what="the reference of the first night")
        finally:
            end(stop, th)
        first_night = gref.read_text(encoding="utf-8")
        for label, kw, expect in (("the same room", dict(light=0.5), False), ("somebody still there", dict(light=0.5, person=True), True),
                                  ("the light left on", dict(light=1.5), True)):
            sent = []
            gref.write_text(first_night, encoding="utf-8")                      # every case starts from the same 'last night'
            night2 = Clock(22, 10, 20)
            wd, ev, inv, stop, th = run_watchdog(Room(jitter=0.02, **kw), guard_hours=H, clock=night2, guard_ref_path=gref, notify=lambda t: (sent.append(t), ["s"])[1])
            try:
                wait_for(lambda: any("the reference of the night is taken" in t for t in notes_of(ev)), what=f"a reference ({label})")
                time.sleep(0.3)
            finally:
                end(stop, th)
            warn = [t for t in sent if t.startswith("GUARD ARMING WARNING")]
            assert bool(warn) is expect, (label, sent)
            if expect:
                assert "not last night's" in warn[0] and "Somebody may still be in the room" in warn[0], warn
            assert any("the reference of the night is taken" in t for t in notes_of(ev)), "the warning does not block the guard"
    finally:
        W.GUARD_SETTLE_S = old_settle
    ok("W21 night guard: a push when it arms; 'I am in the lab' pauses it (no alarm for who is allowed), then the room is measured anew; no beep at night; "
       "a dead microphone is a fault, not silence; a room that is not last night's is told at arming")


def test_postpone_endpoint():
    from fastapi.testclient import TestClient
    tmp = Path(tempfile.mkdtemp())
    models, _ = investigation_models()
    s, bridge = make_session(models, tmp, guard_room=True, guard_hours=W.parse_hours("00:00-23:59"))
    c = TestClient(web.create_app(s), raise_server_exceptions=False)
    try:
        r = c.post("/api/guard/postpone", json={"minutes": 60})
        assert r.status_code == 409 and "not running" in r.json()["error"], r.text
        st = c.get("/api/state").json()
        assert st["guard_room"] is True and st["guard_hours"] == "00:00-23:59", st
        s.start_watch(interval=0.05, tone_every=0)
        wait_for(lambda: s.wd is not None and sum(1 for e in s.hub.events if e["type"] == "sample") >= 2, what="the watch")
        for bad in ({"minutes": "abc"}, {"minutes": -1}, {"minutes": 721}, {"minutes": None}):
            r = c.post("/api/guard/postpone", json=bad)
            assert r.status_code == 400 and r.json()["ok"] is False, (bad, r.text)
        r = c.post("/api/guard/postpone", json={"minutes": 90})
        assert r.status_code == 200 and "paused until" in r.json()["text"] and s.wd._postponed_until > time.time() + 80 * 60, r.text
        assert c.post("/api/guard/postpone", json={"minutes": 0}).status_code == 200 and s.wd._postponed_until == 0.0
        assert c.post("/api/guard/postpone", json={}).status_code == 200, "the default is 60 minutes"
        s.stop()
        wait_for(lambda: s.mode == "idle", what="idle")
        assert 'guardCard' in (HERE / "web_ui.html").read_text(encoding="utf-8") and "/api/guard/postpone" in (HERE / "web_ui.html").read_text(encoding="utf-8")
    finally:
        s.stop()
        bridge.close()
    ok("W22 the page's 'I am in the lab' button: a pause of 0-720 minutes; a letter, a negative or a huge number is refused; 409 when the guard is not running")


def test_heartbeat_and_crash():
    import http.server
    import threading
    pings = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            pings.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    old = os.environ.get("NO_PROXY")
    os.environ["NO_PROXY"] = "127.0.0.1"
    tmp = Path(tempfile.mkdtemp())
    models, _ = investigation_models()
    url = f"http://127.0.0.1:{srv.server_address[1]}/ping/very-secret-uuid"
    s, bridge = make_session(models, tmp, heartbeat_url=url, heartbeat_every=0.3)
    try:
        s.start_watch(interval=0.05, tone_every=0)
        wait_for(lambda: len(pings) >= 3, timeout=30, what="pings while the loop is healthy")
        assert set(pings) == {"/ping/very-secret-uuid"}
        # the loop is not healthy: the pings STOP (nobody sends 'I am fine' for it), and the page says why
        s.wd.health = lambda: (False, "the instrument has not answered for 70 s")
        wait_for(lambda: any("NOT sent" in e.get("text", "") for e in s.hub.events if e["type"] == "note"), timeout=30, what="the note")
        time.sleep(0.5)
        n = len(pings)
        time.sleep(1.0)
        assert len(pings) == n, "no ping while unhealthy"
        s.wd.health = lambda: (True, "")
        wait_for(lambda: len(pings) > n, timeout=30, what="pings again")
        # the address of the service is a secret: it is nowhere on the page
        assert "very-secret-uuid" not in json.dumps(s.hub.events) and "very-secret-uuid" not in json.dumps(s.state())
        s.stop()
        wait_for(lambda: s.mode == "idle", what="idle")
        n = len(pings)
        time.sleep(0.8)
        assert len(pings) <= n + 1, "the heartbeat stops with the watch"
    finally:
        s.stop()
        bridge.close()
    # a service that cannot be reached (no internet) is told on the page once, not every minute
    s, bridge = make_session(models, (lambda d: (d.mkdir(), d)[1])(tmp / "b"), heartbeat_url="http://127.0.0.1:9/dead", heartbeat_every=0.3)
    try:
        s.start_watch(interval=0.05, tone_every=0)
        wait_for(lambda: any("ping FAILED" in e.get("text", "") for e in s.hub.events if e["type"] == "note"), timeout=30, what="the failed ping")
        time.sleep(1.0)
        assert sum(1 for e in s.hub.events if "ping FAILED" in e.get("text", "")) == 1, "told once"
        s.stop()
        wait_for(lambda: s.mode == "idle", what="idle")
    finally:
        s.stop()
        bridge.close()
    # a crash of the loop is pushed: a program that died silently is the worst case of a guard
    sent = []
    s, bridge = make_session(models, (lambda d: (d.mkdir(), d)[1])(tmp / "c"), notify=lambda t: (sent.append(t), ["stub"])[1])
    orig = W.Watchdog.step
    def boom(self):
        raise RuntimeError("boom in the loop")
    W.Watchdog.step = boom
    try:
        s.start_watch(interval=0.05, tone_every=0)
        wait_for(lambda: s.mode == "idle" and sent, timeout=30, what="the crash")
        assert "CRASHED" in sent[0] and "NOT watched" in sent[0] and "boom in the loop" in sent[0] and sent[0].startswith("WATCH TROUBLE"), sent
        assert any(e["type"] == "error" and "boom" in e["text"] for e in s.hub.events)
    finally:
        W.Watchdog.step = orig
        s.stop()
        bridge.close()
        srv.shutdown()
        if old is None:
            os.environ.pop("NO_PROXY", None)
        else:
            os.environ["NO_PROXY"] = old
    ok("W23 heartbeat: a ping while the loop is healthy, none when it is not (the service raises the alarm), none after Stop; the address is not on the page; "
       "an unreachable service is told once; a crash of the loop is pushed")


def test_signed_sensor_messages():
    import hmac as _hmac
    from fastapi.testclient import TestClient
    tmp = Path(tempfile.mkdtemp())
    models, _ = investigation_models()
    sent = []
    s, bridge = make_session(models, tmp, notify=lambda t: (sent.append(t), ["stub"])[1])
    token = "0123456789abcdef"
    signed = lambda body, ts=None, secret=token: (lambda t: {"X-Timestamp": t, "X-Signature": web.sign_message(secret, t, body)})(ts if ts is not None else str(time.time()))
    body = json.dumps({"sensor": "smoke-1", "alarm": True, "detail": "room 2"}).encode("utf-8")
    c = TestClient(web.create_sensor_app(s, token), raise_server_exceptions=False)
    c_signed = TestClient(web.create_sensor_app(s, token, signed_only=True), raise_server_exceptions=False)
    try:
        expect = _hmac.new(token.encode(), b"1700000000." + body, "sha256").hexdigest()
        assert web.sign_message(token, "1700000000", body) == expect, "HMAC-SHA256 of 'timestamp.body'"
        # a good signature is accepted; the secret itself is not on the wire
        h = signed(body)
        assert token not in json.dumps(h) and "x-token" not in {k.lower() for k in h}
        r = c.post("/api/sensor", content=body, headers=h)
        assert r.status_code == 200 and r.json()["alarm"] is True and len(sent) == 1, r.text
        # the same message again (copied from the network): refused, and nothing is announced
        r = c.post("/api/sensor", content=body, headers=h)
        assert r.status_code == 401 and "already used" in r.json()["error"] and len(sent) == 1, r.text
        # refused: a wrong secret, a changed body, an old message, a message from the future, a timestamp that is not a number, a signature without timestamp
        bad = [(signed(body, secret="another-secret-xx"), body, "wrong signature"),
               (signed(body), body.replace(b"room 2", b"room 3"), "wrong signature"),
               (signed(body, ts=str(time.time() - 3600)), body, "too old"),
               (signed(body, ts=str(time.time() + 3600)), body, "too old"),
               ({"X-Timestamp": "yesterday", "X-Signature": "00"}, body, "seconds since 1970"),
               ({"X-Signature": "00"}, body, "seconds since 1970")]
        for hdr, b, word in bad:
            r = c.post("/api/sensor", content=b, headers=hdr)
            assert r.status_code == 401 and word in r.json()["error"], (hdr, word, r.text)
        assert len(sent) == 1 and len([e for e in s.hub.events if e["type"] == "room_alert"]) >= 1
        # a hub that cannot sign still works with the secret (the default) ...
        s.sensor_event("smoke-1", False)
        r = c.post("/api/sensor", content=body, headers={"X-Token": token})
        assert r.status_code == 200 and r.json()["alarm"] is True, r.text
        # ... but not when the door is set to signed_only; the signed message still works there
        r = c_signed.post("/api/sensor", content=body, headers={"X-Token": token})
        assert r.status_code == 401 and "X-Signature" in r.json()["error"] and "X-Token" not in r.json()["error"], r.text
        s.sensor_event("smoke-1", False)
        assert c_signed.post("/api/sensor", content=body, headers=signed(body)).status_code == 200
        # a signed but broken message is a 400, not an alarm
        junk = b"not json"
        assert c.post("/api/sensor", content=junk, headers=signed(junk)).status_code == 400
        # an unsigned message without any header
        assert c.post("/api/sensor", content=body).status_code == 401
    finally:
        s.stop()
        bridge.close()
    ok("W24 signed sensor messages: HMAC of timestamp+body (the secret never travels); a copied, changed, old or future message is refused; "
       "the plain X-Token still works unless --sensor-signed-only")


def test_sensor_send_script_and_cli():
    import http.server
    import socket
    import subprocess
    import threading
    import urllib.error
    import urllib.request
    import uvicorn
    env0 = dict(os.environ, NO_PROXY="127.0.0.1", no_proxy="127.0.0.1")
    free_port = lambda: (lambda so: (so.bind(("127.0.0.1", 0)), so.getsockname()[1], so.close())[1])(socket.socket())
    # 1. sensor_send.py against a real door on a real port
    tmp = Path(tempfile.mkdtemp())
    models, _ = investigation_models()
    sent = []
    s, bridge = make_session(models, tmp, notify=lambda t: (sent.append(t), ["stub"])[1])
    token = "0123456789abcdef"
    port = free_port()
    srv = uvicorn.Server(uvicorn.Config(web.create_sensor_app(s, token), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    try:
        wait_for(lambda: srv.started, timeout=30, what="the sensor door")
        url = f"http://127.0.0.1:{port}/api/sensor"
        run = lambda *a, secret=token, **kw: subprocess.run([sys.executable, str(HERE / "sensor_send.py"), *a, "--url", url], capture_output=True, text=True, timeout=60,
                                                          env=dict(env0, **({"HW3_SENSOR_TOKEN": secret} if secret is not None else {})), **kw)
        r = run("smoke-1", "alarm", "--detail", "room 2")
        assert r.returncode == 0 and "200" in r.stdout and '"alarm":true' in r.stdout.replace(" ", ""), (r.stdout, r.stderr)
        assert len(sent) == 1 and "room 2" in sent[0]
        r = run("smoke-1", "clear")
        assert r.returncode == 0 and "alarm" in r.stdout and s.sensors["smoke-1"] is False
        r = run("smoke-1", "alarm", "--unsigned")
        assert r.returncode == 0 and len(sent) == 2, (r.stdout, r.stderr)
        r = run("smoke-1", "alarm", secret="a-different-secret")
        assert r.returncode == 1 and "401" in r.stderr and "wrong signature" in r.stderr and len(sent) == 2, (r.stdout, r.stderr)
        r = run("smoke-1", "alarm", secret="short")
        assert r.returncode == 2 and "at least 12" in r.stderr
        r = run("smoke-1", "alarm", secret=None)
        assert r.returncode == 2
        r = subprocess.run([sys.executable, str(HERE / "sensor_send.py"), "smoke-1", "alarm", "--url", "http://127.0.0.1:9/x"], capture_output=True, text=True, timeout=60,
                           env=dict(env0, HW3_SENSOR_TOKEN=token))
        assert r.returncode == 1 and "cannot reach" in r.stderr, (r.stdout, r.stderr)
    finally:
        srv.should_exit = True
        s.stop()
        bridge.close()
    # 2. the program itself, started as a person starts it: refusals first (they must be one clear sentence, not a traceback) ...
    base = [sys.executable, str(HERE / "web_app.py"), "--fake", "--model", "x", "--work", str(tmp / "cli_work"), "--runs", str(tmp / "cli_runs"),
            "--alerts", str(tmp / "cli_alerts.log")]
    envc = dict(env0, OPENROUTER_API_KEY="not-a-real-key", HW3_SENSOR_TOKEN="")
    for extra, word in ((["--heartbeat-every", "1"], "at least 5 seconds"), (["--sensor-port", "8123"], "at least 12 characters"),
                        (["--guard-hours", "25:00-07:00"], "guard hours"), (["--hints", str(tmp / "nope.json")], "--hints")):
        r = subprocess.run(base + extra, capture_output=True, text=True, timeout=120, env=envc)
        assert r.returncode != 0 and word in (r.stderr + r.stdout) and "Traceback" not in r.stderr, (extra, r.returncode, r.stdout[-300:], r.stderr[-300:])
    # ... then a start with everything on: --autostart, the heartbeat, the signed-only door, the guard hours; it must run, ping, answer, and stop on a signal
    pings = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            pings.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, *a):
            pass
    hb = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=hb.serve_forever, daemon=True).start()
    web_port, sensor_port = free_port(), free_port()
    secret = "cli-secret-0123456789"
    envc = dict(env0, OPENROUTER_API_KEY="not-a-real-key", HW3_SENSOR_TOKEN=secret, HW3_HEARTBEAT_URL=f"http://127.0.0.1:{hb.server_address[1]}/ping/cli-secret-uuid")
    log = (tmp / "cli.out").open("w")
    proc = subprocess.Popen(base + ["--autostart", "--port", str(web_port), "--sensor-port", str(sensor_port), "--sensor-signed-only", "--heartbeat-every", "5",
                                    "--guard-hours", "00:00-23:59", "--guard-settle", "1", "--camera-index", "0"], stdout=log, stderr=subprocess.STDOUT, env=envc)
    try:
        wait_for(lambda: "open  http://" in (tmp / "cli.out").read_text(encoding="utf-8", errors="replace") or proc.poll() is not None, timeout=90, what="the server")
        assert proc.poll() is None, (tmp / "cli.out").read_text(encoding="utf-8", errors="replace")[-800:]
        out = (tmp / "cli.out").read_text(encoding="utf-8", errors="replace")
        assert "autostart: watching has begun" in out and "GUARD HOURS 00:00-23:59" in out and "sensor door open" in out, out[-800:]
        def get(path):                                   # the line "open http://" is printed a moment before the port is bound: try again for a while
            t0 = time.time()
            while True:
                try:
                    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{web_port}{path}", timeout=10).read())
                except urllib.error.URLError:
                    if time.time() - t0 > 60 or proc.poll() is not None:
                        raise
                    time.sleep(0.3)
        wait_for(lambda: get("/api/state")["mode"] == "watching", timeout=60, what="watching without a button")
        st = get("/api/state")
        assert st["guard_room"] is True and st["guard_hours"] == "00:00-23:59", st
        wait_for(lambda: len(pings) >= 2, timeout=60, what="two heartbeat pings")
        assert set(pings) == {"/ping/cli-secret-uuid"} and "cli-secret-uuid" not in out and "cli-secret-uuid" not in json.dumps(get("/api/log"))
        # the door: the plain secret is refused (signed-only), the signed message is announced, the main port has no sensor route
        wait_for(lambda: socket.socket().connect_ex(("127.0.0.1", sensor_port)) == 0, timeout=60, what="the sensor door to listen")
        r = subprocess.run([sys.executable, str(HERE / "sensor_send.py"), "smoke-1", "alarm", "--unsigned", "--url", f"http://127.0.0.1:{sensor_port}/api/sensor"],
                           capture_output=True, text=True, timeout=60, env=envc)
        assert r.returncode == 1 and "401" in r.stderr
        r = subprocess.run([sys.executable, str(HERE / "sensor_send.py"), "smoke-1", "alarm", "--url", f"http://127.0.0.1:{sensor_port}/api/sensor"],
                           capture_output=True, text=True, timeout=60, env=envc)
        assert r.returncode == 0 and "200" in r.stdout, (r.stdout, r.stderr)
        alerts = (tmp / "cli_alerts.log").read_text(encoding="utf-8")
        assert "SENSOR ALERT" in alerts and "smoke-1" in alerts and secret not in alerts
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{web_port}/api/sensor", data=b"{}", method="POST"), timeout=10)
            raise AssertionError("the main port must not have the sensor route")
        except urllib.error.HTTPError as e:
            assert e.code in (404, 405)
        # the summary file is written (one line per minute); the process stops cleanly on SIGTERM
        proc.terminate()
        proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
        hb.shutdown()
        log.close()
    wait_for(lambda: (tmp / "cli_work" / "watch_summary.csv").exists(), timeout=15, what="the per-minute summary, written when the watch stops")
    ok("W25 sensor_send.py (signed by default, --unsigned, wrong secret, no secret, no door) against a real door; web_app.py as started by a person: "
       "one-sentence refusals, then --autostart + heartbeat + signed-only door + guard hours run together and stop on SIGTERM")


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
    test_loop_closes_or_escalates()
    test_person_is_asked_when_the_tone_is_lost()
    test_reference_state_is_not_forgotten()
    test_scene_change_and_room_facts()
    test_reference_file()
    test_guard_mode()
    test_notifier()
    test_camera_grid()
    test_guard_whole_path()
    test_guard_hours()
    test_smoke_detector()
    test_summary_report_rotation()
    test_hints_file()
    test_steady_environment()
    test_push_on_trouble_and_health()
    test_night_arming_pause_and_quiet()
    test_postpone_endpoint()
    test_heartbeat_and_crash()
    test_signed_sensor_messages()
    test_sensor_send_script_and_cli()
    print(f"\nall {len(PASSED)} checks passed")
