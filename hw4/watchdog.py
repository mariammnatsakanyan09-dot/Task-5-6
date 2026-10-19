"""The watchdog: a cheap loop WITHOUT a language model that watches the instrument and calls the agent when something changes.

Every few seconds the program measures the scene (camera) and the room noise (microphone), and compares the numbers with the
normal values of the last minute. Nothing is sent to a model while everything is normal. When the numbers leave the normal range in
CONFIRM of the last WINDOW samples, the watchdog builds a request from the facts and hands it to the graph of Homework 3 (orchestrator,
hardware agent, human pause): THE MODEL decides what to check, in which order, and what the person must fix. After the
investigation the PROGRAM measures once more. If the instrument is not back inside the limits of the alert, the loop is not closed yet:
the agent is called again (MAX_RETRIES times) with the numbers of the program's own check, and if that does not help either, the program
says loudly that a person is needed and what stays outside the limits, and asks whether this state may become the new normal. Only an
"accept" of the person does that; otherwise the reference state (the first normal values of the run) stays and the watch goes on against it.

    sample -> compare with the normal values -> (normal: wait) / (changed twice: ALERT -> agent investigates -> program checks
    -> back inside: new normal / not back: agent again -> program checks -> still not back: tell the person -> person accepts: new normal / not: reference stays)

The numbers come from the instrument tools (measure_region, listen_noise, play_tone), exactly as the agent gets them.

Two more things were added after the first version:

* THE REFERENCE FILE. The reference state (the normal values of the first minute) is written to a file (reference.json) and read again at
  the next start. Only a person's "accept" changes it. So the program can tell "the problem was fixed" (the readings are back at the stored
  reference) from "we got used to the problem" (the readings are still away from it), also after a restart.

* GUARD MODE (room events). The watchdog was made to find faults of the INSTRUMENT. With guard=True it also watches the ROOM against the
  stored reference: the picture changed (something moved or appeared), the light changed, the picture got warmer (flame?), the room got louder.
  A change of the room alone proves nothing: it could be a covered lens or a broken microphone. So before the program says "ROOM ALERT" it
  tests the instrument itself (picture sharp, microphone alive, the check tone arrives). If the instrument is in order, the change is real, and the
  program tells a person at once (page, alerts.log, a push message if HW3_NOTIFY_URL is set) with a picture as evidence. If the instrument is not in
  order, the usual loop starts (agent investigates, program checks). The model is not needed for the room alert: it comes from numbers.

* GUARD HOURS. In a lab people are normal by day and suspicious at night. With guard_hours=(start, end) the room is watched only inside that window
  of the day (the clock of the computer). When the window begins the lights go off and the people leave, so the program does not compare with the
  reference of the day: after GUARD_SETTLE_S it takes the room as it is then as the reference of the NIGHT (it assumes the room is empty; it is
  saved with the date of the window, so a restart in the middle of the night does not forget it and a power cut does not hide a person). Whatever
  changes after that is a room alert. When the window ends the normal values are learned anew (the day starts with the lights on).
"""
from __future__ import annotations

import json
import os
import re
import statistics
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

WARMUP = 4                  # samples that make the first normal values
KEEP = 8                    # normal samples that the normal values are made of (the oldest drop out: slow drift is not an alarm)
CONFIRM = 2                 # samples that must disagree ...
WINDOW = 3                  # ... among the last WINDOW samples of that kind (one odd sample is not an alarm; a burst of noise with a quiet sample between is)
BRIGHT_ABS, BRIGHT_REL = 15.0, 0.25      # brightness change that counts: at least 15 grey levels and 25 %
SHARP_RATIO = 3.0                        # sharpness_norm that falls to a third (a rise is no alarm: on the real camera it jumped 187 -> 1000 when the picture settled)
NOISE_RISE_DB = 8.0                      # room noise that rises by this much
TONE_DROP_DB = 10.0                      # the check tone that falls by this much at the microphone (snr_db)
TONE_LEVELS = (-30.0, -20.0, -12.0)      # levels tried to find one that the microphone hears clearly
TONE_HZ = 1000.0
TONE_NEEDED_DB = 15.0                    # a level at which the check tone is heard this far above the noise is taken at once ...
TONE_MIN_DB = 10.0                       # ... otherwise the best level that is heard at all (10 dB is the connector's "heard"); in a reflecting room the
                                         # margin stops growing with the level (real room: 11.6 / 12.3 / 11.7 dB at -30 / -20 / -12 dBFS)
COUPLING_DROP_DB = 6.0                   # the check tone arrives this much weaker at the microphone (tone_db - level_db) = the loudspeaker got quieter
RECAL_S = 45.0                           # a switched-off tone check is tried again this often
SETTLE_RATIO = 2.0                       # warm-up: the last WARMUP samples must agree this well (sharpness max/min), else the picture is still settling
GUARD_SETTLE_S = 120.0                   # guard hours: after the window begins the room is left alone this long (people leave, the lights go off), then it is taken as the reference of the night
ASK_TIMEOUT_S = 300.0                    # a question of the program to the person (reference, ESCALATION, volume) that nobody answers this long counts as "keep" and the watch goes on (a night guard must not wait for ever)
BLIND_S = 60.0                           # the camera or the microphone has not answered for this long: the heartbeat stops (nobody is watching the room any more)
TROUBLE_QUIET_S = 300.0                  # the same trouble is pushed to the phone again at most this often
STEADY_BRIGHT_ABS, STEADY_BRIGHT_REL, STEADY_NOISE_DB = 3.0, 0.03, 2.0     # two checks in a row that agree this well = the numbers do not move: the investigations change nothing
FROZEN_N, FROZEN_DB = 8, 0.005           # night: the microphone level that does not move by even this much in FROZEN_N readings is a dead input, not a quiet room
ROOM_QUIET_S = 60.0                      # guard mode: after a ROOM ALERT the same kind of alert waits this long (a person who stays in the room is not a new alarm every 4 s)
SCENE_CHANGE = 0.05                      # guard mode: this share of the 16 x 9 cells of the picture differs from the reference (5 % = 8 cells) -> "something changed in the room"
CELL_ABS, CELL_REL = 12.0, 0.20          # a cell differs when it is at least 12 grey levels and 20 % away from the reference (after a lamp's change of the whole picture is removed)
WARM_RISE = 0.25                         # guard mode: red / blue ratio of the picture rose by this share (a flame, a warm lamp)
MED_KEYS = ("brightness", "sharpness", "noise_db", "tone_snr", "tone_coupling", "warm")
ESC_QUIET_S = 120.0                      # after an ESCALATION that the person did not accept: no new alarm for this many seconds, then the reference state is watched again
TONE_ASKS = 3                           # the check tone is not heard at any level: the PROGRAM (no model) asks the person this many times to raise the volume / quiet the room
MAX_RETRIES = 5                          # the program's own check says NOT back inside the limits: the agent is called again this many times, then a person is told


def _call(call: Callable, name: str, args: dict) -> dict:
    try:
        r = json.loads(call(name, args))
        return r if isinstance(r, dict) else dict(ok=False, error="unexpected answer")
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, error=f"{type(e).__name__}: {str(e)[:120]}")


def sample(call: Callable, names, tone: Optional[dict] = None, region: str = "scene") -> dict:
    """One set of readings. `tone` (level_db) also plays the check tone (a short beep). `region` is the part of the picture that is measured."""
    s: dict = dict(t=round(time.time(), 2), errors=[])
    if "measure_region" in names:
        r = _call(call, "measure_region", dict(region=region, n_frames=3, thumb=True))
        if r.get("ok"):
            s.update(brightness=float(r["mean_brightness"]), sharpness=float(r["sharpness_norm"]),
                     saturated=float(r["saturated_fraction"]))
            if isinstance(r.get("thumb"), list) and r["thumb"]:
                s["thumb"] = [float(v) for v in r["thumb"]]            # coarse grid of the picture (kept for the guard mode, not shown)
            if r.get("warm_ratio") is not None:
                s["warm"] = float(r["warm_ratio"])
        else:
            s["errors"].append("camera: " + str(r.get("error"))[:140])
    if "listen_noise" in names:
        r = _call(call, "listen_noise", dict(seconds=1.0))
        if r.get("ok"):
            s.update(noise_db=float(r["noise_rms_db"]), silent=bool(r.get("silent_input")))
        else:
            s["errors"].append("microphone: " + str(r.get("error"))[:140])
    if tone and "play_tone" in names:
        r = _call(call, "play_tone", dict(freq_hz=TONE_HZ, level_db=tone["level_db"], seconds=1.6))
        if r.get("ok"):
            s.update(tone_snr=float(r["snr_db"]), tone_heard=bool(r["heard"]), tone_interference=bool(r.get("interference")),
                     tone_silent=bool(r.get("silent_input")))
            if r.get("coupling_db") is not None:
                s["tone_coupling"] = float(r["coupling_db"])          # the tone's level at the microphone minus the level played
        else:
            s["errors"].append("tone: " + str(r.get("error"))[:140])
    return s


class Normal:
    """The normal values: the median of the last KEEP normal samples."""

    def __init__(self):
        self.items: deque = deque(maxlen=KEEP)
        self.tone_ref: Optional[float] = None    # how far above the noise the calibration beep was heard: the normal value of the check tone until beep samples are in
        self.coupling_ref: Optional[float] = None    # the same for the level at the microphone (tone_db - level_db)

    def add(self, s: dict) -> None:
        self.items.append(s)

    def clear(self) -> None:
        self.items.clear()
        self.tone_ref = None
        self.coupling_ref = None

    def snapshot(self) -> "Normal":
        n = Normal()
        n.items.extend(self.items)
        n.tone_ref, n.coupling_ref = self.tone_ref, self.coupling_ref
        return n

    def settled(self) -> bool:
        """Do the last WARMUP samples agree? (A camera that has just started changes its exposure and focus: on the real camera sharpness_norm
        jumped 187 -> 1000 while the picture settled, and in another run the first readings were about 1027 and the stable ones about 400.)"""
        last = list(self.items)[-WARMUP:]
        b = [x["brightness"] for x in last if x.get("brightness") is not None]
        h = [x["sharpness"] for x in last if x.get("sharpness") is not None]
        if b and max(b) - min(b) > max(10.0, 0.2 * statistics.median(b)):
            return False
        return not (h and (max(h) + 1.0) / (min(h) + 1.0) > SETTLE_RATIO)

    def ready(self) -> bool:
        """Enough samples that agree with each other; if they never agree, the normal values are taken after KEEP samples anyway."""
        n = len(self.items)
        return n >= WARMUP and (self.settled() or n >= KEEP)

    def med(self, key: str) -> Optional[float]:
        v = [x[key] for x in self.items if x.get(key) is not None]
        if v:
            return float(statistics.median(v))
        return {"tone_snr": self.tone_ref, "tone_coupling": self.coupling_ref}.get(key)

    def thumb(self) -> Optional[list]:
        """The coarse picture of the normal state: the median of every cell over the normal samples."""
        ts = [x["thumb"] for x in self.items if x.get("thumb")]
        if not ts or len({len(t) for t in ts}) != 1:
            return None
        return [float(statistics.median(c)) for c in zip(*ts)]

    def to_dict(self) -> dict:
        """What goes into the reference file: the median values, the coarse picture and the two beginnings of the check tone."""
        return dict(values={k: round(v, 3) for k in MED_KEYS if (v := self.med(k)) is not None},
                    thumb=[round(v, 1) for v in (self.thumb() or [])] or None, tone_ref=self.tone_ref, coupling_ref=self.coupling_ref)

    @classmethod
    def from_dict(cls, d: dict) -> "Normal":
        """The reverse. The file is not trusted: a wrong shape raises ValueError (the caller says so and measures a new reference)."""
        vals = d["values"]
        if not isinstance(vals, dict) or not isinstance(vals.get("brightness"), (int, float)):
            raise ValueError("the reference file has no brightness")
        item = {k: float(v) for k, v in vals.items() if k in MED_KEYS and isinstance(v, (int, float))}
        if d.get("thumb"):
            item["thumb"] = [float(v) for v in d["thumb"]]
        n = cls()
        for _ in range(KEEP):                               # KEEP equal samples: ready() and settled() are true at once
            n.items.append(dict(item))
        for k in ("tone_ref", "coupling_ref"):
            v = d.get(k)
            setattr(n, k, float(v) if isinstance(v, (int, float)) else None)
        return n


def judge_cats(base: Normal, s: dict, instrument_only: bool = False) -> list[tuple]:
    """What is different from normal in this sample: (category, fact with numbers) pairs. Categories: err, scene, noise, tone.
    instrument_only (guard mode): leave out the two readings that can also be the ROOM (the light, the room noise); what is left can only be a
    fault of the instrument: no answer, a collapse of sharpness (covered or dirty lens), a muted microphone, a weaker check tone."""
    out = [("err", e) for e in (s.get("errors") or [])]
    b0, b = base.med("brightness"), s.get("brightness")
    if not instrument_only and b0 is not None and b is not None and abs(b - b0) > max(BRIGHT_ABS, BRIGHT_REL * b0):
        out.append(("scene", f"scene brightness {b0:.0f} -> {b:.0f} grey levels"))
    h0, h = base.med("sharpness"), s.get("sharpness")
    if h0 is not None and h is not None:
        ratio = (h + 1.0) / (h0 + 1.0)
        if ratio < 1.0 / SHARP_RATIO:
            out.append(("scene", f"scene sharpness {h0:.1f} -> {h:.1f}"))
    n0, n = base.med("noise_db"), s.get("noise_db")
    if s.get("silent"):
        out.append(("noise", "the microphone delivers pure silence (muted?)"))
    elif not instrument_only and n0 is not None and n is not None and n - n0 > NOISE_RISE_DB:
        out.append(("noise", f"room noise {n0:.1f} -> {n:.1f} dBFS"))
    t0, t = base.med("tone_snr"), s.get("tone_snr")
    c0, c = base.med("tone_coupling"), s.get("tone_coupling")
    if s.get("tone_silent"):
        out.append(("tone", "the microphone delivered silence while the check tone played"))
    elif s.get("tone_interference"):
        pass                                                      # the room changed while the tone played: no verdict
    elif c0 is not None and c is not None and c0 - c > COUPLING_DROP_DB:
        out.append(("tone", f"the check tone is weaker at the microphone: {c0:.0f} -> {c:.0f} dB (its level there minus the level played; "
                            "the loudspeaker is quieter?)"))
    elif s.get("tone_heard") is False and t0 is not None and t is not None and t0 - t > TONE_DROP_DB / 2:
        out.append(("tone", f"the check tone is no longer heard (it was {t0:.0f} dB above the noise, now {t:.0f} dB)"))
    elif t0 is not None and t is not None and t0 - t > TONE_DROP_DB:
        out.append(("tone", f"the check tone is weaker at the microphone: {t0:.0f} -> {t:.0f} dB above the noise"))
    return out


def judge(base: Normal, s: dict) -> list[str]:
    return [t for _, t in judge_cats(base, s)]


def scene_change(ref_thumb: Optional[list], thumb: Optional[list]) -> Optional[float]:
    """The share (0..1) of the cells of the coarse picture that differ from the reference. Before the cells are compared, the whole picture is
    scaled to the brightness of the reference: a lamp that is switched on makes every cell brighter by about the same factor, and that is
    'the light', not 'something moved'. A person who stands in the room changes only the cells where he is."""
    if not ref_thumb or not thumb or len(ref_thumb) != len(thumb):
        return None
    a_mean, b_mean = sum(thumb) / len(thumb), sum(ref_thumb) / len(ref_thumb)
    if a_mean < 1.0 or b_mean < 1.0:
        return None
    k = b_mean / a_mean
    changed = sum(1 for a, b in zip(thumb, ref_thumb) if abs(a * k - b) > max(CELL_ABS, CELL_REL * b))
    return changed / len(thumb)


def judge_room(ref: Normal, s: dict) -> list[str]:
    """Guard mode: what in the ROOM differs from the stored reference (facts with numbers). It says nothing about the instrument: a covered lens
    changes the picture too. That is why the program tests the instrument before it calls a change a ROOM ALERT."""
    out = []
    b0, b = ref.med("brightness"), s.get("brightness")
    if b0 is not None and b is not None and abs(b - b0) > max(BRIGHT_ABS, BRIGHT_REL * b0):
        out.append(f"the light in the room changed: scene brightness {b0:.0f} -> {b:.0f} grey levels")
    ch = s.get("change")
    if ch is not None and ch >= SCENE_CHANGE:
        out.append(f"the picture of the room changed: {100 * ch:.0f} % of it differs from the reference (something moved, appeared or disappeared)")
    w0, w = ref.med("warm"), s.get("warm")
    if w0 and w and w / w0 > 1.0 + WARM_RISE:
        out.append(f"the picture became warmer: red/blue ratio {w0:.2f} -> {w:.2f} (a flame? a warm lamp?)")
    n0, n = ref.med("noise_db"), s.get("noise_db")
    if n0 is not None and n is not None and not s.get("silent") and n - n0 > NOISE_RISE_DB:
        out.append(f"room noise {n0:.1f} -> {n:.1f} dBFS (a voice? a bang? an alarm?)")
    return out


def parse_hours(text: str) -> tuple:
    """'22:00-07:00' -> (start, end) in minutes of the day; the window may cross midnight. ValueError with a sentence if it is not a window."""
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", str(text))
    if not m:
        raise ValueError(f"guard hours must look like 22:00-07:00 (start-end, the clock of this computer), not {text!r}")
    h1, m1, h2, m2 = (int(x) for x in m.groups())
    if h1 > 23 or h2 > 23 or m1 > 59 or m2 > 59:
        raise ValueError(f"guard hours {text!r}: hours are 0-23, minutes 0-59")
    a, b = h1 * 60 + m1, h2 * 60 + m2
    if a == b:
        raise ValueError(f"guard hours {text!r}: the start and the end are the same time")
    return a, b


def in_hours(hours: tuple, t: float) -> bool:
    """Is the moment t (seconds since 1970) inside the window of the day, by the clock of this computer?"""
    lt = time.localtime(t)
    m = lt.tm_hour * 60 + lt.tm_min
    a, b = hours
    return a <= m < b if a < b else (m >= a or m < b)


def window_start(hours: tuple, t: float) -> float:
    """When did the window that contains t begin? (an epoch time, to tell one night from the next)"""
    lt = time.localtime(t)
    m = lt.tm_hour * 60 + lt.tm_min
    a, b = hours
    day = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, a // 60, a % 60, 0, 0, 0, -1))
    return day - 86400.0 if (a > b and m < b) else day


def make_notifier(log_path=None, url: Optional[str] = None) -> Callable[[str], list]:
    """notify(text) -> list of sentences about what was done. Always: one line in the log file (if there is a path). If url is given
    (HW3_NOTIFY_URL, for example https://ntfy.sh/<your-secret-topic>, the phone gets a push message from the ntfy app): the text is POSTed there.
    On a Mac also a notification of the system. A failure of one way never stops the others, and the address itself is never printed."""
    import subprocess
    import sys
    import urllib.request
    from pathlib import Path

    def notify(text: str) -> list:
        done = []
        if log_path:
            try:
                with Path(log_path).open("a", encoding="utf-8") as fh:
                    fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + "  " + text.replace("\n", " ") + "\n")
                done.append(f"written to {Path(log_path).name}")
            except OSError as e:
                done.append(f"cannot write {log_path}: {e}")
        if url:
            try:
                kind = ("TROUBLE" if text.startswith("WATCH TROUBLE") else "ARMING" if text.startswith("GUARD ARMING") else "ROOM ALERT")
                req = urllib.request.Request(url, data=text.encode("utf-8"), method="POST",
                                             headers={"Title": f"Lab Watchdog {kind}", "Priority": "default" if kind == "ARMING" else "high",
                                                      "Tags": "warning"})
                with urllib.request.urlopen(req, timeout=6) as r:
                    done.append(f"push message sent (answer {r.status})")
            except Exception as e:  # noqa: BLE001
                done.append(f"push message FAILED: {type(e).__name__}")
        if sys.platform == "darwin":
            try:
                msg = text.replace('"', "'")[:180]
                title = "Lab Watchdog TROUBLE" if text.startswith("WATCH TROUBLE") else "Lab Watchdog ARMING" if text.startswith("GUARD ARMING") else "Lab Watchdog ROOM ALERT"
                subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "{title}"'], timeout=5, check=False)
                done.append("notification of the system shown")
            except Exception as e:  # noqa: BLE001
                done.append(f"system notification FAILED: {type(e).__name__}")
        return done

    return notify


def presence(s: dict) -> set:
    """Which kinds of reading this sample has (a sample without the check tone says nothing about the tone)."""
    p = {"err"}
    if "brightness" in s:
        p.add("scene")
    if "noise_db" in s or "silent" in s:
        p.add("noise")
    if "tone_snr" in s or "tone_silent" in s:
        p.add("tone")
    return p


def clean(s: dict) -> dict:
    """The sample as it goes into the normal values: a check tone that was not cleanly heard is not a normal value."""
    if s.get("tone_heard") is not True:
        return {k: v for k, v in s.items() if not k.startswith("tone_")}
    return s


def normal_text(base: Normal) -> str:
    parts = []
    for key, label, unit in (("brightness", "scene brightness", " grey levels"), ("sharpness", "sharpness_norm", ""),
                             ("noise_db", "room noise", " dBFS"), ("tone_snr", f"{TONE_HZ:g} Hz check tone above the noise", " dB"),
                             ("tone_coupling", "check tone coupling (its level at the microphone minus the level played)", " dB")):
        v = base.med(key)
        if v is not None:
            parts.append(f"{label} {v:.1f}{unit}")
    return ", ".join(parts) or "(not measured)"


def accepted_text(base: Normal, tone: Optional[dict]) -> str:
    """The limits inside which the watchdog itself calls a reading normal (the same rules as judge_cats), with numbers."""
    parts = []
    b0 = base.med("brightness")
    if b0 is not None:
        d = max(BRIGHT_ABS, BRIGHT_REL * b0)
        parts.append(f"scene brightness between {b0 - d:.0f} and {b0 + d:.0f} grey levels")
    h0 = base.med("sharpness")
    if h0 is not None:
        parts.append(f"sharpness_norm at least {(h0 + 1.0) / SHARP_RATIO - 1.0:.0f} (a third of the normal value; a higher value is fine)")
    n0 = base.med("noise_db")
    if n0 is not None:
        parts.append(f"room noise at most {n0 + NOISE_RISE_DB:.1f} dBFS, and not pure silence")
    c0, t0 = base.med("tone_coupling"), base.med("tone_snr")
    if tone and c0 is not None:
        parts.append(f"the {TONE_HZ:g} Hz check tone arriving at the microphone with a coupling (tone_db minus level_db) of at least "
                     f"{c0 - COUPLING_DROP_DB:.0f} dB")
    elif tone and t0 is not None:
        parts.append(f"the {TONE_HZ:g} Hz check tone heard at least {t0 - TONE_DROP_DB:.0f} dB above the noise")
    return "; ".join(parts)


HINTS = {   # what the program knows about this instrument from the real runs, as hypotheses to test; a file (--hints) may replace any of them
    "cover": (
        'observed on this camera: when the lens is covered by a hand the brightness stays near normal (the automatic exposure '
        'brightens the dark picture) and sharpness_norm collapses to a small part of normal; so a sharpness collapse with '
        'normal brightness points to a covered, dirty or blocked lens. Only a person can uncover it: ask for exactly that. '
        'Right after the lens is uncovered the camera needs about 10 s to readjust and the first readings can still be low; the '
        'program waits before your next turn. After uncovering, a reading that is inside the limits above means the instrument '
        'works: say so and finish. Ask to wipe the lens ONLY if two readings a few seconds apart are still BELOW the sharpness '
        'limit above'
    ),
    "fell": (
        'the brightness FELL: the light changed, OR something is in front of the lens (a hand, a paper, a cap, a part of the '
        'lens covered). The readings alone cannot tell these apart: a partly covered lens keeps a normal contrast and sharpness '
        'and darkens only a part of the picture (one region can be darker while the others stay bright), so do NOT conclude '
        "'the light changed' from a good contrast or a bright other region. You cannot change the light, and a person can "
        "remove an obstacle: ask the person ONE concrete action BEFORE you conclude that the light changed: 'Please check that "
        "nothing is in front of the camera lens (a hand, a paper, a cap) and remove it, then reply done.' After the answer "
        'measure once more; only if the readings are still outside the limits report that the light is lower'
    ),
    "rose": (
        'the brightness ROSE: the light changed (a lamp, a window); you cannot change the light: report what you measured, no '
        'person is needed unless the lens or the camera was moved'
    ),
    "camera_only": (
        'the alarm is about the camera, the sound side is not involved: do not play tones; at most one listen_noise as a '
        'cross-check'
    ),
    "noise": (
        'room noise rose. Do NOT ask the person to be quiet or to turn anything down. Act yourself: play the check tone at the '
        'usual level; if heard is true and coupling_db is within about 6 dB of the normal value (given above, if there is one), '
        "report 'the room is louder by X dB, the instrument still works, nothing to do' and finish; if it is not, raise "
        'level_db yourself (-20, then -12, the maximum) until heard is true and report the level you needed'
    ),
    "tone": (
        'the check tone is weaker or lost (the volume of the loudspeaker was lowered, or the room is louder). COMPENSATE '
        "YOURSELF: play the tone again at the usual level and read coupling_db in the answer (the tone's level at the "
        'microphone minus the level played; the normal value is given above): the difference to the normal value is the loss. '
        'Raise level_db by about that loss (-12 dBFS is the maximum) and play again until coupling_db is back near the normal '
        'value; report the loss in dB and that the instrument works again. Do not look for a margin above the noise of 15 dB: '
        'in a reflecting room it stops growing with the level. Ask the person to press the volume button only if even -12 dBFS '
        'does not bring the tone back'
    ),
    "silence": (
        'the microphone delivers pure silence: it is muted, and a louder tone cannot help; only a person can unmute it: ask for '
        'exactly that'
    ),
    "dead": (
        'the instrument itself stopped answering: the cable or the power; only a person can check them'
    ),
}
HINT_KEYS = tuple(HINTS)


def load_hints(path) -> list:
    """Replace hints by those of a JSON file {"fell": "...", ...} (the keys of HINTS). The hints were written after the incidents of ONE camera;
    another device may need other words, and they must be changeable without touching the code. Returns the keys that were replaced; an unknown key
    or a text that is not a string raises ValueError and changes nothing."""
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("the hints file must be a JSON object {\"fell\": \"...\"}")
    bad = [k for k in data if k not in HINTS]
    if bad:
        raise ValueError(f"unknown hint {bad[0]!r}; the keys are: {', '.join(HINT_KEYS)}")
    if any(not isinstance(v, str) or not v.strip() for v in data.values()):
        raise ValueError("every hint must be a non-empty text")
    HINTS.update({k: v.strip() for k, v in data.items()})
    return sorted(data)


def hints(facts: list[str]) -> str:
    """What the program knows about this instrument from the real runs, as hypotheses to test (the agent still chooses its checks)."""
    t = " ".join(facts)
    silence = "silence" in t
    h = []
    if "sharpness" in t and "brightness" not in t:
        h.append(HINTS["cover"])
    if "brightness" in t:
        m = re.search(r"brightness (\d+(?:\.\d+)?) -> (\d+(?:\.\d+)?)", t)
        fell = bool(m) and float(m.group(2)) < float(m.group(1))
        h.append(HINTS["fell"] if fell else HINTS["rose"])
    if ("sharpness" in t or "brightness" in t) and not any(k in t for k in ("room noise", "check tone", "silence", "camera:", "microphone:", "tone:")):
        h.append(HINTS["camera_only"])
    if "room noise" in t:
        h.append(HINTS["noise"])
    if "check tone" in t and not silence:
        h.append(HINTS["tone"])
    if silence:
        h.append(HINTS["silence"])
    if "camera:" in t or "microphone:" in t or "tone:" in t:
        h.append(HINTS["dead"])
    return (" What the program knows (hypotheses to test, the checks are yours): " + "; ".join(h) + ".") if h else ""


def alert_request(facts: list[str], base: Normal, tone: Optional[dict]) -> str:
    tone_note = (f" The check tone is {TONE_HZ:g} Hz at {tone['level_db']:g} dBFS." if tone else "")
    limits = accepted_text(base, tone)
    limits_note = (f" The watchdog itself calls the instrument normal when: {limits}. Use exactly these limits when you say whether a reading "
                   "is back to normal: a reading inside them is normal even if it differs from the normal value (the normal values are only the "
                   "median of the last readings, and sharpness_norm changes a lot with the picture); then no further action is needed."
                   if limits else "")
    return ("ALERT from the watchdog (a program that measures the instrument every few seconds, without a model): "
            + "; ".join(facts) + f". Normal values until now: {normal_text(base)}.{tone_note}{limits_note}{hints(facts)} "
            "Find out what is wrong with the instrument or its surroundings. Take your own readings with the camera and the "
            "microphone, choose the checks yourself and compare them with the normal values (do not repeat the same measurement more than "
            "twice). You act yourself wherever the instrument allows it (for example the level of the check tone). Ask the person ONLY for "
            "what nothing but a person can do (uncover the lens, press the volume or mute button, unmute the microphone), as ONE concrete "
            "action (for example 'Please uncover the lens and reply done'), never a general request to 'check' things, and never to be "
            "quiet, turn music down or change the light. After the person says done, measure once more and compare with the limits; "
            "if the readings are inside them, say so and finish. "
            "Be economical, every step takes about 10 s: one hardware turn in which the agent takes all the independent readings you need "
            "together (not one by one), then at most one question to the person, then one hardware turn to verify, then finish; do not call "
            "camera_info (the program knows the instrument); tell the hardware agent to report in at most 3 sentences, without tables. "
            "Your final message has at most 5 short lines: the diagnosis, what you did, and the key numbers before and after (the program "
            "adds the table of all readings).")


def retry_request(left: list[str], base: Normal, tone: Optional[dict], n: int) -> str:
    """The same alarm once more: the first investigation is over, then the PROGRAM (no model) measured again and the instrument is still
    outside the limits of the alert. The agent gets the numbers of that check, so the loop closes on a measurement and not on its words."""
    return (f"NEXT INVESTIGATION (number {n} of {MAX_RETRIES + 1}). The first investigation is over, and then the PROGRAM, without a model, "
            "measured the instrument again: it is still NOT back inside the limits: " + "; ".join(left) + ". The last answer was not "
            "enough: do not repeat it, take new readings and find what still differs. If a person can fix it (the lens is still covered, the "
            "volume is still too low, the microphone is still muted), ask the person AGAIN, as ONE concrete action, even if you asked before. If the cause cannot be changed by you or by the "
            "person (for example the room stays louder), say so plainly in the diagnosis and give the key numbers. "
            + alert_request(left, base, tone))


def steady_environment(prev: Optional[dict], cur: Optional[dict], left: list[str]) -> bool:
    """Is what stays outside the limits only the ROOM (the light, the noise), and did two checks in a row give the same numbers? Then another
    investigation cannot change anything: an agent cannot switch the light or make the room quiet, and the loop should say ESCALATION now instead of
    spending five more rounds (minutes and money) on it."""
    if not prev or not cur or not left:
        return False
    if not all(f.startswith(("scene brightness", "room noise")) for f in left):
        return False
    b0, b1 = prev.get("brightness"), cur.get("brightness")
    n0, n1 = prev.get("noise_db"), cur.get("noise_db")
    if b0 is not None and b1 is not None and abs(b1 - b0) > max(STEADY_BRIGHT_ABS, STEADY_BRIGHT_REL * abs(b0)):
        return False
    if n0 is not None and n1 is not None and abs(n1 - n0) > STEADY_NOISE_DB:
        return False
    return True


class MinuteSummary:
    """One CSV line per minute: how many samples, how many 'changed', the range of the readings, the alarms. A night is about 100 bytes a minute
    (the full log of the instrument rotates and is large). This is what night_report.py reads to count false alarms and to suggest limits."""
    FIELDS = ("minute", "samples", "changed", "errors", "brightness_min", "brightness_mean", "brightness_max", "sharpness_mean",
              "noise_mean", "noise_max", "change_max", "anomalies", "room_alerts")

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._minute: Optional[str] = None
        self._acc = self._fresh()

    @staticmethod
    def _fresh() -> dict:
        return dict(samples=0, changed=0, errors=0, b=[], h=[], n=[], c=[], anomalies=0, room_alerts=0)

    def feed(self, ev: dict) -> None:
        t = ev.get("type")
        if t not in ("sample", "anomaly", "room_alert"):
            return
        minute = time.strftime("%Y-%m-%d %H:%M", time.localtime(ev.get("t") or time.time()))
        with self._lock:
            if self._minute is not None and minute != self._minute:
                self._flush()
            self._minute = minute
            a = self._acc
            if t == "anomaly":
                a["anomalies"] += 1
            elif t == "room_alert":
                a["room_alerts"] += 1
            else:
                a["samples"] += 1
                a["changed"] += 1 if ev.get("status") == "changed" else 0
                a["errors"] += 1 if ev.get("errors") else 0
                for key, lst in (("brightness", a["b"]), ("sharpness", a["h"]), ("noise_db", a["n"]), ("change", a["c"])):
                    if ev.get(key) is not None:
                        lst.append(float(ev[key]))

    def flush(self) -> None:
        with self._lock:
            self._flush()

    def _flush(self) -> None:
        a, m = self._acc, self._minute
        self._acc, self._minute = self._fresh(), None
        if m is None or not (a["samples"] or a["anomalies"] or a["room_alerts"]):
            return
        mean = lambda v: round(sum(v) / len(v), 2) if v else ""
        row = [m, a["samples"], a["changed"], a["errors"], round(min(a["b"]), 1) if a["b"] else "", mean(a["b"]), round(max(a["b"]), 1) if a["b"] else "",
               mean(a["h"]), mean(a["n"]), round(max(a["n"]), 2) if a["n"] else "", round(max(a["c"]), 3) if a["c"] else "", a["anomalies"], a["room_alerts"]]
        try:
            new = not self.path.exists() or self.path.stat().st_size == 0
            with self.path.open("a", encoding="utf-8") as fh:
                if new:
                    fh.write(",".join(self.FIELDS) + "\n")
                fh.write(",".join(str(x) for x in row) + "\n")
        except OSError:
            pass


class Watchdog:
    """The loop. `call(name, args)` sends a command to the instrument; `emit(event)` shows what happens; `investigate(request,
    facts)` runs the agent on the request and returns when it has finished."""

    def __init__(self, call: Callable, names, emit: Callable[[dict], None], investigate: Callable[[str, list], None],
                 interval: float = 4.0, tone_every: float = 0.0, ask: Optional[Callable[[str], Optional[str]]] = None,
                 guard: bool = False, ref_path=None, fresh_reference: bool = False,
                 notify: Optional[Callable[[str], list]] = None, region: str = "scene",
                 guard_hours: Optional[tuple] = None, guard_ref_path=None, clock: Callable[[], float] = time.time):
        self.call, self.names, self.emit, self.investigate = call, set(names), emit, investigate
        self.region = region               # the part of the picture that is watched ('scene' = the region of roi.json; in guard mode it must cover where a person could stand)
        self.guard_hours = tuple(guard_hours) if guard_hours else None     # (start, end) in minutes of the day: the room is watched only inside this window
        self.guard_ref_path = Path(guard_ref_path) if guard_ref_path else None
        self.clock = clock                 # the clock for the window of the night (a test sets its own)
        self._gactive, self._gref, self._gbuf, self._gstart = False, None, Normal(), 0.0   # the window is open / the reference of this night / its samples / when we began to wait
        self._gwin = 0.0                   # the start of the window that is open
        self.guard = bool(guard) or self.guard_hours is not None   # guard mode: the ROOM is watched against the reference too (somebody should not be here)
        self.ref_path = Path(ref_path) if ref_path else None    # the reference file; None = the reference lives only in memory
        self.fresh_reference = bool(fresh_reference)           # ignore the stored file and measure a new reference (which then replaces it)
        self.notify = notify               # notify(text) -> what was done; called at a ROOM ALERT
        self._stored: Optional[Normal] = None                  # the reference that was read from the file at the start
        self._stored_at = ""
        self._persist_next = self.ref_path is not None          # the next reference that is taken is written to the file (first run, or a person accepted)
        self._room_quiet_until = 0.0
        self._stop = threading.Event()
        self._pushed: dict = {}            # key of a push message -> when it was sent (the same trouble is not pushed again at once)
        self.last_step_t = time.time()     # the last time the loop measured something (the heartbeat asks)
        self.trouble_since: Optional[float] = None    # since when does the instrument answer with errors only
        self._trouble_told = False         # a push about a lost instrument was sent: when it is back, another one says so
        self.state = "sampling"            # sampling | investigating | asking: a loop that waits for the agent or a person is not stuck
        self._postponed_until = 0.0        # 'I am in the lab': guard hours are paused until this time (the clock of the watchdog)
        self._fresh_night = False          # after a pause the room of the night is measured anew (the people changed it)
        self._gprev: Optional[Normal] = None    # the reference of the previous night (to see that tonight's empty room is not another room)
        self._noise_hist: deque = deque(maxlen=FROZEN_N)
        self._last_check: dict = {}        # the numbers of the last check after an investigation
        self.ask = ask                     # ask(question) -> the person's reply, or None if the run was stopped (the pause bar of the page)
        self._asked = False                # the person was asked in this episode of a lost check tone: not again until the tone is heard once
        self.interval, self.tone_every = float(interval), float(tone_every)
        self.base = Normal()
        self.ref: Optional[Normal] = None  # the reference state (the first normal values): an unfixable problem does not silently become the normal
        self._quiet_until = 0.0            # after an ESCALATION the person was told: no new alarm until this time
        self.tone: Optional[dict] = None  # None = no level found yet: the tone check is off and is tried again every RECAL_S seconds
        self._cal_t = 0.0
        self._said_off = False
        self.hist: dict = {}               # per kind of reading: did the last WINDOW samples disagree with the normal values?
        self.settle = 3.0 * self.interval  # after an investigation: wait for the camera (autofocus, exposure) and the room to settle
        self._last_tone = 0.0

    # ---- telling a person outside the page, and showing that the loop is alive
    def push(self, key: str, text: str, quiet: float = TROUBLE_QUIET_S) -> None:
        """Tell a person who is not looking at the page (the notifier: log file, notification of the system, push message). The same key is not
        repeated within `quiet` seconds. What was done is written on the page too."""
        now = time.time()
        if now - self._pushed.get(key, 0.0) < quiet:
            return
        self._pushed[key] = now
        done: list = []
        if self.notify is not None:
            try:
                done = self.notify(text)
            except Exception as e:  # noqa: BLE001
                done = [f"notification FAILED: {type(e).__name__}"]
        self.emit(dict(type="note", text=text + (f"  [{'; '.join(done)}]" if done else "  [page only: no notifier]")))

    def _observe(self, s: dict) -> None:
        """Every reading goes through here: the loop is alive, and an instrument that answers again after trouble is told to the person."""
        self.last_step_t = time.time()
        if s.get("errors"):
            if self.trouble_since is None:
                self.trouble_since = self.last_step_t
        else:
            if self._trouble_told:
                self._trouble_told = False
                self.push("back", "WATCH TROUBLE ended: the instrument answers again.", quiet=0.0)
            self.trouble_since = None

    def health(self) -> tuple:
        """(healthy, why not). The heartbeat uses it: it is sent only while the loop makes its steps and the instrument answers. A dead
        program, a sleeping Mac, a lost camera or a loop that hangs all stop it, and the service that waits for it says so."""
        now = time.time()
        if self.trouble_since is not None and now - self.trouble_since > BLIND_S:
            return False, f"the instrument has not answered for {now - self.trouble_since:.0f} s"
        limit = (20.0 * self.interval + 120.0) if self.state == "sampling" else (ASK_TIMEOUT_S + 900.0)     # waiting for the agent or a person takes longer
        if now - self.last_step_t > limit:
            return False, f"the loop has made no step for {now - self.last_step_t:.0f} s"
        return True, ""

    def _quiet(self) -> bool:
        """Is it guard night right now (hours set, inside the window, not paused)? Then nothing is played: a beep in an empty room is heard by
        whoever is in it. It looks at the clock itself, so it is right before the first step and at the first step of a window."""
        if self.guard_hours is None:
            return False
        now = self.clock()
        return in_hours(self.guard_hours, now) and now >= self._postponed_until

    def _ask(self, question: str) -> Optional[str]:
        prev, self.state = self.state, "asking"
        try:
            return self.ask(question) if self.ask is not None else None
        finally:
            self.state = prev

    def postpone(self, minutes: float) -> str:
        """'I am in the lab': guard hours are paused for `minutes` (0 = resume now). The room is measured anew afterwards."""
        m = max(0.0, min(float(minutes), 720.0))
        now = self.clock()
        self._postponed_until = now + m * 60.0 if m else 0.0
        self._fresh_night = True
        if m:
            until = time.strftime("%H:%M", time.localtime(self._postponed_until))
            text = f"GUARD ARMING: guard hours are paused until {until} (somebody is in the lab); then the room is measured anew."
        else:
            text = "GUARD ARMING: the pause is cancelled, the guard hours go on by the clock."
        self.push("postpone", text, quiet=0.0)
        return text

    def calibrate_tone(self) -> None:
        """Find a level at which the check tone is heard (at once if it is TONE_NEEDED_DB above the noise, else the best level that is heard
        at all: in a reflecting room the margin stops growing with the level). The normal value of the check is the tone's level at the
        microphone, tone_db - level_db (coupling): it falls when the loudspeaker gets quieter, whatever the noise does. If no level is heard,
        the tone check is off for now (it says so, with the numbers) and is tried again every RECAL_S seconds."""
        if not self.tone_every or "play_tone" not in self.names or self._quiet():
            return                                          # (at night nothing is played: a beep in an empty room is heard by whoever is there)
        self._cal_t = time.time()
        tried, best = [], None
        for lv in TONE_LEVELS:
            r = _call(self.call, "play_tone", dict(freq_hz=TONE_HZ, level_db=lv, seconds=1.6))
            if r.get("ok") and r.get("heard") and not r.get("interference") and float(r.get("snr_db", 0)) >= TONE_MIN_DB:
                snr = float(r["snr_db"])
                if best is None or snr > best[1] + 1.0:           # the quieter level wins unless a louder one is clearly better
                    best = (lv, snr, r)
                if snr >= TONE_NEEDED_DB:
                    break
                tried.append(f"{lv:g} dBFS: {snr:g} dB")
                continue
            tried.append(f"{lv:g} dBFS: " + (f"{float(r.get('snr_db', 0)):g} dB" + (", interference" if r.get("interference") else "")
                                             if r.get("ok") else "error " + str(r.get("error"))[:60]))
        if best is not None:
            lv, snr, r = best
            self.tone = dict(level_db=lv, snr=snr)
            self.base.tone_ref = snr                           # the normal values of the check tone from the very start: a volume turned down before the first beep sample is caught too
            self.base.coupling_ref = float(r["coupling_db"]) if r.get("coupling_db") is not None else None
            self._said_off = False
            self._asked = False
            self.emit(dict(type="note", text=f"check tone: {TONE_HZ:g} Hz at {lv:g} dBFS is heard {snr:g} dB above the noise"
                                             + (f", {r['tone_db']:g} dB at the microphone (coupling {r['coupling_db']:g} dB)" if r.get("coupling_db") is not None else "")
                                             + ("" if snr >= TONE_NEEDED_DB else "; the margin does not grow with the level in this room, so the check watches "
                                                                                    "the tone's level at the microphone")))
            return
        self.tone = None
        self.base.tone_ref = self.base.coupling_ref = None
        if not self._said_off:
            self._said_off = True
            self.emit(dict(type="note", text=f"the check tone is not heard ({'; '.join(tried)}; it must be heard, at least {TONE_MIN_DB:g} dB above "
                                             f"the noise, at one of the levels): the tone check is off, I try again every {RECAL_S:g} s (raise "
                                             "the volume of the device or make the room quieter)"))

    def secure_tone(self, stop: threading.Event) -> None:
        """The check tone is not heard at any level (the program may not go above -12 dBFS and cannot turn the volume up). The program, WITHOUT a
        model, asks the person up to TONE_ASKS times for the one thing only a person can do (the volume of the device, a quieter room) and
        calibrates again after every answer. If it is still not heard, an ESCALATION note says so: the sound check stays off and is tried
        again every RECAL_S seconds. The person is asked once per episode, not every RECAL_S seconds."""
        if self.ask is None or self._asked or self.tone is not None or not self.tone_every or "play_tone" not in self.names or self._quiet():
            return
        self._asked = True
        for i in range(TONE_ASKS):
            if stop.is_set():
                return
            q = ("The check tone is not heard at any level, so the sound side cannot be watched. Please raise the volume of the Pano-360 "
                 "to the maximum with its own buttons and keep the room quiet for 20 seconds, then reply done."
                 if i == 0 else
                 f"Still not heard (request {i + 1} of {TONE_ASKS}). Please raise the volume of the Pano-360 to the maximum, move closer to "
                 "it if you can, keep the room quiet for 20 seconds, then reply done.")
            r = self._ask(q)
            if r is None or stop.is_set():
                return
            self.calibrate_tone()
            if self.tone is not None:
                return
        self.emit(dict(type="note", text=f"ESCALATION: the check tone is still not heard after {TONE_ASKS} requests to the person. The sound "
                                         "side is NOT watched now (the room is too loud or the loudspeaker too quiet at the maximum level the "
                                         "program may use, -12 dBFS); the camera is still watched, and the tone check is tried again every "
                                         f"{RECAL_S:g} s."))
        self.push("tone-escalation", "WATCH TROUBLE: ESCALATION: the check tone is not heard at any level; the sound side is NOT watched, the camera still is.")

    def step(self) -> Optional[list[str]]:
        """One sample. Returns the facts if the alarm goes off, else None. A kind of reading (scene, noise, check tone) must disagree
        with the normal values in CONFIRM of the last WINDOW samples that contain it; a sample without the check tone says nothing about the tone."""
        quiet = self._quiet()
        if self.tone is None and self.tone_every and not quiet and time.time() - self._cal_t >= RECAL_S:
            self.calibrate_tone()
        use_tone = bool(self.tone and self.tone_every and not quiet and time.time() - self._last_tone >= self.tone_every)
        s = sample(self.call, self.names, self.tone if use_tone else None, self.region)
        if use_tone:
            self._last_tone = time.time()
        if self.ref is None and self.base.ready():
            self.take_reference()              # the reference state: kept until a person accepts another state, and written to the file
        gnow = self._guard_tick(s)                                              # is the room watched now? (always in --guard, inside the window with --guard-hours)
        if self.guard_hours is not None and gnow and s.get("noise_db") is not None and not s.get("silent"):
            self._noise_hist.append(s["noise_db"])                              # the quiet check of the microphone at night (no beep): its level must move a little
            if len(self._noise_hist) == FROZEN_N and max(self._noise_hist) - min(self._noise_hist) < FROZEN_DB:
                s["errors"].append(f"microphone: the level does not move at all in {FROZEN_N} readings (a dead input, not a quiet room)")
        elif not gnow:
            self._noise_hist.clear()
        self._observe(s)
        rref = self._gref if self.guard_hours is not None else self.ref         # the reference of the room: of the night (guard hours) or the stored one
        shown = rref if rref is not None else self.ref
        if shown is not None and s.get("thumb"):
            s["change"] = scene_change(shown.thumb(), s["thumb"])              # how much of the picture differs from the reference (None = unknown)
        cats = judge_cats(self.base, s, instrument_only=gnow) if self.base.ready() else []
        rooms = judge_room(rref, s) if (gnow and rref is not None and not s["errors"]) else []
        by_cat: dict = {}
        for c, t in cats:
            by_cat.setdefault(c, []).append(t)
        for c in presence(s):
            self.hist.setdefault(c, deque(maxlen=WINDOW)).append(c in by_cat)
        if gnow and rref is not None:
            self.hist.setdefault("room", deque(maxlen=WINDOW)).append(bool(rooms))
        facts = [t for _, t in cats]
        broken = bool(s["errors"]) and not self.base.ready()      # no normal values yet and the instrument does not answer
        status = ("waiting for the instrument" if broken else "warming up" if not self.base.ready()
                  else ("changed" if (facts or rooms) else "normal"))
        self.emit(dict(type="sample", status=status, facts=facts + rooms, **{k: v for k, v in s.items() if k not in ("errors", "t", "thumb")},
                       errors=s["errors"], t=s["t"], normal={k: self.base.med(k) for k in ("brightness", "sharpness", "noise_db", "tone_snr", "tone_coupling")}))
        if broken:
            return None                     # an error is not a normal value, and there is nothing to compare with: no alarm, no model call
        if not facts and not rooms:
            self.base.add(clean(s))
            return None
        if time.time() < self._quiet_until:
            return None                     # a person was just told about this state: no new alarm for a while
        if facts and any(sum(self.hist.get(c, ())) >= CONFIRM for c in by_cat):
            return facts                    # the instrument itself: the loop of the agent
        if rooms and sum(self.hist.get("room", ())) >= CONFIRM and time.time() >= self._room_quiet_until:
            return self.room_event(rooms)   # the room: the instrument is tested first, then a person is told (or, if the instrument is not in order, the loop of the agent)
        return None

    # ---- the reference file
    def load_stored(self) -> None:
        """Read the reference of an earlier run (once, at the start). A damaged file is said aloud and ignored: nothing is guessed from it."""
        self._stored = None
        if self.ref_path is None or self.fresh_reference:
            if self.ref_path is not None and self.fresh_reference:
                self.emit(dict(type="note", text="reference file ignored on purpose (fresh reference): a new reference is measured and replaces it"))
            return
        if not self.ref_path.exists():
            self.emit(dict(type="note", text=f"no reference file yet ({self.ref_path.name}): the first normal values of this run become the reference and are saved"))
            return
        try:
            d = json.loads(self.ref_path.read_text(encoding="utf-8"))
            if d.get("region", self.region) != self.region:
                self.emit(dict(type="note", text=f"the stored reference is of the region {d.get('region')!r}, now {self.region!r} is watched: "
                                                 "a new reference is measured and replaces it"))
                return
            self._stored, self._stored_at = Normal.from_dict(d), str(d.get("saved_at", "an earlier run"))
            self._persist_next = False
            self.emit(dict(type="note", text=f"reference read from {self.ref_path.name} (saved {self._stored_at}): "
                                             f"{normal_text(self._stored)}"))
        except Exception as e:  # noqa: BLE001
            self._stored = None
            self.emit(dict(type="note", text=f"the reference file {self.ref_path.name} cannot be used ({type(e).__name__}: {str(e)[:100]}): "
                                             "a new reference is measured and replaces it"))

    def save_reference(self) -> None:
        if self.ref_path is None or self.ref is None:
            return
        d = self.ref.to_dict()
        d.update(saved_at=time.strftime("%Y-%m-%d %H:%M:%S"), version=1, region=self.region)
        try:
            tmp = self.ref_path.with_suffix(self.ref_path.suffix + ".tmp")
            tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
            os.replace(tmp, self.ref_path)                       # the old file is never half-written
            self.emit(dict(type="note", text=f"the reference is saved to {self.ref_path.name}: {normal_text(self.ref)}"))
        except OSError as e:
            self.emit(dict(type="note", text=f"the reference could NOT be saved ({e}); it stays in memory"))

    def take_reference(self) -> None:
        """The first normal values of this run are ready. No stored reference: they ARE the reference (saved). A stored one: the state now is
        compared with it; if they agree the stored reference stays (it is not rewritten), if not a PERSON decides - program cannot know whether
        the lab was repaired or has become different."""
        if self._gactive:                  # guard hours, night: the state of the night is not the state of the day: nobody is asked, nothing is saved
            self.ref = self.base.snapshot()
            return
        st, self._stored = self._stored, None
        if st is None:
            self.ref = self.base.snapshot()
            if self._persist_next:
                self.save_reference()
                self._persist_next = False
            return
        now = {k: v for k in MED_KEYS if (v := self.base.med(k)) is not None}
        if self.base.thumb():
            now["thumb"] = self.base.thumb()
        now["change"] = scene_change(st.thumb(), now.get("thumb"))
        diff = [t for _, t in judge_cats(st, now)] + (judge_room(st, now) if self.guard and self.guard_hours is None else [])
        if not diff:
            self.ref = st
            self.emit(dict(type="note", text=f"the state now agrees with the stored reference of {self._stored_at}: the stored reference stays"))
            return
        self.emit(dict(type="note", text=f"the state now is NOT the stored reference of {self._stored_at}: " + "; ".join(diff)))
        q = (f"The state now differs from the reference stored on {self._stored_at}: " + "; ".join(diff) + f". The stored reference was: "
             f"{normal_text(st)}. If this is the normal state of the lab now (the light, the camera, the room changed on purpose), reply \"accept\": "
             "I save the present state as the new reference. Reply \"keep\" (or anything else): I watch against the stored reference and call "
             "the agent about the difference.")
        r = self._ask(q) if self.ask is not None else None
        if r is not None and r.strip().lower().startswith(("accept", "yes")):
            self.ref = self.base.snapshot()
            self.save_reference()
            self.emit(dict(type="note", text="the person accepted the present state: it replaces the stored reference"))
            return
        self.ref = st
        self.base = st.snapshot()                                 # the stored state is the normal: the difference is an alarm now
        self.hist.clear()
        self.emit(dict(type="note", text="the stored reference stays; the watch goes on against it" + ("" if self.ask is not None else " (nobody to ask)")))

    # ---- guard hours
    def _guard_tick(self, s: dict) -> bool:
        """Called with every sample. Without hours: guard mode is on or off for the whole run. With hours: the window opens and closes by the clock,
        and at both moments the normal values are learned anew (the lights go off in the evening and on in the morning: that is not a fault). After
        the window opened and GUARD_SETTLE_S passed, the first settled samples are the reference of the NIGHT."""
        if self.guard_hours is None:
            return self.guard
        now = self.clock()
        paused = now < self._postponed_until
        active = in_hours(self.guard_hours, now) and not paused
        if active and not self._gactive:
            self._gactive, self._gstart, self._gwin = True, now, window_start(self.guard_hours, now)
            self._gref, self._gbuf = None, Normal()
            self.base.clear()
            self.hist.clear()
            self._noise_hist.clear()
            fresh, self._fresh_night = self._fresh_night, False
            if not fresh and self._load_night(self._gwin):
                self.emit(dict(type="note", text=f"GUARD HOURS are on: the reference of this night was read back from {self.guard_ref_path.name} "
                                                 f"(a restart in the night does not forget it): {normal_text(self._gref)}"))
            else:
                self.emit(dict(type="note", text=f"GUARD HOURS began: the room is left alone for {GUARD_SETTLE_S:g} s (people leave, the lights go off), "
                                                 "then the room as it is becomes the reference of the night and every change is a room alert"))
                self.push("arming", f"GUARD ARMING: the room is taken as empty in {GUARD_SETTLE_S:g} s and then guarded. If somebody is still in the lab, "
                                    "press \"I am in the lab\" on the page now.", quiet=0.0)
        elif not active and self._gactive:
            self._gactive, self._gref = False, None
            self.base.clear()
            self.hist.clear()
            if self.tone_every:
                self.calibrate_tone()                           # (the beep is allowed again: the night is over)
            self.emit(dict(type="note", text=("GUARD paused (somebody is in the lab): the room is not watched until the pause ends"
                                              if paused else "GUARD HOURS ended: the room is not watched any more, the normal values are learned anew")))
        if active and self._gref is None and now - self._gstart >= GUARD_SETTLE_S and not s["errors"]:
            self._gbuf.add(clean(s))
            if self._gbuf.ready():
                self._gref = self._gbuf.snapshot()
                self.emit(dict(type="note", text="the reference of the night is taken (the room is assumed to be empty now): " + normal_text(self._gref)
                                                 + ". From now on every change of the room is an alert"))
                self._compare_with_last_night()
                self._save_night()
        return active

    def _save_night(self) -> None:
        if self.guard_ref_path is None or self._gref is None:
            return
        d = self._gref.to_dict()
        d.update(saved_at=time.strftime("%Y-%m-%d %H:%M:%S"), version=1, region=self.region, window_start=self._gwin)
        if self._gprev is not None:
            d["previous"] = self._gprev.to_dict()                  # last night's room, to compare tonight's with
        try:
            tmp = self.guard_ref_path.with_suffix(self.guard_ref_path.suffix + ".tmp")
            tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
            os.replace(tmp, self.guard_ref_path)
        except OSError as e:
            self.emit(dict(type="note", text=f"the reference of the night could NOT be saved ({e}); it stays in memory"))

    def _load_night(self, win: float) -> bool:
        """The stored reference of THIS night (the same window, the same region), if there is one. Also remembers last night's room."""
        self._gprev = None
        if self.guard_ref_path is None or self.fresh_reference or not self.guard_ref_path.exists():
            return False
        try:
            d = json.loads(self.guard_ref_path.read_text(encoding="utf-8"))
            same_region = d.get("region", self.region) == self.region
            if abs(float(d["window_start"]) - win) > 60.0 or not same_region:
                if same_region:
                    self._gprev = Normal.from_dict(d)           # that file is of an earlier night: it is 'last night'
                return False                                   # another night (or another region): the room is measured anew
            if isinstance(d.get("previous"), dict):
                self._gprev = Normal.from_dict(d["previous"])
            self._gref = Normal.from_dict(d)
            return True
        except Exception:  # noqa: BLE001
            self._gref = None
            self._gprev = None
            return False

    def _compare_with_last_night(self) -> None:
        """Tonight's 'empty' room against last night's. If they differ strongly, somebody may still be in the room, or the light was left on: the
        reference of the night is taken anyway (nothing is blocked), but a person is told while there is still time to look."""
        if self._gprev is None or self._gref is None:
            return
        now = {k: v for k in MED_KEYS if (v := self._gref.med(k)) is not None}
        now["thumb"] = self._gref.thumb()
        now["change"] = scene_change(self._gprev.thumb(), now["thumb"])
        diff = judge_room(self._gprev, now)
        if diff:
            self.push("arming-diff", "GUARD ARMING WARNING: tonight's empty room is not last night's: " + "; ".join(diff)
                                     + ". Somebody may still be in the room, or the light was left on.", quiet=0.0)

    # ---- guard mode
    def room_event(self, rooms: list[str]) -> Optional[list[str]]:
        """Something in the ROOM differs from the reference for CONFIRM of the last WINDOW samples. A change in the room alone proves nothing (a
        covered lens or a muted microphone looks like one), so the PROGRAM tests the instrument first, with the check tone. If the test finds a
        fault, the facts of the fault are returned and the usual loop (agent, program check) starts. If the instrument is in order, the change is
        real: ROOM ALERT, a picture as evidence, a message to the person. No model is used."""
        test = sample(self.call, self.names, self.tone if (self.tone_every and not self._quiet()) else None, self.region)
        self._observe(test)
        inst = [t for _, t in judge_cats(self.base, test, instrument_only=True)]
        if inst:
            self.emit(dict(type="note", text="the room seems to have changed (" + "; ".join(rooms) + "), but the instrument is not in order itself: "
                                             "the instrument is looked at first"))
            return inst
        proofs = []
        if "sharpness" in test:
            proofs.append(f"picture sharp ({test['sharpness']:.0f})")
        if test.get("noise_db") is not None:
            proofs.append("microphone alive")
        if test.get("tone_coupling") is not None:
            proofs.append(f"check tone arrives at {test['tone_coupling']:.0f} dB")
        evidence = []
        if "capture" in self.names:
            r = _call(self.call, "capture", dict(label="room_alert", n_frames=1))
            if r.get("ok"):
                evidence = list(r.get("saved_crops") or [])
        text = ("ROOM ALERT: " + "; ".join(rooms) + ". The instrument passed its own test just now (" + (", ".join(proofs) or "it answers")
                + "), so this is a change of the room, not a fault of the camera or the microphone. A person must look.")
        done = []
        if self.notify is not None:
            try:
                done = self.notify(text)
            except Exception as e:  # noqa: BLE001
                done = [f"notification FAILED: {type(e).__name__}"]
        self.emit(dict(type="room_alert", facts=rooms, text=text, evidence=evidence, notified=done))
        self._room_quiet_until = time.time() + ROOM_QUIET_S
        self.hist.clear()
        return None

    def accepts_new_normal(self, left: list[str], stop: threading.Event) -> bool:
        """After an ESCALATION: may this state become the new normal? Only a PERSON can say yes (the program cannot tell 'fixed' from 'we
        gave up'). With nobody to ask (no `ask`) or no reference yet, the old behaviour stays: the state is measured as the new normal."""
        if self.ask is None or self.ref is None:
            return True
        q = ("The instrument is still not back to its reference state: " + "; ".join(left) + f". The reference was: {normal_text(self.ref)}. "
             "If you cannot fix it and this state is fine for you now, reply \"accept\": I take it as the new normal. "
             "Reply \"keep\" (or anything else): I keep watching against the reference and call the agent again later.")
        r = self._ask(q)
        if r is None or stop.is_set():
            return False
        yes = r.strip().lower().startswith(("accept", "yes"))
        self.emit(dict(type="note", text="the person accepted this state as the new normal: the reference is measured again" if yes else
                                         "the person did not accept it: the reference state stays, the watch goes on against it"))
        if yes and self.ref_path is not None:
            self._persist_next = True                              # the new reference will be written over the old file (only a person's yes does that)
        return yes

    def keep_reference(self) -> None:
        """Not accepted: the normal values go back to the reference state, so the unfixed problem keeps counting as a problem. For
        ESC_QUIET_S seconds no new alarm starts (the person has just been told), then the watch can call the agent again."""
        if self.ref is None:
            return
        self.base = self.ref.snapshot()
        self.hist.clear()
        self._quiet_until = time.time() + ESC_QUIET_S

    def verify(self, old: Normal) -> list[str]:
        """After an investigation the PROGRAM measures once more (no model) and says whether the instrument is back inside the limits that
        the alert used: the agent's words can be wrong, this check applies the same rules as the alarm. Returns what is still outside the
        limits (an empty list = back inside), so that the loop can act on it."""
        s = sample(self.call, self.names, self.tone if (self.tone_every and not self._quiet()) else None, self.region)   # the check tone too: a loudspeaker that is still too quiet is not 'back'
        self._observe(s)
        self._last_check = {k: s[k] for k in ("brightness", "noise_db", "sharpness") if k in s}
        facts = [t for _, t in judge_cats(old, s)]
        vals = []
        if "brightness" in s:
            vals.append(f"scene brightness {s['brightness']:.0f}")
        if "sharpness" in s:
            vals.append(f"sharpness_norm {s['sharpness']:.0f}")
        if "noise_db" in s:
            vals.append(f"room noise {s['noise_db']:.1f} dBFS")
        limits = accepted_text(old, None)
        head = "check after the investigation (made by the program, no model): " + (", ".join(vals) or "no reading") + " - "
        if facts:
            self.emit(dict(type="note", text=head + "NOT back inside the limits: " + "; ".join(facts)))
        else:
            self.emit(dict(type="note", text=head + "back inside the limits of the alert" + (f" ({limits})" if limits else "")))
        return facts

    def run(self, stop: threading.Event) -> None:
        self._stop = stop
        self.emit(dict(type="note", text=f"watching: a check every {self.interval:g} s; the model is called only when the numbers change"
                                         + ("; GUARD HOURS %02d:%02d-%02d:%02d: the room is watched against the reference of the night"
                                            % (self.guard_hours[0] // 60, self.guard_hours[0] % 60, self.guard_hours[1] // 60, self.guard_hours[1] % 60)
                                            if self.guard_hours else "; GUARD MODE: the room is watched against the reference too" if self.guard else "")))
        self.load_stored()
        self.calibrate_tone()
        self.secure_tone(stop)
        while not stop.is_set():
            t0 = time.time()
            facts = self.step()
            if facts:
                req = alert_request(facts, self.base, self.tone)
                self.emit(dict(type="anomaly", facts=facts, request=req))
                if any(f.startswith(("camera:", "microphone:", "tone:")) for f in facts):
                    self._trouble_told = True                # nobody may be looking at the page: a lost instrument is pushed at once
                    self.push("lost", "WATCH TROUBLE: the instrument does not answer: " + "; ".join(facts)[:300]
                              + ". The room may NOT be watched. A person must check the cable and the power.")
                self.hist.clear()
                old = self.base.snapshot()
                left, tries, prev_check = facts, 0, None
                self.state = "investigating"
                try:
                    while True:
                        try:
                            self.investigate(req, left)
                        finally:
                            self.base.clear()               # the instrument may be different now: the normal values are measured again
                            self._last_tone = 0.0
                            self.hist.clear()
                            self.emit(dict(type="note", text=f"the investigation is over: waiting {self.settle:g} s for the camera and the room "
                                                             "to settle, then checking the instrument once more"))
                            stop.wait(self.settle)
                        if stop.is_set():
                            break
                        left = self.verify(old)             # the program's own check: the loop closes on this measurement, not on the agent's words
                        if not left:
                            break
                        steady = steady_environment(prev_check, self._last_check, left)
                        prev_check = dict(self._last_check)
                        if tries >= MAX_RETRIES or steady:  # the loop could not close by itself: say it loudly, with the numbers
                            why = ("The difference is of the room (the light or the noise) and two checks in a row gave the same numbers: "
                                   "more investigations cannot change it, so the loop does not repeat them. " if steady and tries < MAX_RETRIES else "")
                            self.emit(dict(type="note", text="ESCALATION: still NOT back inside the limits after "
                                                             f"{tries + 1} investigation(s): " + "; ".join(left) + ". " + why + "The loop could not close by "
                                                             "itself: a person has to look at the instrument or the room. "
                                                             + ("The person is asked now whether this state may become the new normal; until "
                                                                "the person accepts it, the watch goes on against the reference state."
                                                                if self.ask is not None and self.ref is not None else
                                                                "There is nobody to ask, so the watch goes on and measures new normal values: "
                                                                "this state is now the normal one, it is written here, not hidden.")))
                            self.push("escalation", "WATCH TROUBLE: ESCALATION: still NOT back inside the limits after "
                                                    f"{tries + 1} investigation(s): " + "; ".join(left)[:300] + ". A person has to look.")
                            if self.accepts_new_normal(left, stop):
                                self.ref = None                 # the person took this state as normal: the reference is measured again
                            else:
                                self.keep_reference()           # not accepted: the old reference stays, the problem is not forgotten
                            break
                        tries += 1
                        self.emit(dict(type="note", text=f"the program's check says NOT back inside the limits: the agent is called again "
                                                         f"(investigation {tries + 1} of {MAX_RETRIES + 1}) with the numbers of that check"))
                        req = retry_request(left, old, self.tone, tries + 1)
                        self.hist.clear()
                        self.emit(dict(type="anomaly", facts=left, request=req))
                finally:
                    self.state = "sampling"
                    if not stop.is_set():
                        self.calibrate_tone()               # the loudspeaker may be quieter or louder now: the check tone is tuned again
                        self.secure_tone(stop)              # and if it is not heard at all, the person is asked (once per episode)
                continue
            stop.wait(max(0.0, self.interval - (time.time() - t0)))
        self.emit(dict(type="note", text="watching stopped"))
