"""The watchdog: a cheap loop WITHOUT a language model that watches the instrument and calls the agent when something changes.

Every few seconds the program measures the scene (camera) and the room noise (microphone), and compares the numbers with the
normal values of the last minute. Nothing is sent to a model while everything is normal. When the numbers leave the normal range in
CONFIRM of the last WINDOW samples, the watchdog builds a request from the facts and hands it to the graph of Homework 3 (orchestrator,
hardware agent, human pause): THE MODEL decides what to check, in which order, and what the person must fix. After the
investigation the normal values are measured again and the watch goes on.

    sample -> compare with the normal values -> (normal: wait) / (changed twice: ALERT -> agent investigates -> new normal)

The numbers come from the instrument tools (measure_region, listen_noise, play_tone), exactly as the agent gets them.
"""
from __future__ import annotations

import json
import statistics
import threading
import time
from collections import deque
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


def _call(call: Callable, name: str, args: dict) -> dict:
    try:
        r = json.loads(call(name, args))
        return r if isinstance(r, dict) else dict(ok=False, error="unexpected answer")
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, error=f"{type(e).__name__}: {str(e)[:120]}")


def sample(call: Callable, names, tone: Optional[dict] = None) -> dict:
    """One set of readings. `tone` (level_db) also plays the check tone (a short beep)."""
    s: dict = dict(t=round(time.time(), 2), errors=[])
    if "measure_region" in names:
        r = _call(call, "measure_region", dict(region="scene", n_frames=3))
        if r.get("ok"):
            s.update(brightness=float(r["mean_brightness"]), sharpness=float(r["sharpness_norm"]),
                     saturated=float(r["saturated_fraction"]))
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


def judge_cats(base: Normal, s: dict) -> list[tuple]:
    """What is different from normal in this sample: (category, fact with numbers) pairs. Categories: err, scene, noise, tone."""
    out = [("err", e) for e in (s.get("errors") or [])]
    b0, b = base.med("brightness"), s.get("brightness")
    if b0 is not None and b is not None and abs(b - b0) > max(BRIGHT_ABS, BRIGHT_REL * b0):
        out.append(("scene", f"scene brightness {b0:.0f} -> {b:.0f} grey levels"))
    h0, h = base.med("sharpness"), s.get("sharpness")
    if h0 is not None and h is not None:
        ratio = (h + 1.0) / (h0 + 1.0)
        if ratio < 1.0 / SHARP_RATIO:
            out.append(("scene", f"scene sharpness {h0:.1f} -> {h:.1f}"))
    n0, n = base.med("noise_db"), s.get("noise_db")
    if s.get("silent"):
        out.append(("noise", "the microphone delivers pure silence (muted?)"))
    elif n0 is not None and n is not None and n - n0 > NOISE_RISE_DB:
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


def hints(facts: list[str]) -> str:
    """What the program knows about this instrument from the real runs, as hypotheses to test (the agent still chooses its checks)."""
    t = " ".join(facts)
    silence = "silence" in t
    h = []
    if "sharpness" in t and "brightness" not in t:
        h.append("observed on this camera: when the lens is covered by a hand the brightness stays near normal (the automatic exposure "
                 "brightens the dark picture) and sharpness_norm collapses to a small part of normal; so a sharpness collapse with normal "
                 "brightness points to a covered, dirty or blocked lens. Only a person can uncover it: ask for exactly that. Right after the "
                 "lens is uncovered the camera needs about 10 s to readjust and the first readings can still be low; the program waits before "
                 "your next turn. After uncovering, a reading that is inside the limits above means the instrument works: say so and finish. "
                 "Ask to wipe the lens ONLY if two readings a few seconds apart are still BELOW the sharpness limit above")
    if "brightness" in t:
        h.append("a brightness change means the light changed or the lens is covered; you cannot change the light: report what you "
                 "measured (for a covered lens ask the person to uncover it)")
    if ("sharpness" in t or "brightness" in t) and not any(k in t for k in ("room noise", "check tone", "silence", "camera:", "microphone:", "tone:")):
        h.append("the alarm is about the camera, the sound side is not involved: do not play tones; at most one listen_noise as a cross-check")
    if "room noise" in t:
        h.append("room noise rose. Do NOT ask the person to be quiet or to turn anything down. Act yourself: play the check tone at the "
                 "usual level; if heard is true and coupling_db is within about 6 dB of the normal value (given above, if there is one), report "
                 "'the room is louder by X dB, the instrument still works, nothing to do' and finish; if it is not, raise level_db yourself "
                 "(-20, then -12, the maximum) until heard is true and report the level you needed")
    if "check tone" in t and not silence:
        h.append("the check tone is weaker or lost (the volume of the loudspeaker was lowered, or the room is louder). COMPENSATE YOURSELF: "
                 "play the tone again at the usual level and read coupling_db in the answer (the tone's level at the microphone minus the level "
                 "played; the normal value is given above): the difference to the normal value is the loss. Raise level_db by about that loss "
                 "(-12 dBFS is the maximum) and play again until coupling_db is back near the normal value; report the loss in dB and that the "
                 "instrument works again. Do not look for a margin above the noise of 15 dB: in a reflecting room it stops growing with the "
                 "level. Ask the person to press the volume button only if even -12 dBFS does not bring the tone back")
    if silence:
        h.append("the microphone delivers pure silence: it is muted, and a louder tone cannot help; only a person can unmute it: ask for exactly that")
    if "camera:" in t or "microphone:" in t or "tone:" in t:
        h.append("the instrument itself stopped answering: the cable or the power; only a person can check them")
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


class Watchdog:
    """The loop. `call(name, args)` sends a command to the instrument; `emit(event)` shows what happens; `investigate(request,
    facts)` runs the agent on the request and returns when it has finished."""

    def __init__(self, call: Callable, names, emit: Callable[[dict], None], investigate: Callable[[str, list], None],
                 interval: float = 4.0, tone_every: float = 0.0):
        self.call, self.names, self.emit, self.investigate = call, set(names), emit, investigate
        self.interval, self.tone_every = float(interval), float(tone_every)
        self.base = Normal()
        self.tone: Optional[dict] = None   # None = no level found yet: the tone check is off and is tried again every RECAL_S seconds
        self._cal_t = 0.0
        self._said_off = False
        self.hist: dict = {}               # per kind of reading: did the last WINDOW samples disagree with the normal values?
        self.settle = 3.0 * self.interval  # after an investigation: wait for the camera (autofocus, exposure) and the room to settle
        self._last_tone = 0.0

    def calibrate_tone(self) -> None:
        """Find a level at which the check tone is heard (at once if it is TONE_NEEDED_DB above the noise, else the best level that is heard
        at all: in a reflecting room the margin stops growing with the level). The normal value of the check is the tone's level at the
        microphone, tone_db - level_db (coupling): it falls when the loudspeaker gets quieter, whatever the noise does. If no level is heard,
        the tone check is off for now (it says so, with the numbers) and is tried again every RECAL_S seconds."""
        if not self.tone_every or "play_tone" not in self.names:
            return
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

    def step(self) -> Optional[list[str]]:
        """One sample. Returns the facts if the alarm goes off, else None. A kind of reading (scene, noise, check tone) must disagree
        with the normal values in CONFIRM of the last WINDOW samples that contain it; a sample without the check tone says nothing about the tone."""
        if self.tone is None and self.tone_every and time.time() - self._cal_t >= RECAL_S:
            self.calibrate_tone()
        use_tone = bool(self.tone and self.tone_every and time.time() - self._last_tone >= self.tone_every)
        s = sample(self.call, self.names, self.tone if use_tone else None)
        if use_tone:
            self._last_tone = time.time()
        cats = judge_cats(self.base, s) if self.base.ready() else []
        by_cat: dict = {}
        for c, t in cats:
            by_cat.setdefault(c, []).append(t)
        for c in presence(s):
            self.hist.setdefault(c, deque(maxlen=WINDOW)).append(c in by_cat)
        facts = [t for _, t in cats]
        broken = bool(s["errors"]) and not self.base.ready()      # no normal values yet and the instrument does not answer
        status = ("waiting for the instrument" if broken else "warming up" if not self.base.ready()
                  else ("changed" if facts else "normal"))
        self.emit(dict(type="sample", status=status, facts=facts, **{k: v for k, v in s.items() if k not in ("errors", "t")},
                       errors=s["errors"], t=s["t"], normal={k: self.base.med(k) for k in ("brightness", "sharpness", "noise_db", "tone_snr", "tone_coupling")}))
        if broken:
            return None                     # an error is not a normal value, and there is nothing to compare with: no alarm, no model call
        if not facts:
            self.base.add(clean(s))
            return None
        if any(sum(self.hist.get(c, ())) >= CONFIRM for c in by_cat):
            return facts
        return None

    def verify(self, old: Normal) -> None:
        """After an investigation the PROGRAM measures once more (no model) and says whether the instrument is back inside the limits that
        the alert used: the agent's words can be wrong, this check applies the same rules as the alarm."""
        s = sample(self.call, self.names)
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

    def run(self, stop: threading.Event) -> None:
        self.emit(dict(type="note", text=f"watching: a check every {self.interval:g} s; the model is called only when the numbers change"))
        self.calibrate_tone()
        while not stop.is_set():
            t0 = time.time()
            facts = self.step()
            if facts:
                req = alert_request(facts, self.base, self.tone)
                self.emit(dict(type="anomaly", facts=facts, request=req))
                self.hist.clear()
                old = self.base.snapshot()
                try:
                    self.investigate(req, facts)
                finally:
                    self.base.clear()                       # the instrument may be different now: the normal values are measured again
                    self._last_tone = 0.0
                    self.hist.clear()
                    self.emit(dict(type="note", text=f"the investigation is over: waiting {self.settle:g} s for the camera and the room "
                                                     "to settle, then checking the instrument once more"))
                    stop.wait(self.settle)
                    if not stop.is_set():
                        self.verify(old)
                        self.calibrate_tone()               # the loudspeaker may be quieter or louder now: the check tone is tuned again
                continue
            stop.wait(max(0.0, self.interval - (time.time() - t0)))
        self.emit(dict(type="note", text="watching stopped"))
