"""(a, second part) THE SOUND CONNECTOR - a loudspeaker that is commanded and a microphone that measures what it did.

Instrument : the Pano-360 (case label THAT DRAGONFLY) is not only a camera, it is a USB conference device: it has a loudspeaker
             (the white mesh band) and microphones. On the computer they appear as one sound OUTPUT and one sound INPUT, both named
             "Pano-360" (python audio_check.py --list shows them).
Interface  : the sound driver of the operating system (Core Audio on macOS, WASAPI on Windows, ALSA on Linux) through the python
             package `sounddevice` (PortAudio).
Command    : play a pure tone through the loudspeaker: frequency (Hz), digital level (dBFS, 0 = full scale), duration.
Reading    : the microphone records while the tone plays - not as one long tone but "chopped": the recording is cut into segments
             of 0.2 s and the tone is ON in half of them and OFF in the other half, in a fixed pseudo-random order (4 to 7 pairs). The
             program finds the delay between playing and recording from the recording itself and measures the loudness at the
             tone's frequency in every segment. Then
               noise_db  the level of the segments WITHOUT the tone (what the room makes at that frequency: a fan, voices far away,
                         silence ...), measured in the same recording, a few tenths of a second from the tone,
               tone_db   the level that the segments WITH the tone have in addition (power difference of the medians),
               snr_db    tone_db - noise_db, and heard = (snr_db >= 10 dB, the tone segments are louder than the others in at
                         least 75 % of the pairs, and the loudness follows the played on/off order better than 80 other orders do:
                         the last test is the answer to a room that happens to change in step with the tone by chance).
             Whatever is there in both kinds of segment is taken away by the subtraction, and a sound that does not follow the
             on/off order cannot pass as the tone: the measurement works in a noisy room as well as in a quiet one. If the tone
             seems present but the segments are not cleanly apart, the reading has "no verdict" (interference) and is repeated.
             Which device is the speaker and which the microphone is chosen, in this order, by: the environment variables HW3_AUDIO_OUT and
             HW3_AUDIO_IN (a part of a name, or a number), then the file audio.json (written by python audio_check.py --pairs;
             HW3_AUDIO_FILE names another file), then the name "Pano". Some speakerphones cancel their own sound out of their
             microphone signal (echo cancellation); then the microphone of the computer is the sensor of the Pano-360 loudspeaker,
             and the Pano-360 microphone can listen to the loudspeaker of the computer (audio_check.py --pairs finds out).

Privacy    : a microphone hears the room. The recording stays in memory for about three seconds and only the numbers above are kept;
             no sound is ever written to disk, nothing is transcribed, nothing is sent anywhere.
Safety     : the level is limited to -12 dBFS (HW3_AUDIO_MAX_DB can lower or raise it between -40 and -3) and a tone lasts at most 3 s.

A backend "fake" (a synthetic speaker-to-microphone path with a room noise, a resonance and a mute switch, held in the JSON file
HW3_FAKE_AUDIO) is used by the offline tests: HW3_AUDIO_BACKEND=fake. It is never used silently.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import random
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

MIN_DB = -80.0            # quietest level that may be commanded (dBFS)
MAX_DB_DEFAULT = -12.0    # loudest level that may be commanded (dBFS); a louder request is cut back to this
MIN_FREQ = 100.0
MAX_FREQ = 12000.0
MIN_SECONDS, DEFAULT_SECONDS, MAX_SECONDS = 1.6, 1.6, 3.0     # the length of the on/off pattern (tone on for half of it)
SEG_S = 0.2               # the pattern is made of segments of this length; in each one the tone is either ON or OFF
LEAD_S = 0.5              # silence before the pattern (the streams start, the loudspeaker wakes up)
TRAIL_S = 0.5             # silence after the pattern
TAIL_S = 0.5              # extra time recorded after the played buffer (the sound arrives late)
FADE_S = 0.01             # the tone is switched on and off with ramps of this length (no clicks)
GUARD_BEFORE_S, GUARD_AFTER_S = 0.05, 0.03    # of each segment, the start and the end are not measured (ramps, room echo)
LAG_MIN_S, LAG_MAX_S = -0.1, 0.9              # the delay between playing and recording (found in each recording) is searched here
MARGIN_DB = 10.0          # the tone is "heard" when it is this far above the noise at its frequency
U_MIN = 0.75              # ... and when the segments with the tone are louder than the segments without it in at least this share of pairs
RUMBLE_HZ = 80.0          # the strongest component that is reported (dominant_hz) is looked for above this (fans, mains hum; a tone is >= 100 Hz)
FLOOR_DB = -120.0         # a level below this reads as this (digital silence)
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
VIRTUAL = ("teams", "zoom", "aggregate", "blackhole", "loopback", "soundflower", "multi-output", "virtual")


class AudioError(RuntimeError):
    pass


def pair_file() -> Path:
    return Path(os.environ.get("HW3_AUDIO_FILE") or "audio.json")


def saved_pair() -> dict:
    """The loudspeaker / microphone chosen by audio_check.py --pairs ({"out": name, "in": name}), {} if there is none."""
    try:
        d = json.loads(pair_file().read_text(encoding="utf-8"))
        return {k: str(d[k]) for k in ("out", "in") if d.get(k)}
    except Exception:  # noqa: BLE001
        return {}


def max_db() -> float:
    try:
        return min(max(float(os.environ.get("HW3_AUDIO_MAX_DB", MAX_DB_DEFAULT)), -40.0), -3.0)
    except ValueError:
        return MAX_DB_DEFAULT


# ----------------------------------------------------------------------------- analysis (pure numpy, tested without any device)
def db(amplitude: float) -> float:
    """Level of an amplitude in dB relative to full scale (a sine of amplitude 1.0 reads 0 dB); silence reads FLOOR_DB."""
    return max(20.0 * math.log10(max(float(amplitude), 1e-12)), FLOOR_DB)


def lockin(x: np.ndarray, sr: float, f: float) -> float:
    """Amplitude (peak units: a sine of amplitude A at f gives A) of the component of x at frequency f (Hann window)."""
    n = len(x)
    if n < 16:
        return 0.0
    t = np.arange(n, dtype=np.float64) / sr
    w = np.hanning(n)
    c = np.dot(w * np.asarray(x, np.float64), np.exp(-2j * math.pi * f * t))
    return float(2.0 * abs(c) / w.sum())


def spectrum_peak(x: np.ndarray, sr: float, lo: float = 50.0) -> tuple[float, float]:
    """(frequency, amplitude) of the strongest component of x between lo and 0.47 * sr."""
    n = len(x)
    w = np.hanning(n)
    nfft = 1 << int(math.ceil(math.log2(max(n, 16) * 4)))
    spec = np.abs(np.fft.rfft(np.asarray(x, np.float64) * w, nfft)) * 2.0 / w.sum()
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    m = (freqs >= lo) & (freqs <= 0.47 * sr)
    k = int(np.argmax(spec[m]))
    return float(freqs[m][k]), float(spec[m][k])


def chop_pattern(n_pairs: int) -> list:
    """The order of the segments of a measurement: 1 = the tone is on, 0 = it is off; n_pairs of each. The order looks random but is
    the same every time (so two runs can be compared); never more than two equal segments in a row and never a plain alternation,
    so nothing periodic in the room (a fan, a ticking clock) can follow it by chance."""
    rng = random.Random(1000 + n_pairs)
    seq = [1] * n_pairs + [0] * n_pairs
    for _ in range(10000):
        rng.shuffle(seq)
        runs_ok = all(not (seq[i] == seq[i + 1] == seq[i + 2]) for i in range(len(seq) - 2))
        alternating = all(seq[i] != seq[i + 1] for i in range(len(seq) - 1))
        if runs_ok and not alternating:
            return list(seq)
    return [1, 0] * n_pairs                    # (unreachable for n_pairs >= 3)


def n_pairs_for(seconds: float) -> int:
    return int(min(7, max(4, round(seconds / (2 * SEG_S)))))


def build_play(freq: float, level_db: float, n_pairs: int, sr: int) -> tuple:
    """The buffer for the loudspeaker: LEAD_S of silence, then the segments (the tone on or off in the order of chop_pattern), then
    TRAIL_S of silence. Returns (float32 samples, pattern)."""
    pattern = chop_pattern(n_pairs)
    seg = int(SEG_S * sr)
    gate = np.concatenate([np.zeros(int(LEAD_S * sr))] + [np.full(seg, float(p)) for p in pattern] + [np.zeros(int(TRAIL_S * sr))])
    nf = max(2, int(FADE_S * sr))
    ker = np.hanning(2 * nf + 1)
    gate = np.convolve(gate, ker / ker.sum(), mode="same")                  # raised-cosine edges
    t = np.arange(len(gate)) / sr
    return ((10 ** (level_db / 20.0)) * np.sin(2 * math.pi * freq * t) * gate).astype(np.float32), pattern


def _amplitude_track(rec: np.ndarray, sr: float, f: float, win_s: float = 0.06, hop_s: float = 0.01) -> tuple:
    """(centre times, amplitude at f) of short windows (Hann, 60 ms) every 10 ms: where in the recording the frequency f is loud."""
    win, hop = max(16, int(win_s * sr)), max(1, int(hop_s * sr))
    if len(rec) < win + hop:
        return np.zeros(0), np.zeros(0)
    t = np.arange(len(rec)) / sr
    z = np.asarray(rec, np.float64) * np.exp(-2j * math.pi * f * t)
    w = np.hanning(win)
    starts = np.arange(0, len(rec) - win, hop)
    view = np.lib.stride_tricks.sliding_window_view(z, win)[starts]
    return (starts + win / 2.0) / sr, 2.0 * np.abs(view @ w) / w.sum()


def _like_a_shift(true_pattern: list, q: list) -> bool:
    """Is q the played pattern moved by a few segments (or nearly)? Such a pattern fits the same sound just as well at another delay,
    so it says nothing about chance and is not used as a comparison."""
    t, u, n = np.asarray(true_pattern, float), np.asarray(q, float), len(true_pattern)
    for d in range(-3, 4):
        x, y = t[max(0, -d):n - max(0, d)], u[max(0, d):n - max(0, -d)]
        if len(x) >= 4 and x.std() > 0 and y.std() > 0 and float(np.corrcoef(x, y)[0, 1]) >= 0.75:
            return True
    return False


@lru_cache(maxsize=32)
def _surrogates_for(true_pattern: tuple, k: int) -> tuple:
    n = len(true_pattern)
    cand = [[1 if i in on else 0 for i in range(n)] for on in itertools.combinations(range(n), n // 2)]    # every balanced pattern
    random.Random(777).shuffle(cand)                                                                       # fixed order, the same every time
    out = []
    for q in cand:
        if tuple(q) != true_pattern and not _like_a_shift(list(true_pattern), q):
            out.append(q)
            if len(out) >= k:
                break
    return tuple(tuple(q) for q in out)


def _surrogate_patterns(true_pattern: list, k: int = 80) -> list:
    """Up to k other balanced on/off patterns of the same length (not the played one, not a shifted copy of it): what the recording
    would be compared with if the room, not the loudspeaker, were making the changes."""
    return [list(q) for q in _surrogates_for(tuple(true_pattern), k)]


def _scores(times: np.ndarray, amp: np.ndarray, pattern: list, lags: np.ndarray) -> np.ndarray:
    """Correlation (-1..1) between the loudness at f over time and the on/off pattern shifted by each lag."""
    idx = np.clip(((times[:, None] - lags[None, :] - LEAD_S) // SEG_S).astype(int), -1, len(pattern))
    pat = np.concatenate([[0.0], np.asarray(pattern, float), [0.0]])
    ref = pat[idx + 1]                                                      # (time, lag): 1 where the tone should be on at that lag
    a = amp - amp.mean()
    r = ref - ref.mean(axis=0, keepdims=True)
    return (a[:, None] * r).sum(axis=0) / (np.sqrt((a ** 2).sum() * (r ** 2).sum(axis=0)) + 1e-30)


def estimate_lag(rec: np.ndarray, sr: float, f: float, pattern: list) -> tuple:
    """(delay in s, match, chance). The delay between playing and recording is the lag at which the loudness at f follows the on/off
    pattern best; `match` is the correlation there. `chance` is the best correlation (over the same lags) that up to 80 OTHER patterns reach
    on the same recording: whatever the room does has no reason to follow the played pattern more closely than another one, so a
    match above the chance means the loudspeaker made it."""
    times, amp = _amplitude_track(rec, sr, f)
    max_lag = len(rec) / sr - (LEAD_S + len(pattern) * SEG_S)
    lags = np.arange(LAG_MIN_S, min(LAG_MAX_S, max_lag) + 1e-9, 0.01)
    if len(times) == 0 or len(lags) == 0:
        raise AudioError("the recording is too short to measure")
    amp = np.argsort(np.argsort(amp)).astype(float)          # ranks: one huge click or slam cannot decide where the pattern lies
    sc = _scores(times, amp, pattern, lags)
    k = int(np.argmax(sc))
    chance = max(float(np.max(_scores(times, amp, q, lags))) for q in _surrogate_patterns(pattern))
    return float(lags[k]), float(sc[k]), float(chance)


def analyse(rec: np.ndarray, sr: float, freq: float, pattern: list) -> dict:
    """Measure a recording made while a tone of `freq` was played in the on/off pattern `pattern` (see build_play).
    The loudness at `freq` is measured in every segment. tone_db is the level that the segments WITH the tone have more than the
    segments WITHOUT it, noise_db the level of the segments without it, both found in the same recording, a few tenths of a second
    apart. Whatever the room does at that frequency in both kinds of segment (a fan, a hum, a voice far away, silence) is in the
    noise and is taken away; only what follows the pattern is the tone. Medians are used, so a loud burst in one or two segments
    does not count."""
    rec = np.asarray(rec, np.float64)
    peak = float(np.max(np.abs(rec))) if len(rec) else 0.0
    silent = peak < 1e-7
    lag, match, chance = estimate_lag(rec, sr, freq, pattern)
    p_on, p_off, on_windows = [], [], []
    for i, on in enumerate(pattern):
        s0 = LEAD_S + i * SEG_S + lag + GUARD_BEFORE_S
        s1 = LEAD_S + (i + 1) * SEG_S + lag - GUARD_AFTER_S
        x = rec[max(0, int(s0 * sr)):max(0, int(s1 * sr))]
        if len(x) < 16:
            raise AudioError("the recording is too short to measure")
        (p_on if on else p_off).append(lockin(x, sr, freq) ** 2)
        if on:
            on_windows.append(x)
    p_on, p_off = np.asarray(p_on), np.asarray(p_off)
    med_on, med_off = float(np.median(p_on)), float(np.median(p_off))
    tone_db = db(math.sqrt(max(med_on - med_off, 0.0)))
    noise_db = db(math.sqrt(med_off / math.log(2.0)))                      # the median of a squared noise amplitude is ln 2 of its mean
    snr = tone_db - noise_db
    sep = float(np.mean(p_on[:, None] > p_off[None, :]))                    # share of (on, off) pairs in which the "on" segment is louder
    matched = match > chance                                               # the pattern is followed better than any other pattern is
    clean = bool(sep >= U_MIN and matched)
    heard = bool((not silent) and snr >= MARGIN_DB and clean)
    interference = bool((not silent) and snr >= MARGIN_DB and not clean)   # the tone seemed to be there but the segments are not cleanly apart
    peaks = [spectrum_peak(x, sr, RUMBLE_HZ) for x in on_windows]
    dom_hz, dom_amp = max(peaks, key=lambda q: q[1])
    amps = np.sqrt(p_on)
    steady = bool(amps.min() >= 0.5 * amps.max()) if heard else None
    return dict(tone_db=round(tone_db, 1), noise_db=round(noise_db, 1), snr_db=round(snr, 1), heard=heard,
                separation=round(sep, 2), match=round(match, 2), chance=round(chance, 2), delay_s=round(lag, 2), dominant_hz=round(dom_hz, 1), dominant_db=round(db(dom_amp), 1),
                interference=interference, steady=steady, clipped=bool(peak >= 0.98), silent_input=bool(silent))


# ----------------------------------------------------------------------------- backends
def device_table(devs) -> str:
    return "\n".join(f"  {i} {d['name']} ({d['max_input_channels']} in, {d['max_output_channels']} out)" for i, d in enumerate(devs))


def pick(devs, selector, direction: str) -> int:
    """The index of a sound device: `selector` is an index (digits) or a part of the name; only devices that have channels in
    that direction are considered."""
    key = "max_output_channels" if direction == "output" else "max_input_channels"
    sel = str(selector).strip()
    if sel.isdigit():
        i = int(sel)
        if 0 <= i < len(devs) and devs[i][key] > 0:
            return i
        raise AudioError(f"sound device {i} has no {direction} channels; the devices are:\n{device_table(devs)}")
    for i, d in enumerate(devs):
        if d[key] > 0 and sel.lower() in d["name"].lower():
            return i
    raise AudioError(f"no sound {direction} device whose name contains {sel!r}; the devices are:\n{device_table(devs)}\n"
                     "(is the Pano-360 plugged in and switched on? choose another with HW3_AUDIO_NAME / HW3_AUDIO_OUT / HW3_AUDIO_IN)")


class SoundBackend:
    """The real loudspeaker and microphone, through sounddevice."""
    kind = "sounddevice"

    def __init__(self, out_sel: Optional[str] = None, in_sel: Optional[str] = None):
        import sounddevice as sd
        self.sd = sd
        devs = sd.query_devices()
        hint = os.environ.get("HW3_AUDIO_NAME") or "Pano"
        saved = saved_pair()
        self.source: dict = {}
        self.out_idx = self._choose(devs, "output", "out", out_sel, os.environ.get("HW3_AUDIO_OUT"), saved.get("out"), hint)
        self.in_idx = self._choose(devs, "input", "in", in_sel, os.environ.get("HW3_AUDIO_IN"), saved.get("in"), hint)
        o, i = devs[self.out_idx], devs[self.in_idx]
        self.out_name, self.in_name = o["name"], i["name"]
        self.sr_out, self.sr_in = int(o["default_samplerate"]), int(i["default_samplerate"])
        self.ch_out = max(1, min(2, int(o["max_output_channels"])))

    def _choose(self, devs, direction: str, key: str, argument, env, saved, hint) -> int:
        """An explicit choice (argument, environment variable) is obeyed or refused, never silently replaced; a saved choice that no
        longer fits (another computer, the device unplugged) falls back to the name hint."""
        for sel, src in ((argument, "argument"), (env, "environment")):
            if sel:
                self.source[key] = src
                return pick(devs, sel, direction)
        if saved:
            try:
                idx = pick(devs, saved, direction)
                self.source[key] = pair_file().name
                return idx
            except AudioError:
                pass
        self.source[key] = "default"
        return pick(devs, hint, direction)

    def capture(self, play: Optional[np.ndarray], rec_seconds: float) -> np.ndarray:
        """Record `rec_seconds` from the microphone; if `play` (mono float32 at sr_out) is given, play it through the loudspeaker
        while recording. The input stream is opened first, so that the start of the recording is silence."""
        sd = self.sd
        n_rec = int(rec_seconds * self.sr_in)
        rec = np.zeros(n_rec, np.float32)
        errors: list = []
        started = threading.Event()

        def recorder():
            try:
                with sd.InputStream(device=self.in_idx, channels=1, samplerate=self.sr_in, dtype="float32") as s:
                    started.set()
                    data, _ = s.read(n_rec)
                    rec[:] = data[:, 0]
            except Exception as e:  # noqa: BLE001
                errors.append(("microphone", e))
                started.set()

        th = threading.Thread(target=recorder, daemon=True)
        th.start()
        started.wait(10.0)
        if play is not None and not errors:
            try:
                out = np.repeat(np.asarray(play, np.float32)[:, None], self.ch_out, axis=1)
                with sd.OutputStream(device=self.out_idx, channels=self.ch_out, samplerate=self.sr_out, dtype="float32") as o:
                    o.write(out)
            except Exception as e:  # noqa: BLE001
                errors.append(("loudspeaker", e))
        th.join(rec_seconds + 8.0)
        if errors:
            what, e = errors[0]
            raise AudioError(f"the {what} could not be used: {type(e).__name__}: {e}. On a Mac the first use asks for permission "
                             "to use the microphone: allow it for the program that runs this (Terminal or Visual Studio Code) in "
                             "System Settings > Privacy & Security > Microphone, then run it again.")
        if th.is_alive():
            raise AudioError("the microphone did not finish recording in time")
        return rec


class FakeSoundBackend:
    """A synthetic stand-in for the tests: a speaker-to-microphone path with a room noise, a coupling loss, a resonance near
    2.5 kHz, a delay, a volume offset and a mute switch. Its settings are read from the JSON file HW3_FAKE_AUDIO before each
    recording: {"noise_db": -60, "coupling_db": -24, "volume_db": 0, "muted": false, "delay_s": 0.12}; noise_db is the RMS of the
    room noise in dBFS, coupling_db the loss between the played level and the level at the microphone at 1 kHz."""
    kind = "fake"
    out_name = "Pano-360 (synthetic loudspeaker)"
    in_name = "Pano-360 (synthetic microphone)"
    sr_out, sr_in, ch_out = 48000, 16000, 2
    source = {"out": "default", "in": "default"}

    def __init__(self, state_file: Optional[str] = None):
        self.state_file = state_file or os.environ.get("HW3_FAKE_AUDIO")
        self.n = 0

    def _state(self) -> dict:
        st = dict(noise_db=-60.0, coupling_db=-24.0, volume_db=0.0, muted=False, delay_s=0.12)
        if self.state_file and Path(self.state_file).exists():
            try:
                st.update(json.loads(Path(self.state_file).read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass
        return st

    @staticmethod
    def response_db(f: float) -> float:
        """A resonance of +9 dB at 2.5 kHz and a loss below 400 Hz (what a small speaker looks like)."""
        r = 9.0 * math.exp(-0.5 * (math.log2(f / 2500.0) / 0.9) ** 2)
        return r - (12.0 * math.log2(400.0 / f) if f < 400.0 else 0.0)

    def capture(self, play: Optional[np.ndarray], rec_seconds: float) -> np.ndarray:
        st = self._state()
        self.n += 1
        n_rec = int(rec_seconds * self.sr_in)
        if st["muted"]:
            return np.zeros(n_rec, np.float32)                       # a muted microphone delivers digital silence
        rng = np.random.default_rng(1000 + self.n)
        rec = rng.normal(0.0, 10 ** (float(st["noise_db"]) / 20.0), n_rec)
        if play is not None and float(np.max(np.abs(play))) > 0:
            seg = np.asarray(play, np.float64)[np.abs(play) > 1e-9]
            freq, _ = spectrum_peak(seg[:self.sr_out], self.sr_out)
            gain = 10 ** ((float(st["coupling_db"]) + float(st["volume_db"]) + self.response_db(freq)) / 20.0)
            t_in = np.arange(n_rec) / self.sr_in - float(st["delay_s"])
            rec = rec + gain * np.interp(t_in, np.arange(len(play)) / self.sr_out, np.asarray(play, np.float64), left=0.0, right=0.0)
        return np.clip(rec, -1.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------- the connector
class AudioConnector:
    def __init__(self, backend):
        self.b = backend
        self.readings: dict[str, dict] = {}       # numbers kept under a label (for compare_audio); never sound

    @property
    def max_freq(self) -> float:
        return float(round(min(0.4 * min(self.b.sr_in, self.b.sr_out), MAX_FREQ)))

    def info(self) -> dict:
        return dict(ok=True,
                    instrument="the loudspeaker and the microphone of the Pano-360 (THAT DRAGONFLY), a USB conference device",
                    loudspeaker=self.b.out_name, microphone=self.b.in_name,
                    sample_rate_out=self.b.sr_out, sample_rate_in=self.b.sr_in, backend=self.b.kind,
                    chosen_by=dict(loudspeaker=self.b.source["out"], microphone=self.b.source["in"]),
                    level_db_range=[MIN_DB, max_db()], freq_hz_range=[MIN_FREQ, self.max_freq],
                    seconds_range=[MIN_SECONDS, MAX_SECONDS], heard_margin_db=MARGIN_DB,
                    labels=sorted(self.readings),
                    note=("level_db is digital (dBFS, 0 = full scale); the loudness in the room also depends on the volume set on "
                          "the device and on the computer, which this tool cannot change. Nothing that is recorded is stored."))

    def _validated(self, freq_hz, level_db, seconds, label) -> tuple:
        try:
            f, lv, sec = float(freq_hz), float(level_db), float(seconds)
        except (TypeError, ValueError):
            raise ValueError("freq_hz, level_db and seconds must be numbers")
        if not all(math.isfinite(v) for v in (f, lv, sec)):
            raise ValueError("freq_hz, level_db and seconds must be finite numbers")
        if not MIN_FREQ <= f <= self.max_freq:
            raise ValueError(f"freq_hz must be between {MIN_FREQ:g} and {self.max_freq:g} Hz for this device (asked {f:g})")
        if label and not NAME_RE.match(str(label)):
            raise ValueError("label may contain only letters, digits, _ and - (at most 40 characters)")
        notes = []
        top = max_db()
        if lv > top:
            notes.append(f"the level {lv:g} dBFS was cut back to the limit {top:g} dBFS")
            lv = top
        if lv < MIN_DB:
            notes.append(f"the level {lv:g} dBFS was raised to {MIN_DB:g} dBFS")
            lv = MIN_DB
        if not MIN_SECONDS <= sec <= MAX_SECONDS:
            sec = min(max(sec, MIN_SECONDS), MAX_SECONDS)
            notes.append(f"seconds was set to {sec:g}")
        return f, lv, sec, notes

    def play_tone(self, freq_hz: float, level_db: float = -40.0, seconds: float = DEFAULT_SECONDS, label: str = "") -> dict:
        try:
            f, lv, sec, notes = self._validated(freq_hz, level_db, seconds, label)
        except ValueError as e:
            return dict(ok=False, error=str(e))
        n_pairs = n_pairs_for(sec)
        play, pattern = build_play(f, lv, n_pairs, self.b.sr_out)
        t0 = time.time()
        rec = self.b.capture(play, len(play) / self.b.sr_out + TAIL_S)
        res = analyse(rec, self.b.sr_in, f, pattern)
        res.update(ok=True, freq_hz=round(f, 1), level_db=round(lv, 1), seconds=round(2 * n_pairs * SEG_S, 2), tone_segments=n_pairs,
                   coupling_db=round(res["tone_db"] - lv, 1), took_s=round(time.time() - t0, 1),
                   loudspeaker=self.b.out_name, microphone=self.b.in_name)
        res["note"] = self._note(res, notes)
        if label:
            self.readings[label] = dict(res)
            res["label"] = label
        return res

    @staticmethod
    def _note(r: dict, notes: list) -> str:
        if r["silent_input"]:
            msg = ("the microphone delivered pure digital silence: it is muted on the device, or the computer does not allow this "
                   "program to use the microphone (System Settings > Privacy & Security > Microphone).")
        elif r["heard"]:
            msg = f"the tone was heard {r['snr_db']:g} dB above the noise at {r['freq_hz']:g} Hz."
            if r["steady"] is False:
                msg += " The tone was uneven from one segment to the next (the device was slow to start it?): the level may read low; repeat."
            if r["clipped"]:
                msg += " The microphone signal reached full scale: lower the level."
        elif r.get("interference"):
            msg = (f"NO VERDICT on this tone: it is {r['snr_db']:g} dB above the noise, but the segments with the tone were louder than "
                   f"the segments without it in only {r['separation']:.0%} of the pairs (at least {U_MIN:.0%} are needed): the room changed "
                   "while the pattern played (talking, music, typing, a notification). Repeat the same reading. This reading is "
                   "neither 'heard' nor 'inaudible'.")
        else:
            msg = (f"the tone was NOT heard: it is only {r['snr_db']:g} dB above the noise that the room makes at that frequency "
                   f"({r['noise_db']:g} dB; at least {MARGIN_DB:g} dB are needed). Louder, or the sound does not reach the microphone.")
        return " ".join(notes + [msg])

    def listen_noise(self, seconds: float = 1.0) -> dict:
        try:
            sec = min(max(float(seconds), 0.5), 5.0)
        except (TypeError, ValueError):
            return dict(ok=False, error="seconds must be a number")
        rec = np.asarray(self.b.capture(None, sec + 0.2), np.float64)[int(0.2 * self.b.sr_in):]
        silent = bool(len(rec) == 0 or float(np.max(np.abs(rec))) < 1e-7)
        hz, amp = spectrum_peak(rec, self.b.sr_in) if not silent else (0.0, 0.0)
        res = dict(ok=True, seconds=round(sec, 2), noise_rms_db=round(db(float(np.sqrt(np.mean(rec ** 2)))), 1) if len(rec) else FLOOR_DB,
                   strongest_hz=round(hz, 1), strongest_db=round(db(amp), 1), silent_input=silent, microphone=self.b.in_name)
        res["note"] = ("the microphone delivered pure digital silence (muted, or no permission)." if silent else
                       "the room noise as the microphone hears it; nothing was played and nothing is stored.")
        return res

    def compare_audio(self, label_a: str, label_b: str) -> dict:
        a, b = self.readings.get(label_a), self.readings.get(label_b)
        if a is None or b is None:
            miss = [x for x, r in ((label_a, a), (label_b, b)) if r is None]
            return dict(ok=False, error=f"no reading with the label {', '.join(map(repr, miss))}; labels kept: "
                                        f"{', '.join(sorted(self.readings)) or '(none)'}. Give label=... to play_tone first.")
        same = a["freq_hz"] == b["freq_hz"] and a["level_db"] == b["level_db"]
        delta = round(b["tone_db"] - a["tone_db"], 1)
        res = dict(ok=True, label_a=label_a, label_b=label_b, same_setting=same,
                   freq_hz_a=a["freq_hz"], freq_hz_b=b["freq_hz"], level_db_a=a["level_db"], level_db_b=b["level_db"],
                   tone_db_a=a["tone_db"], tone_db_b=b["tone_db"], delta_tone_db=delta,
                   heard_a=a["heard"], heard_b=b["heard"], noise_db_a=a["noise_db"], noise_db_b=b["noise_db"],
                   silent_input_a=a["silent_input"], silent_input_b=b["silent_input"], significant=bool(same and abs(delta) >= 6.0))
        if not same:
            res["note"] = "the two readings were made with different frequency or level: the difference does not compare the device."
        elif b["silent_input"] and not a["silent_input"]:
            res["note"] = f"the microphone became silent: the tone fell by {abs(delta):g} dB (to digital silence)."
        elif abs(delta) >= 6.0:
            res["note"] = f"the tone at the microphone became {'quieter' if delta < 0 else 'louder'} by {abs(delta):g} dB (b minus a: {delta:+g} dB); this is more than the noise explains."
        else:
            res["note"] = f"the tone changed by only {delta:+g} dB (b minus a): no real change."
        return res

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------- opening
def backend_kind() -> str:
    return os.environ.get("HW3_AUDIO_BACKEND", "sounddevice").strip().lower()


def available() -> tuple[bool, str]:
    """Whether the sound tools can be offered on this computer, and why not (checked without playing or recording anything)."""
    if os.environ.get("HW3_AUDIO", "").strip().lower() in ("off", "0", "no"):
        return False, "switched off (HW3_AUDIO=off)"
    if backend_kind() == "fake":
        return True, "synthetic sound (tests only)"
    try:
        import sounddevice  # noqa: F401
    except Exception as e:  # noqa: BLE001  (ImportError, or PortAudio missing: OSError)
        return False, f"the python package sounddevice is not usable ({type(e).__name__}: {e}); run: pip install sounddevice"
    try:
        b = SoundBackend()
    except AudioError as e:
        return False, str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    return True, f"loudspeaker {b.out_name!r}, microphone {b.in_name!r}"


def open_audio(out_sel: Optional[str] = None, in_sel: Optional[str] = None) -> AudioConnector:
    if backend_kind() == "fake":
        return AudioConnector(FakeSoundBackend())
    try:
        return AudioConnector(SoundBackend(out_sel, in_sel))
    except ImportError as e:
        raise AudioError(f"the python package sounddevice is missing ({e}); run: pip install sounddevice")
