"""(b) THE MCP SERVER - exposes the device's controls and readings (the camera, and the loudspeaker + microphone of the same
device) as tools that any MCP client (here: the agent) can call.

Run by the client as a child process and spoken to over standard input/output (the MCP "stdio" transport):

    python3 mcp_camera_server.py

Because stdout is the protocol channel, this program never prints to stdout; messages go to stderr.

Environment variables
    HW3_WORKDIR          folder for crops, references and the instrument log (default: hw3_work)
    HW3_CAMERA_INDEX     force a camera number (default: find the Pano-360 by its two-part picture)
    HW3_CAMERA_BACKEND   "fake" selects the synthetic camera used by the offline tests
    HW3_ROI_FILE         path of roi.json (default: roi.json in the current folder)
    HW3_CONTROLS_FILE    path of controls.json made by probe_camera.py (default: controls.json in the current folder)
    HW3_CV_API           Windows: dshow (default) or msmf; see cam_platform.py
    HW3_LOG_MAX_MB       instrument_log.jsonl rotates at this size, default 20 (the previous part stays as instrument_log.jsonl.1)
    HW3_AUDIO            "off" does not offer the sound tools
    HW3_AUDIO_BACKEND    "fake" selects the synthetic loudspeaker/microphone used by the offline tests
    HW3_AUDIO_NAME / HW3_AUDIO_OUT / HW3_AUDIO_IN   which sound devices: part of a name or a number (default: the Pano-360 for both)
    HW3_AUDIO_MAX_DB     loudest tone that may be commanded, dBFS (default -12; allowed -40 .. -3)

Tools (each returns JSON text with "ok": true/false; a refused call says why and is not counted as a measurement)
    camera_info          what the instrument is, the frame size, the regions, what can be commanded
    set_resolution       COMMAND: 640x480, 1280x720 or 1920x1080
    set_control          COMMAND: brightness / contrast / saturation (exposure / gain ...) - offered ONLY if controls.json lists controls that
                         probe_camera.py verified on this camera (the picture really changed)
    reset_controls       COMMAND: put the controls back (offered together with set_control)
    capture              READING: one picture; statistics of every region; saves the crop of the measured region
    measure_region       READING: n pictures; brightness, noise, saturation and sharpness of one region
    save_reference       COMMAND: store the measured region as a named reference (refuses to overwrite)
    compare_to_reference READING: how the region differs now from a stored reference
  sound tools, offered only if the computer has the device's loudspeaker and microphone (see audio_connector.py):
    audio_info           the loudspeaker and the microphone, the allowed levels and frequencies
    play_tone            COMMAND + READING: play a pure tone through the loudspeaker, measure it with the microphone
    listen_noise         READING: the room noise as the microphone hears it (nothing is played)
    compare_audio        READING: the difference between two labelled play_tone readings, computed by the program

Every call is appended to <HW3_WORKDIR>/instrument_log.jsonl (time, tool, arguments, result).
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import audio_connector as ac
import cam_platform as plat
import camera_connector as cc

WORKDIR = Path(os.environ.get("HW3_WORKDIR", "hw3_work"))
WORKDIR.mkdir(parents=True, exist_ok=True)
LOG = WORKDIR / "instrument_log.jsonl"
LOG_MAX_BYTES = int(float(os.environ.get("HW3_LOG_MAX_MB", "20")) * 1024 * 1024)      # the log rotates at this size (instrument_log.jsonl.1 keeps the previous part)

mcp = FastMCP("pano360-camera")
_lock = threading.Lock()
_cam: cc.PanoCamera | None = None
_aud: ac.AudioConnector | None = None


def _log_err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _camera() -> cc.PanoCamera:
    """The camera is opened on first use, so the server starts even when the camera is missing."""
    global _cam
    if _cam is None:
        _log_err("[camera server] opening the camera ...")
        _cam = cc.open_camera(WORKDIR)
        _log_err(f"[camera server] camera opened: index {_cam.index}, backend {_cam.b.kind}")
    return _cam


def _audio() -> ac.AudioConnector:
    """The sound devices are opened on first use; the connector keeps the labelled readings between calls."""
    global _aud
    if _aud is None:
        _aud = ac.open_audio()
        _log_err(f"[server] sound devices: loudspeaker {_aud.b.out_name!r}, microphone {_aud.b.in_name!r}, backend {_aud.b.kind}")
    return _aud


def _shutdown() -> None:
    """The settings of a UVC camera stay in the camera until they are changed back: put them back when the server ends
    (the client closes the pipe, or sends SIGTERM), also if the agent forgot reset_controls."""
    global _cam
    cam, _cam = _cam, None
    if cam is not None:
        try:
            cam.close()
        except Exception:  # noqa: BLE001
            pass


atexit.register(_shutdown)
if hasattr(signal, "SIGTERM"):
    try:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    except (ValueError, OSError):
        pass


def _run(tool: str, args: dict, fn, device: str = "camera") -> str:
    t0 = time.time()
    with _lock:
        try:
            result = fn(_audio() if device == "audio" else _camera())
        except cc.CameraError as e:
            result = dict(ok=False, error=f"camera error: {e}")
        except ac.AudioError as e:
            result = dict(ok=False, error=f"sound error: {e}")
        except Exception as e:  # noqa: BLE001  (a tool must answer, never crash the server)
            _log_err(traceback.format_exc())
            result = dict(ok=False, error=f"unexpected error: {type(e).__name__}: {e}")
    logged = dict(result, thumb="[%d numbers]" % len(result["thumb"])) if isinstance(result.get("thumb"), list) else result   # the coarse grid is not worth a log line
    entry = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), tool=tool, args=args, seconds=round(time.time() - t0, 2),
                 result=logged)
    try:
        plat.rotate_log(LOG, LOG_MAX_BYTES)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError as e:
        _log_err(f"[camera server] cannot write the log: {e}")
    return json.dumps(result)


@mcp.tool()
def camera_info() -> str:
    """Describe the instrument: the camera, the frame size now, the named regions (x, y, width, height in pixels),
    which controls exist and which allowed resolutions can be set. Call it first when unsure what can be done."""
    return _run("camera_info", {}, lambda c: c.info())


@mcp.tool()
def set_resolution(width: int, height: int) -> str:
    """Command the camera to deliver pictures of a new size. Only 640x480, 1280x720 and 1920x1080 are offered; other
    sizes are rejected. The answer says which size the camera really delivers afterwards."""
    return _run("set_resolution", dict(width=width, height=height), lambda c: c.set_resolution(width, height))


@mcp.tool()
def capture(label: str = "", n_frames: int = 1) -> str:
    """Take a picture (the last of n_frames, 1..30) and return statistics of every region: mean_brightness (0..255),
    max, saturated_fraction (share of pixels at 250 or more), sharpness (Laplacian variance, higher is sharper) and
    sharpness_norm (the same after resizing the region to 400 px wide, so different resolutions can be compared).
    The crop of the measured region 'scene' is saved to disk; the whole picture is not."""
    return _run("capture", dict(label=label, n_frames=n_frames), lambda c: c.capture(label, n_frames))


@mcp.tool()
def measure_region(region: str = "scene", n_frames: int = 5, thumb: bool = False) -> str:
    """Take n_frames pictures (1..30) and return the average brightness, its frame-to-frame noise (standard deviation
    of the brightness over the frames), the maximum, the saturated fraction and the sharpness of one region
    ('scene', 'main_view' or 'panorama'). Use it to compare conditions (for example two resolutions).
    thumb=true (used by the watchdog, not needed by the agent) adds a coarse 16 x 9 grid of the region (mean grey level of each cell) and
    the red / blue ratio, so that the program can see that something moved in the room."""
    return _run("measure_region", dict(region=region, n_frames=n_frames, thumb=thumb), lambda c: c.measure_region(region, n_frames, thumb))


@mcp.tool()
def save_reference(name: str, overwrite: bool = False) -> str:
    """Store the current picture of the 'scene' region as a named reference (name: letters, digits, _ or -).
    If the name already exists the call is refused unless overwrite is true; overwriting destroys the old reference."""
    return _run("save_reference", dict(name=name, overwrite=overwrite), lambda c: c.save_reference(name, overwrite))


@mcp.tool()
def compare_to_reference(name: str, region: str = "scene") -> str:
    """Compare the current picture of a region with a stored reference: mean_abs_difference (grey levels),
    brightness_change (now minus reference) and correlation (1.0 = same pattern)."""
    return _run("compare_to_reference", dict(name=name, region=region), lambda c: c.compare_to_reference(name, region))


# ---- the sound tools exist only when the computer has the device's loudspeaker and microphone (the tool list follows the instrument)
AUDIO_OK, AUDIO_WHY = ac.available()
_log_err("[server] sound tools: " + ("offered (" + AUDIO_WHY + ")" if AUDIO_OK else "NOT offered: " + AUDIO_WHY))
if AUDIO_OK:
    @mcp.tool()
    def audio_info() -> str:
        """Describe the sound side of the instrument: the loudspeaker and the microphone of the device, the allowed levels (dBFS)
        and frequencies (Hz), and the labels of the readings kept for compare_audio. Call it when unsure what can be played."""
        return _run("audio_info", {}, lambda a: a.info(), "audio")

    @mcp.tool()
    def play_tone(freq_hz: float, level_db: float = -40.0, seconds: float = 1.6, label: str = "") -> str:
        """COMMAND and READING in one call (about 3 seconds): play a pure tone through the device's loudspeaker and measure it with
        its microphone at the same time. The tone is switched on and off in a fixed pseudo-random pattern of 0.2 s segments, so the
        moments without the tone measure what the room makes at that frequency (a fan, voices far away, silence). freq_hz: 100 Hz up to
        the limit in audio_info (about 6400); level_db: the digital level in dBFS, from -80 (very quiet) to -12 (loud; a louder request
        is cut back); seconds: 1.6 to 3 (the length of the pattern). Returns tone_db (what the tone adds at freq_hz), noise_db (the
        room at that frequency in the moments when the tone is off), snr_db = tone_db - noise_db, heard (true when snr_db is at least
        10 dB and the moments with the tone are louder than those without it in at least 75 % of the pairs: use this verdict,
        do not judge it yourself), coupling_db = tone_db - level_db (how much of the played level arrives at the microphone), separation
        (that share of pairs), delay_s, dominant_hz and dominant_db (the strongest component above 80 Hz), interference (true when
        snr_db is large but the moments with and without the tone are not cleanly apart because the room changed: NO verdict, heard is
        false), clipped, silent_input (the microphone delivered pure silence: muted, or no permission) and a note. label: an optional
        name (letters, digits, _ or -) under which the numbers are kept for compare_audio. Nothing is recorded to disk."""
        return _run("play_tone", dict(freq_hz=freq_hz, level_db=level_db, seconds=seconds, label=label),
                    lambda a: a.play_tone(freq_hz, level_db, seconds, label), "audio")

    @mcp.tool()
    def listen_noise(seconds: float = 1.0) -> str:
        """READING: listen to the room for `seconds` (0.5 to 5) with the device's microphone, playing nothing. Returns noise_rms_db
        (dBFS), the strongest frequency in the room and its level, and silent_input. Use it to see whether the room is quiet enough.
        Nothing is recorded to disk."""
        return _run("listen_noise", dict(seconds=seconds), lambda a: a.listen_noise(seconds), "audio")

    @mcp.tool()
    def compare_audio(label_a: str, label_b: str) -> str:
        """READING: compare two play_tone readings that were kept under labels. The program computes delta_tone_db (b minus a, the
        change of the tone at the microphone), whether both were made at the same frequency and level (same_setting), and whether
        the change is more than the noise explains (significant). Use it for before / after a physical change (a button pressed)."""
        return _run("compare_audio", dict(label_a=label_a, label_b=label_b), lambda a: a.compare_audio(label_a, label_b), "audio")


# ---- tools that exist only when the camera has verified controls (the tool list follows the instrument)
CONTROLS = cc.load_controls()
if CONTROLS:
    def set_control(name: str, value: float) -> str:
        return _run("set_control", dict(name=name, value=value), lambda c: c.set_control(name, value))

    def reset_controls() -> str:
        return _run("reset_controls", {}, lambda c: c.reset_controls())

    mcp.add_tool(
        set_control, name="set_control",
        description=("Command a control of the camera to a new value. Changing brightness, contrast or saturation (or exposure, "
                     "gain) changes what every later reading means, so compare readings only at the same settings. The answer gives the value "
                     "the camera read back (it may differ from the request); measure afterwards to see the effect. "
                     "Verified controls: " + cc.controls_summary(CONTROLS)))
    mcp.add_tool(
        reset_controls, name="reset_controls",
        description="Put every control changed with set_control back to the value it had before the first change.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
