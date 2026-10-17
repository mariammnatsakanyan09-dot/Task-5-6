# Homework 3: hardware in the loop

The Homework 2 system (orchestrator, scientist, general agent, executor, `human`; one LangGraph `StateGraph`) with a sixth
destination, `hardware`. The `hardware` agent is the only role that holds the tools of a real instrument. You type one request; the
agent decides which commands to send, reads the measurement and decides again.

**The instrument is one USB device, the Pano-360 (case label THAT DRAGONFLY)**, a 360-degree conference camera that is also a
speakerphone. The computer sees it as a UVC camera (one 1920 x 1080 picture = main view + panorama strip) and as a sound output and a
sound input, both named "Pano-360" (loudspeaker: 2 channels, microphone: 1 channel). That makes two instruments in one box, and **one
agent holds the tools of both**:

| instrument | command | reading |
|---|---|---|
| the **loudspeaker + microphone** | `play_tone(freq_hz, level_db, seconds)`: a pure tone through the loudspeaker | the microphone measures it at that frequency, the room noise at the same frequency, and the program's verdict `heard` |
| the **camera** | `set_resolution(w, h)` (640x480, 1280x720, 1920x1080) | brightness, noise, sharpness of a region |

```
 request ─▶ orchestrator ─▶ hardware ─▶ MCP client (bridge) ═ stdio ═▶ MCP server ─┬▶ audio_connector ─▶ loudspeaker ··air··▶ microphone
                 ▲   │          │  decide ▸ command ▸ reading ▸ decide…            └▶ camera_connector ─▶ camera
                 │   ├▶ scientist / general / executor (no instrument tools)
                 │   └▶ human  (pauses: ambiguous request / overwriting a reference / "press a button of the device")
```

## Homework 4: from prototype to product (a web page and a watchdog)

The same graph, the same MCP server, the same instrument, now as an application: a server (`web_app.py`, FastAPI) and one web page
(`web_ui.html`). The agents of Homework 3 changed in two small ways: every tool call now carries `why`, the agent's own
one-sentence reason (what it saw, why it chose this call; shown live, kept in the transcript, never sent to the instrument), and the hardware
agent is asked to put independent readings into ONE reply and to report in a few sentences (fewer model requests, a faster run). One bug was
fixed in the camera connector: `camera_info` answered without `ok` and was shown as "REJECTED" although the camera had answered.

```
pip install fastapi uvicorn
export OPENROUTER_API_KEY=...               # typed in the terminal
python3 web_app.py --model "<openrouter model id>"            # open http://127.0.0.1:8000
python3 web_app.py --model "<id>" --fake                      # synthetic instrument; the page has buttons that break it
```

**Watch mode** (the new capability). `watchdog.py` is a cheap loop WITHOUT a model: every few seconds it measures the scene (camera) and the
room noise (microphone), optionally plays a short check tone, and compares the numbers with the normal values (the median of the last 8 normal checks, about half a minute). While
everything is normal no model is called. When a kind of reading (scene, noise, check tone) disagrees with the normal values in two of the last
three samples, the watchdog hands an ALERT with the numbers to the graph. The orchestrator, the hardware agent and the `human` pause then do what
they did in Homework 3: the agent chooses its own readings (with a `why` for each), asks the person to fix what a person can fix (uncover
the lens, press the mute button; the pause is now a button on the page), checks that the readings are back to normal, and the watch goes on
with new normal values. Slow drift (the light changing during the day) is not an alarm, and one odd picture is not an alarm.

Three rules keep the verdict honest. (1) The normal values are taken only when the last samples agree with each other (a camera that has just
started changes exposure and focus; after 8 samples they are taken anyway). (2) The alert tells the agent the exact limits inside which the
watchdog itself calls a reading normal (for example "sharpness_norm at least 342" when the normal value is 1027), because in a real run the agent
compared with the old normal value only, found a reading of 395 "far below normal" and asked for a wipe that was not needed. (3) After the
investigation the PROGRAM measures once more, without a model, and writes whether the instrument is back inside those limits; the agent's words
are never the only evidence.

The check tone is judged by its level at the microphone (`tone_db - level_db`, the "coupling"), not by how far it is above the noise: in a
reflecting room that margin stopped growing with the level (11.6, 12.3 and 11.7 dB at -30, -20 and -12 dBFS), so a rule "15 dB above the noise"
switched the check off, while the coupling falls by the same number of dB as the loudspeaker volume does. A level is accepted if the tone is
heard (10 dB or more); the quietest good level is taken. The **Stop** button ends a run after the step it is doing now (one model request and
its instrument calls); the number of model requests in the final line counts that run only.

**Ask mode.** Type a request or choose a use case of Homework 3 (1-19): the same run as in the terminal, shown live on the page.

Every run writes `runs/<time>-<ask|alert>/` (transcript, report, evidence, events, images) as before. Whole camera pictures are never
written, only crops of the measured region; sound is never written.

| file | what |
|---|---|
| `web_app.py`, `web_ui.html` | the server (state, event stream, ask, reply, stop, crops) and the page (live numbers and charts, the log of decisions / commands / measurements, the pause bar) |
| `watchdog.py` | the cheap loop and its rules (brightness 25 %, sharpness falls to a third, noise +8 dB, silence, check tone 6 dB weaker at the microphone, instrument errors), the limits given to the agent, the program's own check after an investigation |
| `test_web_offline.py` | 11 checks: the rules, the whole loop on the synthetic instrument (covered lens, muted microphone, lowered volume), the HTTP interface, the event stream on a real server, `why`, Stop, the request counter of a run, the settings of the page |

## The experiments

**Sound (use cases 15-19)**, the camera is not even switched on in 15-18:

* **15, the quietest tone the microphone hears.** The agent plays 1000 Hz at four levels in one reply, then halves the interval around the
  threshold, one tone per reply: *decide, play, read, decide again*. It reports the lowest level that was heard and the highest that was not.
* **16, frequency response, then a closer look where it is strongest.** The agent measures 300 ... 6000 Hz at one level, then picks the
  strongest of them and plays one more frequency between it and each neighbour (decide, play, read, decide again). The program counts what is
  still missing (it tells the orchestrator, and refuses a `finish` before every frequency and both closer tones are played) and writes the curve
  summary itself: every reading, the spread of repeated readings, and whether the readings can tell the strongest frequency from the others
  ("NOT RESOLVED" when they cannot: on this device two plays of the same tone differ by about 3 dB).
* **17 and 18, buttons.** Reading *before*, the system pauses (`scene_action`) and asks the person to press the mute button (17) / the
  volume - button three times and then + three times (18) of the device, reading *after*, and `compare_audio` gives the change in dB
  (computed by the program). The device is tested by a measurement, not by trust. 17 needs the **microphone** of the Pano-360 (the
  mute button belongs to it), 18 needs its **loudspeaker** (the volume buttons belong to it): see "Which microphone hears which
  loudspeaker" below.
* **19, acceptance test of the whole device**: camera sharpness at three resolutions, the sound threshold and three frequencies, the
  volume buttons, one table. One hardware agent, two instruments.

**Camera (use cases 1-5).** The camera has one command that really changes what it delivers, the resolution: the agent sets a resolution and measures the scene at it.

## What is what

| part | file | what it does |
|---|---|---|
| (a) connector, sound | `audio_connector.py` | the **loudspeaker and the microphone** of the device through the OS sound driver (`sounddevice`): play a tone, record while it plays, measure the tone at its own frequency (lock-in), the noise just before it, the verdict `heard` (at least 10 dB above the noise and the strongest component), labelled readings and their comparison. Nothing recorded is stored |
| (a) connector, camera | `camera_connector.py`, `cam_platform.py` | the Pano-360 camera through the OS camera driver and OpenCV: the frame size (the command), readings of named regions, references |
| (b) MCP server | `mcp_camera_server.py` | exposes both connectors as tools over stdio: `camera_info`, `set_resolution`, `capture`, `measure_region`, `save_reference`, `compare_to_reference`; `audio_info`, `play_tone`, `listen_noise`, `compare_audio` (only if the computer has the device's loudspeaker and microphone); `set_control`, `reset_controls` **only if** `controls.json` lists camera controls that work. Every call is logged in `instrument_log.jsonl` |
| bridge | `mcp_bridge.py` | keeps one MCP session open in a background thread, so the instrument keeps its state (resolution, labelled readings) between the graph's calls |
| (c, d) agents | `hw3_agents.py` | the graph; the `hardware` node: model call → MCP tool call → measurement → model call … (max 14 model requests and 12 tool calls per turn, 40 tool calls per run) |
| runner | `run_hw3.py` | runs a request live, writes `runs/<time>/` |
| sound set-up | `audio_check.py` | checks without a model that the loudspeaker is heard by a microphone; `--pairs` tries every loudspeaker / microphone pair |
| camera set-up | `probe_camera.py`, `snap.py`, `pick_roi.py`, `smoke_camera.py`, `uvc_controls.py` | find the camera, **test which of its controls really work**, choose a measured region |
| optional | `screen_patch.py`, `pattern_window.py`, `fake_uvc_util.py` | an earlier idea (the monitor as a light source, `screen_level`), not needed; `fake_uvc_util.py` imitates the macOS tool uvc-util for the tests |
| checks | `test_hw3_offline.py`, `test_audio_offline.py`, `test_probe_offline.py`, `test_wire_fake_openrouter.py`, `test_web_offline.py` | 21 + 13 + 5 + 2 + 11 checks, no key, no camera, no sound card, nothing is played |

Rules enforced by the program, not by the prompt: only `hardware` is given the instrument tools; `save_reference(overwrite=true)` is not
sent to the server unless the human accepted that step; `finish` needs at least one reading; every number in the final report is copied
from the tool results; **whether a tone was heard is the program's verdict, and a change between two readings is computed by
`compare_audio`**, not by the model; the level of a tone is limited to -12 dBFS (the server cuts a louder request back) and a tone lasts at
most 3 s; the sound tools are offered only to the sound use cases; **no sound and no whole camera picture is ever written to disk** (only the measured region and numbers).

## What was found out about this device

*Camera.* The Pano-360 lists brightness, contrast, saturation and hue as UVC controls and accepts and reads back new values, but **the
picture does not change** (`probe_camera.py` measures the picture while sweeping each control: mean grey level, spread and colour saturation
stay flat). The probe therefore offers no camera control to the agent. What does change the delivered picture is the frame size, so
that is the camera's command.

*Sound.* `python audio_check.py --list` shows `Pano-360` twice: as an output (0 in, 2 out) and as an input (1 in, 0 out). Whether the
microphone hears the loudspeaker of the same device is **measured, not assumed**: conference devices often cancel their own sound out of
their microphone signal (echo cancellation). `python audio_check.py` plays quiet-to-louder tones and says; `--pairs` tries the other
combinations and saves the working pair in `audio.json`, which the system reads. **On the Pano-360 used for this project the
microphone of the device does not hear its own loudspeaker** (it is a speakerphone: it removes its own sound from its microphone), so
the Pano-360 loudspeaker is commanded and the **microphone of the computer** is the sensor (measured: the tone arrives 5 to 36 dB above
the room noise, depending on the level, the frequency and the room), while the Pano-360 microphone does hear the loudspeaker of the computer (about 18 dB above the noise). That is why
there are two modes (below).

*How a tone is measured, and why the room does not matter.* A room is never quiet: a fan, voices far away, a notification. So the tone
is not played as one long beep. It is **switched on and off in a fixed pseudo-random pattern** (segments of 0.2 s, 4 to 7 "on" and as many
"off", never three alike in a row, the same pattern every time) while the microphone records. The program finds the delay between playing
and recording in the recording itself, and measures the loudness at the tone's frequency in every segment:

* `noise_db` is the level of the "off" segments: what the room makes at that frequency, in the same recording, a fraction of a second away;
* `tone_db` is the level that the "on" segments have **in addition** (medians, so one loud burst does not count);
* `snr_db = tone_db - noise_db`. **heard** means: `snr_db` of at least 10 dB, the "on" segments are louder than the "off" ones in at least
  75 % of the pairs, **and** the loudness follows the played order better than up to 80 other orders do (a test against a room that
  happens to change in step by chance).

Whatever is in both kinds of segment (a fan, a hum, silence, steady voices) is subtracted; whatever does not follow the order cannot pass as
the tone. If the tone seems to be there but the segments are not cleanly apart (the room changed while the pattern played), the reading is
marked `interference` and has **no verdict**: it is neither "heard" nor "inaudible", the agent repeats it, and the report's "Sound levels
summarised by the program" never builds the threshold from such a reading. `test_audio_offline.py` (S1) checks this with a fan, a hum at the
tone's own frequency, voices, slams and a beeping notification, with and without the tone.

## Set up

Windows (PowerShell, in the repository folder):

```powershell
cd hw3
py -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked:  Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
```

macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

Close every program that shows the camera (Camera app, Teams, Zoom, browser tabs) or plays sound through the device.

### The sound (use cases 15-19)

```bash
python audio_check.py --list           # the sound devices: Pano-360 should be there twice (an output and an input)
python audio_check.py                  # beeps, quiet to louder: does the microphone hear the loudspeaker of the Pano-360?
python audio_check.py --pairs          # only if not: tries every loudspeaker / microphone pair and saves the working one in audio.json
```

Set the volume on the device to about the middle and keep the room quiet. On a Mac the first run asks for permission to use the microphone:
allow it for Terminal / Visual Studio Code (System Settings > Privacy & Security > Microphone). `HW3_AUDIO_MAX_DB` changes the loudest
tone (default -12 dBFS, allowed -40 to -3).

**Which microphone hears which loudspeaker.** The devices are chosen in this order: the environment variables `HW3_AUDIO_OUT` /
`HW3_AUDIO_IN` (a part of the name or the number), then the file `audio.json` (written by `audio_check.py --pairs`; delete it to go back
to the default; it is not published), then the name `Pano`. The system prints which one decided (`[chosen by audio.json]`).

| mode | loudspeaker | microphone | use cases | how |
|---|---|---|---|---|
| main (written to `audio.json`) | Pano-360 | microphone of the computer | 15, 16, 18, 19 | `python run_hw3.py --model ... --use-case 15` |
| mute button | loudspeaker of the computer | Pano-360 | 17 | `HW3_AUDIO_OUT="MacBook Pro Speakers" HW3_AUDIO_IN="Pano-360" python run_hw3.py --model ... --use-case 17` |

Before any model request the runner checks that the pair suits the use case (17 needs the Pano-360 microphone, 18 and 19 its
loudspeaker) and, if not, prints the command to use instead.

### The camera (use cases 1-5)

```bash
python snap.py                         # finds the Pano-360 itself, one picture -> probe_out/snap.png (open it; it stays on your computer)
python pick_roi.py probe_out/snap.png scene     # only for use cases 1-5: drag a box around the part of the scene to measure
python probe_camera.py                 # which controls really work (the Pano-360: none) -> controls.json
```

## Run

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."      # macOS: export OPENROUTER_API_KEY=sk-or-...   (never write the key in a file)
python run_hw3.py --model nvidia/nemotron-3-super-120b-a12b:free --use-case 15
```

| use case | request | shows |
|---|---|---|
| 15 | find the **quietest level at which a 1000 Hz tone is still heard** | closed loop with the loudspeaker: tone, reading, decide the next level, narrow the interval |
| 16 | frequency response at 7 frequencies; **then tones between the strongest frequency and its neighbours** | closed loop: the next frequencies come from the readings; the program checks completeness and writes the curve summary |
| 17 | does the **microphone mute button** work (the person presses it, then unmutes); uses the loudspeaker of the computer | the pause `scene_action`; reading before, reading after, the drop from `compare_audio` |
| 18 | do the **volume buttons** (- three times, + three times) change the loudness | the same, with the loudspeaker's own level |
| 19 | **acceptance test**: camera sharpness at 3 resolutions, sound threshold, 3 frequencies, the volume buttons | one agent, two instruments, one table (about 25 model requests) |
| 1 | first look at the scene, store the reference `base` | readings, a new reference |
| 2 | which resolution gives the sharpest picture | decide → set size → measure, repeated |
| 3 | replace the reference `base` | the pause: overwriting needs the human (try `refuse`, then run again with `accept`) |
| 4 | has the scene changed since `base` | comparison after you moved something |
| 5 | "Is the image good?" | the pause for an ambiguous request |
| 6-8 | (optional) the monitor patch as a light source: response curve, tune to 128, saturation point | needs `screen_level` in `controls.json` (`probe_camera.py --screen`) |
| 9-11 | (optional) the camera's own brightness / contrast / saturation | only if `controls.json` lists them (the Pano-360 does not) |

Every use case sends the device a real command and then measures: 15 and 16 command tones, 17-19 command tones around a physical
button press, 1, 3 and 4 first command 1920x1080, the size the
reference is made at, and 5 is the ambiguity pause (what it sends depends on the answer). When the person presses a button, the instrument still reacts to the agent's commands; the person only changes the experiment.

For 1-5 the "scene" is the region from `roi.json`: point the camera at something harmless (a sheet of paper); the saved crops of the region
are all that is written to disk.

`--show-images` opens each saved crop (for the video), `--reply "..."` answers a pause without typing. The free tier allows 20 requests per
minute and 50 per day; one run uses about 6 to 16 (19: about 25). Each run writes `runs/<time>/`: `report.md`, `evidence.md`, `transcript.md`,
`events.json`, `instrument_log.jsonl`, `images/`, `graph.mmd`. Look at `images/` before publishing.

## Optional: the monitor as a light source

An earlier idea for a camera control: a window on the monitor with a grey patch whose level (`screen_level`, 0 to 255) is commanded while
the camera looks at it (`python pattern_window.py --level 128`, `python probe_camera.py --screen`, use cases 6-8). The code and its tests are
in the repository; the experiments above do not need it.

## Tests (no key, no camera, no sound card)

```powershell
python test_hw3_offline.py; python test_audio_offline.py; python test_probe_offline.py; python test_wire_fake_openrouter.py; python test_web_offline.py
```

The camera and the loudspeaker / microphone in the tests are synthetic (`HW3_CAMERA_BACKEND=fake`, `HW3_AUDIO_BACKEND=fake`, also
`run_hw3.py --fake`): a picture that reacts to the commands, and a loudspeaker-to-microphone path with a room
noise, a resonance, a delay, a volume offset and a mute switch. They are never used silently and their readings are not measurements. The
real sound code (device choice, the thread that records while the tone plays) is tested on a stand-in for the `sounddevice` package.

> Use cases 12-14 (QR code on a phone) were removed from this folder; the numbers of the other use cases are unchanged.
