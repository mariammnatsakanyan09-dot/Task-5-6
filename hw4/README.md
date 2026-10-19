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
are never the only evidence. **The loop closes on that measurement:** if the instrument is not back inside the limits, the agent is called again, up to `MAX_RETRIES = 5` times
(the check includes the check tone, so a loudspeaker that is still too quiet is not "back" either; the agent may ask the person again each time) with the numbers of the program's check; if the last check is still outside the limits, an `ESCALATION` note says what stays
outside and that a person has to look. **A state that cannot be fixed (for example a room that stays loud) does not silently become the normal:** the program
keeps a *reference state* (the first normal values of the run) and asks the person on the page whether this state may become the new normal. Reply `accept`: the
reference is measured again at the new state. Reply `keep` (or anything else): the normal values go back to the reference state, no new alarm starts for
`ESC_QUIET_S = 120` seconds, then the unfixed problem is noticed again. Without a person to ask (no `ask` function) the old behaviour stays. The reference is also kept in a file (see "The reference file" below), so a fault that is already there at Start is not silently taken as the reference any more.

The check tone is judged by its level at the microphone (`tone_db - level_db`, the "coupling"), not by how far it is above the noise: in a
reflecting room that margin stopped growing with the level (11.6, 12.3 and 11.7 dB at -30, -20 and -12 dBFS), so a rule "15 dB above the noise"
switched the check off, while the coupling falls by the same number of dB as the loudspeaker volume does. A level is accepted if the tone is
heard (10 dB or more); the quietest good level is taken. The **Stop** button ends a run after the step it is doing now (one model request and
its instrument calls); the number of model requests in the final line counts that run only.

**A fall of brightness always asks a person first.** On the real camera a partly covered lens kept a normal contrast and sharpness, and the agent decided "the light changed" and asked nobody (the dark region was the scene crop, the other regions stayed bright). Now the request for a brightness FALL tells the agent: the readings cannot tell a changed light from an obstacle in front of the lens, so ask the person one concrete action ("check that nothing is in front of the camera lens and remove it") before concluding that the light changed; a brightness RISE is reported without asking.

**A lost check tone.** If the check tone is not heard at any level the program may use (-30, -20, -12 dBFS; it cannot turn the volume of the device up), the program itself, without a model, asks the person on the page (up to 3 times, once per episode) to raise the volume of the Pano-360 and keep the room quiet, and calibrates again after every answer; if it is still not heard, an `ESCALATION` note says that the sound side is not watched.

### The reference file (`reference.json`)

The first normal values of a run are the *reference state*. They are written to `reference.json` (median values, a coarse 16 x 9 grid of the picture, the check
tone's level at the microphone, the time, the watched region) and read again at the next start. **Only a person's `accept` rewrites the file.**

* No file yet: the first normal values become the reference and are saved.
* A file, and the state now agrees with it (the same limits as the alarm): the stored reference stays, the file is not touched.
* A file, and the state now differs: the page asks the person (`accept` = the present state replaces the file, anything else = keep). If kept, the stored
  state is the normal, so the difference is an alarm and the loop (agent, program check, retries, ESCALATION) starts. This is how the program tells
  "the problem was fixed" (the readings are back at the stored reference) from "we got used to the problem" (they are not), also after a restart.
* Nobody to ask (no `ask`): the stored reference is kept; the file is never overwritten without a person.
* A damaged file, or a file of another region, is said aloud and ignored; `--fresh-reference` ignores a good file on purpose; `--no-reference` turns the file off.

### Guard mode (`--guard`): somebody should not be here

The loop above finds faults of the INSTRUMENT. A thief at night or a fire is an event of the ROOM, and the same two sensors can see signs of it. In guard
mode the watchdog also compares every sample with the stored reference and watches four things: the picture changed (more than `SCENE_CHANGE = 5 %` of the
16 x 9 cells differ, after a change of the whole picture by a lamp is removed: a lamp is the light, a figure is a movement), the light changed (the same limits as
before), the picture became warmer (red / blue ratio up by `WARM_RISE = 25 %`: a flame, or a warm lamp), the room became louder (`+8 dB`).

**A change of the room alone proves nothing**: a covered lens or a muted microphone looks like one. So when a room change is confirmed (2 of the last 3 samples),
the PROGRAM tests the instrument first, with the check tone: is the picture sharp, does the microphone answer, does the tone arrive. If the test finds a fault,
the usual loop starts (the agent looks at the instrument). If the instrument is in order, the change is real: a `ROOM ALERT` on the page, a picture of the region as evidence, one line in `alerts.log`, a notification of the system on a Mac and, if
`HW3_NOTIFY_URL` is set (for example `https://ntfy.sh/<your-secret-topic>` with the ntfy app on a phone), a push message. No model is used for it: it comes from numbers, in a few seconds. The same alert is
not repeated for `ROOM_QUIET_S = 60` s.

```bash
python3 web_app.py --model "<id>" --guard                       # the room is watched against reference.json, all the time
python3 web_app.py --model "<id>" --guard-hours 22:00-07:00     # the room is watched only at night, against the room of that night
export HW3_NOTIFY_URL="https://ntfy.sh/<your-secret-topic>"      # optional: a push message to the phone (the address is a password, do not publish it)
python3 web_app.py --model "<id>" --guard --region main_view     # watch the whole main view instead of the region of roi.json
python3 web_app.py --model "<id>" --fake --guard                 # the synthetic room: the page has buttons "Person enters", "Warm light"
```

### Guard hours (`--guard-hours 22:00-07:00`): people are normal by day, suspicious at night

In a lab somebody being in the room is normal; the same person at 03:00 is not. With `--guard-hours` (the clock of the computer, the window may cross midnight; it implies `--guard`) the room is watched only inside the window.
The reference of the day is not used for it: in the evening the people leave and the lights go off, which is no fault and no alarm. So when the window opens the program
leaves the room alone for `GUARD_SETTLE_S = 120` s (`--guard-settle`), then takes the room as it is (assumed empty) as the **reference of the night**, saves it to `guard_reference.json` together with the
date of the window, and from then on every change of the room is a `ROOM ALERT` (the instrument is tested first, as above). A restart in the same night (a power cut) reads the reference of that night back, so a person who is already there is not taken as normal.
Another night: measured anew. When the window ends the normal values are learned anew (the lights come on and the staff arrive: nothing to investigate).

A question that the program asks the person on the page (the reference at the start, the ESCALATION `accept`/`keep`, the volume) is not waited for for ever: after `ASK_TIMEOUT_S = 300` s of silence it counts as `keep` and the watch goes on (questions of the agent
are not affected). Not solved: the program does not know WHO is in the room; a night guard who is allowed there looks the same as a thief (the answer is a person who stops the guard hours, or a badge reader as one more sensor).

### A smoke detector (`--sensor-port`)

A smoke detector is a better sensor for smoke than a camera, so its alarm is **not confirmed and not investigated**: it is announced at once (the page, `alerts.log`, the push message), without a model, whether the watch runs or not, with a picture of the region after the message.
The program has a small separate server for outside sensors (its own port, so the page, which can start agents and spend money, stays on `127.0.0.1`):

```bash
export HW3_SENSOR_TOKEN="a-long-secret-of-at-least-12-characters"      # the shared secret
python3 web_app.py --model "<id>" --guard-hours 22:00-07:00 --sensor-port 8001                 # only this computer
python3 web_app.py --model "<id>" --guard-hours 22:00-07:00 --sensor-port 8001 --sensor-host 0.0.0.0 --sensor-signed-only   # a hub on the network may reach this door, signed messages only
python3 sensor_send.py smoke-1 alarm --detail "room 2"                    # a SIGNED message (this is what a hub or a small board runs)
curl -X POST http://127.0.0.1:8001/api/sensor -H "X-Token: $HW3_SENSOR_TOKEN" -H "Content-Type: application/json" -d '{"sensor":"smoke-1","alarm":true}'   # the plain secret (not with --sensor-signed-only)
```

**Signed messages.** A message carries `X-Timestamp` and `X-Signature` = HMAC-SHA256 of `"<timestamp>.<body>"` with the shared secret (`sensor_send.py` makes it, standard library only, so it also runs on a small board). The secret itself never travels, a message older than 5 minutes (or from the
future) is refused, and a signature is accepted once: somebody who copied the message from the network cannot send it again or change it. The plain secret in `X-Token` still works unless `--sensor-signed-only` is given (it travels in clear text, so use it only on this computer). The traffic itself is
not encrypted (a message can be read, not forged); behind a router that is acceptable, over the internet it is not.

`{"sensor": "<name>", "alarm": true|false, "detail": "<optional>"}`: `true` = alarm, `false` = all clear. A detector that repeats its message is announced again at most every `ROOM_QUIET_S = 60` s. How a real detector gets there:
(1) a smart detector (Zigbee, Z-Wave, Wi-Fi) that is connected to a hub (for example Home Assistant): an automation "when the smoke sensor is on, POST to this address"; no electronics; (2) an ordinary detector with a relay (dry contact) output on a small board
(Raspberry Pi, Arduino) whose script sends the same POST; (3) with no extra hardware the microphone could listen for the siren of the detector (a loud tone of about 3 kHz in a repeating pattern), as the program already finds its own check tone: NOT done here.
The synthetic page (`--fake`) has a button "Smoke detector alarm" that goes through the same door.
**The detector's own siren and the fire service must never depend on this program**: it is one more channel and a picture, not fire protection. Do not connect anything to a certified alarm system yourself.

### Keeping the guard itself alive and honest (what a night needs besides the detection)

A guard that fails silently is worse than none. These parts exist for that, none of them uses a model:

* **A push when the watch itself is in trouble** (`WATCH TROUBLE: ...`, the same notifier as a room alert: `alerts.log`, the system notification, `HW3_NOTIFY_URL`): the camera or the microphone stops answering (a pulled cable), the loop says `ESCALATION` (it could not fix the instrument, or the light or the
  noise stays outside the limits), the program crashes, and `WATCH TROUBLE ended` when the instrument answers again. The same message is not repeated for 5 minutes.
* **A heartbeat** (`--heartbeat-url` or env `HW3_HEARTBEAT_URL`, every `--heartbeat-every` seconds, at least 5): a ping to a service such as healthchecks.io, sent **only while the loop is healthy** (`Watchdog.health()`: the instrument answered within `BLIND_S = 60` s and the loop made its steps). When the pings stop, the
  SERVICE raises the alarm: this is the only way to learn that the program died, the Mac went to sleep or lost the internet, because a dead program cannot send a message. The address is a secret and is never written to the page. Needs an account of such a service (free ones exist); not tested against a real one.
* **Arming** (`--guard-hours`): when the window opens a push says "the room is taken as empty in 120 s; if somebody is still in the lab press 'I am in the lab'". The **page button** (Guard card: 60 minutes / 4 hours / resume now, `POST /api/guard/postpone`, 0-720 minutes) pauses the guard for somebody who is allowed to be there; after the pause the room is measured anew.
  When the reference of tonight is taken it is compared with last night's: if they differ strongly (somebody still in the room, a light left on) a `GUARD ARMING WARNING` is pushed; the guard starts anyway.
* **A quiet night:** inside the guard window the check tone is not played (a beep in an empty room is heard by whoever is in it). Without the tone a dead microphone could look like a quiet room, so at night a microphone level that does not move by even 0.005 dB in 8 readings is reported as a fault.
* **Faster ESCALATION for the room:** an agent cannot switch the light or make a room quiet. When what stays outside the limits is only the light or the noise and two checks in a row gave the same numbers, the loop says `ESCALATION` after 2 investigations instead of 6 (a lens, a cable or a loudspeaker still get the full number).
* **The words for the agent can be changed without code:** `--hints hints.json` (see `hints.example.json`; keys: cover, fell, rose, camera_only, noise, tone, silence, dead). The defaults were written after the incidents of ONE camera; another device may need other words. A wrong key or text stops the start with one sentence.
* **Logs that do not fill the disk:** `instrument_log.jsonl` becomes `.1` when it is larger than `HW3_LOG_MAX_MB` (default 20; two files at most); evidence pictures older than `--keep-days` (default 7; 0 = keep all) are deleted once an hour. The program also writes **one line per minute** to `watch_summary.csv` (samples, changes, brightness / noise ranges, picture change, alarms), and
  `python3 night_report.py hw3_work/watch_summary.csv --hours 22:00-07:00` prints how many alarms a night had and which limits the quiet minutes allow. Run it on a night in which NOBODY was in the room: then every alert was false, and the largest quiet values are the floor under which a limit must not be set.
* **Starting by itself:** `--autostart` begins watching at once (no button). `start_guard.sh` keeps the Mac awake (`caffeinate`), reads the secrets from the Keychain (nothing in a file) and starts the guard; `com.labwatchdog.plist.example` is a launchd file that starts it at login and again after a stop. A stop signal of the system ends the program in the orderly way (the last minute is written).
  One time: `security add-generic-password -a "$USER" -s openrouter -w` (and `labwatch-sensor`, `labwatch-heartbeat`, `labwatch-notify`, each asks for the secret itself), then `MODEL=<id> ./start_guard.sh --guard-hours 22:00-07:00`. **The Mac files were written on Linux and are not tested on a real Mac.**

What this is NOT, said plainly: (1) without `--guard-hours` the program does not know that people are not expected; **the person says it by starting guard mode**. (2) Only
what the camera sees in the watched region and what the microphone hears can raise an alert; `scene` is the region of `roi.json`, which may be a small patch, so for guarding use `--region main_view` or draw a larger region. (3) The limits (5 %, 25 %, 8 dB) are
guesses that were tested on the synthetic room only; they must be tuned on the real room (the page shows the live "picture differs from the reference" value). (4) A flame detector by colour is a hint, not fire protection: a smoke detector is the proper tool. (5) The evidence picture is a crop of the region: look at it before you
publish it. (6) An alert needs somebody who reads it: without `HW3_NOTIFY_URL` and an open page nobody is told at night.

**Ask mode.** Type a request or choose a use case of Homework 3 (1-11 and 15-19): the same run as in the terminal, shown live on the page.

Every run writes `runs/<time>-<ask|alert>/` (transcript, report, evidence, events, images) as before. Whole camera pictures are never
written, only crops of the measured region; sound is never written.

| file | what |
|---|---|
| `web_app.py`, `web_ui.html` | the server (state, event stream, ask, reply, stop, crops) and the page (live numbers and charts, the log of decisions / commands / measurements, the pause bar) |
| `watchdog.py` | the cheap loop and its rules (brightness 25 %, sharpness falls to a third, noise +8 dB, silence, check tone 6 dB weaker at the microphone, instrument errors), the limits given to the agent, the program's own check after an investigation, the reference file, guard mode (room alerts) and the notifier |
| `sensor_send.py`, `night_report.py`, `start_guard.sh`, `com.labwatchdog.plist.example`, `hints.example.json` | a signed message of a sensor; the report of a night from `watch_summary.csv`; the start of the guard on a Mac (awake, Keychain, autostart); the launchd file; an example of the replaceable hints |
| `test_web_offline.py` | 31 checks: the rules, the whole loop on the synthetic instrument (covered lens, muted microphone, lowered volume), the HTTP interface, the event stream on a real server, `why`, Stop, the request counter of a run, the settings of the page, the closed loop (fixed at once / fixed by the second call / escalation, the reference state), the reference file, guard mode (a person, a lamp, a flame colour, a loud room, a covered lens, a muted microphone, a quieter loudspeaker), the notifier, the coarse picture, the whole path from the synthetic camera to the page, guard hours (day / evening / night / morning / restart / next night), the smoke detector door, signed sensor messages, the pushes about the watch itself, the heartbeat, the pause of the guard, the quiet night, the hints file, log rotation and the night report, the programs started as a person starts them |

## The experiments

**Sound (use cases 15-19)**, the camera is not even switched on in 15-18:

* **15, the quietest tone the microphone hears.** With the current prompt (Homework 4) the agent plays 1000 Hz at four levels in one reply, then halves the interval around the
  threshold, one tone per reply: *decide, play, read, decide again*. (The run in the Homework 3 report was made with the earlier prompt and went one tone per reply.) It reports the lowest level that was heard and the highest that was not.
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
| checks | `test_hw3_offline.py`, `test_audio_offline.py`, `test_probe_offline.py`, `test_wire_fake_openrouter.py`, `test_web_offline.py` | 21 + 13 + 5 + 2 + 31 checks, no key, no camera, no sound card, nothing is played |

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
