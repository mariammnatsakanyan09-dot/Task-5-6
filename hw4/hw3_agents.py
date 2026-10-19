"""Homework 3: hardware in the loop.

The Homework 2 system (orchestrator, scientist, general agent, executor, human; one LangGraph StateGraph with a
checkpointer) gets a sixth node, `hardware`.

  * The `hardware` agent is the ONLY role that holds the instrument tools: the camera and, when the computer has them, the
    loudspeaker and microphone of the same device. The tools come from the MCP server (mcp_camera_server.py) and are reached
    through the bridge (mcp_bridge.py); the other roles never see them.
  * Which tool to call, with which arguments, in which order, is a model call of the hardware agent.
  * A hardware turn is a loop:  decide (model call) -> command the camera (MCP tool call) -> receive the measurement
    -> decide again (model call) ... until the agent has nothing more to call and reports.
  * The `human` node (interrupt / resume, as in Homework 2) pauses the graph for an ambiguous request and before a
    hardware step that overwrites a stored reference. The program also enforces it: save_reference(overwrite=true)
    is not sent to the camera server unless the human accepted that step. A third kind of pause, `scene_action`, is how the
    system moves something in front of the camera: the camera cannot, so the person does (press the mute or the volume button of
    the device) and answers "done, muted".
  * Every number in the final report is copied by the program from the tool results, never from model text.
"""
from __future__ import annotations

import json
import operator
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, field_validator

from llm_utils import (STATS, call, emit, extract_code, extract_json, markup_tool_calls, text_of,
                       tools_unsupported, daily_quota)

try:  # name changed between langgraph versions
    from langgraph.checkpoint.memory import InMemorySaver as _Saver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver as _Saver

MAX_DECISIONS = 9           # decisions of the orchestrator (every choice of a destination counts one, human included)
MAX_CAMERA_CALLS = 40       # instrument tool calls in the whole run (calls the program refused to send do not count)
MAX_CALLS_PER_TURN = 12     # instrument tool calls in one hardware turn
MAX_ROUNDS = 14             # model requests in one hardware turn (the last one only writes the report). A model that calls one tool
                            # per reply needs one round per tone; with 8 rounds it was cut off after 7 tones in a frequency curve.
EXEC_TIMEOUT_S = 60
STDOUT_TAIL = 4000
STDERR_TAIL = 3000

READING_TOOLS = ("capture", "measure_region", "compare_to_reference", "save_reference",
                 "play_tone", "listen_noise", "compare_audio")                                          # answers that hold measured numbers
COMMAND_TOOLS = ("set_resolution", "save_reference", "set_control", "reset_controls", "play_tone")        # tools that change the instrument or the files

# ----------------------------------------------------------------------------- prompts
INSTRUMENT_NOTE = """THE INSTRUMENT. One real camera, a Pano-360 (USB video class, 360-degree optics), connected to the computer.
Its picture is 1920 x 1080 and has two parts: the main view (upper part) and a panorama strip (lower part). The project
measures one rectangle called 'scene' (chosen by the person); 'main_view' and 'panorama' are also available.
Readings: mean_brightness (0-255), brightness_noise_std (frame to frame), max, saturated_fraction (share of pixels at 250
or more), rms_contrast (standard deviation of the grey levels), colour_saturation (mean HSV saturation, 0-255), sharpness
(variance of the Laplacian, higher = sharper) and sharpness_norm (the same after resizing the region to 400 px wide: use it to
compare different resolutions, because plain sharpness depends on the number of pixels).
The camera cannot move or press anything: when the physical scene has to change, the person does it.
@@AUDIO_NOTE@@@@CONTROLS@@@@SCENE_NOTE@@"""

CTX = {"controls": "", "audio": False, "curve": None}     # the verified controls, the sound tools that the server offers (set by build_graph
                                                           # from its tool list) and the frequency-curve plan of the use case, if it has one

AUDIO_NOTE = """The same device is also a speakerphone: the tool play_tone plays a pure tone through its loudspeaker and measures it with its
microphone (the command is the tone: frequency and digital level in dBFS; the reading is how loud the microphone hears it, the room noise
at that frequency, and a verdict heard / not heard), listen_noise listens to the room, compare_audio compares two labelled readings.
How loud the sound is in the room also depends on the volume set on the device and on the computer, which the system cannot change:
only the person can (a button of the device). A reading near the threshold (snr_db close to 10) is noisy: repeat it before relying on
it; differences of less than about 3 dB between two readings are not meaningful. Sound is never stored, only numbers.
"""
AUDIO_ORCH = (" It also holds the sound tools audio_info, play_tone, listen_noise and compare_audio (the device's own loudspeaker plays a "
              "tone, its microphone measures it): send here every question about what the loudspeaker and the microphone do. Give it the "
              "goal and the frequencies, but do not compute a tone level for it (for example from the room noise): it finds a suitable "
              "level itself from the field snr_db. Do not ask the person to set the volume unless the request is about the buttons or "
              "the volume of the device: the level of a tone is commanded in dBFS.")


def instrument_note() -> str:
    c = CTX["controls"]
    line = (f"Commands: the resolution (640x480, 1280x720, 1920x1080) and the controls verified on this camera: {c}. "
            if c else
            "No camera control is available on this computer (no brightness, contrast, saturation, exposure, gain or focus "
            "command). The only thing that can be commanded is the resolution (640x480, 1280x720, 1920x1080). ")
    scene = ("The region 'scene' lies on a grey patch shown on the monitor in front of the camera: it changes when the patch level "
             "(screen_level) is commanded, otherwise only if the person changes the scene."
             if "screen_level" in control_names() else "The scene changes only if the person changes it.")
    return (INSTRUMENT_NOTE.replace("@@AUDIO_NOTE@@", AUDIO_NOTE if CTX["audio"] else "")
            .replace("@@CONTROLS@@", line).replace("@@SCENE_NOTE@@", scene))


def fill(template: str) -> str:
    return template.replace("@@INSTRUMENT@@", instrument_note()).replace("@@AUDIO_ORCH@@", AUDIO_ORCH if CTX["audio"] else "")


AUDIO_TOOLS = """  audio_info()                                   what the loudspeaker and the microphone are, the allowed levels and frequencies
  play_tone(freq_hz, level_db, seconds, label)   COMMAND + READING: the loudspeaker plays a tone, the microphone measures it
  listen_noise(seconds)                          READING: the room noise as the microphone hears it
  compare_audio(label_a, label_b)                READING: change of the tone between two labelled play_tone readings
"""
AUDIO_RULES = (
    "Sound readings: play_tone plays a pure tone through the device's loudspeaker and, at the same time, measures it with the device's "
    "microphone (about 3 seconds per call). level_db is the digital level in dBFS from -80 (very quiet) to -12 (loud; louder requests "
    "are cut back); start quiet. It returns tone_db (what the microphone measured at that frequency), noise_db (the room noise at that "
    "frequency, measured in the same recording in the moments when the tone is off: the tone is switched on and off in a fixed "
    "pseudo-random pattern, so a fan, voices far away or silence are measured and taken away), snr_db = tone_db - noise_db, heard (the "
    "program's verdict: snr_db at least 10 dB and the moments with the tone are louder than the moments without it in at least 75 % of "
    "the pairs), coupling_db = tone_db - level_db, clipped, silent_input and a note. NEVER judge 'heard' yourself: "
    "use the field heard. A reading near the threshold is noisy: repeat it once before you rely on it. If interference is true, the reading has "
    "NO VERDICT: the room changed while the pattern played. It is NOT 'inaudible': never use it as 'the highest level "
    "not heard' and never as 'heard'. Repeat the same level once; if it happens again, "
    "report that the room was too noisy for that level, do not say the tone was inaudible.\n"
    "To find the QUIETEST level that is heard at a frequency (the threshold): in ONE reply play 4 levels that span the range (for "
    "example -80, -65, -50 and -35), then narrow the interval between the highest level that was not heard and the lowest that was "
    "heard by halving it, at most 3 more calls (each halving depends on the previous result, so it is one call per reply). Report the "
    "lowest level heard and the highest level not heard, exactly as the tools printed them. If the very first level (-80) is already "
    "heard, say that the threshold lies below the quietest level the system can command and report that reading. If nothing is heard "
    "at -35, go up with -25 and -12 before you say so, and then say what silent_input and the note report: do not guess.\n"
    "For a frequency curve use ONE level at every frequency, a level at which 1000 Hz is heard with snr_db of at least 20. If you "
    "do not know such a level yet, play 1000 Hz at -50 and go up by 15 dB until snr_db is at least 20. The tones of a curve do not "
    "depend on each other: once the level is known, send all the frequencies in ONE reply (several play_tone calls in one reply). "
    "If the orchestrator asks for tones between two frequencies, play them at the SAME level_db as the curve. Report tone_db and heard for "
    "each frequency; frequencies must lie between 100 Hz and the limit that audio_info gives.\n"
    "Report EVERY tone you played (every frequency, also the extra ones), do not say 'one tone' when you played more. If a reading "
    "has steady false or its note asks to repeat it, play that tone once more and report both readings. Do not fit curves, predict "
    "or calculate in your head: report the readings; calculations are done by the program or by the executor.\n"
    "To check a button or any physical change: play the tone with a label ('before') BEFORE the change, wait until the orchestrator "
    "tells you the person has done it, play the SAME freq_hz and level_db with another label ('after'), then call "
    "compare_audio('before', 'after'): the change in dB comes from the program, do not subtract numbers yourself. The tone must be "
    "heard in the 'before' reading, otherwise a drop cannot be seen: raise the level first.\n")
CONTROL_TOOLS = """  set_control(name, value)                       COMMAND: change a verified control (see the instrument description for names and ranges)
  reset_controls()                               COMMAND: put the controls back to the values they had before your first change
"""
CONTROL_HELP = {          # what each control does, said only for the controls that exist on this bench
    "screen_level": "screen_level sets the grey level (0 black ... 255 white) of a uniform patch that a window shows on the monitor "
                    "in front of the camera; the region 'scene' lies on that patch, so its readings follow the level (the camera's "
                    "automatic exposure may flatten the response; the tool already waits about a second for the camera to settle).",
    "brightness": "brightness makes the whole picture lighter or darker (reading: mean_brightness).",
    "contrast": "contrast spreads or squeezes the grey levels around the middle (reading: rms_contrast).",
    "saturation": "saturation makes the colours stronger or weaker (reading: colour_saturation).",
    "exposure": "exposure and gain make the whole picture lighter or darker (reading: mean_brightness).",
    "gain": "",
    "focus": "focus changes the sharpness of the picture (reading: sharpness).",
}


def control_names() -> list[str]:
    """Names of the verified controls, read from the summary that the server puts in the set_control description."""
    c = CTX["controls"]
    return [n for n in CONTROL_HELP if re.search(rf"(?:^|, ){n} -?\d", c)]


def control_rules() -> str:
    helps = " ".join(h for h in (CONTROL_HELP[n] for n in control_names()) if h)
    return ("Controls: set_control changes what every later reading means. " + helps + "\n"
            "Compare readings only at the same settings and say which setting each reading was made at. To find a setting that meets "
            "a target, change it, measure, and decide the next value from the result (two or three points, then interpolate or "
            "halve the interval; do not sweep blindly). Read saturated_fraction: a picture with many saturated pixels hides "
            "differences. Call reset_controls before you stop (the program also puts the controls back when the run ends).\n")



ORCHESTRATOR_SYSTEM = """You are the ORCHESTRATOR of a laboratory system that has one real instrument: a USB device with a camera and, if the
description below says so, a loudspeaker and a microphone.
You are the only role the user talks to. You see the whole dialogue and decide who speaks next.
The dialogue can also hold turns from the HUMAN (a person who answers your questions) and [graph] notes. A [graph] note is
written by the program after a human reply; it is not a role.

@@INSTRUMENT@@

Destinations you can choose:
- hardware: the ONLY role that can command the instrument. It holds the camera tools (camera_info, set_resolution, capture,
  measure_region, save_reference, compare_to_reference and, if the instrument description above lists verified controls,
  set_control and reset_controls).@@AUDIO_ORCH@@ It decides itself which tool to call and in which order, reads each
  result and decides again. Send here whenever a number or a picture must come from the real camera. Give it a goal
  ("measure the brightness and sharpness of the scene at the three resolutions, return to 1920x1080, say which gives the
  sharpest picture"), not a list of calls.
- scientist: plans the measurement: which conditions to compare, what a reading means, how to read the numbers. No tools.
- general: answers a short side question (what a reading means, what the camera can do). No tools.
- executor: runs a short Python script on files that already exist in the working directory (saved crops, references,
  instrument_log.jsonl) for arithmetic or image statistics that the camera tools do not give. It cannot reach the camera.
- human: pause the graph and ask the person. Before EVERY decision run these three checks, in this order. CHECK 1 and
  CHECK 2 are MANDATORY when they apply; CHECK 3 applies only when the goal needs a physical change of the scene. If none
  of them applies, human is forbidden:
  CHECK 1 - ambiguous_query. Read the latest user request. Does it have more than one reasonable reading, where the
  readings would send the camera to different measurements or change what is reported (for example "is the image good?":
  bright enough, sharp, or unchanged since a reference)? If yes, and no [human] turn has answered it yet, you MUST choose
  human now, before hardware. Put the question in `question`: name the readings and ask which to follow. Leave `guarded`
  empty. After the human has answered, follow that reading and never ask about it again. Do not ask about details that a
  reasonable default settles.
  CHECK 2 - hard_decision. Is the step you are about to choose a hardware step that would OVERWRITE a stored reference
  (save_reference with a name that is already in the list of stored references in the facts below)? Overwriting destroys
  the old reference. If yes, and the human has not already accepted this step (look for [human] and [graph] turns), you MUST
  choose human now instead of hardware. Put guarded="hardware", the goal for the hardware agent in `instruction`
  (including that the human accepted the overwrite), and in `question` state which reference would be destroyed and ask
  whether to proceed. The step does NOT run until the human accepts. If the human refuses, the step is not run and the turn
  comes back to you: propose another step (a new reference name) or finish and say plainly that nothing was overwritten.
  CHECK 3 - scene_action. The instrument cannot move or press anything. If the goal needs the person to change the physical
  scene before the next hardware step (for example to press a
  button of the device such as the microphone mute or the volume), choose
  human now, with kind scene_action, BEFORE that hardware step. Put in `question` exactly what to do and ask the person to
  reply "done" with the real value (for example "done, muted"). Leave `guarded` empty. After the reply, send
  the hardware agent and write the condition that the person gave into its instruction. Ask once for each new condition; never ask
  for a condition that is already in place. When a hardware step must take a reading BEFORE the physical change and another AFTER it
  (checking a button), first send hardware to take the reading before, then ask the person (scene_action), then send hardware again
  for the reading after and the comparison: tell it which label the earlier reading has.
  Everything else the instrument does (taking pictures, changing the resolution, playing a tone and listening,
  saving a reference under a NEW name) needs no question. If no check applies, do not choose human.
- finish: end the run. Allowed only after a hardware turn returned at least one reading (a successful capture, measure_region,
  play_tone, listen_noise, compare_audio, compare_to_reference or save_reference; the facts below count them), or
  after the human refused a step that was needed. Put the final message for the user in `instruction`: what was done, which
  agent did what, the readings exactly as the hardware agent reported them. If nothing was measured, say so plainly.

You write no theory and no code. You have at most {max_decisions} decisions in total; {used} are already used. Every choice
of destination, human included, counts as one decision; a step that the human accepts runs without costing another
decision. Plan the budget: typically hardware -> finish; add scientist, executor or human only when they are needed.
Return a decision with:
  next        scientist | general | executor | hardware | human | finish
  instruction what that role should do now, concrete and self-contained (for human + hard_decision: the goal of the guarded
              hardware step; for finish: the final message; for human + ambiguous_query or scene_action: empty)
  reason      ONE sentence: why this destination is next
  question    only for human: the question the person will read
  kind        only for human: ambiguous_query | hard_decision | scene_action
  guarded     only for human + hard_decision: hardware"""

SCIENTIST_SYSTEM = """You are the SCIENTIST. You plan the measurement and read the numbers. You have no tools.
The dialogue can hold HUMAN turns (answers from the person): honour them.
Say which conditions to compare and which readings answer the question, what a difference of a given size would mean
(compare it with the frame-to-frame noise that the readings report), and what could mislead (for example: plain sharpness
depends on the number of pixels of the region, sharpness_norm does not; the scene must not change between readings).
Do not claim that any number was measured unless a hardware turn in the dialogue already reported it. Be concise.

@@INSTRUMENT@@"""

GENERAL_SYSTEM = """You are the GENERAL AGENT. You answer a short side question: what a reading means, what the camera can
do, what a term in the request means. You have no tools and you do not plan measurements. If the instruction asks for a plan
or for camera readings, say in one sentence that it belongs to the scientist or the hardware agent.
Be short and concrete. Do not invent measured numbers; quote only numbers that appear in the dialogue.

@@INSTRUMENT@@"""

HARDWARE_SYSTEM = """You are the HARDWARE agent, the only role that holds the instrument tools. You command the real device (its camera, and
its loudspeaker and microphone when the sound tools are listed below) and read what it measured.

Tools (each returns JSON with "ok": true or false; a rejected call says why and is not a measurement):
  camera_info()                                  what the instrument is, the frame size, the regions, what can be commanded
  set_resolution(width, height)                  COMMAND: 640x480, 1280x720 or 1920x1080
  capture(label, n_frames)                       READING: statistics of every region; saves the crop of the scene
  measure_region(region, n_frames)               READING: n_frames pictures: brightness, noise, saturation, sharpness of a region
  save_reference(name, overwrite)                COMMAND: store the scene as a named reference; refused if the name exists
  compare_to_reference(name, region)             READING: difference between now and a stored reference
@@AUDIO_TOOLS@@@@CONTROL_TOOLS@@
How you work: read the goal and call the tools it needs. Put every independent reading in ONE reply (several tool calls at once: they run
one after another, and every extra reply costs about 10 s); call a tool alone only when its arguments depend on an earlier result (for
example a tone level chosen from the snr_db). Read the results, and decide again:
call more tools or stop. You choose the tools, their order and their arguments. Compare conditions only with the same
region and the same number of frames. After a resolution change the next reading is made at the new size; compare sharpness
between resolutions with sharpness_norm. If you changed the resolution only for a comparison, put it back to the size the goal
names (or 1920x1080) before you stop. save_reference with overwrite=true destroys the old reference: use it ONLY if the goal
says that the human accepted the overwrite. If a call is rejected, the answer says what to change: fix it and call again, or
report the problem. EVERY tool call also carries the extra argument why: ONE short sentence that says what you have seen so far
and why you chose this call (for example "brightness fell from 142 to 4, I suspect a covered lens, so I check another region");
the person reads it live and it goes into the report. It is not a measurement and the instrument never sees it.
@@AUDIO_RULES@@@@CONTROL_RULES@@You may make up to {per_turn} tool calls in this turn; {left} are left in the whole run. When you have what the goal needs,
reply in 2-5 short sentences (no table, no markdown) without a tool call: what you did and the readings EXACTLY as the tools printed them. Never invent,
round or recompute a measurement.

@@INSTRUMENT@@

{facts}"""

EXECUTOR_SYSTEM = """You are the EXECUTOR, the only role with a coding tool: run_python(code, purpose).
The tool writes your code to a NEW file in the working directory and runs it there with a {timeout} second limit, returning
the exit code, stdout and stderr. Call it exactly once per turn.
The working directory holds the files of this laboratory: crops/ (PNG pictures of the measured region), references/ (PNG and
JSON of stored references), instrument_log.jsonl (every instrument call and its result, one JSON object per line: the tool, its
args and its result; a play_tone entry has args freq_hz and level_db and the result fields tone_db, noise_db, snr_db and heard). You have NO
access to the instrument: read files only, never open the camera or the sound devices. Python, numpy and opencv (cv2) are installed. Print what you
compute; never overwrite or delete an existing file. After the tool returns, reply in 2-4 sentences: the exit code and the
numbers exactly as printed, or the cause of the failure. Never invent numbers."""

TEXT_MODE_NOTE = (
    "The tool-calling interface is not available for this model. Reply with ONE JSON object and nothing else: "
    '{"calls": [{"tool": "set_resolution", "args": {"width": 1280, "height": 720}}, ...], "report": ""} - one entry per '
    "camera call (the keys of args are the arguments of the tool). If the goal is reached and no call is needed, reply "
    '{"calls": [], "report": "2-5 sentences with the readings exactly as printed"}. At most {per_turn} calls.'
)
FINAL_NOTE = ("The tools are not available any more in this turn. Reply in 2-5 sentences, without a tool call: what you did and "
              "the readings exactly as the tools printed them; say what is still missing if the goal is not reached.")
NUDGE = ("You answered without calling a tool. The goal needs a reading or a command from the real camera: call the tool now.")
JSON_NOTE = (
    "Reply with ONE JSON object and nothing else (no markdown fence, no commentary) with exactly these keys: "
    '"next" (scientist|general|executor|hardware|human|finish), "instruction" (string), "reason" (one sentence), '
    '"question" (string or null), "kind" ("ambiguous_query"|"hard_decision"|"scene_action"|null), "guarded" ("hardware"|null).'
)
EXEC_TEXT_NOTE = ("The tool-calling interface is not available for this model. Reply with the COMPLETE script in ONE fenced "
                  "```python block and nothing else; it will be written to a new file and run exactly as the tool would.")


# ----------------------------------------------------------------------------- state
class Turn(TypedDict):
    role: str
    text: str


class State(TypedDict, total=False):
    dialogue: Annotated[list[Turn], operator.add]
    events: Annotated[list[dict], operator.add]   # every camera tool call that was sent, with the answer as the tool returned it
    n_calls: int                                  # camera tool calls sent so far
    next: str
    instruction: str
    pending: Optional[dict]                       # the human question that is open (kind, question, guarded step)
    approved_overwrite: bool                      # the human accepted a hardware hard decision: valid for ONE hardware turn
    decisions: int
    refused: int                                  # how many hard decisions the human refused
    final_text: str
    hw_report: str                                # the closing words of the last hardware turn (for the live display)
    stop_reason: str


class Decision(BaseModel):
    next: Literal["scientist", "general", "executor", "hardware", "human", "finish"] = Field(
        description="destination of the next turn, or finish")
    instruction: str = Field(default="", description="what that role should do now; for finish the final message to the user; "
                                                      "for human + hard_decision the goal of the guarded hardware step")
    reason: str = Field(default="", description="one sentence: why this destination is next")
    question: Optional[str] = Field(default=None, description="only for human: the question for the person")
    kind: Optional[Literal["ambiguous_query", "hard_decision", "scene_action"]] = Field(
        default=None, description="only for human: why the person is asked (scene_action: to change something in front of the camera)")
    guarded: Optional[str] = Field(default=None, description="only for human + hard_decision: hardware")

    @field_validator("instruction", "reason", mode="before")
    @classmethod
    def _none_to_empty(cls, v):
        return "" if v is None else v


# ----------------------------------------------------------------------------- helpers
def render(dialogue: list[Turn]) -> str:
    return "\n\n".join(f"[{t['role']}]\n{t['text']}" for t in dialogue)


def current_instruction(state: State) -> str:
    if state.get("instruction"):
        return state["instruction"]
    for t in reversed(state["dialogue"]):
        if t["role"] == "orchestrator":
            return t["text"]
    return ""


def as_decision(obj: Any) -> Decision:
    if isinstance(obj, Decision):
        return obj
    if isinstance(obj, dict):
        return Decision(**obj)
    raise ValueError(f"orchestrator model returned no decision: {obj!r}")


def normalize(d: Decision) -> Decision:
    if d.next == "human":
        kind = d.kind or ("hard_decision" if d.guarded else "ambiguous_query")
        return d.model_copy(update={"kind": kind, "guarded": d.guarded if kind == "hard_decision" else None})
    return d.model_copy(update={"kind": None, "guarded": None, "question": None})


def decision_problem(d: Decision) -> Optional[str]:
    """Why a decision cannot be executed (None if it can)."""
    if d.next != "human":
        return None
    if not (d.question or "").strip():
        return "next=human needs a non-empty `question` for the person."
    if d.kind == "hard_decision":
        if d.guarded != "hardware":
            return "a hard_decision needs `guarded` = hardware (the hardware step that runs after an accept)."
        if not d.instruction.strip():
            return "a hard_decision needs `instruction` (the goal of the guarded hardware step)."
    return None


class DecisionCaller:
    """One model call that returns a Decision. First choice is the provider's structured output; free OpenRouter models
    often lack it, so on failure the caller switches (for the rest of the run) to a plain-text call that asks for one JSON
    object. A decision that cannot be executed is sent back to the model with the reason, up to `attempts` times."""

    def __init__(self, model, attempts: int = 3):
        self.model, self.attempts = model, attempts
        self.structured, self.use_text = None, False

    def _once(self, msgs) -> Decision:
        if not self.use_text:
            try:
                if self.structured is None:
                    self.structured = self.model.with_structured_output(Decision)
                return as_decision(call(self.structured, msgs))
            except Exception as e:  # noqa: BLE001
                if isinstance(e, RuntimeError) and daily_quota(e.__cause__ or e):
                    raise
                emit(f"    [note] structured output unavailable ({type(e).__name__}: {str(e)[:100]}); "
                     f"asking for JSON text from now on")
                self.use_text = True
        reply = text_of(call(self.model, msgs + [HumanMessage(content=JSON_NOTE)]))
        return as_decision(extract_json(reply))

    def __call__(self, messages) -> Decision:
        problem = None
        for _ in range(self.attempts):
            msgs = messages if problem is None else messages + [
                HumanMessage(content=f"Your previous answer was rejected: {problem} Answer again.")]
            d = normalize(self._once(msgs))
            problem = decision_problem(d)
            if problem is None:
                return d
        raise ValueError(f"the orchestrator model kept returning an unusable decision: {problem}")


# ---- what the program counts and shows (facts, not model text)
def readings_of(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("ok") and e["tool"] in READING_TOOLS]


def references_on_disk(workdir: Path) -> list[str]:
    d = Path(workdir) / "references"
    return sorted(p.stem for p in d.glob("*.png")) if d.exists() else []


def gate_facts(dialogue: list[Turn]) -> dict:
    """What the human has already answered / accepted, counted by the program from the dialogue."""
    n_amb = a_hw = n_scene = 0
    last_kind = None
    for t in dialogue:
        if t["role"] == "orchestrator":
            m = re.search(r"^kind: (\S+)", t["text"], re.M)
            last_kind = m.group(1) if m else None
        elif t["role"] == "human" and last_kind == "ambiguous_query":
            n_amb += 1
        elif t["role"] == "human" and last_kind == "scene_action":
            n_scene += 1
        elif t["role"] == "graph" and "accepted by the human: hardware" in t["text"]:
            a_hw += 1
    return dict(n_amb=n_amb, a_hw=a_hw, n_scene=n_scene)


def facts_block(state: State, max_decisions: int, workdir: Path) -> str:
    """Placed at the END of what the orchestrator reads (weaker models follow what is nearest to the answer)."""
    f = gate_facts(state["dialogue"])
    events = state.get("events", [])
    refs = references_on_disk(workdir)
    curve = curve_facts(events)
    return (
        "[facts counted by the program]\n"
        f"decisions used: {state.get('decisions', 0)} of {max_decisions}. Instrument tool calls sent: {state.get('n_calls', 0)} of "
        f"{MAX_CAMERA_CALLS}; successful readings so far: {len(readings_of(events))} (finish needs 1).\n"
        f"The human has answered {f['n_amb']} ambiguity question(s), accepted {f['a_hw']} hardware step(s) and refused "
        f"{state.get('refused', 0)} step(s). Scene actions done by the person (moved something, answered 'done'): {f['n_scene']}.\n"
        f"Stored references: {', '.join(refs) if refs else '(none)'}.\n"
        "[checklist before you answer]\n"
        "CHECK 1: is the latest user request ambiguous and still unanswered by the human? Then choose human (ambiguous_query).\n"
        "CHECK 2: is the hardware step you are about to choose a save_reference under a name that is in the stored references "
        "above, without the human having accepted it? Then choose human (hard_decision) instead, with guarded=hardware.\n"
        "CHECK 3: does the goal need the person to change the scene (for example pressing a button of "
        "the device) before the next hardware step, and has the person not yet done it for that condition? Then choose human "
        "(scene_action) first."
        + ("\n" + curve if curve else "")
    )


def short_result(name: str, r: dict) -> str:
    """One line for the terminal and the report, built from the tool result."""
    if not r.get("ok"):
        return "REJECTED: " + str(r.get("error") or r.get("note") or "?")[:200]
    try:
        if name == "set_resolution":
            return f"the camera delivers {r['accepted'][0]}x{r['accepted'][1]} now"
        if name == "measure_region":
            return (f"{r['region']} {r['resolution'][0]}x{r['resolution'][1]}: brightness {r['mean_brightness']} "
                    f"(noise {r['brightness_noise_std']}), contrast {r['rms_contrast']}, "
                    + (f"colour saturation {r['colour_saturation']}, " if "colour_saturation" in r else "")
                    + f"sharpness_norm {r['sharpness_norm']}, saturated {r['saturated_fraction'] * 100:.2f} %")
        if name == "capture":
            regs = r["regions"]
            pick = "scene" if "scene" in regs else next(iter(regs))
            s = regs[pick]
            return (f"capture {r['capture']} {r['resolution'][0]}x{r['resolution'][1]}: {pick} brightness {s['mean_brightness']}, "
                    f"sharpness_norm {s['sharpness_norm']}, saturated {s['saturated_fraction'] * 100:.2f} %"
                    + (f"; saved {', '.join(r['saved_crops'])}" if r.get("saved_crops") else ""))
        if name == "save_reference":
            s = r["stats"]
            return (f"reference {r['name']} {'OVERWRITTEN' if r.get('overwritten') else 'saved (new)'}: brightness "
                    f"{s['mean_brightness']}, sharpness_norm {s['sharpness_norm']} -> {r['file']}")
        if name == "compare_to_reference":
            return (f"{r['region']} vs reference {r['reference']}: mean abs difference {r['mean_abs_difference']}, "
                    f"brightness change {r['brightness_change']}, correlation {r['correlation']}")
        if name == "set_control":
            return f"{r['control']} requested {r['requested']:g}, the camera read back {r['read_back']} ({r['unit']})"
        if name == "reset_controls":
            return "controls put back: " + (", ".join(f"{k}={v}" for k, v in r["restored"].items()) or "nothing had been changed")
        if name == "camera_info":
            return f"frame {r['frame_size'][0]}x{r['frame_size'][1]}; regions: {', '.join(r['regions'])}"
        if name == "play_tone":
            return (f"tone {r['freq_hz']:g} Hz at {r['level_db']:g} dBFS: microphone {r['tone_db']:g} dB, noise {r['noise_db']:g} dB, "
                    f"{r['snr_db']:g} dB above the noise -> {'HEARD' if r['heard'] else 'NOT heard'}"
                    + (f" (NO VERDICT: the room changed while the tone played, separation {r['separation']:.0%})" if r.get("interference") else "")
                    + ("; the microphone is silent" if r.get("silent_input") else "")
                    + (f" (label {r['label']})" if r.get("label") else ""))
        if name == "listen_noise":
            return (f"room noise {r['noise_rms_db']:g} dBFS, strongest component {r['strongest_hz']:g} Hz at {r['strongest_db']:g} dB"
                    + ("; the microphone is silent" if r.get("silent_input") else ""))
        if name == "compare_audio":
            return (f"{r['label_a']} -> {r['label_b']}: tone {r['tone_db_a']:g} -> {r['tone_db_b']:g} dB, change {r['delta_tone_db']:+g} dB, "
                    f"{'significant' if r['significant'] else 'not significant'}"
                    + ("" if r["same_setting"] else " (different settings: not comparable)"))
        if name == "audio_info":
            return f"loudspeaker {r['loudspeaker']}, microphone {r['microphone']}; levels {r['level_db_range'][0]:g}..{r['level_db_range'][1]:g} dBFS"
    except (KeyError, TypeError, IndexError):
        pass
    return json.dumps(r)[:200]


def events_table_md(events: list[dict]) -> str:
    if not events:
        return "(no instrument tool call was sent)"
    rows = ["| # | tool | arguments | what the instrument answered |", "|---|---|---|---|"]
    for e in events:
        args = ", ".join(f"{k}={v}" for k, v in e["args"].items()) or "-"
        rows.append(f"| {e['n']} | `{e['tool']}` | {args} | {short_result(e['tool'], e['result'])} |")
    return "\n".join(rows)


def sound_summary_md(events: list[dict]) -> list[str]:
    """For every frequency that was played at two or more levels: which levels were heard, which were not, and which have NO
    VERDICT (the room got louder during the tone). Computed by the program from the readings, so the threshold in the report
    does not depend on how a model reads them."""
    by_f: dict = {}
    for e in events:
        r = e.get("result") or {}
        if e.get("tool") == "play_tone" and e.get("ok") and "level_db" in r:
            by_f.setdefault(r["freq_hz"], {}).setdefault(r["level_db"], []).append(r)
    lines = []
    for f, levels in sorted(by_f.items()):
        if len(levels) < 2:
            continue
        heard, not_heard, none = [], [], []
        for lv, rs in sorted(levels.items()):
            if any(x.get("heard") for x in rs):
                heard.append(lv)
            elif all(x.get("interference") for x in rs):
                none.append(lv)
            elif not all(x.get("silent_input") for x in rs):
                not_heard.append(lv)
        fmt = lambda xs: ", ".join(f"{x:g}" for x in xs) or "none"
        low = min(heard) if heard else None
        below = [x for x in not_heard if low is None or x < low]
        lines.append(f"- {f:g} Hz, levels in dBFS: heard at {fmt(heard)}; not heard at {fmt(not_heard)}; no verdict (the room changed "
                     f"while the tone played) at {fmt(none)}. Lowest level heard: {fmt([low]) if low is not None else 'none'}; highest level "
                     f"not heard below it: {fmt([max(below)]) if below else 'none'}.")
    return lines


# ---- the frequency curve (use case 16): what the program counts, checks and writes, so that the result does not depend on how a
# model reads the numbers (in a real run the model said "one more tone" after three, and "confirmed" after a difference of 3 dB)
CURVE_PLAN = dict(freqs=(300.0, 600.0, 1000.0, 2000.0, 3000.0, 4000.0, 6000.0))
FREQ_TOL = 0.01          # a tone within 1 % of a requested frequency counts as that frequency
MIN_SPREAD_DB = 3.0      # two plays of the same tone differ by about this much: a smaller difference between frequencies means nothing
CLEAR_SNR_DB = 20.0      # the request: the level at which 1000 Hz is heard at least this far above the noise


def _planned(f: float, freqs) -> Optional[float]:
    return next((p for p in freqs if abs(f - p) <= FREQ_TOL * p), None)


def _named(text: str, f: float) -> bool:
    """Does the text name the frequency f (600 is not found inside 6000 or 1600.5)?"""
    num = "%g" % f
    return re.search(r"(?<![\d.])" + re.escape(num) + r"(?![\d])", text) is not None


def curve_status(events: list[dict], plan: dict) -> dict:
    """The readings of a frequency curve, grouped by the program. The level of the curve is the level at which most of the requested
    frequencies were played (the latest wins a tie); only readings at that level belong to the curve. A reading with NO VERDICT
    (the room changed) or a silent microphone does not count as played."""
    freqs = sorted(float(f) for f in plan["freqs"])
    rs = [e["result"] for e in events if e.get("tool") == "play_tone" and e.get("ok") and isinstance(e.get("result"), dict)
          and all(k in e["result"] for k in ("freq_hz", "level_db", "tone_db"))]
    st: dict = dict(freqs=freqs, level=None, rows=[], played=[], missing=list(freqs), best_planned=None, refine_missing=[],
                    spread=None, spread_eff=MIN_SPREAD_DB, best=None, within=[], no_verdict=[], extra=[], snr_1000=None)
    seen: dict = {}
    for i, r in enumerate(rs):
        p = _planned(r["freq_hz"], freqs)
        if p is not None and not r.get("interference") and not r.get("silent_input"):
            s = seen.setdefault(round(r["level_db"] * 10), [set(), i])
            s[0].add(p)
            s[1] = i
    if not seen:
        return st
    key = max(seen, key=lambda k: (len(seen[k][0]), seen[k][1]))
    at = [r for r in rs if round(r["level_db"] * 10) == key]
    st["level"] = at[0]["level_db"]
    good = [r for r in at if not r.get("interference") and not r.get("silent_input")]
    st["no_verdict"] = sorted({round(float(r["freq_hz"]), 1) for r in at if r.get("interference") or r.get("silent_input")})
    by_f: dict = {}
    for r in good:
        by_f.setdefault(round(float(r["freq_hz"]), 1), []).append(r)
    for f, g in sorted(by_f.items()):
        tones = [x["tone_db"] for x in g]
        st["rows"].append(dict(freq=f, planned=_planned(f, freqs) is not None, n=len(g), tones=tones, mean=sum(tones) / len(tones),
                               spread=(max(tones) - min(tones)) if len(g) > 1 else None,
                               snr_min=min(x.get("snr_db", 0.0) for x in g),
                               uneven=any(x.get("steady") is False for x in g), not_heard=any(not x.get("heard") for x in g)))
    st["played"] = [p for p in freqs if any(row["planned"] and _planned(row["freq"], [p]) for row in st["rows"])]
    st["missing"] = [p for p in freqs if p not in st["played"]]
    st["extra"] = [row["freq"] for row in st["rows"] if not row["planned"]]
    one = [row for row in st["rows"] if _planned(row["freq"], [1000.0]) and row["planned"]]
    if one:
        st["snr_1000"] = one[0]["snr_min"]
    spreads = [row["spread"] for row in st["rows"] if row["spread"] is not None]
    st["spread"] = max(spreads) if spreads else None
    st["spread_eff"] = max(st["spread"] or 0.0, MIN_SPREAD_DB)
    if st["rows"]:
        best = max(st["rows"], key=lambda row: row["mean"])
        st["best"] = best
        st["within"] = [row for row in st["rows"] if row["mean"] >= best["mean"] - st["spread_eff"]]
    if not st["missing"]:
        first: dict = {}                                     # the first reading of each requested frequency: a stable choice
        for r in good:
            p = _planned(r["freq_hz"], freqs)
            if p is not None and p not in first:
                first[p] = r["tone_db"]
        bp = max(first, key=lambda p: first[p])
        st["best_planned"] = bp
        k = freqs.index(bp)
        for a, b in ((freqs[k - 1] if k > 0 else None, bp), (bp, freqs[k + 1] if k + 1 < len(freqs) else None)):
            if a is None or b is None:
                continue
            if not any(a * (1 + FREQ_TOL) < r["freq_hz"] < b * (1 - FREQ_TOL) for r in good):
                st["refine_missing"].append((a, b))
    return st


def curve_missing(events: list[dict], plan: dict) -> list[str]:
    """What the request still lacks, in words for the orchestrator (an empty list: nothing is missing)."""
    st = curve_status(events, plan)
    fs = lambda xs: ", ".join(f"{x:g}" for x in xs)
    if st["level"] is None:
        return [f"none of the requested frequencies ({fs(st['freqs'])} Hz) has been played with a usable reading yet"]
    out = []
    if st["missing"]:
        out.append(f"still to play at the level of the curve ({st['level']:g} dBFS): {fs(st['missing'])} Hz")
    for a, b in st["refine_missing"]:
        out.append(f"the strongest requested frequency is {st['best_planned']:g} Hz; still to play at {st['level']:g} dBFS: "
                   f"one tone between {a:g} and {b:g} Hz")
    return out


def curve_goal(events: list[dict], plan: dict) -> str:
    """The next instruction for the hardware agent that closes what is missing (only used as a hint to the orchestrator)."""
    st = curve_status(events, plan)
    fs = lambda xs: ", ".join(f"{x:g}" for x in xs)
    if st["level"] is None:
        return (f"find one level at which a 1000 Hz tone has snr_db of at least {CLEAR_SNR_DB:g}, then play {fs(st['freqs'])} Hz at that "
                "same level, all in one reply, and report tone_db of each")
    if st["missing"]:
        return f"play {fs(st['missing'])} Hz at {st['level']:g} dBFS (the level of the curve) and report tone_db of each"
    parts = [f"one frequency between {a:g} and {b:g} Hz" for a, b in st["refine_missing"]]
    return f"play, at {st['level']:g} dBFS, " + " and ".join(parts) + ", and report tone_db of each"


def curve_facts(events: list[dict]) -> str:
    """Lines for the end of what the orchestrator reads (empty when the request has no frequency curve)."""
    plan = CTX.get("curve")
    if not plan:
        return ""
    st = curve_status(events, plan)
    fs = lambda xs: ", ".join(f"{x:g}" for x in xs) or "none"
    lines = ["[checklist counted by the program: the frequency curve of this request]"]
    if st["level"] is None:
        lines.append(f"No requested frequency ({fs(st['freqs'])} Hz) has a usable reading yet.")
    else:
        lines.append(f"Level of the curve: {st['level']:g} dBFS. Requested frequencies played at that level: {fs(st['played'])} Hz. "
                     f"Still missing: {fs(st['missing'])}"
                     + (f". Strongest of the requested frequencies: {st['best_planned']:g} Hz; tones between it and its neighbours still missing: "
                        + (", ".join(f"{a:g}-{b:g} Hz" for a, b in st["refine_missing"]) or "none") if st["best_planned"] else "") + ".")
    miss = curve_missing(events, plan)
    lines.append("Nothing is missing: you may finish." if not miss else
                 "finish is NOT allowed yet (the program will refuse it). Send hardware to: " + curve_goal(events, plan) + ".")
    return "\n".join(lines)


def curve_summary_md(events: list[dict], plan: dict, final_text: str = "") -> list[str]:
    """The frequency curve as the program reads it from the readings: every reading, the repeats, the strongest frequency and whether the
    readings can tell the strongest frequency from the others. This does not depend on what a model wrote."""
    st = curve_status(events, plan)
    fs = lambda xs: ", ".join(f"{x:g}" for x in xs) or "none"
    if st["level"] is None:
        return ["None of the requested frequencies has a usable reading, so there is no curve."]
    out = [f"Level of the curve: {st['level']:g} dBFS (the level at which most of the requested frequencies were played). tone_db is what the "
           "microphone measured; the table lists every reading at that level.", "",
           "| frequency | tone_db of each reading | mean | lowest snr_db | notes |", "|---|---|---|---|---|"]
    for row in st["rows"]:
        notes = []
        if row["uneven"]:
            notes.append("tone uneven, the level may read low; " + ("repeated" if row["n"] > 1 else "NOT repeated"))
        if row["not_heard"]:
            notes.append("not heard (snr_db below 10)")
        out.append(f"| {row['freq']:g} Hz{'' if row['planned'] else ' (extra)'} | {', '.join(f'{t:g}' for t in row['tones'])} | "
                   f"{row['mean']:.1f} | {row['snr_min']:g} | {'; '.join(notes) or '-'} |")
    out.append("")
    out.append(f"Requested frequencies played at this level: {fs(st['played'])} Hz. Not played: {fs(st['missing'])}"
               + (" (the request is NOT complete)." if st["missing"] else "."))
    if st["no_verdict"]:
        out.append(f"Readings with NO VERDICT (the room changed while the tone played) or a silent microphone, not used: {fs(st['no_verdict'])} Hz.")
    if st["snr_1000"] is not None and st["snr_1000"] < CLEAR_SNR_DB:
        out.append(f"Warning: 1000 Hz was only {st['snr_1000']:g} dB above the noise at this level; the request asked for at least {CLEAR_SNR_DB:g} dB.")
    if st["spread"] is not None:
        out.append(f"Largest difference between repeated readings of the same tone: {st['spread']:.1f} dB. Differences between frequencies "
                   f"smaller than {st['spread_eff']:.1f} dB are not meaningful.")
    else:
        out.append(f"No tone was repeated; differences between frequencies smaller than about {MIN_SPREAD_DB:g} dB are not meaningful "
                   "(two plays of the same tone differ by about that much).")
    best, within = st["best"], st["within"]
    out.append(f"Strongest reading: {best['freq']:g} Hz ({best['mean']:.1f} dB).")
    if len(within) == 1:
        out.append(f"Verdict: the response is strongest at {best['freq']:g} Hz; every other frequency played reads more than "
                   f"{st['spread_eff']:.1f} dB lower. The position is known only as finely as the frequencies played.")
    else:
        idx = [st["rows"].index(w) for w in within]
        near = idx == list(range(idx[0], idx[0] + len(idx)))
        out.append(f"Verdict: NOT RESOLVED. These frequencies read within {st['spread_eff']:.1f} dB of the strongest one: "
                   f"{fs([w['freq'] for w in within])} Hz; the readings do not separate them. "
                   + (f"The strongest response lies somewhere between {within[0]['freq']:g} and {within[-1]['freq']:g} Hz."
                      if near else "They are not neighbours: the readings are too uncertain to place the maximum."))
    if st["best_planned"] is not None:
        between = [r for r in st["rows"] if not r["planned"]]
        out.append(f"The strongest of the requested frequencies is {st['best_planned']:g} Hz. Tones played between it and its neighbours: "
                   f"{fs([r['freq'] for r in between])}"
                   + (f"; still missing between {', '.join(f'{a:g} and {b:g}' for a, b in st['refine_missing'])} Hz." if st["refine_missing"] else "."))
    if final_text.strip():
        unnamed = [r["freq"] for r in st["rows"] if not _named(final_text, r["freq"])]
        if unnamed:
            out.append(f"Played, but not named in the final message of the orchestrator: {fs(unnamed)} Hz. The table above is complete.")
    return out


def short_args(a: dict) -> str:
    return ", ".join(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}" for k, v in a.items())


def _truthy(v: Any) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() in ("true", "1", "yes")) or v == 1


# ----------------------------------------------------------------------------- the executor's tool (run_python)
RUN_PYTHON_SPEC = {"type": "function", "function": {
    "name": "run_python",
    "description": "Write the given Python code to a new file in the working directory and run it. Returns exit code, "
                   "stdout and stderr. For reading files and arithmetic only; it cannot reach the camera.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}, "purpose": {"type": "string"}},
                   "required": ["code"]}}}


def run_script(code: str, workdir: Path, timeout_s: int) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    idx = len(list(workdir.glob("exec_*.py"))) + 1
    path = workdir / f"exec_{idx:02d}.py"
    path.write_text(code, encoding="utf-8")
    proc = subprocess.Popen([sys.executable, path.name], cwd=workdir,
                            env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1"),
                            text=True, encoding="utf-8", errors="replace",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout_s)
        code_ = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except Exception:  # noqa: BLE001
            proc.kill()
        out, err = proc.communicate()
        err = (err or "") + f"\nTimeoutExpired: the script exceeded the {timeout_s} s limit and was killed."
        code_ = -9
    return dict(file=path.name, exit_code=code_, stdout=(out or "")[-STDOUT_TAIL:], stderr=(err or "")[-STDERR_TAIL:])


def format_run(res: dict) -> str:
    return (f"--- run_python result ({res['file']}) ---\nexit code: {res['exit_code']}\n"
            f"stdout:\n{res['stdout'] or '(empty)'}\nstderr:\n{res['stderr'] or '(empty)'}")


# ----------------------------------------------------------------------------- nodes
def make_orchestrator(model, max_decisions: int, workdir: Path):
    decide = DecisionCaller(model)

    def node(state: State) -> dict:
        used = state.get("decisions", 0)
        if used >= max_decisions:
            return {"next": "finish", "stop_reason": f"decision cap reached ({max_decisions} decisions)"}
        events = state.get("events", [])
        sys_prompt = fill(ORCHESTRATOR_SYSTEM).replace("{max_decisions}", str(max_decisions)).replace("{used}", str(used))
        messages = [SystemMessage(content=sys_prompt),
                    HumanMessage(content=render(state["dialogue"]) + "\n\n" + facts_block(state, max_decisions, workdir))]
        d = decide(messages)

        # Guards. finish: at least one reading from the camera (or a human refusal); hardware: camera calls left.
        refusal = None
        decisions_left = max_decisions - used - 1
        wants_finish = d.next == "finish"
        if wants_finish and not readings_of(events) and not state.get("refused") and decisions_left >= 2:
            refusal = ("finish was refused by the finish guard: no successful reading from the camera yet and the human "
                       "refused nothing. Decide again: hardware (or scientist / human if that is needed first).")
        elif d.next == "hardware" and state.get("n_calls", 0) >= MAX_CAMERA_CALLS:
            refusal = (f"hardware was refused: all {MAX_CAMERA_CALLS} tool calls of the run are used. "
                       "Decide again: finish with what was measured.")
        elif (wants_finish and CTX.get("curve") and decisions_left >= 2 and state.get("n_calls", 0) < MAX_CAMERA_CALLS
              and curve_missing(events, CTX["curve"])):
            refusal = ("finish was refused by the finish guard: the request is not complete yet: "
                       + "; ".join(curve_missing(events, CTX["curve"])) + ". Decide again: hardware, with the goal to "
                       + curve_goal(events, CTX["curve"]) + ".")
        if refusal:
            return {"dialogue": [{"role": "orchestrator",
                                  "text": f"[orchestrator decision refused]\n{refusal}\nreason: {d.reason}\n"
                                          f"original instruction: {d.instruction}"}],
                    "next": "orchestrator", "decisions": used + 1}

        if d.next == "finish":
            text = f"next: finish\ninstruction: {d.instruction}\nreason: {d.reason}"
            return {"dialogue": [{"role": "orchestrator", "text": text}], "next": "finish", "decisions": used + 1,
                    "final_text": d.instruction, "stop_reason": "finish"}

        if d.next == "human":
            hard = d.kind == "hard_decision"
            pending = dict(kind=d.kind, question=d.question.strip(), reason=d.reason,
                           guarded=d.guarded if hard else None, instruction=d.instruction if hard else "")
            text = f"next: human\nkind: {d.kind}\nquestion: {pending['question']}\n"
            if hard:
                text += f"step if accepted: {d.guarded}\ninstruction: {d.instruction}\n"
            text += f"reason: {d.reason}"
            return {"dialogue": [{"role": "orchestrator", "text": text}], "next": "human", "pending": pending,
                    "instruction": "", "decisions": used + 1}

        text = f"next: {d.next}\ninstruction: {d.instruction}\nreason: {d.reason}"
        return {"dialogue": [{"role": "orchestrator", "text": text}], "next": d.next,
                "instruction": d.instruction, "decisions": used + 1}

    return node


_ACCEPT = re.compile(r"^\W*(accept(ed)?|yes|y|yep|yeah|proceed|approve[d]?|ok(ay)?|go ahead|go|sure|do it)\b", re.I)
_REFUSE = re.compile(r"^\W*(refuse[d]?|no|n|nope|reject(ed)?|den(y|ied)|stop|cancel|abort|don'?t|do not)\b", re.I)


def parse_verdict(text: str) -> Optional[str]:
    """accept / refuse from the first word of the reply; None if it is neither (treated as a refusal)."""
    if _REFUSE.match(text):
        return "refuse"
    if _ACCEPT.match(text):
        return "accept"
    return None


def reply_text(reply: Any) -> str:
    if isinstance(reply, dict):
        reply = reply.get("text") or reply.get("reply") or reply.get("verdict") or ""
    return str(reply).strip()


def make_human(workdir: Path):
    """The pause. interrupt() stops the graph here (it needs a checkpointer and a thread) and returns the reply that the
    caller passes in Command(resume=...) when it resumes the same thread."""

    def node(state: State) -> dict:
        pend = state["pending"]
        hard = pend["kind"] == "hard_decision"
        payload = {"kind": pend["kind"], "question": pend["question"], "reason": pend.get("reason", "")}
        if hard:
            payload["step"] = {"role": pend["guarded"], "instruction": pend["instruction"]}
            payload["answer_with"] = "accept or refuse, optionally followed by a comment"
        elif pend["kind"] == "scene_action":
            payload["answer_with"] = 'when it is done: "done" and the real value, for example "done, muted"'
        else:
            payload["answer_with"] = "the reading to follow, in a sentence"

        reply = interrupt(payload)                      # <- paused here until Command(resume=...)
        text = reply_text(reply) or "(empty reply)"
        out: dict = {"dialogue": [{"role": "human", "text": text}], "pending": None, "instruction": ""}
        if not hard:                                    # ambiguous query / scene action: the orchestrator reads the reply next
            out["next"] = "orchestrator"
            return out

        verdict = parse_verdict(text)
        step = pend["guarded"]
        if verdict == "accept":
            note = f"step accepted by the human: {step} runs now (overwriting a stored reference is allowed in this one turn)."
            out.update(next="hardware", instruction=pend["instruction"], approved_overwrite=True)
        else:
            why = "refused" if verdict == "refuse" else "the reply was not a clear accept, so it counts as a refusal"
            note = (f"step {why}: {step} did NOT run, no camera command was sent for it, and nothing was overwritten "
                    f"(stored references: {', '.join(references_on_disk(workdir)) or 'none'}). The turn returns to the "
                    f"orchestrator.")
            out.update(next="orchestrator", refused=state.get("refused", 0) + 1, approved_overwrite=False)
        out["dialogue"].append({"role": "graph", "text": note})
        return out

    return node


def make_scientist(model):
    def node(state: State) -> dict:
        messages = [SystemMessage(content=fill(SCIENTIST_SYSTEM)),
                    HumanMessage(content=render(state["dialogue"])
                                 + f"\n\n[orchestrator instruction for you]\n{current_instruction(state)}"
                                 + "\n\nInstrument tool calls so far (camera and sound, from the tool results):\n"
                                 + events_table_md(state.get("events", [])))]
        return {"dialogue": [{"role": "scientist", "text": text_of(call(model, messages))}], "next": "orchestrator"}

    return node


def make_general(model):
    def node(state: State) -> dict:
        messages = [SystemMessage(content=fill(GENERAL_SYSTEM)),
                    HumanMessage(content=render(state["dialogue"])
                                 + f"\n\n[orchestrator instruction for you]\n{current_instruction(state)}")]
        return {"dialogue": [{"role": "general", "text": text_of(call(model, messages))}], "next": "orchestrator"}

    return node


def make_executor(model, workdir: Path, timeout_s: int = EXEC_TIMEOUT_S):
    model_with_tools = model.bind_tools([RUN_PYTHON_SPEC])
    flags = {"text_mode": False}

    def node(state: State) -> dict:
        messages = [SystemMessage(content=EXECUTOR_SYSTEM.format(timeout=timeout_s)),
                    HumanMessage(content=render(state["dialogue"])
                                 + f"\n\n[orchestrator instruction for you]\n{current_instruction(state)}")]
        ai = None
        if not flags["text_mode"]:
            try:
                ai = call(model_with_tools, messages)
            except Exception as e:  # noqa: BLE001
                if not tools_unsupported(e):
                    raise
                emit(f"    [note] tool calling unavailable ({str(e)[:100]}); asking for a fenced script")
                flags["text_mode"] = True
        if ai is None:
            ai = call(model, messages + [HumanMessage(content=EXEC_TEXT_NOTE)])
        tcs = list(getattr(ai, "tool_calls", None) or [])
        code = (tcs[0].get("args") or {}).get("code") if tcs else None
        via_tool = bool(code)
        if not code:
            code = extract_code(text_of(ai))
        if not code:
            return {"dialogue": [{"role": "executor", "text": text_of(ai) or "(executor produced no tool call and no text)"}],
                    "next": "orchestrator"}
        emit(f"  [executor] TOOL CALL run_python (purpose: {((tcs[0].get('args') or {}).get('purpose') if tcs else '') or '-'})")
        res = run_script(code, workdir, timeout_s)
        emit(f"  [executor] TOOL RESULT run_python exit code {res['exit_code']}")
        tool_result = format_run(res)
        if via_tool:
            follow = [ai] + [ToolMessage(content=tool_result if i == 0 else "not run: one script per turn",
                                         tool_call_id=tc.get("id", f"run{i}")) for i, tc in enumerate(tcs)]
        else:
            follow = [AIMessage(content=text_of(ai)), HumanMessage(content=tool_result)]
        try:                                              # the script already ran; a failed summary must not lose it
            summary = text_of(call(model, messages + follow))
        except Exception as e:  # noqa: BLE001
            summary = f"(summary unavailable: {type(e).__name__}: {str(e)[:120]})"
        return {"dialogue": [{"role": "executor", "text": tool_result + "\n\n" + (summary or "")}], "next": "orchestrator"}

    return node


# ---- the hardware agent
HOOKS: dict = {"after_tool": None, "before_tool": None}      # the runner may set a function(event) called after / before each camera call


def with_why(specs: list[dict]) -> list[dict]:
    """The tool descriptions for the model, each with the optional argument `why` (the reason for the call, shown live).
    The instrument server does not have this argument: do_call() takes it out before the call is sent."""
    out = []
    for sp in specs:
        sp = json.loads(json.dumps(sp))
        params = sp["function"].setdefault("parameters", {"type": "object", "properties": {}})
        params.setdefault("properties", {})["why"] = {
            "type": "string", "description": "one short sentence: what you have seen so far and why you chose this call"}
        out.append(sp)
    return out


def make_hardware(model, bridge, workdir: Path):
    """(c) The agent that holds the camera tools, and (d) the hardware turn: decide -> command -> measurement -> decide.

    `bridge` offers tool_specs() (descriptions for the model) and call(name, args) (sends the command to the MCP server
    and returns the answer as JSON text). Only this node ever sees them."""
    specs = with_why(bridge.tool_specs())
    names = {s["function"]["name"] for s in specs}
    model_with_tools = model.bind_tools(specs)
    flags = {"text_mode": False}      # sticky: the model has no tool calling, ask for a JSON list of calls

    def node(state: State) -> dict:
        calls_before = state.get("n_calls", 0)
        ctr = {"turn": 0, "total": calls_before}
        approved = bool(state.get("approved_overwrite"))
        has_controls = "set_control" in names
        has_audio = "play_tone" in names
        sys_prompt = (fill(HARDWARE_SYSTEM).replace("@@CONTROL_TOOLS@@", CONTROL_TOOLS if has_controls else "")
                      .replace("@@AUDIO_TOOLS@@", AUDIO_TOOLS if has_audio else "")
                      .replace("@@AUDIO_RULES@@", AUDIO_RULES if has_audio else "")
                      .replace("@@CONTROL_RULES@@", control_rules() if has_controls else "")
                      .replace("{per_turn}", str(MAX_CALLS_PER_TURN))
                      .replace("{left}", str(max(0, MAX_CAMERA_CALLS - calls_before)))
                      .replace("{facts}", "Stored references: " + (", ".join(references_on_disk(workdir)) or "(none)")
                               + f". The human has accepted an overwrite for this turn: {'YES' if approved else 'NO'}."))
        messages = [SystemMessage(content=sys_prompt),
                    HumanMessage(content=render(state["dialogue"])
                                 + f"\n\n[orchestrator instruction for you]\n{current_instruction(state)}")]
        blocks: list[str] = []
        new_events: list[dict] = []

        def not_sent(name: str, args: dict, why: str) -> str:
            """The program refused to send this call to the camera server; the agent and the orchestrator are told why."""
            res = {"ok": False, "error": "not run: " + why}
            blocks.append(f"--- camera tool call NOT SENT: {name}({short_args(args)}) ---\n{json.dumps(res)}")
            emit(f"  [hardware] NOT SENT {name}({short_args(args)}): {why}")
            return json.dumps(res)

        def do_call(name: str, args: dict) -> str:
            args = dict(args or {})
            why = " ".join(str(args.pop("why", "") or "").split())[:300]      # the model's own reason: shown, never sent to the instrument
            if name not in names:
                return not_sent(name, args, f"unknown tool {name!r}; tools: {', '.join(sorted(names))}")
            if ctr["turn"] >= MAX_CALLS_PER_TURN:
                return not_sent(name, args, f"at most {MAX_CALLS_PER_TURN} tool calls per turn")
            if ctr["total"] >= MAX_CAMERA_CALLS:
                return not_sent(name, args, f"all {MAX_CAMERA_CALLS} tool calls of the run are used")
            if name == "save_reference" and _truthy(args.get("overwrite")) and not approved:
                return not_sent(name, args, "overwriting a stored reference needs the human's accept, which has not been "
                                            "given; no command was sent to the camera. Report this; the orchestrator will "
                                            "ask the human.")
            ctr["turn"] += 1
            ctr["total"] += 1
            emit(f"  [hardware] TOOL CALL {ctr['total']}: {name}({short_args(args)})" + (f"\n             why: {why}" if why else ""))
            if HOOKS.get("before_tool"):
                try:
                    HOOKS["before_tool"](dict(n=ctr["total"], tool=name, args=args, why=why))
                except Exception as e:  # noqa: BLE001
                    emit(f"    [note] before_tool hook failed: {e}")
            text = bridge.call(name, args)
            try:
                res = json.loads(text)
                if not isinstance(res, dict):
                    res = {"ok": False, "error": f"unexpected answer: {text[:200]}"}
            except ValueError:
                res = {"ok": False, "error": f"unreadable answer: {text[:200]}"}
            ev = dict(n=ctr["total"], tool=name, args=args, ok=bool(res.get("ok")), result=res)
            if why:
                ev["why"] = why
            new_events.append(ev)
            blocks.append(f"--- camera tool call {ev['n']}: {name}({short_args(args)}) ---\n" + (f"why: {why}\n" if why else "")
                          + json.dumps(res))
            emit(f"  [hardware] TOOL RESULT {ctr['total']}: {short_result(name, res)}")
            if HOOKS.get("after_tool"):
                try:
                    HOOKS["after_tool"](ev)
                except Exception as e:  # noqa: BLE001
                    emit(f"    [note] after_tool hook failed: {e}")
            return json.dumps(res)

        def propose(follow: list, last: bool):
            """One model request. Returns (calls, text, ai_message, kind) with kind tool | markup | json | text."""
            if last:
                ai = call(model, messages + follow + [HumanMessage(content=FINAL_NOTE)])
                return [], text_of(ai), ai, "text"
            if not flags["text_mode"]:
                try:
                    ai = call(model_with_tools, messages + follow)
                except Exception as e:  # noqa: BLE001
                    if blocks or not tools_unsupported(e):
                        raise
                    emit(f"    [note] tool calling unavailable ({str(e)[:100]}); asking for a JSON list of calls")
                    flags["text_mode"] = True
                else:
                    tcs = [{"name": tc.get("name", ""), "args": tc.get("args") or {}, "id": tc.get("id") or f"call{i}"}
                           for i, tc in enumerate(getattr(ai, "tool_calls", None) or [])]
                    if tcs:
                        return tcs, text_of(ai), ai, "tool"
                    mk = markup_tool_calls(text_of(ai))
                    return (mk, text_of(ai), ai, "markup") if mk else ([], text_of(ai), ai, "text")
            note = TEXT_MODE_NOTE.replace("{per_turn}", str(MAX_CALLS_PER_TURN))
            ai = call(model, messages + follow + [HumanMessage(content=note)])
            reply = text_of(ai)
            try:
                obj = extract_json(reply)
            except ValueError:
                return [], reply, ai, "text"
            raw = obj.get("calls") if isinstance(obj.get("calls"), list) else []
            calls = [{"name": c.get("tool") or c.get("name") or "", "args": c.get("args") or {}, "id": f"json{i}"}
                     for i, c in enumerate(raw) if isinstance(c, dict)]
            return calls, str(obj.get("report") or ""), ai, "json"

        follow: list = []
        closing = ""
        nudged = False
        try:
            for rnd in range(MAX_ROUNDS):
                last = rnd == MAX_ROUNDS - 1
                calls, text, ai, kind = propose(follow, last)
                if not calls:
                    if not blocks and not nudged and not last and kind != "json":
                        nudged = True                       # nothing ran yet: ask once more for the tool call
                        follow += [ai, HumanMessage(content=NUDGE)]
                        continue
                    closing = text
                    break
                outs = [do_call(c["name"], c["args"]) for c in calls]
                if kind == "tool":
                    follow.append(ai)
                    follow += [ToolMessage(content=o, tool_call_id=c["id"]) for c, o in zip(calls, outs)]
                else:                                       # no tool_calls field: the results go back as a user turn
                    follow += [AIMessage(content=text_of(ai)), HumanMessage(content="Tool results:\n" + "\n\n".join(outs))]
        except Exception as e:  # noqa: BLE001
            if not blocks:
                raise
            closing = f"(model call failed after the camera calls ran: {type(e).__name__}: {str(e)[:120]})"

        text = "\n\n".join(blocks)
        if closing.strip():
            text = (text + "\n\n" if text else "") + closing.strip()
        out: dict = {"dialogue": [{"role": "hardware", "text": text or "(hardware agent made no tool call)"}],
                     "next": "orchestrator", "n_calls": ctr["total"], "approved_overwrite": False,
                     "hw_report": closing.strip()}
        if new_events:
            out["events"] = new_events
        return out

    return node


# ----------------------------------------------------------------------------- graph
def build_graph(models: dict, bridge, workdir: Path, timeout_s: int = EXEC_TIMEOUT_S, max_decisions: int = MAX_DECISIONS,
                checkpointer=None, curve: Optional[dict] = None):
    """One StateGraph, compiled WITH a checkpointer (interrupt() needs it). Only `hardware` is given the bridge.
    `curve` is the frequency-curve plan of the use case (CURVE_PLAN), or None: the program then also counts what the curve still lacks,
    refuses a finish before it is complete and writes the curve summary into the report."""
    workdir = Path(workdir)
    CTX["curve"] = curve
    CTX["controls"] = ""
    CTX["audio"] = "play_tone" in bridge.tool_names()
    for sp in bridge.tool_specs():
        if sp["function"]["name"] == "set_control" and "Verified controls:" in sp["function"]["description"]:
            CTX["controls"] = sp["function"]["description"].split("Verified controls:", 1)[1].strip()
    g = StateGraph(State)
    g.add_node("orchestrator", make_orchestrator(models["orchestrator"], max_decisions, workdir))
    g.add_node("scientist", make_scientist(models["scientist"]))
    g.add_node("general", make_general(models["general"]))
    g.add_node("executor", make_executor(models["executor"], workdir, timeout_s))
    g.add_node("hardware", make_hardware(models["hardware"], bridge, workdir))
    g.add_node("human", make_human(workdir))
    g.set_entry_point("orchestrator")

    def route(state: State) -> str:
        nxt = state.get("next", "finish")
        if nxt == "finish" or state.get("stop_reason"):
            return END
        return nxt

    g.add_conditional_edges(
        "orchestrator", route,
        {"scientist": "scientist", "general": "general", "executor": "executor", "hardware": "hardware",
         "human": "human", "orchestrator": "orchestrator", END: END})
    # after the pause: orchestrator (ambiguous query, refusal) or hardware (accepted step)
    g.add_conditional_edges("human", lambda s: s["next"], {"orchestrator": "orchestrator", "hardware": "hardware"})
    for role in ("scientist", "general", "executor", "hardware"):
        g.add_edge(role, "orchestrator")
    return g.compile(checkpointer=checkpointer or _Saver())


def initial_state(task: str) -> State:
    return {"dialogue": [{"role": "user", "text": task}], "events": [], "n_calls": 0, "decisions": 0, "refused": 0,
            "approved_overwrite": False}


def drive(graph, task: str, ask: Callable[[dict], str], thread_id: str = "hw3",
          on_update: Optional[Callable[[str, dict], None]] = None, recursion_limit: int = 100,
          max_pauses: int = 12) -> dict:
    """Run the graph on one thread. Every time it pauses at the human node, call ask(payload) for the reply and resume THE
    SAME thread with Command(resume=reply). Returns the final state."""
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    inp: Any = initial_state(task)
    for _ in range(max_pauses + 1):
        payload = None
        for update in graph.stream(inp, config=cfg, stream_mode="updates"):
            for node, upd in update.items():
                if node == "__interrupt__":
                    payload = upd[0].value
                elif on_update and isinstance(upd, dict):
                    on_update(node, upd)
        snap = graph.get_state(cfg)
        if not snap.next:
            return snap.values
        if payload is None:
            payload = snap.tasks[0].interrupts[0].value
        inp = Command(resume=ask(payload))
    raise RuntimeError(f"more than {max_pauses} pauses; stopping")


def build_task(request: str) -> str:
    return (f"Request: {request.strip()}\n\nEnvironment note for the system: the instrument (a camera, and the loudspeaker and "
            "microphone of the same device) is real and is reached only through the hardware agent. Pictures of the measured region "
            "are saved in the working directory; sound is never saved.")


# ----------------------------------------------------------------------------- reports
def transcript_md(state: State) -> str:
    lines = ["# Agent transcript\n"]
    for t in state.get("dialogue", []):
        lines += [f"## {t['role']}\n", t["text"].rstrip() + "\n"]
    lines += ["\n---\n", f"**decisions**: {state.get('decisions')}\n", f"**instrument tool calls**: {state.get('n_calls')}\n",
              f"**readings**: {len(readings_of(state.get('events', [])))}\n",
              f"**refused hard decisions**: {state.get('refused')}\n", f"**stop_reason**: {state.get('stop_reason')}\n",
              f"**model requests**: {STATS['model_calls']}\n"]
    return "\n".join(lines)


def final_report_md(state: State) -> str:
    """The run's ending: the orchestrator's message, then the camera calls and readings written by the program from the tool
    results (so the numbers are the instrument's)."""
    events = state.get("events", [])
    out = ["# Result of the run\n"]
    if state.get("final_text"):
        out += ["## Final message of the orchestrator\n", state["final_text"].strip(), ""]
    else:
        out += [f"The run stopped without a final message ({state.get('stop_reason')}).", ""]
    out += ["## Instrument tool calls and what the instrument answered (copied by the program from the tool results)\n",
            events_table_md(events), ""]
    n = len(readings_of(events))
    out.append(f"{len(events)} instrument tool call(s), {n} reading(s)." if n else
               "No reading was taken in this run: no measured value is reported.")
    snd = sound_summary_md(events)
    if snd:
        out += ["", "## Sound levels summarised by the program (from the readings above)\n"] + snd
    if CTX.get("curve"):
        out += ["", "## Frequency curve summarised by the program (from the readings above)\n"] + \
               curve_summary_md(events, CTX["curve"], state.get("final_text") or "")
    return "\n".join(out)


def evidence_md(dialogue: list[Turn], limit: int = 1500) -> str:
    """For each pause: the question the graph posed, the reply that was typed, and the messages that follow it."""
    clip = lambda s: s if len(s) <= limit else s[:limit] + " [...]"
    out = ["# Human-in-the-loop evidence\n"]
    k = 0
    for i, t in enumerate(dialogue):
        if t["role"] != "human":
            continue
        k += 1
        q = next((x for x in reversed(dialogue[:i]) if x["role"] == "orchestrator"), None)
        kind = re.search(r"^kind: (\S+)", q["text"], re.M) if q else None
        out.append(f"## Pause {k}: {kind.group(1) if kind else '?'}\n")
        out.append("**Question the graph posed** (orchestrator turn):\n\n```\n" + clip(q["text"] if q else "?") + "\n```\n")
        out.append("**Reply typed** (turn `human`):\n\n```\n" + t["text"] + "\n```\n")
        nxt = []
        for x in dialogue[i + 1:]:
            nxt.append(x)
            if x["role"] not in ("graph", "human"):
                break
        out.append("**Next message(s) in the dialogue**:\n")
        for x in nxt:
            out.append(f"`[{x['role']}]`\n\n```\n{clip(x['text'])}\n```\n")
    if k == 0:
        out.append("The graph never paused.\n")
    return "\n".join(out)
