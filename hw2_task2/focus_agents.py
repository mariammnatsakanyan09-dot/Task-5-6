"""Homework 2, Task 2: the Task 6 four-role LangGraph system with a new tool, focus_search, on the executor.

Roles (nodes of one StateGraph, as in Homework 1, Task 6): orchestrator, scientist, general, executor.
New tool of the executor: focus_search(...) - one complete wavefront-shaping experiment on the simulated bench (bench.py).
The executor keeps run_python (the Task 6 tool) for reading files and arithmetic.

The user types one request ("Focus the beam.") and nothing else. The agent chooses the macropixel size, the region sizes
and the number of iterations itself, calls the tool, reads the enhancement the tool returns, and decides what to try
next. Every number that goes into the table comes from the tool result (the program records it); a number that a model
writes in prose is never recorded.
"""
from __future__ import annotations

import json
import math
import operator
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError, field_validator

try:  # name changed between langgraph versions
    from langgraph.checkpoint.memory import InMemorySaver as _Saver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver as _Saver

import bench

MAX_DECISIONS = 10          # decisions of the orchestrator (every choice of a destination counts one)
MAX_FOCUS_CALLS = 14        # focus_search calls in the whole run (rejected calls count too)
MAX_CALLS_PER_TURN = 4      # focus_search calls in one executor turn
MAX_ROUNDS = 3              # model requests in one executor turn that may contain tool calls
MIN_EXPERIMENTS = 3         # finish needs this many successful experiments (unless the decision budget is nearly gone)
EXEC_TIMEOUT_S = 120        # run_python limit
STDOUT_TAIL = 4000
STDERR_TAIL = 3000
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# ----------------------------------------------------------------------------- prompts
BENCH_MANUAL = f"""THE BENCH (simulated). Laser -> DMD -> scattering medium -> camera. The DMD is {bench.DMD_N} x {bench.DMD_N} mirrors
and works in amplitude only (each mirror is on or off). The camera is {bench.CAM_N} x {bench.CAM_N} pixels and records the
intensity |T E_in|^2 with shot noise and read noise, so a second shot of the same pattern is not a copy of the first.
The medium is fixed. The tool focus_search runs one complete search of the following kind:
  1. the DMD is divided into macropixels (squares of macropixel_size x macropixel_size mirrors, all switched together);
     N = ({bench.DMD_N} / macropixel_size)^2 macropixels; the search starts from a random on/off pattern;
  2. after each transmission the intensity is read in the FOCUS region (a square of focus_size camera pixels around
     focus_row, focus_col) and in the proximal BACKGROUND: a ring around the focus, starting bg_gap pixels outside it, bg_width
     pixels wide, split into n_bg_regions sectors;
  3. the enhancement is the mean intensity in the focus divided by the mean of the intensities of the background regions;
  4. n_flip macropixels are flipped, the camera is read again, the flip is kept if the enhancement went up and undone
     otherwise; this is repeated `iterations` times.
The tool returns the enhancement of the final pattern, measured again with fresh shots (the number seen during the search is
biased upward by noise), the same number for the starting pattern, the mean counts in the focus and in the background
before and after, and it saves the camera images. As a check it also gives the focus against the FAR FIELD (camera pixels
outside the ring, never used by the search): the search lowers the ring it measures, so that number shows how much of the
enhancement is a brighter focus. Valid values: macropixel_size in {bench.VALID_MACROPIXEL}; focus_size 1-9;
bg_gap 0-6; bg_width 1-12; n_bg_regions 1-16; iterations 10-{bench.MAX_ITERATIONS}; n_flip >= 1; n_avg (camera shots averaged
per reading) 1-64; the ring must fit on the camera. One experiment takes seconds, long ones (many iterations, many shots)
up to {int(bench.TIME_LIMIT_S)} s."""

ORCHESTRATOR_SYSTEM = """You are the ORCHESTRATOR of a system that focuses a laser beam through a scattering medium on a SIMULATED optical bench.
You are the only role the user talks to. You see the whole dialogue and decide who speaks next.
Nobody answers questions during the run: the user typed one request and left. Never ask the user for settings or
confirmation; the system chooses the settings itself.

Destinations you can choose:
- scientist: owns the physics and the experimental plan: which settings to try, why, and how to read the results.
  Has no tools. Send here when a plan is missing or the results call for a new idea.
- general: answers a short side question (what a setting means, what the enhancement measures). No plans, no tools.
- executor: the only role with tools: focus_search (runs ONE experiment on the bench and returns the enhancement) and
  run_python. Send here to run experiments. Your `instruction` says which experiments to run (the settings, or what
  to vary); the executor sets the arguments of the tool from it and may run several experiments in one turn.
- finish: end the run. Allowed only after at least {min_exp} experiments have run successfully (the program counts them
  and refuses an earlier finish). Put the final message for the user in `instruction`: what was done, the best enhancement
  EXACTLY as the table below shows it with its settings, how the enhancement depended on the settings you varied, and what
  you would try next. Write no number that is not in the table.

How to work: first the scientist writes a short campaign (3 to 6 experiments that vary the macropixel size, the region
sizes and the number of iterations); then the executor runs them; then read the table, and if something is clearly
limiting (for example the search had too few iterations for its number of macropixels, or the numbers disagree with the
plan) send the next set of experiments to the executor, or ask the scientist. Stop when you can say which settings
worked best and why.

Budget: at most {max_decisions} decisions in total ({used} used). At most {max_calls} focus_search calls in total
({calls_used} used). Plan it: typically scientist -> executor -> (executor again with refined settings) -> finish.

Return a decision with:
  next         scientist | general | executor | finish
  instruction  what that role should do now, concrete and self-contained (for finish: the final message)
  reason       ONE sentence: why this destination is next

""" + BENCH_MANUAL

SCIENTIST_SYSTEM = """You are the SCIENTIST. You own the physics and the experimental plan. You have no tools.

""" + BENCH_MANUAL + """

Facts you may use: the speckle grain on the camera is about 2-3 pixels wide, so a focus region of 1-3 pixels holds one or
a few grains. A larger number of macropixels N gives a higher possible enhancement (for binary amplitude of the order of
N/(2 pi) against an unchanged background) but each macropixel needs several trials, so the number of iterations has to
grow with N (a few to tens of iterations per macropixel). The enhancement is a ratio: the search can also lower the
background. The background should be taken from several regions so that it does not depend on one speckle grain, and the
ring should lie outside the focus with a small gap. Noise limits how small a gain the search can see; averaging shots
(n_avg) costs time. Repeating the same settings gives a different result each time (random start, random noise).

From the user's request and the dialogue write the plan: a numbered list of experiments, each with ALL arguments of
focus_search (macropixel_size, focus_size, bg_gap, bg_width, n_bg_regions, iterations, and n_flip, n_avg if you change
them), and one sentence on what each experiment is meant to show. Choose the values yourself. If results are already in
the dialogue, say what they show and revise the plan for scientific reasons. Never claim a number that is not in an
executor turn of the dialogue."""

GENERAL_SYSTEM = """You are the GENERAL AGENT. You answer a short side question about the bench: what a setting means, what the
enhancement measures, what the images show. You have no tools and you do not write the experimental plan. If the
instruction asks for a plan or for runs, say in one sentence that it belongs to the scientist or the executor.
Be short and concrete. Do not invent measured numbers; quote only numbers that appear in the dialogue.

""" + BENCH_MANUAL

EXECUTOR_SYSTEM = """You are the EXECUTOR, the only role with tools:
  focus_search(macropixel_size, focus_size, bg_gap, bg_width, n_bg_regions, iterations, [n_flip, n_avg, focus_row, focus_col])
      one complete experiment on the simulated bench; returns the enhancement and saves camera images.
  run_python(code, purpose)
      runs a short Python script in the working directory (reading experiments.json, arithmetic). It is NOT a way to
      run experiments: only focus_search results count as experiments, and nothing you compute in a script is an enhancement.

Implement the latest instruction of the orchestrator and the latest plan of the scientist: call focus_search with the
settings they ask for (if a value is left open, choose it yourself and keep it consistent with the plan). You may make up to
{per_turn} focus_search calls in this turn, in one reply or one after another; {left} focus_search calls are left in the
whole run. If a call is rejected, the tool says what to change: fix it and call again. When the experiments of this turn
are done, reply in 2-4 sentences: which experiments ran and the enhancement of each EXACTLY as the tool printed it. Never invent,
round or recompute an enhancement.

""" + BENCH_MANUAL + """

Experiments so far (written by the program from the tool results):
{table}"""

TEXT_MODE_NOTE = (
    "The tool-calling interface is not available for this model. Reply with ONE JSON object and nothing else: "
    '{"calls": [{"macropixel_size": 8, "focus_size": 3, "bg_gap": 2, "bg_width": 6, "n_bg_regions": 8, "iterations": 3000}, ...]} '
    "- one entry per experiment to run (the keys are the arguments of focus_search; n_flip, n_avg, focus_row, focus_col are "
    "optional), at most {per_turn} entries. Each entry will be run exactly as the tool would run it."
)
NUDGE = ("You answered without calling a tool. The instruction asks for experiments: call focus_search now with the "
         "settings of the plan.")
JSON_NOTE = (
    "Reply with ONE JSON object and nothing else (no markdown fence, no commentary) with exactly these keys: "
    '"next" (scientist|general|executor|finish), "instruction" (string), "reason" (one sentence).'
)


# ----------------------------------------------------------------------------- state
class Turn(TypedDict):
    role: str
    text: str


class State(TypedDict, total=False):
    dialogue: Annotated[list[Turn], operator.add]
    experiments: Annotated[list[dict], operator.add]     # results of the successful focus_search calls, as the tool returned them
    n_calls: int                                         # focus_search calls so far, rejected ones included
    next: str
    instruction: str
    decisions: int
    final_text: str
    stop_reason: str


class Decision(BaseModel):
    next: Literal["scientist", "general", "executor", "finish"] = Field(description="destination of the next turn, or finish")
    instruction: str = Field(default="", description="what that role should do now; for finish the final message to the user")
    reason: str = Field(default="", description="one sentence: why this destination is next")

    @field_validator("instruction", "reason", mode="before")
    @classmethod
    def _none_to_empty(cls, v):
        return "" if v is None else v


class FocusSearchArgs(BaseModel):
    """Arguments of focus_search. The first six have no default: the agent has to choose them."""
    macropixel_size: int = Field(description=f"side of a macropixel in DMD pixels, one of {bench.VALID_MACROPIXEL}; "
                                             f"N = ({bench.DMD_N}/macropixel_size)^2 macropixels")
    focus_size: int = Field(description="side of the square focus region, in camera pixels (1-9)")
    bg_gap: int = Field(description="gap between the focus region and the background ring, in camera pixels (0-6)")
    bg_width: int = Field(description="width of the background ring, in camera pixels (1-12)")
    n_bg_regions: int = Field(description="number of sector regions the ring is split into (1-16)")
    iterations: int = Field(description=f"number of trials: flip, read the camera, keep or undo (10-{bench.MAX_ITERATIONS})")
    n_flip: int = Field(default=1, description="macropixels flipped in one trial (default 1)")
    n_avg: int = Field(default=1, description="camera shots averaged in one reading (default 1, at most 64)")
    focus_row: int = Field(default=20, description=f"row of the focus centre on the {bench.CAM_N}x{bench.CAM_N} camera (default 20)")
    focus_col: int = Field(default=20, description="column of the focus centre (default 20)")


FOCUS_TOOL_DESCRIPTION = (
    "Run ONE complete focusing experiment on the simulated bench (laser -> DMD, amplitude only -> scattering medium -> camera) "
    "with the given settings, and return the enhancement (focus intensity / mean background intensity) of the final pattern, "
    "measured again with fresh shots, plus the same number for the starting random pattern, the mean counts, and the name of "
    "the saved camera image. A rejected call says which setting to change.")


# ----------------------------------------------------------------------------- model calls
RETRY_SLEEP = [10, 20, 40, 60]          # seconds between retries of a transient provider error
STATS = {"model_calls": 0}              # every request sent to a provider, retries included (free tier: 20/min, 50/day)


def emit(msg: str = "") -> None:
    """Live line for the terminal (the runner shows the run as it happens)."""
    print(msg, flush=True)


def _status(e: Exception):
    return getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)


def _daily_quota(e: Exception) -> bool:
    m = str(e).lower()
    return "per-day" in m or "per day" in m


def _is_transient(e: Exception) -> bool:
    """Worth retrying? With an HTTP status the status decides (a 400/404 from a model that lacks a feature must not be
    retried: every retry spends a request of the free quota). Without a status, look at the message."""
    if _daily_quota(e):
        return False
    s = _status(e)
    if s is not None:
        return s in (408, 409, 425, 429, 500, 502, 503, 504)
    if isinstance(e, TypeError) and "nonetype" in str(e).lower():   # provider returned an error body, no choices
        return True
    m = str(e).lower()
    return any(k in m for k in ("rate limit", "rate-limit", "timed out", "timeout", "overloaded",
                                "temporarily", "connection error", "provider returned error"))


def call(model, messages):
    """model.invoke with retries on rate limits / provider hiccups (OpenRouter free models have both)."""
    for attempt in range(len(RETRY_SLEEP) + 1):
        STATS["model_calls"] += 1
        try:
            return model.invoke(messages)
        except Exception as e:  # noqa: BLE001
            if _daily_quota(e):
                raise RuntimeError(
                    "the provider says the DAILY quota is used up (OpenRouter free models: 50 requests/day, "
                    "1000/day after buying 10 credits). Try again tomorrow or add credits. Original error: "
                    f"{str(e)[:300]}") from e
            if attempt == len(RETRY_SLEEP) or not _is_transient(e):
                raise
            wait = RETRY_SLEEP[attempt]
            emit(f"    [retry] {type(e).__name__}: {str(e)[:140]} - waiting {wait}s")
            time.sleep(wait)


def tools_unsupported(e: Exception) -> bool:
    m = str(e).lower()
    return ("tool" in m or "function" in m) and any(
        k in m for k in ("support", "endpoint", "unsupported", "not available", "not allowed", "invalid"))


# ----------------------------------------------------------------------------- helpers
def render(dialogue: list[Turn]) -> str:
    return "\n\n".join(f"[{t['role']}]\n{t['text']}" for t in dialogue)


def text_of(msg: Any) -> str:
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    parts = []
    for b in c or []:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts)


def as_decision(obj: Any) -> Decision:
    if isinstance(obj, Decision):
        return obj
    if isinstance(obj, dict):
        return Decision(**obj)
    raise ValueError(f"orchestrator model returned no decision: {obj!r}")


def extract_json(text: str) -> dict:
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object in the model reply")


def extract_code(text: str) -> Optional[str]:
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    long_ = [b for b in blocks if b.count("\n") >= 2]
    return max(long_, key=len) if long_ else None


_FN_RE = re.compile(r"<function=([\w.\-]+)>(.*?)(?:</function>|$)", re.S)
_PARAM_RE = re.compile(r"<parameter=([\w.\-]+)>\s*(.*?)\s*(?:</parameter>|(?=<parameter=)|$)", re.S)


def markup_tool_calls(text: str) -> list[dict]:
    """Some models print their tool call as text (<function=NAME><parameter=KEY>VALUE</parameter></function>) instead of
    returning it in the tool_calls field. Read those calls too (values that look like integers become integers)."""
    out = []
    for i, m in enumerate(_FN_RE.finditer(text or "")):
        args = {}
        for k, v in _PARAM_RE.findall(m.group(2)):
            args[k] = int(v) if re.fullmatch(r"-?\d+", v.strip()) else v
        out.append({"name": m.group(1), "args": args, "id": f"markup{i}"})
    return out


PACE = {"s": 0.0}     # seconds to wait after each progress line of an experiment; 0 normally, >0 only to make a video readable
HOOKS: dict = {"after_experiment": None}     # the runner can set a function(result) that is called after each experiment


def current_instruction(state: State) -> str:
    if state.get("instruction"):
        return state["instruction"]
    for t in reversed(state["dialogue"]):
        if t["role"] == "orchestrator":
            return t["text"]
    return ""


class DecisionCaller:
    """One model call that returns a Decision. First choice is the provider's structured output; when it fails (free
    OpenRouter models often have none) the caller switches, for the rest of the run, to plain text with one JSON object."""

    def __init__(self, model):
        self.model, self.structured, self.use_text = model, None, False

    def __call__(self, messages) -> Decision:
        if not self.use_text:
            try:
                if self.structured is None:
                    self.structured = self.model.with_structured_output(Decision)
                return as_decision(call(self.structured, messages))
            except Exception as e:  # noqa: BLE001
                if isinstance(e, RuntimeError) and _daily_quota(e.__cause__ or e):
                    raise
                emit(f"    [note] structured output unavailable ({type(e).__name__}: {str(e)[:100]}); "
                     f"asking for JSON text from now on")
                self.use_text = True
        reply = text_of(call(self.model, messages + [HumanMessage(content=JSON_NOTE)]))
        return as_decision(extract_json(reply))


# ----------------------------------------------------------------------------- the experiments table (written by the program)
SETTING_KEYS = ("macropixel_size", "focus_size", "bg_gap", "bg_width", "n_bg_regions", "iterations", "n_flip", "n_avg")


def fmt_settings(s: dict) -> str:
    return (f"macropixel_size={s['macropixel_size']} focus_size={s['focus_size']} bg_gap={s['bg_gap']} "
            f"bg_width={s['bg_width']} n_bg_regions={s['n_bg_regions']} iterations={s['iterations']} "
            f"n_flip={s['n_flip']} n_avg={s['n_avg']} focus=({s['focus_row']},{s['focus_col']}) seed={s.get('seed')}")


def fmt_far(e: dict) -> str:
    v = e.get("enhancement_vs_far_field")
    return "-" if v is None else f"{v:.1f}"


def experiments_table_md(exps: list[dict]) -> str:
    if not exps:
        return "(no successful experiment yet)"
    rows = ["| # | macropixel (N) | focus px | ring gap / width / regions | iterations | flip | avg | enhancement | start | vs far field | focus x | background x | s |",
            "|--:|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for e in exps:
        s = e["settings"]
        rows.append(
            f"| {e['experiment']} | {s['macropixel_size']} ({e['n_macropixels']}) | {s['focus_size']} | "
            f"{s['bg_gap']} / {s['bg_width']} / {s['n_bg_regions']} | {s['iterations']} | {s['n_flip']} | {s['n_avg']} | "
            f"**{e['enhancement']:.1f}** | {e['enhancement_initial']:.2f} | {fmt_far(e)} | {e['focus_gain']:.1f} | {e['background_ratio']:.2f} | "
            f"{e['seconds']:.1f} |")
    return "\n".join(rows)


def best_experiment(exps: list[dict]) -> Optional[dict]:
    return max(exps, key=lambda e: e["enhancement"]) if exps else None


def facts_block(state: State, max_decisions: int) -> str:
    """Placed at the END of what the orchestrator reads: counts and the table, written by the program."""
    exps = state.get("experiments", [])
    best = best_experiment(exps)
    head = (f"[facts counted by the program]\ndecisions used: {state.get('decisions', 0)} of {max_decisions}; focus_search "
            f"calls: {state.get('n_calls', 0)} of {MAX_FOCUS_CALLS} (successful experiments: {len(exps)}; finish needs "
            f"{MIN_EXPERIMENTS}).")
    if best:
        head += f" Best enhancement so far: {best['enhancement']:.1f} in experiment {best['experiment']}."
    if not exps:
        head += "\nNo experiment has run yet; only the executor can run one."
    elif len(exps) < MIN_EXPERIMENTS:
        head += (f"\n{MIN_EXPERIMENTS - len(exps)} more successful experiment(s) are needed before finish is allowed; "
                 "only the executor can run them.")
    else:
        head += "\nEnough experiments have run for finish to be allowed; finish when you can name the best settings, otherwise refine."
    return head + "\nExperiments so far (from the tool results):\n" + experiments_table_md(exps)


def format_result(res: dict) -> str:
    """The tool result as the executor (and everybody after it) reads it."""
    if not res.get("ok"):
        return f"--- focus_search REJECTED ---\nerror: {res.get('error')}\nsettings asked: {json.dumps(res.get('settings'))}"
    s = res["settings"]
    return (
        f"--- focus_search result (experiment {res['experiment']}) ---\n"
        f"settings: {fmt_settings(s)}\n"
        f"enhancement: {res['enhancement']}   (starting random pattern: {res['enhancement_initial']}; "
        f"seen during the search, biased upward: {res['enhancement_end_of_search']})\n"
        f"enhancement against the far field (camera pixels the search never used): {res.get('enhancement_vs_far_field')} "
        f"(start {res.get('enhancement_vs_far_field_initial')})\n"
        f"focus mean counts: {res['focus_mean_counts']} (start {res['focus_mean_counts_initial']}, x{res['focus_gain']}); "
        f"background mean counts: {res['background_mean_counts']} (start {res['background_mean_counts_initial']}, x{res['background_ratio']})\n"
        f"N = {res['n_macropixels']} macropixels (textbook N/(2 pi) = {res['textbook_N_over_2pi']}); "
        f"{res['flips_accepted']} flips kept in {res['iterations_done']} iterations; {res['seconds']} s"
        + ("; STOPPED EARLY by the time limit" if res.get("stopped_early") else "")
        + f"\ncamera images: {res['figure']}")


# ----------------------------------------------------------------------------- run_python (the Task 6 tool)
def run_script(code: str, workdir: Path, timeout_s: int) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    idx = len(list(workdir.glob("exec_*.py"))) + 1
    path = workdir / f"exec_{idx:02d}.py"
    path.write_text(code)
    proc = subprocess.Popen([sys.executable, path.name], cwd=workdir, env=dict(os.environ, PYTHONUNBUFFERED="1"), text=True,
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
def make_orchestrator(model, max_decisions: int):
    decide = DecisionCaller(model)

    def node(state: State) -> dict:
        used = state.get("decisions", 0)
        if used >= max_decisions:
            return {"next": "finish", "stop_reason": f"decision cap reached ({max_decisions} decisions)"}
        exps = state.get("experiments", [])
        sys_prompt = ORCHESTRATOR_SYSTEM.format(min_exp=MIN_EXPERIMENTS, max_decisions=max_decisions, used=used,
                                                max_calls=MAX_FOCUS_CALLS, calls_used=state.get("n_calls", 0))
        messages = [SystemMessage(content=sys_prompt),
                    HumanMessage(content=render(state["dialogue"]) + "\n\n" + facts_block(state, max_decisions))]
        d = decide(messages)

        # Guards. finish: enough successful experiments (a number from the tool, not from prose); executor: calls left.
        refusal = None
        decisions_left = max_decisions - used - 1
        if d.next == "finish" and len(exps) < MIN_EXPERIMENTS and (len(exps) == 0 or decisions_left >= 2):
            refusal = (f"finish was refused by the finish guard: {len(exps)} successful experiment(s) so far, "
                       f"{MIN_EXPERIMENTS} are needed. Decide again: scientist or executor.")
        elif d.next == "executor" and state.get("n_calls", 0) >= MAX_FOCUS_CALLS:
            refusal = (f"executor was refused: all {MAX_FOCUS_CALLS} focus_search calls are used. Decide again: "
                       "scientist, general or finish.")
        if refusal:
            return {"dialogue": [{"role": "orchestrator",
                                  "text": f"[orchestrator decision refused]\n{refusal}\nreason: {d.reason}\n"
                                          f"original instruction: {d.instruction}"}],
                    "next": "orchestrator", "decisions": used + 1}

        text = f"next: {d.next}\ninstruction: {d.instruction}\nreason: {d.reason}"
        if d.next == "finish":
            return {"dialogue": [{"role": "orchestrator", "text": text}], "next": "finish", "decisions": used + 1,
                    "final_text": d.instruction, "stop_reason": "finish"}
        return {"dialogue": [{"role": "orchestrator", "text": text}], "next": d.next, "instruction": d.instruction,
                "decisions": used + 1}

    return node


def make_scientist(model):
    def node(state: State) -> dict:
        messages = [SystemMessage(content=SCIENTIST_SYSTEM),
                    HumanMessage(content=render(state["dialogue"])
                                 + f"\n\n[orchestrator instruction for you]\n{current_instruction(state)}"
                                 + "\n\nExperiments so far (from the tool results):\n" + experiments_table_md(state.get("experiments", [])))]
        return {"dialogue": [{"role": "scientist", "text": text_of(call(model, messages))}], "next": "orchestrator"}

    return node


def make_general(model):
    def node(state: State) -> dict:
        messages = [SystemMessage(content=GENERAL_SYSTEM),
                    HumanMessage(content=render(state["dialogue"])
                                 + f"\n\n[orchestrator instruction for you]\n{current_instruction(state)}")]
        return {"dialogue": [{"role": "general", "text": text_of(call(model, messages))}], "next": "orchestrator"}

    return node


def short_args(a: dict) -> str:
    return ", ".join(f"{k}={a[k]}" for k in a)


def make_executor(model, workdir: Path, timeout_s: int = EXEC_TIMEOUT_S):
    tools = [
        StructuredTool.from_function(func=lambda **kw: "", name="focus_search", description=FOCUS_TOOL_DESCRIPTION,
                                     args_schema=FocusSearchArgs),
        StructuredTool.from_function(func=lambda code, purpose="": "", name="run_python",
                                     description="Write the given Python code to a new file in the working directory and run it. "
                                                 "Returns exit code, stdout and stderr. Not for experiments."),
    ]
    model_with_tools = model.bind_tools(tools)
    flags = {"text_mode": False}      # sticky: the model has no tool calling, ask for a JSON list of experiments

    def run_focus(args: dict, counters: dict) -> tuple[str, Optional[dict]]:
        """One focus_search call. Returns (text for the dialogue, result dict if the experiment succeeded)."""
        counters["calls_turn"] += 1
        counters["calls_total"] += 1
        try:
            parsed = FocusSearchArgs(**(args or {}))
        except ValidationError as e:
            msg = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            emit(f"  [executor] focus_search REJECTED (arguments): {msg}")
            return f"--- focus_search REJECTED ---\nerror: bad arguments - {msg}\nsettings asked: {json.dumps(args)}", None
        kw = parsed.model_dump()
        emit(f"  [executor] TOOL CALL focus_search({short_args(kw)})")

        def progress(it, total, eta):
            emit(f"      ... iteration {it:>6d}/{total}   enhancement seen so far {eta:7.2f}")
            if PACE["s"] > 0:
                time.sleep(PACE["s"])

        res = bench.focus_search(workdir, progress=progress, **kw)
        text = format_result(res)
        if res.get("ok"):
            emit(f"  [executor] TOOL RESULT experiment {res['experiment']}: enhancement {res['enhancement']} "
                 f"(start {res['enhancement_initial']}), focus x{res['focus_gain']}, background x{res['background_ratio']}, "
                 f"{res['seconds']} s -> {res['figure']}")
            if HOOKS.get("after_experiment"):
                try:
                    HOOKS["after_experiment"](res)
                except Exception as e:  # noqa: BLE001
                    emit(f"    [note] after_experiment hook failed: {e}")
            return text, res
        emit(f"  [executor] TOOL RESULT rejected: {res.get('error')}")
        return text, None

    def run_py(args: dict) -> str:
        code = (args or {}).get("code") or ""
        if not code.strip():
            return "--- run_python REJECTED ---\nerror: no code given"
        emit("  [executor] TOOL CALL run_python (purpose: %s)" % ((args or {}).get("purpose") or "-")[:80])
        res = run_script(code, workdir, timeout_s)
        emit(f"  [executor] TOOL RESULT run_python exit code {res['exit_code']}")
        return format_run(res)

    def node(state: State) -> dict:
        exps_before = state.get("experiments", [])
        calls_total = state.get("n_calls", 0)
        counters = {"calls_turn": 0, "calls_total": calls_total}
        sys_prompt = EXECUTOR_SYSTEM.format(per_turn=MAX_CALLS_PER_TURN, left=max(0, MAX_FOCUS_CALLS - calls_total),
                                            table=experiments_table_md(exps_before))
        messages = [SystemMessage(content=sys_prompt),
                    HumanMessage(content=render(state["dialogue"])
                                 + f"\n\n[orchestrator instruction for you]\n{current_instruction(state)}")]
        blocks: list[str] = []
        new_exps: list[dict] = []
        closing = ""

        def do_call(name: str, args: dict) -> str:
            if name == "focus_search":
                if counters["calls_turn"] >= MAX_CALLS_PER_TURN:
                    return f"not run: at most {MAX_CALLS_PER_TURN} focus_search calls per turn"
                if counters["calls_total"] >= MAX_FOCUS_CALLS:
                    return f"not run: all {MAX_FOCUS_CALLS} focus_search calls of the run are used"
                text, res = run_focus(args, counters)
                if res is not None:
                    new_exps.append(res)
                blocks.append(text)
                return text
            if name == "run_python":
                text = run_py(args)
                blocks.append(text)
                return text
            return f"not run: unknown tool {name!r}"

        if not flags["text_mode"]:
            follow: list = []
            nudged = False
            try:
                for _ in range(MAX_ROUNDS):
                    ai = call(model_with_tools, messages + follow)
                    tcs = getattr(ai, "tool_calls", None) or []
                    as_text = False
                    if not tcs:
                        tcs, as_text = markup_tool_calls(text_of(ai)), True      # a call printed as text
                    if not tcs:
                        if not blocks and not nudged:           # nothing ran yet: ask once more for the tool call
                            nudged = True
                            follow += [ai, HumanMessage(content=NUDGE)]
                            continue
                        closing = text_of(ai)
                        break
                    outs = [do_call(tc.get("name", ""), tc.get("args") or {}) for tc in tcs]
                    if as_text:                                  # no tool_calls field: the results go back as a user turn
                        follow += [AIMessage(content=text_of(ai)),
                                   HumanMessage(content="Tool results:\n" + "\n\n".join(outs))]
                    else:
                        follow.append(ai)
                        follow += [ToolMessage(content=o, tool_call_id=tc.get("id") or f"call{len(follow)}_{i}")
                                   for i, (tc, o) in enumerate(zip(tcs, outs))]
            except Exception as e:  # noqa: BLE001
                if blocks or not tools_unsupported(e):
                    if not blocks:
                        raise
                    closing = f"(model call failed after the experiments ran: {type(e).__name__}: {str(e)[:120]})"
                else:
                    emit(f"    [note] tool calling unavailable ({str(e)[:100]}); asking for a JSON list of experiments")
                    flags["text_mode"] = True
        if flags["text_mode"] and not blocks:
            reply = text_of(call(model, messages + [HumanMessage(content=TEXT_MODE_NOTE.replace("{per_turn}", str(MAX_CALLS_PER_TURN)))]))
            calls: list = []
            try:
                obj = extract_json(reply)
                calls = obj.get("calls") if isinstance(obj.get("calls"), list) else [obj]
            except ValueError:
                code = extract_code(reply)
                if code:
                    do_call("run_python", {"code": code, "purpose": "text mode"})
            for c in calls:
                if isinstance(c, dict):
                    do_call("focus_search", c.get("args", c) if isinstance(c.get("args", c), dict) else c)
            if not blocks:
                closing = reply or "(executor produced no tool call and no text)"

        text = "\n\n".join(blocks)
        if closing.strip():
            text = (text + "\n\n" if text else "") + closing.strip()
        out: dict = {"dialogue": [{"role": "executor", "text": text or "(executor made no tool call)"}], "next": "orchestrator",
                     "n_calls": counters["calls_total"]}
        if new_exps:
            out["experiments"] = new_exps
        return out

    return node


# ----------------------------------------------------------------------------- graph
def build_graph(models: dict, workdir: Path, timeout_s: int = EXEC_TIMEOUT_S, max_decisions: int = MAX_DECISIONS,
                checkpointer=None):
    """The Task 6 graph: orchestrator <-> scientist / general / executor. The executor now has focus_search."""
    workdir = Path(workdir)
    g = StateGraph(State)
    g.add_node("orchestrator", make_orchestrator(models["orchestrator"], max_decisions))
    g.add_node("scientist", make_scientist(models["scientist"]))
    g.add_node("general", make_general(models["general"]))
    g.add_node("executor", make_executor(models["executor"], workdir, timeout_s))
    g.set_entry_point("orchestrator")

    def route(state: State) -> str:
        nxt = state.get("next", "finish")
        if nxt == "finish" or state.get("stop_reason"):
            return END
        return nxt

    g.add_conditional_edges("orchestrator", route, {"scientist": "scientist", "general": "general", "executor": "executor",
                                                    "orchestrator": "orchestrator", END: END})
    for role in ("scientist", "general", "executor"):
        g.add_edge(role, "orchestrator")
    return g.compile(checkpointer=checkpointer or _Saver())


def initial_state(task: str) -> State:
    return {"dialogue": [{"role": "user", "text": task}], "experiments": [], "n_calls": 0, "decisions": 0}


def run_agent(graph, task: str, thread_id: str = "focus", on_update: Optional[Callable[[str, dict], None]] = None,
              recursion_limit: int = 100) -> dict:
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    for update in graph.stream(initial_state(task), config=cfg, stream_mode="updates"):
        for node, upd in update.items():
            if on_update and isinstance(upd, dict):
                on_update(node, upd)
    return graph.get_state(cfg).values


# ----------------------------------------------------------------------------- models
def default_models(model_name: str | None = None, provider: str | None = None):
    """OpenRouter (OPENROUTER_API_KEY) first, then Google (GOOGLE_API_KEY), then Anthropic.
    For OpenRouter `model_name` may be a comma-separated list: the first is the model, the whole list is sent as
    OpenRouter's `models` fallback array (a fallback inside one request costs one request)."""
    name = model_name or os.environ.get("AGENT_MODEL")
    provider = provider or os.environ.get("AGENT_PROVIDER")
    if not provider:
        provider = ("openrouter" if os.environ.get("OPENROUTER_API_KEY") else
                    "google" if os.environ.get("GOOGLE_API_KEY") else "anthropic")

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        if not name:
            raise SystemExit("OpenRouter needs a model: pass --model <id> (for example an id ending in :free) "
                             "or set AGENT_MODEL. python list_openrouter_models.py shows the free ones.")
        ids = [m.strip() for m in name.split(",") if m.strip()]
        extra = {"models": ids} if len(ids) > 1 else {}
        base_url = os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_URL)   # override only for tests / proxies

        def make():
            return ChatOpenAI(
                model=ids[0], api_key=os.environ["OPENROUTER_API_KEY"], base_url=base_url,
                temperature=0.2, timeout=180, max_retries=0,   # retries are done in call(), where they are counted
                default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "hw2-task2-focusing-agent"},
                extra_body=extra or None,
            )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        def make():
            return ChatGoogleGenerativeAI(model=name or "gemini-3.8-flash", temperature=0.2,
                                          convert_system_message_to_human=True)
    else:
        from langchain_anthropic import ChatAnthropic

        def make():
            return ChatAnthropic(model=name or "claude-sonnet-4-5-20250929", temperature=0.2)

    return {role: make() for role in ("orchestrator", "scientist", "general", "executor")}


def build_task(request: str, context: str = "") -> str:
    parts = [f"Request: {request.strip()}"]
    if context.strip():
        parts.append("Background (the laboratory setup and the task):\n" + context.strip())
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- reports
def final_report_md(state: State) -> str:
    """What the run ends with: the orchestrator's message, then the table and the best experiment, written by the program
    from the tool results (so the numbers are the tool's)."""
    exps = state.get("experiments", [])
    best = best_experiment(exps)
    out = ["# Result of the run\n"]
    if state.get("final_text"):
        out += ["## Final message of the orchestrator\n", state["final_text"].strip(), ""]
    elif state.get("stop_reason") == "finish":
        out += ["The orchestrator finished with an empty final message.", ""]
    else:
        out += [f"The run stopped without a final message ({state.get('stop_reason')}).", ""]
    out += ["## Experiments run by the agent (numbers from the tool results)\n", experiments_table_md(exps), ""]
    if best:
        out += [f"**Best enhancement kept from the tool: {best['enhancement']:.1f}** (experiment {best['experiment']}: "
                f"{fmt_settings(best['settings'])}).", ""]
    else:
        out += ["No experiment succeeded; no enhancement was measured.", ""]
    return "\n".join(out)


def transcript_md(state: State) -> str:
    lines = ["# Agent transcript\n"]
    for t in state.get("dialogue", []):
        lines += [f"## {t['role']}\n", t["text"].rstrip() + "\n"]
    lines += ["\n---\n", f"**decisions**: {state.get('decisions')}\n", f"**focus_search calls**: {state.get('n_calls')}\n",
              f"**successful experiments**: {len(state.get('experiments', []))}\n", f"**stop_reason**: {state.get('stop_reason')}\n"]
    return "\n".join(lines)
