"""Homework 2, Task 1: the Task 6 four-role LangGraph system plus a human-in-the-loop stop.

Roles (nodes of one StateGraph): orchestrator, scientist, general, executor  (Homework 1, Task 6)
New destination:                  human

The orchestrator can now send the turn to `human` in two situations, and it decides that with a model call:
  * ambiguous_query - the latest request has more than one reading and the readings would change the work
    or the reported number;
  * hard_decision   - the next step would run the coding tool, overwrite a result already on disk, or end the
    dialogue and report a measurement.

The stop is a LangGraph interrupt: the graph is compiled with a checkpointer, run on a thread, and the `human`
node calls interrupt(...). The graph stays paused until the same thread is resumed with Command(resume=reply).
The reply is appended to the dialogue with the role name `human`.
  * ambiguous query: the next orchestrator decision reads the reply and follows that reading;
  * hard decision : on `accept` the step runs; on `refuse` (or a reply that is not clearly an accept) the turn
    returns to the orchestrator and the step does not run.
"""
from __future__ import annotations

import json
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
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, field_validator

try:  # name changed between langgraph versions
    from langgraph.checkpoint.memory import InMemorySaver as _Saver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver as _Saver

MAX_DECISIONS = 8
EXEC_TIMEOUT_S = 300
STDOUT_TAIL = 6000
STDERR_TAIL = 4000
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# ----------------------------------------------------------------------------- prompts
ORCHESTRATOR_SYSTEM = """You are the ORCHESTRATOR of a system that solves a modelling task.
You are the only role the user talks to. You see the whole dialogue and decide who speaks next.
The dialogue can also hold turns from the HUMAN (a person who answers your questions) and [graph] notes. A [graph]
note is written by the program after a human reply; it is not a role.

Destinations you can choose:
- scientist: owns the theory and the plan (differential equation, network, state equation, training, free-run test,
  numbered coding steps). Send here whenever a plan is missing, wrong, or must change for scientific reasons.
- general: answers a high-level side question (what a record in the file is, what a term in the problem means,
  what a reported error is measuring). It does not write plans or code.
- executor: the only role that can write and run Python. Send here to implement the latest plan from the
  scientist, and also when a run failed (a traceback must go to the executor, not to general).
- human: pause the graph and ask the person a question. This is MANDATORY in exactly two situations and forbidden
  otherwise. Before EVERY decision run these two checks, in this order:
  CHECK 1 - ambiguous_query. Read the latest user request. Does it have more than one reasonable reading, where the
  readings would send the work to different roles or would change the number that gets reported (for example: which
  error is wanted - train or test, free run or one step; an explanation or a fitted number; the whole record or the
  excerpt)? If yes, and no [human] turn has answered it yet, you MUST choose human now, before scientist, general or
  executor. Put the question in `question`: name the readings and ask which to follow. Leave `guarded` empty. After the
  human has answered, follow that reading and never ask about it again. Do not ask about details that a reasonable
  default settles.
  CHECK 2 - hard_decision. Is the step you are about to choose (a) the executor, that is ANY run of the coding tool,
  (b) a step that would overwrite a file listed below, or (c) finish with a reported measurement? If yes, and the human
  has not already accepted this step (look for [human] and [graph] turns), you MUST choose human now instead of
  choosing executor or finish directly. Put the step in `guarded` ("executor" or "finish"), the instruction for that
  step in `instruction` (for "finish": the final message for the user), and in `question` state the step and ask
  whether to proceed. The step does NOT run until the human accepts. If the human refuses, the step is not run and the
  turn comes back to you: propose a different step (another file name, a smaller change, a different plan), or finish
  without a measurement and say that you stopped because the human refused. Exception: repairing a failed script of a
  run the human already accepted needs no new question unless the repair would overwrite a file.
  If neither check applies, do not choose human.
- finish: end the run. Allowed only after an executor turn shows a test RMSE printed by a script that actually ran,
  or after the human refused a step that was needed for the measurement (then say plainly that nothing was measured
  and give no numbers). Put the final message for the user (what was built, which agent did what, the free-run RMSE
  numbers exactly as the executor reported them) in `instruction`.

Files on disk in the working directory:
{files}

You write no theory and no code. You have at most {max_decisions} decisions in total; {used} are already used.
Every choice of destination, human included, counts as one decision; a step that the human accepts runs without
costing another decision. Plan the budget: typically scientist -> executor -> (repair turns if a run failed) -> finish.
Return a decision with:
  next        scientist | general | executor | human | finish
  instruction what that role should do now, concrete and self-contained (for human + hard_decision: the guarded step;
              for finish: the final message; for human + ambiguous_query: empty)
  reason      ONE sentence: why this destination is next
  question    only for human: the question the person will read
  kind        only for human: ambiguous_query | hard_decision
  guarded     only for human + hard_decision: executor | finish"""

SCIENTIST_SYSTEM = """You are the SCIENTIST. You own the theory and the plan. You have no coding tool.
The dialogue can hold HUMAN turns: they are answers from the person; honour them (for example the reading of an
ambiguous request that the person chose).
From the problem and the dialogue, write:
1. the differential equation of the system with a short explanation of each coefficient;
2. the network you propose (inputs, scaling, depth, widths, activation, output scaling);
3. the state equation to integrate (state, integrator and step, how the measured input enters);
4. how to train (loss, optimizer, learning rate, number of steps, rollout/window length, curriculum);
5. the free-run test exactly as the problem defines it (start state, no reset, what the rollout may read, RMSE on
   train and test segments);
6. a numbered list of steps specific enough for someone to code without asking questions, including that the script
   must print two lines: TRAIN_RMSE=<float> and TEST_RMSE=<float> computed by that free run.
Do not claim that any number was measured unless an executor turn in the dialogue already printed that number.
If an executor reported a failure or a result, revise the plan only for scientific reasons and keep it consistent."""

GENERAL_SYSTEM = """You are the GENERAL AGENT. You answer a high-level question that sits beside the fit: what a record in
the file is, what a term in the problem means, what a reported error is measuring. You have no coding tool and you
do not write the fitting plan. If the instruction asks for a plan or code, say in one sentence that it belongs to the
scientist or the executor. The dialogue can hold HUMAN turns (answers from the person); honour them.
Be short and concrete. Do not invent measured numbers; quote only numbers that already appear in the dialogue."""

EXECUTOR_SYSTEM = """You are the EXECUTOR, the only role with a coding tool: run_python(code, purpose).
The tool writes your code to a NEW file in this run's working directory and runs it there with a {timeout} second limit,
returning the exit code, stdout and stderr. Call it exactly once per turn.

Implement the LATEST plan from the scientist in a new file. Do not change the scientist's differential equation or the
free-run test; if a run failed, repair the script and keep the plan. Requirements for the script:
- read the data with scipy.io.loadmat from the path in the environment variable SNLS_DATA (variables V1 = input u,
  V2 = output y); python, numpy, scipy, torch (CPU) and matplotlib are installed;
- write result files into the current directory (the working directory); never overwrite or delete a file that already
  exists there unless the instruction says so explicitly;
- it must finish well inside the time limit (aim for under {budget} seconds of compute: small network, few thousand
  optimizer steps at most; set torch.set_num_threads(os.cpu_count()));
- it must print exactly two result lines at the end: TRAIN_RMSE=<float> and TEST_RMSE=<float>, computed by one free run
  over the whole 4096-sample excerpt as the plan defines it (no reset at the test segment, y is never read after
  sample 0). Never print a number you did not compute.
After the tool returns, reply in 2-4 sentences: the exit code and the numbers exactly as printed, or the cause of the
failure. Never invent or round numbers beyond what was printed.
{previous}"""

TEXT_MODE_NOTE = (
    "The tool-calling interface is not available for this model. Reply with the COMPLETE script in ONE fenced "
    "```python block and nothing else; it will be written to a new file and run exactly as the tool would."
)
JSON_NOTE = (
    "Reply with ONE JSON object and nothing else (no markdown fence, no commentary) with exactly these keys: "
    '"next" (scientist|general|executor|human|finish), "instruction" (string), "reason" (one sentence), '
    '"question" (string or null), "kind" ("ambiguous_query"|"hard_decision"|null), "guarded" ("executor"|"finish"|null).'
)


# ----------------------------------------------------------------------------- state
class Turn(TypedDict):
    role: str
    text: str


class State(TypedDict, total=False):
    dialogue: Annotated[list[Turn], operator.add]
    next: str                      # where the graph goes after the current node
    instruction: str               # what the next specialist should do
    pending: Optional[dict]        # the human question that is open (kind, question, guarded step)
    release: bool                  # the human accepted `finish`: the orchestrator now emits the final message
    final_text: str
    decisions: int
    refused: int                   # how many hard decisions the human refused
    test_rmse: Optional[float]
    train_rmse: Optional[float]
    stop_reason: str


class Decision(BaseModel):
    next: Literal["scientist", "general", "executor", "human", "finish"] = Field(
        description="destination of the next turn, or finish"
    )
    instruction: str = Field(
        default="",
        description="what that role should do now; for finish the final message to the user; "
        "for human + hard_decision the instruction of the guarded step",
    )
    reason: str = Field(default="", description="one sentence: why this destination is next")
    question: Optional[str] = Field(default=None, description="only for human: the question for the person")
    kind: Optional[Literal["ambiguous_query", "hard_decision"]] = Field(
        default=None, description="only for human: why the person is asked"
    )
    guarded: Optional[Literal["executor", "finish"]] = Field(
        default=None, description="only for human + hard_decision: the step that runs after the person accepts"
    )

    @field_validator("instruction", "reason", mode="before")
    @classmethod
    def _none_to_empty(cls, v):
        return "" if v is None else v


# ----------------------------------------------------------------------------- model calls
RETRY_SLEEP = [10, 20, 40, 60]          # seconds between retries of a transient provider error
STATS = {"model_calls": 0}              # every request sent to a provider, retries included (free tier: 20/min, 50/day)


def _status(e: Exception):
    return getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)


def _daily_quota(e: Exception) -> bool:
    m = str(e).lower()
    return "per-day" in m or "per day" in m


def _is_transient(e: Exception) -> bool:
    """Worth retrying? With an HTTP status the status decides (a 400/404 from a model that lacks a feature must not
    be retried: every retry spends a request of the free quota). Without a status, look at the message."""
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
            print(f"    [retry] {type(e).__name__}: {str(e)[:140]} - waiting {wait}s", flush=True)
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
    """The script from a fenced block (used when the model prints code instead of calling the tool)."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    with_result = [b for b in blocks if "TEST_RMSE" in b]
    if with_result:
        return with_result[-1]
    long_ = [b for b in blocks if b.count("\n") >= 5]
    return max(long_, key=len) if long_ else None


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
        if d.guarded not in ("executor", "finish"):
            return "a hard_decision needs `guarded` = executor or finish (the step that runs after an accept)."
        if not d.instruction.strip():
            return "a hard_decision needs `instruction` (the instruction of the guarded step, or the final message)."
    return None


class DecisionCaller:
    """One model call that returns a Decision.

    First choice is the provider's structured output. Free OpenRouter models often do not offer it, so when it
    fails (error, or no parsable object) the caller switches, for the rest of the run, to a plain-text call that
    asks for one JSON object. A decision that cannot be executed (for example human without a question) is sent
    back to the model with the reason, up to `attempts` times.
    """

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
                if isinstance(e, RuntimeError) and _daily_quota(e.__cause__ or e):
                    raise
                print(f"    [note] structured output unavailable ({type(e).__name__}: {str(e)[:100]}); "
                      f"asking for JSON text from now on", flush=True)
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


_RMSE_RE = {k: re.compile(rf"^\s*{k}\s*=\s*([-+0-9.eE]+)\s*$", re.M) for k in ("TRAIN_RMSE", "TEST_RMSE")}


def parse_rmse(stdout: str) -> dict[str, float]:
    out = {}
    for key, rx in _RMSE_RE.items():
        m = rx.findall(stdout)
        if m:
            try:
                val = float(m[-1])
            except ValueError:
                continue
            if val == val and abs(val) != float("inf"):
                out[key] = val
    return out


def run_script(code: str, workdir: Path, data_path: str, timeout_s: int) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    idx = len(list(workdir.glob("exec_*.py"))) + 1
    path = workdir / f"exec_{idx:02d}.py"
    path.write_text(code)
    env = dict(os.environ, SNLS_DATA=str(data_path), PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        [sys.executable, path.name], cwd=workdir, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
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
    return (
        f"--- run_python result ({res['file']}) ---\nexit code: {res['exit_code']}\n"
        f"stdout:\n{res['stdout'] or '(empty)'}\nstderr:\n{res['stderr'] or '(empty)'}"
    )


_BOOKKEEPING = {"graph.mmd", "transcript.md", "state.json", "evidence.md"}


def n_exec_scripts(workdir: Path) -> int:
    return len(list(workdir.glob("exec_*.py")))


def list_files(workdir: Path, limit: int = 40) -> str:
    """What is on disk in the working directory (computed by the program, shown to the orchestrator)."""
    rows = []
    for p in sorted(workdir.rglob("*")) if workdir.exists() else []:
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(workdir).as_posix()
        if re.fullmatch(r"exec_\d+\.py", rel) or rel in _BOOKKEEPING:
            continue
        rows.append(f"- {rel} ({p.stat().st_size} bytes)")
    body = "\n".join(rows[:limit]) + (f"\n- ... and {len(rows) - limit} more" if len(rows) > limit else "")
    return f"{body or '(no result files yet)'}\n(executor scripts written so far: {n_exec_scripts(workdir)})"


_ACCEPT = re.compile(r"^\W*(accept(ed)?|yes|y|yep|yeah|proceed|approve[d]?|ok(ay)?|go ahead|go|sure|run it|do it)\b", re.I)
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


def current_instruction(state: State) -> str:
    if state.get("instruction"):
        return state["instruction"]
    for t in reversed(state["dialogue"]):
        if t["role"] == "orchestrator":
            return t["text"]
    return ""


# ----------------------------------------------------------------------------- nodes
def gate_facts(dialogue: list[Turn]) -> dict:
    """What the human has already answered / accepted, counted by the program from the dialogue (facts only)."""
    n_amb = a_exec = a_fin = n_run = 0
    last_kind = None
    for t in dialogue:
        if t["role"] == "orchestrator":
            m = re.search(r"^kind: (\S+)", t["text"], re.M)
            last_kind = m.group(1) if m else None
        elif t["role"] == "human" and last_kind == "ambiguous_query":
            n_amb += 1
        elif t["role"] == "graph":
            if "accepted by the human: executor" in t["text"]:
                a_exec += 1
            elif "accepted by the human: finish" in t["text"]:
                a_fin += 1
        elif t["role"] == "executor" and "--- run_python result" in t["text"]:
            n_run += 1
    return dict(n_amb=n_amb, a_exec=a_exec, a_fin=a_fin, n_run=n_run)


def gate_reminder(state: State) -> str:
    """Placed at the END of what the orchestrator reads (weaker models follow what is nearest to the answer).
    It states facts and repeats the two checks; the decision itself is still the model's."""
    f = gate_facts(state["dialogue"])
    return (
        "[checklist before you answer]\n"
        f"Facts counted by the program: the human has answered {f['n_amb']} ambiguity question(s); has accepted "
        f"{f['a_exec']} executor run(s) and {f['a_fin']} finish; has refused {state.get('refused', 0)} step(s); "
        f"{f['n_run']} script(s) have run so far.\n"
        "CHECK 1: is the latest user request ambiguous and still unanswered by the human? Then choose human "
        "(ambiguous_query).\n"
        "CHECK 2: are you about to choose executor (any run of the coding tool, or a step that overwrites a file on "
        "disk) or finish, without the human having accepted that step? Then choose human (hard_decision) instead, with "
        "that step in `guarded`."
    )


def make_orchestrator(model, max_decisions: int, workdir: Path):
    decide = DecisionCaller(model)

    def node(state: State) -> dict:
        # the human accepted `finish` on the previous pause: emit the final message, no model call
        if state.get("release"):
            return {
                "dialogue": [{"role": "orchestrator", "text": state.get("final_text", "")}],
                "next": "finish",
                "release": False,
                "stop_reason": "finish",
            }

        used = state.get("decisions", 0)
        if used >= max_decisions:
            return {"next": "finish", "stop_reason": f"decision cap reached ({max_decisions} decisions)"}

        sys_prompt = ORCHESTRATOR_SYSTEM.format(
            max_decisions=max_decisions, used=used, files=list_files(workdir))
        messages = [SystemMessage(content=sys_prompt),
                    HumanMessage(content=render(state["dialogue"]) + "\n\n" + gate_reminder(state))]
        d = decide(messages)

        # Finish guard (kept from Task 6): a measurement from a script that really ran, or a human refusal
        wants_finish = d.next == "finish" or (d.next == "human" and d.guarded == "finish")
        if wants_finish and state.get("test_rmse") is None and not state.get("refused"):
            text = (
                "[orchestrator decision refused]\n"
                "finish was refused by the finish guard (no TEST_RMSE from a successful executor run yet, "
                "and the human has refused nothing). Decide again: scientist, executor or human.\n"
                f"reason: {d.reason}\noriginal instruction: {d.instruction}"
            )
            return {"dialogue": [{"role": "orchestrator", "text": text}], "next": "orchestrator",
                    "decisions": used + 1}

        if d.next == "finish":
            return {"dialogue": [{"role": "orchestrator", "text": d.instruction}], "next": "finish",
                    "decisions": used + 1, "stop_reason": "finish"}

        if d.next == "human":
            hard = d.kind == "hard_decision"
            pending = dict(kind=d.kind, question=d.question.strip(), reason=d.reason,
                           guarded=d.guarded if hard else None, instruction=d.instruction if hard else "")
            text = f"next: human\nkind: {d.kind}\nquestion: {pending['question']}\n"
            if hard:
                text += f"step if accepted: {d.guarded}\ninstruction: {d.instruction}\n"
            text += f"reason: {d.reason}"
            return {"dialogue": [{"role": "orchestrator", "text": text}], "next": "human",
                    "pending": pending, "instruction": "", "decisions": used + 1}

        text = f"next: {d.next}\ninstruction: {d.instruction}\nreason: {d.reason}"
        return {"dialogue": [{"role": "orchestrator", "text": text}], "next": d.next,
                "instruction": d.instruction, "decisions": used + 1}

    return node


def make_human(workdir: Path):
    """The pause. interrupt() stops the graph here (it needs a checkpointer and a thread) and returns the reply
    that the caller passes in Command(resume=...) when it resumes the same thread."""

    def node(state: State) -> dict:
        pend = state["pending"]
        hard = pend["kind"] == "hard_decision"
        payload = {"kind": pend["kind"], "question": pend["question"], "reason": pend.get("reason", "")}
        if hard:
            payload["step"] = {"role": pend["guarded"], "instruction": pend["instruction"]}
            payload["answer_with"] = "accept or refuse, optionally followed by a comment"
        else:
            payload["answer_with"] = "the reading to follow, in a sentence"

        reply = interrupt(payload)                      # <- paused here until Command(resume=...)
        text = reply_text(reply) or "(empty reply)"
        out: dict = {"dialogue": [{"role": "human", "text": text}], "pending": None, "instruction": ""}

        if not hard:                                    # ambiguous query: the orchestrator reads the reading next
            out["next"] = "orchestrator"
            return out

        verdict = parse_verdict(text)
        step = pend["guarded"]
        if verdict == "accept":
            note = f"step accepted by the human: {step} runs now."
            if step == "executor":
                out.update(next="executor", instruction=pend["instruction"])
            else:                                       # finish: the orchestrator emits the final message
                out.update(next="orchestrator", release=True, final_text=pend["instruction"])
        else:
            why = "refused" if verdict == "refuse" else "the reply was not a clear accept, so it counts as a refusal"
            note = (f"step {why}: {step} did NOT run (executor scripts in the working directory: "
                    f"{n_exec_scripts(workdir)}). The turn returns to the orchestrator.")
            out.update(next="orchestrator", refused=state.get("refused", 0) + 1)
        out["dialogue"].append({"role": "graph", "text": note})
        return out

    return node


def make_scientist(model):
    def node(state: State) -> dict:
        messages = [
            SystemMessage(content=SCIENTIST_SYSTEM),
            HumanMessage(content=render(state["dialogue"])
                         + f"\n\n[orchestrator instruction for you]\n{current_instruction(state)}"),
        ]
        reply = text_of(call(model, messages))
        return {"dialogue": [{"role": "scientist", "text": reply}], "next": "orchestrator"}

    return node


def make_general(model):
    def node(state: State) -> dict:
        messages = [
            SystemMessage(content=GENERAL_SYSTEM),
            HumanMessage(content=render(state["dialogue"])
                         + f"\n\n[orchestrator instruction for you]\n{current_instruction(state)}"),
        ]
        reply = text_of(call(model, messages))
        return {"dialogue": [{"role": "general", "text": reply}], "next": "orchestrator"}

    return node


def make_executor(model, workdir: Path, data_path: str, timeout_s: int):
    def _run_python(code: str, purpose: str = "") -> str:
        return format_run(run_script(code, workdir, data_path, timeout_s))

    tool = StructuredTool.from_function(
        func=_run_python, name="run_python",
        description="Write the given Python code to a new file and execute it. Returns exit code, stdout and stderr.",
    )
    model_with_tools = model.bind_tools([tool])
    flags = {"text_mode": False}      # sticky: the model has no tool calling, ask for a fenced script instead

    def node(state: State) -> dict:
        previous = ""
        scripts = sorted(workdir.glob("exec_*.py"))
        prev_results = [t["text"][:1500] for t in state["dialogue"]
                        if t["role"] == "executor" and "--- run_python result" in t["text"]]
        if scripts and prev_results:
            previous = (f"Your previous script ({scripts[-1].name}):\n```python\n{scripts[-1].read_text()[:6000]}\n```\n"
                        "Previous executor turns (for repair context):\n" + "\n---\n".join(prev_results[-2:]))

        sys_prompt = EXECUTOR_SYSTEM.format(timeout=timeout_s, budget=max(60, timeout_s // 3), previous=previous)
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=render(state["dialogue"])
                         + f"\n\n[orchestrator instruction for you]\n{current_instruction(state)}"),
        ]

        ai_msg = None
        if not flags["text_mode"]:
            try:
                ai_msg = call(model_with_tools, messages)
            except Exception as e:  # noqa: BLE001
                if not tools_unsupported(e):
                    raise
                print(f"    [note] tool calling unavailable ({str(e)[:100]}); asking for a fenced script", flush=True)
                flags["text_mode"] = True
        if ai_msg is None:
            ai_msg = call(model, messages + [HumanMessage(content=TEXT_MODE_NOTE)])

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        code = (tool_calls[0].get("args") or {}).get("code") if tool_calls else None
        via_tool = bool(code)
        if not code:
            code = extract_code(text_of(ai_msg))
        if not code:
            reply = text_of(ai_msg) or "(executor produced no tool call and no text)"
            return {"dialogue": [{"role": "executor", "text": reply}], "next": "orchestrator"}

        res = run_script(code, workdir, data_path, timeout_s)
        tool_result = format_run(res)
        updates: dict[str, Any] = {}
        if res["exit_code"] == 0:
            rmses = parse_rmse(res["stdout"])
            if "TEST_RMSE" in rmses:
                updates["test_rmse"] = rmses["TEST_RMSE"]
            if "TRAIN_RMSE" in rmses:
                updates["train_rmse"] = rmses["TRAIN_RMSE"]

        if via_tool:
            follow = [ai_msg] + [
                ToolMessage(content=tool_result if i == 0 else "not run: one script per turn",
                            tool_call_id=tc.get("id", f"run{i}"))
                for i, tc in enumerate(tool_calls)]
        else:
            follow = [AIMessage(content=text_of(ai_msg)), HumanMessage(content=tool_result)]
        try:                                              # the script already ran; a failed summary must not lose it
            summary = text_of(call(model, messages + follow))
        except Exception as e:  # noqa: BLE001
            summary = f"(summary unavailable: {type(e).__name__}: {str(e)[:120]})"
        return {"dialogue": [{"role": "executor", "text": tool_result + "\n\n" + (summary or "")}],
                "next": "orchestrator", **updates}

    return node


# ----------------------------------------------------------------------------- graph
def build_graph(
    models: dict,
    workdir: Path,
    data_path: str,
    timeout_s: int = EXEC_TIMEOUT_S,
    max_decisions: int = MAX_DECISIONS,
    checkpointer=None,
):
    """One StateGraph, compiled WITH a checkpointer (interrupt() needs it)."""
    workdir = Path(workdir)
    g = StateGraph(State)
    g.add_node("orchestrator", make_orchestrator(models["orchestrator"], max_decisions, workdir))
    g.add_node("scientist", make_scientist(models["scientist"]))
    g.add_node("general", make_general(models["general"]))
    g.add_node("executor", make_executor(models["executor"], workdir, data_path, timeout_s))
    g.add_node("human", make_human(workdir))
    g.set_entry_point("orchestrator")

    def route(state: State) -> str:
        nxt = state.get("next", "finish")
        if nxt == "finish" or state.get("stop_reason"):
            return END
        return nxt

    g.add_conditional_edges(
        "orchestrator", route,
        {"scientist": "scientist", "general": "general", "executor": "executor", "human": "human",
         "orchestrator": "orchestrator", END: END},
    )
    # after the pause: orchestrator (ambiguous query, refusal, accepted finish) or executor (accepted step)
    g.add_conditional_edges("human", lambda s: s["next"], {"orchestrator": "orchestrator", "executor": "executor"})
    for role in ("scientist", "general", "executor"):
        g.add_edge(role, "orchestrator")
    return g.compile(checkpointer=checkpointer or _Saver())


def drive(graph, task: str, ask: Callable[[dict], str], thread_id: str = "hitl",
          on_update: Optional[Callable[[str, dict], None]] = None, recursion_limit: int = 100,
          max_pauses: int = 20) -> dict:
    """Run the graph on one thread. Every time it pauses at the human node, call ask(payload) for the reply and
    resume THE SAME thread with Command(resume=reply). Returns the final state."""
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    inp: Any = initial_state(task)
    for _ in range(max_pauses + 1):
        payload = None
        for update in graph.stream(inp, config=cfg, stream_mode="updates"):
            for node, upd in update.items():
                if node == "__interrupt__":
                    payload = upd[0].value
                elif on_update:
                    on_update(node, upd)
        snap = graph.get_state(cfg)
        if not snap.next:
            return snap.values
        if payload is None:
            payload = snap.tasks[0].interrupts[0].value
        inp = Command(resume=ask(payload))
    raise RuntimeError(f"more than {max_pauses} pauses; stopping")


# ----------------------------------------------------------------------------- models
def default_models(model_name: str | None = None, provider: str | None = None):
    """OpenRouter (OPENROUTER_API_KEY) first, then Google (GOOGLE_API_KEY), then Anthropic.

    For OpenRouter `model_name` may be a comma-separated list: the first is the model, the whole list is sent as
    OpenRouter's `models` fallback array (a fallback inside one request costs one request of the free quota).
    """
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
                default_headers={"HTTP-Referer": "https://github.com/", "X-Title": "hw2-task1-human-in-the-loop"},
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


def build_task(request: str, context: str = "", data_note: bool = True) -> str:
    parts = [f"Request: {request.strip()}"]
    if context.strip():
        parts.append("Background - the Silverbox work from Homework 1 (the Task 5 statement):\n" + context.strip())
    if data_note:
        parts.append("Environment note for the system: the data file is available to scripts through the "
                     "environment variable SNLS_DATA. Scripts run on CPU only, in the run's working directory.")
    return "\n\n".join(parts)


def initial_state(task: str) -> State:
    return {"dialogue": [{"role": "user", "text": task}], "decisions": 0, "refused": 0,
            "test_rmse": None, "train_rmse": None}


# ----------------------------------------------------------------------------- reports
def transcript_md(state: State) -> str:
    lines = ["# Agent transcript\n"]
    for t in state.get("dialogue", []):
        lines.append(f"## {t['role']}\n")
        lines.append(t["text"].rstrip() + "\n")
    lines += ["\n---\n", f"**decisions**: {state.get('decisions')}\n", f"**refused hard decisions**: {state.get('refused')}\n",
              f"**stop_reason**: {state.get('stop_reason')}\n", f"**TRAIN_RMSE**: {state.get('train_rmse')}\n",
              f"**TEST_RMSE**: {state.get('test_rmse')}\n"]
    return "\n".join(lines)


def evidence_md(dialogue: list[Turn], limit: int = 1500) -> str:
    """For each pause: the question the graph posed, the reply that was typed, and the messages that follow it
    (the graph note, then the next turn). This is what the homework asks to show in the PDF."""
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
