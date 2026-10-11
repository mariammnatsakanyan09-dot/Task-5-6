"""Task 6: a four-role agentic system built with LangGraph.

Roles (nodes of one StateGraph):  orchestrator, scientist, general, executor.
Shared state: the dialogue, a list of turns {"role": ..., "text": ...}.
"""
from __future__ import annotations

import json
import operator
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

MAX_DECISIONS = 8
EXEC_TIMEOUT_S = 300
STDOUT_TAIL = 6000
STDERR_TAIL = 4000

# ----------------------------------------------------------------------------- prompts
ORCHESTRATOR_SYSTEM = """You are the ORCHESTRATOR of a four-role system that solves a modelling task.
You are the only role the user talks to. You see the whole dialogue and decide who speaks next.

Roles you can hand the turn to:
- scientist: owns the theory and the plan (differential equation, network, state equation, training, free-run test,
  numbered coding steps). Send here whenever a plan is missing, wrong, or must change for scientific reasons.
- general: answers a high-level side question (what a record in the file is, what a term in the problem means,
  what a reported error is measuring). It does not write plans or code.
- executor: the only role that can write and run Python. Send here to implement the latest plan from the
  scientist, and also when a run failed (a traceback must go to the executor, not to general).
- finish: end the run. This is allowed only after an executor turn shows a test RMSE printed by a script that
  actually ran. When you finish, put the final message for the user (what was built, which agent did what, the
  free-run RMSE numbers exactly as the executor reported them) in `instruction`.

You write no theory and no code. You have at most {max_decisions} decisions in total; {used} are already used.
Plan the budget: typically scientist -> executor -> (repair executor turns if a run failed) -> finish.
Return a decision with: next (scientist | general | executor | finish), instruction (what that role should do now,
concrete and self-contained), reason (ONE sentence why this role is next)."""

SCIENTIST_SYSTEM = """You are the SCIENTIST. You own the theory and the plan. You have no coding tool.
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
scientist or the executor. Be short and concrete. Do not invent measured numbers; quote only numbers that already
appear in the dialogue."""

EXECUTOR_SYSTEM = """You are the EXECUTOR, the only role with a coding tool: run_python(code, purpose).
The tool writes your code to a NEW file in this run's working directory and runs it with a {timeout} second limit,
returning the exit code, stdout and stderr. Call it exactly once per turn.

Implement the LATEST plan from the scientist in a new file. Do not change the scientist's differential equation or the
free-run test; if a run failed, repair the script and keep the plan. Requirements for the script:
- read the data with scipy.io.loadmat from the path in the environment variable SNLS_DATA (variables V1 = input u,
  V2 = output y); python, numpy, scipy, torch (CPU) and matplotlib are installed;
- it must finish well inside the time limit (aim for under {budget} seconds of compute: small network, few thousand
  optimizer steps at most; set torch.set_num_threads(os.cpu_count()));
- it must print exactly two result lines at the end: TRAIN_RMSE=<float> and TEST_RMSE=<float>, computed by one free run
  over the whole 4096-sample excerpt as the plan defines it (no reset at the test segment, y is never read after
  sample 0). Never print a number you did not compute.
After the tool returns, reply in 2-4 sentences: the exit code and the numbers exactly as printed, or the cause of the
failure. Never invent or round numbers beyond what was printed.
{previous}"""


# ----------------------------------------------------------------------------- state
class Turn(TypedDict):
    role: str
    text: str


class State(TypedDict, total=False):
    dialogue: Annotated[list[Turn], operator.add]
    next: str
    decisions: int
    test_rmse: Optional[float]
    train_rmse: Optional[float]
    stop_reason: str


class Decision(BaseModel):
    next: Literal["scientist", "general", "executor", "finish"] = Field(
        description="role that speaks next, or finish"
    )
    instruction: str = Field(
        description="what that role should do now; for finish, the final message to the user"
    )
    reason: str = Field(description="one sentence: why this role is next")


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


_RMSE_RE = {
    k: re.compile(rf"^\s*{k}\s*=\s*([-+0-9.eE]+)\s*$", re.M)
    for k in ("TRAIN_RMSE", "TEST_RMSE")
}


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
        [sys.executable, path.name],
        cwd=workdir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout_s)
        code_ = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except Exception:
            proc.kill()
        out, err = proc.communicate()
        err = (err or "") + f"\nTimeoutExpired: the script exceeded the {timeout_s} s limit and was killed."
        code_ = -9
    return dict(
        file=path.name,
        exit_code=code_,
        stdout=(out or "")[-STDOUT_TAIL:],
        stderr=(err or "")[-STDERR_TAIL:],
    )


def format_run(res: dict) -> str:
    return (
        f"--- run_python result ({res['file']}) ---\nexit code: {res['exit_code']}\n"
        f"stdout:\n{res['stdout'] or '(empty)'}\nstderr:\n{res['stderr'] or '(empty)'}"
    )


# ----------------------------------------------------------------------------- nodes
def make_orchestrator(model, max_decisions: int):
    structured = model.with_structured_output(Decision)

    def node(state: State) -> dict:
        used = state.get("decisions", 0)
        if used >= max_decisions:
            return {
                "next": "finish",
                "stop_reason": f"decision cap reached ({max_decisions} decisions)",
            }

        sys_prompt = ORCHESTRATOR_SYSTEM.format(max_decisions=max_decisions, used=used)
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=render(state["dialogue"])),
        ]
        decision = as_decision(structured.invoke(messages))

        # Finish guard
        if decision.next == "finish":
            has_rmse = state.get("test_rmse") is not None
            if not has_rmse:
                roles_so_far = {t["role"] for t in state["dialogue"]}
                fallback = "scientist" if "scientist" not in roles_so_far else "executor"
                text = (
                    f"[orchestrator decision refused]\n"
                    f"next=finish was refused by the finish guard "
                    f"(no TEST_RMSE from a successful executor run yet).\n"
                    f"Routing to {fallback} instead.\n"
                    f"reason: {decision.reason}\n"
                    f"original instruction: {decision.instruction}"
                )
                return {
                    "dialogue": [{"role": "orchestrator", "text": text}],
                    "next": fallback,
                    "decisions": used + 1,
                }
            # accept finish
            return {
                "dialogue": [{"role": "orchestrator", "text": decision.instruction}],
                "next": "finish",
                "decisions": used + 1,
                "stop_reason": "finish",
            }

        text = (
            f"next: {decision.next}\n"
            f"instruction: {decision.instruction}\n"
            f"reason: {decision.reason}"
        )
        return {
            "dialogue": [{"role": "orchestrator", "text": text}],
            "next": decision.next,
            "decisions": used + 1,
        }

    return node


def make_scientist(model):
    def node(state: State) -> dict:
        instruction = ""
        for t in reversed(state["dialogue"]):
            if t["role"] == "orchestrator":
                instruction = t["text"]
                break
        messages = [
            SystemMessage(content=SCIENTIST_SYSTEM),
            HumanMessage(
                content=render(state["dialogue"])
                + f"\n\n[orchestrator instruction for you]\n{instruction}"
            ),
        ]
        reply = text_of(model.invoke(messages))
        return {
            "dialogue": [{"role": "scientist", "text": reply}],
            "next": "orchestrator",
        }

    return node


def make_general(model):
    def node(state: State) -> dict:
        instruction = ""
        for t in reversed(state["dialogue"]):
            if t["role"] == "orchestrator":
                instruction = t["text"]
                break
        messages = [
            SystemMessage(content=GENERAL_SYSTEM),
            HumanMessage(
                content=render(state["dialogue"])
                + f"\n\n[orchestrator instruction for you]\n{instruction}"
            ),
        ]
        reply = text_of(model.invoke(messages))
        return {
            "dialogue": [{"role": "general", "text": reply}],
            "next": "orchestrator",
        }

    return node


def make_executor(model, workdir: Path, data_path: str, timeout_s: int):
    def _run_python(code: str, purpose: str = "") -> str:
        res = run_script(code, workdir, data_path, timeout_s)
        return format_run(res)

    tool = StructuredTool.from_function(
        func=_run_python,
        name="run_python",
        description="Write the given Python code to a new file and execute it. Returns exit code, stdout and stderr.",
    )
    model_with_tools = model.bind_tools([tool])

    def node(state: State) -> dict:
        prev_scripts = []
        for t in state["dialogue"]:
            if t["role"] == "executor" and "--- run_python result" in t["text"]:
                prev_scripts.append(t["text"][:1500])
        previous = ""
        if prev_scripts:
            previous = (
                "Previous executor turns (for repair context):\n"
                + "\n---\n".join(prev_scripts[-2:])
            )

        instruction = ""
        for t in reversed(state["dialogue"]):
            if t["role"] == "orchestrator":
                instruction = t["text"]
                break

        sys_prompt = EXECUTOR_SYSTEM.format(
            timeout=timeout_s,
            budget=max(60, timeout_s // 3),
            previous=previous,
        )
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(
                content=render(state["dialogue"])
                + f"\n\n[orchestrator instruction for you]\n{instruction}"
            ),
        ]

        ai_msg = model_with_tools.invoke(messages)
        tool_calls = getattr(ai_msg, "tool_calls", None) or []

        if not tool_calls:
            reply = text_of(ai_msg) or "(executor produced no tool call and no text)"
            return {
                "dialogue": [{"role": "executor", "text": reply}],
                "next": "orchestrator",
            }

        tc = tool_calls[0]
        args = tc.get("args") or {}
        code = args.get("code", "")
        res = run_script(code, workdir, data_path, timeout_s)
        tool_result = format_run(res)

        updates: dict[str, Any] = {}
        if res["exit_code"] == 0:
            rmses = parse_rmse(res["stdout"])
            if "TEST_RMSE" in rmses:
                updates["test_rmse"] = rmses["TEST_RMSE"]
            if "TRAIN_RMSE" in rmses:
                updates["train_rmse"] = rmses["TRAIN_RMSE"]

        messages2 = messages + [
            ai_msg,
            ToolMessage(content=tool_result, tool_call_id=tc.get("id", "run1")),
        ]
        summary = text_of(model.invoke(messages2))
        full_text = tool_result + "\n\n" + (summary or "")

        return {
            "dialogue": [{"role": "executor", "text": full_text}],
            "next": "orchestrator",
            **updates,
        }

    return node


# ----------------------------------------------------------------------------- graph
def build_graph(
    models: dict,
    workdir: Path,
    data_path: str,
    timeout_s: int = EXEC_TIMEOUT_S,
    max_decisions: int = MAX_DECISIONS,
):
    g = StateGraph(State)

    g.add_node("orchestrator", make_orchestrator(models["orchestrator"], max_decisions))
    g.add_node("scientist", make_scientist(models["scientist"]))
    g.add_node("general", make_general(models["general"]))
    g.add_node(
        "executor",
        make_executor(models["executor"], workdir, data_path, timeout_s),
    )

    g.set_entry_point("orchestrator")

    def route(state: State) -> str:
        nxt = state.get("next", "finish")
        if nxt == "finish" or state.get("stop_reason"):
            return END
        return nxt

    g.add_conditional_edges(
        "orchestrator",
        route,
        {
            "scientist": "scientist",
            "general": "general",
            "executor": "executor",
            END: END,
        },
    )
    for role in ("scientist", "general", "executor"):
        g.add_edge(role, "orchestrator")

    return g.compile()


def default_models(model_name: str | None = None):
    """Prefer Google GenAI when GOOGLE_API_KEY is set; otherwise Anthropic."""
    name = model_name or os.environ.get("AGENT_MODEL")

    if os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_id = name or "gemini-3.8-flash"

        def make():
            return ChatGoogleGenerativeAI(
                model=model_id,
                temperature=0.2,
                convert_system_message_to_human=True,
            )

        return {
            "orchestrator": make(),
            "scientist": make(),
            "general": make(),
            "executor": make(),
        }

    from langchain_anthropic import ChatAnthropic

    model_id = name or "claude-sonnet-4-5-20250929"

    def make():
        return ChatAnthropic(model=model_id, temperature=0.2)

    return {
        "orchestrator": make(),
        "scientist": make(),
        "general": make(),
        "executor": make(),
    }


def initial_state(task: str) -> State:
    return {
        "dialogue": [{"role": "user", "text": task}],
        "decisions": 0,
        "test_rmse": None,
        "train_rmse": None,
    }


def transcript_md(state: State) -> str:
    lines = ["# Agent transcript\n"]
    for t in state.get("dialogue", []):
        lines.append(f"## {t['role']}\n")
        lines.append(t["text"].rstrip() + "\n")
    lines.append("\n---\n")
    lines.append(f"**decisions**: {state.get('decisions')}\n")
    lines.append(f"**stop_reason**: {state.get('stop_reason')}\n")
    lines.append(f"**TRAIN_RMSE**: {state.get('train_rmse')}\n")
    lines.append(f"**TEST_RMSE**: {state.get('test_rmse')}\n")
    return "\n".join(lines)