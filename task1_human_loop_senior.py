"""
Senior implementation - FINAL with working model defaults
Default now gemini-flash-latest which works with your key (you confirmed)
"""

from __future__ import annotations

import logging
import operator
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, model_validator

MAX_DECISIONS = 12
EXEC_TIMEOUT_S = 300
STDOUT_TAIL = 6000
STDERR_TAIL = 4000

logger = logging.getLogger(__name__)

ORCHESTRATOR_SYSTEM = """You are the ORCHESTRATOR for the Silverbox neural ODE task.
You are the ONLY role the user talks to. You see the entire dialogue and you decide who speaks next.

You manage 5 destinations:
- scientist: owns theory + plan (diff eq, network, state eq, training curriculum, free-run test, numbered coding steps). Route here when plan missing/wrong/needs scientific change.
- general: answers high-level side questions (what is a record, what does RMSE measure, what is in file). No tool, no plan, no code.
- executor: ONLY role with tool run_python(code, purpose). Routes here to implement latest scientist plan or repair a failed run.
- human: ask a person. You MUST choose human in exactly two cases:
  1) AMBIGUOUS QUERY: The latest user/human request admits >1 plausible reading, AND those readings would route to different roles OR change the numeric value reported. Example: "Report the RMSE" could be TRAIN_RMSE vs TEST_RMSE — that changes reported number. If ambiguous, do NOT guess. Choose human, state the two readings, ask which to follow. Set human_type=ambiguous, proposed_next=role you would have chosen without clarification, proposed_instruction=instruction for that role.
  2) HARD DECISION: The next step you intend is risky: it would run the coding tool (any executor call), overwrite a file already on disk (exec_*.py exists, TEST_RMSE already exists), or end the dialogue and report final measurement (finish). In that case you must confirm. Choose human, explicitly state the step: e.g. "run executor to create exec_02.py and overwrite exec_01.py which reported TEST_RMSE=0.0123" or "finish and report TEST_RMSE=0.0119". Ask whether to proceed. Set human_type=hard.
- finish: end run. Allowed ONLY after executor printed TEST_RMSE with exit_code 0. Final message goes in instruction.

Constraints:
- At most {max_decisions} decisions total, {used} used.
- You write no theory, no code.
- Return structured Decision: next, instruction, reason (one sentence), plus when next==human: question, human_type, proposed_next, proposed_instruction.
- Never use keyword list. Reason from dialogue and system status.

System status:
- has_result: {has_result}
- disk_files: {disk_files}
- current TEST_RMSE: {current_rmse}
"""

SCIENTIST_SYSTEM = """You are the SCIENTIST. You own theory and plan. No coding tool.
Output diff eq, network, state eq, training, free-run test, numbered coding steps.
"""

GENERAL_SYSTEM = """You are the GENERAL AGENT. High-level side questions. No tool, no plan. Short, concrete.
"""

EXECUTOR_SYSTEM = """You are the EXECUTOR. Only role with tool run_python(code, purpose).
Tool writes code to NEW file in workdir and runs it with {timeout}s limit. Call exactly once per turn.
Implement LATEST scientist plan. Must print TRAIN_RMSE=<float> and TEST_RMSE=<float> from free run.
{previous}
"""

class Turn(TypedDict):
    role: str
    text: str

class State(TypedDict, total=False):
    dialogue: Annotated[List[Turn], operator.add]
    next: str
    decisions: int
    test_rmse: Optional[float]
    train_rmse: Optional[float]
    stop_reason: str
    human_question: Optional[str]
    human_type: Optional[Literal["ambiguous", "hard"]]
    proposed_next: Optional[Literal["scientist", "general", "executor", "finish"]]
    proposed_instruction: Optional[str]
    human_accepted: Optional[bool]

class DecisionWithHuman(BaseModel):
    next: Literal["scientist", "general", "executor", "finish", "human"]
    instruction: str
    reason: str
    question: Optional[str] = None
    human_type: Optional[Literal["ambiguous", "hard"]] = None
    proposed_next: Optional[Literal["scientist", "general", "executor", "finish"]] = None
    proposed_instruction: Optional[str] = None

    @model_validator(mode="after")
    def _validate_human(self):
        if self.next == "human":
            if not self.question or len(self.question.strip()) < 10:
                raise ValueError("question required when next==human")
            if self.human_type not in ("ambiguous", "hard"):
                raise ValueError("human_type must be ambiguous or hard")
            if self.proposed_next is None:
                raise ValueError("proposed_next required")
            if not self.proposed_instruction:
                raise ValueError("proposed_instruction required")
        return self

def render(dialogue: List[Turn]) -> str:
    return "\n\n".join(f"[{t['role']}]\n{t['text']}" for t in dialogue)

def text_of(msg: Any) -> str:
    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    parts: List[str] = []
    for b in c or []:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts)

def as_decision(obj: Any) -> DecisionWithHuman:
    if isinstance(obj, DecisionWithHuman):
        return obj
    if isinstance(obj, dict):
        return DecisionWithHuman(**obj)
    raise ValueError(f"Invalid decision: {obj!r}")

_RMSE_RE = {k: re.compile(rf"^\s*{k}\s*=\s*([-+0-9.eE]+)\s*$", re.M) for k in ("TRAIN_RMSE", "TEST_RMSE")}

def parse_rmse(stdout: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, rx in _RMSE_RE.items():
        m = rx.findall(stdout)
        if m:
            try:
                val = float(m[-1])
                if val == val and abs(val) != float("inf"):
                    out[key] = val
            except ValueError:
                continue
    return out

def run_script(code: str, workdir: Path, data_path: str, timeout_s: int) -> Dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    idx = len(list(workdir.glob("exec_*.py"))) + 1
    path = workdir / f"exec_{idx:02d}.py"
    path.write_text(code, encoding="utf-8")
    env = dict(os.environ, SNLS_DATA=str(data_path), PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([sys.executable, path.name], cwd=str(workdir), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout_s)
        code_ = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except Exception:
            proc.kill()
        out, err = proc.communicate()
        err = (err or "") + f"\nTimeoutExpired after {timeout_s}s"
        code_ = -9
    return dict(file=path.name, exit_code=code_, stdout=(out or "")[-STDOUT_TAIL:], stderr=(err or "")[-STDERR_TAIL:])

def format_run(res: Dict[str, Any]) -> str:
    return f"--- run_python result ({res['file']}) ---\nexit code: {res['exit_code']}\nstdout:\n{res['stdout'] or '(empty)'}\nstderr:\n{res['stderr'] or '(empty)'}"

def make_orchestrator(model, max_decisions: int, workdir: Path):
    structured = model.with_structured_output(DecisionWithHuman)
    def node(state: State):
        used = state.get("decisions", 0)
        if used >= max_decisions:
            return {"next": "finish", "stop_reason": f"decision cap {max_decisions}"}
        has_result = state.get("test_rmse") is not None
        try:
            files = [p.name for p in workdir.glob("exec_*.py")] if workdir.exists() else []
        except Exception:
            files = []
        disk_str = ", ".join(files) if files else "(no exec files yet)"
        current_rmse = state.get("test_rmse")
        sys_prompt = ORCHESTRATOR_SYSTEM.format(max_decisions=max_decisions, used=used, has_result=has_result, disk_files=disk_str, current_rmse=current_rmse if current_rmse is not None else "None yet")
        messages = [SystemMessage(content=sys_prompt), HumanMessage(content=render(state["dialogue"]))]
        decision = as_decision(structured.invoke(messages))
        if decision.next == "finish" and not has_result:
            roles = {t["role"] for t in state["dialogue"]}
            fallback = "scientist" if "scientist" not in roles else "executor"
            text = f"[orchestrator decision refused]\nnext=finish refused (no TEST_RMSE yet). Routing to {fallback} instead.\nreason: {decision.reason}"
            return {"dialogue": [{"role": "orchestrator", "text": text}], "next": fallback, "decisions": used + 1}
        if decision.next == "human":
            text = f"next: human\nhuman_type: {decision.human_type}\nproposed_next: {decision.proposed_next}\nproposed_instruction: {decision.proposed_instruction}\nquestion: {decision.question}\nreason: {decision.reason}"
            return {"dialogue": [{"role": "orchestrator", "text": text}], "next": "human", "decisions": used + 1, "human_question": decision.question, "human_type": decision.human_type, "proposed_next": decision.proposed_next, "proposed_instruction": decision.proposed_instruction}
        text = f"next: {decision.next}\ninstruction: {decision.instruction}\nreason: {decision.reason}"
        return {"dialogue": [{"role": "orchestrator", "text": text}], "next": decision.next, "decisions": used + 1}
    return node

def make_scientist(model):
    def node(state: State):
        instruction = next((t["text"] for t in reversed(state["dialogue"]) if t["role"] == "orchestrator"), "")
        messages = [SystemMessage(content=SCIENTIST_SYSTEM), HumanMessage(content=render(state["dialogue"]) + f"\n[orchestrator instruction]\n{instruction}")]
        reply = text_of(model.invoke(messages))
        return {"dialogue": [{"role": "scientist", "text": reply}], "next": "orchestrator"}
    return node

def make_general(model):
    def node(state: State):
        instruction = next((t["text"] for t in reversed(state["dialogue"]) if t["role"] == "orchestrator"), "")
        messages = [SystemMessage(content=GENERAL_SYSTEM), HumanMessage(content=render(state["dialogue"]) + f"\n[orchestrator instruction]\n{instruction}")]
        reply = text_of(model.invoke(messages))
        return {"dialogue": [{"role": "general", "text": reply}], "next": "orchestrator"}
    return node

def make_executor(model, workdir: Path, data_path: str, timeout_s: int):
    def _run_python(code: str, purpose: str = "") -> str:
        res = run_script(code, workdir, data_path, timeout_s)
        return format_run(res)
    tool = StructuredTool.from_function(func=_run_python, name="run_python", description="Write Python code to new file and execute it.")
    model_with_tools = model.bind_tools([tool])
    def node(state: State):
        prev = [t["text"][:1500] for t in state["dialogue"] if t["role"] == "executor" and "run_python result" in t["text"]]
        previous = ("Previous executor turns:\n" + "\n---\n".join(prev[-2:])) if prev else ""
        instruction = next((t["text"] for t in reversed(state["dialogue"]) if t["role"] == "orchestrator"), "")
        sys_prompt = EXECUTOR_SYSTEM.format(timeout=timeout_s, budget=max(60, timeout_s // 3), previous=previous)
        messages = [SystemMessage(content=sys_prompt), HumanMessage(content=render(state["dialogue"]) + f"\n[orchestrator instruction]\n{instruction}")]
        ai_msg = model_with_tools.invoke(messages)
        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            reply = text_of(ai_msg) or "(executor: no tool call)"
            return {"dialogue": [{"role": "executor", "text": reply}], "next": "orchestrator"}
        tc = tool_calls[0]
        args = tc.get("args") or {}
        code = args.get("code", "")
        res = run_script(code, workdir, data_path, timeout_s)
        tool_result = format_run(res)
        updates: Dict[str, Any] = {}
        if res["exit_code"] == 0:
            rmses = parse_rmse(res["stdout"])
            if "TEST_RMSE" in rmses:
                updates["test_rmse"] = rmses["TEST_RMSE"]
            if "TRAIN_RMSE" in rmses:
                updates["train_rmse"] = rmses["TRAIN_RMSE"]
        messages2 = messages + [ai_msg, ToolMessage(content=tool_result, tool_call_id=tc.get("id", "run1"))]
        summary = text_of(model.invoke(messages2))
        full = tool_result + "\n\n" + (summary or "")
        return {"dialogue": [{"role": "executor", "text": full}], "next": "orchestrator", **updates}
    return node

_ACCEPT_RE = re.compile(r"\b(yes|yep|yeah|proceed|accept|go ahead|ok|okay|continue|allow|confirm)\b", re.I)
_REFUSE_RE = re.compile(r"\b(no|nope|refuse|reject|cancel|stop|wait|don't|do not|abort|hold)\b", re.I)

def _is_acceptance(text: str) -> bool:
    t = text.strip().lower()
    if t in {"y", "yes"}:
        return True
    if t == "no" or t.startswith("no "):
        return False
    has_accept = bool(_ACCEPT_RE.search(t))
    has_refuse = bool(_REFUSE_RE.search(t))
    if has_accept and not has_refuse:
        return True
    if has_accept and has_refuse:
        return "yes" in t or "proceed" in t
    return False

def make_human_node():
    def node(state: State):
        question = state.get("human_question") or "Human input needed."
        human_type = state.get("human_type") or "ambiguous"
        proposed_next = state.get("proposed_next") or "general"
        proposed_instruction = state.get("proposed_instruction") or ""
        payload = {"question": question, "type": human_type, "proposed_next": proposed_next, "proposed_instruction": proposed_instruction}
        human_reply_raw = interrupt(payload)
        if isinstance(human_reply_raw, dict):
            human_reply = str(human_reply_raw.get("text") or human_reply_raw.get("reply") or human_reply_raw)
        else:
            human_reply = str(human_reply_raw)
        dialogue_updates = [{"role": "human", "text": human_reply}]
        if human_type == "hard":
            if _is_acceptance(human_reply):
                if proposed_next == "finish":
                    return {"dialogue": dialogue_updates, "next": "finish", "stop_reason": "finish (human accepted)", "human_question": None, "human_type": None, "proposed_next": None, "proposed_instruction": None, "human_accepted": True}
                return {"dialogue": dialogue_updates, "next": proposed_next, "human_question": None, "human_type": None, "proposed_next": None, "proposed_instruction": None, "human_accepted": True}
            else:
                refusal_note = f"[human decision] The proposed step '{proposed_next}: {proposed_instruction}' was REFUSED by human. Human reply: '{human_reply}'. Do not run it. The refused step did not run."
                return {"dialogue": dialogue_updates + [{"role": "orchestrator", "text": refusal_note}], "next": "orchestrator", "human_question": None, "human_type": None, "proposed_next": None, "proposed_instruction": None, "human_accepted": False}
        else:
            return {"dialogue": dialogue_updates, "next": "orchestrator", "human_question": None, "human_type": None, "proposed_next": None, "proposed_instruction": None}
    return node

def build_graph_with_human(models, workdir: Path, data_path: str, timeout_s: int = 300, max_decisions: int = 8):
    checkpointer = MemorySaver()
    g = StateGraph(State)
    g.add_node("orchestrator", make_orchestrator(models["orchestrator"], max_decisions, workdir))
    g.add_node("scientist", make_scientist(models["scientist"]))
    g.add_node("general", make_general(models["general"]))
    g.add_node("executor", make_executor(models["executor"], workdir, data_path, timeout_s))
    g.add_node("human", make_human_node())
    g.set_entry_point("orchestrator")
    def route(state: State):
        nxt = state.get("next", "finish")
        if nxt == "finish" or state.get("stop_reason"):
            return END
        return nxt
    g.add_conditional_edges("orchestrator", route, {"scientist": "scientist", "general": "general", "executor": "executor", "human": "human", END: END})
    for role in ("scientist", "general", "executor"):
        g.add_edge(role, "orchestrator")
    def human_route(state: State):
        nxt = state.get("next", "orchestrator")
        if nxt == "finish":
            return END
        if nxt in ("scientist", "general", "executor"):
            return nxt
        return "orchestrator"
    g.add_conditional_edges("human", human_route, {"orchestrator": "orchestrator", "scientist": "scientist", "general": "general", "executor": "executor", END: END})
    return g.compile(checkpointer=checkpointer)

def default_models(model_name: Optional[str] = None):
    name = model_name or os.environ.get("AGENT_MODEL")
    if os.environ.get("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        # FINAL WORKING DEFAULTS for your key
        model_id = name or "gemini-3.8-flash"
        def make():
            return ChatGoogleGenerativeAI(model=model_id, temperature=0.2, convert_system_message_to_human=True)
        return {k: make() for k in ("orchestrator", "scientist", "general", "executor")}
    from langchain_anthropic import ChatAnthropic
    model_id = name or "claude-sonnet-4-5-20250929"
    def make():
        return ChatAnthropic(model=model_id, temperature=0.2)
    return {k: make() for k in ("orchestrator", "scientist", "general", "executor")}

def initial_state(task: str) -> State:
    return {"dialogue": [{"role": "user", "text": task}], "decisions": 0, "test_rmse": None, "train_rmse": None}

def transcript_md(state: State) -> str:
    lines = ["# Agent transcript (human-in-the-loop, senior FINAL)\n"]
    for t in state.get("dialogue", []):
        lines.append(f"## {t['role']}\n")
        lines.append(t["text"].rstrip() + "\n")
    lines.append(f"\n**decisions**: {state.get('decisions')}\n**stop_reason**: {state.get('stop_reason')}\n**TRAIN_RMSE**: {state.get('train_rmse')}\n**TEST_RMSE**: {state.get('test_rmse')}\n")
    return "\n".join(lines)
