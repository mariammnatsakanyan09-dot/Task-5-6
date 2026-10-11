"""Offline checks of the Task 6 wiring with scripted fake models (no API calls).

    python test_task6_offline.py
"""
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage

from task6_agents import Decision, build_graph, initial_state


class Scripted:
    """Minimal stand-in for a chat model: returns queued items, records what was asked of it."""

    def __init__(self, items):
        self.items = list(items)
        self.calls = []
        self.tools_bound = False
        self.structured = False

    def bind_tools(self, tools):
        self.tools_bound = True
        return self

    def with_structured_output(self, schema):
        self.structured = True
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.items.pop(0)


def tool_call(code, cid="c1"):
    return AIMessage(content="", tool_calls=[{"name": "run_python", "args": {"code": code, "purpose": "t"}, "id": cid}])


OK_SCRIPT = "print('some log')\nprint('TRAIN_RMSE=0.0100')\nprint('TEST_RMSE=0.0123')\n"
BAD_SCRIPT = "raise ValueError('boom')\n"


def run(orch, sci=(), gen=(), exe=(), timeout=20, max_decisions=8):
    models = dict(orchestrator=Scripted(orch), scientist=Scripted(sci), general=Scripted(gen), executor=Scripted(exe))
    wd = Path(tempfile.mkdtemp())
    g = build_graph(models, wd, "/nonexistent.mat", timeout_s=timeout, max_decisions=max_decisions)
    final = g.invoke(initial_state("task text"), config={"recursion_limit": 60})
    return final, models, wd


def D(nxt, ins="do it", reason="because."):
    return Decision(next=nxt, instruction=ins, reason=reason)


def roles(final):
    return [t["role"] for t in final["dialogue"]]


def test_happy_path():
    final, models, wd = run(
        orch=[D("scientist"), D("executor"), D("finish", "final report")],
        sci=[AIMessage(content="plan: eq, net, state, training, free run, steps")],
        exe=[tool_call(OK_SCRIPT), AIMessage(content="exit 0, TEST_RMSE=0.0123")])
    assert roles(final) == ["user", "orchestrator", "scientist", "orchestrator", "executor", "orchestrator"], roles(final)
    assert final["test_rmse"] == 0.0123 and final["train_rmse"] == 0.01
    assert final["stop_reason"] == "finish" and final["decisions"] == 3
    assert (wd / "exec_01.py").exists()
    ex = final["dialogue"][4]["text"]
    assert "exit code: 0" in ex and "TEST_RMSE=0.0123" in ex
    # tools only on the executor
    assert models["executor"].tools_bound
    assert not models["scientist"].tools_bound and not models["general"].tools_bound and not models["orchestrator"].tools_bound
    assert models["orchestrator"].structured
    # the scientist saw the orchestrator's instruction in the shared dialogue
    seen = models["scientist"].calls[0][-1].content
    assert "[orchestrator]" in seen and "instruction: do it" in seen and "[user]" in seen
    print("ok  happy path")


def test_premature_finish_is_refused():
    # model asks to finish twice with no RMSE: guard forces the scientist (no plan yet), then proceeds normally
    final, _, _ = run(
        orch=[D("finish"), D("finish"), D("executor"), D("finish")],
        sci=[AIMessage(content="plan")],
        exe=[tool_call(OK_SCRIPT), AIMessage(content="done"),
             tool_call(OK_SCRIPT), AIMessage(content="done")],  # extra in case of re-routing
    )
    assert roles(final)[1:3] == ["orchestrator", "scientist"], roles(final)
    assert "refused by the finish guard" in final["dialogue"][1]["text"] or \
       "finish refused by guard" in final["dialogue"][1]["text"]
    assert final["test_rmse"] == 0.0123 and final["stop_reason"] == "finish"
    print("ok  premature finish refused")


def test_failure_then_repair():
    final, _, wd = run(
        orch=[D("scientist"), D("executor"), D("executor", "repair"), D("finish")],
        sci=[AIMessage(content="plan")],
        exe=[tool_call(BAD_SCRIPT), AIMessage(content="it failed with ValueError"),
             tool_call(OK_SCRIPT, "c2"), AIMessage(content="fixed")])
    execs = [t for t in final["dialogue"] if t["role"] == "executor"]
    assert "exit code: 1" in execs[0]["text"] and "ValueError: boom" in execs[0]["text"]
    assert "exit code: 0" in execs[1]["text"]
    assert sorted(p.name for p in wd.glob("exec_*.py")) == ["exec_01.py", "exec_02.py"]   # a new file each time
    assert final["test_rmse"] == 0.0123
    print("ok  failure then repair (new file per turn, traceback returned)")


def test_failed_run_gives_no_rmse():
    # a failing script that printed a TEST_RMSE line must not count
    script = "print('TEST_RMSE=0.001')\nraise SystemExit(3)\n"
    final, _, _ = run(
        orch=[D("scientist"), D("executor")] + [D("finish")] * 20,
        sci=[AIMessage(content="plan")],
        exe=[tool_call(script), AIMessage(content="exit 3")] * 4, max_decisions=5)
    assert final["test_rmse"] is None and "cap" in final["stop_reason"]
    print("ok  RMSE from a failed run is not accepted")


def test_decision_cap():
    final, _, _ = run(orch=[D("general")] * 8, gen=[AIMessage(content="answer")] * 8)
    assert final["decisions"] == 8 and roles(final).count("general") == 8
    assert "cap" in final["stop_reason"]
    print("ok  stops after 8 decisions")


def test_timeout():
    final, _, _ = run(
        orch=[D("scientist"), D("executor"), D("finish", "x"), D("executor"), D("finish")],
        sci=[AIMessage(content="plan")],
        exe=[tool_call("import time\ntime.sleep(30)\n"), AIMessage(content="timed out")] * 2,
        timeout=2, max_decisions=3)
    ex = [t for t in final["dialogue"] if t["role"] == "executor"][0]["text"]
    assert "TimeoutExpired" in ex and "exit code: -9" in ex
    assert final["test_rmse"] is None
    print("ok  time limit enforced")


def test_model_text_cannot_fake_rmse():
    # the executor's prose claims a number, but nothing printed it: finish must still be refused
    final, _, _ = run(
        orch=[D("scientist"), D("executor")] + [D("finish")] * 20,
        sci=[AIMessage(content="plan")],
        exe=[tool_call("print('hello')\n"), AIMessage(content="TEST_RMSE=0.0001, great")] * 4, max_decisions=5)
    assert final["test_rmse"] is None and "cap" in final["stop_reason"]
    print("ok  RMSE claimed in prose is ignored")


if __name__ == "__main__":
    for fn in (test_happy_path, test_premature_finish_is_refused, test_failure_then_repair,
               test_failed_run_gives_no_rmse, test_decision_cap, test_timeout, test_model_text_cannot_fake_rmse):
        fn()
    print("\nall offline checks passed")
