"""Offline checks of the agent wiring with scripted fake models (no API calls; the bench is the real one, with small runs).

    python test_agents_offline.py

The roles are the Task 6 ones (orchestrator, scientist, general, executor); the executor has the new tool focus_search.
"""
import json
import tempfile
from pathlib import Path

import focus_agents as F
from focus_agents import Decision, build_graph, run_agent
from langchain_core.messages import AIMessage

F.RETRY_SLEEP[:] = [0, 0]          # no waiting in tests
F.emit = lambda *a, **k: None      # keep the output of the checks short


class Scripted:
    """Stand-in for a chat model: returns queued items (an Exception item is raised), records what was asked."""

    def __init__(self, items):
        self.items = list(items)
        self.calls = []
        self.tools_bound = None
        self.structured = False

    def bind_tools(self, tools):
        self.tools_bound = [t.name for t in tools]
        return self

    def with_structured_output(self, schema):
        self.structured = True
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


A = dict(macropixel_size=16, focus_size=3, bg_gap=2, bg_width=6, n_bg_regions=8, iterations=300)


def tc(*calls, name="focus_search"):
    """An assistant message with tool calls. Each call is a dict of arguments (or a (name, args) pair)."""
    out = []
    for i, c in enumerate(calls):
        n, a = c if isinstance(c, tuple) else (name, c)
        out.append({"name": n, "args": a, "id": f"c{i}"})
    return AIMessage(content="", tool_calls=out)


def D(nxt, ins="do it", reason="because."):
    return Decision(next=nxt, instruction=ins, reason=reason)


def roles(final):
    return [t["role"] for t in final["dialogue"]]


def run(orch, sci=(), gen=(), exe=(), max_decisions=10, timeout=20):
    models = dict(orchestrator=Scripted(orch), scientist=Scripted(sci), general=Scripted(gen), executor=Scripted(exe))
    wd = Path(tempfile.mkdtemp())
    g = build_graph(models, wd, timeout_s=timeout, max_decisions=max_decisions)
    final = run_agent(g, "Request: Focus the beam.", thread_id="t")
    return final, models, wd


def a(**kw):
    d = dict(A)
    d.update(kw)
    return d


# ---------------------------------------------------------------------------------------------------------------
def test_happy_path():
    final, models, wd = run(
        orch=[D("scientist"), D("executor", "run experiments 1-2"), D("executor", "run experiment 3"), D("finish", "final report")],
        sci=[AIMessage(content="plan: 1) b=16 2) b=8 3) b=8 more iterations")],
        exe=[tc(a(), a(macropixel_size=8, iterations=600)), AIMessage(content="two experiments ran"),
             tc(a(macropixel_size=8, iterations=1500)), AIMessage(content="third ran")])
    assert roles(final) == ["user", "orchestrator", "scientist", "orchestrator", "executor", "orchestrator", "executor",
                            "orchestrator"], roles(final)
    exps = final["experiments"]
    assert [e["experiment"] for e in exps] == [1, 2, 3] and final["n_calls"] == 3
    assert [e["settings"]["macropixel_size"] for e in exps] == [16, 8, 8]
    assert final["stop_reason"] == "finish" and final["decisions"] == 4
    for e in exps:                                                    # files made by the tool, in the working directory
        assert (wd / e["figure"]).exists() and (wd / e["data"]).exists()
    saved = json.loads((wd / "experiments.json").read_text())
    assert [e["enhancement"] for e in saved] == [e["enhancement"] for e in exps]
    # roles and tools
    assert models["executor"].tools_bound == ["focus_search", "run_python"]
    assert models["orchestrator"].tools_bound is None and models["scientist"].tools_bound is None
    assert models["general"].tools_bound is None and models["orchestrator"].structured
    # the executor's turn holds the tool results exactly as the tool returned them
    ex1 = final["dialogue"][4]["text"]
    assert f"enhancement: {exps[0]['enhancement']}" in ex1 and f"enhancement: {exps[1]['enhancement']}" in ex1
    # the orchestrator read the table written by the program before deciding
    seen = models["orchestrator"].calls[2][-1].content
    assert "Experiments so far" in seen and f"**{exps[0]['enhancement']:.1f}**" in seen and "decisions used: 2 of 10" in seen
    # the executor was shown the request, the plan and its instruction
    first = models["executor"].calls[0][-1].content
    assert "Focus the beam" in first and "plan: 1) b=16" in first and "run experiments 1-2" in first
    rep = F.final_report_md(final)
    assert "final report" in rep and f"{max(e['enhancement'] for e in exps):.1f}" in rep
    print("ok  A1 happy path: scientist plan, two tool calls in one reply, one more turn, finish; results recorded from the tool")


def test_premature_finish_is_refused():
    final, _, _ = run(
        orch=[D("finish"), D("executor"), D("finish"), D("executor"), D("finish", "done")],
        exe=[tc(a(), a(macropixel_size=8)), AIMessage(content="ok"), tc(a(macropixel_size=4)), AIMessage(content="ok")])
    assert "refused by the finish guard: 0 successful" in final["dialogue"][1]["text"]
    assert "refused by the finish guard: 2 successful" in final["dialogue"][4]["text"]
    assert len(final["experiments"]) == 3 and final["stop_reason"] == "finish"
    print("ok  A2 finish refused until 3 experiments have really run (counted from the tool results)")


def test_finish_near_the_cap_is_allowed_with_one_experiment():
    final, _, _ = run(orch=[D("executor"), D("finish", "stop")], exe=[tc(a()), AIMessage(content="ok")], max_decisions=3)
    assert final["stop_reason"] == "finish" and len(final["experiments"]) == 1 and final["final_text"] == "stop"
    print("ok  A3 with the decision budget nearly gone, finish is allowed after one experiment")


def test_rejected_call_is_returned_and_repaired():
    final, models, _ = run(
        orch=[D("executor"), D("executor"), D("finish")],
        exe=[tc(a(macropixel_size=3)), tc(a(macropixel_size=8)), AIMessage(content="repaired"),
             tc(a(), a(macropixel_size=4)), AIMessage(content="ok")])
    ex = final["dialogue"][2]["text"]
    assert "REJECTED" in ex and "macropixel_size must be one of" in ex and "experiment 1" in ex
    assert final["n_calls"] == 4 and len(final["experiments"]) == 3          # the rejected call is a call, not an experiment
    msgs = models["executor"].calls[1]                                       # round 2 saw the tool's error text
    assert any(getattr(m, "type", "") == "tool" and "must be one of" in m.content for m in msgs)
    assert [e["experiment"] for e in final["experiments"]] == [1, 2, 3]
    print("ok  A4 an invalid setting is rejected by the tool, the error goes back to the model, it repairs in the same turn")


def test_missing_arguments_are_rejected():
    final, _, _ = run(orch=[D("executor")] + [D("finish")] * 5, exe=[tc({"macropixel_size": 8}), AIMessage(content="no")] * 3,
                      max_decisions=2)
    ex = final["dialogue"][2]["text"]
    assert "REJECTED" in ex and "iterations" in ex and "focus_size" in ex and not final["experiments"]
    print("ok  A5 a call without the required settings is rejected (the agent has to choose them)")


def test_call_caps():
    six = [a(macropixel_size=16, iterations=100 + i) for i in range(6)]
    final, _, _ = run(orch=[D("executor"), D("executor"), D("finish")], exe=[tc(*six), AIMessage(content="x"), tc(*six), AIMessage(content="y")])
    first = final["dialogue"][2]["text"]
    assert first.count("--- focus_search result") == F.MAX_CALLS_PER_TURN
    old = F.MAX_FOCUS_CALLS
    F.MAX_FOCUS_CALLS = 5
    try:
        final, _, _ = run(orch=[D("executor"), D("executor"), D("executor"), D("general"), D("finish")],
                          exe=[tc(*six[:4]), AIMessage(content="x"), tc(*six[:4]), AIMessage(content="y")],
                          gen=[AIMessage(content="g")])
    finally:
        F.MAX_FOCUS_CALLS = old
    assert final["n_calls"] == 5 and len(final["experiments"]) == 5
    assert "executor was refused" in final["dialogue"][5]["text"]                # the third executor decision is refused
    print(f"ok  A6 at most {F.MAX_CALLS_PER_TURN} calls per turn; the run-wide cap stops further experiments")


def test_run_python_is_not_an_experiment():
    script = "print('enhancement: 999.0')\nprint('ENHANCEMENT=999')\n"
    final, _, wd = run(orch=[D("executor")] + [D("finish")] * 5,
                       exe=[tc(("run_python", {"code": script, "purpose": "fake"})), AIMessage(content="enhancement 999")] * 3,
                       max_decisions=2)
    assert "exit code: 0" in final["dialogue"][2]["text"] and (wd / "exec_01.py").exists()
    assert not final["experiments"] and "999" not in F.final_report_md(final).split("Experiments run")[1]
    print("ok  A7 run_python results and numbers claimed in prose never become an experiment")


def test_prose_cannot_change_the_table():
    final, _, _ = run(orch=[D("executor"), D("executor"), D("finish", "best was 12345")],
                      exe=[tc(a(), a(macropixel_size=8), a(macropixel_size=4)), AIMessage(content="enhancement 777 great"),
                           tc(a(macropixel_size=2, iterations=500)), AIMessage(content="ok")])
    rep = F.final_report_md(final)
    table = rep.split("## Experiments run")[1]
    assert "777" not in table and "12345" not in table
    best = max(final["experiments"], key=lambda e: e["enhancement"])
    assert f"Best enhancement kept from the tool: {best['enhancement']:.1f}" in table
    print("ok  A8 the table and the best enhancement come from the tool results, whatever the models write")


def test_text_mode_fallback_is_sticky():
    err = RuntimeError("Error code: 404 - No endpoints found that support tool use. Try disabling tools.")
    final, models, _ = run(
        orch=[D("executor"), D("executor"), D("finish")],
        exe=[err,
             AIMessage(content='{"calls": [%s, %s]}' % (json.dumps(a()), json.dumps(a(macropixel_size=8)))),
             AIMessage(content='```json\n{"calls": [%s]}\n```' % json.dumps(a(macropixel_size=4)))],
        max_decisions=4)
    assert [e["settings"]["macropixel_size"] for e in final["experiments"]] == [16, 8, 4]
    assert len(models["executor"].calls) == 3                      # the second turn went straight to text mode
    assert "JSON object" in models["executor"].calls[1][-1].content
    print("ok  A9 a model without tool calling: one JSON list of experiments in text, same tool, sticky for the run")


def test_tool_call_printed_as_text():
    markup = ("I will run it.\n<tool_call>\n<function=focus_search>\n<parameter=macropixel_size>\n16\n</parameter>\n"
              "<parameter=focus_size>\n3\n</parameter>\n<parameter=bg_gap>\n2\n</parameter>\n<parameter=bg_width>\n6\n</parameter>\n"
              "<parameter=n_bg_regions>\n8\n</parameter>\n<parameter=iterations>\n300\n</parameter>\n</function>\n</tool_call>")
    final, models, _ = run(orch=[D("executor"), D("finish")], exe=[AIMessage(content=markup), AIMessage(content="done")], max_decisions=2)
    assert len(final["experiments"]) == 1 and final["experiments"][0]["settings"]["iterations"] == 300
    print("ok  A10 a tool call that the model printed as text is read and run")


def test_nudge_when_no_tool_call():
    final, models, _ = run(orch=[D("executor"), D("finish")],
                           exe=[AIMessage(content="I will think about it."), tc(a()), AIMessage(content="ok")], max_decisions=2)
    assert len(final["experiments"]) == 1 and F.NUDGE in models["executor"].calls[1][-1].content
    print("ok  A11 an executor reply without a tool call is nudged once")


def test_decision_cap_and_no_experiment():
    final, _, _ = run(orch=[D("general")] * 10, gen=[AIMessage(content="answer")] * 10, max_decisions=4)
    assert final["decisions"] == 4 and "cap" in final["stop_reason"] and not final["experiments"]
    rep = F.final_report_md(final)
    assert "No experiment succeeded" in rep
    print("ok  A12 the decision cap ends the run; with no experiment the report says nothing was measured")


def test_structured_output_fallback():
    class NoStructured(Scripted):
        def with_structured_output(self, schema):
            raise RuntimeError("Error code: 400 - response_format is not supported")

    models = dict(orchestrator=NoStructured([AIMessage(content='{"next": "executor", "instruction": "go", "reason": "r"}'),
                                             AIMessage(content='```json\n{"next": "finish", "instruction": "bye", "reason": "r"}\n```')]),
                  scientist=Scripted([]), general=Scripted([]), executor=Scripted([tc(a()), AIMessage(content="ok")]))
    wd = Path(tempfile.mkdtemp())
    g = build_graph(models, wd, max_decisions=2)
    final = run_agent(g, "Request: Focus the beam.", thread_id="t")
    assert len(final["experiments"]) == 1 and final["final_text"] == "bye"
    assert "JSON object" in models["orchestrator"].calls[0][-1].content
    print("ok  A13 an orchestrator model without structured output is asked for one JSON object")


def test_quota_error_is_clear():
    quota = RuntimeError("Error code: 429 - Rate limit exceeded: free-models-per-day. Add 10 credits")
    models = dict(orchestrator=Scripted([quota]), scientist=Scripted([]), general=Scripted([]), executor=Scripted([]))
    g = build_graph(models, Path(tempfile.mkdtemp()))
    try:
        run_agent(g, "Request: Focus the beam.", thread_id="t")
    except RuntimeError as e:
        assert "DAILY quota" in str(e)
    else:
        raise AssertionError("no error")
    print("ok  A14 a daily-quota error stops the run with a clear message (no retries burned)")


def test_hook_and_prompts():
    seen = []
    F.HOOKS["after_experiment"] = lambda res: seen.append(res["figure"])
    try:
        run(orch=[D("executor"), D("finish")], exe=[tc(a()), AIMessage(content="ok")], max_decisions=2)
    finally:
        F.HOOKS["after_experiment"] = None
    assert seen == ["exp_01_images.png"]
    assert all("{" not in p and "}" not in p for p in (F.BENCH_MANUAL,))
    props = F.FocusSearchArgs.model_json_schema()
    assert set(props["required"]) == {"macropixel_size", "focus_size", "bg_gap", "bg_width", "n_bg_regions", "iterations"}
    print("ok  A15 after-experiment hook is called; prompts format; six settings are required in the tool schema")


if __name__ == "__main__":
    test_happy_path()
    test_premature_finish_is_refused()
    test_finish_near_the_cap_is_allowed_with_one_experiment()
    test_rejected_call_is_returned_and_repaired()
    test_missing_arguments_are_rejected()
    test_call_caps()
    test_run_python_is_not_an_experiment()
    test_prose_cannot_change_the_table()
    test_text_mode_fallback_is_sticky()
    test_tool_call_printed_as_text()
    test_nudge_when_no_tool_call()
    test_decision_cap_and_no_experiment()
    test_structured_output_fallback()
    test_quota_error_is_clear()
    test_hook_and_prompts()
    print("\nall agent checks passed")
