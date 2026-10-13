"""Offline checks of the human-in-the-loop wiring with scripted fake models (no API calls).

    python test_hitl_offline.py

Part A: the Task 6 checks (four roles, 8-decision cap, finish guard, repair, time limit, tools only on the executor).
Part B: the new stop: interrupt on a thread, resume the same thread, accept / refuse, fallbacks for free models.
"""
import tempfile
from pathlib import Path

import hitl_agents as H
from hitl_agents import Decision, build_graph, drive, initial_state, parse_verdict
from langchain_core.messages import AIMessage
from langgraph.types import Command

H.RETRY_SLEEP[:] = [0, 0]          # no waiting in tests


class Scripted:
    """Stand-in for a chat model: returns queued items (an Exception item is raised), records what was asked."""

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
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def tool_call(code, cid="c1"):
    return AIMessage(content="", tool_calls=[{"name": "run_python", "args": {"code": code, "purpose": "t"}, "id": cid}])


OK_SCRIPT = "print('some log')\nprint('TRAIN_RMSE=0.0100')\nprint('TEST_RMSE=0.0123')\n"
BAD_SCRIPT = "raise ValueError('boom')\n"


def D(nxt, ins="do it", reason="because."):
    return Decision(next=nxt, instruction=ins, reason=reason)


def ASK(question="Which reading?"):
    return Decision(next="human", kind="ambiguous_query", question=question, reason="two readings change the number.")


def HARD(step="executor", ins="run the fit", question="I am about to run the coding tool. Proceed?"):
    return Decision(next="human", kind="hard_decision", guarded=step, instruction=ins, question=question,
                    reason="the step runs code.")


def roles(final):
    return [t["role"] for t in final["dialogue"]]


def make(orch, sci=(), gen=(), exe=(), timeout=20, max_decisions=8):
    models = dict(orchestrator=Scripted(orch), scientist=Scripted(sci), general=Scripted(gen), executor=Scripted(exe))
    wd = Path(tempfile.mkdtemp())
    g = build_graph(models, wd, "/nonexistent.mat", timeout_s=timeout, max_decisions=max_decisions)
    return g, models, wd


def run(orch, sci=(), gen=(), exe=(), timeout=20, max_decisions=8, replies=()):
    """Drive the graph to the end, answering every pause from `replies`. Returns final state, models, workdir, asked."""
    g, models, wd = make(orch, sci, gen, exe, timeout, max_decisions)
    replies, asked = list(replies), []

    def ask(payload):
        asked.append(payload)
        return replies.pop(0)

    final = drive(g, "task text", ask, thread_id="t")
    return final, models, wd, asked


# ============================================================================ Part A: Task 6 behaviour
def test_happy_path():
    final, models, wd, asked = run(
        orch=[D("scientist"), D("executor"), D("finish", "final report")],
        sci=[AIMessage(content="plan: eq, net, state, training, free run, steps")],
        exe=[tool_call(OK_SCRIPT), AIMessage(content="exit 0, TEST_RMSE=0.0123")])
    assert roles(final) == ["user", "orchestrator", "scientist", "orchestrator", "executor", "orchestrator"], roles(final)
    assert final["test_rmse"] == 0.0123 and final["train_rmse"] == 0.01
    assert final["stop_reason"] == "finish" and final["decisions"] == 3 and not asked
    assert (wd / "exec_01.py").exists()
    ex = final["dialogue"][4]["text"]
    assert "exit code: 0" in ex and "TEST_RMSE=0.0123" in ex
    assert models["executor"].tools_bound
    assert not models["scientist"].tools_bound and not models["general"].tools_bound and not models["orchestrator"].tools_bound
    assert models["orchestrator"].structured
    seen = models["scientist"].calls[0][-1].content
    assert "[orchestrator]" in seen and "instruction: do it" in seen and "[user]" in seen
    print("ok  A1 happy path (four roles, tools only on the executor, nobody asked)")


def test_premature_finish_is_refused():
    final, _, _, _ = run(
        orch=[D("finish"), D("scientist"), D("executor"), D("finish")],
        sci=[AIMessage(content="plan")],
        exe=[tool_call(OK_SCRIPT), AIMessage(content="done")])
    assert roles(final)[1:3] == ["orchestrator", "orchestrator"], roles(final)
    assert "refused by the finish guard" in final["dialogue"][1]["text"]
    assert final["test_rmse"] == 0.0123 and final["stop_reason"] == "finish"
    print("ok  A2 premature finish refused, the orchestrator decides again")


def test_failure_then_repair():
    final, _, wd, _ = run(
        orch=[D("scientist"), D("executor"), D("executor", "repair"), D("finish")],
        sci=[AIMessage(content="plan")],
        exe=[tool_call(BAD_SCRIPT), AIMessage(content="it failed with ValueError"),
             tool_call(OK_SCRIPT, "c2"), AIMessage(content="fixed")])
    execs = [t for t in final["dialogue"] if t["role"] == "executor"]
    assert "exit code: 1" in execs[0]["text"] and "ValueError: boom" in execs[0]["text"]
    assert "exit code: 0" in execs[1]["text"]
    assert sorted(p.name for p in wd.glob("exec_*.py")) == ["exec_01.py", "exec_02.py"]
    assert final["test_rmse"] == 0.0123
    print("ok  A3 failure then repair (new file per turn, traceback returned)")


def test_failed_run_gives_no_rmse():
    script = "print('TEST_RMSE=0.001')\nraise SystemExit(3)\n"
    final, _, _, _ = run(
        orch=[D("scientist"), D("executor")] + [D("finish")] * 20,
        sci=[AIMessage(content="plan")],
        exe=[tool_call(script), AIMessage(content="exit 3")] * 4, max_decisions=5)
    assert final["test_rmse"] is None and "cap" in final["stop_reason"]
    print("ok  A4 RMSE from a failed run is not accepted")


def test_decision_cap():
    final, _, _, _ = run(orch=[D("general")] * 8, gen=[AIMessage(content="answer")] * 8)
    assert final["decisions"] == 8 and roles(final).count("general") == 8 and "cap" in final["stop_reason"]
    print("ok  A5 stops after 8 decisions")


def test_timeout():
    final, _, _, _ = run(
        orch=[D("scientist"), D("executor"), D("finish", "x"), D("executor"), D("finish")],
        sci=[AIMessage(content="plan")],
        exe=[tool_call("import time\ntime.sleep(30)\n"), AIMessage(content="timed out")] * 2,
        timeout=2, max_decisions=3)
    ex = [t for t in final["dialogue"] if t["role"] == "executor"][0]["text"]
    assert "TimeoutExpired" in ex and "exit code: -9" in ex and final["test_rmse"] is None
    print("ok  A6 time limit enforced")


def test_model_text_cannot_fake_rmse():
    final, _, _, _ = run(
        orch=[D("scientist"), D("executor")] + [D("finish")] * 20,
        sci=[AIMessage(content="plan")],
        exe=[tool_call("print('hello')\n"), AIMessage(content="TEST_RMSE=0.0001, great")] * 4, max_decisions=5)
    assert final["test_rmse"] is None and "cap" in final["stop_reason"]
    print("ok  A7 RMSE claimed in prose is ignored")


# ============================================================================ Part B: the human stop
def test_pause_and_resume_same_thread():
    """The graph really stops at `human`, stays stopped, and continues only when THAT thread is resumed."""
    g, models, wd = make(orch=[ASK("Train or test error?"), D("scientist"), D("executor"), D("finish", "report")],
                         sci=[AIMessage(content="plan for the TEST segment")],
                         exe=[tool_call(OK_SCRIPT), AIMessage(content="ok")])
    assert g.checkpointer is not None                                  # compiled with a checkpointer
    cfg = {"configurable": {"thread_id": "A"}, "recursion_limit": 100}
    g.invoke(initial_state("task"), cfg)
    snap = g.get_state(cfg)
    assert snap.next == ("human",)                                     # paused AT the human node
    payload = snap.tasks[0].interrupts[0].value
    assert payload["kind"] == "ambiguous_query" and payload["question"] == "Train or test error?"
    assert roles(snap.values) == ["user", "orchestrator"]              # nothing after the question has run
    assert len(models["scientist"].calls) == 0 and len(models["orchestrator"].calls) == 1

    assert g.get_state(cfg).next == ("human",)                         # still paused when looked at again
    assert len(models["orchestrator"].calls) == 1                      # ... and nothing else was decided meanwhile

    g.invoke(Command(resume="the test segment"), cfg)                  # resume the same thread
    # the reply is a turn with role `human`, and the NEXT orchestrator decision was made after reading it
    snap = g.get_state(cfg)
    assert roles(snap.values)[2:4] == ["human", "orchestrator"], roles(snap.values)
    assert snap.values["dialogue"][2]["text"] == "the test segment"
    assert snap.values["dialogue"][3]["text"].startswith("next: scientist")
    prompt = models["orchestrator"].calls[1][-1].content
    assert "[human]\nthe test segment" in prompt                       # the 2nd decision read the reply
    assert "[graph]" not in prompt                                     # ambiguous query: no graph note
    print("ok  B1 interrupt on a thread; stays paused; resume the same thread; reply = turn `human`; next decision reads it")
    return g, cfg


def test_ambiguous_reading_reaches_the_roles():
    final, models, _, asked = run(
        orch=[ASK(), D("scientist"), D("executor"), D("finish", "done")],
        sci=[AIMessage(content="plan")], exe=[tool_call(OK_SCRIPT), AIMessage(content="ok")],
        replies=["the free-run TEST error"])
    assert len(asked) == 1 and asked[0]["kind"] == "ambiguous_query" and "step" not in asked[0]
    assert roles(final) == ["user", "orchestrator", "human", "orchestrator", "scientist", "orchestrator",
                            "executor", "orchestrator"], roles(final)
    assert "[human]\nthe free-run TEST error" in models["scientist"].calls[0][-1].content   # specialists see it too
    assert final["decisions"] == 4                                      # asking the human is one decision
    print("ok  B2 ambiguous query: the answer is in the shared dialogue for every role; costs one decision")


def test_hard_decision_accept_runs_the_step():
    final, models, wd, asked = run(
        orch=[D("scientist"), HARD("executor", "RUN-THE-FIT-INSTRUCTION"), D("finish", "report")],
        sci=[AIMessage(content="plan")], exe=[tool_call(OK_SCRIPT), AIMessage(content="ok")],
        replies=["accept, go"])
    assert asked[0]["kind"] == "hard_decision" and asked[0]["step"] == {"role": "executor", "instruction": "RUN-THE-FIT-INSTRUCTION"}
    assert roles(final) == ["user", "orchestrator", "scientist", "orchestrator", "human", "graph", "executor",
                            "orchestrator"], roles(final)
    assert "accepted" in final["dialogue"][5]["text"]
    assert (wd / "exec_01.py").exists() and final["test_rmse"] == 0.0123
    assert "RUN-THE-FIT-INSTRUCTION" in models["executor"].calls[0][-1].content    # the guarded step is what ran
    assert final["decisions"] == 3                                      # the accepted step cost no extra decision
    print("ok  B3 hard decision accepted: the guarded step runs")


def test_hard_decision_refuse_does_not_run_the_step():
    final, models, wd, asked = run(
        orch=[D("scientist"), HARD("executor", "overwrite results/best_model.pt"),
              HARD("executor", "write to best_model_v2.pt instead"), D("finish", "report")],
        sci=[AIMessage(content="plan")], exe=[tool_call(OK_SCRIPT), AIMessage(content="ok")],
        replies=["refuse, do not overwrite", "yes"])
    r = roles(final)
    # after the refusal: graph note, then the orchestrator again. No executor turn in between.
    i = r.index("human")
    assert r[i:i + 3] == ["human", "graph", "orchestrator"], r
    note = final["dialogue"][i + 1]["text"]
    assert "did NOT run" in note and "executor scripts in the working directory: 0" in note
    assert not any(t["role"] == "executor" for t in final["dialogue"][:i + 2])
    assert final["refused"] == 1
    # the second ask was accepted and only then did exec_01.py appear
    assert r.count("human") == 2 and r.count("executor") == 1 and (wd / "exec_01.py").exists()
    assert (wd / "exec_01.py").read_text() == OK_SCRIPT
    assert "write to best_model_v2.pt" in models["executor"].calls[0][-1].content
    assert len(models["executor"].calls) == 2                           # 1 tool turn + 1 summary: one run only
    print("ok  B4 hard decision refused: step did not run, no script written, turn back to the orchestrator; "
          "second ask accepted")


def test_refusal_is_visible_to_the_next_decision():
    final, models, _, _ = run(
        orch=[D("scientist"), HARD("executor"), D("finish", "stopped: the human refused the run; nothing measured")],
        sci=[AIMessage(content="plan")], replies=["no"])
    prompt = models["orchestrator"].calls[2][-1].content
    assert "[human]\nno" in prompt and "did NOT run" in prompt
    assert final["test_rmse"] is None and final["stop_reason"] == "finish"       # honest finish allowed after a refusal
    assert models["executor"].calls == []
    print("ok  B5 the next decision sees the refusal; finishing without a measurement is allowed after it")


def test_unclear_reply_counts_as_refusal():
    final, models, wd, _ = run(
        orch=[D("scientist"), HARD("executor"), D("finish", "stopped")],
        sci=[AIMessage(content="plan")], replies=["hmm, what would that do exactly?"])
    assert "not a clear accept" in final["dialogue"][5]["text"] and models["executor"].calls == []
    assert not list(wd.glob("exec_*.py"))
    for text, want in [("accept", "accept"), ("Yes, run it", "accept"), ("ok", "accept"), ("Refuse once", "refuse"),
                       ("no, keep the old file", "refuse"), ("don't", "refuse"), ("maybe", None), ("", None)]:
        assert parse_verdict(text) == want, (text, parse_verdict(text))
    print("ok  B6 a reply that is not a clear accept fails closed (step not run)")


def test_hard_finish_accept_and_refuse():
    # accept: the final message is released only after the human says yes
    final, _, _, asked = run(
        orch=[D("scientist"), D("executor"), HARD("finish", "FINAL MESSAGE with TEST_RMSE=0.0123", "Report and end?")],
        sci=[AIMessage(content="plan")], exe=[tool_call(OK_SCRIPT), AIMessage(content="ok")], replies=["accept"])
    assert asked[0]["step"]["role"] == "finish"
    assert roles(final)[-3:] == ["human", "graph", "orchestrator"] and final["dialogue"][-1]["text"] == "FINAL MESSAGE with TEST_RMSE=0.0123"
    assert final["stop_reason"] == "finish"
    # refuse: the run does not end; the orchestrator decides again
    final, _, _, _ = run(
        orch=[D("scientist"), D("executor"), HARD("finish", "FINAL", "Report and end?"), D("finish", "FINAL, revised")],
        sci=[AIMessage(content="plan")], exe=[tool_call(OK_SCRIPT), AIMessage(content="ok")], replies=["refuse"])
    r = roles(final)
    assert r[-3:] == ["human", "graph", "orchestrator"], r
    orch_texts = [t["text"] for t in final["dialogue"] if t["role"] == "orchestrator"]
    assert "FINAL" not in orch_texts and orch_texts[-1] == "FINAL, revised"      # the refused message was never sent
    print("ok  B7 finish as a hard decision: accept releases the final message, refuse keeps the run going")


def test_finish_guard_still_applies_to_a_hard_finish():
    # the model asks the human to approve finishing before anything was measured: refused by code, human not asked
    final, _, _, asked = run(
        orch=[HARD("finish", "report", "End?"), D("scientist"), D("executor"), D("finish", "report")],
        sci=[AIMessage(content="plan")], exe=[tool_call(OK_SCRIPT), AIMessage(content="ok")])
    assert not asked and "refused by the finish guard" in final["dialogue"][1]["text"]
    print("ok  B8 finish guard also stops a premature 'ask the human to finish' (no pointless question)")


def test_human_decisions_count_toward_the_cap():
    final, _, _, asked = run(orch=[ASK(f"q{i}") for i in range(8)], replies=["a"] * 8)
    assert final["decisions"] == 8 and len(asked) == 8 and "cap" in final["stop_reason"]
    print("ok  B9 asking the human counts as a decision; the 8-decision cap still ends the run")


def test_unusable_human_decision_is_sent_back():
    bad = Decision(next="human", kind="ambiguous_query", question="  ", reason="x")
    bad2 = Decision(next="human", kind="hard_decision", question="Run it?", instruction="go", reason="x")  # no guarded
    final, models, _, asked = run(orch=[bad, bad2, ASK("Which?"), D("general")], gen=[AIMessage(content="a")],
                                  replies=["r"], max_decisions=2)
    assert len(asked) == 1 and asked[0]["question"] == "Which?" and len(models["orchestrator"].calls) == 4
    assert any("rejected" in m.content and "`question`" in m.content for m in models["orchestrator"].calls[1])
    assert any("rejected" in m.content and "`guarded`" in m.content for m in models["orchestrator"].calls[2])
    print("ok  B10 a human decision without a question / without a guarded step is sent back to the model")


def test_json_text_fallback_for_models_without_structured_output():
    js = ('Sure!\n```json\n{"next": "human", "kind": "ambiguous_query", "question": "Which segment?", '
          '"reason": "r", "instruction": null, "guarded": null}\n```')
    final, models, _, asked = run(
        orch=[RuntimeError("404 No endpoints found that support response_format"), AIMessage(content=js),
              AIMessage(content='{"next":"general","instruction":"x","reason":"r"}')],
        gen=[AIMessage(content="a")], replies=["test"], max_decisions=2)
    assert asked and asked[0]["question"] == "Which segment?"
    # sticky: the third call went straight to the text path (no second failed structured attempt)
    assert len(models["orchestrator"].calls) == 3
    print("ok  B11 no structured output on the model: JSON text is used (and stays in use)")


def test_executor_without_tool_calling():
    code = "```python\n" + OK_SCRIPT + "```"
    final, models, wd, _ = run(
        orch=[D("scientist"), D("executor"), D("executor"), D("finish", "x")],
        sci=[AIMessage(content="plan")],
        exe=[RuntimeError("404 No endpoints found that support tool use"), AIMessage(content=code), AIMessage(content="ran it"),
             AIMessage(content=code), AIMessage(content="again")])
    assert final["test_rmse"] == 0.0123 and (wd / "exec_01.py").read_text() == OK_SCRIPT
    assert len(models["executor"].calls) == 5                            # failed tool call, then text mode for good
    print("ok  B12 no tool calling on the model: the fenced script is run instead")


def test_rate_limit_retry_and_daily_quota():
    class E429(Exception):
        status_code = 429
    final, models, _, _ = run(orch=[D("scientist"), D("executor"), D("finish", "x")],
                              sci=[E429("Rate limit exceeded: free-models-per-min"), AIMessage(content="plan")],
                              exe=[tool_call(OK_SCRIPT), AIMessage(content="ok")])
    assert len(models["scientist"].calls) == 2 and final["test_rmse"] == 0.0123     # one retry, then success
    try:
        run(orch=[D("scientist")], sci=[E429("Rate limit exceeded: free-models-per-day")])
        raise AssertionError("daily quota must raise")
    except RuntimeError as e:
        assert "DAILY quota" in str(e)
    print("ok  B13 429 is retried; the daily quota stops the run with a clear message")


def test_evidence_report():
    final, _, _, _ = run(
        orch=[ASK("Which error?"), D("scientist"), HARD("executor", "overwrite results.json", "Overwrite results.json and run?"),
              HARD("executor", "write results_v2.json", "Run with a new file name?"), D("finish", "done")],
        sci=[AIMessage(content="plan")], exe=[tool_call(OK_SCRIPT), AIMessage(content="ok")],
        replies=["test error", "refuse: keep the old file", "accept"])
    ev = H.evidence_md(final["dialogue"])
    assert ev.count("## Pause") == 3 and "Which error?" in ev and "refuse: keep the old file" in ev
    assert "did NOT run" in ev and "step accepted by the human" in ev
    assert "# Agent transcript" in H.transcript_md(final)
    print("ok  B14 evidence report lists question, reply and next message for every pause")


def test_gate_reminder_tracks_what_the_human_answered():
    final, models, _, _ = run(
        orch=[ASK(), D("scientist"), HARD("executor"), HARD("executor", "again"), D("finish", "x")],
        sci=[AIMessage(content="plan")], exe=[tool_call(OK_SCRIPT), AIMessage(content="ok")],
        replies=["test", "refuse", "accept"])
    prompts = [c[-1].content for c in models["orchestrator"].calls]
    assert "answered 0 ambiguity" in prompts[0] and "answered 1 ambiguity" in prompts[1]
    assert "accepted 0 executor" in prompts[3] and "refused 1 step" in prompts[3]
    assert "accepted 1 executor" in prompts[4] and "1 script(s) have run" in prompts[4]
    assert prompts[0].rstrip().endswith("with that step in `guarded`.")          # the checklist is the last thing read
    print("ok  B15 the orchestrator is shown program-counted facts (answers, accepts, refusals, runs) at the end of its input")


if __name__ == "__main__":
    for fn in (test_happy_path, test_premature_finish_is_refused, test_failure_then_repair, test_failed_run_gives_no_rmse,
               test_decision_cap, test_timeout, test_model_text_cannot_fake_rmse,
               test_pause_and_resume_same_thread, test_ambiguous_reading_reaches_the_roles,
               test_hard_decision_accept_runs_the_step, test_hard_decision_refuse_does_not_run_the_step,
               test_refusal_is_visible_to_the_next_decision, test_unclear_reply_counts_as_refusal,
               test_hard_finish_accept_and_refuse, test_finish_guard_still_applies_to_a_hard_finish,
               test_human_decisions_count_toward_the_cap, test_unusable_human_decision_is_sent_back,
               test_json_text_fallback_for_models_without_structured_output, test_executor_without_tool_calling,
               test_rate_limit_retry_and_daily_quota, test_evidence_report,
               test_gate_reminder_tracks_what_the_human_answered):
        fn()
    print("\nall offline checks passed")
