"""Offline checks of Homework 3 (no API calls, no camera).

    python3 test_hw3_offline.py

The models are scripted stand-ins. The camera is the synthetic backend of camera_connector.py, but everything between the
agent and the camera is the real thing: the MCP server runs as a child process and is spoken to over stdio through the
bridge.

A  connector       sizes, seam, readings, reference overwrite refusal, covered lens
B  MCP server      tools are listed, the camera keeps its state between calls, bad arguments are answered, not fatal
C  agent graph     only `hardware` holds the camera tools; the hardware turn (decide -> command -> reading -> decide);
                   the human gate (ambiguous query; refuse / accept an overwrite); the program blocks an overwrite that the
                   human did not accept; finish guard; call caps; fallbacks for models without tool calling
S  scene_action    the pause that asks the person to change the scene (press a button); the reply is information, not a verdict
"""
import json
import os
import tempfile
from pathlib import Path

os.environ["HW3_CAMERA_BACKEND"] = "fake"
os.environ["HW3_AUDIO"] = "off"             # the sound tools are tested separately (test_audio_offline.py); never use a real speaker here

import numpy as np  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

import camera_connector as cc  # noqa: E402
import hw3_agents as A  # noqa: E402
import llm_utils as U  # noqa: E402
from hw3_agents import Decision, build_graph, drive  # noqa: E402
from mcp_bridge import McpBridge, default_env  # noqa: E402

U.RETRY_SLEEP[:] = [0, 0]          # no waiting in tests
PASSED = []


def ok(name):
    PASSED.append(name)
    print("ok ", name)


def make_controls(tmp: Path, name="controls.json") -> Path:
    """What probe_camera.py would write for a camera whose exposure and gain work (exposure only in manual mode)."""
    p = tmp / name
    p.write_text(json.dumps({"camera": {"index": 0}, "controls": {
        "exposure": {"min": -9, "max": -4, "unit": "log2 s", "needs": {"auto_exposure": 0.25}, "default": -5},
        "gain": {"min": 0, "max": 255, "unit": "driver units", "needs": {"auto_exposure": 0.25}, "default": 0}}}), encoding="utf-8")
    return p


def make_roi(tmp: Path) -> Path:
    p = tmp / "roi.json"
    p.write_text(json.dumps({"scene": dict(x=400, y=830, w=800, h=200, frame_width=1920, frame_height=1080)}))
    return p


# ============================================================================ A: connector
def test_connector():
    tmp = Path(tempfile.mkdtemp())
    os.environ["HW3_ROI_FILE"] = str(make_roi(tmp))
    os.environ["HW3_CONTROLS_FILE"] = str(tmp / "no_controls.json")          # none verified
    state = tmp / "state.json"
    os.environ["HW3_FAKE_STATE"] = str(state)
    cam = cc.open_camera(tmp / "w")
    info = cam.info()
    assert info["frame_size"] == [1920, 1080] and info["layout"] == dict(seam_row=816, seam_found=True)
    assert info["regions"]["scene"] == [400, 830, 800, 200] and info["layout_verified"]
    assert "no camera control" in info["controls"]
    assert cam.set_control("exposure", -5)["ok"] is False and "no verified controls" in cam.set_control("exposure", -5)["error"]
    m = cam.measure_region("scene", 4)
    assert m["ok"] and 100 < m["mean_brightness"] < 255 and m["sharpness_norm"] > 0 and m["n_frames"] == 4
    assert cam.measure_region("nope")["ok"] is False
    bad = cam.set_resolution(800, 600)
    assert bad["ok"] is False and "not offered" in bad["error"]
    r = cam.set_resolution(1280, 720)
    assert r["ok"] and r["accepted"] == [1280, 720]
    m2 = cam.measure_region("scene", 3)
    assert m2["resolution"] == [1280, 720] and m2["region_size"] == [533, 133]       # the region scales with the frame
    assert abs(m2["sharpness_norm"] - m["sharpness_norm"]) < 0.5 * m["sharpness_norm"]
    cam.set_resolution(1920, 1080)
    s1 = cam.save_reference("base")
    s2 = cam.save_reference("base")
    assert s1["ok"] and not s1["overwritten"] and s2["ok"] is False and "already exists" in s2["error"]
    assert cam.save_reference("../evil")["ok"] is False
    same = cam.compare_to_reference("base")
    assert same["ok"] and same["mean_abs_difference"] < 3 and same["correlation"] > 0.99
    state.write_text(json.dumps(dict(covered=True)))
    dark = cam.compare_to_reference("base")
    assert dark["brightness_change"] < -100 and dark["mean_abs_difference"] > 100
    assert cam.layout(cam.grab(1)[0])["seam_found"] is False                       # a dark picture hides the seam
    state.write_text(json.dumps(dict(shift=300)))
    moved = cam.compare_to_reference("base")
    assert moved["correlation"] < same["correlation"] and moved["mean_abs_difference"] > same["mean_abs_difference"]
    assert cam.save_reference("base", overwrite=True)["overwritten"] is True
    cap = cam.capture("t", 2)
    assert cap["saved_crops"] == ["crops/capture_001_scene.png"] and (tmp / "w" / cap["saved_crops"][0]).exists()
    assert not list((tmp / "w").rglob("*main_view*")), "only the measured region is written to disk"
    del os.environ["HW3_FAKE_STATE"]

    class Stubborn(cc.FakeBackend):                       # a driver that only takes a new size when the device is reopened
        n_reopen = 0

        def set_size(self, w, h):
            pass

        def reopen(self, w, h):
            self.n_reopen += 1
            self.size = (w, h)

    sb = Stubborn(0)
    cam2 = cc.PanoCamera(sb, 0, tmp / "w2")
    r = cam2.set_resolution(1280, 720)
    assert r["ok"] and r["accepted"] == [1280, 720] and sb.n_reopen == 1 and "reopened" in r["note"]

    class Dead(cc.FakeBackend):                           # a camera that stops delivering pictures
        reopened_to = None

        def read(self):
            return (False, None) if self.size != (1920, 1080) else super().read()

        def reopen(self, w, h):
            self.reopened_to = (w, h)
            self.size = (w, h)

    db = Dead(0)
    cam3 = cc.PanoCamera(db, 0, tmp / "w3")
    try:
        cam3.set_resolution(640, 480)
        raise AssertionError("should have raised")
    except cc.CameraError as e:
        assert "gave no picture at 640x480" in str(e) and db.reopened_to == (1920, 1080)
    ok("A  connector: sizes, regions, readings, reference refusal/overwrite, covered lens, only the scene crop on disk")


# ============================================================================ B: MCP server through the bridge
def start_bridge(tmp: Path, state: Path | None = None, controls: bool = False) -> McpBridge:
    env = default_env(str(tmp / "work"), roi_file=str(make_roi(tmp)), fake=True, fake_state=str(state) if state else None,
                      controls_file=str(make_controls(tmp) if controls else tmp / "no_controls.json"))
    b = McpBridge(env=env)
    b.start()
    return b


def test_mcp_server():
    tmp = Path(tempfile.mkdtemp())
    b = start_bridge(tmp)
    try:
        assert b.tool_names() == ["camera_info", "set_resolution", "capture", "measure_region", "save_reference",
                                  "compare_to_reference"], b.tool_names()
        specs = {s["function"]["name"]: s["function"] for s in b.tool_specs()}
        assert specs["set_resolution"]["parameters"]["required"] == ["width", "height"]
        assert "overwrite" in specs["save_reference"]["parameters"]["properties"]
        assert json.loads(b.call("set_resolution", {"width": 640, "height": 480}))["accepted"] == [640, 480]
        m = json.loads(b.call("measure_region", {"region": "scene", "n_frames": 2}))
        assert m["resolution"] == [640, 480], "the camera state survives between calls"
        assert json.loads(b.call("measure_region", {"n_frames": "abc"}))["ok"] is False
        assert json.loads(b.call("compare_to_reference", {}))["ok"] is False
        assert json.loads(b.call("no_such_tool"))["ok"] is False
        assert json.loads(b.call("set_resolution", {"width": 1, "height": 1}))["ok"] is False
        log = [json.loads(x) for x in (tmp / "work" / "instrument_log.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [e["tool"] for e in log] == ["set_resolution", "measure_region", "set_resolution"], log   # validation errors never reach a tool
        assert log[0]["args"] == {"width": 640, "height": 480} and log[0]["result"]["ok"] is True
    finally:
        b.close()
    assert json.loads(b.call("camera_info"))["ok"] is False
    ok("B  MCP server over stdio: 7 tools, state kept between calls, bad arguments answered, every call logged")


# ============================================================================ C: agent graph
class Scripted:
    """Stand-in for a chat model: returns queued items (an Exception item is raised) and records what it was asked."""

    def __init__(self, items=()):
        self.items = list(items)
        self.calls = []
        self.bound = []
        self.structured = False

    def bind_tools(self, tools):
        self.bound.extend(tools)
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


def TC(*calls):
    """An AI message that calls tools: TC(("measure_region", {...}), ("capture", {...}))."""
    return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"c{i}"} for i, (n, a) in enumerate(calls)])


def D(nxt, ins="do it", reason="because."):
    return Decision(next=nxt, instruction=ins, reason=reason)


def ASK(q="Which reading?"):
    return Decision(next="human", kind="ambiguous_query", question=q, reason="two readings change the measurement.")


def HARD(ins="save the scene as base, overwriting", q="This overwrites the reference base. Proceed?"):
    return Decision(next="human", kind="hard_decision", guarded="hardware", instruction=ins, question=q,
                    reason="overwriting destroys the old reference.")


MEASURE = ("measure_region", {"region": "scene", "n_frames": 3})


def roles(final):
    return [t["role"] for t in final["dialogue"]]


class Env:
    def __init__(self, tmp=None, bridge=None, work=None, controls=False):
        self.tmp = tmp or Path(tempfile.mkdtemp())
        self.work = work or (self.tmp / "work")
        self.state = self.tmp / "state.json"
        self.bridge = bridge or start_bridge(self.tmp, self.state, controls)

    def run(self, orch, hw=(), sci=(), gen=(), exe=(), replies=(), max_decisions=9, task="task text", curve=None):
        models = dict(orchestrator=Scripted(orch), scientist=Scripted(sci), general=Scripted(gen),
                      hardware=Scripted(hw), executor=Scripted(exe))
        g = build_graph(models, self.bridge, self.work, max_decisions=max_decisions, curve=curve)
        replies, asked = list(replies), []

        def ask(payload):
            asked.append(payload)
            r = replies.pop(0)
            return r(payload) if callable(r) else r          # a callable is "the person does something, then answers"

        final = drive(g, task, ask, thread_id=f"t{id(models)}")
        return final, models, asked

    def close(self):
        self.bridge.close()


def test_happy_path_and_tool_ownership():
    e = Env()
    try:
        final, models, asked = e.run(
            orch=[D("hardware", "measure the scene at 1280x720"), D("finish", "final report")],
            hw=[TC(("set_resolution", {"width": 1280, "height": 720}), MEASURE),
                AIMessage(content="Scene brightness measured at 1280x720.")])
        assert roles(final) == ["user", "orchestrator", "hardware", "orchestrator"], roles(final)
        assert final["stop_reason"] == "finish" and final["decisions"] == 2 and not asked
        assert [x["tool"] for x in final["events"]] == ["set_resolution", "measure_region"] and final["n_calls"] == 2
        assert all(x["ok"] for x in final["events"]) and len(A.readings_of(final["events"])) == 1
        hw_text = final["dialogue"][2]["text"]
        assert "camera tool call 1: set_resolution(width=1280, height=720)" in hw_text and "mean_brightness" in hw_text
        assert hw_text.rstrip().endswith("Scene brightness measured at 1280x720.")
        # the decision to call a tool is a model call of the hardware agent: it saw the instruction, then the results
        hw_model = models["hardware"]
        assert len(hw_model.calls) == 2 and "[orchestrator instruction for you]" in hw_model.calls[0][-1].content
        assert type(hw_model.calls[1][-1]).__name__ == "ToolMessage", "second decision is made after the readings came back"
        # who holds which tools
        cam_tools = {"camera_info", "set_resolution", "capture", "measure_region", "save_reference", "compare_to_reference"}
        assert {s["function"]["name"] for s in hw_model.bound} == cam_tools
        for role in ("orchestrator", "scientist", "general"):
            assert not models[role].bound, role
        assert [s["function"]["name"] for s in models["executor"].bound] == ["run_python"]
        assert models["orchestrator"].structured
        # the numbers in the report are the camera's
        rep = A.final_report_md(final)
        reading = json.loads(final["events"][1]["result"] and json.dumps(final["events"][1]["result"]))
        assert str(reading["mean_brightness"]) in rep and "1280x720" in rep and "final report" in rep
        # and are in the instrument log written by the server
        log = [json.loads(x) for x in (e.work / "instrument_log.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [x["tool"] for x in log] == ["set_resolution", "measure_region"]
    finally:
        e.close()
    ok("C1 hardware turn (decide -> command -> reading -> decide); only `hardware` holds the camera tools; report numbers are the camera's")


def test_finish_guard_and_cap():
    e = Env()
    try:
        final, _, _ = e.run(orch=[D("finish", "done"), D("hardware"), D("finish", "now done")],
                            hw=[TC(MEASURE), AIMessage(content="ok")])
        assert "refused by the finish guard" in final["dialogue"][1]["text"]
        assert roles(final) == ["user", "orchestrator", "orchestrator", "hardware", "orchestrator"], roles(final)
        # late in the budget the guard lets finish through, and the report says nothing was measured
        final2, _, _ = e.run(orch=[D("finish", "x")], max_decisions=2)
        assert final2["stop_reason"] == "finish"
        # decision cap
        final3, _, _ = e.run(orch=[D("general"), D("general")], gen=[AIMessage(content="a"), AIMessage(content="b")],
                             max_decisions=2)
        assert "decision cap" in final3["stop_reason"]
        assert "No reading was taken" in A.final_report_md(final3)
    finally:
        e.close()
    ok("C2 finish guard (needs a camera reading), decision cap, report says plainly when nothing was measured")


def test_ambiguous_query_pause():
    e = Env()
    try:
        final, models, asked = e.run(
            orch=[ASK("Do you mean bright enough, sharp or unchanged?"), D("hardware", "measure sharpness"), D("finish", "report")],
            hw=[TC(MEASURE), AIMessage(content="sharpness measured")], replies=["sharpness"])
        assert len(asked) == 1 and asked[0]["kind"] == "ambiguous_query" and "bright enough" in asked[0]["question"]
        assert roles(final) == ["user", "orchestrator", "human", "orchestrator", "hardware", "orchestrator"], roles(final)
        assert final["dialogue"][2]["text"] == "sharpness"
        seen = models["orchestrator"].calls[1][-1].content
        assert "[human]\nsharpness" in seen and "answered 1 ambiguity question" in seen
        assert "[human]" in models["hardware"].calls[0][-1].content
        assert "Pause 1: ambiguous_query" in A.evidence_md(final["dialogue"])
    finally:
        e.close()
    ok("C3 ambiguous request: graph pauses (interrupt), the same thread resumes, the reading is followed")


def test_overwrite_refused_then_accepted():
    e = Env()
    try:
        # 1. a new reference is saved without any question
        final, _, asked = e.run(orch=[D("hardware", "save the scene as reference base"), D("finish", "saved")],
                                hw=[TC(("save_reference", {"name": "base"})), AIMessage(content="saved base")])
        assert not asked and final["events"][0]["result"]["overwritten"] is False
        png = e.work / "references" / "base.png"
        before = png.read_bytes()
        e.state.write_text(json.dumps(dict(shift=300)))             # the scene changes
        # 2. overwriting needs the human; the human refuses; nothing is sent to the camera; the file is unchanged
        final, models, asked = e.run(
            orch=[HARD(), D("finish", "Nothing was overwritten; the human refused.")], hw=[],
            replies=["refuse, keep the old one"])
        assert asked[0]["kind"] == "hard_decision" and asked[0]["step"]["role"] == "hardware"
        assert final["refused"] == 1 and final["n_calls"] == 0 and final["events"] == []
        assert "did NOT run" in [t for t in final["dialogue"] if t["role"] == "graph"][0]["text"]
        assert roles(final) == ["user", "orchestrator", "human", "graph", "orchestrator"], roles(final)
        assert models["hardware"].calls == [] and png.read_bytes() == before
        assert "base" in models["orchestrator"].calls[0][-1].content          # the orchestrator is told which references exist
        # 3. a reply that is not clearly an accept counts as a refusal
        final, _, _ = e.run(orch=[HARD(), D("finish", "stopped")], replies=["hmm, maybe later"])
        assert final["refused"] == 1 and "not a clear accept" in final["dialogue"][3]["text"] and png.read_bytes() == before
        # 4. accepted: the hardware agent overwrites
        final, models, asked = e.run(
            orch=[HARD(), D("finish", "Overwritten.")],
            hw=[TC(("save_reference", {"name": "base", "overwrite": True})), AIMessage(content="base overwritten")],
            replies=["accept"])
        assert roles(final) == ["user", "orchestrator", "human", "graph", "hardware", "orchestrator"], roles(final)
        assert final["events"][0]["result"]["overwritten"] is True and final["events"][0]["ok"]
        assert png.read_bytes() != before and final["approved_overwrite"] is False        # approval is used up
        assert "accepted by the human: hardware" in final["dialogue"][3]["text"]
        assert "accepted an overwrite for this turn: YES" in models["hardware"].calls[0][0].content
        ev = A.evidence_md(final["dialogue"])
        assert "Pause 1: hard_decision" in ev and "accept" in ev
    finally:
        e.close()
    ok("C4 overwrite of a stored reference: refused (nothing sent, file unchanged) / unclear reply = refusal / accepted (overwritten)")


def test_program_blocks_overwrite_without_accept():
    e = Env()
    try:
        e.run(orch=[D("hardware"), D("finish", "x")], hw=[TC(("save_reference", {"name": "base"})), AIMessage(content="ok")])
        png = e.work / "references" / "base.png"
        before = png.read_bytes()
        e.state.write_text(json.dumps(dict(shift=300)))
        # the orchestrator skips the human gate; the hardware agent asks for the overwrite anyway
        final, models, asked = e.run(
            orch=[D("hardware", "overwrite base"), HARD(), D("finish", "not overwritten")],
            hw=[TC(("save_reference", {"name": "base", "overwrite": True})), AIMessage(content="blocked, report it")],
            replies=["refuse"])
        assert final["events"] == [] and final["n_calls"] == 0, "a blocked call is not sent to the camera"
        assert "NOT SENT" in final["dialogue"][2]["text"] and "needs the human's accept" in final["dialogue"][2]["text"]
        assert png.read_bytes() == before
        # the orchestrator reads the block, finish is refused (nothing measured), and it asks the human as it should
        assert "needs the human's accept" in models["orchestrator"].calls[1][-1].content
        assert roles(final)[:4] == ["user", "orchestrator", "hardware", "orchestrator"] and asked[0]["kind"] == "hard_decision"
        # also when the model writes the flag as a string, and when it first tries without overwrite
        final, _, _ = e.run(
            orch=[D("hardware", "overwrite base"), HARD(), D("finish", "not overwritten")],
            hw=[TC(("save_reference", {"name": "base"}), ("save_reference", {"name": "base", "overwrite": "true"})),
                AIMessage(content="not overwritten")], replies=["refuse"])
        assert [x["ok"] for x in final["events"]] == [False] and "already exists" in final["events"][0]["result"]["error"]
        assert png.read_bytes() == before
    finally:
        e.close()
    ok("C5 the program itself blocks save_reference(overwrite=true) that the human did not accept")


def test_rejected_calls_and_caps():
    e = Env()
    try:
        calls = [("set_resolution", {"width": 800, "height": 600})] + [("camera_info", {})] * (A.MAX_CALLS_PER_TURN + 1)
        final, _, _ = e.run(orch=[D("hardware"), D("finish", "x")], hw=[TC(*calls), AIMessage(content="done")],
                            max_decisions=3)         # no reading is possible here: late in the budget the finish guard yields
        assert len(final["events"]) == A.MAX_CALLS_PER_TURN and final["n_calls"] == A.MAX_CALLS_PER_TURN
        assert final["events"][0]["ok"] is False and "not offered" in final["events"][0]["result"]["error"]
        assert not A.readings_of(final["events"])                        # a rejected call and camera_info are no measurement
        txt = final["dialogue"][2]["text"]
        assert txt.count("NOT SENT") == 2 and f"at most {A.MAX_CALLS_PER_TURN} tool calls per turn" in txt
        # unknown tool: not sent to the server, not an event
        final, _, _ = e.run(orch=[D("hardware"), D("finish", "x")], max_decisions=3,
                            hw=[TC(("set_exposure", {"value": 3})), AIMessage(content="no such tool")])
        assert final["events"] == [] and "unknown tool 'set_exposure'" in final["dialogue"][2]["text"]
    finally:
        e.close()
    ok(f"C6 a rejected call is recorded but is not a reading; at most {A.MAX_CALLS_PER_TURN} tool calls per turn; unknown tools are not sent")


def test_rounds_and_report():
    e = Env()
    try:
        # every round but the last calls tools: the last request is a plain report request
        hw = ([TC(("set_resolution", {"width": 640, "height": 480})), TC(MEASURE)]
              + [TC(("camera_info", {}))] * (A.MAX_ROUNDS - 3) + [AIMessage(content="final summary")])
        final, models, _ = e.run(orch=[D("hardware"), D("finish", "x")], hw=hw)
        assert len(models["hardware"].calls) == A.MAX_ROUNDS and final["n_calls"] == min(A.MAX_ROUNDS - 1, A.MAX_CALLS_PER_TURN)
        assert "final summary" in final["dialogue"][2]["text"]
        assert "tools are not available" in models["hardware"].calls[-1][-1].content
    finally:
        e.close()
    ok(f"C7 at most {A.MAX_ROUNDS} model requests per hardware turn; the last one only writes the report")


def test_fallbacks():
    e = Env()
    try:
        # (a) no tool calling at all: JSON list of calls, then results go back as a user turn, then a report
        hw = [Exception("tool use is not supported by this endpoint"),
              AIMessage(content='{"calls": [{"tool": "set_resolution", "args": {"width": 1280, "height": 720}}, '
                                '{"tool": "measure_region", "args": {"region": "scene", "n_frames": 2}}], "report": ""}'),
              AIMessage(content='{"calls": [], "report": "measured at 1280x720"}')]
        final, models, _ = e.run(orch=[D("hardware"), D("finish", "x")], hw=hw)
        assert [x["tool"] for x in final["events"]] == ["set_resolution", "measure_region"]
        assert final["dialogue"][2]["text"].rstrip().endswith("measured at 1280x720")
        assert "Tool results:" in models["hardware"].calls[-1][-2].content
        # (b) a call printed as text instead of in tool_calls
        hw = [AIMessage(content="<function=measure_region><parameter=region>scene</parameter><parameter=n_frames>2</parameter></function>"),
              AIMessage(content="done")]
        final, _, _ = e.run(orch=[D("hardware"), D("finish", "x")], hw=hw)
        assert final["events"][0]["args"] == {"region": "scene", "n_frames": 2} and final["events"][0]["ok"]
        # (c) the model answers without a tool: one nudge, then the call
        hw = [AIMessage(content="I will measure now."), TC(MEASURE), AIMessage(content="done")]
        final, models, _ = e.run(orch=[D("hardware"), D("finish", "x")], hw=hw)
        assert len(final["events"]) == 1 and "without calling a tool" in models["hardware"].calls[1][-1].content
        # (d) the orchestrator model has no structured output: JSON text is asked for
        orch = Scripted([Exception("response_format is not supported"),
                         AIMessage(content='{"next": "hardware", "instruction": "measure", "reason": "r"}'),
                         AIMessage(content='{"next": "finish", "instruction": "done", "reason": "r"}')])
        models = dict(orchestrator=orch, scientist=Scripted(), general=Scripted(),
                      hardware=Scripted([TC(MEASURE), AIMessage(content="ok")]), executor=Scripted())
        g = build_graph(models, e.bridge, e.work)
        final = drive(g, "t", lambda p: "", thread_id="fb")
        assert final["stop_reason"] == "finish" and len(final["events"]) == 1
        # (e) an unusable decision (human without a question) is sent back
        bad = Decision(next="human", kind="ambiguous_query", question="", reason="r")
        final, models, _ = e.run(orch=[bad, D("hardware"), D("finish", "x")], hw=[TC(MEASURE), AIMessage(content="ok")])
        assert "was rejected" in models["orchestrator"].calls[1][-1].content and final["decisions"] == 2
    finally:
        e.close()
    ok("C8 fallbacks: no tool calling (JSON calls), call printed as text, nudge, no structured output, unusable decision")


def test_executor_and_other_roles():
    e = Env()
    try:
        code = "import json\nprint(sum(1 for _ in open('instrument_log.jsonl')))\n"
        final, models, _ = e.run(
            orch=[D("hardware"), D("scientist"), D("general"), D("executor", "count the log lines"), D("finish", "x")],
            hw=[TC(MEASURE), AIMessage(content="ok")],
            sci=[AIMessage(content="compare sharpness_norm, not sharpness")], gen=[AIMessage(content="sharpness is ...")],
            exe=[TC(("run_python", {"code": code, "purpose": "count"})), AIMessage(content="1 line")])
        assert roles(final) == ["user", "orchestrator", "hardware", "orchestrator", "scientist", "orchestrator", "general",
                                "orchestrator", "executor", "orchestrator"], roles(final)
        assert "exit code: 0" in final["dialogue"][8]["text"] and "\n1\n" in final["dialogue"][8]["text"]
        assert (e.work / "exec_01.py").exists()
        assert "instrument tool calls so far" in models["scientist"].calls[0][-1].content.lower()
        assert not models["scientist"].bound and not models["general"].bound
    finally:
        e.close()
    ok("C9 scientist, general and executor take their turns; the executor reads files only (run_python), no camera tools")


def test_instrument_error_is_not_fatal():
    e = Env()
    try:
        e.bridge.close()                                               # the camera server dies
        final, _, _ = e.run(orch=[D("hardware"), D("finish", "the camera server is down")], max_decisions=3,
                            hw=[TC(MEASURE), AIMessage(content="the camera could not be reached")])
        assert final["events"][0]["ok"] is False and not A.readings_of(final["events"])
        assert "not running" in final["events"][0]["result"]["error"] and final["stop_reason"] == "finish"
        assert "No reading was taken" in A.final_report_md(final)
    finally:
        pass
    ok("C10 a dead camera server gives a failed tool result, not a crash")


def test_controls_connector_and_server():
    tmp = Path(tempfile.mkdtemp())
    os.environ["HW3_ROI_FILE"] = str(make_roi(tmp))
    os.environ["HW3_CONTROLS_FILE"] = str(make_controls(tmp))
    os.environ.pop("HW3_FAKE_STATE", None)
    cam = cc.open_camera(tmp / "w")
    info = cam.info()
    assert set(info["controls"]["verified"]) == {"exposure", "gain"} and info["controls"]["verified"]["exposure"]["current"] == -5.0
    b0 = cam.measure_region("scene", 3)["mean_brightness"]
    r = cam.set_control("exposure", -8)
    assert r["ok"] and r["read_back"] == -8 and r["before"] == -5 and r["unit"] == "log2 s"
    assert cam.b.get_prop("auto_exposure") == 0.25, "the prerequisite (manual exposure) was applied"
    b1 = cam.measure_region("scene", 3)["mean_brightness"]
    assert b1 < 0.4 * b0, (b0, b1)                                           # a darker exposure gives a darker picture
    assert cam.set_control("exposure", -2)["ok"] is False and "between -9 and -4" in cam.set_control("exposure", -2)["error"]
    assert "not a number" in cam.set_control("gain", "lots")["error"] and "unknown control" in cam.set_control("focus", 1)["error"]
    cam.set_control("gain", 200)
    assert cam.measure_region("scene", 3)["mean_brightness"] > b1
    rr = cam.reset_controls()
    assert rr["ok"] and rr["restored"]["exposure"] == -5 and cam.b.get_prop("auto_exposure") == 0.75 and cam.b.get_prop("gain") == 0
    assert abs(cam.measure_region("scene", 3)["mean_brightness"] - b0) < 3
    assert cam.reset_controls()["ok"]
    # the MCP server offers set_control / reset_controls only when controls are verified
    b = start_bridge(tmp, controls=True)
    try:
        assert b.tool_names() == ["camera_info", "set_resolution", "capture", "measure_region", "save_reference",
                                  "compare_to_reference", "set_control", "reset_controls"], b.tool_names()
        spec = {sp["function"]["name"]: sp["function"] for sp in b.tool_specs()}
        assert "Verified controls: exposure -9..-4 (log2 s), gain 0..255 (driver units)" in spec["set_control"]["description"]
        assert json.loads(b.call("set_control", {"name": "exposure", "value": -7}))["ok"] is True
        assert json.loads(b.call("set_control", {"name": "exposure", "value": -1}))["ok"] is False
        assert json.loads(b.call("set_control", {"name": "exposure"}))["ok"] is False                  # missing argument
        log = [json.loads(x) for x in (tmp / "work" / "instrument_log.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [e["tool"] for e in log] == ["set_control", "set_control"] and log[0]["result"]["read_back"] == -7
    finally:
        b.close()
    ok("A2 controls: set_control validates range/name, applies manual-exposure prerequisite, changes the picture, "
       "reset_controls restores; the MCP server offers the two tools only when controls.json lists verified controls")


def test_agent_with_controls():
    e = Env(controls=True)
    try:
        hw = [TC(("set_control", {"name": "exposure", "value": -8}), ("measure_region", {"region": "scene", "n_frames": 3})),
              TC(("set_control", {"name": "exposure", "value": -6}), ("measure_region", {"region": "scene", "n_frames": 3})),
              TC(("set_control", {"name": "exposure", "value": 3}), ("reset_controls", {})),
              AIMessage(content="exposure -6 is closest to 128; controls restored.")]
        final, models, _ = e.run(orch=[D("hardware", "find the exposure closest to 128 grey levels"), D("finish", "done")], hw=hw)
        ev = final["events"]
        assert [x["tool"] for x in ev] == ["set_control", "measure_region", "set_control", "measure_region", "set_control",
                                           "reset_controls"], [x["tool"] for x in ev]
        assert [x["ok"] for x in ev] == [True, True, True, True, False, True]          # exposure 3 is outside the verified range
        assert "between -9 and -4" in ev[4]["result"]["error"] and len(A.readings_of(ev)) == 2
        # the prompts follow the instrument: the verified controls are named, the rules for controls are given
        sys0 = models["hardware"].calls[0][0].content
        assert "exposure -9..-4 (log2 s), gain 0..255 (driver units)" in sys0 and "set_control(name, value)" in sys0
        assert "reset_controls before you stop" in sys0 and "No camera control is available" not in sys0
        orch0 = models["orchestrator"].calls[0][0].content
        assert "verified on this camera: exposure -9..-4" in orch0
        assert {s["function"]["name"] for s in models["hardware"].bound} >= {"set_control", "reset_controls"}
        rep = A.final_report_md(final)
        assert "set_control" in rep and "requested -8, the camera read back -8" in rep and "REJECTED" in rep
        # after the run the camera is back to its original settings
        assert json.loads(e.bridge.call("camera_info"))["controls"]["verified"]["exposure"]["current"] == -5.0
    finally:
        e.close()
    # and without verified controls the prompts say so and no control tool exists
    e = Env()
    try:
        final, models, _ = e.run(orch=[D("hardware"), D("finish", "x")], hw=[TC(MEASURE), AIMessage(content="ok")])
        assert "No camera control is available" in models["hardware"].calls[0][0].content
        assert "set_control" not in {s["function"]["name"] for s in models["hardware"].bound}
        assert "set_control(name, value)" not in models["hardware"].calls[0][0].content
    finally:
        e.close()
    ok("C11 controls reach the agent: tools, ranges and rules appear in the prompts only when verified; out-of-range "
       "values are rejected by the server; reset_controls restores; without controls the agent is told there are none")


def make_uvc_controls(tmp: Path) -> Path:
    """What probe_camera.py writes on a Mac with uvc-util: the three controls that change the picture."""
    p = tmp / "controls_uvc.json"
    p.write_text(json.dumps({"camera": {"index": 0, "platform": "darwin", "via": "uvc-util"}, "controls": {
        "brightness": {"min": -64, "max": 64, "step": 1, "unit": "camera units (UVC)", "needs": {}, "default": 0, "via": "uvc-util"},
        "contrast": {"min": 0, "max": 100, "step": 1, "unit": "camera units (UVC)", "needs": {}, "default": 50, "via": "uvc-util"},
        "saturation": {"min": 0, "max": 100, "step": 1, "unit": "camera units (UVC)", "needs": {}, "default": 50, "via": "uvc-util"}}}),
        encoding="utf-8")
    return p


class UvcEnv:
    """A fake uvc-util on PATH-like env vars (HW3_UVC_UTIL) + its state file; removed again afterwards."""

    def __enter__(self):
        import fake_uvc_util
        self.tmp = Path(tempfile.mkdtemp())
        self.keys = ("HW3_UVC_UTIL", "HW3_FAKE_UVC_STATE", "HW3_CONTROLS_FILE", "HW3_ROI_FILE", "HW3_FAKE_STATE")
        self.old = {k: os.environ.get(k) for k in self.keys}
        os.environ["HW3_UVC_UTIL"] = fake_uvc_util.make_launcher(self.tmp / "uvc-util")
        os.environ["HW3_FAKE_UVC_STATE"] = str(self.tmp / "uvc_state.json")
        os.environ["HW3_CONTROLS_FILE"] = str(make_uvc_controls(self.tmp))
        os.environ["HW3_ROI_FILE"] = str(make_roi(self.tmp))
        os.environ.pop("HW3_FAKE_STATE", None)
        cc.FakeBackend._cache.clear()
        return self

    def state(self) -> dict:
        p = self.tmp / "uvc_state.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_uvc_controls_end_to_end():
    import time
    with UvcEnv() as u:
        cam = cc.open_camera(u.tmp / "w")
        info = cam.info()
        assert "uvc-util" in info["interface"] and set(info["controls"]["verified"]) == {"brightness", "contrast", "saturation"}
        assert info["controls"]["verified"]["brightness"]["current"] == 0
        m0 = cam.measure_region("scene", 3)
        assert "colour_saturation" in m0 and m0["rms_contrast"] > 5 and m0["colour_saturation"] > 20, m0
        r = cam.set_control("brightness", 30.4)                                   # whole numbers only: 30.4 -> 30
        assert r["ok"] and r["requested"] == 30 and r["read_back"] == 30 and "rounded from 30.4" in r["note"], r
        assert u.state()["brightness"] == 30                                      # the command really went through the tool
        m1 = cam.measure_region("scene", 3)
        assert 27 < m1["mean_brightness"] - m0["mean_brightness"] < 33, (m0["mean_brightness"], m1["mean_brightness"])
        cam.set_control("contrast", 25)                                           # half the contrast: half the spread of grey levels
        assert cam.measure_region("scene", 3)["rms_contrast"] < 0.7 * m0["rms_contrast"]
        cam.set_control("saturation", 0)
        assert cam.measure_region("scene", 3)["colour_saturation"] < 3
        before = dict(u.state())
        bad = cam.set_control("brightness", 90)
        assert bad["ok"] is False and "between -64 and 64" in bad["error"] and u.state() == before     # refused before the tool
        assert cam.set_control("hue", 5)["ok"] is False and "unknown control" in cam.set_control("hue", 5)["error"]
        rr = cam.reset_controls()
        assert rr["ok"] and rr["restored"] == {"brightness": 0, "contrast": 50, "saturation": 50}, rr
        st = u.state()
        assert (st["brightness"], st["contrast"], st["saturation"]) == (0, 50, 50)
        m2 = cam.measure_region("scene", 3)
        assert abs(m2["mean_brightness"] - m0["mean_brightness"]) < 2.5 and abs(m2["colour_saturation"] - m0["colour_saturation"]) < 3

        # through the MCP server (a child process): the environment reaches it, the description lists the ranges,
        # and when the server ends the settings are put back even though the agent never called reset_controls
        env = default_env(str(u.tmp / "work2"), roi_file=str(make_roi(u.tmp)), fake=True, controls_file=str(make_uvc_controls(u.tmp)))
        b = McpBridge(env=env)
        b.start()
        try:
            spec = {sp["function"]["name"]: sp["function"] for sp in b.tool_specs()}
            assert "brightness -64..64 (camera units (UVC)), contrast 0..100" in spec["set_control"]["description"]
            res = json.loads(b.call("set_control", {"name": "brightness", "value": 20}))
            assert res["ok"] and res["read_back"] == 20, res
            res = json.loads(b.call("set_control", {"name": "saturation", "value": 80}))
            assert res["ok"] and u.state()["brightness"] == 20 and u.state()["saturation"] == 80, u.state()
            mm = json.loads(b.call("measure_region", {"region": "scene", "n_frames": 2}))
            assert "colour_saturation" in mm and mm["ok"]
        finally:
            b.close()
        for _ in range(40):                                                       # the server needs a moment to end
            if u.state().get("brightness") == 0 and u.state().get("saturation") == 50:
                break
            time.sleep(0.1)
        assert u.state()["brightness"] == 0 and u.state()["saturation"] == 50, u.state()
    ok("A3 controls through uvc-util (macOS path, fake tool): whole-number rounding, range refusal before the tool is called, brightness / "
       "contrast / saturation each move their own reading, reset restores, the server puts the settings back when it ends")


def test_agent_with_uvc_brightness():
    with UvcEnv() as u:
        bridge = McpBridge(env=default_env(str(u.tmp / "work"), roi_file=str(make_roi(u.tmp)), fake=True,
                                           controls_file=str(make_uvc_controls(u.tmp))))
        bridge.start()
        e = Env(tmp=u.tmp, bridge=bridge)
        try:
            hw = [TC(("measure_region", {"region": "scene", "n_frames": 3}), ("set_control", {"name": "brightness", "value": 30})),
                  TC(("measure_region", {"region": "scene", "n_frames": 3}), ("set_control", {"name": "brightness", "value": 200})),
                  TC(("reset_controls", {})),
                  AIMessage(content="brightness 30 raised the scene by about 30 grey levels; controls restored.")]
            final, models, _ = e.run(orch=[D("hardware", "make the scene 30 grey levels brighter with the brightness control"),
                                           D("finish", "done")], hw=hw)
            ev = final["events"]
            assert [x["tool"] for x in ev] == ["measure_region", "set_control", "measure_region", "set_control", "reset_controls"]
            assert [x["ok"] for x in ev] == [True, True, True, False, True] and "between -64 and 64" in ev[3]["result"]["error"]
            gain = ev[2]["result"]["mean_brightness"] - ev[0]["result"]["mean_brightness"]
            assert 27 < gain < 33, gain
            sys0 = models["hardware"].calls[0][0].content
            assert "brightness -64..64 (camera units (UVC)), contrast 0..100" in sys0 and "rms_contrast" in sys0
            rep = A.final_report_md(final)
            assert "requested 30, the camera read back 30" in rep and "contrast" in rep
            assert u.state()["brightness"] == 0                                    # reset_controls was called by the agent
        finally:
            e.close()
    ok("C12 hardware turn with the uvc-util camera: brightness +30 measured before and after, an out-of-range value refused, controls restored")


def make_screen_controls(tmp: Path) -> Path:
    """What probe_camera.py --screen writes: the monitor patch is the verified control."""
    p = tmp / "controls_screen.json"
    p.write_text(json.dumps({"camera": {"index": 0, "platform": "darwin", "via": "screen"}, "controls": {
        "screen_level": {"min": 0, "max": 255, "step": 1, "unit": "grey level of the patch on the monitor (0 black, 255 white)",
                         "needs": {}, "via": "screen", "default": 128, "settle_s": 0}}}), encoding="utf-8")
    return p


class ScreenEnv:
    def __enter__(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.keys = ("HW3_CONTROLS_FILE", "HW3_ROI_FILE", "HW3_FAKE_STATE", "HW3_FAKE_SCREEN", "HW3_UVC_UTIL")
        self.old = {k: os.environ.get(k) for k in self.keys}
        os.environ["HW3_CONTROLS_FILE"] = str(make_screen_controls(self.tmp))
        os.environ["HW3_ROI_FILE"] = str(make_roi(self.tmp))
        for k in ("HW3_FAKE_STATE", "HW3_FAKE_SCREEN", "HW3_UVC_UTIL"):
            os.environ.pop(k, None)
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_screen_level_connector_and_server():
    import time
    with ScreenEnv() as u:
        cam = cc.open_camera(u.tmp / "w")
        assert cam.screen is not None and cam.screen.kind == "fake"
        info = cam.info()
        assert set(info["controls"]["verified"]) == {"screen_level"} and info["controls"]["verified"]["screen_level"]["current"] == 128
        assert "monitor" in info["interface"]
        m128 = cam.measure_region("scene", 3)["mean_brightness"]
        assert cam.set_control("screen_level", 0)["ok"] and cam.set_control("screen_level", 0)["read_back"] == 0
        m0 = cam.measure_region("scene", 3)["mean_brightness"]
        r = cam.set_control("screen_level", 254.6)
        assert r["ok"] and r["requested"] == 255 and "rounded" in r["note"], r
        m255 = cam.measure_region("scene", 3)["mean_brightness"]
        assert m0 < m128 < m255 and m255 - m0 > 150, (m0, m128, m255)                     # the camera follows the patch
        assert cam.set_control("screen_level", 300)["ok"] is False and "between 0 and 255" in cam.set_control("screen_level", 300)["error"]
        rr = cam.reset_controls()
        assert rr["ok"] and rr["restored"] == {"screen_level": 128}, rr
        assert abs(cam.measure_region("scene", 3)["mean_brightness"] - m128) < 1.5
        cam.close()
        assert not Path(os.environ["HW3_FAKE_SCREEN"]).exists()                           # the patch is gone with the camera
        os.environ.pop("HW3_FAKE_SCREEN", None)

        # through the MCP server: description, a command, and the patch is put away when the server ends
        env = default_env(str(u.tmp / "work2"), roi_file=str(make_roi(u.tmp)), fake=True, controls_file=str(make_screen_controls(u.tmp)))
        b = McpBridge(env=env)
        b.start()
        try:
            spec = {sp["function"]["name"]: sp["function"] for sp in b.tool_specs()}
            assert "screen_level 0..255 (grey level of the patch on the monitor (0 black, 255 white))" in spec["set_control"]["description"]
            assert json.loads(b.call("set_control", {"name": "screen_level", "value": 200}))["ok"] is True
            mm = json.loads(b.call("measure_region", {"region": "scene", "n_frames": 2}))
            assert mm["ok"] and mm["mean_brightness"] > m128
            patch = u.tmp / "work2" / "screen" / "fake_screen.json"
            assert json.loads(patch.read_text(encoding="utf-8"))["level"] == 200
        finally:
            b.close()
        for _ in range(40):
            if not patch.exists():
                break
            time.sleep(0.1)
        assert not patch.exists()
    ok("A4 screen_level (monitor patch as a control): range check, rounding, the camera reading follows the level, reset restores, "
       "the patch is put away when the camera or the server ends")


def test_screen_window_process():
    """The real window backend (a separate process, files for commands and answers), run without a screen (HW3_VIEWER_HEADLESS)."""
    with ScreenEnv() as u:
        old = {k: os.environ.get(k) for k in ("HW3_SCREEN_BACKEND", "HW3_VIEWER_HEADLESS")}
        os.environ["HW3_SCREEN_BACKEND"], os.environ["HW3_VIEWER_HEADLESS"] = "window", "1"
        try:
            cam = cc.open_camera(u.tmp / "w")
            assert cam.screen.kind == "window" and cam.screen.proc.poll() is None
            r = cam.set_control("screen_level", 200)
            assert r["ok"] and r["read_back"] == 200 and r["before"] == 128, r
            ack = json.loads((u.tmp / "w" / "screen" / "ack.json").read_text(encoding="utf-8"))
            assert ack["level"] == 200 and ack["seq"] == cam.screen.seq                       # the window confirmed this very command
            assert cam.reset_controls()["restored"] == {"screen_level": 128}
            cam.screen.proc.kill()                                                           # the window process dies
            cam.screen.proc.wait()
            try:
                cam.set_control("screen_level", 10)
                raise AssertionError("a dead window must be an error")
            except cc.CameraError as e:
                assert "window process ended" in str(e), e
            proc = cam.screen.proc
            cam.close()
            assert proc.poll() is not None
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    ok("A5 monitor window as a separate process: every command is confirmed by the window before the tool answers; "
       "a dead window is an error, not a silent wrong reading")


def test_agent_with_screen_level():
    with ScreenEnv() as u:
        bridge = McpBridge(env=default_env(str(u.tmp / "work"), roi_file=str(make_roi(u.tmp)), fake=True,
                                           controls_file=str(make_screen_controls(u.tmp))))
        bridge.start()
        e = Env(tmp=u.tmp, bridge=bridge)
        try:
            hw = [TC(("set_control", {"name": "screen_level", "value": 0}), ("measure_region", {"region": "scene", "n_frames": 3}),
                     ("set_control", {"name": "screen_level", "value": 128}), ("measure_region", {"region": "scene", "n_frames": 3})),
                  TC(("set_control", {"name": "screen_level", "value": 255}), ("measure_region", {"region": "scene", "n_frames": 3}),
                     ("set_control", {"name": "screen_level", "value": 999}), ("reset_controls", {})),
                  AIMessage(content="The reading grows with the level; the camera does not saturate at 255; screen put back.")]
            final, models, _ = e.run(orch=[D("hardware", "measure the camera response at the screen levels 0, 128 and 255"),
                                           D("finish", "done")], hw=hw)
            ev = final["events"]
            assert [x["tool"] for x in ev] == ["set_control", "measure_region", "set_control", "measure_region", "set_control",
                                               "measure_region", "set_control", "reset_controls"]
            assert [x["ok"] for x in ev] == [True] * 6 + [False, True] and "between 0 and 255" in ev[6]["result"]["error"]
            means = [ev[i]["result"]["mean_brightness"] for i in (1, 3, 5)]
            assert means == sorted(means) and means[2] - means[0] > 150, means
            sys0 = models["hardware"].calls[0][0].content
            assert "screen_level sets the grey level" in sys0 and "lies on a grey patch" in sys0
            assert "contrast spreads" not in sys0                                         # rules only for the controls that exist
            assert "screen_level 0..255" in models["orchestrator"].calls[0][0].content
            assert "screen_level" in A.final_report_md(final)
        finally:
            e.close()
    ok("C14 hardware turn with the monitor patch: three levels commanded, three readings (growing with the level), an out-of-range "
       "level refused, the screen put back; the prompts name only the controls that exist")


# ============================================================================ S: the scene_action pause
def test_scene_action_is_not_a_verdict():
    e = Env()
    try:
        bad = Decision(next="human", kind="scene_action", question="", reason="r")
        assert "non-empty `question`" in A.decision_problem(bad)
        norm = A.normalize(Decision(next="human", kind="scene_action", question="q", guarded="hardware", reason="r"))
        assert norm.kind == "scene_action" and norm.guarded is None
        ask = Decision(next="human", kind="scene_action", question="Press the mute button of the device, then reply done.", reason="r")
        # an unusable decision is sent back; a reply such as "no" is just information (it is not accept / refuse of a hard decision)
        final, models, asked = e.run(orch=[bad, ask, D("hardware", "measure the scene at 1920x1080"), D("finish", "x")],
                                     hw=[TC(("measure_region", {"n_frames": 2})), AIMessage(content="measured")], replies=["no, I cannot press it"])
        assert "was rejected" in models["orchestrator"].calls[1][-1].content and len(asked) == 1
        assert final["refused"] == 0 and "graph" not in roles(final), roles(final)
        assert "[human]\nno, I cannot press it" in models["orchestrator"].calls[2][-1].content
    finally:
        e.close()
    ok("S1 scene_action: an empty question is rejected, `guarded` is dropped, and the person's reply is information for the orchestrator "
       "(it is not an accept or a refusal)")


def test_use_cases_and_runner_text():
    import run_hw3
    assert sorted(run_hw3.USE_CASES, key=int) == [str(i) for i in list(range(1, 12)) + list(range(15, 20))]       # 12-14 (QR) were removed
    for k, control in run_hw3.NEEDS.items():
        assert control in run_hw3.USE_CASES[k][1].lower(), (k, control)
    assert "exposure" not in " ".join(v[1].lower() for v in run_hw3.USE_CASES.values())
    import re
    for k in ("1", "2", "3", "4"):                  # each of these makes the agent command the camera's resolution, then measure
        assert re.search(r"\d{3,4}x\d{3,4}|resolution", run_hw3.USE_CASES[k][1]), k
    assert "ambiguous" == run_hw3.USE_CASES["5"][0] and run_hw3.USE_CASES["5"][1] == "Is the image good?"       # the pause test stays ambiguous
    ok("C13 use cases 6-11 name the control they need (screen_level for 6-8, the camera's own controls for 9-11); the runner knows which each needs")


if __name__ == "__main__":
    test_connector()
    test_controls_connector_and_server()
    test_mcp_server()
    test_happy_path_and_tool_ownership()
    test_finish_guard_and_cap()
    test_ambiguous_query_pause()
    test_overwrite_refused_then_accepted()
    test_program_blocks_overwrite_without_accept()
    test_rejected_calls_and_caps()
    test_rounds_and_report()
    test_fallbacks()
    test_executor_and_other_roles()
    test_instrument_error_is_not_fatal()
    test_agent_with_controls()
    test_uvc_controls_end_to_end()
    test_agent_with_uvc_brightness()
    test_scene_action_is_not_a_verdict()
    test_screen_level_connector_and_server()
    test_screen_window_process()
    test_agent_with_screen_level()
    test_use_cases_and_runner_text()
    print(f"\nall {len(PASSED)} checks passed")
