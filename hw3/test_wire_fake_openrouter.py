"""End-to-end check of the whole Homework 3 path WITHOUT a key, a network or a camera.

A tiny local server speaks the OpenAI chat-completions protocol (what OpenRouter exposes) and plays the models.
run_hw3.py is started as a real subprocess with OPENROUTER_BASE_URL pointing at it, so the real ChatOpenAI client, the real
tool-call messages, the real MCP server process, the real interrupt/resume and the real files are used. Only the models
(this script) and the camera (synthetic picture, --fake) are stand-ins. Two modes:

  full  - a model with structured output and tool calling
  free  - a model WITHOUT them (400 for response_format, 404 for tools): the JSON-text fallbacks

    python3 test_wire_fake_openrouter.py

Run 1: a new reference is stored (capture + save_reference, no question).
Run 2 (same working folder, the scene has changed): an ambiguous request (pause 1), then overwriting the stored reference
needs the human (pause 2, accepted); the hardware agent overwrites it.
Run 3 (a scene_action pause): the orchestrator pauses ("switch the room light off"), the person answers "done, light off",
the hardware agent sets the resolution and measures the scene (the synthetic picture gets darker).
Run 4 (the sound experiment, use case 15): the hardware agent plays tones through the (synthetic) loudspeaker and reads the
(synthetic) microphone; the camera is not even opened.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).parent

ORCH_RUN1 = [
    dict(next="hardware", instruction="Capture the scene and store it as the reference base.", reason="A reading is needed."),
    dict(next="finish", instruction="Stored the reference base from a capture of the scene.", reason="The reading is there."),
]
ORCH_RUN2 = [
    dict(next="human", kind="ambiguous_query", question="Do you mean: sharp enough, bright enough, or unchanged since base?",
         reason="The readings differ.", instruction=""),
    dict(next="human", kind="hard_decision", guarded="hardware",
         question="Replacing the stored reference base destroys the old one. Proceed?",
         instruction="Save the scene as reference base with overwrite=true; the human accepted the overwrite.",
         reason="Overwrites a stored reference."),
    dict(next="finish", instruction="The reference base was replaced after the human accepted.", reason="Done."),
]
ORCH_RUN3 = [
    dict(next="human", kind="scene_action", question="Please switch the room light off, then reply 'done, light off'.",
         reason="Only the person can change the light.", instruction=""),
    dict(next="hardware", instruction="The room light is off. Measure the scene at 1280x720.", reason="The light is changed."),
    dict(next="finish", instruction="With the room light off the scene was measured at 1280x720.", reason="The reading is there."),
]
ORCH_RUN4 = [
    dict(next="hardware", instruction="Find the quietest level at which a 1000 Hz tone is heard: try -70 and -40 dBFS.", reason="A reading is needed."),
    dict(next="finish", instruction="The 1000 Hz tone was heard at -40 dBFS and not at -70 dBFS.", reason="The readings are there."),
]
PLAN4 = [("play_tone", {"freq_hz": 1000, "level_db": -70, "label": "quiet"}), ("play_tone", {"freq_hz": 1000, "level_db": -40})]
PLAN1 = [("capture", {"label": "first look", "n_frames": 2}), ("save_reference", {"name": "base"})]
PLAN3 = [("set_resolution", {"width": 1280, "height": 720}), ("measure_region", {"region": "scene", "n_frames": 3})]
PLAN2 = [("save_reference", {"name": "base", "overwrite": True})]


def make_server(mode):
    log = dict(orch=[], n_orch=0, rejected=[], hw_requests=0, bound_tools=[], hw_systems=[])
    state = dict(orch_script=ORCH_RUN1)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _msg(self, body, content=None, tool_calls=None):
            msg = {"role": "assistant", "content": content}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            self._send(200, {"id": "g", "object": "chat.completion", "created": 0, "model": body["model"],
                             "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop", "message": msg}],
                             "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            msgs = body["messages"]
            system = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
            if mode == "free" and body.get("response_format"):
                log["rejected"].append("response_format")
                return self._send(400, {"error": {"message": "Provider returned error: response_format is not supported", "code": 400}})
            if mode == "free" and body.get("tools"):
                log["rejected"].append("tools")
                return self._send(404, {"error": {"message": "No endpoints found that support tool use. Try disabling tools.", "code": 404}})

            if "You are the ORCHESTRATOR" in system:
                assert not body.get("tools"), "the orchestrator must not get tools"
                if mode == "free":
                    assert "JSON object" in msgs[-1]["content"] or "rejected" in msgs[-1]["content"]
                log["orch"].append("\n".join(m["content"] for m in msgs if isinstance(m.get("content"), str)))
                d = state["orch_script"][log["n_orch"]]
                log["n_orch"] += 1
                return self._msg(body, content=json.dumps(d))

            if "You are the HARDWARE agent" in system:
                log["hw_requests"] += 1
                log["hw_systems"].append(system)
                last_user = next(m["content"] for m in reversed(msgs) if m["role"] == "user")
                plan = (PLAN2 if "overwrite=true" in msgs[1]["content"] else PLAN3 if "room light" in msgs[1]["content"]
                        else PLAN4 if "1000 Hz tone" in msgs[1]["content"] else PLAN1)
                got_results = any(m["role"] == "tool" for m in msgs) or any(
                    isinstance(m.get("content"), str) and m["content"].startswith("Tool results:") for m in msgs)
                if body.get("tools"):
                    log["bound_tools"] = [t["function"]["name"] for t in body["tools"]]
                    if got_results:
                        if any(m["role"] == "tool" and '"ok": false' in m["content"] for m in msgs):
                            return self._msg(body, content="A call was rejected.")
                        return self._msg(body, content="Done: the scene was read and the reference handled.")
                    calls = [{"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                             for i, (n, a) in enumerate(plan)]
                    return self._msg(body, tool_calls=calls)
                # text mode
                assert "JSON object" in last_user, last_user[:200]
                if got_results:
                    return self._msg(body, content=json.dumps({"calls": [], "report": "Done (text mode): the reference was handled."}))
                return self._msg(body, content=json.dumps({"calls": [{"tool": n, "args": a} for n, a in plan], "report": ""}))

            return self._msg(body, content="general answer")

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, log, state


def run_cli(tmp, srv, request, replies, use_case=None):
    env = dict(os.environ, OPENROUTER_API_KEY="sk-or-test", NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
               OPENROUTER_BASE_URL=f"http://127.0.0.1:{srv.server_port}/api/v1")
    for k in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "HW3_CAMERA_BACKEND", "HW3_FAKE_STATE", "HW3_ROI_FILE"):
        env.pop(k, None)
    cmd = [sys.executable, str(HERE / "run_hw3.py"), "--fake", "--model", "vendor/free-model:free", "--provider", "openrouter",
           "--runs", str(tmp / "runs"), "--work", str(tmp / "work"), "--roi", str(tmp / "roi.json")]
    cmd += ["--use-case", use_case] if use_case else ["--request", request]
    for r in replies:
        cmd += ["--reply", r]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=240, cwd=tmp)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    run_dir = sorted((tmp / "runs").iterdir())[-1]
    return run_dir, json.loads((run_dir / "state.json").read_text()), p.stdout


def run_mode(mode):
    srv, log, state = make_server(mode)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "roi.json").write_text(json.dumps({"scene": dict(x=400, y=830, w=800, h=200, frame_width=1920, frame_height=1080)}))

    # ---- run 1: a new reference, no question
    run1, st, out = run_cli(tmp, srv, "Take a first look at the scene and store it as reference base.", [])
    roles = [t["role"] for t in st["dialogue"]]
    assert roles == ["user", "orchestrator", "hardware", "orchestrator"], roles
    assert [e["tool"] for e in st["events"]] == ["capture", "save_reference"] and all(e["ok"] for e in st["events"])
    base = tmp / "work" / "references" / "base.png"
    assert base.exists() and st["stop_reason"] == "finish" and st["n_calls"] == 2
    before = base.read_bytes()
    assert "TOOL CALL 1: capture(" in out and "TOOL RESULT 2: reference base saved (new)" in out   # the live lines of the video
    assert (run1 / "images").exists() and any((run1 / "images").glob("*capture_001_scene.png"))
    lines = [json.loads(x) for x in (run1 / "instrument_log.jsonl").read_text().splitlines()]
    assert [x["tool"] for x in lines] == ["camera_info", "audio_info", "capture", "save_reference"], [x["tool"] for x in lines]
    cam_tools = ["camera_info", "set_resolution", "capture", "measure_region", "save_reference", "compare_to_reference",
                 "audio_info", "play_tone", "listen_noise", "compare_audio"]
    if mode == "full":
        assert log["bound_tools"] == cam_tools, log["bound_tools"]

    # ---- run 2: the scene changed; ambiguous request, then the human accepts the overwrite
    (tmp / "work" / "fake_state.json").write_text(json.dumps({"shift": 300}))
    state["orch_script"], log["n_orch"] = ORCH_RUN2, 0
    run2, st, out = run_cli(tmp, srv, "Is the image still good? If not, replace the reference base.",
                            ["sharpness", "accept, replace it"])
    roles = [t["role"] for t in st["dialogue"]]
    assert roles == ["user", "orchestrator", "human", "orchestrator", "human", "graph", "hardware", "orchestrator"], roles
    assert st["decisions"] == 3 and st["refused"] == 0 and st["stop_reason"] == "finish"
    assert len(st["events"]) == 1 and st["events"][0]["result"]["overwritten"] is True
    assert base.read_bytes() != before, "the accepted overwrite replaced the reference"
    assert "accepted by the human: hardware" in st["dialogue"][5]["text"]
    assert "Stored references: base" in log["orch"][-1]                      # the orchestrator saw what is on disk
    assert "accepted an overwrite for this turn: YES" in log["hw_systems"][-1]
    ev = (run2 / "evidence.md").read_text()
    assert ev.count("## Pause") == 2 and "Pause 2: hard_decision" in ev and "accept, replace it" in ev
    assert "OVERWRITTEN" in (run2 / "report.md").read_text()
    if mode == "free":
        assert log["rejected"].count("response_format") == 2 and log["rejected"].count("tools") == 2, log["rejected"]
    else:
        assert not log["rejected"]
    # ---- run 3: a scene_action pause; the person is asked to switch the room light off, answers, the camera measures the darker scene
    (tmp / "work" / "fake_state.json").write_text(json.dumps({"light": 0.5}))
    state["orch_script"], log["n_orch"] = ORCH_RUN3, 0
    run3, st, out = run_cli(tmp, srv, "Measure the scene at 1280x720 after the room light is switched off.", ["done, light off"])
    roles = [t["role"] for t in st["dialogue"]]
    assert roles == ["user", "orchestrator", "human", "orchestrator", "hardware", "orchestrator"], roles
    assert "kind: scene_action" in st["dialogue"][1]["text"] and st["dialogue"][2]["text"] == "done, light off"
    assert [e["tool"] for e in st["events"]] == ["set_resolution", "measure_region"] and all(e["ok"] for e in st["events"])
    rq = st["events"][1]["result"]
    assert rq["resolution"] == [1280, 720] and "mean_brightness" in rq, rq
    assert "Scene actions done by the person (moved something, answered 'done'): 1" in log["orch"][-1]
    assert "GRAPH PAUSED (scene_action)" in out and "TOOL RESULT 2: scene 1280x720" in out
    assert "Pause 1: scene_action" in (run3 / "evidence.md").read_text()
    assert not list(run3.rglob("*main_view*"))
    # ---- run 4: the sound experiment (use case 15): tones through the synthetic loudspeaker, the camera is never opened
    state["orch_script"], log["n_orch"] = ORCH_RUN4, 0
    run4, st, out = run_cli(tmp, srv, "", [], use_case="15")
    roles = [t["role"] for t in st["dialogue"]]
    assert roles == ["user", "orchestrator", "hardware", "orchestrator"], roles
    assert [e["tool"] for e in st["events"]] == ["play_tone", "play_tone"] and all(e["ok"] for e in st["events"])
    heard = [e["result"]["heard"] for e in st["events"]]
    assert heard == [False, True], st["events"]                              # -70 dBFS is under the synthetic noise, -40 is above it
    lines = [json.loads(x) for x in (run4 / "instrument_log.jsonl").read_text().splitlines()]
    assert [x["tool"] for x in lines] == ["audio_info", "play_tone", "play_tone"], [x["tool"] for x in lines]   # no camera call at all
    assert "camera    : not used in this use case" in out and "TOOL RESULT 1: tone 1000 Hz at -70 dBFS" in out and "NOT heard" in out
    assert "Sound readings" in log["hw_systems"][-1] and "play_tone(freq_hz, level_db, seconds, label)" in log["hw_systems"][-1]
    assert "audio_info, play_tone, listen_noise and compare_audio" in log["orch"][-1]
    assert not (run4 / "images").exists() and not list(run4.rglob("*.wav")) and not list((tmp / "work").rglob("*.wav"))
    assert "HEARD" in (run4 / "report.md").read_text() and "NOT heard" in (run4 / "report.md").read_text()
    print(f"ok  wire test, mode={mode}: run 1 stored a reference; run 2 paused twice (ambiguous, overwrite accepted) and "
          f"overwrote it; run 3 paused for a scene_action and measured the scene; run 4 played tones and left the camera off; "
          f"{log['hw_requests']} hardware requests; "
          f"rejected by the server: {log['rejected'] or 'nothing'}")
    srv.shutdown()


if __name__ == "__main__":
    run_mode("full")
    run_mode("free")
    print("\nwire checks passed")
