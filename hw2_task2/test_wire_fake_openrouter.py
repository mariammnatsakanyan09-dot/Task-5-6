"""End-to-end check of the OpenRouter path WITHOUT a key or network.

A tiny local server speaks the OpenAI chat-completions protocol (what OpenRouter exposes) and plays the models.
run_task2.py is started as a real subprocess with OPENROUTER_BASE_URL pointing at it, so the real ChatOpenAI client, the
real tool calls (JSON schema of focus_search included), the real bench and the real report builder run. Two modes:

  full  - a model with structured output and tool calling
  free  - a model WITHOUT them (400 for response_format, 404 for tools): the JSON-text fallbacks

    python test_wire_fake_openrouter.py

The play: the scientist writes a plan; the executor first makes a call with an invalid macropixel size (the tool rejects it),
repairs it in the same turn and runs two experiments, then runs two more in a second turn; the orchestrator refuses a
premature finish (the guard) and finishes after four successful experiments.
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

BASE = dict(focus_size=3, bg_gap=2, bg_width=6, n_bg_regions=8)
BAD = dict(BASE, macropixel_size=3, iterations=300)
TURN1 = [dict(BASE, macropixel_size=16, iterations=300), dict(BASE, macropixel_size=8, iterations=800)]
TURN2 = [dict(BASE, macropixel_size=4, iterations=3000), dict(BASE, macropixel_size=4, iterations=3000, n_avg=2)]

ORCH = [
    dict(next="scientist", instruction="Plan a short campaign that varies macropixel size and iterations.", reason="No plan yet."),
    dict(next="executor", instruction="Run the first experiments of the plan.", reason="A plan exists."),
    dict(next="finish", instruction="Premature final message.", reason="Looks done."),                 # the guard refuses this
    dict(next="executor", instruction="Run the finer experiments: 256 macropixels.", reason="More macropixels, more iterations."),
    dict(next="finish", instruction="FINAL: the best enhancement is in the table.", reason="Four experiments done."),
]


def make_server(mode):
    log = dict(orch_prompts=[], orch_user=[], rejected=[], n_orch=0, exec_turns=0, repaired=False, tool_schema=None)

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
                if mode == "free":
                    assert "JSON object" in msgs[-1]["content"], "text mode must ask for JSON"
                log["orch_prompts"].append(system)
                log["orch_user"].append(msgs[-2]["content"] if mode == "free" else msgs[-1]["content"])
                d = ORCH[log["n_orch"]]
                log["n_orch"] += 1
                return self._msg(body, content=json.dumps(d))
            if "You are the SCIENTIST" in system:
                return self._msg(body, content="PLAN: 1) 16 macropixels, 300 iterations; 2) 64 macropixels; 3) 256 macropixels with more iterations.")
            if "You are the EXECUTOR" in system:
                if body.get("tools"):
                    log["tool_schema"] = {t["function"]["name"]: t["function"]["parameters"] for t in body["tools"]}
                    tool_msgs = [m for m in msgs if m["role"] == "tool"]
                    mk = lambda args, i: {"id": f"call_{log['exec_turns']}_{i}", "type": "function",
                                          "function": {"name": "focus_search", "arguments": json.dumps(args)}}
                    if not tool_msgs:
                        log["exec_turns"] += 1
                        calls = [BAD] if log["exec_turns"] == 1 else TURN2
                        return self._msg(body, tool_calls=[mk(c, i) for i, c in enumerate(calls)])
                    if "REJECTED" in tool_msgs[-1]["content"] and not log["repaired"]:
                        log["repaired"] = True
                        return self._msg(body, tool_calls=[mk(c, i) for i, c in enumerate(TURN1)])
                    return self._msg(body, content="The experiments of this turn ran; the enhancements are in the tool results above.")
                # text mode: one JSON list of experiments
                assert "JSON object" in msgs[-1]["content"]
                log["exec_turns"] += 1
                calls = [BAD] + TURN1 if log["exec_turns"] == 1 else TURN2
                return self._msg(body, content=json.dumps({"calls": calls}))
            return self._msg(body, content="general answer")

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, log


def run_mode(mode):
    srv, log = make_server(mode)
    tmp = Path(tempfile.mkdtemp())
    env = dict(os.environ, OPENROUTER_API_KEY="sk-or-test", NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
               OPENROUTER_BASE_URL=f"http://127.0.0.1:{srv.server_port}/api/v1")
    for k in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        env.pop(k, None)
    cmd = [sys.executable, str(HERE / "run_task2.py"), "--model", "vendor/free-model:free", "--provider", "openrouter",
           "--runs", str(tmp / "runs")]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=240, cwd=tmp)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    out = p.stdout

    run_dir = next((tmp / "runs").iterdir())
    state = json.loads((run_dir / "state.json").read_text())
    exps = json.loads((run_dir / "experiments.json").read_text())
    roles = [t["role"] for t in state["dialogue"]]
    assert roles == ["user", "orchestrator", "scientist", "orchestrator", "executor", "orchestrator", "orchestrator",
                    "executor", "orchestrator"], roles
    assert "Focus the beam." in state["dialogue"][0]["text"] and "DMD" in state["dialogue"][0]["text"]
    assert "refused by the finish guard: 2 successful" in state["dialogue"][5]["text"]
    assert len(exps) == 4 and state["n_calls"] == 5 and state["stop_reason"] == "finish" and state["decisions"] == 5
    assert [e["settings"]["macropixel_size"] for e in exps] == [16, 8, 4, 4]
    assert "REJECTED" in state["dialogue"][4]["text"] and "macropixel_size must be one of" in state["dialogue"][4]["text"]
    assert all((run_dir / e["figure"]).exists() for e in exps) and state["final_text"].startswith("FINAL")
    assert "decisions used: 3 of 10" in log["orch_user"][3] and f"**{exps[0]['enhancement']:.1f}**" in log["orch_user"][3]
    # what the terminal (and the video) shows
    for needle in ("[orchestrator]   decision 1", "next: scientist", "TOOL CALL focus_search(macropixel_size=3",
                   "TOOL RESULT rejected", "TOOL CALL focus_search(macropixel_size=16", "... iteration", "TOOL RESULT experiment 1: enhancement",
                   "=== summary ==="):
        assert needle in out, (needle, out[-2500:])
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["request"] == "Focus the beam." and info["model"] == "vendor/free-model:free"
    assert (run_dir / "report.md").read_text().count("| 4 |") == 1
    if mode == "free":
        assert log["rejected"] == ["response_format", "tools"], log["rejected"]      # each unsupported feature tried once
    else:
        assert not log["rejected"]
        props = log["tool_schema"]["focus_search"]
        assert set(props["required"]) == {"macropixel_size", "focus_size", "bg_gap", "bg_width", "n_bg_regions", "iterations"}
        assert "run_python" in log["tool_schema"]
    # the report builder on the same run
    r = subprocess.run([sys.executable, str(HERE / "make_report.py"), str(run_dir), "--lecture-enhancement", "50",
                        "--lecture-macropixels", "100"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and (run_dir / "task2_section.pdf").stat().st_size > 20000, r.stdout + r.stderr
    assert (run_dir / "experiments_table.csv").read_text().count("\n") == 5
    print(f"ok  wire test, mode={mode}: 4 experiments (1 rejected call repaired), finish guard refused once, {log['n_orch']} orchestrator "
          f"calls; rejected by the server: {log['rejected'] or 'nothing'}; report PDF built")
    srv.shutdown()


if __name__ == "__main__":
    run_mode("full")
    run_mode("free")
    print("\nwire checks passed")
