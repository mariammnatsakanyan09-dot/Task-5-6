"""End-to-end check of the OpenRouter path WITHOUT a key or network.

A tiny local server speaks the OpenAI chat-completions protocol (what OpenRouter exposes) and plays the models.
run_hitl.py is started as a real subprocess with OPENROUTER_BASE_URL pointing at it, so the real ChatOpenAI client,
the real tool calls, the real interrupt/resume and the real executor script run. Two modes:

  full  - a model with structured output and tool calling
  free  - a model WITHOUT them (400 for response_format, 404 for tools): the JSON-text and fenced-script fallbacks

    python test_wire_fake_openrouter.py

The script of the run is the demonstration of the homework in one go: an ambiguous request (pause 1), a hard decision
that overwrites a result already on disk, refused once (pause 2), proposed again with another file name and accepted
(pause 3), and a hard 'finish' accepted (pause 4).
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import scipy.io

HERE = Path(__file__).parent

EXEC_CODE = '''import os, numpy as np, scipy.io
d = scipy.io.loadmat(os.environ["SNLS_DATA"])
u = d["V1"].ravel()[42650:46746]; y = d["V2"].ravel()[42650:46746]
y = y - y.mean()
rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
open("results/best_model_v2.pt", "w").write("new model")          # new file; the old one is not touched
print("TRAIN_RMSE=%.6f" % rms(y[:3072]))
print("TEST_RMSE=%.6f" % rms(y[3072:]))
'''

ORCH = [
    dict(next="human", kind="ambiguous_query", question="Which error do you mean: the free-run RMSE on the 1024 test samples, or on the 3072 training samples?",
         reason="The two readings report different numbers.", instruction=""),
    dict(next="scientist", instruction="Write the plan for the free-run test error.", reason="No plan yet."),
    dict(next="human", kind="hard_decision", guarded="executor",
         question="The next step runs the coding tool and would overwrite results/best_model.pt, which is already on disk. Proceed?",
         instruction="Fit the network and save it as results/best_model.pt.", reason="Overwrites an existing result."),
    dict(next="human", kind="hard_decision", guarded="executor",
         question="Instead I will run the coding tool and save the model as results/best_model_v2.pt (nothing overwritten). Proceed?",
         instruction="Fit the network and save it as results/best_model_v2.pt.", reason="Avoids the overwrite."),
    dict(next="human", kind="hard_decision", guarded="finish",
         question="I am about to end the dialogue and report the measured test RMSE. Proceed?",
         instruction="Final report: the free-run test RMSE was measured by exec_01.py.", reason="A measurement exists."),
]
REPLIES = ["the free-run test error", "refuse, do not overwrite best_model.pt", "accept, use the v2 name", "accept"]


def make_server(mode):
    log = dict(orch_prompts=[], rejected=[], n_orch=0, roles=[])

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

            if "ORCHESTRATOR" in system:
                if mode == "free":
                    assert "JSON object" in msgs[-1]["content"] or "rejected" in msgs[-1]["content"], "text mode must ask for JSON"
                log["orch_prompts"].append(system)
                d = ORCH[log["n_orch"]]
                log["n_orch"] += 1
                log["roles"].append("orchestrator")
                return self._msg(body, content=json.dumps(d))
            if "SCIENTIST" in system:
                log["roles"].append("scientist")
                return self._msg(body, content="PLAN: ... print TRAIN_RMSE= and TEST_RMSE= from one free run.")
            if "EXECUTOR" in system:
                log["roles"].append("executor")
                if any(m["role"] == "tool" for m in msgs) or msgs[-1]["content"].startswith("--- run_python result"):
                    return self._msg(body, content="Exit code 0; the script printed TRAIN_RMSE and TEST_RMSE.")
                if body.get("tools"):
                    call = {"id": "call_1", "type": "function",
                            "function": {"name": "run_python", "arguments": json.dumps({"code": EXEC_CODE, "purpose": "fit"})}}
                    return self._msg(body, tool_calls=[call])
                return self._msg(body, content="```python\n" + EXEC_CODE + "```")
            return self._msg(body, content="general answer")

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, log


def run_mode(mode):
    srv, log = make_server(mode)
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    scipy.io.savemat(tmp / "fake.mat", {"V1": rng.normal(0, .05, (1, 131072)), "V2": rng.normal(0, .06, (1, 131072))})
    seed = tmp / "results"
    seed.mkdir()
    (seed / "best_model.pt").write_text("old model")

    env = dict(os.environ, OPENROUTER_API_KEY="sk-or-test", NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
               OPENROUTER_BASE_URL=f"http://127.0.0.1:{srv.server_port}/api/v1")
    for k in ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        env.pop(k, None)
    cmd = [sys.executable, str(HERE / "run_hitl.py"), "--data", str(tmp / "fake.mat"), "--model", "vendor/free-model:free",
           "--provider", "openrouter", "--runs", str(tmp / "runs"), "--seed", str(seed),
           "--request", "How good is the Silverbox model? Give me one error number."]
    for r in REPLIES:
        cmd += ["--reply", r]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=240, cwd=tmp)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]

    run_dir = next((tmp / "runs").iterdir())
    state = json.loads((run_dir / "state.json").read_text())
    roles = [t["role"] for t in state["dialogue"]]
    want = ["user", "orchestrator", "human", "orchestrator", "scientist", "orchestrator", "human", "graph", "orchestrator",
            "human", "graph", "executor", "orchestrator", "human", "graph", "orchestrator"]
    assert roles == want, roles
    d = state["dialogue"]
    assert d[2]["text"] == REPLIES[0]                                           # pause 1: reply typed, then ...
    assert "free-run test error" in d[2]["text"] and d[3]["text"].startswith("next: scientist")   # ... next decision
    assert "did NOT run" in d[7]["text"] and "scripts in the working directory: 0" in d[7]["text"]    # pause 2: refused
    assert "accepted" in d[10]["text"] and d[11]["role"] == "executor"                               # pause 3: accepted
    assert state["decisions"] == 5 and state["refused"] == 1 and state["stop_reason"] == "finish"
    assert state["test_rmse"] is not None and d[-1]["text"].startswith("Final report")
    assert not (run_dir / "exec_02.py").exists() and (run_dir / "exec_01.py").exists()
    assert (run_dir / "results" / "best_model.pt").read_text() == "old model"          # refused step never overwrote it
    assert (run_dir / "results" / "best_model_v2.pt").read_text() == "new model"
    assert "results/best_model.pt" in log["orch_prompts"][2]                           # the orchestrator saw what is on disk
    ev = (run_dir / "evidence.md").read_text()
    assert ev.count("## Pause") == 4 and "refuse, do not overwrite best_model.pt" in ev
    if mode == "free":
        assert log["rejected"] == ["response_format", "tools"], log["rejected"]      # each unsupported feature tried once
    else:
        assert not log["rejected"]
    print(f"ok  wire test, mode={mode}: 4 pauses (ambiguous, refused, accepted, accepted finish); "
          f"{log['n_orch']} orchestrator calls; rejected by the server: {log['rejected'] or 'nothing'}")
    srv.shutdown()


if __name__ == "__main__":
    run_mode("full")
    run_mode("free")
    print("\nwire checks passed")
