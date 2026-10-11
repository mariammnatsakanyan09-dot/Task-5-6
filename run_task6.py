"""Run the four-role agentic system on Task 5.

    export GOOGLE_API_KEY=...          # or ANTHROPIC_API_KEY=...
    python3 run_task6.py --data /path/to/SNLS80mV.mat

Writes runs/<timestamp>/ with: the executor's scripts (exec_01.py, ...),
transcript.md, state.json, graph.mmd.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from task6_agents import build_graph, default_models, initial_state, transcript_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to SNLS80mV.mat")
    ap.add_argument("--task", default=str(Path(__file__).with_name("task5_prompt.txt")))
    ap.add_argument("--runs", default="runs")
    ap.add_argument(
        "--model",
        default=None,
        help="chat model name (default: env AGENT_MODEL or gemini-2.0-flash / claude-sonnet)",
    )
    ap.add_argument("--timeout", type=int, default=300, help="seconds per script run")
    args = ap.parse_args()

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("Set either ANTHROPIC_API_KEY or GOOGLE_API_KEY first.")

    data = str(Path(args.data).resolve())
    if not Path(data).exists():
        sys.exit(f"data file not found: {data}")

    workdir = Path(args.runs) / time.strftime("%Y%m%d-%H%M%S")
    workdir.mkdir(parents=True, exist_ok=True)
    workdir = workdir.resolve()

    task = Path(args.task).read_text()
    task += (
        "\n\nEnvironment note for the system: the data file is available to scripts "
        "through the environment variable SNLS_DATA. Scripts run on CPU only."
    )

    graph = build_graph(default_models(args.model), workdir, data, timeout_s=args.timeout)
    (workdir / "graph.mmd").write_text(graph.get_graph().draw_mermaid())

    final = initial_state(task)
    print(f"working directory: {workdir}\n")

    for update in graph.stream(final, config={"recursion_limit": 60}, stream_mode="updates"):
        for node, upd in update.items():
            for t in upd.get("dialogue", []):
                head = t["text"].strip().splitlines()[0][:110] if t["text"].strip() else ""
                print(f"[{node:12s}] {head}")
            for k, v in upd.items():  # merge the update into our copy of the state
                final[k] = final["dialogue"] + v if k == "dialogue" else v

    (workdir / "transcript.md").write_text(transcript_md(final))
    (workdir / "state.json").write_text(json.dumps(final, indent=2, default=str))

    print("\n=== summary ===")
    print("turn order :", " -> ".join(t["role"] for t in final["dialogue"][1:]))
    print("decisions  :", final.get("decisions"))
    print("stop reason:", final.get("stop_reason"))
    print("TRAIN_RMSE :", final.get("train_rmse"))
    print("TEST_RMSE  :", final.get("test_rmse"))
    print("files      :", workdir)


if __name__ == "__main__":
    main()