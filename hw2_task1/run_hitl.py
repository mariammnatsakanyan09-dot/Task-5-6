"""Run the five-destination system (orchestrator, scientist, general, executor, human) on a request.

    export OPENROUTER_API_KEY=sk-or-...
    python run_hitl.py --data SNLS80mV.mat --model "<openrouter model id>" --request "..."

When the orchestrator sends the turn to `human`, the graph PAUSES (LangGraph interrupt), this script prints the
question and waits for your reply, then resumes the same thread with that reply.

Writes runs/<timestamp>/ with: exec_NN.py (the executor's scripts), transcript.md, evidence.md (the question, the
reply and the next message of every pause - what the PDF asks for), state.json, graph.mmd.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import hitl_agents as H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to SNLS80mV.mat")
    ap.add_argument("--request", help="what you ask the system (you talk only to the orchestrator)")
    ap.add_argument("--request-file", help="read the request from a text file instead")
    ap.add_argument("--context", default=str(Path(__file__).with_name("task5_prompt.txt")),
                    help="background text put under the request (default: the Task 5 statement); 'none' = no background")
    ap.add_argument("--seed", action="append", default=[],
                    help="file or folder copied into the working directory before the run, so that it is 'already "
                         "on disk' (repeatable). Example: --seed results   (your Homework 1 results folder)")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--workdir", default=None, help="use this working directory instead of runs/<timestamp>")
    ap.add_argument("--provider", choices=["openrouter", "google", "anthropic"], default=None,
                    help="default: openrouter if OPENROUTER_API_KEY is set, else google, else anthropic")
    ap.add_argument("--model", default=None,
                    help="model id (env AGENT_MODEL). OpenRouter: one id, or several separated by commas (fallbacks)")
    ap.add_argument("--timeout", type=int, default=300, help="seconds per script run")
    ap.add_argument("--max-decisions", type=int, default=H.MAX_DECISIONS)
    ap.add_argument("--reply", action="append", default=[],
                    help="a scripted reply for the n-th pause (repeatable). Without it you are asked in the terminal.")
    ap.add_argument("--thread-id", default=None)
    args = ap.parse_args()

    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit("Set OPENROUTER_API_KEY (or GOOGLE_API_KEY / ANTHROPIC_API_KEY) first.")
    request = args.request or (Path(args.request_file).read_text() if args.request_file else None)
    if not request:
        sys.exit("Give --request \"...\" or --request-file path.")
    data = Path(args.data).resolve()
    if not data.exists():
        sys.exit(f"data file not found: {data}")

    workdir = Path(args.workdir) if args.workdir else Path(args.runs) / time.strftime("%Y%m%d-%H%M%S")
    workdir.mkdir(parents=True, exist_ok=True)
    workdir = workdir.resolve()
    for s in args.seed:
        src = Path(s)
        if not src.exists():
            sys.exit(f"--seed path not found: {src}")
        if src.is_dir():
            shutil.copytree(src, workdir / src.name, dirs_exist_ok=True)
        else:
            shutil.copy2(src, workdir / src.name)

    context = "" if args.context == "none" else Path(args.context).read_text()
    task = H.build_task(request, context)

    graph = H.build_graph(H.default_models(args.model, args.provider), workdir, str(data),
                          timeout_s=args.timeout, max_decisions=args.max_decisions)
    (workdir / "graph.mmd").write_text(graph.get_graph().draw_mermaid())
    thread_id = args.thread_id or "hitl-" + time.strftime("%Y%m%d-%H%M%S")
    print(f"working directory: {workdir}\nthread: {thread_id}")
    print("on disk before the run:\n" + H.list_files(workdir) + "\n")

    scripted = list(args.reply)

    def ask(payload: dict) -> str:
        print("\n" + "=" * 78)
        print(f"GRAPH PAUSED ({payload['kind']}) - waiting for the human")
        print("=" * 78)
        print(payload["question"])
        if payload.get("step"):
            print(f"\nstep that would run: {payload['step']['role']}\n  {payload['step']['instruction']}")
        print(f"\n(answer with: {payload['answer_with']})")
        if scripted:
            reply = scripted.pop(0)
            print(f"your reply> {reply}   [scripted]")
        else:
            try:
                reply = input("your reply> ")
            except EOFError:
                sys.exit("\nNo terminal to read the reply from. The graph is paused; pass --reply \"...\" for each pause.")
        print("=" * 78 + "\n")
        return reply

    def show(node: str, upd: dict):
        for t in upd.get("dialogue", []):
            head = t["text"].strip().splitlines()[0][:110] if t["text"].strip() else ""
            print(f"[{t['role']:12s}] {head}")

    final = H.drive(graph, task, ask, thread_id=thread_id, on_update=show)

    (workdir / "transcript.md").write_text(H.transcript_md(final))
    (workdir / "evidence.md").write_text(H.evidence_md(final["dialogue"]))
    (workdir / "state.json").write_text(json.dumps(final, indent=2, default=str))

    print("\n=== summary ===")
    print("turn order    :", " -> ".join(t["role"] for t in final["dialogue"][1:]))
    print("decisions     :", final.get("decisions"), "  refused hard decisions:", final.get("refused"))
    print("stop reason   :", final.get("stop_reason"))
    print("TRAIN_RMSE    :", final.get("train_rmse"))
    print("TEST_RMSE     :", final.get("test_rmse"))
    print("model requests:", H.STATS["model_calls"], "(OpenRouter free models: 20 per minute, 50 per day)")
    print("files         :", workdir, "  (evidence.md = what to put in the PDF)")


if __name__ == "__main__":
    main()
