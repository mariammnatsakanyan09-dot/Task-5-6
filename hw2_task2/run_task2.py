"""Task 2: ask the agent to focus the beam, and watch it work on its own.

    export OPENROUTER_API_KEY=sk-or-...
    python run_task2.py --model "<openrouter model id>"

You type nothing after this command. The agent (orchestrator, scientist, general agent, executor with the focus_search tool)
chooses the macropixel size, the region sizes and the number of iterations, calls the tool, reads the enhancement, and
decides what to try next. The terminal shows each decision and each tool call live (this is what the video shows).

Writes runs/<timestamp>/ with: experiments.json (one record per experiment, as the tool returned it), exp_NN_images.png
(camera images of each experiment), exp_NN_data.npz, report.md (the table), transcript.md, state.json, graph.mmd.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import focus_agents as F


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", default="Focus the beam.", help="what you ask the system (default: Focus the beam.)")
    ap.add_argument("--context", default=str(Path(__file__).with_name("task2_prompt.txt")),
                    help="background text under the request (default: the Task 2 statement); 'none' = no background")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--workdir", default=None, help="use this working directory instead of runs/<timestamp>")
    ap.add_argument("--provider", choices=["openrouter", "google", "anthropic"], default=None,
                    help="default: openrouter if OPENROUTER_API_KEY is set, else google, else anthropic")
    ap.add_argument("--model", default=None,
                    help="model id (env AGENT_MODEL). OpenRouter: one id, or several separated by commas (fallbacks)")
    ap.add_argument("--max-decisions", type=int, default=F.MAX_DECISIONS)
    ap.add_argument("--show-images", action="store_true",
                    help="open each experiment's image file when the experiment ends (macOS: 'open'); good for the video")
    ap.add_argument("--pace", type=float, default=0.0,
                    help="wait this many seconds after each progress line of an experiment (try 0.3) so that a screen "
                         "recording can follow it; it changes no result")
    ap.add_argument("--thread-id", default=None)
    args = ap.parse_args()

    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit("Set OPENROUTER_API_KEY (or GOOGLE_API_KEY / ANTHROPIC_API_KEY) first.")

    workdir = Path(args.workdir) if args.workdir else Path(args.runs) / time.strftime("%Y%m%d-%H%M%S")
    workdir.mkdir(parents=True, exist_ok=True)
    workdir = workdir.resolve()
    context = "" if args.context == "none" else Path(args.context).read_text()
    task = F.build_task(args.request, context)

    if args.show_images:
        def open_image(res):
            opener = "open" if sys.platform == "darwin" else ("xdg-open" if sys.platform.startswith("linux") else None)
            if opener:
                subprocess.Popen([opener, str(workdir / res["figure"])], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        F.HOOKS["after_experiment"] = open_image

    F.PACE["s"] = args.pace
    graph = F.build_graph(F.default_models(args.model, args.provider), workdir, max_decisions=args.max_decisions)
    (workdir / "graph.mmd").write_text(graph.get_graph().draw_mermaid())
    thread_id = args.thread_id or "focus-" + time.strftime("%Y%m%d-%H%M%S")
    print(f"working directory: {workdir}\nrequest to the system: {args.request!r}\n"
          f"from now on nobody types anything: the agent chooses the settings\n")

    def show(node: str, upd: dict):
        for t in upd.get("dialogue", []):
            body = t["text"].strip()
            if t["role"] == "executor":
                continue                                  # its tool calls and results were printed live
            lines = body.splitlines()
            limit = 8 if t["role"] == "orchestrator" else 10
            print("-" * 78)
            print(f"[{t['role']}]" + (f"   decision {upd.get('decisions')}" if t["role"] == "orchestrator" and upd.get("decisions") else ""))
            for ln in lines[:limit]:
                print("   " + (ln if len(ln) <= 200 else ln[:200] + " ..."))
            if len(lines) > limit:
                print(f"   ... ({len(lines) - limit} more lines in transcript.md)")
            print("-" * 78)

    final = F.run_agent(graph, task, thread_id=thread_id, on_update=show)

    (workdir / "run_info.json").write_text(json.dumps(dict(
        model=args.model or os.environ.get("AGENT_MODEL"), provider=args.provider, request=args.request,
        max_decisions=args.max_decisions, model_requests=F.STATS["model_calls"], finished=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    (workdir / "transcript.md").write_text(F.transcript_md(final))
    (workdir / "report.md").write_text(F.final_report_md(final))
    (workdir / "state.json").write_text(json.dumps(final, indent=2, default=str))

    print("\n=== summary ===")
    print(F.experiments_table_md(final.get("experiments", [])))
    best = F.best_experiment(final.get("experiments", []))
    if best:
        print(f"\nenhancement kept from the tool (best experiment): {best['enhancement']:.1f}  (experiment {best['experiment']})")
    if final.get("final_text"):
        print("\nfinal message of the orchestrator:\n" + final["final_text"].strip())
    print("\ndecisions        :", final.get("decisions"), " stop reason:", final.get("stop_reason"))
    print("focus_search calls:", final.get("n_calls"))
    print("model requests   :", F.STATS["model_calls"], "(OpenRouter free models: 20 per minute, 50 per day)")
    print("files            :", workdir)
    print("next step        : python make_report.py", workdir)


if __name__ == "__main__":
    main()
