"""Final working runner - handles Command correctly"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from langgraph.types import Command
from task1_human_loop_senior import (
    build_graph_with_human,
    default_models,
    initial_state,
    transcript_md,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="gemini-3.8-flash", help="use gemini-3.8-flash or gemini-flash-latest")
    ap.add_argument("--mode", default="ambiguous", choices=["ambiguous", "hard"])
    ap.add_argument("--runs", default="runs_task1_senior")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("export GOOGLE_API_KEY=... first")

    data = str(Path(args.data).resolve())
    if not Path(data).exists():
        sys.exit(f"data not found: {data}")

    workdir = Path(args.runs) / time.strftime("%Y%m%d-%H%M%S")
    workdir.mkdir(parents=True, exist_ok=True)
    workdir = workdir.resolve()

    if args.mode == "ambiguous":
        task_text = "You have SNLS80mV.mat. Train the neural ODE and report the RMSE."
    else:
        task_text = "Train the neural ODE on SNLS80mV.mat. After you get a result, run again to overwrite exec_01.py with a better model and then finish with final TEST_RMSE."
    task_text += "\n\nEnvironment note: data file available via env var SNLS_DATA. CPU only."

    thread_id = f"senior-{args.mode}-{time.strftime('%H%M%S')}"
    print(f"Workdir: {workdir}  Thread: {thread_id}  Model: {args.model}")

    models = default_models(args.model)
    graph = build_graph_with_human(models, workdir, data, timeout_s=args.timeout, max_decisions=8)
    try:
        (workdir / "graph.mmd").write_text(graph.get_graph().draw_mermaid())
    except:
        pass

    state_dict = initial_state(task_text)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}
    current_input = state_dict  # first input is state dict, later Command

    while True:
        # stream updates
        for update in graph.stream(current_input, config=config, stream_mode="updates"):
            if isinstance(update, tuple):
                # older langgraph: (node, patch)
                node_name, patch = update
                items = [(node_name, patch)]
            elif isinstance(update, dict):
                items = list(update.items())
            else:
                continue

            for node_name, patch in items:
                if node_name == "__interrupt__":
                    continue
                if not isinstance(patch, dict):
                    continue
                for t in patch.get("dialogue", []):
                    txt = (t.get("text") or "")[:300].splitlines()[0]
                    print(f"[{node_name:12s}] {t.get('role','')}: {txt}")

        # check state for interrupt or finish
        snapshot = graph.get_state(config)
        if not snapshot.tasks:  # no tasks = finished
            final = snapshot.values
            break

        # look for interrupt
        interrupted = False
        for task in snapshot.tasks:
            if getattr(task, "interrupts", None):
                intr = task.interrupts[0]
                payload = intr.value if hasattr(intr, "value") else intr
                print("\n" + "="*70)
                print("HUMAN INPUT REQUIRED")
                print("="*70)
                if isinstance(payload, dict):
                    print(f"Type: {payload.get('type')}")
                    print(f"Question: {payload.get('question')}")
                    print(f"Proposed next: {payload.get('proposed_next')}")
                    print(f"Proposed instruction: {payload.get('proposed_instruction')}")
                else:
                    print(str(payload))
                print("="*70)
                human_text = input("\nYour answer: ").strip()
                if not human_text:
                    human_text = "yes"
                print(f"\nResuming with: {human_text}\n")
                current_input = Command(resume={"text": human_text})
                interrupted = True
                break

        if interrupted:
            continue
        else:
            # tasks exist but no interrupt = still running? loop again with None?
            # In langgraph, after non-interrupt tasks you need to continue streaming with None
            # But our loop already streamed once; get new snapshot
            # To avoid infinite loop, break if no interrupt
            final = snapshot.values
            # if still has tasks, try streaming again with no input (continue)
            # Actually we need to pass None to continue
            current_input = None
            # small guard to prevent infinite loop
            if snapshot.next:
                continue
            else:
                break

    (workdir / "transcript.md").write_text(transcript_md(final))
    (workdir / "state.json").write_text(json.dumps(final, indent=2, default=str))

    print("\n=== summary ===")
    try:
        print("turn order :", " -> ".join(t["role"] for t in final["dialogue"][1:]))
    except:
        pass
    print("decisions  :", final.get("decisions"))
    print("stop reason:", final.get("stop_reason"))
    print("TRAIN_RMSE :", final.get("train_rmse"))
    print("TEST_RMSE  :", final.get("test_rmse"))
    print("files      :", workdir)


if __name__ == "__main__":
    main()
