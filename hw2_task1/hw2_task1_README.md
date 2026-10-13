# Homework 2, Task 1: human-in-the-loop agent system

The Homework 1 Task 6 system (orchestrator, scientist, general agent, executor; one LangGraph `StateGraph`) with a fifth
destination, `human`. The orchestrator sends the turn to `human` for an **ambiguous request** or a **hard decision**
(running the coding tool, overwriting a file on disk, ending with a reported measurement).

## How it works

* The choice of the next destination is one model call. It returns the destination, an instruction and a one-sentence
  reason; for `human` also the question and the kind (`ambiguous_query` or `hard_decision`).
* The `human` node calls LangGraph `interrupt()`. The graph is compiled with a checkpointer (`InMemorySaver`), runs on a
  thread and stays paused until the same thread is resumed with `Command(resume=reply)`.
* The reply is appended to the shared dialogue with the role `human`.
* **Ambiguous request:** the turn returns to the orchestrator, whose next decision follows the reading you gave.
* **Hard decision:** a reply that starts with `accept` (or `yes`, `ok`, `proceed`) lets the step run. `refuse` or any other
  reply means the step does not run: no script is written, the graph adds a `[graph]` note saying so, and the turn returns to the orchestrator.
* At most 8 decisions per run; asking the human counts as one.

## Files

| file | what it is |
|---|---|
| `hitl_agents.py` | the system: graph, `human` node, interrupt/resume, OpenRouter models |
| `run_hitl.py` | runs it: prints the question at each pause, reads your reply, resumes the thread, writes `evidence.md` |
| `task5_prompt.txt` | the Homework 1 Task 5 statement, given to the system as background |
| `list_openrouter_models.py` | lists free OpenRouter models that support tool calling |
| `test_hitl_offline.py` | 22 offline checks with scripted models |
| `test_wire_fake_openrouter.py` | the OpenRouter path against a local fake server (no key) |

## Run

```bash
pip3 install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...     # free key; read from the environment, never stored in a file
python3 test_hitl_offline.py            # no key needed
```

Put `SNLS80mV.mat` ([nonlinearbenchmark.org](https://www.nonlinearbenchmark.org/), Silverbox) in the repository root, next
to this folder. The OpenRouter free tier allows 20 requests per minute and 50 per day; one run used 10 to 14 requests.

**(a) Ambiguous request**

```bash
python3 run_hitl.py --data ../SNLS80mV.mat --model nvidia/nemotron-3-super-120b-a12b:free \
  --request "How big is the cubic term in the Silverbox model?"
```

**(b) Hard decision, refused once, then accepted**

```bash
python3 run_hitl.py --data ../SNLS80mV.mat --model nvidia/nemotron-3-super-120b-a12b:free --seed ../results \
  --request "Re-train the Silverbox network and save the model as results/best_model.pt, then report the test error."
```

`--seed ../results` copies a `results/` folder into the run's working directory, so `results/best_model.pt` is already on
disk (the original is not touched). Type the replies in the terminal when the graph pauses.

Each run writes `runs/<timestamp>/`: `evidence.md` (for every pause: the question, the reply, the next message),
`transcript.md` (whole dialogue), `exec_NN.py` (executor scripts), `graph.mmd` (graph as Mermaid).

## Results (model `nvidia/nemotron-3-super-120b-a12b:free`, both runs)

| run | request | pauses |
|---|---|---|
| `runs/20261013-203406` (a) | cubic term | 1 `ambiguous_query`; reply: *the typical size of the cubic term c\*y^3 relative to the linear term at peak \|y\| of about 0.2 V*; the next decision followed it. 8 decisions, 14 model requests |
| `runs/20261013-204607` (b) | re-train, save `results/best_model.pt` | 3 `hard_decision`: 1st **refused** (executor did not run, 0 scripts written), 2nd accepted (new file name; its script hit the 300 s limit), 3rd accepted (repaired script ran). 5 decisions, 10 model requests |

In (b) `results/best_model.pt` was a 4-byte placeholder, and the final test RMSE was 2.19 V, worse than predicting zero
(about 0.06 V). This task tests the pause mechanism, not the model quality. Whether a model pauses is the model's decision:
the request "How good is the Silverbox model? Give me one error number." did not make it ask, so it was not used.
