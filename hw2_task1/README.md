# Homework 2, Task 1: human-in-the-loop for the four-role agent system

Starts from Homework 1, Task 6 (orchestrator, scientist, general agent, executor on one LangGraph `StateGraph`)
and adds a fifth destination, `human`, that pauses the graph with a LangGraph interrupt.

## Files

| file | what it is |
|---|---|
| `hitl_agents.py` | the system: Task 6 graph + `human` node, interrupt/resume, OpenRouter models, fallbacks for free models |
| `run_hitl.py` | runs it: prints the question when the graph pauses, reads your reply, resumes the same thread, writes `evidence.md` |
| `test_hitl_offline.py` | 22 offline checks with scripted fake models (the 7 Task 6 checks + 15 for the human stop) |
| `test_wire_fake_openrouter.py` | end-to-end check of the OpenRouter path against a local fake server (no key, no network) |
| `list_openrouter_models.py` | lists the free OpenRouter models that support tool calling (run it on your machine) |
| `task5_prompt.txt` | the Task 5 statement, given to the system as background for requests about the Silverbox work |

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...        # your free key; it is read from the environment and never written to a file
python list_openrouter_models.py           # pick a free model that supports tools
python test_hitl_offline.py                # 22 checks, a few seconds
python test_wire_fake_openrouter.py        # OpenRouter path end to end, offline
```

Download `SNLS80mV.mat` from https://www.nonlinearbenchmark.org/ (Silverbox) as before.

### OpenRouter free tier: what to know

* Limits: **20 requests per minute, 50 per day** (1000 per day after buying 10 credits). One full run here is about
  8 to 15 requests, and `run_hitl.py` prints the number of requests it sent. Two demo runs fit in a day; do not loop.
* The code counts retries (a 429 is retried after 10, 20, 40, 60 s) and stops with a clear message when the daily
  quota is gone. A 400/404 is never retried (a retry would only burn quota).
* Free models differ a lot. The system needs structured output for the orchestrator and tool calling for the executor.
  If a model has neither, the code detects it once and switches for the rest of the run: the orchestrator is asked for
  one JSON object in text, the executor for one fenced ```python block. Both paths are tested, but a model that
  supports tools and structured output is more reliable. Model ids ending in `:free` come and go, so use
  `list_openrouter_models.py` on the day. `--model a:free,b:free` sends the list as OpenRouter's fallback array
  (a fallback inside one request costs one request).

## The two runs the task asks for

The system is interactive: when the graph pauses you type the reply in the terminal. Use the same model for both runs.

**(a) Ambiguous request about the Silverbox work**

```bash
python run_hitl.py --data SNLS80mV.mat --model <free-model-id> \
  --request "How good is the Silverbox model? Give me one error number."
```

The two readings are different numbers (free-run RMSE on the 1024 test samples, on the 3072 training samples,
a one-step error). The orchestrator should pause with `ambiguous_query`; answer with a reading, for example
`the free-run RMSE on the 1024 test samples`. Other requests that work the same way: "Tell me about the cubic term c"
(explanation by the general agent, or an estimate of its value by the executor), "Evaluate the model on the held-out data".

**(b) A request that leads to a hard decision, refused once, then accepted**

```bash
python run_hitl.py --data SNLS80mV.mat --model <free-model-id> --seed results \
  --request "Re-train the Silverbox network and save the model as results/best_model.pt, then report the test error."
```

`--seed results` copies your Homework 1 `results/` folder into the run's working directory, so `results/best_model.pt` is
"already on disk" (your original folder is never touched; the orchestrator is shown the file list). The orchestrator
should pause with `hard_decision` before the executor runs. Reply `refuse, do not overwrite best_model.pt`; the turn
returns to the orchestrator, the executor does not run (no `exec_*.py` is written), and the orchestrator proposes
another step (for example another file name), which you answer with `accept`.
No `results/` folder? `mkdir results && echo old > results/best_model.pt` is enough.

Reply format for a hard decision: **start the reply with `accept` or `refuse`**, then optionally a comment. A reply that
is neither is treated as a refusal (the step does not run on a doubtful answer).

### What goes into the PDF

Each run writes `runs/<timestamp>/evidence.md`: for every pause the question the graph posed, the reply you typed, and the
next message(s) in the dialogue (for a refusal: the `[graph]` note saying the step did not run and how many executor
scripts exist, then the orchestrator's next turn). `transcript.md` is the whole dialogue, `graph.mmd` the Mermaid source of
the graph, `exec_NN.py` the executor's scripts. Push the folder to GitHub and put the link in the PDF:

```bash
git init && git add . && git commit -m "HW2 task 1: human-in-the-loop"
git branch -M main && git remote add origin <your repo url> && git push -u origin main
```

(`runs/` is not git-ignored, so the two demo runs go with the code; `*.mat` and `__pycache__` are.)

## Design

```
                       ┌──────────────────────────────────────────────┐
                       │  orchestrator  (one model call per decision) │◄───────────────┐
                       └──┬──────┬───────┬────────┬────────┬──────────┘                │
                  scientist  general  executor   human   finish→END                    │
                          │      │       │        │ interrupt()                        │
                          └──────┴───────┘        │ resume: Command(resume=reply)      │
                                  │               ├─ ambiguous query ─────────────────►┤ (next decision reads the reply)
                                  └──────────────►│ hard decision, accept ► executor ──►┤
                                                  └ hard decision, refuse ─────────────►┘ (step does NOT run)
```

* **The choice is a model call.** The orchestrator returns `next` (scientist | general | executor | human | finish),
  `instruction`, a one-sentence `reason`, and for `human`: `question`, `kind` (`ambiguous_query` | `hard_decision`) and
  `guarded` (the step that runs after an accept: `executor` or `finish`). No keyword list, no fixed order. It is told when
  `human` is allowed (the two situations of the task) and is shown, by code, the list of files on disk.
* **The stop is a LangGraph interrupt.** `build_graph` compiles with a checkpointer (`InMemorySaver`). The `human` node calls
  `interrupt(payload)` with the question (and, for a hard decision, the step). The graph stays paused on its thread until
  the caller resumes **that same thread** with `Command(resume=reply)` (`drive()` in `hitl_agents.py` does exactly this).
* **The reply** is appended to the shared dialogue as a turn with the role `human`, so every later model call reads it.
* **Ambiguous query:** control returns to the orchestrator; its next decision follows the reading you gave (the prompt says so,
  and the reply is in the dialogue it reads).
* **Hard decision, accept:** the guarded step runs: the executor with the instruction that was shown to you, or `finish`
  (the orchestrator then emits the final message that you approved). **Refuse (or unclear reply):** the turn returns to the
  orchestrator, the executor is not called and no script is written. The graph appends a `[graph]` note that says so.
* **Budget:** asking the human counts as one decision, so the 8-decision cap still ends every run; a step the human accepts
  runs without costing another decision.

### What changed compared with Task 6

* The finish guard (finish only after a script that really ran printed `TEST_RMSE=`) is kept. A refused finish no longer forces
  a role chosen by code: the turn goes back to the orchestrator, so every destination is a model choice, including the ones
  that would run code. A finish is also allowed when the human has refused a step (then the run ends honestly with no
  measurement), otherwise a refusal could never end.
* The executor sees its previous script when it repairs a failed run, and a failed summary call no longer loses a script
  that already ran.
* Model calls retry on 429/5xx, and fall back for free models without structured output or tool calling (see above).

## What was verified (and what was not)

* `python test_hitl_offline.py`: 22 checks pass. They show that the graph stops at `human`, that the paused thread
  does not advance until resumed, that the reply becomes a `human` turn which the next orchestrator call reads, that an accepted
  step runs the instruction that was shown, that a refused step leaves no `exec_*.py` and no executor call, that an unclear
  reply fails closed, that `finish` works as a hard decision (accept releases the message, refuse keeps the run going), that the
  cap counts human decisions, and the free-model fallbacks.
* `python test_wire_fake_openrouter.py`: the real `run_hitl.py`, the real `ChatOpenAI` client and the real executor script against
  a local server that speaks the OpenAI protocol, in a model-with-tools mode and a model-without-tools mode: four pauses
  (ambiguous, refused, accepted, accepted finish), the refused step leaves `best_model.pt` untouched.
* **Not verified: a live OpenRouter run.** The development sandbox could not reach openrouter.ai and had no key, so the two
  demo runs for the PDF have to be made by you. Whether a particular free model recognises an ambiguous request or a hard
  decision when it should is a property of that model; the code makes the pause work, the model decides when to ask. If a
  model never asks, try another one or make the request more clearly ambiguous.
* The orchestrator is told that running the coding tool, overwriting a file and ending with a measurement are hard decisions
  (the three cases in the task), so a run usually has two or three pauses: the one you are demonstrating plus an approval of
  the run and of the final report. Answer `accept` to those. It is told not to ask again about a step you already accepted.
* `--reply "..."` (repeatable) answers the pauses from the command line instead of the terminal; the wire test uses it.
  For the PDF, type the replies yourself.
