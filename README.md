# Tasks 5 and 6: Silverbox neural ODE and the four-role agent system

## Files

| file | what it is |
|---|---|
| `task5_pinn.py` | Task 5: data excerpt, acceleration network, RK4 free run, architecture/hyperparameter sweep, plot |
| `task6_agents.py` | Task 6: LangGraph `StateGraph` with orchestrator, scientist, general agent, executor |
| `run_task6.py` | runs the agent system on Task 5 and saves the transcript, scripts and graph |
| `task5_prompt.txt` | the text of Task 5 that is given to the agent system |
| `test_task6_offline.py` | offline checks of the Task 6 wiring (scripted fake models, no API calls) |
| `make_synthetic.py` | writes a synthetic Silverbox-like `.mat` to test Task 5 without the real file |

## Setup

```
pip install torch numpy scipy matplotlib h5py langgraph langchain-core langchain-anthropic
```
Download `SNLS80mV.mat` from https://www.nonlinearbenchmark.org/ (Silverbox benchmark).

## Task 5

```
python task5_pinn.py --data SNLS80mV.mat --out results --budget full     # all 7 configurations
python task5_pinn.py --data SNLS80mV.mat --out results --budget medium   # 3 configurations, ~5 min each on 2 cores
python task5_pinn.py --data SNLS80mV.mat --out results --budget smoke    # sanity check, ~30 s
```
Outputs in `results/`: `results.json` (kept version, both free-run RMSEs, every run), `sweep_table.md`
(settings tried and their RMSE: copy it into the PDF), `test_prediction.png` (measured vs predicted on the test
segment, the free run), `best_model.pt`. The last lines printed are `TRAIN_RMSE=` and `TEST_RMSE=`.

What it does, as the task defines it:
- excerpt `[42650:46746]`, mean of u and mean of y of the excerpt subtracted, first 3072 samples train, last 1024 test;
- state `(y, v)`, `y' = v`, `v' = alpha_theta(y, v, u)`; `alpha_theta` is a fully connected network on the scaled
  inputs `(y, v, u)` with scaled output (optionally plus a linear skip path); the circuit coefficients a, b, c, g are
  not parameters of the model. A least-squares fit of eq. (1) is printed as a diagnostic only (it is biased by noise in
  the finite differences);
- integration with classical RK4 at `dt = 1/610.35`, the input at the half step is the mean of its two neighbours;
- training: stage 0 regresses on finite-difference acceleration (stable start), stage 1 is multiple shooting: random
  windows of the training record are integrated from the measured position and finite-difference velocity, with a
  curriculum on the window length (for example 32, 96, 256 samples), Adam or AdamW, cosine learning-rate decay,
  gradient clipping;
- metric: one free run over all 4096 samples from the first sample (measured position, finite-difference velocity),
  no reset at sample 3072, `u` read everywhere, `y` never read after sample 0; RMSE on samples 3072..4095 (test) and on
  0..3071 (train); baseline = RMS of the test output (predict zero);
- the kept version is the one with the smallest test RMSE, as the task says.

## Task 6

```
export ANTHROPIC_API_KEY=...
python run_task6.py --data /path/to/SNLS80mV.mat
```
The model name comes from `--model` or the env var `AGENT_MODEL` (default `claude-sonnet-5-5`).
`runs/<timestamp>/` receives the executor's scripts (`exec_01.py`, ...), `transcript.md` (who said what),
`state.json` and `graph.mmd` (Mermaid source of the graph, for the PDF).

Design:
- one `StateGraph`; the shared state is the dialogue, a list of turns `{role, text}`; every node reads it and appends
  its own turn; `orchestrator` is the entry node and every specialist returns to it;
- orchestrator: one structured-output model call per turn returns `next` (scientist | general | executor | finish),
  `instruction`, `reason`; no keyword table, no fixed order; it writes no theory or code and has no tool;
  it stops on `finish` or after 8 decisions, whichever comes first;
- finish guard in code: `finish` is accepted only if an executor turn ran a script with exit code 0 that printed
  `TEST_RMSE=<number>`; the number is parsed from the tool's stdout, not from model text. If the model tries to finish
  early it is told once, then routed to the scientist (no plan yet) or the executor;
- scientist: theory and plan, no tool; general agent: high-level side questions, no tool, no plan;
- executor: the only node with a tool, `run_python(code, purpose)`: writes a new file `exec_NN.py` in the run's working
  directory and runs it with a 300 s limit, returning exit code, stdout, stderr. One script per turn; a failed run
  returns the traceback and a later executor turn repairs the script (it is shown its previous script) while the
  scientist's plan stays the same.

## What was and was not verified here

- Task 5: the pipeline ran end to end on a synthetic Duffing record with the same file layout (`make_synthetic.py`).
  On that stand-in the free-run test RMSE was about 0.002 V against 0.058 V for predicting zero, which only shows the
  code works. No number from the real Silverbox file was produced here: the benchmark site is not reachable from this
  workspace. Run it on the real file and take the numbers for the PDF from that run.
- Task 6: `python test_task6_offline.py` passes 7 checks with scripted models: routing, the 8-decision cap, the finish
  guard, repair after a failing script with a new file each time, the 5-minute limit, tools only on the executor, and
  that an RMSE claimed in prose is ignored. The graph was not run against a live model here (no API access from this
  workspace), so the end-to-end result of the agent system on Task 5 has to come from your own run.
- Still to do by you for the hand-in: push the code to a GitHub repository and put the link in the PDF, and write the
  PDF with the architecture, the settings tried and their test RMSE (`sweep_table.md`), both free-run RMSEs, the plot,
  and for Task 6 the transcript (who did what) and the free-run error the run printed.

Limit of the guard: it trusts what a script prints. A script that prints a hard-coded `TEST_RMSE` line would pass; the
executor prompt forbids it and the transcript shows the script, but the code does not check how the number was made.
# Task-5-6
