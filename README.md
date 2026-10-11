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

```bash
pip install torch numpy scipy matplotlib h5py langgraph langchain-core langchain-google-genai
# optional: langchain-anthropic if you prefer Claude
```

Download `SNLS80mV.mat` from https://www.nonlinearbenchmark.org/ (Silverbox benchmark).

## Task 5

```bash
python3 task5_pinn.py --data SNLS80mV.mat --out results --budget full     # all 7 configurations
python3 task5_pinn.py --data SNLS80mV.mat --out results --budget medium   # 3 configurations, ~5 min each on 2 cores
python3 task5_pinn.py --data SNLS80mV.mat --out results --budget smoke    # sanity check, ~30 s
```

Outputs in `results/`: `results.json` (kept version, both free-run RMSEs, every run), `sweep_table.md`
(settings tried and their RMSE), `test_prediction.png` (measured vs predicted on the test
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

**Results on real SNLS80mV.mat (full budget):**
- Kept model: `mlp-2x64-silu-adam`
- TRAIN_RMSE = 0.001545 V
- TEST_RMSE = 0.003778 V
- Baseline (predict zero) ≈ 0.0605 V

## Task 6

```bash
export GOOGLE_API_KEY=...          # or ANTHROPIC_API_KEY=...
python3 run_task6.py --data /path/to/SNLS80mV.mat --model gemini-3.8-flash
```

The model name comes from `--model` or the env var `AGENT_MODEL`
(default: `gemini-3.8-flash` when `GOOGLE_API_KEY` is set; otherwise Claude).

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

**Live run on Task 5 (Gemini 3.8 Flash):**
- Turn order: orchestrator → scientist → orchestrator → executor → orchestrator
- TRAIN_RMSE = 0.001793, TEST_RMSE = 0.001596
- Artefacts: `runs/20261011-233946/`

## What was verified

- Task 5: full budget on real `SNLS80mV.mat`. Kept model `mlp-2x64-silu-adam`, free-run
  TRAIN_RMSE=0.001545, TEST_RMSE=0.003778 (baseline ≈0.0605 V). Plot in `results/test_prediction.png`.
- Task 6: `python3 test_task6_offline.py` passes 7 checks (routing, 8-decision cap, finish guard,
  repair after a failing script, 5-minute limit, tools only on the executor, RMSE claimed in prose ignored).
  Live run with Gemini completed successfully (see above).
- Code: https://github.com/mariammnatsakanyan09-dot/Task-5-6

Limit of the guard: it trusts what a script prints. A script that prints a hard-coded `TEST_RMSE` line would pass;
the executor prompt forbids it and the transcript shows the script, but the code does not check how the number was made.
