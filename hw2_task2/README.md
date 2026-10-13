# Homework 2, Task 2: focusing through a simulated bench, with an agent

A simulated bench (laser, DMD with amplitude only, scattering medium, noisy camera), a tool `focus_search` that runs the
whole focusing search on it, and the Homework 1, Task 6 agent (orchestrator, scientist, general agent, executor) with that
tool added to the executor. You type one request, **"Focus the beam."**, and from then on the agent works alone: it chooses
the macropixel size, the region sizes and the number of iterations, calls the tool, keeps the enhancement the tool returns
and decides what to try next.

## Files

| file | what it is |
|---|---|
| `bench.py` | the simulated bench and the tool `focus_search(...)` (physics, search, camera images, `experiments.json`) |
| `focus_agents.py` | the agent: the Task 6 graph, the executor with `focus_search` and `run_python`, guards, OpenRouter models |
| `run_task2.py` | runs the agent on "Focus the beam." and prints every decision and tool call live (this is what the video shows) |
| `make_report.py` | builds the Task 2 section of the PDF (table, camera images, comparison with the lecture) from a finished run |
| `task2_prompt.txt` | the Task 2 statement, given to the system as background |
| `test_bench.py` | 10 checks of the bench and the tool (no model needed) |
| `test_agents_offline.py` | 15 checks of the agent with scripted fake models |
| `test_wire_fake_openrouter.py` | the real runner, the real OpenRouter client and the real report builder against a local fake server (no key) |
| `list_openrouter_models.py` | lists the free OpenRouter models that support tool calling |

## Step by step

**1. Put the folder in your repo, next to `hw2_task1`.** Copy `hw2_task2` into `/Users/mari/Desktop/Task5-6/`.

**2. Install and test** (a few seconds, no key needed):

```bash
cd /Users/mari/Desktop/Task5-6/hw2_task2
pip install -r requirements.txt
python test_bench.py
python test_agents_offline.py
python test_wire_fake_openrouter.py
```

**3. Key.** Use a NEW OpenRouter key (the first one was printed in a terminal log: delete it at https://openrouter.ai/keys).
Never paste a key into a chat or a file.

```bash
export OPENROUTER_API_KEY=sk-or-...          # the new key; valid for this terminal window only
```

**4. Free-tier budget.** A run sends about 11 to 20 requests to the model (the terminal prints the number at the end).
The free tier allows 50 requests per day and 20 per minute. Check what is left at https://openrouter.ai/activity; if
Task 1 used most of today's 50, wait for the daily reset (usually 00:00 UTC) or buy 10 credits (then 1000 per day).
When the daily quota is gone the run stops with a clear message instead of retrying.

**5. The run** (same model as in Task 1):

```bash
python run_task2.py --model nvidia/nemotron-3-super-120b-a12b:free
```

You type nothing after this. The terminal shows the orchestrator's decisions, the scientist's plan, every tool call with its
settings, the progress of each experiment, and the result. At the end it prints the table and the best enhancement.
Everything is saved in `runs/<timestamp>/` (`experiments.json`, `exp_NN_images.png`, `report.md`, `transcript.md`, ...).
If the run is poor (no experiments, a model that never calls the tool), run it again with another free model
(`python list_openrouter_models.py`); a second run costs another 11 to 20 requests.

**6. The video** (the task asks to show the agent making a decision, calling the tool and running an experiment).
Record the run of step 5, started like this so the experiments are slow enough to follow and each image opens when it is made:

```bash
python run_task2.py --model nvidia/nemotron-3-super-120b-a12b:free --pace 0.3 --show-images
```

On a Mac: press **Shift-Command-5**, choose "Record Selected Portion", drag a frame around the Terminal window, click
**Record**, then switch to the Terminal and start the command. When the summary table appears, press the stop button in the
menu bar (or Shift-Command-5 again). The `.mov` file lands on the Desktop. Put it in `hw2_task2/video/`. A few minutes of screen
recording can be large: GitHub's web upload takes at most 25 MB per file (100 MB by `git push`); QuickTime
"File > Export As > 720p" makes it smaller, or send it by email together with the PDF.

**7. The report section.**

```bash
python make_report.py runs/<timestamp> \
    --lecture-enhancement <the enhancement of the lecture's focusing result> \
    --lecture-macropixels <its number of macropixels, if the lecture says> \
    --repo-url https://github.com/mariammnatsakanyan09-dot/Task-5-6/tree/main/hw2_task2 \
    --video hw2_task2/video/task2_demo.mov
```

It writes `runs/<timestamp>/task2_section.pdf` (the table of experiments with the settings of each run and the enhancement
obtained, the camera images of every experiment, the agent's final message, the comparison with the lecture) and
`experiments_table.csv`. Every number in it is copied from the tool's records; the lecture numbers are the two you pass.
Merge the PDF into the homework PDF (or copy the table and images into your document).

**8. Push to GitHub** (same as Task 1):

```bash
cd /Users/mari/Desktop/Task5-6
git add hw2_task2
git commit -m "HW2 task 2: focusing agent"
git push
```

## The bench

```
laser -> DMD (64x64, on/off) -> scattering medium T -> camera (40x40)        E_out = |T E_in|^2
```

* **DMD and macropixels.** 64 x 64 mirrors, each on or off. The agent picks the macropixel size (1, 2, 4, 8, 16 or 32 mirrors
  per side), so N = 4096, 1024, 256, 64, 16 or 4 macropixels.
* **Medium.** One fixed complex Gaussian transmission matrix (the same medium in every experiment), smoothed over the camera
  plane so that the speckle grain is about 2 to 3 camera pixels. A random pattern with half of the DMD on gives about
  30 counts per pixel.
* **Camera.** Linear; every shot has Poisson (shot) noise and Gaussian read noise, so a second shot of the same pattern is not
  a copy of the first. `n_avg` shots can be averaged for one reading.
* **The search** (the algorithm of the task): random starting pattern; after each transmission read the focus region (a square
  of `focus_size` pixels) and the proximal background (a ring `bg_gap` pixels outside the focus, `bg_width` pixels wide, split into
  `n_bg_regions` sectors); enhancement = mean focus intensity / mean of the background regions; flip `n_flip` macropixels; keep
  the flip if the enhancement went up, otherwise undo it; repeat `iterations` times.
* **What the tool returns.** The enhancement of the final pattern **re-measured with 50 fresh shots** (the number seen during
  the search is optimistic: the search keeps whatever looked best under noise, and that is also returned, marked as biased);
  the same number for the starting pattern; the mean counts in the focus and in the ring before and after; the enhancement
  against the far field; the saved image `exp_NN_images.png` (camera before, camera after in log scale, the DMD pattern, the
  enhancement during the search) and `exp_NN_data.npz`. A setting that cannot run is rejected with a message that says what to
  change; a rejected call is not an experiment.

### Two things to know when reading the numbers

1. **The search also lowers the ring.** The enhancement is a ratio, focus / background, and the search keeps any flip that
   raises the ratio, so it makes the focus brighter *and* the ring darker. In the sweep of `python bench.py sweep` (not the
   agent) the mean counts in the focus rose 1.5 to 20 times while the ring fell to 0.26 to 0.47 of its start. That is why the
   tool also reports the focus against the **far field** (camera pixels outside the ring, never used by the search), and why
   the PDF shows both numbers. The task's definition (focus against the proximal background) is the headline number.
2. **The usual estimate** for N binary-amplitude macropixels is N / (2 pi) (the field of the pixels that help, against a
   background of half of the pixels on and unchanged). It is printed next to every result as `textbook_N_over_2pi`. It is an
   estimate for a single focus pixel with an unchanged background, not a limit: with a ring that is lowered by the search, the
   task's enhancement can be higher, and against the far field it is lower (a 3x3 focus, finite iterations and noise).
   With very many macropixels (N = 1024 or 4096) one flip changes the focus by a tiny fraction, noise hides it, and the search
   needs many iterations and averaged shots; with very few macropixels the ceiling is low. The agent has to find the middle.

## The agent

```
user: "Focus the beam."
        |
   orchestrator  <-----------------------------+
     |     |      \                             |
 scientist general executor ----------------> (each returns to the orchestrator)
 (plan)   (side Q) tools: focus_search, run_python
        finish (only after >= 3 successful experiments)
```

* **Roles are the Task 6 ones**, one shared dialogue, one StateGraph. The orchestrator's choice is one model call (structured
  output; for free models without it, one JSON object in text). The scientist writes the experimental plan (it is told what the
  settings mean, not what to pick). The executor turns the plan into tool calls.
* **The executor has two tools.** `focus_search(macropixel_size, focus_size, bg_gap, bg_width, n_bg_regions, iterations,
  [n_flip, n_avg, focus_row, focus_col])`: the first six have no default, so the agent has to choose them. `run_python(code,
  purpose)`: the Task 6 tool, kept for reading files and arithmetic; it cannot create an experiment. An executor turn can make
  up to 4 calls (in one reply or one after another); a rejected call returns the tool's message to the model, which repairs it
  in the same turn. A model without tool calling is asked for one JSON list of experiments instead (detected once, then sticky);
  a tool call that a model prints as text (`<function=...>`) is read too.
* **The enhancement is the tool's, not the model's.** The program records each successful `focus_search` result (state
  `experiments`); the table the orchestrator reads before each decision and the table and the best enhancement in the final
  report are written by the program from those records. A number that a model writes in prose, or that a `run_python` script
  prints, is never recorded.
* **Guards.** `finish` is refused (and the orchestrator decides again) until 3 experiments have succeeded, unless the decision
  budget is nearly gone; 10 decisions in total, 14 `focus_search` calls in total, 4 per executor turn.
* **Autonomy.** Nothing in the graph asks the user anything; after the request the run is unattended.

## What was verified, and what was not

* `test_bench.py` (10 checks): the macropixel field equals the pixel-level field `T E_in`; speckle grain; camera noise (two shots
  differ); regions; the search raises the enhancement about 18 times over the random start and the far-field enhancement is a real
  focus (above 1, below the ring-based number); the number seen during the search is above the re-measured one; more iterations
  and more macropixels help; same seed gives the same experiment; 17 invalid settings are rejected; files and records; progress
  lines; the time limit.
* `test_agents_offline.py` (15 checks): roles and tools, the table the orchestrator reads, the finish guard counted from tool
  results, rejected and incomplete calls, the call caps, `run_python` and prose cannot create or change an enhancement, text-mode
  and structured-output fallbacks, daily-quota message.
* `test_wire_fake_openrouter.py`: `run_task2.py`, the real `ChatOpenAI` client (tool schema included), the real bench and the
  real `make_report.py` against a local server that plays the models, in a with-tools mode and a without-tools mode.
* **Not verified: a live OpenRouter run.** The development sandbox could not reach openrouter.ai and had no key. What a particular
  free model decides (which settings, when to stop) is that model's behaviour; the code makes every step work and records what
  really happened. The table and images in your report must come from your own run.
* **Not known to the code: the lecture's focusing result.** `make_report.py` takes it from you
  (`--lecture-enhancement`, `--lecture-macropixels`); without it the comparison section shows only this run's numbers.
