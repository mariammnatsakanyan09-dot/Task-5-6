# Homework 2, Task 2: focusing through a simulated bench

A simulated wavefront-shaping bench, a tool `focus_search` that runs a whole focusing search on it, and the Homework 1
Task 6 agent (orchestrator, scientist, general agent, executor) with that tool added to the executor. You type one request,
**"Focus the beam."**; the agent then chooses the settings itself, calls the tool, reads the enhancement and decides what
to try next. Nothing asks the user anything after the request.

## The bench and the tool

```
laser -> DMD (64x64, on/off) -> scattering medium T -> camera (40x40)        E_out = |T E_in|^2
```

* **DMD:** 64 x 64 mirrors, on or off, grouped into macropixels of 1, 2, 4, 8, 16 or 32 mirrors per side
  (N = 4096, 1024, 256, 64, 16 or 4).
* **Medium and camera:** one fixed random complex transmission matrix; linear camera with shot and read noise.
* **Search:** random start; after each transmission read the focus square and the ring of background sectors around it;
  enhancement = mean focus intensity / mean of the background sectors; flip `n_flip` macropixels, keep the flip if the
  enhancement went up, otherwise undo it; repeat `iterations` times.
* **`focus_search(macropixel_size, focus_size, bg_gap, bg_width, n_bg_regions, iterations, [n_flip, n_avg, ...])`:** the first
  six settings are required, so the agent must choose them. It returns the enhancement of the final pattern
  re-measured with 50 fresh shots (the value seen during the search is biased upward), the same for the starting pattern,
  the enhancement against the far field (pixels outside the ring), and saves the camera images `exp_NN_images.png`.
  Invalid settings are rejected with a message; a rejected call is not an experiment.

## The agent

* Same roles as Task 6, one shared dialogue. The executor has two tools: `focus_search` and `run_python` (for reading files
  and arithmetic; it cannot create an experiment).
* The enhancement is the tool's: the program records every successful `focus_search` result, and the table and the best
  value in the final report are written from those records, never from numbers a model writes in text.
* Limits: `finish` is refused until 3 experiments have succeeded (unless the decisions are nearly used up); 10 decisions and
  14 `focus_search` calls per run.

## Files

| file | what it is |
|---|---|
| `bench.py` | the simulated bench and the tool `focus_search` |
| `focus_agents.py` | the agent: graph, executor with the tools, guards, OpenRouter models |
| `run_task2.py` | runs the agent on "Focus the beam." and prints every decision and tool call live |
| `make_report.py` | builds the PDF section (table, camera images, comparison with the lecture) from a finished run |
| `task2_prompt.txt` | the Task 2 statement, given to the system as background |
| `list_openrouter_models.py` | lists free OpenRouter models that support tool calling |
| `test_bench.py`, `test_agents_offline.py`, `test_wire_fake_openrouter.py` | 10 + 15 offline checks and an end-to-end check against a fake server (no key) |
| `video/task2_demo.mp4` | screen recording of the run below (compressed copy, about 4 MB) |

## Run

```bash
pip3 install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...        # free key; read from the environment, never stored in a file
python3 test_bench.py && python3 test_agents_offline.py        # no key needed
python3 run_task2.py --model nvidia/nemotron-3-super-120b-a12b:free --pace 0.3 --show-images
```

`--pace 0.3` slows the search so it can be followed on screen and `--show-images` opens each camera image when it is made
(this is what the video shows). The run is saved in `runs/<timestamp>/` (`experiments.json`, `exp_NN_images.png`, `report.md`,
`transcript.md`). The free tier allows 20 requests per minute and 50 per day; a run used 10 requests.

**Report section:**

```bash
python3 make_report.py runs/<timestamp> --lecture-enhancement 33 \
    --repo-url https://github.com/mariammnatsakanyan09-dot/Task-5-6/tree/main/hw2_task2 \
    --video hw2_task2/video/task2_demo.mp4
```

It writes `task2_section.pdf`, `experiments_table.csv` and `comparison.md` into the run folder. Without `--repo-url` and
`--video` the PDF keeps a placeholder link and has no video line. `--lecture-macropixels N` adds the lecture's N if known.

## Result (run `runs/20261013-231903`, model `nvidia/nemotron-3-super-120b-a12b:free`)

4 decisions, 4 `focus_search` calls, 10 model requests. In all four experiments the agent chose focus 3 px, ring gap 2,
ring width 4, 4 regions, flip 1, average 4, and varied the macropixel size and the iterations:

| # | macropixel (N) | iterations | enhancement, start → final | vs far field |
|---|---|---|---|---|
| 1 | 4 (256) | 200 | 1.37 → **29.1** (best) | 13.5 |
| 2 | 8 (64) | 200 | 1.49 → 22.0 | 12.6 |
| 3 | 16 (16) | 200 | 0.69 → 6.4 | 5.1 |
| 4 | 8 (64) | 800 | 1.17 → 24.7 | 10.0 |

The lecture's focusing result is an enhancement of 33; the best experiment reached 29.1, a ratio of 0.88. The usual estimate
N/(2π) for binary-amplitude macropixels is 40.7 for N = 256, so 29.1 is 0.71 of it.
In experiment 1 the focus became 9.3 times brighter (36.1 → 336.1 counts) and the ring 2.3 times darker (26.3 → 11.6 counts):
the enhancement is a ratio, so the search lowers the ring as well. Against the far field, which the search never uses, the
enhancement is 13.5.
