"""Task 2: a simulated focusing bench and the focus-search tool.

Physics (the part of the laboratory path that forms the image):
    laser -> DMD (amplitude only) -> scattering medium -> camera
    E_out = |T E_in|^2,   T = complex transmission matrix of that path, E_in = the DMD amplitude pattern.

The simulation
    * DMD: 64 x 64 pixels, binary amplitude (on/off). Pixels are grouped in square MACROPIXELS (the agent picks the size).
    * Scattering medium: T is one fixed complex Gaussian matrix (same medium in every experiment, MEDIUM_SEED), with a
      speckle grain of about two camera pixels (T is smoothed over the camera plane).
    * Camera: 40 x 40 pixels, linear. Every shot has Poisson (shot) noise and Gaussian read noise, so a second shot of the
      same pattern is not a copy of the first. n_avg shots can be averaged.

The search (the algorithm of the task)
    1. divide the DMD into macropixels, start from a random pattern;
    2. after each transmission read the intensity in the focus region and in the proximal background;
    3. the background is a ring around the focus region, split into several sector regions;
       enhancement = mean focus intensity / mean of the background-region intensities;
    4. flip some macropixels; keep the flip if the enhancement went up, otherwise undo it; repeat.

The reported enhancement is NOT the last number seen during the search (that one is biased upward by noise: the search keeps
whatever looked best). It is re-measured on the final pattern with 50 fresh shots, and the starting pattern is re-measured
the same way, so enhancement / enhancement_initial are comparable.

    python bench.py sweep      # a reference sweep (NOT the agent): enhancement against macropixel size
"""
from __future__ import annotations

import json
import math
import secrets
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

DMD_N = 64            # DMD pixels per side
CAM_N = 40            # camera pixels per side
GRAIN_SIGMA = 1.2     # camera-plane smoothing of T (speckle grain ~ 2-3 camera pixels)
BG_LEVEL = 30.0       # mean counts per camera pixel for a random pattern with half of the DMD on
READ_NOISE = 3.0      # camera read noise, counts rms
EVAL_FRAMES = 50      # shots averaged when the initial and the final pattern are re-measured
MEDIUM_SEED = 7
VALID_MACROPIXEL = [b for b in (1, 2, 4, 8, 16, 32) if DMD_N % b == 0]
MAX_ITERATIONS = 200_000
TIME_LIMIT_S = 150.0

_T_CACHE: dict = {}


# ----------------------------------------------------------------------------------- the medium
def medium() -> np.ndarray:
    """T at DMD-pixel level, shape (CAM_N, CAM_N, DMD_N, DMD_N), complex64. Built once."""
    if "T" not in _T_CACHE:
        rng = np.random.default_rng(MEDIUM_SEED)
        shape = (CAM_N, CAM_N, DMD_N, DMD_N)
        parts = []
        for _ in range(2):
            a = rng.standard_normal(shape, dtype=np.float32)
            parts.append(ndimage.gaussian_filter(a, sigma=(GRAIN_SIGMA, GRAIN_SIGMA, 0, 0), mode="wrap"))
        T = (parts[0] + 1j * parts[1]).astype(np.complex64)
        # a random pattern with half of the DMD pixels on gives BG_LEVEL counts on average
        T *= np.float32(math.sqrt(2.0 * BG_LEVEL / DMD_N ** 2 / float(np.mean(np.abs(T) ** 2))))
        _T_CACHE["T"] = T
    return _T_CACHE["T"]


def macro_T(b: int) -> np.ndarray:
    """T aggregated to macropixels of b x b DMD pixels: shape (CAM_N*CAM_N, M), M = (DMD_N/b)^2."""
    key = ("macro", b)
    if key not in _T_CACHE:
        T = medium()
        m = DMD_N // b
        _T_CACHE[key] = np.ascontiguousarray(
            T.reshape(CAM_N, CAM_N, m, b, m, b).sum(axis=(3, 5)).reshape(CAM_N * CAM_N, m * m))
    return _T_CACHE[key]


# ----------------------------------------------------------------------------------- regions
def regions(focus_size: int, gap: int, width: int, n_bg: int, row: int, col: int):
    """Boolean focus mask, ring mask and sector index of every camera pixel (the ring is split in n_bg sectors)."""
    yy, xx = np.mgrid[0:CAM_N, 0:CAM_N]
    r0, c0 = row - focus_size // 2, col - focus_size // 2
    focus = (yy >= r0) & (yy < r0 + focus_size) & (xx >= c0) & (xx < c0 + focus_size)
    cy, cx = r0 + (focus_size - 1) / 2.0, c0 + (focus_size - 1) / 2.0
    d = np.maximum(np.abs(yy - cy), np.abs(xx - cx))
    h = focus_size / 2.0
    ring = (d > h + gap) & (d <= h + gap + width)
    ang = np.arctan2(yy - cy, xx - cx)
    sector = np.floor((ang + np.pi) / (2 * np.pi / n_bg)).astype(int) % n_bg
    return focus, ring, sector, (cy, cx, h)


def far_mask(focus_size: int, gap: int, width: int, row: int, col: int, margin: int = 2) -> np.ndarray:
    """Camera pixels the search never looks at: farther than the outer edge of the ring plus a margin."""
    yy, xx = np.mgrid[0:CAM_N, 0:CAM_N]
    r0, c0 = row - focus_size // 2, col - focus_size // 2
    cy, cx = r0 + (focus_size - 1) / 2.0, c0 + (focus_size - 1) / 2.0
    d = np.maximum(np.abs(yy - cy), np.abs(xx - cx))
    return d > focus_size / 2.0 + gap + width + margin


def check_settings(s: dict) -> Optional[str]:
    """None if the settings can run, else a message that says what to change."""
    if s["macropixel_size"] not in VALID_MACROPIXEL:
        return f"macropixel_size must be one of {VALID_MACROPIXEL} (the DMD is {DMD_N}x{DMD_N} pixels)."
    if not 1 <= s["focus_size"] <= 9:
        return "focus_size must be between 1 and 9 camera pixels."
    if not 0 <= s["bg_gap"] <= 6:
        return "bg_gap must be between 0 and 6 camera pixels."
    if not 1 <= s["bg_width"] <= 12:
        return "bg_width must be between 1 and 12 camera pixels."
    if not 1 <= s["n_bg_regions"] <= 16:
        return "n_bg_regions must be between 1 and 16."
    if not 10 <= s["iterations"] <= MAX_ITERATIONS:
        return f"iterations must be between 10 and {MAX_ITERATIONS}."
    n_mac = (DMD_N // s["macropixel_size"]) ** 2
    if not 1 <= s["n_flip"] <= max(1, n_mac // 2):
        return f"n_flip must be between 1 and {max(1, n_mac // 2)} for {n_mac} macropixels."
    if not 1 <= s["n_avg"] <= 64:
        return "n_avg must be between 1 and 64 camera shots per measurement."
    extent = s["focus_size"] / 2.0 + s["bg_gap"] + s["bg_width"]
    for name, v in (("focus_row", s["focus_row"]), ("focus_col", s["focus_col"])):
        if not (extent - 0.5 <= v <= CAM_N - 1 - extent + 0.5):
            return (f"{name}={v} leaves the background ring outside the {CAM_N}x{CAM_N} camera; "
                    f"use a value between {math.ceil(extent - 0.5)} and {math.floor(CAM_N - 1 - extent + 0.5)} "
                    f"or a smaller ring (bg_gap + bg_width).")
    return None


# ----------------------------------------------------------------------------------- the search
def run_search(s: dict, rng: np.random.Generator, progress=None) -> dict:
    b = s["macropixel_size"]
    m_side = DMD_N // b
    M = m_side * m_side
    focus, ring, sector, geom = regions(s["focus_size"], s["bg_gap"], s["bg_width"], s["n_bg_regions"],
                                        s["focus_row"], s["focus_col"])
    fidx = np.flatnonzero(focus.ravel())
    ridx = np.flatnonzero(ring.ravel())
    sec = sector.ravel()[ridx]
    counts = np.bincount(sec, minlength=s["n_bg_regions"])
    if (counts == 0).any():
        raise ValueError("the background ring is too small for that many regions; use fewer n_bg_regions "
                         "or a wider ring (bg_width) / larger focus_size.")
    nf = fidx.size
    pix = np.concatenate([fidx, ridx])
    Tfull = macro_T(b)                              # (cam pixels, M)
    TsubT = np.ascontiguousarray(Tfull[pix].T)      # (M, n region pixels): a flip adds rows of this

    def measure(E, n_avg):
        """eta from one camera measurement (n_avg shots averaged) of the field E at the region pixels."""
        I = (E.real ** 2 + E.imag ** 2).astype(np.float64)
        vals = (rng.poisson(I * n_avg) + rng.normal(0.0, READ_NOISE * math.sqrt(n_avg), I.shape)) / n_avg
        bg = np.bincount(sec, weights=vals[nf:], minlength=s["n_bg_regions"]) / counts
        return float(vals[:nf].mean() / max(bg.mean(), 1e-9)), float(vals[:nf].mean()), float(bg.mean())

    state = (rng.random(M) < 0.5).astype(np.float32)             # 1. random starting pattern
    state_start = state.copy()
    E = (state.astype(np.complex64) @ TsubT).astype(np.complex64)
    E_start = E.copy()
    eta_cur, _, _ = measure(E, s["n_avg"])
    eta0_search = eta_cur
    trace_every = max(1, s["iterations"] // 200)
    trace, t0, stopped_early, accepted = [(0, eta_cur)], time.time(), False, 0
    for it in range(1, s["iterations"] + 1):
        idx = rng.choice(M, size=s["n_flip"], replace=False)       # 4. flip some macropixels
        delta = (1.0 - 2.0 * state[idx]).astype(np.complex64)      # off->on = +1, on->off = -1
        E_new = E + delta @ TsubT[idx]
        eta_new, _, _ = measure(E_new, s["n_avg"])                 # 2./3. read focus and background
        if eta_new > eta_cur:                                      # keep the flip only if the enhancement went up
            state[idx] = 1.0 - state[idx]
            E, eta_cur = E_new, eta_new
            accepted += 1
        if it % trace_every == 0:
            trace.append((it, eta_cur))
            if progress is not None and (it // trace_every) % 20 == 0:     # about ten lines per experiment
                progress(it, s["iterations"], eta_cur)
        if it % 500 == 0 and time.time() - t0 > TIME_LIMIT_S:
            stopped_early = True
            break
    iters_done = it

    # unbiased numbers: fresh shots (EVAL_FRAMES averaged) of the final pattern and of the starting pattern
    seconds = time.time() - t0
    fin = measure(E, EVAL_FRAMES)
    init = measure(E_start, EVAL_FRAMES)
    far = far_mask(s["focus_size"], s["bg_gap"], s["bg_width"], s["focus_row"], s["focus_col"]).ravel()

    def far_counts(st):                     # mean counts over the far field, from fresh shots of the whole camera
        If = np.abs(Tfull @ st.astype(np.complex64)) ** 2
        shots = (rng.poisson(If * EVAL_FRAMES) + rng.normal(0.0, READ_NOISE * math.sqrt(EVAL_FRAMES), If.shape)) / EVAL_FRAMES
        return float(shots[far].mean()) if far.any() else float("nan")
    far_fin, far_init = far_counts(state), far_counts(state_start)
    return dict(state=state, state_start=state_start, eta_search_end=eta_cur, eta_search_start=eta0_search,
                trace=trace, accepted=accepted, iterations_done=iters_done, stopped_early=stopped_early,
                fresh_final=fin, fresh_initial=init, far_final=far_fin, far_initial=far_init, geom=geom, M=M,
                seconds=seconds)


def frame_from_pattern(state: np.ndarray, b: int, rng: np.random.Generator) -> tuple:
    """One noisy camera shot of the whole camera (and the noiseless intensity) for a macropixel pattern."""
    E = macro_T(b) @ state.astype(np.complex64)
    I = (E.real ** 2 + E.imag ** 2).astype(np.float64).reshape(CAM_N, CAM_N)
    shot = rng.poisson(I) + rng.normal(0.0, READ_NOISE, I.shape)
    return np.maximum(shot, 0.0), I


def focus_search(workdir, macropixel_size: int = 8, focus_size: int = 3, bg_gap: int = 2, bg_width: int = 6,
                 n_bg_regions: int = 8, iterations: int = 3000, n_flip: int = 1, n_avg: int = 1,
                 focus_row: int = 20, focus_col: int = 20, seed: Optional[int] = None, progress=None) -> dict:
    """Run one focusing experiment on the simulated bench. Saves a figure and the arrays, appends the experiment to
    experiments.json in `workdir`, and returns the result (a plain dict, JSON-serialisable).
    progress(iteration, iterations, enhancement_seen) is called about ten times during the search (optional)."""
    s = dict(macropixel_size=int(macropixel_size), focus_size=int(focus_size), bg_gap=int(bg_gap),
             bg_width=int(bg_width), n_bg_regions=int(n_bg_regions), iterations=int(iterations),
             n_flip=int(n_flip), n_avg=int(n_avg), focus_row=int(focus_row), focus_col=int(focus_col))
    problem = check_settings(s)
    if problem:
        return dict(ok=False, error=problem, settings=s)
    seed = int(seed) if seed is not None else secrets.randbelow(2 ** 31)
    s["seed"] = seed
    rng = np.random.default_rng(seed)
    try:
        r = run_search(s, rng, progress)
    except ValueError as e:
        return dict(ok=False, error=str(e), settings=s)

    b = s["macropixel_size"]
    init, fin = r["fresh_initial"], r["fresh_final"]         # (enhancement, focus counts, background counts)
    state0 = r["state_start"]

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    exp_file = workdir / "experiments.json"
    experiments = json.loads(exp_file.read_text()) if exp_file.exists() else []
    k = len(experiments) + 1
    rng_img = np.random.default_rng(seed + 1)
    frame0, _ = frame_from_pattern(state0, b, rng_img)
    frame1, I1 = frame_from_pattern(r["state"], b, rng_img)
    npz = workdir / f"exp_{k:02d}_data.npz"
    np.savez_compressed(npz, camera_initial=frame0, camera_final=frame1, pattern=r["state"].reshape(DMD_N // b, -1),
                        trace=np.array(r["trace"]))
    fig_path = workdir / f"exp_{k:02d}_images.png"
    far_enh = None if math.isnan(r["far_final"]) else fin[1] / r["far_final"]
    make_figure(fig_path, k, s, frame0, frame1, r, init[0], fin[0], far_enh)

    n_mac = r["M"]
    result = dict(
        ok=True, experiment=k, settings=s,
        enhancement=round(fin[0], 3), enhancement_initial=round(init[0], 3),
        enhancement_end_of_search=round(r["eta_search_end"], 3),
        focus_mean_counts=round(fin[1], 2), background_mean_counts=round(fin[2], 2),
        focus_mean_counts_initial=round(init[1], 2), background_mean_counts_initial=round(init[2], 2),
        focus_gain=round(fin[1] / max(init[1], 1e-9), 2), background_ratio=round(fin[2] / max(init[2], 1e-9), 3),
        enhancement_vs_far_field=(None if math.isnan(r["far_final"]) else round(fin[1] / r["far_final"], 3)),
        enhancement_vs_far_field_initial=(None if math.isnan(r["far_initial"]) else round(init[1] / r["far_initial"], 3)),
        far_field_mean_counts=(None if math.isnan(r["far_final"]) else round(r["far_final"], 2)),
        n_macropixels=n_mac, textbook_N_over_2pi=round(n_mac / (2 * math.pi), 2),
        iterations_done=r["iterations_done"], flips_accepted=r["accepted"], stopped_early=r["stopped_early"],
        seconds=round(r["seconds"], 2), figure=fig_path.name, data=npz.name,
        note=("enhancement = mean focus intensity / mean of the background regions, re-measured on the final pattern with "
              f"{EVAL_FRAMES} fresh shots (enhancement_end_of_search is the biased number seen during the search). "
              "The search maximises focus/background, so the ring background can fall as well as the focus rising: "
              "focus_gain = focus after / focus before, background_ratio = background after / background before. "
              "enhancement_vs_far_field is the same focus intensity divided by the mean over camera pixels outside the ring "
              "that the search never used (a check that the enhancement is not only a lowered ring). "
              "textbook_N_over_2pi is the usual estimate for N binary-amplitude macropixels when the background stays unchanged."))
    experiments.append(result)
    exp_file.write_text(json.dumps(experiments, indent=1))
    return result


# ----------------------------------------------------------------------------------- figure
def make_figure(path, k, s, frame0, frame1, r, eta0, eta1, far1=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    cy, cx, h = r["geom"]
    vmax = max(frame1.max(), 10.0)
    norm = LogNorm(vmin=1.0, vmax=vmax)
    fig, axs = plt.subplots(1, 4, figsize=(14.5, 3.5), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25]))
    far_txt = "" if far1 is None else f" (vs far field {far1:.1f})"
    for ax, fr, title in ((axs[0], frame0, f"start (random pattern)\nenhancement {eta0:.1f}"),
                          (axs[1], frame1, f"after the search\nenhancement {eta1:.1f}{far_txt}")):
        im = ax.imshow(np.maximum(fr, 1.0), cmap="inferno", norm=norm, origin="upper")
        ax.add_patch(Rectangle((cx - h, cy - h), 2 * h, 2 * h, fill=False, ec="cyan", lw=1.2))
        for rad in (h + s["bg_gap"], h + s["bg_gap"] + s["bg_width"]):
            ax.add_patch(Rectangle((cx - rad, cy - rad), 2 * rad, 2 * rad, fill=False, ec="white", lw=0.7, ls="--"))
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.03)
    cb.set_label("counts (log)", fontsize=8)
    axs[2].imshow(r["state"].reshape(DMD_N // s["macropixel_size"], -1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axs[2].set_title(f"DMD pattern, {r['M']} macropixels of {s['macropixel_size']}x{s['macropixel_size']}", fontsize=9)
    axs[2].set_xticks([]); axs[2].set_yticks([])
    it, eta = zip(*r["trace"])
    axs[3].plot(it, eta, color="#2a78d6", lw=1.4, label="seen during the search")
    axs[3].axhline(eta1, color="#eb6834", lw=1.2, ls="--", label=f"re-measured at the end: {eta1:.1f}")
    axs[3].axhline(r["M"] / (2 * math.pi), color="gray", lw=1, ls=":", label=f"textbook N/(2 pi) = {r['M'] / (2 * math.pi):.1f}")
    axs[3].set_xlabel("iteration", fontsize=8); axs[3].set_ylabel("enhancement", fontsize=8)
    axs[3].tick_params(labelsize=8); axs[3].legend(fontsize=7, frameon=False, loc="lower right")
    for sp in ("top", "right"):
        axs[3].spines[sp].set_visible(False)
    fig.suptitle(f"experiment {k}: macropixel {s['macropixel_size']}, focus {s['focus_size']}px, background ring gap {s['bg_gap']} "
                 f"width {s['bg_width']} in {s['n_bg_regions']} regions, {r['iterations_done']} iterations, flip {s['n_flip']}, "
                 f"average {s['n_avg']}", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------- reference sweep (not the agent)
def sweep(workdir="bench_sweep"):
    print("reference sweep with fixed settings (NOT the agent): focus 3 px, ring gap 2 / width 6 / 8 regions, 10 N iterations")
    print(f"{'macropixel':>10s} {'N':>6s} {'N/2pi':>7s} {'initial':>8s} {'final':>8s} {'focus x':>8s} {'bg x':>7s} {'s':>6s}")
    for b in (16, 8, 4, 2):
        n = (DMD_N // b) ** 2
        r = focus_search(workdir, macropixel_size=b, iterations=10 * n, seed=1)
        print(f"{b:10d} {n:6d} {r['textbook_N_over_2pi']:7.1f} {r['enhancement_initial']:8.2f} {r['enhancement']:8.1f} "
              f"{r['focus_gain']:8.1f} {r['background_ratio']:7.2f} {r['seconds']:6.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep()
    else:
        print(__doc__)
