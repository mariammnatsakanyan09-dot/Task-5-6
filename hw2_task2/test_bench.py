"""Checks of the simulated bench and of the focus_search tool (no model, no network).

    python test_bench.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np

import bench


def quick(workdir, **kw):
    base = dict(macropixel_size=8, focus_size=3, bg_gap=2, bg_width=6, n_bg_regions=8, iterations=1500, seed=1)
    base.update(kw)
    return bench.focus_search(workdir, **base)


def test_forward_model():
    """E_out = |T E_in|^2 with E_in the DMD pattern: the macropixel matrix gives the same field as the pixel-level one."""
    T = bench.medium()
    assert T.shape == (bench.CAM_N, bench.CAM_N, bench.DMD_N, bench.DMD_N) and np.iscomplexobj(T)
    rng = np.random.default_rng(3)
    for b in (8, 4):
        m = bench.DMD_N // b
        s = (rng.random(m * m) < 0.5).astype(np.float32)
        pix = np.kron(s.reshape(m, m), np.ones((b, b), np.float32))              # the DMD pattern, pixel by pixel
        want = T.reshape(bench.CAM_N ** 2, -1) @ pix.ravel().astype(np.complex64)
        got = bench.macro_T(b) @ s.astype(np.complex64)
        assert np.allclose(want, got, rtol=1e-3, atol=1e-3 * np.abs(want).max()), b
    # a random pattern with half of the DMD on gives about BG_LEVEL counts per pixel
    s = (rng.random(64) < 0.5).astype(np.float32)
    mean_I = np.mean(np.abs(bench.macro_T(8) @ s.astype(np.complex64)) ** 2)
    assert 0.5 * bench.BG_LEVEL < mean_I < 2.0 * bench.BG_LEVEL, mean_I
    print("ok  B1 forward model: macropixel field = pixel-level field, mean level of a random pattern", round(float(mean_I), 1))


def test_speckle_grain():
    """The speckle grain is a few camera pixels (neighbouring pixels are correlated, far ones are not)."""
    rng = np.random.default_rng(5)
    c1, c4 = [], []
    for _ in range(20):
        s = (rng.random(64) < 0.5).astype(np.float32)
        I = (np.abs(bench.macro_T(8) @ s.astype(np.complex64)) ** 2).reshape(40, 40)
        I = I - I.mean()
        c1.append((I[:, :-1] * I[:, 1:]).mean() / I.var())
        c4.append((I[:, :-6] * I[:, 6:]).mean() / I.var())
    assert np.mean(c1) > 0.4 and abs(np.mean(c4)) < 0.15, (np.mean(c1), np.mean(c4))
    print(f"ok  B2 speckle grain: intensity correlation {np.mean(c1):.2f} at 1 px, {np.mean(c4):.2f} at 6 px")


def test_camera_noise():
    """A second shot of the same pattern is not a copy of the first."""
    s = (np.random.default_rng(0).random(64) < 0.5).astype(np.float32)
    rng = np.random.default_rng(1)
    a, I = bench.frame_from_pattern(s, 8, rng)
    b, I2 = bench.frame_from_pattern(s, 8, rng)
    assert np.array_equal(I, I2) and not np.array_equal(a, b)
    rel = np.std(a - b) / np.sqrt(2) / np.sqrt(I.mean())
    assert 0.6 < rel < 1.6, rel                 # about Poisson: std ~ sqrt(mean)
    print("ok  B3 camera noise: two shots of one pattern differ (std of the difference is about shot noise)")


def test_regions():
    focus, ring, sector, _ = bench.regions(3, 2, 6, 8, 20, 20)
    assert focus.sum() == 9 and focus[20, 20] and not (focus & ring).any()
    assert not ring[18:23, 18:23].any()                       # the gap: nothing within 2 pixels of the focus edge
    counts = np.bincount(sector[ring], minlength=8)
    assert counts.min() > 0 and counts.max() < 1.4 * counts.min(), counts
    print("ok  B4 regions: 3x3 focus, ring outside the gap, 8 sectors of similar size", counts.tolist())


def test_search_improves_and_is_unbiased():
    with tempfile.TemporaryDirectory() as d:
        res = [quick(d, seed=s) for s in range(1, 11)]
    assert all(r["ok"] for r in res)
    gain = np.mean([r["enhancement"] / r["enhancement_initial"] for r in res])
    assert gain > 5, gain
    assert np.mean([r["enhancement"] for r in res]) > 0.8 * res[0]["textbook_N_over_2pi"]
    biased = np.mean([r["enhancement_end_of_search"] for r in res])
    fresh = np.mean([r["enhancement"] for r in res])
    assert biased > fresh, (biased, fresh)                    # the number seen during the search is optimistic
    assert all(r["focus_gain"] > 1 for r in res)
    far = np.mean([r["enhancement_vs_far_field"] for r in res])
    assert 1 < far < fresh, (far, fresh)          # against an area the search never used the number is smaller, still a real focus
    print(f"ok  B5 the search works: enhancement x{gain:.0f} over the random start; seen-during-search {biased:.1f} > re-measured {fresh:.1f}; "
          f"against the far field {far:.1f}")


def test_more_iterations_and_pixels_help():
    with tempfile.TemporaryDirectory() as d:
        few = np.mean([quick(d, seed=s, iterations=40)["enhancement"] for s in range(1, 5)])
        many = np.mean([quick(d, seed=s, iterations=1500)["enhancement"] for s in range(1, 5)])
        small = np.mean([quick(d, seed=s, macropixel_size=16, iterations=1500)["enhancement"] for s in range(1, 5)])
        big = np.mean([quick(d, seed=s, macropixel_size=4, iterations=6000)["enhancement"] for s in range(1, 5)])
    assert many > 2 * few and big > 2 * small, (few, many, small, big)
    print(f"ok  B6 40 -> 1500 iterations: {few:.1f} -> {many:.1f}; 16 -> 256 macropixels: {small:.1f} -> {big:.1f}")


def test_seed_and_repeat():
    with tempfile.TemporaryDirectory() as d:
        a, b = quick(d, seed=11), quick(d, seed=11)
        c, e = quick(d, seed=None), quick(d, seed=None)
    assert a["enhancement"] == b["enhancement"] and a["settings"]["seed"] == 11
    assert c["settings"]["seed"] != e["settings"]["seed"] and c["enhancement"] != e["enhancement"]
    print("ok  B7 same seed = same experiment; no seed = a new random start and new noise each time")


def test_rejects_bad_settings():
    with tempfile.TemporaryDirectory() as d:
        bad = [dict(macropixel_size=3), dict(macropixel_size=64), dict(focus_size=0), dict(focus_size=12), dict(bg_gap=9),
               dict(bg_width=0), dict(n_bg_regions=0), dict(n_bg_regions=40), dict(iterations=3), dict(iterations=10 ** 7),
               dict(n_flip=0), dict(n_flip=10 ** 6), dict(n_avg=0), dict(n_avg=500), dict(focus_row=2), dict(focus_col=39),
               dict(macropixel_size=32, focus_size=1, bg_gap=0, bg_width=1, n_bg_regions=16)]
        for kw in bad:
            r = quick(d, **kw)
            assert r["ok"] is False and r["error"], kw
        assert not (Path(d) / "experiments.json").exists()          # a rejected call is not an experiment
        assert "one of" in quick(d, macropixel_size=3)["error"]
    print(f"ok  B8 {len(bad)} invalid settings are rejected with a message that says what to change; nothing recorded")


def test_files_and_records():
    with tempfile.TemporaryDirectory() as d:
        r1, r2 = quick(d, seed=1), quick(d, seed=2, macropixel_size=16)
        exps = json.loads((Path(d) / "experiments.json").read_text())
        assert [e["experiment"] for e in exps] == [1, 2] and exps[1]["settings"]["macropixel_size"] == 16
        assert exps[0]["enhancement"] == r1["enhancement"]
        for k in (1, 2):
            assert (Path(d) / f"exp_{k:02d}_images.png").stat().st_size > 5000
            z = np.load(Path(d) / f"exp_{k:02d}_data.npz")
            assert z["camera_initial"].shape == (40, 40) and z["camera_final"].shape == (40, 40)
        z = np.load(Path(d) / "exp_01_data.npz")
        fi = z["camera_final"]
        assert fi[18:23, 18:23].max() > 5 * np.median(fi)           # the spot is in the image
        json.dumps(r1)                                              # JSON-serialisable
    print("ok  B9 experiments.json, images and arrays are written; the focus is visible in the final camera frame")


def test_progress_and_time_limit():
    calls = []
    with tempfile.TemporaryDirectory() as d:
        quick(d, iterations=4000, progress=lambda i, n, e: calls.append((i, n)))
        assert 8 <= len(calls) <= 12 and calls[-1][0] <= 4000, calls
        old = bench.TIME_LIMIT_S
        bench.TIME_LIMIT_S = 0.0
        try:
            r = quick(d, iterations=20000)
        finally:
            bench.TIME_LIMIT_S = old
        assert r["stopped_early"] and r["iterations_done"] < 20000
    print(f"ok  B10 progress is reported ({len(calls)} lines); the time limit stops a long search and says so")


if __name__ == "__main__":
    test_forward_model()
    test_speckle_grain()
    test_camera_noise()
    test_regions()
    test_search_improves_and_is_unbiased()
    test_more_iterations_and_pixels_help()
    test_seed_and_repeat()
    test_rejects_bad_settings()
    test_files_and_records()
    test_progress_and_time_limit()
    print("\nall bench checks passed")
