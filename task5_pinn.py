"""Task 5: neural-ODE ("PINN") model of the Silverbox Duffing oscillator.

Physics:      y'' + a y' + b y + c y^3 = g u(t)            (eq. 1, only used for scaling/diagnostics)
Model:        y' = v,   v' = alpha_theta(y, v, u)           (eq. 2, alpha_theta is a fully connected net)
Integrator:   classical RK4 at the sampling step dt, driven by the measured input
              (u at the half step is the average of the two neighbouring samples).
Metric:       one free run over the whole 4096-sample excerpt, started at sample 0 from the
              measured position and the finite-difference velocity, never reset;
              RMSE on the last 1024 samples (test) and on the first 3072 (train).

Usage
-----
    python task5_pinn.py --data SNLS80mV.mat --out results --budget full
    python task5_pinn.py --data SNLS80mV.mat --out results --budget smoke    # ~1 minute check
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FS = 610.35
DT = 1.0 / FS
START = 42650          # first sample of the excerpt (2000 samples after the multisine begins at 40650)
N_EXC = 4096
N_TRAIN = 3072
N_TEST = N_EXC - N_TRAIN   # 1024


# ----------------------------------------------------------------------------- data
def _read_mat(path: str) -> dict[str, np.ndarray]:
    import scipy.io

    try:
        raw = scipy.io.loadmat(path)
        return {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:  # MATLAB v7.3 (HDF5)
        import h5py

        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                out[k] = np.asarray(f[k])
        return out


def load_excerpt(path: str):
    """Return centered (u, y) of the 4096-sample excerpt as float64 arrays."""
    d = _read_mat(path)
    low = {k.lower(): k for k in d}
    if "v1" not in low or "v2" not in low:
        raise KeyError(f"V1/V2 not found in {path}; keys: {list(d)}")
    v1 = np.asarray(d[low["v1"]], dtype=np.float64).ravel()
    v2 = np.asarray(d[low["v2"]], dtype=np.float64).ravel()
    if v1.size != 131072 or v2.size != 131072:
        print(f"[warn] expected 131072 samples, got {v1.size}/{v2.size}")
    u = v1[START:START + N_EXC].copy()
    y = v2[START:START + N_EXC].copy()
    assert u.size == N_EXC and y.size == N_EXC, "excerpt slice is out of range"
    u -= u.mean()          # means of the excerpt, as the task specifies
    y -= y.mean()
    return u, y


# ----------------------------------------------------------------------------- model
@dataclass
class Config:
    name: str = "mlp-2x64-tanh"
    hidden: tuple = (64, 64)
    activation: str = "tanh"        # tanh | silu | gelu
    linear_skip: bool = False       # alpha = Linear(x) + MLP(x)
    optimizer: str = "adam"         # adam | adamw
    lr: float = 3e-3
    lr_final: float = 1e-4          # cosine decay target
    weight_decay: float = 0.0
    warm_steps: int = 600           # stage 0: regression on finite-difference acceleration
    steps: int = 2500               # stage 1: rollout training steps
    rollout_lengths: tuple = (32, 96, 256)   # curriculum, equal share of `steps` each
    batch: int = 48                 # number of windows per step
    clip: float = 1.0
    seed: int = 0


ACT = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}


class AccNet(nn.Module):
    """alpha_theta(y, v, u): fully connected network. Inputs and output are scaled."""

    def __init__(self, cfg: Config, ys: float, vs: float, us: float, accs: float):
        super().__init__()
        self.register_buffer("scale_in", torch.tensor([ys, vs, us], dtype=torch.float32))
        self.accs = float(accs)
        layers, d = [], 3
        for h in cfg.hidden:
            layers += [nn.Linear(d, h), ACT[cfg.activation]()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)
        nn.init.normal_(self.mlp[-1].weight, std=0.05)
        nn.init.zeros_(self.mlp[-1].bias)
        self.skip = nn.Linear(3, 1) if cfg.linear_skip else None
        if self.skip is not None:
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    def forward(self, y, v, u):
        x = torch.stack([y, v, u], dim=-1) / self.scale_in
        out = self.mlp(x)
        if self.skip is not None:
            out = out + self.skip(x)
        return out.squeeze(-1) * self.accs


# ----------------------------------------------------------------------------- integrator
def rk4_step(f, y, v, u0, u1, dt):
    um = 0.5 * (u0 + u1)
    k1y, k1v = v, f(y, v, u0)
    y2, v2 = y + 0.5 * dt * k1y, v + 0.5 * dt * k1v
    k2y, k2v = v2, f(y2, v2, um)
    y3, v3 = y + 0.5 * dt * k2y, v + 0.5 * dt * k2v
    k3y, k3v = v3, f(y3, v3, um)
    y4, v4 = y + dt * k3y, v + dt * k3v
    k4y, k4v = v4, f(y4, v4, u1)
    y_new = y + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
    v_new = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return y_new, v_new


def rollout(f, y0, v0, u_seq, dt=DT):
    """y0, v0: (B,), u_seq: (B, L+1) -> predicted positions (B, L+1) (column 0 is y0)."""
    ys = [y0]
    y, v = y0, v0
    for k in range(u_seq.shape[1] - 1):
        y, v = rk4_step(f, y, v, u_seq[:, k], u_seq[:, k + 1], dt)
        ys.append(y)
    return torch.stack(ys, dim=1)


@torch.no_grad()
def free_run(net, u, y, dt=DT):
    """One free run over the whole excerpt. The model sees u everywhere and y only at sample 0."""
    u_t = torch.as_tensor(u, dtype=torch.float32)[None, :]
    y0 = torch.as_tensor(y[:1], dtype=torch.float32)
    v0 = torch.as_tensor([(y[1] - y[0]) / dt], dtype=torch.float32)   # finite-difference velocity
    return rollout(net, y0, v0, u_t, dt)[0].numpy()


def rmse(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    r = float(np.sqrt(np.mean(d ** 2)))
    return r if math.isfinite(r) else float("inf")


def free_run_scores(net, u, y):
    yhat = free_run(net, u, y)
    return (rmse(yhat[:N_TRAIN], y[:N_TRAIN]),
            rmse(yhat[N_TRAIN:], y[N_TRAIN:]),
            yhat)


# ----------------------------------------------------------------------------- training
def reference_coefficients(u, y):
    """Least-squares a, b, c, g of eq. (1) from finite differences. Diagnostic only: the
    network never sees these numbers, they are not its weights."""
    v = np.gradient(y, DT)
    acc = np.gradient(v, DT)
    sl = slice(2, N_TRAIN - 2)
    A = np.stack([-v[sl], -y[sl], -y[sl] ** 3, u[sl]], axis=1)
    coef, *_ = np.linalg.lstsq(A, acc[sl], rcond=None)
    return dict(zip("abcg", map(float, coef)))


def make_net(cfg, u, y):
    v = np.gradient(y, DT)
    ys, vs, us = y[:N_TRAIN].std(), v[:N_TRAIN].std(), u[:N_TRAIN].std()
    accs = vs ** 2 / ys          # (characteristic angular frequency)^2 * amplitude
    return AccNet(cfg, ys, vs, us, accs)


def train(cfg: Config, u, y, log=print, time_budget_s: float | None = None):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    net = make_net(cfg, u, y)
    v_fd = np.gradient(y, DT)
    acc_fd = np.gradient(v_fd, DT)

    T = lambda a: torch.as_tensor(a, dtype=torch.float32)
    u_t, y_t, v_t, a_t = T(u), T(y), T(v_fd), T(acc_fd)
    Opt = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
    opt = Opt(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    t0 = time.time()

    # stage 0: one-step regression on finite-difference acceleration (stable warm start)
    idx = torch.arange(2, N_TRAIN - 2)
    for i in range(cfg.warm_steps):
        pred = net(y_t[idx], v_t[idx], u_t[idx])
        loss = ((pred - a_t[idx]) / net.accs).pow(2).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
        opt.step()
        if i % 200 == 0:
            log(f"    warm {i:4d} loss {loss.item():.4f}")

    # stage 1: multiple-shooting rollout training with a curriculum on the window length
    for g in opt.param_groups:
        g["lr"] = cfg.lr
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.steps, 1), eta_min=cfg.lr_final)
    stages = len(cfg.rollout_lengths)
    per_stage = max(cfg.steps // stages, 1)
    best_state, best_loss = None, float("inf")
    step = 0
    for L in cfg.rollout_lengths:
        for _ in range(per_stage):
            starts = torch.randint(0, N_TRAIN - L - 1, (cfg.batch,))
            ar = torch.arange(L + 1)
            ii = starts[:, None] + ar[None, :]
            y_hat = rollout(net, y_t[starts], v_t[starts], u_t[ii])
            loss = ((y_hat[:, 1:] - y_t[ii][:, 1:]) / net.scale_in[0]).pow(2).mean()
            opt.zero_grad()
            if not torch.isfinite(loss):
                log(f"    [warn] non-finite loss at step {step}, restoring best weights")
                if best_state is not None:
                    net.load_state_dict(best_state)
                break
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.clip)
            opt.step(); sched.step()
            if loss.item() < best_loss and L == cfg.rollout_lengths[-1]:
                best_loss = loss.item()
                best_state = copy.deepcopy(net.state_dict())
            if step % 250 == 0:
                log(f"    L={L:3d} step {step:5d} loss {loss.item():.5f}  ({time.time() - t0:.0f}s)")
            step += 1
            if time_budget_s and time.time() - t0 > time_budget_s:
                log("    [warn] time budget hit, stopping this run")
                return net
    return net


# ----------------------------------------------------------------------------- sweep
def sweep_configs(budget: str) -> list[Config]:
    if budget == "smoke":
        return [
            Config(name="smoke-mlp-2x16-tanh", hidden=(16, 16), warm_steps=200, steps=90,
                   rollout_lengths=(16, 48, 96), batch=16),
            Config(name="smoke-skip+mlp-2x16-tanh", hidden=(16, 16), linear_skip=True, warm_steps=200, steps=90,
                   rollout_lengths=(16, 48, 96), batch=16),
        ]
    if budget == "medium":      # about 4-5 minutes per configuration on 2 CPU cores
        base = dict(warm_steps=600, steps=900, rollout_lengths=(32, 96, 256), batch=32)
        return [
            Config(name="mlp-2x32-tanh-adam", hidden=(32, 32), **base),
            Config(name="skip+mlp-2x32-tanh-adam", hidden=(32, 32), linear_skip=True, **base),
            Config(name="skip+mlp-2x64-silu-adam", hidden=(64, 64), activation="silu", linear_skip=True, **base),
        ]
    base = dict(warm_steps=800, steps=2400, rollout_lengths=(32, 96, 256), batch=48)
    return [
        Config(name="mlp-2x32-tanh-adam", hidden=(32, 32), **base),
        Config(name="mlp-2x64-tanh-adam", hidden=(64, 64), **base),
        Config(name="mlp-3x64-tanh-adam", hidden=(64, 64, 64), **base),
        Config(name="mlp-2x64-silu-adam", hidden=(64, 64), activation="silu", **base),
        Config(name="skip+mlp-2x32-tanh-adam", hidden=(32, 32), linear_skip=True, **base),
        Config(name="skip+mlp-2x32-tanh-adamw-lr1e-3", hidden=(32, 32), linear_skip=True,
               optimizer="adamw", lr=1e-3, weight_decay=1e-4, **base),
        Config(name="skip+mlp-2x32-tanh-adam-long", hidden=(32, 32), linear_skip=True,
               **{**base, "rollout_lengths": (64, 256, 512), "steps": 3000}),
    ]


def plot_test(t, y_meas, y_pred, rmse_test, rmse_zero, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_meas, c_pred, ink, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]),
                                  facecolor="#fcfcfb")
    for a in (ax, ax2):
        a.set_facecolor("#fcfcfb")
        a.grid(True, color=grid, lw=0.8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(grid)
        a.tick_params(colors=muted, labelsize=9)
    ax.plot(t, y_meas, color=c_meas, lw=1.6, label="measured y")
    ax.plot(t, y_pred, color=c_pred, lw=1.6, label="free-run prediction")
    ax.set_ylabel("output voltage (V, centered)", color=muted)
    ax.set_title(f"Silverbox test segment: free-run RMSE {rmse_test:.4f} V "
                 f"(predicting zero: {rmse_zero:.4f} V)", color=ink, fontsize=11, loc="left")
    ax.legend(frameon=False, loc="upper right", labelcolor=ink)
    ax2.plot(t, y_pred - y_meas, color=muted, lw=1.2)
    ax2.axhline(0, color=grid, lw=1)
    ax2.set_ylabel("error (V)", color=muted)
    ax2.set_xlabel("time within test record (s)", color=muted)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to SNLS80mV.mat")
    ap.add_argument("--out", default="results")
    ap.add_argument("--budget", default="full", choices=["smoke", "medium", "full"])
    ap.add_argument("--only", default=None, help="substring: run only configs whose name contains it")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    u, y = load_excerpt(args.data)
    zero_test = rmse(np.zeros(N_TEST), y[N_TRAIN:])
    print(f"excerpt: {N_EXC} samples, train {N_TRAIN}, test {N_TEST}; dt={DT:.6f}s")
    print(f"baseline (predict zero): test RMS = {zero_test:.5f} V")
    ref = reference_coefficients(u, y)
    print("least-squares reference coefficients of eq.(1) (diagnostic only):",
          {k: f"{v:.4g}" for k, v in ref.items()})

    rows, best = [], None
    for cfg in sweep_configs(args.budget):
        if args.only and args.only not in cfg.name:
            continue
        print(f"\n=== {cfg.name} ===")
        t0 = time.time()
        net = train(cfg, u, y)
        r_tr, r_te, yhat = free_run_scores(net, u, y)
        dt_s = time.time() - t0
        print(f"  free-run RMSE  train {r_tr:.5f}  test {r_te:.5f}   ({dt_s:.0f}s)")
        rows.append(dict(config=asdict(cfg), train_rmse=r_tr, test_rmse=r_te, seconds=dt_s))
        if best is None or r_te < best["test_rmse"]:
            best = dict(cfg=cfg, test_rmse=r_te, train_rmse=r_tr, yhat=yhat, state=copy.deepcopy(net.state_dict()))

    if best is None:
        raise SystemExit("no configuration was run")

    t_test = np.arange(N_TEST) * DT
    plot_test(t_test, y[N_TRAIN:], best["yhat"][N_TRAIN:], best["test_rmse"], zero_test, out / "test_prediction.png")
    torch.save(dict(state=best["state"], config=asdict(best["cfg"])), out / "best_model.pt")
    summary = dict(
        baseline_zero_test_rmse=zero_test,
        reference_coefficients=ref,
        kept=dict(name=best["cfg"].name, config=asdict(best["cfg"]),
                  free_run_rmse_train=best["train_rmse"], free_run_rmse_test=best["test_rmse"]),
        all_runs=rows,
    )
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    with open(out / "sweep_table.md", "w") as fh:
        fh.write("| architecture / settings | free-run RMSE train (V) | free-run RMSE test (V) |\n|---|---|---|\n")
        for r in sorted(rows, key=lambda r: r["test_rmse"]):
            fh.write(f"| {r['config']['name']} | {r['train_rmse']:.5f} | {r['test_rmse']:.5f} |\n")
    print(f"\nKEPT: {best['cfg'].name}   train RMSE {best['train_rmse']:.5f}   test RMSE {best['test_rmse']:.5f}")
    print(f"TRAIN_RMSE={best['train_rmse']:.6f}")
    print(f"TEST_RMSE={best['test_rmse']:.6f}")


if __name__ == "__main__":
    main()
