"""Write a SYNTHETIC stand-in for SNLS80mV.mat (same variables, length and excerpt position).

It is a Duffing oscillator  y'' + a y' + b y + c y^3 = g u  driven by a random-phase multisine that
starts at sample 40650, plus small output noise. It exists only to test the pipeline offline.
Numbers obtained on it say nothing about the real Silverbox record.

    python make_synthetic.py synthetic_SNLS80mV.mat
"""
import sys

import numpy as np
import scipy.io

FS = 610.35
DT = 1.0 / FS
N = 131072
MULTISINE_START = 40650
PERIOD = 8192


def simulate(u, a=40.0, b=(2 * np.pi * 70.0) ** 2 * 0.9, c=7.0e5, g=1.0):
    n = len(u)
    y = np.zeros(n)
    v = np.zeros(n)

    def f(y_, v_, u_):
        return v_, g * u_ - a * v_ - b * y_ - c * y_ ** 3

    yy, vv = 0.0, 0.0
    for k in range(n - 1):
        um = 0.5 * (u[k] + u[k + 1])
        k1 = f(yy, vv, u[k])
        k2 = f(yy + 0.5 * DT * k1[0], vv + 0.5 * DT * k1[1], um)
        k3 = f(yy + 0.5 * DT * k2[0], vv + 0.5 * DT * k2[1], um)
        k4 = f(yy + DT * k3[0], vv + DT * k3[1], u[k + 1])
        yy += DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        vv += DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        y[k + 1], v[k + 1] = yy, vv
    return y


def main(path):
    rng = np.random.default_rng(1)
    kmax = int(200.0 / (FS / PERIOD))                      # excite up to ~200 Hz
    ks = np.arange(1, kmax + 1)
    phases = rng.uniform(0, 2 * np.pi, ks.size)
    t = np.arange(PERIOD) / FS
    one_period = np.sum(np.cos(2 * np.pi * np.outer(t, ks * FS / PERIOD) + phases), axis=1)
    one_period *= 0.05 / one_period.std()                  # ~50 mV rms like the real input
    u = np.zeros(N)
    m = N - MULTISINE_START
    u[MULTISINE_START:] = np.tile(one_period, m // PERIOD + 1)[:m]

    y_unit = simulate(u, g=1.0)
    ex = slice(42650, 42650 + 4096)
    g = 0.06 / y_unit[ex].std()                            # scale gain so the output rms is ~0.06 V
    y = simulate(u, g=g)
    y_noisy = y + rng.normal(0, 1e-3, N)
    scipy.io.savemat(path, {"V1": u[None, :], "V2": y_noisy[None, :]})
    print(f"wrote {path}: V1,V2 of {N} samples; excerpt y rms = {y[ex].std():.4f} V, max|y| = {np.abs(y[ex]).max():.3f} V")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic_SNLS80mV.mat")
