"""
closure_balance_check.py -- affirmative closure receipt for the stationary
balance E[I] = -C1 = -nu*E[<w_xx, DP1>] on the SDE (cross-model referee item,
log #23), and the SNR diagnosis of the direct-C1 estimator at large nu.

Findings (committed log closure_balance_check.log):
  * nu=10, T=2000 tau: E[I] = 4.43e-5 +/- 1.1e-6 (44 sigma), while
    nu*E[Y] has block-SE 1.6e-4 >> |target| 4.4e-5: the direct-C1 estimator is
    statistically unresolvable there (mean/rms(Y) ~ eps ~ 1/430) -- this is the
    mechanism behind the earlier "sign-mixed C1 at nu>=5" (theta_extend.py),
    which was therefore guaranteed noise, not evidence of theta -> 0.
  * nu=2, T=20000 tau (n=200k): E[I] = 1.1124e-3 +/- 7.9e-6 vs
    -nu*E[Y] = 1.1730e-3 +/- 1.1e-4, ratio 0.948, z(diff) = -0.54 -- the
    balance CLOSES within noise (PASS).
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "instrument"))
from theta_large_nu_exact_ou import step_exact           # noqa: E402
from measure_cascade import NOISE, B0                    # noqa: E402
import sgclm as sg                                       # noqa: E402
from measure_theta import make, H, dx, vel, ip           # noqa: E402


def I_W1_Y(w, k, N, a=-2.0):
    v = H(w, k); wx = dx(w, k); wxx = dx(w, k, 2); u = vel(w, k)
    DP1 = -H(wx * wx, k) - 2 * H(wx, k) * wx - 2 * v * wxx
    bnl = -a * u * wx + v * w
    return ip(bnl, DP1, N), 6 * ip(v * v, wx * wx, N), ip(wxx, DP1, N)


def closure(nu, T_units, seed, N=64, dtfac=1.0):
    dt = min(2e-3, 0.05 / (16 * nu)) / dtfac
    tau = 1 / nu
    Tb, Ts = 50 * tau, T_units * tau
    x, k = sg.make_grid(N)
    _, kf = make(N)
    rng = np.random.default_rng(seed)
    what = sg.random_band_limited_ic(N, k, rng, 1, 8,
                                     amp=np.sqrt(B0 / (4 * np.pi * nu)))
    rec = max(1, int(round((tau / 10) / dt)))
    Is, Ws, Ys = [], [], []
    for s in range(int(round((Tb + Ts) / dt))):
        what = step_exact(what, k, N, dt, rng, nu, -2.0, NOISE)
        if s % rec == 0 and (s + 1) * dt > Tb:
            I, W1, Y = I_W1_Y(np.fft.irfft(what, N), kf, N)
            Is.append(I); Ws.append(W1); Ys.append(Y)
    Is, Ws, Ys = map(np.array, (Is, Ws, Ys))
    L = 50; nb = len(Ys) // L

    def bse(x):
        return x[:nb * L].reshape(nb, L).mean(axis=1).std(ddof=1) / np.sqrt(nb)
    seI, seY = bse(Is), bse(Ys)
    diff_z = (Is.mean() + nu * Ys.mean()) / np.sqrt(seI ** 2 + (nu * seY) ** 2)
    print(f"CLOSURE nu={nu} T={T_units}tau n={len(Is)} dt={dt:.2e}: "
          f"E[I]={Is.mean():+.5e}+-{seI:.1e}   "
          f"-nu*E[Y]={-nu*Ys.mean():+.5e}+-{nu*seY:.1e}   "
          f"ratio={Is.mean()/(-nu*Ys.mean()):+.4f}  z(diff)={diff_z:+.2f}",
          flush=True)


if __name__ == "__main__":
    closure(nu=10.0, T_units=2000, seed=42)   # SNR diagnosis point
    closure(nu=2.0, T_units=20000, seed=7)    # affirmative closure (PASS)
