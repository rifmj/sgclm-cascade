"""
p2_convergence_matrix.py -- the convergence matrix for the stationary campaign
(referee rounds 2-3: R2-numerics-error-budget / R3-coarse-sweep-claims).

Extends p2_stationary_campaign.run_point with an integrated-autocorrelation-time
estimate (Sokal windowing, c=6) for I, W1 and the dissipation, and runs the
bracketing grid:

  * dt/2 at every campaign nu (0.01 already has one in p2_results.jsonl);
  * dt/4 at nu=0.01 (3-point Richardson check of the first-order-in-time bias);
  * spatial resolution up (N=384) at the two smallest nu;
  * an independent, deliberately different initial condition at nu=0.01
    (high-amplitude, wide band) -- initial-state sensitivity;

Each job appends a JSON line to swing/p2_convergence.jsonl.  Jobs are selected
by --job so the driver can parallelize across processes:

  python swing/p2_convergence_matrix.py --job dt2_0.2
  python swing/p2_convergence_matrix.py --list
"""
import os, sys, json, time, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "instrument"))
import sgclm as sg                                    # noqa: E402
from measure_theta import make, I_and_W1              # noqa: E402
from measure_cascade import NOISE, B0                 # noqa: E402
from theta_large_nu_exact_ou import step_exact        # noqa: E402
from p2_stationary_campaign import (                  # noqa: E402
    KS_RHO, K0, rho_K, flux_decomposition, block_stats, THETA_G)


def tau_int(x, c=6.0):
    """Integrated autocorrelation time (in samples) via Sokal's adaptive window."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    xc = x - x.mean()
    var = np.mean(xc ** 2)
    if var == 0 or n < 16:
        return 1.0
    # FFT autocovariance
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xc, m)
    acov = np.fft.irfft(f * np.conjugate(f), m)[:n] / n
    rho = acov / acov[0]
    tau = 1.0
    for W in range(1, n // 2):
        tau = 1.0 + 2.0 * np.sum(rho[1:W + 1])
        if W >= c * tau:
            break
    return float(max(tau, 1.0))


def run_point(nu, N, dt, seed, burn_visc=5.0, stat_visc=20.0,
              rec_dt=0.5, block_tu=50.0, min_burn=100.0, min_stat=2000.0,
              ic=("band", 1, 8, 0.5), tag=""):
    """p2_stationary_campaign.run_point + tau_int fields + configurable IC."""
    T_burn = max(min_burn, burn_visc / nu)
    T_stat = max(min_stat, stat_visc / nu)
    T = T_burn + T_stat
    x, k = sg.make_grid(N)
    _, kf = make(N)
    rng = np.random.default_rng(seed)
    _, lo, hi, amp = ic
    what = sg.random_band_limited_ic(N, k, rng, k_lo=lo, k_hi=hi, amp=amp)
    nsteps = int(round(T / dt))
    rec = max(1, int(round(rec_dt / dt)))
    rows = {q: [] for q in
            ("I", "W1", "E1", "D", "PiLL", "PiLH", "PiHH",
             *(f"rho{K}" for K in KS_RHO))}
    spec = np.zeros(k.size)
    nspec = 0
    t0 = time.time()
    for s in range(nsteps):
        what = step_exact(what, k, N, dt, rng, nu, -2.0, NOISE, dealias=True)
        if s % rec == 0 and (s + 1) * dt > T_burn:
            w = np.fft.irfft(what, N)
            I, W1 = I_and_W1(w, kf, N, a=-2.0)
            rows["I"].append(I); rows["W1"].append(W1)
            rows["E1"].append(sg.palinstrophy(what, k, N))
            rows["D"].append(sg.dissipation(what, k, N, nu))
            LL, LH, HH = flux_decomposition(what, k, N, K0)
            rows["PiLL"].append(LL); rows["PiLH"].append(LH); rows["PiHH"].append(HH)
            for K in KS_RHO:
                rows[f"rho{K}"].append(rho_K(what, k, N, nu, K))
            spec += np.abs(what / N) ** 2
            nspec += 1
    wall = time.time() - t0
    L = max(4, int(round(block_tu / rec_dt)))
    out = dict(tag=tag, nu=nu, N=N, dt=dt, seed=seed, T_burn=T_burn,
               T_stat=T_stat, n=nspec, wall=round(wall, 1), block_tu=block_tu,
               ic=list(ic))
    for q, v in rows.items():
        m, se, z = block_stats(np.array(v), L)
        out[q] = m; out[q + "_se"] = se; out[q + "_z"] = z
    I_arr, W_arr = np.array(rows["I"]), np.array(rows["W1"])
    out["theta"] = I_arr.mean() / W_arr.mean()
    nb = len(I_arr) // L
    bI = I_arr[:nb * L].reshape(nb, L).mean(axis=1)
    bW = W_arr[:nb * L].reshape(nb, L).mean(axis=1)
    jk = np.array([(bI.sum() - bI[j]) / (bW.sum() - bW[j]) for j in range(nb)])
    out["theta_se"] = float(np.sqrt((nb - 1) * np.mean((jk - jk.mean()) ** 2)))
    out["EY2"] = float((I_arr ** 2).mean())
    out["anti_frac"] = float((I_arr <= 0).mean())
    out["L0_ratio"] = out["D"] / (0.5 * B0)
    # integrated autocorrelation times, in time units (samples are rec_dt apart)
    ratio_series = I_arr / np.maximum(W_arr, 1e-300)
    for name, series in (("I", I_arr), ("W1", W_arr),
                         ("D", np.array(rows["D"])), ("ratio", ratio_series)):
        out[f"tau_{name}"] = round(tau_int(series) * rec_dt, 2)
    spec /= max(nspec, 1)
    cut = int(np.floor(2 / 3 * (N // 2)))
    out["tail_over_peak"] = float(spec[cut - 2:cut + 1].max() / spec.max())
    out["spec_forced_frac"] = float(spec[1:5].sum() / spec[1:].sum())
    return out


BASE = {0.2: (128, 1e-3), 0.1: (128, 1e-3), 0.05: (128, 1e-3),
        0.02: (192, 7.5e-4), 0.01: (256, 5e-4), 0.005: (256, 5e-4)}

JOBS = {}
for _nu, (_N, _dt) in BASE.items():
    JOBS[f"dt2_{_nu}"] = dict(nu=_nu, N=_N, dt=_dt / 2, seed=0, tag="dt/2")
JOBS["dt4_0.01"] = dict(nu=0.01, N=256, dt=1.25e-4, seed=0, tag="dt/4")
JOBS["Nup_0.01"] = dict(nu=0.01, N=384, dt=5e-4, seed=0, tag="N-up")
JOBS["Nup_0.005"] = dict(nu=0.005, N=384, dt=5e-4, seed=0, tag="N-up")
JOBS["ic_0.01"] = dict(nu=0.01, N=256, dt=5e-4, seed=7, tag="far-IC",
                       ic=("band", 3, 20, 3.0))
# tau reference points at base dt (tau fields for the existing campaign rows)
JOBS["tau_0.2"] = dict(nu=0.2, N=128, dt=1e-3, seed=2, tag="tau-ref")
JOBS["tau_0.05"] = dict(nu=0.05, N=128, dt=1e-3, seed=2, tag="tau-ref")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", default=os.path.join(_HERE, "p2_convergence.jsonl"))
    args = ap.parse_args()
    if args.list or not args.job:
        print("\n".join(JOBS))
        sys.exit(0)
    o = run_point(**JOBS[args.job])
    print(f"[{args.job}] nu={o['nu']} N={o['N']} dt={o['dt']:.2e} "
          f"theta={o['theta']:.4f}+-{o['theta_se']:.4f} "
          f"tau_I={o['tau_I']} tau_ratio={o['tau_ratio']} "
          f"zmax={max(abs(o['I_z']), abs(o['W1_z'])):.2f} [{o['wall']:.0f}s]",
          flush=True)
    with open(args.json, "a") as f:
        f.write(json.dumps(o) + "\n")
    print("DONE", flush=True)
