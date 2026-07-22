"""
p2_campaign_v18.py -- the v1.8 stationary campaign (referee round 5).

Two corrections over v17:
  (a) STRICT 2/3 mask (sgclm.dealias_mask, M = (N-1)//3): the boundary mode kept at
      3 | N by the old floor(N/3) cutoff aliased its self-product onto -M;
  (b) the diagnostic functionals I, I_M, R_M, W1, Yd are computed on a ZERO-PADDED
      grid L = 2N (L/2 > 2M and L > 4M), so the quadratic fields N[w], DP1 and the
      quartic quadratures are alias-free EXACT evaluations of the paper's functionals
      -- previously they were N-point pseudo-spectral (aliased) analogues.

Per sample we also record the UNPADDED values, so each campaign point carries the
padded-vs-unpadded receipt the round-5 report asks for (delta_alias columns).

Jobs:  python swing/p2_campaign_v18.py --job nu0.05_s0     (see --list)
Appends JSON lines to swing/p2_results_v18.jsonl.
"""
import os, sys, json, time, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "instrument"))
import sgclm as sg                                    # noqa: E402
from measure_theta import make                        # noqa: E402
from measure_cascade import NOISE, B0                 # noqa: E402
from theta_large_nu_exact_ou import step_exact        # noqa: E402
from p2_stationary_campaign import (                  # noqa: E402
    KS_RHO, K0, rho_K, flux_decomposition, block_stats, THETA_G)


def _ip_hat(ah, bh, L):
    """int a*b dx from rfft coefficients on an L-grid (real fields, Parseval)."""
    t = (ah * np.conj(bh)).real / L ** 2
    s = 2.0 * t[1:].sum() + t[0]
    if L % 2 == 0:
        s -= t[-1]
    return 2.0 * np.pi * s


def diagnostics(what, k, N, M, pad=2):
    """I, I_M, R_M, W1, Yd on an L = pad*N grid (pad=2: exact; pad=1: aliased analogue)."""
    L = pad * N
    kL = np.arange(L // 2 + 1)
    whL = np.zeros(L // 2 + 1, dtype=complex)
    whL[:k.size] = what * pad                # irfft scaling: preserves the function values
    def ifft(z):
        return np.fft.irfft(z, L)
    def hilb(z):
        o = z.copy(); o[1:] *= -1j; o[0] = 0; return o
    w = ifft(whL); wx = ifft(whL * 1j * kL); wxx = ifft(-whL * kL ** 2)
    v = ifft(hilb(whL))
    uh = whL.copy(); uh[1:] = -uh[1:] / kL[1:]; uh[0] = 0
    u = ifft(uh)
    def H_(z):
        zh = np.fft.rfft(z)[:kL.size]
        return ifft(zh * (-1j) * np.sign(kL))
    DP1 = -H_(wx * wx) - 2.0 * H_(wx) * wx - 2.0 * v * wxx
    bnl = 2.0 * u * wx + v * w               # a = -2
    Nh = np.fft.rfft(bnl)[:kL.size]
    Dh = np.fft.rfft(DP1)[:kL.size]
    qm = (kL > M).astype(float)
    I = _ip_hat(Nh, Dh, L)
    RM = _ip_hat(Nh * qm, Dh * qm, L)
    W1 = 6.0 * _ip_hat(np.fft.rfft(v * v)[:kL.size], np.fft.rfft(wx * wx)[:kL.size], L)
    Yd = _ip_hat(np.fft.rfft(wxx)[:kL.size], Dh, L)
    return I, I - RM, RM, W1, Yd


def run_point(nu, N, dt, seed, burn_visc=5.0, stat_visc=20.0,
              rec_dt=0.5, block_tu=50.0, min_burn=100.0, min_stat=2000.0, tag=""):
    T_burn = max(min_burn, burn_visc / nu)
    T_stat = max(min_stat, stat_visc / nu)
    T = T_burn + T_stat
    x, k = sg.make_grid(N)
    M = (N - 1) // 3                          # strict 2/3 rule (v1.8)
    rng = np.random.default_rng(seed)
    what = sg.random_band_limited_ic(N, k, rng, k_lo=1, k_hi=8, amp=0.5)
    nsteps = int(round(T / dt))
    rec = max(1, int(round(rec_dt / dt)))
    rows = {q: [] for q in ("I", "IM", "RM", "W1", "Yd",
                            "Iun", "RMun", "W1un",
                            "E1", "D", "PiLL", "PiLH", "PiHH",
                            *(f"rho{K}" for K in KS_RHO))}
    spec = np.zeros(k.size); nspec = 0
    t0 = time.time()
    for s in range(nsteps):
        what = step_exact(what, k, N, dt, rng, nu, -2.0, NOISE, dealias=True)
        if s % rec == 0 and (s + 1) * dt > T_burn:
            I, IM, RM, W1, Yd = diagnostics(what, k, N, M, pad=2)     # exact
            Iu, _, RMu, W1u, _ = diagnostics(what, k, N, M, pad=1)    # aliased receipt
            rows["I"].append(I); rows["IM"].append(IM); rows["RM"].append(RM)
            rows["W1"].append(W1); rows["Yd"].append(Yd)
            rows["Iun"].append(Iu); rows["RMun"].append(RMu); rows["W1un"].append(W1u)
            rows["E1"].append(sg.palinstrophy(what, k, N))
            rows["D"].append(sg.dissipation(what, k, N, nu))
            LL, LH, HH = flux_decomposition(what, k, N, K0)
            rows["PiLL"].append(LL); rows["PiLH"].append(LH); rows["PiHH"].append(HH)
            for K in KS_RHO:
                rows[f"rho{K}"].append(rho_K(what, k, N, nu, K))
            spec += np.abs(what / N) ** 2; nspec += 1
    wall = time.time() - t0
    L = max(4, int(round(block_tu / rec_dt)))
    # v1.9: ONE sample window for every estimator -- trim all series to the block
    # grid (nb*L samples). Previously the block-stats fields averaged the trimmed
    # window while theta/C1/delta_bal/receipts used the full series; with
    # dt = 7.5e-4 (the one non-divisor of rec_dt = 0.5) n = 3998, and the
    # 98-sample tail of the heavy-tailed Yd shifted the full-mean C1 by ~1\% at
    # nu = 0.02 (referee round 6), breaking C1 = nu*mean(Yd) across fields.
    _nt = (min(len(v) for v in rows.values()) // L) * L
    rows = {q: v[:_nt] for q, v in rows.items()}
    out = dict(tag=tag, nu=nu, N=N, M=M, dt=dt, seed=seed, T_burn=T_burn,
               T_stat=T_stat, n=_nt, n_raw=nspec, wall=round(wall, 1),
               block_tu=block_tu)
    for q, v in rows.items():
        m, se, z = block_stats(np.array(v), L)
        out[q] = m; out[q + "_se"] = se; out[q + "_z"] = z
    I_arr, IM_arr, W_arr = (np.array(rows[q]) for q in ("I", "IM", "W1"))
    out["theta"] = I_arr.mean() / W_arr.mean()
    out["theta_M"] = IM_arr.mean() / W_arr.mean()
    nb = len(I_arr) // L
    bI = I_arr[:nb * L].reshape(nb, L).mean(axis=1)
    bW = W_arr[:nb * L].reshape(nb, L).mean(axis=1)
    jk = np.array([(bI.sum() - bI[j]) / (bW.sum() - bW[j]) for j in range(nb)])
    out["theta_se"] = float(np.sqrt((nb - 1) * np.mean((jk - jk.mean()) ** 2)))
    C1 = nu * np.mean(rows["Yd"])
    out["C1"] = C1
    out["C1_se"] = nu * out["Yd_se"]
    out["delta_M"] = np.mean(rows["RM"]) / np.mean(rows["W1"])        # signed
    out["delta_bal"] = abs(np.mean(rows["IM"]) + C1) / np.mean(rows["W1"])
    # padded-vs-unpadded (aliasing) receipts
    out["theta_unpadded"] = float(np.mean(rows["Iun"]) / np.mean(rows["W1un"]))
    out["alias_theta"] = out["theta_unpadded"] - out["theta"]
    RMm = np.mean(rows["RM"])
    out["alias_RM_rel"] = float((np.mean(rows["RMun"]) - RMm) / RMm) if RMm != 0 else 0.0
    out["EY2"] = float((I_arr ** 2).mean())
    out["anti_frac"] = float((I_arr <= 0).mean())
    out["L0_ratio"] = out["D"] / (0.5 * B0)
    spec /= max(nspec, 1)
    out["state_above_M"] = float(spec[M + 1:].sum() / max(spec[1:].sum(), 1e-300))
    return out


GRID = [(0.2, 128, 1e-3), (0.1, 128, 1e-3), (0.05, 128, 1e-3),
        (0.02, 192, 7.5e-4), (0.01, 256, 5e-4), (0.005, 256, 5e-4)]

JOBS = {}
for _nu, _N, _dt in GRID:
    for _s in (0, 1):
        JOBS[f"nu{_nu}_s{_s}"] = dict(nu=_nu, N=_N, dt=_dt, seed=_s, tag="v18")
JOBS["dt2_0.01"] = dict(nu=0.01, N=256, dt=2.5e-4, seed=0, tag="dt/2")
JOBS["Nup384_0.005"] = dict(nu=0.005, N=384, dt=5e-4, seed=0, tag="N-up")
JOBS["Nup384_0.01"] = dict(nu=0.01, N=384, dt=5e-4, seed=0, tag="N-up")
JOBS["largenu_2"] = dict(nu=2.0, N=128, dt=1e-3, seed=0, tag="large-nu")
JOBS["largenu_10"] = dict(nu=10.0, N=128, dt=1e-3, seed=0, tag="large-nu")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", default=os.path.join(_HERE, "p2_results_v18.jsonl"))
    args = ap.parse_args()
    if args.list or not args.job:
        print("\n".join(JOBS)); sys.exit(0)
    o = run_point(**JOBS[args.job])
    print(f"[{args.job}] theta={o['theta']:.4f}+-{o['theta_se']:.4f} "
          f"theta_M={o['theta_M']:.4f} dM={o['delta_M']:+.2e} "
          f"alias(theta)={o['alias_theta']:+.1e} alias(RM)={o['alias_RM_rel']:+.1e} "
          f"bal={o['delta_bal']:.1e} zmax={max(abs(o['I_z']), abs(o['W1_z'])):.2f} "
          f"[{o['wall']:.0f}s]", flush=True)
    with open(args.json, "a") as f:
        f.write(json.dumps(o) + "\n")
    print("DONE", flush=True)
