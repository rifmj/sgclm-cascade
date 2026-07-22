"""
theta_large_nu_exact_ou.py -- decisive large-nu test: what is lim_{nu->inf} theta(nu)?

RESOLVES (2026-07-14, log #23) the S6 question in favor of a POSITIVE limit:

  theta(nu) -> theta_G(shape) = E_G[I]/E_G[W1]  (band [1,4]: 0.8119 +/- 1e-4),

refuting the earlier S6 claim theta(nu) = O(B_0/nu) -> 0 ("null cascade").
theta is a ratio of two quartic functionals, hence 0-homogeneous in the field
amplitude, and the rescaled linear (OU) stationary measure has a nu-INDEPENDENT
shape (per-mode variance b_k^2/(4 pi nu k^2)); the S6 bookkeeping compared a
first-order numerator against a zeroth-order denominator, forgetting that the nu
prefactor in C1 = nu E[Y] restores the order balance (nu s^3 eps = s^4).
Adjudicated by: this session's scaling audit + an independent external referee
derivation + a cross-model (Sol) referee + the measurements below.

Why the earlier nu=10,20 runs (theta_extend.py) got "sign-mixed C1":
  (1) DOMINANT: they estimated theta via C1 = nu*E[Y] directly; Y's mean is the
      O(eps) omega->-omega asymmetry (Gaussian order ZERO by parity) while its
      rms is O(1)*s^3, so the estimator SNR ~ eps ~ nu^{-3/2} (at nu=10:
      signal 4.4e-5 vs block-SE 1.6e-4 at 20k samples) -- guaranteed noise.
      The E[I] route (Gaussian order nonzero) is a clean ~44 sigma signal.
  (2) SECONDARY: the repo stepper adds the EM noise increment UNWEIGHTED after
      the exact viscous decay, so the linear stationary variance is biased by
      the K-DEPENDENT factor 2*g*dt/(1-exp(-2*g*dt)) (x6.4 at nu=20, k=4,
      dt=1e-2 vs x1.2 at k=1) -- it distorts the SHAPE of the measure, the only
      thing the 0-homogeneous theta sees.  Here the noise uses the EXACT
      per-step OU variance (1-exp(-2*g*dt))/(2*g): linear dynamics exact at
      any dt.

Receipts (committed alongside, theta_large_nu_exact_ou.log):
  theta_G(band[1,4]) = +0.8119 +/- 1e-4 (direct Gaussian sampling, M=200k);
  anchor nu=0.05 -> theta = 0.790 (reproduces the banked 0.78--0.81);
  grid nu in {0.5,1,2,5,10,20} x 2 seeds: ALL theta in [0.8107, 0.8126]
  (block-bootstrap CI95, split-half z clean, dt/5-refinement stable);
  repo-EM comparison at nu=10, dt=1e-3: ratio still 0.8118 (bias partially
  cancels in the ratio at that dt; the banked failures used dt=8e-3..1e-2, N=32
  AND the noisy C1 route).

Functionals: certified I_and_W1 imported verbatim from instrument/measure_theta.py.
Usage: python swing/theta_large_nu_exact_ou.py [--coarse|--grid|--refine]
"""
import os, sys, time, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "instrument"))
import sgclm as sg                                   # noqa: E402
from measure_theta import I_and_W1, make             # noqa: E402  (certified formula)
from measure_cascade import NOISE, B0                # noqa: E402  (band [1,4], B0=0.16)


# --------------------------------------------------------------------------
# exact-OU noise increment (replaces the sqrt(dt) EM weight by the exact
# per-step OU variance; reduces to the repo increment as gamma*dt -> 0)
# --------------------------------------------------------------------------
def noise_increment_exact(k, N, noise, dt, nu, rng):
    b2 = noise.b2(k)
    b = np.sqrt(b2)
    gamma = nu * k.astype(float) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        veff = np.where(gamma > 0.0,
                        (1.0 - np.exp(-2.0 * gamma * dt)) / (2.0 * gamma),
                        dt)
    active = b2 > 0
    if (N % 2 == 0) and (k[-1] == N // 2):
        active = active.copy(); active[-1] = False
    xi1 = rng.standard_normal(k.shape)
    xi2 = rng.standard_normal(k.shape)
    d_wtilde = np.zeros(k.shape, dtype=complex)
    d_wtilde[active] = (b[active] / (2.0 * np.sqrt(np.pi))) \
        * np.sqrt(veff[active]) * (xi1[active] + 1j * xi2[active])
    d_what = N * d_wtilde
    d_what[k == 0] = 0.0
    return d_what


def step_exact(what, k, N, dt, rng, nu, a, noise, dealias=True):
    """Integrating-factor drift step + exact-OU noise (linear part exact at any dt)."""
    Nhat = sg.nonlinear(what, k, N, a=a, dealias=dealias)
    E = np.exp(-nu * k ** 2 * dt)
    what_new = E * (what + dt * Nhat) + noise_increment_exact(k, N, noise, dt, nu, rng)
    what_new[k == 0] = 0.0
    if dealias:                       # v1.7: keep the state on E_M exactly (belt-and-braces;
        what_new = what_new * sg.dealias_mask(k, N)   # redundant once nonlinear() projects)
    return what_new


# --------------------------------------------------------------------------
# Gaussian reference: exact sampling of the linear (OU) stationary measure
#   E|wtilde(k)|^2 = b_k^2 / (4*pi*nu*k^2)   (band modes only; degenerate off-band)
# theta_G(band) := E_G[I]/E_G[W1] -- the measured large-nu limit value.
# --------------------------------------------------------------------------
def sample_gaussian_batch(k, N, noise, nu, rng, M):
    b2 = noise.b2(k)
    gamma = nu * k.astype(float) ** 2
    var = np.where(b2 > 0, b2 / (4.0 * np.pi * np.maximum(gamma, 1e-300)), 0.0)
    s = np.sqrt(var / 2.0)                    # per real/imag component of wtilde
    xi1 = rng.standard_normal((M, k.size))
    xi2 = rng.standard_normal((M, k.size))
    wtilde = s[None, :] * (xi1 + 1j * xi2)
    what = N * wtilde
    what[:, k == 0] = 0.0
    return what


def theta_gaussian_mc(nu=1.0, N=64, M=400_000, seed=123, batches=40):
    x, k = sg.make_grid(N)
    _, kf = make(N)
    rng = np.random.default_rng(seed)
    per = M // batches
    bI, bW = [], []
    for b in range(batches):
        what = sample_gaussian_batch(k, N, NOISE, nu, rng, per)
        Is = np.empty(per); Ws = np.empty(per)
        for i in range(per):
            w = np.fft.irfft(what[i], N)
            Is[i], Ws[i] = I_and_W1(w, kf, N, a=-2.0)
        bI.append(Is.mean()); bW.append(Ws.mean())
    bI = np.array(bI); bW = np.array(bW)
    th = bI.sum() / bW.sum()
    jk = np.array([(bI.sum() - bI[j]) / (bW.sum() - bW[j]) for j in range(batches)])
    se = np.sqrt((batches - 1) * np.mean((jk - jk.mean()) ** 2))
    return th, se, bI.mean(), bW.mean()


# --------------------------------------------------------------------------
# SDE measurement: theta as ratio of means + moving-block bootstrap CI
# --------------------------------------------------------------------------
def run_theta(nu, N, dt, T_relax_units, seed, a=-2.0,
              burn_relax_units=50.0, samples_per_tau=10, exact_noise=True):
    """T and burn-in are specified in units of tau = 1/nu (slowest forced mode)."""
    tau = 1.0 / nu
    T_burn = burn_relax_units * tau
    T_stat = T_relax_units * tau
    T = T_burn + T_stat
    x, k = sg.make_grid(N)
    _, kf = make(N)
    rng = np.random.default_rng(seed)
    what = sg.random_band_limited_ic(N, k, rng, k_lo=1, k_hi=8,
                                     amp=np.sqrt(B0 / (4 * np.pi * nu)))
    nsteps = int(round(T / dt))
    rec = max(1, int(round((tau / samples_per_tau) / dt)))
    Is, Ws, ts = [], [], []
    stepper = step_exact if exact_noise else \
        (lambda wh, k_, N_, dt_, rng_, nu_, a_, noise_, dealias=True:
         sg.step(wh, k_, N_, dt_, rng_, nu=nu_, a=a_, noise=noise_, dealias=dealias))
    t0 = time.time()
    for s in range(nsteps):
        what = stepper(what, k, N, dt, rng, nu, a, NOISE, dealias=True)
        if s % rec == 0 and (s + 1) * dt > T_burn:
            w = np.fft.irfft(what, N)
            I, W1 = I_and_W1(w, kf, N, a=a)
            Is.append(I); Ws.append(W1); ts.append((s + 1) * dt)
    return dict(nu=nu, N=N, dt=dt, seed=seed, n=len(Is),
                I=np.array(Is), W1=np.array(Ws), t=np.array(ts),
                wall=time.time() - t0,
                samples_per_tau=samples_per_tau)


def block_bootstrap_ratio(I, W1, samples_per_tau, blocks_tau=5.0, B=2000, seed=7):
    """theta = mean(I)/mean(W1); moving-block bootstrap with blocks of
    blocks_tau relaxation times (>> autocorr time ~1 tau)."""
    n = len(I)
    L = max(2, int(round(blocks_tau * samples_per_tau)))
    nb = n // L
    if nb < 8:
        return np.nan, (np.nan, np.nan), nb
    I = I[:nb * L].reshape(nb, L)
    W = W1[:nb * L].reshape(nb, L)
    bI = I.sum(axis=1); bW = W.sum(axis=1)
    th = bI.sum() / bW.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(B, nb))
    ths = bI[idx].sum(axis=1) / bW[idx].sum(axis=1)
    lo, hi = np.percentile(ths, [2.5, 97.5])
    return th, (lo, hi), nb


def split_half_z(x, samples_per_tau, blocks_tau=5.0):
    """Stationarity drift z between the two halves, block-based SEs."""
    n = len(x); h = n // 2
    L = max(2, int(round(blocks_tau * samples_per_tau)))

    def mean_se(y):
        nb = len(y) // L
        if nb < 4:
            return np.mean(y), np.inf
        m = y[:nb * L].reshape(nb, L).mean(axis=1)
        return m.mean(), m.std(ddof=1) / np.sqrt(nb)
    m1, s1 = mean_se(x[:h]); m2, s2 = mean_se(x[h:])
    return (m2 - m1) / np.sqrt(s1 ** 2 + s2 ** 2)


def analyze(res, label="", blocks_tau=5.0):
    th, (lo, hi), nb = block_bootstrap_ratio(res["I"], res["W1"],
                                             res["samples_per_tau"],
                                             blocks_tau=blocks_tau)
    zI = split_half_z(res["I"], res["samples_per_tau"], blocks_tau=blocks_tau)
    zW = split_half_z(res["W1"], res["samples_per_tau"], blocks_tau=blocks_tau)
    print(f"  {label} nu={res['nu']:<6g} seed={res['seed']} N={res['N']} "
          f"dt={res['dt']:.2e} n={res['n']} blocks={nb} | "
          f"E[I]={res['I'].mean():+.4e} E[W1]={res['W1'].mean():.4e} | "
          f"theta={th:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] | "
          f"z_I={zI:+.1f} z_W1={zW:+.1f} | wall={res['wall']:.0f}s",
          flush=True)
    return th, lo, hi


def dt_for(nu):
    """Exact linear part => dt limited only by the (slow) nonlinearity and
    drift-noise interaction; resolve the fastest banded relaxation ~20x."""
    return min(2e-3, 0.05 / (16.0 * nu))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--refine", action="store_true")
    args = ap.parse_args()

    print("=" * 90)
    print("large-nu limit of theta -- theta_G(band) > 0 vs 0 (S6)   "
          f"[B0={B0}, band=[1,4]]")
    print("=" * 90, flush=True)

    if args.coarse or not (args.grid or args.refine):
        print("[1] Gaussian reference theta_G(band) -- exact sampling of the "
              "linear OU stationary measure:", flush=True)
        t0 = time.time()
        thG, seG, EI_G, EW_G = theta_gaussian_mc(M=200_000)
        print(f"  theta_G(band[1,4]) = {thG:+.4f} +/- {seG:.4f}  "
              f"(E_G[I]={EI_G:+.4e}, E_G[W1]={EW_G:.4e})  [{time.time()-t0:.0f}s]")
        print(f"  banked Gaussian family floor (min over shapes m): 0.7948",
              flush=True)

        print("[2] ANCHOR nu=0.05 (banked theta ~ 0.78-0.81; validates the "
              "exact-OU stepper at moderate nu):", flush=True)
        res = run_theta(nu=0.05, N=64, dt=2e-3, T_relax_units=8.0, seed=10,
                        burn_relax_units=3.0, samples_per_tau=200)
        analyze(res, "[anchor]", blocks_tau=0.5)  # eddy time << visc tau here

        print("[3] DECISIVE POINT nu=10 (coarse, 1 seed):", flush=True)
        res = run_theta(nu=10.0, N=64, dt=dt_for(10.0), T_relax_units=2000.0,
                        seed=0)
        analyze(res, "[nu=10]")

    if args.grid:
        print("[4] FULL GRID nu in {0.5,1,2,5,10,20} x seeds {0,1}:", flush=True)
        for nu in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            for seed in [0, 1]:
                res = run_theta(nu=nu, N=64, dt=dt_for(nu),
                                T_relax_units=2000.0, seed=seed)
                analyze(res, "[grid]")

    if args.refine:
        print("[5] dt-REFINEMENT RECEIPT nu=10 (dt/5) + repo-EM-noise "
              "comparison (quantifies the banked-run bias):", flush=True)
        res = run_theta(nu=10.0, N=64, dt=dt_for(10.0) / 5.0,
                        T_relax_units=2000.0, seed=0)
        analyze(res, "[dt/5]")
        res = run_theta(nu=10.0, N=64, dt=1e-3, T_relax_units=2000.0, seed=0,
                        exact_noise=False)
        analyze(res, "[repo-EM dt=1e-3]")
    print("DONE", flush=True)
