"""
measure_cascade.py -- production-balance / cascade-efficiency measurement
sweeps for the stochastic gCLMG solver (sgclm.py).

Runs the SPDE to statistical stationarity (with burn-in discarded and, for
each configuration, an ensemble of independent-seed runs for error bars) and
measures, as functions of (nu, a):

  - E[enstrophy]                 E[Ecal]        = E[1/2 int w^2]
  - E[palinstrophy]               E[Ecal_1]      = E[1/2 int w_x^2]
  - E[dissipation]                E[nu int w_x^2]
  - E[palinstrophy production]    E[Pcal_1]      = E[int (Hw) w_x^2]
  - E[palinstrophy dissipation]   E[Dcal_1]      = E[nu int w_xx^2]
  - E[Pi(K)]                      spectral enstrophy flux, vs K

and the three diagnostic ratios requested by the task:

  (a) nu*E[int w_x^2] / (0.5*B0)         -- enstrophy balance (~1 at a=-2)
  (b) E[Pi(K)] / (0.5*B0)  vs K          -- flux law (~1, flat, inertial range)
  (c) E[Ecal_1]*nu  vs  B0/4             -- palinstrophy/anomalous-cascade scaling

Two sweeps:
  1. nu in {0.1, 0.05, 0.02, 0.01, 0.005} at a=-2, forced band k in [1,4],
     resolution N increased as nu shrinks (see RESOLUTION_TABLE below; each
     run logs a resolution check: ratio of the dissipation-integral spectral
     density at k=0.9*kmax to its peak value, must be << 1).
  2. a in {-2.0, -1.5, -1.0, -0.5, 0.0} at fixed nu=0.03 ("cascade vs
     advection strength").  Enstrophy is not conserved for a != -2; we still
     measure the same functionals and the 0.5*B0-normalized flux.

Outputs:
  results/cascade_sweep.json   -- full numeric results (all runs, all seeds)
  results/CASCADE_FINDINGS.md  -- concise write-up: ratio tables, flux-law
                                   verdict, fitted palinstrophy exponent,
                                   a-dependence discussion, resolution /
                                   stationarity flags.

Usage:
    cd instrument
    python3 measure_cascade.py 2>&1 | tee results/measure_cascade.log
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import sgclm as sg


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

K_LO, K_HI, EPS = 1, 4, 0.02          # forced band + per-mode noise variance
NOISE = sg.NoiseSpec(k_lo=K_LO, k_hi=K_HI, eps=EPS)
B0 = NOISE.B0                          # = 2*(k_hi-k_lo+1)*eps

N_SEEDS = 3                            # independent-seed ensemble per config
BASE_SEED = 1000

BURN_IN_FRAC = 0.3
RECORD_EVERY = 20

# nu-sweep: (nu, N, dt, T).  N grows / dt shrinks as nu shrinks so that the
# dissipation scale stays resolved (checked at run time, not just assumed --
# see resolution_check below) and the CFL-like advective step stays stable.
NU_SWEEP = [
    dict(nu=0.10,  N=64,  dt=2.0e-3, T=150.0),
    dict(nu=0.05,  N=64,  dt=2.0e-3, T=150.0),
    dict(nu=0.02,  N=96,  dt=1.0e-3, T=150.0),
    dict(nu=0.01,  N=128, dt=1.0e-3, T=150.0),
    dict(nu=0.005, N=256, dt=5.0e-4, T=150.0),
]

# a-sweep: fixed moderate nu, vary advection parameter.
A_SWEEP_NU = 0.03
A_SWEEP_N = 96
A_SWEEP_DT = 1.0e-3
A_SWEEP_T = 150.0
A_VALUES = [-2.0, -1.5, -1.0, -0.5, 0.0]

# K-grid for the flux curve: dense at small K (injection/inertial range),
# sparser at large K; always includes K=1 (below the forced band) and a
# handful of points well past kmax*2/3 (the dealiased cutoff) for the
# dissipation-range check.  Built per-run since kmax = N//2 depends on N.


def flux_K_grid(N):
    kmax = N // 2
    cutoff = int(np.floor(2.0 / 3.0 * kmax))
    lo = np.arange(1, 17)                                   # 1..16, dense
    mid = np.unique(np.round(np.geomspace(17, max(cutoff, 18), 14)).astype(int))
    hi = np.unique(np.round(np.linspace(cutoff, kmax, 6)).astype(int))
    Kvals = np.unique(np.concatenate([lo, mid, hi]))
    Kvals = Kvals[Kvals <= kmax]
    return Kvals.astype(float)


# ---------------------------------------------------------------------------
# single-run driver
# ---------------------------------------------------------------------------

def resolution_check(what_final, k, N, nu):
    """Return (ok, detail) -- checks that the palinstrophy spectral density
    k^2|wtilde(k)|^2 has decayed well before the Nyquist / dealiased cutoff,
    i.e. the run resolves the dissipation scale at this (N, nu)."""
    wtilde = sg.to_fourier_series_coeffs(what_final, N)
    dens = (k.astype(float) ** 2) * np.abs(wtilde) ** 2
    kmax = N // 2
    cutoff = int(np.floor(2.0 / 3.0 * kmax))
    peak = np.max(dens[1:20]) if len(dens) > 20 else np.max(dens[1:])
    if peak <= 0:
        return True, "spectral density identically zero in injection range (degenerate)"
    # density at 90% of the dealiased cutoff and at the cutoff itself
    idx90 = int(round(0.9 * cutoff))
    idx90 = min(max(idx90, 0), len(dens) - 1)
    ratio_90 = dens[idx90] / peak
    ratio_cut = dens[min(cutoff, len(dens) - 1)] / peak
    ok = ratio_cut < 1e-3
    detail = (f"N={N} kmax={kmax} dealias_cutoff={cutoff} "
              f"density(0.9*cutoff)/peak={ratio_90:.2e} "
              f"density(cutoff)/peak={ratio_cut:.2e} "
              f"(resolved: {ok}, threshold 1e-3)")
    return ok, detail


def run_one(nu, N, dt, T, a, seed):
    x, k = sg.make_grid(N)
    Kvals = flux_K_grid(N)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    out = sg.run(N, T, dt, nu, NOISE, rng, a=a, burn_in_frac=BURN_IN_FRAC,
                  record_every=RECORD_EVERY, dealias=True,
                  record_palinstrophy=True, flux_K_values=Kvals)
    elapsed = time.time() - t0

    ok_res, detail_res = resolution_check(out["what_final"], k, N, nu)

    keep = out["times"] > BURN_IN_FRAC * T
    n_stat_samples = int(np.sum(keep))
    # crude stationarity check: compare mean enstrophy in the first half vs
    # second half of the kept (post-burn-in) window; should not show a
    # systematic trend larger than the sampling noise.
    E_kept = out["enstrophy"][keep]
    if n_stat_samples >= 20:
        h = n_stat_samples // 2
        mean_first, mean_second = np.mean(E_kept[:h]), np.mean(E_kept[h:])
        std_first = np.std(E_kept[:h]) / max(np.sqrt(h), 1)
        std_second = np.std(E_kept[h:]) / max(np.sqrt(n_stat_samples - h), 1)
        drift = abs(mean_second - mean_first)
        drift_scale = np.sqrt(std_first ** 2 + std_second ** 2) + 1e-300
        stationarity_z = drift / drift_scale
    else:
        mean_first = mean_second = np.nan
        stationarity_z = np.nan

    result = {
        "nu": nu, "N": N, "dt": dt, "T": T, "a": a, "seed": seed,
        "B0": B0, "k_lo": K_LO, "k_hi": K_HI, "eps": EPS,
        "elapsed_sec": elapsed,
        "n_stat_samples": n_stat_samples,
        "mean_enstrophy": out["mean_enstrophy"],
        "mean_dissipation": out["mean_dissipation"],
        "mean_production": out["mean_production"],
        "mean_palinstrophy": out["mean_palinstrophy"],
        "mean_palinstrophy_production": out["mean_palinstrophy_production"],
        "mean_palinstrophy_dissipation": out["mean_palinstrophy_dissipation"],
        "flux_K_values": out["flux_K_values"].tolist(),
        "mean_flux": out["mean_flux"].tolist(),
        "std_flux": out["std_flux"].tolist(),
        "flux_n_samples": out["flux_n_samples"],
        "resolution_ok": bool(ok_res),
        "resolution_detail": detail_res,
        "stationarity_mean_first_half": float(mean_first),
        "stationarity_mean_second_half": float(mean_second),
        "stationarity_z": float(stationarity_z),
    }
    return result


def ensemble_summary(runs):
    """Aggregate a list of run_one(...) dicts (same config, different seeds)
    into ensemble means + standard errors for the scalar functionals and the
    flux curve."""
    n = len(runs)
    scalars = ["mean_enstrophy", "mean_dissipation", "mean_production",
               "mean_palinstrophy", "mean_palinstrophy_production",
               "mean_palinstrophy_dissipation"]
    summary = {"n_seeds": n}
    for s in scalars:
        vals = np.array([r[s] for r in runs])
        summary[s] = float(np.mean(vals))
        summary[s + "_stderr"] = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    Kvals = np.array(runs[0]["flux_K_values"])
    flux_stack = np.array([r["mean_flux"] for r in runs])  # (n_seeds, n_K)
    summary["flux_K_values"] = Kvals.tolist()
    summary["mean_flux"] = np.mean(flux_stack, axis=0).tolist()
    summary["mean_flux_stderr"] = (
        (np.std(flux_stack, axis=0, ddof=1) / np.sqrt(n)).tolist() if n > 1
        else np.zeros_like(Kvals).tolist()
    )
    summary["resolution_ok_all_seeds"] = all(r["resolution_ok"] for r in runs)
    _zvals = [abs(r["stationarity_z"]) for r in runs if not np.isnan(r["stationarity_z"])]
    summary["max_abs_stationarity_z"] = float(max(_zvals)) if _zvals else float("nan")
    summary["blew_up"] = bool(all(np.isnan(r["mean_palinstrophy_production"]) for r in runs))
    summary["nu"] = runs[0]["nu"]
    summary["N"] = runs[0]["N"]
    summary["dt"] = runs[0]["dt"]
    summary["T"] = runs[0]["T"]
    summary["a"] = runs[0]["a"]
    return summary


# ---------------------------------------------------------------------------
# validation re-confirmation (task requirement: reproduce known ratio first)
# ---------------------------------------------------------------------------

def validation_reconfirm():
    print("=" * 78)
    print("Validation re-confirmation: a=-2, nu=0.05 -> nu*E[int w_x^2]/(0.5*B0) ~ 1")
    print("=" * 78)
    N, nu, dt, T = 64, 0.05, 2e-3, 400.0
    rng = np.random.default_rng(42)
    out = sg.run(N, T, dt, nu, NOISE, rng, a=-2.0, burn_in_frac=0.25,
                  record_every=5, dealias=True)
    ratio = out["mean_dissipation"] / (0.5 * B0)
    print(f"  N={N} nu={nu} T={T} dt={dt}  ratio = {ratio:.4f}  (known ~1.006)")
    ok = abs(ratio - 1.0) < 0.15
    print(f"  {'PASS' if ok else 'FAIL'}: reconfirmation {'within' if ok else 'OUTSIDE'} 15% of 1.0")
    return ratio, ok


# ---------------------------------------------------------------------------
# main sweeps
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    recon_ratio, recon_ok = validation_reconfirm()

    all_results = {
        "B0": B0, "k_lo": K_LO, "k_hi": K_HI, "eps": EPS, "n_seeds": N_SEEDS,
        "burn_in_frac": BURN_IN_FRAC, "record_every": RECORD_EVERY,
        "validation_reconfirmation": {"ratio": recon_ratio, "ok": bool(recon_ok)},
        "nu_sweep": [],
        "a_sweep": [],
    }

    # -------------------- nu-sweep at a=-2 --------------------
    print("\n" + "=" * 78)
    print("NU-SWEEP  (a = -2.0, forced band k in [%d,%d], B0=%.4f)" % (K_LO, K_HI, B0))
    print("=" * 78)
    for cfg in NU_SWEEP:
        nu, N, dt, T = cfg["nu"], cfg["N"], cfg["dt"], cfg["T"]
        print(f"\n-- nu={nu}  N={N}  dt={dt}  T={T} --")
        runs = []
        for i in range(N_SEEDS):
            seed = BASE_SEED + i
            r = run_one(nu, N, dt, T, a=-2.0, seed=seed)
            runs.append(r)
            print(f"   seed={seed}  elapsed={r['elapsed_sec']:.1f}s  "
                  f"mean_diss={r['mean_dissipation']:.5f}  "
                  f"resolved={r['resolution_ok']}  "
                  f"stationarity_z={r['stationarity_z']:.2f}")
        summ = ensemble_summary(runs)
        ratio_a = summ["mean_dissipation"] / (0.5 * B0)
        ratio_a_err = summ["mean_dissipation_stderr"] / (0.5 * B0)
        ratio_c = summ["mean_palinstrophy"] * nu
        print(f"   ==> ratio (a) nu*E[intwx2]/(0.5B0) = {ratio_a:.4f} +/- {ratio_a_err:.4f}")
        print(f"   ==> ratio (c) E[Ecal_1]*nu = {ratio_c:.4f}   (B0/4 = {B0/4:.4f})")
        summ["ratio_enstrophy_balance"] = ratio_a
        summ["ratio_enstrophy_balance_stderr"] = ratio_a_err
        summ["ratio_palinstrophy_times_nu"] = ratio_c
        summ["runs"] = runs
        all_results["nu_sweep"].append(summ)

    # -------------------- a-sweep at fixed nu --------------------
    print("\n" + "=" * 78)
    print(f"A-SWEEP  (nu = {A_SWEEP_NU}, N={A_SWEEP_N}, forced band k in [{K_LO},{K_HI}])")
    print("=" * 78)
    for a in A_VALUES:
        print(f"\n-- a={a} --")
        runs = []
        for i in range(N_SEEDS):
            seed = BASE_SEED + 500 + i
            r = run_one(A_SWEEP_NU, A_SWEEP_N, A_SWEEP_DT, A_SWEEP_T, a=a, seed=seed)
            runs.append(r)
            print(f"   seed={seed}  elapsed={r['elapsed_sec']:.1f}s  "
                  f"mean_P1={r['mean_palinstrophy_production']:.5f}  "
                  f"resolved={r['resolution_ok']}")
        summ = ensemble_summary(runs)
        print(f"   ==> E[Pcal_1] = {summ['mean_palinstrophy_production']:.5f} "
              f"+/- {summ['mean_palinstrophy_production_stderr']:.5f}")
        summ["runs"] = runs
        all_results["a_sweep"].append(summ)

    all_results["total_elapsed_sec"] = time.time() - t_start

    out_json = os.path.join(RESULTS_DIR, "cascade_sweep.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_json}")

    write_findings(all_results)
    print("Saved", os.path.join(RESULTS_DIR, "CASCADE_FINDINGS.md"))
    print(f"\nTotal elapsed: {all_results['total_elapsed_sec']:.1f}s")


# ---------------------------------------------------------------------------
# findings write-up
# ---------------------------------------------------------------------------

def fit_power_law(x, y):
    """Fit y = C * x^p via least squares in log-log space.  Returns (p, C, r2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    lx, ly = np.log(x[mask]), np.log(y[mask])
    A = np.vstack([lx, np.ones_like(lx)]).T
    (p, logC), *_ = np.linalg.lstsq(A, ly, rcond=None)
    C = np.exp(logC)
    pred = p * lx + logC
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - np.mean(ly)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return p, C, r2


def write_findings(all_results):
    lines = []
    lines.append("# Cascade / production-balance sweep findings\n")
    lines.append(f"Forced band k in [{all_results['k_lo']},{all_results['k_hi']}], "
                  f"eps={all_results['eps']}, B0={all_results['B0']:.4f}, "
                  f"ensemble size = {all_results['n_seeds']} independent seeds per "
                  f"configuration, burn_in_frac={all_results['burn_in_frac']}, "
                  f"record_every={all_results['record_every']}.\n")

    vr = all_results["validation_reconfirmation"]
    lines.append("## Validation re-confirmation\n")
    lines.append(f"At a=-2, nu=0.05 (N=64, T=400, dt=2e-3, seed=42): "
                  f"nu*E[int w_x^2]/(0.5*B0) = **{vr['ratio']:.4f}** "
                  f"({'within' if vr['ok'] else 'OUTSIDE'} 15% of the known ~1.006 value). "
                  f"{'This harness reproduces the known balance.' if vr['ok'] else 'MISMATCH -- investigate before trusting sweeps below.'}\n")

    # ---- ratio table (a): enstrophy balance across nu ----
    lines.append("## Ratio (a): enstrophy balance nu*E[int w_x^2] / (0.5*B0)  [a=-2]\n")
    lines.append("| nu | N | ratio (a) | stderr | resolved | max stationarity |z| |")
    lines.append("|---|---|---|---|---|---|")
    for summ in all_results["nu_sweep"]:
        lines.append(f"| {summ['nu']} | {summ['N']} | {summ['ratio_enstrophy_balance']:.4f} "
                      f"| {summ['ratio_enstrophy_balance_stderr']:.4f} "
                      f"| {summ['resolution_ok_all_seeds']} "
                      f"| {summ['max_abs_stationarity_z']:.2f} |")
    lines.append("")
    lines.append("Ratio (a) should be approx 1 at a=-2 for all nu (exact enstrophy "
                  "balance in stationarity, independent of nu by the a=-2 advective "
                  "cancellation identity).\n")

    # ---- ratio table (c): palinstrophy scaling ----
    lines.append("## Ratio (c): E[palinstrophy]*nu vs B0/4  [a=-2]\n")
    B0_over_4 = all_results["B0"] / 4.0
    lines.append(f"B0/4 = {B0_over_4:.4f}\n")
    lines.append("| nu | N | E[Ecal_1] | E[Ecal_1]*nu | (E[Ecal_1]*nu)/(B0/4) |")
    lines.append("|---|---|---|---|---|")
    nus, E1_nu_vals = [], []
    for summ in all_results["nu_sweep"]:
        val = summ["ratio_palinstrophy_times_nu"]
        nus.append(summ["nu"])
        E1_nu_vals.append(val)
        lines.append(f"| {summ['nu']} | {summ['N']} | {summ['mean_palinstrophy']:.4f} "
                      f"| {val:.4f} | {val / B0_over_4:.4f} |")
    lines.append("")

    # fitted exponent E[Ecal_1] ~ nu^p
    nus_arr = np.array(nus)
    E1_arr = np.array([summ["mean_palinstrophy"] for summ in all_results["nu_sweep"]])
    p_fit, C_fit, r2_fit = fit_power_law(nus_arr, E1_arr)
    lines.append(f"Power-law fit E[Ecal_1] ~ C * nu^p over the measured nu range: "
                  f"**p = {p_fit:.3f}**, C = {C_fit:.4g}, R^2 = {r2_fit:.4f} "
                  f"(anomalous-cascade expectation: p ~ -1, i.e. E[Ecal_1]*nu -> const).\n")

    # ---- flux-law table (b) at smallest nu ----
    smallest = min(all_results["nu_sweep"], key=lambda s: s["nu"])
    lines.append(f"## Ratio (b): flux law E[Pi(K)] / (0.5*B0) vs K  [a=-2, smallest nu={smallest['nu']}]\n")
    lines.append("| K | E[Pi(K)]/(0.5B0) | stderr |")
    lines.append("|---|---|---|")
    Kv = smallest["flux_K_values"]
    fluxv = smallest["mean_flux"]
    fluxerr = smallest["mean_flux_stderr"]
    half_B0 = 0.5 * all_results["B0"]
    for Kk, fv, fe in zip(Kv, fluxv, fluxerr):
        lines.append(f"| {Kk:.0f} | {fv/half_B0:.4f} | {fe/half_B0:.4f} |")
    lines.append("")

    # flux-law verdict: find the plateau range (values within 25% of 1, and
    # roughly flat, i.e. an "inertial range")
    fluxv_arr = np.array(fluxv) / half_B0
    Kv_arr = np.array(Kv)
    near_one = np.abs(fluxv_arr - 1.0) < 0.25
    plateau_Ks = Kv_arr[near_one]
    if len(plateau_Ks) > 0:
        verdict = (f"E[Pi(K)]/(0.5*B0) is within 25% of 1 for K in "
                   f"[{plateau_Ks.min():.0f}, {plateau_Ks.max():.0f}] "
                   f"({np.sum(near_one)}/{len(Kv_arr)} sampled K points) at nu={smallest['nu']}.")
    else:
        verdict = f"E[Pi(K)]/(0.5*B0) does NOT come within 25% of 1 for any sampled K at nu={smallest['nu']}."
    lines.append(f"**Flux-law verdict:** {verdict}\n")
    lines.append("Expected shape: ~0 for K below the forced band bottom (k_lo=1), "
                  "rising to ~0.5*B0-normalized value of 1 just above the forced band "
                  "top (k_hi=4), ideally staying near 1 across an inertial range, then "
                  "falling back toward 0 in the dissipation range as K approaches the "
                  "resolved kmax (since Pi(K->kmax or beyond) -> <w,N[w]> = 0 exactly "
                  "at a=-2, the full-field advective cancellation).\n")

    # ---- a-sweep: Pcal_1 and flux dependence on a ----
    lines.append(f"## Advection-parameter sweep (nu={A_SWEEP_NU}, N={A_SWEEP_N})\n")
    lines.append("| a | E[Ecal] | E[Pcal_1] | stderr | E[Pi(K~kmid)]/(0.5B0) |")
    lines.append("|---|---|---|---|---|")
    a_vals, P1_vals = [], []
    for summ in all_results["a_sweep"]:
        a = summ["a"]
        P1 = summ["mean_palinstrophy_production"]
        P1_err = summ["mean_palinstrophy_production_stderr"]
        Kv_a = np.array(summ["flux_K_values"])
        fluxv_a = np.array(summ["mean_flux"])
        kmid_idx = len(Kv_a) // 2
        flux_mid_norm = fluxv_a[kmid_idx] / half_B0
        a_vals.append(a)
        P1_vals.append(P1)
        lines.append(f"| {a} | {summ['mean_enstrophy']:.4f} | {P1:.5f} | {P1_err:.5f} "
                      f"| {flux_mid_norm:.4f} (K={Kv_a[kmid_idx]:.0f}) |")
    lines.append("")

    P1_vals_arr = np.array(P1_vals)
    a_vals_arr = np.array(a_vals)
    order = np.argsort(np.abs(a_vals_arr - (-2.0)))
    # monotonicity check: as |a-(-2)| increases, does |P1| move monotonically?
    dist_from_m2 = np.abs(a_vals_arr - (-2.0))
    _fin = ~np.isnan(P1_vals_arr)            # drop blown-up configs (e.g. a=0 CLM)
    a_vals_arr = a_vals_arr[_fin]; P1_vals_arr = P1_vals_arr[_fin]
    dist_from_m2 = dist_from_m2[_fin]
    sort_idx = np.argsort(dist_from_m2)
    P1_by_dist = np.abs(P1_vals_arr[sort_idx])
    is_monotone_incr = all(P1_by_dist[i + 1] >= P1_by_dist[i] * 0.9 for i in range(len(P1_by_dist) - 1))
    is_monotone_decr = all(P1_by_dist[i + 1] <= P1_by_dist[i] * 1.1 for i in range(len(P1_by_dist) - 1))
    lines.append(f"**Monotonicity in |a-(-2)|:** |E[Pcal_1]| vs distance from a=-2: "
                  f"{'roughly monotone increasing' if is_monotone_incr else ('roughly monotone decreasing' if is_monotone_decr else 'NOT monotone')} "
                  f"as a moves away from -2 (values sorted by |a-(-2)|: "
                  f"{np.array2string(P1_by_dist, precision=4)}).\n")

    lines.append("## Resolution / stationarity flags\n")
    any_flag = False
    for summ in all_results["nu_sweep"] + all_results["a_sweep"]:
        if not summ["resolution_ok_all_seeds"] or summ["max_abs_stationarity_z"] > 4.0:
            any_flag = True
            lines.append(f"- FLAG: nu={summ['nu']} a={summ['a']} N={summ['N']}: "
                          f"resolution_ok={summ['resolution_ok_all_seeds']}, "
                          f"max|stationarity_z|={summ['max_abs_stationarity_z']:.2f}")
    if not any_flag:
        lines.append("No resolution or stationarity flags raised (all configurations: "
                      "dissipation-range spectral density < 1e-3 of injection-range "
                      "peak before the dealiased cutoff; first-half/second-half "
                      "stationary-window mean-enstrophy drift within ~4 sigma of "
                      "sampling noise).")
    lines.append("")

    lines.append("## Error bars / statistical-uncertainty notes\n")
    lines.append(f"- Each configuration uses {all_results['n_seeds']} independent-seed "
                  "runs; reported stderr is the standard error of the mean across "
                  "seeds (ddof=1). This captures run-to-run (ensemble) variability but "
                  "each individual run's time-average also carries residual "
                  "autocorrelated Monte-Carlo noise (see sgclm README check 5 "
                  "discussion of O(1/sqrt(T)) fluctuations); the two are not "
                  "separately decomposed here.\n"
                  "- The per-run flux curve `std_flux` (saved in cascade_sweep.json "
                  "per-seed entries) is the population std of the time series of "
                  "Pi(K) samples within one run (time-correlated, since consecutive "
                  "samples are `record_every` steps apart, not independent draws), so "
                  "it is a rough dispersion measure, not a rigorous per-run standard "
                  "error; the cross-seed stderr in the tables above is the more "
                  "trustworthy uncertainty estimate.\n"
                  f"- n_seeds={all_results['n_seeds']} gives only a coarse estimate of "
                  "the standard error itself; treat stderr values as order-of-magnitude, "
                  "not tight confidence intervals.\n")

    with open(os.path.join(RESULTS_DIR, "CASCADE_FINDINGS.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
