#!/usr/bin/env python3
"""Generate the paper figures from committed data (no new simulation).

Sources (all committed):
  swing/p2_results.jsonl                 -- hardened small-nu campaign: theta, rho5 per (nu,seed)
  swing/theta_large_nu_exact_ou.log      -- large-nu grid: theta per nu (exact-OU stepping)
  instrument/results/cascade_sweep.json  -- flux Pi(K) per nu (coarse exploratory sweep)

Outputs PDF figures into theory/figs/ for \\includegraphics.
Run:  python3 swing/make_figures.py
"""
import json, os, re
from collections import defaultdict
from fractions import Fraction
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "theory", "figs")
os.makedirs(OUT, exist_ok=True)

B0 = 0.16
HALF_B0 = 0.5 * B0                      # 0.08
THETA_G = float(Fraction(2017, 2484))   # 0.811996779...

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

# ---- small-nu hardened campaign: theta(nu), rho5(nu) ---------------------
_p2_path = os.path.join(HERE, "p2_results_v18.jsonl")     # v1.8 campaign (strict mask + padded diagnostics)
for _fb in ("p2_results_v17.jsonl", "p2_results.jsonl"):
    if not os.path.exists(_p2_path):
        _p2_path = os.path.join(HERE, _fb)
p2 = [json.loads(l) for l in open(_p2_path) if l.strip()]
p2 = [r for r in p2 if r.get("tag", "v18") in ("v17", "v18")]  # campaign rows only (dt/2, N-up, large-nu checks live in the matrix figure/tables)
theta_by_nu = defaultdict(list); rho5_by_nu = defaultdict(list)
for r in p2:
    theta_by_nu[r["nu"]].append(r["theta"])
    if "rho5" in r:
        rho5_by_nu[r["nu"]].append(r["rho5"])
nus_s = sorted(theta_by_nu)
th_s = np.array([np.mean(theta_by_nu[n]) for n in nus_s])
th_s_e = np.array([np.std(theta_by_nu[n])/max(1,len(theta_by_nu[n])-1)**0.5 for n in nus_s])

# ---- large-nu grid from the log -----------------------------------------
big = defaultdict(list)
for line in open(os.path.join(HERE, "theta_large_nu_exact_ou.log")):
    if "[grid]" not in line:
        continue
    m = re.search(r"nu=([0-9.]+).*?theta=\+?(-?[0-9.]+)", line)
    if m:
        big[float(m.group(1))].append(float(m.group(2)))
nus_b = sorted(big)
th_b = np.array([np.mean(big[n]) for n in nus_b])

# ============ Figure 1: theta(nu) vs nu, with theta_G anchor =============
fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.axhline(THETA_G, ls="--", color="k", lw=1,
           label=r"$\theta_G=2017/2484$")
ax.errorbar(nus_s, th_s, yerr=th_s_e, fmt="o-", color="C0", capsize=2,
            label="stationary campaign (small $\\nu$)")
if nus_b:
    ax.plot(nus_b, th_b, "s-", color="C3", label="exact-OU grid (large $\\nu$)")
ax.set_xscale("log")
ax.set_xlabel(r"$\nu$"); ax.set_ylabel(r"$\theta(\nu)$")
ax.set_title(r"Normalized production diagnostic $\theta$ vs. viscosity")
ax.legend(fontsize=9, loc="lower right")
fig.savefig(os.path.join(OUT, "theta_vs_nu.pdf")); plt.close(fig)

# ============ Figure 2: deficit theta_G - theta(nu), log-law ============
defic = THETA_G - th_s
fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.errorbar(nus_s, defic, yerr=th_s_e, fmt="o", color="C0", capsize=2,
            label="measured deficit")
xx = np.logspace(np.log10(min(nus_s)), np.log10(max(nus_s)), 100)
_b, _a = np.polyfit(np.log(nus_s), defic, 1)       # defic ~ _a + _b ln(nu)
ax.plot(xx, _a + _b*np.log(xx), "-", color="C1",
        label=rf"${_a:+.4f}{_b:+.4f}\,\ln\nu$")
print(f"deficit log-fit: a={_a:+.4f} b={_b:+.4f} resid={np.abs(defic-(_a+_b*np.log(nus_s))).max():.4f}")
ax.set_xscale("log")
ax.set_xlabel(r"$\nu$"); ax.set_ylabel(r"$\theta_G-\theta(\nu)$")
ax.set_title(r"Finite-$\nu$ deficit below the Gaussian anchor")
ax.legend(fontsize=9, loc="upper right")
fig.savefig(os.path.join(OUT, "theta_deficit.pdf")); plt.close(fig)

# ============ Figure 3: rho_nu([-5,5])/(B0/2) drain =====================
if rho5_by_nu:
    nus_r = sorted(rho5_by_nu)
    rho = np.array([np.mean(rho5_by_nu[n]) for n in nus_r]) / HALF_B0
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(nus_r, rho, "o-", color="C2")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$\rho_\nu(\{|k|<5\})/(\frac{1}{2}B_0)$")
    ax.set_title(r"Low-mode dissipation share drains as $\nu\to0$ ($(\dagger$-weak$)$)")
    ax.set_ylim(0, 1.05)
    fig.savefig(os.path.join(OUT, "rho_drain.pdf")); plt.close(fig)

# ============ Figure 4: flux Pi(K)/(B0/2) vs K, several nu ==============
# Sourced from the HARDENED campaign via the exact identity (Theorem A1):
#   Pi(K)/(B0/2) = 1 - rho_nu([-K,K))/(B0/2),  rho_K committed in p2_results.jsonl.
RHO_KS = (5, 6, 8, 12)
flux_by_nu = defaultdict(dict)
for r in p2:
    if r.get("dt") == 2.5e-4:           # skip the dt/2 receipt row
        continue
    for K in RHO_KS:
        flux_by_nu[r["nu"]].setdefault(K, []).append(1.0 - r[f"rho{K}"] / HALF_B0)
fig, ax = plt.subplots(figsize=(5.4, 3.6))
for nu in sorted(flux_by_nu, reverse=True):
    K = np.array(RHO_KS, dtype=float)
    F = np.array([np.mean(flux_by_nu[nu][k]) for k in RHO_KS])
    ax.plot(K, F, "o-", ms=3, label=rf"$\nu={nu:g}$")
ax.axhline(1.0, ls=":", color="k", lw=0.8)
ax.set_xlabel(r"$K$"); ax.set_ylabel(r"$\Pi(K)/(\frac{1}{2}B_0)$")
ax.set_title(r"Enstrophy flux: broad maximum approaching $\frac{1}{2}B_0$ as $\nu\to0$"
             "\n" r"(stationary campaign, via $\Pi(K)=\frac{1}{2}B_0-\rho_\nu([-K,K))$)")
ax.legend(fontsize=8, ncol=2)
fig.savefig(os.path.join(OUT, "flux_pi.pdf")); plt.close(fig)

# ============ Figure 5: convergence matrix (theta vs nu, all checks) ======
conv_path = os.path.join(HERE, "p2_convergence_v18.jsonl")
for _fb in ("p2_convergence_v17.jsonl", "p2_convergence.jsonl"):
    if not os.path.exists(conv_path):
        conv_path = os.path.join(HERE, _fb)
if os.path.exists(conv_path):
    conv = [json.loads(l) for l in open(conv_path) if l.strip()]
    # v1.9: merge the tagged receipt rows that live in the campaign file (N-up 384,
    # dt/2 at nu=0.01) -- advertised in the caption, previously omitted from the plot
    _all_rows = [json.loads(l) for l in open(_p2_path) if l.strip()]
    conv += [r for r in _all_rows if r.get("tag") in ("N-up", "dt/2")]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    # base campaign: pooled 2-seed points with jackknife SE
    base_se = {}
    for nu in nus_s:
        rs = [r for r in p2 if r["nu"] == nu and r.get("dt") != 2.5e-4]
        base_se[nu] = np.mean([r["theta_se"] for r in rs])
    ax.errorbar(nus_s, th_s, yerr=[base_se[n] for n in nus_s], fmt="o", ms=5,
                color="k", label="base grid (2 seeds)", zorder=5)
    MARK = {"dt/2": ("s", "C0"), "dt/4": ("D", "C1"), "N-up": ("^", "C2"),
            "far-IC": ("v", "C3"), "tau-ref": ("x", "C4")}
    seen = set()
    for r in conv:
        m, c = MARK.get(r["tag"], ("*", "C5"))
        lbl = r["tag"] if r["tag"] not in seen else None
        seen.add(r["tag"])
        ax.errorbar([r["nu"]], [r["theta"]], yerr=[r["theta_se"]], fmt=m, ms=5,
                    color=c, label=lbl, alpha=0.85)
    ax.axhline(THETA_G, ls=":", color="gray", lw=0.8)
    ax.text(0.006, THETA_G + 0.001, r"$\theta_G$", color="gray")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\nu$"); ax.set_ylabel(r"$\theta$")
    ax.set_title("Convergence matrix: step, resolution and initial-state checks\n"
                 "agree with the base grid within statistical bands")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(os.path.join(OUT, "theta_convergence.pdf")); plt.close(fig)
    print("convergence rows:", len(conv))

print("theta_G =", THETA_G)
print("small-nu:", list(zip(nus_s, np.round(th_s,4))))
print("large-nu:", list(zip(nus_b, np.round(th_b,4))))
if rho5_by_nu:
    print("rho5/(B0/2):", list(zip(nus_r, np.round(rho,3))))
print("wrote 4 figures to", OUT)
