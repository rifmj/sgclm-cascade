"""
p2_analysis.py -- analysis layer over the P2 stationary campaign
(p2_results.jsonl from p2_stationary_campaign.py; log #25).

Produces:
  (1) the hardened theta(nu) table (seed-pooled, with z receipts) and the
      finite-nu deficit d(nu) = theta_G - theta(nu) under the exact anchor
      theta_G = 2017/2484 (Thm F), with power-law and log fits of d(nu)
      (reported for the measured range only -- no extrapolation claims);
  (2) the empirical (dagger-weak) statement WITH A RATE: rho_nu([-K,K])
      local decay slopes alpha_loc between adjacent nu (rho ~ nu^alpha);
      (dagger-weak) needs only o(1) -- any alpha > 0 is a strict margin;
  (3) the (star) phase-homogeneity verdict: E[Pi_HH]/E[Pi(K0)] vs nu --
      the (star)-consequence is mean-HH ~ 0; a growing share refutes (star)
      as a small-nu mechanism (pre-registered falsifier #3, SWING_RESULT.md).
"""
import json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "p2_results.jsonl")
THETA_G = 2017.0 / 2484.0

runs = [json.loads(l) for l in open(PATH) if l.strip()]
# pool the two seeds per (nu, dt) at production dt. Base pooling is restricted to
# campaign rows (tag v17/v18): receipt rows (dt/2, N-up, large-nu) can share
# (nu, dt) with a base point and must not contaminate the pooled values (v1.9 fix).
by = defaultdict(list)
receipts = []
for r in runs:
    if r.get("tag", "v18") in ("v17", "v18"):
        by[(r["nu"], r["dt"])].append(r)
    else:
        receipts.append(r)
prod = {}   # nu -> pooled dict, production dt = the one with 2 seeds (or first)
for (nu, dt), rs in sorted(by.items()):
    if len(rs) >= 2:
        prod[nu] = rs
    else:
        receipts.append(rs[0])

print(f"P2 analysis over {len(runs)} runs; anchor theta_G = 2017/2484 = {THETA_G:.6f}")
print("=" * 96)
print(f"{'nu':>6} | {'theta (pooled)':>16} | {'deficit':>8} | {'max|z|':>6} | "
      f"{'rho5/(B0/2)':>11} | {'HH share':>8} | {'anti':>6}")
nus, defs_, rho5s, hhshare = [], [], [], []
for nu in sorted(prod, reverse=True):
    rs = prod[nu]
    # v1.9: pooled theta = ratio of the seeds' combined sample means (the Table 1
    # caption's definition), n-weighted -- not the mean of per-seed ratios (the two
    # differ by 1 ulp at the printed precision at nu = 0.005, 0.01)
    th = (sum(r["I"] * r.get("n", 1) for r in rs)
          / sum(r["W1"] * r.get("n", 1) for r in rs))
    the = np.sqrt(sum(r["theta_se"] ** 2 for r in rs)) / len(rs)
    spread = abs(rs[0]["theta"] - rs[-1]["theta"]) / 2 if len(rs) > 1 else 0.0
    zmax = max(abs(r[q + "_z"]) for r in rs for q in ("I", "W1", "E1", "D"))
    rho5 = np.mean([r["rho5"] for r in rs])
    B0half = 0.08
    hh = np.mean([r["PiHH"] / (r["PiLH"] + r["PiHH"]) for r in rs])
    anti = max(r["anti_frac"] for r in rs)
    print(f"{nu:>6g} | {th:.4f} +- {max(the, spread):.4f} | {THETA_G-th:+.4f} | "
          f"{zmax:>6.2f} | {rho5/B0half:>11.3f} | {hh:>8.3f} | {anti:>6.4f}")
    nus.append(nu); defs_.append(THETA_G - th); rho5s.append(rho5); hhshare.append(hh)

nus = np.array(nus); defs_ = np.array(defs_); rho5s = np.array(rho5s)

print("-" * 96)
# deficit fits over the measured range
lp = np.polyfit(np.log(nus), np.log(defs_), 1)
resid_p = np.std(np.log(defs_) - np.polyval(lp, np.log(nus)))
ll = np.polyfit(np.log(nus), defs_, 1)
resid_l = np.std(defs_ - np.polyval(ll, np.log(nus)))
print(f"deficit d(nu) = theta_G - theta:  power fit d ~ nu^{lp[0]:+.3f} "
      f"(log-resid {resid_p:.3f});  log fit d = {ll[1]:.4f} {ll[0]:+.4f} ln(nu) "
      f"(resid {resid_l:.4f})")
print("  (fits are DESCRIPTIVE over the measured range; the small-nu limit of "
      "theta is not extrapolated)")

# rho5 local slopes (the (dagger-weak) rate)
print("rho_nu([-5,5]) local decay slopes alpha_loc (rho ~ nu^alpha between "
      "adjacent nu):")
for i in range(len(nus) - 1):
    a = np.log(rho5s[i] / rho5s[i + 1]) / np.log(nus[i] / nus[i + 1])
    print(f"  nu: {nus[i]:>5g} -> {nus[i+1]:<6g}  alpha_loc = {a:+.3f}")
print("  (dagger-weak) requires only rho_nu -> 0, i.e. o(1); a positive "
      "alpha_loc is a strict quantitative margin.")

# (star) verdict
print("(star) phase-homogeneity consequence (mean-HH kill): HH share of Pi(K0):")
print("  " + "  ".join(f"{nu:g}:{s:.3f}" for nu, s in zip(nus, hhshare)))
mono = all(hhshare[i] < hhshare[i + 1] for i in range(len(hhshare) - 1))
print(f"  monotone increasing as nu decreases: {mono} -> the (star)-consequence "
      f"DEGRADES toward small nu;\n  verdict: (star) REFUTED as a small-nu "
      f"mechanism (falsifier #3), consistent with SWING_RESULT's own caveat\n"
      f"  ('plausible only at large nu' -- where the cascade needs no help).")

for r in receipts:
    print(f"receipt run (dt={r['dt']:.1e}, nu={r['nu']}): theta={r['theta']:.4f}"
          f"+-{r['theta_se']:.4f}  max|z|="
          f"{max(abs(r[q+'_z']) for q in ('I','W1','E1','D')):.2f}")
print("=" * 96)
