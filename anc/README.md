# Ancillary verification package

Machine verification for "Production balances and a normalized production diagnostic for invariant
measures of the stochastic gCLM (a = −2)". The seven CAS certificates are an **independent**
symbolic-certification layer; the proofs are complete in the paper (`../paper/`). The solver
validator and the reproduction scripts pin the numerical identities and regenerate the campaign
data and figures.

Reference environment: Python 3.13.13, numpy 2.4.6, scipy 1.17.1, sympy 1.14.0 (see
`requirements.txt`).

    make verify             # 7 CAS certificates + 12-identity solver validator
    make verify-mutations   # corruption battery (each mutated certifier must be REJECTED)

`verify.py` runs each certifier and gates on its success verdict — several print a verdict but
always exit 0, so exit code alone is not a gate — and exits nonzero if any check fails.

## Contents

**Certification (the `make verify` set):**

- `certify_identities_symbolic.py` — every load-bearing algebraic identity as a zero polynomial
  over `ℚ[a]` with **symbolic** `a`, for band-limited fields (advective cancellation, `D𝒫₁`, the
  five-term `I(a)`, the `W₁/N₁` split, the palinstrophy secondary term).
- `certify_flux_law.py` — the spectral enstrophy-flux law and the secondary-term identity
  `∫ v_x ω ω_x = −½ ∫ (H ω_xx) ω²`.
- `certify_ito_vanishing.py` — the homogeneous-noise Itô injection `½ Tr(Q D²𝒫₁) = 0` (and its
  anisotropic non-vanishing).
- `certify_palinstrophy_mine.py` — the inertial closed form `I(a)`.
- `certify_projected_generator.py` — `I = I_M + R_M`, the support lemma (`R_M = 0` for `2K ≤ M`),
  and the `M = 1` example (`I = 15π/4 A⁴`, `W₁ = 9π/2 A⁴`, `I/W₁ = 5/6`, `I_M = 0`). Contains a
  built-in corrupted-decomposition check: an unprojected pairing is required to FAIL.
- `certify_theta_G_exact.py` — `θ_G = 2017/2484` by exact rational Wick sums, with a Monte-Carlo
  cross-check and a scale-invariance check.
- `certify_cubic_balance.py` — the telescoped, non-circular cubic reduction (the double-counting
  candidate is shown to fail; the telescoped one is certified).
- `validate.py` — 12 solver identities checked against the pseudo-spectral instrument `sgclm.py`.

**Corruption battery:**

- `mutation_tests.py` — applies one single-site mathematical mutation to each of a representative
  set of certifiers (a temporary in-place edit, restored in a `finally` block), re-runs it, and
  asserts the corrupted certifier is REJECTED (its success verdict disappears or it exits nonzero).
  A certifier that accepts a corrupted certificate is not a verifier. Representative — one probe
  per certifier surface class — not exhaustive.

**Solver and reproduction:**

- `sgclm.py` — the pseudo-spectral solver (strict `2/3` de-aliasing; exact-OU integrating-factor
  time stepping).
- `p2_campaign_v18.py`, `p2_analysis.py`, `p2_convergence_matrix.py`,
  `theta_large_nu_exact_ou.py`, `closure_balance_check.py` — the stationary campaign, its analysis,
  the resolution/step convergence matrix, the large-`ν` exact-OU grid, and the balance-closure
  check. They append to / read the committed data files below.
- `measure_theta.py`, `measure_cascade.py` — the independent cross-code re-measurement (a plain
  Euler–Maruyama solver, blind to the integrating-factor campaign) and the shared noise spec.
- `make_figures.py` — regenerates the five figures in `../paper/figs/`.

**Committed data:**

- `p2_results_v18.jsonl` — the stationary small-`ν` campaign (the primary dataset for Table 1 and
  Figure 2; two seeds per `ν`, plus `N`-up and `dt`-halving convergence rows).
- `p2_convergence_v18.jsonl` — the convergence matrix.
- `theta_large_nu_exact_ou.log`, `closure_balance_check.log` — the large-`ν` limit measurements and
  the balance-closure log.

**Integrity:** `SHA256SUMS` pins every file here (verify with `shasum -a 256 -c SHA256SUMS` on
macOS/BSD or `sha256sum -c SHA256SUMS` on GNU/Linux; the explicit `-a 256` avoids the SHA-1 default
of older `shasum`).
