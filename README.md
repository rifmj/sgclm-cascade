# Production balances and a normalized production diagnostic for the stochastic gCLM (a = −2)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21492681.svg)](https://doi.org/10.5281/zenodo.21492681)

Paper and exact-verification package for

> R. Jumagulov, *Production balances and a normalized production diagnostic for invariant
> measures of the stochastic gCLM (a = −2): Galerkin-level identities and a large-viscosity
> limit* (2026).

**Setting.** The stochastically forced generalized Constantin–Lax–Majda–De Gregorio equation at
`a = −2`, the enstrophy-conserving case (Fujita–Fukuizumi–Sakajo). From the invariant-measure
identity `E_μ[𝓛F] = 0`, applied at each fixed Galerkin truncation, one obtains a family of exact
stationary balances.

**Results.**

- **Production balance & diagnostic.** For the palinstrophy production functional
  `𝒫₁ = ∫(Hω)ω_x²`, the exact generator contribution is the *projected* inertial pairing
  `I_M = ⟨P_M N, P_M D𝒫₁⟩` — not the full functional `I`, from which it differs by an explicit
  Galerkin defect `R_M` that vanishes exactly for band-limited fields. This defines the
  **normalized production diagnostic** `θ = 1 − E[N₁]/E[W₁]`.
- **Scale-blindness (Lemma).** `θ` is dilation-invariant while spectral localization is not, so
  no bound factoring through `θ` alone controls the low-mode dissipation.
- **Grouping-dependence.** `θ` is a diagnostic, not a canonical physical magnitude — only `E[I]`,
  `E[I_M] = −𝒞₁`, and the spectral flux are convention-free.
- **Large-viscosity limit (Theorem F, main result).** For every invariant measure of the Galerkin
  system, `θ → θ_G(shape) + O(ε)` (`M ≥ k_f`) and `θ_M → θ_G` with the balance coefficient positive
  (`M ≥ 2k_f`), where `ε = √B₀·ν^(−3/2)`; the exact Gaussian value for the flat band `[1,4]` is
  `θ_G = 2017/2484`.
- **Scope.** All identities and Theorem F are theorems at each fixed Galerkin truncation; the
  cubic/quintic production balances are formal for the untruncated SPDE.

> **Paper:** [`paper/paper.tex`](paper/paper.tex) → [`paper/paper.pdf`](paper/paper.pdf) (30 pp).

## Verification

The CAS certificates are an **independent** symbolic-certification layer; the proofs are complete
in the paper. The package ([`anc/`](anc/)) reproduces the certified identities, the exact Gaussian
constant, and the solver identities with one command:

    cd anc
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt        # pinned: numpy 2.4.6, scipy 1.17.1, sympy 1.14.0
    make verify                            # all seven CAS certificates + the 12-identity solver validator
    make verify-mutations                  # corruption battery: each mutated certifier must be REJECTED

`verify.py` gates on each certifier's success verdict (several print a verdict but always exit 0),
and exits nonzero if any check fails.

| script | certifies |
|---|---|
| `certify_identities_symbolic.py` | every load-bearing algebraic identity as a zero polynomial over `ℚ[a]` (symbolic `a`, band-limited fields) |
| `certify_flux_law.py` | the spectral enstrophy-flux law and the palinstrophy secondary-term identity |
| `certify_ito_vanishing.py` | the homogeneous-noise Itô injection vanishes (and is nonzero anisotropically) |
| `certify_palinstrophy_mine.py` | the inertial closed form `I(a)` |
| `certify_projected_generator.py` | `I = I_M + R_M`, the support lemma, and the `M = 1` example (`I/W₁ = 5/6`, `I_M = 0`); includes a built-in corrupted-decomposition trap |
| `certify_theta_G_exact.py` | `θ_G = 2017/2484` by exact rational Wick sums + a Monte-Carlo cross-check and a scale-invariance check |
| `certify_cubic_balance.py` | the telescoped, non-circular cubic reduction |
| `validate.py` | 12 solver identities against the pseudo-spectral instrument |

Reproduction scripts (`p2_campaign_v18.py`, `p2_analysis.py`, `p2_convergence_matrix.py`,
`theta_large_nu_exact_ou.py`, `measure_theta.py`, `make_figures.py`) regenerate the committed
campaign data (`p2_results_v18.jsonl`, `p2_convergence_v18.jsonl`, `*.log`) and the figures.

Integrity chain: the top-level `SHA256SUMS` pins the paper source and `anc/SHA256SUMS`, which pins
every ancillary file. Verify with `shasum -a 256 -c SHA256SUMS` (macOS/BSD) or
`sha256sum -c SHA256SUMS` (GNU/Linux).

## Provenance and disclosure

This work was carried out with substantial AI assistance in derivation, computation, and drafting.
Every result is backed by the proofs in the paper and the independent verification here; the
manuscript went through fourteen numbered rounds of independent review, including cross-model
audits. The author takes full responsibility for the results.

## License

MIT (see [`LICENSE`](LICENSE)). The model and its large-viscosity ergodic theory are due to
Fujita–Fukuizumi–Sakajo (arXiv:2603.06182).
