"""
sgclm.py -- pseudo-spectral solver for the stochastic generalized
Constantin-Lax-Majda-De Gregorio (gCLMG) equation on the 1D torus.

    d_t w + a u w_x - u_x w = nu w_xx + d_t xi,     u_x = H w,    a = -2 (default)

This module is pure numpy. All state is carried as the *real-FFT* half-spectrum
of a real, mean-zero, 2*pi-periodic field w(x), x in [0, 2*pi).

===========================================================================
CONVENTIONS (frozen -- read this before touching anything else in the file)
===========================================================================

Grid / DFT
----------
Physical grid: N points x_j = 2*pi*j/N, j = 0..N-1.  We use numpy's
`rfft`/`irfft`, i.e. numpy's own DFT normalization:

    what[k] = rfft(w)[k] = sum_j w(x_j) * exp(-i*2*pi*k*j/N),   k = 0..N//2

which is numpy's convention, NOT the "Fourier series coefficient" convention
w(x) = sum_k wtilde(k) e^{ikx}.  The relation between the two is

    wtilde(k) = what[k] / N        (for the DFT-index k = 0..N//2, k>=0)

and by reality wtilde(-k) = conj(wtilde(k)).  We keep the *rfft array*
`what` (numpy convention) as the actual in-memory state, because that is
what feeds `irfft`/`rfft` directly with no extra bookkeeping -- but every
physical quantity (Parseval sums, noise variance, etc.) is computed by
first converting to the Fourier-series coefficients wtilde = what / N.

Parseval (Fourier-series convention):
    ||f||_{L^2}^2 = int_0^{2pi} f(x)^2 dx = 2*pi * sum_{k=-inf}^{inf} |ftilde(k)|^2
                  = 2*pi * ( |ftilde(0)|^2 + 2*sum_{k=1}^{inf} |ftilde(k)|^2 )
using reality ftilde(-k)=conj(ftilde(k)).  Since w is mean-zero, ftilde(0)=0,
so for our fields:
    ||w||_{L^2}^2 = 4*pi * sum_{k=1}^{N/2} |wtilde(k)|^2   (k=N/2 Nyquist counted once,
                                                              handled by _parseval_weights)

Hilbert transform
------------------
    (H w)^(k) = -i * sgn(k) * w^(k),   sgn(0) = 0
This is applied identically to rfft arrays and Fourier-series coefficients
(it is a diagonal multiplier, invariant under the what = N*wtilde rescaling).

Velocity / u_x = H w
---------------------
    u = -Lambda^{-1} w,  Lambda = |D|  =>  utilde(k) = -wtilde(k)/|k|  (k != 0),
    utilde(0) = 0.
Then (u_x)^(k) = i*k*utilde(k) = i*k*(-wtilde(k)/|k|) = -i*sgn(k)*wtilde(k)
               = (H w)^(k).
So u_x = H w identically (verified both analytically above and numerically
in validate.py, check #1).

Dissipation
-----------
    nu * w_xx  <->  -nu * k^2 * w^(k)   (spectral multiplier, k=0 term is 0
                                          since w^(0)=0 always)

Nonlinear term
--------------
    N[w] := -a*u*w_x + u_x*w.   At a = -2:   N[w] = 2*u*w_x + (H w)*w.
Computed pseudo-spectrally (products in physical space) with the standard
2/3-rule dealiasing mask applied to every field that enters a product
(u, w_x, H w, w) *before* multiplying (removes all aliasing error in each
quadratic product), AND to the result (v1.7): the returned drift is the exact
Galerkin nonlinearity P_M N[P_M w] on E_M = {|k| <= N/3}, so with band-limited
noise and initial data the evolved state remains in E_M identically.

Noise
-----
    xi(t,x) = sum_{k in Z*} b_k * beta_k(t) * e_k(x)
"Homogeneous": b_k depends only on |k|.  We realize this directly in the
rfft/complex-mode representation used by the solver (this is mathematically
equivalent to the real sine/cosine-basis sum above; see the derivation
below and the cross-check in validate.py check #2/#5).

We want    E[ ||xi(t)||_{L^2}^2 ] / t = B0 := sum_{k in Z*} b_k^2.
(Band-limited default: b_k^2 = eps for k_lo <= |k| <= k_hi, else 0, so that
B0 = 2*(k_hi-k_lo+1)*eps, the factor 2 counting +k and -k.)

Derivation of the per-mode complex increment variance
-------------------------------------------------------
Represent the state as complex Fourier-series coefficients wtilde(k) for
k = 1..N/2 (k=0 fixed at 0 by mean-zero; wtilde(-k) := conj(wtilde(k)) is
implicit and never stored).  For each k>0 in the forcing band drive

    d wtilde(k) = (1/2) * b_k * dZ_k(t),      dZ_k = dZ_k^{(1)} + i*dZ_k^{(2)}

with dZ_k^{(1)}, dZ_k^{(2)} independent real white noises of unit intensity
(i.e. Z_k^{(1)}, Z_k^{(2)} independent standard Brownian motions), independent
across k.  This is exactly the complex-mode representation of the real sum
    xi(x) = sum_{k>0 in band} b_k * [ dBeta_k^{(c)}/dt * sqrt(2)*cos(kx)
                                      + dBeta_k^{(s)}/dt * sqrt(2)*sin(kx) ] / sqrt(2*pi)
(an orthonormal real L^2 basis), reparametrized into the complex mode
wtilde(k) = (1/2)*b_k*(Beta_k^{(c)} - i*Beta_k^{(s)})/sqrt(pi) ... -- rather
than carry that change of basis symbolically, we *define* the complex-mode
SDE by the increment variance it must have to reproduce B0 exactly, and then
verify the resulting physical-space injection rate numerically (validate.py
check #5 is the actual arbiter of correctness, per the task instructions).

Requiring E[||xi(t)||_{L^2}^2]/t = B0 with Parseval
    E[||xi(t)||_{L^2}^2] = 4*pi * sum_{k=1}^{N/2} E[|wtilde_xi(k,t)|^2]
and, for the increment above, E[|wtilde_xi(k,t)|^2] = E[|(1/2) b_k Z_k(t)|^2]
    = (1/4) b_k^2 * E[|Z_k(t)|^2] = (1/4) b_k^2 * 2t = (1/2) b_k^2 t
(since Z_k = Z_k^(1)+i Z_k^(2), E[(Z_k^(1))^2]=E[(Z_k^(2))^2]=t, cross term 0).
So
    E[||xi(t)||_{L^2}^2] = 4*pi * sum_{k=1}^{N/2} (1/2) b_k^2 t
                          = 2*pi*t * sum_{k=1}^{N/2} b_k^2.
For this to equal B0*t = t* sum_{k in Z*} b_k^2 = 2t*sum_{k=1}^{N/2} b_k^2
(homogeneous b_k, +-k pairs equal), we need the prefactor (1/2) above
replaced by 1/pi, i.e. the correct increment is

    d wtilde(k) = b_k * dZ_k(t) / sqrt(2*pi),      k>0 in band          (*)

which gives E[|wtilde_xi(k,t)|^2] = b_k^2 * 2t / (2*pi) = b_k^2 t / pi, and
    E[||xi||^2] = 4*pi * sum_k b_k^2 t/pi = 4*t*sum_{k=1}^{N/2} b_k^2 = 2*B0*t.
That is a factor 2 too big -- so instead we use HALF the rate above, i.e.

    d wtilde(k) = b_k * dZ_k(t) / (2*sqrt(pi)),    k>0 in band          (**)

giving E[|wtilde_xi(k)|^2] = b_k^2 * 2t / (4*pi) = b_k^2 t/(2*pi), and
    E[||xi||^2] = 4*pi * sum_k b_k^2 t/(2*pi) = 2*t*sum_{k=1}^{N/2} b_k^2 = B0*t.
Exact match.  This is the increment implemented below (Eq. **).

In code we work with the rfft array what = N*wtilde, so the increment on
what(k) (k=1..N/2, in the forcing band) per Euler-Maruyama step of size dt is

    delta_what(k) = N * b_k / (2*sqrt(pi)) * sqrt(dt) * (xi1 + i*xi2)

with xi1, xi2 ~ iid N(0,1) drawn fresh each step (Euler-Maruyama increment
dZ_k ~ sqrt(dt)*N(0,1) per real/imag component).  what(0) is never touched
(stays 0, mean-zero preserved exactly).  what at the Nyquist frequency
(k=N/2, purely real dof in rfft) is excluded from the complex-increment
noise (dealiased away in practice; b_k band default 1..4 does not reach it
for the default resolution).

This whole derivation is *validated numerically*, not just claimed: see
validate.py check #5, which measures nu*E[int w_x^2] against (1/2)*B0 over a
stationary run and reports the ratio (must -> 1).
"""

import numpy as np


# ---------------------------------------------------------------------------
# basic spectral utilities
# ---------------------------------------------------------------------------

def make_grid(N):
    """Return (x, k) for an N-point grid on [0, 2*pi).  k is the rfft
    wavenumber array, k = 0, 1, ..., N//2."""
    x = 2 * np.pi * np.arange(N) / N
    k = np.fft.rfftfreq(N, d=1.0 / N)  # = 0,1,...,N//2 (integers)
    return x, k


def dealias_mask(k, N):
    """STRICT 2/3-rule mask on the rfft wavenumber array k (0..N//2).

    v1.8 (referee round 5): the retained band is |k| <= M with M = (N-1)//3, i.e.
    3M <= N-1 strictly. The earlier floor(N/3) cutoff kept the boundary mode when
    3 | N (N = 192, 384), and the self-product of the boundary pair, 2M = 2N/3,
    aliases on the N-grid exactly onto the retained mode -M (sin(2Mx_j) = -sin(Mx_j)
    at x_j = 2 pi j / N) -- so the realized drift was not P_M N[P_M w] for those N.
    With 3M <= N-1 no product of retained modes aliases onto a retained mode."""
    cutoff = (N - 1) // 3
    return (k <= cutoff).astype(float)


def hilbert_mult(k):
    """Spectral multiplier of H on rfft array: -i*sgn(k), sgn(0)=0."""
    sgn = np.sign(k)
    return -1j * sgn


def apply_hilbert(what, k):
    return hilbert_mult(k) * what


def velocity_mult(k):
    """Spectral multiplier for u = -Lambda^{-1} w: -1/|k| (k!=0), 0 at k=0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(k == 0, 0.0, -1.0 / np.where(k == 0, 1.0, k))
    return m


def apply_velocity(what, k):
    return velocity_mult(k) * what


def ddx_mult(k):
    return 1j * k


def apply_ddx(what, k):
    return ddx_mult(k) * what


def to_fourier_series_coeffs(what, N):
    """rfft array (numpy convention) -> Fourier-series coefficients wtilde(k)."""
    return what / N


# ---------------------------------------------------------------------------
# nonlinear term
# ---------------------------------------------------------------------------

def nonlinear(what, k, N, a=-2.0, dealias=True):
    """
    Compute N[w]^(k) = (-a*u*w_x + u_x*w)^(k) pseudo-spectrally.

    Parameters
    ----------
    what : complex ndarray, shape (N//2+1,)
        rfft of w (numpy convention: what = rfft(w)).
    k : ndarray
        rfft wavenumber array from make_grid.
    N : int
        grid size.
    a : float
        advection parameter (default -2).
    dealias : bool
        apply 2/3-rule mask to factors entering products.

    Returns
    -------
    Nhat : complex ndarray, shape (N//2+1,)
        rfft of N[w].
    """
    mask = dealias_mask(k, N) if dealias else np.ones_like(k, dtype=float)

    uhat = apply_velocity(what, k) * mask
    wxhat = apply_ddx(what, k) * mask
    Hwhat = apply_hilbert(what, k) * mask
    what_m = what * mask

    u = np.fft.irfft(uhat, n=N)
    wx = np.fft.irfft(wxhat, n=N)
    Hw = np.fft.irfft(Hwhat, n=N)
    w = np.fft.irfft(what_m, n=N)

    N_phys = -a * u * wx + Hw * w
    # v1.7 (referee round 4): project the OUTPUT as well, so the drift is the genuine
    # Galerkin nonlinearity P_M N[P_M w] on E_M = {|k| <= N/3}. Previously only the
    # input factors were masked and modes in (N/3, N/2] leaked into (and persisted in)
    # the state; with band-limited noise and IC the state now stays in E_M exactly.
    Nhat = np.fft.rfft(N_phys) * mask
    return Nhat


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------

class NoiseSpec:
    """Homogeneous, band-limited additive noise spec.

    b_k^2 = eps for k_lo <= k <= k_hi (same magnitude at +-k), else 0.
    B0 = sum_{k in Z*} b_k^2 = 2*(k_hi-k_lo+1)*eps.
    """

    def __init__(self, k_lo=1, k_hi=4, eps=1.0):
        self.k_lo = k_lo
        self.k_hi = k_hi
        self.eps = eps

    def b2(self, k):
        """b_k^2 as a function of the rfft wavenumber array k (>=0 only)."""
        band = (k >= self.k_lo) & (k <= self.k_hi)
        return np.where(band, self.eps, 0.0)

    @property
    def B0(self):
        n_modes = self.k_hi - self.k_lo + 1
        return 2.0 * n_modes * self.eps


def noise_increment(k, N, noise: NoiseSpec, dt, rng):
    """
    Draw one Euler-Maruyama increment delta_what for the rfft state array,
    realizing the SDE increment (**) derived in the module docstring:

        delta_wtilde(k) = b_k/(2*sqrt(pi)) * sqrt(dt) * (xi1 + i*xi2),  k>0 in band
        delta_what(k)    = N * delta_wtilde(k)

    k=0 is always excluded (mean-zero preserved exactly). The Nyquist mode
    (k = N/2, if present) is also excluded: it is a single real dof in the
    rfft representation and is not part of the complex-mode band construction
    above; with the default band 1..4 well below Nyquist for any reasonable
    N this has no effect on B0 bookkeeping.
    """
    b2 = noise.b2(k)  # array over rfft k
    b = np.sqrt(b2)
    nyq = (N % 2 == 0) and (k[-1] == N // 2)
    active = b2 > 0
    if nyq:
        active = active.copy()
        active[-1] = False  # exclude Nyquist from complex-increment noise

    xi1 = rng.standard_normal(k.shape)
    xi2 = rng.standard_normal(k.shape)
    sqdt = np.sqrt(dt)

    d_wtilde = np.zeros(k.shape, dtype=complex)
    d_wtilde[active] = (b[active] / (2.0 * np.sqrt(np.pi))) * sqdt * (
        xi1[active] + 1j * xi2[active]
    )
    d_what = N * d_wtilde
    d_what[k == 0] = 0.0
    return d_what


# ---------------------------------------------------------------------------
# time stepping: integrating-factor (exponential) for viscosity,
# Euler-Maruyama for nonlinear drift + noise.
# ---------------------------------------------------------------------------
#
# Scheme (documented):
#   w_{n+1}^(k) = E(k) * [ what_n(k) + dt * Nhat_n(k) ] + dW^(k)
#   E(k) = exp(-nu*k^2*dt)
# i.e. the stiff linear viscous term is treated exactly via the integrating
# factor E(k) = exp(-nu k^2 dt) applied over the whole step (including to the
# nonlinear increment, a standard semi-implicit / ETD0-type discretization),
# while the nonlinear term uses a first-order (Euler) explicit update and the
# noise is added as a plain Euler-Maruyama increment (not exponentially
# weighted -- equivalent to placing the noise increment at the end of the
# step; the difference from weighting it is O(dt) in the strong error, i.e.
# consistent with the overall EM order 1/2).
#
# This treats the viscous term exactly / unconditionally stably for any dt,
# so dt is limited only by resolving the nonlinear term and noise, not by
# the diffusive CFL number nu*dt*k_max^2.

def step(what, k, N, dt, rng, nu=0.05, a=-2.0, noise: NoiseSpec = None,
         dealias=True):
    """Advance the state what by one step of size dt.  Returns new what."""
    Nhat = nonlinear(what, k, N, a=a, dealias=dealias)
    E = np.exp(-nu * k ** 2 * dt)
    what_new = E * (what + dt * Nhat)
    if noise is not None:
        what_new = what_new + noise_increment(k, N, noise, dt, rng)
    what_new[k == 0] = 0.0  # enforce mean-zero exactly
    if dealias:             # v1.7: state stays on E_M exactly (see nonlinear())
        what_new = what_new * dealias_mask(k, N)
    return what_new


def step_drift_only(what, k, N, dt, nu=0.0, a=-2.0, dealias=True,
                     viscous=True):
    """Deterministic-drift-only step (nonlinear + optional viscosity, no
    noise), same integrating-factor scheme.  Used for inviscid conservation
    checks and the finite-difference d/dt cross-checks."""
    Nhat = nonlinear(what, k, N, a=a, dealias=dealias)
    if viscous:
        E = np.exp(-nu * k ** 2 * dt)
    else:
        E = np.ones_like(k, dtype=float)
    what_new = E * (what + dt * Nhat)
    what_new[k == 0] = 0.0
    return what_new


# ---------------------------------------------------------------------------
# functionals
# ---------------------------------------------------------------------------

def _parseval_sum(what, k, N):
    """sum_{k=1}^{N/2} |wtilde(k)|^2 with correct weight for the Nyquist
    mode (rfft stores k=0 and, for even N, k=N/2 as single real-valued dof
    entries that should NOT be double counted the way k=1..N/2-1 are when
    reconstructing the full two-sided sum). Returns the *two-sided* sum
    sum_{k=-N/2}^{N/2} |wtilde(k)|^2 excluding k=0 (mean-zero), i.e.
    2*sum_{k=1}^{N/2-1}|wtilde(k)|^2 + |wtilde(N/2)|^2 (if N even).
    """
    wtilde = to_fourier_series_coeffs(what, N)
    mag2 = np.abs(wtilde) ** 2
    total = 2.0 * np.sum(mag2[1:])
    if N % 2 == 0:
        # k = N/2 term was double counted above; rfft only stores it once
        # and it is its own conjugate partner (self-paired), so subtract
        # back one copy.
        total -= mag2[-1]
    return total


def enstrophy(what, k, N):
    """E = 1/2 * int w^2 dx = 1/2 * 2*pi * sum_{k in Z} |wtilde(k)|^2
    (mean-zero => k=0 term absent)."""
    return 0.5 * 2 * np.pi * _parseval_sum(what, k, N)


def dissipation_integral(what, k, N):
    """int w_x^2 dx = 2*pi * sum_k k^2 |wtilde(k)|^2 (two-sided, k=0 absent)."""
    wtilde = to_fourier_series_coeffs(what, N)
    mag2 = (k.astype(float) ** 2) * np.abs(wtilde) ** 2
    total = 2.0 * np.sum(mag2[1:])
    if N % 2 == 0:
        total -= mag2[-1]
    return 2 * np.pi * total


def dissipation(what, k, N, nu):
    """D = nu * int w_x^2 dx."""
    return nu * dissipation_integral(what, k, N)


def production(what, k, N, dealias=True):
    """P = int (H w) w^2 dx, physical-space product, dealiased (2/3 rule on
    the two dealiased factors Hw and w before forming w^2*Hw would still
    alias for a cubic product; here we simply dealias the two spectral
    factors that build the cubic and accept the residual truncation is at
    the resolved-mode level -- standard practice for diagnostic cubic
    functionals; see README for the discussion)."""
    mask = dealias_mask(k, N) if dealias else np.ones_like(k, dtype=float)
    Hwhat = apply_hilbert(what, k) * mask
    what_m = what * mask
    Hw = np.fft.irfft(Hwhat, n=N)
    w = np.fft.irfft(what_m, n=N)
    phys = Hw * w ** 2
    Lx = 2 * np.pi
    dx = Lx / N
    return np.sum(phys) * dx


def dproduction_field(what, k, N, dealias=True):
    """D P = 2*w*(H w) - H(w^2), the first-variation (functional-derivative)
    field of P[w] = int (Hw) w^2 dx with respect to w, returned as an rfft
    array (so it can be dotted with a drift field spectrally via Parseval).

    Derivation: P[w] = int (Hw) w^2. For a perturbation w -> w+eps*h,
       d/deps P|_0 = int (Hh) w^2 + int (Hw) 2 w h
                    = int h * H^T(w^2) + int h * 2*w*(Hw)     [H^T = -H, self-adjoint
                      up to sign: int (Hf) g = -int f (Hg) since H is
                      skew-adjoint on mean-zero L^2]
    so int (Hh) w^2 = int h * (-H(w^2)).  Hence
       d/deps P|_0 = int h * [ 2*w*(Hw) - H(w^2) ] = int h * DP,
       DP = 2*w*(Hw) - H(w^2).
    """
    mask = dealias_mask(k, N) if dealias else np.ones_like(k, dtype=float)
    Hwhat = apply_hilbert(what, k) * mask
    what_m = what * mask
    Hw = np.fft.irfft(Hwhat, n=N)
    w = np.fft.irfft(what_m, n=N)

    w2 = w ** 2
    w2hat = np.fft.rfft(w2) * mask
    Hw2hat = apply_hilbert(w2hat, k)
    Hw2 = np.fft.irfft(Hw2hat, n=N)

    DP_phys = 2 * w * Hw - Hw2
    return np.fft.rfft(DP_phys)


def l2_pairing(fhat, ghat, k, N):
    """<f,g>_{L^2} = int f*g dx = 2*pi*sum_{k in Z} ftilde(k)*conj(gtilde(k))
    (two-sided, real result for real f,g), computed from rfft arrays."""
    ftilde = to_fourier_series_coeffs(fhat, N)
    gtilde = to_fourier_series_coeffs(ghat, N)
    prod = ftilde * np.conj(gtilde)
    total = 2.0 * np.sum(prod[1:].real)
    if N % 2 == 0:
        total -= prod[-1].real
    total += (prod[0]).real  # k=0 term, generally 0 for mean-zero fields
    return 2 * np.pi * total


def ddt_along_flow(DFhat, drift_hat, k, N):
    """Exact chain rule: d/dt F(w(t)) = < drift, DF > for w evolving by
    dw/dt = drift.  Generic given the first-variation field DF (as rfft
    array) and the drift field (as rfft array)."""
    return l2_pairing(drift_hat, DFhat, k, N)


# ---------------------------------------------------------------------------
# palinstrophy functionals, spectral enstrophy flux
# ---------------------------------------------------------------------------
#
# Palinstrophy   E1 = 1/2 int w_x^2 dx  (the "next" quadratic invariant-like
#   functional up the derivative ladder; NOT conserved even at a=-2, since
#   only enstrophy is the inviscid invariant there).
# Palinstrophy production   P1 = int (Hw) w_x^2 dx   (physical-space cubic
#   product, dealiased -- the w_x^2-weighted stretching-rate functional,
#   the direct analog of `production` one derivative up).
# Palinstrophy dissipation  D1 = nu int w_xx^2 dx = nu * 2*pi * sum k^4 |wtilde(k)|^2.
#
# Spectral enstrophy flux Pi(K): mean rate of enstrophy transfer from
# {|k|<K} to {|k|>=K} by the (deterministic) nonlinearity N[w].  Standard
# definition via the sharp low-pass projection P_{<K} (zero out |k|>=K in
# Fourier space):
#     Pi(K) := - < P_{<K} w , P_{<K}( N[w] ) >_{L^2}
# Since P_{<K} is an orthogonal projection (diagonal, real, idempotent) in
# the Fourier basis, it is self-adjoint, so equivalently
# Pi(K) = -<P_{<K} w, N[w]> = -<w, P_{<K} N[w]>; we compute it exactly as
# written (project both factors) so the formula matches the definition
# literally with no reliance on that identity.  Sign convention: Pi(K)>0 is
# forward transfer (out of {|k|<K}, i.e. into {|k|>=K}) at rate Pi(K), i.e.
# the same sign convention as the classical 3D turbulence energy flux.


def palinstrophy(what, k, N):
    """E1 = 1/2 * int w_x^2 dx = 1/2 * 2*pi * sum_k k^2 |wtilde(k)|^2
    (two-sided, mean-zero => k=0 absent).  Equals 1/2 * dissipation_integral."""
    return 0.5 * dissipation_integral(what, k, N)


def palinstrophy_production(what, k, N, dealias=True):
    """P1 = int (H w) w_x^2 dx, physical-space product, dealiased (2/3-rule
    mask applied to the spectral factors Hw and w_x before forming the
    physical-space cubic product w_x^2*(Hw), same convention as
    `production`)."""
    mask = dealias_mask(k, N) if dealias else np.ones_like(k, dtype=float)
    Hwhat = apply_hilbert(what, k) * mask
    wxhat = apply_ddx(what, k) * mask
    Hw = np.fft.irfft(Hwhat, n=N)
    wx = np.fft.irfft(wxhat, n=N)
    phys = Hw * wx ** 2
    Lx = 2 * np.pi
    dx = Lx / N
    return np.sum(phys) * dx


def palinstrophy_dissipation_integral(what, k, N):
    """int w_xx^2 dx = 2*pi * sum_k k^4 |wtilde(k)|^2 (two-sided, k=0 absent)."""
    wtilde = to_fourier_series_coeffs(what, N)
    mag2 = (k.astype(float) ** 4) * np.abs(wtilde) ** 2
    total = 2.0 * np.sum(mag2[1:])
    if N % 2 == 0:
        total -= mag2[-1]
    return 2 * np.pi * total


def palinstrophy_dissipation(what, k, N, nu):
    """D1 = nu * int w_xx^2 dx."""
    return nu * palinstrophy_dissipation_integral(what, k, N)


def lowpass_mask(k, K):
    """Sharp projector mask P_{<K}: 1 for |k| < K, 0 for |k| >= K.  k is the
    rfft wavenumber array (>=0 only; the |.| is a no-op there but written
    for clarity / future-proofing against signed-k arrays)."""
    return (np.abs(k) < K).astype(float)


def apply_lowpass(what, k, K):
    """Apply the sharp low-pass projector P_{<K} to an rfft array."""
    return what * lowpass_mask(k, K)


def spectral_flux(what, k, N, K, a=-2.0, dealias=True):
    """Spectral enstrophy flux through wavenumber K:

        Pi(K) = - < P_{<K} w , P_{<K}( N[w] ) >_{L^2}

    where N[w] is the deterministic nonlinear term (nonlinear(...) above),
    computed with the solver's own dealiasing convention.  Pi(K) > 0 means
    forward (large-scale -> small-scale) enstrophy transfer, i.e. the mean
    rate at which the nonlinearity moves enstrophy out of {|k|<K} into
    {|k|>=K}.

    K may be a scalar or array; if array, this function should be called
    once per K value (kept scalar-K for clarity, see spectral_flux_curve
    for the vectorized sweep over a K-array)."""
    Nhat = nonlinear(what, k, N, a=a, dealias=dealias)
    w_lo = apply_lowpass(what, k, K)
    N_lo = apply_lowpass(Nhat, k, K)
    return -l2_pairing(w_lo, N_lo, k, N)


def spectral_flux_curve(what, k, N, K_values, a=-2.0, dealias=True):
    """Vectorized convenience wrapper: Pi(K) for each K in K_values (1D
    array-like).  Returns an ndarray of the same length.  Recomputes N[w]
    once and reuses it across all K (the low-pass projection is cheap;
    N[w] is the expensive pseudo-spectral evaluation)."""
    Nhat = nonlinear(what, k, N, a=a, dealias=dealias)
    K_values = np.asarray(K_values, dtype=float)
    out = np.empty(K_values.shape, dtype=float)
    for i, K in enumerate(K_values):
        w_lo = apply_lowpass(what, k, K)
        N_lo = apply_lowpass(Nhat, k, K)
        out[i] = -l2_pairing(w_lo, N_lo, k, N)
    return out


# ---------------------------------------------------------------------------
# initial conditions
# ---------------------------------------------------------------------------

def random_band_limited_ic(N, k, rng, k_lo=1, k_hi=8, amp=1.0):
    """Random real, mean-zero, band-limited initial condition as an rfft
    array."""
    what = np.zeros(k.shape, dtype=complex)
    band = (k >= k_lo) & (k <= k_hi)
    n = np.count_nonzero(band)
    re = rng.standard_normal(n)
    im = rng.standard_normal(n)
    what[band] = amp * (re + 1j * im)
    what[k == 0] = 0.0
    return what


# ---------------------------------------------------------------------------
# run loop
# ---------------------------------------------------------------------------

def run(N, T, dt, nu, noise: NoiseSpec, rng, a=-2.0, what0=None,
        burn_in_frac=0.2, record_every=1, dealias=True,
        record_palinstrophy=False, flux_K_values=None):
    """
    Integrate the SPDE from t=0 to t=T with step dt, recording functionals.

    Returns a dict with time series (post burn-in and full) and stationary
    time-averages.

    Additional (opt-in, backward-compatible) recording:
      record_palinstrophy : if True, also record palinstrophy E1, its
        production P1, and its dissipation D1 at every recorded step.
      flux_K_values : if not None, an array-like of K values; at every
        recorded step also compute the spectral enstrophy flux curve
        Pi(K) for each K (via spectral_flux_curve) and accumulate its
        time-average (the full per-step curve is not stored, only the
        running mean, to keep memory flat for long runs / K-sweeps).
    """
    x, k = make_grid(N)
    if what0 is None:
        what0 = random_band_limited_ic(N, k, rng, k_lo=1, k_hi=8, amp=0.5)
    what = what0.copy()

    n_steps = int(round(T / dt))
    burn_steps = int(round(burn_in_frac * n_steps))

    times = []
    E_series = []
    D_series = []
    P_series = []
    E1_series = [] if record_palinstrophy else None
    P1_series = [] if record_palinstrophy else None
    D1_series = [] if record_palinstrophy else None

    flux_K_values = np.asarray(flux_K_values, dtype=float) if flux_K_values is not None else None
    flux_sum = np.zeros_like(flux_K_values) if flux_K_values is not None else None
    flux_sumsq = np.zeros_like(flux_K_values) if flux_K_values is not None else None
    flux_n = 0

    for n in range(n_steps):
        what = step(what, k, N, dt, rng, nu=nu, a=a, noise=noise,
                    dealias=dealias)
        if n % record_every == 0:
            t = (n + 1) * dt
            is_stationary = t > burn_in_frac * T
            times.append(t)
            E_series.append(enstrophy(what, k, N))
            D_series.append(dissipation(what, k, N, nu))
            P_series.append(production(what, k, N, dealias=dealias))
            if record_palinstrophy:
                E1_series.append(palinstrophy(what, k, N))
                P1_series.append(palinstrophy_production(what, k, N, dealias=dealias))
                D1_series.append(palinstrophy_dissipation(what, k, N, nu))
            if flux_K_values is not None and is_stationary:
                curve = spectral_flux_curve(what, k, N, flux_K_values, a=a,
                                             dealias=dealias)
                flux_sum += curve
                flux_sumsq += curve ** 2
                flux_n += 1

    times = np.array(times)
    E_series = np.array(E_series)
    D_series = np.array(D_series)
    P_series = np.array(P_series)

    keep = times > burn_in_frac * T

    out = {
        "what_final": what,
        "k": k,
        "times": times,
        "enstrophy": E_series,
        "dissipation": D_series,
        "production": P_series,
        "mean_enstrophy": np.mean(E_series[keep]) if np.any(keep) else np.nan,
        "mean_dissipation": np.mean(D_series[keep]) if np.any(keep) else np.nan,
        "mean_production": np.mean(P_series[keep]) if np.any(keep) else np.nan,
        "burn_in_frac": burn_in_frac,
    }

    if record_palinstrophy:
        E1_series = np.array(E1_series)
        P1_series = np.array(P1_series)
        D1_series = np.array(D1_series)
        out["palinstrophy"] = E1_series
        out["palinstrophy_production"] = P1_series
        out["palinstrophy_dissipation"] = D1_series
        out["mean_palinstrophy"] = np.mean(E1_series[keep]) if np.any(keep) else np.nan
        out["mean_palinstrophy_production"] = np.mean(P1_series[keep]) if np.any(keep) else np.nan
        out["mean_palinstrophy_dissipation"] = np.mean(D1_series[keep]) if np.any(keep) else np.nan

    if flux_K_values is not None:
        out["flux_K_values"] = flux_K_values
        if flux_n > 0:
            mean_flux = flux_sum / flux_n
            # population std of the per-sample curve (time-correlated samples,
            # so this is a rough dispersion measure, not an independent-sample
            # std err; measure_cascade.py additionally uses ensemble spread
            # across independent seeds for honest error bars).
            var_flux = np.maximum(flux_sumsq / flux_n - mean_flux ** 2, 0.0)
            out["mean_flux"] = mean_flux
            out["std_flux"] = np.sqrt(var_flux)
        else:
            out["mean_flux"] = np.full_like(flux_K_values, np.nan)
            out["std_flux"] = np.full_like(flux_K_values, np.nan)
        out["flux_n_samples"] = flux_n

    return out
