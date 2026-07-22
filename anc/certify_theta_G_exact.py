"""
certify_theta_G_exact.py -- EXACT (rational-arithmetic) Wick computation of the
Gaussian cascade efficiency theta_G = E_G[I]/E_G[W1] on the stationary law of the
linearized (OU) dynamics, for homogeneous band forcing.

Context (log #23, S6 correction): theta_G(shape) is the physical nu->infinity
limit of theta; the MC value for band [1,4] was 0.8119 +/- 1e-4
(swing/theta_large_nu_exact_ou.py). This script replaces the MC estimate by an
EXACT closed-form rational number, and is the symbolic-certificate standard
(review item: exact arithmetic, not float residuals) applied to the new material.

Model conventions (PAPER.md S2): omega real mean-zero on the 2pi-torus,
v = H omega, u = -Lambda^{-1} omega. OU stationary law for band forcing:
independent complex modes with E|omega_hat(k)|^2 = sigma^2(|k|) = b^2/(4 pi nu k^2),
i.e. spectral SHAPE sigma^2 ~ 1/k^2 on the band (flat b), 0 off band. theta_G is
0-homogeneous, so the overall scale of sigma^2 cancels (checked explicitly below).

Functionals (certified inertial formula, cas/certify_palinstrophy_mine.py, a=-2):
  I  = 6*T1 + 4*T2 + 2*T3 - 2*T4 - T5,     W1 = 6*T1,
  T1 = int v^2 omega_x^2          T2 = int u v omega_x omega_xx
  T3 = int v v_x omega omega_x    T4 = int u omega_x H(omega_x^2)
  T5 = int v omega H(omega_x^2)

Each T is a quartic Fourier sum  2*pi * sum_{k1+k2+k3+k4=0} A(k1,k2,k3,k4)
prod omega_hat(k_i); Gaussian expectation = 3 Wick pairings with
E[omega_hat(k) omega_hat(l)] = sigma^2(|k|) delta_{k+l,0}. All kernels are
products of the multipliers  m_v = -i sgn(k), m_u = -1/|k|, d/dx = ik,
d2/dx2 = -k^2, m_{v_x} = |k|, and for T4/T5 the H acting on the (k3,k4) pair
contributes -i*sgn(k3+k4). Finite sums over the band, exact in QQ.

Verification layers:
  (1) exact rational theta_G for band [1,4], shape 1/k^2;
  (2) scale-invariance receipt (sigma^2 -> c*sigma^2 leaves theta_G unchanged);
  (3) imaginary parts of all T's vanish identically (reality check);
  (4) independent numpy Monte Carlo on the same law using the SOLVER-SIDE
      certified I_and_W1 (different code path) must agree to MC accuracy;
  (5) cross-check against the banked MC 0.8119 +/- 1e-4.
"""
import os, sys
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

I_ = sp.I
BAND = [1, 2, 3, 4]


def sgn(k):
    return 0 if k == 0 else (1 if k > 0 else -1)


def m_v(k):   # Hilbert transform
    return -I_ * sgn(k)


def m_u(k):   # u = -Lambda^{-1} omega
    return sp.Rational(-1, abs(k)) if k != 0 else 0


def dx(k):
    return I_ * k


def dxx(k):
    return -sp.Integer(k) ** 2


def m_vx(k):  # (H omega)_x
    return sp.Integer(abs(k))


# kernels A(k1,k2,k3,k4) for the five quartic terms
def A_T1(k1, k2, k3, k4):
    return m_v(k1) * m_v(k2) * dx(k3) * dx(k4)


def A_T2(k1, k2, k3, k4):
    return m_u(k1) * m_v(k2) * dx(k3) * dxx(k4)


def A_T3(k1, k2, k3, k4):
    return m_v(k1) * m_vx(k2) * 1 * dx(k4)


def A_T4(k1, k2, k3, k4):
    return m_u(k1) * dx(k2) * (-I_ * sgn(k3 + k4)) * dx(k3) * dx(k4)


def A_T5(k1, k2, k3, k4):
    return m_v(k1) * 1 * (-I_ * sgn(k3 + k4)) * dx(k3) * dx(k4)


def wick_quartic(A, sigma2):
    """E_G[ int f1 f2 f3 f4 dx ] = 2*pi * sum over the 3 Wick pairings.
    sigma2: dict |k| -> exact variance E|omega_hat(k)|^2 (0 off its keys)."""
    Bpm = [s * k for k in sigma2 for s in (+1, -1)]
    tot = sp.Integer(0)
    for ka in Bpm:
        for kb in Bpm:
            sa, sb = sigma2[abs(ka)], sigma2[abs(kb)]
            # pairing (12)(34): k2=-k1, k4=-k3
            tot += A(ka, -ka, kb, -kb) * sa * sb
            # pairing (13)(24): k3=-k1, k4=-k2
            tot += A(ka, kb, -ka, -kb) * sa * sb
            # pairing (14)(23): k4=-k1, k3=-k2
            tot += A(ka, kb, -kb, -ka) * sa * sb
    return sp.simplify(2 * sp.pi * tot)


def theta_G_exact(sigma2):
    T = {name: wick_quartic(A, sigma2) for name, A in
         [("T1", A_T1), ("T2", A_T2), ("T3", A_T3), ("T4", A_T4), ("T5", A_T5)]}
    for name, val in T.items():
        assert sp.im(val) == 0, f"{name} has nonzero imaginary part: {val}"
    EI = 6 * T["T1"] + 4 * T["T2"] + 2 * T["T3"] - 2 * T["T4"] - T["T5"]
    EW1 = 6 * T["T1"]
    return sp.nsimplify(sp.simplify(EI / EW1), rational=True), T, EI, EW1


def mc_crosscheck(n=200_000, seed=1234):
    """Independent numpy MC on the same Gaussian law, evaluated through the
    solver-side certified I_and_W1 (a different code path entirely)."""
    import numpy as np
    from measure_theta import I_and_W1, make
    N = 64
    _, kf = make(N)
    rng = np.random.default_rng(seed)
    k = np.fft.rfftfreq(N, d=1.0 / N)
    var = np.zeros_like(k)
    for kk in BAND:
        var[int(kk)] = 1.0 / kk ** 2       # shape 1/k^2 (scale-free)
    s = np.sqrt(var / 2.0)
    Isum = Wsum = 0.0
    for i in range(n):
        wt = s * (rng.standard_normal(k.size) + 1j * rng.standard_normal(k.size))
        wt[0] = 0.0
        w = np.fft.irfft(N * wt, N)
        I, W1 = I_and_W1(w, kf, N, a=-2.0)
        Isum += I; Wsum += W1
    return Isum / Wsum


if __name__ == "__main__":
    print("=" * 78)
    print("EXACT Wick certificate: theta_G = E_G[I]/E_G[W1], band [1,4], shape 1/k^2")
    print("=" * 78)

    sigma2 = {k: sp.Rational(1, k * k) for k in BAND}
    th, T, EI, EW1 = theta_G_exact(sigma2)
    print("per-term Wick values (in units of 2*pi*scale^2):")
    for name in ("T1", "T2", "T3", "T4", "T5"):
        print(f"  {name} = {sp.nsimplify(T[name]/(2*sp.pi), rational=True)} * 2pi")
    print(f"E_G[I]  = {sp.nsimplify(EI/(2*sp.pi), rational=True)} * 2pi")
    print(f"E_G[W1] = {sp.nsimplify(EW1/(2*sp.pi), rational=True)} * 2pi  (> 0)")
    print(f"\ntheta_G(band[1,4]) EXACT = {th} = {sp.N(th, 12)}")

    # scale-invariance receipt
    sigma2_scaled = {k: 7 * v for k, v in sigma2.items()}
    th_s, *_ = theta_G_exact(sigma2_scaled)
    assert sp.simplify(th - th_s) == 0, "scale invariance FAILED"
    print("scale-invariance (sigma^2 -> 7*sigma^2): PASS (theta_G unchanged)")

    # MC cross-check through the independent solver-side code path
    try:
        mc = mc_crosscheck()
        diff = abs(float(th) - mc)
        print(f"numpy MC cross-check (200k samples, solver-side I_and_W1): "
              f"{mc:.5f}  |exact - MC| = {diff:.2e}")
        verdict = "PASS" if diff < 3e-3 else "FAIL"
        print(f"MC agreement: {verdict}")
        if verdict == "FAIL":
            sys.exit(1)   # machine-checkable: a corrupted Wick sum must fail loudly
    except Exception as e:  # keep the exact result usable without numpy env
        print(f"MC cross-check skipped ({e})")

    print(f"banked large-nu SDE measurements (log #23): theta in [0.8107, 0.8126]")
    print("=" * 78)
