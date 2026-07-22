"""
certify_flux_law.py  --  band-limited spectral certification of the ALGEBRAIC
pieces of Carrier A (theory/FLUX_LAW.md): the L0 enstrophy-balance inertial part
and the L1 palinstrophy-balance inertial part, for the stochastic gCLM at a=-2.

Self-contained (no solver import), by design an INDEPENDENT check of the
main-loop pen derivation. Band-limited omega on |k|<=K0 => all cubic products
have max wavenumber 3*K0 < Nyquist, so every integral is computed EXACTLY by the
periodic trapezoid rule (== spectral quadrature). Residuals ~ 1e-13 => certified.

Conventions (frozen, PROBLEM.md §1): torus [0,2pi); (H w)^(k) = -i sgn(k) w^(k);
v := H w; integral f g dx = (2pi/N) sum_i f_i g_i (exact here).
"""
import numpy as np

N = 96          # grid points; Nyquist = 48 > 3*K0
K0 = 6          # band limit of omega
TWO_PI = 2*np.pi
x = np.arange(N) * (TWO_PI / N)
k = np.fft.fftfreq(N, d=1.0/N)          # integer wavenumbers
sgn = np.sign(k)

def rand_field(seed):
    """Random real mean-zero field band-limited to 1<=|k|<=K0."""
    rng = np.random.default_rng(seed)
    ch = np.zeros(N, dtype=complex)
    for kk in range(1, K0+1):
        c = rng.standard_normal() + 1j*rng.standard_normal()
        ch[kk] = c
        ch[N-kk] = np.conj(c)           # reality
    w = np.fft.ifft(ch).real
    return w - w.mean()

def H(f):
    fh = np.fft.fft(f)
    return np.fft.ifft(-1j*sgn*fh).real

def dx(f, n=1):
    fh = np.fft.fft(f)
    return np.fft.ifft((1j*k)**n * fh).real

def ip(f, g):
    """integral_T f g dx (exact for band-limited below Nyquist)."""
    return (TWO_PI/N) * np.sum(f*g)

def velocity(w):
    """u = -Lambda^{-1} w : u^(k) = -w^(k)/|k| (k!=0)."""
    wh = np.fft.fft(w)
    uh = np.zeros(N, dtype=complex)
    nz = k != 0
    uh[nz] = -wh[nz]/np.abs(k[nz])
    return np.fft.ifft(uh).real

def nonlinear(w, a):
    """N[w] = -a u w_x + u_x w,  u_x = H w."""
    u = velocity(w)
    return -a*u*dx(w) + H(w)*w

def report(name, lhs, rhs):
    absdiff = abs(lhs-rhs)
    denom = max(abs(lhs), abs(rhs), 1e-300)
    rel = absdiff/denom
    # pass on small RELATIVE error, OR small ABSOLUTE error (handles the exact
    # a=-2 cancellation where both sides are ~machine-zero and rel is undefined).
    good = (rel < 1e-11) or (absdiff < 1e-12)
    status = "PASS" if good else "FAIL"
    tag = " [0=0 cancellation]" if (absdiff < 1e-12 and denom < 1e-10) else ""
    print(f"[{status}] {name:52s} lhs={lhs:+.6e} rhs={rhs:+.6e} rel={rel:.2e} abs={absdiff:.2e}{tag}")
    return good

ok = True
print("="*100)
print("Carrier A algebraic certification (band-limited, N=%d, K0=%d)" % (N, K0))
print("="*100)

for seed in range(5):
    w = rand_field(seed)
    v = H(w)                      # = u_x
    wx = dx(w); wxx = dx(w, 2)
    vx = H(wx)                    # = v_x = (H w)_x
    P0 = ip(v, w*w)               # enstrophy production integral (Hw)w^2

    # --- L0: inertial part <N[w], w> = (1+a/2) P0 ; = 0 at a=-2 --------------
    for a in (-2.0, -1.0, 0.0, 1.0):
        lhs = ip(nonlinear(w, a), w)
        rhs = (1.0 + a/2.0) * P0
        ok &= report(f"L0 seed{seed} a={a:+.1f}: <N,w>=(1+a/2)P0", lhs, rhs)

    # --- L1 at a=-2: inertial part <N[w], -w_xx> = 2 int v wx^2 + int vx w wx
    a = -2.0
    lhs = ip(nonlinear(w, a), -wxx)
    rhs = 2.0*ip(v, wx*wx) + ip(vx, w*wx)
    ok &= report(f"L1 seed{seed} a=-2: <N,-wxx>=2∫v wx²+∫vx w wx", lhs, rhs)

    # --- L1 secondary-term identity: int vx w wx = -1/2 int (H wxx) w^2 -------
    lhs2 = ip(vx, w*wx)
    rhs2 = -0.5*ip(H(wxx), w*w)
    ok &= report(f"L1 seed{seed}: ∫vx w wx = -½∫(Hwxx)w²", lhs2, rhs2)

print("="*100)
print("ALL CERTIFIED" if ok else "SOME CHECKS FAILED")
print("="*100)
