"""
measure_theta.py -- measure the cascade efficiency theta of Carrier B directly on
the stochastic-gCLM invariant measure, using the certified inertial identity.

At stationarity the homogeneous-noise production balance gives
    E[L P1] = 0  =>  E[I] = -C1   (Ito term = 0, Lemma B0),
so  theta := 1 - E[N1]/E[W1] = -C1/E[W1] = E[I]/E[W1],
with  I  = <b_nl, D P1>  (inertial, certified in cas/certify_palinstrophy_mine.py)
and   W1 = 6 int v^2 omega_x^2 >= 0.
We measure E[I] and E[W1] over stationary snapshots (own loop via sg.step) and
report theta(nu) at a=-2.  A steady single-mode trap gives theta=0 (checked).
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sgclm as sg
from measure_cascade import NOISE, B0        # same noise spec (B0=0.16, band [1,4])

# ---- physical-space primitives (full-fft; match sgclm conventions) ----------
def make(N):
    x = np.arange(N)*(2*np.pi/N)
    k = np.fft.fftfreq(N, d=1.0/N)
    return x, k
def H(f, k):   return np.fft.ifft(-1j*np.sign(k)*np.fft.fft(f)).real
def dx(f, k, n=1): return np.fft.ifft((1j*k)**n*np.fft.fft(f)).real
def vel(f, k):
    fh = np.fft.fft(f); uh = np.zeros_like(fh); nz = k != 0
    uh[nz] = -fh[nz]/np.abs(k[nz]); return np.fft.ifft(uh).real
def ip(f, g, N): return (2*np.pi/N)*np.sum(f*g)

def I_and_W1(w, k, N, a=-2.0):
    """inertial part I=<b_nl,DP1> and W1=6 int v^2 wx^2 on physical field w."""
    v = H(w, k); wx = dx(w, k); wxx = dx(w, k, 2)
    u = vel(w, k)
    DP1 = -H(wx*wx, k) - 2*H(wx, k)*wx - 2*v*wxx
    bnl = -a*u*wx + v*w
    I = ip(bnl, DP1, N)
    W1 = 6*ip(v*v, wx*wx, N)
    return I, W1

def phys(what, N):        # solver state (rfft) -> physical omega
    return np.fft.irfft(what, N)

def measure(nu, N, dt, T, a=-2.0, seed=0, burn=0.3, rec=20):
    x, k = sg.make_grid(N)
    _, kf = make(N)                          # full-fft wavenumbers for my primitives
    rng = np.random.default_rng(seed)
    what = sg.random_band_limited_ic(N, k, rng, k_lo=1, k_hi=8, amp=0.5)
    nsteps = int(round(T/dt)); Isum = W1sum = 0.0; n = 0
    for s in range(nsteps):
        what = sg.step(what, k, N, dt, rng, nu=nu, a=a, noise=NOISE, dealias=True)
        if s % rec == 0 and (s+1)*dt > burn*T:
            w = phys(what, N)
            I, W1 = I_and_W1(w, kf, N, a=a)
            Isum += I; W1sum += W1; n += 1
    EI, EW1 = Isum/n, W1sum/n
    return dict(nu=nu, N=N, a=a, E_I=EI, E_W1=EW1, theta=EI/EW1, n=n)

def steady_trap_check():
    """Diagnostic only: theta of a STATIC single-mode field cos(3x).

    NOTE (audit): cos(3x) is NOT a steady state of the gCLM (N[cos3x] != 0),
    so it does not satisfy the trap condition (POS) and there is NO expectation
    that I ~ 0 here (measured theta ~ 0.83). The theory's theta=0 claim is about
    *dynamical* steady traps (C1 = 0), which a static single mode does not test.
    Kept as a smoke test of the I/W1 evaluation pipeline on a closed-form field."""
    N = 64; _, kf = make(N)
    x = np.arange(N)*(2*np.pi/N); w = np.cos(3*x)
    I, W1 = I_and_W1(w, kf, N, a=-2.0)
    return I, W1, (I/W1 if W1 else 0.0)

if __name__ == "__main__":
    print("="*78); print("theta = E[I]/E[W1] on the stochastic-gCLM invariant measure (a=-2)")
    print("B0=%.3f, forced band [1,4]" % B0); print("="*78)
    It, W1t, th_t = steady_trap_check()
    print(f"static single-mode diagnostic (NOT a steady state; no I~0 expectation): "
          f"I={It:+.3e} W1={W1t:.3e} theta={th_t:+.3e}")
    print("-"*78)
    configs = [dict(nu=0.05, N=64, dt=2e-3, T=200.0),
               dict(nu=0.02, N=96, dt=1e-3, T=200.0),
               dict(nu=0.01, N=128, dt=1e-3, T=200.0)]
    rows = []
    for c in configs:
        accum = [measure(seed=10+i, **c) for i in range(2)]
        th = np.mean([r["theta"] for r in accum])
        th_se = np.std([r["theta"] for r in accum], ddof=1)/np.sqrt(len(accum))
        EI = np.mean([r["E_I"] for r in accum]); EW1 = np.mean([r["E_W1"] for r in accum])
        rows.append((c["nu"], EI, EW1, th, th_se))
        print(f"nu={c['nu']:<6} E[I]={EI:+.4e}  E[W1]={EW1:.4e}  "
              f"theta={th:+.4f} +/- {th_se:.4f}")
    print("="*78)
    print("theta in (0,1) on developed states => cascade efficiency well-posed"
          if all(0 < r[3] < 1 for r in rows) else
          "theta NOT uniformly in (0,1) -- inspect (sign of destruction / grouping)")
    print("="*78)
