"""
certify_palinstrophy_mine.py -- INDEPENDENT (main-loop, dual-blind) certification
of the inertial part of the palinstrophy production balance E_mu[L P1]=0, for the
stochastic gCLM. Anchors the sgclm-palinstrophy-theta workflow.

P1 = int (H w) w_x^2.   v := H w,  vx := H w_x,  u := -Lambda^{-1} w (u_x = v).
First variation:  D P1 = -H(w_x^2) - 2 (H w_x) w_x - 2 (H w) w_xx.
Drift b_nl(a) = -a u w_x + v w.  Inertial part  I(a) = <b_nl, D P1>.

Hand-derived formula (this file certifies it):
  I(a) = (2-2a) int v^2 w_x^2            [leading square, = 6 int v^2 w_x^2 >=0 at a=-2]
         - 2a int u v w_x w_xx
         + 2   int v vx w w_x
         + a   int u w_x H(w_x^2)
         -     int v w H(w_x^2).
Viscous cross-term V = <w_xx, D P1> (destruction candidate); Ito term = 0 (homogeneous).
Stationary balance:  E[I] = -nu E[V].  Split I = W1 - N1 with W1 = (2-2a) int v^2 w_x^2 >= 0.
"""
import numpy as np

N = 96
K0 = 6
TWO_PI = 2*np.pi
x = np.arange(N)*(TWO_PI/N)
k = np.fft.fftfreq(N, d=1.0/N)
sgn = np.sign(k)

def H(f):  return np.fft.ifft(-1j*sgn*np.fft.fft(f)).real
def dx(f, n=1): return np.fft.ifft((1j*k)**n*np.fft.fft(f)).real
def ip(f, g): return (TWO_PI/N)*np.sum(f*g)
def vel(w):
    wh = np.fft.fft(w); uh = np.zeros(N, complex); nz = k != 0
    uh[nz] = -wh[nz]/np.abs(k[nz]); return np.fft.ifft(uh).real
def rand(seed):
    rng = np.random.default_rng(seed); ch = np.zeros(N, complex)
    for kk in range(1, K0+1):
        c = rng.standard_normal()+1j*rng.standard_normal(); ch[kk]=c; ch[N-kk]=np.conj(c)
    w = np.fft.ifft(ch).real; return w-w.mean()

def DP1(w):
    wx, wxx = dx(w), dx(w, 2)
    return -H(wx*wx) - 2*H(wx)*wx - 2*H(w)*wxx

def bnl(w, a):
    return -a*vel(w)*dx(w) + H(w)*w

def report(name, lhs, rhs):
    d = abs(lhs-rhs); den = max(abs(lhs), abs(rhs), 1e-300)
    good = (d/den < 1e-10) or (d < 1e-11)
    print(f"[{'PASS' if good else 'FAIL'}] {name:46s} lhs={lhs:+.6e} rhs={rhs:+.6e} rel={d/den:.1e}")
    return good

ok = True
print("="*94); print("Palinstrophy production balance -- independent inertial certification"); print("="*94)
for seed in range(4):
    w = rand(seed)
    v, vx = H(w), H(dx(w)); u = vel(w); wx, wxx = dx(w), dx(w, 2)

    # (1) first variation D P1 via finite difference
    h = rand(seed+100); eps = 1e-6
    P1 = lambda ww: ip(H(ww), dx(ww)**2)
    fd = (P1(w+eps*h) - P1(w-eps*h))/(2*eps)
    ok &= report(f"s{seed} D P1 vs FD", ip(DP1(w), h), fd)

    # (2) inertial I(a) vs hand formula, several a
    for a in (-2.0, -1.0, 0.5):
        I = ip(bnl(w, a), DP1(w))
        hand = ((2-2*a)*ip(v*v, wx*wx) - 2*a*ip(u*v, wx*wxx) + 2*ip(v*vx, w*wx)
                + a*ip(u*wx, H(wx*wx)) - ip(v*w, H(wx*wx)))
        ok &= report(f"s{seed} I(a={a:+.1f}) vs hand formula", I, hand)

    # (3) leading square W1 = 6 int v^2 wx^2 >= 0 at a=-2
    W1 = 6*ip(v*v, wx*wx)
    ok &= (W1 >= -1e-14)
    print(f"      W1 = 6∫v²wx² = {W1:+.4e}  (>=0: {W1>=-1e-14})")

print("="*94); print("INERTIAL FORMULA CERTIFIED" if ok else "FAILED"); print("="*94)
