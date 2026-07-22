"""
certify_ito_vanishing.py -- independent check of the Ito-trace vanishing for the
palinstrophy production functional P1 = int (H omega) omega_x^2 (Carrier B).

Claim (homogeneous-noise lemma): (1/2) Tr(Q D2 P1) = 0 when b_k depends only on
|k| (translation-invariant covariance), and is generically nonzero otherwise.

D2 P1(h,h) = 2 int (H omega) h_x^2 + 4 int (H h) omega_x h_x.
(1/2)Tr(Q D2 P1) = sum_k b_k^2 [ int (H omega)(e_k')^2 + 2 int (H e_k) omega_x e_k' ].

Clean proof (each term): for a (cos_k, sin_k) pair with equal weight, the
k-summand is (const in x) times a mean-zero integrand:
  Term1 pair -> b_k^2 (k^2/pi) * int (H omega) = 0    (H omega has zero mean);
  Term2 pair -> b_k^2 (-k/pi) * 2 int omega_x   = 0    (omega_x has zero mean).
Numerics below confirm to machine precision and show anisotropic b breaks it.
Basis: e_k = cos(kx)/sqrt(pi), sin(kx)/sqrt(pi)  (so int e_k^2 dx = 1 on [0,2pi]).
"""
import numpy as np

N = 128
K0 = 6                 # omega band
KF = 8                 # noise band 1..KF
TWO_PI = 2*np.pi
x = np.arange(N)*(TWO_PI/N)
k = np.fft.fftfreq(N, d=1.0/N)
sgn = np.sign(k)

def H(f):
    return np.fft.ifft(-1j*sgn*np.fft.fft(f)).real
def dx(f):
    return np.fft.ifft((1j*k)*np.fft.fft(f)).real
def ip(f, g):
    return (TWO_PI/N)*np.sum(f*g)

def rand_omega(seed):
    rng = np.random.default_rng(seed)
    ch = np.zeros(N, dtype=complex)
    for kk in range(1, K0+1):
        c = rng.standard_normal()+1j*rng.standard_normal()
        ch[kk] = c; ch[N-kk] = np.conj(c)
    w = np.fft.ifft(ch).real
    return w - w.mean()

sq = 1.0/np.sqrt(np.pi)
def basis():
    """yield (e_k, weight-key |k|) over the real ON basis for 1<=|k|<=KF."""
    for kk in range(1, KF+1):
        yield (sq*np.cos(kk*x), kk)
        yield (sq*np.sin(kk*x), kk)

def half_tr(omega, bfun):
    """(1/2) Tr(Q D2 P1) with per-mode weight b_k^2 = bfun(|k|)^2."""
    Hom = H(omega); omx = dx(omega)
    tot = 0.0
    for e, kk in basis():
        b2 = bfun(kk)**2
        ex = dx(e)
        t1 = ip(Hom, ex*ex)
        t2 = 2.0*ip(H(e), omx*ex)
        tot += b2*(t1+t2)
    return tot

print("="*90)
print("Ito-trace vanishing for P1 (Carrier B), N=%d" % N)
print("="*90)
homog = lambda kk: 1.0                       # b_k = f(|k|): homogeneous
aniso = lambda kk: (1.0 if kk % 2 == 0 else 0.3)   # depends on k in a non-|k| way? still f(|k|)...
# A genuinely anisotropic covariance must weight cos_k and sin_k DIFFERENTLY.
def half_tr_aniso(omega):
    Hom = H(omega); omx = dx(omega); tot = 0.0
    for kk in range(1, KF+1):
        for e, w in ((sq*np.cos(kk*x), 1.0), (sq*np.sin(kk*x), 0.2)):  # cos!=sin weight
            ex = dx(e)
            tot += w*(ip(Hom, ex*ex) + 2.0*ip(H(e), omx*ex))
    return tot

ok = True
for seed in range(5):
    w = rand_omega(seed)
    hh = half_tr(w, homog)
    aa = half_tr_aniso(w)
    p1 = abs(hh) < 1e-11
    p2 = abs(aa) > 1e-6
    ok &= p1 and p2
    print(f"seed{seed}: homogeneous (1/2)Tr = {hh:+.3e}  [{'PASS ~0' if p1 else 'FAIL'}]"
          f"   anisotropic = {aa:+.3e}  [{'PASS !=0' if p2 else 'FAIL'}]")
print("="*90)
print("LEMMA CERTIFIED: homogeneous-noise Ito trace vanishes; anisotropic breaks it"
      if ok else "CHECK FAILED")
print("="*90)
