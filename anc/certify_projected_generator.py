#!/usr/bin/env python3
"""
certify_projected_generator.py -- referee round 4 / v1.7 ledger item.

Certifies the projected-generator bookkeeping of §4:

  (1) decomposition   I = I_M + R_M with I_M = <P_M N, P_M DP1>, R_M = <Q_M N, Q_M DP1>
      (machine-zero residual on random band fields; NB the cross terms vanish because
      P_M and Q_M are complementary orthogonal projectors);
  (2) support lemma   R_M(omega) = 0 whenever omega is supported on |k| <= K with 2K <= M
      (exact zero), and its sharpness: R_M != 0 generically for M < 2K;
  (3) the M = 1 counterexample of Remark 4.x (prompted by a referee):
      omega = A cos x  =>  N = (3/2)A^2 sin 2x, DP1 = (5/2)A^2 sin 2x,
      I = (15*pi/4) A^4,  W1 = (9*pi/2) A^4,  I/W1 = 5/6,
      while P_1 N = 0, I_1 = 0 and <w_xx, DP1> = 0 (so C1 = 0):
      the full pairing is NOT the generator contribution.

Exit 0 iff all checks pass. A corrupted-decomposition check (deliberately unprojected
"I_M") is required to FAIL, so the certificate rejects a broken verifier.
"""
import sys
import numpy as np

N = 512
TOL = 1e-12


def ifft(w):
    return np.fft.irfft(w, N)


def dd(w):
    return w * 1j * np.arange(len(w))


def hilb(w):
    out = w.copy(); out[1:] *= -1j; out[0] = 0
    return out


def lam_inv(w):
    out = w.copy(); out[1:] = out[1:] / np.arange(1, len(w)); out[0] = 0
    return out


def integ(f):
    return f.sum() * 2 * np.pi / N


def proj(w, M):
    out = w.copy(); out[M + 1:] = 0
    return out


def fields(wh):
    w = ifft(wh); wx = ifft(dd(wh)); wxx = ifft(dd(dd(wh)))
    v = ifft(hilb(wh)); vx = ifft(dd(hilb(wh))); u = ifft(-lam_inv(wh))
    return w, wx, wxx, v, vx, u


def N_hat(wh, a=-2.0):
    w, wx, wxx, v, vx, u = fields(wh)
    return np.fft.rfft(-a * u * wx + v * w)[:len(wh)]


def DP1_hat(wh):
    w, wx, wxx, v, vx, u = fields(wh)
    Hwx2 = ifft(hilb(np.fft.rfft(wx ** 2)[:len(wh)]))
    return np.fft.rfft(-Hwx2 - 2 * vx * wx - 2 * v * wxx)[:len(wh)]


def pair(ah, bh):
    return integ(ifft(ah) * ifft(bh))


def band_field(rng, K, amp=1.0):
    wh = np.zeros(N // 2 + 1, dtype=complex)
    wh[1:K + 1] = (rng.normal(size=K) + 1j * rng.normal(size=K)) * amp * N / 8
    return wh


def main():
    rng = np.random.default_rng(41)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + f"  {name}" + (f"  ({detail})" if detail else ""))
        ok = ok and cond

    # (1) decomposition I = I_M + R_M, random fields, several (K, M)
    worst = 0.0
    for K, M in [(8, 12), (8, 20), (16, 24), (24, 40)]:
        wh = proj(band_field(rng, K), M)
        Nh, Dh = N_hat(wh), DP1_hat(wh)
        I = pair(Nh, Dh)
        IM = pair(proj(Nh, M), proj(Dh, M))
        RM = pair(Nh - proj(Nh, M), Dh - proj(Dh, M))
        worst = max(worst, abs(I - IM - RM) / max(abs(I), 1e-300))
    check("decomposition I = I_M + R_M (4 configs)", worst < TOL, f"max rel resid {worst:.1e}")

    # (2) support lemma: R_M = 0 exactly for 2K <= M; nonzero for M < 2K
    z, nz = [], []
    for K, M in [(8, 16), (8, 20), (6, 12), (10, 40)]:
        wh = band_field(rng, K)
        Nh, Dh = N_hat(wh), DP1_hat(wh)
        z.append(abs(pair(Nh - proj(Nh, M), Dh - proj(Dh, M))))
    for K, M in [(8, 12), (8, 15), (10, 12)]:
        wh = band_field(rng, K)
        Nh, Dh = N_hat(wh), DP1_hat(wh)
        w, wx, wxx, v, vx, u = fields(wh)
        W1 = 6 * integ(v * v * wx * wx)
        nz.append(abs(pair(Nh - proj(Nh, M), Dh - proj(Dh, M))) / W1)
    check("support lemma: R_M = 0 for 2K <= M", max(z) < 1e-10, f"max |R_M| {max(z):.1e}")
    check("sharpness: R_M != 0 for M < 2K", min(nz) > 1e-3, f"min |R_M|/W1 {min(nz):.1e}")

    # (3) M = 1 counterexample, exact rationals
    A = 1.7
    wh = np.zeros(N // 2 + 1, dtype=complex); wh[1] = A * N / 2       # A cos x
    Nh, Dh = N_hat(wh), DP1_hat(wh)
    w, wx, wxx, v, vx, u = fields(wh)
    I = pair(Nh, Dh); W1 = 6 * integ(v * v * wx * wx)
    I1 = pair(proj(Nh, 1), proj(Dh, 1))
    C1pair = integ(wxx * ifft(Dh))
    check("M=1: I = (15*pi/4) A^4", abs(I - 15 * np.pi / 4 * A ** 4) < 1e-9 * abs(I))
    check("M=1: W1 = (9*pi/2) A^4", abs(W1 - 9 * np.pi / 2 * A ** 4) < 1e-9 * W1)
    check("M=1: I/W1 = 5/6", abs(I / W1 - 5 / 6) < 1e-12)
    check("M=1: I_1 = 0 (projected pairing vanishes)", abs(I1) < 1e-12)
    check("M=1: <w_xx, DP1> = 0 (C1 = 0)", abs(C1pair) < 1e-12)

    # corrupted-certificate check: an UNPROJECTED "I_M" must break the balance claim
    whb = proj(band_field(rng, 8), 12)
    Nh, Dh = N_hat(whb), DP1_hat(whb)
    fake_IM = pair(Nh, Dh)                     # deliberately unprojected
    true_IM = pair(proj(Nh, 12), proj(Dh, 12))
    check("corrupted check: unprojected pairing detected as wrong",
          abs(fake_IM - true_IM) > 1e-6 * abs(fake_IM))

    print("VERIFIED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
