"""
certify_identities_symbolic.py -- SYMBOLIC (exact, rational/polynomial) proofs of
the paper's load-bearing algebraic identities, replacing float-residual checks by
polynomial-identity certificates (review item: exact arithmetic).

Method: a band-limited real field  omega = sum_{k=1..K} (a_k cos kx + b_k sin kx)
with SYMBOLIC coefficients a_k, b_k is represented by its complex Fourier
coefficients c_k = (a_k - i b_k)/2, c_{-k} = (a_k + i b_k)/2 (exact sympy
symbols). All operators (H, d/dx, Lambda^{-1}) are diagonal multipliers; products
are exact convolutions; int_0^{2pi} f dx = 2*pi*f_0. Every identity below is then
a POLYNOMIAL in {a_k, b_k} (and the advection parameter a, kept symbolic where
stated), and is certified by exact expansion to the ZERO polynomial -- a proof
for all band-limited fields of the given band, over QQ (not a numerical check).

Certified identities (all with symbolic a where meaningful):
  (1) L0 advective cancellation:  <N[w], w> = (1+a/2) * int (Hw) w^2
      (in particular == 0 at a = -2), N[w] = -a u w_x + (Hw) w.
  (2) First variation of P1 = int (Hw) w_x^2:
      <DP1, h> = d/deps P1(w + eps h)|_0 for an independent symbolic field h,
      with DP1 = -H(w_x^2) - 2 (H w_x) w_x - 2 (H w) w_xx.
  (3) Inertial closed form: <b_nl, DP1> = (2-2a) int v^2 w_x^2
      - 2a int u v w_x w_xx + 2 int v v_x w w_x + a int u w_x H(w_x^2)
      - int v w H(w_x^2),  b_nl = -a u w_x + v w, v = Hw.
  (4) Lemma B0 (homogeneous injection): sum_k q_|k| * [ D^2 P1(e_k, e_k) ] = 0
      with symbolic per-|k| rates q_j, e_k running over the cos/sin pair basis.
  (5) A2 secondary term: int (H w_x) w w_x = -(1/2) int (H w_xx) w^2.

Band size K is a parameter (default 5; the identities are certified for ALL
fields band-limited to |k| <= K; the pen derivations in FLUX_LAW.md/PAPER.md
give the general-K statements, of which these are exact-QQ instances).
"""
import sys
import sympy as sp

K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
I_ = sp.I
a_sym = sp.Symbol('a')                      # advection parameter, symbolic


# ---------- exact convolution engine over dict {wavenumber: coeff} -----------
def field_symbols(prefix, K):
    """Real band-limited field as complex coefficients from real symbols."""
    A = [sp.Symbol(f"{prefix}a{j}") for j in range(1, K + 1)]
    B = [sp.Symbol(f"{prefix}b{j}") for j in range(1, K + 1)]
    f = {}
    for j in range(1, K + 1):
        f[j] = (A[j - 1] - I_ * B[j - 1]) / 2
        f[-j] = (A[j - 1] + I_ * B[j - 1]) / 2
    return f


def mult(f, m):
    """Apply diagonal multiplier m(k)."""
    return {k: m(k) * v for k, v in f.items() if m(k) != 0}


def H(f):    return mult(f, lambda k: -I_ * sp.sign(k))
def DX(f):   return mult(f, lambda k: I_ * k)
def U(f):    return mult(f, lambda k: sp.Rational(-1, abs(k)) if k else 0)


def prod(f, g):
    out = {}
    for kf, vf in f.items():
        for kg, vg in g.items():
            out[kf + kg] = out.get(kf + kg, 0) + vf * vg
    return out


def integral(f):
    return 2 * sp.pi * f.get(0, 0)


def N_full(w, a=a_sym):
    u = U(w); wx = DX(w); v = H(w)
    t1 = prod(u, wx)
    t2 = prod(v, w)
    out = {}
    for k, val in t1.items():
        out[k] = out.get(k, 0) - a * val
    for k, val in t2.items():
        out[k] = out.get(k, 0) + val
    return out


def is_zero(expr, label):
    z = sp.expand(expr)
    ok = (z == 0)
    print(f"  {label}: {'PROVED (zero polynomial)' if ok else 'FAIL'}")
    if not ok:
        print("   residual:", sp.simplify(z))
    return ok


print(f"Symbolic QQ-certificates, band K={K} "
      f"({2*K} real symbols per field; advection parameter a symbolic)")
print("=" * 74)
ALL = True
w = field_symbols("w", K)

# (1) L0 cancellation, symbolic a
Nw = N_full(w)
lhs = integral(prod(Nw, w))
rhs = (1 + a_sym / 2) * integral(prod(H(w), prod(w, w)))
ALL &= is_zero(lhs - rhs, "(1) <N[w],w> = (1+a/2) int (Hw) w^2   [symbolic a]")

# (2) first variation of P1 against an independent symbolic field h
h = field_symbols("h", K)
eps = sp.Symbol('eps')
weps = {k: w.get(k, 0) + eps * h.get(k, 0) for k in set(w) | set(h)}
P1_eps = integral(prod(H(weps), prod(DX(weps), DX(weps))))
dP1 = sp.expand(sp.diff(P1_eps, eps)).subs(eps, 0)
wx = DX(w)
DP1 = {}
for part in (mult(H(prod(wx, wx)), lambda k: -1),
             {k: -2 * v for k, v in prod(H(wx), wx).items()},
             {k: -2 * v for k, v in prod(H(w), DX(DX(w))).items()}):
    for k, v in part.items():
        DP1[k] = DP1.get(k, 0) + v
pair = integral(prod(DP1, h))
ALL &= is_zero(dP1 - pair, "(2) <DP1,h> = dP1(w+eps h)/deps|_0   [independent h]")

# (3) inertial closed form, symbolic a
u = U(w); v = H(w); vx = DX(H(w)); wxx = DX(DX(w))
bnl = {}
for k, val in prod(u, wx).items():
    bnl[k] = bnl.get(k, 0) - a_sym * val
for k, val in prod(v, w).items():
    bnl[k] = bnl.get(k, 0) + val
lhs3 = integral(prod(bnl, DP1))
Hwx2 = H(prod(wx, wx))
rhs3 = ((2 - 2 * a_sym) * integral(prod(prod(v, v), prod(wx, wx)))
        - 2 * a_sym * integral(prod(prod(u, v), prod(wx, wxx)))
        + 2 * integral(prod(prod(v, vx), prod(w, wx)))
        + a_sym * integral(prod(prod(u, wx), Hwx2))
        - integral(prod(prod(v, w), Hwx2)))
ALL &= is_zero(lhs3 - rhs3, "(3) <b_nl,DP1> = I(a) five-term formula   [symbolic a]")

# (4) Lemma B0: homogeneous Ito trace of D^2 P1 vanishes
# D^2 P1(h,h) = 2 int (Hw) h_x^2 + 4 int (Hh) w_x h_x   (second variation of P1)
qs = [sp.Symbol(f"q{j}") for j in range(1, K + 1)]
trace = 0
for j in range(1, K + 1):
    for phase in ("c", "s"):     # e = cos(jx)/sqrt(pi) or sin(jx)/sqrt(pi) (ON)
        if phase == "c":
            e = {j: sp.Rational(1, 2), -j: sp.Rational(1, 2)}
        else:
            e = {j: -I_ / 2, -j: I_ / 2}
        e = {k: v / sp.sqrt(sp.pi) for k, v in e.items()}
        ex = DX(e)
        d2 = 2 * integral(prod(H(w), prod(ex, ex))) \
            + 4 * integral(prod(H(e), prod(wx, ex)))
        trace += qs[j - 1] * d2
ALL &= is_zero(trace, "(4) Lemma B0: sum_k q_|k| D^2P1(e_k,e_k) = 0   [symbolic q_j]")

# (5) A2 secondary-term identity
lhs5 = integral(prod(H(wx), prod(w, wx)))
rhs5 = -sp.Rational(1, 2) * integral(prod(H(wxx), prod(w, w)))
ALL &= is_zero(lhs5 - rhs5, "(5) int (Hw_x) w w_x = -1/2 int (Hw_xx) w^2")

print("=" * 74)
print("ALL IDENTITIES PROVED (zero polynomials over QQ[a])" if ALL else
      "SOME IDENTITY FAILED -- see residuals above")
sys.exit(0 if ALL else 1)
