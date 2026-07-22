#!/usr/bin/env python3
"""
Corruption battery for the sgclm_cascade certifiers.

A certifier that accepts a corrupted certificate is not a verifier. This script applies one
single-site mathematical mutation to each of a representative set of certifiers (a temporary
in-place edit, restored in a finally block), re-runs it through the same success-marker check
that verify.py uses, and asserts the corrupted certifier is now REJECTED (its success verdict
disappears, or it exits nonzero). It is representative -- one probe per certifier surface
class -- not an exhaustive corruption of every checked identity.

Usage:  python3 mutation_tests.py     (from the anc/ directory; exits nonzero if any survives)
"""
import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))

# (script, old_substring, mutated_substring, success_marker)
# each mutation flips a load-bearing constant; the certifier must then fail its success marker.
MUTATIONS = [
    # symbolic advective-cancellation coefficient (1 + a/2) -> (1 + a/3): residual no longer zero
    ("certify_identities_symbolic.py", "(1 + a_sym / 2)", "(1 + a_sym / 3)", "ALL IDENTITIES PROVED"),
    # projected-generator M=1 numeric anchor I = (15*pi/4) A^4 -> (16*pi/4): the check fails
    ("certify_projected_generator.py", "15 * np.pi / 4", "16 * np.pi / 4", "VERIFIED"),
    # theta_G Wick sum E[I] leading coefficient 6*T1 -> 7*T1: theta_G leaves its banked interval
    ("certify_theta_G_exact.py", "6 * T[\"T1\"] + 4 * T[\"T2\"]", "7 * T[\"T1\"] + 4 * T[\"T2\"]", "MC agreement: PASS"),
    # flux-law secondary-term identity constant -1/2 -> -3/5: identity no longer holds
    ("certify_flux_law.py", "-0.5*ip(H(wxx), w*w)", "-0.6*ip(H(wxx), w*w)", "ALL CERTIFIED"),
]

def run_capture(script):
    p = subprocess.run([sys.executable, os.path.join(HERE, script)],
                       capture_output=True, text=True, timeout=600)
    return p.stdout + p.stderr, p.returncode

def main():
    print("=" * 70)
    print("sgclm_cascade — corruption battery (each mutation must be REJECTED)")
    print("=" * 70)
    all_ok = True
    for script, old, new, marker in MUTATIONS:
        path = os.path.join(HERE, script)
        original = open(path, encoding="utf-8").read()
        if old not in original:
            print(f"  [ERROR] {script:34s}  mutation anchor not found -- battery is stale")
            all_ok = False
            continue
        try:
            open(path, "w", encoding="utf-8").write(original.replace(old, new, 1))
            out, rc = run_capture(script)
        finally:
            open(path, "w", encoding="utf-8").write(original)   # always restore
        rejected = (marker not in out) or (rc != 0)
        all_ok &= rejected
        print(f"  [{'REJECTED' if rejected else 'SURVIVED!'}]  {script:34s}  "
              f"(corrupted certifier {'failed as required' if rejected else 'still reported success — NOT a verifier'})")
    print("=" * 70)
    print("ALL CORRUPTIONS REJECTED" if all_ok else "A CORRUPTION SURVIVED — certifier is not discriminating")
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
