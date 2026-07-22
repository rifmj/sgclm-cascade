#!/usr/bin/env python3
"""
Single-command verifier for the sgclm_cascade verification package.

Runs every load-bearing certifier and the solver-identity validator, and checks each
reaches its success verdict. Several certifiers print a verdict but always exit 0, so this
driver gates on the verdict string (and the absence of a failure token), not on exit code
alone. Exits 0 only if every check passes; nonzero otherwise.

Usage:  python3 verify.py        (from the anc/ directory)
"""
import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))

# (script, must-appear success marker, must-NOT-appear failure token within the verdict)
CHECKS = [
    ("certify_identities_symbolic.py", "ALL IDENTITIES PROVED",        "SOME IDENTITY FAILED"),
    ("certify_flux_law.py",            "ALL CERTIFIED",                "SOME CHECKS FAILED"),
    ("certify_ito_vanishing.py",       "LEMMA CERTIFIED",              "CHECK FAILED"),
    ("certify_palinstrophy_mine.py",   "INERTIAL FORMULA CERTIFIED",   "\nFAILED"),
    ("certify_projected_generator.py", "VERIFIED",                     "FAILED"),
    ("certify_theta_G_exact.py",       "MC agreement: PASS",           "MC agreement: FAIL"),
    ("certify_cubic_balance.py",       "PASS(4b-aniso-nonzero)=True",  "=False"),
    ("validate.py",                    "ALL CHECKS PASS",              "[FAIL]"),
]

def run(script, marker, antimarker):
    p = subprocess.run([sys.executable, os.path.join(HERE, script)],
                       capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    ok = (marker in out) and (antimarker not in out) and p.returncode == 0
    return ok, p.returncode

def main():
    print("=" * 70)
    print("sgclm_cascade — verification package")
    print("=" * 70)
    all_ok = True
    for script, marker, anti in CHECKS:
        try:
            ok, rc = run(script, marker, anti)
        except Exception as e:                       # timeout, missing file, crash
            ok, rc = False, f"EXC:{e}"
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {script:34s}  (rc={rc}; marker={'found' if ok else 'MISSING/failed'})")
    print("=" * 70)
    print("ALL VERIFIED" if all_ok else "VERIFICATION FAILED")
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
