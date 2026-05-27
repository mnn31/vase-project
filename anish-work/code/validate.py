"""Independent validation of the MAV-QV implementation (synthesis_volume.py).

Three checks:

  A. Divergence-theorem (V3) gives the correct volume on a closed analytic
     surface (a sphere) at the same triangulation density. Confirms the
     mesh-building code and the (1/6) Σ v1·(v2×v3) formula are correct.

  B. The outer profile passes exactly through the three tape-measurement
     anchors (R_base at z = 0, R_widest at the belly, R_rim at z = 17.5).
     Confirms the geometric setup is internally consistent with the direct
     measurements.

  C. V4 reproduces the partner's source-direct frustum at his widest-point
     assumption (h_widest = 8.0). His value = 2942 cm³.

@author Manan Gupta
@author Claude (Anthropic AI assistant), code co-author
"""
from __future__ import annotations
import math
import numpy as np

# Import the MAV-QV module (kept on disk as synthesis_volume.py for path stability)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthesis_volume as syn


def check_A_sphere():
    """Build a closed triangle mesh of a sphere; the divergence theorem
    should give (4/3)π R³ to high precision."""
    R = 5.0
    N_theta = 72
    N_phi   = 36
    theta = np.linspace(0, 2*math.pi, N_theta, endpoint=False)
    phi = np.linspace(0, math.pi, N_phi + 1)   # 0 = north pole, π = south
    cos_t = np.cos(theta); sin_t = np.sin(theta)

    pts = np.zeros((N_phi + 1, N_theta, 3))
    for i, p in enumerate(phi):
        for j in range(N_theta):
            pts[i, j] = (R * math.sin(p) * cos_t[j],
                         R * math.sin(p) * sin_t[j],
                         R * math.cos(p))

    tris = []
    for i in range(N_phi):
        for j in range(N_theta):
            jn = (j + 1) % N_theta
            a = pts[i,   j]; b = pts[i,   jn]
            c = pts[i+1, j]; d = pts[i+1, jn]
            # Outward winding: looking from outside the sphere, vertices
            # should be counter-clockwise. (i, j) is "above-and-left",
            # (i+1, jn) is "below-and-right". The two triangles are
            # (a → c → d) and (a → d → b), which gives outward-pointing
            # normals (positive r·n).
            tris.append((a, c, d))
            tris.append((a, d, b))

    tris = np.array(tris)
    v1, v2, v3 = tris[:, 0], tris[:, 1], tris[:, 2]
    signed = np.einsum("ij,ij->i", v1, np.cross(v2, v3))
    V = signed.sum() / 6.0       # Should be positive — no abs(), so a
                                 # winding bug here would FAIL the test
                                 # instead of being silently corrected.
    assert V > 0, ("Sphere test winding is inward — outward winding lost. "
                    "Would otherwise mask orientation bugs in the main V3.")

    V_true = (4.0/3.0) * math.pi * R**3
    err_pct = abs(V - V_true) / V_true * 100
    print(f"  A. Divergence-theorem on a sphere (R={R} cm):")
    print(f"     V_computed = {V:.4f} cm³")
    print(f"     V_true     = {V_true:.4f} cm³  (= 4/3·π·R³)")
    print(f"     Error      = {err_pct:.4f}%  "
          f"({'PASS' if err_pct < 1.0 else 'FAIL'})")
    return err_pct < 1.0


def check_B_anchors_consistent():
    """The hand-coded outer profile must pass exactly through the three
    tape-measurement anchors. This is a direct sanity check on the geometric
    setup — if any anchor drifted away from its measured circumference, the
    integration would be working from an inconsistent model."""
    h_prof, r_prof = syn.load_outer_profile()
    checks = [
        ("R_base  at z = 0.00",  0.00,  syn.R_BASE),
        ("R_widest at z = 7.00", 7.00,  syn.R_WIDEST),
        ("R_rim   at z = 17.50", 17.50, syn.R_RIM),
    ]
    print(f"  B. Outer profile passes through the three tape anchors:")
    all_ok = True
    for label, z_target, r_expected in checks:
        r_at_z = float(np.interp(z_target, h_prof, r_prof))
        diff = r_at_z - r_expected
        ok = abs(diff) < 0.001  # 10 µm tolerance
        all_ok = all_ok and ok
        flag = "OK" if ok else "FAIL"
        print(f"     {label:>20s}: profile r = {r_at_z:.4f} cm, "
              f"expected {r_expected:.4f}, diff {diff:+.4f}  ({flag})")
    return all_ok


def check_C_partner_m3():
    """V4 with widest_h = 8.0 should match partner's M3 = 2942 cm³ closely."""
    V4_at_8 = syn.method_V4_frustum_raw(widest_h=8.0)
    partner_M3 = 2942.0
    diff = V4_at_8 - partner_M3
    print(f"  C. V4 vs partner's M3 (both at widest_h = 8.0 cm):")
    print(f"     partner M3   = {partner_M3:.2f} cm³")
    print(f"     MAV-QV V4    = {V4_at_8:.2f} cm³")
    print(f"     diff         = {diff:+.2f} cm³  ({diff/partner_M3*100:+.2f}%)")
    pct = abs(diff) / partner_M3 * 100
    print(f"     {'PASS' if pct < 5.0 else 'FAIL'} (< 5 % is acceptable; "
          f"small differences come from rim-wall handling and C_base 41.1 vs 41.2)")
    return pct < 5.0


def main():
    print("═" * 72)
    print("  MAV-QV VALIDATION SUITE")
    print("═" * 72)
    results = []
    results.append(check_A_sphere())
    print()
    results.append(check_B_anchors_consistent())
    print()
    results.append(check_C_partner_m3())
    print()
    print("═" * 72)
    print(f"  Summary: {sum(results)}/{len(results)} checks passed")
    print("═" * 72)


if __name__ == "__main__":
    main()
