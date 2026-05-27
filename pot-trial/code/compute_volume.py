"""Final disk-method volume calculation for the POT trial.

Same method-shape as the vase: photogrammetry mesh → slice profile → anchor
at the tape-measured rim circumference → integrate.

Two pot-specific fixes that the original (vase) code didn't need:

  1. NO CYLINDER FILTER. A cap centered on a slightly-offset axis excludes
     bowl rim points on the far side; without the cap, we get the true
     bbox. Above the desk plane only the bowl is present anyway.

  2. ANCHOR ON UPPER-HALF MAX. The slices near the desk plane have
     textbook contamination (r > R_top), and the very first slice into
     the bowl region is a transition. So we anchor at the max r in the
     TOP HALF of the bowl height — definitely the rim, never the textbook
     or the transition.

Cone-extrapolate the profile from h=0 (apex) to the clean bowl start, since
photogrammetry can't capture the bowl's curved base where it sits on the
desk.

@author Manan Gupta
@author Claude (Anthropic AI assistant), code co-author
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.integrate import simpson


# === Inputs from pot-measurements.pdf ================================
C_TOP = 54.0
T_WALL = 0.49
H_OUTER = 7.0
R_TOP = C_TOP / (2 * math.pi)


# === Load raw slice profile ==========================================
ROOT = Path(__file__).resolve().parent.parent
mesh = json.loads((ROOT / "data" / "mesh_profile.json").read_text())
HS = np.array(mesh["heights_cm"])
RS = np.array(mesh["r_outer_cm_raw"])


def identify_bowl_and_anchor(hs, rs, R_top, H_outer):
    """Return (h_clean, r_anchored, anchor_info)."""
    # Bowl region: r below textbook-contamination threshold
    bowl_mask = rs < R_top * 1.2
    h_bowl = hs[bowl_mask]
    r_bowl = rs[bowl_mask]

    # Anchor: max r in the UPPER HALF of the bowl region. This is
    # definitely the rim — no transition or textbook in this band.
    upper_half = h_bowl > 0.5 * H_outer
    rim_r_mesh = float(r_bowl[upper_half].max())
    factor = R_top / rim_r_mesh
    r_anchored = r_bowl * factor

    # Drop the transitional first slice (immediately above the desk plane;
    # photogrammetry tends to record the textbook edge there at slightly
    # inflated r).
    bowl_start = 0
    while (bowl_start + 1 < len(r_anchored)
           and r_anchored[bowl_start] > r_anchored[bowl_start + 1] * 1.1):
        bowl_start += 1

    h_clean = h_bowl[bowl_start:]
    r_clean = r_anchored[bowl_start:]
    info = {
        "rim_r_mesh_cm": rim_r_mesh,
        "anchor_factor": float(factor),
        "bowl_start_h_cm": float(h_clean[0]),
    }
    return h_clean, r_clean, info


def main():
    h_clean, r_clean, info = identify_bowl_and_anchor(HS, RS, R_TOP, H_OUTER)
    print(f"Bowl actually starts at h = {info['bowl_start_h_cm']:.2f} cm")
    print(f"Mesh rim radius (pre-anchor): {info['rim_r_mesh_cm']:.3f} cm")
    print(f"Anchor factor: {info['anchor_factor']:.4f}")

    # Cone-extrapolate from (h=0, r=0) up to clean bowl start
    def r_at(h):
        if h <= h_clean[0]:
            return r_clean[0] * h / h_clean[0]
        if h >= h_clean[-1]:
            return r_clean[-1]
        return float(np.interp(h, h_clean, r_clean))

    N = 10_000
    grid = np.linspace(T_WALL, H_OUTER, N + 1)
    r_out = np.array([r_at(h) for h in grid])
    r_in = np.clip(r_out - T_WALL, 0.0, None)
    V_int = float(simpson(math.pi * r_in ** 2, x=grid))
    V_out = float(simpson(math.pi * r_out ** 2, x=grid))

    results = {
        "object": "pot trial (bowl on textbook stack)",
        "measurements": {
            "C_top_cm": C_TOP, "R_top_cm": R_TOP,
            "H_outer_cm": H_OUTER, "t_wall_cm": T_WALL,
        },
        "method": {
            "name": "Photogrammetry mesh + upper-half-max rim anchor + "
                    "cone-extrapolation below clean bowl start",
            "anchor_info": info,
            "N_integration": N,
        },
        "volume": {
            "V_outer_L": V_out / 1000.0,
            "V_interior_L": V_int / 1000.0,
            "V_interior_cm3": V_int,
            "V_3sig_L": float(f"{V_int/1000.0:.3g}"),
        },
        "profile": [
            {"h_cm": float(h), "r_out_cm": float(r),
             "r_in_cm": float(max(r - T_WALL, 0))}
            for h, r in zip(h_clean, r_clean)
        ],
    }
    (ROOT / "output" / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nV_outer    = {V_out:.2f} cm³ = {V_out/1000:.4f} L")
    print(f"V_interior = {V_int:.2f} cm³ = {V_int/1000:.4f} L")
    print(f"3 sig figs = {V_int/1000:.3g} L")


if __name__ == "__main__":
    main()
