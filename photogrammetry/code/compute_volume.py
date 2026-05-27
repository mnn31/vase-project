"""Final disk-method volume calculation for the vase project.

METHOD: photogrammetry + axisymmetric spin + tape-measurement anchors,
        with position-dependent wall thickness.

  1. Apple Object Capture turned 29 photos into a 3-D triangle mesh of the
     vase's outer surface. The "a" axis (front-facing semi-axis) is well-
     photographed; the "b" axis (back) is incomplete. We take the front-side
     profile a(h) and spin it around the vertical axis — that gives us a
     complete axisymmetric vase, equivalent to assuming the (un-photographed)
     back side mirrors the (well-photographed) front side.

  2. The mesh's belly radius matches the tape-measured C_max within 0.3%,
     but it underestimates the base by ~30% and the rim by ~9% (photogrammetry
     struggles where photo coverage is uneven). To fix that, we keep the
     mesh's SHAPE (how r varies with h) but RESCALE so the profile passes
     exactly through the three tape-measurement anchors.

  3. Wall thickness was measured at 8 ring depths going from rim to base.
     Each block below is multiple angular readings AROUND the vase at the
     same depth — averaging the block gives the angular-mean wall thickness
     at that single depth. We interpolate t_wall(h) through these 8 ring
     depths and use it position-dependently in the integral.

  4. r_inner(h) = r_outer(h) − t_wall(h), then integrate by Simpson's rule.

Inputs read by this file:
  - Three tape-measured outer circumferences and interior height (constants)
  - Wall-thickness caliper readings (constants, grouped by ring depth)
  - Mesh slice profile from data/mesh_profile.json

@author Manan Gupta
@author Claude (Anthropic AI assistant), code co-author
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from scipy.integrate import simpson


# === Inputs from the physical vase ===================================

C_bot = 41.2       # cm, bottom outer circumference
C_max = 60.0       # cm, widest belly outer circumference
C_top = 47.0       # cm, just under the rim
H_INTERIOR = 17.2  # cm, bottom-dent peak to rim

R_bot = C_bot / (2 * math.pi)
R_max = C_max / (2 * math.pi)
R_top = C_top / (2 * math.pi)

# Wall-thickness caliper readings in mm. Each block is the readings taken
# AROUND THE VASE at one depth (multiple angular readings at the same height).
# Blocks are ordered top -> bottom of the vase.
WALL_BLOCKS = [
    ("shallow", [8.4, 7.3, 7.2, 7.7, 8.0, 8.3, 7.8, 7.5]),
    ("deep",    [9.8, 8.4, 7.8, 7.5, 7.5, 7.5, 8.1, 7.6, 7.6]),
    ("shallow", [7.1, 7.2, 7.3, 7.2, 7.5, 7.6, 7.3, 6.8, 6.7, 6.9]),
    ("deep",    [8.3, 8.0, 8.0, 8.1, 8.0, 7.4, 7.7, 7.9, 10.7, 9.4, 9.6, 8.7]),
    ("shallow", [8.0, 8.1, 7.9, 7.9, 7.8, 7.4, 7.8, 8.1, 8.0]),
    ("deep",    [8.8, 8.7, 9.2, 8.3, 8.2, 8.0, 8.1, 8.0, 8.1, 7.6, 7.6, 7.7,
                 8.6, 8.4, 8.3, 7.6, 7.3, 7.7, 7.7]),
    ("shallow", [7.8, 7.8, 8.1, 8.2, 8.9, 8.3, 7.9, 8.1, 8.1, 8.1, 8.1, 8.2,
                 8.1, 8.2, 7.9, 7.7, 8.5, 8.1]),
    ("deep",    [8.6, 8.4, 8.8, 7.5, 8.1, 7.8, 8.1, 8.2, 8.7, 8.7, 8.2, 8.2, 7.9]),
]
N_BLOCKS = len(WALL_BLOCKS)

# Angular-mean wall thickness at each ring depth (mm).
RING_T_MM = np.array([float(np.mean(vals)) for _, vals in WALL_BLOCKS])
# Heights of the 8 ring depths in INTERIOR coordinates, evenly spaced from
# top to bottom across H_INTERIOR.
RING_H_INTERIOR = np.array(
    [H_INTERIOR * (1 - (i + 0.5) / N_BLOCKS) for i in range(N_BLOCKS)])

# Useful summaries
ALL_T_MM = np.array([v for _, vals in WALL_BLOCKS for v in vals])
T_MEAN_MM = float(ALL_T_MM.mean())
T_STDEV_MM = float(ALL_T_MM.std(ddof=1))
RING_T_MEAN_MM = float(RING_T_MM.mean())
T_BASE = RING_T_MEAN_MM / 10.0    # cm, used as lower integration bound
H_OUTER = H_INTERIOR + T_BASE     # rim is t_base above outer bottom


# === Load mesh slices from the photogrammetry pipeline ===============

ROOT = Path(__file__).resolve().parent.parent
mesh = json.loads((ROOT / "data" / "mesh_profile.json").read_text())
MESH_H = np.array(mesh["sample_heights_cm"])
MESH_A = np.array(mesh["sample_a_outer_cm"])


def rescale_to_anchors(mesh_h, mesh_a, R_bot, R_max, R_top, H_outer):
    """Stretch the mesh's a-profile in r so it passes through the three
    tape-measurement anchors at h = 0, h = h_belly, h = H_outer."""
    i_belly = int(np.argmax(mesh_a))
    h_belly = mesh_h[i_belly]
    a_base, a_belly, a_top = mesh_a[0], mesh_a[i_belly], mesh_a[-1]
    r_rescaled = np.zeros_like(mesh_a)
    for i, (h, a) in enumerate(zip(mesh_h, mesh_a)):
        if h <= h_belly:
            frac = (a - a_base) / (a_belly - a_base) if a_belly != a_base else 0
            r_rescaled[i] = R_bot + frac * (R_max - R_bot)
        else:
            frac = (a_belly - a) / (a_belly - a_top) if a_belly != a_top else 0
            r_rescaled[i] = R_max - frac * (R_max - R_top)
    return r_rescaled, h_belly


def t_wall_at(h_outer):
    """Position-dependent wall thickness in cm, interpolated through the
    8 ring measurements. h_outer is the outer height in cm."""
    h_int = h_outer - T_BASE  # convert outer -> interior height
    # Boundary handling: use the nearest ring outside the measured range.
    h_anchors = np.concatenate(([H_INTERIOR], RING_H_INTERIOR, [0.0]))
    t_anchors_mm = np.concatenate(([RING_T_MM[0]], RING_T_MM, [RING_T_MM[-1]]))
    order = np.argsort(h_anchors)
    return np.interp(h_int, h_anchors[order], t_anchors_mm[order]) / 10.0


def integrate_disk_position_dependent(profile_h, profile_r, N=10_000):
    """V_int = π ∫ (r(h) − t(h))² dh, with t varying through the 8 ring
    depths and r from the anchored mesh profile."""
    grid = np.linspace(T_BASE, H_OUTER, N + 1)
    r_out = np.interp(grid, profile_h, profile_r)
    t_grid = np.array([t_wall_at(h) for h in grid])
    r_in = np.clip(r_out - t_grid, 0.0, None)
    V_int = float(simpson(math.pi * r_in ** 2, x=grid))
    V_out = float(simpson(math.pi * r_out ** 2, x=grid))
    return V_int, V_out


def main():
    r_anchored, h_belly = rescale_to_anchors(
        MESH_H, MESH_A, R_bot, R_max, R_top, H_OUTER)

    V_int, V_outer = integrate_disk_position_dependent(MESH_H, r_anchored)

    # Cross-checks: constant-t versions
    grid = np.linspace(T_BASE, H_OUTER, 10001)
    r_out_grid = np.interp(grid, MESH_H, r_anchored)
    r_in_const_98 = np.clip(r_out_grid - T_MEAN_MM/10.0, 0, None)
    V_const_98 = float(simpson(math.pi * r_in_const_98**2, x=grid))
    r_in_const_8 = np.clip(r_out_grid - RING_T_MEAN_MM/10.0, 0, None)
    V_const_8 = float(simpson(math.pi * r_in_const_8**2, x=grid))

    # Sensitivity over ring mean ± 1σ of the 8 ring means
    ring_stdev = float(RING_T_MM.std(ddof=1))
    sens = []
    for delta in (-ring_stdev, 0.0, ring_stdev):
        t_grid = np.array([t_wall_at(h) + delta/10.0 for h in grid])
        ri = np.clip(r_out_grid - t_grid, 0, None)
        Vi = float(simpson(math.pi * ri**2, x=grid))
        sens.append({"t_offset_mm": delta, "V_L": Vi / 1000.0})

    # Sampled profile for the writeup
    sample_h = np.linspace(0, H_OUTER, 19)
    sample_rout = np.interp(sample_h, MESH_H, r_anchored)
    sample_t = np.array([t_wall_at(h) for h in sample_h])
    sample_rin = np.clip(sample_rout - sample_t, 0, None)

    results = {
        "student": {"name": "Manan Gupta", "period": "5", "date": "2026-05-22"},
        "measurements": {
            "C_bot_cm": C_bot, "C_max_cm": C_max, "C_top_cm": C_top,
            "R_bot_cm": R_bot, "R_max_cm": R_max, "R_top_cm": R_top,
            "H_interior_cm": H_INTERIOR, "H_outer_cm": H_OUTER,
            "t_wall_mean_mm_all_98": T_MEAN_MM,
            "t_wall_stdev_mm_all_98": T_STDEV_MM,
            "t_wall_mean_mm_ring_means": RING_T_MEAN_MM,
            "t_base_cm_used": T_BASE,
            "n_wall_readings": int(len(ALL_T_MM)),
            "n_ring_depths": N_BLOCKS,
            "n_shallow": sum(len(vals) for typ, vals in WALL_BLOCKS if typ == "shallow"),
            "n_deep": sum(len(vals) for typ, vals in WALL_BLOCKS if typ == "deep"),
        },
        "ring_wall_thicknesses": [
            {"block_index": i + 1, "type": typ,
             "h_interior_cm": float(RING_H_INTERIOR[i]),
             "n_angular_readings": len(vals),
             "mean_mm": float(np.mean(vals)),
             "stdev_mm": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
             "readings_mm": list(vals)}
            for i, (typ, vals) in enumerate(WALL_BLOCKS)
        ],
        "method": {
            "name": "Photogrammetry mesh + axisymmetric spin + tape-anchor rescale + position-dependent t",
            "h_belly_cm": float(h_belly),
            "n_mesh_slices": int(len(MESH_H)),
            "N_integration": 10000,
            "mesh_source": "data/mesh_profile.json (Apple Object Capture, 29 photos)",
        },
        "volume": {
            "V_outer_L": V_outer / 1000.0,
            "V_interior_L": V_int / 1000.0,
            "V_const_t_98mean_L": V_const_98 / 1000.0,
            "V_const_t_8ringmean_L": V_const_8 / 1000.0,
            "V_interior_cm3": V_int,
            "V_3sig_L": float(f"{V_int/1000.0:.3g}"),
        },
        "sensitivity": sens,
        "profile_table": [
            {"h_outer_cm": float(h),
             "r_out_cm": float(ro),
             "t_wall_cm": float(t),
             "r_in_cm": float(ri)}
            for h, ro, t, ri in zip(sample_h, sample_rout, sample_t, sample_rin)
        ],
    }

    out = ROOT / "output" / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"V_interior (position-dependent t)   = {V_int/1000:.4f} L  (3 sig: {V_int/1000:.3g} L)")
    print(f"V_interior (const t = mean of 98)   = {V_const_98/1000:.4f} L  (cross-check)")
    print(f"V_interior (const t = mean of 8 rings) = {V_const_8/1000:.4f} L  (cross-check)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
