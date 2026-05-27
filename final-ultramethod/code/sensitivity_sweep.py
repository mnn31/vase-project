"""Sensitivity sweep: investigate the 3.13 L → 2.90 L gap.

Runs five parameter sweeps and reports ΔV for each, plus combinations.
Does NOT modify the main synthesis_volume.py code — operates on copies of
the same data so the baseline remains intact.

Sweeps:
  A. Replace photogrammetry-mesh outer profile with Anish's hand-coded
     OUTER_PROFILE (still anchored to the 3 tape measurements).
  B. V4 frustum with widest_h sweep over [6.0, 10.0] cm.
  C. Lower-wall thickness boost: add Δt for z < 8.2 cm, sweep [0, 3] mm.
  D. INNER_DOME_RISE sweep over [0.5, 1.5] cm.
  E. Rim wall thickness + taper-down length sweep:
       T_RIM_PEAK ∈ [1.0, 1.8] cm, taper length ∈ [0.5, 2.0] cm.

@author Manan Gupta
@author Claude (Anthropic AI assistant), code co-author
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import simpson

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthesis_volume as syn


# ====================================================================
# Helper: a generic V1 disk-method evaluator that takes a callable for
# outer radius and a wall-thickness array on the z_grid.
# ====================================================================

def evaluate_V1(
    r_outer_grid: np.ndarray,
    t_wall_grid: np.ndarray,
    z_grid: np.ndarray,
    z_floor_edge: float,
    z_floor_center: float,
    R_dome_edge_override: float | None = None,
) -> float:
    """Disk-method V1 with arbitrary outer profile and wall grid."""
    t_mean = t_wall_grid.mean(axis=1) if t_wall_grid.ndim > 1 else t_wall_grid
    z_dense = np.linspace(z_floor_edge, syn.H_EXT, syn.N_DISK + 1)
    r_outer_dense = np.interp(z_dense, z_grid, r_outer_grid)
    t_mean_dense  = np.interp(z_dense, z_grid, t_mean)
    r_inner_dense = np.clip(r_outer_dense - t_mean_dense, 0.0, None)
    V_full = float(simpson(math.pi * r_inner_dense**2, x=z_dense))

    if R_dome_edge_override is None:
        r_outer_at_edge = float(np.interp(z_floor_edge, z_grid, r_outer_grid))
        t_wall_at_edge  = float(np.interp(z_floor_edge, z_grid, t_mean))
        R_dome_edge = r_outer_at_edge - t_wall_at_edge
    else:
        R_dome_edge = R_dome_edge_override

    V_dome = syn.dome_volume(R_dome_edge, z_floor_center - z_floor_edge)
    return V_full - V_dome


# Build baseline once
def build_baseline():
    mesh_h, mesh_a = syn.load_mesh_profile()
    r_anchored, h_belly = syn.anchor_mesh_to_tape(
        mesh_h, mesh_a, syn.R_BASE, syn.R_WIDEST, syn.R_RIM, syn.H_EXT)
    z_grid = np.linspace(0.0, syn.H_EXT, syn.N_GRID_Z)
    r_outer_grid = np.interp(z_grid, mesh_h, r_anchored)
    r_outer_grid = np.where(z_grid < mesh_h[0], syn.R_BASE, r_outer_grid)
    t_wall_grid = syn.build_wall_thickness_grid(z_grid, syn.N_THETA)
    return z_grid, r_outer_grid, t_wall_grid, h_belly


# ====================================================================
# Anish's hand-coded OUTER_PROFILE (read from partner-work/archive_extracted/build_vase.py).
# These are (h_cm, r_outer_cm) tuples — no rescaling needed; he already
# anchored at the same 3 tape circumferences.
# ====================================================================

ANISH_OUTER_PROFILE = [
    (0.00, 6.5413),
    (0.30, 6.75), (0.80, 6.95), (1.30, 7.40), (2.00, 7.95),
    (2.50, 8.30), (3.00, 8.60), (3.60, 8.85), (4.00, 8.98),
    (4.50, 9.12), (5.00, 9.24), (5.50, 9.34), (6.00, 9.43),
    (6.50, 9.50), (7.00, 9.5493),   # widest = exact R_widest
    (7.50, 9.52), (8.00, 9.45), (8.50, 9.36), (9.00, 9.24),
    (9.50, 9.12), (10.00, 8.98), (10.50, 8.83), (11.00, 8.66),
    (11.50, 8.48), (12.00, 8.28), (12.50, 8.08), (13.00, 7.88),
    (13.50, 7.68), (14.00, 7.50), (14.50, 7.35), (15.00, 7.22),
    (15.50, 7.10), (16.00, 7.00), (16.30, 6.93), (16.50, 6.88),
    (16.80, 6.95), (17.00, 7.08), (17.20, 7.25), (17.40, 7.40),
    (17.50, 7.4803),  # rim peak = exact R_rim
    (17.70, 7.44), (17.85, 7.35), (18.00, 7.20),
]


def anish_r_outer(z_grid):
    """Interpolate Anish's hand-coded profile onto z_grid."""
    h_arr = np.array([p[0] for p in ANISH_OUTER_PROFILE])
    r_arr = np.array([p[1] for p in ANISH_OUTER_PROFILE])
    return np.interp(z_grid, h_arr, r_arr)


# ====================================================================
# SWEEPS
# ====================================================================

def sweep_A_anish_shape(z_grid, t_wall_grid):
    """A: Replace mesh-derived r_outer with Anish's hand-coded profile."""
    r_anish = anish_r_outer(z_grid)
    V = evaluate_V1(r_anish, t_wall_grid, z_grid,
                    syn.Z_FLOOR_EDGE, syn.Z_FLOOR_CENTER)
    return V, r_anish


def sweep_B_widest_h():
    """B: V4 frustum with various widest_h."""
    rows = []
    for h_w in [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.4, 10.0]:
        V = syn.method_V4_frustum_raw(widest_h=h_w)
        rows.append((h_w, V))
    return rows


def sweep_C_lower_wall(z_grid, r_outer_grid, t_wall_grid):
    """C: Add constant Δt to t_wall for z < 8.2 cm (the extrapolated region)."""
    rows = []
    for delta_mm in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        delta_cm = delta_mm / 10.0
        t_modified = t_wall_grid.copy()
        # Boost wall in z < 8.2 region
        lower_mask = z_grid < 8.2
        t_modified[lower_mask, :] += delta_cm
        V = evaluate_V1(r_outer_grid, t_modified, z_grid,
                        syn.Z_FLOOR_EDGE, syn.Z_FLOOR_CENTER)
        rows.append((delta_mm, V))
    return rows


def sweep_D_dome_rise(z_grid, r_outer_grid, t_wall_grid):
    """D: Vary the inner-floor dome rise from 0.5 to 1.5 cm.
    For each, z_floor_center shifts so it stays consistent with z_floor_edge."""
    rows = []
    for rise_cm in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
        # Keep z_floor_edge fixed at 1.10; center moves up.
        z_edge = syn.Z_FLOOR_EDGE
        z_center = z_edge + rise_cm
        # R_dome_edge depends on z_floor_edge geometry, same as before
        r_outer_at_edge = float(np.interp(z_edge, z_grid, r_outer_grid))
        t_wall_at_edge  = float(np.interp(z_edge, z_grid,
                                           t_wall_grid.mean(axis=1)))
        R_dome_edge = r_outer_at_edge - t_wall_at_edge
        V = evaluate_V1(r_outer_grid, t_wall_grid, z_grid,
                        z_edge, z_center, R_dome_edge_override=R_dome_edge)
        rows.append((rise_cm, V))
    return rows


def sweep_E_rim(z_grid, r_outer_grid):
    """E: Vary T_RIM_PEAK and taper-down length."""
    rows = []
    for rim_peak_cm in [1.0, 1.2, 1.3, 1.5, 1.7, 2.0]:
        for taper_len in [0.5, 1.0, 1.5, 2.0]:
            # Build modified wall grid
            t_grid = syn.build_wall_thickness_grid(z_grid, syn.N_THETA).copy()
            # Override the rim region: smooth taper from body to rim_peak
            for j in range(syn.N_THETA):
                for i, z in enumerate(z_grid):
                    if z > syn.H_EXT - taper_len:
                        ramp = (z - (syn.H_EXT - taper_len)) / taper_len
                        body_t = t_grid[i, j]
                        # Linear blend toward rim_peak
                        t_grid[i, j] = body_t * (1 - ramp) + rim_peak_cm * ramp
            V = evaluate_V1(r_outer_grid, t_grid, z_grid,
                            syn.Z_FLOOR_EDGE, syn.Z_FLOOR_CENTER)
            rows.append((rim_peak_cm, taper_len, V))
    return rows


# ====================================================================
# COMBINED scenarios
# ====================================================================

def combo(label, z_grid, r_outer_grid, t_wall_grid, *,
          use_anish_shape=False, lower_wall_delta_mm=0.0,
          dome_rise_cm=0.70, rim_peak_cm=1.30, rim_taper_len=1.0,
          rmax_bump_correction_cm=0.0, z_floor_edge_override=None):
    """Compose multiple adjustments and report V.

    rmax_bump_correction_cm: reduce R_widest by this much to account for the
        tape circumference being measured across the outward "deep" ridges
        rather than the smooth envelope. Only affects the belly anchor.
    z_floor_edge_override: shift the cavity bottom upward (e.g. if the inner-
        floor dome edge sits higher than 1.10 cm).
    """
    r_grid = anish_r_outer(z_grid) if use_anish_shape else r_outer_grid.copy()
    t_grid = t_wall_grid.copy()

    # Bump-correct R_max in the bulge band (re-anchor the profile)
    if rmax_bump_correction_cm > 0:
        # Find belly region and squeeze it inward
        i_belly = int(np.argmax(r_grid))
        # Apply a Gaussian-like falloff centered at belly
        # so we don't touch the base or rim anchors
        z_belly = z_grid[i_belly]
        sigma = 4.0  # cm — width of bulge region affected
        falloff = np.exp(-0.5 * ((z_grid - z_belly) / sigma) ** 2)
        # Only reduce where the original is near R_max (the bulge band)
        r_grid = r_grid - rmax_bump_correction_cm * falloff

    # Lower-wall delta
    if lower_wall_delta_mm > 0:
        t_grid[z_grid < 8.2, :] += lower_wall_delta_mm / 10.0

    # Rim adjustment
    if rim_peak_cm != 1.30 or rim_taper_len != 1.0:
        for j in range(syn.N_THETA):
            for i, z in enumerate(z_grid):
                if z > syn.H_EXT - rim_taper_len:
                    ramp = (z - (syn.H_EXT - rim_taper_len)) / rim_taper_len
                    body_t = t_grid[i, j]
                    t_grid[i, j] = body_t * (1 - ramp) + rim_peak_cm * ramp

    # Dome geometry
    z_edge = z_floor_edge_override if z_floor_edge_override else syn.Z_FLOOR_EDGE
    z_center = z_edge + dome_rise_cm
    r_outer_at_edge = float(np.interp(z_edge, z_grid, r_grid))
    t_wall_at_edge  = float(np.interp(z_edge, z_grid, t_grid.mean(axis=1)))
    R_dome_edge = r_outer_at_edge - t_wall_at_edge

    V = evaluate_V1(r_grid, t_grid, z_grid, z_edge, z_center,
                    R_dome_edge_override=R_dome_edge)
    return V


# ====================================================================
# MAIN
# ====================================================================

def main():
    sep = "═" * 72
    print(sep)
    print("  SENSITIVITY SWEEPS — investigating the 3.13 L → 2.90 L gap")
    print(sep)

    z_grid, r_outer_grid, t_wall_grid, h_belly = build_baseline()
    V_baseline = evaluate_V1(r_outer_grid, t_wall_grid, z_grid,
                              syn.Z_FLOOR_EDGE, syn.Z_FLOOR_CENTER)
    print(f"\n  Baseline V1 (current MAV-QV):  {V_baseline:.2f} cm³ = "
          f"{V_baseline/1000:.4f} L")
    print(f"  Target:                        2900.00 cm³ = 2.9000 L")
    print(f"  Gap to close:                  {V_baseline-2900:.2f} cm³")

    # ── A: Anish's shape ──────────────────────────────────────────
    print(f"\n{sep}")
    print("  SWEEP A — Replace photogrammetry shape with Anish's hand-coded profile")
    print(sep)
    V_A, r_anish = sweep_A_anish_shape(z_grid, t_wall_grid)
    print(f"  V1 with Anish's r_outer(z):    {V_A:.2f} cm³  "
          f"(ΔV = {V_A - V_baseline:+.2f} cm³)")
    print(f"\n  Side-by-side comparison at sample heights:")
    print(f"  {'z (cm)':>8} {'mesh r':>10} {'Anish r':>10} {'Δr':>8}")
    for z in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
              11.0, 12.0, 14.0, 16.0, 17.5]:
        rm = float(np.interp(z, z_grid, r_outer_grid))
        ra = float(np.interp(z, z_grid, r_anish))
        print(f"  {z:>8.1f} {rm:>10.3f} {ra:>10.3f} {ra-rm:>+8.3f}")

    # ── B: widest-h sweep ────────────────────────────────────────
    print(f"\n{sep}")
    print("  SWEEP B — V4 frustum with various widest_h assumptions")
    print(sep)
    rows = sweep_B_widest_h()
    for h_w, V in rows:
        marker = ""
        if abs(h_w - 9.4) < 0.05: marker = "  ← mesh-derived (default)"
        if abs(h_w - 7.0) < 0.05: marker = "  ← Anish's value"
        print(f"  widest_h = {h_w:4.1f} cm:  V4 = {V:7.2f} cm³"
              f"  ({V/1000:.4f} L){marker}")

    # ── C: lower-wall thickness boost ───────────────────────────
    print(f"\n{sep}")
    print("  SWEEP C — Boost wall thickness for z < 8.2 cm (extrapolated region)")
    print(sep)
    rows = sweep_C_lower_wall(z_grid, r_outer_grid, t_wall_grid)
    print(f"  {'Δt (mm)':>10}  {'V (cm³)':>10}  {'ΔV (cm³)':>10}")
    for delta_mm, V in rows:
        print(f"  {delta_mm:>10.1f}  {V:>10.2f}  {V-V_baseline:>+10.2f}")

    # ── D: dome rise sweep ──────────────────────────────────────
    print(f"\n{sep}")
    print("  SWEEP D — Inner-floor dome rise")
    print(sep)
    rows = sweep_D_dome_rise(z_grid, r_outer_grid, t_wall_grid)
    print(f"  {'rise (cm)':>12}  {'V (cm³)':>10}  {'ΔV (cm³)':>10}")
    for rise_cm, V in rows:
        marker = "  ← default" if abs(rise_cm - 0.7) < 0.01 else ""
        print(f"  {rise_cm:>12.2f}  {V:>10.2f}  {V-V_baseline:>+10.2f}{marker}")

    # ── E: rim sweep ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("  SWEEP E — Rim wall peak + taper length")
    print(sep)
    rows = sweep_E_rim(z_grid, r_outer_grid)
    print(f"  {'rim_peak':>10} {'taper_len':>10}  {'V (cm³)':>10}  "
          f"{'ΔV (cm³)':>10}")
    for rim_peak, taper_len, V in rows:
        marker = ""
        if abs(rim_peak - 1.30) < 0.01 and abs(taper_len - 1.0) < 0.01:
            marker = "  ← default"
        print(f"  {rim_peak:>10.2f} {taper_len:>10.2f}  {V:>10.2f}  "
              f"{V-V_baseline:>+10.2f}{marker}")

    # ── COMBINATIONS ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("  COMBINATIONS — can stacked adjustments reach 2.90 L honestly?")
    print(sep)
    scenarios = [
        ("Baseline (default everything)", dict()),
        ("A only (Anish's shape)", dict(use_anish_shape=True)),
        ("A + C(+1 mm lower wall)", dict(use_anish_shape=True,
                                          lower_wall_delta_mm=1.0)),
        ("A + C(+2 mm)", dict(use_anish_shape=True,
                               lower_wall_delta_mm=2.0)),
        ("A + C(+3 mm)", dict(use_anish_shape=True,
                               lower_wall_delta_mm=3.0)),
        ("A + D(dome 1.0 cm)", dict(use_anish_shape=True, dome_rise_cm=1.0)),
        ("A + E(rim 1.5 cm)", dict(use_anish_shape=True, rim_peak_cm=1.5)),
        ("A + C(+2) + D(1.0) + E(1.5/1.5)",
         dict(use_anish_shape=True, lower_wall_delta_mm=2.0,
              dome_rise_cm=1.0, rim_peak_cm=1.5, rim_taper_len=1.5)),
        ("A + C(+1) + D(0.9) + E(1.4/1.2)",
         dict(use_anish_shape=True, lower_wall_delta_mm=1.0,
              dome_rise_cm=0.9, rim_peak_cm=1.4, rim_taper_len=1.2)),
        ("A + C(+3) + D(1.0) + E(1.5/1.5)  — aggressive",
         dict(use_anish_shape=True, lower_wall_delta_mm=3.0,
              dome_rise_cm=1.0, rim_peak_cm=1.5, rim_taper_len=1.5)),
        ("Only mesh + C(+3) + D(1.2) — no Anish",
         dict(use_anish_shape=False, lower_wall_delta_mm=3.0,
              dome_rise_cm=1.2, rim_peak_cm=1.5, rim_taper_len=1.5)),
        # ── Honest combinations using only defensible adjustments ──
        # (F dropped — see comment above; bump correction only valid if
        #  a deep ridge is at the belly height, which it isn't.)
        ("A + C(+2) + z_edge=1.30",
         dict(use_anish_shape=True, lower_wall_delta_mm=2.0,
              z_floor_edge_override=1.30)),
        ("A + C(+2) + D(1.0) + E(1.5/1.5) + z_edge=1.30",
         dict(use_anish_shape=True, lower_wall_delta_mm=2.0,
              dome_rise_cm=1.0, rim_peak_cm=1.5, rim_taper_len=1.5,
              z_floor_edge_override=1.30)),
        ("A + C(+1.5) + D(0.9) + z_edge=1.25",
         dict(use_anish_shape=True, lower_wall_delta_mm=1.5,
              dome_rise_cm=0.9, z_floor_edge_override=1.25)),
        ("A + C(+1) + D(1.0) + E(1.5/1.5) + z_edge=1.30",
         dict(use_anish_shape=True, lower_wall_delta_mm=1.0,
              dome_rise_cm=1.0, rim_peak_cm=1.5, rim_taper_len=1.5,
              z_floor_edge_override=1.30)),
    ]
    print(f"  {'Scenario':<50} {'V (cm³)':>10} {'V (L)':>8} {'gap':>8}")
    for label, kwargs in scenarios:
        V = combo(label, z_grid, r_outer_grid, t_wall_grid, **kwargs)
        gap = V - 2900
        flag = "  ← within 50 cm³ of 2.90 L" if abs(gap) < 50 else ""
        print(f"  {label:<50} {V:>10.2f} {V/1000:>8.4f} {gap:>+8.2f}{flag}")

    print(f"\n{sep}")
    print("  DONE — interpret combinations honestly. If only the aggressive")
    print("  scenario lands at 2.90 L, that's force-fitting. If a moderate")
    print("  scenario lands there with all parameters in defensible ranges,")
    print("  that's evidence the calculation was off.")
    print(sep)


if __name__ == "__main__":
    main()
