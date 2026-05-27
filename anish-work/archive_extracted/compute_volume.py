#!/usr/bin/env python3
"""
AP Calc BC — Interior Volume Computation
=========================================

We compute the interior (water-holding) volume of the vase by THREE
independent methodologies, then compare them.

  M1 (model-based, surface):   Divergence Theorem on the triangulated
                               inner-cavity mesh.  V = (1/6) Σ v1·(v2×v3).

  M2 (model-based, volume):    Cross-section integration with angular
                               variation.  V = ∫∫ ½(R_inner)² dθ dz
                               (minus inner floor dome).

  M3 (source-direct, raw):     Uses ONLY the raw tape & caliper
                               measurements — no photo interpretation,
                               no 3D model.  Linearly interpolates a
                               radius profile between known
                               (z, R_inner) data points and sums
                               truncated-cone (frustum) volumes.

If M1 ≈ M2 ≈ M3 (within ≈1%), the entire methodology validates itself:
both the 3D model and the math operating on it agree with the raw data.
A larger gap between M1/M2 and M3 quantifies the uncertainty introduced
by the photo-to-3D-model step.

Output:
  volume_report.txt       Numeric results and comparison
  volume_report.png       Visualization of the three computations
"""

import numpy as np
import json
import struct
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))


# ════════════════════════════════════════════════════════════════════
#  Load 3D model data
# ════════════════════════════════════════════════════════════════════

with open(os.path.join(BASE, "vase_data.json")) as f:
    D = json.load(f)

Z       = np.array(D["z_cm"])
R_O     = np.array(D["r_outer_cm"])
R_I     = np.array(D["inner_r_by_angle"])   # (n_z, n_theta)
THETA   = np.array(D["theta_rad"])
WALL_T  = np.array(D["wall_t_avg_cm"])
IF_R    = np.array(D["inner_floor_r_cm"])   # inner dome radii (edge→center)
IF_Z    = np.array(D["inner_floor_z_cm"])   # inner dome z values

HEIGHT      = D["measurements"]["height_cm"]
FLOOR_H     = D["measurements"]["floor_height_cm"]
CONCAVITY   = D["measurements"]["concavity_depth_cm"]
BOT_T_C     = D["measurements"]["bottom_thickness_center_cm"]
BOT_T_E     = D["measurements"]["bottom_thickness_edge_cm"]
C_WIDEST    = D["measurements"]["widest_circ_cm"]
C_RIM       = D["measurements"]["rim_circ_cm"]
C_BASE      = D["measurements"]["base_circ_cm"]

LAYERS_RAW  = D["layer_data"]   # 12 entries: h_cm, type, measurements_mm

PI2 = 2 * math.pi


# ════════════════════════════════════════════════════════════════════
#  METHODOLOGY 1: Divergence Theorem on triangulated inner mesh
# ════════════════════════════════════════════════════════════════════
#
# Mathematical basis:
#   ∫∫∫_V (∇·F) dV = ∮∮_S F·n dA       (divergence theorem)
#   choose F(x,y,z) = (x,y,z)/3 ⇒ ∇·F = 1
#   ⇒ V = ∮∮_S (r/3)·n dA
#
# For a triangulated closed surface with triangles {v1,v2,v3}, this
# becomes:    V = (1/6) Σ_T  v1 · (v2 × v3)
#
# Each term is the signed volume of the tetrahedron formed by the
# triangle and the origin. Interior tetrahedra cancel; only the
# enclosed volume remains.


def stl_load_triangles(path):
    """Load triangles from a binary STL file. Returns (n,3,3) array."""
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        tris = np.empty((n, 3, 3), dtype=np.float64)
        for i in range(n):
            f.read(12)                       # skip normal
            for j in range(3):
                tris[i, j] = struct.unpack("<3f", f.read(12))
            f.read(2)                        # skip attribute
    return tris


def divergence_theorem_volume(triangles, verbose=False):
    """V = (1/6) Σ v1·(v2 × v3)  for each triangle."""
    v1 = triangles[:, 0]
    v2 = triangles[:, 1]
    v3 = triangles[:, 2]
    cross = np.cross(v2, v3)
    signed = np.einsum("ij,ij->i", v1, cross)
    if verbose:
        total = signed.sum() / 6.0
        sum_abs = np.abs(signed).sum() / 6.0
        print(f"    Signed sum / 6 = {total:.3f}  (this should equal ±V if consistent)")
        print(f"    Sum of |contributions| / 6 = {sum_abs:.3f} (much larger if mixed signs)")
    return abs(signed.sum()) / 6.0


def build_closed_cavity_triangles():
    """
    Build the FULL closed surface of the interior cavity by directly
    triangulating the 3D model data (with full angular variation).

    Closed surface = inner side walls + inner floor dome + top cap.
    All normals oriented OUTWARD with respect to the cavity volume.

    Returns (N, 3, 3) array of triangles.
    """
    n_z, n_t = R_I.shape
    cos_t = np.cos(THETA)
    sin_t = np.sin(THETA)

    z_floor_min = float(IF_Z[0])
    # Index of first z value at or above z_floor_min (where side wall begins)
    floor_idx = int(np.searchsorted(Z, z_floor_min))

    tris = []

    # ── 1. Inner side wall (full angular detail) ──
    # The cavity has the wall on its outer side; outward normal points
    # AWAY from the axis (into the wall material).
    for i in range(floor_idx, n_z - 1):
        z0, z1 = Z[i], Z[i+1]
        for j in range(n_t):
            jn = (j + 1) % n_t
            r0a, r0b = R_I[i,  j], R_I[i,  jn]
            r1a, r1b = R_I[i+1,j], R_I[i+1,jn]
            v0 = (r0a*cos_t[j],  r0a*sin_t[j],  z0)
            v3 = (r0b*cos_t[jn], r0b*sin_t[jn], z0)
            v1 = (r1a*cos_t[j],  r1a*sin_t[j],  z1)
            v2 = (r1b*cos_t[jn], r1b*sin_t[jn], z1)
            # Winding so normal points AWAY from axis (outward from cavity)
            tris.append((v0, v3, v1))
            tris.append((v1, v3, v2))

    # ── 2. Inner floor dome (revolution of IF_R, IF_Z) ──
    # The dome is the top surface of the bottom clay; the cavity sits ABOVE.
    # Outward normal of cavity points DOWN (into the clay below the dome).
    #
    # Winding: reverse order so the right-hand rule gives a downward normal.
    n_if = len(IF_R)
    for i in range(n_if - 1):
        r0, r1 = IF_R[i], IF_R[i+1]
        z0, z1 = IF_Z[i], IF_Z[i+1]
        if r1 < 1e-6:
            apex = (0.0, 0.0, float(z1))
            for j in range(n_t):
                jn = (j + 1) % n_t
                v0 = (r0*cos_t[j],  r0*sin_t[j],  z0)
                v1 = (r0*cos_t[jn], r0*sin_t[jn], z0)
                # Normal points DOWN (outward from cavity into clay)
                tris.append((apex, v0, v1))
        else:
            for j in range(n_t):
                jn = (j + 1) % n_t
                v0 = (r0*cos_t[j],  r0*sin_t[j],  z0)
                v3 = (r0*cos_t[jn], r0*sin_t[jn], z0)
                v1 = (r1*cos_t[j],  r1*sin_t[j],  z1)
                v2 = (r1*cos_t[jn], r1*sin_t[jn], z1)
                # Reversed winding for DOWNWARD-pointing normal
                tris.append((v0, v1, v3))
                tris.append((v1, v2, v3))

    # ── 3. TOP CAP at z=HEIGHT (closes the cavity) ──
    # This is the imaginary water surface. Outward normal points UP.
    z_top = float(Z[-1])
    apex_top = (0.0, 0.0, z_top)
    for j in range(n_t):
        jn = (j + 1) % n_t
        ra = R_I[-1, j]
        rb = R_I[-1, jn]
        v1 = (ra*cos_t[j],  ra*sin_t[j],  z_top)
        v2 = (rb*cos_t[jn], rb*sin_t[jn], z_top)
        # Normal points UP (outward from cavity)
        tris.append((apex_top, v1, v2))

    return np.array(tris)


def method_1_divergence():
    """
    Interior cavity volume by divergence theorem on a closed triangulated
    surface built from the 3D model data (full angular variation).
    """
    tris = build_closed_cavity_triangles()
    V = divergence_theorem_volume(tris, verbose=False)
    return V, tris


# ════════════════════════════════════════════════════════════════════
#  METHODOLOGY 2: Cross-section integration with angular variation
# ════════════════════════════════════════════════════════════════════
#
# The inner cavity occupies a 3D region. Its volume is:
#
#   V = ∫ over the cavity dV
#     = ∫_{z_floor_min}^{HEIGHT} A(z) dz
#
# where A(z) is the cross-sectional area of the cavity at height z.
#
# Above the floor dome edge (z > z_floor_min) but below FLOOR_HEIGHT:
#   the dome cuts into the cavity; A(z) is an annulus
#   from r_floor(z) (where the dome passes through this z) to R_inner(z,θ).
#
# Above FLOOR_HEIGHT: the dome is entirely below; A(z) is a full disk
#   of (varying) inner radius. With angular variation:
#       A(z) = ∫_0^{2π} ½ R_inner(z, θ)² dθ
#
# Below z_floor_min: no cavity, A = 0.


def method_2_cross_section():
    """
    Inner cavity volume by 2D area integration.
    Handles the angular variation in R_inner AND the inner floor dome.
    """
    n_z = len(Z)
    n_t = len(THETA)
    z_floor_min = float(IF_Z[0])   # lowest point of inner floor dome

    # For each z, compute the cross-sectional cavity area.
    A = np.zeros(n_z)

    # Build a function for the inner floor dome: at any z in
    # [z_floor_min, FLOOR_H], the dome passes through z at some radius r_d.
    # If z > FLOOR_H: cavity is full disk; r_d = 0.
    # If z < z_floor_min: cavity is empty; A = 0.

    # IF_Z is monotonically increasing from edge to center? Let's check:
    if IF_Z[0] > IF_Z[-1]:
        # Decreasing: reverse it
        if_z_sorted = IF_Z[::-1]
        if_r_sorted = IF_R[::-1]
    else:
        if_z_sorted = IF_Z
        if_r_sorted = IF_R

    for i, z_val in enumerate(Z):
        if z_val < z_floor_min:
            A[i] = 0.0
            continue

        # Compute angular integral of R_inner(z, θ)² / 2 (full disk area)
        # If the dome is below z_val (z_val >= FLOOR_H), full disk.
        # Otherwise, subtract the dome-occupied region (an inner disk of
        # radius r_d, which we get from the dome curve).
        r_inner_at_z = R_I[i]                  # (n_t,) inner radius at this z
        # Trapezoidal angular integral with periodic theta
        full_area = 0.5 * np.trapezoid(
            np.concatenate([r_inner_at_z**2, [r_inner_at_z[0]**2]]),
            np.concatenate([THETA, [THETA[0] + PI2]])
        )

        if z_val >= FLOOR_H:
            A[i] = full_area
        else:
            # Subtract the inner disk that the dome occupies at this z.
            # The dome passes through z=z_val at some radius r_d:
            r_d = float(np.interp(z_val, if_z_sorted, if_r_sorted))
            dome_area = math.pi * r_d**2
            A[i] = max(full_area - dome_area, 0.0)

    # Integrate A(z) dz
    V = float(np.trapezoid(A, Z))
    return V, A


# ════════════════════════════════════════════════════════════════════
#  METHODOLOGY 3: Source-direct (raw measurements only)
# ════════════════════════════════════════════════════════════════════
#
# Uses ONLY direct measurements — no photos, no 3D model:
#   • Tape circumferences: 60.0, 47.0, 41.1 cm  at three known heights
#   • Caliper wall thickness at 8 measured layers (multiple readings each)
#   • Total height: 18.0 cm
#   • Floor height (interior): 1.80 cm
#   • Concavity depth: 0.70 cm
#
# Method:
#   1. Build R_outer(z) by linear interpolation between the 3 tape anchors.
#      Widest anchor is placed at h = WIDEST_H (estimated visually as 8 cm,
#      varied below for sensitivity).
#   2. Compute R_inner(z) at each MEASURED LAYER by subtracting the
#      caliper-measured wall thickness (mean of all readings at that layer).
#   3. Linearly interpolate R_inner(z) between caliper data points.
#   4. The interior cavity is then a stack of truncated cones (frustums)
#      between successive (z, R_inner) data points.
#   5. Subtract the inner floor dome volume (treated as a paraboloid:
#      V_dome = π R² h / 2, with R = R_inner at floor edge, h = (FLOOR_H - bottom_thickness)).


def method_3_source_direct(widest_height_cm=8.0):
    """
    Volume from raw measurements only. No photos, no 3D model.

    Parameter: widest_height_cm — best-guess height of the widest point.
    We vary this for sensitivity below.
    """
    # 1. R_outer anchors from tape measure (the ONLY R_outer values used)
    R_outer_anchors_z = np.array([0.0, widest_height_cm, HEIGHT])
    R_outer_anchors_r = np.array([
        C_BASE   / PI2,
        C_WIDEST / PI2,
        C_RIM    / PI2,
    ])

    # 2. Wall thickness at each measured layer (only the MEASURED ones,
    #    ignoring extrapolated layers in the 3D model)
    layer_z = []
    layer_wall = []   # in cm
    for L in LAYERS_RAW:
        if L["measurements_mm"] is None:
            continue
        layer_z.append(L["h_cm"])
        layer_wall.append(L["mean_mm"] / 10.0)
    layer_z = np.array(layer_z)
    layer_wall = np.array(layer_wall)
    # Sort by z
    sort_idx = np.argsort(layer_z)
    layer_z = layer_z[sort_idx]
    layer_wall = layer_wall[sort_idx]

    # 3. Build R_inner at each measured layer
    #    R_inner(z_layer) = R_outer(z_layer) - wall(z_layer)
    R_outer_at_layer = np.interp(layer_z, R_outer_anchors_z, R_outer_anchors_r)
    R_inner_at_layer = R_outer_at_layer - layer_wall

    # 4. Add a top anchor at z=HEIGHT.  Inner radius at top = rim - rim_wall.
    #    The 3D model uses ~1.3cm rim wall thickness, but we don't have direct
    #    caliper data at the rim. Best approximation: use the highest measured
    #    layer's wall (~7.8mm) as a conservative estimate (the rim is thicker
    #    but the interior opening is what matters; the rim folds outward).
    rim_R_inner = (C_RIM / PI2) - layer_wall[-1]   # ~6.7 cm
    # Add base of interior — at z = FLOOR_H (just above the dome).
    # The inner radius at z=FLOOR_H = R_outer(FLOOR_H) - wall_at_FLOOR_H.
    # We don't have a direct caliper measurement at FLOOR_H, so we use the
    # lowest measured layer's wall thickness (~8.2mm at h=8.2 cm).
    floor_R_inner = (np.interp(FLOOR_H, R_outer_anchors_z, R_outer_anchors_r)
                     - layer_wall[0])

    # Combined data points for interior radius profile (z, R_inner)
    z_pts = np.concatenate([[FLOOR_H], layer_z, [HEIGHT]])
    r_pts = np.concatenate([[floor_R_inner], R_inner_at_layer, [rim_R_inner]])
    # Ensure sorted
    order = np.argsort(z_pts)
    z_pts = z_pts[order]
    r_pts = r_pts[order]

    # 5. Sum frustum volumes between consecutive (z, r) data points
    V_cavity = 0.0
    for i in range(len(z_pts) - 1):
        h = z_pts[i+1] - z_pts[i]
        R1 = r_pts[i]
        R2 = r_pts[i+1]
        V_frustum = (math.pi * h / 3.0) * (R1**2 + R1*R2 + R2**2)
        V_cavity += V_frustum

    # 6. Subtract interior floor dome (raised in center, reduces cavity volume).
    #    Model as a paraboloid:  V = (π/2) R² h
    #    where R = floor_R_inner (edge of dome) and h = CONCAVITY (rise above
    #    floor edge to center).
    R_dome = floor_R_inner
    h_dome = CONCAVITY
    V_inner_dome = (math.pi / 2.0) * R_dome**2 * h_dome
    V_cavity -= V_inner_dome

    return V_cavity, dict(
        z_pts=z_pts, r_pts=r_pts,
        widest_height=widest_height_cm,
        V_inner_dome=V_inner_dome,
    )


# ════════════════════════════════════════════════════════════════════
#  RUN ALL METHODS AND COMPARE
# ════════════════════════════════════════════════════════════════════

def main():
    sep = "=" * 70
    print(sep)
    print("  AP Calc BC — Interior Volume Computation")
    print("  Three independent methodologies for cross-validation")
    print(sep)

    # ── Method 1 ──
    print("\nMethod 1: Divergence Theorem on inner-cavity triangle mesh")
    print("  V = (1/6) Σ v₁·(v₂×v₃)   for each triangle")
    V1, tris = method_1_divergence()
    print(f"  Triangles processed: {len(tris):,}")
    print(f"  Volume: {V1:.2f} cm³  =  {V1/1000:.4f} L")

    # ── Method 2 ──
    print("\nMethod 2: Cross-section integration with angular variation")
    print("  V = ∫ A(z) dz,   A(z) = ½ ∮ R_inner(z,θ)² dθ  (minus dome)")
    V2, A_of_z = method_2_cross_section()
    print(f"  Height steps: {len(Z)}")
    print(f"  Volume: {V2:.2f} cm³  =  {V2/1000:.4f} L")

    # ── Method 3 ──
    print("\nMethod 3: Source-direct (raw measurements only, NO 3D model)")
    print("  Frustum stack from tape circumferences + caliper wall thicknesses")
    # Run with several widest-point heights for sensitivity
    sensitivity = {}
    for h_w in [6.0, 7.0, 8.0, 9.0, 10.0]:
        V3_h, _ = method_3_source_direct(widest_height_cm=h_w)
        sensitivity[h_w] = V3_h
        print(f"    widest at z={h_w:4.1f} cm  →  V = {V3_h:.2f} cm³")

    # Best estimate uses h=8.0 (visually consistent with photos)
    V3, M3_detail = method_3_source_direct(widest_height_cm=8.0)

    # ── Comparison ──
    print("\n" + sep)
    print("  COMPARISON")
    print(sep)
    avg_model = (V1 + V2) / 2
    print(f"\n  M1 (model + divergence theorem):  {V1:7.2f} cm³")
    print(f"  M2 (model + cross-section int.):  {V2:7.2f} cm³")
    print(f"  M3 (source-direct, raw data):     {V3:7.2f} cm³")
    print()
    print(f"  |M1 - M2| = {abs(V1-V2):.2f} cm³  ({abs(V1-V2)/V1*100:.2f}%)")
    print(f"  |M1 - M3| = {abs(V1-V3):.2f} cm³  ({abs(V1-V3)/V1*100:.2f}%)")
    print(f"  |M2 - M3| = {abs(V2-V3):.2f} cm³  ({abs(V2-V3)/V2*100:.2f}%)")

    spread = max(V1, V2, V3) - min(V1, V2, V3)
    print()
    print(f"  Total spread: {spread:.2f} cm³ "
          f"= {spread/avg_model*100:.2f}% of mean")

    # ── Sensitivity from M3 widest-position assumption ──
    print(f"\n  M3 sensitivity to widest-point height assumption:")
    V3_range = max(sensitivity.values()) - min(sensitivity.values())
    print(f"    Range over h_widest ∈ [6, 10] cm: {V3_range:.2f} cm³")
    print(f"    (this is the bound on uncertainty from NOT using photos)")

    # ── Final answer ──
    print("\n" + sep)
    final = avg_model     # average of M1 and M2 as the best 3D-model answer
    print(f"  FINAL ANSWER: Interior volume = {final:.1f} cm³ "
          f"({final/1000:.3f} L)")
    print(f"  ± uncertainty: {spread/2:.1f} cm³ "
          f"({spread/2/final*100:.1f}%)")
    print(sep)

    # ── Write text report ──
    report_path = os.path.join(BASE, "volume_report.txt")
    with open(report_path, "w") as f:
        f.write("AP Calc BC — Vase Interior Volume Computation\n")
        f.write("=" * 60 + "\n\n")
        f.write("THREE INDEPENDENT METHODOLOGIES\n\n")
        f.write(f"M1 (Divergence Theorem on mesh):  {V1:.3f} cm³\n")
        f.write(f"  V = (1/6) Σ v₁·(v₂×v₃) over {len(tris):,} triangles\n")
        f.write(f"  Mathematical basis: ∫∫∫∇·F dV = ∮F·n dA, F=r/3\n\n")
        f.write(f"M2 (Cross-section integration):    {V2:.3f} cm³\n")
        f.write(f"  V = ∫ A(z) dz, A(z) = ½ ∮R_inner(z,θ)² dθ\n")
        f.write(f"  {len(Z)} height steps × {len(THETA)} angular samples\n\n")
        f.write(f"M3 (Source-direct, raw data):     {V3:.3f} cm³\n")
        f.write(f"  Frustum stack from raw measurements only\n")
        f.write(f"  Uses: 3 tape circumferences, 8 caliper layers,\n")
        f.write(f"        height, floor height, concavity depth\n")
        f.write(f"  NO photo interpretation, NO 3D model\n\n")
        f.write(f"  Sensitivity to widest-point height assumption:\n")
        for h_w, V_h in sensitivity.items():
            f.write(f"    h_widest = {h_w:.1f} cm → V = {V_h:.3f} cm³\n")
        f.write("\n" + "-" * 60 + "\n\n")
        f.write("COMPARISON\n")
        f.write(f"|M1 - M2| = {abs(V1-V2):.3f} cm³ ({abs(V1-V2)/V1*100:.3f}%)\n")
        f.write(f"|M1 - M3| = {abs(V1-V3):.3f} cm³ ({abs(V1-V3)/V1*100:.3f}%)\n")
        f.write(f"|M2 - M3| = {abs(V2-V3):.3f} cm³ ({abs(V2-V3)/V2*100:.3f}%)\n\n")
        f.write(f"Total spread: {spread:.3f} cm³ ({spread/avg_model*100:.3f}% of mean)\n\n")
        f.write(f"FINAL ANSWER\n")
        f.write(f"  Interior volume = {final:.2f} ± {spread/2:.2f} cm³\n")
        f.write(f"                  = {final/1000:.4f} ± {spread/2/1000:.4f} L\n")
    print(f"\nReport saved: {os.path.basename(report_path)}")

    # ── Visualization ──
    make_visualization(V1, V2, V3, A_of_z, sensitivity, M3_detail)


def make_visualization(V1, V2, V3, A_of_z, sensitivity, M3_detail):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Cross-section area A(z) from Method 2
    ax = axes[0]
    ax.plot(A_of_z, Z, "b-", lw=2)
    ax.fill_betweenx(Z, 0, A_of_z, alpha=0.3)
    ax.set_xlabel("Cross-section cavity area A(z)  [cm²]")
    ax.set_ylabel("Height z  [cm]")
    ax.set_title(f"Method 2: ∫A(z)dz = {V2:.1f} cm³\n(cross-section integration)")
    ax.grid(True, alpha=0.3)
    ax.axhline(FLOOR_H, color="red", lw=1, ls="--", alpha=0.5,
               label=f"floor center z={FLOOR_H}")
    ax.legend(loc="lower right", fontsize=9)

    # Panel 2: Bar chart comparing the three methods
    ax = axes[1]
    methods = ["M1\nDivergence\nTheorem\n(on 3D mesh)",
               "M2\nCross-section\nIntegration\n(on 3D model)",
               "M3\nSource-direct\n(raw measurements\nonly, no 3D model)"]
    vals = [V1, V2, V3]
    colors = ["#3a76c5", "#5fa55a", "#d97a47"]
    bars = ax.bar(methods, vals, color=colors, edgecolor="black", lw=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{v:.0f} cm³", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Interior Volume  [cm³]")
    ax.set_title("Three Methodologies Compared")
    ax.grid(True, axis="y", alpha=0.3)
    # Horizontal line at mean
    mean_v = (V1 + V2 + V3) / 3
    ax.axhline(mean_v, color="black", ls=":", alpha=0.6,
               label=f"mean = {mean_v:.1f} cm³")
    ax.legend(loc="lower right", fontsize=9)

    # Panel 3: M3 source-direct profile + sensitivity
    ax = axes[2]
    z_pts = M3_detail["z_pts"]
    r_pts = M3_detail["r_pts"]
    ax.plot(r_pts, z_pts, "o-", color="#d97a47", lw=2, ms=8,
            label="M3: inner radius from raw data")
    ax.fill_betweenx(z_pts, 0, r_pts, alpha=0.2, color="#d97a47")
    # Overlay the 3D model's inner radius (averaged)
    inner_avg = np.array(D["r_inner_avg_cm"])
    ax.plot(inner_avg, Z, "g-", lw=1, alpha=0.7,
            label="3D model (M1/M2) avg inner")
    ax.set_xlabel("Inner radius  [cm]")
    ax.set_ylabel("Height z  [cm]")
    ax.set_title(f"M3: frustum stack  V = {V3:.1f} cm³\n(source-only profile)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(inner_avg.max(), r_pts.max()) + 0.5)

    plt.tight_layout()
    path = os.path.join(BASE, "volume_report.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved: {os.path.basename(path)}")


if __name__ == "__main__":
    main()
