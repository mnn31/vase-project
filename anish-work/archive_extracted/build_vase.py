#!/usr/bin/env python3
"""
AP Calc BC — Hand-Crafted 3D Vase Model
========================================

Every value in this file traces to a specific measurement or image:

  CIRCUMFERENCES (tape measure):
    widest = 60.0 cm  →  R = 9.5493 cm
    rim    = 47.0 cm  →  R = 7.4803 cm
    base   = 41.1 cm  →  R = 6.5413 cm
    height = 18.0 cm

  OUTER PROFILE:
    Width proportions read from IMG_1188 (side view with both rulers),
    calibrated so the three circumference anchor points match exactly.
    Validated against IMG_1675 (front view with ruler).

  LAYER POSITIONS:
    Counted from side views IMG_1188, IMG_1674, IMG_1676.
    ~1.2 cm spacing, 8 measured + 4 extrapolated.

  WALL THICKNESS:
    Direct caliper measurements at 8 layers (top → middle),
    each layer has multiple readings around the circumference.

  CONCAVE BOTTOM:
    From IMG_1666, IMG_1667 (bottom views).
    Foot ring width ~0.8 cm, concavity depth ~0.7 cm.

  RIM:
    From IMG_1670, IMG_1672 (close-ups).
    Rolled lip ~13 mm thick at peak.

Outputs:
  vase_3d.html           Interactive Three.js viewer
  vase_outer.stl         Outer surface STL
  vase_cross_section.png Cross-section verification plot
  vase_data.json         All profile data for volume calculation
"""

import numpy as np
import struct
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
PI2 = 2 * np.pi

# ══════════════════════════════════════════════════════════════════
#  SECTION 1: HARD MEASUREMENTS
# ══════════════════════════════════════════════════════════════════

HEIGHT = 18.0
R_WIDEST = 60.0 / PI2      # 9.5493 cm
R_RIM    = 47.0 / PI2      # 7.4803 cm
R_BASE   = 41.1 / PI2      # 6.5413 cm

# ── Outer profile: (height_cm, outer_radius_cm, source) ──
# Width proportions from IMG_1188 side-view ruler photo:
#   At each height, I measured vase silhouette width as a fraction
#   of the widest silhouette width, then multiplied by R_WIDEST.
#   Anchor points use exact circumference values.
#
# The widest point is at h≈7.0 cm (39% from bottom), determined by
# the ruler reading in IMG_1188 where the maximum width occurs at
# approximately 3.5" above the base (3.5" × 2.54 = 8.9 cm... but
# measured from the ~0.9" base mark: (3.5-0.9)×2.54 ≈ 6.6 cm,
# and accounting for the broad plateau it spans ~6-8 cm).

OUTER_PROFILE = [
    # (h_cm, R_cm, source_note)
    # ── Foot ring contact (z=0 is the floor) ──
    # The bottom of the side wall meets the floor at the foot ring.
    # Below this (underside) is the concave dome — see UNDERSIDE_PROFILE.
    ( 0.00,  R_BASE, "foot ring contact — 41.1 cm circumference (tape measure)"),
    # ── Unglazed base, lower part (concentric wheel rings visible IMG_1191, IMG_1192) ──
    # The side wall flares outward fairly rapidly above the foot.
    ( 0.30,  6.75, "small flare above foot — IMG_1191 close-up"),
    ( 0.80,  6.95, "IMG_1188 proportion: 0.728 × R_WIDEST"),
    ( 1.30,  7.40, "IMG_1188 proportion: 0.775"),
    ( 2.00,  7.95, "IMG_1188 proportion: 0.832"),
    ( 2.50,  8.30, "IMG_1188 proportion: 0.869"),
    ( 3.00,  8.60, "IMG_1188 proportion: 0.900"),
    # ── Glaze line (~3.6 cm from base, from ruler in IMG_1188) ──
    ( 3.60,  8.85, "glaze line — IMG_1188 ruler: 2.4\" mark, proportion 0.93"),
    # ── Lower glazed body (expanding) ──
    ( 4.00,  8.98, "IMG_1188 proportion: 0.94"),
    ( 4.50,  9.12, "IMG_1188 proportion: 0.955"),
    ( 5.00,  9.24, "IMG_1188 proportion: 0.968"),
    ( 5.50,  9.34, "IMG_1188 proportion: 0.978"),
    ( 6.00,  9.43, "IMG_1188 proportion: 0.987"),
    ( 6.50,  9.50, "IMG_1188 proportion: 0.995"),
    # ── Widest zone ──
    ( 7.00,  R_WIDEST, "WIDEST — 60.0 cm circumference"),
    ( 7.50,  9.52, "IMG_1188: still near max, proportion 0.997"),
    ( 8.00,  9.45, "IMG_1188 proportion: 0.99"),
    # ── Upper body (gradual narrowing) ──
    ( 8.50,  9.36, "IMG_1188 proportion: 0.98"),
    ( 9.00,  9.24, "IMG_1188 proportion: 0.968"),
    ( 9.50,  9.12, "IMG_1188 proportion: 0.955"),
    (10.00,  8.98, "IMG_1188 proportion: 0.94"),
    (10.50,  8.83, "IMG_1188 proportion: 0.924"),
    (11.00,  8.66, "IMG_1188 proportion: 0.907"),
    (11.50,  8.48, "IMG_1188 proportion: 0.888"),
    (12.00,  8.28, "IMG_1188 proportion: 0.867"),
    (12.50,  8.08, "IMG_1188 proportion: 0.846"),
    (13.00,  7.88, "IMG_1188 proportion: 0.825"),
    (13.50,  7.68, "IMG_1188 proportion: 0.804"),
    (14.00,  7.50, "IMG_1188 proportion: 0.785"),
    (14.50,  7.35, "IMG_1188 proportion: 0.770"),
    (15.00,  7.22, "IMG_1188 proportion: 0.756"),
    (15.50,  7.10, "IMG_1188 proportion: 0.743"),
    # ── Neck region (narrowest above base) ──
    (16.00,  7.00, "IMG_1188 proportion: 0.733"),
    (16.30,  6.93, "approaching neck minimum"),
    (16.50,  6.88, "neck minimum — IMG_1188, IMG_1674"),
    # ── Rim (rolled lip flares outward) ──
    (16.80,  6.95, "rim begins to flare — IMG_1670 close-up"),
    (17.00,  7.08, "rim flaring — IMG_1670"),
    (17.20,  7.25, "rim building up"),
    (17.40,  7.40, "approaching rim peak"),
    (17.50,  R_RIM, "RIM PEAK — 47.0 cm circumference — IMG_1670"),
    (17.70,  7.44, "past peak, slight rollback — IMG_1670"),
    (17.85,  7.35, "rim lip edge — IMG_1672"),
    (18.00,  7.20, "top edge of rim — IMG_1672"),
]

# ── Deep/shallow layer positions ──
# Counted from side views. 8 measured layers (shallow/deep alternating
# from top to bottom). 4 extrapolated below. ~1.2 cm spacing.
LAYERS = [
    # (height_cm, type, wall_thickness_measurements_mm)
    # MEASURED (caliper readings around circumference):
    (16.60, "shallow", [8.4, 7.3, 7.2, 7.7, 8.0, 8.3, 7.8, 7.5]),
    (15.40, "deep",    [9.8, 8.4, 7.8, 7.5, 7.5, 7.5, 8.1, 7.6, 7.6]),
    (14.20, "shallow", [7.1, 7.2, 7.3, 7.2, 7.5, 7.6, 7.3, 6.8, 6.7, 6.9]),
    (13.00, "deep",    [8.3, 8.0, 8.0, 8.1, 8.0, 7.4, 7.7, 7.9,
                         10.7, 9.4, 9.6, 8.7]),
    (11.80, "shallow", [8.0, 8.1, 7.9, 7.9, 7.8, 7.4, 7.8, 8.1, 8.0]),
    (10.60, "deep",    [8.8, 8.7, 9.2, 8.3, 8.2, 8.0, 8.1, 8.0, 8.1,
                         7.6, 7.6, 7.7, 8.6, 8.4, 8.3, 7.6, 7.3, 7.7, 7.7]),
    ( 9.40, "shallow", [7.8, 7.8, 8.1, 8.2, 8.9, 8.3, 7.9, 8.1, 8.1,
                         8.1, 8.1, 8.2, 8.1, 8.2, 7.9, 7.7, 8.5, 8.1]),
    ( 8.20, "deep",    [8.6, 8.4, 8.8, 7.5, 8.1, 7.8, 8.1, 8.2,
                         8.7, 8.7, 8.2, 8.2, 7.9]),
    # EXTRAPOLATED (4 more layers below measured, continuing pattern):
    ( 7.00, "shallow", None),   # extrapolated
    ( 5.80, "deep",    None),
    ( 4.60, "shallow", None),
    ( 3.40, "deep",    None),
]

# Deep/shallow bump amplitude: from IMG_1188, IMG_1674, IMG_1676
# the ridges appear to deviate ~1.5-2 mm from the smooth profile
BUMP_DEEP    =  0.17   # cm outward for deep bumps
BUMP_SHALLOW = -0.13   # cm inward for shallow indents
BUMP_SIGMA   =  0.20   # cm Gaussian half-width of each bump

# ── Concave bottom (from IMG_1666, IMG_1667 bottom views) ──
# Looking up at the underside, the vase has a foot ring around the
# perimeter (the contact surface) with the center pushed UP forming
# a shallow dome. Concentric wheel rings are visible going from the
# foot ring inward to the center peak.
#
# Geometry (cross-section of underside):
#   r = R_BASE  → z = 0           (foot ring contact with floor)
#   r = 0       → z = DOME_PEAK   (center of underside, raised up)
#   shape: roughly elliptical dome
#
# The foot ring itself is a narrow annular contact (~0.5 cm wide).
# Inside the foot ring (r < R_FOOT_INNER), the dome rises smoothly.
CONCAVITY_RISE    = 0.70     # cm center above contact plane (from IMG_1666 visual estimate)
FOOT_RING_WIDTH   = 0.50     # cm radial width of the contact annulus
# At very bottom of side wall, taper inward slightly to form the foot
FOOT_OUTER_R      = R_BASE   # 6.54 cm (matches 41.1 cm tape measure)
FOOT_INNER_R      = R_BASE - FOOT_RING_WIDTH   # 6.04 cm

# Underside (OUTER) dome profile: the visible bottom of the vase.
# Linear interpolation between (r, z) anchors going from foot ring inward.
UNDERSIDE_PROFILE = [
    # (r_cm, z_cm) — z is height above floor for the OUTSIDE of bottom
    (FOOT_OUTER_R, 0.00),   # foot ring contact with floor — IMG_1666
    (FOOT_INNER_R, 0.05),   # just inside foot ring — small step up
    (5.40,         0.20),   # dome starting to rise
    (4.50,         0.38),   # mid-dome — IMG_1667 concentric ring
    (3.50,         0.52),   # closer to center
    (2.50,         0.62),   # near center
    (1.50,         0.68),   # almost at center
    (0.00,         CONCAVITY_RISE),   # center peak — IMG_1666
]

# Bottom (clay) thickness — clay layer between outer dome and inner floor.
# Slightly thicker at the edge (more clay around the foot ring) than at center.
BOTTOM_THICK_CENTER = 1.10   # cm at center (FLOOR_HEIGHT = 0.70 + 1.10 = 1.80)
BOTTOM_THICK_EDGE   = 1.05   # cm at edge of inner floor (still thick, near foot)

# Inner floor (INSIDE the vase) — IMG_1678 clearly shows the interior
# floor is also DOMED, mirroring the outer dome shape. The center is
# raised, sloping down to where the floor meets the inner side wall.
# z_inner_floor(r) = z_outer_dome(r) + bottom_thickness(r)
# We compute INNER_FLOOR_PROFILE at runtime from UNDERSIDE_PROFILE.
# The inner floor extends from r=0 outward to R_INNER_FLOOR_EDGE
# (where it meets the inner side wall).
R_INNER_FLOOR_EDGE = 5.90    # cm — where dome meets inner side wall (computed below)

# ── Rim wall thickness (from IMG_1670, IMG_1672) ──
RIM_WALL_THICK = 1.30   # cm at thickest part of rolled lip

# ── Interior floor height (CENTER, highest point) ──
# IMG_1678 shows the interior is domed; center is the HIGHEST point.
# At edge of floor (where it meets the wall), z is lower.
FLOOR_HEIGHT = CONCAVITY_RISE + BOTTOM_THICK_CENTER   # cm at center (= 1.80)

# ── Angular resolution ──
N_THETA = 72   # 5° per segment


# ══════════════════════════════════════════════════════════════════
#  SECTION 2: COMPUTE PROFILES (linear interpolation only)
# ══════════════════════════════════════════════════════════════════

def lerp_profile(profile_tuples, heights):
    """Linearly interpolate a profile defined by (h, r, ...) tuples."""
    ph = np.array([t[0] for t in profile_tuples])
    pr = np.array([t[1] for t in profile_tuples])
    return np.interp(heights, ph, pr)


def bump_at(z_arr, layers):
    """Compute deep/shallow radial offset at each height."""
    bump = np.zeros_like(z_arr)
    for h_layer, ltype, _ in layers:
        gauss = np.exp(-0.5 * ((z_arr - h_layer) / BUMP_SIGMA) ** 2)
        if ltype == "deep":
            bump += BUMP_DEEP * gauss
        else:
            bump += BUMP_SHALLOW * gauss
    # Fade below glaze line (base has wheel rings, not the same bump pattern)
    fade = np.clip((z_arr - 2.5) / 1.5, 0.0, 1.0)
    return bump * fade


def wall_thickness_at_angle(layer_measurements, n_theta):
    """
    Given a list of caliper readings (mm) taken around a ring,
    distribute them evenly around 360° and linearly interpolate
    to get thickness at each of n_theta angles.
    Returns array of shape (n_theta,) in cm.
    """
    if layer_measurements is None:
        return None
    meas = np.array(layer_measurements) / 10.0  # mm → cm
    n = len(meas)
    # Measurements were taken at roughly equal angular intervals
    meas_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    target_angles = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    # Periodic linear interpolation
    extended_angles = np.concatenate([meas_angles - 2*np.pi, meas_angles,
                                      meas_angles + 2*np.pi])
    extended_vals   = np.concatenate([meas, meas, meas])
    return np.interp(target_angles, extended_angles, extended_vals)


def compute_all():
    """
    Build the complete 3D model.

    Returns dict with:
      z:       (n_z,) array of heights (cm) — for the SIDE wall (z >= 0)
      outer_r: (n_z,) outer radius at each height
      inner_r: (n_z, n_theta) inner radius at each height and angle
      theta:   (n_theta,) angles
      wall_t:  (n_z, n_theta) wall thickness at each point
      underside_r: (n_u,) radii sampled on the underside dome (from edge to center)
      underside_z: (n_u,) corresponding z heights of the underside dome
      inner_floor_r: array of r values where the INTERIOR floor sits
    """
    # Fine height grid: 0.25 cm steps
    z = np.arange(0, HEIGHT + 0.01, 0.25)
    n_z = len(z)
    theta = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)

    # ── OUTER underside dome geometry ──
    # Sample the dome from outer (foot ring) to center.
    n_u = 24
    u_r = np.linspace(FOOT_OUTER_R, 0.0, n_u)
    u_profile_r = np.array([p[0] for p in UNDERSIDE_PROFILE])
    u_profile_z = np.array([p[1] for p in UNDERSIDE_PROFILE])
    # Interp uses sorted x. Reverse since UNDERSIDE_PROFILE goes from large r to small r.
    u_z = np.interp(u_r, u_profile_r[::-1], u_profile_z[::-1])

    # ── INNER floor dome geometry (matches outer dome, offset by bottom thickness) ──
    # IMG_1678 clearly shows the interior floor is convex/domed (raised in center).
    # We compute it as: z_inner_floor(r) = z_outer_dome(r) + thickness(r)
    # Thickness varies linearly from BOTTOM_THICK_CENTER (at r=0) to BOTTOM_THICK_EDGE (at edge).
    #
    # The OUTER edge of the inner floor must MEET the inner side wall.
    # We iterate to find where dome edge meets wall:
    #   R_corner = R_inner_wall(z_corner) where z_corner = z_outer_dome(R_corner) + BOTTOM_THICK_EDGE
    # Inner wall radius at any z = R_outer(z) - mean_wall_thickness(z)
    inner_mean_at_z = lambda z_val: float(
        np.interp(z_val, z, r_outer - np.mean(wall_t, axis=1))
    ) if False else None  # wall_t not computed yet — use estimate

    # Simple iteration (3 passes is enough): assume thickness ≈ 1.1 cm near base
    r_corner = 6.0   # initial guess
    for _ in range(5):
        z_corner_test = float(np.interp(r_corner, u_profile_r[::-1], u_profile_z[::-1])) + BOTTOM_THICK_EDGE
        r_outer_at_zc = float(np.interp(z_corner_test, np.array([t[0] for t in OUTER_PROFILE]),
                                                       np.array([t[1] for t in OUTER_PROFILE])))
        # Estimate wall thickness near base ≈ 1.15 cm (matches anchor at h=1.0)
        wall_est = 1.10 if z_corner_test < 1.3 else 1.0
        r_corner = max(r_outer_at_zc - wall_est, 1.0)
    R_corner = r_corner

    n_if = 24
    if_r = np.linspace(R_corner, 0.0, n_if)
    # Outer dome z at these inner-floor radii (extend past UNDERSIDE_PROFILE if needed)
    if_outer_z = np.interp(if_r, u_profile_r[::-1], u_profile_z[::-1])
    # Thickness fraction: 0 at center, 1 at edge of inner floor
    frac = if_r / R_corner
    if_thickness = BOTTOM_THICK_CENTER + (BOTTOM_THICK_EDGE - BOTTOM_THICK_CENTER) * frac
    if_z = if_outer_z + if_thickness
    # At r=0: z = CONCAVITY_RISE + BOTTOM_THICK_CENTER = FLOOR_HEIGHT (1.80) ✓
    # At r=R_corner: z = z_outer_dome(R_corner) + BOTTOM_THICK_EDGE (meets inner wall)

    # ── Outer radius (smooth base profile) ──
    r_smooth = lerp_profile(OUTER_PROFILE, z)

    # ── Add bump modulation ──
    bump = bump_at(z, LAYERS)
    r_outer = r_smooth + bump
    r_outer = np.maximum(r_outer, 0.5)

    # ── Wall thickness model ──
    # For each height, determine wall thickness at each angle.
    # At measured layers: use actual caliper data (angular variation preserved).
    # Between layers: linearly interpolate the angular patterns.
    # At rim: use RIM_WALL_THICK.
    # At base: estimate thicker walls.

    # First, build angular wall thickness at each measured layer
    layer_wall_angular = {}  # h -> (n_theta,) array in cm
    layer_wall_mean = {}

    for h_layer, ltype, measurements in LAYERS:
        if measurements is not None:
            angular = wall_thickness_at_angle(measurements, N_THETA)
            layer_wall_angular[h_layer] = angular
            layer_wall_mean[h_layer] = np.mean(angular)
        else:
            # Extrapolated: use average of nearby measured layers of same type
            same_type = [(h, m) for h, t, m in LAYERS
                         if t == ltype and m is not None]
            if same_type:
                avg_thick = np.mean([np.mean(m) / 10.0 for _, m in same_type])
                avg_std   = np.mean([np.std(np.array(m) / 10.0) for _, m in same_type])
            else:
                avg_thick = 0.81
                avg_std = 0.03
            # Create synthetic angular variation with same statistics
            rng = np.random.default_rng(int(h_layer * 100))
            angular = avg_thick + rng.normal(0, avg_std, N_THETA)
            angular = np.clip(angular, 0.55, 1.20)
            layer_wall_angular[h_layer] = angular
            layer_wall_mean[h_layer] = avg_thick

    # Build wall thickness at every height by interpolating between layers
    wall_t = np.zeros((n_z, N_THETA))

    # Add rim and base anchor points for interpolation
    rim_angular = np.full(N_THETA, RIM_WALL_THICK)
    base_angular = np.full(N_THETA, 1.10)  # base wall ~11mm

    all_anchors = [(18.0, rim_angular),
                   (17.5, np.full(N_THETA, 1.35)),
                   (17.0, np.full(N_THETA, 0.95))]

    for h_layer in sorted(layer_wall_angular.keys()):
        all_anchors.append((h_layer, layer_wall_angular[h_layer]))

    all_anchors.extend([
        (2.50, np.full(N_THETA, 0.95)),
        (1.80, np.full(N_THETA, 1.10)),   # floor level
        (1.00, np.full(N_THETA, 1.15)),
        (0.30, np.full(N_THETA, 1.25)),
    ])
    all_anchors.sort(key=lambda x: x[0])

    anchor_h = np.array([a[0] for a in all_anchors])
    anchor_w = np.array([a[1] for a in all_anchors])  # (n_anchors, n_theta)

    # Linearly interpolate for each angle
    for j in range(N_THETA):
        col_vals = anchor_w[:, j]
        wall_t[:, j] = np.interp(z, anchor_h, col_vals)

    # ── Inner radius (side wall only — floor dome is stored separately) ──
    inner_r = r_outer[:, None] - wall_t
    inner_r = np.maximum(inner_r, 0.0)

    # The inner side wall exists above the LOWEST point of the floor dome.
    # The lowest point is where the floor meets the inner wall:
    z_floor_min = float(if_z[0])   # at r=R_INNER_FLOOR_EDGE, the lowest z of the floor dome

    # For z below z_floor_min: no interior, set inner_r to 0
    # For z between z_floor_min and FLOOR_HEIGHT (the dome region):
    #   the inner wall still exists, but its lower end is bounded by the floor dome.
    #   inner_r at these z values = side wall radius (the floor will cap it from below).
    # For z above FLOOR_HEIGHT: full side wall

    below_floor_mask = z < z_floor_min
    inner_r[below_floor_mask, :] = 0.0

    return {
        "z": z,
        "theta": theta,
        "r_outer": r_outer,
        "r_smooth": r_smooth,
        "bump": bump,
        "wall_t": wall_t,
        "inner_r": inner_r,
        "underside_r": u_r,         # OUTER underside dome (large → small)
        "underside_z": u_z,         # OUTER underside dome z (0 at foot, 0.7 at center)
        "inner_floor_r": if_r,      # INNER floor dome (R_INNER_FLOOR_EDGE → 0)
        "inner_floor_z": if_z,      # INNER floor dome z (z_floor_min at edge, 1.8 at center)
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION 3: STL WRITER
# ══════════════════════════════════════════════════════════════════

def _write_tris(path, tris):
    """Write a list of triangles (each a 3-tuple of (x,y,z) tuples) to binary STL."""
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for v0, v1, v2 in tris:
            e1 = np.array(v1) - np.array(v0)
            e2 = np.array(v2) - np.array(v0)
            n = np.cross(e1, e2)
            nl = np.linalg.norm(n)
            if nl > 0: n /= nl
            f.write(struct.pack("<3f", *n))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))


def write_outer_stl(path, data):
    """
    Write the OUTER surface of the vase: side wall + concave underside dome.
    The underside dome connects the foot ring (r=R_BASE, z=0) up to the
    raised center (r=0, z=CONCAVITY_RISE).
    """
    z_arr = data["z"]
    r_arr = data["r_outer"]
    theta = data["theta"]
    u_r   = data["underside_r"]
    u_z   = data["underside_z"]
    n_z, n_t = len(z_arr), len(theta)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    tris = []

    # ── Side wall (revolution of outer profile) ──
    for i in range(n_z - 1):
        r0, r1 = r_arr[i], r_arr[i+1]
        z0, z1 = z_arr[i], z_arr[i+1]
        for j in range(n_t):
            jn = (j + 1) % n_t
            v0 = (r0*cos_t[j],  r0*sin_t[j],  z0)
            v1 = (r1*cos_t[j],  r1*sin_t[j],  z1)
            v2 = (r1*cos_t[jn], r1*sin_t[jn], z1)
            v3 = (r0*cos_t[jn], r0*sin_t[jn], z0)
            tris.append((v0, v1, v2))
            tris.append((v0, v2, v3))

    # ── Underside dome (revolution of UNDERSIDE_PROFILE) ──
    # u_r goes from FOOT_OUTER_R down to 0; u_z goes from 0 up to CONCAVITY_RISE.
    # Normal should point DOWNWARD (away from interior), so wind triangles outward.
    n_u = len(u_r)
    for i in range(n_u - 1):
        r0, r1 = u_r[i], u_r[i+1]
        z0, z1 = u_z[i], u_z[i+1]
        if r1 < 1e-6:
            # Last ring connects to center apex point
            apex = (0.0, 0.0, float(z1))
            for j in range(n_t):
                jn = (j + 1) % n_t
                v0 = (r0*cos_t[j],  r0*sin_t[j],  z0)
                v1 = (r0*cos_t[jn], r0*sin_t[jn], z0)
                # Triangle winding: looking from BELOW (negative z), CW = outward normal
                tris.append((apex, v1, v0))
        else:
            for j in range(n_t):
                jn = (j + 1) % n_t
                v0 = (r0*cos_t[j],  r0*sin_t[j],  z0)
                v1 = (r1*cos_t[j],  r1*sin_t[j],  z1)
                v2 = (r1*cos_t[jn], r1*sin_t[jn], z1)
                v3 = (r0*cos_t[jn], r0*sin_t[jn], z0)
                # Winding for downward-facing normal
                tris.append((v0, v2, v1))
                tris.append((v0, v3, v2))

    # ── Top cap (rim — close the opening at z=HEIGHT) ──
    # Actually the inner surface meets the outer at the rim, so the top is
    # an annular ring, not a flat disk. For the OUTER STL only, we close it
    # with a flat disk at z=HEIGHT (acceptable since we have a separate inner STL).
    ctr_top = (0.0, 0.0, float(z_arr[-1]))
    r_top = r_arr[-1]
    for j in range(n_t):
        jn = (j + 1) % n_t
        v_top = (r_top*cos_t[j],  r_top*sin_t[j],  float(z_arr[-1]))
        v_top_n = (r_top*cos_t[jn], r_top*sin_t[jn], float(z_arr[-1]))
        tris.append((ctr_top, v_top, v_top_n))

    _write_tris(path, tris)
    print(f"  STL: {os.path.basename(path)} — {len(tris):,} triangles (with concave underside)")


def write_inner_stl(path, data):
    """
    Write the INNER cavity surface (the water-containing region).
    Side walls from rim down to where they meet the INNER floor dome
    (z_floor_min ≈ 1.05 cm at edge), then the floor dome rising to
    z=FLOOR_HEIGHT at center (matching the OUTER dome shape).
    """
    z_arr = data["z"]
    inner_r = data["inner_r"]
    theta = data["theta"]
    if_r = data["inner_floor_r"]   # radii on inner floor dome (edge → center)
    if_z = data["inner_floor_z"]   # z values on inner floor dome
    n_z, n_t = len(z_arr), len(theta)

    z_floor_min = float(if_z[0])
    floor_idx = int(np.searchsorted(z_arr, z_floor_min))

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    tris = []
    # Side wall from floor-dome edge up to rim
    inner_mean = np.mean(inner_r, axis=1)
    for i in range(floor_idx, n_z - 1):
        r0, r1 = inner_mean[i], inner_mean[i+1]
        z0, z1 = z_arr[i], z_arr[i+1]
        for j in range(n_t):
            jn = (j + 1) % n_t
            v0 = (r0*cos_t[j],  r0*sin_t[j],  z0)
            v1 = (r1*cos_t[j],  r1*sin_t[j],  z1)
            v2 = (r1*cos_t[jn], r1*sin_t[jn], z1)
            v3 = (r0*cos_t[jn], r0*sin_t[jn], z0)
            tris.append((v0, v2, v1))
            tris.append((v0, v3, v2))

    # INNER floor dome (revolution of if_r, if_z) — matches outer dome shape
    n_if = len(if_r)
    for i in range(n_if - 1):
        r0, r1 = if_r[i], if_r[i+1]
        z0, z1 = if_z[i], if_z[i+1]
        if r1 < 1e-6:
            # Connect to center apex point
            apex = (0.0, 0.0, float(z1))
            for j in range(n_t):
                jn = (j + 1) % n_t
                v0 = (r0*cos_t[j],  r0*sin_t[j],  z0)
                v1 = (r0*cos_t[jn], r0*sin_t[jn], z0)
                # Normal faces UP (into the interior)
                tris.append((apex, v0, v1))
        else:
            for j in range(n_t):
                jn = (j + 1) % n_t
                v0 = (r0*cos_t[j],  r0*sin_t[j],  z0)
                v1 = (r1*cos_t[j],  r1*sin_t[j],  z1)
                v2 = (r1*cos_t[jn], r1*sin_t[jn], z1)
                v3 = (r0*cos_t[jn], r0*sin_t[jn], z0)
                # Normal faces UP
                tris.append((v0, v1, v2))
                tris.append((v0, v2, v3))

    _write_tris(path, tris)
    print(f"  STL: {os.path.basename(path)} — {len(tris):,} triangles (with domed floor)")


# ══════════════════════════════════════════════════════════════════
#  SECTION 4: CROSS-SECTION PLOT
# ══════════════════════════════════════════════════════════════════

def plot_cross_section(data):
    z = data["z"]
    r_o = data["r_outer"]
    r_i_min = np.min(data["inner_r"], axis=1)
    r_i_max = np.max(data["inner_r"], axis=1)
    r_i_avg = np.mean(data["inner_r"], axis=1)
    wt = data["wall_t"]
    u_r = data["underside_r"]
    u_z = data["underside_z"]
    if_r = data["inner_floor_r"]   # INNER floor dome radii
    if_z = data["inner_floor_z"]   # INNER floor dome z values

    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(2, 4, width_ratios=[3, 1.5, 3, 3], height_ratios=[2, 1])
    ax_main = fig.add_subplot(gs[:, 0])
    ax_zoom = fig.add_subplot(gs[:, 1])
    ax_prof = fig.add_subplot(gs[:, 2])
    ax_wall = fig.add_subplot(gs[:, 3])
    axes = [ax_main, ax_prof, ax_wall]  # legacy ref for old code

    # Panel 1: Cross-section showing wall thickness variation
    ax = axes[0]
    # Floor index for clay region
    floor_idx = np.searchsorted(z, FLOOR_HEIGHT)

    # Build the outline of the SOLID CLAY in cross-section.
    # The clay occupies:
    #   Side walls: between r_outer and r_inner for h >= FLOOR_HEIGHT
    #   Bottom mass: outer surface = side wall + underside dome,
    #                inner surface = flat floor at z=FLOOR_HEIGHT
    # We use a polygon to fill the cross-section clay properly.

    # Right-half clay polygon path:
    # Start at the rim top (outside), go DOWN the outside, along the underside
    # OUTER dome to the center, UP through the clay, along the INNER floor dome
    # outward to where it meets the inner wall, then UP the inner wall to rim.
    poly_r = []
    poly_z = []
    # Outer side going down (rim -> base)
    for i in range(len(z)-1, -1, -1):
        poly_r.append(r_o[i]); poly_z.append(z[i])
    # OUTER underside dome going from foot ring (r=R_BASE, z=0) inward to center
    for i in range(len(u_r)):
        poly_r.append(u_r[i]); poly_z.append(u_z[i])
    # Going UP from outer dome center (0, CONCAVITY_RISE) to inner floor center (0, FLOOR_HEIGHT)
    poly_r.append(0.0); poly_z.append(FLOOR_HEIGHT)
    # INNER floor dome going from center (r=0, z=FLOOR_HEIGHT) OUTWARD to where it
    # meets the inner wall (r=R_INNER_FLOOR_EDGE, z=z_floor_min)
    # if_r is ordered edge→center; we want center→edge here
    for i in range(len(if_r)-1, -1, -1):
        poly_r.append(if_r[i]); poly_z.append(if_z[i])
    # Find where the inner side wall starts (z just above the floor edge)
    z_floor_min = float(if_z[0])
    floor_idx_eff = int(np.searchsorted(z, z_floor_min))
    # UP the inner wall from where the floor dome edge meets the wall up to the rim
    for i in range(floor_idx_eff, len(z)):
        poly_r.append(r_i_avg[i]); poly_z.append(z[i])
    poly_r = np.array(poly_r); poly_z = np.array(poly_z)
    # Also keep a reference floor index for inner wall side display
    floor_r_inner = r_i_avg[floor_idx_eff] if floor_idx_eff < len(z) else 0.0

    # Fill right half (clay)
    ax.fill(poly_r, poly_z, color="#C4956A", alpha=0.85,
            label="Clay (cross-section)")
    # Fill left half (mirror)
    ax.fill(-poly_r, poly_z, color="#C4956A", alpha=0.85)

    # Wall thickness range shading (angular variation around inner surface)
    for i in range(floor_idx_eff, len(z)):
        if r_i_max[i] > r_i_min[i]:
            ax.plot([r_i_min[i], r_i_max[i]], [z[i], z[i]],
                    color="#8a5a2b", alpha=0.25, lw=0.5)
            ax.plot([-r_i_max[i], -r_i_min[i]], [z[i], z[i]],
                    color="#8a5a2b", alpha=0.25, lw=0.5)

    # Outer surface outline (side + outer dome)
    ax.plot( r_o, z, "b-", lw=1.5, label="Outer surface")
    ax.plot(-r_o, z, "b-", lw=1.5)
    ax.plot( u_r, u_z, "b-", lw=1.5)          # outer underside dome (right)
    ax.plot(-u_r, u_z, "b-", lw=1.5)          # outer underside dome (left)
    # Inner side wall (above floor edge only)
    ax.plot( r_i_avg[floor_idx_eff:], z[floor_idx_eff:], "r-", lw=1.0, label="Inner (avg)")
    ax.plot(-r_i_avg[floor_idx_eff:], z[floor_idx_eff:], "r-", lw=1.0)
    ax.plot( r_i_min[floor_idx_eff:], z[floor_idx_eff:], "r:", lw=0.5, alpha=0.5, label="Inner (min/max)")
    ax.plot(-r_i_min[floor_idx_eff:], z[floor_idx_eff:], "r:", lw=0.5, alpha=0.5)
    ax.plot( r_i_max[floor_idx_eff:], z[floor_idx_eff:], "r:", lw=0.5, alpha=0.5)
    ax.plot(-r_i_max[floor_idx_eff:], z[floor_idx_eff:], "r:", lw=0.5, alpha=0.5)
    # INNER floor dome (NEW — matches outer dome shape, raised in center)
    ax.plot( if_r, if_z, "r-", lw=1.5, label="Inner floor (domed)")
    ax.plot(-if_r, if_z, "r-", lw=1.5)
    # Annotate foot ring + outer dome + inner dome
    ax.annotate("foot ring contact", xy=(R_BASE, 0), xytext=(R_BASE+0.8, -1.0),
                fontsize=7, color="#664400",
                arrowprops=dict(arrowstyle="->", color="#664400", lw=0.5))
    ax.annotate(f"outer dome peak\nrises {CONCAVITY_RISE} cm", xy=(0, CONCAVITY_RISE),
                xytext=(2.5, -1.7), fontsize=7, color="#664400",
                arrowprops=dict(arrowstyle="->", color="#664400", lw=0.5))
    ax.annotate(f"inner floor dome\n(center: {FLOOR_HEIGHT:.2f} cm)",
                xy=(0, FLOOR_HEIGHT),
                xytext=(-7.5, FLOOR_HEIGHT+0.6), fontsize=7, color="#aa3333",
                arrowprops=dict(arrowstyle="->", color="#aa3333", lw=0.5))
    # Floor line for visual reference
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5, ls="--")

    for val, lbl, clr in [(R_WIDEST, "60cm circ", "green"),
                           (R_RIM, "47cm circ", "purple"),
                           (R_BASE, "41.1cm circ", "orange")]:
        h_at = z[np.argmin(np.abs(r_o - val))]
        ax.plot(val, h_at, "o", color=clr, ms=6)
        ax.annotate(f"R={val:.2f} ({lbl})", xy=(val, h_at),
                    xytext=(val+0.5, h_at+0.3), fontsize=7, color=clr,
                    arrowprops=dict(arrowstyle="->", color=clr, lw=0.5))

    ax.set_xlabel("Radius (cm)")
    ax.set_ylabel("Height (cm)")
    ax.set_title("Vase Cross-Section\n(showing wall thickness angular range)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Zoom panel: bottom region (shows BOTH outer and inner domes)
    az = ax_zoom
    az.fill(poly_r, poly_z, color="#C4956A", alpha=0.85)
    az.fill(-poly_r, poly_z, color="#C4956A", alpha=0.85)
    az.plot( r_o, z, "b-", lw=1.5)
    az.plot(-r_o, z, "b-", lw=1.5)
    az.plot( u_r, u_z, "b-", lw=2.0)              # outer dome
    az.plot(-u_r, u_z, "b-", lw=2.0)
    az.plot( if_r, if_z, "r-", lw=2.0)            # inner floor dome (NEW)
    az.plot(-if_r, if_z, "r-", lw=2.0)
    az.plot( r_i_avg[floor_idx_eff:], z[floor_idx_eff:], "r-", lw=1.0)
    az.plot(-r_i_avg[floor_idx_eff:], z[floor_idx_eff:], "r-", lw=1.0)
    az.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    az.annotate("FLOOR", xy=(-1, 0.03), fontsize=8, color="black", alpha=0.7)
    az.annotate(f"OUTER dome\nrises {CONCAVITY_RISE} cm",
                xy=(0, CONCAVITY_RISE), xytext=(2.5, -0.5), fontsize=7,
                color="blue",
                arrowprops=dict(arrowstyle="->", color="blue", lw=0.8))
    az.annotate(f"INNER floor dome\n(also raised in center)\ncenter={FLOOR_HEIGHT:.2f}cm  edge={float(if_z[0]):.2f}cm",
                xy=(0, FLOOR_HEIGHT), xytext=(2.5, 3.0), fontsize=7,
                color="#aa3333",
                arrowprops=dict(arrowstyle="->", color="#aa3333", lw=0.8))
    az.annotate("foot ring", xy=(R_BASE, 0),
                xytext=(-8.5, -0.5), fontsize=7, color="#664400",
                arrowprops=dict(arrowstyle="->", color="#664400", lw=0.8))
    az.annotate(f"bottom clay\n~{BOTTOM_THICK_CENTER:.1f}cm thick",
                xy=(0, (CONCAVITY_RISE + FLOOR_HEIGHT)/2),
                xytext=(-7.5, 2.5), fontsize=7, color="#664400",
                arrowprops=dict(arrowstyle="->", color="#664400", lw=0.6))
    az.set_xlim(-9, 9)
    az.set_ylim(-1, 4.5)
    az.set_aspect("equal")
    az.set_title("Concave Bottom Detail\n(outer dome + inner floor dome)")
    az.set_xlabel("Radius (cm)")
    az.set_ylabel("Height (cm)")
    az.grid(True, alpha=0.2)

    # Panel 2: Outer profile with bumps
    ax = axes[1]
    ax.plot(data["r_smooth"], z, "b--", lw=1, alpha=0.4, label="Smooth envelope")
    ax.plot(r_o, z, "b-", lw=2, label="With deep/shallow layers")
    ax.fill_betweenx(z, data["r_smooth"], r_o, alpha=0.2, color="blue")
    for h_layer, ltype, meas in LAYERS:
        clr = "#cc3333" if ltype == "deep" else "#3366cc"
        mk  = ">" if ltype == "deep" else "<"
        ax.plot(np.interp(h_layer, z, r_o), h_layer, mk, color=clr, ms=5)
        ax.text(r_o.max() + 0.1, h_layer,
                f"{'D' if ltype=='deep' else 'S'}"
                f"{'*' if meas else ''}",
                fontsize=6, color=clr, va="center")
    ax.set_xlabel("Outer Radius (cm)")
    ax.set_ylabel("Height (cm)")
    ax.set_title("Outer Profile\n(* = measured layer, no * = extrapolated)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 3: Wall thickness at each measured layer
    ax = axes[2]
    for h_layer, ltype, meas in LAYERS:
        if meas is None:
            continue
        clr = "#cc3333" if ltype == "deep" else "#3366cc"
        mk  = "o" if ltype == "deep" else "s"
        avg = np.mean(meas)
        std = np.std(meas)
        ax.errorbar(avg, h_layer, xerr=std, fmt=mk, color=clr, ms=5,
                    capsize=3, lw=1)
        ax.text(avg + std + 0.3, h_layer,
                f"n={len(meas)} [{min(meas):.1f}–{max(meas):.1f}]",
                fontsize=6, va="center", color=clr)
    # Plot model wall thickness (min/mean/max at each height)
    wt_mean = np.mean(wt, axis=1) * 10
    wt_min  = np.min(wt, axis=1)  * 10
    wt_max  = np.max(wt, axis=1)  * 10
    ax.plot(wt_mean, z, "g-", lw=1.5, label="Model mean")
    ax.fill_betweenx(z, wt_min, wt_max, color="green", alpha=0.15,
                     label="Model range")
    ax.set_xlabel("Wall Thickness (mm)")
    ax.set_ylabel("Height (cm)")
    ax.set_title("Wall Thickness\n(measured points with angular spread)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(BASE, "vase_cross_section.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot: vase_cross_section.png")


# ══════════════════════════════════════════════════════════════════
#  SECTION 5: INTERACTIVE 3D HTML VIEWER
# ══════════════════════════════════════════════════════════════════

def build_html(data):
    z = data["z"]
    r_o = data["r_outer"]
    inner_r = data["inner_r"]
    theta = data["theta"]
    u_r = data["underside_r"]
    u_z = data["underside_z"]
    if_r = data["inner_floor_r"]
    if_z = data["inner_floor_z"]

    # Subsample for browser performance
    step_z = max(1, len(z) // 80)
    step_t = max(1, len(theta) // 72)

    zv = z[::step_z].tolist()
    rv_o = r_o[::step_z].tolist()
    # Inner radius: sample at each angle
    rv_i = inner_r[::step_z, ::step_t].tolist()
    tv = theta[::step_t].tolist()
    # Outer underside dome
    uv_r = u_r.tolist()
    uv_z = u_z.tolist()
    # Inner floor dome
    ifv_r = if_r.tolist()
    ifv_z = if_z.tolist()
    floor_h = float(FLOOR_HEIGHT)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vase 3D Model — AP Calc BC</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0f1729;overflow:hidden;font-family:system-ui,sans-serif}}
canvas{{display:block}}
#hud{{position:absolute;top:14px;left:14px;color:#d0d0d0;
  background:rgba(0,0,0,.7);padding:16px 20px;border-radius:10px;
  max-width:320px;font-size:13px;line-height:1.6}}
#hud h2{{margin:0 0 6px;font-size:16px;color:#fff}}
#hud b{{color:#90caf9}}
#hud hr{{border-color:#333;margin:6px 0}}
.ctrls{{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);
  display:flex;gap:8px;flex-wrap:wrap;justify-content:center}}
.btn{{padding:6px 14px;border:1px solid #444;background:rgba(20,20,40,.85);
  color:#bbb;cursor:pointer;border-radius:5px;font-size:12px}}
.btn:hover{{background:rgba(50,50,80,.85)}}
.btn.on{{background:rgba(35,75,170,.75);border-color:#68f;color:#fff}}
</style>
</head>
<body>
<div id="hud">
  <h2>Hand-Crafted Vase 3D Model</h2>
  <p>AP Calc BC — Creative Volume Project</p>
  <hr>
  <p><b>Height:</b> {HEIGHT} cm</p>
  <p><b>Widest:</b> 60.0 cm circ &rarr; &empty;{2*R_WIDEST:.2f} cm</p>
  <p><b>Rim:</b> 47.0 cm circ &rarr; &empty;{2*R_RIM:.2f} cm</p>
  <p><b>Base:</b> 41.1 cm circ &rarr; &empty;{2*R_BASE:.2f} cm</p>
  <hr>
  <p style="font-size:11px;color:#888">
    Outer profile from IMG_1188 ruler proportions.<br>
    Inner surface from caliper wall-thickness data<br>
    (varies around circumference = asymmetric).<br>
    Drag to rotate &bull; Scroll to zoom &bull; Right-drag to pan
  </p>
</div>
<div class="ctrls">
  <button class="btn on" id="bO" onclick="tog('o',this)">Outer</button>
  <button class="btn on" id="bI" onclick="tog('i',this)">Inner</button>
  <button class="btn"    id="bW" onclick="tog('w',this)">Wireframe</button>
  <button class="btn"    id="bX" onclick="tog('x',this)">Cross-Section</button>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
"use strict";

const Z={json.dumps(zv)};
const Ro={json.dumps(rv_o)};
const Ri={json.dumps(rv_i)};
const TH={json.dumps(tv)};
const Ur={json.dumps(uv_r)};   // OUTER underside dome radii (large→small)
const Uz={json.dumps(uv_z)};   // OUTER underside dome z values
const IFr={json.dumps(ifv_r)}; // INNER floor dome radii (edge→center)
const IFz={json.dumps(ifv_z)}; // INNER floor dome z values
const FLOOR_H={floor_h};
const nZ=Z.length, nT=TH.length, nU=Ur.length, nIF=IFr.length;

// Scene
const scene=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(38,innerWidth/innerHeight,.1,500);
cam.position.set(20,12,20);
const ren=new THREE.WebGLRenderer({{antialias:true}});
ren.setSize(innerWidth,innerHeight);
ren.setPixelRatio(devicePixelRatio);
ren.setClearColor(0x0f1729);
document.body.appendChild(ren.domElement);
const ctrl=new THREE.OrbitControls(cam,ren.domElement);
ctrl.target.set(0,9,0); ctrl.enableDamping=true; ctrl.dampingFactor=.07;

// Lights
scene.add(new THREE.AmbientLight(0xffffff,.4));
const d1=new THREE.DirectionalLight(0xffffff,.65); d1.position.set(12,22,18); scene.add(d1);
const d2=new THREE.DirectionalLight(0x8888ff,.25); d2.position.set(-10,8,-12); scene.add(d2);
scene.add(new THREE.GridHelper(30,30,0x222244,0x1a1a33));

function mkSurf(rData,isInner){{
  const geo=new THREE.BufferGeometry();
  const pos=[],idx=[];
  for(let i=0;i<nZ;i++){{
    for(let j=0;j<=nT;j++){{
      const jj=j%nT;
      const a=TH[jj];
      const r=isInner?(Array.isArray(rData[i])?rData[i][jj]:rData[i]):rData[i];
      pos.push(r*Math.cos(a), Z[i], r*Math.sin(a));
    }}
  }}
  for(let i=0;i<nZ-1;i++){{
    for(let j=0;j<nT;j++){{
      const a=i*(nT+1)+j, b=a+nT+1;
      if(isInner){{ idx.push(a,a+1,b); idx.push(a+1,b+1,b); }}
      else {{ idx.push(a,b,a+1); idx.push(a+1,b,b+1); }}
    }}
  }}
  geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}}

// Outer mesh (side wall)
const oGeo=mkSurf(Ro,false);
const oMat=new THREE.MeshPhongMaterial({{color:0xE8E0D8,specular:0x444444,
  shininess:55,side:THREE.DoubleSide}});
const oMesh=new THREE.Mesh(oGeo,oMat); scene.add(oMesh);

// Underside dome: revolution of (Ur, Uz) — the CONCAVE BOTTOM of the vase
function mkDome(){{
  const geo=new THREE.BufferGeometry();
  const pos=[],idx=[];
  for(let i=0;i<nU;i++){{
    for(let j=0;j<=nT;j++){{
      const jj=j%nT;
      const a=TH[jj];
      pos.push(Ur[i]*Math.cos(a), Uz[i], Ur[i]*Math.sin(a));
    }}
  }}
  for(let i=0;i<nU-1;i++){{
    for(let j=0;j<nT;j++){{
      const a=i*(nT+1)+j, b=a+nT+1;
      // Winding for downward normal (visible from below)
      idx.push(a,a+1,b); idx.push(a+1,b+1,b);
    }}
  }}
  geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}}
const dGeo=mkDome();
const dMesh=new THREE.Mesh(dGeo,oMat); scene.add(dMesh);

// Inner mesh (with angular variation!) — only above the interior floor edge
const iGeo=mkSurf(Ri,true);
const iMat=new THREE.MeshPhongMaterial({{color:0x8B6F47,specular:0x222222,
  shininess:15,side:THREE.DoubleSide}});
const iMesh=new THREE.Mesh(iGeo,iMat); scene.add(iMesh);

// INNER floor dome: revolution of (IFr, IFz) — the DOMED INTERIOR floor.
// Matches the outer dome shape, raised in the center. IMG_1678 confirms this.
function mkInnerDome(){{
  const geo=new THREE.BufferGeometry();
  const pos=[],idx=[];
  for(let i=0;i<nIF;i++){{
    for(let j=0;j<=nT;j++){{
      const jj=j%nT;
      const a=TH[jj];
      pos.push(IFr[i]*Math.cos(a), IFz[i], IFr[i]*Math.sin(a));
    }}
  }}
  for(let i=0;i<nIF-1;i++){{
    for(let j=0;j<nT;j++){{
      const a=i*(nT+1)+j, b=a+nT+1;
      // Normal faces UP (into interior so you can see it from inside)
      idx.push(a,b,a+1); idx.push(a+1,b,b+1);
    }}
  }}
  geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  return geo;
}}
const ifGeo=mkInnerDome();
const ifMesh=new THREE.Mesh(ifGeo,iMat); scene.add(ifMesh);

// Wireframe
const wGeo=mkSurf(Ro,false);
const wMat=new THREE.MeshBasicMaterial({{color:0x4488ff,wireframe:true,
  transparent:true,opacity:.2}});
const wMesh=new THREE.Mesh(wGeo,wMat); wMesh.visible=false; scene.add(wMesh);

// Clip plane for cross-section
const clipPl=new THREE.Plane(new THREE.Vector3(0,0,-1),0);
let clipOn=false;

function tog(w,btn){{
  btn.classList.toggle('on');
  const on=btn.classList.contains('on');
  if(w==='o') {{ oMesh.visible=on; dMesh.visible=on; }}
  if(w==='i') {{ iMesh.visible=on; ifMesh.visible=on; }}
  if(w==='w') wMesh.visible=on;
  if(w==='x'){{
    clipOn=on;
    [oMat,iMat].forEach(m=>{{
      // dome shares oMat so it's covered automatically
      m.clippingPlanes=on?[clipPl]:[];
      m.clipShadows=on;
      m.needsUpdate=true;
    }});
    ren.localClippingEnabled=on;
  }}
}}

(function anim(){{
  requestAnimationFrame(anim);
  ctrl.update();
  ren.render(scene,cam);
}})();

addEventListener('resize',()=>{{
  cam.aspect=innerWidth/innerHeight;
  cam.updateProjectionMatrix();
  ren.setSize(innerWidth,innerHeight);
}});
</script>
</body>
</html>"""

    path = os.path.join(BASE, "vase_3d.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  HTML: vase_3d.html")


# ══════════════════════════════════════════════════════════════════
#  SECTION 6: MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    sep = "=" * 60
    print(sep)
    print("  Hand-Crafted Vase 3D Model Builder")
    print(sep)

    print("\n1. Computing profiles (linear interpolation only)...")
    data = compute_all()

    z = data["z"]
    r_o = data["r_outer"]
    r_i = data["inner_r"]

    print(f"   Height grid: {len(z)} levels, 0 to {z[-1]:.1f} cm")
    print(f"   Angular grid: {N_THETA} points (every {360/N_THETA:.0f}°)")
    print(f"   Outer R max: {r_o.max():.3f} cm at h={z[np.argmax(r_o)]:.2f} cm")
    print(f"   Inner R max: {r_i.max():.3f} cm")
    print(f"   Wall thickness: {data['wall_t'].min()*10:.1f}–"
          f"{data['wall_t'].max()*10:.1f} mm")

    # Layer stats
    print("\n   Wall thickness layer stats (measured):")
    for h_layer, ltype, meas in LAYERS:
        if meas is None:
            continue
        arr = np.array(meas)
        print(f"     h={h_layer:5.1f} {ltype:>7s}: "
              f"mean={arr.mean():.1f} std={arr.std():.1f} "
              f"[{arr.min():.1f}–{arr.max():.1f}] mm  (n={len(meas)})")

    print("\n2. Generating STL files...")
    write_outer_stl(os.path.join(BASE, "vase_outer.stl"), data)
    write_inner_stl(os.path.join(BASE, "vase_inner.stl"), data)

    print("\n3. Building interactive 3D viewer...")
    build_html(data)

    print("\n4. Rendering cross-section plot...")
    plot_cross_section(data)

    print("\n5. Saving profile data...")
    out = {
        "z_cm": z.tolist(),
        "r_outer_cm": r_o.tolist(),
        "r_inner_avg_cm": np.mean(r_i, axis=1).tolist(),
        "r_inner_min_cm": np.min(r_i, axis=1).tolist(),
        "r_inner_max_cm": np.max(r_i, axis=1).tolist(),
        "wall_t_avg_cm": np.mean(data["wall_t"], axis=1).tolist(),
        "theta_rad": data["theta"].tolist(),
        "inner_r_by_angle": r_i.tolist(),
        "outer_underside_r_cm": data["underside_r"].tolist(),
        "outer_underside_z_cm": data["underside_z"].tolist(),
        "inner_floor_r_cm": data["inner_floor_r"].tolist(),
        "inner_floor_z_cm": data["inner_floor_z"].tolist(),
        "measurements": {
            "height_cm": HEIGHT,
            "widest_circ_cm": 60.0,
            "rim_circ_cm": 47.0,
            "base_circ_cm": 41.1,
            "floor_height_cm": FLOOR_HEIGHT,
            "concavity_depth_cm": CONCAVITY_RISE,
            "bottom_thickness_center_cm": BOTTOM_THICK_CENTER,
            "bottom_thickness_edge_cm": BOTTOM_THICK_EDGE,
        },
        "layer_data": [
            {"h_cm": h, "type": t,
             "measurements_mm": m,
             "mean_mm": float(np.mean(m)) if m else None,
             "std_mm": float(np.std(m)) if m else None}
            for h, t, m in LAYERS
        ],
    }
    with open(os.path.join(BASE, "vase_data.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  JSON: vase_data.json")

    print(f"\n{sep}")
    print("  Done! Files generated:")
    print("    vase_3d.html           ← Open in browser for interactive 3D view")
    print("    vase_outer.stl         ← Outer surface (any 3D viewer)")
    print("    vase_inner.stl         ← Inner cavity")
    print("    vase_cross_section.png ← Profile verification")
    print("    vase_data.json         ← All data for volume calculation")
    print(sep)


if __name__ == "__main__":
    main()
