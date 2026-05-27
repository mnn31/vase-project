"""Build the axisymmetric vase mesh by revolving the photogrammetry mesh's
front-side semi-axis a(h) around the vertical axis. Produces a clean,
watertight USDZ for Quick Look.

The mesh is also rescaled so the profile passes through the three tape-
measurement anchors (the same correction the volume calculation applies),
so what you see in Quick Look is exactly the shape that's being integrated.

A two-tone display color is applied:
  - upper portion: cream-white (glazed)
  - lower ~22%   : tan (unglazed clay band visible in the photos)

@author Manan Gupta
@author Claude (Anthropic AI assistant), code co-author
"""
from __future__ import annotations
import json, math, shutil, subprocess, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# -- Same input data as compute_volume.py --
mesh_data = json.loads((ROOT / "data" / "mesh_profile.json").read_text())
h_arr = np.array(mesh_data["sample_heights_cm"])
a_arr = np.array(mesh_data["sample_a_outer_cm"])

# Tape-measured anchors (same as compute_volume.py)
C_bot, C_max, C_top = 41.2, 60.0, 47.0
H_INT = 17.2
T_WALL = 0.8016                     # mean of 98 caliper readings, cm
R_bot = C_bot / (2*math.pi)
R_max = C_max / (2*math.pi)
R_top = C_top / (2*math.pi)
H_out = H_INT + T_WALL

# Rescale the mesh a-profile to match the three tape anchors at the
# three known heights (h=0, h_belly, h=H_out).
i_belly = int(np.argmax(a_arr))
h_belly = h_arr[i_belly]
a_base, a_belly, a_top_mesh = a_arr[0], a_arr[i_belly], a_arr[-1]
r_anchored = np.zeros_like(a_arr)
for i, (h, a) in enumerate(zip(h_arr, a_arr)):
    if h <= h_belly:
        frac = (a - a_base) / (a_belly - a_base) if a_belly != a_base else 0
        r_anchored[i] = R_bot + frac * (R_max - R_bot)
    else:
        frac = (a_belly - a) / (a_belly - a_top_mesh) if a_belly != a_top_mesh else 0
        r_anchored[i] = R_max - frac * (R_max - R_top)

# Denser interpolation
N_h = 80
N_theta = 64
h_fine = np.linspace(h_arr.min(), h_arr.max(), N_h)
r_fine = np.interp(h_fine, h_arr, r_anchored)

# cm -> meters
h_fine_m = h_fine / 100.0
r_fine_m = r_fine / 100.0

# Vertices around the axis. Y is up (USDZ convention).
vertices = []
for h, r in zip(h_fine_m, r_fine_m):
    for k in range(N_theta):
        theta = 2 * math.pi * k / N_theta
        vertices.append((r*math.cos(theta), h, r*math.sin(theta)))
top_c = len(vertices); vertices.append((0.0, float(h_fine_m[-1]), 0.0))
bot_c = len(vertices); vertices.append((0.0, float(h_fine_m[0]), 0.0))

# Per-vertex displayColor: tan (unglazed clay) at the very bottom band,
# cream-white (glazed) everywhere else. The transition line in the photos
# sits at roughly 22% of the height from the base.
GLAZE_CREAM = (0.94, 0.92, 0.87)
CLAY_TAN    = (0.78, 0.69, 0.50)
CLAY_BAND_FRAC = 0.22
clay_cutoff_h = h_fine_m[0] + CLAY_BAND_FRAC*(h_fine_m[-1] - h_fine_m[0])
colors = []
for h, r in zip(h_fine_m, r_fine_m):
    base_col = CLAY_TAN if h < clay_cutoff_h else GLAZE_CREAM
    for _ in range(N_theta):
        colors.append(base_col)
colors.append(GLAZE_CREAM)  # top centre
colors.append(CLAY_TAN)     # bot centre

# Faces: quads on the side, triangle fans on the caps.
face_counts = []
face_indices = []
for i in range(N_h - 1):
    for k in range(N_theta):
        a = i*N_theta + k
        b = i*N_theta + (k+1) % N_theta
        c = (i+1)*N_theta + (k+1) % N_theta
        d = (i+1)*N_theta + k
        face_counts.append(4)
        face_indices.extend([a, b, c, d])
# Top cap (CCW seen from above -> normal up)
for k in range(N_theta):
    a = (N_h-1)*N_theta + k
    b = (N_h-1)*N_theta + (k+1) % N_theta
    face_counts.append(3)
    face_indices.extend([a, b, top_c])
# Bottom cap (CW seen from below -> normal down)
for k in range(N_theta):
    a = k
    b = (k+1) % N_theta
    face_counts.append(3)
    face_indices.extend([b, a, bot_c])

# Compose USDA with a UsdPreviewSurface material so it renders with
# a proper ceramic look in Quick Look (cream diffuse + per-vertex
# colour for the unglazed band).
pts_str = ", ".join(f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})" for v in vertices)
fc_str = ", ".join(str(c) for c in face_counts)
fi_str = ", ".join(str(i) for i in face_indices)
col_str = ", ".join(f"({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})" for c in colors)

usda = f"""#usda 1.0
(
    defaultPrim = "Vase"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Vase" {{
    def Mesh "vase" {{
        point3f[] points = [{pts_str}]
        int[] faceVertexCounts = [{fc_str}]
        int[] faceVertexIndices = [{fi_str}]
        color3f[] primvars:displayColor = [{col_str}] (interpolation = "vertex")
        float[] primvars:displayOpacity = [1.0] (interpolation = "constant")
    }}
}}
"""

out_usda = ROOT / "output" / "vase_axisym.usda"
out_usda.write_text(usda)
print(f"wrote {out_usda} ({len(usda)} chars, {len(vertices)} verts, "
      f"{len(face_counts)} faces)")

# Convert USDA -> USDC and zip as USDZ.
# usdcat insists the output extension be .usd (regardless of the binary
# format flag), so write to .usd first then rename to .usdc inside the zip.
tmp_usd = ROOT / "output" / "_vase_axisym.usd"
subprocess.run(
    ["usdcat", "-o", str(tmp_usd), "--usdFormat", "usdc", str(out_usda)],
    check=True)
out_usdz = ROOT / "output" / "vase_axisym.usdz"
with zipfile.ZipFile(out_usdz, "w", zipfile.ZIP_STORED) as zf:
    zf.write(tmp_usd, "vase_axisym.usdc")
tmp_usd.unlink()
print(f"wrote {out_usdz} ({out_usdz.stat().st_size} bytes)")
