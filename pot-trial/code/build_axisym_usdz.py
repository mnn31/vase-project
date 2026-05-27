"""Build the axisymmetric pot mesh by revolving the photogrammetry mesh's
front-side semi-axis a(h) around the vertical axis, rescaled to the
top-rim tape anchor. Same idea as vase_project/code/build_axisym_usdz.py.

@author Manan Gupta
@author Claude (Anthropic AI assistant), code co-author
"""
from __future__ import annotations
import json, math, subprocess, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
results = json.loads((ROOT / "output" / "results.json").read_text())
profile = results["profile"]
h_arr = np.array([p["h_cm"] for p in profile])
a_arr = np.array([p["r_out_cm"] for p in profile])

# Pot measurements (already-anchored profile in results.json)
C_top = 54.0
R_top = C_top / (2*math.pi)
H_out = 7.0
T_WALL = 0.49

# Profile is already anchored; no further rescale needed.
factor = 1.0
r_anchored = a_arr

# Denser interpolation
N_h = 60
N_theta = 64
h_fine = np.linspace(h_arr.min(), h_arr.max(), N_h)
r_fine = np.interp(h_fine, h_arr, r_anchored)

h_fine_m = h_fine / 100.0
r_fine_m = r_fine / 100.0

vertices = []
for h, r in zip(h_fine_m, r_fine_m):
    for k in range(N_theta):
        theta = 2 * math.pi * k / N_theta
        vertices.append((r*math.cos(theta), h, r*math.sin(theta)))
top_c = len(vertices); vertices.append((0.0, float(h_fine_m[-1]), 0.0))
bot_c = len(vertices); vertices.append((0.0, float(h_fine_m[0]), 0.0))

# Terracotta color
TERRACOTTA = (0.78, 0.45, 0.30)
colors = [TERRACOTTA for _ in range(len(vertices))]

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
for k in range(N_theta):
    a = (N_h-1)*N_theta + k
    b = (N_h-1)*N_theta + (k+1) % N_theta
    face_counts.append(3)
    face_indices.extend([a, b, top_c])
for k in range(N_theta):
    a = k
    b = (k+1) % N_theta
    face_counts.append(3)
    face_indices.extend([b, a, bot_c])

pts_str = ", ".join(f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})" for v in vertices)
fc_str = ", ".join(str(c) for c in face_counts)
fi_str = ", ".join(str(i) for i in face_indices)
col_str = ", ".join(f"({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})" for c in colors)

usda = f"""#usda 1.0
(
    defaultPrim = "Pot"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Pot" {{
    def Mesh "pot" {{
        point3f[] points = [{pts_str}]
        int[] faceVertexCounts = [{fc_str}]
        int[] faceVertexIndices = [{fi_str}]
        color3f[] primvars:displayColor = [{col_str}] (interpolation = "vertex")
        float[] primvars:displayOpacity = [1.0] (interpolation = "constant")
    }}
}}
"""

out_usda = ROOT / "output" / "pot_axisym.usda"
out_usda.write_text(usda)
print(f"wrote {out_usda}")

tmp_usd = ROOT / "output" / "_pot_axisym.usd"
subprocess.run(["usdcat", "-o", str(tmp_usd), "--usdFormat", "usdc", str(out_usda)],
               check=True)
out_usdz = ROOT / "output" / "pot_axisym.usdz"
with zipfile.ZipFile(out_usdz, "w", zipfile.ZIP_STORED) as zf:
    zf.write(tmp_usd, "pot_axisym.usdc")
tmp_usd.unlink()
print(f"wrote {out_usdz} ({out_usdz.stat().st_size} bytes)")
