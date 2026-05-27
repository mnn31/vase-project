"""Extract the bowl's slice profile from the photogrammetry USDA mesh.

The pot photos place the bowl on a stack of textbooks. The bowl is the only
object that extends above the textbook (densest plane) — so we can slice
above the desk plane and the only object captured is the bowl itself, no
cylinder filter needed.

Important: a cylinder filter centered on a SLIGHTLY-OFFSET bowl axis can
EXCLUDE bowl rim points on the far side. Use bbox half-extents of ALL points
at each height above the desk instead.

@author Manan Gupta
@author Claude (Anthropic AI assistant), code co-author
"""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
import numpy as np


def parse_usda_mesh(path: Path):
    text = path.read_text()
    def find_array(name):
        m = re.search(rf"{re.escape(name)}\s*=\s*\[", text)
        start = m.end(); depth = 1; i = start
        while i < len(text) and depth > 0:
            if text[i] == '[': depth += 1
            elif text[i] == ']': depth -= 1
            i += 1
        return text[start:i-1]
    points_str = find_array("point3f[] points")
    tuples = re.findall(r"\(([^()]+)\)", points_str)
    return np.array([[float(x) for x in t.split(",")] for t in tuples])


def find_desk_z(points_zup, N=50):
    """Densest height bucket (the textbook/desk plane)."""
    z = points_zup[:, 2]
    edges = np.linspace(z.min(), z.max(), N + 1)
    counts, _ = np.histogram(z, bins=edges)
    i = int(np.argmax(counts))
    return float(0.5 * (edges[i] + edges[i+1]))


def slice_radii(points_zup, z_lo, z_hi, N):
    """At each height take bbox half-extents (no cylinder filter — the
    above-desk region contains only the bowl)."""
    heights = np.linspace(z_lo, z_hi, N)
    band = (z_hi - z_lo) / (N * 0.5)
    a_list = np.zeros(N)
    b_list = np.zeros(N)
    n_list = np.zeros(N, dtype=int)
    for i, h in enumerate(heights):
        mask = np.abs(points_zup[:, 2] - h) < band
        p = points_zup[mask, :2]
        if len(p) < 3: continue
        a_list[i] = 0.5*(p[:, 0].max() - p[:, 0].min())
        b_list[i] = 0.5*(p[:, 1].max() - p[:, 1].min())
        n_list[i] = len(p)
    valid = (a_list > 0) & (b_list > 0)
    return heights[valid], a_list[valid], b_list[valid], n_list[valid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usda", type=Path)
    ap.add_argument("--H", type=float, required=True)
    ap.add_argument("--R-top-cm", type=float, required=True)
    ap.add_argument("--N-slice", type=int, default=80)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    points = parse_usda_mesh(args.usda)
    zup = points.copy()[:, [0, 2, 1]]
    zup[:, 2] -= zup[:, 2].min()
    H_mesh = zup[:, 2].max()

    desk_z = find_desk_z(zup)
    bowl_extent = H_mesh - desk_z
    scale = args.H / bowl_extent
    print(f"Desk plane: z = {desk_z:.4f} units")
    print(f"Bowl extent: {bowl_extent:.4f} units → scale = {scale:.3f} cm/unit")

    h_arr, a_arr, b_arr, n_arr = slice_radii(zup, desk_z, H_mesh, args.N_slice)
    h_cm = (h_arr - desk_z) * scale
    a_cm = a_arr * scale
    b_cm = b_arr * scale
    r_cm = np.maximum(a_cm, b_cm)

    out = {
        "mesh_full_height_units": float(H_mesh),
        "desk_z_units": float(desk_z),
        "bowl_extent_units": float(bowl_extent),
        "scale_cm_per_unit": float(scale),
        "H_cm": args.H,
        "n_slices": int(len(h_cm)),
        "heights_cm": h_cm.tolist(),
        "r_outer_cm_raw": r_cm.tolist(),
        "n_points_per_slice": n_arr.tolist(),
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nProfile (raw, no anchor yet):")
    for h, r, n in zip(h_cm, r_cm, n_arr):
        flag = ""
        if r > args.R_top_cm * 1.2: flag = "  ← textbook contamination"
        print(f"  h={h:5.2f}  r={r:6.3f}  (n={n:5d}){flag}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
