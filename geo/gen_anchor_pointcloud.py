"""
Generate a point cloud map of the anchor scene (walls, no snake).

Geometry comes directly from xml_anchor_scene.xml:
  Wall1: pos=(0.04175, 0, 0.2),  half-size=(0.01, 0.5,  0.5)
  Wall2: pos=(-0.04175, 0, 0.2), half-size=(0.01, 0.5,  0.5)

Outputs:
  anchor_scene.npy  — (N, 3) float32 array of [x, y, z] points
  anchor_scene.ply  — PLY file for visualization (e.g. MeshLab / Open3D)
"""

import numpy as np


def sample_box_surface(center, half_sizes, density=500):
    """Sample points uniformly on the surface of an axis-aligned box.

    density: target points per square meter
    """
    cx, cy, cz = center
    hx, hy, hz = half_sizes

    faces = [
        # (fixed_axis, fixed_val, range_ax1, range_ax2)   — order: x,y,z
        ("x", cx + hx, (cy - hy, cy + hy), (cz - hz, cz + hz)),   # +x
        ("x", cx - hx, (cy - hy, cy + hy), (cz - hz, cz + hz)),   # -x
        ("y", cy + hy, (cx - hx, cx + hx), (cz - hz, cz + hz)),   # +y
        ("y", cy - hy, (cx - hx, cx + hx), (cz - hz, cz + hz)),   # -y
        ("z", cz + hz, (cx - hx, cx + hx), (cy - hy, cy + hy)),   # +z
        ("z", cz - hz, (cx - hx, cx + hx), (cy - hy, cy + hy)),   # -z
    ]

    pts = []
    for axis, val, r1, r2 in faces:
        area = (r1[1] - r1[0]) * (r2[1] - r2[0])
        n = max(1, int(area * density))
        a1 = np.random.uniform(r1[0], r1[1], n)
        a2 = np.random.uniform(r2[0], r2[1], n)
        fixed = np.full(n, val)
        if axis == "x":
            face_pts = np.stack([fixed, a1, a2], axis=1)
        elif axis == "y":
            face_pts = np.stack([a1, fixed, a2], axis=1)
        else:
            face_pts = np.stack([a1, a2, fixed], axis=1)
        pts.append(face_pts)

    return np.concatenate(pts, axis=0)


def save_ply(path, points):
    """Write a binary PLY file from an (N, 3) float32 array."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode()
    with open(path, "wb") as f:
        f.write(header)
        f.write(points.astype(np.float32).tobytes())


np.random.seed(42)

boxes = [
    # (label,         center,              half_sizes,       density)
    ("Wall1",   ( 0.04175, 0.0, 0.20), (0.01, 0.50, 0.50), 500),
    ("Wall2",   (-0.04175, 0.0, 0.20), (0.01, 0.50, 0.50), 500),
]

all_pts = []

for label, center, half_sizes, density in boxes:
    pts = sample_box_surface(center, half_sizes, density)
    print(f"{label}: {len(pts)} points")
    all_pts.append(pts)

cloud = np.concatenate(all_pts, axis=0).astype(np.float32)
print(f"Total:  {len(cloud)} points")

out_dir = __file__[: __file__.rfind("/")]
npy_path = out_dir + "/anchor_scene.npy"
ply_path = out_dir + "/anchor_scene.ply"

np.save(npy_path, cloud)
save_ply(ply_path, cloud)

print(f"Saved {npy_path}")
print(f"Saved {ply_path}")
