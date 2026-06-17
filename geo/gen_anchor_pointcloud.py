import argparse
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pathlib import Path


def euler_to_rot(euler_str):
    """Parse 'rx ry rz' (radians) → 3×3 rotation matrix.

    MuJoCo applies extrinsic XYZ rotations (around global axes in order X→Y→Z),
    giving R = Rz @ Ry @ Rx.
    """
    if not euler_str:
        return np.eye(3)
    rx, ry, rz = [float(v) for v in euler_str.split()]
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,   0,  0], [0,  cx, -sx], [0,  sx, cx]])
    Ry = np.array([[cy,  0, sy], [0,   1,   0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz,   0], [0,   0,  1]])
    return Rz @ Ry @ Rx


def parse_wall_geoms(xml_path):
    """Return list of (name, world_center, half_sizes, R) for all box geoms in Wall bodies."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    if worldbody is None:
        return []

    walls = []
    for body in worldbody.iter('body'):
        body_pos = np.array([float(v) for v in body.get('pos', '0 0 0').split()])
        R = euler_to_rot(body.get('euler', ''))
        for geom in body.findall('geom'):
            if geom.get('type', 'sphere') != 'box':
                continue
            geom_pos_local = np.array([float(v) for v in geom.get('pos', '0 0 0').split()])
            half_sizes = np.array([float(v) for v in geom.get('size', '').split()])
            world_center = body_pos + R @ geom_pos_local
            walls.append((geom.get('name'), world_center, half_sizes, R))

    return walls


def inner_face_info(center, half_sizes, R, toward):
    """Compute the inner-face geometry of an oriented box.

    toward: world-space vector from wall center toward the gap centroid.
    R:      rotation matrix whose columns are the wall's local axes in world space.

    Returns:
        face_center : world-space center of the inner face
        face_normal : world-space unit normal pointing INTO the gap
        tangent_axes: two local-frame axis indices that span the face
    """
    toward_local = R.T @ toward
    local_ax = int(np.argmax(np.abs(toward_local)))
    local_sign = float(np.sign(toward_local[local_ax]))

    face_normal = local_sign * R[:, local_ax]
    face_center = center + face_normal * half_sizes[local_ax]
    tangent_axes = [a for a in range(3) if a != local_ax]
    return face_center, face_normal, tangent_axes


def sample_face_uniform(face_center, half_sizes, R, tangent_axes, n_points):
    """Sample n_points uniformly on a wall's inner face. No clipping applied."""
    a0, a1 = tangent_axes
    h0, h1 = half_sizes[a0], half_sizes[a1]
    u = np.random.uniform(-h0, h0, n_points)
    v = np.random.uniform(-h1, h1, n_points)
    return face_center + u[:, None] * R[:, a0] + v[:, None] * R[:, a1]


def clip_pts(pts, face_center, half_sizes, R, tangent_axes, halfspaces, wall_idx, max_depth):
    """Apply Z-depth clamp and interior half-space clipping."""
    a0, a1 = tangent_axes
    z_bot = face_center[2] - half_sizes[a0] * abs(R[2, a0]) - half_sizes[a1] * abs(R[2, a1])
    pts = pts[pts[:, 2] <= z_bot + max_depth]
    for j, (fc, fn) in enumerate(halfspaces):
        if j == wall_idx:
            continue
        pts = pts[(pts - fc) @ fn >= -1e-6]
    return pts


def visualize_cloud(cloud, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=1, alpha=0.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


TARGET_POINTS = 1_200  # desired total points after all clipping
N_PILOT      = 500   # pilot samples per wall to estimate survival fraction
MAX_DEPTH    = 0.30    # keep only the bottom MAX_DEPTH metres in world Z

parser = argparse.ArgumentParser()
parser.add_argument('--visualize', action='store_true', help='Show 3D plot of each point cloud after generation')
args = parser.parse_args()

np.random.seed(42)

scenes_dir = Path("anchor_scenes")
clouds_dir = Path("point_clouds")
clouds_dir.mkdir(exist_ok=True)

for xml_file in sorted(scenes_dir.glob("*.xml")):
    walls = parse_wall_geoms(xml_file)

    if len(walls) < 2:
        print(f"[{xml_file.name}] fewer than 2 Wall geoms found, skipping")
        continue

    centers  = np.array([w[1] for w in walls])
    centroid = centers.mean(axis=0)

    # Precompute each wall's face geometry and half-space constraint
    face_infos = []  # (face_center, face_normal, tangent_axes, raw_area)
    halfspaces = []  # (face_center, face_normal) used when clipping other walls
    for _, center, half_sizes, R in walls:
        toward = centroid - center
        fc, fn, tax = inner_face_info(center, half_sizes, R, toward)
        a0, a1 = tax
        raw_area = 4 * half_sizes[a0] * half_sizes[a1]
        face_infos.append((fc, fn, tax, raw_area))
        halfspaces.append((fc, fn))

    # Pilot pass: measure each wall's survival fraction under both Z clamp and
    # half-space clipping together, so heavily-clipped walls receive proportionally
    # more raw samples in the final pass to compensate.
    fracs = []
    for i, (name, center, half_sizes, R) in enumerate(walls):
        fc, fn, tax, _ = face_infos[i]
        pilot     = sample_face_uniform(fc, half_sizes, R, tax, N_PILOT)
        surviving = clip_pts(pilot, fc, half_sizes, R, tax, halfspaces, i, MAX_DEPTH)
        fracs.append(len(surviving) / N_PILOT)

    # Allocate raw samples so the expected surviving count from each wall is
    # proportional to its clipped area (raw_area * frac):
    #   n_raw_i = TARGET * raw_area_i / sum(raw_area_j * frac_j)
    #   E[survivors_i] = n_raw_i * frac_i = TARGET * clipped_area_i / total_clipped_area
    clipped_areas = [face_infos[i][3] * fracs[i] for i in range(len(walls))]
    total_clipped = sum(clipped_areas)

    all_pts = []
    for i, (name, center, half_sizes, R) in enumerate(walls):
        fc, fn, tax, raw_area = face_infos[i]
        n_raw = max(1, round(TARGET_POINTS * raw_area / total_clipped))
        pts   = sample_face_uniform(fc, half_sizes, R, tax, n_raw)
        pts   = clip_pts(pts, fc, half_sizes, R, tax, halfspaces, i, MAX_DEPTH)
        print(f"  {name}: {len(pts)} points")
        all_pts.append(pts)

    cloud    = np.concatenate(all_pts).astype(np.float32)
    out_path = clouds_dir / f"{xml_file.stem}.npy"
    np.save(out_path, cloud)
    print(f"[{xml_file.name}] {len(cloud)} total points → {out_path}")

    if args.visualize:
        visualize_cloud(cloud, xml_file.stem)
