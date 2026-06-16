import argparse
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pathlib import Path


def parse_wall_geoms(xml_path):
    """Return list of (name, world_center, half_sizes) for all Wall_* box geoms."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find('worldbody')
    if worldbody is None:
        return []

    walls = []
    for body in worldbody.iter('body'):
        body_pos = np.array([float(v) for v in body.get('pos', '0 0 0').split()])
        for geom in body.findall('geom'):
            if geom.get('type', 'sphere') != 'box':
                continue
            geom_pos = np.array([float(v) for v in geom.get('pos', '0 0 0').split()])
            half_sizes = np.array([float(v) for v in geom.get('size', '').split()])
            walls.append((geom.get('name'), body_pos + geom_pos, half_sizes))

    return walls


def sample_inner_face(center, half_sizes, toward, density=5000, max_depth=0.20):
    """Sample points uniformly on the single face of a box that faces the gap.

    toward:    unit-ish vector pointing from this wall's center toward the gap centroid
    max_depth: only sample this many meters above the bottom of the wall (Z axis)
    """
    inner_axis = int(np.argmax(np.abs(toward)))
    inner_dir  = int(np.sign(toward[inner_axis]))
    face_coord = center[inner_axis] + inner_dir * half_sizes[inner_axis]

    free_axes = [a for a in range(3) if a != inner_axis]
    a0, a1 = free_axes

    ranges = {}
    for a in free_axes:
        lo = center[a] - half_sizes[a]
        hi = center[a] + half_sizes[a]
        if a == 2:  # Z is vertical — clamp to bottom max_depth metres
            hi = min(hi, lo + max_depth)
        ranges[a] = (lo, hi)

    r0, r1 = ranges[a0], ranges[a1]
    area = (r0[1] - r0[0]) * (r1[1] - r1[0])
    n = max(1, int(area * density))

    pts = np.zeros((n, 3))
    pts[:, inner_axis] = face_coord
    pts[:, a0] = np.random.uniform(r0[0], r0[1], n)
    pts[:, a1] = np.random.uniform(r1[0], r1[1], n)
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

    centers = np.array([w[1] for w in walls])
    centroid = centers.mean(axis=0)

    all_pts = []
    for name, center, half_sizes in walls:
        toward = centroid - center
        pts = sample_inner_face(center, half_sizes, toward, density=5000)
        print(f"  {name}: {len(pts)} points")
        all_pts.append(pts)

    cloud = np.concatenate(all_pts).astype(np.float32)
    out_path = clouds_dir / f"{xml_file.stem}.npy"
    np.save(out_path, cloud)
    print(f"[{xml_file.name}] {len(cloud)} total points → {out_path}")

    if args.visualize:
        visualize_cloud(cloud, xml_file.stem)
