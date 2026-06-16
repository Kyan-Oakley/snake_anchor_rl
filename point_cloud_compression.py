import heapq
import math
import numpy as np

def _dist_sq(a, b):
    d = a - b
    return d @ d


class _Node:
    __slots__ = ('point', 'idx', 'axis', 'left', 'right', 'deleted')

    def __init__(self, point, idx, axis, left=None, right=None):
        self.point = point
        self.idx = idx
        self.axis = axis
        self.left = left
        self.right = right
        self.deleted = False


class _KDTree:
    def __init__(self, indexed):  # indexed: [(idx, (x, y, z)), ...]
        self._nodes = {}
        self.size = len(indexed)
        self.root = self._build(indexed, 0)

    def _build(self, pts, depth):
        if not pts:
            return None
        axis = depth % 3
        pts.sort(key=lambda p: p[1][axis])
        mid = len(pts) // 2
        idx, point = pts[mid]
        node = _Node(point, idx, axis)
        self._nodes[idx] = node
        node.left  = self._build(pts[:mid],     depth + 1)
        node.right = self._build(pts[mid + 1:], depth + 1)
        return node

    def delete(self, idx):
        self._nodes[idx].deleted = True

    def nearest(self, query, exclude):
        best = [None]  # (dist_sq, idx)
        self._search(self.root, query, exclude, best)
        return best[0]

    def _search(self, node, query, exclude, best):
        if node is None:
            return
        if not node.deleted and node.idx != exclude:
            d = _dist_sq(query, node.point)
            if best[0] is None or d < best[0][0]:
                best[0] = (d, node.idx)
        diff = query[node.axis] - node.point[node.axis]
        near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._search(near, query, exclude, best)
        if best[0] is None or diff * diff < best[0][0]:
            self._search(far, query, exclude, best)


class ClosestPairTracker:
    def __init__(self, points):
        """
        points: list of (x, y, z) tuples
        """
        self._points = list(points)
        self._active = set(range(len(points)))
        self._build()

    def _build(self):
        indexed = [(i, self._points[i]) for i in self._active]
        self._tree = _KDTree(indexed)
        self._heap = []
        for i, p in indexed:
            result = self._tree.nearest(p, exclude=i)
            if result:
                d_sq, j = result
                heapq.heappush(self._heap, (d_sq, i, j))

    def pop_closest(self):
        """
        Finds the globally closest pair and removes one point from it.
        Returns (removed_idx, partner_idx, distance) or None if fewer than 2 points remain.
        """
        # Rebuild when over half the tree nodes are deleted (keeps NN
        if len(self._active) < self._tree.size // 2:
            self._build()

        while self._heap:
            d_sq, i, j = heapq.heappop(self._heap)

            if i in self._active and j in self._active:
                self._active.remove(i)
                self._tree.delete(i)
                # j's nearest neighbor may have changed — requeue it
                result = self._tree.nearest(self._points[j], exclude=j)
                if result:
                    heapq.heappush(self._heap, (result[0], j, result[1]))
                return i, j, math.sqrt(d_sq)

            if i in self._active:
                # j was already removed — find i's new nearest neighbor
                result = self._tree.nearest(self._points[i], exclude=i)
                if result:
                    heapq.heappush(self._heap, (result[0], i, result[1]))
            # if i is no longer active, entry is fully stale — discard

        return None

def closest_point_filter(cloud, number_of_points):
    tracker = ClosestPairTracker(cloud)
    while len(tracker._active) > number_of_points:
        tracker.pop_closest()
    remaining = sorted(tracker._active)
    return cloud[remaining]
