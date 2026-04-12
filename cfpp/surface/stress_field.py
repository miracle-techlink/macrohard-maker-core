"""Project volume stress field onto surface mesh.

Uses IDW interpolation from nearest volume element centroids to surface
triangle centroids, then projects stress directions onto tangent planes.
"""

import numpy as np
import sys as _sys
_print = print
def print(*a, **kw):
    try: _print(*a, **kw, flush=True)
    except (BrokenPipeError, OSError): pass
from scipy.spatial import cKDTree


class SurfaceStressField:
    """Surface stress field projected from volume FEA results."""

    def __init__(self, surface_mesh, stress_npz_path):
        """
        Args:
            surface_mesh: SurfaceMesh instance
            stress_npz_path: path to stress NPZ from Phase 1
                (must contain centroids, sigma_1_dir, sigma_1_val, von_mises, etc.)
        """
        self.surface = surface_mesh
        data = np.load(stress_npz_path)

        vol_centroids = data["centroids"]      # (N_vol, 3)
        vol_sigma1_dir = data["sigma_1_dir"]   # (N_vol, 3)
        vol_sigma1_val = data["sigma_1_val"]   # (N_vol,)
        vol_sigma2_dir = data["sigma_2_dir"]
        vol_sigma2_val = data["sigma_2_val"]
        vol_von_mises = data["von_mises"]

        # Build dominant stress direction (same logic as StressField)
        if "sigma_3_dir" in data:
            vol_sigma3_dir = data["sigma_3_dir"]
            vol_sigma3_val = data["sigma_3_val"]
        else:
            vol_sigma3_dir = vol_sigma2_dir
            vol_sigma3_val = vol_sigma2_val

        abs1 = np.abs(vol_sigma1_val)
        abs3 = np.abs(vol_sigma3_val)
        use_3 = abs3 > abs1
        vol_dom_dir = vol_sigma1_dir.copy()
        vol_dom_dir[use_3] = vol_sigma3_dir[use_3]
        vol_dom_val = vol_sigma1_val.copy()
        vol_dom_val[use_3] = vol_sigma3_val[use_3]

        # KD-tree on volume centroids
        vol_tree = cKDTree(vol_centroids)

        n_tri = len(surface_mesh.triangles)
        k = 4  # number of nearest neighbors for IDW

        # For each surface triangle, IDW interpolate stress from volume
        self.stress_dir = np.zeros((n_tri, 3))    # tangent-projected direction
        self.stress_val = np.zeros(n_tri)          # stress magnitude
        self.von_mises = np.zeros(n_tri)           # von Mises on surface
        self.dom_dir = np.zeros((n_tri, 3))        # dominant direction (tangent)
        self.dom_val = np.zeros(n_tri)

        # Batch query
        dists, indices = vol_tree.query(surface_mesh.centroids, k=k)
        dists = np.maximum(dists, 1e-10)

        for i in range(n_tri):
            w = 1.0 / dists[i]
            w /= w.sum()
            idx = indices[i]
            n = surface_mesh.normals[i]

            # IDW interpolate sigma_1 direction with sign alignment
            ref = vol_sigma1_dir[idx[0]]
            dirs = vol_sigma1_dir[idx].copy()
            for j in range(1, k):
                if np.dot(dirs[j], ref) < 0:
                    dirs[j] = -dirs[j]
            d_interp = np.average(dirs, axis=0, weights=w)

            # Project onto tangent plane: d_surf = d - (d.n)n
            d_surf = d_interp - np.dot(d_interp, n) * n
            norm_d = np.linalg.norm(d_surf)
            if norm_d > 1e-12:
                d_surf /= norm_d
            self.stress_dir[i] = d_surf
            self.stress_val[i] = np.average(vol_sigma1_val[idx], weights=w)

            # Dominant direction
            ref_dom = vol_dom_dir[idx[0]]
            dirs_dom = vol_dom_dir[idx].copy()
            for j in range(1, k):
                if np.dot(dirs_dom[j], ref_dom) < 0:
                    dirs_dom[j] = -dirs_dom[j]
            d_dom = np.average(dirs_dom, axis=0, weights=w)
            d_dom_surf = d_dom - np.dot(d_dom, n) * n
            norm_dom = np.linalg.norm(d_dom_surf)
            if norm_dom > 1e-12:
                d_dom_surf /= norm_dom
            self.dom_dir[i] = d_dom_surf
            self.dom_val[i] = np.average(vol_dom_val[idx], weights=w)

            # Von Mises
            self.von_mises[i] = np.average(vol_von_mises[idx], weights=w)

        # KD-tree on surface centroids for point queries
        self._tree = cKDTree(surface_mesh.centroids)

        print(f"SurfaceStressField: {n_tri} triangles, "
              f"von Mises range [{self.von_mises.min():.2f}, {self.von_mises.max():.2f}] MPa")

    def query(self, point, component="sigma_1"):
        """Query stress at a surface point.

        Args:
            point: (3,) coordinate
            component: "sigma_1" or "dom"

        Returns:
            direction: (3,) tangent-plane stress direction
            value: stress magnitude
        """
        _, tri_idx = self._tree.query(point, k=1)
        return self.query_at_triangle(tri_idx, component)

    def query_at_triangle(self, tri_idx, component="sigma_1"):
        """Query stress at a given triangle.

        Args:
            tri_idx: triangle index
            component: "sigma_1" or "dom"

        Returns:
            direction: (3,) tangent-plane stress direction
            value: stress magnitude
        """
        if component == "dom":
            return self.dom_dir[tri_idx].copy(), self.dom_val[tri_idx]
        else:
            return self.stress_dir[tri_idx].copy(), self.stress_val[tri_idx]
