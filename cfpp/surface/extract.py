"""Surface mesh extraction from tetrahedral volume mesh.

Extracts boundary triangles (faces appearing exactly once) from a tet mesh,
computes normals, builds edge-to-triangle adjacency, and provides spatial
queries for surface projection.
"""

import numpy as np
import sys as _sys
_print = print
def print(*a, **kw):
    try: _print(*a, **kw, flush=True)
    except (BrokenPipeError, OSError): pass
from scipy.spatial import cKDTree
import meshio


class SurfaceMesh:
    """Triangulated surface mesh with spatial query support."""

    def __init__(self, vertices, triangles, outward_normals=True):
        """
        Args:
            vertices: (V, 3) vertex coordinates
            triangles: (T, 3) triangle vertex indices
            outward_normals: if True, orient normals outward
        """
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64)

        # Compute triangle normals
        v0 = self.vertices[self.triangles[:, 0]]
        v1 = self.vertices[self.triangles[:, 1]]
        v2 = self.vertices[self.triangles[:, 2]]
        e1 = v1 - v0
        e2 = v2 - v0
        normals = np.cross(e1, e2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        self.normals = normals / norms  # (T, 3)

        # Orient normals outward (away from centroid of mesh)
        if outward_normals:
            mesh_center = self.vertices.mean(axis=0)
            self.centroids = (v0 + v1 + v2) / 3.0
            outward_dir = self.centroids - mesh_center
            flip = np.sum(self.normals * outward_dir, axis=1) < 0
            self.normals[flip] *= -1
            # Also flip triangle winding for consistency
            self.triangles[flip] = self.triangles[flip][:, [0, 2, 1]]
        else:
            self.centroids = (v0 + v1 + v2) / 3.0

        # Vertex normals (area-weighted average of adjacent triangle normals)
        self.vertex_normals = np.zeros_like(self.vertices)
        # weight by triangle area (proportional to norm of cross product)
        areas = np.linalg.norm(np.cross(e1, e2), axis=1) * 0.5
        for i in range(3):
            np.add.at(self.vertex_normals, self.triangles[:, i],
                       self.normals * areas[:, np.newaxis])
        vn_norms = np.linalg.norm(self.vertex_normals, axis=1, keepdims=True)
        vn_norms = np.maximum(vn_norms, 1e-15)
        self.vertex_normals /= vn_norms

        # Edge-to-triangle adjacency
        self.edge_to_tri = {}
        for ti in range(len(self.triangles)):
            tri = self.triangles[ti]
            for j in range(3):
                edge = tuple(sorted((tri[j], tri[(j + 1) % 3])))
                if edge not in self.edge_to_tri:
                    self.edge_to_tri[edge] = []
                self.edge_to_tri[edge].append(ti)

        # KD-tree on centroids for spatial queries
        self._centroid_tree = cKDTree(self.centroids)
        # KD-tree on vertices
        self._vertex_tree = cKDTree(self.vertices)

        print(f"SurfaceMesh: {len(self.vertices)} vertices, "
              f"{len(self.triangles)} triangles")

    def find_triangle(self, point):
        """Find nearest triangle to a 3D point.

        Args:
            point: (3,) coordinate

        Returns:
            tri_idx: index of nearest triangle (by centroid distance)
        """
        _, idx = self._centroid_tree.query(point, k=1)
        return idx

    def project_to_surface(self, point):
        """Project a 3D point onto the nearest triangle plane.

        Args:
            point: (3,) coordinate

        Returns:
            projected: (3,) projected point on triangle plane
            tri_idx: index of the triangle
        """
        tri_idx = self.find_triangle(point)
        n = self.normals[tri_idx]
        c = self.centroids[tri_idx]
        # Project to plane: p_proj = p - ((p-c) . n) * n
        diff = point - c
        projected = point - np.dot(diff, n) * n
        return projected, tri_idx

    def project_vector_to_tangent(self, vec, tri_idx):
        """Project a vector onto the tangent plane of a triangle.

        Args:
            vec: (3,) vector
            tri_idx: triangle index

        Returns:
            tangent_vec: (3,) projected and normalized vector
        """
        n = self.normals[tri_idx]
        # Remove normal component: v_tan = v - (v.n)n
        tangent = vec - np.dot(vec, n) * n
        norm = np.linalg.norm(tangent)
        if norm < 1e-12:
            return tangent
        return tangent / norm


def extract_surface(mesh_path):
    """Extract boundary surface from a tetrahedral mesh file.

    Reads the mesh, finds tetrahedra, extracts faces appearing exactly once
    (boundary faces), and returns a SurfaceMesh.

    Args:
        mesh_path: path to mesh file (any format meshio supports)

    Returns:
        SurfaceMesh instance
    """
    mio = meshio.read(str(mesh_path))
    vertices = mio.points

    # Find tetrahedral cells
    tet_cells = None
    for cell_block in mio.cells:
        if cell_block.type == "tetra":
            tet_cells = cell_block.data
            break

    if tet_cells is None:
        raise ValueError(f"No tetrahedral cells found in {mesh_path}")

    print(f"Volume mesh: {len(vertices)} vertices, {len(tet_cells)} tetrahedra")

    # Extract boundary faces: faces appearing exactly once
    face_count = {}
    # Each tet has 4 faces (combinations of 3 vertices out of 4)
    face_indices = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

    for tet in tet_cells:
        for fi in face_indices:
            face = tuple(sorted(tet[list(fi)]))
            if face in face_count:
                face_count[face] += 1
            else:
                face_count[face] = 1

    # Boundary faces appear exactly once
    boundary_faces = []
    for face, count in face_count.items():
        if count == 1:
            boundary_faces.append(face)

    triangles = np.array(boundary_faces, dtype=np.int64)
    print(f"Boundary faces: {len(triangles)} triangles")

    return SurfaceMesh(vertices, triangles)
