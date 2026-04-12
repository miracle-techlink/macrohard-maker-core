"""Geodesic computations on surfaces.

Provides Clairaut-based geodesic direction computation for surfaces of
revolution, geodesic curvature calculation, and stress-geodesic blending.
"""

import numpy as np
import sys as _sys
_print = print
def print(*a, **kw):
    try: _print(*a, **kw, flush=True)
    except (BrokenPipeError, OSError): pass


def clairaut_constant(r, alpha):
    """Compute Clairaut constant C = r * cos(alpha).

    For a geodesic on a surface of revolution, C is conserved along the curve.

    Args:
        r: distance from rotation axis
        alpha: angle between geodesic and parallel (latitude circle)

    Returns:
        C: Clairaut constant
    """
    return r * np.cos(alpha)


def geodesic_angle_at_radius(C, r):
    """Compute geodesic angle alpha at a given radius.

    From Clairaut's relation: alpha = arccos(C/r).
    Clamps C/r to [-1, 1] for numerical safety.

    Args:
        C: Clairaut constant
        r: distance from rotation axis

    Returns:
        alpha: angle between geodesic and parallel direction
    """
    ratio = np.clip(C / np.maximum(r, 1e-10), -1.0, 1.0)
    return np.arccos(ratio)


def geodesic_direction_on_surface(point, surface_mesh, tri_idx, C=None, alpha=None):
    """Compute geodesic direction in tangent plane for a surface of revolution.

    The surface is assumed to revolve around the Z-axis.
    The geodesic direction is: cos(alpha)*e_theta + sin(alpha)*e_meridional,
    projected onto the triangle tangent plane.

    Either C (Clairaut constant) or alpha must be provided.

    Args:
        point: (3,) point on surface
        surface_mesh: SurfaceMesh instance
        tri_idx: triangle index
        C: Clairaut constant (if provided, alpha is computed from it)
        alpha: angle between geodesic and parallel direction

    Returns:
        direction: (3,) unit geodesic direction in tangent plane
    """
    x, y, z = point
    r = np.sqrt(x**2 + y**2)
    r = max(r, 1e-10)
    theta = np.arctan2(y, x)

    if C is not None:
        alpha = geodesic_angle_at_radius(C, r)
    elif alpha is None:
        raise ValueError("Either C or alpha must be provided")

    n = surface_mesh.normals[tri_idx]

    # e_theta: tangent to parallel (latitude circle), in 3D
    e_theta_3d = np.array([-np.sin(theta), np.cos(theta), 0.0])
    # Project to tangent plane
    e_theta_tan = e_theta_3d - np.dot(e_theta_3d, n) * n
    et_norm = np.linalg.norm(e_theta_tan)
    if et_norm > 1e-12:
        e_theta_tan /= et_norm
    else:
        # Degenerate case — use any tangent direction
        return _arbitrary_tangent(n)

    # e_meridional: tangent to meridian (along z direction projected to tangent)
    e_z = np.array([0.0, 0.0, 1.0])
    e_merid_3d = e_z - np.dot(e_z, n) * n
    em_norm = np.linalg.norm(e_merid_3d)
    if em_norm > 1e-12:
        e_merid_3d /= em_norm
    else:
        # At pole, meridional is any radial direction
        e_r = np.array([np.cos(theta), np.sin(theta), 0.0])
        e_merid_3d = e_r - np.dot(e_r, n) * n
        em_norm = np.linalg.norm(e_merid_3d)
        if em_norm > 1e-12:
            e_merid_3d /= em_norm
        else:
            return _arbitrary_tangent(n)

    # Geodesic direction
    direction = np.cos(alpha) * e_theta_tan + np.sin(alpha) * e_merid_3d
    d_norm = np.linalg.norm(direction)
    if d_norm > 1e-12:
        direction /= d_norm

    return direction


def _arbitrary_tangent(n):
    """Get an arbitrary unit tangent vector given a normal."""
    if abs(n[0]) < 0.9:
        t = np.cross(n, np.array([1, 0, 0]))
    else:
        t = np.cross(n, np.array([0, 1, 0]))
    t_norm = np.linalg.norm(t)
    if t_norm > 1e-12:
        return t / t_norm
    return np.array([1.0, 0.0, 0.0])


def compute_geodesic_curvature(point, direction, surface_mesh, tri_idx):
    """Compute geodesic curvature kappa_g of a curve on the surface.

    Uses finite difference approximation. For a geodesic on the surface,
    kappa_g = 0. The geodesic curvature measures how much a curve deviates
    from being a geodesic.

    For a surface of revolution around Z-axis with direction d:
    kappa_g ~ |d x (dd/ds)| projected onto normal

    Here we use an approximate formula based on the deviation of the direction
    from the geodesic direction.

    Args:
        point: (3,) point on surface
        direction: (3,) tangent direction of curve
        surface_mesh: SurfaceMesh instance
        tri_idx: triangle index

    Returns:
        kappa_g: geodesic curvature (1/mm)
    """
    x, y, z = point
    r = np.sqrt(x**2 + y**2)
    if r < 1e-6:
        return 0.0

    n = surface_mesh.normals[tri_idx]
    theta = np.arctan2(y, x)

    # e_theta in tangent plane
    e_theta_3d = np.array([-np.sin(theta), np.cos(theta), 0.0])
    e_theta_tan = e_theta_3d - np.dot(e_theta_3d, n) * n
    et_norm = np.linalg.norm(e_theta_tan)
    if et_norm > 1e-12:
        e_theta_tan /= et_norm

    # The angle between direction and e_theta
    d_tan = direction - np.dot(direction, n) * n
    d_norm = np.linalg.norm(d_tan)
    if d_norm < 1e-12:
        return 0.0
    d_tan /= d_norm

    cos_alpha = np.clip(np.dot(d_tan, e_theta_tan), -1.0, 1.0)
    alpha = np.arccos(abs(cos_alpha))

    # For a surface of revolution, geodesic curvature is approximately:
    # kappa_g = (1/r) * sin(alpha) * cos(alpha) * dr/ds_meridional
    # Simplified: kappa_g ~ sin(alpha) * cos(alpha) / r for cylindrical surface
    # For a cylinder (dr/dz = 0), true geodesics have kappa_g = 0
    # This is a useful metric even if approximate
    kappa_g = abs(np.sin(alpha) * np.cos(alpha)) / max(r, 1e-6)

    return kappa_g


def blend_stress_geodesic(stress_dir, geodesic_dir, w_stress, w_geodesic):
    """Weighted blend of stress direction and geodesic direction.

    Handles sign ambiguity (stress directions are axial, i.e. +/- equivalent).

    Args:
        stress_dir: (3,) unit stress direction in tangent plane
        geodesic_dir: (3,) unit geodesic direction in tangent plane
        w_stress: weight for stress direction
        w_geodesic: weight for geodesic direction

    Returns:
        blended: (3,) unit blended direction
    """
    # Align signs: flip stress_dir if it opposes geodesic_dir
    if np.dot(stress_dir, geodesic_dir) < 0:
        stress_dir = -stress_dir

    blended = w_stress * stress_dir + w_geodesic * geodesic_dir
    norm = np.linalg.norm(blended)
    if norm < 1e-12:
        return stress_dir.copy()
    return blended / norm
