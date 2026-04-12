"""CF-Path-Planner Web Server

Serves the original Three.js visualization + provides API endpoints
for FEA simulation, XY+A path planning, and G-code generation.

Architecture:
  - Static files: original visualization HTML + Three.js + data JSON
  - API endpoints: /api/mesh, /api/fea, /api/xyza_paths, /api/gcode,
                   /api/optimize, /api/upload_stl
  - Frontend calls API, gets back JSON data, renders in Three.js
"""

import os
import sys
import signal
import json
import uuid
import tempfile
import traceback

# Ignore SIGPIPE to prevent server crash on client disconnect
signal.signal(signal.SIGPIPE, signal.SIG_IGN)
import numpy as np
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import io

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Visualization directory (serves static files from here)
VIZ_DIR = os.path.join(PROJECT_ROOT, "visualization")
DATA_DIR = os.path.join(VIZ_DIR, "data")

# Global pipeline state
state = {
    "mesh_path": None,
    "surface": None,
    "surf_stress": None,
    "xyza_waypoints": None,
    "gcode_path": None,
}


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs, flush=True)
    except (BrokenPipeError, OSError):
        pass


# ======================================================================
# API handlers
# ======================================================================

def api_mesh(params):
    """Generate mesh. params: model, r_outer, r_inner, height, r_top, wall, mesh_size"""
    import subprocess

    model = params.get("model", "cylinder")
    mesh_size = float(params.get("mesh_size", 2.5))
    mesh_path = os.path.join(tempfile.gettempdir(), f"cfpp_{uuid.uuid4().hex[:8]}.msh")

    if model == "cylinder":
        r_outer = float(params.get("r_outer", 25))
        r_inner = float(params.get("r_inner", 20))
        height = float(params.get("height", 80))
        code = (f'import sys; sys.path.insert(0,"{PROJECT_ROOT}")\n'
                f'from cfpp.mesh.generator import create_cylinder_mesh\n'
                f'create_cylinder_mesh({r_outer},{r_inner},{height},{mesh_size},"{mesh_path}")')
    elif model == "cone":
        r_bottom = float(params.get("r_bottom", 25))
        r_top = float(params.get("r_top", 15))
        height = float(params.get("height", 60))
        wall = float(params.get("wall", 5))
        code = (f'import sys; sys.path.insert(0,"{PROJECT_ROOT}")\n'
                f'from cfpp.mesh.generator import create_cone_mesh\n'
                f'create_cone_mesh({r_bottom},{r_top},{height},{wall},{mesh_size},"{mesh_path}")')
    else:
        return {"error": f"Unknown model: {model}"}

    result = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return {"error": result.stderr[-300:]}

    state["mesh_path"] = mesh_path
    state["surface"] = None
    state["surf_stress"] = None
    state["xyza_waypoints"] = None

    return {"status": "ok", "mesh_path": mesh_path, "info": result.stdout.strip()}


def api_fea(params):
    """Run FEA + surface extraction + stress projection."""
    if state["mesh_path"] is None:
        return {"error": "No mesh. Run /api/mesh first."}

    from cfpp.solver.elastic import ElasticSolver
    from cfpp.surface.extract import extract_surface
    from cfpp.surface.stress_field import SurfaceStressField

    E = float(params.get("E_gpa", 60)) * 1e3  # GPa to MPa
    nu = float(params.get("nu", 0.3))
    P = float(params.get("P", 500))

    solver = ElasticSolver(state["mesh_path"])
    solver.set_isotropic_material(E, nu)

    # Auto-detect boundary faces for any mesh (cylinder, cone, or STL)
    import meshio
    m = meshio.read(state["mesh_path"])
    z_min = float(m.points[:, 2].min())
    z_max = float(m.points[:, 2].max())
    z_range = z_max - z_min
    z_tol = z_range * 0.05  # 5% tolerance for face detection

    # Estimate load area from top face
    top_pts = m.points[m.points[:, 2] > z_max - z_tol]
    if len(top_pts) > 3:
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(top_pts[:, :2])
            A_top = float(hull.volume)  # 2D convex hull area
        except Exception:
            A_top = float(np.pi * np.max(np.sqrt(top_pts[:,0]**2 + top_pts[:,1]**2))**2)
    else:
        A_top = 100.0  # fallback
    A_top = max(A_top, 1.0)
    traction_x = P / A_top

    # Solve with auto-detected boundaries (Z-min = fixed, Z-max = load)
    mesh_sk = solver.mesh_sk
    from skfem import Basis, ElementVector, ElementTetP1, FacetBasis, LinearForm, solve, condense
    from skfem.models.elasticity import linear_elasticity, lame_parameters

    e = ElementVector(ElementTetP1(), 3)
    ib = Basis(mesh_sk, e)
    solver._basis = ib

    lam, mu = lame_parameters(solver.E, solver.nu)
    K = linear_elasticity(lam, mu).assemble(ib)

    # Fixed: bottom face (Z ≈ Z_min)
    fixed_facets = mesh_sk.facets_satisfying(
        lambda x: np.abs(x[2] - z_min) < z_tol)

    # Load: top face (Z ≈ Z_max)
    load_facets = mesh_sk.facets_satisfying(
        lambda x: np.abs(x[2] - z_max) < z_tol)

    traction_vec = np.array([traction_x, 0, 0], dtype=float)

    fb = FacetBasis(mesh_sk, e, facets=load_facets)

    @LinearForm
    def surface_load(v, w):
        return (traction_vec[0] * v.value[0] +
                traction_vec[1] * v.value[1] +
                traction_vec[2] * v.value[2])

    f_vec = surface_load.assemble(fb)
    fixed_dofs = ib.get_dofs(fixed_facets).all()

    solver.displacement = solve(*condense(K, f_vec, D=fixed_dofs))
    _safe_print(f"Solved. max|u| = {np.max(np.abs(solver.displacement)):.6f} mm")
    ps = solver.extract_principal_stresses()

    stress_npz = os.path.join(tempfile.gettempdir(), f"cfpp_stress_{uuid.uuid4().hex[:8]}.npz")
    np.savez_compressed(stress_npz, **{k: ps[k] for k in ps})

    surface = extract_surface(state["mesh_path"])
    surf_stress = SurfaceStressField(surface, stress_npz)

    state["surface"] = surface
    state["surf_stress"] = surf_stress

    # Export stress data as JSON for visualization
    centroids = surface.centroids
    von_mises = surf_stress.von_mises
    stress_dir = surf_stress.dom_dir

    # Subsample for browser
    n = len(centroids)
    if n > 8000:
        idx = np.random.choice(n, 8000, replace=False)
    else:
        idx = np.arange(n)

    stress_json = {
        "centroids": centroids[idx].tolist(),
        "von_mises": von_mises[idx].tolist(),
        "vm_max": float(von_mises.max()),
        "vm_min": float(von_mises.min()),
        "sigma_1_dir": stress_dir[idx].tolist(),
    }

    # Save to data dir for viz
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "live_stress.json"), "w") as f:
        json.dump(stress_json, f)

    # Export surface mesh for 3D rendering (triangles, not just point cloud)
    mesh_verts = surface.vertices
    mesh_tris = surface.triangles
    # Subsample if too many triangles
    if len(mesh_tris) > 15000:
        step = len(mesh_tris) // 15000
        mesh_tris_sub = mesh_tris[::step]
    else:
        mesh_tris_sub = mesh_tris

    # Get vertex indices used by subsampled triangles
    used_verts = np.unique(mesh_tris_sub.ravel())
    vert_map = -np.ones(len(mesh_verts), dtype=int)
    vert_map[used_verts] = np.arange(len(used_verts))
    new_verts = mesh_verts[used_verts]
    new_tris = vert_map[mesh_tris_sub]

    mesh_json = {
        "vertices": new_verts.tolist(),
        "triangles": new_tris.tolist(),
    }
    with open(os.path.join(DATA_DIR, "live_mesh.json"), "w") as f:
        json.dump(mesh_json, f)

    max_disp = float(np.max(np.abs(solver.displacement)))
    return {
        "status": "ok",
        "max_displacement": max_disp,
        "max_von_mises": float(von_mises.max()),
        "n_surface_tris": len(surface.triangles),
    }


def api_xyza_paths(params):
    """Generate XY+A winding waypoints (unified endpoint).

    Parameters:
      strategy   — "constant" | "variable" | "stress" | "fill" | "combined"
                   (default: "constant")
                   "fill"     — FFF layer-fill only (concentric rings per X-layer)
                   "combined" — FFF fill + CF winding on outer surface
      angle      — winding angle in degrees (default: 45)
      n_layers   — number of winding passes (default: 4)
      a_offset_z — A-axis center height in mm (default: 50)
      spacing    — path spacing in mm (default: 2.0)
      layer_height     — fill layer height in mm (default: 0.2)
      extrusion_width  — extrusion line width in mm (default: 0.4)
      n_walls          — number of solid perimeter rings (default: 2)
      infill_density   — 0.0–1.0, inner fill density (default: 0.25)
      radius     — optional override; auto-detect from surface centroids if omitted
      length     — optional override; auto-detect from surface centroids if omitted
    """
    from cfpp.surface.planner_v2 import XYAPathPlanner

    # Auto-detect geometry from surface if available
    if state["surface"] is not None:
        centroids = state["surface"].centroids
        # Mesh is Z-axis cylinder (XY cross-section, Z = height)
        radii = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)
        auto_radius = float(np.percentile(radii, 90))
        z_vals = centroids[:, 2]
        auto_length = float(z_vals.max() - z_vals.min())
    else:
        auto_radius = 25.0
        auto_length = 100.0

    radius = float(params.get("radius", auto_radius))
    length = float(params.get("length", auto_length))
    r_inner = float(params.get("r_inner", 0.0))
    angle = float(params.get("angle", 45.0))
    n_layers = int(params.get("n_layers", 4))
    strategy = "combined"   # single unified algorithm (rectilinear fill + CF winding)
    a_offset_z = float(params.get("a_offset_z", 50.0))
    layer_height = float(params.get("layer_height", 0.2))
    extrusion_width = float(params.get("extrusion_width", 0.4))
    n_walls = int(params.get("n_walls", 2))
    infill_density = float(params.get("infill_density", 0.25))

    planner = XYAPathPlanner(a_offset_y=0.0, a_offset_z=a_offset_z)

    # Unified strategy: rectilinear base-material fill + CF surface winding
    waypoints = planner.combined_print_paths(
        radius=radius, length=length,
        r_inner=r_inner,
        winding_angle_deg=angle, n_cf_layers=n_layers,
        layer_height=layer_height, extrusion_width=extrusion_width,
        n_walls=n_walls, infill_density=infill_density,
    )

    state["xyza_waypoints"] = waypoints

    # Convert XYZA waypoints → 3D mesh coords (Z-axis cylinder):
    #   x_axis → z_mesh (height along Z)
    #   z_model * cos(a_deg) → x_mesh  (radius in XY plane)
    #   z_model * sin(a_deg) → y_mesh
    import math as _math
    def _wp_to_3d(wp):
        x_axis, y_model, z_model, a_deg = wp
        a = _math.radians(a_deg)
        return [float(z_model * _math.cos(a)),
                float(z_model * _math.sin(a)),
                float(x_axis)]

    pts_3d = [_wp_to_3d(wp) for wp in waypoints]
    a_vals = [wp[3] for wp in waypoints]
    a_min = float(min(a_vals)) if a_vals else 0.0
    a_max = float(max(a_vals)) if a_vals else 360.0

    os.makedirs(DATA_DIR, exist_ok=True)

    # live_xyza_paths.json — raw waypoints for XYZA preview view
    wps_sub = waypoints[::max(1, len(waypoints)//5000)]
    with open(os.path.join(DATA_DIR, "live_xyza_paths.json"), "w") as f:
        json.dump({
            "waypoints": [[float(v) for v in wp] for wp in wps_sub],
            "n_total": len(waypoints),
            "a_min": round(a_min, 1),
            "a_max": round(a_max, 1),
        }, f)

    # ── split paths + total length ──────────────────────────────────────
    # For fill/combined: split at x-position boundaries (each X slice is a layer)
    # For winding:       split at large position jumps (each fiber pass is a layer)
    path_list = []
    total_len = 0.0
    layer_types = []   # "fill" | "fiber"

    if pts_3d:
        jump_thresh = radius * 0.5

        if strategy in ("fill", "combined"):
            # --- group by rounded x-position (= each printed cross-section) ---
            # waypoints[i][0] = x (axial position).
            # Within a fill layer all points share the same x; between layers
            # x steps by exactly layer_height.  Use dx > 0.6*layer_height as
            # the layer boundary detector.
            #
            # For 'combined': fill waypoints come first (clustered x), then CF
            # winding waypoints (x spans 0→length).  Detect the CF start when
            # d3 > jump_thresh AND the new x is near 0 or near length (winding
            # start positions).
            in_fiber_phase = False   # turns True once CF winding starts in "combined"
            current_x = round(waypoints[0][0], 4)
            current_r = round(waypoints[0][2], 4)  # z_model = ring radius
            current_path = [pts_3d[0]]
            current_type = "fill"

            for i in range(1, len(pts_3d)):
                new_x = waypoints[i][0]
                new_r = waypoints[i][2]   # z_model
                dx = abs(new_x - current_x)
                dr = abs(new_r - current_r)
                p0, p1 = pts_3d[i-1], pts_3d[i]
                d3 = _math.sqrt(sum((p1[k]-p0[k])**2 for k in range(3)))

                # Boundary detection:
                # - Fill phase (rectilinear boustrophedon): split on new X-layer or
                #   travel-move jump only.  Do NOT split on dr — z_model varies
                #   continuously along each scan line and would fragment every line.
                # - Fiber phase: only large 3D jumps (x grows monotonically along helix)
                if in_fiber_phase:
                    new_layer = d3 > jump_thresh
                else:
                    new_layer = (dx > layer_height * 0.6) or (d3 > jump_thresh)
                if new_layer:
                    if len(current_path) >= 2:
                        path_list.append(current_path)
                        layer_types.append(current_type)
                        for j in range(len(current_path)-1):
                            total_len += _math.sqrt(
                                sum((current_path[j+1][k]-current_path[j][k])**2 for k in range(3)))
                    current_x = round(new_x, 4)
                    current_r = round(new_r, 4)
                    current_path = [p1]

                    # Determine type of the NEW segment:
                    #   strategy=="fill"     → always "fill"
                    #   strategy=="combined" → "fill" until winding starts;
                    #     winding starts with a large d3 jump to x≈0 or x≈length
                    if strategy == "fill":
                        current_type = "fill"
                    else:
                        # Only flip to fiber phase when we've finished the fill
                        # region (current_x near length) AND jump to CF winding
                        # start (new_x near 0 or near length).  This prevents the
                        # large wall→scan-line diameter jump (d3≈2r) on the very
                        # first fill layer from being mistaken for a fill→CF transition.
                        if (d3 > jump_thresh
                                and current_x > length * 0.8
                                and (new_x < layer_height
                                     or abs(new_x - length) < layer_height)):
                            in_fiber_phase = True
                        current_type = "fiber" if in_fiber_phase else "fill"
                else:
                    current_path.append(p1)

            if len(current_path) >= 2:
                path_list.append(current_path)
                layer_types.append(current_type)
                for j in range(len(current_path)-1):
                    total_len += _math.sqrt(
                        sum((current_path[j+1][k]-current_path[j][k])**2 for k in range(3)))

        else:
            # --- winding strategies: split at large 3D jumps ---
            current_path = [pts_3d[0]]
            for i in range(1, len(pts_3d)):
                p0, p1 = pts_3d[i-1], pts_3d[i]
                d = _math.sqrt(sum((p1[k]-p0[k])**2 for k in range(3)))
                if d > jump_thresh:
                    if len(current_path) >= 2:
                        path_list.append(current_path)
                        layer_types.append("fiber")
                        for j in range(len(current_path)-1):
                            total_len += _math.sqrt(
                                sum((current_path[j+1][k]-current_path[j][k])**2 for k in range(3)))
                    current_path = [p1]
                else:
                    current_path.append(p1)
            if len(current_path) >= 2:
                path_list.append(current_path)
                layer_types.append("fiber")
                for j in range(len(current_path)-1):
                    total_len += _math.sqrt(
                        sum((current_path[j+1][k]-current_path[j][k])**2 for k in range(3)))

    # ── live_paths.json (legacy compat, subsampled) ──────────────────────
    path_list_sub = [p[::max(1, len(p)//60)] for p in path_list]
    with open(os.path.join(DATA_DIR, "live_paths.json"), "w") as f:
        json.dump({
            "paths": path_list_sub,
            "n_paths": len(path_list_sub),
            "n_paths_total": len(path_list),
            "total_length_mm": round(total_len, 1),
        }, f)

    # ── volumetric fill rate ──────────────────────────────────────────────
    # fill_len = total path length of all fill-type segments
    # fill_volume ≈ fill_len × extrusion_width × layer_height
    # cylinder_volume = π(r_outer² − r_inner²) × length
    import math as _math2
    fill_len = 0.0
    fiber_len = 0.0
    for i, p in enumerate(path_list):
        seg_len = sum(
            _math2.sqrt(sum((p[j+1][k]-p[j][k])**2 for k in range(3)))
            for j in range(len(p)-1)
        )
        if i < len(layer_types) and layer_types[i] == "fill":
            fill_len += seg_len
        else:
            fiber_len += seg_len

    fill_volume = fill_len * extrusion_width * layer_height
    cyl_volume = _math2.pi * (radius**2 - r_inner**2) * length
    fill_vf = (fill_volume / cyl_volume) if cyl_volume > 0 else 0.0

    # ── live_layers.json — layer-by-layer paths for the viewport slider ──
    MAX_PTS_PER_RING = 200    # enough for a smooth circle
    MAX_VIS_SEGMENTS = 2000   # total segments (rings/passes) sent to browser

    if strategy in ("fill", "combined"):
        # CRITICAL: build slice_map from FILL paths only.
        # Fiber paths start at x≈0 or x≈length; including them inflates
        # rings_per_slice to n_fiber_paths ≈ 1000+, which collapses
        # max_slices to 1 and shows only one fill cross-section.
        fill_indices  = [i for i, t in enumerate(layer_types) if t == "fill"]
        fiber_indices = [i for i, t in enumerate(layer_types) if t == "fiber"]

        # Group fill segments by axial x-position
        slice_map: dict = {}
        for idx in fill_indices:
            p = path_list[idx]
            if not p:
                continue
            x_key = round(p[0][2], 3)   # pts[0][2] = x_axial in 3D
            slice_map.setdefault(x_key, []).append(idx)

        x_keys_sorted = sorted(slice_map.keys())

        # Budget: 2/3 of MAX_VIS_SEGMENTS for fill slices, 1/3 for fiber passes
        fill_budget  = MAX_VIS_SEGMENTS * 2 // 3
        fiber_budget = MAX_VIS_SEGMENTS - fill_budget

        rings_per_slice = max(1, max((len(v) for v in slice_map.values()), default=1))
        max_slices = max(1, fill_budget // rings_per_slice)

        # Pick evenly-spaced slice keys to show
        if len(x_keys_sorted) <= max_slices:
            shown_keys = set(x_keys_sorted)
        else:
            step_f = len(x_keys_sorted) / max_slices
            shown_keys = set(x_keys_sorted[int(i * step_f)]
                             for i in range(max_slices))
            shown_keys.add(x_keys_sorted[-1])  # always include last slice

        layers_vis = []
        layer_types_vis = []
        for x_key in x_keys_sorted:
            if x_key not in shown_keys:
                continue
            for idx in slice_map[x_key]:
                p = path_list[idx]
                step = max(1, len(p) // MAX_PTS_PER_RING)
                layers_vis.append(p[::step])
                layer_types_vis.append("fill")

        # Fiber passes: evenly subsample to fit fiber_budget
        fiber_step = max(1, len(fiber_indices) // fiber_budget)
        for idx in fiber_indices[::fiber_step]:
            p = path_list[idx]
            step = max(1, len(p) // MAX_PTS_PER_RING)
            layers_vis.append(p[::step])
            layer_types_vis.append("fiber")
    else:
        # Winding strategies: simple per-segment subsample
        layers_vis = []
        layer_types_vis = []
        for idx, p in enumerate(path_list):
            step = max(1, len(p) // MAX_PTS_PER_RING)
            layers_vis.append(p[::step])
            layer_types_vis.append(layer_types[idx] if idx < len(layer_types) else "fiber")
        # Global point cap
        total_vis = sum(len(lp) for lp in layers_vis)
        if total_vis > 200000:
            thin = max(2, int(total_vis / 200000))
            layers_vis = [lp[::thin] for lp in layers_vis]
    layer_types = layer_types_vis

    with open(os.path.join(DATA_DIR, "live_layers.json"), "w") as f:
        json.dump({
            "n_layers": len(layers_vis),
            "strategy": strategy,
            "layer_types": layer_types,
            "layers": layers_vis,
        }, f)

    # live_gcode.json — positions + is_print for G-code replay animation
    sub_step = max(1, len(pts_3d) // 8000)
    gc_pos = pts_3d[::sub_step]
    gc_is_print = [True] * len(gc_pos)
    with open(os.path.join(DATA_DIR, "live_gcode.json"), "w") as f:
        json.dump({"positions": gc_pos, "is_print": gc_is_print}, f)

    # live_topo.json — coverage density map on surface
    if state["surface"] is not None:
        from scipy.spatial import cKDTree as _KDT
        centroids = state["surface"].centroids
        tree = _KDT(np.array(pts_3d))
        dists, _ = tree.query(centroids, k=1)
        densities = np.clip(1.0 - dists / max(radius * 0.25, 2.0), 0, 1).tolist()
        idx = (np.random.choice(len(centroids), 8000, replace=False)
               if len(centroids) > 8000 else np.arange(len(centroids)))
        with open(os.path.join(DATA_DIR, "live_topo.json"), "w") as f:
            json.dump({
                "centroids": centroids[idx].tolist(),
                "densities": [densities[i] for i in idx],
            }, f)

    fill_vis_count  = sum(1 for t in layer_types if t == "fill")
    fiber_vis_count = sum(1 for t in layer_types if t == "fiber")

    return {
        "status": "ok",
        "n_waypoints": len(waypoints),
        "radius": radius,
        "length": length,
        "r_inner": r_inner,
        "strategy": strategy,
        "angle": angle,
        "n_layers": n_layers,
        "n_vis_layers": len(layers_vis),
        "fill_vis_layers": fill_vis_count,
        "fiber_vis_layers": fiber_vis_count,
        "total_length_mm": round(total_len, 1),
        "fill_length_mm": round(fill_len, 1),
        "fill_volume_mm3": round(fill_volume, 1),
        "cyl_volume_mm3": round(cyl_volume, 1),
        "fill_vf": round(fill_vf, 4),
        "fill_vf_pct": round(fill_vf * 100, 1),
        "a_min": round(a_min, 1),
        "a_max": round(a_max, 1),
    }


def api_gcode(params):
    """Generate G-code for the XY+A axis system.

    Uses state["xyza_waypoints"]. Run /api/xyza_paths first.

    Parameters:
      feed_rate       — mm/min (default: 600)
      a_offset_z      — A-axis chuck center height above bed in mm (default: 50.0)
      layer_height    — mm (default: 0.2)
      extrusion_width — mm (default: 0.4)
    """
    if state["xyza_waypoints"] is None:
        return {"error": "No waypoints. Run /api/xyza_paths first."}

    from cfpp.gcode.xyza_backend import PrinterConfig, waypoints_to_gcode

    feed_rate = float(params.get("feed_rate", 600))
    a_offset_z = float(params.get("a_offset_z", 50.0))
    layer_height = float(params.get("layer_height", 0.2))
    extrusion_width = float(params.get("extrusion_width", 0.4))

    config = PrinterConfig(
        a_offset_y=0.0,
        a_offset_z=a_offset_z,
        print_speed=feed_rate,
        travel_speed=feed_rate * 5,
        extrusion_height=layer_height,
        extrusion_width=extrusion_width,
    )

    waypoints = state["xyza_waypoints"]
    gcode_str = waypoints_to_gcode(waypoints, config)

    os.makedirs(DATA_DIR, exist_ok=True)
    gcode_path = os.path.join(DATA_DIR, "output.gcode")
    with open(gcode_path, "w") as f:
        f.write(gcode_str)

    state["gcode_path"] = gcode_path

    a_vals = [wp[3] for wp in waypoints]
    fiber_mm = float(sum(
        np.sqrt(sum((waypoints[i+1][j] - waypoints[i][j])**2 for j in range(3)))
        for i in range(len(waypoints) - 1)
    ))

    return {
        "status": "ok",
        "axis_system": "xyza",
        "n_lines": gcode_str.count("\n"),
        "n_waypoints": len(waypoints),
        "fiber_mm": round(fiber_mm, 1),
        "travel_mm": 0.0,
        "a_min": round(min(a_vals), 1) if a_vals else 0.0,
        "a_max": round(max(a_vals), 1) if a_vals else 0.0,
        "file": "/data/output.gcode",
    }


def api_optimize(params):
    """Run layup optimization."""
    if state["mesh_path"] is None:
        return {"error": "No mesh. Run /api/mesh first."}

    from cfpp.optimizer.layup import LayupOptimizer

    n_starts = int(params.get("n_starts", 6))
    max_iter = int(params.get("max_iter", 15))

    # Detect geometry
    radii = np.sqrt(state["surface"].centroids[:, 0]**2 +
                    state["surface"].centroids[:, 1]**2)
    r_outer = float(np.percentile(radii, 95))
    r_inner = float(np.percentile(radii, 5))

    import meshio
    m = meshio.read(state["mesh_path"])
    z_max = m.points[:, 2].max()
    top_pts = m.points[m.points[:, 2] > z_max - 1]
    r_vals = np.sqrt(top_pts[:, 0]**2 + top_pts[:, 1]**2)
    A_top = np.pi * (r_vals.max()**2 - r_vals.min()**2)
    P = float(params.get("P", 500))
    traction_x = P / max(A_top, 1.0)

    optimizer = LayupOptimizer(
        state["mesh_path"],
        n_layers=5,
        r_outer=r_outer, r_inner=r_inner)

    result = optimizer.optimize(
        traction=(traction_x, 0, 0),
        n_starts=n_starts, max_iter=max_iter)

    return {
        "status": "ok",
        "optimal_layup": result["best"]["alpha_opt"],
        "current_layup": result["current_layup"],
        "C_optimal": result["best"]["C_final"],
        "C_current": result["C_current"],
        "improvement_pct": result["improvement_pct"],
        "disp_max": result["best"]["disp_max"],
        "n_starts": n_starts,
        "best_start": result["best"]["start_idx"],
        "n_iterations": result["best"]["n_iter"],
    }


MODELS_DIR = os.path.join(VIZ_DIR, "models")

MODELS_META = {
    # ── 基础管件 ──────────────────────────────────────────────────
    "hollow_tube":     {"name": "空心圆管",    "desc": "基础圆柱管件，CF缠绕演示",   "icon": "🔵"},
    "t_joint":         {"name": "T形三通",     "desc": "管道三通接头",               "icon": "🔀"},
    "cross_joint":     {"name": "十字四通",    "desc": "X形管道交叉接头",            "icon": "➕"},
    "elbow_pipe":      {"name": "90°弯管",     "desc": "直角弯头，各向异性明显",     "icon": "↩️"},
    "y_junction":      {"name": "Y形分叉",     "desc": "45°管道分叉接头",            "icon": "🍴"},
    "flange":          {"name": "法兰盘",      "desc": "标准管道法兰连接件",          "icon": "🔩"},
    "s_curve_pipe":    {"name": "S形弯管",     "desc": "双向弯曲管道",               "icon": "〰️"},
    "helix_coil":      {"name": "螺旋线圈",    "desc": "多圈螺旋管，弹簧形态",        "icon": "🌀"},
    # ── 结构件 ────────────────────────────────────────────────────
    "l_bracket":       {"name": "L形支架",     "desc": "结构支架，应力集中典型",      "icon": "📐"},
    "i_beam":          {"name": "工字梁",      "desc": "标准工字截面结构梁",          "icon": "🏗️"},
    "wing_spar":       {"name": "翼梁截面",    "desc": "航空矩形中空翼梁",            "icon": "✈️"},
    "saddle_bracket":  {"name": "U形夹座",     "desc": "管道鞍形夹紧支座",            "icon": "🐎"},
    "box_frame":       {"name": "矩形箱梁",    "desc": "封闭矩形截面箱型梁",          "icon": "📦"},
    "rect_frame":      {"name": "矩形框架",    "desc": "平面矩形边框结构",            "icon": "⬜"},
    "lattice_cube":    {"name": "点阵框架",    "desc": "空间桁架十二棱骨架",          "icon": "🔲"},
    "curved_beam":     {"name": "C形曲梁",     "desc": "半圆环形曲线梁",              "icon": "🌙"},
    # ── 壳体 ──────────────────────────────────────────────────────
    "pressure_vessel": {"name": "压力容器",    "desc": "球端封头圆柱壳体",            "icon": "🫙"},
    "dome":            {"name": "半球壳",      "desc": "薄壁半球形封头",              "icon": "⛺"},
    "sphere_shell":    {"name": "球形壳",      "desc": "完整球形薄壁壳体",            "icon": "🔮"},
    "thermos":         {"name": "双层筒",      "desc": "双壁圆柱容器",               "icon": "🧊"},
    "cone_shell":      {"name": "锥形壳",      "desc": "空心圆锥台壳体",              "icon": "🔺"},
    "nozzle":          {"name": "收缩喷管",    "desc": "航空发动机缩放喷嘴",          "icon": "🚀"},
    # ── 机械零件 ──────────────────────────────────────────────────
    "propeller_hub":   {"name": "四轴桨座",    "desc": "无人机电机座 + 四臂",         "icon": "🚁"},
    "hex_tube":        {"name": "六棱管",      "desc": "六边形截面蜂窝管件",          "icon": "⬡"},
    "bearing_housing": {"name": "轴承座",      "desc": "圆柱滚子轴承外壳",            "icon": "⚙️"},
    "stepped_shaft":   {"name": "阶梯轴",      "desc": "三段变径传动轴",              "icon": "🎯"},
    "torus_shell":     {"name": "环形壳",      "desc": "圆形截面空心圆环",            "icon": "💍"},
    "annular_disk":    {"name": "环形盘",      "desc": "平面圆环 / 垫片",             "icon": "🪙"},
    # ── 航空航天 ──────────────────────────────────────────────────
    "turbine_blade":   {"name": "涡轮叶片",    "desc": "翼型截面叶片，CF铺层关键",    "icon": "🌬️"},
    "rocket_fin":      {"name": "火箭尾翼",    "desc": "梯形平板尾翼稳定面",          "icon": "🛸"},
}


def api_list_models():
    models = []
    for key, meta in MODELS_META.items():
        stl_path = os.path.join(MODELS_DIR, f"{key}.stl")
        if os.path.exists(stl_path):
            models.append({
                "id": key,
                "name": meta["name"],
                "desc": meta["desc"],
                "icon": meta["icon"],
                "url": f"/models/{key}.stl",
                "size": os.path.getsize(stl_path),
            })
    return {"models": models}


# ======================================================================
# HTTP Server with API routing
# ======================================================================

class CFPPHandler(SimpleHTTPRequestHandler):
    """Serves static files from VIZ_DIR + handles /api/* endpoints."""

    def translate_path(self, path):
        """Override to serve from VIZ_DIR instead of cwd."""
        # Don't translate API paths
        if path.startswith("/api/"):
            return path
        # Serve from VIZ_DIR
        path = super().translate_path(path)
        # Replace cwd with VIZ_DIR
        rel = os.path.relpath(path, os.getcwd())
        return os.path.join(VIZ_DIR, rel)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            self._handle_api(parsed.path, params)
        else:
            super().do_GET()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload_stl":
            self._handle_stl_upload()
        elif parsed.path.startswith("/api/"):
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else "{}"
            try:
                params = json.loads(body)
            except json.JSONDecodeError:
                params = {}
            self._handle_api(parsed.path, params)
        else:
            self.send_error(405)

    def _handle_stl_upload(self):
        """Handle STL file upload via multipart form data (no cgi module)."""
        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self._send_json({"error": "Expected multipart/form-data"})
            return

        # Read entire body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        # Extract boundary from content-type
        boundary = None
        for part in content_type.split(';'):
            part = part.strip()
            if part.startswith('boundary='):
                boundary = part.split('=', 1)[1].strip()
                break
        if not boundary:
            self._send_json({"error": "No boundary in multipart"})
            return

        # Split by boundary and find the file part
        boundary_bytes = ('--' + boundary).encode()
        parts = body.split(boundary_bytes)

        stl_data = None
        mesh_size = 2.5
        for part in parts:
            if b'Content-Disposition' not in part:
                continue
            # Parse headers
            header_end = part.find(b'\r\n\r\n')
            if header_end < 0:
                continue
            headers_str = part[:header_end].decode('utf-8', errors='ignore')
            file_data = part[header_end + 4:]
            # Remove trailing \r\n--
            if file_data.endswith(b'\r\n'):
                file_data = file_data[:-2]
            if file_data.endswith(b'--'):
                file_data = file_data[:-2]
            if file_data.endswith(b'\r\n'):
                file_data = file_data[:-2]

            if 'name="file"' in headers_str:
                stl_data = file_data
            elif 'name="mesh_size"' in headers_str:
                try:
                    mesh_size = float(file_data.decode().strip())
                except (ValueError, UnicodeDecodeError):
                    pass

        if stl_data is None or len(stl_data) < 100:
            self._send_json({"error": "No STL file data found"})
            return

        # Save uploaded file
        stl_path = os.path.join(tempfile.gettempdir(),
                                f"upload_{uuid.uuid4().hex[:8]}.stl")
        with open(stl_path, 'wb') as f:
            f.write(stl_data)

        # Convert STL to mesh
        import subprocess
        mesh_path = stl_path.replace('.stl', '.msh')
        code = (f'import sys; sys.path.insert(0,"{PROJECT_ROOT}")\n'
                f'from cfpp.mesh.generator import stl_to_mesh\n'
                f'stl_to_mesh("{stl_path}", {mesh_size}, output_path="{mesh_path}")')
        result = subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            self._send_json({"error": result.stderr[-300:]})
            return

        state["mesh_path"] = mesh_path
        state["surface"] = None
        state["surf_stress"] = None
        state["xyza_waypoints"] = None

        self._send_json({
            "status": "ok",
            "mesh_path": mesh_path,
            "info": f"STL imported: {os.path.basename(stl_path)}",
        })

    def _handle_api(self, path, params=None):
        """Route API calls."""
        if params is None:
            params = {}
        endpoint = path.replace("/api/", "").strip("/")

        try:
            if endpoint == "status":
                result = {"status": "ok", "version": "1.0"}
            elif endpoint == "models":
                result = api_list_models()
            elif endpoint == "mesh":
                result = api_mesh(params)
            elif endpoint == "fea":
                result = api_fea(params)
            elif endpoint == "xyza_paths":
                result = api_xyza_paths(params)
            elif endpoint == "gcode":
                result = api_gcode(params)
            elif endpoint == "optimize":
                result = api_optimize(params)
            else:
                result = {"error": f"Unknown endpoint: {endpoint}"}
        except Exception as e:
            result = {"error": str(e), "traceback": traceback.format_exc()[-500:]}

        try:
            self._send_json(result)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # Client disconnected, don't crash server

    def _send_json(self, data):
        try:
            body = json.dumps(data).encode('utf-8')
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def handle(self):
        """Override to catch ALL exceptions in request handling."""
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as e:
            _safe_print(f"[ERROR] Request handler: {e}")

    def log_message(self, format, *args):
        try:
            msg = str(args[0]) if args else ""
            if "/api/" in msg:
                _safe_print(f"[API] {msg}")
        except Exception:
            pass


def main():
    from http.server import ThreadingHTTPServer
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadingHTTPServer(("0.0.0.0", port), CFPPHandler)
    _safe_print(f"CF-Path-Planner server: http://localhost:{port}/")
    _safe_print(f"  Viz dir: {VIZ_DIR}")
    _safe_print(f"  API: /api/mesh, /api/fea, /api/xyza_paths, /api/gcode, /api/optimize")
    server.serve_forever()


if __name__ == "__main__":
    main()
