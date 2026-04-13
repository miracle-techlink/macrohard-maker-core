"""
业务逻辑层 — 把 cfpp 算法包装成 JobManager 可调用的函数。
每个函数签名：fn(params: dict, job_state: JobState, progress_cb) -> dict
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np

# Project root（webapp/ 的上一级）
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "..")))

# Data dir for live_*.json outputs（visualization/data/）
VIZ_DIR  = os.path.normpath(os.path.join(PROJECT_ROOT, "visualization"))
DATA_DIR = os.path.join(VIZ_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ── Helper ────────────────────────────────────────────────────────────────────

def _tmpfile(suffix: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"cfpp_{uuid.uuid4().hex[:8]}{suffix}")


# ── Mesh ─────────────────────────────────────────────────────────────────────

def run_mesh(params: dict, job_state, progress_cb=None) -> dict:
    """生成网格（同步，供 geometry router 直接调用或作为短任务）。"""
    model     = params.get("model", "cylinder")
    mesh_size = float(params.get("mesh_size", 3.0))
    mesh_path = _tmpfile(".msh")

    if model == "cylinder":
        r_outer = float(params.get("r_outer", 25))
        r_inner = float(params.get("r_inner", 0))
        height  = float(params.get("height", 50))
        code = (f'import sys; sys.path.insert(0,"{PROJECT_ROOT}")\n'
                f'from cfpp.mesh.generator import create_cylinder_mesh\n'
                f'create_cylinder_mesh({r_outer},{r_inner},{height},{mesh_size},"{mesh_path}")')
    elif model == "cone":
        r_bottom = float(params.get("r_bottom", 25))
        r_top    = float(params.get("r_top", 15))
        height   = float(params.get("height", 60))
        wall     = float(params.get("wall", 5))
        code = (f'import sys; sys.path.insert(0,"{PROJECT_ROOT}")\n'
                f'from cfpp.mesh.generator import create_cone_mesh\n'
                f'create_cone_mesh({r_bottom},{r_top},{height},{wall},{mesh_size},"{mesh_path}")')
    else:
        raise ValueError(f"Unknown model: {model}")

    if progress_cb:
        progress_cb("mesh_generate", 10, f"生成 {model} 网格...")

    t0  = time.time()
    res = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"Mesh generation failed: {res.stderr[-300:]}")

    job_state.mesh_path      = mesh_path
    job_state.waypoints_path = None
    job_state.gcode_path     = None

    if progress_cb:
        progress_cb("mesh_generate", 100, f"网格就绪 {time.time()-t0:.2f}s")

    return {"status": "ok", "mesh_path": mesh_path,
            "elapsed_s": round(time.time()-t0, 2), "info": res.stdout.strip()}


def run_builtin_mesh(params: dict, job_state, progress_cb=None) -> dict:
    from webapp.server import MODELS_META  # 复用旧代码的元数据
    model_key = params.get("model_key", "")
    mesh_size = float(params.get("mesh_size", 3.0))
    mesh_path = os.path.join(tempfile.gettempdir(), f"builtin_{model_key}.msh")
    code = (f'import sys; sys.path.insert(0,"{PROJECT_ROOT}")\n'
            f'from cfpp.mesh.generator import create_builtin_mesh\n'
            f'create_builtin_mesh("{model_key}", {mesh_size}, "{mesh_path}")')
    res = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(res.stderr[-400:])

    job_state.mesh_path      = mesh_path
    job_state.waypoints_path = None
    job_state.gcode_path     = None
    return {"status": "ok", "mesh_path": mesh_path}


# ── Path Planning ─────────────────────────────────────────────────────────────

def run_plan(params: dict, job_state, progress_cb=None) -> dict:
    """XY+A 路径规划，写 live_layers.json，结果存入 job_state。"""
    mesh_path = job_state.mesh_path or params.get("mesh_path")
    if not mesh_path or not os.path.exists(mesh_path):
        raise RuntimeError("No mesh available. Submit a mesh job first.")
    job_state.mesh_path = mesh_path

    from cfpp.surface.planner_v2 import XYAPathPlanner
    from cfpp.surface.extract    import extract_surface

    strategy        = params.get("strategy", "combined")
    angle           = float(params.get("angle", 45.0))
    n_layers        = int(params.get("n_layers", 4))
    n_walls         = int(params.get("n_walls", 2))
    infill_density  = float(params.get("infill_density", 0.25))
    layer_height    = float(params.get("layer_height", 0.18))
    ext_width       = float(params.get("extrusion_width", 0.4))
    r_inner         = float(params.get("r_inner", 0.0))
    a_offset_z      = float(params.get("a_offset_z", 50.0))

    if progress_cb:
        progress_cb("plan_init", 5, "初始化规划器...")

    surface = extract_surface(mesh_path)
    centroids = surface.centroids
    radii = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)
    auto_radius = float(np.percentile(radii, 90)) if len(radii) else 25.0
    radius = float(params.get("radius", auto_radius))
    length = float(params.get("length", float(centroids[:, 2].max() - centroids[:, 2].min())))

    if progress_cb:
        progress_cb("plan_paths", 15, f"combined_print_paths r={radius:.1f} l={length:.1f}...")

    planner = XYAPathPlanner(a_offset_y=0.0, a_offset_z=a_offset_z)
    t0 = time.time()
    waypoints = planner.combined_print_paths(
        radius=radius, length=length, r_inner=r_inner,
        winding_angle_deg=angle, n_cf_layers=n_layers, n_walls=n_walls,
        infill_density=infill_density,
        layer_height=layer_height, extrusion_width=ext_width,
    )
    t_plan = time.time() - t0

    if progress_cb:
        progress_cb("plan_convert", 60, f"{len(waypoints)} 路径点，转换 3D...")

    # 3D 坐标转换 — 与 server.py _wp_to_3d 完全一致
    # waypoint: (x_axis, y_model, z_model=radius, a_deg)
    # 3D output: [radius*cos(A), radius*sin(A), x_axis]
    def _wp_to_3d(wp):
        x_axis, _y_model, z_model, a_deg = wp
        a = math.radians(a_deg)
        return [float(z_model * math.cos(a)),
                float(z_model * math.sin(a)),
                float(x_axis)]

    pts_3d = [_wp_to_3d(wp) for wp in waypoints]

    if progress_cb:
        progress_cb("plan_split", 70, "切分可视化层...")

    import json

    # 层切分 — 与 server.py combined 策略一致
    jump_thresh = radius * 0.5
    all_segs, layer_types = [], []
    current_path = [pts_3d[0]]
    in_fiber_phase = False
    current_type = "fill"

    for i in range(1, len(pts_3d)):
        p0, p1 = pts_3d[i-1], pts_3d[i]
        dx = abs(p1[2] - p0[2])   # z = x_axis
        d3 = math.sqrt(sum((p1[k]-p0[k])**2 for k in range(3)))
        if in_fiber_phase:
            new_layer = d3 > jump_thresh
        else:
            new_layer = (dx > layer_height * 0.6) or (d3 > jump_thresh)
        if new_layer:
            if len(current_path) >= 2:
                all_segs.append(current_path)
                layer_types.append(current_type)
            if not in_fiber_phase and current_type == "fill":
                a_prev = waypoints[i-1][3]
                a_cur  = waypoints[i][3]
                if abs(a_cur - a_prev) > 10 or (i > len(waypoints) * 0.4 and current_type == "fill"):
                    in_fiber_phase = True
            current_type = "fiber" if in_fiber_phase else "fill"
            current_path = []
        current_path.append(p1)
    if len(current_path) >= 2:
        all_segs.append(current_path)
        layer_types.append(current_type)

    # 可视化预算采样（最多 1342 段，与旧版一致）
    MAX_VIS = 1342
    MAX_PTS = 200
    fill_idx  = [i for i,t in enumerate(layer_types) if t == "fill"]
    fiber_idx = [i for i,t in enumerate(layer_types) if t == "fiber"]
    fill_budget  = MAX_VIS * 2 // 3
    fiber_budget = MAX_VIS - fill_budget

    layers_vis, layer_types_vis = [], []
    fill_step  = max(1, len(fill_idx)  // fill_budget)
    fiber_step = max(1, len(fiber_idx) // fiber_budget)
    for idx in fill_idx[::fill_step]:
        p = all_segs[idx]
        layers_vis.append(p[::max(1, len(p)//MAX_PTS)])
        layer_types_vis.append("fill")
    for idx in fiber_idx[::fiber_step]:
        p = all_segs[idx]
        layers_vis.append(p[::max(1, len(p)//MAX_PTS)])
        layer_types_vis.append("fiber")

    if progress_cb:
        progress_cb("plan_write", 85, f"写 live_layers.json ({len(layers_vis)} 层)...")

    # live_paths.json（原始 waypoints，供 G-code 使用）
    wps_path = os.path.join(DATA_DIR, "live_paths.json")
    with open(wps_path, "w") as f:
        json.dump({"waypoints": [list(w) for w in waypoints]}, f)

    # live_layers.json — 格式与 server.py 完全一致，index.html 直接读取
    layers_path = os.path.join(DATA_DIR, "live_layers.json")
    with open(layers_path, "w") as f:
        json.dump({
            "n_layers":    len(layers_vis),
            "strategy":    strategy,
            "layer_types": layer_types_vis,
            "layers":      layers_vis,
        }, f)

    # 保存 waypoints 到 job_state（通过路径传递，避免跨线程大对象）
    wp_arr = np.array(waypoints, dtype=np.float32)
    wp_npz = _tmpfile(".npz")
    np.savez_compressed(wp_npz, waypoints=wp_arr)
    job_state.waypoints_path = wp_npz
    job_state.vis_path       = layers_path

    diffs = np.diff(wp_arr[:, :3], axis=0)
    total_len  = float(np.sum(np.linalg.norm(diffs, axis=1)))
    fill_count  = sum(1 for t in layer_types if t == "fill")
    fiber_count = len(layer_types) - fill_count

    if progress_cb:
        progress_cb("plan_done", 100, f"完成 {time.time()-t0:.2f}s")

    return {
        "status":           "ok",
        "n_waypoints":      len(waypoints),
        "n_vis_layers":     len(layers_vis),
        "fill_vis_layers":  fill_count,
        "fiber_vis_layers": fiber_count,
        "radius":           round(radius, 2),
        "length":           round(length, 2),
        "r_inner":          r_inner,
        "strategy":         strategy,
        "angle":            angle,
        "n_layers":         n_layers,
        "total_length_mm":  round(total_len, 1),
        "plan_elapsed_s":   round(t_plan, 2),
    }


# ── G-code ────────────────────────────────────────────────────────────────────

def run_gcode(params: dict, job_state, progress_cb=None) -> dict:
    """从 job_state.waypoints_path 生成 G-code。"""
    if not job_state.waypoints_path or not os.path.exists(job_state.waypoints_path):
        raise RuntimeError("No waypoints. Submit a plan job first.")

    if progress_cb:
        progress_cb("gcode_load", 5, "加载路径点...")

    data = np.load(job_state.waypoints_path)
    waypoints = data["waypoints"].tolist()

    feed_rate    = float(params.get("feed_rate", 3000.0))
    layer_height = float(params.get("layer_height", 0.18))
    ext_width    = float(params.get("extrusion_width", 0.4))
    travel_rate  = feed_rate * 2
    filament_r   = 0.875
    e_per_mm     = ext_width * layer_height / (math.pi * filament_r ** 2)

    if progress_cb:
        progress_cb("gcode_gen", 10, f"生成 G-code n={len(waypoints)}...")

    t0  = time.time()
    buf = [
        "; CF-Path-Planner XYZA G-code",
        f"; waypoints={len(waypoints)}  feed={feed_rate}mm/min",
        "G28 ; home all",
        "G92 E0",
        f"M104 S240 ; nozzle temp",
        f"M140 S80  ; bed temp",
        "M109 S240",
        "M190 S80",
        f"G1 F{travel_rate:.0f}",
    ]
    E = 0.0
    prev_x = prev_y = prev_z = None
    for wp in waypoints:
        x, y, z, a = wp
        if prev_x is None:
            buf.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} A{a:.2f} E0.0000 F{feed_rate:.0f}")
        else:
            dx = x - prev_x; dy = y - prev_y; dz = z - prev_z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            E += dist * e_per_mm
            buf.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} A{a:.2f} E{E:.4f}")
        prev_x, prev_y, prev_z = x, y, z
    buf += ["M104 S0 ; cool down", "M140 S0"]
    gcode_str = "\n".join(buf)

    if progress_cb:
        progress_cb("gcode_write", 90, f"写文件 {len(gcode_str)//1024}KB...")

    gcode_path = os.path.join(DATA_DIR, "output.gcode")
    with open(gcode_path, "w") as f:
        f.write(gcode_str)

    job_state.gcode_path = gcode_path

    wps_arr = np.array(waypoints, dtype=np.float32)
    a_vals  = wps_arr[:, 3]
    diffs   = np.diff(wps_arr[:, :3], axis=0)
    fiber_mm = float(np.sum(np.linalg.norm(diffs, axis=1)))

    if progress_cb:
        progress_cb("gcode_done", 100, f"完成 {time.time()-t0:.2f}s")

    return {
        "status":      "ok",
        "axis_system": "xyza",
        "n_lines":     gcode_str.count("\n"),
        "n_waypoints": len(waypoints),
        "fiber_mm":    round(fiber_mm, 1),
        "a_min":       round(float(a_vals.min()), 1) if len(a_vals) else 0.0,
        "a_max":       round(float(a_vals.max()), 1) if len(a_vals) else 0.0,
        "file":        "/data/output.gcode",
        "elapsed_s":   round(time.time()-t0, 2),
    }


# ── FEA ───────────────────────────────────────────────────────────────────────

def run_fea(params: dict, job_state, progress_cb=None) -> dict:
    """FEA 分析 — 直接移植旧 api_fea 逻辑。"""
    mesh_path = job_state.mesh_path or params.get("mesh_path")
    if not mesh_path or not os.path.exists(mesh_path):
        raise RuntimeError("No mesh. Submit a mesh job first.")
    job_state.mesh_path = mesh_path

    if progress_cb:
        progress_cb("fea_setup", 5, "初始化 FEA 求解器...")

    from cfpp.solver.elastic      import ElasticSolver
    from cfpp.surface.extract     import extract_surface
    from cfpp.surface.stress_field import SurfaceStressField
    import meshio, json

    E  = float(params.get("E_gpa", 60)) * 1e3
    nu = float(params.get("nu", 0.3))
    P  = float(params.get("P", 500))

    solver = ElasticSolver(mesh_path)
    solver.set_isotropic_material(E, nu)

    m     = meshio.read(mesh_path)
    z_min = float(m.points[:, 2].min())
    z_max = float(m.points[:, 2].max())
    z_tol = (z_max - z_min) * 0.05

    top_pts = m.points[m.points[:, 2] > z_max - z_tol]
    if len(top_pts) > 3:
        from scipy.spatial import ConvexHull
        try:
            hull  = ConvexHull(top_pts[:, :2])
            A_top = float(hull.volume)
        except Exception:
            A_top = float(math.pi * np.max(np.sqrt(top_pts[:,0]**2 + top_pts[:,1]**2))**2)
    else:
        A_top = 100.0
    A_top = max(A_top, 1.0)
    traction_x = P / A_top

    if progress_cb:
        progress_cb("fea_solve", 30, "组装刚度矩阵并求解...")

    from skfem import Basis, ElementVector, ElementTetP1, FacetBasis, LinearForm, solve, condense
    from skfem.models.elasticity import linear_elasticity, lame_parameters

    e   = ElementVector(ElementTetP1(), 3)
    ib  = Basis(solver.mesh_sk, e)
    solver._basis = ib
    lam, mu = lame_parameters(solver.E, solver.nu)
    K = linear_elasticity(lam, mu).assemble(ib)

    fixed_facets = solver.mesh_sk.facets_satisfying(lambda x: np.abs(x[2]-z_min) < z_tol)
    load_facets  = solver.mesh_sk.facets_satisfying(lambda x: np.abs(x[2]-z_max) < z_tol)
    fb = FacetBasis(solver.mesh_sk, e, facets=load_facets)

    tv = np.array([traction_x, 0, 0], dtype=float)
    @LinearForm
    def surface_load(v, w):
        return tv[0]*v.value[0] + tv[1]*v.value[1] + tv[2]*v.value[2]

    f_vec     = surface_load.assemble(fb)
    fixed_dofs = ib.get_dofs(fixed_facets).all()
    solver.displacement = solve(*condense(K, f_vec, D=fixed_dofs))

    if progress_cb:
        progress_cb("fea_post", 70, "提取主应力场...")

    ps = solver.extract_principal_stresses()
    stress_npz = _tmpfile("_stress.npz")
    np.savez_compressed(stress_npz, **{k: ps[k] for k in ps})

    surface     = extract_surface(mesh_path)
    surf_stress = SurfaceStressField(surface, stress_npz)
    job_state.surface    = surface      # type: ignore
    job_state.surf_stress = surf_stress # type: ignore

    if progress_cb:
        progress_cb("fea_export", 85, "写可视化 JSON...")

    centroids  = surface.centroids
    von_mises  = surf_stress.von_mises
    stress_dir = surf_stress.dom_dir
    n = len(centroids)
    idx = np.random.choice(n, 8000, replace=False) if n > 8000 else np.arange(n)
    stress_json = {
        "centroids":  centroids[idx].tolist(),
        "von_mises":  von_mises[idx].tolist(),
        "vm_max":     float(von_mises.max()),
        "vm_min":     float(von_mises.min()),
        "sigma_1_dir": stress_dir[idx].tolist(),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "live_stress.json"), "w") as f:
        json.dump(stress_json, f)

    if progress_cb:
        progress_cb("fea_done", 100, "FEA 完成")

    return {
        "status":          "ok",
        "max_displacement": round(float(np.max(np.abs(solver.displacement))), 6),
        "max_von_mises":   round(float(von_mises.max()), 2),
        "n_surface_tris":  len(surface.triangles),
    }


# ── Optimize ──────────────────────────────────────────────────────────────────

def run_optimize(params: dict, job_state, progress_cb=None) -> dict:
    """打印方向优化 — 枚举多起点，最小化悬垂面积。"""
    mesh_path = job_state.mesh_path or params.get("mesh_path")
    if not mesh_path or not os.path.exists(mesh_path):
        raise RuntimeError("No mesh. Submit a mesh job first.")
    job_state.mesh_path = mesh_path

    n_starts = int(params.get("n_starts", 6))
    max_iter  = int(params.get("max_iter", 15))
    P         = float(params.get("P", 500.0))

    if progress_cb:
        progress_cb("optimize_init", 5, f"初始化优化器 n_starts={n_starts}...")

    from cfpp.surface.extract import extract_surface
    from scipy.optimize import minimize
    import meshio, json

    mesh = meshio.read(mesh_path)
    pts  = mesh.points

    def overhang_area(angles):
        """代理目标：表面法线与重力方向夹角 < 45° 的面积。"""
        rx, ry = float(angles[0]), float(angles[1])
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        R  = Ry @ Rx
        g  = R @ np.array([0, 0, -1])
        surface = extract_surface(mesh_path)
        normals = surface.normals
        cos_a   = normals @ g
        overhang_mask = cos_a > math.cos(math.radians(45))
        return float(overhang_mask.sum())

    best_val  = float("inf")
    best_rxry = (0.0, 0.0)
    results   = []

    for i in range(n_starts):
        pct = 10 + int(80 * i / n_starts)
        if progress_cb:
            progress_cb("optimize_search", pct, f"起点 {i+1}/{n_starts}...")
        rx0 = math.radians((i / n_starts) * 360)
        ry0 = math.radians(((i * 7) % 360))
        res = minimize(overhang_area, [rx0, ry0], method="Nelder-Mead",
                       options={"maxiter": max_iter, "xatol": 0.01, "fatol": 1})
        if res.fun < best_val:
            best_val  = res.fun
            best_rxry = (float(res.x[0]), float(res.x[1]))
        results.append({"rx": float(res.x[0]), "ry": float(res.x[1]), "score": res.fun})

    if progress_cb:
        progress_cb("optimize_done", 100, "优化完成")

    opt_json = {
        "best_rx_deg": round(math.degrees(best_rxry[0]), 2),
        "best_ry_deg": round(math.degrees(best_rxry[1]), 2),
        "best_overhang_count": int(best_val),
        "all_results": results,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "live_optimize.json"), "w") as f:
        json.dump(opt_json, f)

    return {"status": "ok", **opt_json}
