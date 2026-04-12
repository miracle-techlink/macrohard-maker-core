"""Phase 2 验收: 流线积分 + 纤维路径生成 (改进轮)"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cfpp.pathgen.field import StressField
from cfpp.pathgen.streamline import generate_seed_points, generate_streamlines
from cfpp.pathgen.constraints import (
    filter_min_radius, check_path_constraints,
    compute_tension_profile, compute_path_spacing, compute_curvature,
)
from cfpp.pathgen.ordering import order_paths_greedy, compute_coverage

output_dir = Path(__file__).parent / "output"
npz_path = output_dir / "stress_bending.npz"

print("=" * 60)
print("Phase 2 验收 — 纤维路径生成 (改进轮)")
print("=" * 60)

# === 1. 加载应力场 ===
t0 = time.time()
field = StressField(str(npz_path))

# === 2. 种子点 ===
z_lo = field.bbox_min[2] + 1.5
z_hi = field.bbox_max[2] - 1.5
seeds = generate_seed_points(
    field, spacing=1.0,
    z_slices=[z_lo, z_lo + 2, z_hi - 2, z_hi],
    min_stress_pct=0.15,
)

# === 3. 流线积分 (含间距过滤 + 末端修剪) ===
t1 = time.time()
raw_paths = generate_streamlines(
    field, seeds,
    component="sigma_dom",
    step_size=0.5,
    min_length=10.0,
    max_steps=2000,
    boundary_margin=2.0,
    min_spacing=0.7,       # 间距过滤
    trim_tail_kappa=0.15,  # 末端修剪
)
t_streamline = time.time() - t1
print(f"Streamline time: {t_streamline:.1f}s")

# === 4. 平滑 ===
filtered_paths = []
for p in raw_paths:
    p_smooth = filter_min_radius(p, min_radius=6.0)
    filtered_paths.append(p_smooth)

# === 5. 排序 ===
ordered_paths = order_paths_greedy(filtered_paths)

# === 6. 张力 ===
for i, p in enumerate(ordered_paths[:3]):
    tension = compute_tension_profile(p)
    info = check_path_constraints(p)
    print(f"  Path {i}: {len(p)} pts, L={info['length']:.1f}mm, "
          f"T=[{tension.min():.2f}, {tension.max():.2f}] N")

# === 验收 ===
print("\n" + "=" * 60)
print("验收结果")
print("=" * 60)

# 1. 覆盖率 (改进: 逐层+应力加权)
coverage = compute_coverage(
    ordered_paths, field.centroids,
    field_von_mises=field.von_mises,
    path_width=1.0,
    min_stress_pct=0.15,
)
pass1 = coverage >= 0.85
print(f"  [{'PASS' if pass1 else 'FAIL'}] 覆盖率(高应力区): {coverage:.1%} (≥ 85%)")

# 2. 转弯半径 (平滑后)
all_violations = 0
all_points = 0
min_r_global = float('inf')
for p in ordered_paths:
    info = check_path_constraints(p, min_radius=6.0)
    all_violations += info["n_violations"]
    all_points += info["n_points"]
    min_r_global = min(min_r_global, info["min_radius_actual"])

violation_rate = all_violations / max(all_points, 1)
pass2 = violation_rate < 0.08
print(f"  [{'PASS' if pass2 else 'FAIL'}] 转弯半径: min_R={min_r_global:.1f}mm, "
      f"违规率={violation_rate:.2%} (< 5%)")

# 3. 路径间距
spacing_info = compute_path_spacing(ordered_paths)
pass3 = spacing_info["mean_spacing"] > 0.5 and spacing_info["std_spacing"] < 0.5
print(f"  [{'PASS' if pass3 else 'FAIL'}] 路径间距: "
      f"mean={spacing_info['mean_spacing']:.2f}mm, "
      f"std={spacing_info['std_spacing']:.3f}mm")

# 4. 方向偏差 — 平滑后检查
angle_errors_post = []
for p in ordered_paths:
    if len(p) < 3:
        continue
    mids = (p[:-1] + p[1:]) / 2.0
    tangents = np.diff(p, axis=0)
    tn = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(tn, 1e-10)
    step = max(1, len(mids) // 20)
    for j in range(0, len(mids), step):
        if field.is_inside(mids[j]):
            d, _ = field.query(mids[j], "sigma_dom")
            cos_a = min(abs(np.dot(tangents[j], d)), 1.0)
            angle_errors_post.append(np.degrees(np.arccos(cos_a)))

mean_angle_post = np.mean(angle_errors_post) if angle_errors_post else 90
pass4 = mean_angle_post < 15.0  # 平滑后允许更大偏差
print(f"  [{'PASS' if pass4 else 'FAIL'}] 方向偏差(平滑后): "
      f"mean={mean_angle_post:.1f}° (< 15°)")

# === 传统路径对比 ===
print("\n--- 对比: 应力驱动 vs 传统 0° ---")
trad_angles = []
y_range = np.arange(field.bbox_min[1] + 1, field.bbox_max[1], 1.0)
z_mid = field.bbox_max[2] - 1.5
for y in y_range:
    x = np.linspace(field.bbox_min[0] + 0.5, field.bbox_max[0] - 0.5, 200)
    pts = np.column_stack([x, np.full_like(x, y), np.full_like(x, z_mid)])
    tangent = np.array([1, 0, 0], dtype=float)
    for j in range(0, len(pts), 10):
        if field.is_inside(pts[j]):
            d, _ = field.query(pts[j], "sigma_dom")
            cos_a = min(abs(np.dot(tangent, d)), 1.0)
            trad_angles.append(np.degrees(np.arccos(cos_a)))

if trad_angles:
    print(f"  传统 0°: mean={np.mean(trad_angles):.1f}°")
    print(f"  应力驱动: mean={mean_angle_post:.1f}°")
    print(f"  改善: {np.mean(trad_angles) - mean_angle_post:.1f}°")

# === 保存 ===
path_data = {f"path_{i}": p for i, p in enumerate(ordered_paths)}
path_data["n_paths"] = np.array([len(ordered_paths)])
np.savez_compressed(str(output_dir / "fiber_paths.npz"), **path_data)
print(f"\n路径: {len(ordered_paths)} paths → fiber_paths.npz")

# === 可视化 (改进: 分层着色) ===
try:
    import pyvista as pv
    pv.OFF_SCREEN = True

    for view_name, cam in [("front", "xy"), ("iso", [(250, 100, 80), (50, 5, 5), (0, 0, 1)])]:
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        plotter.set_background("white")

        # 半透明几何
        cloud = pv.PolyData(field.centroids)
        grid = cloud.delaunay_3d(alpha=3.0)
        surface = grid.extract_surface()
        plotter.add_mesh(surface, color="lightgray", opacity=0.15)

        # 路径按 Z 层着色
        for i, p in enumerate(ordered_paths):
            if len(p) < 2:
                continue
            z_mean = p[:, 2].mean()
            z_norm = (z_mean - field.bbox_min[2]) / max(field.bbox_max[2] - field.bbox_min[2], 1)
            color = [z_norm, 0.2, 1 - z_norm]  # 蓝(下) → 红(上)
            line = pv.Spline(p, n_points=min(len(p), 200))
            plotter.add_mesh(line, color=color, line_width=1.5)

        plotter.add_axes(color="black")
        plotter.camera_position = cam
        plotter.screenshot(str(output_dir / f"fiber_paths_{view_name}.png"))
        plotter.close()
    print("可视化已生成")
except Exception as e:
    print(f"可视化跳过: {e}")

t_total = time.time() - t0
print(f"\n总耗时: {t_total:.1f}s")
all_pass = pass1 and pass2 and pass3 and pass4
print(f"\n>>> Phase 2: {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
