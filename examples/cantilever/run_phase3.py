"""Phase 3 验收: 切片 + G-code 生成"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cfpp.pathgen.field import StressField
from cfpp.pathgen.constraints import compute_tension_profile
from cfpp.gcode.slicer import slice_paths_by_layer
from cfpp.gcode.transform import transform_layer_paths
from cfpp.gcode.generator import GCodeGenerator
from cfpp.gcode.validator import GCodeValidator

output_dir = Path(__file__).parent / "output"

print("=" * 60)
print("Phase 3 验收 — 切片 + G-code 生成")
print("=" * 60)

# === 1. 加载 Phase 2 路径 ===
t0 = time.time()
data = np.load(str(output_dir / "fiber_paths.npz"))
n_paths = int(data["n_paths"][0])
paths = [data[f"path_{i}"] for i in range(n_paths)]
print(f"Loaded {n_paths} paths")

# === 2. 切片 ===
layers = slice_paths_by_layer(paths, layer_height=0.15)

# === 3. 5轴坐标变换 ===
layers_5axis = {}
tension_profiles = {}
for li, layer_paths in layers.items():
    # 平面打印: normal = [0,0,1], A=0, C=0
    paths_5ax = transform_layer_paths(layer_paths)
    layers_5axis[li] = paths_5ax

    # 张力预设
    layer_tensions = []
    for p in layer_paths:
        t = compute_tension_profile(p)
        layer_tensions.append(t)
    tension_profiles[li] = layer_tensions

# === 4. G-code 生成 ===
gen = GCodeGenerator()
gen.nozzle_temp = 240
gen.bed_temp = 80
gen.feed_rate = 600
gcode_path = gen.generate(
    layers_5axis,
    tension_profiles=tension_profiles,
    output_path=str(output_dir / "cantilever.gcode"),
)

# === 5. G-code 验证 ===
validator = GCodeValidator(gcode_path)
result = validator.validate()

print("\n" + "=" * 60)
print("验收结果")
print("=" * 60)

# 检查1: G-code 合法性
pass1 = result["valid"]
print(f"  [{'PASS' if pass1 else 'FAIL'}] G-code 合法性: "
      f"{result['n_errors']} errors, {result['n_warnings']} warnings")
if result["errors"]:
    for e in result["errors"][:3]:
        print(f"    ERROR: {e}")

# 检查2: 坐标连续性 (无大跳变)
n_jumps = sum(1 for w in result.get("warnings", []) if "跳变" in str(w))
pass2 = n_jumps == 0
print(f"  [{'PASS' if pass2 else 'FAIL'}] 坐标连续性: {n_jumps} 跳变")

# 检查3: 打印距离合理
print_dist = result.get("print_distance", 0)
travel_dist = result.get("travel_distance", 0)
ratio = travel_dist / max(print_dist, 1)
pass3 = ratio < 1.0  # 空行程不超过打印距离
print(f"  [{'PASS' if pass3 else 'FAIL'}] 行程效率: "
      f"print={print_dist:.0f}mm, travel={travel_dist:.0f}mm, ratio={ratio:.2f}")

# 检查4: 剪切次数 = 路径数
n_cuts = result.get("n_cuts", 0)
pass4 = n_cuts == n_paths
print(f"  [{'PASS' if pass4 else 'FAIL'}] 剪切次数: {n_cuts} (expect {n_paths})")

# 检查5: 时间估算合理
est_time = result.get("est_time_min", 0)
pass5 = est_time > 0
print(f"  [{'PASS' if pass5 else 'FAIL'}] 打印时间: ≈{est_time:.1f} min")

# === 统计 ===
print(f"\n--- G-code 统计 ---")
print(f"  总行数: {result['n_lines']}")
print(f"  运动指令: {result['n_moves']} ({result['n_print_moves']} print + {result['n_travel_moves']} travel)")
print(f"  工作范围: {result['bbox_min']} → {result['bbox_max']}")
print(f"  纤维总长: {print_dist:.0f} mm")

# === 可视化: G-code 轨迹回放 ===
try:
    import pyvista as pv
    pv.OFF_SCREEN = True

    traj = validator.get_trajectory()
    if len(traj) > 0:
        for view_name, cam in [("front", "xy"), ("iso", [(250, 100, 80), (50, 5, 5), (0, 0, 1)])]:
            plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
            plotter.set_background("white")

            # 打印路径 (红) vs 空行程 (灰虚线)
            print_mask = traj[:, 3] > 0.5
            if print_mask.any():
                pts = traj[print_mask, :3]
                cloud = pv.PolyData(pts)
                plotter.add_mesh(cloud, color="red", point_size=2,
                               render_points_as_spheres=True)

            travel_mask = ~print_mask
            if travel_mask.any():
                pts = traj[travel_mask, :3]
                cloud = pv.PolyData(pts)
                plotter.add_mesh(cloud, color="gray", point_size=1, opacity=0.3)

            plotter.add_axes(color="black")
            plotter.camera_position = cam
            plotter.screenshot(str(output_dir / f"gcode_replay_{view_name}.png"))
            plotter.close()

        print("G-code 回放图已生成")
except Exception as e:
    print(f"可视化跳过: {e}")

t_total = time.time() - t0
all_pass = pass1 and pass2 and pass3 and pass4 and pass5
print(f"\n总耗时: {t_total:.1f}s")
print(f"\n>>> Phase 3: {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")
