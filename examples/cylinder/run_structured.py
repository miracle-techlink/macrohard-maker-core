"""Cylinder Benchmark — Structured Helical Winding (Regular Paths)

Uses StructuredWindingPlanner to generate organized, parallel fiber paths
like real filament winding. Produces clean diamond/lattice patterns.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cfpp.mesh.generator import create_cylinder_mesh
from cfpp.solver.elastic import ElasticSolver
from cfpp.surface.extract import extract_surface
from cfpp.surface.stress_field import SurfaceStressField
from cfpp.surface.structured import StructuredWindingPlanner
from cfpp.surface.machine_coords import (
    path_to_machine_coords, generate_gcode_xzc, order_paths_minimize_c_jumps
)

output_dir = Path(__file__).parent / "output_structured"
output_dir.mkdir(exist_ok=True)

R_outer, R_inner, H = 25.0, 20.0, 80.0
P = 500.0
E_fiber = 60e3
nu = 0.3
A_top = np.pi * (R_outer**2 - R_inner**2)
traction_x = P / A_top

print("=" * 70)
print("Cylinder — Structured Helical Winding")
print("=" * 70)
print(f"Geometry: R={R_outer}/{R_inner}mm, H={H}mm")
t0 = time.time()

# === Phase 1: FEA + Surface ===
print("\n" + "=" * 70)
print("Phase 1: FEA + Surface")
print("=" * 70)

mesh_path = str(output_dir / "cylinder.msh")
create_cylinder_mesh(R_outer, R_inner, H, mesh_size=2.0, output_path=mesh_path)

solver = ElasticSolver(mesh_path)
solver.set_isotropic_material(E_fiber, nu)
solver.solve(traction=(traction_x, 0, 0))
ps = solver.extract_principal_stresses()

stress_npz = str(output_dir / "stress_volume.npz")
np.savez_compressed(stress_npz, **{k: ps[k] for k in ps})

surface = extract_surface(mesh_path)
surf_stress = SurfaceStressField(surface, stress_npz)

np.savez_compressed(str(output_dir / "surface_mesh.npz"),
    vertices=surface.vertices, triangles=surface.triangles,
    normals=surface.normals, vertex_normals=surface.vertex_normals,
    centroids=surface.centroids)
np.savez_compressed(str(output_dir / "surface_stress.npz"),
    stress_dir=surf_stress.stress_dir, stress_val=surf_stress.stress_val,
    dom_dir=surf_stress.dom_dir, dom_val=surf_stress.dom_val,
    von_mises=surf_stress.von_mises)

t1 = time.time()
print(f"Phase 1: {t1-t0:.1f}s")

# === Phase 2: Structured Winding ===
print("\n" + "=" * 70)
print("Phase 2: Structured Helical Winding")
print("=" * 70)

planner = StructuredWindingPlanner(surface, surf_stress)

# Config A: Standard ±45 winding (like traditional filament winding)
print("\n--- Config A: Standard ±45 ---")
paths_45, info_45 = planner.generate_full_winding(
    layup=[(+45, +1), (-45, -1)],
    path_spacing_mm=2.0, step_size=0.4, max_steps=3000)

# Config B: Stress-driven angles
print("\n--- Config B: Stress-driven angles ---")
paths_stress, info_stress = planner.generate_stress_driven_winding(
    n_angle_bins=4, path_spacing_mm=2.0, step_size=0.4, max_steps=3000)

# Config C: Multi-angle layup (±30/±60/85) for comprehensive coverage
print("\n--- Config C: Multi-angle layup ---")
paths_multi, info_multi = planner.generate_full_winding(
    layup=[(+30, +1), (-30, -1), (+60, +1), (-60, -1), (+85, +1)],
    path_spacing_mm=2.0, step_size=0.4, max_steps=3000)

# Compute coverage for each config
def compute_coverage(paths_list, surface, surf_stress):
    if not paths_list:
        return 0.0, 0
    all_pts = np.vstack([p['points'] for p in paths_list])
    tree = cKDTree(all_pts)

    # Outer wall: normal outward + radius above median
    c = surface.centroids
    n = surface.normals
    radial = c.copy()
    radial[:, 2] = 0
    r_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    r_norm = np.maximum(r_norm, 1e-10)
    radial /= r_norm
    dot = np.sum(n * radial, axis=1)
    normal_out = dot > 0.3
    surf_r = np.sqrt(c[:, 0]**2 + c[:, 1]**2)
    r_med = np.median(surf_r[normal_out]) if normal_out.any() else 0
    outer = normal_out & (surf_r > r_med)
    high_stress = surf_stress.von_mises > surf_stress.von_mises.max() * 0.15
    target = np.where(outer & high_stress)[0]

    if len(target) == 0:
        return 0.0, 0
    dists, _ = tree.query(c[target], k=1)
    covered = np.sum(dists < 2.0)
    return covered / len(target), len(paths_list)

from scipy.spatial import cKDTree

configs = [
    ("±45 standard", paths_45, info_45),
    ("stress-driven", paths_stress, info_stress),
    ("multi-angle", paths_multi, info_multi),
]

print(f"\n{'Config':<20} {'Paths':>6} {'Coverage':>10} {'Fiber(mm)':>10}")
print("-" * 50)
results = {}
for name, paths, info in configs:
    cov, n_p = compute_coverage(paths, surface, surf_stress)
    total_len = sum(i['total_length'] for i in info)
    print(f"  {name:<18} {n_p:>6} {cov:>9.1%} {total_len:>10.0f}")
    results[name] = {'paths': paths, 'info': info, 'coverage': cov,
                     'n_paths': n_p, 'total_length': total_len}

# Save best config paths
best_name = max(results, key=lambda k: results[k]['coverage'])
best = results[best_name]
print(f"\nBest config: {best_name} ({best['coverage']:.1%})")

path_data = {}
for i, p in enumerate(best['paths']):
    path_data[f"path_{i}"] = p['points']
    path_data[f"normals_{i}"] = p['normals']
    path_data[f"winding_{i}"] = p['winding']
path_data["n_paths"] = np.array([len(best['paths'])])
np.savez_compressed(str(output_dir / "structured_paths.npz"), **path_data)

# === Visualization ===
print("\nGenerating visualizations...")

# 1. 3D + Unwrapped view
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
    subplot_kw={'projection': '3d'} if True else {})
fig.delaxes(ax1)
fig.delaxes(ax2)

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# Color by layer
layer_colors = plt.cm.Set1(np.linspace(0, 1, 10))
path_idx = 0
for li, info_item in enumerate(best['info']):
    n_in_layer = info_item['n_paths_actual']
    color = layer_colors[li % len(layer_colors)]
    for j in range(n_in_layer):
        if path_idx >= len(best['paths']):
            break
        p = best['paths'][path_idx]['points']
        ax1.plot(p[:, 0], p[:, 1], p[:, 2], color=color, linewidth=0.5, alpha=0.7)
        # Unwrapped
        theta = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        ax2.plot(theta, p[:, 2], color=color, linewidth=0.5, alpha=0.7)
        path_idx += 1

ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.set_title(f'Structured Winding 3D ({best_name})')
ax2.set_xlabel('Theta (deg)'); ax2.set_ylabel('Z (mm)')
ax2.set_title(f'Structured Winding Unwrapped ({best_name})')
ax2.set_xlim(-180, 180)
ax2.grid(True, alpha=0.3)

# Add legend
for li, info_item in enumerate(best['info']):
    color = layer_colors[li % len(layer_colors)]
    label = f"α={info_item['alpha_deg']:+.0f}° {'↑' if info_item['direction']>0 else '↓'}"
    ax2.plot([], [], color=color, linewidth=2, label=label)
ax2.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(str(output_dir / "phase2_structured_paths.png"), dpi=150)
plt.close()

# 2. Coverage comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
names = [n for n, _, _ in configs]
coverages = [results[n]['coverage'] * 100 for n in names]
n_paths_list = [results[n]['n_paths'] for n in names]
colors = ['#2196F3', '#4CAF50', '#FF9800']
bars = ax.bar(names, coverages, color=colors)
for bar, cov, np_ in zip(bars, coverages, n_paths_list):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{cov:.1f}%\n({np_} paths)', ha='center', fontsize=9)
ax.set_ylabel('Coverage (%)')
ax.set_title('Structured Winding Coverage Comparison')
ax.axhline(y=99.9, color='red', linestyle='--', alpha=0.5, label='Target 99.9%')
ax.set_ylim(0, 105)
ax.legend()
plt.tight_layout()
plt.savefig(str(output_dir / "coverage_comparison.png"), dpi=150)
plt.close()

print("  Visualizations saved")

# === Phase 3: G-code ===
print("\n" + "=" * 70)
print("Phase 3: Klipper G-code")
print("=" * 70)

machine_paths = []
tensions_list = []
for p in best['paths']:
    pts, nrm = p['points'], p['normals']
    if len(pts) < 2:
        continue
    mc = path_to_machine_coords(pts, nrm, clamp_radius=R_outer)
    machine_paths.append(mc)
    tensions_list.append(np.full(len(pts), 2.0))

machine_paths, tensions_list = order_paths_minimize_c_jumps(machine_paths, tensions_list)

lines, stats = generate_gcode_xzc(
    machine_paths, tensions=tensions_list,
    feed_rate=600, travel_rate=3000,
    klipper_mode=True)

gcode_path = str(output_dir / "cylinder_structured_klipper.gcode")
with open(gcode_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"  G-code: {len(lines)} lines, {stats['n_cuts']} cuts, "
      f"{stats['total_fiber_length']:.0f}mm fiber")

# === Summary ===
t_total = time.time() - t0
print(f"\n{'=' * 70}")
print(f"Summary")
print(f"{'=' * 70}")
for name, _, _ in configs:
    r = results[name]
    print(f"  {name:<18} {r['n_paths']:>4} paths  {r['coverage']:>6.1%}  {r['total_length']:>8.0f}mm")
print(f"\nBest: {best_name} ({best['coverage']:.1%})")
print(f"Total time: {t_total:.1f}s")
print(f"\n>>> Structured Winding: {'DONE' if best['coverage'] > 0.99 else 'NEEDS MORE COVERAGE'} <<<")
