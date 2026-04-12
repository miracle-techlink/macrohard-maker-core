"""拓扑优化驱动的纤维路径规划 Demo

流程:
  1. SIMP 拓扑优化 -> 确定纤维放置区域
  2. FEA 应力分析 (仅 fiber region)
  3. 在 fiber region 内生成种子点 + 流线路径
  4. 对比: 有/无拓扑优化的路径规划效果
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cfpp.mesh.generator import create_cylinder_mesh
from cfpp.solver.elastic import ElasticSolver
from cfpp.topo.simp import SIMPOptimizer
from cfpp.io.export import export_principal_stresses_npz

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# === Geometry & load parameters ===
R_outer = 25.0   # mm
R_inner = 20.0   # mm
H = 80.0         # mm
P = 500.0        # N (lateral force)
E_fiber = 60e3   # MPa
E_base = 3.5e3   # MPa
nu = 0.3

A_top = np.pi * (R_outer ** 2 - R_inner ** 2)
traction_x = P / A_top  # MPa

print("=" * 70)
print("Topology-Optimized Fiber Path Planning (SIMP)")
print("=" * 70)
print(f"Geometry: R_outer={R_outer}, R_inner={R_inner}, H={H} mm")
print(f"Load: P={P} N lateral, traction_x={traction_x:.4f} MPa")
print()

# ============================================================
# Step 1: Generate mesh
# ============================================================
print("--- Step 1: Mesh generation ---")
mesh_path = str(output_dir / "cylinder_topo.msh")
create_cylinder_mesh(R_outer, R_inner, H, mesh_size=3.0, output_path=mesh_path)

# ============================================================
# Step 2: SIMP topology optimization
# ============================================================
print("\n--- Step 2: SIMP topology optimization ---")
t0 = time.time()

optimizer = SIMPOptimizer(
    mesh_path=mesh_path,
    volume_fraction=0.4,
    penalty=3.0,
)

result = optimizer.optimize(
    E_fiber=E_fiber,
    E_base=E_base,
    nu=nu,
    traction=(traction_x, 0, 0),
    fixed_boundary="fixed",
    load_boundary="load",
    max_iter=40,
    tol=0.01,
    filter_radius=4.0,
)

t_topo = time.time() - t0

densities = result["densities"]
fiber_mask = result["fiber_region"]
n_fiber = np.sum(fiber_mask)
n_total = len(densities)

print(f"\nTopology optimization: {t_topo:.1f}s")
print(f"  Fiber region: {n_fiber}/{n_total} elements ({n_fiber/n_total:.1%})")
print(f"  Final compliance: {result['compliance_history'][-1]:.4f}")
print(f"  Iterations: {result['n_iterations']}")

# Export density field
density_path = str(output_dir / "topo_density.npz")
optimizer.export_density_field(density_path)

# ============================================================
# Step 3: Stress analysis on full model
# ============================================================
print("\n--- Step 3: Stress analysis ---")

solver = ElasticSolver(mesh_path)
solver.set_isotropic_material(E_fiber, nu)
solver.solve(traction=(traction_x, 0, 0))
stress_data = solver.extract_principal_stresses()

# Export stress field
npz_path = export_principal_stresses_npz(stress_data, str(output_dir / "stress_field_topo.npz"))

# ============================================================
# Step 4: Seed points only in fiber region
# ============================================================
print("\n--- Step 4: Topo-guided seed generation ---")

from cfpp.pathgen.field import StressField
from cfpp.pathgen.streamline import generate_seed_points, generate_streamlines
from cfpp.pathgen.constraints import filter_min_radius, check_path_constraints
from cfpp.pathgen.ordering import order_paths_greedy, compute_coverage

field = StressField(str(output_dir / "stress_field_topo.npz"))

# Fiber region centroids
fiber_centroids = result["centroids"][fiber_mask]  # (n_fiber, 3)

# Use the fiber region centroids to create a spatial mask for seeds
from scipy.spatial import cKDTree
fiber_tree = cKDTree(fiber_centroids)

# Generate seeds broadly
z_lo = field.bbox_min[2] + 2.0
z_hi = field.bbox_max[2] - 2.0
z_slices = np.linspace(z_lo, z_hi, 6).tolist()

all_seeds = generate_seed_points(
    field, spacing=2.0,
    z_slices=z_slices,
    min_stress_pct=0.10,
)

# Filter seeds: keep only those near fiber region
seed_dists, _ = fiber_tree.query(all_seeds, k=1)
max_seed_dist = 5.0  # mm, generous to catch boundary seeds
topo_seeds = all_seeds[seed_dists < max_seed_dist]

print(f"  Total seeds: {len(all_seeds)} -> topo-filtered: {len(topo_seeds)}")

# ============================================================
# Step 5: Generate streamlines in fiber region
# ============================================================
print("\n--- Step 5: Topo-guided path generation ---")

topo_paths = generate_streamlines(
    field, topo_seeds,
    component="sigma_dom",
    step_size=0.5,
    min_length=8.0,
    max_steps=2000,
    boundary_margin=2.5,
    min_spacing=1.0,
    trim_tail_kappa=0.15,
)

# Smooth
filtered_paths = []
for p in topo_paths:
    p_smooth = filter_min_radius(p, min_radius=6.0)
    filtered_paths.append(p_smooth)

ordered_paths = order_paths_greedy(filtered_paths)

print(f"  Generated: {len(ordered_paths)} paths (topo-guided)")

# ============================================================
# Step 6: Compare with non-topo baseline
# ============================================================
print("\n--- Step 6: Baseline (no topo) path generation ---")

baseline_seeds = generate_seed_points(
    field, spacing=2.0,
    z_slices=z_slices,
    min_stress_pct=0.10,
)

baseline_raw = generate_streamlines(
    field, baseline_seeds,
    component="sigma_dom",
    step_size=0.5,
    min_length=8.0,
    max_steps=2000,
    boundary_margin=2.5,
    min_spacing=1.0,
    trim_tail_kappa=0.15,
)

baseline_paths = []
for p in baseline_raw:
    baseline_paths.append(filter_min_radius(p, min_radius=6.0))
baseline_paths = order_paths_greedy(baseline_paths)

print(f"  Generated: {len(baseline_paths)} paths (baseline)")

# ============================================================
# Comparison
# ============================================================
print("\n" + "=" * 70)
print("Comparison: Topo-guided vs Baseline")
print("=" * 70)

# Fiber length
def total_length(paths):
    total = 0.0
    for p in paths:
        if len(p) < 2:
            continue
        total += np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
    return total

topo_length = total_length(ordered_paths)
base_length = total_length(baseline_paths)

# Coverage of high-stress region
topo_cov = compute_coverage(
    ordered_paths, field.centroids,
    field_von_mises=field.von_mises,
    path_width=1.5, min_stress_pct=0.10,
)
base_cov = compute_coverage(
    baseline_paths, field.centroids,
    field_von_mises=field.von_mises,
    path_width=1.5, min_stress_pct=0.10,
)

# Coverage of TOPO fiber region specifically
def topo_region_coverage(paths, fiber_cents, path_width=1.5):
    """Fraction of fiber-region elements covered by paths."""
    if len(fiber_cents) == 0 or len(paths) == 0:
        return 0.0
    all_pts = np.vstack([p for p in paths if len(p) >= 2])
    path_tree = cKDTree(all_pts)
    dists, _ = path_tree.query(fiber_cents, k=1)
    covered = np.sum(dists < path_width)
    return covered / len(fiber_cents)

topo_region_cov_topo = topo_region_coverage(ordered_paths, fiber_centroids)
topo_region_cov_base = topo_region_coverage(baseline_paths, fiber_centroids)

print(f"\n{'Metric':<35} {'Topo-guided':>15} {'Baseline':>15}")
print("-" * 70)
print(f"  {'Number of paths':<33} {len(ordered_paths):>15d} {len(baseline_paths):>15d}")
print(f"  {'Total fiber length (mm)':<33} {topo_length:>15.1f} {base_length:>15.1f}")
print(f"  {'High-stress coverage':<33} {topo_cov:>15.1%} {base_cov:>15.1%}")
print(f"  {'Fiber-region coverage':<33} {topo_region_cov_topo:>15.1%} {topo_region_cov_base:>15.1%}")
print(f"  {'Fiber efficiency (cov/length)':<33} "
      f"{topo_cov/max(topo_length,1)*1000:>15.3f} "
      f"{base_cov/max(base_length,1)*1000:>15.3f}")

# Key insight
print(f"\nKey insight:")
print(f"  Topo-guided places fibers WHERE material is most needed,")
print(f"  achieving {topo_region_cov_topo:.0%} coverage of the optimal region")
print(f"  vs {topo_region_cov_base:.0%} for the baseline approach.")

# ============================================================
# Visualization (optional)
# ============================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Convergence history
    ax = axes[0]
    ax.semilogy(result["compliance_history"], "b-o", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Compliance")
    ax.set_title("SIMP Convergence")
    ax.grid(True, alpha=0.3)

    # 2. Density distribution (histogram)
    ax = axes[1]
    ax.hist(densities, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(x=0.5, color="red", linestyle="--", label="threshold=0.5")
    ax.set_xlabel("Element Density")
    ax.set_ylabel("Count")
    ax.set_title(f"Density Distribution (vf={optimizer.volume_fraction})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Comparison bar chart
    ax = axes[2]
    labels = ["Paths", "Length\n(mm/10)", "High-stress\nCov (%)", "Fiber-region\nCov (%)"]
    topo_vals = [len(ordered_paths), topo_length/10, topo_cov*100, topo_region_cov_topo*100]
    base_vals = [len(baseline_paths), base_length/10, base_cov*100, topo_region_cov_base*100]

    x_pos = np.arange(len(labels))
    w = 0.35
    ax.bar(x_pos - w/2, topo_vals, w, label="Topo-guided", color="#2E7D32")
    ax.bar(x_pos + w/2, base_vals, w, label="Baseline", color="#FF9800")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title("Topo-guided vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(str(output_dir / "topo_comparison.png"), dpi=150)
    plt.close()
    print(f"\nVisualization saved: {output_dir / 'topo_comparison.png'}")
except Exception as e:
    print(f"\nVisualization skipped: {e}")

# Density field visualization
try:
    import pyvista as pv
    pv.OFF_SCREEN = True

    for view_name, cam in [("front", "xy"), ("iso", [(200, 150, 120), (0, 0, 40), (0, 0, 1)])]:
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        plotter.set_background("white")

        # Show density as point cloud
        cloud = pv.PolyData(result["centroids"])
        cloud["density"] = densities
        plotter.add_mesh(cloud, scalars="density", cmap="RdYlBu_r",
                        point_size=5, render_points_as_spheres=True,
                        clim=[0, 1], scalar_bar_args={"title": "Density"})

        plotter.add_axes(color="black")
        plotter.camera_position = cam
        plotter.screenshot(str(output_dir / f"topo_density_{view_name}.png"))
        plotter.close()

    print(f"Density field visualization saved")
except Exception as e:
    print(f"3D visualization skipped: {e}")

print(f"\nTotal time: {time.time() - t0:.1f}s")
print("Done.")
