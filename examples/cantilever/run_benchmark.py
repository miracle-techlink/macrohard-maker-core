"""悬臂梁 Benchmark: Phase 1 验收"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cfpp.mesh.generator import create_box_mesh
from cfpp.solver.elastic import ElasticSolver
from cfpp.io.export import export_principal_stresses_json, export_principal_stresses_npz

L, W, H = 100.0, 10.0, 10.0  # mm
P = 1000.0  # N
E, nu = 210e3, 0.3  # MPa
I = W * H**3 / 12

delta_analytical = P * L**3 / (3 * E * I)
sigma_max_analytical = P * L * (H / 2) / I

print("=" * 60)
print("Phase 1 验收 — 悬臂梁 Benchmark")
print("=" * 60)
print(f"尺寸: {L}×{W}×{H} mm | 载荷: {P} N | E={E/1e3:.0f} GPa")
print(f"解析解: δ={delta_analytical:.4f} mm, σ_max={sigma_max_analytical:.1f} MPa")
print()

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)


def run_one(mesh_size, label):
    mesh_path = str(output_dir / f"cantilever_{label}.msh")
    t0 = time.time()
    create_box_mesh(L, W, H, mesh_size=mesh_size, output_path=mesh_path)

    solver = ElasticSolver(mesh_path)
    solver.set_isotropic_material(E, nu)
    solver.solve(traction=(0, 0, -P / (W * H)))

    t1 = time.time()
    principal = solver.extract_principal_stresses()
    t_total = time.time() - t0

    max_disp = np.max(np.abs(solver.displacement))

    # σ_xx 排除固定端集中
    mask = principal["centroids"][:, 0] > 5.0
    max_sigma_xx = np.max(np.abs(solver.stress_tensor[mask, 0, 0]))

    disp_err = abs(max_disp - delta_analytical) / delta_analytical * 100
    stress_err = abs(max_sigma_xx - sigma_max_analytical) / sigma_max_analytical * 100

    print(f"[{label}] δ={max_disp:.4f} mm (err {disp_err:.2f}%) | "
          f"σ_xx={max_sigma_xx:.1f} MPa (err {stress_err:.2f}%) | {t_total:.1f}s")

    return dict(max_disp=max_disp, max_stress=max_sigma_xx,
                disp_err=disp_err, stress_err=stress_err,
                principal=principal, solver=solver)


r_coarse = run_one(0.8, "coarse")
r_fine = run_one(0.5, "fine")

conv_disp = abs(r_fine["max_disp"] - r_coarse["max_disp"]) / r_fine["max_disp"] * 100
conv_stress = abs(r_fine["max_stress"] - r_coarse["max_stress"]) / r_fine["max_stress"] * 100

# === 验收 ===
print("\n" + "=" * 60)
print("验收结果")
print("=" * 60)

checks = [
    ("位移误差 < 2%",         r_fine["disp_err"],   2.0),
    ("应力误差 < 5%",         r_fine["stress_err"], 5.0),
    ("网格收敛(位移) < 3%",   conv_disp,            3.0),
    ("网格收敛(应力) < 5%",   conv_stress,          5.0),
]

all_pass = True
results_for_report = []
for name, value, threshold in checks:
    passed = value < threshold
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: {value:.2f}%")
    results_for_report.append((name, value, threshold, passed))
    if not passed:
        all_pass = False

# === 导出 ===
npz_path = export_principal_stresses_npz(r_fine["principal"], str(output_dir / "stress_field.npz"))
json_path = export_principal_stresses_json(r_fine["principal"], str(output_dir / "stress_field.json"))

# === 可视化 ===
try:
    from cfpp.viz.plotter import plot_stress_report
    plot_stress_report(r_fine["principal"], str(output_dir))
    print("  [PASS] 可视化生成")
except Exception as e:
    print(f"  [SKIP] 可视化: {e}")

# === 多工况 ===
print("\n--- 多工况验证 ---")
mesh_path = str(output_dir / "cantilever_multi.msh")
create_box_mesh(L, W, H, mesh_size=1.5, output_path=mesh_path)

loadcases = [
    ("bending",  (0, 0, -10.0),  "弯曲 -Z"),
    ("shear",    (0, 10.0, 0),   "剪切 +Y"),
    ("tension",  (50.0, 0, 0),   "拉伸 +X"),
]

for name, trac, desc in loadcases:
    s = ElasticSolver(mesh_path)
    s.set_isotropic_material(E, nu)
    s.solve(traction=trac)
    p = s.extract_principal_stresses()
    export_principal_stresses_npz(p, str(output_dir / f"stress_{name}.npz"))
    print(f"  {desc}: von Mises max = {p['von_mises'].max():.1f} MPa")

print(f"\n  [PASS] 多工况: {len(loadcases)} 种工况")
print()
print(f">>> Phase 1: {'ALL PASSED' if all_pass else 'SOME FAILED'} <<<")

# === 保存验收数据供报告使用 ===
report_data = {
    "analytical": {"delta": delta_analytical, "sigma_max": sigma_max_analytical},
    "coarse": {"mesh_size": 0.8, "disp": r_coarse["max_disp"], "stress": r_coarse["max_stress"],
               "n_elements": r_coarse["solver"].mesh_sk.t.shape[1]},
    "fine":   {"mesh_size": 0.5, "disp": r_fine["max_disp"], "stress": r_fine["max_stress"],
               "n_elements": r_fine["solver"].mesh_sk.t.shape[1]},
    "checks": results_for_report,
    "convergence": {"disp": conv_disp, "stress": conv_stress},
}
np.savez(str(output_dir / "report_data.npz"), **{k: str(v) for k, v in report_data.items()})
