"""多工况求解示例: 验收项3 — 至少3种载荷工况可配置

工况:
  1. 弯曲 (Z方向端部载荷)
  2. 扭转 (端面剪力模拟)
  3. 拉伸 (X方向轴向力)
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cfpp.mesh.generator import create_box_mesh
from cfpp.solver.elastic import ElasticSolver
from cfpp.io.export import export_principal_stresses_json, export_principal_stresses_npz

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# 生成网格
mesh_path = str(output_dir / "cantilever_multi.msh")
create_box_mesh(100, 10, 10, mesh_size=2.0, output_path=mesh_path)

# 三种工况
loadcases = [
    {"name": "bending",  "traction": (0, 0, -10.0), "desc": "弯曲: Z方向端部面力 -10 MPa"},
    {"name": "shear",    "traction": (0, 10.0, 0),   "desc": "剪切: Y方向端部面力 10 MPa"},
    {"name": "tension",  "traction": (50.0, 0, 0),   "desc": "拉伸: X方向轴向面力 50 MPa"},
]

results = {}
for lc in loadcases:
    print(f"\n{'='*50}")
    print(f"工况: {lc['desc']}")
    print(f"{'='*50}")

    solver = ElasticSolver(mesh_path)
    solver.set_isotropic_material(E=210e3, nu=0.3)
    solver.solve(traction=lc["traction"])
    principal = solver.extract_principal_stresses()

    out_path = str(output_dir / f"stress_field_{lc['name']}.json")
    export_principal_stresses_json(principal, out_path)
    results[lc["name"]] = principal

# 多工况包络: 取每个单元在所有工况下的最大 von Mises
n = len(results["bending"]["von_mises"])
envelope = np.zeros(n)
for name, data in results.items():
    envelope = np.maximum(envelope, data["von_mises"])

print(f"\n多工况包络: max von Mises = {envelope.max():.2f} MPa")
print(f"多工况求解完成, {len(loadcases)} 种工况已导出。")
