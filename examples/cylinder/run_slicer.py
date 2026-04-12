"""Cylinder benchmark with proper layer-first slicing pipeline.

This replaces the old approach (3D streamlines -> projection)
with the correct approach (slice first -> plan paths per layer).
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cfpp.mesh.generator import create_cylinder_mesh
from cfpp.solver.elastic import ElasticSolver
from cfpp.pathgen.field import StressField
from cfpp.slicer.pipeline import SlicingPipeline
from cfpp.gcode.generator import GCodeGenerator
from cfpp.gcode.validator import GCodeValidator

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("Cylinder Benchmark -- Layer-First Slicing Pipeline")
print("=" * 70)

t0 = time.time()

# === Parameters ===
R_outer, R_inner, H = 25.0, 20.0, 80.0
P = 500.0
E_fiber = 60e3
nu = 0.3

traction_x = P / (np.pi * (R_outer**2 - R_inner**2))
print(f"Geometry: R={R_outer}/{R_inner}mm, H={H}mm")
print(f"Load: P={P}N, traction_x={traction_x:.4f} MPa")

# === Step 1: Mesh + FEA ===
print("\n" + "=" * 70)
print("Step 1: Mesh + FEA")
print("=" * 70)

mesh_path = str(output_dir / "cylinder_slicer.msh")
create_cylinder_mesh(R_outer, R_inner, H, mesh_size=5.0, output_path=mesh_path)

solver = ElasticSolver(mesh_path)
solver.set_isotropic_material(E_fiber, nu)
solver.solve(traction=(traction_x, 0, 0))
ps = solver.extract_principal_stresses()

stress_npz = str(output_dir / "stress_field_slicer.npz")
np.savez_compressed(stress_npz,
    centroids=ps["centroids"],
    sigma_1_dir=ps["sigma_1_dir"],
    sigma_2_dir=ps["sigma_2_dir"],
    sigma_3_dir=ps["sigma_3_dir"],
    sigma_1_val=ps["sigma_1_val"],
    sigma_2_val=ps["sigma_2_val"],
    sigma_3_val=ps["sigma_3_val"],
    von_mises=ps["von_mises"],
)

field = StressField(stress_npz)

# === Step 2: Layer-First Slicing ===
print("\n" + "=" * 70)
print("Step 2: Layer-First Slicing Pipeline")
print("=" * 70)

pipeline = SlicingPipeline(
    mesh_path=mesh_path,
    stress_field=field,
    layer_height=2.0,         # coarse layers for benchmark speed
    first_layer_height=2.0,
    n_walls=2,
    wall_spacing=0.4,
    fiber_spacing=2.0,
    infill_density=0.0,  # fiber only, no plastic infill
    min_fiber_length=5.0,
    min_turn_radius=6.0,
    support_angle=45.0,
    fiber_component="sigma_dom",
)

result = pipeline.run()

# === Step 3: G-code ===
print("\n" + "=" * 70)
print("Step 3: G-code Generation")
print("=" * 70)

gcode_layers = pipeline.get_all_paths_for_gcode(result)

gen = GCodeGenerator()
gen.nozzle_temp = 240
gen.bed_temp = 80
gen.feed_rate = 600

gcode_path = gen.generate(
    gcode_layers,
    output_path=str(output_dir / "cylinder_slicer.gcode"),
)

# Validate
validator = GCodeValidator(gcode_path)
vresult = validator.validate()

print(f"\n  G-code: {vresult['n_lines']} lines, {vresult.get('n_cuts', 0)} cuts")
print(f"  Validity: {vresult['n_errors']} errors, {vresult['n_warnings']} warnings")
print(f"  Print distance: {vresult.get('print_distance', 0):.0f}mm")
print(f"  Travel distance: {vresult.get('travel_distance', 0):.0f}mm")
print(f"  Est. time: {vresult.get('est_time_min', 0):.1f} min")

# === Step 4: Export for visualization ===
print("\n" + "=" * 70)
print("Step 4: Export for Visualization")
print("=" * 70)

# Save all paths (walls + fibers) for viz
all_paths = []
for key in sorted(result['layers'].keys()):
    layer = result['layers'][key]
    all_paths.extend(layer['wall_paths'])
    all_paths.extend(layer['fiber_paths'])

path_data = {f"path_{i}": p for i, p in enumerate(all_paths)}
path_data["n_paths"] = np.array([len(all_paths)])
np.savez_compressed(str(output_dir / "fiber_paths_slicer.npz"), **path_data)

print(f"  Total paths exported: {len(all_paths)}")
print(f"  Wall paths: {result['stats']['total_walls']}")
print(f"  Fiber paths: {result['stats']['total_fibers']}")
print(f"  Fiber length: {result['stats']['total_fiber_length']:.0f}mm")

# === Summary ===
t_total = time.time() - t0
print(f"\n{'=' * 70}")
print(f"Total time: {t_total:.1f}s")

# Comparison with old approach
old_paths_file = output_dir / "fiber_paths.npz"
if old_paths_file.exists():
    old = np.load(str(old_paths_file))
    n_old = int(old["n_paths"][0])
    old_len = sum(
        np.sum(np.linalg.norm(np.diff(old[f"path_{i}"], axis=0), axis=1))
        for i in range(n_old)
    )
    print(f"\n--- Old vs New Comparison ---")
    print(f"  Old (3D streamlines): {n_old} paths, {old_len:.0f}mm fiber")
    print(f"  New (layer-first):    {len(all_paths)} paths, {result['stats']['total_fiber_length']:.0f}mm fiber")
    print(f"  New has {result['stats']['total_walls']} wall paths + {result['stats']['total_fibers']} fiber paths")

print(f"\n>>> Layer-First Pipeline: DONE <<<")
