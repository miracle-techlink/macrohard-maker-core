"""Prepare visualization data from NPZ files."""
import numpy as np
import json
import re
from pathlib import Path


def export_viz_data(example_dir: Path, output_dir: Path, name: str):
    """Export one example's data to JSON for web visualization."""

    # 1. Stress field
    sf = np.load(str(example_dir / "output" / "stress_field.npz"))
    centroids = sf["centroids"]
    von_mises = sf["von_mises"]
    sigma_1_dir = sf["sigma_1_dir"]

    # Downsample if too many elements (>10k)
    if len(centroids) > 10000:
        idx = np.random.choice(len(centroids), 10000, replace=False)
        centroids = centroids[idx]
        von_mises = von_mises[idx]
        sigma_1_dir = sigma_1_dir[idx]

    stress_data = {
        "centroids": centroids.tolist(),
        "von_mises": von_mises.tolist(),
        "vm_max": float(von_mises.max()),
        "vm_min": float(von_mises.min()),
        "sigma_1_dir": sigma_1_dir.tolist(),
    }

    with open(str(output_dir / f"{name}_stress.json"), "w") as f:
        json.dump(stress_data, f)

    # 2. Fiber paths
    fp = np.load(str(example_dir / "output" / "fiber_paths.npz"))
    n_paths = int(fp["n_paths"][0])
    paths = []
    total_length = 0.0
    for i in range(min(n_paths, 200)):  # cap at 200 paths
        p = fp[f"path_{i}"]
        # Compute length before downsampling
        if len(p) > 1:
            diffs = np.diff(p, axis=0)
            total_length += float(np.sum(np.linalg.norm(diffs, axis=1)))
        # Downsample long paths
        if len(p) > 100:
            idx_ds = np.linspace(0, len(p) - 1, 100, dtype=int)
            p = p[idx_ds]
        paths.append(p.tolist())

    path_data = {
        "paths": paths,
        "n_paths": len(paths),
        "n_paths_total": n_paths,
        "total_length_mm": round(total_length, 1),
    }
    with open(str(output_dir / f"{name}_paths.json"), "w") as f:
        json.dump(path_data, f)

    # 3. Topology density (if exists)
    topo_path = example_dir / "output" / "topo_density.npz"
    has_topo = False
    if topo_path.exists():
        td = np.load(str(topo_path))
        densities = td["densities"]
        topo_centroids = td["centroids"]

        if len(topo_centroids) > 10000:
            idx = np.random.choice(len(topo_centroids), 10000, replace=False)
            topo_centroids = topo_centroids[idx]
            densities = densities[idx]

        topo_data = {
            "centroids": topo_centroids.tolist(),
            "densities": densities.tolist(),
        }
        with open(str(output_dir / f"{name}_topo.json"), "w") as f:
            json.dump(topo_data, f)
        has_topo = True

    # 4. G-code trajectory
    gcode_files = list((example_dir / "output").glob("*.gcode"))
    n_gcode = 0
    if gcode_files:
        gcode_path = gcode_files[0]
        positions = []
        is_print = []
        pos = [0.0, 0.0, 0.0]
        for line in open(str(gcode_path)):
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            cmd_match = re.match(r"(G[01])\s", line)
            if cmd_match:
                cmd = cmd_match.group(1)
                for axis, ai in [("X", 0), ("Y", 1), ("Z", 2)]:
                    m = re.search(rf"{axis}([-\d.]+)", line)
                    if m:
                        pos[ai] = float(m.group(1))
                positions.append(list(pos))
                is_print.append(cmd == "G1")

        # Downsample
        if len(positions) > 20000:
            step = len(positions) // 20000
            positions = positions[::step]
            is_print = is_print[::step]

        n_gcode = len(positions)
        gcode_data = {"positions": positions, "is_print": is_print}
        with open(str(output_dir / f"{name}_gcode.json"), "w") as f:
            json.dump(gcode_data, f)

    print(
        f"  {name}: stress({len(stress_data['centroids'])} pts), "
        f"paths({path_data['n_paths']}/{n_paths} total, {total_length:.0f}mm), "
        f"topo({'yes' if has_topo else 'no'}), "
        f"gcode({n_gcode} pts)"
    )


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    out = Path(__file__).resolve().parent / "data"
    out.mkdir(exist_ok=True)

    # Cantilever
    cantilever_dir = base / "examples" / "cantilever"
    if (cantilever_dir / "output" / "stress_field.npz").exists():
        export_viz_data(cantilever_dir, out, "cantilever")

    # Cylinder
    cylinder_dir = base / "examples" / "cylinder"
    if (cylinder_dir / "output" / "stress_field.npz").exists():
        export_viz_data(cylinder_dir, out, "cylinder")

    # Cylinder (new slicer pipeline)
    slicer_paths = cylinder_dir / "output" / "fiber_paths_slicer.npz"
    slicer_stress = cylinder_dir / "output" / "stress_field_slicer.npz"
    if slicer_paths.exists() and slicer_stress.exists():
        # Export stress data for cylinder_v2
        sf = np.load(str(slicer_stress))
        centroids = sf["centroids"]
        von_mises = sf["von_mises"]
        sigma_1_dir = sf["sigma_1_dir"]

        if len(centroids) > 10000:
            idx = np.random.choice(len(centroids), 10000, replace=False)
            centroids = centroids[idx]
            von_mises = von_mises[idx]
            sigma_1_dir = sigma_1_dir[idx]

        stress_data = {
            "centroids": centroids.tolist(),
            "von_mises": von_mises.tolist(),
            "vm_max": float(von_mises.max()),
            "vm_min": float(von_mises.min()),
            "sigma_1_dir": sigma_1_dir.tolist(),
        }
        with open(str(out / "cylinder_v2_stress.json"), "w") as f:
            json.dump(stress_data, f)

        # Export paths for cylinder_v2
        fp = np.load(str(slicer_paths))
        n_paths = int(fp["n_paths"][0])
        paths = []
        total_length = 0.0
        for i in range(min(n_paths, 200)):
            p = fp[f"path_{i}"]
            if len(p) > 1:
                diffs = np.diff(p, axis=0)
                total_length += float(np.sum(np.linalg.norm(diffs, axis=1)))
            if len(p) > 100:
                idx_ds = np.linspace(0, len(p) - 1, 100, dtype=int)
                p = p[idx_ds]
            paths.append(p.tolist())

        path_data = {
            "paths": paths,
            "n_paths": len(paths),
            "n_paths_total": n_paths,
            "total_length_mm": round(total_length, 1),
        }
        with open(str(out / "cylinder_v2_paths.json"), "w") as f:
            json.dump(path_data, f)

        # Export gcode trajectory for cylinder_v2
        slicer_gcode = cylinder_dir / "output" / "cylinder_slicer.gcode"
        n_gcode = 0
        if slicer_gcode.exists():
            positions = []
            is_print = []
            pos = [0.0, 0.0, 0.0]
            for line in open(str(slicer_gcode)):
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                cmd_match = re.match(r"(G[01])\s", line)
                if cmd_match:
                    cmd = cmd_match.group(1)
                    for axis, ai in [("X", 0), ("Y", 1), ("Z", 2)]:
                        m = re.search(rf"{axis}([-\d.]+)", line)
                        if m:
                            pos[ai] = float(m.group(1))
                    positions.append(list(pos))
                    is_print.append(cmd == "G1")

            if len(positions) > 20000:
                step = len(positions) // 20000
                positions = positions[::step]
                is_print = is_print[::step]

            n_gcode = len(positions)
            gcode_data = {"positions": positions, "is_print": is_print}
            with open(str(out / "cylinder_v2_gcode.json"), "w") as f:
                json.dump(gcode_data, f)

        print(
            f"  cylinder_v2: stress({len(stress_data['centroids'])} pts), "
            f"paths({path_data['n_paths']}/{n_paths} total, {total_length:.0f}mm), "
            f"gcode({n_gcode} pts)"
        )

    print("Done! Open visualization/index.html to view.")
