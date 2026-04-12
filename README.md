# CF-Path-Planner

**Stress-driven continuous carbon fiber path planning for multi-axis 3D printing.**

A full-stack tool that combines FEA-based stress analysis, geodesic surface path planning, and topology optimization to generate optimized fiber deposition paths for continuous carbon fiber (CCF) reinforced polymer 3D printing.

---

## Features

- **FEA Solver** — Linear elastic solver (isotropic & orthotropic) via `scikit-fem`, supporting per-element fiber orientation
- **Geodesic Path Planning** — Stress-field-guided geodesic paths on curved surfaces (cylinder, cone, arbitrary mesh)
- **XY+A Path Planner** — 4-axis (XY+A rotary) waypoint generation for helical winding, hoop/axial patterns, and custom fiber angles
- **Topology Optimization** — SIMP-based TO for optimal material distribution
- **G-code Generator** — Outputs machine-ready G-code for Klipper-based multi-axis printers
- **Interactive Web UI** — Three.js visualization with real-time path display, layer control, and mesh inspection

## Architecture

```
cf-path-planner/
├── cfpp/                  # Core Python library
│   ├── solver/            # FEA: isotropic & orthotropic elastic solvers
│   ├── surface/           # Geodesic path planning, stress field extraction, surface planner v2
│   ├── mesh/              # Mesh generation (gmsh-based)
│   ├── topo/              # SIMP topology optimization
│   ├── optimizer/         # Fiber layup optimizer
│   └── gcode/             # G-code post-processing
├── webapp/
│   └── server.py          # HTTP server: static files + REST API
├── visualization/
│   ├── index.html         # Three.js frontend (2400+ lines, no build step)
│   ├── lib/               # three.min.js, OrbitControls.js
│   └── data/              # Live JSON data from pipeline
├── examples/
│   ├── cylinder/          # Helical winding examples
│   ├── cone/              # Cone surface examples
│   ├── cantilever/        # Cantilever beam topology optimization
│   └── leg_link/          # Robotic leg link fiber layout
├── tests/                 # Unit + integration tests
├── klipper/               # Klipper printer config files
└── docs/                  # LaTeX reports: geodesic theory, FEA, experiments
```

## Quick Start

### Requirements

```bash
pip install numpy scipy meshio scikit-fem gmsh
```

### Run the Web App

```bash
cd webapp
python server.py
# Open http://localhost:8080
```

The web UI exposes:
- `POST /api/mesh` — Generate mesh from model parameters
- `POST /api/fea` — Run FEA and extract stress field
- `POST /api/xyza_paths` — Compute fiber winding paths
- `POST /api/gcode` — Generate G-code
- `POST /api/upload_stl` — Upload custom STL

### Run Path Planning Directly

```python
from cfpp.surface.planner_v2 import XYAPathPlanner

planner = XYAPathPlanner(a_offset_z=50.0)
waypoints = planner.helical_path(radius=15, length=80, winding_angle=45, n_layers=4)
# Returns list of (x, y_model, z_model, a_deg) tuples
```

### Run Examples

```bash
cd examples/cylinder
python run_surface.py      # Geodesic paths on cylinder surface
python run_all_phases.py   # Full pipeline: mesh → FEA → paths → G-code
```

## Pipeline Overview

```
STL / Parametric Model
        ↓
   Mesh Generation (gmsh)
        ↓
   FEA Stress Analysis (scikit-fem)
        ↓
   Stress Field → Principal Directions
        ↓
   Geodesic Path Planning on Surface
        ↓
   XY+A Waypoint Generation
        ↓
   G-code Output (Klipper)
```

## Hardware Target

Designed for **XY+A** multi-axis FFF printers:
- X axis: axial translation along fiber direction
- Y axis: nozzle transverse
- A axis: rotary platform (part rotation)

Klipper configuration examples in `klipper/`.

## Documentation

Technical reports (PDF + LaTeX source) in `docs/`:
- `geodesic_theory` — Geodesic path theory on curved surfaces
- `surface_geodesic_report` — Surface geodesic experiments
- `full_report` — Complete algorithm report

## Tests

```bash
cd tests
python test_xyza_backend.py
python benchmark_webapp.py
```

## License

MIT
