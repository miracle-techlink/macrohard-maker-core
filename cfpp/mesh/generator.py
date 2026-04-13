"""网格生成模块: STL → 自适应四面体网格"""

import gmsh
import math
import numpy as np
from pathlib import Path


def _build_builtin_occ(model_key: str) -> None:
    """Build OCC geometry for a built-in model. Must be called inside an active gmsh session."""
    occ = gmsh.model.occ

    if model_key == "hollow_tube":
        outer = occ.addCylinder(0, 0, 0, 0, 0, 80, 25)
        inner = occ.addCylinder(0, 0, 0, 0, 0, 80, 20)
        occ.cut([(3, outer)], [(3, inner)])

    elif model_key == "t_joint":
        mo = occ.addCylinder(-40, 0, 0, 80, 0, 0, 14)
        mi = occ.addCylinder(-40, 0, 0, 80, 0, 0, 10)
        bo = occ.addCylinder(0, 0, 0, 0, 40, 0, 14)
        bi = occ.addCylinder(0, 0, 0, 0, 40, 0, 10)
        out_f, _ = occ.fuse([(3, mo)], [(3, bo)])
        in_f, _ = occ.fuse([(3, mi)], [(3, bi)])
        occ.cut(out_f, in_f)

    elif model_key == "cross_joint":
        xo = occ.addCylinder(-50, 0, 0, 100, 0, 0, 13)
        xi = occ.addCylinder(-50, 0, 0, 100, 0, 0, 9)
        yo = occ.addCylinder(0, -50, 0, 0, 100, 0, 13)
        yi = occ.addCylinder(0, -50, 0, 0, 100, 0, 9)
        out, _ = occ.fuse([(3, xo)], [(3, yo)])
        inn, _ = occ.fuse([(3, xi)], [(3, yi)])
        occ.cut(out, inn)

    elif model_key == "elbow_pipe":
        full = occ.addTorus(0, 0, 0, 40, 12)
        inner = occ.addTorus(0, 0, 0, 40, 8)
        body, _ = occ.cut([(3, full)], [(3, inner)])
        box = occ.addBox(-80, -80, -20, 160, 80, 40)
        occ.intersect(body, [(3, box)])

    elif model_key == "y_junction":
        ang = math.radians(45)
        ao = occ.addCylinder(-40, 0, 0, 80, 0, 0, 12)
        ai = occ.addCylinder(-40, 0, 0, 80, 0, 0, 8)
        bo = occ.addCylinder(0, 0, 0, 40 * math.cos(ang), 40 * math.sin(ang), 0, 12)
        bi = occ.addCylinder(0, 0, 0, 40 * math.cos(ang), 40 * math.sin(ang), 0, 8)
        out, _ = occ.fuse([(3, ao)], [(3, bo)])
        inn, _ = occ.fuse([(3, ai)], [(3, bi)])
        occ.cut(out, inn)

    elif model_key == "flange":
        pipe_o = occ.addCylinder(0, 0, 0, 0, 0, 50, 18)
        pipe_i = occ.addCylinder(0, 0, 0, 0, 0, 50, 13)
        disk_o = occ.addCylinder(0, 0, 0, 0, 0, 8, 38)
        disk2_o = occ.addCylinder(0, 0, 42, 0, 0, 8, 38)
        body, _ = occ.fuse([(3, pipe_o)], [(3, disk_o), (3, disk2_o)])
        body, _ = occ.cut(body, [(3, pipe_i)])
        bolt_holes = []
        for deg in range(0, 360, 60):
            r = math.radians(deg)
            bolt_holes.append((3, occ.addCylinder(28 * math.cos(r), 28 * math.sin(r), -1, 0, 0, 12, 4)))
            bolt_holes.append((3, occ.addCylinder(28 * math.cos(r), 28 * math.sin(r), 41, 0, 0, 12, 4)))
        occ.cut(body, bolt_holes)

    elif model_key == "s_curve_pipe":
        t1 = occ.addTorus(20, 0, 0, 20, 8)
        t2 = occ.addTorus(-20, 0, 0, 20, 8)
        t1i = occ.addTorus(20, 0, 0, 20, 5)
        t2i = occ.addTorus(-20, 0, 0, 20, 5)
        box1 = occ.addBox(-10, -50, -20, 60, 50, 40)
        box2 = occ.addBox(-50, 0, -20, 60, 50, 40)
        b1, _ = occ.intersect([(3, t1)], [(3, box1)])
        b2, _ = occ.intersect([(3, t2)], [(3, box2)])
        box3 = occ.addBox(-10, -50, -20, 60, 50, 40)
        box4 = occ.addBox(-50, 0, -20, 60, 50, 40)
        h1, _ = occ.intersect([(3, t1i)], [(3, box3)])
        h2, _ = occ.intersect([(3, t2i)], [(3, box4)])
        body, _ = occ.fuse(b1, b2)
        hole, _ = occ.fuse(h1, h2)
        occ.cut(body, hole)

    elif model_key == "helix_coil":
        # 3 stacked torus rings approximating helix coil (reliable mesh generation)
        R, pitch, r = 30, 12, 5
        tags = [(3, occ.addTorus(0, 0, i * pitch, R, r)) for i in range(3)]
        occ.fuse([tags[0]], tags[1:])

    elif model_key == "l_bracket":
        a = occ.addBox(0, 0, 0, 80, 10, 8)
        b = occ.addBox(0, 0, 0, 8, 60, 8)
        occ.fuse([(3, a)], [(3, b)])

    elif model_key == "i_beam":
        top = occ.addBox(-30, -4, 0, 60, 8, 120)
        bottom = occ.addBox(-30, -54, 0, 60, 8, 120)
        web = occ.addBox(-4, -54, 0, 8, 58, 120)
        occ.fuse([(3, top)], [(3, bottom), (3, web)])

    elif model_key == "wing_spar":
        outer = occ.addBox(-6, -18, 0, 12, 36, 160)
        inner = occ.addBox(-4, -16, 0, 8, 32, 160)
        occ.cut([(3, outer)], [(3, inner)])

    elif model_key == "saddle_bracket":
        base = occ.addBox(-35, -20, 0, 70, 40, 8)
        saddle_cut = occ.addCylinder(0, 0, -5, 0, 0, 18, 20)
        ear1 = occ.addBox(-35, -20, 8, 12, 40, 30)
        ear2 = occ.addBox(23, -20, 8, 12, 40, 30)
        body, _ = occ.fuse([(3, base)], [(3, ear1), (3, ear2)])
        occ.cut(body, [(3, saddle_cut)])

    elif model_key == "box_frame":
        outer = occ.addBox(-40, -25, 0, 80, 50, 100)
        inner = occ.addBox(-34, -19, 0, 68, 38, 100)
        occ.cut([(3, outer)], [(3, inner)])

    elif model_key == "rect_frame":
        outer = occ.addBox(-50, -30, 0, 100, 60, 8)
        inner = occ.addBox(-40, -20, 0, 80, 40, 8)
        occ.cut([(3, outer)], [(3, inner)])

    elif model_key == "lattice_cube":
        r = 4
        edges = [
            ((0,0,0),(60,0,0)), ((0,60,0),(60,60,0)),
            ((0,0,60),(60,0,60)), ((0,60,60),(60,60,60)),
            ((0,0,0),(0,60,0)), ((60,0,0),(60,60,0)),
            ((0,0,60),(0,60,60)), ((60,0,60),(60,60,60)),
            ((0,0,0),(0,0,60)), ((60,0,0),(60,0,60)),
            ((0,60,0),(0,60,60)), ((60,60,0),(60,60,60)),
        ]
        bars = [(3, occ.addCylinder(x1-30, y1-30, z1-30, x2-x1, y2-y1, z2-z1, r))
                for (x1,y1,z1),(x2,y2,z2) in edges]
        occ.fuse([bars[0]], bars[1:])

    elif model_key == "curved_beam":
        full = occ.addTorus(0, 0, 0, 50, 8)
        box = occ.addBox(-80, -80, -12, 80, 160, 24)
        occ.cut([(3, full)], [(3, box)])

    elif model_key == "pressure_vessel":
        co = occ.addCylinder(0, 0, -40, 0, 0, 80, 28)
        c1o = occ.addSphere(0, 0, 40, 28)
        c2o = occ.addSphere(0, 0, -40, 28)
        ci = occ.addCylinder(0, 0, -40, 0, 0, 80, 23)
        c1i = occ.addSphere(0, 0, 40, 23)
        c2i = occ.addSphere(0, 0, -40, 23)
        outer, _ = occ.fuse([(3, co)], [(3, c1o), (3, c2o)])
        inner, _ = occ.fuse([(3, ci)], [(3, c1i), (3, c2i)])
        occ.cut(outer, inner)

    elif model_key == "dome":
        outer = occ.addSphere(0, 0, 0, 40)
        inner = occ.addSphere(0, 0, 0, 34)
        body, _ = occ.cut([(3, outer)], [(3, inner)])
        cut_box = occ.addBox(-50, -50, -50, 100, 100, 50)
        occ.cut(body, [(3, cut_box)])

    elif model_key == "sphere_shell":
        outer = occ.addSphere(0, 0, 0, 40)
        inner = occ.addSphere(0, 0, 0, 35)
        occ.cut([(3, outer)], [(3, inner)])

    elif model_key == "thermos":
        outer_wall = occ.addCylinder(0, 0, 0, 0, 0, 100, 30)
        inner_wall = occ.addCylinder(0, 0, 5, 0, 0, 90, 22)
        lid = occ.addCylinder(0, 0, 95, 0, 0, 8, 30)
        bottom = occ.addCylinder(0, 0, 0, 0, 0, 6, 30)
        body, _ = occ.fuse([(3, outer_wall)], [(3, lid), (3, bottom)])
        occ.cut(body, [(3, inner_wall)])

    elif model_key == "cone_shell":
        outer = occ.addCone(0, 0, 0, 0, 0, 80, 30, 15)
        inner = occ.addCone(0, 0, 0, 0, 0, 80, 25, 11)
        occ.cut([(3, outer)], [(3, inner)])

    elif model_key == "nozzle":
        outer = occ.addCone(0, 0, 0, 0, 0, 80, 28, 10)
        inner = occ.addCone(0, 0, 0, 0, 0, 80, 22, 6)
        occ.cut([(3, outer)], [(3, inner)])

    elif model_key == "propeller_hub":
        hub_o = occ.addCylinder(0, 0, 0, 0, 0, 18, 16)
        hub_i = occ.addCylinder(0, 0, 0, 0, 0, 18, 8)
        arms = []
        for deg in [0, 90, 180, 270]:
            rad = math.radians(deg)
            arm = occ.addBox(
                -3 * abs(math.sin(rad)) - 14 * abs(math.cos(rad)),
                -3 * abs(math.cos(rad)) - 14 * abs(math.sin(rad)),
                4,
                28 * abs(math.cos(rad)) + 6 * abs(math.sin(rad)),
                28 * abs(math.sin(rad)) + 6 * abs(math.cos(rad)),
                10,
            )
            arms.append((3, arm))
        body, _ = occ.fuse([(3, hub_o)], arms)
        occ.cut(body, [(3, hub_i)])

    elif model_key == "hex_tube":
        n, R, t = 6, 24.0, 4.0
        pts_o, pts_i = [], []
        for k in range(n):
            a = math.pi / 6 + k * 2 * math.pi / n
            pts_o.append(occ.addPoint(R * math.cos(a), R * math.sin(a), 0))
            pts_i.append(occ.addPoint((R - t) * math.cos(a), (R - t) * math.sin(a), 0))
        pts_o.append(pts_o[0])
        pts_i.append(pts_i[0])
        lo = occ.addCurveLoop([occ.addSpline(pts_o)])
        li = occ.addCurveLoop([occ.addSpline(pts_i)])
        face = occ.addPlaneSurface([lo, li])
        occ.extrude([(2, face)], 0, 0, 80)

    elif model_key == "bearing_housing":
        body_o = occ.addCylinder(0, 0, -15, 0, 0, 30, 35)
        body_i = occ.addCylinder(0, 0, -15, 0, 0, 30, 22)
        flange = occ.addCylinder(0, 0, -5, 0, 0, 10, 45)
        flange_i = occ.addCylinder(0, 0, -5, 0, 0, 10, 22)
        out, _ = occ.fuse([(3, body_o)], [(3, flange)])
        inn, _ = occ.fuse([(3, body_i)], [(3, flange_i)])
        occ.cut(out, inn)

    elif model_key == "stepped_shaft":
        s1 = occ.addCylinder(0, 0, 0, 0, 0, 30, 20)
        s2 = occ.addCylinder(0, 0, 30, 0, 0, 40, 14)
        s3 = occ.addCylinder(0, 0, 70, 0, 0, 25, 10)
        occ.fuse([(3, s1)], [(3, s2), (3, s3)])

    elif model_key == "torus_shell":
        outer = occ.addTorus(0, 0, 0, 35, 12)
        inner = occ.addTorus(0, 0, 0, 35, 8)
        occ.cut([(3, outer)], [(3, inner)])

    elif model_key == "annular_disk":
        outer = occ.addCylinder(0, 0, 0, 0, 0, 8, 40)
        inner = occ.addCylinder(0, 0, 0, 0, 0, 8, 15)
        occ.cut([(3, outer)], [(3, inner)])

    elif model_key == "turbine_blade":
        # NACA 0012 profile — skip degenerate leading-edge line, close via spline
        def naca(x):
            t = 0.12
            return 5 * t * (0.2969 * x**0.5 - 0.1260 * x - 0.3516 * x**2
                            + 0.2843 * x**3 - 0.1015 * x**4)
        scale = 40
        # Skip x=0 (degenerate LE) and use xs starting from small offset
        xs = [i / 20 for i in range(1, 21)]   # x from 0.05 to 1.0
        pts_top = [occ.addPoint(x * scale - 20, naca(x) * scale, 0) for x in xs]
        pts_bot = [occ.addPoint(x * scale - 20, -naca(x) * scale, 0) for x in xs]
        le_pt = occ.addPoint(-20, 0, 0)   # leading edge (x=0, y=0)
        # top: LE → TE, bot: TE → LE
        top_s = occ.addSpline([le_pt] + pts_top)
        bot_s = occ.addSpline(list(reversed(pts_bot)) + [le_pt])
        te = occ.addLine(pts_top[-1], pts_bot[-1])
        loop = occ.addCurveLoop([top_s, te, bot_s])
        face = occ.addPlaneSurface([loop])
        occ.extrude([(2, face)], 0, 0, 80)

    elif model_key == "rocket_fin":
        gpts = [occ.addPoint(x, y, 0) for x, y in [(0,0),(60,0),(40,40),(10,40)]]
        gpts.append(gpts[0])
        lines = [occ.addLine(gpts[i], gpts[i+1]) for i in range(4)]
        loop = occ.addCurveLoop(lines)
        face = occ.addPlaneSurface([loop])
        occ.extrude([(2, face)], 0, 0, 4)

    else:
        raise ValueError(f"Unknown built-in model: {model_key}")

    occ.synchronize()


def create_builtin_mesh(
    model_key: str,
    mesh_size: float = 3.0,
    output_path: str | None = None,
) -> str:
    """Generate 3D tetrahedral mesh from OCC primitives for a built-in model.

    Bypasses the STL round-trip for reliable meshing of all 30 built-in models.

    Args:
        model_key: model identifier (e.g. "hollow_tube", "l_bracket")
        mesh_size: target element size (mm)
        output_path: output .msh file path

    Returns:
        output file path
    """
    if output_path is None:
        output_path = f"/tmp/builtin_{model_key}.msh"

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add(model_key)

    _build_builtin_occ(model_key)

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size * 2.0)

    last_err = None
    for algo in [10, 4, 1]:
        try:
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.Algorithm3D", algo)
            gmsh.model.mesh.generate(3)
            last_err = None
            break
        except Exception as e:
            last_err = e

    if last_err is not None:
        gmsh.finalize()
        raise RuntimeError(f"3D mesh failed for {model_key}: {last_err}")

    gmsh.write(output_path)

    info = gmsh.model.mesh.getNodes()
    n_nodes = len(info[0])
    elems = gmsh.model.mesh.getElements(3)
    n_elems = sum(len(e) for e in elems[1])

    gmsh.finalize()
    print(f"Builtin mesh [{model_key}]: {n_nodes} nodes, {n_elems} elements → {output_path}")
    return output_path


def create_box_mesh(
    length: float,
    width: float,
    height: float,
    mesh_size: float = 2.0,
    output_path: str | None = None,
) -> str:
    """创建长方体网格 (用于悬臂梁benchmark)

    Args:
        length, width, height: 尺寸 (mm)
        mesh_size: 目标网格尺寸 (mm)
        output_path: 输出 .msh 文件路径

    Returns:
        输出文件路径
    """
    if output_path is None:
        output_path = f"box_{length}x{width}x{height}.msh"

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("box")

    gmsh.model.occ.addBox(0, 0, 0, length, width, height)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT

    # 标记物理组: 用于施加边界条件
    # 找到 x=0 的面 (固定端) 和 x=length 的面 (加载端)
    surfaces = gmsh.model.occ.getEntities(2)
    fixed_surfs = []
    load_surfs = []

    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(com[0]) < 1e-6:
            fixed_surfs.append(tag)
        elif abs(com[0] - length) < 1e-6:
            load_surfs.append(tag)

    gmsh.model.addPhysicalGroup(2, fixed_surfs, tag=1, name="fixed")
    gmsh.model.addPhysicalGroup(2, load_surfs, tag=2, name="load")
    gmsh.model.addPhysicalGroup(3, [1], tag=1, name="volume")

    gmsh.model.mesh.generate(3)
    gmsh.write(output_path)

    info = gmsh.model.mesh.getNodes()
    n_nodes = len(info[0])
    elems = gmsh.model.mesh.getElements(3)
    n_elems = sum(len(e) for e in elems[1])

    gmsh.finalize()
    print(f"Mesh: {n_nodes} nodes, {n_elems} elements → {output_path}")
    return output_path


def create_cylinder_mesh(
    outer_radius: float,
    inner_radius: float,
    height: float,
    mesh_size: float = 2.0,
    output_path: str | None = None,
) -> str:
    """创建筒状(空心圆柱)网格

    Args:
        outer_radius: 外径 (mm)
        inner_radius: 内径 (mm), 0=实心
        height: 高度 (mm)
        mesh_size: 目标网格尺寸 (mm)
        output_path: 输出路径

    Returns:
        输出文件路径
    """
    if output_path is None:
        output_path = f"cylinder_R{outer_radius}_r{inner_radius}_H{height}.msh"

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("cylinder")

    # 外圆柱 (沿 Z 轴, 底面圆心在原点)
    outer = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, outer_radius)

    if inner_radius > 0:
        # 内圆柱 (布尔减去得到空心)
        inner = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, inner_radius)
        result = gmsh.model.occ.cut([(3, outer)], [(3, inner)])
        gmsh.model.occ.synchronize()
        # cut 返回的体积 tag
        vol_tags = [t for d, t in result[0] if d == 3]
    else:
        gmsh.model.occ.synchronize()
        vol_tags = [outer]

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT

    # 标记物理组: z=0 底面 (fixed), z=height 顶面 (load)
    surfaces = gmsh.model.occ.getEntities(2)
    fixed_surfs = []
    load_surfs = []

    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(com[2]) < 1e-6:
            fixed_surfs.append(tag)
        elif abs(com[2] - height) < 1e-6:
            load_surfs.append(tag)

    gmsh.model.addPhysicalGroup(2, fixed_surfs, tag=1, name="fixed")
    gmsh.model.addPhysicalGroup(2, load_surfs, tag=2, name="load")
    gmsh.model.addPhysicalGroup(3, vol_tags, tag=1, name="volume")

    gmsh.model.mesh.generate(3)
    gmsh.write(output_path)

    info = gmsh.model.mesh.getNodes()
    n_nodes = len(info[0])
    elems = gmsh.model.mesh.getElements(3)
    n_elems = sum(len(e) for e in elems[1])

    gmsh.finalize()
    print(f"Cylinder mesh: {n_nodes} nodes, {n_elems} elements → {output_path}")
    return output_path


def create_cone_mesh(
    r_bottom: float,
    r_top: float,
    height: float,
    wall_thickness: float = 5.0,
    mesh_size: float = 2.0,
    output_path: str | None = None,
) -> str:
    """Create a hollow tapered cone/frustum mesh.

    Args:
        r_bottom: outer radius at z=0 (mm)
        r_top: outer radius at z=height (mm)
        wall_thickness: wall thickness (mm)
        height: height (mm)
        mesh_size: target mesh size (mm)
        output_path: output .msh path

    Returns:
        output file path
    """
    if output_path is None:
        output_path = f"cone_Rb{r_bottom}_Rt{r_top}_H{height}.msh"

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("cone")

    # Outer cone frustum (z=0 to z=height)
    outer = gmsh.model.occ.addCone(0, 0, 0, 0, 0, height, r_bottom, r_top)

    # Inner cone frustum (same axis, reduced radii)
    r_bottom_inner = r_bottom - wall_thickness
    r_top_inner = r_top - wall_thickness
    # Ensure inner radii are positive
    r_bottom_inner = max(r_bottom_inner, 0.5)
    r_top_inner = max(r_top_inner, 0.5)

    inner = gmsh.model.occ.addCone(0, 0, 0, 0, 0, height, r_bottom_inner, r_top_inner)

    # Boolean subtract inner from outer
    result = gmsh.model.occ.cut([(3, outer)], [(3, inner)])
    gmsh.model.occ.synchronize()

    vol_tags = [t for d, t in result[0] if d == 3]

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT

    # Tag physical groups: z=0 bottom face (fixed), z=height top face (load)
    surfaces = gmsh.model.occ.getEntities(2)
    fixed_surfs = []
    load_surfs = []

    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(com[2]) < 1e-6:
            fixed_surfs.append(tag)
        elif abs(com[2] - height) < 1e-6:
            load_surfs.append(tag)

    gmsh.model.addPhysicalGroup(2, fixed_surfs, tag=1, name="fixed")
    gmsh.model.addPhysicalGroup(2, load_surfs, tag=2, name="load")
    gmsh.model.addPhysicalGroup(3, vol_tags, tag=1, name="volume")

    gmsh.model.mesh.generate(3)
    gmsh.write(output_path)

    info = gmsh.model.mesh.getNodes()
    n_nodes = len(info[0])
    elems = gmsh.model.mesh.getElements(3)
    n_elems = sum(len(e) for e in elems[1])

    gmsh.finalize()
    print(f"Cone mesh: {n_nodes} nodes, {n_elems} elements → {output_path}")
    return output_path


def stl_to_mesh(
    stl_path: str,
    mesh_size: float = 2.0,
    refine_regions: list[dict] | None = None,
    output_path: str | None = None,
) -> str:
    """STL文件 → 自适应四面体网格（带算法回退，兼容非完美水密 STL）

    Args:
        stl_path: 输入STL文件路径
        mesh_size: 全局目标网格尺寸 (mm)
        refine_regions: 局部加密区域列表
        output_path: 输出 .msh 文件路径

    Returns:
        输出文件路径
    """
    stl_path = str(Path(stl_path).resolve())
    if output_path is None:
        output_path = str(Path(stl_path).with_suffix(".msh"))

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("stl_model")

    gmsh.merge(stl_path)

    # 宽松角度分类，兼容复杂/薄壁 STL
    gmsh.model.mesh.classifySurfaces(
        angle=np.pi,           # 180°：将所有相邻三角形归为同一曲面
        boundary=True,
        forReparametrization=False,
        curveAngle=np.pi,
    )
    gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()

    surfaces = gmsh.model.getEntities(2)
    if not surfaces:
        gmsh.finalize()
        raise RuntimeError("STL 导入后未发现曲面，文件可能损坏")

    sl = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
    gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.3)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

    if refine_regions:
        for i, region in enumerate(refine_regions):
            fid = i + 1
            gmsh.model.mesh.field.add("Ball", fid)
            gmsh.model.mesh.field.setNumber(fid, "VIn", region["size"])
            gmsh.model.mesh.field.setNumber(fid, "VOut", mesh_size)
            gmsh.model.mesh.field.setNumber(fid, "Radius", region["radius"])
            gmsh.model.mesh.field.setNumber(fid, "XCenter", region["center"][0])
            gmsh.model.mesh.field.setNumber(fid, "YCenter", region["center"][1])
            gmsh.model.mesh.field.setNumber(fid, "ZCenter", region["center"][2])
        mf = len(refine_regions) + 1
        gmsh.model.mesh.field.add("Min", mf)
        gmsh.model.mesh.field.setNumbers(mf, "FieldsList", list(range(1, mf)))
        gmsh.model.mesh.field.setAsBackgroundMesh(mf)

    # 算法回退：HXT(10) → Frontal-Delaunay(4) → Delaunay(1)
    last_err = None
    for algo in [10, 4, 1]:
        try:
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.Algorithm3D", algo)
            gmsh.model.mesh.generate(3)
            last_err = None
            break
        except Exception as e:
            last_err = e

    if last_err is not None:
        gmsh.finalize()
        raise RuntimeError(f"3D 网格生成失败 (所有算法均不可用): {last_err}")

    gmsh.write(output_path)
    gmsh.finalize()

    print(f"STL mesh → {output_path}")
    return output_path
