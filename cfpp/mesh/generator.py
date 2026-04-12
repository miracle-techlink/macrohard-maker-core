"""网格生成模块: STL → 自适应四面体网格"""

import gmsh
import numpy as np
from pathlib import Path


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
    """STL文件 → 自适应四面体网格

    Args:
        stl_path: 输入STL文件路径
        mesh_size: 全局目标网格尺寸 (mm)
        refine_regions: 局部加密区域列表, e.g. [{"center": [x,y,z], "radius": r, "size": s}]
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
    gmsh.model.mesh.classifySurfaces(angle=40 * np.pi / 180, boundary=True, forReparametrization=True)
    gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()

    # 创建体积
    sl = gmsh.model.geo.addSurfaceLoop(
        [s[1] for s in gmsh.model.getEntities(2)]
    )
    gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()

    # 全局网格尺寸
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.3)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

    # 局部加密
    if refine_regions:
        for i, region in enumerate(refine_regions):
            field_id = i + 1
            gmsh.model.mesh.field.add("Ball", field_id)
            gmsh.model.mesh.field.setNumber(field_id, "VIn", region["size"])
            gmsh.model.mesh.field.setNumber(field_id, "VOut", mesh_size)
            gmsh.model.mesh.field.setNumber(field_id, "Radius", region["radius"])
            gmsh.model.mesh.field.setNumber(field_id, "XCenter", region["center"][0])
            gmsh.model.mesh.field.setNumber(field_id, "YCenter", region["center"][1])
            gmsh.model.mesh.field.setNumber(field_id, "ZCenter", region["center"][2])

        min_field = len(refine_regions) + 1
        gmsh.model.mesh.field.add("Min", min_field)
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", list(range(1, min_field)))
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.model.mesh.generate(3)
    gmsh.write(output_path)
    gmsh.finalize()

    print(f"STL mesh → {output_path}")
    return output_path
