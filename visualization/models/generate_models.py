"""Generate built-in model library STL files using gmsh."""
import gmsh
import math
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
gmsh.option.setNumber("General.Verbosity", 0)


def _save(name: str):
    path = os.path.join(OUT_DIR, name)
    gmsh.model.mesh.generate(2)
    gmsh.write(path)
    gmsh.finalize()
    print(f"  ✓ {name}")


def gen_hollow_tube():
    """Hollow cylindrical tube — basic CF winding demo."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("hollow_tube")
    outer = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 80, 25)
    inner = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 80, 20)
    gmsh.model.occ.cut([(3, outer)], [(3, inner)])
    gmsh.model.occ.synchronize()
    _save("hollow_tube.stl")


def gen_l_bracket():
    """L-shaped structural bracket."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("l_bracket")
    a = gmsh.model.occ.addBox(0, 0, 0, 80, 10, 8)
    b = gmsh.model.occ.addBox(0, 0, 0, 8, 60, 8)
    gmsh.model.occ.fuse([(3, a)], [(3, b)])
    gmsh.model.occ.synchronize()
    _save("l_bracket.stl")


def gen_t_joint():
    """T-shaped pipe junction."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("t_joint")
    # Main tube outer
    mo = gmsh.model.occ.addCylinder(-40, 0, 0, 80, 0, 0, 14)
    mi = gmsh.model.occ.addCylinder(-40, 0, 0, 80, 0, 0, 10)
    # Branch tube outer
    bo = gmsh.model.occ.addCylinder(0, 0, 0, 0, 40, 0, 14)
    bi = gmsh.model.occ.addCylinder(0, 0, 0, 0, 40, 0, 10)
    # Fuse outers, fuse inners, cut
    out_fused, _ = gmsh.model.occ.fuse([(3, mo)], [(3, bo)])
    in_fused, _ = gmsh.model.occ.fuse([(3, mi)], [(3, bi)])
    gmsh.model.occ.cut(out_fused, in_fused)
    gmsh.model.occ.synchronize()
    _save("t_joint.stl")


def gen_pressure_vessel():
    """Pressure vessel: cylinder + hemispherical caps (hollow)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("pressure_vessel")
    co = gmsh.model.occ.addCylinder(0, 0, -40, 0, 0, 80, 28)
    c1o = gmsh.model.occ.addSphere(0, 0, 40, 28)
    c2o = gmsh.model.occ.addSphere(0, 0, -40, 28)
    ci = gmsh.model.occ.addCylinder(0, 0, -40, 0, 0, 80, 23)
    c1i = gmsh.model.occ.addSphere(0, 0, 40, 23)
    c2i = gmsh.model.occ.addSphere(0, 0, -40, 23)
    outer, _ = gmsh.model.occ.fuse([(3, co)], [(3, c1o), (3, c2o)])
    inner, _ = gmsh.model.occ.fuse([(3, ci)], [(3, c1i), (3, c2i)])
    gmsh.model.occ.cut(outer, inner)
    gmsh.model.occ.synchronize()
    _save("pressure_vessel.stl")


def gen_propeller_hub():
    """Quadcopter motor hub with 4 arms."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("propeller_hub")
    hub_o = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 18, 16)
    hub_i = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 18, 8)
    arms = []
    for deg in [0, 90, 180, 270]:
        rad = math.radians(deg)
        arm = gmsh.model.occ.addBox(
            -3 * abs(math.sin(rad)) - 14 * abs(math.cos(rad)),
            -3 * abs(math.cos(rad)) - 14 * abs(math.sin(rad)),
            4,
            28 * abs(math.cos(rad)) + 6 * abs(math.sin(rad)),
            28 * abs(math.sin(rad)) + 6 * abs(math.cos(rad)),
            10,
        )
        arms.append((3, arm))
    body, _ = gmsh.model.occ.fuse([(3, hub_o)], arms)
    gmsh.model.occ.cut(body, [(3, hub_i)])
    gmsh.model.occ.synchronize()
    _save("propeller_hub.stl")


def gen_wing_spar():
    """Hollow rectangular wing spar — aerospace beam."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("wing_spar")
    outer = gmsh.model.occ.addBox(-6, -18, 0, 12, 36, 160)
    inner = gmsh.model.occ.addBox(-4, -16, 0, 8, 32, 160)
    gmsh.model.occ.cut([(3, outer)], [(3, inner)])
    gmsh.model.occ.synchronize()
    _save("wing_spar.stl")


def gen_hex_tube():
    """Hexagonal cross-section tube."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("hex_tube")
    n, R, t = 6, 24.0, 4.0
    pts_o, pts_i = [], []
    for k in range(n):
        a = math.pi / 6 + k * 2 * math.pi / n
        pts_o.append(gmsh.model.occ.addPoint(R * math.cos(a), R * math.sin(a), 0))
        pts_i.append(gmsh.model.occ.addPoint((R - t) * math.cos(a), (R - t) * math.sin(a), 0))
    pts_o.append(pts_o[0]); pts_i.append(pts_i[0])
    lo = gmsh.model.occ.addCurveLoop([gmsh.model.occ.addBSpline(pts_o)])
    li = gmsh.model.occ.addCurveLoop([gmsh.model.occ.addBSpline(pts_i)])
    face = gmsh.model.occ.addPlaneSurface([lo, li])
    gmsh.model.occ.extrude([(2, face)], 0, 0, 80)
    gmsh.model.occ.synchronize()
    _save("hex_tube.stl")


def gen_curved_beam():
    """Curved C-shaped structural beam."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("curved_beam")
    # Torus sector: R=50 (major), r=8 (minor), angle=180°
    full = gmsh.model.occ.addTorus(0, 0, 0, 50, 8)
    # Cut away half with a box
    box = gmsh.model.occ.addBox(-80, -80, -12, 80, 160, 24)
    gmsh.model.occ.cut([(3, full)], [(3, box)])
    gmsh.model.occ.synchronize()
    _save("curved_beam.stl")


if __name__ == "__main__":
    print("Generating built-in model library...")
    gen_hollow_tube()
    gen_l_bracket()
    gen_t_joint()
    gen_pressure_vessel()
    gen_propeller_hub()
    gen_wing_spar()
    gen_hex_tube()
    gen_curved_beam()
    print("Done.")
