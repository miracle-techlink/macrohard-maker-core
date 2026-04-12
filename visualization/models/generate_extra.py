"""Generate 22 additional built-in model STL files using gmsh."""
import gmsh, math, os, sys

OUT = os.path.dirname(os.path.abspath(__file__))
gmsh.option.setNumber("General.Verbosity", 0)

def save(name):
    gmsh.model.mesh.generate(2)
    gmsh.write(os.path.join(OUT, name))
    gmsh.finalize()
    print(f"  ✓ {name}")

def new(name):
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add(name)

# ── 1. 90° elbow pipe ────────────────────────────────────────────────
def gen_elbow():
    new("elbow")
    # Torus sector as elbow body, cut to 90°
    full = gmsh.model.occ.addTorus(0,0,0, 40, 12)
    inner = gmsh.model.occ.addTorus(0,0,0, 40, 8)
    # Keep only first quadrant (x>0, y>0)
    box1 = gmsh.model.occ.addBox(-80,-80,-20, 80,160,40)
    box2 = gmsh.model.occ.addBox(-80,-80,-20, 160,80,40)
    body, _ = gmsh.model.occ.cut([(3,full)], [(3,inner)])
    gmsh.model.occ.intersect(body, [(3,box2)])
    gmsh.model.occ.synchronize()
    save("elbow_pipe.stl")

# ── 2. I-beam ────────────────────────────────────────────────────────
def gen_ibeam():
    new("ibeam")
    top    = gmsh.model.occ.addBox(-30,-4,0, 60,8,120)
    bottom = gmsh.model.occ.addBox(-30,-4,0, 60,8,0)  # reuse top offset
    bottom = gmsh.model.occ.addBox(-30,-54,0, 60,8,120)
    web    = gmsh.model.occ.addBox(-4,-54,0, 8,58,120)
    gmsh.model.occ.fuse([(3,top),(3,bottom),(3,web)], [])
    gmsh.model.occ.synchronize()
    save("i_beam.stl")

# ── 3. Cross / X pipe junction ───────────────────────────────────────
def gen_cross():
    new("cross")
    xo = gmsh.model.occ.addCylinder(-50,0,0,100,0,0,13)
    xi = gmsh.model.occ.addCylinder(-50,0,0,100,0,0,9)
    yo = gmsh.model.occ.addCylinder(0,-50,0,0,100,0,13)
    yi = gmsh.model.occ.addCylinder(0,-50,0,0,100,0,9)
    out, _ = gmsh.model.occ.fuse([(3,xo)],[(3,yo)])
    inn, _ = gmsh.model.occ.fuse([(3,xi)],[(3,yi)])
    gmsh.model.occ.cut(out, inn)
    gmsh.model.occ.synchronize()
    save("cross_joint.stl")

# ── 4. Hollow cone / frustum ─────────────────────────────────────────
def gen_cone():
    new("cone")
    outer = gmsh.model.occ.addCone(0,0,0, 0,0,80, 30,15)
    inner = gmsh.model.occ.addCone(0,0,0, 0,0,80, 25,11)
    gmsh.model.occ.cut([(3,outer)],[(3,inner)])
    gmsh.model.occ.synchronize()
    save("cone_shell.stl")

# ── 5. Hemisphere dome ───────────────────────────────────────────────
def gen_dome():
    new("dome")
    outer = gmsh.model.occ.addSphere(0,0,0,40)
    inner = gmsh.model.occ.addSphere(0,0,0,34)
    cut_box = gmsh.model.occ.addBox(-50,-50,-50, 100,100,50)
    body, _ = gmsh.model.occ.cut([(3,outer)],[(3,inner)])
    gmsh.model.occ.cut(body, [(3,cut_box)])
    gmsh.model.occ.synchronize()
    save("dome.stl")

# ── 6. Y-junction pipe ───────────────────────────────────────────────
def gen_yjoint():
    new("yjoint")
    ao = gmsh.model.occ.addCylinder(-40,0,0,80,0,0,12)
    ai = gmsh.model.occ.addCylinder(-40,0,0,80,0,0,8)
    ang = math.radians(45)
    bo = gmsh.model.occ.addCylinder(0,0,0, 40*math.cos(ang),40*math.sin(ang),0, 12)
    bi = gmsh.model.occ.addCylinder(0,0,0, 40*math.cos(ang),40*math.sin(ang),0, 8)
    out, _ = gmsh.model.occ.fuse([(3,ao)],[(3,bo)])
    inn, _ = gmsh.model.occ.fuse([(3,ai)],[(3,bi)])
    gmsh.model.occ.cut(out, inn)
    gmsh.model.occ.synchronize()
    save("y_junction.stl")

# ── 7. Pipe flange ───────────────────────────────────────────────────
def gen_flange():
    new("flange")
    pipe_o = gmsh.model.occ.addCylinder(0,0,0,0,0,50,18)
    pipe_i = gmsh.model.occ.addCylinder(0,0,0,0,0,50,13)
    disk_o = gmsh.model.occ.addCylinder(0,0,0,0,0,8,38)
    disk_i = gmsh.model.occ.addCylinder(0,0,0,0,0,8,13)
    disk2_o= gmsh.model.occ.addCylinder(0,0,42,0,0,8,38)
    disk2_i= gmsh.model.occ.addCylinder(0,0,42,0,0,8,13)
    body, _ = gmsh.model.occ.fuse([(3,pipe_o),(3,disk_o),(3,disk2_o)],[])
    holes, _ = gmsh.model.occ.fuse([(3,pipe_i),(3,disk_i),(3,disk2_i)],[])
    gmsh.model.occ.cut(body, holes)
    # Bolt holes on flange
    bolt_holes = []
    for deg in range(0,360,60):
        r = math.radians(deg)
        bh = gmsh.model.occ.addCylinder(28*math.cos(r),28*math.sin(r),-1, 0,0,12, 4)
        bolt_holes.append((3,bh))
        bh2 = gmsh.model.occ.addCylinder(28*math.cos(r),28*math.sin(r),41, 0,0,12, 4)
        bolt_holes.append((3,bh2))
    gmsh.model.occ.cut([(3,1)], bolt_holes, removeObject=False, removeTool=True)
    gmsh.model.occ.synchronize()
    save("flange.stl")

# ── 8. Nozzle (converging tube) ──────────────────────────────────────
def gen_nozzle():
    new("nozzle")
    outer = gmsh.model.occ.addCone(0,0,0,0,0,80,28,10)
    inner = gmsh.model.occ.addCone(0,0,0,0,0,80,22,6)
    gmsh.model.occ.cut([(3,outer)],[(3,inner)])
    gmsh.model.occ.synchronize()
    save("nozzle.stl")

# ── 9. Box frame / chassis ───────────────────────────────────────────
def gen_boxframe():
    new("boxframe")
    outer = gmsh.model.occ.addBox(-40,-25,0,80,50,100)
    inner = gmsh.model.occ.addBox(-34,-19,0,68,38,100)
    gmsh.model.occ.cut([(3,outer)],[(3,inner)])
    gmsh.model.occ.synchronize()
    save("box_frame.stl")

# ── 10. Flat annular disk ────────────────────────────────────────────
def gen_disk():
    new("disk")
    outer = gmsh.model.occ.addCylinder(0,0,0,0,0,8,40)
    inner = gmsh.model.occ.addCylinder(0,0,0,0,0,8,15)
    gmsh.model.occ.cut([(3,outer)],[(3,inner)])
    gmsh.model.occ.synchronize()
    save("annular_disk.stl")

# ── 11. Bearing housing ──────────────────────────────────────────────
def gen_bearing():
    new("bearing")
    body_o = gmsh.model.occ.addCylinder(0,0,-15,0,0,30,35)
    body_i = gmsh.model.occ.addCylinder(0,0,-15,0,0,30,22)
    flange = gmsh.model.occ.addCylinder(0,0,-5,0,0,10,45)
    flange_i= gmsh.model.occ.addCylinder(0,0,-5,0,0,10,22)
    out, _ = gmsh.model.occ.fuse([(3,body_o),(3,flange)],[])
    inn, _ = gmsh.model.occ.fuse([(3,body_i),(3,flange_i)],[])
    gmsh.model.occ.cut(out, inn)
    gmsh.model.occ.synchronize()
    save("bearing_housing.stl")

# ── 12. Helix coil tube ──────────────────────────────────────────────
def gen_helix():
    new("helix")
    n_turns = 3
    n_pts = n_turns * 36
    R, pitch, r = 30, 12, 5
    pts = []
    for i in range(n_pts+1):
        t = i / n_pts * n_turns * 2 * math.pi
        pts.append(gmsh.model.occ.addPoint(R*math.cos(t), R*math.sin(t), pitch/(2*math.pi)*t))
    spline = gmsh.model.occ.addSpline(pts)
    # Disk profile to sweep (use pipe)
    cp = gmsh.model.occ.addPoint(R,0,0)
    circle_pts = [gmsh.model.occ.addPoint(R+r*math.cos(a), r*math.sin(a), 0) for a in [0,math.pi/2,math.pi,3*math.pi/2,2*math.pi]]
    circle_pts[-1] = circle_pts[0]
    arc1 = gmsh.model.occ.addCircleArc(circle_pts[0],cp,circle_pts[1])
    arc2 = gmsh.model.occ.addCircleArc(circle_pts[1],cp,circle_pts[2])
    arc3 = gmsh.model.occ.addCircleArc(circle_pts[2],cp,circle_pts[3])
    arc4 = gmsh.model.occ.addCircleArc(circle_pts[3],cp,circle_pts[0])
    loop = gmsh.model.occ.addCurveLoop([arc1,arc2,arc3,arc4])
    profile = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.addPipe([(2,profile)], spline)
    gmsh.model.occ.synchronize()
    save("helix_coil.stl")

# ── 13. Double wall cylinder (thermos) ───────────────────────────────
def gen_thermos():
    new("thermos")
    outer_wall = gmsh.model.occ.addCylinder(0,0,0,0,0,100,30)
    outer_void = gmsh.model.occ.addCylinder(0,0,0,0,0,100,27)
    inner_wall = gmsh.model.occ.addCylinder(0,0,5,0,0,90,22)
    inner_void = gmsh.model.occ.addCylinder(0,0,5,0,0,90,19)
    lid = gmsh.model.occ.addCylinder(0,0,95,0,0,8,30)
    bottom = gmsh.model.occ.addCylinder(0,0,0,0,0,6,30)
    body, _ = gmsh.model.occ.fuse([(3,outer_wall),(3,lid),(3,bottom)],[])
    voids, _ = gmsh.model.occ.fuse([(3,outer_void),(3,inner_void)],[])
    gmsh.model.occ.cut(body, [(3,inner_wall)])
    gmsh.model.occ.synchronize()
    save("thermos.stl")

# ── 14. Turbine blade profile ────────────────────────────────────────
def gen_blade():
    new("blade")
    # NACA-like profile extruded + twisted
    def naca(x):
        t = 0.12
        y = 5*t*(0.2969*x**0.5 - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
        return y
    pts_top, pts_bot = [], []
    xs = [i/20 for i in range(21)]
    for x in xs:
        y = naca(x)
        scale = 40
        pts_top.append(gmsh.model.occ.addPoint(x*scale - 20, y*scale, 0))
        pts_bot.append(gmsh.model.occ.addPoint(x*scale - 20, -y*scale, 0))
    top_spline = gmsh.model.occ.addSpline(pts_top)
    bot_spline = gmsh.model.occ.addSpline(list(reversed(pts_bot)))
    te_line = gmsh.model.occ.addLine(pts_top[-1], pts_bot[-1])
    le_line = gmsh.model.occ.addLine(pts_bot[0], pts_top[0])
    loop = gmsh.model.occ.addCurveLoop([top_spline, te_line, bot_spline, le_line])
    face = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.extrude([(2,face)], 0, 0, 80)
    gmsh.model.occ.synchronize()
    save("turbine_blade.stl")

# ── 15. Rocket fin ───────────────────────────────────────────────────
def gen_fin():
    new("fin")
    pts = [
        (0,0), (60,0), (40,40), (10,40)
    ]
    gpts = [gmsh.model.occ.addPoint(x,y,0) for x,y in pts]
    gpts.append(gpts[0])
    lines = [gmsh.model.occ.addLine(gpts[i],gpts[i+1]) for i in range(len(pts))]
    loop = gmsh.model.occ.addCurveLoop(lines)
    face = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.extrude([(2,face)],0,0,4)
    gmsh.model.occ.synchronize()
    save("rocket_fin.stl")

# ── 16. Saddle bracket ───────────────────────────────────────────────
def gen_saddle():
    new("saddle")
    base = gmsh.model.occ.addBox(-35,-20,0,70,40,8)
    # Saddle cutout (half cylinder removed from bottom)
    saddle_cut = gmsh.model.occ.addCylinder(0,0,-5,0,0,18,20)
    # Two upright ears
    ear1 = gmsh.model.occ.addBox(-35,-20,8,12,40,30)
    ear2 = gmsh.model.occ.addBox(23,-20,8,12,40,30)
    body, _ = gmsh.model.occ.fuse([(3,base),(3,ear1),(3,ear2)],[])
    gmsh.model.occ.cut(body, [(3,saddle_cut)])
    gmsh.model.occ.synchronize()
    save("saddle_bracket.stl")

# ── 17. Torus (donut) ────────────────────────────────────────────────
def gen_torus():
    new("torus")
    outer = gmsh.model.occ.addTorus(0,0,0,35,12)
    inner = gmsh.model.occ.addTorus(0,0,0,35,8)
    gmsh.model.occ.cut([(3,outer)],[(3,inner)])
    gmsh.model.occ.synchronize()
    save("torus_shell.stl")

# ── 18. Rectangular frame plate ─────────────────────────────────────
def gen_frame():
    new("frame")
    outer = gmsh.model.occ.addBox(-50,-30,0,100,60,8)
    inner = gmsh.model.occ.addBox(-40,-20,0,80,40,8)
    gmsh.model.occ.cut([(3,outer)],[(3,inner)])
    gmsh.model.occ.synchronize()
    save("rect_frame.stl")

# ── 19. Sphere shell ─────────────────────────────────────────────────
def gen_sphere():
    new("sphere")
    outer = gmsh.model.occ.addSphere(0,0,0,40)
    inner = gmsh.model.occ.addSphere(0,0,0,35)
    gmsh.model.occ.cut([(3,outer)],[(3,inner)])
    gmsh.model.occ.synchronize()
    save("sphere_shell.stl")

# ── 20. Stepped shaft ────────────────────────────────────────────────
def gen_shaft():
    new("shaft")
    s1 = gmsh.model.occ.addCylinder(0,0,0,0,0,30,20)
    s2 = gmsh.model.occ.addCylinder(0,0,30,0,0,40,14)
    s3 = gmsh.model.occ.addCylinder(0,0,70,0,0,25,10)
    gmsh.model.occ.fuse([(3,s1),(3,s2),(3,s3)],[])
    gmsh.model.occ.synchronize()
    save("stepped_shaft.stl")

# ── 21. Lattice cube frame ───────────────────────────────────────────
def gen_lattice():
    new("lattice")
    r = 4
    bars = []
    # 12 edges of a 60mm cube
    edges = [
        ((0,0,0),(60,0,0)),((0,60,0),(60,60,0)),
        ((0,0,60),(60,0,60)),((0,60,60),(60,60,60)),
        ((0,0,0),(0,60,0)),((60,0,0),(60,60,0)),
        ((0,0,60),(0,60,60)),((60,0,60),(60,60,60)),
        ((0,0,0),(0,0,60)),((60,0,0),(60,0,60)),
        ((0,60,0),(0,60,60)),((60,60,0),(60,60,60)),
    ]
    for (x1,y1,z1),(x2,y2,z2) in edges:
        dx,dy,dz = x2-x1,y2-y1,z2-z1
        bars.append((3, gmsh.model.occ.addCylinder(x1-30,y1-30,z1-30,dx,dy,dz,r)))
    gmsh.model.occ.fuse([bars[0]], bars[1:])
    gmsh.model.occ.synchronize()
    save("lattice_cube.stl")

# ── 22. S-curve pipe ─────────────────────────────────────────────────
def gen_scurve():
    new("scurve")
    # Two half-tori forming S shape
    t1 = gmsh.model.occ.addTorus(20,0,0, 20, 8)
    t2 = gmsh.model.occ.addTorus(-20,0,0, 20, 8)
    t1i = gmsh.model.occ.addTorus(20,0,0, 20, 5)
    t2i = gmsh.model.occ.addTorus(-20,0,0, 20, 5)
    # Cut to half each: t1 keep y>0, t2 keep y<0
    box1 = gmsh.model.occ.addBox(-10,-50,-20,60,50,40)
    box2 = gmsh.model.occ.addBox(-50,0,-20,60,50,40)
    b1, _ = gmsh.model.occ.intersect([(3,t1)],[(3,box1)])
    b2, _ = gmsh.model.occ.intersect([(3,t2)],[(3,box2)])
    h1, _ = gmsh.model.occ.intersect([(3,t1i)],[(3,gmsh.model.occ.addBox(-10,-50,-20,60,50,40))])
    h2, _ = gmsh.model.occ.intersect([(3,t2i)],[(3,gmsh.model.occ.addBox(-50,0,-20,60,50,40))])
    body, _ = gmsh.model.occ.fuse(b1, b2)
    hole, _ = gmsh.model.occ.fuse(h1, h2)
    gmsh.model.occ.cut(body, hole)
    gmsh.model.occ.synchronize()
    save("s_curve_pipe.stl")


if __name__ == "__main__":
    print("Generating 22 additional models...")
    jobs = [
        ("elbow_pipe",      gen_elbow),
        ("i_beam",          gen_ibeam),
        ("cross_joint",     gen_cross),
        ("cone_shell",      gen_cone),
        ("dome",            gen_dome),
        ("y_junction",      gen_yjoint),
        ("flange",          gen_flange),
        ("nozzle",          gen_nozzle),
        ("box_frame",       gen_boxframe),
        ("annular_disk",    gen_disk),
        ("bearing_housing", gen_bearing),
        ("helix_coil",      gen_helix),
        ("thermos",         gen_thermos),
        ("turbine_blade",   gen_blade),
        ("rocket_fin",      gen_fin),
        ("saddle_bracket",  gen_saddle),
        ("torus_shell",     gen_torus),
        ("rect_frame",      gen_frame),
        ("sphere_shell",    gen_sphere),
        ("stepped_shaft",   gen_shaft),
        ("lattice_cube",    gen_lattice),
        ("s_curve_pipe",    gen_scurve),
    ]
    ok, fail = 0, []
    for name, fn in jobs:
        try:
            fn()
            ok += 1
        except Exception as e:
            fail.append((name, str(e)))
            try: gmsh.finalize()
            except: pass
            print(f"  ✗ {name}: {e}")
    print(f"\nDone: {ok} ok, {len(fail)} failed")
    if fail:
        print("Failed:", [n for n,_ in fail])
