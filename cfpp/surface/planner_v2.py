"""Unified XY+A path planning backend for continuous carbon fiber 3D printing.

Coordinate conventions:
  - Cylinder axis is along X.
  - A-axis (rotary platform) rotates about X.
  - Winding angle: 0° = axial (along X), 90° = hoop (circumferential).
  - Waypoint tuple: (x, y_model, z_model, a_deg)
      x       — axial position along cylinder (mm)
      y_model — nozzle Y in model coords (0 for top-of-cylinder deposition)
      z_model — nozzle Z in model coords (= radius for top-of-cylinder)
      a_deg   — A-axis rotation angle (degrees, cumulative)

The nozzle is assumed to deposit at the "top" of the cylinder (y_model=0,
z_model=radius).  The rotary platform carries the cylinder, so the A-axis
drives the circumferential position.

Winding geometry for a cylinder of radius r and length L:
  - circumference C = 2*pi*r
  - pitch per revolution p = C / tan(alpha)  (alpha = winding angle)
  - number of revolutions for one pass: L / p = L*tan(alpha) / C
"""

import numpy as np
from math import pi, sin, cos, tan, radians, degrees, atan2, sqrt
from typing import Callable, List, Optional, Tuple


# Type alias for a single waypoint
Waypoint = Tuple[float, float, float, float]  # (x, y_model, z_model, a_deg)


class XYAPathPlanner:
    """Generate fiber-winding waypoints for a cylinder mounted on an XY+A printer.

    All path methods return a list of Waypoint tuples:
        (x, y_model, z_model, a_deg)

    Parameters
    ----------
    a_offset_y : float
        Y-offset from part origin to the A-axis centre of rotation (default 0).
    a_offset_z : float
        Z-offset from part origin to the A-axis centre of rotation (default 50).
    """

    def __init__(self, a_offset_y: float = 0.0, a_offset_z: float = 50.0):
        self.a_offset_y = a_offset_y
        self.a_offset_z = a_offset_z

    # ------------------------------------------------------------------ #
    #  1. Constant-angle helical winding
    # ------------------------------------------------------------------ #
    def constant_angle_winding(
        self,
        radius: float,
        length: float,
        angle_deg: float,
        n_layers: int = 4,
        fiber_width: float = 2.0,
        points_per_pass: int = 48,
    ) -> List[Waypoint]:
        """Generate full-coverage helical winding at a constant angle.

        Each layer consists of multiple parallel fiber passes evenly spaced
        around the circumference so the whole surface is covered.  Alternating
        layers wind at +alpha / -alpha for a balanced ±alpha layup.

        Parameters
        ----------
        radius : float
            Cylinder outer radius (mm).
        length : float
            Cylinder axial length (mm).
        angle_deg : float
            Winding angle in degrees (0° = axial, 90° = hoop).
        n_layers : int
            Number of winding layers.  Each layer fully covers the surface.
        fiber_width : float
            Effective fiber/tow width in mm (default 2.0 mm for CFRP tow).
            Determines the number of parallel fibers needed per layer.
        points_per_pass : int
            Number of G-code sample points per one-way fiber pass.

        Returns
        -------
        list[Waypoint]
            Flat list of (x, y_model, z_model, a_deg) waypoints.
            Consecutive waypoints within the same fiber are connected;
            between fibers there is a position jump (travel move).
        """
        circumference = 2.0 * pi * radius
        a_eff = max(0.5, min(89.5, abs(angle_deg)))
        angle_rad = radians(a_eff)

        # Pitch: axial distance travelled per full revolution
        pitch = circumference / tan(angle_rad)
        # Turns per one-way traverse (may be < 1 for long pitch)
        turns_per_pass = length / pitch

        # Number of fibers needed to fill circumference in one layer
        n_fibers = max(1, round(circumference / fiber_width))
        # Circumferential spacing between adjacent fibers (degrees)
        fiber_spacing_deg = 360.0 / n_fibers

        waypoints: List[Waypoint] = []

        for layer in range(n_layers):
            direction = 1 if layer % 2 == 0 else -1   # +α / −α alternation
            # Offset starting angle per layer so layers interleave nicely
            layer_offset_deg = (layer * fiber_spacing_deg / n_layers) % fiber_spacing_deg

            for fi in range(n_fibers):
                # Starting circumferential angle for this fiber
                a_start = layer_offset_deg + fi * fiber_spacing_deg

                # X start position
                x_start = 0.0 if direction > 0 else length

                for i in range(points_per_pass + 1):
                    t = i / points_per_pass
                    x = x_start + direction * t * length
                    x = max(0.0, min(length, x))
                    # A rotation: proportional to axial travel
                    a = a_start + direction * t * turns_per_pass * 360.0
                    waypoints.append((float(x), 0.0, float(radius), float(a)))

        return waypoints

    # ------------------------------------------------------------------ #
    #  2. Variable-angle winding
    # ------------------------------------------------------------------ #
    def variable_angle_winding(
        self,
        radius: float,
        length: float,
        angle_func: Callable[[float, int], float],
        n_layers: int = 1,
        n_steps: int = 500,
    ) -> List[Waypoint]:
        """Generate winding with spatially varying angle.

        The winding angle is a function of axial position x and layer index.
        This is useful for stress-adaptive winding where the angle is tuned
        to the local principal stress direction.

        Parameters
        ----------
        radius : float
            Cylinder outer radius (mm).
        length : float
            Cylinder axial length (mm).
        angle_func : Callable[[float, int], float]
            Function(x, layer) → winding angle in degrees.
        n_layers : int
            Number of winding passes.
        n_steps : int
            Number of integration steps per layer.

        Returns
        -------
        list[Waypoint]
        """
        circumference = 2.0 * pi * radius
        waypoints = []
        cumulative_a = 0.0

        for layer in range(n_layers):
            direction = 1 if layer % 2 == 0 else -1
            x_step = direction * length / n_steps

            x = 0.0 if direction > 0 else length

            for i in range(n_steps + 1):
                x_clamped = max(0.0, min(length, x))
                angle_deg = angle_func(x_clamped, layer)
                angle_rad = radians(np.clip(float(angle_deg), 0.1, 89.9))

                # Arc length along circumference for one x_step
                # tan(alpha) = dC/dX  → dC = dX * tan(alpha)
                dx = abs(x_step)
                dtheta_deg = degrees(dx * tan(angle_rad) / radius)  # rotation in degrees

                cumulative_a += direction * dtheta_deg

                y_model = 0.0
                z_model = float(radius)
                waypoints.append((float(x_clamped), y_model, z_model, float(cumulative_a)))

                x += x_step

        return waypoints

    # ------------------------------------------------------------------ #
    #  3. Stress-aligned winding
    # ------------------------------------------------------------------ #
    def stress_aligned_winding(
        self,
        radius: float,
        length: float,
        stress_field: "StressFieldLike",
        n_layers: int = 2,
        n_x_samples: int = 50,
        n_steps_per_layer: int = 400,
    ) -> List[Waypoint]:
        """Generate winding paths aligned with principal stress directions.

        The stress field is sampled at evenly-spaced axial positions on the
        cylinder outer surface.  At each position the principal stress
        direction is projected onto the cylinder tangent plane and converted
        to a winding angle.

        Parameters
        ----------
        radius : float
            Cylinder outer radius (mm).
        length : float
            Cylinder axial length (mm).
        stress_field : object
            Object with a ``query(point)`` method returning
            ``(direction_3d, value)`` where direction_3d is a unit vector
            in 3D Cartesian space (XZC model — old convention, we translate).
            Alternatively, accepts a dict ``{x_pos: angle_deg}`` as a simple
            lookup table.
        n_layers : int
            Number of winding passes.
        n_x_samples : int
            Number of axial positions to sample the stress field.
        n_steps_per_layer : int
            Integration steps per layer.

        Returns
        -------
        list[Waypoint]
        """
        x_samples = np.linspace(0.0, length, n_x_samples)

        # Build an angle lookup table from the stress field
        angle_table: List[Tuple[float, float]] = []  # (x, angle_deg)

        for x_pos in x_samples:
            if isinstance(stress_field, dict):
                # Simple dict lookup: find nearest key
                keys = np.array(sorted(stress_field.keys()))
                nearest_key = keys[np.argmin(np.abs(keys - x_pos))]
                alpha_deg = float(stress_field[nearest_key])
            else:
                # Object with query(point) → (direction, value)
                # Sample at the top of the cylinder (y=0, z=radius)
                # Note: the old code uses XZC where x=axial, z=height, c=rotation
                # Here we use XY+A: point on surface top = (x, 0, radius)
                point = np.array([x_pos, 0.0, radius])
                try:
                    direction, value = stress_field.query(point)
                    # direction is a unit vector on the surface tangent plane
                    # On the outer cylinder, the tangent plane is spanned by:
                    #   e_x = (1, 0, 0)  — axial
                    #   e_circ = (0, 1, 0)  — circumferential (at top)
                    # Winding angle alpha = atan2(|e_x component|, |e_circ component|)
                    d = np.asarray(direction, dtype=float)
                    if np.linalg.norm(d) > 1e-10:
                        d /= np.linalg.norm(d)
                    comp_axial = abs(d[0])
                    comp_circ = sqrt(d[1]**2 + d[2]**2)
                    alpha_deg = degrees(atan2(comp_axial, comp_circ))
                    alpha_deg = float(np.clip(alpha_deg, 5.0, 85.0))
                except Exception:
                    alpha_deg = 45.0

            angle_table.append((float(x_pos), alpha_deg))

        # Interpolation function for angle vs x
        x_arr = np.array([t[0] for t in angle_table])
        a_arr = np.array([t[1] for t in angle_table])

        def angle_func(x: float, layer: int) -> float:
            return float(np.interp(x, x_arr, a_arr))

        return self.variable_angle_winding(
            radius, length, angle_func, n_layers, n_steps_per_layer)

    # ------------------------------------------------------------------ #
    #  4. Fermat spiral layup (flat / planar parts)
    # ------------------------------------------------------------------ #
    def fermat_spiral_layup(
        self,
        contour_points: List[Tuple[float, float]],
        layer_height: float,
        n_layers: int,
        n_arms: int = 2,
        density: int = 300,
    ) -> List[Waypoint]:
        """Generate Fermat spiral space-filling paths for flat/planar parts.

        The spiral is computed in the XY plane (Y = 0 in model coords for the
        flat part lying on the print bed).  The A-axis stays at 0° (no rotation).

        Parameters
        ----------
        contour_points : list[(x, y)]
            Bounding contour of the flat part in XY plane.
        layer_height : float
            Layer height in Z (mm).
        n_layers : int
            Number of layers to stack.
        n_arms : int
            Number of interleaved spiral arms (1 = single Fermat, 2 = dual).
        density : int
            Points per arm.

        Returns
        -------
        list[Waypoint]
            Waypoints with a_deg=0.0 (flat layup, no rotation).
        """
        contour_arr = np.array(contour_points, dtype=float)
        x_min, x_max = contour_arr[:, 0].min(), contour_arr[:, 0].max()
        y_min, y_max = contour_arr[:, 1].min(), contour_arr[:, 1].max()

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        r_max = sqrt((x_max - cx)**2 + (y_max - cy)**2)

        waypoints = []

        for layer_idx in range(n_layers):
            z = (layer_idx + 1) * layer_height

            for arm in range(n_arms):
                arm_offset = (arm * 2.0 * pi) / n_arms

                # Fermat spiral parametric: r = sqrt(t), theta = c*t + offset
                # We use t from 0 → 1, mapping r from 0 → r_max
                for i in range(density + 1):
                    t = i / density  # 0 → 1
                    r = sqrt(t) * r_max
                    # Angular density proportional to 1/r to keep arc-length uniform
                    # theta = k * t + arm_offset
                    # k chosen so outer edge has reasonable spacing
                    k = 4.0 * pi  # 2 full revolutions over the spiral
                    theta = k * t + arm_offset

                    x = cx + r * cos(theta)
                    y = cy + r * sin(theta)

                    # Keep within bounding box (simple clamp)
                    x = max(x_min, min(x_max, x))
                    y = max(y_min, min(y_max, y))

                    waypoints.append((float(x), float(y), float(z), 0.0))

        return waypoints

    # ------------------------------------------------------------------ #
    #  5. Mixed transition path
    # ------------------------------------------------------------------ #
    def mixed_transition_path(
        self,
        winding_segment: List[Waypoint],
        layup_segment: List[Waypoint],
        transition_length: int = 20,
    ) -> List[Waypoint]:
        """Smoothly transition between a winding segment and a flat layup.

        The transition interpolates the A-axis angle from the last winding
        waypoint to 0° over ``transition_length`` waypoints, then appends
        the flat layup segment.

        Parameters
        ----------
        winding_segment : list[Waypoint]
            Winding waypoints ending with some A angle.
        layup_segment : list[Waypoint]
            Flat layup waypoints starting at A=0.
        transition_length : int
            Number of intermediate waypoints for the A-angle ramp.

        Returns
        -------
        list[Waypoint]
            Combined path: winding + smooth transition + flat layup.
        """
        if not winding_segment:
            return list(layup_segment)
        if not layup_segment:
            return list(winding_segment)

        # Last winding waypoint
        x_w, y_w, z_w, a_w = winding_segment[-1]
        # First layup waypoint
        x_l, y_l, z_l, a_l = layup_segment[0]

        transition: List[Waypoint] = []
        for i in range(1, transition_length + 1):
            t = i / transition_length  # 0 → 1
            # Smooth cubic interpolation
            t_smooth = t * t * (3.0 - 2.0 * t)

            x_t = x_w + t_smooth * (x_l - x_w)
            y_t = y_w + t_smooth * (y_l - y_w)
            z_t = z_w + t_smooth * (z_l - z_w)
            a_t = a_w + t_smooth * (a_l - a_w)

            transition.append((float(x_t), float(y_t), float(z_t), float(a_t)))

        return list(winding_segment) + transition + list(layup_segment)

    # ------------------------------------------------------------------ #
    #  6. Layer-fill paths  (FFF infill for solid cylinder cross-sections)
    # ------------------------------------------------------------------ #
    def layer_fill_paths(
        self,
        radius: float,
        length: float,
        layer_height: float = 0.2,
        extrusion_width: float = 0.4,
        n_walls: int = 2,
        infill_density: float = 0.25,
    ) -> List[Waypoint]:
        """Generate FFF layer-by-layer fill paths for a solid/hollow cylinder body.

        Uses the XY+A kinematics to fill each cross-section with concentric
        rings: for every X position (layer along the cylinder axis) the A-axis
        rotates 360° at each ring radius, stepping from the outer radius
        inward.

        Physical meaning (XYZA0 coordinates):
          - x    : position along cylinder axis (0 → length)
          - z_model : radial distance from A-axis (outer_radius → 0)
          - a_deg  : A-axis rotation (0 → 360 per ring)
          - y_model : 0.0 throughout (nozzle in the horizontal plane of the axis)

        Parameters
        ----------
        radius : float
            Cylinder outer radius (mm).
        length : float
            Cylinder axial length (mm).
        layer_height : float
            Step along X between adjacent cross-sections (mm).
        extrusion_width : float
            Extrusion line width = radial spacing between rings (mm).
        n_walls : int
            Number of solid outer perimeter rings (always fully dense).
        infill_density : float
            0.0–1.0.  Inner fill ring spacing = extrusion_width / infill_density.
            0.0 → walls only; 1.0 → fully dense concentric rings.

        Returns
        -------
        list[Waypoint]
        """
        waypoints: List[Waypoint] = []
        n_x = max(1, int(round(length / layer_height)))

        for li in range(n_x):
            x_pos = (li + 0.5) * layer_height   # centre of this slice

            # ---- outer walls ----
            for w in range(n_walls):
                r = radius - (w + 0.5) * extrusion_width
                if r <= 0:
                    break
                # points around the ring – roughly 1 pt per extrusion_width arc
                n_pts = max(12, int(2.0 * pi * r / (extrusion_width * 0.6)))
                if (li + w) % 2 == 0:
                    a_vals = np.linspace(0.0, 360.0, n_pts, endpoint=False)
                else:
                    a_vals = np.linspace(360.0, 0.0, n_pts, endpoint=False)
                for a in a_vals:
                    waypoints.append((x_pos, 0.0, r, float(a)))
                # close the ring
                waypoints.append((x_pos, 0.0, r, float(a_vals[0])))

            if infill_density <= 0.0:
                continue

            # ---- infill: concentric rings from first inner to centre ----
            r_start = radius - n_walls * extrusion_width - extrusion_width * 0.5
            if r_start <= extrusion_width * 0.5:
                continue

            # Area-based ring_spacing:
            #   total_area = π·R²
            #   wall_area  = Σ 2π·r_w·w   (already printed above)
            #   target infill area = infill_density·total_area - wall_area
            #   For N evenly-spaced rings from r_start down with spacing s:
            #     infill_area ≈ 2π·w·(r_start²/(2s))  (integral approximation)
            #   Solving for s:  s = π·w·r_start² / target_infill_area
            total_area = pi * radius ** 2
            wall_area = sum(
                2 * pi * (radius - (ww + 0.5) * extrusion_width) * extrusion_width
                for ww in range(n_walls)
                if radius - (ww + 0.5) * extrusion_width > 0
            )
            target_infill_area = max(0.0, infill_density * total_area - wall_area)
            if target_infill_area <= 0.0:
                continue
            # area-based spacing (never less than extrusion_width → fully dense)
            ring_spacing = max(
                extrusion_width,
                pi * extrusion_width * r_start ** 2 / target_infill_area,
            )
            r = r_start
            ring_idx = 0
            while r > extrusion_width * 0.5:
                n_pts = max(8, int(2.0 * pi * r / (extrusion_width * 0.8)))
                if (li + n_walls + ring_idx) % 2 == 0:
                    a_vals = np.linspace(0.0, 360.0, n_pts, endpoint=False)
                else:
                    a_vals = np.linspace(360.0, 0.0, n_pts, endpoint=False)
                for a in a_vals:
                    waypoints.append((x_pos, 0.0, r, float(a)))
                waypoints.append((x_pos, 0.0, r, float(a_vals[0])))
                r -= ring_spacing
                ring_idx += 1

        return waypoints

    # ------------------------------------------------------------------ #
    #  7. Rectilinear fill  (boustrophedon grid, XY+A kinematics)
    # ------------------------------------------------------------------ #
    def layer_rectilinear_paths(
        self,
        radius: float,
        length: float,
        r_inner: float = 0.0,
        layer_height: float = 0.2,
        extrusion_width: float = 0.4,
        n_walls: int = 2,
        infill_density: float = 0.25,
    ) -> List[Waypoint]:
        """Fill a cylinder cross-section with a rectilinear (grid) pattern.

        Each axial layer (fixed x_pos) is filled with parallel straight lines
        that span the full chord of the circle.  Adjacent layers rotate 90° so
        the two directions interleave into an orthogonal grid.

        For a hollow cylinder (r_inner > 0), each scan line is split into two
        segments that skip the hollow core:
          - If |d| >= r_inner: one full chord segment (line doesn't pass through hole)
          - If |d| < r_inner:  two segments [−outer_chord, −inner_chord] and
                               [+inner_chord, +outer_chord]

        The outer perimeter is still printed as n_walls concentric rings.
        Line density is based on the annular fill area π(r_fill² − r_inner²).

        XY+A parametrisation of a straight line at perpendicular distance d
        from the cylinder axis, in fill-direction phi:

            x_c = d·cos(phi) - t·sin(phi)
            y_c = d·sin(phi) + t·cos(phi)
            z_model = sqrt(x_c² + y_c²)         (radial distance from axis)
            a_deg   = atan2(y_c, x_c) × 180/π   (A-axis angle)

        t ∈ [−chord, +chord],  chord = sqrt(r_fill² − d²)

        Line spacing uses area-based formula on the annular fill region so
        that the requested infill_density is met accurately.
        """
        waypoints: List[Waypoint] = []
        n_x = max(1, int(round(length / layer_height)))
        r_fill = radius - n_walls * extrusion_width   # fill starts inside walls
        MIN_Z = extrusion_width * 0.5                 # avoid axis singularity
        # r_inner effective: clamp so it never exceeds r_fill
        r_hole = max(0.0, min(r_inner, r_fill - extrusion_width))

        # Area-based line spacing on the annular infill region:
        #   annular_fill_area = π(r_fill² − r_hole²)
        #   target = infill_density * total_area - wall_area
        #   spacing s: ew * annular_fill_area / s = target  →  s = ew * annular_area / target
        total_area = pi * (radius ** 2 - r_hole ** 2)
        wall_area = sum(
            2 * pi * (radius - (w + 0.5) * extrusion_width) * extrusion_width
            for w in range(n_walls)
            if radius - (w + 0.5) * extrusion_width > r_hole
        )
        target_infill_area = max(0.0, infill_density * total_area - wall_area)
        annular_fill_area = pi * max(0.0, r_fill ** 2 - r_hole ** 2)

        if target_infill_area > 0 and annular_fill_area > 0 and r_fill > MIN_Z:
            line_spacing = max(
                extrusion_width,
                extrusion_width * annular_fill_area / target_infill_area,
            )
        else:
            line_spacing = extrusion_width   # fully dense (or walls cover target)

        # Points per mm along each line – dense enough for smooth IK
        pts_per_mm = max(2, int(2.0 / extrusion_width))

        def _append_segment(x_pos, t_a, t_b, d, phi, seg_idx):
            """Emit one straight scan-line segment from t=t_a to t=t_b."""
            n_pts = max(4, int(abs(t_b - t_a) * pts_per_mm))
            for t in np.linspace(t_a, t_b, n_pts):
                x_c = d * cos(phi) - t * sin(phi)
                y_c = d * sin(phi) + t * cos(phi)
                z_model = sqrt(x_c * x_c + y_c * y_c)
                if z_model < MIN_Z:
                    z_model = MIN_Z
                    a_deg_v = 0.0
                else:
                    a_deg_v = degrees(atan2(y_c, x_c))
                waypoints.append((x_pos, 0.0, float(z_model), float(a_deg_v)))

        for li in range(n_x):
            x_pos = (li + 0.5) * layer_height

            # ── outer perimeter walls (concentric rings) ──────────────────
            for w in range(n_walls):
                r = radius - (w + 0.5) * extrusion_width
                if r <= max(r_hole, 0.0):
                    break
                n_pts = max(12, int(2.0 * pi * r / (extrusion_width * 0.6)))
                a_vals = np.linspace(
                    0.0, 360.0, n_pts, endpoint=False
                ) if (li + w) % 2 == 0 else np.linspace(
                    360.0, 0.0, n_pts, endpoint=False
                )
                for a in a_vals:
                    waypoints.append((x_pos, 0.0, r, float(a)))
                waypoints.append((x_pos, 0.0, r, float(a_vals[0])))

            if infill_density <= 0.0 or r_fill <= MIN_Z:
                continue

            # ── rectilinear infill ─────────────────────────────────────────
            # Fill direction: 0° for even layers, 90° for odd layers
            phi = 0.0 if li % 2 == 0 else pi / 2.0

            # Perpendicular-distance offsets of the scan lines
            ds = np.arange(-r_fill + line_spacing / 2.0, r_fill, line_spacing)

            global_seg_idx = 0  # for boustrophedon direction alternation
            for d in ds:
                outer_chord = sqrt(max(0.0, r_fill ** 2 - d * d))
                if outer_chord < MIN_Z:
                    continue

                inner_chord = sqrt(max(0.0, r_hole ** 2 - d * d)) if r_hole > MIN_Z else 0.0

                if inner_chord < MIN_Z:
                    # Line doesn't pass through the hole — single full segment
                    t_start = -outer_chord if global_seg_idx % 2 == 0 else outer_chord
                    t_end   =  outer_chord if global_seg_idx % 2 == 0 else -outer_chord
                    _append_segment(x_pos, t_start, t_end, d, phi, global_seg_idx)
                    global_seg_idx += 1
                else:
                    # Line passes through the hollow core — two segments, skip [−inner, +inner]
                    # Direction: alternate the pair direction for boustrophedon continuity
                    if global_seg_idx % 2 == 0:
                        # left segment: −outer → −inner_chord, then right: +inner → +outer
                        _append_segment(x_pos, -outer_chord, -inner_chord, d, phi, 0)
                        _append_segment(x_pos, +inner_chord, +outer_chord, d, phi, 1)
                    else:
                        # reversed: +outer → +inner, then −inner → −outer
                        _append_segment(x_pos, +outer_chord, +inner_chord, d, phi, 0)
                        _append_segment(x_pos, -inner_chord, -outer_chord, d, phi, 1)
                    global_seg_idx += 1

        return waypoints

    # ------------------------------------------------------------------ #
    #  8. Combined print: rectilinear fill + CF winding
    # ------------------------------------------------------------------ #
    def combined_print_paths(
        self,
        radius: float,
        length: float,
        r_inner: float = 0.0,
        winding_angle_deg: float = 45.0,
        n_cf_layers: int = 4,
        layer_height: float = 0.2,
        extrusion_width: float = 0.4,
        n_walls: int = 2,
        infill_density: float = 0.25,
    ) -> List[Waypoint]:
        """Full cylinder print: base-material fill layers + CF winding on surface.

        Generates fill paths for the entire cylinder body first, then appends
        the continuous-fibre winding paths on the outer surface.  In the
        generated G-code the operator would switch to the CF extruder before
        the winding segment (handled by ``multi_path_to_gcode``).

        Parameters
        ----------
        radius, length : geometry (mm)
        r_inner : inner radius of hollow cylinder (0 = solid, mm)
        winding_angle_deg : CF winding angle (0°=axial, 90°=hoop)
        n_cf_layers : winding layers for fibre reinforcement
        layer_height, extrusion_width, n_walls, infill_density : fill params

        Returns
        -------
        list[Waypoint]  – fill_waypoints followed by cf_waypoints
        """
        fill_wps = self.layer_rectilinear_paths(
            radius=radius,
            length=length,
            r_inner=r_inner,
            layer_height=layer_height,
            extrusion_width=extrusion_width,
            n_walls=n_walls,
            infill_density=infill_density,
        )
        cf_wps = self.constant_angle_winding(
            radius=radius,
            length=length,
            angle_deg=winding_angle_deg,
            n_layers=n_cf_layers,
            fiber_width=extrusion_width,
        )
        return fill_wps + cf_wps

    # ------------------------------------------------------------------ #
    #  Utility: unwrap A angles for G-code continuity
    # ------------------------------------------------------------------ #
    @staticmethod
    def unwrap_a_angles(waypoints: List[Waypoint]) -> List[Waypoint]:
        """Unwrap the A-axis angles to prevent 360° jumps in G-code.

        Parameters
        ----------
        waypoints : list[Waypoint]

        Returns
        -------
        list[Waypoint]
            Same waypoints with A angles unwrapped for continuity.
        """
        if not waypoints:
            return waypoints

        a_vals = np.array([wp[3] for wp in waypoints])
        a_unwrapped = np.degrees(np.unwrap(np.radians(a_vals)))

        return [
            (wp[0], wp[1], wp[2], float(a_unwrapped[i]))
            for i, wp in enumerate(waypoints)
        ]
