#!/usr/bin/env python3
"""Tests for the XYZA0 path planning + G-code backend.

Covers:
  1. Constant-angle winding → G-code
  2. Variable-angle winding with sin-wave angle variation → G-code
  3. Stress-aligned winding with a mock stress field → G-code
  4. Fermat spiral layup (flat part, A=0) → G-code
  5. Mixed transition path → G-code

Run from the project root:
    cd /home/liuyue/Research/连续碳纤维3D打印/cf-path-planner
    python tests/test_xyza_backend.py
"""

import sys
import os
from math import sin, pi

# ── path setup ────────────────────────────────────────────────────────────────
# Project root (cf-path-planner/)
_PROJ_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

# FullControl root (contains 'fullcontrol/' and 'lab/' packages)
_FC_ROOT = os.path.normpath(os.path.join(_PROJ_ROOT, '..', 'fullcontrol'))
if _FC_ROOT not in sys.path:
    sys.path.insert(0, _FC_ROOT)

from cfpp.surface.planner_v2 import XYAPathPlanner
from cfpp.gcode.xyza_backend import waypoints_to_gcode, multi_path_to_gcode, PrinterConfig

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── shared printer config ─────────────────────────────────────────────────────
CONFIG = PrinterConfig(
    a_offset_y=0.0,
    a_offset_z=50.0,
    print_speed=600.0,
    travel_speed=3000.0,
    nozzle_temp=240,
    bed_temp=80,
    extrusion_width=1.0,
    extrusion_height=0.3,
)

RADIUS = 15.0    # mm
LENGTH = 100.0   # mm

planner = XYAPathPlanner(a_offset_y=CONFIG.a_offset_y, a_offset_z=CONFIG.a_offset_z)


# ── helpers ───────────────────────────────────────────────────────────────────

def save_and_report(gcode_str: str, filename: str, test_name: str):
    """Save G-code to output dir and print a brief report."""
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w') as fh:
        fh.write(gcode_str)
    lines = gcode_str.strip().split('\n')
    motion_lines = [l for l in lines if l.startswith('G0') or l.startswith('G1')]
    print(f"  [PASS] {test_name}")
    print(f"         Total lines: {len(lines):,}  |  Motion lines: {len(motion_lines):,}")
    print(f"         Saved → {path}")
    # Show first 3 and last 3 motion lines for visual check
    sample = motion_lines[:3] + ['  ...'] + motion_lines[-3:] if len(motion_lines) > 6 else motion_lines
    for l in sample:
        print(f"         {l}")
    print()


# ── Test 1: Constant-angle winding ────────────────────────────────────────────

def test_constant_angle_winding():
    print("=" * 60)
    print("Test 1: Constant-angle winding (±45°, 2 layers)")
    print("=" * 60)

    waypoints = planner.constant_angle_winding(
        radius=RADIUS,
        length=LENGTH,
        angle_deg=45.0,
        n_layers=2,
        points_per_pass=36,   # coarser for testing
    )

    assert len(waypoints) > 0, "No waypoints generated"

    # Validate waypoint structure
    for i, wp in enumerate(waypoints[:5]):
        x, y, z, a = wp
        assert isinstance(x, float) and isinstance(y, float)
        assert isinstance(z, float) and isinstance(a, float)
        assert 0.0 <= x <= LENGTH, f"x={x} out of bounds"
        assert z == RADIUS, f"z_model should equal radius, got {z}"
        assert y == 0.0, f"y_model should be 0, got {y}"

    print(f"  Waypoints: {len(waypoints)}")
    print(f"  First waypoint: {waypoints[0]}")
    print(f"  Last waypoint:  {waypoints[-1]}")

    gcode_str = waypoints_to_gcode(waypoints, CONFIG)
    assert len(gcode_str) > 100, "G-code too short"
    assert 'G1' in gcode_str or 'G0' in gcode_str, "No motion commands found"
    assert 'A' in gcode_str, "No A-axis commands found"

    save_and_report(gcode_str, 'test1_constant_angle_45deg.gcode', 'Constant-angle winding ±45°')


# ── Test 2: Variable-angle winding with sin-wave angle variation ──────────────

def test_variable_angle_winding():
    print("=" * 60)
    print("Test 2: Variable-angle winding (sin-wave angle variation)")
    print("=" * 60)

    def sin_angle_func(x: float, layer: int) -> float:
        """Angle varies sinusoidally from 20° to 70° along the cylinder."""
        t = x / LENGTH  # normalised position 0→1
        base_angle = 45.0
        amplitude = 25.0
        layer_offset = layer * 10.0  # phase shift per layer
        return base_angle + amplitude * sin(2.0 * pi * t + layer_offset * pi / 180.0)

    waypoints = planner.variable_angle_winding(
        radius=RADIUS,
        length=LENGTH,
        angle_func=sin_angle_func,
        n_layers=2,
        n_steps=200,
    )

    assert len(waypoints) > 0, "No waypoints generated"

    # Check A angle is monotonically increasing (integration accumulates)
    a_vals = [wp[3] for wp in waypoints]
    # Not strictly monotone (layer direction alternates), just check range
    assert max(a_vals) > min(a_vals), "A angles all equal — no rotation"

    print(f"  Waypoints: {len(waypoints)}")
    print(f"  A range: [{min(a_vals):.1f}°, {max(a_vals):.1f}°]")

    gcode_str = waypoints_to_gcode(waypoints, CONFIG)
    assert 'A' in gcode_str, "No A-axis commands found"

    save_and_report(gcode_str, 'test2_variable_angle_sinwave.gcode', 'Variable-angle winding (sin-wave)')


# ── Test 3: Stress-aligned winding with mock stress field ────────────────────

def test_stress_aligned_winding():
    print("=" * 60)
    print("Test 3: Stress-aligned winding (mock stress field dict)")
    print("=" * 60)

    # Mock stress field: a dict mapping x-position to winding angle
    # Simulates a bending load: near ends need high angle (near 60°),
    # mid-span needs low angle (near 30°) to resist bending.
    import numpy as np

    x_positions = np.linspace(0, LENGTH, 10)
    angles = 30.0 + 30.0 * (1.0 - np.abs(x_positions / (LENGTH / 2.0) - 1.0))
    # values range from 30° (ends) to 60° (mid-span)
    mock_stress_field = {float(x): float(a) for x, a in zip(x_positions, angles)}

    waypoints = planner.stress_aligned_winding(
        radius=RADIUS,
        length=LENGTH,
        stress_field=mock_stress_field,
        n_layers=2,
        n_x_samples=10,
        n_steps_per_layer=200,
    )

    assert len(waypoints) > 0, "No waypoints generated"

    print(f"  Waypoints: {len(waypoints)}")
    print(f"  Sample: {waypoints[0]}, ..., {waypoints[len(waypoints)//2]}")

    gcode_str = waypoints_to_gcode(waypoints, CONFIG)
    assert 'A' in gcode_str, "No A-axis commands found"

    save_and_report(gcode_str, 'test3_stress_aligned_winding.gcode', 'Stress-aligned winding (mock field)')


# ── Test 3b: Stress-aligned with object-based stress field ───────────────────

def test_stress_aligned_winding_object():
    print("=" * 60)
    print("Test 3b: Stress-aligned winding (object-based mock field)")
    print("=" * 60)

    import numpy as np

    class MockStressField:
        """Object-based mock: returns principal stress direction at a point."""

        def query(self, point):
            """Return a stress direction that varies with x position."""
            x = point[0]
            # Stress direction tilted between axial and circumferential
            t = x / LENGTH
            angle_rad = (0.3 + 0.4 * t) * (3.14159 / 2.0)  # 0.3→0.7 * (pi/2)
            direction = np.array([cos(angle_rad), sin(angle_rad), 0.0])
            value = 50.0 + 100.0 * t  # MPa, increasing along x
            return direction, value

    mock_field = MockStressField()

    waypoints = planner.stress_aligned_winding(
        radius=RADIUS,
        length=LENGTH,
        stress_field=mock_field,
        n_layers=1,
        n_x_samples=20,
        n_steps_per_layer=150,
    )

    assert len(waypoints) > 0, "No waypoints generated"
    print(f"  Waypoints: {len(waypoints)}")

    gcode_str = waypoints_to_gcode(waypoints, CONFIG)
    assert 'A' in gcode_str

    save_and_report(gcode_str, 'test3b_stress_object_field.gcode', 'Stress-aligned winding (object field)')


# ── Test 4: Fermat spiral layup ───────────────────────────────────────────────

def test_fermat_spiral_layup():
    print("=" * 60)
    print("Test 4: Fermat spiral layup (flat part, A=0)")
    print("=" * 60)

    # Rectangular contour for a 60mm × 40mm flat part
    contour = [(0.0, 0.0), (60.0, 0.0), (60.0, 40.0), (0.0, 40.0)]

    waypoints = planner.fermat_spiral_layup(
        contour_points=contour,
        layer_height=0.3,
        n_layers=3,
        n_arms=2,
        density=100,
    )

    assert len(waypoints) > 0, "No waypoints generated"

    # All A angles should be 0 for flat layup
    a_vals = [wp[3] for wp in waypoints]
    assert all(a == 0.0 for a in a_vals), f"Expected all A=0, got {set(a_vals)}"

    # Z should increment by layer
    z_vals = sorted(set(wp[2] for wp in waypoints))
    assert len(z_vals) == 3, f"Expected 3 z-levels, got {len(z_vals)}: {z_vals}"

    print(f"  Waypoints: {len(waypoints)}")
    print(f"  Z levels: {z_vals}")
    print(f"  A angles (all should be 0): {set(a_vals)}")

    gcode_str = waypoints_to_gcode(waypoints, CONFIG)
    # For flat layup, A should NOT appear in motion lines (since A stays 0)
    # The FullControl backend only emits A when it changes from 0
    print(f"  Note: A=0 throughout, so A token may not appear in G-code (correct)")

    save_and_report(gcode_str, 'test4_fermat_spiral_layup.gcode', 'Fermat spiral layup (flat, A=0)')


# ── Test 5: Mixed transition path ─────────────────────────────────────────────

def test_mixed_transition_path():
    print("=" * 60)
    print("Test 5: Mixed transition (winding → flat layup)")
    print("=" * 60)

    # Winding segment: 45° helical on the cylinder end
    winding_wp = planner.constant_angle_winding(
        radius=RADIUS,
        length=LENGTH,
        angle_deg=45.0,
        n_layers=1,
        points_per_pass=24,
    )

    # Flat layup segment: Fermat spiral on a small flange at x=0, z=0
    flat_wp = planner.fermat_spiral_layup(
        contour_points=[(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)],
        layer_height=0.3,
        n_layers=1,
        density=50,
    )

    combined = planner.mixed_transition_path(
        winding_wp, flat_wp, transition_length=15)

    assert len(combined) >= len(winding_wp) + len(flat_wp), \
        f"Combined path shorter than sum of parts: {len(combined)}"

    # Check transition smoothness: A angle should change monotonically in the transition
    # region (from last winding A to flat A=0)
    last_winding_a = winding_wp[-1][3]
    first_flat_a = flat_wp[0][3]
    transition_start = len(winding_wp)
    transition_end = transition_start + 15

    transition_a = [combined[i][3] for i in range(transition_start, transition_end)]
    # A should move from last_winding_a towards first_flat_a (not necessarily monotone
    # but should be interpolating)
    mid_a = combined[transition_start + 7][3]
    expected_mid = last_winding_a + 0.5 * (first_flat_a - last_winding_a)
    assert abs(mid_a - expected_mid) < abs(last_winding_a - first_flat_a) * 0.6, \
        f"Transition mid-point A={mid_a:.1f} not near expected {expected_mid:.1f}"

    print(f"  Winding segment:  {len(winding_wp)} waypoints")
    print(f"  Flat segment:     {len(flat_wp)} waypoints")
    print(f"  Combined:         {len(combined)} waypoints")
    print(f"  Transition A: {last_winding_a:.1f}° → {mid_a:.1f}° → {first_flat_a:.1f}°")

    gcode_str = waypoints_to_gcode(combined, CONFIG)

    save_and_report(gcode_str, 'test5_mixed_transition.gcode', 'Mixed transition (winding → flat layup)')


# ── Test 6: Multi-path G-code (multiple independent fiber segments) ───────────

def test_multi_path_gcode():
    print("=" * 60)
    print("Test 6: Multi-path G-code (3 independent fiber segments)")
    print("=" * 60)

    paths = []
    for angle in [30.0, 45.0, 60.0]:
        wp = planner.constant_angle_winding(
            radius=RADIUS, length=LENGTH, angle_deg=angle,
            n_layers=1, points_per_pass=24)
        paths.append(wp)

    gcode_str = multi_path_to_gcode(paths, CONFIG)
    assert len(gcode_str) > 100, "G-code too short"

    # Should contain extruder on/off transitions
    assert 'G0' in gcode_str, "No travel moves found"
    assert 'G1' in gcode_str, "No print moves found"

    print(f"  Paths: {len(paths)}, total waypoints: {sum(len(p) for p in paths)}")

    save_and_report(gcode_str, 'test6_multi_path.gcode', 'Multi-path G-code (3 fiber segments)')


# ── Test 7: Unwrap A angles utility ──────────────────────────────────────────

def test_unwrap_a():
    print("=" * 60)
    print("Test 7: A-angle unwrapping utility")
    print("=" * 60)

    # Construct waypoints where A angle wraps around 360°
    waypoints_wrap = [
        (0.0, 0.0, 15.0, 350.0),
        (1.0, 0.0, 15.0, 360.0),
        (2.0, 0.0, 15.0, 370.0),   # equivalent to 10° but not wrapped
        (3.0, 0.0, 15.0, 10.0),    # this would be a 360° jump if not unwrapped
        (4.0, 0.0, 15.0, 20.0),
    ]

    unwrapped = XYAPathPlanner.unwrap_a_angles(waypoints_wrap)
    a_vals = [wp[3] for wp in unwrapped]

    # After unwrapping, differences should be small
    import numpy as np
    diffs = [abs(a_vals[i+1] - a_vals[i]) for i in range(len(a_vals)-1)]
    assert max(diffs) < 180.0, f"Large jump after unwrapping: {max(diffs):.1f}°"

    print(f"  Original A values:  {[wp[3] for wp in waypoints_wrap]}")
    print(f"  Unwrapped A values: {[round(a, 1) for a in a_vals]}")
    print(f"  Max jump after unwrap: {max(diffs):.1f}°")
    print(f"  [PASS] A-angle unwrapping")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("XYZA Backend Tests")
    print("Output directory:", OUTPUT_DIR)
    print("=" * 60 + "\n")

    tests = [
        test_constant_angle_winding,
        test_variable_angle_winding,
        test_stress_aligned_winding,
        test_stress_aligned_winding_object,
        test_fermat_spiral_layup,
        test_mixed_transition_path,
        test_multi_path_gcode,
        test_unwrap_a,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as exc:
            print(f"  [FAIL] {test_fn.__name__}: {exc}\n")
            failed += 1
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {test_fn.__name__}: {exc}")
            traceback.print_exc()
            print()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    # Make sure XYAPathPlanner is importable for test_unwrap_a
    from cfpp.surface.planner_v2 import XYAPathPlanner
    main()
