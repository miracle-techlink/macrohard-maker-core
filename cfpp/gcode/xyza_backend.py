"""XYZA0 G-code backend for cf-path-planner.

Converts (x, y_model, z_model, a_deg) waypoints from planner_v2 into
G-code via the FullControl XYZA0 module.

The FullControl XYZA0 module lives at:
    /home/liuyue/Research/连续碳纤维3D打印/fullcontrol/

Import convention::

    from cfpp.gcode.xyza_backend import waypoints_to_gcode, PrinterConfig

Usage example::

    from cfpp.surface.planner_v2 import XYAPathPlanner
    from cfpp.gcode.xyza_backend import waypoints_to_gcode, PrinterConfig

    planner = XYAPathPlanner(a_offset_y=0.0, a_offset_z=50.0)
    waypoints = planner.constant_angle_winding(radius=15, length=100, angle_deg=45, n_layers=2)

    config = PrinterConfig(a_offset_z=50.0)
    gcode_str = waypoints_to_gcode(waypoints, config)
    print(gcode_str[:500])
"""

import sys
import os
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

# Ensure the FullControl library is on the Python path.
# The path is relative to this file: ../../../../fullcontrol/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FULLCONTROL_ROOT = os.path.normpath(
    os.path.join(_THIS_DIR, '..', '..', '..', 'fullcontrol'))
if _FULLCONTROL_ROOT not in sys.path:
    sys.path.insert(0, _FULLCONTROL_ROOT)

# Also ensure the fullcontrol package root is importable (contains fullcontrol/ and lab/)
if _FULLCONTROL_ROOT not in sys.path:
    sys.path.insert(0, _FULLCONTROL_ROOT)

# Import the FullControl XYZA0 API
try:
    import lab.fullcontrol.xyza_axis as fca
except ImportError as exc:
    raise ImportError(
        f"Could not import FullControl XYZA0 module from {_FULLCONTROL_ROOT}. "
        f"Original error: {exc}"
    ) from exc

# Type alias
Waypoint = Tuple[float, float, float, float]  # (x, y_model, z_model, a_deg)


@dataclass
class PrinterConfig:
    """Printer configuration for G-code generation.

    Parameters
    ----------
    a_offset_y : float
        Y-offset from part origin to A-axis centre of rotation (default 0.0).
    a_offset_z : float
        Z-offset from part origin to A-axis centre of rotation (default 50.0).
        This is the height of the rotary chuck above the bed surface.
    print_speed : float
        Printing feed rate in mm/min (default 600).
    travel_speed : float
        Travel / rapid feed rate in mm/min (default 3000).
    nozzle_temp : int
        Nozzle temperature in °C (default 240).
    bed_temp : int
        Bed temperature in °C (default 80).
    extrusion_width : float
        Extrusion width in mm (default 1.0).
    extrusion_height : float
        Extrusion / layer height in mm (default 0.3).
    save_as : str | None
        If set, G-code is saved to a file with this prefix + timestamp.
        If None (default), G-code string is returned directly.
    """
    a_offset_y: float = 0.0
    a_offset_z: float = 50.0
    print_speed: float = 600.0
    travel_speed: float = 3000.0
    nozzle_temp: int = 240
    bed_temp: int = 80
    extrusion_width: float = 1.0
    extrusion_height: float = 0.3
    save_as: Optional[str] = None


def waypoints_to_steps(
    waypoints: List[Waypoint],
    with_extrusion: bool = True,
) -> list:
    """Convert a list of waypoints to FullControl XYZA0 steps.

    Parameters
    ----------
    waypoints : list[Waypoint]
        List of (x, y_model, z_model, a_deg) tuples.
    with_extrusion : bool
        If True, wrap the path with Extruder on/off controls so that the
        first move travels (no extrusion) and subsequent moves extrude.

    Returns
    -------
    list
        List of ``fca.Point`` (and optionally ``fca.Extruder``) objects
        ready to pass to ``fca.transform``.
    """
    if not waypoints:
        return []

    steps = []

    if with_extrusion:
        steps.append(fca.Extruder(on=False))

    for i, (x, y_model, z_model, a_deg) in enumerate(waypoints):
        steps.append(fca.Point(x=float(x), y=float(y_model),
                               z=float(z_model), a=float(a_deg)))
        if with_extrusion and i == 0:
            # Travel to first point complete; start extruding from point 1
            steps.append(fca.Extruder(on=True))

    return steps


def waypoints_to_gcode(
    waypoints: List[Waypoint],
    config: Optional[PrinterConfig] = None,
) -> str:
    """Convert waypoints to a G-code string using the FullControl XYZA0 backend.

    Parameters
    ----------
    waypoints : list[Waypoint]
        List of (x, y_model, z_model, a_deg) tuples produced by XYAPathPlanner.
    config : PrinterConfig | None
        Printer configuration.  Defaults to PrinterConfig() if None.

    Returns
    -------
    str
        Complete G-code string (or empty string if waypoints is empty).
    """
    if not waypoints:
        return ""

    if config is None:
        config = PrinterConfig()

    steps = waypoints_to_steps(waypoints, with_extrusion=True)

    controls = fca.GcodeControls(
        a_offset_y=config.a_offset_y,
        a_offset_z=config.a_offset_z,
        initialization_data={
            'print_speed': config.print_speed,
            'travel_speed': config.travel_speed,
            'nozzle_temp': config.nozzle_temp,
            'bed_temp': config.bed_temp,
            'extrusion_width': config.extrusion_width,
            'extrusion_height': config.extrusion_height,
        },
        save_as=config.save_as,
    )

    result = fca.transform(steps, 'gcode', controls)

    # fca.transform returns None when save_as is set (writes to file).
    # Return empty string in that case so callers always get a str.
    if result is None:
        return ""
    return result


def multi_path_to_gcode(
    path_list: List[List[Waypoint]],
    config: Optional[PrinterConfig] = None,
) -> str:
    """Convert multiple separate winding paths to a single G-code string.

    Each path in path_list is treated as an independent fiber segment:
    the machine travels (no extrusion) between paths.

    Parameters
    ----------
    path_list : list[list[Waypoint]]
        List of waypoint lists, each representing one fiber path.
    config : PrinterConfig | None

    Returns
    -------
    str
        Complete G-code string.
    """
    if not path_list:
        return ""

    if config is None:
        config = PrinterConfig()

    all_steps = []

    for path_idx, path in enumerate(path_list):
        if not path:
            continue
        # Travel move to start of this path (extrusion off)
        all_steps.append(fca.Extruder(on=False))
        for i, (x, y_model, z_model, a_deg) in enumerate(path):
            all_steps.append(fca.Point(x=float(x), y=float(y_model),
                                       z=float(z_model), a=float(a_deg)))
            if i == 0:
                all_steps.append(fca.Extruder(on=True))

    controls = fca.GcodeControls(
        a_offset_y=config.a_offset_y,
        a_offset_z=config.a_offset_z,
        initialization_data={
            'print_speed': config.print_speed,
            'travel_speed': config.travel_speed,
            'nozzle_temp': config.nozzle_temp,
            'bed_temp': config.bed_temp,
            'extrusion_width': config.extrusion_width,
            'extrusion_height': config.extrusion_height,
        },
        save_as=config.save_as,
    )

    result = fca.transform(all_steps, 'gcode', controls)
    if result is None:
        return ""
    return result
