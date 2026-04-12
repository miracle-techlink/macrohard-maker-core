#!/usr/bin/env python3
"""Generate principles document PDF for surface geodesic path planning."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import numpy as np
from datetime import date

# ── Font setup ──────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# ── Colours ─────────────────────────────────────────────────────────────
DARK_BLUE   = '#1a3a5c'
LIGHT_BLUE  = '#4a90d9'
ACCENT_BLUE = '#2c5f8a'
PASS_GREEN  = '#2e7d32'
FAIL_RED    = '#c62828'
ROW_LIGHT   = '#f0f4fa'
ROW_WHITE   = '#ffffff'
HEADER_BG   = '#1a3a5c'
HEADER_FG   = '#ffffff'
BORDER_CLR  = '#b0bec5'

OUT_PDF = '/home/liuyue/Research/连续碳纤维3D打印/docs/曲面路径规划原理.pdf'

# ── Helper: draw a styled table ─────────────────────────────────────────
def draw_table(ax, headers, rows, col_widths=None, font_size=9, title=None):
    """Draw a professional table on the given axes."""
    ax.axis('off')
    n_cols = len(headers)
    n_rows = len(rows)

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_widths,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.6)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER_CLR)
        cell.set_linewidth(0.5)
        if r == 0:  # header
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color=HEADER_FG, fontweight='bold', fontsize=font_size)
        else:
            bg = ROW_LIGHT if r % 2 == 1 else ROW_WHITE
            cell.set_facecolor(bg)
            txt = cell.get_text().get_text()
            if txt.strip() == 'PASS':
                cell.get_text().set_color(PASS_GREEN)
                cell.get_text().set_fontweight('bold')
            elif txt.strip() == 'FAIL':
                cell.get_text().set_color(FAIL_RED)
                cell.get_text().set_fontweight('bold')

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', color=DARK_BLUE, pad=12)


def add_page_number(fig, page_num, total=10):
    fig.text(0.5, 0.02, f'— {page_num} / {total} —', ha='center', fontsize=8, color='#888888')


def section_header(fig, text, y=0.92):
    fig.text(0.5, y, text, ha='center', fontsize=16, fontweight='bold', color=DARK_BLUE)


def sub_header(fig, text, y=0.86):
    fig.text(0.5, y, text, ha='center', fontsize=11, color=ACCENT_BLUE)


def text_block(fig, title, lines, y_start, title_size=11, line_size=9,
               title_color=DARK_BLUE, line_color='#333333', x_title=0.10,
               x_line=0.12, line_spacing=0.025, mono=False):
    """Draw a titled text block and return the y position after the last line."""
    fig.text(x_title, y_start, title, fontsize=title_size,
             fontweight='bold', color=title_color)
    y = y_start - 0.03
    ff = 'monospace' if mono else None
    for line in lines:
        kwargs = {}
        if ff:
            kwargs['fontfamily'] = ff
        fig.text(x_line, y, line, fontsize=line_size, color=line_color, **kwargs)
        y -= line_spacing
    return y


# ═══════════════════════════════════════════════════════════════════════
#                         BUILD PDF
# ═══════════════════════════════════════════════════════════════════════
with PdfPages(OUT_PDF) as pdf:
    TOTAL = 10

    # ── Page 1: Cover ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.patch.set_facecolor('white')

    # Decorative top bar
    fig.patches.append(mpatches.FancyBboxPatch(
        (0.05, 0.82), 0.9, 0.12, boxstyle="round,pad=0.02",
        facecolor=DARK_BLUE, edgecolor='none', transform=fig.transFigure,
        figure=fig))

    fig.text(0.5, 0.89, '曲面测地线路径规划原理', ha='center', fontsize=24,
             fontweight='bold', color='white')
    fig.text(0.5, 0.84, 'Surface Geodesic Path Planning', ha='center',
             fontsize=18, color='#a0c4ff')

    fig.text(0.5, 0.74, 'Surface Geodesic Path Planning for Continuous Fiber 3D Printing',
             ha='center', fontsize=13, color=ACCENT_BLUE, style='italic')

    fig.text(0.5, 0.64, '2026-03-31', ha='center', fontsize=12, color='#666666')

    # Description box
    box_y = 0.34
    fig.patches.append(mpatches.FancyBboxPatch(
        (0.12, box_y), 0.76, 0.22, boxstyle="round,pad=0.02",
        facecolor='#f5f8fc', edgecolor=BORDER_CLR, linewidth=1,
        transform=fig.transFigure, figure=fig))

    desc_lines = [
        'XZ+C',
    ]
    fig.text(0.5, box_y + 0.16,
             '面向 XZ+C 三轴连续碳纤维打印机的最优纤维铺放路径理论',
             ha='center', fontsize=12, color=DARK_BLUE, fontweight='bold')
    fig.text(0.5, box_y + 0.11,
             'Optimal Fiber Placement Path Theory for XZ+C 3-Axis',
             ha='center', fontsize=10, color=ACCENT_BLUE)
    fig.text(0.5, box_y + 0.07,
             'Continuous Carbon Fiber Printer',
             ha='center', fontsize=10, color=ACCENT_BLUE)

    sub_topics = [
        'Geodesic Theory  |  Lateral Force Analysis  |  Stress-Driven Optimization',
        'Cylindrical Coordinates  |  Machine Kinematics  |  General Surfaces',
    ]
    for i, line in enumerate(sub_topics):
        fig.text(0.5, box_y + 0.02 - i * 0.03, line,
                 ha='center', fontsize=9, color='#888888')

    fig.text(0.5, 0.22, 'CF Path Planner — Principles & Theory Document',
             ha='center', fontsize=10, color='#888888')

    add_page_number(fig, 1, TOTAL)
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 2: Table of Contents ───────────────────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    section_header(fig, '目录 / Table of Contents')

    toc = [
        ('Chapter 1', '机器运动学 / Machine Kinematics', '3'),
        ('Chapter 2', '柱坐标系下的路径描述 / Path in Cylindrical Coords', '4'),
        ('Chapter 3', '测地线理论 / Geodesic Theory', '5'),
        ('Chapter 4', '侧向沉积力分析 / Lateral Deposition Force', '6'),
        ('Chapter 5', '应力-测地线混合优化 / Stress-Geodesic Optimization', '7'),
        ('Chapter 6', '圆柱面上的具体分析 / Cylindrical Surface Analysis', '8'),
        ('Chapter 7', '一般曲面的推广 / Extension to General Surfaces', '9'),
        ('Chapter 8', '与现有方案对比 / Comparison with Existing Approaches', '10'),
    ]

    y_start = 0.78
    for i, (chap, desc, pg) in enumerate(toc):
        y = y_start - i * 0.055
        fig.text(0.12, y, chap, fontsize=12, color=DARK_BLUE, fontweight='bold')
        fig.text(0.25, y, desc, fontsize=11, color='#333333')
        fig.text(0.88, y, pg, fontsize=11, color='#888888', ha='right')
        line = mlines.Line2D([0.12, 0.88], [y - 0.008, y - 0.008],
                 color=BORDER_CLR, linewidth=0.3, transform=fig.transFigure)
        fig.add_artist(line)

    add_page_number(fig, 2, TOTAL)
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 3: Chapter 1 — Machine Kinematics ──────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    section_header(fig, 'Chapter 1: 机器运动学 / Machine Kinematics')
    sub_header(fig, 'XZ+C 三轴构型分析')

    ax1 = fig.add_axes([0.06, 0.62, 0.88, 0.20])
    draw_table(ax1,
        ['Axis', 'Motion', 'Physical Meaning', 'Range'],
        [
            ['X', 'Linear (horizontal)', 'Radial position (head to part)', '0-300 mm'],
            ['Z', 'Linear (vertical)', 'Height position', '0-350 mm'],
            ['C', 'Rotary (platform)', 'Angular position', '0-360° (cont.)'],
        ],
        col_widths=[0.10, 0.25, 0.40, 0.20],
        title='Axis Definitions')

    text_block(fig, 'Coordinate Mapping:', [
        'The machine operates in cylindrical coordinates (r, \u03b8, z):',
        '  X axis controls radial distance r',
        '  C axis controls angular position \u03b8',
        '  Z axis controls height z',
        '',
        'This is fundamentally different from Cartesian XYZ printers (Markforged).',
        'The rotary platform enables natural helical/spiral paths \u2014',
        'matching geodesics on cylindrical surfaces.',
    ], y_start=0.56, line_size=9)

    ax2 = fig.add_axes([0.06, 0.10, 0.88, 0.24])
    draw_table(ax2,
        ['Feature', 'XYZ Cartesian (Markforged)', 'XZ+C Rotary (This Machine)'],
        [
            ['Coordinate system', 'Cartesian (x,y,z)', 'Cylindrical (r,\u03b8,z)'],
            ['Natural path type', 'Planar 2D lines', 'Helical spirals on surface'],
            ['Fiber placement', 'Layer-by-layer flat', 'Surface-following'],
            ['Suitable for', 'Flat plates, low curvature', 'Cylindrical, tubular, revolute'],
        ],
        col_widths=[0.22, 0.38, 0.38],
        font_size=8,
        title='Machine Architecture Comparison')

    add_page_number(fig, 3, TOTAL)
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 4: Chapter 2 — Path in Cylindrical Coordinates ─────────────
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    section_header(fig, 'Chapter 2: 柱坐标系下的路径描述')
    sub_header(fig, 'From Machine Axes to Surface Paths')

    text_block(fig, 'Path on Cylindrical Surface:', [
        'A path on a cylindrical surface of radius R is described by:',
    ], y_start=0.80, line_size=9)

    fig.text(0.14, 0.74, 'P(s) = ( R\u00b7cos(\u03b8(s)),  R\u00b7sin(\u03b8(s)),  z(s) )',
             fontsize=11, color=DARK_BLUE, fontfamily='monospace', fontweight='bold')

    text_block(fig, 'Winding Angle Definition:', [
        'The winding angle \u03b1 (angle between fiber direction and circumferential):',
    ], y_start=0.70, line_size=9)

    eqs = [
        'tan(\u03b1) = (dz/ds) / (R\u00b7d\u03b8/ds) = Vz / (R\u00b7\u03c9c)',
        '',
        'where Vz = Z-axis velocity,  \u03c9c = C-axis angular velocity',
    ]
    y = 0.64
    for eq in eqs:
        fig.text(0.14, y, eq, fontsize=10, color=DARK_BLUE, fontfamily='monospace')
        y -= 0.025

    text_block(fig, 'Machine Coordinate Mapping:', [
        'X(t) = R              (constant for cylindrical surface)',
        'C(t) = \u03b8(t) = \u03c9c \u00b7 t',
        'Z(t) = z0 + Vz \u00b7 t',
        '',
        'For a helical path: constant Vz and \u03c9c \u2192 constant winding angle \u03b1.',
    ], y_start=0.55, line_size=9, mono=True)

    ax1 = fig.add_axes([0.06, 0.10, 0.88, 0.24])
    draw_table(ax1,
        ['Winding Angle \u03b1', 'Path Type', 'Z velocity', 'C velocity', 'Geodesic?'],
        [
            ['0\u00b0 (circumferential)', 'Hoop', '0', '\u03c9c', 'NO'],
            ['30\u00b0', 'Low-angle helix', 'Vz\u00b7sin(30\u00b0)', '\u03c9c\u00b7cos(30\u00b0)', 'YES'],
            ['45\u00b0', 'Balanced helix', 'Vz/\u221a2', '\u03c9c/\u221a2', 'YES'],
            ['90\u00b0 (axial)', 'Longitudinal', 'Vz', '0', 'YES'],
        ],
        col_widths=[0.22, 0.18, 0.20, 0.20, 0.12],
        font_size=8,
        title='Winding Angle Examples')

    add_page_number(fig, 4, TOTAL)
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 5: Chapter 3 — Geodesic Theory ─────────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    section_header(fig, 'Chapter 3: 测地线理论 / Geodesic Theory')
    sub_header(fig, 'Why Geodesics Minimize Lateral Force')

    text_block(fig, 'Definition:', [
        'A geodesic is the shortest path between two points on a surface.',
        'Equivalently, it is a curve whose geodesic curvature \u03bag = 0 everywhere.',
        '',
        'On a flat plane:  geodesic = straight line',
        'On a sphere:      geodesic = great circle arc',
        'On a cylinder:    geodesic = helix (any angle except pure circumferential)',
    ], y_start=0.80, line_size=9)

    text_block(fig, 'Geodesic Curvature on Surfaces of Revolution:', [
        'For a surface of revolution with radius r(z):',
    ], y_start=0.62, line_size=9)

    fig.text(0.14, 0.57, '\u03bag = (d\u03b1/ds) + cos(\u03b1)\u00b7sin(\u03b1) / r',
             fontsize=11, color=DARK_BLUE, fontfamily='monospace', fontweight='bold')

    text_block(fig, 'Cylinder Case (r = R = const, d\u03b1/ds = 0):', [
        '\u03bag = cos(\u03b1)\u00b7sin(\u03b1)/R - cos(\u03b1)\u00b7sin(\u03b1)/R = 0  \u2713',
        '',
        'Note: Pure hoop winding (\u03b1=0\u00b0) has \u03bag=0 in surface sense,',
        'but sits in UNSTABLE equilibrium \u2014 any perturbation causes axial slip.',
    ], y_start=0.52, line_size=9, mono=True)

    # Clairaut box
    box_y = 0.32
    fig.patches.append(mpatches.FancyBboxPatch(
        (0.08, box_y), 0.84, 0.10, boxstyle="round,pad=0.02",
        facecolor='#f5f8fc', edgecolor=ACCENT_BLUE, linewidth=1.5,
        transform=fig.transFigure, figure=fig))
    fig.text(0.5, box_y + 0.065, "Clairaut's Relation for Surfaces of Revolution:",
             ha='center', fontsize=11, fontweight='bold', color=DARK_BLUE)
    fig.text(0.5, box_y + 0.03, 'r(z) \u00b7 cos(\u03b1) = C   (constant along geodesic)',
             ha='center', fontsize=12, fontfamily='monospace', color=DARK_BLUE)

    text_block(fig, '', [
        'For cylinder (r = R = const): cos(\u03b1) = C/R = constant',
        '\u2192 The winding angle \u03b1 is constant along a geodesic on a cylinder.',
    ], y_start=0.28, line_size=9)

    ax1 = fig.add_axes([0.06, 0.04, 0.88, 0.18])
    draw_table(ax1,
        ['Surface', 'Geodesic Form', 'Clairaut Relation', 'Special Case'],
        [
            ['Cylinder (r=R)', 'Helix, constant \u03b1', 'R\u00b7cos(\u03b1) = C', 'Any \u03b1 works'],
            ['Cone (r=kz)', 'Variable-angle spiral', 'kz\u00b7cos(\u03b1) = C', '\u03b1 varies with z'],
            ['Sphere (r=Rsin\u03c6)', 'Great circle', 'Rsin\u03c6\u00b7cos(\u03b1)=C', '\u2014'],
            ['General revolution', 'Numerical integration', 'r(z)\u00b7cos(\u03b1) = C', '\u2014'],
        ],
        col_widths=[0.22, 0.26, 0.28, 0.20],
        font_size=8,
        title='Geodesic Properties on Common Surfaces')

    add_page_number(fig, 5, TOTAL)
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 6: Chapter 4 — Lateral Deposition Force ────────────────────
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    section_header(fig, 'Chapter 4: 侧向沉积力分析')
    sub_header(fig, 'The Critical Manufacturing Constraint')

    text_block(fig, 'Lateral Force on Fiber During Deposition:', [
    ], y_start=0.80, line_size=9)

    fig.text(0.14, 0.76, 'F_lateral = T \u00b7 \u03bag',
             fontsize=13, color=DARK_BLUE, fontfamily='monospace', fontweight='bold')

    text_block(fig, '', [
        'where:  T  = fiber tension (controlled by tension system, typically 1-4 N)',
        '        \u03bag = geodesic curvature of the path',
    ], y_start=0.73, line_size=9, mono=True)

    text_block(fig, 'Non-Slip Condition:', [
    ], y_start=0.66, line_size=9)

    fig.text(0.14, 0.62, 'F_lateral < \u03bc \u00b7 F_normal',
             fontsize=12, color=DARK_BLUE, fontfamily='monospace', fontweight='bold')
    fig.text(0.14, 0.59, 'T \u00b7 \u03bag < \u03bc \u00b7 Fn',
             fontsize=11, color=DARK_BLUE, fontfamily='monospace')
    fig.text(0.14, 0.56, '\u03bag_max = \u03bc \u00b7 Fn / T',
             fontsize=11, color=DARK_BLUE, fontfamily='monospace', fontweight='bold')

    ax1 = fig.add_axes([0.06, 0.38, 0.88, 0.16])
    draw_table(ax1,
        ['Parameter', 'Symbol', 'Typical Value', 'Unit'],
        [
            ['Fiber tension', 'T', '2.0', 'N'],
            ['Normal force (compaction)', 'Fn', '10', 'N'],
            ['Friction coefficient (CF/PA6)', '\u03bc', '0.3\u20130.5', '\u2014'],
            ['Max geodesic curvature', '\u03bag_max', '1.5\u20132.5', '1/m'],
            ['Min deviation radius', '1/\u03bag_max', '400\u2013670', 'mm'],
        ],
        col_widths=[0.35, 0.15, 0.20, 0.10],
        font_size=8,
        title='Typical Manufacturing Parameters')

    text_block(fig, 'Practical Implications:', [
        '1. For R=25mm cylinder: \u03bag must be < 2.0/m',
        '2. Pure hoop winding (\u03b1=0\u00b0) on cylinder: stable but poor for axial loads',
        '3. Helical winding (any \u03b1>0\u00b0): \u03bag=0, F_lateral=0 \u2192 always safe',
        '4. On complex surfaces: \u03bag\u22600 even for "geodesic-like" paths \u2192 must check',
    ], y_start=0.32, line_size=9)

    text_block(fig, 'Tension Control System (\u5f20\u529b\u95ed\u73af):', [
        'Lower T \u2192 lower F_lateral \u2192 wider path freedom',
        'But too low T \u2192 poor fiber consolidation, waviness',
        'Optimal T depends on local curvature:  T = T_base \u00b7 f(\u03bag)',
    ], y_start=0.18, line_size=9)

    add_page_number(fig, 6, TOTAL)
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 7: Chapter 5 — Stress-Geodesic Optimization ────────────────
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    section_header(fig, 'Chapter 5: 应力-测地线混合优化')
    sub_header(fig, 'Balancing Structural Performance and Manufacturing Quality')

    text_block(fig, 'Core Innovation:', [
        'Find the optimal fiber direction that balances structural performance',
        '(align with principal stress) and manufacturing quality (minimize \u03bag).',
        '',
        'At each point on the surface, two directions compete:',
        '  d_stress:   principal stress direction projected onto surface',
        '  d_geodesic: local geodesic direction (from Clairaut relation)',
        '',
        'The deviation angle \u03b2 between them:',
    ], y_start=0.80, line_size=9)

    fig.text(0.14, 0.61, '\u03b2 = arccos( |d_stress \u00b7 d_geodesic| )',
             fontsize=11, color=DARK_BLUE, fontfamily='monospace', fontweight='bold')

    # Optimization objective box
    box_y = 0.48
    fig.patches.append(mpatches.FancyBboxPatch(
        (0.08, box_y), 0.84, 0.10, boxstyle="round,pad=0.02",
        facecolor='#f5f8fc', edgecolor=ACCENT_BLUE, linewidth=1.5,
        transform=fig.transFigure, figure=fig))
    fig.text(0.5, box_y + 0.065, 'Optimization Objective:',
             ha='center', fontsize=11, fontweight='bold', color=DARK_BLUE)
    fig.text(0.5, box_y + 0.025,
             'd_opt = argmin [ w1*(1 - |d*d_sigma|^2) + w2*kappa_g(d)^2 ]',
             ha='center', fontsize=12, fontfamily='monospace', color=DARK_BLUE,
             fontweight='bold')

    text_block(fig, 'Where:', [
        'w1 = structural weight (importance of stress alignment)',
        'w2 = manufacturing weight (importance of low lateral force)',
        '\u03bag(d) = geodesic curvature if fiber follows direction d',
        '',
        'For cylindrical surfaces: \u03b2 is often small (stress directions',
        'naturally close to geodesics), so both objectives can be satisfied.',
        'For complex surfaces: \u03b2 can be large, requiring careful trade-off.',
    ], y_start=0.44, line_size=9)

    ax1 = fig.add_axes([0.06, 0.06, 0.88, 0.18])
    draw_table(ax1,
        ['Application', 'w1 (Structural)', 'w2 (Manufacturing)', 'Rationale'],
        [
            ['High-load structural', '0.8', '0.2', 'Max strength, harder to print'],
            ['General purpose', '0.5', '0.5', 'Balanced trade-off'],
            ['Complex geometry', '0.3', '0.7', 'Prioritize printability'],
            ['Near-geodesic (\u03b2<10\u00b0)', '0.9', '0.1', 'Both aligned, max structure'],
        ],
        col_widths=[0.22, 0.18, 0.22, 0.32],
        font_size=8,
        title='Weight Selection Guidelines')

    add_page_number(fig, 7, TOTAL)
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 8: Chapter 6 — Cylindrical Surface Analysis ────────────────
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    section_header(fig, 'Chapter 6: 圆柱面上的具体分析')
    sub_header(fig, 'The Benchmark Case: Hollow Cylinder Under Lateral Load')

    text_block(fig, 'Benchmark Geometry:', [
        'Hollow cylinder: R_outer=25mm, R_inner=20mm, H=80mm',
        'Load: 500N lateral force on top face',
    ], y_start=0.80, line_size=9)

    text_block(fig, 'Stress Analysis Results:', [
        'Tension side (\u03b8=0\u00b0):          \u03c31 direction \u2248 axial (\u03b1\u224885-90\u00b0)',
        'Compression side (\u03b8=180\u00b0):  \u03c31 direction \u2248 axial (\u03b1\u224885-90\u00b0)',
        'Shear sides (\u03b8=90\u00b0,270\u00b0):   \u03c31 direction \u2248 \u00b145\u00b0 helical',
    ], y_start=0.72, line_size=9)

    # Key insight box
    box_y = 0.56
    fig.patches.append(mpatches.FancyBboxPatch(
        (0.08, box_y), 0.84, 0.08, boxstyle="round,pad=0.02",
        facecolor='#e8f5e9', edgecolor=PASS_GREEN, linewidth=1.5,
        transform=fig.transFigure, figure=fig))
    fig.text(0.5, box_y + 0.045, 'Key Insight: ALL stress directions are geodesics on the cylinder!',
             ha='center', fontsize=11, fontweight='bold', color=PASS_GREEN)
    fig.text(0.5, box_y + 0.015,
             'Stress-driven paths \u2248 geodesic paths \u2192 optimal for BOTH structure AND manufacturing',
             ha='center', fontsize=9, color='#333333')

    text_block(fig, 'Why This Works:', [
        'Axial (\u03b1=90\u00b0):     geodesic \u2713   (straight line on unrolled cylinder)',
        '\u00b145\u00b0 helical:     geodesic \u2713   (constant-angle helix)',
        'Any constant \u03b1: geodesic \u2713   (Clairaut: R\u00b7cos\u03b1 = const)',
        '',
        'The deviation angle \u03b2 \u2248 0\u00b0 everywhere on the cylinder.',
    ], y_start=0.50, line_size=9)

    ax1 = fig.add_axes([0.06, 0.08, 0.88, 0.24])
    draw_table(ax1,
        ['Path Strategy', 'Winding Angle', '\u03bag', 'F_lateral', 'Stress Align', 'Overall'],
        [
            ['Stress-driven helix', 'Variable (45-90\u00b0)', '0', '0', 'Optimal', 'BEST'],
            ['Fixed 0\u00b0 (hoop)', '0\u00b0', '~0', '~0', 'Poor for bending', 'Worst'],
            ['Fixed 90\u00b0 (axial)', '90\u00b0', '0', '0', 'Good tension side', 'OK'],
            ['\u00b145\u00b0 alternating', '\u00b145\u00b0', '0', '0', 'Good for shear', 'Good'],
        ],
        col_widths=[0.22, 0.18, 0.08, 0.12, 0.22, 0.10],
        font_size=8,
        title='Path Comparison on Cylinder')

    add_page_number(fig, 8, TOTAL)
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 9: Chapter 7 — Extension to General Surfaces ───────────────
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    section_header(fig, 'Chapter 7: 一般曲面的推广')
    sub_header(fig, 'From Cylinders to Humanoid Robot Structural Parts')

    text_block(fig, 'Pipeline for General Surfaces of Revolution:', [
        '1. Surface extraction: extract outer surface mesh from 3D volume',
        '2. Parameterization: map surface to (\u03b8, z) or (u, v) coordinates',
        '3. Stress projection: project 3D stress field onto tangent plane',
        '4. Geodesic direction: compute from Clairaut relation r(z)\u00b7cos(\u03b1)=C',
        '5. Direction blending: find optimal \u03b1 at each point',
        '6. Streamline integration: trace paths following blended direction',
        '7. 5-axis conversion: path + surface normal \u2192 (X, Z, C) machine coords',
    ], y_start=0.80, line_size=9)

    text_block(fig, 'For Non-Revolute Surfaces:', [
        'Use discrete geodesic computation (heat method or fast marching)',
        'Surface parameterization via conformal/harmonic maps',
        'Direction field optimization (FieldGen or similar)',
    ], y_start=0.54, line_size=9)

    # Clairaut general form
    box_y = 0.38
    fig.patches.append(mpatches.FancyBboxPatch(
        (0.08, box_y), 0.84, 0.10, boxstyle="round,pad=0.02",
        facecolor='#f5f8fc', edgecolor=ACCENT_BLUE, linewidth=1.5,
        transform=fig.transFigure, figure=fig))
    fig.text(0.5, box_y + 0.07, 'Clairaut for General Surface of Revolution r(z):',
             ha='center', fontsize=10, fontweight='bold', color=DARK_BLUE)
    fig.text(0.5, box_y + 0.04,
             '\u03b1(z) = arccos( C / r(z) ),    C = r(z0)\u00b7cos(\u03b10)',
             ha='center', fontsize=11, fontfamily='monospace', color=DARK_BLUE)
    fig.text(0.5, box_y + 0.01,
             'If C > r(z) at some height \u2192 geodesic cannot reach there \u2192 turnaround point',
             ha='center', fontsize=9, color='#555555')

    ax1 = fig.add_axes([0.06, 0.06, 0.88, 0.24])
    draw_table(ax1,
        ['Part', 'Shape', 'Surface Type', 'Geodesic Complexity'],
        [
            ['Upper arm link', 'Near-cylinder', 'Revolution', 'Low (like benchmark)'],
            ['Forearm link', 'Tapered cylinder', 'Revolution', 'Medium (variable \u03b1)'],
            ['Knee joint cover', 'Double curvature', 'General', 'High (need numerical)'],
            ['Shin guard', 'Curved plate', 'Near-developable', 'Medium'],
            ['Foot sole', 'Complex 3D', 'General', 'High'],
        ],
        col_widths=[0.20, 0.20, 0.22, 0.30],
        font_size=8,
        title='Humanoid Robot Part Complexity')

    add_page_number(fig, 9, TOTAL)
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 10: Chapter 8 — Comparison with Existing Approaches ────────
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor('white')
    section_header(fig, 'Chapter 8: 与现有方案对比')
    sub_header(fig, 'Why Surface Geodesic Path Planning is the Right Approach')

    ax1 = fig.add_axes([0.04, 0.52, 0.92, 0.32])
    draw_table(ax1,
        ['Criterion', 'Markforged\n(2D Planar)', 'Old cfpp\n(3D Volume)',
         'Filament\nWinding', 'New cfpp\n(Surface Geodesic)'],
        [
            ['Path dimension', '2D (flat layers)', '3D (volume)', 'Surface (helical)', 'Surface (stress)'],
            ['Machine compat.', 'XYZ only', 'Cannot mfg.', 'Mandrel winding', 'XZ+C \u2713'],
            ['Lateral force ctrl', 'N/A (flat)', 'N/A (not surface)', 'Geodesic (\u03bag=0)', 'Geodesic-constrained'],
            ['Stress alignment', 'Fixed angles', 'Optimal', 'Not stress-driven', 'Optimal + constraint'],
            ['Part geometry', 'Flat/prismatic', 'Any (theoretical)', 'Convex revolution', 'Any revolution'],
            ['Path continuity', 'Short (per layer)', 'Medium', 'Long (multi-wrap)', 'Long (helical)'],
            ['Fiber cuts/part', 'Many (per layer)', 'Many (per path)', 'Few', 'Few'],
            ['Academic novelty', 'None (commercial)', 'Low (known)', 'None (1960s)', 'HIGH (novel)'],
        ],
        col_widths=[0.18, 0.18, 0.18, 0.18, 0.22],
        font_size=7,
        title='Comprehensive Approach Comparison')

    # Summary box
    box_y = 0.14
    fig.patches.append(mpatches.FancyBboxPatch(
        (0.08, box_y), 0.84, 0.32, boxstyle="round,pad=0.02",
        facecolor='#e8f5e9', edgecolor=PASS_GREEN, linewidth=2,
        transform=fig.transFigure, figure=fig))

    fig.text(0.5, box_y + 0.27,
             'Our Approach: Surface Geodesic Streamline Planning',
             ha='center', fontsize=13, fontweight='bold', color=DARK_BLUE)

    summary_items = [
        '\u2713  Stress-driven optimization (from FEA, like aerospace AFP)',
        '\u2713  Geodesic curvature constraint (from filament winding theory)',
        '\u2713  Machine-aware path planning (matched to XZ+C kinematics)',
        '\u2713  Applicable to general surfaces of revolution',
    ]
    for i, item in enumerate(summary_items):
        fig.text(0.14, box_y + 0.22 - i * 0.03, item,
                 fontsize=10, color='#333333')

    fig.text(0.5, box_y + 0.09,
             'Simultaneously optimizes:',
             ha='center', fontsize=10, fontweight='bold', color=DARK_BLUE)
    opt_items = [
        '1. Structural performance (fibers along stress)',
        '2. Manufacturing quality (minimal lateral force)',
        '3. Machine feasibility (paths achievable by XZ+C system)',
    ]
    for i, item in enumerate(opt_items):
        fig.text(0.18, box_y + 0.06 - i * 0.025, item,
                 fontsize=9, color='#333333')

    # Key innovation highlight
    fig.text(0.5, box_y + -0.02,
             'Key Innovation:  d = argmin[ w1\u00b7(1-|d\u00b7d_\u03c3|\u00b2) + w2\u00b7\u03bag\u00b2 ]',
             ha='center', fontsize=10, fontfamily='monospace',
             fontweight='bold', color=PASS_GREEN)

    add_page_number(fig, 10, TOTAL)
    pdf.savefig(fig)
    plt.close(fig)

print(f'Report saved to: {OUT_PDF}')
print(f'Total pages: {TOTAL}')
