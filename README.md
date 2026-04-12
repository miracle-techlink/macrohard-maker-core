# CF-Path-Planner

**应力驱动的连续碳纤维多轴3D打印路径规划工具**

结合有限元分析（FEA）、曲面测地线路径规划和拓扑优化，为连续碳纤维增强聚合物（CCF）3D打印生成最优纤维沉积路径的全栈工具。

---

## 功能特性

- **FEA 求解器** — 基于 `scikit-fem` 的线弹性求解器，支持各向同性与正交各向异性材料，逐单元纤维取向
- **测地线路径规划** — 应力场引导的曲面测地线路径（圆柱、锥面、任意网格）
- **XY+A 路径规划器** — 四轴（XY+A 旋转）航点生成，支持螺旋缠绕、环向/轴向模式及自定义纤维角度
- **拓扑优化** — 基于 SIMP 法的材料分布优化
- **G-code 生成器** — 输出适用于 Klipper 多轴打印机的机器码
- **交互式 Web UI** — 基于 Three.js 的可视化界面，支持实时路径显示、分层控制和网格查看

## 项目结构

```
cf-path-planner/
├── cfpp/                  # 核心 Python 库
│   ├── solver/            # FEA：各向同性与正交各向异性弹性求解器
│   ├── surface/           # 测地线路径规划、应力场提取、曲面规划器 v2
│   ├── mesh/              # 网格生成（基于 gmsh）
│   ├── topo/              # SIMP 拓扑优化
│   ├── optimizer/         # 纤维铺层优化器
│   └── gcode/             # G-code 后处理
├── webapp/
│   └── server.py          # HTTP 服务器：静态文件 + REST API
├── visualization/
│   ├── index.html         # Three.js 前端（2400+ 行，无需构建）
│   ├── lib/               # three.min.js、OrbitControls.js
│   └── data/              # Pipeline 实时 JSON 数据
├── examples/
│   ├── cylinder/          # 螺旋缠绕示例
│   ├── cone/              # 锥面示例
│   ├── cantilever/        # 悬臂梁拓扑优化
│   └── leg_link/          # 机器人腿杆纤维铺层
├── tests/                 # 单元测试与集成测试
├── klipper/               # Klipper 打印机配置文件
└── docs/                  # LaTeX 报告：测地线理论、FEA、实验方案
```

## 快速开始

### 依赖安装

```bash
pip install numpy scipy meshio scikit-fem gmsh
```

### 启动 Web 应用

```bash
cd webapp
python server.py
# 打开浏览器访问 http://localhost:8080
```

Web UI 提供以下接口：
- `POST /api/mesh` — 根据模型参数生成网格
- `POST /api/fea` — 运行 FEA 并提取应力场
- `POST /api/xyza_paths` — 计算纤维缠绕路径
- `POST /api/gcode` — 生成 G-code
- `POST /api/upload_stl` — 上传自定义 STL 文件

### 直接调用路径规划

```python
from cfpp.surface.planner_v2 import XYAPathPlanner

planner = XYAPathPlanner(a_offset_z=50.0)
waypoints = planner.helical_path(radius=15, length=80, winding_angle=45, n_layers=4)
# 返回 (x, y_model, z_model, a_deg) 元组列表
```

### 运行示例

```bash
cd examples/cylinder
python run_surface.py      # 圆柱曲面上的测地线路径
python run_all_phases.py   # 完整流程：网格 → FEA → 路径 → G-code
```

## 流程总览

```
STL / 参数化模型
       ↓
  网格生成 (gmsh)
       ↓
  FEA 应力分析 (scikit-fem)
       ↓
  应力场 → 主方向提取
       ↓
  曲面测地线路径规划
       ↓
  XY+A 航点生成
       ↓
  G-code 输出 (Klipper)
```

## 硬件目标

适用于 **XY+A** 多轴 FFF 打印机：
- X 轴：沿纤维方向的轴向平移
- Y 轴：喷嘴横向移动
- A 轴：旋转平台（零件旋转）

Klipper 配置示例见 `klipper/` 目录。

## 技术文档

`docs/` 目录包含 PDF 及 LaTeX 源文件：
- `geodesic_theory` — 曲面测地线路径理论
- `surface_geodesic_report` — 曲面测地线实验报告
- `full_report` — 完整算法报告
- `experiment_plan` — 实验方案

## 测试

```bash
cd tests
python test_xyza_backend.py
python benchmark_webapp.py
```

## 许可证

MIT
