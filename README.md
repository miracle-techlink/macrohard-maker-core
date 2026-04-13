# CF-Path-Planner

**应力驱动的连续碳纤维多轴3D打印路径规划工具**

结合有限元分析（FEA）、曲面测地线路径规划和拓扑优化，为连续碳纤维增强聚合物（CCF）3D打印生成最优纤维沉积路径。

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## 功能特性

- **XY+A combined 路径规划** — 缠绕层 + 填充层一体化，1.7M 路径点 / 3.4s
- **G-code 生成（16× 提速）** — 直接字符串格式化，22.5s → 1.35s，输出 75MB G-code
- **FEA 应力分析** — 基于 scikit-fem 的线弹性求解器，识别主应力方向
- **打印方向优化** — 多起点枚举，最小化悬垂面积
- **参数化网格生成** — cylinder / cone / hollow 等内置模型 + STL 导入
- **Three.js 3D 可视化** — 实时路径预览，层切片滑块，1342 可视层段
- **后端日志面板** — 实时 500 条环形缓冲，可拖拽，带清空反馈

## 后端架构（v1.1）

```
webapp/
├── run.py                    # uvicorn 入口
└── app/
    ├── main.py               # FastAPI app + 旧接口兼容层
    ├── core/
    │   └── job_manager.py    # ThreadPoolExecutor 任务管理器，per-job JobState
    ├── models/
    │   └── schemas.py        # Pydantic v2 请求/响应 Schema
    ├── services/
    │   └── pipeline.py       # 业务逻辑：run_mesh/plan/gcode/fea/optimize
    └── routers/
        ├── jobs.py           # /api/v1/jobs/* + SSE 进度流
        ├── geometry.py       # /api/v1/geometry/mesh + upload
        └── system.py         # /api/v1/system/health|logs|metrics
```

## API

### v1 接口（推荐）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/geometry/mesh` | 提交网格生成任务 → `job_id` |
| POST | `/api/v1/geometry/upload` | 上传 STL → `job_id` |
| POST | `/api/v1/jobs/plan` | 提交路径规划任务 |
| POST | `/api/v1/jobs/gcode` | 提交 G-code 生成任务 |
| POST | `/api/v1/jobs/fea` | 提交 FEA 任务 |
| POST | `/api/v1/jobs/optimize` | 提交打印方向优化任务 |
| GET  | `/api/v1/jobs/{id}` | 查询任务状态 |
| GET  | `/api/v1/jobs/{id}/stream` | SSE 实时进度流 |
| DELETE | `/api/v1/jobs/{id}` | 取消任务 |
| GET  | `/api/v1/system/health` | 健康检查 |
| GET  | `/api/v1/system/logs` | 后端日志（增量拉取） |

### 旧接口（向后兼容）

`POST /api/mesh` · `/api/fea` · `/api/xyza_paths` · `/api/gcode` · `/api/optimize`

Swagger UI：启动后访问 `http://localhost:8080/docs`

## 快速开始

### 安装依赖

```bash
pip install fastapi "uvicorn[standard]" python-multipart numpy scipy meshio scikit-fem gmsh
```

### 启动服务

```bash
python webapp/run.py
# 访问 http://localhost:8080
# Swagger: http://localhost:8080/docs
```

### 直接调用路径规划

```python
from cfpp.surface.planner_v2 import XYAPathPlanner

planner = XYAPathPlanner(a_offset_z=50.0)
waypoints = planner.helical_path(radius=15, length=80, winding_angle=45, n_layers=4)
```

## 流程总览

```
STL / 参数化模型
       ↓
  网格生成 (gmsh)          POST /api/v1/geometry/mesh
       ↓
  FEA 应力分析             POST /api/v1/jobs/fea
       ↓
  主方向提取 + 路径规划     POST /api/v1/jobs/plan
       ↓
  G-code 生成              POST /api/v1/jobs/gcode
       ↓
  Klipper 打印机
```

## 项目结构

```
cf-path-planner/
├── cfpp/                  # 核心 Python 库
│   ├── solver/            # FEA 弹性求解器
│   ├── surface/           # 测地线路径、应力场、XYA 规划器
│   ├── mesh/              # 网格生成 (gmsh)
│   ├── topo/              # SIMP 拓扑优化
│   └── gcode/             # G-code 后处理
├── webapp/
│   ├── run.py             # uvicorn 启动入口
│   └── app/               # FastAPI 应用（见上方架构图）
├── visualization/
│   ├── index.html         # Three.js 前端
│   └── data/              # Pipeline 实时 JSON 输出
├── examples/              # 使用示例
├── klipper/               # Klipper 打印机配置
└── docs/                  # LaTeX 技术报告
```

## 硬件目标

适用于 **XY+A** 多轴 FFF 打印机（X/Y 平动 + A 旋转平台）。Klipper 配置示例见 `klipper/`。

## 测试

```bash
python tests/test_xyza_backend.py
python tests/benchmark_webapp.py
```

## 桌面版

提供 Electron 封装的桌面应用：[cf-path-planner-desktop](https://github.com/miracle-techlink/cf-path-planner-desktop)

## License

MIT
