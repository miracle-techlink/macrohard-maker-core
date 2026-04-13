"""Pydantic v2 请求/响应 Schema。"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── 通用 ─────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    CANCELLED = "cancelled"


class JobProgress(BaseModel):
    phase:   str
    percent: float
    message: str = ""


class JobRef(BaseModel):
    job_id: str
    status: JobStatus


class JobDetail(BaseModel):
    id:         str
    type:       str
    status:     JobStatus
    progress:   Optional[JobProgress] = None
    result:     Optional[dict[str, Any]] = None
    error:      Optional[str] = None
    created_at: float
    updated_at: float


# ── Geometry ──────────────────────────────────────────────────────────────────

class MeshRequest(BaseModel):
    model:     str   = Field("cylinder", description="cylinder | cone")
    mesh_size: float = Field(3.0, ge=0.5, le=10.0)
    # cylinder
    r_outer:   float = Field(25.0, ge=1.0)
    r_inner:   float = Field(0.0,  ge=0.0)
    height:    float = Field(50.0, ge=1.0)
    # cone
    r_bottom:  float = Field(25.0, ge=1.0)
    r_top:     float = Field(15.0, ge=0.0)
    wall:      float = Field(5.0,  ge=0.5)


class MeshResult(BaseModel):
    status:    str
    mesh_path: str
    info:      str = ""


# ── Jobs ─────────────────────────────────────────────────────────────────────

class PlanRequest(BaseModel):
    """路径规划任务请求。"""
    mesh_job_id: Optional[str] = Field(None, description="前序 mesh job 的 id")
    mesh_path:   Optional[str] = Field(None, description="直接指定网格路径（优先）")

    strategy:       str   = Field("combined", description="combined | fill | constant | stress")
    angle:          float = Field(45.0,  ge=5.0,  le=85.0)
    n_layers:       int   = Field(4,     ge=1,    le=50)
    n_walls:        int   = Field(2,     ge=0,    le=8)
    infill_density: float = Field(0.25,  ge=0.05, le=1.0)
    layer_height:   float = Field(0.18,  ge=0.05, le=0.5)
    extrusion_width:float = Field(0.4,   ge=0.1,  le=2.0)
    r_inner:        float = Field(0.0,   ge=0.0)
    a_offset_z:     float = Field(50.0,  ge=0.0)


class GcodeRequest(BaseModel):
    """G-code 生成任务请求。"""
    plan_job_id:    Optional[str] = Field(None, description="前序 plan job 的 id")
    feed_rate:      float = Field(3000.0, ge=100.0,  le=20000.0)
    a_offset_z:     float = Field(50.0,  ge=0.0)
    layer_height:   float = Field(0.18,  ge=0.05, le=0.5)
    extrusion_width:float = Field(0.4,   ge=0.1,  le=2.0)


class FEARequest(BaseModel):
    """有限元分析任务请求。"""
    mesh_job_id: Optional[str] = None
    mesh_path:   Optional[str] = None
    E_gpa: float = Field(60.0,  ge=1.0,  le=500.0)
    nu:    float = Field(0.3,   ge=0.0,  lt=0.5)
    P:     float = Field(500.0, ge=0.0)


class OptimizeRequest(BaseModel):
    mesh_job_id: Optional[str] = None
    mesh_path:   Optional[str] = None
    n_starts: int   = Field(6,   ge=1, le=20)
    max_iter: int   = Field(15,  ge=1, le=100)
    P:        float = Field(500.0, ge=0.0)


# ── System ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.1.0"


class LogEntry(BaseModel):
    t:   str
    lvl: str
    msg: str


class LogsResponse(BaseModel):
    logs: list[LogEntry]
    n:    int


class CancelResponse(BaseModel):
    cancelled: bool
    job_id:    str
