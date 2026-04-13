"""
/api/v1/geometry/* 路由
同步网格接口（快速，返回 mesh_job_id 供后续任务继承）。
"""
from __future__ import annotations

import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from ..core.job_manager import get_job_manager
from ..models.schemas import JobRef, MeshRequest, MeshResult
from ..services import pipeline as svc

router = APIRouter(prefix="/api/v1/geometry", tags=["geometry"])

# ── 内置参数生成网格 ──────────────────────────────────────────────────────────

@router.post("/mesh", response_model=JobRef, status_code=202)
def submit_mesh(req: MeshRequest):
    """提交网格生成任务，返回 job_id（内置几何参数）。"""
    mgr = get_job_manager()
    params = req.model_dump()
    job_id = mgr.submit(svc.run_mesh, params, "mesh")
    return JobRef(job_id=job_id, status="pending")


# ── STL 上传 ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=JobRef, status_code=202)
async def upload_stl(file: UploadFile = File(...)):
    """上传 STL，提交网格任务。"""
    import tempfile, uuid
    if not file.filename.lower().endswith((".stl", ".step", ".iges")):
        raise HTTPException(400, "Only STL/STEP/IGES files are accepted")
    suffix = os.path.splitext(file.filename)[1]
    tmp = os.path.join(tempfile.gettempdir(), f"cfpp_upload_{uuid.uuid4().hex[:8]}{suffix}")
    content = await file.read()
    with open(tmp, "wb") as f:
        f.write(content)
    mgr = get_job_manager()
    job_id = mgr.submit(svc.run_mesh, {"stl_path": tmp, "mesh_size": 3.0}, "mesh")
    return JobRef(job_id=job_id, status="pending")


# ── 内置模型列表 ──────────────────────────────────────────────────────────────

_BUILTIN_MODELS = [
    "cylinder", "hollow_cylinder", "cone", "tee_pipe",
    "elbow_pipe", "spar", "turbine_blade", "pressure_vessel",
]

@router.get("/models")
def list_models():
    return {"models": _BUILTIN_MODELS}
