"""
CF-Path-Planner FastAPI 应用入口。
Phase 1: FastAPI + JobManager 替换 HTTPServer，保留 /api/* 旧接口兼容。
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .core.job_manager import get_job_manager
from .models.schemas import JobRef
from .routers import geometry, jobs, system
from .routers.system import setup_log_handler
from .services import pipeline as svc

# ── 路径 ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
VIZ_DIR      = os.path.join(PROJECT_ROOT, "visualization")
DATA_DIR     = os.path.join(VIZ_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger("cfpp")


# ── 生命周期 ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_log_handler("cfpp")
    mgr = get_job_manager()
    logger.info("CF-Path-Planner FastAPI server started")
    yield
    mgr.shutdown(wait=False)
    logger.info("Server shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CF-Path-Planner API",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── v1 路由 ───────────────────────────────────────────────────────────────────

app.include_router(jobs.router)
app.include_router(geometry.router)
app.include_router(system.router)


# ── 旧 /api/* 兼容层（前端未迁移前继续可用）──────────────────────────────────
# 通过 JobManager 提交，但立即轮询等待结果（阻塞语义，与旧接口一致）

def _run_sync(fn, params: dict, job_type: str, parent_job_id: str | None = None) -> dict:
    """提交任务并同步等待结果，超时 120s。"""
    mgr = get_job_manager()
    job_id = mgr.submit(fn, params, job_type, parent_job_id=parent_job_id)
    deadline = time.time() + 120
    while time.time() < deadline:
        job = mgr.get(job_id)
        if job and job["status"] == "completed":
            return job["result"] or {}
        if job and job["status"] in ("failed", "cancelled"):
            return {"error": job.get("error") or "job failed"}
        time.sleep(0.2)
    return {"error": "timeout"}


@app.post("/api/mesh")
async def legacy_mesh(request: Request):
    body = await _parse_body(request)
    result = _run_sync(svc.run_mesh, body, "mesh")
    return JSONResponse(result)


@app.post("/api/fea")
async def legacy_fea(request: Request):
    body = await _parse_body(request)
    result = _run_sync(svc.run_fea, body, "fea")
    return JSONResponse(result)


@app.post("/api/xyza_paths")
async def legacy_plan(request: Request):
    body = await _parse_body(request)
    result = _run_sync(svc.run_plan, body, "plan")
    return JSONResponse(result)


@app.post("/api/gcode")
async def legacy_gcode(request: Request):
    body = await _parse_body(request)
    result = _run_sync(svc.run_gcode, body, "gcode")
    return JSONResponse(result)


@app.post("/api/optimize")
async def legacy_optimize(request: Request):
    body = await _parse_body(request)
    result = _run_sync(svc.run_optimize, body, "optimize")
    return JSONResponse(result)


@app.get("/api/logs")
async def legacy_logs(since: int = 0):
    from .routers.system import _log_buffer, _log_lock
    with _log_lock:
        entries = list(_log_buffer)
    return JSONResponse({"logs": entries[since:], "n": len(entries[since:])})


@app.get("/api/waypoints_npz")
async def legacy_waypoints_npz():
    """返回最近一次 plan 任务生成的 .npz 文件（向后兼容）。"""
    mgr = get_job_manager()
    for job_dict in reversed(mgr.list_recent(50)):
        if job_dict["type"] == "plan" and job_dict["status"] == "completed":
            job = mgr._jobs.get(job_dict["id"])
            if job and job.state.waypoints_path and os.path.exists(job.state.waypoints_path):
                return FileResponse(job.state.waypoints_path, media_type="application/octet-stream")
    return JSONResponse({"error": "No waypoints available"}, status_code=404)


# ── STL 上传（旧接口）────────────────────────────────────────────────────────

@app.post("/api/upload_stl")
async def legacy_upload_stl(request: Request):
    import tempfile, uuid as _uuid
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("file")
        if file is None:
            return JSONResponse({"error": "No file"}, status_code=400)
        data = await file.read()
        suffix = os.path.splitext(file.filename)[1] or ".stl"
    else:
        data = await request.body()
        suffix = ".stl"
    tmp = os.path.join(tempfile.gettempdir(), f"cfpp_upload_{_uuid.uuid4().hex[:8]}{suffix}")
    with open(tmp, "wb") as f:
        f.write(data)
    result = _run_sync(svc.run_mesh, {"stl_path": tmp, "mesh_size": 3.0}, "mesh")
    return JSONResponse(result)


# ── 静态文件 & SPA ─────────────────────────────────────────────────────────────

if os.path.isdir(VIZ_DIR):
    # data/ 目录单独挂载（优先级高，避免被 StaticFiles catch-all 吃掉）
    app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
    app.mount("/", StaticFiles(directory=VIZ_DIR, html=True), name="static")


# ── 工具函数 ──────────────────────────────────────────────────────────────────

async def _parse_body(request: Request) -> dict:
    try:
        return await request.json()
    except Exception:
        return {}
