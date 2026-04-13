"""
/api/v1/jobs/* 路由
支持：提交任务、查询状态、取消、SSE 进度流、列出最近任务。
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..core.job_manager import get_job_manager
from ..models.schemas import (
    CancelResponse, FEARequest, GcodeRequest,
    JobDetail, JobRef, OptimizeRequest, PlanRequest,
)
from ..services import pipeline as svc

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


# ── 提交 ──────────────────────────────────────────────────────────────────────

@router.post("/plan", response_model=JobRef, status_code=202)
def submit_plan(req: PlanRequest):
    mgr = get_job_manager()
    parent = req.mesh_job_id
    params = req.model_dump()
    job_id = mgr.submit(svc.run_plan, params, "plan", parent_job_id=parent)
    return JobRef(job_id=job_id, status="pending")


@router.post("/gcode", response_model=JobRef, status_code=202)
def submit_gcode(req: GcodeRequest):
    mgr = get_job_manager()
    parent = req.plan_job_id
    params = req.model_dump()
    job_id = mgr.submit(svc.run_gcode, params, "gcode", parent_job_id=parent)
    return JobRef(job_id=job_id, status="pending")


@router.post("/fea", response_model=JobRef, status_code=202)
def submit_fea(req: FEARequest):
    mgr = get_job_manager()
    parent = req.mesh_job_id
    params = req.model_dump()
    job_id = mgr.submit(svc.run_fea, params, "fea", parent_job_id=parent)
    return JobRef(job_id=job_id, status="pending")


@router.post("/optimize", response_model=JobRef, status_code=202)
def submit_optimize(req: OptimizeRequest):
    mgr = get_job_manager()
    parent = req.mesh_job_id
    params = req.model_dump()
    job_id = mgr.submit(svc.run_optimize, params, "optimize", parent_job_id=parent)
    return JobRef(job_id=job_id, status="pending")


# ── 查询 ──────────────────────────────────────────────────────────────────────

@router.get("", response_model=list[JobDetail])
def list_jobs(limit: int = Query(20, ge=1, le=100)):
    return get_job_manager().list_recent(limit)


@router.get("/{job_id}", response_model=JobDetail)
def get_job(job_id: str):
    job = get_job_manager().get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    return job


# ── 取消 ──────────────────────────────────────────────────────────────────────

@router.delete("/{job_id}", response_model=CancelResponse)
def cancel_job(job_id: str):
    ok = get_job_manager().cancel(job_id)
    return CancelResponse(cancelled=ok, job_id=job_id)


# ── SSE 进度流 ────────────────────────────────────────────────────────────────

@router.get("/{job_id}/stream")
def stream_job(job_id: str):
    """Server-Sent Events：每 300ms 推送一次进度，任务结束后关闭流。"""
    mgr = get_job_manager()

    def _generate():
        while True:
            job = mgr.get(job_id)
            if job is None:
                yield f"event: error\ndata: {json.dumps({'error': 'not found'})}\n\n"
                break
            payload = json.dumps(job)
            yield f"data: {payload}\n\n"
            if job["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.3)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
