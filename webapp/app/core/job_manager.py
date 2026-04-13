"""
JobManager — 异步任务管理核心
每个任务拿到唯一 job_id，在 ThreadPoolExecutor 中执行，
支持状态查询、进度回调、取消（pending 阶段）。
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable


class JobState:
    """单个任务的隔离状态（替代全局 state dict）。"""

    def __init__(self) -> None:
        self.mesh_path:      str | None = None
        self.waypoints_path: str | None = None   # .npz 文件路径
        self.vis_path:       str | None = None   # live_layers.json
        self.gcode_path:     str | None = None


class Job:
    __slots__ = (
        "id", "type", "status", "progress", "result",
        "error", "created_at", "updated_at", "state", "_future",
    )

    def __init__(self, job_id: str, job_type: str) -> None:
        self.id          = job_id
        self.type        = job_type
        self.status      = "pending"          # pending|running|completed|failed|cancelled
        self.progress: dict | None = None
        self.result:   dict | None = None
        self.error:    str  | None = None
        self.created_at  = time.time()
        self.updated_at  = time.time()
        self.state       = JobState()
        self._future: Future | None = None

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "type":        self.type,
            "status":      self.status,
            "progress":    self.progress,
            "result":      self.result,
            "error":       self.error,
            "created_at":  self.created_at,
            "updated_at":  self.updated_at,
        }


class JobManager:
    """
    线程池任务管理器。
    - 每个任务返回 job_id（8位 hex）
    - 进度通过 progress_cb(phase, pct, msg) 回调
    - pending 任务可取消
    - 保留最近 200 条已完成任务记录
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._pool   = ThreadPoolExecutor(max_workers=max_workers,
                                          thread_name_prefix="cfpp-worker")
        self._jobs:  dict[str, Job] = {}
        self._done:  deque[str]     = deque(maxlen=200)   # 已完成任务 id 队列
        self._lock   = threading.Lock()

    # ── 提交 ─────────────────────────────────────────────────────────────

    def submit(
        self,
        fn: Callable,
        params: dict[str, Any],
        job_type: str,
        parent_job_id: str | None = None,
    ) -> str:
        """提交任务，立即返回 job_id。"""
        job_id = uuid.uuid4().hex[:8]
        job    = Job(job_id, job_type)

        # 如果有父任务，继承其 JobState（共享 mesh_path 等）
        if parent_job_id:
            parent = self._jobs.get(parent_job_id)
            if parent:
                job.state = parent.state

        with self._lock:
            self._jobs[job_id] = job

        future = self._pool.submit(self._run, job_id, fn, params)
        with self._lock:
            self._jobs[job_id]._future = future

        return job_id

    # ── 查询 ─────────────────────────────────────────────────────────────

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def list_recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            ids = list(self._done)[-limit:]
            return [self._jobs[i].to_dict() for i in ids if i in self._jobs]

    # ── 取消 ─────────────────────────────────────────────────────────────

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            if job.status == "pending":
                job.status = "cancelled"
                job.updated_at = time.time()
                if job._future:
                    job._future.cancel()
                return True
        return False

    # ── 内部执行 ─────────────────────────────────────────────────────────

    def _run(self, job_id: str, fn: Callable, params: dict) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status == "cancelled":
                return
            job.status     = "running"
            job.updated_at = time.time()

        def progress_cb(phase: str, pct: float, msg: str = "") -> None:
            with self._lock:
                j = self._jobs.get(job_id)
                if j:
                    j.progress    = {"phase": phase, "percent": pct, "message": msg}
                    j.updated_at  = time.time()

        try:
            result = fn(params, job_state=job.state, progress_cb=progress_cb)
            with self._lock:
                job.status     = "completed"
                job.result     = result
                job.updated_at = time.time()
                self._done.append(job_id)
        except Exception as exc:
            with self._lock:
                job.status     = "failed"
                job.error      = str(exc)
                job.updated_at = time.time()
                self._done.append(job_id)

    # ── 生命周期 ─────────────────────────────────────────────────────────

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)


# 全局单例（main.py 启动时初始化）
_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager(max_workers=4)
    return _manager
