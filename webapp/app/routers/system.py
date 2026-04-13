"""
/api/v1/system/* 路由
health check、日志查询、metrics。
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ..models.schemas import HealthResponse, LogEntry, LogsResponse

router = APIRouter(prefix="/api/v1/system", tags=["system"])

# ── 全局日志 ring buffer（与旧 server.py 风格一致）────────────────────────────

_log_buffer: deque[dict] = deque(maxlen=500)
_log_lock = threading.Lock()


class _BufHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        with _log_lock:
            _log_buffer.append({
                "t":   time.strftime("%H:%M:%S") + f".{record.msecs:03.0f}",
                "lvl": record.levelname,
                "msg": record.getMessage(),
            })


def setup_log_handler(logger_name: str = "cfpp") -> logging.Logger:
    """在 app 启动时调用一次，挂载 ring-buffer handler。"""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    h = _BufHandler()
    h.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)-5s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(h)
    return logger


# ── 路由 ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@router.get("/logs", response_model=LogsResponse)
def get_logs(
    since: int = Query(0, ge=0, description="从第 N 条开始返回（增量拉取）"),
    level: str = Query("", description="过滤级别: DEBUG|INFO|WARNING|ERROR"),
):
    with _log_lock:
        entries = list(_log_buffer)
    if level:
        level = level.upper()
        entries = [e for e in entries if e["lvl"] == level]
    entries = entries[since:]
    return LogsResponse(
        logs=[LogEntry(**e) for e in entries],
        n=len(entries),
    )


@router.get("/metrics")
def metrics():
    """简单 Prometheus-style 文本指标（后续可接 prometheus_client）。"""
    from ..core.job_manager import get_job_manager
    mgr = get_job_manager()
    recent = mgr.list_recent(100)
    counts = {}
    for j in recent:
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    lines = ["# HELP cfpp_jobs_total Total jobs by status", "# TYPE cfpp_jobs_total gauge"]
    for status, n in counts.items():
        lines.append(f'cfpp_jobs_total{{status="{status}"}} {n}')
    return JSONResponse({"text": "\n".join(lines)})
