"""
CF-Path-Planner uvicorn 启动入口。
用法:
    python webapp/run.py              # default port 8080
    python webapp/run.py --port 5000
    uvicorn webapp.app.main:app --reload
"""
from __future__ import annotations

import argparse
import os
import sys

# 确保 project root 在 sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="CF-Path-Planner server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    print(f"Starting CF-Path-Planner on http://{args.host}:{args.port}")
    uvicorn.run(
        "webapp.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
