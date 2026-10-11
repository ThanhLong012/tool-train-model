"""Quản lý tiến trình train chạy riêng (subprocess), UI chỉ theo dõi.

Chỉ cho phép 1 lần train chạy cùng lúc (YOLO hoặc RF-DETR) để máy không bị quá tải
nhiệt/RAM khi train 2 model song song.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import ROOT


class Job:
    def __init__(self, kind: str, runs_dir: Path):
        self.kind = kind
        self.runs_dir = runs_dir
        self.proc: subprocess.Popen | None = None
        self.name = ""
        self.log_path: Path | None = None
        self.stop_at: float | None = None
        self.note = ""
        self.total_epochs: int | None = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def launch(self, cmd: list[str], name: str, total_epochs: int | None = None) -> str:
        run_dir = self.runs_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = run_dir / "train.log"
        log = self.log_path.open("a")
        log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(cmd[2:])}\n")
        log.flush()
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.name, self.stop_at, self.note, self.total_epochs = name, None, "", total_epochs
        return name

    def stop(self, reason: str = "Đã bấm dừng") -> None:
        if self.running():
            self.proc.send_signal(signal.SIGINT)  # như Ctrl+C: dừng gọn
            self.stop_at, self.note = time.time(), reason

    def poll_autostop(self) -> None:
        if self.running() and self.stop_at and time.time() - self.stop_at > 30:
            self.proc.kill()  # không chịu dừng sau 30s thì dừng cứng


_jobs: dict[str, Job] = {}


def get(kind: str, runs_dir: Path) -> Job:
    if kind not in _jobs:
        _jobs[kind] = Job(kind, runs_dir)
    return _jobs[kind]


def any_running(except_kind: str | None = None) -> Job | None:
    for j in _jobs.values():
        if j.kind != except_kind and j.running():
            return j
    return None


def split_updates(text: str) -> list[str]:
    # Thanh tiến trình (Ultralytics / tqdm) ghi đè dòng bằng \r hoặc ESC[K khi ghi ra file
    return [l for l in re.split(r"\r|\n|\x1b\[K", text) if l.strip()]


def log_tail(log_path: Path | None, n: int = 25) -> str:
    if not log_path or not log_path.exists():
        return ""
    out: list[str] = []
    for raw in log_path.read_text(errors="ignore")[-30000:].split("\n"):
        parts = split_updates(raw)
        if parts:
            out.append(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", parts[-1]))  # chỉ giữ trạng thái cuối mỗi dòng
    return "\n".join(out[-n:])
