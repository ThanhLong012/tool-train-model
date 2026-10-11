"""Đọc trạng thái nhiệt của macOS (nominal / fair / serious / critical).

Dùng ProcessInfo.thermalState qua một binary Swift nhỏ, biên dịch một lần vào
.cache/ — không cần sudo. Trên máy không có swiftc (không phải macOS) trả về
"unknown".
"""

from __future__ import annotations

import shutil
import subprocess

from .config import CACHE_DIR

LEVELS = ["nominal", "fair", "serious", "critical"]
LABELS = {"nominal": "🟢 Mát", "fair": "🟡 Ấm (bình thường khi train)",
          "serious": "🟠 Nóng", "critical": "🔴 Quá nóng", "unknown": "⚪ Không đọc được"}

_SRC = 'import Foundation\nprint(["nominal","fair","serious","critical"][ProcessInfo.processInfo.thermalState.rawValue])\n'
_BIN = CACHE_DIR / "thermal_state"


def _ensure_binary() -> bool:
    if _BIN.exists():
        return True
    if not shutil.which("swiftc"):
        return False
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    src = CACHE_DIR / "thermal_state.swift"
    src.write_text(_SRC)
    r = subprocess.run(["swiftc", "-O", str(src), "-o", str(_BIN)], capture_output=True)
    return r.returncode == 0


def state() -> str:
    if not _ensure_binary():
        return "unknown"
    try:
        out = subprocess.run([str(_BIN)], capture_output=True, text=True, timeout=5).stdout.strip()
        return out if out in LEVELS else "unknown"
    except Exception:
        return "unknown"


def at_least(s: str, level: str) -> bool:
    return s in LEVELS and level in LEVELS and LEVELS.index(s) >= LEVELS.index(level)
