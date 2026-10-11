"""Theo dõi RAM và gửi thông báo hệ thống (macOS) khi máy nóng hoặc đầy RAM.

Thông báo chỉ gửi khi mức độ *tăng lên* (ví dụ fair -> serious), không gửi lại
liên tục mỗi lần poll, và có thời gian chờ tối thiểu giữa 2 thông báo cùng loại.
"""

from __future__ import annotations

import subprocess
import time

import psutil

from . import thermal

RAM_LEVELS = ["ok", "cao", "rat_cao"]
RAM_LABELS = {"ok": "🟢 Bình thường", "cao": "🟡 Cao", "rat_cao": "🔴 Rất cao (dễ bị out-of-memory)"}
RAM_WARN_PCT = 85.0
RAM_CRITICAL_PCT = 95.0

_COOLDOWN_S = 90.0  # không gửi lại thông báo cùng loại trong khoảng này
_last_level: dict[str, str] = {}
_last_notify_at: dict[str, float] = {}


def ram() -> dict:
    m = psutil.virtual_memory()
    if m.percent >= RAM_CRITICAL_PCT:
        level = "rat_cao"
    elif m.percent >= RAM_WARN_PCT:
        level = "cao"
    else:
        level = "ok"
    return {"percent": m.percent, "used_gb": m.used / 1e9, "total_gb": m.total / 1e9, "level": level}


def notify(title: str, message: str) -> None:
    """Gửi thông báo hệ thống macOS (Notification Center). Bỏ qua lặng lẽ nếu không gửi được."""
    try:
        subprocess.run(["osascript", "-e", f'display notification "{message}" with title "{title}"'],
                       capture_output=True, timeout=5)
    except Exception:
        pass


def _notify_on_rise(kind: str, level: str, levels: list[str], title: str, message: str) -> None:
    prev = _last_level.get(kind)
    rose = prev is not None and levels.index(level) > levels.index(prev)
    first_bad = prev is None and level != levels[0]
    now = time.time()
    cooled_down = now - _last_notify_at.get(kind, 0) > _COOLDOWN_S
    if (rose or first_bad) and cooled_down:
        notify(title, message)
        _last_notify_at[kind] = now
    _last_level[kind] = level


def check_and_notify() -> tuple[str, dict]:
    """Đọc nhiệt độ + RAM hiện tại, gửi thông báo nếu mức độ tăng lên. Trả về (thermal_state, ram_info)."""
    t = thermal.state()
    r = ram()
    _notify_on_rise("thermal", t, thermal.LEVELS, "🌡️ Máy đang nóng",
                    f"Nhiệt độ máy: {thermal.LABELS.get(t, t)}. Cân nhắc dừng train để máy nguội.")
    _notify_on_rise("ram", r["level"], RAM_LEVELS, "💾 RAM sắp đầy",
                    f"Đang dùng {r['percent']:.0f}% RAM ({r['used_gb']:.1f}/{r['total_gb']:.1f} GB).")
    return t, r
