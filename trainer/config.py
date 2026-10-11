"""Đường dẫn dùng chung cho toàn bộ package."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
RUNS_DIR = ROOT / "runs"              # runs/yolo/<tên>/ và runs/rfdetr/<tên>/
WEIGHTS_DIR = ROOT / "weights"        # trọng số khởi đầu tải về (yolo11n.pt...)
CACHE_DIR = ROOT / ".cache"

YOLO_RUNS_DIR = RUNS_DIR / "yolo"
RFDETR_RUNS_DIR = RUNS_DIR / "rfdetr"

YOLO_MODEL = "yolo11n.pt"
RFDETR_VARIANT = "nano"
