"""Train YOLO11n trên một dataset (định dạng Ultralytics: data.yaml). Kết quả lưu ở runs/yolo/<name>/.

UI gọi file này như một tiến trình riêng, nhưng cũng chạy thẳng được:
    python -m trainer.yolo_train --data datasets/my_ds/data.yaml --name v1 --epochs 50
    python -m trainer.yolo_train --resume --name v1      # train tiếp từ epoch đã xong gần nhất

Dừng giữa chừng (Ctrl+C hoặc nút Dừng trên UI) chỉ mất epoch đang chạy dở;
last.pt được lưu sau mỗi epoch nên --resume tiếp tục được.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import YOLO_MODEL, YOLO_RUNS_DIR, WEIGHTS_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="Đường dẫn data.yaml")
    ap.add_argument("--name", required=True, help="Tên lần train (thư mục runs/yolo/<name>)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="mps", help="mps (GPU Apple) / cpu / 0 (GPU NVIDIA)")
    ap.add_argument("--fraction", type=float, default=1.0, help="Dùng bao nhiêu phần tập train (0-1)")
    ap.add_argument("--patience", type=int, default=15, help="Dừng sớm nếu N epoch không tiến bộ")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO  # import muộn để --help chạy nhanh

    run_dir = YOLO_RUNS_DIR / args.name
    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"Không có {last} để train tiếp (chưa xong epoch nào?)")
        YOLO(str(last)).train(resume=True)
        return

    if not args.data or not args.data.exists():
        raise SystemExit(f"Không tìm thấy data.yaml: {args.data}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    start = str(WEIGHTS_DIR / YOLO_MODEL)  # Ultralytics tự tải về đây nếu chưa có
    YOLO(start).train(
        data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, fraction=args.fraction, patience=args.patience, workers=args.workers,
        project=str(YOLO_RUNS_DIR), name=args.name, exist_ok=True, pretrained=True,
    )


if __name__ == "__main__":
    main()
