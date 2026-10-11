"""Train RF-DETR Nano trên một dataset định dạng COCO (train/valid/test, mỗi thư mục có
_annotations.coco.json — đúng format Roboflow export). Kết quả lưu ở runs/rfdetr/<name>/.

UI gọi file này như một tiến trình riêng, nhưng cũng chạy thẳng được:
    python -m trainer.rfdetr_train --data datasets/my_ds --name v1 --epochs 50
    python -m trainer.rfdetr_train --data datasets/my_ds --name v1 --resume   # train tiếp từ checkpoint gần nhất

RF-DETR tự lưu checkpoint mỗi epoch (last.ckpt) nên --resume tiếp tục được.
Cần cài `pip install "rfdetr[train]"` (yêu cầu Python >= 3.10).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RFDETR_RUNS_DIR

_META = "train_meta.json"  # lưu lại dataset_dir gốc để --resume không cần nhập lại


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="Thư mục dataset COCO (train/valid/test); bắt buộc trừ khi --resume")
    ap.add_argument("--name", required=True, help="Tên lần train (thư mục runs/rfdetr/<name>)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4, help="batch_size")
    ap.add_argument("--grad-accum", type=int, default=4, help="grad_accum_steps (batch hiệu dụng = batch * grad-accum)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="mps", help="mps (GPU Apple) / cpu / cuda")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    run_dir = RFDETR_RUNS_DIR / args.name
    meta_path = run_dir / _META

    if args.resume:
        if not (run_dir / "last.ckpt").exists():
            raise SystemExit(f"Không có {run_dir / 'last.ckpt'} để train tiếp (chưa xong epoch nào?)")
        if not meta_path.exists():
            raise SystemExit(f"Không có {meta_path} — không rõ dataset gốc để train tiếp")
        dataset_dir = json.loads(meta_path.read_text())["dataset_dir"]
    else:
        if not args.data or not args.data.exists():
            raise SystemExit(f"Không tìm thấy thư mục dataset: {args.data}")
        dataset_dir = str(args.data.resolve())
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"dataset_dir": dataset_dir}))

    from rfdetr import RFDETRNano  # import muộn để --help chạy nhanh

    model = RFDETRNano()
    model.train(
        dataset_dir=dataset_dir, output_dir=str(run_dir),
        epochs=args.epochs, batch_size=args.batch, grad_accum_steps=args.grad_accum,
        lr=args.lr, device=args.device, resume="last" if args.resume else None,
    )


if __name__ == "__main__":
    main()
