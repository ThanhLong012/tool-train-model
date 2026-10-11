# tool-train-model

Giao diện web để train model object detection, tách riêng khỏi phần gán nhãn
(xem project `auto-label-studio`). Chỉ làm một việc: train, theo dõi tiến
trình, và cảnh báo khi máy nóng hoặc gần hết RAM.

## Chạy giao diện

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # cần Python >= 3.10 (rfdetr yêu cầu); máy này có python3.12 ở ~/.local/bin
pip install -r requirements.txt
python3 -m trainer.ui                                    # mở http://localhost:7861
```

| Tab | Làm gì |
|---|---|
| 🟡 YOLO11n | Chọn dataset `data.yaml` (định dạng Ultralytics YOLO) → chọn tham số → Bắt đầu / Dừng / Train tiếp; biểu đồ loss + mAP theo epoch |
| 🟢 RF-DETR Nano | Chọn thư mục dataset COCO (`train/`, `valid/`, `test/`, mỗi thư mục có `_annotations.coco.json` — format Roboflow export) → chọn tham số → Bắt đầu / Dừng / Train tiếp; biểu đồ loss + mAP theo epoch |

Banner trên cùng hiện **nhiệt độ máy** (đọc `ProcessInfo.thermalState` của macOS)
và **RAM đang dùng** (qua `psutil`), cập nhật mỗi 5 giây. Khi nhiệt độ lên mức
`serious`/`critical` hoặc RAM vượt 95%, máy gửi **thông báo hệ thống** (macOS
Notification Center) và — nếu bật ô "Tự dừng" (mặc định bật) — tự dừng lần
train đang chạy. Dừng giữa chừng chỉ mất epoch đang chạy dở, bấm "Train tiếp"
để chạy lại từ checkpoint gần nhất.

Chỉ train được **1 lần tại một thời điểm** (YOLO hoặc RF-DETR, không phải cả
hai) để máy không bị quá tải nhiệt/RAM khi chạy song song.

Train chạy thành tiến trình riêng (subprocess): đóng trình duyệt vẫn tiếp tục chạy.

Train không qua giao diện:
```bash
python3 -m trainer.yolo_train --data datasets/my_ds/data.yaml --name v1 --epochs 50
python3 -m trainer.yolo_train --resume --name v1

python3 -m trainer.rfdetr_train --data datasets/my_ds --name v1 --epochs 50
python3 -m trainer.rfdetr_train --name v1 --resume
```

## Cấu trúc

```
trainer/
  config.py       đường dẫn dùng chung
  thermal.py      đọc trạng thái nhiệt của macOS
  resources.py    đọc RAM, gửi thông báo hệ thống khi nóng / gần đầy RAM
  job.py          quản lý tiến trình train chạy riêng (subprocess), khoá 1 lần train/lúc
  yolo_train.py   train YOLO11n (chạy như tiến trình riêng)
  rfdetr_train.py train RF-DETR Nano (chạy như tiến trình riêng)
  ui.py           giao diện Gradio, 2 tab + banner tài nguyên máy
datasets/         mỗi dataset một thư mục:
                    - YOLO: có data.yaml (định dạng Ultralytics)
                    - RF-DETR: có train/valid/test, mỗi thư mục có _annotations.coco.json (COCO)
runs/
  yolo/<tên>/     weights/best.pt, results.csv, train.log
  rfdetr/<tên>/   last.ckpt, metrics.csv, train.log, train_meta.json (nhớ dataset gốc để resume)
weights/          trọng số khởi đầu (yolo11n.pt...) tự tải về lần đầu
```

## Vì sao tách riêng khỏi auto-label-studio

`auto-label-studio` làm toàn bộ vòng lặp gán nhãn (CVAT, xem dataset, camera
realtime...) — train chỉ là một tab trong đó. Repo này chỉ cần phần train,
nhưng train thêm một model thứ hai (RF-DETR, Apache-2.0 — dùng thương mại được,
khác YOLO là AGPL-3.0) với giao diện theo dõi giống nhau, nên tách thành
project riêng cho gọn, không phải mang theo CVAT/camera/dataset-QA không dùng tới.

Giấy phép: Ultralytics YOLO là AGPL-3.0; RF-DETR (Roboflow) là Apache-2.0.
