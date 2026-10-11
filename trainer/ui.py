"""Giao diện web: train YOLO11n hoặc RF-DETR Nano, theo dõi tiến trình, nhiệt độ máy và RAM.

Chạy từ thư mục gốc project:
    python3 -m trainer.ui            # mở http://localhost:7861

Tab 🟡 YOLO11n     — chọn dataset (data.yaml), tham số train, biểu đồ loss/mAP cập nhật trực tiếp
Tab 🟢 RF-DETR Nano — chọn thư mục dataset COCO (train/valid/test), tham số train, biểu đồ loss/mAP
Banner trên cùng    — nhiệt độ máy và RAM hiện tại, tự dừng train + gửi thông báo macOS khi quá nóng / sắp đầy RAM
"""

from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # không gửi thống kê sử dụng ra ngoài

import re
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
from matplotlib.figure import Figure

from . import job, resources, thermal
from .config import DATASETS_DIR, ROOT, RFDETR_RUNS_DIR, YOLO_RUNS_DIR


def rel(p: Path) -> str:
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", (name or "").strip())


# ---------------------------------------------------------------------------
# Tải dataset lên dưới dạng file .zip, tự giải nén vào datasets/
# ---------------------------------------------------------------------------

def _unique_dataset_dir(stem: str) -> Path:
    stem = sanitize_name(stem) or "dataset"
    d = DATASETS_DIR / stem
    i = 2
    while d.exists():
        d = DATASETS_DIR / f"{stem}_{i}"
        i += 1
    return d


def _safe_extract(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest.resolve()) + os.sep) and target != dest.resolve():
                raise gr.Error("File zip chứa đường dẫn không an toàn, đã huỷ giải nén")
        zf.extractall(dest)


def _flatten_single_subdir(dest: Path) -> Path:
    """Nếu zip chỉ bọc 1 thư mục con duy nhất (thường gặp khi zip cả folder), coi thư mục đó là root."""
    entries = [p for p in dest.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dest


def _dezip(zip_file) -> Path:
    """Giải nén zip vào datasets/<tên duy nhất>/, bỏ 1 lớp thư mục bọc ngoài nếu có
    (tránh kiểu datasets/my_ds/my_ds/... khi người dùng zip cả folder dataset)."""
    if not zip_file:
        raise gr.Error("Chưa chọn file zip")
    src = Path(zip_file if isinstance(zip_file, str) else zip_file.name)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=DATASETS_DIR, prefix=".extract_"))
    try:
        _safe_extract(src, staging)
        content = _flatten_single_subdir(staging)
        dest = _unique_dataset_dir(content.name if content != staging else src.stem)
        shutil.move(str(content), str(dest))
        return dest
    except zipfile.BadZipFile:
        raise gr.Error("File không phải .zip hợp lệ")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Biểu đồ (matplotlib, không dùng gr.LinePlot — xem ghi chú trong dataset Gold Studio
# gốc: BarPlot/LinePlot của Gradio 4.44 lỗi JS khi nằm trong tab đang ẩn)
# ---------------------------------------------------------------------------

def _line_fig(df: pd.DataFrame, cols: dict[str, str], title: str, ylim=None) -> Figure:
    fig = Figure(figsize=(6, 3.4), dpi=100)
    ax = fig.add_subplot()
    for c, label in cols.items():
        if c in df:
            ax.plot(df["epoch"], df[c], label=label, linewidth=1.6)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("epoch")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Banner tài nguyên máy (chung cho mọi tab)
# ---------------------------------------------------------------------------

def poll_resources():
    t, r = resources.check_and_notify()
    return (f"Nhiệt độ máy: **{thermal.LABELS.get(t, t)}**  ·  "
            f"RAM: **{resources.RAM_LABELS[r['level']]}** ({r['percent']:.0f}% — "
            f"{r['used_gb']:.1f}/{r['total_gb']:.1f} GB)")


def _start(kind: str, runs_dir: Path, cmd: list[str], name: str, total_epochs: int | None):
    other = job.any_running()
    if other and not (other.kind == kind and other.name == name):
        raise gr.Error(f"Đang train '{other.name}' ({other.kind}). Dừng lần train đó trước khi bắt đầu lần khác.")
    j = job.get(kind, runs_dir)
    if j.running():
        raise gr.Error(f"Đang train '{j.name}'. Dừng lần train đó trước.")
    return j.launch(cmd, name, total_epochs)


def _autostop_check(j: job.Job, autostop: bool) -> None:
    if not (j.running() and autostop and not j.stop_at):
        return
    t = thermal.state()
    r = resources.ram()
    if thermal.at_least(t, "serious"):
        j.stop(f"Tự dừng vì máy {thermal.LABELS[t]}")
    elif r["level"] == "rat_cao":
        j.stop(f"Tự dừng vì RAM gần đầy ({r['percent']:.0f}%)")


# ---------------------------------------------------------------------------
# Tab YOLO11n
# ---------------------------------------------------------------------------

def find_yolo_datasets() -> list[str]:
    return [rel(p) for p in sorted(DATASETS_DIR.glob("*/data.yaml"))]


def yolo_upload_zip(zip_file):
    dest = _dezip(zip_file)
    root = _flatten_single_subdir(dest)
    yamls = sorted(root.rglob("data.yaml"))
    if not yamls:
        raise gr.Error(f"Không tìm thấy data.yaml trong file zip (đã giải nén vào {rel(dest)})")
    chosen = rel(yamls[0])
    choices = sorted(set(find_yolo_datasets()) | {chosen})
    return gr.update(choices=choices, value=chosen)


def yolo_start(data_yaml, name, epochs, imgsz, batch, device, fraction, patience):
    name = sanitize_name(name)
    if not name:
        raise gr.Error("Đặt tên cho lần train")
    if not data_yaml:
        raise gr.Error("Chọn dataset (data.yaml)")
    if (YOLO_RUNS_DIR / name / "weights" / "last.pt").exists():
        raise gr.Error(f"'{name}' đã có kết quả. Đổi tên khác, hoặc bấm 'Train tiếp'.")
    cmd = [sys.executable, "-m", "trainer.yolo_train", "--data", str(ROOT / data_yaml), "--name", name,
           "--epochs", str(int(epochs)), "--imgsz", str(int(imgsz)), "--batch", str(int(batch)),
           "--device", device, "--fraction", str(fraction), "--patience", str(int(patience))]
    return _start("yolo", YOLO_RUNS_DIR, cmd, name, int(epochs))


def yolo_resume(name):
    name = (name or "").strip()
    if not (YOLO_RUNS_DIR / name / "weights" / "last.pt").exists():
        raise gr.Error(f"'{name}' chưa có last.pt (chưa xong epoch nào) — không train tiếp được")
    cmd = [sys.executable, "-m", "trainer.yolo_train", "--resume", "--name", name]
    return _start("yolo", YOLO_RUNS_DIR, cmd, name, None)


def yolo_stop():
    job.get("yolo", YOLO_RUNS_DIR).stop()


def yolo_read_results(name: str) -> pd.DataFrame:
    if not name:
        return pd.DataFrame()
    f = YOLO_RUNS_DIR / name / "results.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    return df


def yolo_progress(log_path: Path) -> str:
    try:
        with log_path.open("rb") as f:
            f.seek(max(0, log_path.stat().st_size - 20000))
            tail = f.read().decode("utf-8", "ignore")
    except FileNotFoundError:
        return ""
    for line in reversed(job.split_updates(tail)):
        m = re.search(r"^\s*(\d+)/(\d+)\s+[\d.]+G.*?(\d+)/(\d+)\s+([\d.]+(?:s/it|it/s))\s+([\d:.]+s?)<([\d:.]+s?)",
                      line)
        if m:
            ep, eps, b, bs, speed, spent, left = m.groups()
            return f"Epoch **{ep}/{eps}** · batch {b}/{bs} ({int(b) * 100 // int(bs)}%) · {speed} · còn ~{left} cho epoch này"
    return ""


def yolo_charts(df: pd.DataFrame):
    if df.empty:
        return None, None
    loss_cols = {"train/box_loss": "box", "train/cls_loss": "cls", "train/dfl_loss": "dfl",
                 "val/box_loss": "val box", "val/cls_loss": "val cls"}
    met_cols = {"metrics/mAP50(B)": "mAP50", "metrics/mAP50-95(B)": "mAP50-95",
                "metrics/precision(B)": "Precision", "metrics/recall(B)": "Recall"}
    return (_line_fig(df, loss_cols, "Loss (càng thấp càng tốt)"),
            _line_fig(df, met_cols, "Điểm trên tập val (càng cao càng tốt)", (0, 1)))


_yolo_chart_key: list = [None]


def yolo_poll(watch_name: str, autostop: bool):
    j = job.get("yolo", YOLO_RUNS_DIR)
    j.poll_autostop()
    running = j.running()
    _autostop_check(j, autostop)
    name = j.name if running else watch_name
    df = yolo_read_results(name)
    if running:
        state = "⏳ Đang dừng..." if j.stop_at else "🏃 Đang train"
        prog = yolo_progress(j.log_path) or "Đang chuẩn bị (nạp dữ liệu, tải model)..."
    elif j.proc is not None and j.name == name:
        code = j.proc.returncode
        state = "✅ Đã xong" if code == 0 else f"⏹ Đã dừng (mã {code})"
        prog = j.note or ""
    else:
        state, prog = "💤 Không có tiến trình train", ""
    best = ""
    if not df.empty and "metrics/mAP50(B)" in df:
        i = df["metrics/mAP50(B)"].idxmax()
        best = (f" · tốt nhất: epoch {int(df.epoch[i])}, mAP50 **{df['metrics/mAP50(B)'][i]:.3f}**, "
                f"mAP50-95 {df['metrics/mAP50-95(B)'][i]:.3f} ({len(df)} epoch đã xong)")
    status = f"**{state}** — `{name or '—'}`{best}\n\n{prog}"
    key = (name, len(df))
    if key == _yolo_chart_key[0]:
        losses = mets = gr.update()
    else:
        losses, mets = yolo_charts(df)
        _yolo_chart_key[0] = key
    log_path = j.log_path if running or j.name == name else YOLO_RUNS_DIR / name / "train.log"
    return status, losses, mets, job.log_tail(log_path)


# ---------------------------------------------------------------------------
# Tab RF-DETR Nano
# ---------------------------------------------------------------------------

def find_rfdetr_datasets() -> list[str]:
    return [rel(p) for p in sorted(DATASETS_DIR.glob("*")) if (p / "train").is_dir()]


def rfdetr_upload_zip(zip_file):
    dest = _dezip(zip_file)
    root = _flatten_single_subdir(dest)
    if not (root / "train").is_dir():
        raise gr.Error(f"Không tìm thấy thư mục train/ trong file zip (đã giải nén vào {rel(dest)})")
    chosen = rel(root)
    choices = sorted(set(find_rfdetr_datasets()) | {chosen})
    return gr.update(choices=choices, value=chosen)


def rfdetr_start(data_dir, name, epochs, batch, grad_accum, lr, device):
    name = sanitize_name(name)
    if not name:
        raise gr.Error("Đặt tên cho lần train")
    if not data_dir:
        raise gr.Error("Chọn thư mục dataset (COCO: train/valid/test)")
    if (RFDETR_RUNS_DIR / name / "last.ckpt").exists():
        raise gr.Error(f"'{name}' đã có kết quả. Đổi tên khác, hoặc bấm 'Train tiếp'.")
    cmd = [sys.executable, "-m", "trainer.rfdetr_train", "--data", str(ROOT / data_dir), "--name", name,
           "--epochs", str(int(epochs)), "--batch", str(int(batch)), "--grad-accum", str(int(grad_accum)),
           "--lr", str(lr), "--device", device]
    return _start("rfdetr", RFDETR_RUNS_DIR, cmd, name, int(epochs))


def rfdetr_resume(name):
    name = (name or "").strip()
    if not (RFDETR_RUNS_DIR / name / "last.ckpt").exists():
        raise gr.Error(f"'{name}' chưa có last.ckpt (chưa xong epoch nào) — không train tiếp được")
    cmd = [sys.executable, "-m", "trainer.rfdetr_train", "--name", name, "--resume"]
    return _start("rfdetr", RFDETR_RUNS_DIR, cmd, name, None)


def rfdetr_stop():
    job.get("rfdetr", RFDETR_RUNS_DIR).stop()


def rfdetr_read_results(name: str) -> pd.DataFrame:
    if not name:
        return pd.DataFrame()
    f = RFDETR_RUNS_DIR / name / "metrics.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    if "epoch" not in df:
        return pd.DataFrame()
    agg = {c: "mean" if c.startswith("train/") else "max" for c in df.columns if c not in ("epoch", "step")}
    return df.groupby("epoch", as_index=False).agg(agg)


def rfdetr_charts(df: pd.DataFrame):
    if df.empty:
        return None, None
    loss_cols = {"train/loss": "loss"}
    met_cols = {"val/mAP_50": "mAP50", "val/mAP_50_95": "mAP50-95", "val/mAR": "mAR"}
    return (_line_fig(df, loss_cols, "Loss (càng thấp càng tốt)"),
            _line_fig(df, met_cols, "Điểm trên tập val (càng cao càng tốt)", (0, 1)))


_rfdetr_chart_key: list = [None]


def rfdetr_poll(watch_name: str, autostop: bool):
    j = job.get("rfdetr", RFDETR_RUNS_DIR)
    j.poll_autostop()
    running = j.running()
    _autostop_check(j, autostop)
    name = j.name if running else watch_name
    df = rfdetr_read_results(name)
    if running:
        state = "⏳ Đang dừng..." if j.stop_at else "🏃 Đang train"
        n_done = int(df["epoch"].max()) + 1 if not df.empty else 0
        total = f"/{j.total_epochs}" if j.total_epochs else ""
        prog = f"Epoch đã xong: **{n_done}{total}**" if n_done or total else "Đang chuẩn bị (nạp dữ liệu, tải model)..."
    elif j.proc is not None and j.name == name:
        code = j.proc.returncode
        state = "✅ Đã xong" if code == 0 else f"⏹ Đã dừng (mã {code})"
        prog = j.note or ""
    else:
        state, prog = "💤 Không có tiến trình train", ""
    best = ""
    if not df.empty and "val/mAP_50" in df:
        i = df["val/mAP_50"].idxmax()
        best = f" · tốt nhất: epoch {int(df.epoch[i])}, mAP50 **{df['val/mAP_50'][i]:.3f}**"
        if "val/mAP_50_95" in df:
            best += f", mAP50-95 {df['val/mAP_50_95'][i]:.3f}"
        best += f" ({len(df)} epoch đã xong)"
    status = f"**{state}** — `{name or '—'}`{best}\n\n{prog}"
    key = (name, len(df))
    if key == _rfdetr_chart_key[0]:
        losses = mets = gr.update()
    else:
        losses, mets = rfdetr_charts(df)
        _rfdetr_chart_key[0] = key
    log_path = j.log_path if running or j.name == name else RFDETR_RUNS_DIR / name / "train.log"
    return status, losses, mets, job.log_tail(log_path)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build() -> gr.Blocks:
    yolo_datasets = find_yolo_datasets()
    rfdetr_datasets = find_rfdetr_datasets()

    with gr.Blocks(title="Tool Train Model", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏋️ Tool Train Model\nTrain YOLO11n hoặc RF-DETR Nano, theo dõi tiến trình trực tiếp.")
        resource_banner = gr.Markdown()
        gr.Markdown("Chỉ chạy **1 lần train tại một thời điểm** (YOLO hoặc RF-DETR) để máy không bị quá tải.")

        with gr.Tab("🟡 YOLO11n"):
            with gr.Row():
                with gr.Column(scale=1):
                    y_ds = gr.Dropdown(yolo_datasets, value=yolo_datasets[0] if yolo_datasets else None,
                                       allow_custom_value=True,
                                       label="Dataset (data.yaml) — chọn hoặc dán đường dẫn")
                    y_refresh = gr.Button("🔄 Quét lại datasets/")
                    y_zip = gr.File(label="...hoặc tải lên file .zip (tự giải nén vào datasets/)",
                                    file_types=[".zip"], type="filepath")
                    with gr.Accordion("📦 File zip cần có gì?", open=False):
                        gr.Markdown(
                            "Định dạng Ultralytics YOLO. Zip cả thư mục dataset (có bọc 1 folder ngoài hay không "
                            "đều được, tool tự bỏ lớp bọc), cấu trúc bên trong:\n\n"
                            "```\n"
                            "ten_dataset/\n"
                            "  data.yaml          # path, train, val (và test nếu có), names\n"
                            "  images/\n"
                            "    train/xxx.jpg\n"
                            "    val/xxx.jpg\n"
                            "  labels/            # mỗi ảnh 1 file .txt cùng tên, cùng cấu trúc thư mục với images/\n"
                            "    train/xxx.txt    # mỗi dòng: <class_id> <x_center> <y_center> <w> <h> (toạ độ 0-1)\n"
                            "    val/xxx.txt\n"
                            "```\n\n"
                            "`train`/`val` trong `data.yaml` có thể là đường dẫn thư mục ảnh hoặc file `.txt` "
                            "liệt kê đường dẫn ảnh. Ảnh không có file nhãn tương ứng được coi là ảnh nền (không có "
                            "vật thể) — vẫn hợp lệ nếu đó là chủ ý. Xuất từ CVAT: chọn format \"Ultralytics YOLO "
                            "Detection 1.0\"; xuất từ Roboflow: chọn format \"YOLOv11\" hoặc tương đương."
                        )
                    y_name = gr.Textbox(value=f"yolo11n-{datetime.now():%m%d}", label="Tên lần train")
                    y_epochs = gr.Slider(1, 300, value=50, step=1, label="Epochs")
                    with gr.Row():
                        y_imgsz = gr.Dropdown([640, 800, 960, 1280], value=640, label="imgsz")
                        y_batch = gr.Dropdown([2, 4, 8, 16], value=8, label="batch")
                    y_device = gr.Radio(["mps", "cpu"], value="mps", label="Thiết bị (mps = GPU Apple)")
                    with gr.Accordion("Nâng cao", open=False):
                        y_frac = gr.Slider(0.05, 1.0, value=1.0, step=0.05, label="Dùng bao nhiêu phần tập train")
                        y_pat = gr.Slider(0, 50, value=15, step=1, label="Dừng sớm nếu N epoch không tiến bộ")
                    y_auto = gr.Checkbox(True, label="Tự dừng khi máy nóng (serious+) hoặc RAM gần đầy")
                    with gr.Row():
                        y_start = gr.Button("▶ Bắt đầu", variant="primary")
                        y_stop = gr.Button("⏹ Dừng", variant="stop")
                    y_resume = gr.Button("⏯ Train tiếp (từ epoch đã xong)")
                    gr.Markdown("Dừng giữa chừng chỉ mất epoch đang chạy dở. Train vẫn chạy khi đóng trình duyệt.")
                with gr.Column(scale=2):
                    y_status = gr.Markdown()
                    with gr.Row():
                        y_loss = gr.Plot(label="Loss")
                        y_met = gr.Plot(label="Điểm")
                    y_log = gr.Textbox(label="Log", lines=12, max_lines=12, autoscroll=True)

            y_timer = gr.Timer(3)
            y_timer.tick(yolo_poll, [y_name, y_auto], [y_status, y_loss, y_met, y_log])
            y_start.click(yolo_start, [y_ds, y_name, y_epochs, y_imgsz, y_batch, y_device, y_frac, y_pat], y_name)
            y_resume.click(yolo_resume, y_name, y_name)
            y_stop.click(yolo_stop, None, None)
            y_refresh.click(lambda: gr.update(choices=find_yolo_datasets()), None, y_ds)
            y_zip.upload(yolo_upload_zip, y_zip, y_ds)

        with gr.Tab("🟢 RF-DETR Nano"):
            with gr.Row():
                with gr.Column(scale=1):
                    r_ds = gr.Dropdown(rfdetr_datasets, value=rfdetr_datasets[0] if rfdetr_datasets else None,
                                       allow_custom_value=True,
                                       label="Thư mục dataset — chọn hoặc dán đường dẫn")
                    r_refresh = gr.Button("🔄 Quét lại datasets/")
                    r_zip = gr.File(label="...hoặc tải lên file .zip (tự giải nén vào datasets/)",
                                    file_types=[".zip"], type="filepath")
                    with gr.Accordion("📦 File zip cần có gì?", open=False):
                        gr.Markdown(
                            "Định dạng COCO, kiểu Roboflow export. Zip cả thư mục dataset (có bọc 1 folder ngoài "
                            "hay không đều được, tool tự bỏ lớp bọc), cấu trúc bên trong:\n\n"
                            "```\n"
                            "ten_dataset/\n"
                            "  train/\n"
                            "    _annotations.coco.json   # nhãn: bbox + category cho ảnh trong thư mục này\n"
                            "    xxx.jpg\n"
                            "  valid/\n"
                            "    _annotations.coco.json\n"
                            "    xxx.jpg\n"
                            "  test/                      # không bắt buộc\n"
                            "    _annotations.coco.json\n"
                            "    xxx.jpg\n"
                            "```\n\n"
                            "Chỉ cần có thư mục `train/` là nhận ra được dataset; thiếu `valid/` thì không chấm "
                            "được điểm trên tập val trong lúc train. Xuất từ Roboflow: chọn format \"COCO\"; "
                            "xuất từ CVAT: chọn format \"COCO 1.0\" rồi tự chia ảnh vào `train/`/`valid/`."
                        )
                    r_name = gr.Textbox(value=f"rfdetr-nano-{datetime.now():%m%d}", label="Tên lần train")
                    r_epochs = gr.Slider(1, 300, value=50, step=1, label="Epochs")
                    with gr.Row():
                        r_batch = gr.Dropdown([2, 4, 8, 16], value=4, label="batch_size")
                        r_accum = gr.Dropdown([1, 2, 4, 8], value=4, label="grad_accum_steps")
                    r_device = gr.Radio(["mps", "cpu"], value="mps", label="Thiết bị (mps = GPU Apple)")
                    with gr.Accordion("Nâng cao", open=False):
                        r_lr = gr.Number(value=1e-4, label="Learning rate")
                    r_auto = gr.Checkbox(True, label="Tự dừng khi máy nóng (serious+) hoặc RAM gần đầy")
                    with gr.Row():
                        r_start = gr.Button("▶ Bắt đầu", variant="primary")
                        r_stop = gr.Button("⏹ Dừng", variant="stop")
                    r_resume = gr.Button("⏯ Train tiếp (từ epoch đã xong)")
                    gr.Markdown("Dừng giữa chừng chỉ mất epoch đang chạy dở. Train vẫn chạy khi đóng trình duyệt.")
                with gr.Column(scale=2):
                    r_status = gr.Markdown()
                    with gr.Row():
                        r_loss = gr.Plot(label="Loss")
                        r_met = gr.Plot(label="Điểm")
                    r_log = gr.Textbox(label="Log", lines=12, max_lines=12, autoscroll=True)

            r_timer = gr.Timer(3)
            r_timer.tick(rfdetr_poll, [r_name, r_auto], [r_status, r_loss, r_met, r_log])
            r_start.click(rfdetr_start, [r_ds, r_name, r_epochs, r_batch, r_accum, r_lr, r_device], r_name)
            r_resume.click(rfdetr_resume, r_name, r_name)
            r_stop.click(rfdetr_stop, None, None)
            r_refresh.click(lambda: gr.update(choices=find_rfdetr_datasets()), None, r_ds)
            r_zip.upload(rfdetr_upload_zip, r_zip, r_ds)

        banner_timer = gr.Timer(5)
        banner_timer.tick(poll_resources, None, resource_banner)
        demo.load(poll_resources, None, resource_banner)
    return demo


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    YOLO_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RFDETR_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    build().queue().launch(server_name=a.host, server_port=a.port)


if __name__ == "__main__":
    main()
