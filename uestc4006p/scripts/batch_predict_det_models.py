from __future__ import annotations

from pathlib import Path
from typing import Iterable
import time
import json

from ultralytics import YOLO

# ==============================
# 配置区
# ==============================
RUNS_ROOT = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\要使用的训练结果汇总")
SOURCE_DIR = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\test\det")
OUT_ROOT = RUNS_ROOT / "_predict_exports" / "det_direct"

# 只扫描 train_det_ 开头目录；若想只跑部分模型，在这里填目录名
WHITELIST: list[str] = []
EXCLUDE: list[str] = []

DEVICE = 0            # 没有 GPU 改成 "cpu"
IMGSZ = 1024
CONF = 0.25
IOU = 0.50
SAVE_TXT = True
SAVE_CONF = True
LINE_WIDTH = 2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def list_images(src: Path) -> list[Path]:
    if not src.exists():
        raise FileNotFoundError(f"source 不存在: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"source 不是文件夹: {src}")
    return sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def iter_run_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.startswith("train_det_"):
            continue
        if WHITELIST and p.name not in WHITELIST:
            continue
        if p.name in EXCLUDE:
            continue
        if not (p / "weights" / "best.pt").exists():
            continue
        yield p


def main() -> None:
    ensure_dir(OUT_ROOT)
    images = list_images(SOURCE_DIR)
    run_dirs = list(iter_run_dirs(RUNS_ROOT))

    if not images:
        raise RuntimeError(f"{SOURCE_DIR} 下没有图片")
    if not run_dirs:
        raise RuntimeError(f"{RUNS_ROOT} 下没有可用 det 权重")

    summary: list[dict] = []

    print(f"[INFO] det 图片数量: {len(images)}")
    print("[INFO] 将运行这些 det 模型:")
    for p in run_dirs:
        print(" -", p.name)

    for run_dir in run_dirs:
        weight = run_dir / "weights" / "best.pt"
        save_dir = OUT_ROOT / run_dir.name
        ensure_dir(save_dir)

        print(f"\n{'=' * 90}")
        print(f"[START] {run_dir.name}")
        print(f"[WEIGHT] {weight}")
        print(f"[OUT]    {save_dir}")

        t0 = time.perf_counter()
        model = YOLO(str(weight))
        results = model.predict(
            source=str(SOURCE_DIR),
            device=DEVICE,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            save=True,
            save_txt=SAVE_TXT,
            save_conf=SAVE_CONF,
            verbose=False,
            project=str(OUT_ROOT),
            name=run_dir.name,
            exist_ok=True,
            line_width=LINE_WIDTH,
        )
        dt = time.perf_counter() - t0

        meta = {
            "run_name": run_dir.name,
            "weight": str(weight),
            "source": str(SOURCE_DIR),
            "save_dir": str(save_dir),
            "num_images": len(images),
            "imgsz": IMGSZ,
            "conf": CONF,
            "iou": IOU,
            "device": DEVICE,
            "elapsed_sec": dt,
            "num_results": len(results),
        }
        with open(save_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        summary.append(meta)
        print(f"[DONE] {run_dir.name}  elapsed={dt:.2f}s")

    with open(OUT_ROOT / "summary_det_predict.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n全部完成，输出目录: {OUT_ROOT}")


if __name__ == "__main__":
    main()
