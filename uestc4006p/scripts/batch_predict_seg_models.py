from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
import time
import json

import cv2
import numpy as np
from ultralytics import YOLO

# ==============================
# 配置区
# ==============================
RUNS_ROOT = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\要使用的训练结果汇总")
SOURCE_DIR = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\test\seg")
OUT_ROOT = RUNS_ROOT / "_predict_exports" / "seg_direct"

# repo-relative defaults, overridable via environment variables
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = Path(os.environ.get("UESTC4006P_RUNS_ROOT", str(REPO_ROOT / "uestc4006p" / "runs"))).expanduser()
SOURCE_DIR = Path(os.environ.get("UESTC4006P_SEG_SOURCE", str(REPO_ROOT / "data" / "seg_samples"))).expanduser()
OUT_ROOT = Path(os.environ.get("UESTC4006P_SEG_OUT_ROOT", str(RUNS_ROOT / "_predict_exports" / "seg_direct"))).expanduser()

WHITELIST: list[str] = []
EXCLUDE: list[str] = []
EXCLUDE_KEYWORDS: list[str] = ["26"]  # 普通脚本默认跳过 yolo26 系列

DEVICE = 0
IMGSZ = 1024
CONF = 0.25
IOU = 0.50
MASK_THR = 0.50
ALPHA = 0.45
SAVE_TXT = True
SAVE_CONF = True

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
        if not p.name.startswith("train_seg_"):
            continue
        if WHITELIST and p.name not in WHITELIST:
            continue
        if p.name in EXCLUDE:
            continue
        if any(k in p.name for k in EXCLUDE_KEYWORDS):
            print(f"[SKIP] {p.name} -> 命中 EXCLUDE_KEYWORDS={EXCLUDE_KEYWORDS}，请改用 yolo26 专用脚本")
            continue
        if not (p / "weights" / "best.pt").exists():
            continue
        yield p


def overlay_mask_red(img_bgr: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = img_bgr.copy()
    red = np.zeros_like(img_bgr)
    red[:, :, 2] = 255
    m = mask_u8 > 0
    out[m] = (img_bgr[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)
    return out


def union_mask_from_result(seg_result, h: int, w: int, conf_thr: float, mask_thr: float) -> np.ndarray:
    if seg_result.masks is None:
        return np.zeros((h, w), dtype=np.uint8)

    masks = seg_result.masks.data.detach().cpu().numpy()
    if seg_result.boxes is not None and len(seg_result.boxes) == masks.shape[0]:
        scores = seg_result.boxes.conf.detach().cpu().numpy()
        keep = scores >= conf_thr
        masks = masks[keep] if keep.any() else masks[:0]

    if masks.shape[0] == 0:
        return np.zeros((h, w), dtype=np.uint8)

    union = np.max(masks, axis=0)
    union = (union > mask_thr).astype(np.uint8) * 255
    union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)
    return union


def main() -> None:
    ensure_dir(OUT_ROOT)
    images = list_images(SOURCE_DIR)
    run_dirs = list(iter_run_dirs(RUNS_ROOT))

    if not images:
        raise RuntimeError(f"{SOURCE_DIR} 下没有图片")
    if not run_dirs:
        raise RuntimeError(f"{RUNS_ROOT} 下没有可用的非 yolo26 seg 权重")

    summary: list[dict] = []

    print(f"[INFO] seg 图片数量: {len(images)}")
    print("[INFO] 将运行这些非 yolo26 seg 模型:")
    for p in run_dirs:
        print(" -", p.name)

    for run_dir in run_dirs:
        weight = run_dir / "weights" / "best.pt"
        save_dir = OUT_ROOT / run_dir.name
        label_dir = save_dir / "labels"
        ensure_dir(save_dir)
        ensure_dir(label_dir)

        print(f"\n{'=' * 90}")
        print(f"[START] {run_dir.name}")
        print(f"[WEIGHT] {weight}")
        print(f"[OUT]    {save_dir}")

        model = YOLO(str(weight))
        t0 = time.perf_counter()
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] 读图失败: {img_path}")
                continue
            h, w = img.shape[:2]
            res = model.predict(
                source=img,
                device=DEVICE,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                verbose=False,
                retina_masks=True,
                save_txt=False,
                save_conf=False,
            )[0]

            mask_u8 = union_mask_from_result(res, h, w, CONF, MASK_THR)
            overlay = overlay_mask_red(img, mask_u8, alpha=ALPHA)

            stem = img_path.stem
            cv2.imwrite(str(save_dir / f"{stem}__mask.png"), mask_u8)
            cv2.imwrite(str(save_dir / f"{stem}__overlay.jpg"), overlay)

            txt_path = label_dir / f"{stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                if res.masks is not None and res.boxes is not None:
                    polys = res.masks.xy
                    clses = res.boxes.cls.detach().cpu().numpy().astype(int)
                    confs = res.boxes.conf.detach().cpu().numpy()
                    for poly, cls_id, score in zip(polys, clses, confs):
                        if score < CONF:
                            continue
                        pts = poly.copy()
                        pts[:, 0] /= w
                        pts[:, 1] /= h
                        pts = np.clip(pts, 0.0, 1.0)
                        flat = " ".join(f"{v:.6f}" for v in pts.reshape(-1))
                        if SAVE_CONF:
                            f.write(f"{cls_id} {flat} {float(score):.6f}\n")
                        else:
                            f.write(f"{cls_id} {flat}\n")

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
            "mask_thr": MASK_THR,
            "device": DEVICE,
            "elapsed_sec": dt,
            "exclude_keywords": EXCLUDE_KEYWORDS,
        }
        with open(save_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        summary.append(meta)
        print(f"[DONE] {run_dir.name}  elapsed={dt:.2f}s")

    with open(OUT_ROOT / "summary_seg_predict.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n全部完成，输出目录: {OUT_ROOT}")


if __name__ == "__main__":
    main()
