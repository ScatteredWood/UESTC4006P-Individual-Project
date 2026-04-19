from __future__ import annotations

import json
import sys
import importlib.util
import time
from pathlib import Path
from typing import Iterable

import cv2
from ultralytics import YOLO

# ==============================
# 配置区
# ==============================
REPO_ROOT = Path(r"E:\repositories\ultralytics")
CASCADE_SCRIPT = REPO_ROOT / "uestc4006p" / "scripts" / "cascade_infer_detseg.py"

RUNS_ROOT = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\要使用的训练结果汇总")
SOURCE_DIR = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\test\seg")
OUT_ROOT = RUNS_ROOT / "_predict_exports" / "cascade_standard"

DET_WHITELIST: list[str] = []
SEG_WHITELIST: list[str] = []
DET_EXCLUDE_KEYWORDS: list[str] = ["26"]
SEG_EXCLUDE_KEYWORDS: list[str] = ["26"]
PAIR_MODE = "cartesian"   # cartesian / zip

ARGS = {
    "det_conf": 0.15,
    "det_iou": 0.50,
    "det_imgsz": 0,
    "seg_conf": 0.10,
    "seg_thr": 0.30,
    "seg_iou": 0.50,
    "seg_imgsz": 1280,
    "pad_ratio": 0.15,
    "pad_min": 16,
    "max_rois": 80,
    "det_classes": None,
    "max_area_ratio": 0.60,
    "tile_min_side": 1400,
    "tile": 1280,
    "overlap": 256,
    "no_tile": False,
    "big_classes": [2, 3],
    "big_seg_conf": 0.08,
    "big_seg_thr": 0.25,
    "big_seg_iou": 0.50,
    "big_seg_imgsz": 1280,
    "big_force_tile": True,
    "big_tile": 1280,
    "big_overlap": 384,
    "big_clahe": False,
    "clahe_clip": 2.0,
    "clahe_grid": 8,
    "post_open": 0,
    "post_close": 3,
    "post_min_area": 25,
    "roi_v3": True,
    "d20_class_id": 2,
    "d20_pad_ratio": 0.20,
    "d20_pad_min": 24,
    "d20_seg_conf": 0.08,
    "d20_seg_thr": 0.22,
    "d20_seg_iou": 0.50,
    "d20_seg_imgsz": 1280,
    "d20_clahe": False,
    "adaptive_tile": True,
    "adaptive_min_tile": 768,
    "adaptive_max_tile": 1536,
    "adaptive_target_long_tiles": 4,
    "tile_overlap_ratio": 0.30,
    "tile_fusion": "max",
    "tile_fusion_thr": 0.30,
    "highres_seg": False,
    "highres_imgsz": 1600,
    "highres_min_side": 1600,
    "debug_levels": False,
    "enable_d20_structure_score": False,
    "d20_structure_thr": 0.45,
    "enable_lane_marking_suppress": False,
    "lane_marking_bright_thr": 200,
    "lane_marking_sat_thr": 45,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_cascade_module():
    if not CASCADE_SCRIPT.exists():
        raise FileNotFoundError(f"找不到 cascade 脚本: {CASCADE_SCRIPT}")
    sys.path.insert(0, str(CASCADE_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("cascade_infer_detseg_local", str(CASCADE_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 cascade 脚本")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def list_images(src: Path) -> list[Path]:
    if not src.exists():
        raise FileNotFoundError(f"source 不存在: {src}")
    return sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def iter_det_runs(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not p.name.startswith("train_det_"):
            continue
        if DET_WHITELIST and p.name not in DET_WHITELIST:
            continue
        if any(k in p.name for k in DET_EXCLUDE_KEYWORDS):
            print(f"[SKIP DET] {p.name} -> 命中 DET_EXCLUDE_KEYWORDS={DET_EXCLUDE_KEYWORDS}")
            continue
        if not (p / "weights" / "best.pt").exists():
            continue
        yield p


def iter_seg_runs(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not p.name.startswith("train_seg_"):
            continue
        if SEG_WHITELIST and p.name not in SEG_WHITELIST:
            continue
        if any(k in p.name for k in SEG_EXCLUDE_KEYWORDS):
            print(f"[SKIP SEG] {p.name} -> 命中 SEG_EXCLUDE_KEYWORDS={SEG_EXCLUDE_KEYWORDS}")
            continue
        if not (p / "weights" / "best.pt").exists():
            continue
        yield p


def iter_pairs(det_runs: list[Path], seg_runs: list[Path]):
    if PAIR_MODE == "zip":
        n = min(len(det_runs), len(seg_runs))
        for i in range(n):
            yield det_runs[i], seg_runs[i]
    else:
        for d in det_runs:
            for s in seg_runs:
                yield d, s


def main() -> None:
    ensure_dir(OUT_ROOT)
    images = list_images(SOURCE_DIR)
    det_runs = list(iter_det_runs(RUNS_ROOT))
    seg_runs = list(iter_seg_runs(RUNS_ROOT))
    mod = load_cascade_module()

    if not images:
        raise RuntimeError(f"{SOURCE_DIR} 下没有图片")
    if not det_runs:
        raise RuntimeError("没有非 yolo26 det 权重")
    if not seg_runs:
        raise RuntimeError("没有非 yolo26 seg 权重")

    summary = []
    print("[INFO] 标准版 ROI 级联脚本默认排除 yolo26。")
    print(f"[INFO] SOURCE_DIR = {SOURCE_DIR}")

    for det_run, seg_run in iter_pairs(det_runs, seg_runs):
        det_weight = det_run / "weights" / "best.pt"
        seg_weight = seg_run / "weights" / "best.pt"
        pair_name = f"{det_run.name}__{seg_run.name}"
        save_dir = OUT_ROOT / pair_name
        ensure_dir(save_dir)

        det_model = YOLO(str(det_weight))
        seg_model = YOLO(str(seg_weight))
        det_names = det_model.names if hasattr(det_model, "names") else {}

        print(f"\n{'=' * 90}")
        print(f"[START] {pair_name}")
        t0 = time.perf_counter()

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] 读图失败: {img_path}")
                continue
            mask_u8, det_res = mod.cascade_one_image_v3c(
                img,
                det_model,
                seg_model,
                det_conf=ARGS["det_conf"],
                det_iou=ARGS["det_iou"],
                det_imgsz=ARGS["det_imgsz"],
                seg_conf=ARGS["seg_conf"],
                seg_thr=ARGS["seg_thr"],
                seg_iou=ARGS["seg_iou"],
                seg_imgsz=ARGS["seg_imgsz"],
                pad_ratio=ARGS["pad_ratio"],
                pad_min=ARGS["pad_min"],
                max_rois=ARGS["max_rois"],
                allowed_det_classes=ARGS["det_classes"],
                max_area_ratio=ARGS["max_area_ratio"],
                tile_min_side=ARGS["tile_min_side"],
                tile=ARGS["tile"],
                overlap=ARGS["overlap"],
                use_tile_for_big_roi=(not ARGS["no_tile"]),
                big_damage_class_ids=tuple(ARGS["big_classes"]),
                big_seg_conf=ARGS["big_seg_conf"],
                big_seg_thr=ARGS["big_seg_thr"],
                big_seg_iou=ARGS["big_seg_iou"],
                big_seg_imgsz=ARGS["big_seg_imgsz"],
                big_force_tile=ARGS["big_force_tile"],
                big_tile=ARGS["big_tile"],
                big_overlap=ARGS["big_overlap"],
                big_use_clahe=ARGS["big_clahe"],
                clahe_clip=ARGS["clahe_clip"],
                clahe_grid=ARGS["clahe_grid"],
                debug_dir=None,
                debug_prefix=img_path.stem,
                post_open_ksize=ARGS["post_open"],
                post_close_ksize=ARGS["post_close"],
                post_min_area=ARGS["post_min_area"],
                roi_v3=ARGS["roi_v3"],
                d20_class_id=ARGS["d20_class_id"],
                d20_pad_ratio=ARGS["d20_pad_ratio"],
                d20_pad_min=ARGS["d20_pad_min"],
                d20_seg_conf=ARGS["d20_seg_conf"],
                d20_seg_thr=ARGS["d20_seg_thr"],
                d20_seg_iou=ARGS["d20_seg_iou"],
                d20_seg_imgsz=ARGS["d20_seg_imgsz"],
                d20_clahe=ARGS["d20_clahe"],
                adaptive_tile=ARGS["adaptive_tile"],
                adaptive_min_tile=ARGS["adaptive_min_tile"],
                adaptive_max_tile=ARGS["adaptive_max_tile"],
                adaptive_target_long_tiles=ARGS["adaptive_target_long_tiles"],
                tile_overlap_ratio=ARGS["tile_overlap_ratio"],
                tile_fusion=ARGS["tile_fusion"],
                tile_fusion_thr=ARGS["tile_fusion_thr"],
                highres_enabled=ARGS["highres_seg"],
                highres_imgsz=ARGS["highres_imgsz"],
                highres_min_side=ARGS["highres_min_side"],
                debug_levels=ARGS["debug_levels"],
                enable_d20_structure_score=ARGS["enable_d20_structure_score"],
                d20_structure_thr=ARGS["d20_structure_thr"],
                enable_lane_marking_suppress=ARGS["enable_lane_marking_suppress"],
                lane_marking_bright_thr=ARGS["lane_marking_bright_thr"],
                lane_marking_sat_thr=ARGS["lane_marking_sat_thr"],
            )
            det_vis = mod.draw_det_boxes(img, det_res, det_names, conf_thr=ARGS["det_conf"])
            overlay = mod.overlay_mask_red(det_vis, mask_u8, alpha=0.45)
            cv2.imwrite(str(save_dir / f"{img_path.stem}__mask.png"), mask_u8)
            cv2.imwrite(str(save_dir / f"{img_path.stem}__overlay.jpg"), overlay)

        dt = time.perf_counter() - t0
        meta = {
            "pair_name": pair_name,
            "det_weight": str(det_weight),
            "seg_weight": str(seg_weight),
            "source": str(SOURCE_DIR),
            "save_dir": str(save_dir),
            "num_images": len(images),
            "elapsed_sec": dt,
            "args": ARGS,
            "det_exclude_keywords": DET_EXCLUDE_KEYWORDS,
            "seg_exclude_keywords": SEG_EXCLUDE_KEYWORDS,
        }
        with open(save_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        summary.append(meta)
        print(f"[DONE] {pair_name}  elapsed={dt:.2f}s")

    with open(OUT_ROOT / "summary_cascade_standard.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n全部完成，输出目录: {OUT_ROOT}")


if __name__ == "__main__":
    main()
