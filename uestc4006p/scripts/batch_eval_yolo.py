from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from ultralytics import YOLO


# =============================================================================
# 0) 配置区
# =============================================================================

# 训练结果根目录（里面就是 train_det_v8n、train_seg_v11m 等这些文件夹）
RUNS_ROOT = Path(
    r"E:\Large Files\UESTC4006P Individual Project (2025-26)\要使用的训练结果汇总"
)

# 导出目录
EXPORT_ROOT = RUNS_ROOT / "_eval_exports"
DATA_YAML_DIR = EXPORT_ROOT / "data_yaml"
RUN_REPORTS_DIR = EXPORT_ROOT / "run_reports"

# det / seg 的验证集图片目录（脚本会自动去同级 labels/val 找 txt）
DET_VAL_IMAGES = Path(
    r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\RDD2022_China\yolo_det_D00D10D20D40_seed42_v1\images\val"
)
SEG_VAL_IMAGES = Path(
    r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\custom\crack500_SS305_CSa100_cs4029\crack500_SS305_CSa100_cs4029\images\val"
)

# 统一评估输入尺寸：为了公平比较，请固定
DET_IMGSZ = 1024
SEG_IMGSZ = 1024

# 设备
DEVICE: int | str = 0   # GPU 0；没有 GPU 可改成 "cpu"
WORKERS = 4
BATCH = 1               # 为了“单张图推理时间/FPS”统一口径，建议 batch=1
VERBOSE = True

# 是否半精度
HALF = torch.cuda.is_available() and DEVICE != "cpu"

# det 数据集类别名
DET_NAMES = ["D00", "D10", "D20", "D40"]

# seg 数据集类别名
# 你的 crack 数据集通常是单类；若不是单类，请自己改这里
SEG_NAMES = ["crack"]


# =============================================================================
# 1) 工具函数
# =============================================================================

def ensure_dirs() -> None:
    DATA_YAML_DIR.mkdir(parents=True, exist_ok=True)
    RUN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def infer_task(run_dir: Path) -> str:
    """
    根据目录名判断任务类型：
    train_det_xxx -> detect
    train_seg_xxx -> segment
    """
    n = run_dir.name.lower()
    if n.startswith("train_det_"):
        return "detect"
    if n.startswith("train_seg_"):
        return "segment"
    raise ValueError(f"无法从目录名判断任务类型：{run_dir.name}")


def find_best_weight(run_dir: Path) -> Path | None:
    p = run_dir / "weights" / "best.pt"
    return p if p.exists() else None


def load_ckpt_meta(weight_path: Path) -> dict:
    try:
        ckpt = torch.load(weight_path, map_location="cpu")
        if isinstance(ckpt, dict):
            return ckpt
        return {}
    except Exception:
        return {}


def get_best_epoch_from_ckpt(ckpt: dict) -> int | None:
    """
    Ultralytics checkpoint 里的 epoch 通常是 0-based。
    写论文/表格一般展示为 1-based，所以这里 +1。
    """
    epoch = ckpt.get("epoch", None)
    if epoch is None:
        return None
    return int(epoch) + 1


def get_weight_size_mb(weight_path: Path) -> float:
    return weight_path.stat().st_size / (1024 ** 2)


def build_val_yaml(images_val_dir: Path, names: list[str], yaml_path: Path) -> None:
    """
    只为本次 val 评估生成一个临时 YAML。
    这里把 train/val/test 都指向 images/val，避免无关目录缺失导致解析问题。
    labels 会按 Ultralytics 规则自动从 labels/val 找。
    """
    dataset_root = images_val_dir.parent.parent  # .../images/val -> dataset_root

    data = {
        "path": str(dataset_root),
        "train": "images/val",
        "val": "images/val",
        "test": "images/val",
        "names": names,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def check_image_label_pairs(images_val_dir: Path) -> None:
    """
    检查 images/val 与 labels/val 是否一一对应。
    """
    if not images_val_dir.exists():
        raise FileNotFoundError(f"缺少图片目录：{images_val_dir}")

    label_val_dir = images_val_dir.parent.parent / "labels" / "val"
    if not label_val_dir.exists():
        raise FileNotFoundError(f"缺少标注目录：{label_val_dir}")

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = sorted(
        [p for p in images_val_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts]
    )

    missing_labels = []
    for img in image_files:
        txt = label_val_dir / f"{img.stem}.txt"
        if not txt.exists():
            missing_labels.append(img.name)

    print(f"\n[检查] {images_val_dir}")
    print(f"图片数量: {len(image_files)}")
    print(f"缺少标注数量: {len(missing_labels)}")

    if missing_labels:
        preview = missing_labels[:20]
        raise ValueError(f"以下图片缺少对应 txt 标注（仅展示前20个）：{preview}")


def get_names_map(results, fallback_names: list[str]) -> dict[int, str]:
    """
    优先从 results.names 取类别名；取不到就用 fallback_names。
    """
    names = getattr(results, "names", None)

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}

    return {i: str(v) for i, v in enumerate(fallback_names)}


def get_model_info(weight_path: Path, imgsz: int) -> dict:
    """
    尽量获取 layers / params / grads / flops_g。
    若 info() 不稳定，则至少回退到 params / grads。
    """
    out = {
        "layers": None,
        "params": None,
        "grads": None,
        "flops_g": None,
    }

    try:
        model = YOLO(str(weight_path))

        # 尝试 fuse 后再统计，更接近部署口径
        try:
            model.fuse()
        except Exception:
            pass

        # 先尝试官方 info 返回
        try:
            info = model.info(verbose=False, imgsz=imgsz)
            if isinstance(info, (list, tuple)) and len(info) >= 4:
                out["layers"] = safe_int(info[0])
                out["params"] = safe_int(info[1])
                out["grads"] = safe_int(info[2])
                out["flops_g"] = safe_float(info[3])
        except Exception:
            pass

        # 若 params / grads 没拿到，就自己数
        try:
            inner_model = model.model
            if out["params"] is None:
                out["params"] = int(sum(p.numel() for p in inner_model.parameters()))
            if out["grads"] is None:
                out["grads"] = int(sum(p.numel() for p in inner_model.parameters() if p.requires_grad))
        except Exception:
            pass

    except Exception:
        pass

    return out


def run_val(
    weight_path: Path,
    data_yaml: Path,
    imgsz: int,
    run_name: str,
) -> tuple[Any, Path]:
    """
    跑 val。plots=True 会自动保存 PR/P/R/F1 曲线和混淆矩阵图。
    """
    save_dir = RUN_REPORTS_DIR / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weight_path))
    results = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        half=HALF,
        plots=True,
        verbose=VERBOSE,
        project=str(RUN_REPORTS_DIR),
        name=run_name,
        exist_ok=True,
    )
    return results, save_dir


def export_confusion_matrix(results, save_dir: Path, fallback_names: list[str]) -> None:
    """
    导出原始混淆矩阵和按行归一化混淆矩阵。
    注意：这里的矩阵方向保持与 Ultralytics 内部一致，不额外改动。
    """
    cm_obj = getattr(results, "confusion_matrix", None)
    if cm_obj is None:
        return

    matrix = getattr(cm_obj, "matrix", None)
    if matrix is None:
        return

    try:
        import numpy as np

        mat = np.array(matrix, dtype=float)
        names_map = get_names_map(results, fallback_names)

        n = len(names_map)
        labels = [names_map[i] for i in range(n)]

        # Ultralytics 混淆矩阵通常会多一个 background 列/行
        if mat.shape[0] == n + 1 and mat.shape[1] == n + 1:
            labels_with_bg = labels + ["background"]
        else:
            labels_with_bg = [f"class_{i}" for i in range(mat.shape[0])]

        raw_df = pd.DataFrame(mat, index=labels_with_bg, columns=labels_with_bg)
        raw_df.to_csv(save_dir / "confusion_matrix_raw.csv", encoding="utf-8-sig")

        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        norm = mat / row_sums
        norm_df = pd.DataFrame(norm, index=labels_with_bg, columns=labels_with_bg)
        norm_df.to_csv(save_dir / "confusion_matrix_normalized.csv", encoding="utf-8-sig")

    except Exception:
        # 兜底：若有 summary 方法，就导出 summary
        try:
            raw_df = pd.DataFrame(cm_obj.summary(normalize=False, decimals=6))
            raw_df.to_csv(save_dir / "confusion_matrix_raw.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pass

        try:
            norm_df = pd.DataFrame(cm_obj.summary(normalize=True, decimals=6))
            norm_df.to_csv(save_dir / "confusion_matrix_normalized.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pass


def export_per_class_detect(results, save_dir: Path, fallback_names: list[str]) -> None:
    """
    检测任务导出每类：
    Precision / Recall / F1 / AP50 / AP50-95
    """
    rows: list[dict[str, Any]] = []

    try:
        names_map = get_names_map(results, fallback_names)
        box = results.box

        ap_class_index = list(getattr(box, "ap_class_index", []))
        p_arr = list(getattr(box, "p", []))
        r_arr = list(getattr(box, "r", []))
        f1_arr = list(getattr(box, "f1", []))

        # 先给所有类别占位，避免某类没有结果时整行丢失
        base = {}
        for cls_id, cls_name in names_map.items():
            base[int(cls_id)] = {
                "class_id": int(cls_id),
                "class_name": cls_name,
                "precision": None,
                "recall": None,
                "f1": None,
                "AP50": None,
                "AP50_95": None,
            }

        # 再填充真正有统计的类别
        for local_i, cls_id in enumerate(ap_class_index):
            p = safe_float(p_arr[local_i]) if local_i < len(p_arr) else None
            r = safe_float(r_arr[local_i]) if local_i < len(r_arr) else None
            f1 = safe_float(f1_arr[local_i]) if local_i < len(f1_arr) else None

            ap50 = None
            ap = None
            try:
                # class_result(local_i) 通常返回 p, r, ap50, ap
                _, _, ap50_v, ap_v = box.class_result(local_i)
                ap50 = safe_float(ap50_v)
                ap = safe_float(ap_v)
            except Exception:
                pass

            if int(cls_id) not in base:
                base[int(cls_id)] = {
                    "class_id": int(cls_id),
                    "class_name": names_map.get(int(cls_id), f"class_{cls_id}"),
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "AP50": None,
                    "AP50_95": None,
                }

            base[int(cls_id)]["precision"] = p
            base[int(cls_id)]["recall"] = r
            base[int(cls_id)]["f1"] = f1
            base[int(cls_id)]["AP50"] = ap50
            base[int(cls_id)]["AP50_95"] = ap

        rows = [base[k] for k in sorted(base.keys())]

    except Exception:
        # 兜底：直接导出 summary()
        try:
            rows = results.summary()
        except Exception:
            rows = []

    if rows:
        pd.DataFrame(rows).to_csv(save_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")


def export_per_class_generic(results, save_dir: Path) -> None:
    """
    通用导出：直接存 results.summary()。
    对 seg 可以作为补充参考。
    """
    try:
        rows = results.summary()
        if rows:
            pd.DataFrame(rows).to_csv(save_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    except Exception:
        pass


def build_common_row(
    run_name: str,
    task: str,
    weight_path: Path,
    ckpt: dict,
    info: dict,
    results,
    imgsz: int,
) -> dict[str, Any]:
    speed = getattr(results, "speed", {}) or {}

    preprocess_ms = safe_float(speed.get("preprocess", None))
    inference_ms = safe_float(speed.get("inference", None))
    postprocess_ms = safe_float(speed.get("postprocess", None))
    loss_ms = safe_float(speed.get("loss", None))

    end2end_ms = None
    if preprocess_ms is not None and inference_ms is not None and postprocess_ms is not None:
        end2end_ms = preprocess_ms + inference_ms + postprocess_ms

    fps_inference = None
    if inference_ms is not None and inference_ms > 0:
        fps_inference = 1000.0 / inference_ms

    fps_end2end = None
    if end2end_ms is not None and end2end_ms > 0:
        fps_end2end = 1000.0 / end2end_ms

    row = {
        "run_name": run_name,
        "task": task,
        "weight_path": str(weight_path),
        "best_epoch": get_best_epoch_from_ckpt(ckpt),
        "imgsz_eval": imgsz,
        "params": info.get("params"),
        "flops_g": info.get("flops_g"),
        "weight_size_mb": round(get_weight_size_mb(weight_path), 3),
        "speed_preprocess_ms_per_img": preprocess_ms,
        "speed_inference_ms_per_img": inference_ms,
        "speed_postprocess_ms_per_img": postprocess_ms,
        "speed_loss_ms_per_img": loss_ms,
        "speed_end2end_ms_per_img": end2end_ms,
        "fps_inference_only": fps_inference,
        "fps_end2end": fps_end2end,
    }

    train_args = ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {}
    for k in ["model", "data", "imgsz", "batch", "epochs", "device"]:
        if k in train_args:
            row[f"train_arg_{k}"] = train_args[k]

    return row


def build_det_row(common_row: dict[str, Any], results) -> dict[str, Any]:
    row = dict(common_row)
    row.update({
        "precision": safe_float(getattr(results.box, "mp", None)),
        "recall": safe_float(getattr(results.box, "mr", None)),
        "mAP50": safe_float(getattr(results.box, "map50", None)),
        "mAP50_95": safe_float(getattr(results.box, "map", None)),
    })
    return row


def build_seg_row(common_row: dict[str, Any], results) -> dict[str, Any]:
    row = dict(common_row)
    row.update({
        "box_precision": safe_float(getattr(results.box, "mp", None)),
        "box_recall": safe_float(getattr(results.box, "mr", None)),
        "box_mAP50": safe_float(getattr(results.box, "map50", None)),
        "box_mAP50_95": safe_float(getattr(results.box, "map", None)),
        "mask_precision": safe_float(getattr(results.seg, "mp", None)),
        "mask_recall": safe_float(getattr(results.seg, "mr", None)),
        "mask_mAP50": safe_float(getattr(results.seg, "map50", None)),
        "mask_mAP50_95": safe_float(getattr(results.seg, "map", None)),
    })
    return row


def save_metrics_json(row: dict[str, Any], save_dir: Path) -> None:
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)


# =============================================================================
# 2) 主流程
# =============================================================================

def main() -> None:
    ensure_dirs()

    # 先检查数据集
    check_image_label_pairs(DET_VAL_IMAGES)
    check_image_label_pairs(SEG_VAL_IMAGES)

    # 生成临时 YAML
    det_yaml = DATA_YAML_DIR / "det_val.yaml"
    seg_yaml = DATA_YAML_DIR / "seg_val.yaml"
    build_val_yaml(DET_VAL_IMAGES, DET_NAMES, det_yaml)
    build_val_yaml(SEG_VAL_IMAGES, SEG_NAMES, seg_yaml)

    det_rows: list[dict[str, Any]] = []
    seg_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    # 只处理 train_ 开头的目录
    run_dirs = sorted(
        [p for p in RUNS_ROOT.iterdir() if p.is_dir() and p.name.startswith("train_")]
    )

    if not run_dirs:
        raise RuntimeError(f"在 {RUNS_ROOT} 下没有找到 train_ 开头的训练结果目录。")

    print("\n将处理以下目录：")
    for p in run_dirs:
        print(" -", p.name)

    for run_dir in run_dirs:
        run_name = run_dir.name
        print(f"\n{'=' * 100}")
        print(f"[开始] {run_name}")

        try:
            task = infer_task(run_dir)
            weight_path = find_best_weight(run_dir)
            if weight_path is None:
                raise FileNotFoundError(f"缺少 best.pt：{run_dir / 'weights' / 'best.pt'}")

            data_yaml = det_yaml if task == "detect" else seg_yaml
            imgsz = DET_IMGSZ if task == "detect" else SEG_IMGSZ
            fallback_names = DET_NAMES if task == "detect" else SEG_NAMES

            ckpt = load_ckpt_meta(weight_path)
            info = get_model_info(weight_path, imgsz)

            results, save_dir = run_val(
                weight_path=weight_path,
                data_yaml=data_yaml,
                imgsz=imgsz,
                run_name=run_name,
            )

            # 导出混淆矩阵 CSV
            export_confusion_matrix(results, save_dir, fallback_names)

            # 导出 per-class 指标
            if task == "detect":
                export_per_class_detect(results, save_dir, fallback_names)
            else:
                export_per_class_generic(results, save_dir)

            # 构造总表记录
            common_row = build_common_row(
                run_name=run_name,
                task=task,
                weight_path=weight_path,
                ckpt=ckpt,
                info=info,
                results=results,
                imgsz=imgsz,
            )

            if task == "detect":
                row = build_det_row(common_row, results)
                det_rows.append(row)
            else:
                row = build_seg_row(common_row, results)
                seg_rows.append(row)

            save_metrics_json(row, save_dir)

            print(f"[完成] {run_name}")
            print(f"输出目录：{save_dir}")

        except Exception as e:
            err = {
                "run_name": run_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            failed.append(err)
            print(f"[失败] {run_name}: {e}")

    # 保存总表
    if det_rows:
        det_df = pd.DataFrame(det_rows).sort_values("run_name")
        det_df.to_csv(EXPORT_ROOT / "summary_det.csv", index=False, encoding="utf-8-sig")

    if seg_rows:
        seg_df = pd.DataFrame(seg_rows).sort_values("run_name")
        seg_df.to_csv(EXPORT_ROOT / "summary_seg.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(EXPORT_ROOT / "summary_all.xlsx", engine="openpyxl") as writer:
        if det_rows:
            pd.DataFrame(det_rows).sort_values("run_name").to_excel(
                writer, sheet_name="det_summary", index=False
            )
        if seg_rows:
            pd.DataFrame(seg_rows).sort_values("run_name").to_excel(
                writer, sheet_name="seg_summary", index=False
            )
        if failed:
            pd.DataFrame(failed).to_excel(writer, sheet_name="failed", index=False)

    if failed:
        with open(EXPORT_ROOT / "failed_runs.json", "w", encoding="utf-8") as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)

    print("\n全部结束。")
    print(f"导出根目录：{EXPORT_ROOT}")


if __name__ == "__main__":
    main()