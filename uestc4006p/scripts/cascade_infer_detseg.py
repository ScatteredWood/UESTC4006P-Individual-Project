"""
UESTC4006P - Cascade Inference (Det -> Crop -> Seg -> Stitch Back)
===============================================================
级联推理：先用检测模型定位裂缝候选区域（D00/D10/D20/D40），再对候选区域裁剪后做分割，
最后把 ROI 的 mask 映射回原图并融合，输出整图裂缝掩膜与可视化结果。

设计要点（写进毕设/中期报告也好用）：
1) Det 负责“召回”：阈值适当放松，避免漏检导致 Seg 无法补救
2) ROI 做 padding：防止裁剪切断裂缝，提升 Seg 边界完整性
3) Seg 输出回贴：ROI mask resize 回 ROI 尺寸，再 union 融合到全图
4) 多类 Det：可选择只用某些类（如 D00）或全部合并为“裂缝候选”
"""

import argparse
from pathlib import Path
import os
import cv2
import numpy as np
from ultralytics import YOLO


# ===================== 工具函数 =====================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W))
    y2 = max(0, min(int(y2), H))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2

def expand_box(x1, y1, x2, y2, W, H, pad_ratio=0.15, pad_min=16):
    bw, bh = (x2 - x1), (y2 - y1)
    pad = int(max(pad_min, pad_ratio * max(bw, bh)))
    return clamp_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, W, H)

def overlay_mask_red(img_bgr: np.ndarray, mask_u8: np.ndarray, alpha=0.45):
    """
    用红色把 mask 叠加在原图上（alpha 越大越红）
    """
    overlay = img_bgr.copy()
    red = np.zeros_like(img_bgr)
    red[:, :, 2] = 255
    m = (mask_u8 > 0)
    overlay[m] = (img_bgr[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)
    return overlay

def yolo_seg_union_mask(seg_result, roi_h, roi_w, seg_conf=0.25, thr=0.5):
    """
    将 Ultralytics segmentation 输出转换为 ROI 尺寸的 union 二值 mask。
    - seg_result.masks.data: [N, h, w]，通常为模型内部尺度
    - 这里做：按实例置信度过滤 -> max union -> resize 回 ROI 尺寸 -> 二值化
    """
    if seg_result.masks is None:
        return np.zeros((roi_h, roi_w), dtype=np.uint8)

    m = seg_result.masks.data
    m = m.detach().cpu().numpy()  # float32, 0~1

    # 按实例置信度过滤（boxes.conf 与 masks 一一对应）
    if seg_result.boxes is not None and len(seg_result.boxes) == m.shape[0]:
        scores = seg_result.boxes.conf.detach().cpu().numpy()
        keep = scores >= seg_conf
        m = m[keep] if keep.any() else m[:0]

    if m.shape[0] == 0:
        return np.zeros((roi_h, roi_w), dtype=np.uint8)

    union = np.max(m, axis=0)  # [mh, mw]
    union = (union > thr).astype(np.uint8) * 255
    union = cv2.resize(union, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    return union

def draw_det_boxes(img_bgr: np.ndarray, det_res, names, conf_thr=0.2):
    """
    可视化 det 框（可选）
    """
    out = img_bgr.copy()
    if det_res.boxes is None or len(det_res.boxes) == 0:
        return out

    xyxy = det_res.boxes.xyxy.detach().cpu().numpy()
    conf = det_res.boxes.conf.detach().cpu().numpy()
    cls  = det_res.boxes.cls.detach().cpu().numpy().astype(int)

    for (b, c, k) in zip(xyxy, conf, cls):
        if c < conf_thr:
            continue
        x1, y1, x2, y2 = map(int, b.tolist())
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{names.get(k, str(k))} {c:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out

# ===================== 主流程 =====================

def cascade_one_image(
    img_bgr: np.ndarray,
    det_model: YOLO,
    seg_model: YOLO,
    det_conf=0.20,
    det_iou=0.50,
    seg_conf=0.25,
    seg_thr=0.50,
    pad_ratio=0.15,
    pad_min=16,
    max_rois=80,
    allowed_det_classes=None,  # e.g. [0,1,2,3] or None=all
):
    H, W = img_bgr.shape[:2]
    full_mask = np.zeros((H, W), dtype=np.uint8)

    # 1) det on full image
    det_res = det_model.predict(img_bgr, conf=det_conf, iou=det_iou, verbose=False)[0]
    if det_res.boxes is None or len(det_res.boxes) == 0:
        return full_mask, det_res

    xyxy = det_res.boxes.xyxy.detach().cpu().numpy()
    conf = det_res.boxes.conf.detach().cpu().numpy()
    cls  = det_res.boxes.cls.detach().cpu().numpy().astype(int)

    # 2) 过滤类别（如只用 D00）
    idx = np.arange(len(xyxy))
    if allowed_det_classes is not None:
        allowed_det_classes = set(allowed_det_classes)
        idx = np.array([i for i in idx if cls[i] in allowed_det_classes], dtype=int)

    if idx.size == 0:
        return full_mask, det_res

    xyxy = xyxy[idx]
    conf = conf[idx]

    # 3) 限制 ROI 数量，避免极端情况太慢（按 conf 排序）
    if len(xyxy) > max_rois:
        order = np.argsort(-conf)[:max_rois]
        xyxy = xyxy[order]
        conf = conf[order]

    # 4) crop -> seg -> stitch back
    for (x1, y1, x2, y2) in xyxy:
        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, W, H, pad_ratio, pad_min)
        roi = img_bgr[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]
        if rh < 2 or rw < 2:
            continue

        seg_res = seg_model.predict(roi, conf=seg_conf, iou=0.5, verbose=False)[0]
        roi_mask = yolo_seg_union_mask(seg_res, rh, rw, seg_conf=seg_conf, thr=seg_thr)

        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], roi_mask)

    return full_mask, det_res


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--det", required=True, help="path to det best.pt")
    ap.add_argument("--seg", required=True, help="path to seg best.pt")
    ap.add_argument("--source", required=True, help="image file or directory")
    ap.add_argument("--outdir", default="uestc4006p/runs/cascade", help="output dir (relative or absolute)")

    # 阈值与级联参数（中期建议先别乱调）
    ap.add_argument("--det-conf", type=float, default=0.20)
    ap.add_argument("--det-iou", type=float, default=0.50)
    ap.add_argument("--seg-conf", type=float, default=0.25)
    ap.add_argument("--seg-thr", type=float, default=0.50)

    ap.add_argument("--pad-ratio", type=float, default=0.15)
    ap.add_argument("--pad-min", type=int, default=16)
    ap.add_argument("--max-rois", type=int, default=80)

    # 选择 det 类别：默认全用（D00/D10/D20/D40）
    # 例如：--det-classes 0  表示只用 D00
    ap.add_argument("--det-classes", nargs="*", type=int, default=None)

    args = ap.parse_args()

    det_path = Path(args.det).resolve()
    seg_path = Path(args.seg).resolve()

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (Path.cwd() / outdir).resolve()
    ensure_dir(outdir)

    det_model = YOLO(str(det_path))
    seg_model = YOLO(str(seg_path))

    # 读取 det 的类别名（来自训练时的 names）
    det_names = det_model.names if hasattr(det_model, "names") else {}

    src = Path(args.source)
    if src.is_dir():
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
            images += sorted(src.glob(ext))
    else:
        images = [src]

    if len(images) == 0:
        raise FileNotFoundError(f"No images found in: {src}")

    from datetime import datetime

    def short_tag_from_ckpt_parent(parent_name: str, fallback: str) -> str:
        """
        把超长实验名压缩成“中等长度、可读、可追溯”的短 tag。
        策略：
        1) 先取 '__' 前的骨架（通常是 task_dataset_domain_model_imgsz...）
        2) 再做轻量替换：yolov8->v8, yolov11->v11（更短）
        3) 最后截断到前 5 段，避免路径过长导致 Windows 写文件失败
        """
        core = parent_name.split("__")[0]  # e.g. det_RDD2022_China_yolov8n_1024_ep300_bs16_seed42_baseline
        core = core.replace("yolov8", "v8").replace("yolov11", "v11")
        parts = core.split("_")
        if len(parts) > 5:
            core = "_".join(parts[:5])
        return core if core else fallback

    det_short = short_tag_from_ckpt_parent(det_path.parent.name, "det")
    seg_short = short_tag_from_ckpt_parent(seg_path.parent.name, "seg")

    run_dir = outdir / f"cascade__{det_short}__{seg_short}"
    ensure_dir(run_dir)

    # 运行说明：记录权重、测试集来源、参数、输出文件含义（中期报告可直接引用）
    info_path = run_dir / "RUN_INFO.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("UESTC4006P - Cascade Inference Run Info\n")
        f.write("====================================================\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n[Weights]\n")
        f.write(f"DET: {det_path}\n")
        f.write(f"SEG: {seg_path}\n")
        f.write("\n[Test Source]\n")
        f.write(f"SOURCE: {src}\n")
        f.write("\n[Params]\n")
        f.write(f"det_conf={args.det_conf}, det_iou={args.det_iou}\n")
        f.write(f"seg_conf={args.seg_conf}, seg_thr={args.seg_thr}\n")
        f.write(f"pad_ratio={args.pad_ratio}, pad_min={args.pad_min}\n")
        f.write(f"max_rois={args.max_rois}\n")
        f.write(f"det_classes={args.det_classes if args.det_classes is not None else 'ALL'}\n")
        f.write("\n[Outputs]\n")
        f.write("- <image>__mask.png : final stitched binary mask (255=crack)\n")
        f.write("- <image>__overlay.jpg : det boxes + red mask overlay\n")
        f.write("====================================================\n")

    print("RUN_INFO:", info_path)

    print("======================================================")
    print("DET:", det_path)
    print("SEG:", seg_path)
    print("SOURCE:", src)
    print("RUN_DIR:", run_dir)
    print("PARAMS:", f"det_conf={args.det_conf}, det_iou={args.det_iou}, seg_conf={args.seg_conf}, seg_thr={args.seg_thr}, pad_ratio={args.pad_ratio}, pad_min={args.pad_min}")
    print("DET_CLASSES:", args.det_classes if args.det_classes is not None else "ALL")
    print("======================================================")

    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] cannot read: {p}")
            continue

        full_mask, det_res = cascade_one_image(
            img,
            det_model,
            seg_model,
            det_conf=args.det_conf,
            det_iou=args.det_iou,
            seg_conf=args.seg_conf,
            seg_thr=args.seg_thr,
            pad_ratio=args.pad_ratio,
            pad_min=args.pad_min,
            max_rois=args.max_rois,
            allowed_det_classes=args.det_classes,
        )

        # 可视化：det 框 + seg mask overlay
        det_vis = draw_det_boxes(img, det_res, det_names, conf_thr=args.det_conf)
        overlay = overlay_mask_red(det_vis, full_mask, alpha=0.45)

        stem = p.stem
        mask_path = run_dir / f"{stem}__mask.png"
        ov_path   = run_dir / f"{stem}__overlay.jpg"

        ok1 = cv2.imwrite(str(mask_path), full_mask)
        ok2 = cv2.imwrite(str(ov_path), overlay)

        if not ok1 or not ok2:
            raise RuntimeError(
                f"cv2.imwrite failed: {p.name}\n"
                f"  ok_mask={ok1}, ok_overlay={ok2}\n"
                f"  mask_path={mask_path}\n"
                f"  overlay_path={ov_path}\n"
                f"  run_dir={run_dir}"
            )

        print(f"[OK] {p.name} -> saved")

    print("DONE.")


if __name__ == "__main__":
    main()
