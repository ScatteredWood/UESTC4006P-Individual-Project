"""
UESTC4006P - Cascade Inference v3 (Det -> Crop/Tile -> Seg -> Stitch Back)
=========================================================================

适用场景：
- Seg 用 CRACK500 等近景裂缝数据训练（patch 分布）
- 推理在 RDD 等远景整图（尺度差异巨大）
- Det 框可能很大（如 D20 框到整块路面/斑马线），直接 ROI->Seg 会尺度塌缩导致 mask 全空

核心改进（v3）：
1) Seg 推理显式 imgsz 可调（默认 1280）
2) ROI 过大时自动 tile（滑窗 + overlap），把远景问题“变回近景分布”
3) 可选：只用部分 det 类、过滤超大框、后处理去噪、debug 输出 ROI 可视化

输出：
- <image>__mask.png      : stitched binary mask (255=crack)
- <image>__overlay.jpg   : det boxes + red mask overlay
- RUN_INFO.txt           : 运行参数与权重、数据来源记录
- debug_rois/* (可选)    : ROI/Tile 叠加图，方便定位“为什么 seg 空”
"""

import argparse
from pathlib import Path
import os
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime


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
    overlay = img_bgr.copy()
    red = np.zeros_like(img_bgr)
    red[:, :, 2] = 255
    m = (mask_u8 > 0)
    overlay[m] = (img_bgr[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)
    return overlay

def yolo_seg_union_mask(seg_result, roi_h, roi_w, seg_conf=0.25, thr=0.5):
    """
    将 Ultralytics segmentation 输出转换为 ROI 尺寸的 union 二值 mask。
    - seg_result.masks.data: [N, h, w] float(0~1)
    - 按实例置信度过滤 -> union -> resize 回 ROI -> 二值化
    """
    if seg_result.masks is None:
        return np.zeros((roi_h, roi_w), dtype=np.uint8)

    m = seg_result.masks.data
    m = m.detach().cpu().numpy()

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

def postprocess_mask(mask_u8: np.ndarray, open_ksize=0, min_area=0):
    """
    简单后处理：
    - open 去噪点（不建议核太大，避免伤细裂缝）
    - 连通域面积过滤（过滤极小碎片）
    """
    out = mask_u8.copy()

    if open_ksize and open_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)

    if min_area and min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros_like(out, dtype=np.uint8)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                keep[labels == i] = 255
        out = keep

    return out

def seg_on_roi_with_tiling(
    seg_model: YOLO,
    roi_bgr: np.ndarray,
    seg_conf: float,
    seg_thr: float,
    seg_imgsz: int,
    tile: int,
    overlap: int,
    debug_dir: Path = None,
    debug_prefix: str = "",
):
    """
    对 ROI 进行 tile 分割（滑窗 + overlap），并 union 拼回 ROI mask。
    """
    rh, rw = roi_bgr.shape[:2]
    out = np.zeros((rh, rw), dtype=np.uint8)

    tile = int(tile)
    overlap = int(overlap)
    overlap = max(0, min(overlap, tile - 1))
    step = max(1, tile - overlap)

    ys = list(range(0, max(1, rh - tile + 1), step))
    xs = list(range(0, max(1, rw - tile + 1), step))
    if (rh - tile) % step != 0:
        ys.append(max(0, rh - tile))
    if (rw - tile) % step != 0:
        xs.append(max(0, rw - tile))

    tcount = 0
    for y0 in ys:
        for x0 in xs:
            patch = roi_bgr[y0:y0 + tile, x0:x0 + tile]
            ph, pw = patch.shape[:2]
            if ph < 2 or pw < 2:
                continue

            seg_res = seg_model.predict(
                patch,
                conf=seg_conf,
                iou=0.5,
                imgsz=seg_imgsz,
                verbose=False
            )[0]
            patch_mask = yolo_seg_union_mask(seg_res, ph, pw, seg_conf=seg_conf, thr=seg_thr)
            out[y0:y0 + ph, x0:x0 + pw] = np.maximum(out[y0:y0 + ph, x0:x0 + pw], patch_mask)

            if debug_dir is not None:
                tcount += 1
                ov = overlay_mask_red(patch, patch_mask, alpha=0.55)
                cv2.imwrite(str(debug_dir / f"{debug_prefix}__tile_{tcount:03d}_{x0}_{y0}_{x0+pw}_{y0+ph}.jpg"), ov)

    return out


# ===================== 主流程（单图） =====================

def cascade_one_image_v3(
    img_bgr: np.ndarray,
    det_model: YOLO,
    seg_model: YOLO,
    det_conf=0.20,
    det_iou=0.50,
    seg_conf=0.15,
    seg_thr=0.30,
    seg_imgsz=1280,
    pad_ratio=0.15,
    pad_min=16,
    max_rois=80,
    allowed_det_classes=None,      # e.g. [0] or None
    max_area_ratio=0.60,           # 超大框过滤（占整图比例过大直接跳过，或交给 tile）
    tile_min_side=1400,            # ROI 过大触发 tile
    tile=1280,
    overlap=256,
    use_tile_for_big_roi=True,
    debug_dir: Path = None,
    debug_prefix: str = "",
    post_open_ksize=0,
    post_min_area=0,
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
        allowed = set(allowed_det_classes)
        idx = np.array([i for i in idx if cls[i] in allowed], dtype=int)

    if idx.size == 0:
        return full_mask, det_res

    xyxy = xyxy[idx]
    conf = conf[idx]

    # 3) 限制 ROI 数量
    if len(xyxy) > max_rois:
        order = np.argsort(-conf)[:max_rois]
        xyxy = xyxy[order]
        conf = conf[order]

    # 4) crop -> seg/tile -> stitch
    roi_id = 0
    for (x1, y1, x2, y2) in xyxy:
        roi_id += 1

        # 扩框
        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, W, H, pad_ratio, pad_min)
        bw, bh = (x2 - x1), (y2 - y1)

        # 超大 det 框处理（避免整图级框直接拖垮）
        if (bw * bh) > max_area_ratio * (W * H):
            # 你也可以选择这里 continue；我默认“仍允许 tile”，但不建议直接单次 seg
            pass

        roi = img_bgr[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]
        if rh < 2 or rw < 2:
            continue

        # debug：保存 ROI 原图（可选）
        if debug_dir is not None:
            cv2.imwrite(str(debug_dir / f"{debug_prefix}__roi_{roi_id:03d}_{x1}_{y1}_{x2}_{y2}.jpg"), roi)

        # 是否 tile
        big_roi = (max(rh, rw) >= tile_min_side)
        if use_tile_for_big_roi and big_roi:
            roi_mask = seg_on_roi_with_tiling(
                seg_model,
                roi,
                seg_conf=seg_conf,
                seg_thr=seg_thr,
                seg_imgsz=seg_imgsz,
                tile=tile,
                overlap=overlap,
                debug_dir=debug_dir,
                debug_prefix=f"{debug_prefix}__roi{roi_id:03d}"
            )
        else:
            seg_res = seg_model.predict(
                roi,
                conf=seg_conf,
                iou=0.5,
                imgsz=seg_imgsz,
                verbose=False
            )[0]
            roi_mask = yolo_seg_union_mask(seg_res, rh, rw, seg_conf=seg_conf, thr=seg_thr)

        # 可选后处理（在 ROI 层做，减少噪点但别伤细裂缝）
        if post_open_ksize or post_min_area:
            roi_mask = postprocess_mask(roi_mask, open_ksize=post_open_ksize, min_area=post_min_area)

        # debug：保存 ROI overlay（可选）
        if debug_dir is not None:
            roi_ov = overlay_mask_red(roi, roi_mask, alpha=0.55)
            cv2.imwrite(str(debug_dir / f"{debug_prefix}__roi_{roi_id:03d}_{x1}_{y1}_{x2}_{y2}__overlay.jpg"), roi_ov)

        # stitch back
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], roi_mask)

    return full_mask, det_res


# ===================== main =====================

def short_tag_from_ckpt_parent(parent_name: str, fallback: str) -> str:
    core = parent_name.split("__")[0]
    core = core.replace("yolov8", "v8").replace("yolov11", "v11")
    parts = core.split("_")
    if len(parts) > 5:
        core = "_".join(parts[:5])
    return core if core else fallback

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--det", required=True, help="path to det best.pt")
    ap.add_argument("--seg", required=True, help="path to seg best.pt (yolov8n-seg)")
    ap.add_argument("--source", required=True, help="image file or directory")
    ap.add_argument("--outdir", default="uestc4006p/runs/cascade", help="output dir")

    # Det 参数（召回优先）
    ap.add_argument("--det-conf", type=float, default=0.15)
    ap.add_argument("--det-iou", type=float, default=0.50)

    # Seg 参数（建议先偏召回）
    ap.add_argument("--seg-conf", type=float, default=0.10)
    ap.add_argument("--seg-thr",  type=float, default=0.30)
    ap.add_argument("--seg-imgsz", type=int, default=1280, help="seg inference imgsz, e.g. 1024/1280/1536")

    # ROI 与级联
    ap.add_argument("--pad-ratio", type=float, default=0.15)
    ap.add_argument("--pad-min", type=int, default=16)
    ap.add_argument("--max-rois", type=int, default=80)
    ap.add_argument("--det-classes", nargs="*", type=int, default=None, help="e.g. --det-classes 0 (only D00)")

    # 超大框/超大 ROI 控制
    ap.add_argument("--max-area-ratio", type=float, default=0.60, help="det box area ratio vs full image")
    ap.add_argument("--tile-min-side", type=int, default=1400, help="ROI side >= this triggers tiling")
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--no-tile", action="store_true", help="disable tiling even for big ROI")

    # 后处理（可选）
    ap.add_argument("--post-open", type=int, default=0, help="morph open ksize (0=off, suggest 3)")
    ap.add_argument("--post-min-area", type=int, default=0, help="min connected component area (0=off)")

    # debug（可选）
    ap.add_argument("--debug-rois", action="store_true", help="save ROI/tile overlays for debugging")

    args = ap.parse_args()

    det_path = Path(args.det).resolve()
    seg_path = Path(args.seg).resolve()

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (Path.cwd() / outdir).resolve()
    ensure_dir(outdir)

    det_model = YOLO(str(det_path))
    seg_model = YOLO(str(seg_path))

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

    det_short = short_tag_from_ckpt_parent(det_path.parent.name, "det")
    seg_short = short_tag_from_ckpt_parent(seg_path.parent.name, "seg")

    run_dir = outdir / f"cascade_v3__{det_short}__{seg_short}"
    ensure_dir(run_dir)

    debug_dir = None
    if args.debug_rois:
        debug_dir = run_dir / "debug_rois"
        ensure_dir(debug_dir)

    # RUN_INFO
    info_path = run_dir / "RUN_INFO.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("UESTC4006P - Cascade Inference v3 Run Info\n")
        f.write("====================================================\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n[Weights]\n")
        f.write(f"DET: {det_path}\n")
        f.write(f"SEG: {seg_path}\n")
        f.write("\n[Test Source]\n")
        f.write(f"SOURCE: {src}\n")
        f.write("\n[Params]\n")
        f.write(f"det_conf={args.det_conf}, det_iou={args.det_iou}\n")
        f.write(f"seg_conf={args.seg_conf}, seg_thr={args.seg_thr}, seg_imgsz={args.seg_imgsz}\n")
        f.write(f"pad_ratio={args.pad_ratio}, pad_min={args.pad_min}\n")
        f.write(f"max_rois={args.max_rois}, det_classes={args.det_classes if args.det_classes is not None else 'ALL'}\n")
        f.write(f"max_area_ratio={args.max_area_ratio}\n")
        f.write(f"use_tile_for_big_roi={not args.no_tile}, tile_min_side={args.tile_min_side}\n")
        f.write(f"tile={args.tile}, overlap={args.overlap}\n")
        f.write(f"post_open={args.post_open}, post_min_area={args.post_min_area}\n")
        f.write(f"debug_rois={args.debug_rois}\n")
        f.write("\n[Outputs]\n")
        f.write("- <image>__mask.png : final stitched binary mask (255=crack)\n")
        f.write("- <image>__overlay.jpg : det boxes + red mask overlay\n")
        if args.debug_rois:
            f.write("- debug_rois/* : ROI and tile overlays for debugging\n")
        f.write("====================================================\n")

    print("RUN_INFO:", info_path)
    print("======================================================")
    print("DET:", det_path)
    print("SEG:", seg_path)
    print("SOURCE:", src)
    print("RUN_DIR:", run_dir)
    print("PARAMS:",
          f"det_conf={args.det_conf}, det_iou={args.det_iou}, "
          f"seg_conf={args.seg_conf}, seg_thr={args.seg_thr}, seg_imgsz={args.seg_imgsz}, "
          f"tile={'OFF' if args.no_tile else 'ON'}(min_side={args.tile_min_side}, tile={args.tile}, ov={args.overlap}), "
          f"pad_ratio={args.pad_ratio}, pad_min={args.pad_min}, max_area_ratio={args.max_area_ratio}, "
          f"post_open={args.post_open}, post_min_area={args.post_min_area}, debug_rois={args.debug_rois}"
          )
    print("DET_CLASSES:", args.det_classes if args.det_classes is not None else "ALL")
    print("======================================================")

    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] cannot read: {p}")
            continue

        stem = p.stem
        full_mask, det_res = cascade_one_image_v3(
            img,
            det_model,
            seg_model,
            det_conf=args.det_conf,
            det_iou=args.det_iou,
            seg_conf=args.seg_conf,
            seg_thr=args.seg_thr,
            seg_imgsz=args.seg_imgsz,
            pad_ratio=args.pad_ratio,
            pad_min=args.pad_min,
            max_rois=args.max_rois,
            allowed_det_classes=args.det_classes,
            max_area_ratio=args.max_area_ratio,
            tile_min_side=args.tile_min_side,
            tile=args.tile,
            overlap=args.overlap,
            use_tile_for_big_roi=(not args.no_tile),
            debug_dir=debug_dir,
            debug_prefix=stem,
            post_open_ksize=args.post_open,
            post_min_area=args.post_min_area,
        )

        det_vis = draw_det_boxes(img, det_res, det_names, conf_thr=args.det_conf)
        overlay = overlay_mask_red(det_vis, full_mask, alpha=0.45)

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
