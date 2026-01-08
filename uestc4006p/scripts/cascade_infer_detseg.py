"""
UESTC4006P - Cascade Inference v3c (Class-Aware) (Det -> Crop/Tile -> Seg -> Stitch Back)
=======================================================================================

适用场景：
- Seg 用 CRACK500 等近景裂缝数据训练（patch 分布）
- 推理在 RDD 等远景整图（尺度差异巨大）
- Det 框可能很大（如 D20/D40 框到整块路面/斑马线），直接 ROI->Seg 会尺度塌缩导致 mask 全空

v3c 核心增强（适合中期展示）：
1) Seg 推理显式 imgsz 可调（默认 1280）
2) ROI 过大时自动 tile（滑窗 + overlap）
3) Class-aware 策略：对 D20/D40（大纹理型病害）启用更强 tile、更松阈值、可选 CLAHE
4) 可选：只用部分 det 类、过滤超大框、后处理去噪、debug 输出 ROI/Tile 可视化

输出：
- <image>__mask.png      : stitched binary mask (255=crack-like)
- <image>__overlay.jpg   : det boxes + red mask overlay
- RUN_INFO.txt           : 运行参数与权重、数据来源记录
- debug_rois/* (可选)    : ROI/Tile 叠加图，方便定位“为什么 seg 空”
"""

import argparse
from pathlib import Path
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

def clahe_bgr(img_bgr: np.ndarray, clip=2.0, grid=8):
    """
    轻量局部对比度增强（仅建议对 D20/D40 类 ROI 使用）
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

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

def cascade_one_image_v3c(
    img_bgr: np.ndarray,
    det_model: YOLO,
    seg_model: YOLO,
    det_conf=0.15,
    det_iou=0.50,

    # base seg policy (for D00/D10)
    seg_conf=0.10,
    seg_thr=0.30,
    seg_imgsz=1280,

    pad_ratio=0.15,
    pad_min=16,
    max_rois=80,
    allowed_det_classes=None,      # e.g. [0] or None

    max_area_ratio=0.60,           # det box area ratio vs full image (only for logging/guard)

    # base tiling policy
    tile_min_side=1400,
    tile=1280,
    overlap=256,
    use_tile_for_big_roi=True,

    # class-aware big damage policy (for D20/D40)
    big_damage_class_ids=(2, 3),
    big_seg_conf=0.08,
    big_seg_thr=0.25,
    big_seg_imgsz=1280,
    big_force_tile=True,
    big_tile=1280,
    big_overlap=384,
    big_use_clahe=False,
    clahe_clip=2.0,
    clahe_grid=8,

    debug_dir: Path = None,
    debug_prefix: str = "",

    post_open_ksize=0,
    post_min_area=0,
):
    H, W = img_bgr.shape[:2]
    full_mask = np.zeros((H, W), dtype=np.uint8)

    det_res = det_model.predict(img_bgr, conf=det_conf, iou=det_iou, verbose=False)[0]
    if det_res.boxes is None or len(det_res.boxes) == 0:
        return full_mask, det_res

    xyxy = det_res.boxes.xyxy.detach().cpu().numpy()
    conf = det_res.boxes.conf.detach().cpu().numpy()
    cls  = det_res.boxes.cls.detach().cpu().numpy().astype(int)

    # filter classes
    idx = np.arange(len(xyxy))
    if allowed_det_classes is not None:
        allowed = set(allowed_det_classes)
        idx = np.array([i for i in idx if cls[i] in allowed], dtype=int)

    if idx.size == 0:
        return full_mask, det_res

    xyxy = xyxy[idx]
    conf = conf[idx]
    cls  = cls[idx]

    # limit rois
    if len(xyxy) > max_rois:
        order = np.argsort(-conf)[:max_rois]
        xyxy = xyxy[order]
        conf = conf[order]
        cls  = cls[order]

    roi_id = 0
    big_ids = set(int(x) for x in big_damage_class_ids)

    for (box, ccls) in zip(xyxy, cls):
        roi_id += 1
        det_class = int(ccls)
        x1, y1, x2, y2 = box.tolist()

        # expand
        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, W, H, pad_ratio, pad_min)
        bw, bh = (x2 - x1), (y2 - y1)
        _ = (bw * bh) / max(1.0, (W * H))  # for potential future guard

        roi = img_bgr[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]
        if rh < 2 or rw < 2:
            continue

        # choose policy by det class
        is_big = det_class in big_ids

        if is_big and big_use_clahe:
            roi = clahe_bgr(roi, clip=clahe_clip, grid=clahe_grid)

        # pick local parameters
        if is_big:
            local_seg_conf  = float(big_seg_conf)
            local_seg_thr   = float(big_seg_thr)
            local_seg_imgsz = int(max(seg_imgsz, big_seg_imgsz))
            local_tile      = int(big_tile)
            local_overlap   = int(big_overlap)
            force_tile      = bool(big_force_tile)
            local_tile_min_side = 0  # force trigger
        else:
            local_seg_conf  = float(seg_conf)
            local_seg_thr   = float(seg_thr)
            local_seg_imgsz = int(seg_imgsz)
            local_tile      = int(tile)
            local_overlap   = int(overlap)
            force_tile      = False
            local_tile_min_side = int(tile_min_side)

        if debug_dir is not None:
            cv2.imwrite(str(debug_dir / f"{debug_prefix}__c{det_class}__roi_{roi_id:03d}_{x1}_{y1}_{x2}_{y2}.jpg"), roi)

        # tile decision
        big_roi = True if force_tile else (max(rh, rw) >= local_tile_min_side)
        use_tile = use_tile_for_big_roi and (big_roi if (not force_tile) else True)

        if use_tile:
            roi_mask = seg_on_roi_with_tiling(
                seg_model,
                roi,
                seg_conf=local_seg_conf,
                seg_thr=local_seg_thr,
                seg_imgsz=local_seg_imgsz,
                tile=local_tile,
                overlap=local_overlap,
                debug_dir=debug_dir,
                debug_prefix=f"{debug_prefix}__c{det_class}__roi{roi_id:03d}"
            )
        else:
            seg_res = seg_model.predict(
                roi,
                conf=local_seg_conf,
                iou=0.5,
                imgsz=local_seg_imgsz,
                verbose=False
            )[0]
            roi_mask = yolo_seg_union_mask(seg_res, rh, rw, seg_conf=local_seg_conf, thr=local_seg_thr)

        if post_open_ksize or post_min_area:
            roi_mask = postprocess_mask(roi_mask, open_ksize=post_open_ksize, min_area=post_min_area)

        if debug_dir is not None:
            roi_ov = overlay_mask_red(roi, roi_mask, alpha=0.55)
            cv2.imwrite(str(debug_dir / f"{debug_prefix}__c{det_class}__roi_{roi_id:03d}_{x1}_{y1}_{x2}_{y2}__overlay.jpg"), roi_ov)

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
    ap.add_argument("--outdir", default="uestc4006p/runs", help="output dir")

    # Det params
    ap.add_argument("--det-conf", type=float, default=0.15)
    ap.add_argument("--det-iou", type=float, default=0.50)

    # Base seg params (for D00/D10)
    ap.add_argument("--seg-conf", type=float, default=0.10)
    ap.add_argument("--seg-thr",  type=float, default=0.30)
    ap.add_argument("--seg-imgsz", type=int, default=1280)

    # ROI & cascade
    ap.add_argument("--pad-ratio", type=float, default=0.15)
    ap.add_argument("--pad-min", type=int, default=16)
    ap.add_argument("--max-rois", type=int, default=80)
    ap.add_argument("--det-classes", nargs="*", type=int, default=None)

    ap.add_argument("--max-area-ratio", type=float, default=0.60)

    # Base tiling params
    ap.add_argument("--tile-min-side", type=int, default=1400)
    ap.add_argument("--tile", type=int, default=1280)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--no-tile", action="store_true")

    # Class-aware: big damage classes (D20/D40)
    ap.add_argument("--big-classes", nargs="*", type=int, default=[2, 3],
                    help="det class ids treated as big-damage (default: 2 3)")
    ap.add_argument("--big-seg-conf", type=float, default=0.08)
    ap.add_argument("--big-seg-thr",  type=float, default=0.25)
    ap.add_argument("--big-seg-imgsz", type=int, default=1280)
    ap.add_argument("--big-force-tile", action="store_true",
                    help="force tiling for big classes (recommended)")
    ap.add_argument("--big-tile", type=int, default=1280)
    ap.add_argument("--big-overlap", type=int, default=384)

    ap.add_argument("--big-clahe", action="store_true",
                    help="apply CLAHE on ROI for big classes (optional)")
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-grid", type=int, default=8)

    # Postprocess (optional)
    ap.add_argument("--post-open", type=int, default=0)
    ap.add_argument("--post-min-area", type=int, default=0)

    # Debug
    ap.add_argument("--debug-rois", action="store_true")

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

    run_dir = outdir / f"cascade_v3c__{det_short}__{seg_short}"
    ensure_dir(run_dir)

    debug_dir = None
    if args.debug_rois:
        debug_dir = run_dir / "debug_rois"
        ensure_dir(debug_dir)

    info_path = run_dir / "RUN_INFO.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("UESTC4006P - Cascade Inference v3c (Class-Aware) Run Info\n")
        f.write("====================================================\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n[Weights]\n")
        f.write(f"DET: {det_path}\n")
        f.write(f"SEG: {seg_path}\n")
        f.write("\n[Test Source]\n")
        f.write(f"SOURCE: {src}\n")
        f.write("\n[Params]\n")
        f.write(f"det_conf={args.det_conf}, det_iou={args.det_iou}\n")
        f.write(f"base_seg_conf={args.seg_conf}, base_seg_thr={args.seg_thr}, base_seg_imgsz={args.seg_imgsz}\n")
        f.write(f"pad_ratio={args.pad_ratio}, pad_min={args.pad_min}\n")
        f.write(f"max_rois={args.max_rois}, det_classes={args.det_classes if args.det_classes is not None else 'ALL'}\n")
        f.write(f"max_area_ratio={args.max_area_ratio}\n")
        f.write(f"use_tile_for_big_roi={not args.no_tile}, tile_min_side={args.tile_min_side}, tile={args.tile}, overlap={args.overlap}\n")
        f.write("\n[Class-aware Big-Damage Policy]\n")
        f.write(f"big_classes={args.big_classes}\n")
        f.write(f"big_seg_conf={args.big_seg_conf}, big_seg_thr={args.big_seg_thr}, big_seg_imgsz={args.big_seg_imgsz}\n")
        f.write(f"big_force_tile={args.big_force_tile}, big_tile={args.big_tile}, big_overlap={args.big_overlap}\n")
        f.write(f"big_clahe={args.big_clahe}, clahe_clip={args.clahe_clip}, clahe_grid={args.clahe_grid}\n")
        f.write("\n[Postprocess]\n")
        f.write(f"post_open={args.post_open}, post_min_area={args.post_min_area}\n")
        f.write("\n[Debug]\n")
        f.write(f"debug_rois={args.debug_rois}\n")
        f.write("\n[Outputs]\n")
        f.write("- <image>__mask.png : final stitched binary mask (255=crack-like)\n")
        f.write("- <image>__overlay.jpg : det boxes + red mask overlay\n")
        if args.debug_rois:
            f.write("- debug_rois/* : ROI and tile overlays for debugging\n")
        f.write("====================================================\n")

    print("RUN_INFO:", info_path)
    print("RUN_DIR :", run_dir)

    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] cannot read: {p}")
            continue

        stem = p.stem

        full_mask, det_res = cascade_one_image_v3c(
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

            big_damage_class_ids=tuple(int(x) for x in args.big_classes),
            big_seg_conf=args.big_seg_conf,
            big_seg_thr=args.big_seg_thr,
            big_seg_imgsz=args.big_seg_imgsz,
            big_force_tile=args.big_force_tile,
            big_tile=args.big_tile,
            big_overlap=args.big_overlap,
            big_use_clahe=args.big_clahe,
            clahe_clip=args.clahe_clip,
            clahe_grid=args.clahe_grid,

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
