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

# 级联推理v3c：Det出框→ROI裁剪/超大ROI滑窗切块→Seg分割→拼回整图；支持D20/D40类策略(更强tile/更松阈值/可选CLAHE)；输出mask/overlay/RUN_INFO与debug_rois

import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from naming_utils import (
    make_output_dir_name,
    resolve_unique_dir,
    shorten_dataset_name,
    shorten_model_name,
    write_run_meta_yaml,
    extract_weight_meta,
    now_iso,
)

# ===================== ROI-v3 统一默认配置（便于集中调参） =====================
# 说明：
# - 仅推理与后处理参数，保持模型结构不变
# - 默认尽量兼容当前逻辑；ROI-v3 增强能力通过开关启用
ROI_V3_DEFAULTS = {
    # 主开关
    "enabled": False,
    # A) ROI padding（基础 + D20 专用）
    "pad_ratio": 0.15,
    "pad_min": 16,
    "d20_class_id": 2,
    "d20_pad_ratio": 0.20,
    "d20_pad_min": 24,
    # B) oversized ROI 自适应切块
    "adaptive_tile": False,
    "adaptive_min_tile": 768,
    "adaptive_max_tile": 1536,
    "adaptive_target_long_tiles": 4,
    # C) overlap + stitch 融合
    "tile_fusion": "max",  # max / avg
    "tile_fusion_thr": 0.30,
    "tile_overlap_ratio": 0.30,
    # D) 可选高分辨率 seg inference
    "highres_enabled": False,
    "highres_imgsz": 1600,
    "highres_min_side": 1600,
    # E) D20 单独阈值
    "d20_seg_conf": 0.08,
    "d20_seg_thr": 0.22,
    "d20_seg_iou": 0.50,
    "d20_seg_imgsz": 1280,
    # F) CLAHE（D20 专用）
    "d20_clahe": False,
    "clahe_clip": 2.0,
    "clahe_grid": 8,
    # G) 轻量后处理
    "post_open": 0,
    "post_close": 0,
    "post_min_area": 0,
    # H) 多级调试
    "debug_levels": False,
    # I) D20结构判别 + 道路标线抑制
    "enable_d20_structure_score": False,
    "d20_structure_thr": 0.45,
    "enable_lane_marking_suppress": False,
    "lane_marking_bright_thr": 200,
    "lane_marking_sat_thr": 45,
}


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

def yolo_seg_union_mask(seg_result, roi_h, roi_w, seg_conf=0.25, thr=0.5, return_prob=False):
    """
    将 Ultralytics segmentation 输出转换为 ROI 尺寸的 union 二值 mask。
    - seg_result.masks.data: [N, h, w] float(0~1)
    - 按实例置信度过滤 -> union -> resize 回 ROI -> 二值化
    """
    if seg_result.masks is None:
        if return_prob:
            return np.zeros((roi_h, roi_w), dtype=np.float32)
        return np.zeros((roi_h, roi_w), dtype=np.uint8)

    m = seg_result.masks.data
    m = m.detach().cpu().numpy()

    # 按实例置信度过滤（boxes.conf 与 masks 一一对应）
    if seg_result.boxes is not None and len(seg_result.boxes) == m.shape[0]:
        scores = seg_result.boxes.conf.detach().cpu().numpy()
        keep = scores >= seg_conf
        m = m[keep] if keep.any() else m[:0]

    if m.shape[0] == 0:
        if return_prob:
            return np.zeros((roi_h, roi_w), dtype=np.float32)
        return np.zeros((roi_h, roi_w), dtype=np.uint8)

    union = np.max(m, axis=0)  # [mh, mw]
    if return_prob:
        # 概率图用于 tile 融合（avg/vote）；使用线性插值减少边界块感
        prob = cv2.resize(union.astype(np.float32), (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
        return np.clip(prob, 0.0, 1.0)
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

def postprocess_mask(mask_u8: np.ndarray, open_ksize=0, close_ksize=0, min_area=0):
    """
    简单后处理：
    - open 去噪点（不建议核太大，避免伤细裂缝）
    - 连通域面积过滤（过滤极小碎片）
    """
    out = mask_u8.copy()

    if open_ksize and open_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)

    # 轻量闭运算：用于补小裂缝断点、提升连通性（可选）
    if close_ksize and close_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)

    if min_area and min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
        keep = np.zeros_like(out, dtype=np.uint8)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                keep[labels == i] = 255
        out = keep

    return out


def _skeletonize_mask(mask_u8: np.ndarray):
    """
    轻量骨架化：仅用OpenCV形态学迭代，避免引入额外依赖。
    """
    work = ((mask_u8 > 0).astype(np.uint8)) * 255
    skel = np.zeros_like(work, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    # 迭代上限用于防止极端输入导致长循环
    for _ in range(2048):
        if cv2.countNonZero(work) == 0:
            break
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(work, opened)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded
    return skel


def extract_d20_structure_features(roi_mask_u8: np.ndarray):
    """
    从ROI二值mask提取D20结构特征（轻量实现）。
    """
    bw = (roi_mask_u8 > 0).astype(np.uint8)
    h, w = bw.shape[:2]
    roi_area = max(1, int(h * w))
    mask_pixels = int(np.count_nonzero(bw))
    crack_area_ratio = float(mask_pixels / roi_area)

    feats = {
        "crack_area_ratio": crack_area_ratio,
        "connected_components_count": 0,
        "largest_component_ratio": 0.0,
        "skeleton_length": 0,
        "branch_points_count": 0,
        "branch_density": 0.0,
        "orientation_entropy": 0.0,
        "mask_pixels": mask_pixels,
    }
    if mask_pixels == 0:
        return feats

    num, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    comp_areas = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.empty((0,), dtype=np.int32)
    cc_count = int(comp_areas.size)
    largest_ratio = float(comp_areas.max() / max(1, mask_pixels)) if cc_count > 0 else 0.0

    skel = _skeletonize_mask(roi_mask_u8)
    sk_bin = (skel > 0).astype(np.uint8)
    skeleton_length = int(np.count_nonzero(sk_bin))

    branch_points = 0
    if skeleton_length > 0:
        # 统计8邻域>=3的骨架像素作为分支点
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int16)
        nei = cv2.filter2D(sk_bin, cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT)
        branch_points = int(np.logical_and(sk_bin > 0, nei >= 3).sum())
    branch_density = float(branch_points / max(1, skeleton_length))

    # 方向熵（简化版）：基于骨架梯度方向的加权直方图熵，归一化到0~1
    orientation_entropy = 0.0
    if skeleton_length > 8:
        sk_f = sk_bin.astype(np.float32)
        gx = cv2.Sobel(sk_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sk_f, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        valid = mag > 1e-6
        if np.any(valid):
            angles = np.mod(np.arctan2(gy[valid], gx[valid]), np.pi)
            hist, _ = np.histogram(angles, bins=12, range=(0.0, np.pi), weights=mag[valid])
            hsum = float(hist.sum())
            if hsum > 1e-8:
                p = hist / hsum
                p = p[p > 1e-8]
                orientation_entropy = float((-np.sum(p * np.log(p))) / np.log(12.0))

    feats["connected_components_count"] = cc_count
    feats["largest_component_ratio"] = largest_ratio
    feats["skeleton_length"] = skeleton_length
    feats["branch_points_count"] = branch_points
    feats["branch_density"] = branch_density
    feats["orientation_entropy"] = orientation_entropy
    return feats


def score_d20_structure(features: dict):
    """
    结构评分：0~1，分数越高越像裂缝网络结构。
    """
    area_ratio = float(features.get("crack_area_ratio", 0.0))
    cc_count = float(features.get("connected_components_count", 0))
    largest_ratio = float(features.get("largest_component_ratio", 0.0))
    skeleton_length = float(features.get("skeleton_length", 0))
    branch_density = float(features.get("branch_density", 0.0))
    orient_entropy = float(features.get("orientation_entropy", 0.0))
    mask_pixels = float(max(1, int(features.get("mask_pixels", 1))))

    if area_ratio <= 0.0 or skeleton_length <= 0.0:
        return 0.0

    # 面积过小给低分，面积过大给惩罚（大块白线/斑马线误检常见）
    area_support = float(np.clip((area_ratio - 0.001) / 0.035, 0.0, 1.0))
    area_penalty = float(np.clip((area_ratio - 0.22) / 0.30, 0.0, 1.0))
    area_score = max(0.0, area_support * (1.0 - 0.6 * area_penalty))

    cc_score = float(np.clip(cc_count / 8.0, 0.0, 1.0))
    largest_score = float(1.0 - np.clip((largest_ratio - 0.90) / 0.10, 0.0, 1.0))

    # 骨架长度与mask像素比值，反映细长结构占比
    sk_ratio = skeleton_length / mask_pixels
    skeleton_score = float(np.clip((sk_ratio - 0.10) / 0.50, 0.0, 1.0))

    branch_score = float(np.clip(branch_density / 0.035, 0.0, 1.0))
    entropy_score = float(np.clip((orient_entropy - 0.15) / 0.65, 0.0, 1.0))

    score = (
        0.20 * area_score
        + 0.15 * cc_score
        + 0.15 * largest_score
        + 0.20 * skeleton_score
        + 0.15 * branch_score
        + 0.15 * entropy_score
    )
    return float(np.clip(score, 0.0, 1.0))


def suppress_lane_marking_regions(
    roi_bgr: np.ndarray,
    roi_mask_u8: np.ndarray,
    bright_thr: int = 200,
    sat_thr: int = 45,
):
    """
    检测疑似白色道路标线区域，返回用于抑制的mask（255=建议抑制）。
    """
    if roi_bgr is None or roi_mask_u8 is None:
        return np.zeros((0, 0), dtype=np.uint8)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    lane_like = np.logical_and(v >= int(bright_thr), s <= int(sat_thr)).astype(np.uint8) * 255
    if cv2.countNonZero(lane_like) == 0:
        return np.zeros_like(roi_mask_u8, dtype=np.uint8)

    # 轻量形态学，让白线区域更连续稳定
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lane_like = cv2.morphologyEx(lane_like, cv2.MORPH_OPEN, k)
    lane_like = cv2.morphologyEx(lane_like, cv2.MORPH_CLOSE, k)
    lane_like = cv2.dilate(lane_like, k, iterations=1)
    return lane_like


def compute_adaptive_tile_params(
    roi_h: int,
    roi_w: int,
    base_tile: int,
    base_overlap: int,
    enable_adaptive: bool = False,
    min_tile: int = 768,
    max_tile: int = 1536,
    target_long_tiles: int = 4,
    overlap_ratio: float = 0.30,
):
    """
    ROI-v3 自适应切块：
    - 根据 ROI 长边估算 tile 尺寸（限制在 min/max）
    - overlap 可按比例自适应，降低拼接缝
    """
    tile = int(base_tile)
    overlap = int(base_overlap)
    if not enable_adaptive:
        overlap = max(0, min(overlap, max(1, tile - 1)))
        return tile, overlap

    long_side = int(max(roi_h, roi_w))
    target_long_tiles = max(1, int(target_long_tiles))
    est_tile = int(np.ceil(long_side / target_long_tiles))
    tile = int(np.clip(est_tile, int(min_tile), int(max_tile)))
    tile = max(32, int(np.ceil(tile / 32.0) * 32))  # 与常见 YOLO 步长对齐

    if overlap_ratio is not None and overlap_ratio > 0:
        overlap = int(round(tile * float(overlap_ratio)))
    overlap = max(0, min(overlap, max(1, tile - 1)))
    return tile, overlap

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
    seg_iou: float = 0.5,
    adaptive_tile: bool = False,
    adaptive_min_tile: int = 768,
    adaptive_max_tile: int = 1536,
    adaptive_target_long_tiles: int = 4,
    overlap_ratio: float = 0.30,
    fusion_mode: str = "max",
    fusion_thr: float = 0.30,
    debug_levels: bool = False,
    debug_dir: Path = None,
    debug_prefix: str = "",
):
    """
    对 ROI 进行 tile 分割（滑窗 + overlap），并 union 拼回 ROI mask。
    """
    rh, rw = roi_bgr.shape[:2]
    out = np.zeros((rh, rw), dtype=np.uint8)
    prob_sum = np.zeros((rh, rw), dtype=np.float32)
    prob_cnt = np.zeros((rh, rw), dtype=np.float32)

    tile, overlap = compute_adaptive_tile_params(
        roi_h=rh,
        roi_w=rw,
        base_tile=tile,
        base_overlap=overlap,
        enable_adaptive=adaptive_tile,
        min_tile=adaptive_min_tile,
        max_tile=adaptive_max_tile,
        target_long_tiles=adaptive_target_long_tiles,
        overlap_ratio=overlap_ratio,
    )
    step = max(1, int(tile) - int(overlap))

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
                iou=seg_iou,
                imgsz=seg_imgsz,
                verbose=False
            )[0]

            if fusion_mode == "avg":
                patch_prob = yolo_seg_union_mask(
                    seg_res, ph, pw, seg_conf=seg_conf, thr=seg_thr, return_prob=True
                )
                prob_sum[y0:y0 + ph, x0:x0 + pw] += patch_prob
                prob_cnt[y0:y0 + ph, x0:x0 + pw] += 1.0
                patch_mask = (patch_prob > seg_thr).astype(np.uint8) * 255
            else:
                patch_mask = yolo_seg_union_mask(seg_res, ph, pw, seg_conf=seg_conf, thr=seg_thr)
                out[y0:y0 + ph, x0:x0 + pw] = np.maximum(out[y0:y0 + ph, x0:x0 + pw], patch_mask)

            if debug_dir is not None:
                tcount += 1
                ov = overlay_mask_red(patch, patch_mask, alpha=0.55)
                if debug_levels:
                    d_tiles = debug_dir / "03_tiles"
                    d_local = debug_dir / "04_local_masks"
                    ensure_dir(d_tiles)
                    ensure_dir(d_local)
                    cv2.imwrite(str(d_tiles / f"{debug_prefix}__tile_{tcount:03d}_{x0}_{y0}_{x0+pw}_{y0+ph}.jpg"), ov)
                    cv2.imwrite(str(d_local / f"{debug_prefix}__tilemask_{tcount:03d}_{x0}_{y0}_{x0+pw}_{y0+ph}.png"), patch_mask)
                else:
                    cv2.imwrite(str(debug_dir / f"{debug_prefix}__tile_{tcount:03d}_{x0}_{y0}_{x0+pw}_{y0+ph}.jpg"), ov)

    if fusion_mode == "avg":
        valid = prob_cnt > 0
        mean_prob = np.zeros((rh, rw), dtype=np.float32)
        mean_prob[valid] = prob_sum[valid] / np.maximum(prob_cnt[valid], 1e-6)
        out = (mean_prob >= float(fusion_thr)).astype(np.uint8) * 255

    return out


# ===================== 主流程（单图） =====================

def cascade_one_image_v3c(
    img_bgr: np.ndarray,
    det_model: YOLO,
    seg_model: YOLO,
    det_conf=0.15,
    det_iou=0.50,
    det_imgsz=0,

    # base seg policy (for D00/D10)
    seg_conf=0.10,
    seg_thr=0.30,
    seg_iou=0.50,
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
    big_seg_iou=0.5,
    big_force_tile=True,
    big_tile=1280,
    big_overlap=384,
    big_use_clahe=False,
    clahe_clip=2.0,
    clahe_grid=8,

    debug_dir: Path = None,
    debug_prefix: str = "",

    post_open_ksize=0,
    post_close_ksize=0,
    post_min_area=0,

    # ROI-v3 开关与增强参数（默认关闭，兼容旧逻辑）
    roi_v3=False,
    d20_class_id=2,
    d20_pad_ratio=0.20,
    d20_pad_min=24,
    d20_seg_conf=0.08,
    d20_seg_thr=0.22,
    d20_seg_iou=0.50,
    d20_seg_imgsz=1280,
    d20_clahe=False,
    adaptive_tile=False,
    adaptive_min_tile=768,
    adaptive_max_tile=1536,
    adaptive_target_long_tiles=4,
    tile_overlap_ratio=0.30,
    tile_fusion="max",
    tile_fusion_thr=0.30,
    highres_enabled=False,
    highres_imgsz=1600,
    highres_min_side=1600,
    debug_levels=False,
    enable_d20_structure_score=False,
    d20_structure_thr=0.45,
    enable_lane_marking_suppress=False,
    lane_marking_bright_thr=200,
    lane_marking_sat_thr=45,
):
    H, W = img_bgr.shape[:2]
    full_mask = np.zeros((H, W), dtype=np.uint8)

    debug_canvas = None
    if debug_dir is not None and debug_levels:
        ensure_dir(debug_dir / "00_box_debug")
        ensure_dir(debug_dir / "01_rois_raw")
        ensure_dir(debug_dir / "02_rois_expanded")
        ensure_dir(debug_dir / "03_tiles")
        ensure_dir(debug_dir / "04_local_masks")
        ensure_dir(debug_dir / "05_fused_masks")
        debug_canvas = img_bgr.copy()

    det_kwargs = dict(conf=det_conf, iou=det_iou, verbose=False)
    if det_imgsz is not None and int(det_imgsz) > 0:
        det_kwargs["imgsz"] = int(det_imgsz)
    det_res = det_model.predict(img_bgr, **det_kwargs)[0]
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
        raw_x1, raw_y1, raw_x2, raw_y2 = clamp_box(x1, y1, x2, y2, W, H)

        # expand
        local_pad_ratio = pad_ratio
        local_pad_min = pad_min
        if roi_v3 and det_class == int(d20_class_id):
            local_pad_ratio = float(d20_pad_ratio)
            local_pad_min = int(d20_pad_min)
        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, W, H, local_pad_ratio, local_pad_min)
        bw, bh = (x2 - x1), (y2 - y1)
        _ = (bw * bh) / max(1.0, (W * H))  # for potential future guard

        if debug_canvas is not None:
            cv2.rectangle(debug_canvas, (raw_x1, raw_y1), (raw_x2, raw_y2), (0, 255, 255), 2)  # 原框
            cv2.rectangle(debug_canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 扩框

        raw_roi = img_bgr[raw_y1:raw_y2, raw_x1:raw_x2]
        roi = img_bgr[y1:y2, x1:x2]
        rh, rw = roi.shape[:2]
        if rh < 2 or rw < 2:
            continue

        # choose policy by det class
        is_big = det_class in big_ids

        if is_big and big_use_clahe:
            roi = clahe_bgr(roi, clip=clahe_clip, grid=clahe_grid)
        if roi_v3 and det_class == int(d20_class_id) and d20_clahe:
            roi = clahe_bgr(roi, clip=clahe_clip, grid=clahe_grid)

        # pick local parameters
        if is_big:
            local_seg_conf  = float(big_seg_conf)
            local_seg_thr   = float(big_seg_thr)
            local_seg_imgsz = int(max(seg_imgsz, big_seg_imgsz))
            local_seg_iou   = float(big_seg_iou)
            local_tile      = int(big_tile)
            local_overlap   = int(big_overlap)
            force_tile      = bool(big_force_tile)
            local_tile_min_side = 0  # force trigger
        else:
            local_seg_conf  = float(seg_conf)
            local_seg_thr   = float(seg_thr)
            local_seg_iou   = float(seg_iou)
            local_seg_imgsz = int(seg_imgsz)
            local_tile      = int(tile)
            local_overlap   = int(overlap)
            force_tile      = False
            local_tile_min_side = int(tile_min_side)

        # ROI-v3: D20 单独阈值/分辨率/IoU 策略
        if roi_v3 and det_class == int(d20_class_id):
            local_seg_conf = float(d20_seg_conf)
            local_seg_thr = float(d20_seg_thr)
            local_seg_iou = float(d20_seg_iou)
            local_seg_imgsz = int(max(local_seg_imgsz, d20_seg_imgsz))

        # ROI-v3: 大 ROI 可选高分辨率推理
        if roi_v3 and highres_enabled and max(rh, rw) >= int(highres_min_side):
            local_seg_imgsz = int(max(local_seg_imgsz, highres_imgsz))

        if debug_dir is not None:
            if debug_levels:
                cv2.imwrite(
                    str(debug_dir / "01_rois_raw" / f"{debug_prefix}__c{det_class}__roi_{roi_id:03d}_{raw_x1}_{raw_y1}_{raw_x2}_{raw_y2}.jpg"),
                    raw_roi,
                )
                cv2.imwrite(
                    str(debug_dir / "02_rois_expanded" / f"{debug_prefix}__c{det_class}__roi_{roi_id:03d}_{x1}_{y1}_{x2}_{y2}.jpg"),
                    roi,
                )
            else:
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
                seg_iou=local_seg_iou,
                adaptive_tile=bool(roi_v3 and adaptive_tile),
                adaptive_min_tile=adaptive_min_tile,
                adaptive_max_tile=adaptive_max_tile,
                adaptive_target_long_tiles=adaptive_target_long_tiles,
                overlap_ratio=tile_overlap_ratio,
                fusion_mode=tile_fusion if roi_v3 else "max",
                fusion_thr=tile_fusion_thr,
                debug_levels=debug_levels,
                debug_dir=debug_dir,
                debug_prefix=f"{debug_prefix}__c{det_class}__roi{roi_id:03d}"
            )
        else:
            seg_res = seg_model.predict(
                roi,
                conf=local_seg_conf,
                iou=local_seg_iou,
                imgsz=local_seg_imgsz,
                verbose=False
            )[0]
            roi_mask = yolo_seg_union_mask(seg_res, rh, rw, seg_conf=local_seg_conf, thr=local_seg_thr)

        roi_mask_before_post = roi_mask.copy()
        if debug_dir is not None and debug_levels:
            cv2.imwrite(
                str(debug_dir / "04_local_masks" / f"{debug_prefix}__c{det_class}__roi_{roi_id:03d}_{x1}_{y1}_{x2}_{y2}__prepost.png"),
                roi_mask_before_post,
            )

        if post_open_ksize or post_close_ksize or post_min_area:
            roi_mask = postprocess_mask(
                roi_mask,
                open_ksize=post_open_ksize,
                close_ksize=post_close_ksize,
                min_area=post_min_area,
            )

        # D20/大病害结构判别：放在后处理之后、写回full_mask之前
        low_structure_score = False
        lane_suppress_applied = False
        lane_overlap_ratio = 0.0
        d20_structure_score = 1.0
        d20_feats = None
        is_structure_target = (det_class == int(d20_class_id))
        if enable_d20_structure_score and is_structure_target:
            d20_feats = extract_d20_structure_features(roi_mask)
            d20_structure_score = score_d20_structure(d20_feats)
            low_structure_score = d20_structure_score < float(d20_structure_thr)

            # 低结构分时，不删整块ROI；仅在疑似白线区域做局部抑制
            if low_structure_score and enable_lane_marking_suppress:
                lane_sup_mask = suppress_lane_marking_regions(
                    roi,
                    roi_mask,
                    bright_thr=lane_marking_bright_thr,
                    sat_thr=lane_marking_sat_thr,
                )
                if lane_sup_mask.shape[:2] == roi_mask.shape[:2]:
                    overlap_px = int(np.logical_and(roi_mask > 0, lane_sup_mask > 0).sum())
                    mask_px = int(np.count_nonzero(roi_mask))
                    lane_overlap_ratio = float(overlap_px / max(1, mask_px))
                    if overlap_px > 0:
                        keep_mask = np.where(lane_sup_mask > 0, 0, 255).astype(np.uint8)
                        before_px = mask_px
                        roi_mask = cv2.bitwise_and(roi_mask, keep_mask)
                        after_px = int(np.count_nonzero(roi_mask))
                        lane_suppress_applied = after_px < before_px

        if debug_dir is not None:
            roi_ov = overlay_mask_red(roi, roi_mask, alpha=0.55)
            if low_structure_score:
                cv2.putText(
                    roi_ov,
                    f"low-structure-score:{d20_structure_score:.2f}",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2,
                )
                if lane_suppress_applied:
                    cv2.putText(
                        roi_ov,
                        f"lane-suppress overlap:{lane_overlap_ratio:.2f}",
                        (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (0, 255, 255),
                        1,
                    )

            if debug_levels:
                cv2.imwrite(
                    str(debug_dir / "05_fused_masks" / f"{debug_prefix}__c{det_class}__roi_{roi_id:03d}_{x1}_{y1}_{x2}_{y2}__overlay.jpg"),
                    roi_ov,
                )
                cv2.imwrite(
                    str(debug_dir / "05_fused_masks" / f"{debug_prefix}__c{det_class}__roi_{roi_id:03d}_{x1}_{y1}_{x2}_{y2}__mask.png"),
                    roi_mask,
                )
                if low_structure_score and d20_feats is not None:
                    info_file = debug_dir / "05_fused_masks" / f"{debug_prefix}__c{det_class}__roi_{roi_id:03d}_{x1}_{y1}_{x2}_{y2}__structure.txt"
                    with open(info_file, "w", encoding="utf-8") as df:
                        df.write(f"low_structure_score=1, score={d20_structure_score:.4f}, thr={float(d20_structure_thr):.4f}\n")
                        df.write(f"lane_suppress_applied={lane_suppress_applied}, lane_overlap_ratio={lane_overlap_ratio:.4f}\n")
                        for k, v in d20_feats.items():
                            df.write(f"{k}={v}\n")
            else:
                cv2.imwrite(str(debug_dir / f"{debug_prefix}__c{det_class}__roi_{roi_id:03d}_{x1}_{y1}_{x2}_{y2}__overlay.jpg"), roi_ov)
                if low_structure_score and d20_feats is not None:
                    info_file = debug_dir / f"{debug_prefix}__c{det_class}__roi_{roi_id:03d}_{x1}_{y1}_{x2}_{y2}__structure.txt"
                    with open(info_file, "w", encoding="utf-8") as df:
                        df.write(f"low_structure_score=1, score={d20_structure_score:.4f}, thr={float(d20_structure_thr):.4f}\n")
                        df.write(f"lane_suppress_applied={lane_suppress_applied}, lane_overlap_ratio={lane_overlap_ratio:.4f}\n")
                        for k, v in d20_feats.items():
                            df.write(f"{k}={v}\n")

        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], roi_mask)

    if debug_canvas is not None:
        cv2.imwrite(str(debug_dir / "00_box_debug" / f"{debug_prefix}__raw_vs_expanded_boxes.jpg"), debug_canvas)

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
    ap.add_argument("--name-mode", choices=["new", "legacy"], default="new",
                    help="输出命名：new=mode_task_data_model[_tag]，legacy=旧命名")
    ap.add_argument("--run-tag", default="", help="可选核心标签")

    # Det params
    ap.add_argument("--det-conf", type=float, default=0.15)
    ap.add_argument("--det-iou", type=float, default=0.50)
    ap.add_argument("--det-imgsz", type=int, default=0,
                    help="det推理imgsz，<=0表示沿用模型默认")

    # Base seg params (for D00/D10)
    ap.add_argument("--seg-conf", type=float, default=0.10)
    ap.add_argument("--seg-thr",  type=float, default=0.30)
    ap.add_argument("--seg-iou",  type=float, default=0.50)
    ap.add_argument("--seg-imgsz", type=int, default=1280)

    # ROI & cascade
    ap.add_argument("--pad-ratio", type=float, default=ROI_V3_DEFAULTS["pad_ratio"])
    ap.add_argument("--pad-min", type=int, default=ROI_V3_DEFAULTS["pad_min"])
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
    ap.add_argument("--big-seg-iou",  type=float, default=0.50)
    ap.add_argument("--big-seg-imgsz", type=int, default=1280)
    ap.add_argument("--big-force-tile", action="store_true",
                    help="force tiling for big classes (recommended)")
    ap.add_argument("--big-tile", type=int, default=1280)
    ap.add_argument("--big-overlap", type=int, default=384)

    ap.add_argument("--big-clahe", action="store_true",
                    help="apply CLAHE on ROI for big classes (optional)")
    ap.add_argument("--clahe-clip", type=float, default=ROI_V3_DEFAULTS["clahe_clip"])
    ap.add_argument("--clahe-grid", type=int, default=ROI_V3_DEFAULTS["clahe_grid"])

    # Postprocess (optional)
    ap.add_argument("--post-open", type=int, default=ROI_V3_DEFAULTS["post_open"])
    ap.add_argument("--post-close", type=int, default=ROI_V3_DEFAULTS["post_close"])
    ap.add_argument("--post-min-area", type=int, default=ROI_V3_DEFAULTS["post_min_area"])

    # ROI-v3（默认关闭，确保可回退）
    ap.add_argument("--roi-v3", action="store_true",
                    help="启用ROI-v3增强策略（默认关闭，兼容旧逻辑）")
    ap.add_argument("--d20-class-id", type=int, default=ROI_V3_DEFAULTS["d20_class_id"])
    ap.add_argument("--d20-pad-ratio", type=float, default=ROI_V3_DEFAULTS["d20_pad_ratio"])
    ap.add_argument("--d20-pad-min", type=int, default=ROI_V3_DEFAULTS["d20_pad_min"])
    ap.add_argument("--d20-seg-conf", type=float, default=ROI_V3_DEFAULTS["d20_seg_conf"])
    ap.add_argument("--d20-seg-thr", type=float, default=ROI_V3_DEFAULTS["d20_seg_thr"])
    ap.add_argument("--d20-seg-iou", type=float, default=ROI_V3_DEFAULTS["d20_seg_iou"])
    ap.add_argument("--d20-seg-imgsz", type=int, default=ROI_V3_DEFAULTS["d20_seg_imgsz"])
    ap.add_argument("--d20-clahe", action="store_true",
                    help="仅对D20类别ROI启用CLAHE")
    ap.add_argument("--adaptive-tile", action="store_true",
                    help="启用oversized ROI自适应切块")
    ap.add_argument("--adaptive-min-tile", type=int, default=ROI_V3_DEFAULTS["adaptive_min_tile"])
    ap.add_argument("--adaptive-max-tile", type=int, default=ROI_V3_DEFAULTS["adaptive_max_tile"])
    ap.add_argument("--adaptive-target-long-tiles", type=int, default=ROI_V3_DEFAULTS["adaptive_target_long_tiles"])
    ap.add_argument("--tile-overlap-ratio", type=float, default=ROI_V3_DEFAULTS["tile_overlap_ratio"])
    ap.add_argument("--tile-fusion", choices=["max", "avg"], default=ROI_V3_DEFAULTS["tile_fusion"])
    ap.add_argument("--tile-fusion-thr", type=float, default=ROI_V3_DEFAULTS["tile_fusion_thr"])
    ap.add_argument("--highres-seg", action="store_true", help="启用高分辨率分割推理")
    ap.add_argument("--highres-imgsz", type=int, default=ROI_V3_DEFAULTS["highres_imgsz"])
    ap.add_argument("--highres-min-side", type=int, default=ROI_V3_DEFAULTS["highres_min_side"])
    ap.add_argument("--debug-levels", action="store_true",
                    help="输出多级debug结果：原框/扩框/tile/局部mask/融合mask")
    ap.add_argument("--enable-d20-structure-score", action="store_true",
                    help="启用D20/大病害ROI结构判别（后处理后生效）")
    ap.add_argument("--d20-structure-thr", type=float, default=ROI_V3_DEFAULTS["d20_structure_thr"],
                    help="D20结构判别阈值，低于该分数触发保守策略")
    ap.add_argument("--enable-lane-marking-suppress", action="store_true",
                    help="低结构分时启用疑似白色道路标线区域抑制")
    ap.add_argument("--lane-marking-bright-thr", type=int, default=ROI_V3_DEFAULTS["lane_marking_bright_thr"],
                    help="道路标线抑制：HSV-V亮度阈值")
    ap.add_argument("--lane-marking-sat-thr", type=int, default=ROI_V3_DEFAULTS["lane_marking_sat_thr"],
                    help="道路标线抑制：HSV-S饱和度阈值（低饱和更像白线）")

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

    det_parent = det_path.parent.parent.name if det_path.parent.name.lower() == "weights" else det_path.parent.name
    seg_parent = seg_path.parent.parent.name if seg_path.parent.name.lower() == "weights" else seg_path.parent.name

    det_short = short_tag_from_ckpt_parent(det_parent, "det")
    seg_short = short_tag_from_ckpt_parent(seg_parent, "seg")

    if args.name_mode == "legacy":
        run_dir = outdir / f"cascade__{det_short}__{seg_short}"
    else:
        data_token = src.name if src.is_dir() else src.parent.name
        model_token = shorten_model_name(seg_path.stem)
        main_name = make_output_dir_name(
            mode="pred",
            task="cascade",
            data=data_token,
            model=model_token,
            tag=args.run_tag,
        )
        _, run_dir = resolve_unique_dir(outdir, main_name)
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
        f.write(f"det_conf={args.det_conf}, det_iou={args.det_iou}, det_imgsz={args.det_imgsz}\n")
        f.write(f"base_seg_conf={args.seg_conf}, base_seg_thr={args.seg_thr}, base_seg_iou={args.seg_iou}, base_seg_imgsz={args.seg_imgsz}\n")
        f.write(f"pad_ratio={args.pad_ratio}, pad_min={args.pad_min}\n")
        f.write(f"max_rois={args.max_rois}, det_classes={args.det_classes if args.det_classes is not None else 'ALL'}\n")
        f.write(f"max_area_ratio={args.max_area_ratio}\n")
        f.write(f"use_tile_for_big_roi={not args.no_tile}, tile_min_side={args.tile_min_side}, tile={args.tile}, overlap={args.overlap}\n")
        f.write("\n[Class-aware Big-Damage Policy]\n")
        f.write(f"big_classes={args.big_classes}\n")
        f.write(f"big_seg_conf={args.big_seg_conf}, big_seg_thr={args.big_seg_thr}, big_seg_iou={args.big_seg_iou}, big_seg_imgsz={args.big_seg_imgsz}\n")
        f.write(f"big_force_tile={args.big_force_tile}, big_tile={args.big_tile}, big_overlap={args.big_overlap}\n")
        f.write(f"big_clahe={args.big_clahe}, clahe_clip={args.clahe_clip}, clahe_grid={args.clahe_grid}\n")
        f.write("\n[ROI-v3]\n")
        f.write(f"roi_v3={args.roi_v3}, d20_class_id={args.d20_class_id}\n")
        f.write(f"d20_pad_ratio={args.d20_pad_ratio}, d20_pad_min={args.d20_pad_min}\n")
        f.write(f"d20_seg_conf={args.d20_seg_conf}, d20_seg_thr={args.d20_seg_thr}, d20_seg_iou={args.d20_seg_iou}, d20_seg_imgsz={args.d20_seg_imgsz}\n")
        f.write(f"d20_clahe={args.d20_clahe}\n")
        f.write(f"adaptive_tile={args.adaptive_tile}, adaptive_min_tile={args.adaptive_min_tile}, adaptive_max_tile={args.adaptive_max_tile}, adaptive_target_long_tiles={args.adaptive_target_long_tiles}\n")
        f.write(f"tile_overlap_ratio={args.tile_overlap_ratio}, tile_fusion={args.tile_fusion}, tile_fusion_thr={args.tile_fusion_thr}\n")
        f.write(f"highres_seg={args.highres_seg}, highres_imgsz={args.highres_imgsz}, highres_min_side={args.highres_min_side}\n")
        f.write("\n[D20 Structure]\n")
        f.write(f"enable_d20_structure_score={args.enable_d20_structure_score}, d20_structure_thr={args.d20_structure_thr}\n")
        f.write(f"enable_lane_marking_suppress={args.enable_lane_marking_suppress}, lane_marking_bright_thr={args.lane_marking_bright_thr}, lane_marking_sat_thr={args.lane_marking_sat_thr}\n")
        f.write("\n[Postprocess]\n")
        f.write(f"post_open={args.post_open}, post_close={args.post_close}, post_min_area={args.post_min_area}\n")
        f.write("\n[Debug]\n")
        f.write(f"debug_rois={args.debug_rois}, debug_levels={args.debug_levels}\n")
        f.write("\n[Outputs]\n")
        f.write("- <image>__mask.png : final stitched binary mask (255=crack-like)\n")
        f.write("- <image>__overlay.jpg : det boxes + red mask overlay\n")
        if args.debug_rois:
            f.write("- debug_rois/* : ROI and tile overlays for debugging\n")
        f.write("====================================================\n")

    seg_wm = extract_weight_meta(str(seg_path))
    run_meta = {
        "mode": "pred",
        "task": "cascade",
        "dataset": str(src),
        "model": str(seg_path.stem),
        "tag": args.run_tag,
        "created_at": now_iso(),
        "weight_path": str(seg_path),
        "source": str(src),
        "notes": "",
        "weight_file": seg_wm["weight_file"],
        "weight_parent_dir": seg_wm["weight_parent_dir"],
        "used_from_train_dir": seg_wm["used_from_train_dir"],
        "det_weight_path": str(det_path),
    }
    write_run_meta_yaml(run_dir, run_meta)

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
            det_imgsz=args.det_imgsz,
            seg_conf=args.seg_conf,
            seg_thr=args.seg_thr,
            seg_imgsz=args.seg_imgsz,
            seg_iou=args.seg_iou,
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
            big_seg_iou=args.big_seg_iou,
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
            post_close_ksize=args.post_close,
            post_min_area=args.post_min_area,
            roi_v3=args.roi_v3,
            d20_class_id=args.d20_class_id,
            d20_pad_ratio=args.d20_pad_ratio,
            d20_pad_min=args.d20_pad_min,
            d20_seg_conf=args.d20_seg_conf,
            d20_seg_thr=args.d20_seg_thr,
            d20_seg_iou=args.d20_seg_iou,
            d20_seg_imgsz=args.d20_seg_imgsz,
            d20_clahe=args.d20_clahe,
            adaptive_tile=args.adaptive_tile,
            adaptive_min_tile=args.adaptive_min_tile,
            adaptive_max_tile=args.adaptive_max_tile,
            adaptive_target_long_tiles=args.adaptive_target_long_tiles,
            tile_overlap_ratio=args.tile_overlap_ratio,
            tile_fusion=args.tile_fusion,
            tile_fusion_thr=args.tile_fusion_thr,
            highres_enabled=args.highres_seg,
            highres_imgsz=args.highres_imgsz,
            highres_min_side=args.highres_min_side,
            debug_levels=args.debug_levels,
            enable_d20_structure_score=args.enable_d20_structure_score,
            d20_structure_thr=args.d20_structure_thr,
            enable_lane_marking_suppress=args.enable_lane_marking_suppress,
            lane_marking_bright_thr=args.lane_marking_bright_thr,
            lane_marking_sat_thr=args.lane_marking_sat_thr,
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
