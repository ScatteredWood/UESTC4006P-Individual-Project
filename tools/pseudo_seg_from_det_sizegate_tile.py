# pseudo_seg_from_det_sizegate_tile.py
# 伪标签生成主流水线：det出框→按尺寸门控保留ROI→seg生成伪mask→过滤(面积占比/细长度)→mask转YOLO多边形；可选大ROI切块；输出YOLO数据集+yaml+meta.csv
# ------------------------------------------------------------
# det -> ROI (size gate) -> seg pseudo label -> (optional tile) -> YOLO-seg dataset
# + fallback: if one source image produces NO ROI samples, run FULL-image pseudo labeling once
# + unique output folder: add short run tag, so same params won't overwrite same folder
# ------------------------------------------------------------
import csv
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ------------------------ 配置区（你后续主要改这里） ------------------------
RAW_ROOT = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\RDD2022_China_Pseudo_Raw_v0")
RAW_IMG_DIR = RAW_ROOT / "China"  # 你那 300 张

DET_WEIGHTS = r"E:\repositories\ultralytics\uestc4006p\checkpoints\det_RDD2022_China_yolov8n_1024_ep300_bs16_seed42_baseline__drone-and-motorbike-mixed\best.pt"
SEG_WEIGHTS = r"E:\repositories\ultralytics\uestc4006p\checkpoints\seg_CRACK500_ALL_yolov8n-seg_800_ep200_bs24_seed42_baseline__mask2poly-v2__open1close1__eps0p001__max500\best.pt"

# det 覆盖率（你现在在用 det20）
DET_CONF = 0.20
DET_IOU = 0.70
DET_IMGSZ = 512  # 你当前原图就是 512×512，设成 512 最匹配；以后换大图可改 1024

# seg 伪标签（你现在用 sc55）
SEG_CONF = 0.35
SEG_IOU = 0.50
SEG_IMGSZ = 1280  # 你的图就是 512，建议直接 512；想更慢更“细”可改 1024/1280

# ROI padding（防裁断裂缝）
PAD_RATIO = 0.10

# 尺寸 gate：大框全保，小框随机保留一部分（你现在全保了，所以 small_keep=1.0）
SIZE_GATE_MIN = 400
SMALL_KEEP_PROB = 1.00

# tile（保留逻辑，方便你以后换大图；当前 512×512 图像基本不会触发）
TILE_TRIGGER = 900
TILE_SIZE = 768
TILE_OVERLAP = 0.30

# 强过滤：面积占比 + thinness（压误检）
MIN_AREA_RATIO = 0.00005
MAX_AREA_RATIO = 0.08
MAX_THINNESS = 0.006

# mask2poly-v2：approxPolyDP eps=0.001 + max500
EPS_RATIO = 0.001
MAX_POLY_POINTS = 500

# 形态学（细裂缝容易被 OPEN 抹掉；你要“别太空”，建议 OPEN=0）
OPEN_ITERS = 0
CLOSE_ITERS = 1

# train/val split（按源图名 hash，保证同源图不会跨 split）
VAL_RATIO = 0.20
SEED = 42

# ✅ fallback：如果某张图没产出任何 ROI 样本，则整图做一次伪标签（保证至少 300 张样本）
FALLBACK_FULL_IF_EMPTY = False
# ------------------------------------------------------------------------------


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def clip_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def square_crop_xyxy(x1, y1, x2, y2, W, H, pad_ratio=0.10):
    bw, bh = (x2 - x1), (y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    side = max(bw, bh)
    side = side * (1.0 + 2.0 * pad_ratio)  # 四周留 padding

    nx1 = cx - side / 2.0
    ny1 = cy - side / 2.0
    nx2 = cx + side / 2.0
    ny2 = cy + side / 2.0

    return clip_xyxy(nx1, ny1, nx2, ny2, W, H)


def thinness_score(mask_bin: np.ndarray) -> float:
    # area / perimeter^2 （越小越细长，越像裂缝）
    area = float(mask_bin.sum() / 255.0)
    if area <= 0:
        return 1e9
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    peri = 0.0
    for c in contours:
        peri += cv2.arcLength(c, True)
    if peri <= 1e-6:
        return 1e9
    return area / (peri * peri)


def mask_to_polygons(mask_bin: np.ndarray, eps_ratio=0.001, max_points=500):
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = mask_bin.shape[:2]
    polys = []
    for cnt in contours:
        if cnt.shape[0] < 5:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(1.0, eps_ratio * peri)
        approx = cv2.approxPolyDP(cnt, eps, True)  # (N,1,2)
        pts = approx.reshape(-1, 2)
        if pts.shape[0] > max_points:
            idx = np.linspace(0, pts.shape[0] - 1, max_points).astype(int)
            pts = pts[idx]
        pts = pts.astype(np.float32)
        pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1)
        pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
        polys.append(pts)
    return polys


def split_by_source_stem(stem: str, val_ratio: float) -> str:
    key = (hash(stem) % 1000) / 1000.0
    return "val" if key < val_ratio else "train"


def tile_coords(w: int, h: int, tile: int, overlap: float):
    stride = max(1, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, w - tile + 1), stride))
    ys = list(range(0, max(1, h - tile + 1), stride))
    if xs and xs[-1] != w - tile:
        xs.append(max(0, w - tile))
    if ys and ys[-1] != h - tile:
        ys.append(max(0, h - tile))
    for yi, y0 in enumerate(ys):
        for xi, x0 in enumerate(xs):
            x1 = x0 + tile
            y1 = y0 + tile
            yield xi, yi, x0, y0, x1, y1


def merge_masks_to_binary(result_masks, out_h: int, out_w: int) -> np.ndarray | None:
    if result_masks is None or result_masks.data is None or len(result_masks.data) == 0:
        return None
    m = result_masks.data.cpu().numpy()  # (N, mh, mw)
    m_bin = (m.max(axis=0) > 0.5).astype(np.uint8) * 255
    mh, mw = m_bin.shape[:2]
    if (mh != out_h) or (mw != out_w):
        m_bin = cv2.resize(m_bin, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return m_bin


def apply_open_close(mask_bin: np.ndarray, open_iters: int, close_iters: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if open_iters > 0:
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, k, iterations=open_iters)
    if close_iters > 0:
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k, iterations=close_iters)
    return mask_bin


def should_keep_box_by_size(max_side: float) -> bool:
    if max_side >= SIZE_GATE_MIN:
        return True
    return random.random() < SMALL_KEEP_PROB


def decide_is_positive(mask_bin: np.ndarray) -> tuple[bool, float, float]:
    h, w = mask_bin.shape[:2]
    area_ratio = float((mask_bin > 0).sum() / (h * w))
    thin = thinness_score(mask_bin)
    is_pos = (area_ratio >= MIN_AREA_RATIO) and (area_ratio <= MAX_AREA_RATIO) and (thin <= MAX_THINNESS)
    return is_pos, area_ratio, thin


def make_unique_out_dir(root: Path, base_name: str) -> Path:
    """
    给输出目录加一个短 run tag，避免同参数覆盖同文件夹。
    """
    run_tag = datetime.now().strftime("r%m%d_%H%M%S")  # 例如 r0122_153012
    name = f"{base_name}__{run_tag}"
    out = root / name
    # 极少数情况下同秒重复，做一次自增兜底
    if out.exists():
        i = 2
        while (root / f"{name}_{i}").exists():
            i += 1
        out = root / f"{name}_{i}"
    return out


def write_one_sample(seg: YOLO, img: np.ndarray, img_out: Path, lab_out: Path) -> tuple[bool, float, float]:
    """
    对 img 跑 seg，写 label（正样本写 polygon，负样本写空文件）
    返回 is_pos, area_ratio, thin
    """
    h, w = img.shape[:2]
    sr = seg.predict(img, conf=SEG_CONF, iou=SEG_IOU, imgsz=SEG_IMGSZ, verbose=False)[0]
    m_bin = merge_masks_to_binary(sr.masks, h, w)

    is_pos = False
    area_ratio = 0.0
    thin = 1e9

    if m_bin is not None:
        m_bin = apply_open_close(m_bin, OPEN_ITERS, CLOSE_ITERS)
        is_pos, area_ratio, thin = decide_is_positive(m_bin)

        if is_pos:
            polys = mask_to_polygons(m_bin, eps_ratio=EPS_RATIO, max_points=MAX_POLY_POINTS)
            if polys:
                with open(lab_out, "w", encoding="utf-8") as f:
                    for pts in polys:
                        flat = pts.reshape(-1)
                        f.write("0 " + " ".join([f"{v:.6f}" for v in flat.tolist()]) + "\n")
            else:
                is_pos = False

    if not is_pos:
        lab_out.write_text("", encoding="utf-8")

    return is_pos, area_ratio, thin


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # ✅ 短且稳定的 base_name（参数+不覆盖）
    base_name = (
        f"pseg_v0"
        f"_det{int(DET_CONF*100):02d}"
        f"_sg{SIZE_GATE_MIN}"
        f"_sc{int(SEG_CONF*100):02d}"
        f"_im{SEG_IMGSZ}"
        f"_oc{OPEN_ITERS}{CLOSE_ITERS}"
    )
    OUT_DIR = make_unique_out_dir(RAW_ROOT, base_name)

    img_tr = OUT_DIR / "images/train"
    img_va = OUT_DIR / "images/val"
    lab_tr = OUT_DIR / "labels/train"
    lab_va = OUT_DIR / "labels/val"
    for p in [img_tr, img_va, lab_tr, lab_va]:
        ensure_dir(p)

    # 写 data yaml
    yaml_path = OUT_DIR / "seg_pseudo.yaml"
    yaml_path.write_text(
        f"path: {OUT_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: crack\n",
        encoding="utf-8",
    )

    meta_path = OUT_DIR / "meta.csv"
    mf = open(meta_path, "w", newline="", encoding="utf-8")
    mw = csv.writer(mf)
    mw.writerow([
        "split", "dst_image", "dst_label",
        "src_image", "det_cls", "det_conf", "roi_xyxy",
        "mode", "tile_xi", "tile_yi", "tile_xyxy_in_roi",
        "seg_conf", "area_ratio", "thinness", "is_pos"
    ])

    det = YOLO(DET_WEIGHTS)
    seg = YOLO(SEG_WEIGHTS)

    img_paths = sorted([p for p in RAW_IMG_DIR.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    if not img_paths:
        raise ValueError(f"RAW_IMG_DIR 下未找到图片：{RAW_IMG_DIR}")

    total_boxes = 0
    kept_boxes = 0
    total_samples = 0
    pos_samples = 0
    neg_samples = 0
    tiled_rois = 0
    fallback_full_used = 0

    for src in img_paths:
        im0 = cv2.imread(str(src))
        if im0 is None:
            continue
        H, W = im0.shape[:2]

        produced_any = False  # ✅ 本张图是否产出过任何样本（ROI/tile/full）

        # det 生成候选框（不依赖分类准确）
        dr = det.predict(im0, conf=DET_CONF, iou=DET_IOU, imgsz=DET_IMGSZ, verbose=False)[0]
        if dr.boxes is not None and len(dr.boxes) > 0:
            boxes = dr.boxes.xyxy.cpu().numpy()
            clss = dr.boxes.cls.cpu().numpy().astype(int)
            confs = dr.boxes.conf.cpu().numpy()

            roi_idx = 0
            for (x1, y1, x2, y2), c, dc in zip(boxes, clss, confs):
                total_boxes += 1

                bw, bh = (x2 - x1), (y2 - y1)
                max_side = max(bw, bh)
                if not should_keep_box_by_size(max_side):
                    continue
                kept_boxes += 1

                # padding
                x1p, y1p, x2p, y2p = square_crop_xyxy(x1, y1, x2, y2, W, H, pad_ratio=PAD_RATIO)
                roi = im0[y1p:y2p, x1p:x2p].copy()
                rh, rw = roi.shape[:2]
                if rh < 32 or rw < 32:
                    continue

                split = split_by_source_stem(src.stem, VAL_RATIO)
                img_out_dir = img_va if split == "val" else img_tr
                lab_out_dir = lab_va if split == "val" else lab_tr

                det_cls_name = str(det.names.get(int(c), c))

                # ✅ 文件名短一些：src前16 + roi idx + cls
                base_stem = f"{src.stem[:16]}_r{roi_idx:03d}_c{det_cls_name}"
                roi_idx += 1

                # tile（未来大图用；当前 512×512 基本不会触发）
                if max_side > TILE_TRIGGER and (rw >= TILE_SIZE and rh >= TILE_SIZE):
                    tiled_rois += 1
                    for xi, yi, tx0, ty0, tx1, ty1 in tile_coords(rw, rh, TILE_SIZE, TILE_OVERLAP):
                        tile_img = roi[ty0:ty1, tx0:tx1].copy()
                        th, tw = tile_img.shape[:2]
                        if th < 32 or tw < 32:
                            continue

                        dst_stem = f"{base_stem}_t{xi:02d}{yi:02d}"
                        img_out = img_out_dir / f"{dst_stem}.jpg"
                        lab_out = lab_out_dir / f"{dst_stem}.txt"

                        cv2.imwrite(str(img_out), tile_img)
                        produced_any = True

                        is_pos, area_ratio, thin = write_one_sample(seg, tile_img, img_out, lab_out)
                        total_samples += 1
                        pos_samples += int(is_pos)
                        neg_samples += int(not is_pos)

                        mw.writerow([
                            split, str(img_out), str(lab_out),
                            str(src), det_cls_name, f"{dc:.4f}", f"{x1p},{y1p},{x2p},{y2p}",
                            "tile", xi, yi, f"{tx0},{ty0},{tx1},{ty1}",
                            f"{SEG_CONF:.2f}", f"{area_ratio:.6f}", f"{thin:.6f}", int(is_pos)
                        ])
                else:
                    # ROI 直接做样本
                    dst_stem = base_stem
                    img_out = img_out_dir / f"{dst_stem}.jpg"
                    lab_out = lab_out_dir / f"{dst_stem}.txt"

                    cv2.imwrite(str(img_out), roi)
                    produced_any = True

                    is_pos, area_ratio, thin = write_one_sample(seg, roi, img_out, lab_out)
                    total_samples += 1
                    pos_samples += int(is_pos)
                    neg_samples += int(not is_pos)

                    mw.writerow([
                        split, str(img_out), str(lab_out),
                        str(src), det_cls_name, f"{dc:.4f}", f"{x1p},{y1p},{x2p},{y2p}",
                        "roi", "", "", "",
                        f"{SEG_CONF:.2f}", f"{area_ratio:.6f}", f"{thin:.6f}", int(is_pos)
                    ])

        # ✅ fallback：本图没有任何 ROI/tile 样本 -> 用整图做一次伪标签（保证覆盖）
        if FALLBACK_FULL_IF_EMPTY and (not produced_any):
            split = split_by_source_stem(src.stem, VAL_RATIO)
            img_out_dir = img_va if split == "val" else img_tr
            lab_out_dir = lab_va if split == "val" else lab_tr

            dst_stem = f"{src.stem[:16]}_FULL"
            img_out = img_out_dir / f"{dst_stem}.jpg"
            lab_out = lab_out_dir / f"{dst_stem}.txt"

            cv2.imwrite(str(img_out), im0)
            produced_any = True
            fallback_full_used += 1

            is_pos, area_ratio, thin = write_one_sample(seg, im0, img_out, lab_out)
            total_samples += 1
            pos_samples += int(is_pos)
            neg_samples += int(not is_pos)

            mw.writerow([
                split, str(img_out), str(lab_out),
                str(src), "", "", "",
                "full", "", "", "",
                f"{SEG_CONF:.2f}", f"{area_ratio:.6f}", f"{thin:.6f}", int(is_pos)
            ])

    mf.close()

    print("=== DONE ===")
    print("RAW_IMG_DIR:", RAW_IMG_DIR)
    print("OUT_DIR:", OUT_DIR)
    print("YAML:", yaml_path)
    print(f"det boxes total={total_boxes}, kept={kept_boxes}")
    print(f"samples total={total_samples}, pos={pos_samples}, neg={neg_samples}")
    print(f"tiled rois={tiled_rois}")
    print(f"fallback full used={fallback_full_used}")
    print("meta.csv:", meta_path)


if __name__ == "__main__":
    main()
