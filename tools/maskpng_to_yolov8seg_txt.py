# 将语义分割的二值PNG裂缝标签(mask)转换为YOLOv8-seg所需的多边形TXT标签（单类crack=0），便于与其他数据集混合训练。

import os
import glob
import cv2
import numpy as np

# ===================== 你只需要改这里 3 个路径 =====================
IMG_DIR = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\SematicSeg_Dataset\SematicSeg_Dataset\Original Image"
MASK_DIR = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\SematicSeg_Dataset\SematicSeg_Dataset\Labels"
OUT_LABEL_DIR = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\SematicSeg_Dataset\labels_yolo"
# ================================================================

CLASS_ID = 0
THRESH = 127
MIN_AREA = 20
DILATE = 0
AUTO_INVERT = True

# 轮廓自适应简化参数
MAX_POINTS = 800
EPS_MIN = 0.0005
EPS_MAX = 0.01
GROW = 1.35

# 填洞/过滤参数
FILL_SMALL_HOLES = True
MAX_HOLE_AREA_RATIO = 0.002

DROP_HUGE = True
MAX_COMP_AREA_RATIO = 0.05   # 建议先 0.05，更保守；大块还多再降到 0.03

os.makedirs(OUT_LABEL_DIR, exist_ok=True)

img_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    img_paths += glob.glob(os.path.join(IMG_DIR, ext))

def maybe_invert(binm: np.ndarray) -> np.ndarray:
    white_ratio = (binm > 0).mean()
    if white_ratio > 0.5:
        return 255 - binm
    return binm

def fill_small_holes(binm: np.ndarray, max_hole_area_ratio=0.002) -> np.ndarray:
    h, w = binm.shape
    ff = binm.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 255)
    holes = cv2.bitwise_not(ff)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    keep = np.zeros_like(holes)
    max_area = int(max_hole_area_ratio * h * w)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= max_area:
            keep[lab == i] = 255

    return cv2.bitwise_or(binm, keep)

def drop_huge_components(binm: np.ndarray, max_comp_area_ratio=0.05) -> np.ndarray:
    h, w = binm.shape
    num, lab, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
    out = np.zeros_like(binm)
    max_area = int(max_comp_area_ratio * h * w)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= max_area:
            out[lab == i] = 255
    return out

converted = 0
skipped = 0

for ip in img_paths:
    stem = os.path.splitext(os.path.basename(ip))[0]
    mp = os.path.join(MASK_DIR, stem + ".png")
    if not os.path.exists(mp):
        skipped += 1
        continue

    img = cv2.imread(ip, cv2.IMREAD_COLOR)
    if img is None:
        skipped += 1
        continue
    h, w = img.shape[:2]

    mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        skipped += 1
        continue

    _, binm = cv2.threshold(mask, THRESH, 255, cv2.THRESH_BINARY)

    if AUTO_INVERT:
        binm = maybe_invert(binm)

    if DILATE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1))
        binm = cv2.dilate(binm, k, iterations=1)

    if FILL_SMALL_HOLES:
        binm = fill_small_holes(binm, max_hole_area_ratio=MAX_HOLE_AREA_RATIO)

    if DROP_HUGE:
        binm = drop_huge_components(binm, max_comp_area_ratio=MAX_COMP_AREA_RATIO)

    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue

        peri = cv2.arcLength(c, True)
        eps = EPS_MIN * peri
        poly = cv2.approxPolyDP(c, eps, True)
        while len(poly) > MAX_POINTS and eps < EPS_MAX * peri:
            eps *= GROW
            poly = cv2.approxPolyDP(c, eps, True)

        poly = poly.reshape(-1, 2)
        if poly.shape[0] < 3:
            continue

        poly = poly.astype(np.float32)
        poly[:, 0] /= w
        poly[:, 1] /= h

        coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in poly])
        lines.append(f"{CLASS_ID} {coords}")

    out_txt = os.path.join(OUT_LABEL_DIR, stem + ".txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    converted += 1

print(f"Done. converted={converted}, skipped={skipped}")
print(f"Output labels in: {OUT_LABEL_DIR}")
