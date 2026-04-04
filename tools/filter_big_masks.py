# 基于面积占比、连通块大小及形状特征（如线框感）自动筛选裂缝数据集，并将合格样本提取至 clean 目录。

import os
import glob
import shutil
import cv2
import numpy as np

# ============== 原始路径（默认保持不动） ==============
IMG_DIR = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\SematicSeg_Dataset\SematicSeg_Dataset\Original Image"
MASK_DIR = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\SematicSeg_Dataset\SematicSeg_Dataset\Labels"
YOLO_LABEL_DIR = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\SematicSeg_Dataset\labels_yolo"

# ============== 输出路径 ==============
CLEAN_ROOT = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\SematicSeg_Dataset\cleaned_dataset"
BAD_ROOT   = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\SematicSeg_Dataset\bad_samples"

# ============== 行为开关（你最关心的在这里） ==============
RESET_OUTPUT_DIRS = True          # 每次运行前清空 clean/bad，避免“上次残留导致像没删掉”
SAVE_BAD_BACKUP   = True          # bad 样本是否备份到 BAD_ROOT（推荐先 True）
DELETE_SOURCE_FILES = False       # !!! 谨慎：True 会从原始目录物理删除 img/mask/label

# ============== 原阈值（保持不变） ==============
THRESH = 127
AUTO_INVERT = True
FG_RATIO_MAX = 0.10
MAX_COMP_RATIO_MAX = 0.05

# ============== 新增：线框/轮廓 mask 判坏阈值（可微调） ==============
OUTLINE_BBOX_RATIO_MIN = 0.25     # mask 非零像素 bbox 占整图比例超过这个，算“范围很大”
OUTLINE_FILL_RATIO_MAX = 0.35     # bbox 内填充率低于这个，算“很空/像线框”
OUTLINE_MEAN_THICK_MAX = 12.0     # 平均厚度（像素）低于这个，算“偏线条”

# -------------------------------------

def init_dirs(root):
    os.makedirs(root, exist_ok=True)
    dirs = {
        "img": os.path.join(root, "images"),
        "mask": os.path.join(root, "masks"),
        "lbl": os.path.join(root, "labels")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

if RESET_OUTPUT_DIRS:
    reset_dir(CLEAN_ROOT)
    reset_dir(BAD_ROOT)

CLEAN_DIRS = init_dirs(CLEAN_ROOT)
BAD_DIRS = init_dirs(BAD_ROOT)

def maybe_invert(binm: np.ndarray) -> np.ndarray:
    white_ratio = (binm > 0).mean()
    return 255 - binm if white_ratio > 0.5 else binm

def max_component_ratio(binm: np.ndarray) -> float:
    h, w = binm.shape
    num, lab, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if num <= 1:
        return 0.0
    max_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    return max_area / float(h * w)

def estimate_mean_thickness(binm: np.ndarray) -> float:
    # 用距离变换估计“平均厚度”：thickness ≈ 2 * mean(dist)
    fg = (binm > 0).astype(np.uint8)
    if fg.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(fg, distanceType=cv2.DIST_L2, maskSize=3)
    return float(2.0 * dist[fg > 0].mean())

def bbox_ratio_and_fill_ratio(binm: np.ndarray):
    h, w = binm.shape
    ys, xs = np.where(binm > 0)
    if len(xs) == 0:
        return 0.0, 0.0
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bbox_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox_ratio = bbox_area / float(h * w)
    fg_area = int((binm > 0).sum())
    fill_ratio = fg_area / float(bbox_area) if bbox_area > 0 else 0.0
    return bbox_ratio, fill_ratio

def copy_if_exists(src_path, dst_dir):
    if src_path and os.path.exists(src_path):
        shutil.copy2(src_path, os.path.join(dst_dir, os.path.basename(src_path)))

def delete_if_exists(src_path):
    if src_path and os.path.exists(src_path):
        os.remove(src_path)

# 收集图片
img_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    img_paths += glob.glob(os.path.join(IMG_DIR, ext))

good, bad, skipped = 0, 0, 0

print("正在处理，请稍候...")

for ip in img_paths:
    stem = os.path.splitext(os.path.basename(ip))[0]
    mp = os.path.join(MASK_DIR, stem + ".png")
    tp = os.path.join(YOLO_LABEL_DIR, stem + ".txt")

    if not os.path.exists(mp):
        skipped += 1
        continue

    mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        skipped += 1
        continue

    _, binm = cv2.threshold(mask, THRESH, 255, cv2.THRESH_BINARY)
    if AUTO_INVERT:
        binm = maybe_invert(binm)

    fg_ratio = (binm > 0).mean()
    mcr = max_component_ratio(binm)

    # 新增：线框/轮廓判坏
    bbox_ratio, fill_ratio = bbox_ratio_and_fill_ratio(binm)
    mean_thick = estimate_mean_thickness(binm)

    is_outline_like = (
        (bbox_ratio > OUTLINE_BBOX_RATIO_MIN) and
        (fill_ratio < OUTLINE_FILL_RATIO_MAX) and
        (mean_thick < OUTLINE_MEAN_THICK_MAX)
    )

    is_bad = (fg_ratio > FG_RATIO_MAX) or (mcr > MAX_COMP_RATIO_MAX) or is_outline_like

    if is_bad:
        bad += 1

        # 1) 不进入 clean —— 等价于“删掉（从清洗结果里剔除）”
        # 2) 可选：备份到 bad_samples 方便你回看
        if SAVE_BAD_BACKUP:
            copy_if_exists(ip, BAD_DIRS["img"])
            copy_if_exists(mp, BAD_DIRS["mask"])
            copy_if_exists(tp, BAD_DIRS["lbl"])

        # 3) 可选：物理删除原始目录文件（谨慎）
        if DELETE_SOURCE_FILES:
            delete_if_exists(ip)
            delete_if_exists(mp)
            delete_if_exists(tp)

    else:
        good += 1
        copy_if_exists(ip, CLEAN_DIRS["img"])
        copy_if_exists(mp, CLEAN_DIRS["mask"])
        copy_if_exists(tp, CLEAN_DIRS["lbl"])

print("-" * 30)
print(f"处理完成！")
print(f"✅ 干净样本(已复制到 clean): {good} -> {CLEAN_ROOT}")
print(f"❌ 异常样本(已从 clean 剔除): {bad} -> {'并备份到 ' + BAD_ROOT if SAVE_BAD_BACKUP else '未备份'}")
print(f"⚠️ 跳过样本: {skipped}")
print(f"DELETE_SOURCE_FILES={DELETE_SOURCE_FILES}（True 会物理删除原始目录文件）")
