# 裁剪伪标签数据集中的负样本(空label)：随机删除多余负样本，使neg数量不超过pos*指定倍数，并同步删除对应图片

import random
from pathlib import Path

# ====== 改这里 ======
DATASET_DIR = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\RDD2022_China_Pseudo_Raw_v0\pseg_v0_det20_sg400_sc35_im1280_oc01__r0122_184521")
NEG_PER_POS = 2          # 目标：neg <= pos * 2
SEED = 42
# ====================

random.seed(SEED)

def label_is_pos(label_path: Path) -> bool:
    return label_path.exists() and label_path.stat().st_size > 0

def find_image_for_label(img_dir: Path, stem: str):
    for ext in [".jpg", ".png", ".jpeg", ".bmp", ".webp"]:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def prune_split(split: str):
    lab_dir = DATASET_DIR / f"labels/{split}"
    img_dir = DATASET_DIR / f"images/{split}"

    labels = sorted(lab_dir.glob("*.txt"))
    pos = [p for p in labels if label_is_pos(p)]
    neg = [p for p in labels if not label_is_pos(p)]

    target_neg = min(len(neg), max(0, len(pos) * NEG_PER_POS))
    keep_neg = set(random.sample(neg, target_neg)) if target_neg < len(neg) else set(neg)

    removed = 0
    for lp in neg:
        if lp in keep_neg:
            continue
        stem = lp.stem
        ip = find_image_for_label(img_dir, stem)
        if ip and ip.exists():
            ip.unlink()
        if lp.exists():
            lp.unlink()
        removed += 1

    # 重新统计
    labels2 = sorted(lab_dir.glob("*.txt"))
    pos2 = sum(1 for p in labels2 if label_is_pos(p))
    neg2 = len(labels2) - pos2
    print(f"[{split}] before pos={len(pos)} neg={len(neg)}  -> after pos={pos2} neg={neg2} (removed neg={removed})")

def main():
    prune_split("train")
    prune_split("val")
    print("DONE.")

if __name__ == "__main__":
    main()
