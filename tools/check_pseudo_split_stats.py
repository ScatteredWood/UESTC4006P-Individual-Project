# check_pseudo_split_stats.py
# 统计伪标签数据集：train/val 的正样本(非空label)与负样本(空label)数量

from pathlib import Path

# ====== 改这里：指向你的伪标签数据集根目录（包含 images/ labels/ seg_pseudo.yaml 的那个目录）======
DATASET_DIR = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\RDD2022_China_Pseudo_Raw_v0\pseg_v0_det20_sg400_tt900_t768o30_sc55_im1280_m2p2_oc11")
# ============================================================================================

def count_labels(label_dir: Path):
    """返回 (total, pos, neg)；pos=非空txt，neg=空txt或缺失"""
    if not label_dir.exists():
        return 0, 0, 0

    total = 0
    pos = 0
    neg = 0

    for txt in label_dir.glob("*.txt"):
        total += 1
        try:
            s = txt.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            s = txt.read_text(errors="ignore").strip()

        if len(s) > 0:
            pos += 1
        else:
            neg += 1

    return total, pos, neg


def main():
    for split in ["train", "val"]:
        label_dir = DATASET_DIR / "labels" / split
        total, pos, neg = count_labels(label_dir)

        if total == 0:
            print(f"[{split}] labels not found or empty -> {label_dir}")
            continue

        pos_ratio = pos / total * 100.0
        neg_ratio = neg / total * 100.0
        print(f"[{split}] total={total}  pos(non-empty)={pos} ({pos_ratio:.2f}%)  neg(empty)={neg} ({neg_ratio:.2f}%)")

    # 额外：检查 images 数量是否和 labels 对得上（只做提示）
    for split in ["train", "val"]:
        img_dir = DATASET_DIR / "images" / split
        lab_dir = DATASET_DIR / "labels" / split
        if img_dir.exists() and lab_dir.exists():
            imgs = sum(1 for _ in img_dir.glob("*.jpg")) + sum(1 for _ in img_dir.glob("*.png"))
            labs = sum(1 for _ in lab_dir.glob("*.txt"))
            if imgs != labs:
                print(f"[WARN] {split}: images={imgs} != labels={labs} (可能有缺失/多余文件，但不一定致命)")

if __name__ == "__main__":
    main()
