# 伪标签数据集重采样：按train/val统计正/负样本，随机裁剪负样本使neg≤pos*NEG_PER_POS，拷贝到新目录并生成seg_pseudo.yaml

from pathlib import Path
import random, shutil

# ====== 改这里：你的伪标签数据集目录（包含 images/labels）======
SRC_DIR = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\RDD2022_China_Pseudo_Raw_v0\pseg_v0_det20_sg400_tt900_t768o30_sc55_im1280_m2p2_oc11")
# ==============================================================

DST_DIR = SRC_DIR.parent / (SRC_DIR.name + "__bal")  # 输出目录
NEG_PER_POS = 2   # 负样本最多 = 2 * 正样本（建议 1~2）
SEED = 42

def ensure(p: Path): p.mkdir(parents=True, exist_ok=True)

def is_pos_label(txt: Path) -> bool:
    try:
        s = txt.read_text(encoding="utf-8").strip()
    except:
        s = txt.read_text(errors="ignore").strip()
    return len(s) > 0

def copy_pair(img_src: Path, lab_src: Path, img_dst: Path, lab_dst: Path):
    ensure(img_dst.parent); ensure(lab_dst.parent)
    shutil.copy2(img_src, img_dst)
    shutil.copy2(lab_src, lab_dst)

def main():
    random.seed(SEED)

    for split in ["train", "val"]:
        img_dir = SRC_DIR / "images" / split
        lab_dir = SRC_DIR / "labels" / split
        if not lab_dir.exists():
            print("missing:", lab_dir); continue

        pos = []
        neg = []
        for lab in lab_dir.glob("*.txt"):
            stem = lab.stem
            # 图片可能是 jpg 或 png
            img = (img_dir / f"{stem}.jpg")
            if not img.exists():
                img = (img_dir / f"{stem}.png")
            if not img.exists():
                continue

            if is_pos_label(lab):
                pos.append((img, lab))
            else:
                neg.append((img, lab))

        random.shuffle(neg)
        neg_keep = min(len(neg), int(len(pos) * NEG_PER_POS))
        kept = pos + neg[:neg_keep]

        out_img = DST_DIR / "images" / split
        out_lab = DST_DIR / "labels" / split
        ensure(out_img); ensure(out_lab)

        for img, lab in kept:
            copy_pair(img, lab, out_img / img.name, out_lab / lab.name)

        print(f"[{split}] pos={len(pos)} neg={len(neg)} kept_neg={neg_keep} total_out={len(kept)}")

    # 写 yaml
    yaml = DST_DIR / "seg_pseudo.yaml"
    yaml.write_text(
        f"path: {DST_DIR.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: crack\n",
        encoding="utf-8"
    )
    print("DONE:", DST_DIR)
    print("YAML:", yaml)

if __name__ == "__main__":
    main()
