# make_empty_yolo_labels.py
# 为images目录下每张图生成对应的空YOLO标签txt（若不存在），用于保证数据集结构完整

import os
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    root = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\RDD2022_China_Pseudo_Raw_v0\roi_label_v2")
    img_dir = root / "images"
    lab_dir = root / "labels"

    if not img_dir.exists():
        raise FileNotFoundError(f"images dir not found: {img_dir}")
    lab_dir.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    created = 0
    existed = 0

    for p in imgs:
        out = lab_dir / (p.stem + ".txt")
        if out.exists():
            existed += 1
            continue
        out.write_text("", encoding="utf-8")
        created += 1

    print(f"images={len(imgs)} labels_exist={existed} labels_created={created} labels_dir={lab_dir}")

if __name__ == "__main__":
    main()
