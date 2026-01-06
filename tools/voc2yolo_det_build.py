# voc2yolo_det_build.py
# 通用 VOC(XML) -> YOLO(det) 数据集构建：复制图片、生成labels、划分train/val、生成data.yaml
# 适用于 RDD2022 / Pascal VOC 风格的任意检测数据集

import os
import shutil
import random
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def voc_box_to_yolo(w, h, xmin, ymin, xmax, ymax):
    # VOC是像素坐标，YOLO要归一化中心点与宽高
    x = (xmin + xmax) / 2.0 / w
    y = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h

    # clamp，避免极端标注越界
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    return x, y, bw, bh

def parse_voc_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path} 缺少 <size> 字段")
    w = int(float(size.find("width").text))
    h = int(float(size.find("height").text))

    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        b = obj.find("bndbox")
        xmin = int(float(b.find("xmin").text))
        ymin = int(float(b.find("ymin").text))
        xmax = int(float(b.find("xmax").text))
        ymax = int(float(b.find("ymax").text))
        objs.append((name, xmin, ymin, xmax, ymax))
    return w, h, objs

def find_images(images_dir: Path):
    imgs = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return imgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", required=True, help="VOC训练图片目录")
    ap.add_argument("--train-xml", required=True, help="VOC训练XML目录（包含*.xml）")
    ap.add_argument("--out", required=True, help="输出YOLO数据集根目录")
    ap.add_argument("--classes", required=True,
                    help="类别列表，逗号分隔，例如 D00,D10,D20,D40")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true",
                    help="是否复制图片到out/images/...（默认是复制；不加此参数也会复制，建议保留）")
    ap.add_argument("--include-test-images", default=None,
                    help="可选：测试集图片目录（无标注），若提供则复制到 out/images/test")
    args = ap.parse_args()

    train_images_dir = Path(args.train_images)
    train_xml_dir = Path(args.train_xml)
    out_root = Path(args.out)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    if not train_images_dir.exists():
        raise FileNotFoundError(f"train images 目录不存在：{train_images_dir}")
    if not train_xml_dir.exists():
        raise FileNotFoundError(f"train xml 目录不存在：{train_xml_dir}")
    out_root.mkdir(parents=True, exist_ok=True)

    # 输出目录
    for split in ["train", "val"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    if args.include_test_images:
        (out_root / "images" / "test").mkdir(parents=True, exist_ok=True)

    # 收集训练图片（必须有同名xml）
    imgs = find_images(train_images_dir)
    pairs = []
    for img in imgs:
        xml = train_xml_dir / (img.stem + ".xml")
        if xml.exists():
            pairs.append((img, xml))

    if not pairs:
        raise RuntimeError("未找到任何 image-xml 配对。请检查路径是否正确、xml是否同名。")

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = int(len(pairs) * args.val_ratio)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    def convert_and_copy(pairs_list, split):
        n_obj_total = 0
        for img_path, xml_path in pairs_list:
            # copy image
            dst_img = out_root / "images" / split / img_path.name
            shutil.copy2(img_path, dst_img)

            # convert xml -> yolo txt
            w, h, objs = parse_voc_xml(xml_path)
            lines = []
            for (name, xmin, ymin, xmax, ymax) in objs:
                if name not in classes:
                    continue
                cls_id = classes.index(name)
                x, y, bw, bh = voc_box_to_yolo(w, h, xmin, ymin, xmax, ymax)
                lines.append(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
            dst_lbl = out_root / "labels" / split / (img_path.stem + ".txt")
            dst_lbl.write_text("\n".join(lines), encoding="utf-8")
            n_obj_total += len(lines)
        return n_obj_total

    n_train_obj = convert_and_copy(train_pairs, "train")
    n_val_obj = convert_and_copy(val_pairs, "val")

    # 可选：复制 test images（无标注）
    if args.include_test_images:
        test_dir = Path(args.include_test_images)
        if not test_dir.exists():
            raise FileNotFoundError(f"test images 目录不存在：{test_dir}")
        test_imgs = find_images(test_dir)
        for img_path in test_imgs:
            shutil.copy2(img_path, out_root / "images" / "test" / img_path.name)

    # 写 data.yaml（Ultralytics 格式）
    yaml_path = out_root / "data.yaml"
    yaml_text = f"""path: {out_root.as_posix()}
train: images/train
val: images/val
"""
    if args.include_test_images:
        yaml_text += "test: images/test\n"
    yaml_text += f"""
nc: {len(classes)}
names: {classes}
"""
    yaml_path.write_text(yaml_text.strip() + "\n", encoding="utf-8")

    print("✅ VOC -> YOLO(det) 构建完成")
    print(f"   out: {out_root}")
    print(f"   train: {len(train_pairs)} images, {n_train_obj} objects")
    print(f"   val:   {len(val_pairs)} images, {n_val_obj} objects")
    if args.include_test_images:
        print(f"   test:  {len(find_images(Path(args.include_test_images)))} images (no labels)")
    print(f"   data.yaml: {yaml_path}")

if __name__ == "__main__":
    main()
