# 随机划分语义分割数据集(images/labels)为train/val，并复制成YOLO常用目录结构(images/train|val, labels/train|val)
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
LABEL_EXT_PRIORITY = [".txt", ".png", ".jpg", ".jpeg"]  # 优先按这个顺序找同名标签


def find_label_for_image(img_path: Path, label_dir: Path) -> Path | None:
    stem = img_path.stem
    for ext in LABEL_EXT_PRIORITY:
        cand = label_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def copy_pair(img_path: Path, lab_path: Path, img_out: Path, lab_out: Path):
    img_out.parent.mkdir(parents=True, exist_ok=True)
    lab_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, img_out)
    shutil.copy2(lab_path, lab_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="原始images文件夹")
    ap.add_argument("--labels", required=True, help="原始labels文件夹（与images同名）")
    ap.add_argument("--out_root", default="", help="输出根目录（默认=images/labels的同级目录下 split_train_val）")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例，默认0.2")
    ap.add_argument("--seed", type=int, default=3407, help="随机种子，默认3407")
    ap.add_argument("--require_label", action="store_true", help="若开启：遇到缺标签直接报错；否则跳过并统计")
    args = ap.parse_args()

    img_dir = Path(args.images)
    lab_dir = Path(args.labels)

    if not img_dir.exists():
        raise FileNotFoundError(f"images目录不存在: {img_dir}")
    if not lab_dir.exists():
        raise FileNotFoundError(f"labels目录不存在: {lab_dir}")

    # 输出目录：默认放到 images/labels 同级目录下
    if args.out_root.strip():
        out_root = Path(args.out_root)
    else:
        out_root = img_dir.parent / "split_train_val"

    out_img_train = out_root / "images" / "train"
    out_img_val = out_root / "images" / "val"
    out_lab_train = out_root / "labels" / "train"
    out_lab_val = out_root / "labels" / "val"

    # 收集图片
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(img_dir.glob(f"*{ext}"))
        imgs.extend(img_dir.glob(f"*{ext.upper()}"))
    imgs = sorted(set(imgs))

    if not imgs:
        raise RuntimeError(f"在images目录未找到图片（支持后缀: {IMG_EXTS}）: {img_dir}")

    # 配对标签
    pairs = []
    missing = []
    for ip in imgs:
        lp = find_label_for_image(ip, lab_dir)
        if lp is None:
            missing.append(ip.name)
            if args.require_label:
                raise RuntimeError(f"缺少标签: {ip.name}（在 {lab_dir} 未找到同名 .txt/.png/...）")
            continue
        pairs.append((ip, lp))

    if not pairs:
        raise RuntimeError("图片-标签配对为0，请检查labels文件后缀是否为.txt或.png且与图片同名。")

    # 随机划分
    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_val = int(round(n * args.val_ratio))
    n_val = max(1, n_val) if n >= 2 else 0
    n_train = n - n_val

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    # 复制文件
    for ip, lp in train_pairs:
        copy_pair(ip, lp, out_img_train / ip.name, out_lab_train / lp.name)

    for ip, lp in val_pairs:
        copy_pair(ip, lp, out_img_val / ip.name, out_lab_val / lp.name)

    # 写清单（方便你检查/复现实验）
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "train_list.txt").write_text(
        "\n".join([p[0].name for p in train_pairs]) + "\n", encoding="utf-8"
    )
    (out_root / "val_list.txt").write_text(
        "\n".join([p[0].name for p in val_pairs]) + "\n", encoding="utf-8"
    )
    if missing:
        (out_root / "missing_labels.txt").write_text(
            "\n".join(missing) + "\n", encoding="utf-8"
        )

    print(f"[DONE] images: {len(imgs)}")
    print(f"[DONE] paired: {n} (train={n_train}, val={n_val}), val_ratio={args.val_ratio}, seed={args.seed}")
    print(f"[DONE] output: {out_root}")
    if missing:
        print(f"[WARN] missing labels: {len(missing)} -> 已写入 {out_root / 'missing_labels.txt'}")


if __name__ == "__main__":
    main()
