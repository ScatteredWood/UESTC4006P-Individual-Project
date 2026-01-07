import cv2
import numpy as np
from pathlib import Path
import shutil

# ========= 你只需要改这里 =========
TRAIN_DIR = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\pavement_crack_datasets\pavement crack datasets\CRACK500\traincrop\traincrop")
VAL_DIR   = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\pavement_crack_datasets\pavement crack datasets\CRACK500\valcrop\valcrop")

OUT_ROOT  = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\pavement_crack_datasets\pavement crack datasets\CRACK500\CRACK500_yolo_seg")
# =================================

IMG_EXT = ".jpg"
MASK_EXT = ".png"

def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def mask_to_polygons(mask_path: Path, min_area=8, approx_eps_ratio=0.01, max_points=300):
    """
    mask PNG -> list of polygons (normalized coords)
    Each polygon: [x1,y1,x2,y2,...] normalized to [0,1]
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    # 二值化（兼容 0/255 或灰度）
    _, bw = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = bw.shape[:2]

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        eps = approx_eps_ratio * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if approx.shape[0] < 3:
            continue

        # 控制点数，避免 label 太长
        if approx.shape[0] > max_points:
            idx = np.linspace(0, approx.shape[0] - 1, max_points).astype(int)
            approx = approx[idx]

        xy = approx.astype(np.float32)
        xy[:, 0] = np.clip(xy[:, 0] / w, 0, 1)
        xy[:, 1] = np.clip(xy[:, 1] / h, 0, 1)
        polys.append(xy.flatten().tolist())

    return polys

def process_split(split: str, src_dir: Path):
    img_out = OUT_ROOT / "images" / split
    lbl_out = OUT_ROOT / "labels" / split
    msk_out = OUT_ROOT / "masks" / split  # 备份可选
    ensure(img_out); ensure(lbl_out); ensure(msk_out)

    imgs = sorted(src_dir.glob(f"*{IMG_EXT}"))
    missing = 0
    converted = 0

    for im in imgs:
        mask = im.with_suffix(MASK_EXT)  # 同名 .png
        if not mask.exists():
            missing += 1
            continue

        # 拷贝 image/mask
        shutil.copy2(im, img_out / im.name)
        shutil.copy2(mask, msk_out / mask.name)

        # 生成 yolo-seg label
        polys = mask_to_polygons(mask)
        label_path = lbl_out / (im.stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            for poly in polys:
                f.write("0 " + " ".join(f"{v:.6f}" for v in poly) + "\n")

        converted += 1

    print(f"[{split}] images={len(imgs)} converted={converted} missing_mask={missing}")

def write_yaml():
    yaml_path = OUT_ROOT / "crack500-seg.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {OUT_ROOT.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write("  0: crack\n")
    print("data.yaml:", yaml_path)

def main():
    ensure(OUT_ROOT)
    process_split("train", TRAIN_DIR)
    process_split("val", VAL_DIR)
    write_yaml()
    print("DONE.")

if __name__ == "__main__":
    main()
