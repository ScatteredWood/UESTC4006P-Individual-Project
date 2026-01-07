import cv2
import numpy as np
from pathlib import Path
import shutil

# ========= 你只需要改这里 =========
TRAIN_DIR = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\pavement_crack_datasets\pavement crack datasets\CRACK500\traincrop\traincrop")
VAL_DIR   = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\pavement_crack_datasets\pavement crack datasets\CRACK500\valcrop\valcrop")

OUT_ROOT  = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\pavement_crack_datasets\pavement crack datasets\CRACK500\CRACK500_yolo_seg_v2")
# =================================

IMG_EXT = ".jpg"
MASK_EXT = ".png"


def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def mask_to_polygons(
    mask_path: Path,
    # ------------------ 关键超参数（建议先别乱改） ------------------
    min_area: int = 20,
    approx_eps_ratio: float = 0.001,  # 0.1% 周长作为 eps：贴边但不过度爆点
    max_points: int = 500,
    # ------------------ 形态学平滑（抑制锯齿/小孔/毛刺） ------------------
    morph_kernel: int = 3,
    morph_open_iter: int = 1,
    morph_close_iter: int = 1,
):
    """
    mask PNG -> list of polygons (normalized coords)
    输出为 YOLO-seg 标注格式的多边形点序列：
      每个 polygon: [x1,y1,x2,y2,...]，均为归一化到 [0,1] 的坐标

    设计目标（针对 crack 这类细长目标）：
    1) 先做轻量形态学 open + close，减少边界锯齿与小噪点，防止轮廓点爆炸；
    2) findContours 用 CHAIN_APPROX_SIMPLE，避免先拿到“每个像素一个点”的超长轮廓；
    3) approxPolyDP 用较小 eps（0.001），让多边形足够贴边，但可控；
    4) max_points 再做一次上限约束，防止极端样本生成超长 label。
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    # 二值化（兼容 0/255 或灰度）；Otsu 自动阈值
    _, bw = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = bw.shape[:2]

    # ------------------ 形态学平滑 ------------------
    # 说明：open 去掉小噪点；close 填补细小断裂/孔洞，使轮廓更连贯且减少锯齿点。
    # 注意：kernel 不要太大，否则会“抹粗/抹断”细裂缝；3x3 通常最稳。
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    if morph_open_iter > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=morph_open_iter)
    if morph_close_iter > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=morph_close_iter)

    # ------------------ 轮廓提取 ------------------
    # CHAIN_APPROX_SIMPLE：减少冗余点（尤其是长直线段），显著降低 label 长度与训练噪声
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 多边形逼近：eps 越小越贴边，但点越多；这里用 0.001 的折中
        eps = approx_eps_ratio * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if approx.shape[0] < 3:
            continue

        # 点数上限：防止极端轮廓导致 label 超长（影响 IO / 训练稳定性）
        if approx.shape[0] > max_points:
            idx = np.linspace(0, approx.shape[0] - 1, max_points).astype(int)
            approx = approx[idx]

        # 归一化到 [0,1]
        xy = approx.astype(np.float32)
        xy[:, 0] = np.clip(xy[:, 0] / w, 0, 1)
        xy[:, 1] = np.clip(xy[:, 1] / h, 0, 1)

        polys.append(xy.flatten().tolist())

    return polys


def process_split(split: str, src_dir: Path):
    img_out = OUT_ROOT / "images" / split
    lbl_out = OUT_ROOT / "labels" / split
    # msk_out = OUT_ROOT / "masks" / split  # 如需备份 mask 可打开
    ensure(img_out)
    ensure(lbl_out)
    # ensure(msk_out)

    imgs = sorted(src_dir.glob(f"*{IMG_EXT}"))
    missing = 0
    converted = 0

    for im in imgs:
        mask = im.with_suffix(MASK_EXT)  # 同名 .png
        if not mask.exists():
            missing += 1
            continue

        # 拷贝 image（训练用）
        shutil.copy2(im, img_out / im.name)

        # 如需备份 mask（可选）
        # shutil.copy2(mask, msk_out / mask.name)

        # 生成 yolo-seg label（每行一个 polygon）
        polys = mask_to_polygons(mask)

        # 注意：即便没有 polygon（全黑图），也要生成空 txt（避免 dataloader 行为不一致）
        label_path = lbl_out / (im.stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            for poly in polys:
                f.write("0 " + " ".join(f"{v:.6f}" for v in poly) + "\n")

        converted += 1

    print(f"[{split}] images={len(imgs)} converted={converted} missing_mask={missing}")


def write_yaml():
    # 为了最大兼容性，这里保持“手写 yaml”（避免不同 yaml 库输出风格差异）
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
