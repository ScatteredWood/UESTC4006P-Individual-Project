# extract_roi_for_label.py
# 用det模型从原图批量裁正方形ROI patch(带padding并resize)，输出到roi_pool/images并记录meta.csv，供后续标注

import hashlib

import argparse
import csv
import random
from pathlib import Path

import cv2
from ultralytics import YOLO


def list_images(src: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in src.rglob("*") if p.suffix.lower() in exts])


def clamp(v, lo, hi):
    return max(lo, min(int(v), hi))


def square_crop_xyxy(x1, y1, x2, y2, W, H, pad_ratio=0.15):
    """把框扩成正方形并加padding，返回裁剪坐标（int）"""
    bw = (x2 - x1)
    bh = (y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    side = max(bw, bh)
    side = side * (1.0 + 2.0 * pad_ratio)  # 两边padding

    x1n = cx - side / 2.0
    y1n = cy - side / 2.0
    x2n = cx + side / 2.0
    y2n = cy + side / 2.0

    x1i = clamp(x1n, 0, W - 1)
    y1i = clamp(y1n, 0, H - 1)
    x2i = clamp(x2n, 1, W)     # 右边界允许到 W
    y2i = clamp(y2n, 1, H)

    # 保证至少 2px
    if x2i <= x1i + 1:
        x2i = min(W, x1i + 2)
    if y2i <= y1i + 1:
        y2i = min(H, y1i + 2)
    return x1i, y1i, x2i, y2i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="校园原图目录（可递归）")
    ap.add_argument("--out", required=True, help="输出 roi_pool 目录")
    ap.add_argument("--weights", required=True, help="det best.pt 路径")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--iou", type=float, default=0.70)
    ap.add_argument("--imgsz", type=int, default=1024, help="det推理尺寸（不改变原图尺寸，只影响推理）")
    ap.add_argument("--pad", type=float, default=0.15, help="正方形裁剪padding比例")
    ap.add_argument("--min_side", type=int, default=0, help="框最短边阈值，小于此可随机丢弃")
    ap.add_argument("--keep_small", type=float, default=1.0, help="小框保留概率(0~1)")
    ap.add_argument("--topk", type=int, default=0, help="每张图最多保留topk框(按置信度)，0=不限制")
    ap.add_argument("--out_size", type=int, default=512, help="输出patch统一resize到这个尺寸(正方形)")
    ap.add_argument("--export_full_if_empty", action="store_true", help="没检出框时也导出整图patch用于标注")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    src_dir = Path(args.src)
    out_root = Path(args.out)
    img_out = out_root / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    meta_path = out_root / "meta.csv"
    fmeta = open(meta_path, "w", newline="", encoding="utf-8")
    mw = csv.writer(fmeta)
    mw.writerow(["patch", "src", "cls_id", "cls_name", "conf", "crop_xyxy", "orig_wh", "patch_wh"])

    model = YOLO(args.weights)
    imgs = list_images(src_dir)
    if not imgs:
        print("No images found:", src_dir)
        return

    n_patch = 0
    n_full = 0

    for p in imgs:
        im0 = cv2.imread(str(p))
        if im0 is None:
            continue
        H, W = im0.shape[:2]

        r = model.predict(im0, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cid, cf in zip(xyxy, cls, conf):
                bw = x2 - x1
                bh = y2 - y1
                max_side = max(bw, bh)

                if args.min_side > 0 and max_side < args.min_side:
                    if random.random() > args.keep_small:
                        continue

                boxes.append((float(cf), int(cid), float(x1), float(y1), float(x2), float(y2)))

            boxes.sort(key=lambda t: t[0], reverse=True)
            if args.topk and len(boxes) > args.topk:
                boxes = boxes[:args.topk]

        stem = p.stem  # 防Win路径过长
        if not boxes:
            if args.export_full_if_empty:
                patch = cv2.resize(im0, (args.out_size, args.out_size), interpolation=cv2.INTER_LINEAR)
                sid = hashlib.md5(str(p).encode("utf-8")).hexdigest()[:6]
                out_name = f"{stem}_{sid}_r{i:03d}_c{cid}.jpg"
                out_path = img_out / out_name
                cv2.imwrite(str(out_path), patch)
                mw.writerow([str(out_path), str(p), "", "", "", "", f"{W}x{H}", f"{args.out_size}x{args.out_size}"])
                n_patch += 1
                n_full += 1
            continue

        for i, (cf, cid, x1, y1, x2, y2) in enumerate(boxes):
            x1i, y1i, x2i, y2i = square_crop_xyxy(x1, y1, x2, y2, W, H, pad_ratio=args.pad)
            roi = im0[y1i:y2i, x1i:x2i].copy()
            if roi.size == 0:
                continue

            roi = cv2.resize(roi, (args.out_size, args.out_size), interpolation=cv2.INTER_LINEAR)

            cls_name = model.names.get(cid, str(cid))
            out_name = f"{stem}_r{i:03d}_c{cid}.jpg"  # 用cid，避免cls_name太长
            out_path = img_out / out_name
            cv2.imwrite(str(out_path), roi)

            mw.writerow([
                str(out_path), str(p),
                cid, cls_name, f"{cf:.4f}",
                f"{x1i},{y1i},{x2i},{y2i}",
                f"{W}x{H}", f"{args.out_size}x{args.out_size}"
            ])
            n_patch += 1

    fmeta.close()
    print("DONE")
    print("SRC:", src_dir)
    print("OUT:", out_root)
    print("patches:", n_patch, "full_fallback:", n_full)
    print("meta.csv:", meta_path)


if __name__ == "__main__":
    main()
