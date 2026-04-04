# 将Labelme的crack多边形与crack_skel线标注栅格化，并按半径膨胀crack_skel生成裂缝mask，且可选导出YOLOv8-seg多边形标签

import argparse, json
from pathlib import Path

import cv2
import numpy as np

def draw_poly(mask, pts):
    pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)

def draw_lines(mask, pts, thickness=1, closed=False):
    pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], closed, 255, thickness=thickness, lineType=cv2.LINE_AA)

def dilate(mask, radius):
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)

def mask_to_yolo_seg(mask, out_txt: Path, cls_id=0, min_area=20, epsilon=1.5):
    H, W = mask.shape[:2]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        approx = cv2.approxPolyDP(c, epsilon, True)
        pts = approx.reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        xy = []
        for x, y in pts:
            xy.append(f"{x / W:.6f}")
            xy.append(f"{y / H:.6f}")
        lines.append(f"{cls_id} " + " ".join(xy))
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def parse_label_radii(s: str):
    # 形如 "crack_skel:6,crack:0"
    out = {}
    s = s.strip()
    if not s:
        return out
    for kv in s.split(","):
        k, v = kv.split(":")
        out[k.strip()] = int(v)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_mask_dir", required=True)
    ap.add_argument("--out_yolo_dir", default="")
    ap.add_argument("--poly_label", default="crack", help="多边形/矩形用哪个label（默认 crack）")
    ap.add_argument("--line_base_thick", type=int, default=1)

    # ✅ 核心：按 label 决定线膨胀半径（不看文件名 c0/c1/c2）
    ap.add_argument("--line_radii_by_label", default="crack_skel:6",
                    help="linestrip/line 的膨胀半径，label:radius，例如 crack_skel:6,crack:2")
    ap.add_argument("--save_overlay", action="store_true")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    json_dir = Path(args.json_dir)
    out_mask = Path(args.out_mask_dir); out_mask.mkdir(parents=True, exist_ok=True)

    out_yolo = Path(args.out_yolo_dir) if args.out_yolo_dir else None
    if out_yolo:
        out_yolo.mkdir(parents=True, exist_ok=True)

    radii = parse_label_radii(args.line_radii_by_label)

    imgs = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    for ip in imgs:
        jp = json_dir / (ip.stem + ".json")
        if not jp.exists():
            continue

        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]

        data = json.loads(jp.read_text(encoding="utf-8"))

        mask_poly = np.zeros((H, W), dtype=np.uint8)
        # 每个 label 单独一张线mask，方便分别膨胀
        line_masks = {}

        for s in data.get("shapes", []):
            label = s.get("label", "")
            pts = s.get("points", [])
            st = s.get("shape_type", "polygon").lower()

            if st in ["polygon", "rectangle"]:
                if label != args.poly_label:
                    continue
                if st == "rectangle" and len(pts) == 2:
                    (x1, y1), (x2, y2) = pts
                    pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                if len(pts) >= 3:
                    draw_poly(mask_poly, pts)

            elif st in ["line", "linestrip"]:
                # 只处理你配置了半径的 label（比如 crack_skel）
                if label not in radii:
                    continue
                if len(pts) >= 2:
                    if label not in line_masks:
                        line_masks[label] = np.zeros((H, W), dtype=np.uint8)
                    draw_lines(line_masks[label], pts, thickness=args.line_base_thick, closed=False)

        mask = mask_poly.copy()
        for label, lm in line_masks.items():
            r = radii.get(label, 0)
            lm_d = dilate(lm, r)
            mask = np.maximum(mask, lm_d)

        out_path = out_mask / f"{ip.stem}_mask.png"
        cv2.imwrite(str(out_path), mask)

        if out_yolo:
            out_txt = out_yolo / f"{ip.stem}.txt"
            mask_to_yolo_seg(mask, out_txt, cls_id=0, min_area=20, epsilon=1.5)

        if args.save_overlay:
            ov = im.copy()
            ov[mask > 0] = (0.5 * ov[mask > 0] + 0.5 * np.array([255, 255, 255])).astype(np.uint8)
            cv2.imwrite(str(out_mask / f"{ip.stem}_overlay.jpg"), ov)

    print("DONE.")

if __name__ == "__main__":
    main()
