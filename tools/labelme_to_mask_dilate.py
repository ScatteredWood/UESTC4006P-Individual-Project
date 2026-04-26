# 将Labelme的json标注转为二值mask；polygon直接填充，line按label指定半径膨胀；可选导出YOLOv8-seg多边形txt与overlay可视化

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

def mask_to_yolo_seg(mask, out_txt: Path, cls_id=0, min_area=10, epsilon=1.5):
    # 轮廓 -> YOLOv8-seg 多边形（每个连通域一行）
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

def parse_line_label_radii(s: str):
    """
    解析 --line_label_radii
    例: "crack_skel:6,crack_thin:3"
    返回 dict[label]=radius(int)
    """
    m = {}
    s = (s or "").strip()
    if not s:
        return m
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split(":")
        m[k.strip()] = int(v.strip())
    return m

def parse_labels_csv(s: str):
    return set([x.strip() for x in (s or "").split(",") if x.strip()])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, help="ROI images 目录")
    ap.add_argument("--json_dir", required=True, help="Labelme json 目录（与图片同名）")
    ap.add_argument("--out_mask_dir", required=True, help="输出 mask 的目录")
    ap.add_argument("--out_yolo_dir", default="", help="可选：输出 YOLO-seg label txt 目录")

    # ✅ 关键改动：按 label 决定是否膨胀（只对 line/linestrip 生效）
    ap.add_argument("--poly_labels", default="crack",
                    help="作为polygon/rectangle填充的标签（不膨胀），逗号分隔。默认 crack")
    ap.add_argument("--line_label_radii", default="crack_skel:6",
                    help="对 line/linestrip 进行膨胀的标签及半径，格式 label:radius，用逗号分隔。默认 crack_skel:6")

    ap.add_argument("--line_base_thick", type=int, default=1, help="line/linestrip绘制的基础厚度（再膨胀前）")
    ap.add_argument("--save_overlay", action="store_true", help="可选：保存叠加可视化")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    json_dir = Path(args.json_dir)
    out_mask = Path(args.out_mask_dir); out_mask.mkdir(parents=True, exist_ok=True)

    out_yolo = Path(args.out_yolo_dir) if args.out_yolo_dir else None
    if out_yolo:
        out_yolo.mkdir(parents=True, exist_ok=True)

    poly_labels = parse_labels_csv(args.poly_labels)
    line_radii = parse_line_label_radii(args.line_label_radii)  # dict: label->radius

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

        # ✅ 为不同半径的线建立独立mask：radius -> mask
        line_masks = {}  # {radius(int): np.uint8 mask}

        for s in data.get("shapes", []):
            lbl = s.get("label", "")
            pts = s.get("points", [])
            st = s.get("shape_type", "polygon").lower()

            if st in ["polygon", "rectangle"]:
                if lbl not in poly_labels:
                    continue
                if st == "rectangle" and len(pts) == 2:
                    (x1, y1), (x2, y2) = pts
                    pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                if len(pts) >= 3:
                    draw_poly(mask_poly, pts)

            elif st in ["line", "linestrip"]:
                if lbl not in line_radii:
                    continue
                if len(pts) >= 2:
                    r = int(line_radii[lbl])
                    if r not in line_masks:
                        line_masks[r] = np.zeros((H, W), dtype=np.uint8)
                    draw_lines(line_masks[r], pts, thickness=args.line_base_thick, closed=False)

            else:
                # 其他类型忽略
                pass

        mask = mask_poly.copy()
        for r, m in line_masks.items():
            md = dilate(m, r)
            mask = np.maximum(mask, md)

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
