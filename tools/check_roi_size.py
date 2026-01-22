import csv

path = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\RDD2022_China_Pseudo_Raw_v0\pseg_v0_det20_sg400_tt900_t768o30_sc55_im1280_m2p2_oc11\meta.csv"

maxw = 0.0
maxh = 0.0
both768 = 0
cnt = 0

with open(path, newline="", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    for r in rd:
        x1, y1, x2, y2 = map(float, r["roi_xyxy"].split(","))
        w = x2 - x1
        h = y2 - y1
        maxw = max(maxw, w)
        maxh = max(maxh, h)
        if w >= 768 and h >= 768:
            both768 += 1
        cnt += 1

print("rows:", cnt, "maxw:", maxw, "maxh:", maxh, "both>=768:", both768)
