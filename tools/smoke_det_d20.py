# Det权重冒烟测试：随机抽取若干张图跑检测，自动找到D20类别id并逐图打印D20框数量，用于快速确认权重与类别映射是否正常

from ultralytics import YOLO
from pathlib import Path
import random

IMG_DIR = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\RDD2022_China_Pseudo_Raw_v0\China")
DET_W  = r"E:\repositories\ultralytics\uestc4006p\checkpoints\det_RDD2022_China_yolov8n_1024_ep300_bs16_seed42_baseline__drone-and-motorbike-mixed\best.pt"  # <-- 改成你找到的 det best.pt

model = YOLO(DET_W)
print("det_model.names =", model.names)

imgs = sorted([p for p in IMG_DIR.rglob("*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]])
random.shuffle(imgs)
imgs = imgs[:50]

# 你默认的类名一般就是 D00/D10/D20/D40；如果不是，下面会看到 names
TARGET = "D20"
d20_id = None
for k,v in model.names.items():
    if str(v).upper() == TARGET:
        d20_id = int(k); break
print("D20 id =", d20_id)

for p in imgs:
    r = model.predict(str(p), conf=0.35, iou=0.7, verbose=False)[0]
    n = 0
    if d20_id is not None and r.boxes is not None and len(r.boxes) > 0:
        cls = r.boxes.cls.cpu().numpy().astype(int)
        n = int((cls == d20_id).sum())
    print(p.name, "D20 boxes =", n)
