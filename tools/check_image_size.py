# check_image_size.py
# 统计目录下图片尺寸分布：抽样读取前50张，输出总数与宽高min/max

from pathlib import Path
import cv2

p = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\RDD2022_China_Pseudo_Raw_v0\China")
imgs = [x for x in p.rglob("*") if x.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
ws = []
hs = []
for x in imgs[:50]:
    im = cv2.imread(str(x))
    if im is not None:  # 添加一个检查，确保图片被正确读取
        hs.append(im.shape[0])
        ws.append(im.shape[1])
print("count", len(imgs), "minW", min(ws), "maxW", max(ws), "minH", min(hs), "maxH", max(hs))
print("example", imgs[0].name, cv2.imread(str(imgs[0])).shape)