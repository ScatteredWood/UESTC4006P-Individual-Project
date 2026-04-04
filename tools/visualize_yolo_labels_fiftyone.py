# 脚本功能：清空旧的 FiftyOne 数据库记录，重新加载清洗后的图片与 YOLO 分割标签并启动可视化界面。

import fiftyone as fo
import fiftyone.utils.yolo as fouy
import os

img_dir = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\crack500_SematicSeg305_China100_20260129\images"
yolo_label_dir = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\crack500_SematicSeg305_China100_20260129\labels"

classes = ["crack"] 

# --- 修改部分开始 ---
dataset_name = "crack_yolo_vis"

# 检查是否已存在同名数据集，如果存在则先删除
if dataset_name in fo.list_datasets():
    print(f"检测到已有数据集 '{dataset_name}'，正在清理...")
    fo.delete_dataset(dataset_name)

# 1) 加载图片，这里不再需要担心重名报错
dataset = fo.Dataset.from_images_dir(img_dir, name=dataset_name)
# --- 修改部分结束 ---

# 2) 把 YOLO polygon 标签加进去
fouy.add_yolo_labels(
    dataset,
    label_field="ground_truth",
    labels_path=yolo_label_dir,
    classes=classes,
    label_type="instances", 
)

# 3) 打开可视化界面
session = fo.launch_app(dataset)
session.wait()