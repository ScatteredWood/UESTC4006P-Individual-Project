# 该脚本用于同步图片文件夹与标注文件夹：
# 遍历指定的 JSON 文件夹，仅保留图片文件夹中具有同名 JSON 文件的图片，删除多余的图片文件。

import os

def sync_images_with_labels():
    # 路径配置
    json_dir = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\RDD2022_China_Pseudo_Raw_v0\roi_label_v3\data_annotated"
    img_dir = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\RDD2022_China_Pseudo_Raw_v0\roi_label_v3\img_data"

    # 1. 获取所有 JSON 文件的基本文件名（不含后缀）
    # 例如：'China_Drone_000064_r000_c3.json' -> 'China_Drone_000064_r000_c3'
    json_basenames = {os.path.splitext(f)[0] for f in os.listdir(json_dir) if f.endswith('.json')}
    
    print(f"找到标注文件数量: {len(json_basenames)}")

    # 2. 遍历图片文件夹
    count = 0
    for img_name in os.listdir(img_dir):
        # 获取图片的基础文件名和后缀
        img_basename, img_ext = os.path.splitext(img_name)
        
        # 检查后缀，确保只处理常见的图片格式（可根据需要增删）
        if img_ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # 如果图片名不在 JSON 集合中，则删除
            if img_basename not in json_basenames:
                img_path = os.path.join(img_dir, img_name)
                try:
                    os.remove(img_path)
                    print(f"已删除多余图片: {img_name}")
                    count += 1
                except Exception as e:
                    print(f"删除 {img_name} 失败: {e}")

    print("-" * 30)
    print(f"同步完成！共删除 {count} 张多余图片。")

if __name__ == "__main__":
    # 建议在执行前确认路径是否正确
    confirm = input("确定要删除不匹配的图片吗？(y/n): ")
    if confirm.lower() == 'y':
        sync_images_with_labels()
    else:
        print("操作已取消。")