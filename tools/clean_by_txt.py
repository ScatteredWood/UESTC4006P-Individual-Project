# 脚本功能：根据 delect.txt 里的文件名清单，同步删除指定的图片、Mask 掩码和 YOLO 标签文件。

import os

# ============== 1. 路径配置 ==============
# 包含要删除名单的 txt 文件
TXT_PATH = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\SematicSeg_Dataset\delect.txt"

# 待清理的三个目录
IMG_DIR = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\SematicSeg_Dataset\cleaned_dataset\images"
MASK_DIR = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\SematicSeg_Dataset\cleaned_dataset\masks"
YOLO_LABEL_DIR = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\SematicSeg_Dataset\cleaned_dataset\labels"

# ============== 2. 执行逻辑 ==============

def delete_files():
    # 读取 txt 中的编号
    if not os.path.exists(TXT_PATH):
        print(f"错误：找不到名单文件 {TXT_PATH}")
        return

    # 使用 utf-8 编码读取，防止中文路径或特殊字符报错
    with open(TXT_PATH, 'r', encoding='utf-8') as f:
        # strip() 去掉每一行末尾的换行符
        stems = [line.strip() for line in f.readlines() if line.strip()]

    print(f"名单读取成功，共计: {len(stems)} 个待删除样本")
    
    deleted_count = 0
    not_found_count = 0

    for stem in stems:
        found_in_this_round = False
        
        # 1. 尝试删除图片 (兼容多种格式)
        for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
            img_path = os.path.join(IMG_DIR, stem + ext)
            if os.path.exists(img_path):
                os.remove(img_path)
                found_in_this_round = True
                print(f"已删除图片: {stem + ext}")
                break 

        # 2. 删除对应的 Mask (.png)
        mask_path = os.path.join(MASK_DIR, stem + ".png")
        if os.path.exists(mask_path):
            os.remove(mask_path)
            found_in_this_round = True
            print(f"已删除 Mask: {stem}.png")

        # 3. 删除对应的 YOLO Label (.txt)
        label_path = os.path.join(YOLO_LABEL_DIR, stem + ".txt")
        if os.path.exists(label_path):
            os.remove(label_path)
            found_in_this_round = True
            print(f"已删除 Label: {stem}.txt")

        if found_in_this_round:
            deleted_count += 1
        else:
            print(f"⚠️ 警告：未找到样本 {stem} 的任何相关文件")
            not_found_count += 1

    print("-" * 30)
    print(f"清理总结：")
    print(f" - 成功清理样本总数: {deleted_count}")
    print(f" - 未找到的样本总数: {not_found_count}")

if __name__ == "__main__":
    # 二次确认，防止误操作
    confirm = input("！！！此操作将永久删除文件，确定要执行吗？(y/n): ")
    if confirm.lower() == 'y':
        delete_files()
    else:
        print("操作已取消。")