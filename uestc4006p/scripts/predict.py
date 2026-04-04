from ultralytics import YOLO
from config import ExpCfg
from naming_utils import extract_weight_meta, now_iso, write_run_meta_yaml

def main():
    # 命名模式：new=新短命名，legacy=旧命名回退
    name_mode = "new"

    best_pt = r"E:\repositories\ultralytics\uestc4006p\runs\seg_CSC_2649_yolov8n-seg_800_ep200_bs24_seed42_ftbest__mask2poly-v2__open1close1__eps0p001__max500\weights\best.pt"
    source = r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets\public\seg_test"  # 0=默认摄像头；也可以换成图片/文件夹/视频路径

    cfg = ExpCfg(
        dataset="seg_test",
        task="segmentation",
        model_family="yolov8",
        model_name="yolov8n.pt",
        exp_name="baseline",
        tag="",  # tag可选；为空时目录名不追加tag
        naming_mode=name_mode,
    )

    run_root = cfg.run_root(mode="pred")
    run_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(best_pt)
    model.predict(
        source=source,
        device=cfg.device,
        save=True,          # 保存可视化结果
        imgsz=1280,
        conf=0.25,
        iou=0.6,
        project=str(run_root.parent),
        name=run_root.name,
        exist_ok=False,

        # ✅ 不生成 labels 子目录（只留可视化结果）
        save_txt=False,
        save_conf=False,
    )

    wm = extract_weight_meta(best_pt)
    meta = {
        "mode": "pred",
        "task": "seg",
        "dataset": cfg.dataset,
        "model": cfg.model_name,
        "tag": cfg.tag,
        "created_at": now_iso(),
        "weight_path": best_pt,
        "source": source,
        "notes": "",
        "weight_file": wm["weight_file"],
        "weight_parent_dir": wm["weight_parent_dir"],
        "used_from_train_dir": wm["used_from_train_dir"],
    }
    write_run_meta_yaml(run_root, meta)

    print("PRED DONE:", run_root)

if __name__ == "__main__":
    main()
