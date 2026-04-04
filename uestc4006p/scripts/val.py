
from ultralytics import YOLO
from config import ExpCfg
from naming_utils import extract_weight_meta, now_iso, write_run_meta_yaml

def main():
    # 命名模式：new=新短命名，legacy=旧命名回退
    name_mode = "new"
    best_pt = r"E:\path\to\best.pt"  # TODO 修改
    cfg = ExpCfg(
        dataset=r"public\crack_public_01",
        task="detect",
        model_family="yolo11",
        model_name="yolo11n.pt",
        exp_name="baseline",
        tag="",  # tag可选；为空时目录名不追加tag
        naming_mode=name_mode,
    )

    run_root = cfg.run_root(mode="val")
    run_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(best_pt)
    model.val(
        data=str(cfg.dataset_yaml()),
        device=cfg.device,
        project=str(run_root.parent),
        name=run_root.name,
        exist_ok=True,
    )

    wm = extract_weight_meta(best_pt)
    meta = {
        "mode": "val",
        "task": "det",
        "dataset": cfg.dataset,
        "model": cfg.model_name,
        "tag": cfg.tag,
        "created_at": now_iso(),
        "weight_path": best_pt,
        "source": str(cfg.dataset_yaml()),
        "notes": "",
        "weight_file": wm["weight_file"],
        "weight_parent_dir": wm["weight_parent_dir"],
        "used_from_train_dir": wm["used_from_train_dir"],
    }
    write_run_meta_yaml(run_root, meta)

if __name__ == "__main__":
    main()
