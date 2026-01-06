
from ultralytics import YOLO
from config import ExpCfg

def main():
    best_pt = r"E:\path\to\best.pt"  # TODO 修改
    cfg = ExpCfg(
        dataset=r"public\crack_public_01",
        task="detect",
        model_family="yolo11",
        model_name="yolo11n.pt",
        exp_name="baseline",
    )

    run_root = cfg.run_root()
    paths = cfg.ensure_dirs(run_root)

    model = YOLO(best_pt)
    model.val(
        data=str(cfg.dataset_yaml()),
        device=cfg.device,
        project=str(paths["val"]),
        name="run",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()
