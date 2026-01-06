
from ultralytics import YOLO
from config import ExpCfg

def main():
    best_pt = r"E:\path\to\best.pt"  # TODO 修改
    source = 0  # 图片/视频路径 或 摄像头 0

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
    model.predict(
        source=source,
        device=cfg.device,
        save=True,
        project=str(paths["predict"]),
        name="run",
        exist_ok=True,
        conf=0.25
    )

    print("PRED DONE:", run_root)

if __name__ == "__main__":
    main()
