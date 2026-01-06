
from ultralytics import YOLO
from config import ExpCfg

def main():
    cfg = ExpCfg(
        dataset=r"public\crack_public_01",
        task="detect",
        model_family="yolo11",
        model_name="yolo11n.pt",
        exp_name="baseline",
    )

    run_root = cfg.run_root()
    paths = cfg.ensure_dirs(run_root)

    model = YOLO(cfg.model_name)

    model.train(
        data=str(cfg.dataset_yaml()),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        project=str(paths["train"]),
        name="run",
        exist_ok=True,
        save=True,
        save_period=10,
    )

    run_dir = paths["train"] / "run"
    for w in ["best.pt", "last.pt"]:
        src = run_dir / "weights" / w
        if src.exists():
            (paths["weights"] / w).write_bytes(src.read_bytes())

    print("TRAIN DONE:", run_root)

if __name__ == "__main__":
    main()
