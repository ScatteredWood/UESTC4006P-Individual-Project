from pathlib import Path
import textwrap

# ===============================
# 你只需要确认这两个路径
# ===============================
REPO_ROOT = Path(r"E:\repositories\ultralytics")
DATA_ROOT = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets")

APP_ROOT = REPO_ROOT / "uestc4006p"

# ===============================
# 目录结构定义
# ===============================
dirs = [
    APP_ROOT / "configs",
    APP_ROOT / "scripts",
    APP_ROOT / "utils",
    APP_ROOT / "runs",
    APP_ROOT / ".yolo_config",

    DATA_ROOT / "public",
    DATA_ROOT / "custom",
]

# ===============================
# 文件模板
# ===============================
config_py = textwrap.dedent(r"""
from dataclasses import dataclass
from pathlib import Path
import os
from datetime import datetime

DATA_ROOT = Path(r"E:\Large Files\UESTC4006P Individual Project (2025-26)\datasets")
REPO_ROOT = Path(r"E:\repositories\ultralytics")
APP_ROOT  = REPO_ROOT / "uestc4006p"

YOLO_CONFIG_DIR = APP_ROOT / ".yolo_config"
os.environ["YOLO_CONFIG_DIR"] = str(YOLO_CONFIG_DIR)

@dataclass
class ExpCfg:
    dataset: str
    task: str
    model_family: str
    model_name: str
    exp_name: str
    imgsz: int = 640
    batch: int = 16
    epochs: int = 100
    device: int = 0
    workers: int = 4

    def dataset_yaml(self) -> Path:
        return DATA_ROOT / self.dataset / "data.yaml"

    def run_root(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return APP_ROOT / "runs" / self.dataset.replace("\\", "_").replace("/", "_") / self.task / self.model_family / self.exp_name / ts

    def ensure_dirs(self, run_root: Path) -> dict:
        paths = {
            "root": run_root,
            "train": run_root / "train",
            "val": run_root / "val",
            "predict": run_root / "predict",
            "weights": run_root / "weights",
            "exports": run_root / "exports",
            "logs": run_root / "logs",
        }
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)
        return paths
""")

train_py = textwrap.dedent(r"""
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
""")

val_py = textwrap.dedent(r"""
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
""")

predict_py = textwrap.dedent(r"""
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
""")

files = {
    APP_ROOT / "scripts" / "config.py": config_py,
    APP_ROOT / "scripts" / "train.py": train_py,
    APP_ROOT / "scripts" / "val.py": val_py,
    APP_ROOT / "scripts" / "predict.py": predict_py,
}

# ===============================
# 执行创建
# ===============================
def main():
    print("Creating directories...")
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(" OK:", d)

    print("\nCreating files...")
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            print(" OK:", path)
        else:
            print(" SKIP (exists):", path)

    print("\nBootstrap completed successfully.")

if __name__ == "__main__":
    main()
