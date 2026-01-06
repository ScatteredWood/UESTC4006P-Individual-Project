
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
