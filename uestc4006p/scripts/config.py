
from dataclasses import dataclass
from pathlib import Path
import os
from naming_utils import make_output_dir_name, resolve_unique_dir

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
APP_ROOT = REPO_ROOT / "uestc4006p"
DATA_ROOT = Path(os.environ.get("UESTC4006P_DATA_ROOT", str(REPO_ROOT / "datasets"))).expanduser().resolve()

YOLO_CONFIG_DIR = APP_ROOT / ".yolo_config"
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

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
    tag: str = ""
    naming_mode: str = "new"  # new | legacy
    run_name: str = ""

    def dataset_yaml(self) -> Path:
        return DATA_ROOT / self.dataset / "data.yaml"

    def run_root(self, mode: str) -> Path:
        if self.naming_mode == "legacy":
            return APP_ROOT / "runs" / self.dataset.replace("\\", "_").replace("/", "_") / self.task / self.model_family / self.exp_name

        if not self.run_name:
            base = make_output_dir_name(
            mode=mode,
            task=self.task,
            data=self.dataset,
            model=self.model_name,
            tag=self.tag,
        )
            self.run_name, _ = resolve_unique_dir(APP_ROOT / "runs", base)
        return APP_ROOT / "runs" / self.run_name

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
