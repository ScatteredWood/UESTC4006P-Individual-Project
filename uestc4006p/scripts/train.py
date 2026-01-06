"""
UESTC4006P - Unified Training Entry (Ultralytics YOLO)
======================================================

【命名规范（写进毕设/日志里）】
exp_id = {task}_{dataset}_{domain}_{model}_{imgsz}_ep{epochs}_bs{batch}_seed{seed}_{tag}[__note]
例：
det_RDD2022_Japan_yolov8n_640_ep50_bs16_seed42_baseline
seg_CRACK500_all_yolov8n-seg_640_ep100_bs8_seed42_baseline__augv1

- task: det / seg
- dataset: RDD2022 / CRACK500 / ...
- domain: Japan / China / all / ...
- model: yolov8n / yolo11n / yolov8n-seg / ...
- note: 额外备注（可选），用于记录“这次改了什么”

【默认路径（与你的 .gitignore 对齐）】
- runs 输出：<REPO_ROOT>/uestc4006p/runs/<exp_id>/
- checkpoint 归档：<REPO_ROOT>/uestc4006p/checkpoints/<exp_id>/{best,last}.pt
- 预训练权重：<REPO_ROOT>/weights/*.pt
- Ultralytics 缓存/配置：<REPO_ROOT>/uestc4006p/.yolo_config/
- 数据集：放在 E:\Large Files\...\datasets\public（你已统一）
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import os

from ultralytics import YOLO


# ===================== 路径修复：永远以“仓库根目录”为基准 =====================
# train.py 当前放在：<REPO_ROOT>/uestc4006p/scripts/train.py
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]

WEIGHTS_ROOT = REPO_ROOT / "weights"                  # 官方/预训练权重统一放这里
RUNS_ROOT = REPO_ROOT / "uestc4006p" / "runs"         # 训练输出（Ultralytics 会在里面建 exp_id 子目录）
CHECKPOINT_ROOT = REPO_ROOT / "uestc4006p" / "checkpoints"  # ✅ 归档 best/last（方案A）
YOLO_CONFIG_DIR = REPO_ROOT / "uestc4006p" / ".yolo_config"

# 固定 ultralytics 的 config/cache 目录（避免写到 AppData\Roaming\Ultralytics）
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
# ============================================================================


# ***************** 以后你要改超参数，就改这里（也可用命令行覆盖） *****************
DEFAULT_IMGSZ   = 640
DEFAULT_EPOCHS  = 50
DEFAULT_BATCH   = 16
DEFAULT_DEVICE  = "0"    # "0" / "cpu" / "0,1"
DEFAULT_WORKERS = 8
DEFAULT_SEED    = 42

DEFAULT_RUNS_ROOT = RUNS_ROOT
DEFAULT_CHECKPOINT_ROOT = CHECKPOINT_ROOT
# ***************** 以后你要改超参数，就改这里（也可用命令行覆盖） *****************


@dataclass
class ExpCfg:
    task: str                 # "detect" or "segment"
    data_yaml: Path
    model: Path               # 绝对路径（统一后不会乱）
    imgsz: int
    epochs: int
    batch: int
    device: str
    workers: int
    seed: int

    dataset_name: str
    domain: str
    model_alias: str          # e.g. yolov8n / yolo11n / yolov8n-seg
    tag: str                  # baseline / smoke / ablation...
    note: str = ""            # 备注，可空

    runs_root: Path = DEFAULT_RUNS_ROOT
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT
    exist_ok: bool = False

    def task_short(self) -> str:
        return "det" if self.task == "detect" else ("seg" if self.task == "segment" else self.task)

    def exp_id(self) -> str:
        model_part = self.model_alias.strip() if self.model_alias else self.model.stem
        base = f"{self.task_short()}_{self.dataset_name}_{self.domain}_{model_part}_{self.imgsz}_ep{self.epochs}_bs{self.batch}_seed{self.seed}_{self.tag}"
        if self.note.strip():
            safe_note = self.note.strip().replace(" ", "-")
            base += f"__{safe_note}"
        return base

    def ul_run_dir(self) -> Path:
        return self.runs_root / self.exp_id()

    def checkpoint_dir(self) -> Path:
        return self.checkpoint_root / self.exp_id()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def resolve_model_path(model_arg: str) -> Path:
    """
    规则：
    - 传绝对路径：原样用
    - 传 weights/xxx.pt 或 weights\\xxx.pt：统一映射到 <REPO_ROOT>/weights/xxx.pt
    - 只传 xxx.pt：也映射到 <REPO_ROOT>/weights/xxx.pt（方便）
    - 其他相对路径：按 <REPO_ROOT>/<relative> 解析
    """
    p = Path(model_arg)

    if p.is_absolute():
        return p

    # 只给了一个文件名（比如 yolov8n.pt）
    if p.parent == Path(".") and p.suffix == ".pt":
        return WEIGHTS_ROOT / p.name

    # 给了 weights/xxx.pt
    parts = [x.lower() for x in p.parts]
    if len(parts) >= 2 and parts[0] == "weights":
        return WEIGHTS_ROOT / p.name

    # 其他相对路径：按仓库根目录解析
    return (REPO_ROOT / p).resolve()


def main():
    ap = argparse.ArgumentParser()

    # 必填
    ap.add_argument("--task", required=True, choices=["detect", "segment"])
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--model", required=True, help="e.g. weights/yolov8n.pt or yolov8n.pt or absolute path")

    # ***************** 以后你要改超参数，就改这里（或用命令行参数覆盖） *****************
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # ***************** 以后你要改超参数，就改这里（或用命令行参数覆盖） *****************

    # 命名字段
    ap.add_argument("--dataset-name", required=True, help="e.g. RDD2022 / CRACK500")
    ap.add_argument("--domain", required=True, help="e.g. Japan / China / all")
    ap.add_argument("--model-alias", required=True, help="e.g. yolov8n / yolo11n / yolov8n-seg")
    ap.add_argument("--tag", default="baseline", help="baseline/smoke/ablation...")
    ap.add_argument("--note", default="", help="备注：记录这次改了什么（可空）")

    # 输出路径（默认已经是绝对路径，不会再重复拼）
    ap.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    ap.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))

    # 覆盖保护
    ap.add_argument("--exist-ok", action="store_true", help="允许覆盖同名实验目录（不推荐）")
    args = ap.parse_args()

    data_yaml = Path(args.data).expanduser()
    if not data_yaml.is_absolute():
        data_yaml = data_yaml.resolve()

    model_path = resolve_model_path(args.model)

    cfg = ExpCfg(
        task=args.task,
        data_yaml=data_yaml,
        model=model_path,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        dataset_name=args.dataset_name,
        domain=args.domain,
        model_alias=args.model_alias,
        tag=args.tag,
        note=args.note,
        runs_root=Path(args.runs_root).resolve(),
        checkpoint_root=Path(args.checkpoint_root).resolve(),
        exist_ok=args.exist_ok,
    )

    run_dir = cfg.ul_run_dir()
    if run_dir.exists() and not cfg.exist_ok:
        raise FileExistsError(
            f"Run dir already exists:\n  {run_dir}\n"
            f"解决：换 --tag 或加 --note 或用 --exist-ok（不推荐覆盖）。"
        )

    ensure_dir(cfg.runs_root)
    ensure_dir(cfg.checkpoint_dir())
    ensure_dir(WEIGHTS_ROOT)
    ensure_dir(YOLO_CONFIG_DIR)

    print("======================================================")
    print("REPO_ROOT:", REPO_ROOT)
    print("YOLO_CONFIG_DIR:", YOLO_CONFIG_DIR)
    print("TASK:", cfg.task)
    print("EXP_ID:", cfg.exp_id())
    print("DATA:", cfg.data_yaml)
    print("MODEL:", cfg.model)
    print("RUN_DIR:", cfg.ul_run_dir())
    print("CHECKPOINT_DIR:", cfg.checkpoint_dir())
    print("HYPERPARAMS:", f"imgsz={cfg.imgsz}, epochs={cfg.epochs}, batch={cfg.batch}, device={cfg.device}, seed={cfg.seed}")
    print("======================================================")

    model = YOLO(str(cfg.model))

    # 训练：project/name 都是绝对路径 -> 不会再出现 scripts\uestc4006p\runs 的重复
    model.train(
        task=cfg.task,
        data=str(cfg.data_yaml),
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        seed=cfg.seed,
        project=str(cfg.runs_root),
        name=cfg.exp_id(),
        exist_ok=cfg.exist_ok,
        save=True,
        save_period=10,
    )

    # 归档 best/last（稳定位置）
    wdir = cfg.ul_run_dir() / "weights"
    copy_if_exists(wdir / "best.pt", cfg.checkpoint_dir() / "best.pt")
    copy_if_exists(wdir / "last.pt", cfg.checkpoint_dir() / "last.pt")

    print("✅ TRAIN DONE")
    print("RUN_DIR:", cfg.ul_run_dir())
    print("CHECKPOINTED:", cfg.checkpoint_dir())


if __name__ == "__main__":
    main()
