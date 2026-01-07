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
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import os

from ultralytics import YOLO

# ===================== [新增] wakepy 安全导入 =====================
try:
    from wakepy import keep
    HAS_WAKEPY = True
except ImportError:
    HAS_WAKEPY = False
# =================================================================


# ===================== 路径修复：永远以“仓库根目录”为基准 =====================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]

WEIGHTS_ROOT = REPO_ROOT / "weights"
RUNS_ROOT = REPO_ROOT / "uestc4006p" / "runs"
CHECKPOINT_ROOT = REPO_ROOT / "uestc4006p" / "checkpoints"
YOLO_CONFIG_DIR = REPO_ROOT / "uestc4006p" / ".yolo_config"

# 固定 ultralytics 的 config/cache 目录（避免写到 AppData\Roaming\Ultralytics）
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
# ============================================================================


# ***************** 以后你要改超参数，就改这里（也可用命令行覆盖） *****************
DEFAULT_IMGSZ   = 800
DEFAULT_EPOCHS  = 300
DEFAULT_BATCH   = 32
DEFAULT_DEVICE  = "0"
DEFAULT_WORKERS = 8
DEFAULT_SEED    = 42
DEFAULT_PATIENCE = 50

# [ADD] 是否缓存数据集：
# False    -> 不缓存（最省内存，I/O 较多，最安全）
# "disk"  -> 缓存到磁盘（推荐，速度快但不占 RAM）
# True/"ram" -> 缓存到内存（不推荐用于大数据集）
DEFAULT_CACHE = False

DEFAULT_RUNS_ROOT = RUNS_ROOT
DEFAULT_CHECKPOINT_ROOT = CHECKPOINT_ROOT
# ********************************************************************************


@dataclass
class ExpCfg:
    task: str
    data_yaml: Path
    model: Path
    imgsz: int
    epochs: int
    batch: int
    device: str
    workers: int
    seed: int

    patience: int
    cache: object   # bool | str

    dataset_name: str
    domain: str
    model_alias: str
    tag: str
    note: str = ""

    runs_root: Path = DEFAULT_RUNS_ROOT
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT
    exist_ok: bool = False

    def task_short(self) -> str:
        return "det" if self.task == "detect" else ("seg" if self.task == "segment" else self.task)

    def exp_id(self) -> str:
        model_part = self.model_alias.strip() if self.model_alias else self.model.stem
        base = f"{self.task_short()}_{self.dataset_name}_{self.domain}_{model_part}_{self.imgsz}_ep{self.epochs}_bs{self.batch}_seed{self.seed}_{self.tag}"
        if self.note.strip():
            base += f"__{self.note.strip().replace(' ', '-')}"
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
    模型路径解析规则（保证路径稳定、可复现）：
    1) 绝对路径：直接使用
    2) xxx.pt：默认映射到 <REPO_ROOT>/weights/xxx.pt
    3) weights/xxx.pt：同样映射到 weights 目录
    4) 其他相对路径：相对于仓库根目录解析
    """
    p = Path(model_arg)
    if p.is_absolute():
        return p

    if p.parent == Path(".") and p.suffix == ".pt":
        return WEIGHTS_ROOT / p.name

    parts = [x.lower() for x in p.parts]
    if len(parts) >= 2 and parts[0] == "weights":
        return WEIGHTS_ROOT / p.name

    return (REPO_ROOT / p).resolve()


# [ADD] cache 参数规范化，避免版本差异导致异常
def normalize_cache_arg(v):
    """
    支持：
      False / True
      "disk" / "ram"
      "false" / "true"（字符串形式，来自命令行）
    """
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in {"false", "0", "no", "off", "none"}:
        return False
    if s in {"true", "1", "yes", "on"}:
        return True
    if s in {"disk", "ram"}:
        return s
    return v


def main():
    ap = argparse.ArgumentParser()

    # 必填
    ap.add_argument("--task", required=True, choices=["detect", "segment"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)

    # 超参数
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)

    # [ADD] cache 参数（默认 False：不缓存到内存）
    ap.add_argument("--cache", default=DEFAULT_CACHE)

    # 命名字段
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--model-alias", required=True)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--note", default="")

    # 输出路径
    ap.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    ap.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))

    ap.add_argument("--exist-ok", action="store_true")
    args = ap.parse_args()

    cfg = ExpCfg(
        task=args.task,
        data_yaml=Path(args.data).resolve(),
        model=resolve_model_path(args.model),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        cache=normalize_cache_arg(args.cache),
        dataset_name=args.dataset_name,
        domain=args.domain,
        model_alias=args.model_alias,
        tag=args.tag,
        note=args.note,
        runs_root=Path(args.runs_root).resolve(),
        checkpoint_root=Path(args.checkpoint_root).resolve(),
        exist_ok=args.exist_ok,
    )

    ensure_dir(cfg.runs_root)
    ensure_dir(cfg.checkpoint_dir())
    ensure_dir(WEIGHTS_ROOT)
    ensure_dir(YOLO_CONFIG_DIR)

    print("======================================================")
    print("EXP_ID:", cfg.exp_id())
    print("CACHE MODE:", cfg.cache)
    print("======================================================")

    model = YOLO(str(cfg.model))

    # ===================== 训练主逻辑（保持原结构） =====================
    def start_training():
        is_resume = "last.pt" in str(cfg.model).lower()
        print(f"--- 续训状态: {is_resume} ---")

        model.train(
            task=cfg.task,
            data=str(cfg.data_yaml),
            imgsz=cfg.imgsz,
            epochs=cfg.epochs,
            batch=cfg.batch,
            device=cfg.device,
            workers=cfg.workers,
            seed=cfg.seed,
            patience=cfg.patience,

            # [ADD] 关键参数：控制数据缓存方式
            cache=cfg.cache,

            resume=is_resume,
            exist_ok=cfg.exist_ok,
            project=str(cfg.runs_root),
            name=cfg.exp_id(),
            save=True,
            save_period=10,
        )

        wdir = cfg.ul_run_dir() / "weights"
        copy_if_exists(wdir / "best.pt", cfg.checkpoint_dir() / "best.pt")
        copy_if_exists(wdir / "last.pt", cfg.checkpoint_dir() / "last.pt")

    if HAS_WAKEPY:
        with keep.presenting():
            start_training()
    else:
        start_training()
    # ===============================================================================

    print("✅ TRAIN DONE")


if __name__ == "__main__":
    main()
