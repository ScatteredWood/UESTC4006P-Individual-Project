"""
UESTC4006P - Unified Training Entry (Ultralytics YOLO)
======================================================

训练目录命名规则（仅保留新规则）：
- 基础格式：{mode}_{task}_{data}_{model}
- 带标签：{mode}_{task}_{data}_{model}_{tag}

字段说明：
- mode: 固定为 train
- task: det / seg
- data: {dataset_name}_{domain}
- model: 优先 model_alias，若为空则退回 model.stem
- tag: 非空时追加

示例目录名：
- train_det_rdd2022_japan_yolo11n
- train_seg_crack500_all_v8nseg_augv1

默认路径：
- runs 输出：<REPO_ROOT>/uestc4006p/runs/<run_name>/
- checkpoint 归档：<REPO_ROOT>/uestc4006p/checkpoints/<run_name>/{best,last}.pt
- 预训练权重：<REPO_ROOT>/weights/*.pt
- Ultralytics 缓存/配置：<REPO_ROOT>/uestc4006p/.yolo_config/

Resume 规则：
- 若 --model 指向 last.pt，则视为 resume 模式
- resume 模式下复用原 run_name/runs_root/checkpoint_root
- resume 模式下优先读取原 run_meta.yaml 继承 dataset_name/domain/model_alias/tag/note
- 命令行非空参数优先级高于原 run_meta.yaml
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import os
from typing import Any
import yaml

from ultralytics import YOLO
from naming_utils import (
    make_output_dir_name,
    write_run_meta_yaml,
    now_iso,
)

# ===================== [可选] wakepy 安全导入 =====================
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

# 固定 ultralytics 的 config/cache 目录
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
# ============================================================================


# ***************** 以后你要改超参数，就改这里（也可用命令行覆盖） *****************
DEFAULT_IMGSZ = 800
DEFAULT_EPOCHS = 300
DEFAULT_BATCH = 32
DEFAULT_DEVICE = "0"
DEFAULT_WORKERS = 8
DEFAULT_SEED = 42
DEFAULT_PATIENCE = 50

# False -> 不缓存；"disk" -> 缓存到磁盘；True/"ram" -> 缓存到内存
DEFAULT_CACHE = False

DEFAULT_RUNS_ROOT = RUNS_ROOT
DEFAULT_CHECKPOINT_ROOT = CHECKPOINT_ROOT
# ********************************************************************************


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def is_last_checkpoint(weight_path: Path) -> bool:
    return weight_path.name.lower() == "last.pt"


def infer_resume_run_context(weight_path: Path) -> tuple[str, Path, Path]:
    """
    从 last.pt 推断 run_name/runs_root/checkpoint_root。
    支持两种路径：
    1) .../runs/<run_name>/weights/last.pt
    2) .../checkpoints/<run_name>/last.pt
    """
    p = weight_path.resolve()

    # case 1: .../runs/<run_name>/weights/last.pt
    if p.parent.name.lower() == "weights":
        run_dir = p.parent.parent
        if run_dir.parent.name.lower() == "runs":
            run_name = run_dir.name
            runs_root = run_dir.parent
            checkpoint_root = runs_root.parent / "checkpoints"
            return run_name, runs_root, checkpoint_root

    # case 2: .../checkpoints/<run_name>/last.pt
    ckpt_dir = p.parent
    if ckpt_dir.parent.name.lower() == "checkpoints":
        run_name = ckpt_dir.name
        checkpoint_root = ckpt_dir.parent
        runs_root = checkpoint_root.parent / "runs"
        return run_name, runs_root, checkpoint_root

    raise ValueError(
        "Resume 模式要求 --model 指向以下之一："
        ".../runs/<run_name>/weights/last.pt 或 "
        ".../checkpoints/<run_name>/last.pt"
    )


def resolve_unique_run_name(runs_root: Path, checkpoint_root: Path, base_name: str) -> str:
    """在 runs/checkpoints 两侧同时检查重名，重名时追加 _2/_3..."""
    idx = 1
    while True:
        cand = base_name if idx == 1 else f"{base_name}_{idx}"
        if not (runs_root / cand).exists() and not (checkpoint_root / cand).exists():
            return cand
        idx += 1


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


def resolve_pretrained_path(pretrained_arg: str) -> Path | None:
    """解析可选 warm start 权重路径；空值返回 None。"""
    s = str(pretrained_arg).strip()
    if not s:
        return None
    return resolve_model_path(s)


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


def read_yaml_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_resume_run_meta(runs_root: Path, checkpoint_root: Path, run_name: str) -> dict[str, Any]:
    """
    resume 模式下优先读取旧实验的 run_meta.yaml：
    1) runs/<run_name>/run_meta.yaml
    2) checkpoints/<run_name>/run_meta.yaml
    """
    candidates = [
        runs_root / run_name / "run_meta.yaml",
        checkpoint_root / run_name / "run_meta.yaml",
    ]
    for p in candidates:
        meta = read_yaml_if_exists(p)
        if meta:
            return meta
    return {}


def first_non_empty(*values) -> str:
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def split_dataset_field(dataset_field: str) -> tuple[str, str]:
    """
    兼容旧 run_meta.yaml 只有 dataset='dataset_domain' 的情况。
    使用 rsplit('_', 1) 以尽量保留 dataset_name 中可能存在的下划线。
    """
    s = str(dataset_field).strip()
    if not s:
        return "", ""
    if "_" not in s:
        return s, ""
    left, right = s.rsplit("_", 1)
    return left.strip(), right.strip()


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
    cache: object  # bool | str

    dataset_name: str = ""
    domain: str = ""
    model_alias: str = ""
    tag: str = ""
    note: str = ""
    pretrained: Path | None = None

    runs_root: Path = DEFAULT_RUNS_ROOT
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT
    exist_ok: bool = False
    run_name: str = ""
    resume_meta: dict[str, Any] = field(default_factory=dict)

    def task_short(self) -> str:
        return "det" if self.task == "detect" else ("seg" if self.task == "segment" else self.task)

    def model_token(self) -> str:
        alias = self.model_alias.strip()
        return alias if alias else self.model.stem

    def is_resume_mode(self) -> bool:
        return is_last_checkpoint(self.model)

    def apply_resume_context(self):
        """resume 模式下复用原始 run_name 与目录根，不走新建唯一名逻辑。"""
        if not self.is_resume_mode():
            return
        run_name, runs_root, checkpoint_root = infer_resume_run_context(self.model)
        self.run_name = run_name
        self.runs_root = runs_root
        self.checkpoint_root = checkpoint_root

    def apply_resume_metadata(self):
        """
        resume 模式下优先读取旧 run_meta.yaml，并在命令行对应字段为空时自动继承。
        命令行非空值优先级更高。
        """
        if not self.is_resume_mode():
            return

        self.resume_meta = load_resume_run_meta(self.runs_root, self.checkpoint_root, self.run_name)

        meta_dataset_name = first_non_empty(self.resume_meta.get("dataset_name"))
        meta_domain = first_non_empty(self.resume_meta.get("domain"))

        if not meta_dataset_name or not meta_domain:
            merged_dataset = first_non_empty(self.resume_meta.get("dataset"))
            ds_name, ds_domain = split_dataset_field(merged_dataset)
            meta_dataset_name = first_non_empty(meta_dataset_name, ds_name)
            meta_domain = first_non_empty(meta_domain, ds_domain)

        meta_model_alias = first_non_empty(
            self.resume_meta.get("model_alias"),
            self.resume_meta.get("model"),
        )
        meta_tag = first_non_empty(self.resume_meta.get("tag"))
        meta_note = first_non_empty(
            self.resume_meta.get("notes"),
            self.resume_meta.get("note"),
        )

        self.dataset_name = first_non_empty(self.dataset_name, meta_dataset_name)
        self.domain = first_non_empty(self.domain, meta_domain)
        self.model_alias = first_non_empty(self.model_alias, meta_model_alias)
        self.tag = first_non_empty(self.tag, meta_tag)
        self.note = first_non_empty(self.note, meta_note)

    def validate_required_fields(self):
        if not self.dataset_name:
            raise ValueError("dataset_name 不能为空；新训练请传 --dataset-name，续训请保证旧 run_meta.yaml 中可恢复该字段。")
        if not self.domain:
            raise ValueError("domain 不能为空；新训练请传 --domain，续训请保证旧 run_meta.yaml 中可恢复该字段。")

    def main_dir_name(self) -> str:
        return make_output_dir_name(
            mode="train",
            task=self.task_short(),
            data=f"{self.dataset_name}_{self.domain}",
            model=self.model_token(),
            tag=self.tag,
        )

    def ensure_run_name(self):
        if self.run_name:
            return
        self.run_name = resolve_unique_run_name(
            runs_root=self.runs_root,
            checkpoint_root=self.checkpoint_root,
            base_name=self.main_dir_name(),
        )

    def run_project_and_name(self) -> tuple[Path, str]:
        self.ensure_run_name()
        return self.runs_root, self.run_name

    def ul_run_dir(self) -> Path:
        self.ensure_run_name()
        return self.runs_root / self.run_name

    def checkpoint_dir(self) -> Path:
        self.ensure_run_name()
        return self.checkpoint_root / self.run_name


def main():
    ap = argparse.ArgumentParser()

    # 必填
    ap.add_argument("--task", required=True, choices=["detect", "segment"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pretrained", default="", help="可选 warm start 权重；主要用于 --model 为 yaml 时先建模再加载")

    # 超参数
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)

    # cache 参数
    ap.add_argument("--cache", default=DEFAULT_CACHE)

    # 命名字段
    # 非 resume 模式下必填；resume 模式下可留空并从旧 run_meta.yaml 继承
    ap.add_argument("--dataset-name", default="", help="数据集名；新训练必填，续训可留空自动继承")
    ap.add_argument("--domain", default="", help="域名/子域；新训练必填，续训可留空自动继承")
    ap.add_argument("--model-alias", default="", help="目录名中的 model 字段；为空时先尝试继承旧 meta，再回退 model.stem")
    ap.add_argument("--tag", default="", help="可选标签；续训时若留空则优先继承旧 meta")
    ap.add_argument("--note", default="", help="仅写入 run_meta.yaml，不参与目录命名；续训时若留空则优先继承旧 meta")

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
        pretrained=resolve_pretrained_path(args.pretrained),
        runs_root=Path(args.runs_root).resolve(),
        checkpoint_root=Path(args.checkpoint_root).resolve(),
        exist_ok=args.exist_ok,
    )

    ensure_dir(WEIGHTS_ROOT)
    ensure_dir(YOLO_CONFIG_DIR)

    if cfg.is_resume_mode():
        cfg.apply_resume_context()
        ensure_dir(cfg.runs_root)
        ensure_dir(cfg.checkpoint_root)
        cfg.apply_resume_metadata()
    else:
        ensure_dir(cfg.runs_root)
        ensure_dir(cfg.checkpoint_root)
        cfg.validate_required_fields()
        cfg.ensure_run_name()

    # resume 模式在 metadata 继承后再校验
    cfg.validate_required_fields()

    ensure_dir(cfg.checkpoint_dir())

    print("======================================================")
    print("RUN_NAME:", cfg.run_name)
    print("RUN_DIR:", cfg.ul_run_dir())
    print("CKPT_DIR:", cfg.checkpoint_dir())
    print("RESUME MODE:", cfg.is_resume_mode())
    print("CACHE MODE:", cfg.cache)
    print("======================================================")

    model = YOLO(str(cfg.model))
    if cfg.pretrained:
        # resume 优先，避免与断点续训语义冲突
        if cfg.is_resume_mode():
            print(f"[INFO] 检测到 resume 模式，忽略 --pretrained: {cfg.pretrained}")
        elif cfg.model.suffix.lower() == ".pt":
            # --model 已是权重文件时，再传 --pretrained 存在语义歧义，安全起见忽略
            print(f"[INFO] 检测到 --model 为 .pt，忽略 --pretrained 以避免歧义: {cfg.pretrained}")
        else:
            print(f"[INFO] 使用 warm start 权重: {cfg.pretrained}")
            model.load(str(cfg.pretrained))

    def start_training():
        is_resume = cfg.is_resume_mode()
        print(f"--- 续训状态: {is_resume} ---")
        run_project, run_name = cfg.run_project_and_name()
        ensure_dir(run_project)

        results = model.train(
            task=cfg.task,
            data=str(cfg.data_yaml),
            imgsz=cfg.imgsz,
            epochs=cfg.epochs,
            batch=cfg.batch,
            device=cfg.device,
            workers=cfg.workers,
            seed=cfg.seed,
            patience=cfg.patience,
            cache=cfg.cache,
            resume=is_resume,
            exist_ok=cfg.exist_ok,
            project=str(run_project),
            name=run_name,
            save=True,
            save_period=10,
        )

        actual_run_dir = Path(results.save_dir)
        wdir = actual_run_dir / "weights"
        copy_if_exists(wdir / "best.pt", cfg.checkpoint_dir() / "best.pt")
        copy_if_exists(wdir / "last.pt", cfg.checkpoint_dir() / "last.pt")

        run_meta = {
            "mode": "train",
            "task": cfg.task_short(),
            "dataset": f"{cfg.dataset_name}_{cfg.domain}",
            "dataset_name": cfg.dataset_name,
            "domain": cfg.domain,
            "model": cfg.model_token(),
            "model_alias": cfg.model_token(),
            "tag": cfg.tag,
            "created_at": now_iso(),
            "weight_path": str(cfg.model),
            "source": str(cfg.data_yaml),
            "notes": cfg.note,
        }
        if is_resume:
            run_meta["resume_from"] = str(cfg.model)

        write_run_meta_yaml(actual_run_dir, run_meta)
        write_run_meta_yaml(cfg.checkpoint_dir(), run_meta)

    if HAS_WAKEPY:
        with keep.presenting():
            start_training()
    else:
        start_training()

    print("TRAIN DONE")


if __name__ == "__main__":
    main()
