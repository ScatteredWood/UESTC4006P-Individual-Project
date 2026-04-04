from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any

import yaml


DATASET_ALIAS_MAP = {
    "rdd2022_china": "rddcn",
    "csc_2649": "csc2649",
    "china_motorbike": "motorbike",
    "china_drone": "drone",
    "crack500": "crack500",
    "crack_public_01": "cp01",
}

MODEL_ALIAS_MAP = {
    "yolov8n": "v8n",
    "yolov8n-seg": "v8nseg",
    "yolo11n": "v11n",
    "yolo11n-seg": "v11nseg",
}

VALID_MODES = {"train", "val", "pred"}
VALID_TASKS = {"det", "seg", "cascade"}


def sanitize_token(s: str) -> str:
    s = str(s or "").strip().lower().replace("\\", "_").replace("/", "_")
    s = re.sub(r"[^a-z0-9_-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-_")
    return s or "na"


def normalize_mode(mode: str) -> str:
    m = sanitize_token(mode)
    if m not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got: {mode}")
    return m


def normalize_task(task: str) -> str:
    t = sanitize_token(task)
    if t in {"detect", "det"}:
        t = "det"
    elif t in {"segment", "segmentation", "seg"}:
        t = "seg"
    elif t == "cascade":
        t = "cascade"
    if t not in VALID_TASKS:
        raise ValueError(f"task must be one of {sorted(VALID_TASKS)}, got: {task}")
    return t


def shorten_model_name(model_name: str) -> str:
    raw = sanitize_token(model_name).replace(".pt", "")
    if raw in MODEL_ALIAS_MAP:
        return MODEL_ALIAS_MAP[raw]
    m = re.match(r"yolov?(\d+)([nslmx])(-seg)?$", raw)
    if m:
        num, scale, seg = m.groups()
        return f"v{num}{scale}{'seg' if seg else ''}"
    return raw[:18]


def shorten_dataset_name(dataset_name: str) -> str:
    raw = sanitize_token(dataset_name)
    if raw in DATASET_ALIAS_MAP:
        return DATASET_ALIAS_MAP[raw]
    for k, v in DATASET_ALIAS_MAP.items():
        if k in raw:
            return v
    return raw[:16]


def shorten_tag(tag: str, max_len: int = 24) -> str:
    raw = sanitize_token(tag)
    if not raw or raw == "na":
        return ""
    return raw[:max_len] if len(raw) > max_len else raw


def make_output_dir_name(mode: str, task: str, data: str, model: str, tag: str = "") -> str:
    m = normalize_mode(mode)
    t = normalize_task(task)
    d = shorten_dataset_name(data)
    md = shorten_model_name(model)
    tg = shorten_tag(tag)
    base = f"{m}_{t}_{d}_{md}"
    return f"{base}_{tg}" if tg else base


def resolve_unique_dir(root: Path, base_name: str) -> tuple[str, Path]:
    p = root / base_name
    if not p.exists():
        return base_name, p
    idx = 2
    while True:
        name = f"{base_name}_{idx}"
        p = root / name
        if not p.exists():
            return name, p
        idx += 1


def extract_weight_meta(weight_path: str) -> dict[str, str]:
    p = Path(str(weight_path))
    parent = p.parent
    used_from_train_dir = ""
    if parent.name.lower() == "weights" and parent.parent.name:
        used_from_train_dir = parent.parent.name
    elif parent.name:
        used_from_train_dir = parent.name
    return {
        "weight_file": p.name,
        "weight_parent_dir": parent.name,
        "used_from_train_dir": used_from_train_dir,
    }


def write_run_meta_yaml(out_dir: Path, meta: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "run_meta.yaml"
    with meta_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
    return meta_path


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_main_dir_name(task: str, data: str, model: str, imgsz: int, batch: int, tag: str) -> str:
    return make_output_dir_name("train", task, data, model, tag)


def make_timestamp_subdir(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    return dt.strftime("%y%m%d-%H%M")
