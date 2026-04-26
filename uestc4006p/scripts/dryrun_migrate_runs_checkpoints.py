from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import yaml

from naming_utils import make_output_dir_name, normalize_task


@dataclass
class MapRow:
    old_name: str
    new_name: str
    status: str
    reason: str


def _safe_yaml(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        o = yaml.safe_load(f) or {}
    return o if isinstance(o, dict) else {}


def _meta_base_dir(d: Path) -> Path:
    """兼容历史结构：如果元文件在唯一子目录里，优先返回该子目录。"""
    if (d / "run_meta.yaml").exists() or (d / "args.yaml").exists() or (d / "RUN_INFO.txt").exists():
        return d
    subs = [x for x in d.iterdir() if x.is_dir()]
    if len(subs) == 1:
        c = subs[0]
        if (c / "run_meta.yaml").exists() or (c / "args.yaml").exists() or (c / "RUN_INFO.txt").exists():
            return c
    return d


def _infer_dataset_from_path(s: str) -> str:
    raw = str(s or "").replace("\\", "/").lower()
    candidates = [
        ("rdd2022_china", "RDD2022_China"),
        ("csc_2649", "CSC_2649"),
        ("china_motorbike", "China_MotorBike"),
        ("china_drone", "China_Drone"),
        ("crack500", "CRACK500"),
        ("crack_public_01", "crack_public_01"),
    ]
    for k, v in candidates:
        if k in raw:
            return v
    return Path(s).stem if s else "unknown"


def _infer_from_run_meta(d: Path) -> tuple[str, str, str, str, str] | None:
    m = _safe_yaml(_meta_base_dir(d) / "run_meta.yaml")
    if not m:
        return None
    return (
        str(m.get("mode", "")),
        str(m.get("task", "")),
        str(m.get("dataset", "")),
        str(m.get("model", "")),
        str(m.get("tag", "")),
    )


def _infer_from_args_yaml(d: Path) -> tuple[str, str, str, str, str] | None:
    a = _safe_yaml(_meta_base_dir(d) / "args.yaml")
    if not a:
        return None
    mode = "train"
    task = normalize_task(str(a.get("task", "det")))
    data = _infer_dataset_from_path(str(a.get("data", "")))
    model = Path(str(a.get("model", "model.pt"))).stem
    tag = ""
    n = str(a.get("name", ""))
    if "__" in n:
        tag = n.split("__", 1)[1]
    return (mode, task, data, model, tag)


def _infer_from_run_info(d: Path) -> tuple[str, str, str, str, str] | None:
    p = _meta_base_dir(d) / "RUN_INFO.txt"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8", errors="ignore")
    src = ""
    seg = ""
    for line in txt.splitlines():
        if line.startswith("SOURCE:"):
            src = line.split(":", 1)[1].strip()
        elif line.startswith("SEG:"):
            seg = line.split(":", 1)[1].strip()
    data = _infer_dataset_from_path(src)
    model = Path(seg).stem if seg else "model"
    return ("pred", "cascade", data, model, "")


def _infer_from_old_name(name: str, for_checkpoints: bool = False) -> tuple[str, str, str, str, str] | None:
    n = name.strip()
    if not n:
        return None
    # 典型历史格式：det_RDD2022_China_yolov8n_1024_ep300_bs16_seed42_baseline__note
    m = re.match(r"^(det|seg)_([^_]+(?:_[^_]+)*)_(yolo[v]?\d+[nslmx](?:-seg)?)_", n, re.IGNORECASE)
    if m:
        task = m.group(1).lower()
        data = m.group(2)
        model = m.group(3)
        tag = ""
        if "__" in n:
            tag = n.split("__", 1)[1]
        else:
            parts = n.split("_")
            if parts:
                tag = parts[-1]
        return ("train", task, data, model, tag)

    # 已短名历史：det_xxx 或 seg_xxx
    if n.startswith("det_"):
        return ("train", "det", n[4:], "v8n", "")
    if n.startswith("seg_"):
        return ("train", "seg", n[4:], "v8nseg", "")
    if "cascade" in n.lower():
        return ("pred", "cascade", n, "v8nseg", "")
    if for_checkpoints:
        return ("train", "seg", n, "v8nseg", "")
    return None


def _next_unique(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    idx = 2
    while True:
        n = f"{base}_{idx}"
        if n not in used:
            used.add(n)
            return n
        idx += 1


def _rows_for_runs(runs_root: Path) -> list[MapRow]:
    rows: list[MapRow] = []
    used: set[str] = set()
    for d in sorted([x for x in runs_root.iterdir() if x.is_dir()], key=lambda x: x.name.lower()):
        meta = _infer_from_run_meta(d) or _infer_from_args_yaml(d) or _infer_from_run_info(d) or _infer_from_old_name(d.name)
        if not meta:
            rows.append(MapRow(d.name, d.name, "manual", "missing run_meta/args/RUN_INFO"))
            continue
        mode, task, data, model, tag = meta
        try:
            base = make_output_dir_name(mode=mode, task=task, data=data, model=model, tag=tag)
            new_name = _next_unique(base, used)
            rows.append(MapRow(d.name, new_name, "auto", "inferred"))
        except Exception as e:
            rows.append(MapRow(d.name, d.name, "manual", f"infer_failed: {e}"))
    return rows


def _rows_for_checkpoints(ckpt_root: Path, run_rows: list[MapRow]) -> list[MapRow]:
    rows: list[MapRow] = []
    run_map = {r.old_name: r.new_name for r in run_rows}
    used: set[str] = set()
    for d in sorted([x for x in ckpt_root.iterdir() if x.is_dir()], key=lambda x: x.name.lower()):
        if d.name in run_map:
            base = run_map[d.name]
            new_name = _next_unique(base, used)
            rows.append(MapRow(d.name, new_name, "auto", "matched runs mapping"))
            continue

        meta = _infer_from_run_meta(d) or _infer_from_args_yaml(d) or _infer_from_old_name(d.name, for_checkpoints=True)
        if not meta:
            rows.append(MapRow(d.name, d.name, "manual", "missing run_meta/args and no runs match"))
            continue
        mode, task, data, model, tag = meta
        try:
            # checkpoints 对齐训练目录：固定 train mode
            base = make_output_dir_name(mode="train", task=task, data=data, model=model, tag=tag)
            new_name = _next_unique(base, used)
            rows.append(MapRow(d.name, new_name, "auto", "inferred"))
        except Exception as e:
            rows.append(MapRow(d.name, d.name, "manual", f"infer_failed: {e}"))
    return rows


def _write_csv(path: Path, rows: list[MapRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["old_name", "new_name", "status", "reason"])
        for r in rows:
            w.writerow([r.old_name, r.new_name, r.status, r.reason])


def _write_report(path: Path, run_rows: list[MapRow], ck_rows: list[MapRow]) -> None:
    auto_runs = sum(1 for r in run_rows if r.status == "auto")
    manual_runs = sum(1 for r in run_rows if r.status != "auto")
    auto_ck = sum(1 for r in ck_rows if r.status == "auto")
    manual_ck = sum(1 for r in ck_rows if r.status != "auto")
    with path.open("w", encoding="utf-8") as f:
        f.write("# Migration Report (Dry-run)\n\n")
        f.write("## Runs\n")
        f.write(f"- total: {len(run_rows)}\n")
        f.write(f"- auto: {auto_runs}\n")
        f.write(f"- manual: {manual_runs}\n\n")
        f.write("## Checkpoints\n")
        f.write(f"- total: {len(ck_rows)}\n")
        f.write(f"- auto: {auto_ck}\n")
        f.write(f"- manual: {manual_ck}\n\n")
        if manual_runs or manual_ck:
            f.write("## Manual Review Needed\n")
            for r in run_rows + ck_rows:
                if r.status != "auto":
                    f.write(f"- {r.old_name}: {r.reason}\n")


def main():
    ap = argparse.ArgumentParser(description="runs/checkpoints 命名迁移 dry-run")
    ap.add_argument("--runs-root", default=str(Path(__file__).resolve().parents[1] / "runs"))
    ap.add_argument("--checkpoints-root", default=str(Path(__file__).resolve().parents[1] / "checkpoints"))
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "dryrun_outputs"))
    args = ap.parse_args()

    runs_root = Path(args.runs_root).resolve()
    ck_root = Path(args.checkpoints_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    run_rows = _rows_for_runs(runs_root)
    ck_rows = _rows_for_checkpoints(ck_root, run_rows)

    rename_csv = outdir / "rename_mapping.csv"
    ck_csv = outdir / "checkpoint_mapping.csv"
    report_md = outdir / "migration_report.md"

    _write_csv(rename_csv, run_rows)
    _write_csv(ck_csv, ck_rows)
    _write_report(report_md, run_rows, ck_rows)

    print(f"rename_mapping: {rename_csv}")
    print(f"checkpoint_mapping: {ck_csv}")
    print(f"migration_report: {report_md}")


if __name__ == "__main__":
    main()
