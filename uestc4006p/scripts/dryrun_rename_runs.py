from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import yaml

from naming_utils import (
    DATASET_ALIAS_MAP,
    make_main_dir_name,
    make_timestamp_subdir,
    shorten_dataset_name,
    shorten_model_name,
    shorten_tag,
    task_short,
)


@dataclass
class MappingRow:
    old_name: str
    new_main: str
    new_ts: str
    new_rel: str
    status: str
    reason: str


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _extract_tag(old_name: str, args_name: str = "") -> str:
    src = str(args_name or old_name)
    if "__" in src:
        src = src.split("__", 1)[1]
    else:
        parts = src.split("_")
        src = parts[-1] if parts else src
    t = shorten_tag(src)
    return t if t else "legacy"


def _infer_dataset(raw: str) -> str:
    s = (raw or "").lower().replace("\\", "_").replace("/", "_")
    for k in DATASET_ALIAS_MAP.keys():
        if k in s:
            return k
    return Path(raw).stem if raw else "unknown"


def _load_args_yaml(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        obj = yaml.safe_load(f) or {}
    return obj if isinstance(obj, dict) else {}


def _parse_run_info(p: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not p.exists():
        return out
    txt = p.read_text(encoding="utf-8", errors="ignore")
    for line in txt.splitlines():
        if line.startswith("SOURCE:"):
            out["source"] = line.split(":", 1)[1].strip()
        elif line.startswith("DET:"):
            out["det"] = line.split(":", 1)[1].strip()
        elif line.startswith("SEG:"):
            out["seg"] = line.split(":", 1)[1].strip()
        elif "base_seg_imgsz=" in line:
            m = re.search(r"base_seg_imgsz=(\d+)", line)
            if m:
                out["imgsz"] = m.group(1)
    return out


def _build_target_from_args(old_name: str, args_obj: dict[str, Any], ts: str) -> MappingRow:
    task = task_short(str(args_obj.get("task", "unk")))
    data = _infer_dataset(str(args_obj.get("data", "")))
    model = Path(str(args_obj.get("model", "model.pt"))).stem
    imgsz = _safe_int(args_obj.get("imgsz", 0))
    batch = _safe_int(args_obj.get("batch", 0))
    tag = _extract_tag(old_name, str(args_obj.get("name", "")))
    new_main = make_main_dir_name(task, data, model, imgsz, batch, tag)
    return MappingRow(
        old_name=old_name,
        new_main=new_main,
        new_ts=ts,
        new_rel=f"{new_main}/{ts}",
        status="auto",
        reason="from args.yaml",
    )


def _build_target_from_runinfo(old_name: str, info: dict[str, str], ts: str) -> MappingRow:
    source = info.get("source", "")
    det = Path(info.get("det", "det.pt")).stem
    seg = Path(info.get("seg", "seg.pt")).stem
    imgsz = _safe_int(info.get("imgsz", 0))
    data = _infer_dataset(source if source else old_name)
    model = f"{shorten_model_name(det)}-{shorten_model_name(seg)}"
    new_main = make_main_dir_name("seg", data, model, imgsz, 0, "cascade")
    return MappingRow(
        old_name=old_name,
        new_main=new_main,
        new_ts=ts,
        new_rel=f"{new_main}/{ts}",
        status="auto",
        reason="from RUN_INFO.txt",
    )


def resolve_conflicts(rows: list[MappingRow]) -> list[MappingRow]:
    used: dict[str, int] = {}
    out: list[MappingRow] = []
    for r in rows:
        key = f"{r.new_main}/{r.new_ts}"
        if key not in used:
            used[key] = 0
            out.append(r)
            continue
        used[key] += 1
        dup = used[key]
        r.new_main = f"{r.new_main}_dup{dup}"
        r.new_rel = f"{r.new_main}/{r.new_ts}"
        r.reason += f"; conflict->_dup{dup}"
        out.append(r)
    return out


def write_outputs(rows: list[MappingRow], out_csv: Path, out_md: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["old_name", "new_main", "new_ts", "new_rel", "status", "reason"])
        for r in rows:
            w.writerow([r.old_name, r.new_main, r.new_ts, r.new_rel, r.status, r.reason])

    with out_md.open("w", encoding="utf-8") as f:
        f.write("| old_name | new_rel | status | reason |\n")
        f.write("|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r.old_name} | {r.new_rel} | {r.status} | {r.reason} |\n")


def main():
    ap = argparse.ArgumentParser(description="历史 runs 目录重命名预演（dry-run，不执行重命名）")
    ap.add_argument("--runs-root", default=r"E:\repositories\ultralytics\uestc4006p\runs")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--stdout-only", action="store_true", help="仅打印映射，不写文件")
    args = ap.parse_args()

    runs_root = Path(args.runs_root).resolve()
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root not found: {runs_root}")

    now_ts = make_timestamp_subdir(datetime.now())
    outdir = Path(args.outdir).resolve() if args.outdir else (runs_root / "_dryrun_preview")
    out_csv = outdir / f"mapping_{now_ts}.csv"
    out_md = outdir / f"mapping_{now_ts}.md"

    rows: list[MappingRow] = []
    for d in sorted([x for x in runs_root.iterdir() if x.is_dir()], key=lambda x: x.name.lower()):
        if d.name.startswith("_dryrun_preview"):
            continue
        ts = d.stat().st_mtime
        ts_dir = make_timestamp_subdir(datetime.fromtimestamp(ts))
        args_yaml = d / "args.yaml"
        run_info = d / "RUN_INFO.txt"
        if args_yaml.exists():
            rows.append(_build_target_from_args(d.name, _load_args_yaml(args_yaml), ts_dir))
        elif run_info.exists():
            rows.append(_build_target_from_runinfo(d.name, _parse_run_info(run_info), ts_dir))
        else:
            # 无结构化信息，保留人工确认标记
            main = make_main_dir_name("misc", d.name, "na", 0, 0, "manual")
            rows.append(
                MappingRow(
                    old_name=d.name,
                    new_main=main,
                    new_ts=ts_dir,
                    new_rel=f"{main}/{ts_dir}",
                    status="manual",
                    reason="missing args.yaml/RUN_INFO.txt",
                )
            )

    rows = resolve_conflicts(rows)
    if args.stdout_only:
        print("old_name,new_rel,status,reason")
        for r in rows:
            print(f"{r.old_name},{r.new_rel},{r.status},{r.reason}")
        return

    write_outputs(rows, out_csv, out_md)
    print(f"DRY-RUN DONE: {len(rows)} entries")
    print(f"CSV: {out_csv}")
    print(f"MD : {out_md}")


if __name__ == "__main__":
    main()
