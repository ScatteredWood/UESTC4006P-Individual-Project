from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from config import ExpCfg
from naming_utils import extract_weight_meta, now_iso, write_run_meta_yaml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prediction entry for UESTC4006P experiments.")
    ap.add_argument("--best", required=True, help="Path to model checkpoint (best.pt/last.pt).")
    ap.add_argument("--source", required=True, help="Image/video path, folder, webcam id, etc.")
    ap.add_argument("--task", choices=["detect", "segment"], default="segment")
    ap.add_argument("--dataset", default="predict_input")
    ap.add_argument("--model-family", default="yolo")
    ap.add_argument("--model-name", default="model.pt")
    ap.add_argument("--exp-name", default="baseline")
    ap.add_argument("--tag", default="")
    ap.add_argument("--name-mode", choices=["new", "legacy"], default="new")
    ap.add_argument("--run-name", default="", help="Optional fixed run name.")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--save-txt", action="store_true", help="Save label txt files.")
    ap.add_argument("--save-conf", action="store_true", help="Save confidence in txt labels.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    task_for_cfg = "detect" if args.task == "detect" else "segmentation"
    task_short = "det" if args.task == "detect" else "seg"

    best_pt = Path(args.best).expanduser().resolve()

    cfg = ExpCfg(
        dataset=args.dataset,
        task=task_for_cfg,
        model_family=args.model_family,
        model_name=args.model_name,
        exp_name=args.exp_name,
        tag=args.tag,
        naming_mode=args.name_mode,
        run_name=args.run_name,
    )

    run_root = cfg.run_root(mode="pred")
    run_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(best_pt))
    model.predict(
        source=args.source,
        device=args.device,
        save=True,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        project=str(run_root.parent),
        name=run_root.name,
        exist_ok=False,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )

    wm = extract_weight_meta(str(best_pt))
    meta = {
        "mode": "pred",
        "task": task_short,
        "dataset": cfg.dataset,
        "model": cfg.model_name,
        "tag": cfg.tag,
        "created_at": now_iso(),
        "weight_path": str(best_pt),
        "source": args.source,
        "notes": "",
        "weight_file": wm["weight_file"],
        "weight_parent_dir": wm["weight_parent_dir"],
        "used_from_train_dir": wm["used_from_train_dir"],
    }
    write_run_meta_yaml(run_root, meta)
    print("PRED DONE:", run_root)


if __name__ == "__main__":
    main()
