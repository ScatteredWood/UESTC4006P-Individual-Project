# Tools Reference Update Preview (Dry-run)

## Hardcoded runs/checkpoints path hits
- `tools\pseudo_seg_from_det_sizegate_tile.py:22`: `DET_WEIGHTS = r"E:\repositories\ultralytics\uestc4006p\checkpoints\det_RDD2022_China_yolov8n_1024_ep300_bs16_seed42_baseline__drone-and-motorbike-mixed\best.pt"`
- `tools\pseudo_seg_from_det_sizegate_tile.py:23`: `SEG_WEIGHTS = r"E:\repositories\ultralytics\uestc4006p\checkpoints\seg_CRACK500_ALL_yolov8n-seg_800_ep200_bs24_seed42_baseline__mask2poly-v2__open1close1__eps0p001__max500\best.pt"`
- `tools\smoke_det_d20.py:8`: `DET_W  = r"E:\repositories\ultralytics\uestc4006p\checkpoints\det_RDD2022_China_yolov8n_1024_ep300_bs16_seed42_baseline__drone-and-motorbike-mixed\best.pt"  # <-- 改成你找到的 det best.pt`
- `uestc4006p\scripts\apply_rename_runs.py:38`: `ap.add_argument("--runs-root", default=r"E:\repositories\ultralytics\uestc4006p\runs")`
- `uestc4006p\scripts\cascade_infer_detseg.py:621`: `ap.add_argument("--outdir", default="uestc4006p/runs", help="output dir")`
- `uestc4006p\scripts\dryrun_migrate_runs_checkpoints.py:182`: `ap.add_argument("--runs-root", default=r"E:\repositories\ultralytics\uestc4006p\runs")`
- `uestc4006p\scripts\dryrun_migrate_runs_checkpoints.py:183`: `ap.add_argument("--checkpoints-root", default=r"E:\repositories\ultralytics\uestc4006p\checkpoints")`
- `uestc4006p\scripts\dryrun_rename_runs.py:158`: `ap.add_argument("--runs-root", default=r"E:\repositories\ultralytics\uestc4006p\runs")`
- `uestc4006p\scripts\predict.py:9`: `best_pt = r"E:\repositories\ultralytics\uestc4006p\runs\seg_CSC_2649_yolov8n-seg_800_ep200_bs24_seed42_ftbest__mask2poly-v2__open1close1__eps0p001__max500\weights\best.pt"`
- `uestc4006p\scripts\train.py:18`: `- runs 输出：<REPO_ROOT>/uestc4006p/runs/<exp_id>/`
- `uestc4006p\scripts\train.py:19`: `- checkpoint 归档：<REPO_ROOT>/uestc4006p/checkpoints/<exp_id>/{best,last}.pt`

## Suggested tools rename mapping file
- `E:\repositories\ultralytics\uestc4006p\scripts\dryrun_outputs\tools_rename_mapping.csv`
