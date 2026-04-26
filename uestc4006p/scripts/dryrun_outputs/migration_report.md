# Migration Report (Finalized)

## Runs
- renamed: 10
- manual_keep: 3
- skipped_missing: 0
- error: 0

## Checkpoints
- renamed: 9
- manual_keep: 0
- skipped_missing: 0
- error: 0

## Manual Review Final Handling
- migrated:
  - `checkpoints/segFT__pseg512_det20_sg400_sc55_mr2_seed42` -> `checkpoints/train_seg_rddcn_best_pseg512_det20_sg400_sc55`
- manual_keep:
  - `runs/misc_crack500_csc_con_na_0_b0_manual`
  - `runs/seg_r22c_best-best_0_b0_cascade`
  - `runs/seg_r22c_best-best_1280_b0_cascade`

## Output Files
- runs report: `E:\repositories\ultralytics\uestc4006p\scripts\dryrun_outputs\rename_report.csv`
- checkpoints report: `E:\repositories\ultralytics\uestc4006p\scripts\dryrun_outputs\checkpoint_rename_report.csv`
