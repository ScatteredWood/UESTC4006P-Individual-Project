# 2026-04-04 Naming Cleanup（里程碑）

## 本轮修改文件
- `uestc4006p/scripts/apply_migrate_runs_checkpoints.py`（新增）
- `uestc4006p/scripts/dryrun_outputs/rename_report.csv`（真实迁移报告）
- `uestc4006p/scripts/dryrun_outputs/checkpoint_rename_report.csv`（真实迁移报告）
- `uestc4006p/scripts/dryrun_outputs/migration_report.md`（汇总报告）
- `uestc4006p/CODEX_STATUS.md`（当前状态）

说明：
- 本轮未对 `tools` 做真实改名或目录移动，仅保留 dry-run 结果。
- 本轮未修改 `AGENTS.md`。

## runs 迁移结果（最终）
- renamed: 10
- manual_keep: 3
- skipped_missing: 0
- error: 0

### runs manual_keep（保留原名）
- `misc_crack500_csc_con_na_0_b0_manual`
- `seg_r22c_best-best_0_b0_cascade`
- `seg_r22c_best-best_1280_b0_cascade`

## checkpoints 迁移结果（最终）
- renamed: 9
- manual_keep: 0
- skipped_missing: 0
- error: 0

### manual_review 补处理（本次）
- migrated:
  - `segFT__pseg512_det20_sg400_sc55_mr2_seed42`
  - `-> train_seg_rddcn_best_pseg512_det20_sg400_sc55`
- manual_keep:
  - `misc_crack500_csc_con_na_0_b0_manual`
  - `seg_r22c_best-best_0_b0_cascade`
  - `seg_r22c_best-best_1280_b0_cascade`

## 报告文件
- `uestc4006p/scripts/dryrun_outputs/rename_report.csv`
- `uestc4006p/scripts/dryrun_outputs/checkpoint_rename_report.csv`
- `uestc4006p/scripts/dryrun_outputs/migration_report.md`
