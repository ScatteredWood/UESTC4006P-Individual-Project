# CODEX Status

## 当前已完成
- 已统一新命名规则并落地到 train / val / pred / cascade 输出路径生成逻辑。
- 已去除新逻辑中的时间戳子目录依赖。
- 已完成 runs/checkpoints 迁移收尾：
  - 1 个 manual_review checkpoint 已按人工确认迁移：
    - `segFT__pseg512_det20_sg400_sc55_mr2_seed42`
    - `-> train_seg_rddcn_best_pseg512_det20_sg400_sc55`
  - 3 个 runs 目录已按人工确认标记 `manual_keep`（保留原名）：
    - `misc_crack500_csc_con_na_0_b0_manual`
    - `seg_r22c_best-best_0_b0_cascade`
    - `seg_r22c_best-best_1280_b0_cascade`
- 迁移报告已同步更新：
  - `uestc4006p/scripts/dryrun_outputs/rename_report.csv`
  - `uestc4006p/scripts/dryrun_outputs/checkpoint_rename_report.csv`
  - `uestc4006p/scripts/dryrun_outputs/migration_report.md`

## 当前命名规则
- 目录名格式：
  - `{mode}_{task}_{data}_{model}`
  - 或 `{mode}_{task}_{data}_{model}_{tag}`
- `mode` 仅允许：`train` / `val` / `pred`
- `task` 支持：`det` / `seg` / `cascade`
- 无 `tag` 时完全省略
- 不使用时间戳子目录
- 重名冲突自动追加：`_2` / `_3` / `_4` ...
- `checkpoints` 与对应训练目录同名
- `train` / `val` / `pred` / `cascade` 输出目录都写 `run_meta.yaml`

## 当前未完成
- `tools` 目录仍仅完成 dry-run 映射与引用预览，未做真实改名/移动。

## 下一步建议
1. 如果要继续命名治理，下一线程只处理 `tools` 的真实重命名与引用修复。
