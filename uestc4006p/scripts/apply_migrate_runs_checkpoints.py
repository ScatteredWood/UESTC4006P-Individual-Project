from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class RenameRow:
    old_name: str
    new_name: str
    status: str
    reason: str


# 这些映射在当前数据上语义不清，先人工确认后再迁移
MANUAL_REVIEW_OLD_NAMES = {
    "misc_crack500_csc_con_na_0_b0_manual",
    "seg_r22c_best-best_0_b0_cascade",
    "seg_r22c_best-best_1280_b0_cascade",
    "segFT__pseg512_det20_sg400_sc55_mr2_seed42",
}


def read_mapping_csv(path: Path) -> list[RenameRow]:
    rows: list[RenameRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                RenameRow(
                    old_name=str(r.get("old_name", "")).strip(),
                    new_name=str(r.get("new_name", "")).strip(),
                    status=str(r.get("status", "")).strip(),
                    reason=str(r.get("reason", "")).strip(),
                )
            )
    return rows


def resolve_unique_name(root: Path, desired: str) -> tuple[str, str]:
    p = root / desired
    if not p.exists():
        return desired, "ok"
    idx = 2
    while True:
        cand = f"{desired}_{idx}"
        cp = root / cand
        if not cp.exists():
            return cand, f"conflict_resolved_to_{cand}"
        idx += 1


def apply_mapping(root: Path, rows: Iterable[RenameRow], report_path: Path) -> dict[str, int]:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"renamed": 0, "manual_review": 0, "skipped_missing": 0, "error": 0}

    with report_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["old_name", "planned_new_name", "final_new_name", "status", "reason"])

        for row in rows:
            old_name = row.old_name
            new_name = row.new_name
            src = root / old_name

            if (
                row.status.lower() != "auto"
                or old_name in MANUAL_REVIEW_OLD_NAMES
                or not new_name
            ):
                reason = row.reason or "manual_review_required"
                if old_name in MANUAL_REVIEW_OLD_NAMES:
                    reason = f"{reason}; flagged_as_unreasonable_mapping"
                w.writerow([old_name, new_name, "", "manual_review", reason])
                stats["manual_review"] += 1
                continue

            if not src.exists() or not src.is_dir():
                w.writerow([old_name, new_name, "", "skipped_missing", "source_dir_not_found"])
                stats["skipped_missing"] += 1
                continue

            final_name, conflict_note = resolve_unique_name(root, new_name)
            dst = root / final_name

            try:
                src.rename(dst)
                reason = "ok" if conflict_note == "ok" else conflict_note
                w.writerow([old_name, new_name, final_name, "renamed", reason])
                stats["renamed"] += 1
            except Exception as e:
                w.writerow([old_name, new_name, "", "error", str(e)])
                stats["error"] += 1

    return stats


def write_summary(
    out_path: Path,
    run_stats: dict[str, int],
    ckpt_stats: dict[str, int],
    run_report: Path,
    ckpt_report: Path,
) -> None:
    total_manual = run_stats["manual_review"] + ckpt_stats["manual_review"]
    total_error = run_stats["error"] + ckpt_stats["error"]
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Migration Report\n\n")
        f.write("## Runs\n")
        f.write(f"- renamed: {run_stats['renamed']}\n")
        f.write(f"- manual_review: {run_stats['manual_review']}\n")
        f.write(f"- skipped_missing: {run_stats['skipped_missing']}\n")
        f.write(f"- error: {run_stats['error']}\n\n")
        f.write("## Checkpoints\n")
        f.write(f"- renamed: {ckpt_stats['renamed']}\n")
        f.write(f"- manual_review: {ckpt_stats['manual_review']}\n")
        f.write(f"- skipped_missing: {ckpt_stats['skipped_missing']}\n")
        f.write(f"- error: {ckpt_stats['error']}\n\n")
        f.write("## Totals\n")
        f.write(f"- total_manual_review: {total_manual}\n")
        f.write(f"- total_error: {total_error}\n\n")
        f.write("## Output Files\n")
        f.write(f"- runs report: `{run_report}`\n")
        f.write(f"- checkpoints report: `{ckpt_report}`\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply runs/checkpoints migration from dry-run mapping CSV")
    ap.add_argument("--runs-root", default=str(Path(__file__).resolve().parents[1] / "runs"))
    ap.add_argument("--checkpoints-root", default=str(Path(__file__).resolve().parents[1] / "checkpoints"))
    ap.add_argument("--runs-mapping", default=str(Path(__file__).resolve().parent / "dryrun_outputs" / "rename_mapping.csv"))
    ap.add_argument("--checkpoints-mapping", default=str(Path(__file__).resolve().parent / "dryrun_outputs" / "checkpoint_mapping.csv"))
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "migration_outputs"))
    args = ap.parse_args()

    runs_root = Path(args.runs_root).resolve()
    ckpt_root = Path(args.checkpoints_root).resolve()
    runs_mapping = Path(args.runs_mapping).resolve()
    ckpt_mapping = Path(args.checkpoints_mapping).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    run_rows = read_mapping_csv(runs_mapping)
    ckpt_rows = read_mapping_csv(ckpt_mapping)

    run_report = outdir / "rename_report.csv"
    ckpt_report = outdir / "checkpoint_rename_report.csv"
    summary = outdir / "migration_report.md"

    run_stats = apply_mapping(runs_root, run_rows, run_report)
    ckpt_stats = apply_mapping(ckpt_root, ckpt_rows, ckpt_report)
    write_summary(summary, run_stats, ckpt_stats, run_report, ckpt_report)

    print(f"runs_report={run_report}")
    print(f"checkpoints_report={ckpt_report}")
    print(f"summary_report={summary}")
    print(
        "stats="
        f"runs:{run_stats['renamed']}/{run_stats['manual_review']}/{run_stats['skipped_missing']}/{run_stats['error']},"
        f"ckpt:{ckpt_stats['renamed']}/{ckpt_stats['manual_review']}/{ckpt_stats['skipped_missing']}/{ckpt_stats['error']}"
    )


if __name__ == "__main__":
    main()
