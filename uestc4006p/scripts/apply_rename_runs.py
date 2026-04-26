from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ResultRow:
    old_name: str
    new_rel: str
    status: str
    message: str


def load_mapping(mapping_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with mapping_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: (v or "") for k, v in row.items()})
    return rows


def write_report(report_csv: Path, results: list[ResultRow]) -> None:
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["old_name", "new_rel", "status", "message"])
        for x in results:
            w.writerow([x.old_name, x.new_rel, x.status, x.message])


def main():
    ap = argparse.ArgumentParser(description="按 mapping.csv 执行 runs 目录重命名")
    ap.add_argument("--runs-root", default=str(Path(__file__).resolve().parents[1] / "runs"))
    ap.add_argument("--mapping-csv", required=True)
    ap.add_argument("--report-csv", default="")
    args = ap.parse_args()

    runs_root = Path(args.runs_root).resolve()
    mapping_csv = Path(args.mapping_csv).resolve()
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root not found: {runs_root}")
    if not mapping_csv.exists():
        raise FileNotFoundError(f"mapping csv not found: {mapping_csv}")

    if args.report_csv:
        report_csv = Path(args.report_csv).resolve()
    else:
        ts = datetime.now().strftime("%y%m%d-%H%M")
        report_csv = mapping_csv.parent / f"rename_report_{ts}.csv"

    mapping = load_mapping(mapping_csv)
    results: list[ResultRow] = []

    for row in mapping:
        old_name = row.get("old_name", "").strip()
        new_rel = row.get("new_rel", "").strip()
        if not old_name or not new_rel:
            results.append(ResultRow(old_name, new_rel, "skipped", "invalid mapping row"))
            continue

        src = runs_root / old_name
        dst = runs_root / new_rel

        try:
            if not src.exists():
                results.append(ResultRow(old_name, new_rel, "skipped_missing", "source not found"))
                continue
            if dst.exists():
                results.append(ResultRow(old_name, new_rel, "conflict", "target already exists"))
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            results.append(ResultRow(old_name, new_rel, "renamed", "ok"))
        except Exception as e:
            results.append(ResultRow(old_name, new_rel, "error", str(e)))
            continue

    write_report(report_csv, results)

    renamed = sum(1 for x in results if x.status == "renamed")
    skipped = sum(1 for x in results if x.status.startswith("skipped"))
    conflicts = sum(1 for x in results if x.status == "conflict")
    errors = sum(1 for x in results if x.status == "error")

    print(f"RENAMED={renamed}")
    print(f"SKIPPED={skipped}")
    print(f"CONFLICTS={conflicts}")
    print(f"ERRORS={errors}")
    print(f"REPORT={report_csv}")


if __name__ == "__main__":
    main()

