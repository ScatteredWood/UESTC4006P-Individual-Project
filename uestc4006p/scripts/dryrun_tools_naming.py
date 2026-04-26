from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re


def suggest_tools_target(name: str) -> tuple[str, str]:
    stem = name.replace(".py", "")
    s = stem.lower()
    if s.startswith(("check_", "smoke_")):
        return "tools/run", f"run_{s}.py"
    if s.startswith(("split_", "extract_", "prepare_", "labelme_", "maskpng_", "voc2yolo_", "sync_", "clean_", "merge_", "prune_", "rebalance_")):
        return "tools/data", f"data_{s}.py"
    if s.startswith(("visualize_", "vis_")):
        return "tools/vis", f"vis_{s}.py"
    if s.startswith(("pseudo_", "filter_")):
        return "tools/eval", f"eval_{s}.py"
    return "tools/ops", f"ops_{s}.py"


def scan_hardcoded_paths(root: Path, target_tools: Path) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    pattern = re.compile(r"(uestc4006p[\\/](runs|checkpoints)[^\"'\n]*)", re.IGNORECASE)
    for p in root.rglob("*.py"):
        if ".git" in p.parts:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(txt.splitlines(), 1):
            if pattern.search(line):
                hits.append((str(p.relative_to(root)), i, line.strip()))
    return hits


def main():
    ap = argparse.ArgumentParser(description="tools 命名迁移 dry-run（不做真实改动）")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--tools-dir", default=str(Path(__file__).resolve().parents[2] / "tools"))
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "dryrun_outputs"))
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    tools_dir = Path(args.tools_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    mapping_csv = outdir / "tools_rename_mapping.csv"
    preview_md = outdir / "tools_ref_update_preview.md"

    rows = []
    for p in sorted(tools_dir.glob("*.py"), key=lambda x: x.name.lower()):
        tgt_dir, tgt_name = suggest_tools_target(p.name)
        rows.append((str(p.relative_to(repo_root)), f"{tgt_dir}/{tgt_name}", "dry-run"))

    with mapping_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["old_path", "suggested_new_path", "status"])
        for r in rows:
            w.writerow(r)

    hits = scan_hardcoded_paths(repo_root, tools_dir)
    with preview_md.open("w", encoding="utf-8") as f:
        f.write("# Tools Reference Update Preview (Dry-run)\n\n")
        f.write("## Hardcoded runs/checkpoints path hits\n")
        if not hits:
            f.write("- none\n")
        else:
            for rel, ln, txt in hits:
                f.write(f"- `{rel}:{ln}`: `{txt}`\n")
        f.write("\n## Suggested tools rename mapping file\n")
        f.write(f"- `{mapping_csv}`\n")

    print(f"tools_mapping: {mapping_csv}")
    print(f"ref_preview: {preview_md}")


if __name__ == "__main__":
    main()

