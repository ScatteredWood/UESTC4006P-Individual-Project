# tools/merge_rdd2022_all_norename.py
# Merge RDD2022 multi-country VOC folders into ONE VOC-style dataset (NO renaming).
# Assumes filenames are already unique (e.g., Japan_000007.jpg).

import argparse
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return False
    shutil.copy2(src, dst)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-public", required=True,
                    help=r'public root, e.g. E:\Large Files\...\datasets\public')
    ap.add_argument("--out-voc", required=True,
                    help=r'output VOC root, e.g. ...\RDD2022_ALL\ALL')
    ap.add_argument("--include-test", action="store_true",
                    help="also merge test/images (no labels)")
    args = ap.parse_args()

    src_public = Path(args.src_public)
    out_voc = Path(args.out_voc)

    out_train_img = out_voc / "train" / "images"
    out_train_xml = out_voc / "train" / "annotations" / "xmls"
    out_test_img  = out_voc / "test" / "images"

    out_train_img.mkdir(parents=True, exist_ok=True)
    out_train_xml.mkdir(parents=True, exist_ok=True)
    if args.include_test:
        out_test_img.mkdir(parents=True, exist_ok=True)

    # Find all RDD2022_* folders under public
    rdd_folders = sorted([p for p in src_public.iterdir() if p.is_dir() and p.name.startswith("RDD2022_")])
    if not rdd_folders:
        raise RuntimeError(f"No RDD2022_* folders found under: {src_public}")

    n_img = n_xml = n_pair = 0
    n_img_skip = n_xml_skip = 0
    n_missing_xml = 0

    for rdd in rdd_folders:
        # Expect structure: RDD2022_India/India/train/images ...
        # The country folder name is usually the suffix after "RDD2022_"
        country = rdd.name.replace("RDD2022_", "")
        country_root = rdd / country
        train_img_dir = country_root / "train" / "images"
        train_xml_dir = country_root / "train" / "annotations" / "xmls"
        test_img_dir  = country_root / "test" / "images"

        if not train_img_dir.exists() or not train_xml_dir.exists():
            # Skip unexpected folder layouts
            continue

        # Merge train
        for img in train_img_dir.rglob("*"):
            if not is_image(img):
                continue
            xml = train_xml_dir / (img.stem + ".xml")
            if not xml.exists():
                n_missing_xml += 1
                continue

            # Copy image/xml as-is (no rename)
            copied_img = safe_copy(img, out_train_img / img.name)
            copied_xml = safe_copy(xml, out_train_xml / xml.name)

            n_img += 1
            n_xml += 1
            n_pair += 1
            if not copied_img:
                n_img_skip += 1
            if not copied_xml:
                n_xml_skip += 1

        # Merge test (optional)
        if args.include_test and test_img_dir.exists():
            for img in test_img_dir.rglob("*"):
                if not is_image(img):
                    continue
                safe_copy(img, out_test_img / img.name)

    print("✅ MERGE DONE (NO RENAME)")
    print("OUT_VOC:", out_voc)
    print(f"TRAIN pairs processed: {n_pair}")
    print(f"TRAIN images processed: {n_img} (skipped existing: {n_img_skip})")
    print(f"TRAIN xmls processed:   {n_xml} (skipped existing: {n_xml_skip})")
    print(f"Missing xml for images: {n_missing_xml}")
    if n_missing_xml > 0:
        print("⚠️ Some images had no matching XML. They were skipped.")

if __name__ == "__main__":
    main()
