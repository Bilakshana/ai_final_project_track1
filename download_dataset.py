"""
download_dataset.py — Automated TrashNet dataset download and setup.

Option A (Kaggle API — recommended):
    1. Get your Kaggle API key from https://www.kaggle.com/settings/account
    2. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\\.kaggle\\ (Windows)
    3. Run: python download_dataset.py --source kaggle

Option B (Direct download — no API key):
    python download_dataset.py --source manual
    (Opens the download page in your browser)

Option C (If you already have the zip):
    python download_dataset.py --source local --zip /path/to/dataset.zip
"""

import argparse
import os
import sys
import zipfile
import shutil
from pathlib import Path


KAGGLE_DATASET = "feyzazkefe/trashnet"    # Kaggle dataset slug
DATASET_DIR    = "dataset/"
DATA_DIR       = "data/"
CLASSES        = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def download_kaggle():
    """Download via official Kaggle API."""
    try:
        import kaggle  # pip install kaggle
    except ImportError:
        print("❌ kaggle package not found. Install it with:")
        print("   pip install kaggle")
        sys.exit(1)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ kaggle.json not found at", kaggle_json)
        print("   Download it from: https://www.kaggle.com/settings/account")
        print("   → Account → API → Create New Token")
        sys.exit(1)

    print(f"📥 Downloading {KAGGLE_DATASET} from Kaggle…")
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATASET_DIR} --unzip")
    print("✅ Download complete!")


def download_manual():
    """Open browser to manual download page."""
    import webbrowser
    url = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
    print(f"🌐 Opening: {url}")
    webbrowser.open(url)
    print("\nAfter downloading:")
    print("  1. Unzip the file")
    print(f"  2. Move folders to {DATASET_DIR}")
    print("     so you have: dataset/cardboard/, dataset/glass/, …")
    print("  3. Run: python download_dataset.py --source local")


def setup_from_local(zip_path: str = None):
    """Extract and organise a local zip file."""
    if zip_path:
        print(f"📦 Extracting {zip_path}…")
        os.makedirs(DATASET_DIR, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATASET_DIR)
        print("✅ Extracted!")

    # Auto-detect nested folder structure and flatten
    _flatten_dataset()


def _flatten_dataset():
    """
    TrashNet may unzip as dataset/dataset-resized/cardboard/ etc.
    Flatten to dataset/cardboard/.
    """
    src = Path(DATASET_DIR)
    # Find class folders wherever they are
    for cls in CLASSES:
        found = list(src.rglob(cls))
        if found:
            target = src / cls
            if found[0] != target:
                print(f"  Moving {found[0]} → {target}")
                shutil.copytree(str(found[0]), str(target), dirs_exist_ok=True)
    print("✅ Dataset structure ready")
    _print_counts()


def _print_counts():
    src = Path(DATASET_DIR)
    print("\nDataset image counts:")
    total = 0
    for cls in CLASSES:
        p = src / cls
        n = len(list(p.glob("*.*"))) if p.exists() else 0
        print(f"  {cls:12s}: {n}")
        total += n
    print(f"  {'TOTAL':12s}: {total}")


def split_and_verify():
    """Run the train/val split after downloading."""
    from preprocess import split_dataset, dataset_stats
    print("\n🔀 Splitting into train / val…")
    split_dataset(DATASET_DIR, DATA_DIR, val_split=0.20)
    dataset_stats(DATA_DIR)
    print("\n✅ All done! You can now run: python train.py")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrashNet dataset downloader")
    parser.add_argument("--source", choices=["kaggle", "manual", "local"],
                        default="manual",
                        help="kaggle=API, manual=browser, local=use existing zip")
    parser.add_argument("--zip", default=None,
                        help="Path to local zip (used with --source local)")
    parser.add_argument("--split", action="store_true",
                        help="Also run train/val split after download")
    args = parser.parse_args()

    if args.source == "kaggle":
        download_kaggle()
        _flatten_dataset()
    elif args.source == "manual":
        download_manual()
    elif args.source == "local":
        setup_from_local(args.zip)

    if args.split and Path(DATASET_DIR).exists():
        split_and_verify()
