"""
Manifest Generator and Data Integrity Verifier.
Domain-Generalized Deep Learning for Real-World Plant Disease Recognition.

Performs:
1. Deep integrity verification (PIL image open, RGB check, dimension check, corrupt file filter).
2. Canonical taxonomy alignment for all 28 classes.
3. Deterministic stratified 80/20 train/validation split on PlantVillage (Source Domain).
4. Strictly isolates 100% of PlantDoc images as the Unseen Target Domain Test set.
5. Exports CSV manifests and class metadata into data/processed/.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import pandas as pd
from tqdm import tqdm

from src.taxonomy import (
    CANONICAL_CLASSES,
    CANONICAL_TO_ID,
    PLANTVILLAGE_TO_CANONICAL,
    PLANTDOC_TO_CANONICAL,
    CANONICAL_DISPLAY_NAMES,
    NUM_CLASSES,
)
RANDOM_SEED = 42


def verify_image(path: Path) -> Tuple[bool, Optional[str]]:
    """Check image file integrity, readability, and valid dimensions."""
    if not path.is_file():
        return False, "File does not exist"
    if path.stat().st_size == 0:
        return False, "0-byte empty file"
    try:
        with Image.open(path) as img:
            img.verify()
        # Reopen to test conversion to RGB
        with Image.open(path) as img:
            w, h = img.size
            if w < 10 or h < 10:
                return False, f"Degenerate dimensions ({w}x{h})"
            _ = img.convert("RGB")
        return True, None
    except Exception as e:
        return False, str(e)


def build_manifests() -> Dict[str, Any]:
    print("=" * 65)
    print("   PHASE 1: DATA ENGINEERING & CANONICAL MANIFEST GENERATOR   ")
    print("=" * 65)

    random.seed(RANDOM_SEED)

    # 1. Locate dataset roots
    pv_root = PROJECT_ROOT / "data" / "plantvillage"
    if (pv_root / "plantvillage dataset" / "color").exists():
        pv_root = pv_root / "plantvillage dataset" / "color"

    doc_root = PROJECT_ROOT / "data" / "plantdoc"
    doc_train = doc_root / "train"
    doc_test = doc_root / "test"

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1] Source Domain Path (PlantVillage): {pv_root}")
    print(f"    Target Domain Path (PlantDoc):     {doc_root}")
    print(f"    Processed Manifest Output:        {processed_dir}\n")

    # 2. Collect & verify PlantVillage images (Source Domain)
    print("[2] Scanning & Verifying Source Domain (PlantVillage)...")
    pv_records: List[Dict[str, Any]] = []
    pv_corrupt_count = 0

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for pv_folder_name, canonical_name in PLANTVILLAGE_TO_CANONICAL.items():
        folder_path = pv_root / pv_folder_name
        if not folder_path.exists():
            print(f"    [WARN] PlantVillage folder not found: {folder_path}")
            continue

        class_id = CANONICAL_TO_ID[canonical_name]
        display_name = CANONICAL_DISPLAY_NAMES[canonical_name]

        for img_path in folder_path.iterdir():
            if img_path.suffix.lower() not in valid_exts:
                continue

            is_valid, err = verify_image(img_path)
            if is_valid:
                # Store relative path for portability
                rel_path = img_path.relative_to(PROJECT_ROOT).as_posix()
                pv_records.append({
                    "filepath": rel_path,
                    "class_id": class_id,
                    "canonical_name": canonical_name,
                    "display_name": display_name,
                    "domain": "plantvillage",
                    "domain_type": "source_lab",
                })
            else:
                pv_corrupt_count += 1

    print(f"    Verified PlantVillage Images: {len(pv_records)} valid, {pv_corrupt_count} corrupt.")

    # 3. Collect & verify PlantDoc images (Target Domain)
    print("\n[3] Scanning & Verifying Target Domain (PlantDoc)...")
    doc_records: List[Dict[str, Any]] = []
    doc_corrupt_count = 0

    for doc_sub in [doc_train, doc_test]:
        if not doc_sub.exists():
            continue
        for doc_folder_name, canonical_name in PLANTDOC_TO_CANONICAL.items():
            folder_path = doc_sub / doc_folder_name
            if not folder_path.exists():
                continue

            class_id = CANONICAL_TO_ID[canonical_name]
            display_name = CANONICAL_DISPLAY_NAMES[canonical_name]

            for img_path in folder_path.iterdir():
                if img_path.suffix.lower() not in valid_exts:
                    continue

                is_valid, err = verify_image(img_path)
                if is_valid:
                    rel_path = img_path.relative_to(PROJECT_ROOT).as_posix()
                    doc_records.append({
                        "filepath": rel_path,
                        "class_id": class_id,
                        "canonical_name": canonical_name,
                        "display_name": display_name,
                        "domain": "plantdoc",
                        "domain_type": "target_field",
                    })
                else:
                    doc_corrupt_count += 1

    print(f"    Verified PlantDoc Images: {len(doc_records)} valid, {doc_corrupt_count} corrupt.")

    # 4. Stratified 80/20 Split on PlantVillage (Source Domain)
    print("\n[4] Performing Stratified 80/20 Split on Source Domain...")
    # Group by class_id
    by_class: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(NUM_CLASSES)}
    for rec in pv_records:
        by_class[rec["class_id"]].append(rec)

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    for cid in range(NUM_CLASSES):
        items = by_class[cid]
        random.shuffle(items)
        split_idx = int(0.8 * len(items))
        train_records.extend(items[:split_idx])
        val_records.extend(items[split_idx:])

    # Shuffle train and val
    random.shuffle(train_records)
    random.shuffle(val_records)

    # Test records are 100% of PlantDoc
    test_records = doc_records
    random.shuffle(test_records)

    # 5. Export DataFrames to CSV
    df_train = pd.DataFrame(train_records)
    df_val = pd.DataFrame(val_records)
    df_test = pd.DataFrame(test_records)

    train_csv = processed_dir / "train_manifest.csv"
    val_csv = processed_dir / "val_manifest.csv"
    test_csv = processed_dir / "test_manifest.csv"

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    # 6. Export taxonomy metadata JSON
    meta = {
        "num_classes": NUM_CLASSES,
        "classes": CANONICAL_CLASSES,
        "class_to_id": CANONICAL_TO_ID,
        "display_names": CANONICAL_DISPLAY_NAMES,
        "splits": {
            "train": {"count": len(df_train), "domain": "plantvillage (80%)"},
            "val": {"count": len(df_val), "domain": "plantvillage (20%)"},
            "test": {"count": len(df_test), "domain": "plantdoc (100% held-out)"},
        },
    }
    meta_path = processed_dir / "classes.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # 7. Print Class Distribution Summary Table
    print("\n" + "=" * 65)
    print(f"{'Class ID':<9} {'Canonical Name':<32} {'Train':<7} {'Val':<6} {'Test':<6}")
    print("-" * 65)

    train_counts = df_train["class_id"].value_counts().to_dict()
    val_counts = df_val["class_id"].value_counts().to_dict()
    test_counts = df_test["class_id"].value_counts().to_dict()

    for cid in range(NUM_CLASSES):
        cname = CANONICAL_CLASSES[cid]
        tr_c = train_counts.get(cid, 0)
        va_c = val_counts.get(cid, 0)
        te_c = test_counts.get(cid, 0)
        print(f"[{cid:02d}]      {cname:<32} {tr_c:<7} {va_c:<6} {te_c:<6}")

    print("=" * 65)
    print(f"TOTAL:    {'All 28 Classes':<32} {len(df_train):<7} {len(df_val):<6} {len(df_test):<6}")
    print("=" * 65)

    print(f"\n[SUCCESS] Manifests saved successfully:")
    print(f"  -> {train_csv} ({len(df_train)} rows)")
    print(f"  -> {val_csv}   ({len(df_val)} rows)")
    print(f"  -> {test_csv}  ({len(df_test)} rows)")
    print(f"  -> {meta_path}\n")

    return meta


if __name__ == "__main__":
    build_manifests()
