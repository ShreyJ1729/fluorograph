"""
Script 00: Filter and stratified-split dataset.

Filters fpbase_merged.csv to complete proteins with seq_len >= 100,
assigns emission bin labels, and creates stratified train/val/test splits.

Output: data/splits/train.csv, data/splits/val.csv, data/splits/test.csv
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Paths — fpbase_merged.csv lives at project root or data/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
# Check both data/ and project root for the merged CSV
_merged_in_data = os.path.join(DATA_DIR, "fpbase_merged.csv")
_merged_in_root = os.path.join(_PROJECT_ROOT, "fpbase_merged.csv")
MERGED_CSV = _merged_in_data if os.path.exists(_merged_in_data) else _merged_in_root
SPLITS_DIR = os.path.join(DATA_DIR, "splits")


def assign_em_bin(em_max: float) -> str:
    """Assign emission bin label based on em_max wavelength (nm)."""
    if em_max < 460:
        return "blue"
    elif em_max < 500:
        return "cyan"
    elif em_max < 530:
        return "green"
    elif em_max < 570:
        return "yellow"
    elif em_max < 600:
        return "orange"
    elif em_max < 650:
        return "red"
    elif em_max < 700:
        return "far-red"
    else:
        return "infrared"


def main() -> None:
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # Load and filter
    df = pd.read_csv(MERGED_CSV)
    print(f"Loaded {len(df)} total rows from fpbase_merged.csv")

    filtered = df[(df["complete"] == 1) & (df["seq_len"] >= 100)].copy()
    print(f"After filter (complete==1, seq_len>=100): {len(filtered)} rows")

    # Assign emission bin
    filtered["em_bin"] = filtered["em_max"].apply(assign_em_bin)

    # First split: carve out test (15%)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(sss_test.split(filtered, filtered["em_bin"]))
    train_val = filtered.iloc[train_val_idx].reset_index(drop=True)
    test = filtered.iloc[test_idx].reset_index(drop=True)

    # Second split: carve out val from train_val (~15% of total = 0.176 of train_val)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.176, random_state=42)
    train_idx, val_idx = next(sss_val.split(train_val, train_val["em_bin"]))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)

    # Write splits
    train.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(SPLITS_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)

    print(f"\nSplit sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Total: {len(train) + len(val) + len(test)} (expected ~{len(filtered)})")

    # Bin distribution per split
    print("\nBin distribution per split:")
    bins = sorted(filtered["em_bin"].unique())
    header = f"{'bin':<12} {'train':>8} {'val':>8} {'test':>8}"
    print(header)
    print("-" * len(header))
    missing_from_train = []
    for b in bins:
        t = (train["em_bin"] == b).sum()
        v = (val["em_bin"] == b).sum()
        te = (test["em_bin"] == b).sum()
        print(f"{b:<12} {t:>8} {v:>8} {te:>8}")
        if t == 0:
            missing_from_train.append(b)

    if missing_from_train:
        print(f"\nWARNING: bins missing from train: {missing_from_train}")
        sys.exit(1)
    else:
        print("\nAll bins appear in train set. ✓")
        print("Splits written to data/splits/")


if __name__ == "__main__":
    main()
