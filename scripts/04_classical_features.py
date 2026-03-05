"""
Script 04: Classical feature extraction.

Computes per-protein biochemical and structural features for XGBoost ablation models:
  - 20-dim AA composition (fraction)
  - Sequence length
  - Isoelectric point (Bio.SeqUtils.ProtParam)
  - Molecular weight
  - Chromophore triplet one-hot (alignment positions 736-738 = EGFP positions 65-67)
  - Oligomerization one-hot: m/d/t/wd/other (5-dim)
  - Switch type one-hot: b/ps/pa/pc/other (5-dim)

Output: features/classical_features.csv with columns: name, <feature columns>
"""

import os
import sys

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
ALIGNED_FASTA = os.path.join(DATA_DIR, "aligned", "all.fasta")
FEATURES_DIR = os.path.join(_PROJECT_ROOT, "features")
OUTPUT_CSV = os.path.join(FEATURES_DIR, "classical_features.csv")

# 20 standard amino acids (alphabetical order)
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Chromophore triplet alignment columns (EGFP positions 65-67 in the MSA)
CHROM_COLS = (736, 737, 738)

# Oligomerization categories (5-dim one-hot)
OLIGO_CATS = ["m", "d", "t", "wd", "other"]

# Switch type categories (5-dim one-hot)
SWITCH_CATS = ["b", "ps", "pa", "pc", "other"]


def load_alignment(fasta_path: str) -> dict[str, str]:
    """Parse aligned FASTA into {name: aligned_seq} dict."""
    if not os.path.exists(fasta_path):
        print(f"ERROR: Aligned FASTA not found at {fasta_path}")
        print("Run scripts/01_mafft_align.py first.")
        sys.exit(1)
    return {r.description: str(r.seq).upper() for r in SeqIO.parse(fasta_path, "fasta")}


def aa_composition(seq: str) -> list[float]:
    """Return 20-dim AA composition vector (fraction of each standard AA)."""
    seq_clean = seq.upper().replace("-", "")
    n = len(seq_clean) if seq_clean else 1
    return [seq_clean.count(aa) / n for aa in AMINO_ACIDS]


def isoelectric_point(seq: str) -> float:
    """Compute isoelectric point; unknown AAs replaced with Ala."""
    clean = "".join(c if c in AMINO_ACIDS else "A" for c in seq.upper())
    if not clean:
        return float("nan")
    try:
        return float(ProteinAnalysis(clean).isoelectric_point())
    except Exception:
        return float("nan")


def molecular_weight(seq: str) -> float:
    """Compute molecular weight; unknown AAs replaced with Ala."""
    clean = "".join(c if c in AMINO_ACIDS else "A" for c in seq.upper())
    if not clean:
        return float("nan")
    try:
        return float(ProteinAnalysis(clean).molecular_weight())
    except Exception:
        return float("nan")


def chromophore_triplet(aligned_seq: str) -> str:
    """Extract chromophore triplet from alignment columns 736-738."""
    triplet = "".join(aligned_seq[c] for c in CHROM_COLS)
    # All-gap triplet -> unknown
    if triplet == "---":
        return "unknown"
    return triplet


def _normalize_cat(value: object, cats: list[str]) -> str:
    """Return category string or 'other' if missing/unknown."""
    if value is None:
        return "other"
    s = str(value).strip().lower()
    if s in ("nan", "none", ""):
        return "other"
    return s if s in cats else "other"


def onehot_oligo(value: object) -> list[float]:
    """One-hot encode oligomerization (m/d/t/wd/other)."""
    v = _normalize_cat(value, OLIGO_CATS)
    return [1.0 if v == cat else 0.0 for cat in OLIGO_CATS]


def onehot_switch(value: object) -> list[float]:
    """One-hot encode switch_type (b/ps/pa/pc/other)."""
    v = _normalize_cat(value, SWITCH_CATS)
    return [1.0 if v == cat else 0.0 for cat in SWITCH_CATS]


def main() -> None:
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # Load all splits
    splits = {}
    for split in ("train", "val", "test"):
        path = os.path.join(SPLITS_DIR, f"{split}.csv")
        splits[split] = pd.read_csv(path)
    df_all = pd.concat(splits.values(), ignore_index=True)
    print(f"Total proteins across splits: {len(df_all)}")

    # Load alignment
    print(f"Loading alignment from {ALIGNED_FASTA} ...")
    alignment = load_alignment(ALIGNED_FASTA)
    print(f"Alignment records: {len(alignment)}, width: {len(next(iter(alignment.values())))}")

    # Discover all chromophore triplet categories from full dataset
    triplet_counts: dict[str, int] = {}
    for _, row in df_all.iterrows():
        name = str(row["name"])
        if name in alignment:
            t = chromophore_triplet(alignment[name])
            triplet_counts[t] = triplet_counts.get(t, 0) + 1
    # Sort by frequency descending for reproducibility
    triplet_cats = sorted(triplet_counts, key=lambda x: -triplet_counts[x])
    print(f"Unique chromophore triplets: {len(triplet_cats)} (+ unknown if any)")

    # Build feature rows
    rows: list[dict[str, object]] = []
    missing_aln = 0

    for _, row in df_all.iterrows():
        name = str(row["name"])
        seq = str(row["seq"])

        feat: dict[str, object] = {"name": name}

        # 20-dim AA composition
        comp = aa_composition(seq)
        for aa, val in zip(AMINO_ACIDS, comp):
            feat[f"aa_{aa}"] = val

        # Sequence length
        feat["seq_len"] = float(len(seq.replace("-", "")))

        # Isoelectric point
        feat["isoelectric_point"] = isoelectric_point(seq)

        # Molecular weight
        feat["molecular_weight"] = molecular_weight(seq)

        # Chromophore triplet one-hot
        if name in alignment:
            t = chromophore_triplet(alignment[name])
        else:
            t = "unknown"
            missing_aln += 1

        for cat in triplet_cats:
            feat[f"triplet_{cat}"] = 1.0 if t == cat else 0.0

        # Oligomerization one-hot
        oligo_vec = onehot_oligo(row.get("oligomerization", None))
        for cat, val in zip(OLIGO_CATS, oligo_vec):
            feat[f"oligo_{cat}"] = val

        # Switch type one-hot
        switch_vec = onehot_switch(row.get("switch_type", None))
        for cat, val in zip(SWITCH_CATS, switch_vec):
            feat[f"switch_{cat}"] = val

        rows.append(feat)

    if missing_aln:
        print(f"WARNING: {missing_aln} proteins not found in alignment; triplet set to unknown")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False)

    feature_cols = [c for c in df_out.columns if c != "name"]
    print(f"\nFeature count: {len(feature_cols)}")
    print(f"Output shape:  {df_out.shape}")
    print(f"Written to:    {OUTPUT_CSV}")

    # Summary of feature groups
    print(f"\nFeature breakdown:")
    print(f"  AA composition:       20")
    print(f"  seq_len:               1")
    print(f"  isoelectric_point:     1")
    print(f"  molecular_weight:      1")
    print(f"  chromophore triplet:  {len(triplet_cats)}")
    print(f"  oligomerization:       5")
    print(f"  switch_type:           5")
    print(f"  Total:                {len(feature_cols)}")


if __name__ == "__main__":
    main()
