"""
Script 02: One-hot encode alignment per split.

Loads the MAFFT-aligned FASTA and encodes each protein as a flat one-hot
vector (alignment_width * 21). Writes per-split .npz feature files.

Alphabet: 20 standard amino acids + '-' gap = 21 characters.
Unknown characters are mapped to gap ('-').

Output: features/onehot_train.npz, features/onehot_val.npz, features/onehot_test.npz
Each npz has keys: X (float32), names (str array), em_max (float64), ex_max (float64)
"""

import os
import sys

import numpy as np
import pandas as pd
from Bio import SeqIO

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
ALIGNED_DIR = os.path.join(DATA_DIR, "aligned")
ALIGNED_FASTA = os.path.join(ALIGNED_DIR, "all.fasta")
FEATURES_DIR = os.path.join(_PROJECT_ROOT, "features")

# Alphabet: 20 standard amino acids in alphabetical order + gap
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
GAP_CHAR = "-"
ALPHABET = AMINO_ACIDS + [GAP_CHAR]
ALPHABET_SIZE = len(ALPHABET)  # 21
CHAR_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(ALPHABET)}


def load_alignment(aligned_fasta: str) -> dict[str, str]:
    """Parse aligned FASTA into {name: aligned_sequence} dict."""
    if not os.path.exists(aligned_fasta):
        print(f"ERROR: Aligned FASTA not found at {aligned_fasta}")
        print("Run scripts/01_mafft_align.py first.")
        sys.exit(1)
    records = list(SeqIO.parse(aligned_fasta, "fasta"))
    if not records:
        print(f"ERROR: No records found in {aligned_fasta}")
        sys.exit(1)
    # Use rec.description (full header) as key: FASTA truncates rec.id at whitespace,
    # so names with spaces (e.g. "mKate M41G S158C") require description for full name.
    return {rec.description: str(rec.seq).upper() for rec in records}


def onehot_encode_sequence(seq: str) -> np.ndarray:
    """
    Encode a single aligned sequence as a flat one-hot vector.

    Args:
        seq: Aligned sequence string of length alignment_width.

    Returns:
        Float32 array of shape (alignment_width * ALPHABET_SIZE,).
    """
    alignment_width = len(seq)
    matrix = np.zeros((alignment_width, ALPHABET_SIZE), dtype=np.float32)
    for pos, char in enumerate(seq):
        idx = CHAR_TO_IDX.get(char, CHAR_TO_IDX[GAP_CHAR])  # unknown -> gap
        matrix[pos, idx] = 1.0
    return matrix.reshape(-1)


def encode_split(
    split_name: str,
    alignment: dict[str, str],
    alignment_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode all proteins in a split.

    Returns:
        X: float32 array (n_proteins, alignment_width * ALPHABET_SIZE)
        names: str array (n_proteins,)
        em_max: float64 array (n_proteins,)
        ex_max: float64 array (n_proteins,)
    """
    csv_path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    df = pd.read_csv(csv_path)

    X_list: list[np.ndarray] = []
    names_list: list[str] = []
    em_list: list[float] = []
    ex_list: list[float] = []
    missing: list[str] = []

    for _, row in df.iterrows():
        name = str(row["name"])
        if name not in alignment:
            missing.append(name)
            continue
        seq = alignment[name]
        X_list.append(onehot_encode_sequence(seq))
        names_list.append(name)
        em_list.append(float(row["em_max"]))
        ex_list.append(float(row["ex_max"]))

    if missing:
        print(f"  WARNING: {len(missing)} proteins in {split_name} not found in alignment:")
        for m in missing[:5]:
            print(f"    {m}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    X = np.array(X_list, dtype=np.float32)
    names = np.array(names_list, dtype=object)
    em_max = np.array(em_list, dtype=np.float64)
    ex_max = np.array(ex_list, dtype=np.float64)
    return X, names, em_max, ex_max


def main() -> None:
    os.makedirs(FEATURES_DIR, exist_ok=True)

    print(f"Loading alignment from {ALIGNED_FASTA} ...")
    alignment = load_alignment(ALIGNED_FASTA)
    alignment_width = len(next(iter(alignment.values())))
    feature_dim = alignment_width * ALPHABET_SIZE

    print(f"Alignment width: {alignment_width} columns")
    print(f"Alphabet size:   {ALPHABET_SIZE} (20 AAs + gap)")
    print(f"Feature vector:  {alignment_width} x {ALPHABET_SIZE} = {feature_dim} dims")

    for split in ("train", "val", "test"):
        print(f"\nEncoding {split} split...")
        X, names, em_max, ex_max = encode_split(split, alignment, alignment_width)
        out_path = os.path.join(FEATURES_DIR, f"onehot_{split}.npz")
        np.savez(out_path, X=X, names=names, em_max=em_max, ex_max=ex_max)
        print(f"  X shape:    {X.shape}")
        print(f"  em_max:     {em_max.shape}, ex_max: {ex_max.shape}")
        print(f"  Written to: {out_path}")

    print("\nOne-hot encoding complete.")
    print(f"Feature files written to {FEATURES_DIR}/")


if __name__ == "__main__":
    main()
