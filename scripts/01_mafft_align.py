"""
Script 01: MAFFT multiple sequence alignment.

Combines all 898 protein sequences from train/val/test splits into a single
FASTA file and runs MAFFT alignment. Prints alignment width and mean gap fraction.

Output: data/aligned/input.fasta, data/aligned/all.fasta
"""

import os
import shutil
import subprocess
import sys

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
ALIGNED_DIR = os.path.join(DATA_DIR, "aligned")
INPUT_FASTA = os.path.join(ALIGNED_DIR, "input.fasta")
ALIGNED_FASTA = os.path.join(ALIGNED_DIR, "all.fasta")


def check_mafft() -> str:
    """Return path to mafft binary, or exit with instructions if not found."""
    mafft_path = shutil.which("mafft")
    if mafft_path is None:
        print("ERROR: mafft binary not found in PATH.")
        print()
        print("Install mafft with one of:")
        print("  brew install mafft          # macOS with Homebrew")
        print("  conda install -c bioconda mafft  # Conda")
        print("  sudo apt-get install mafft  # Debian/Ubuntu")
        print()
        print("Then re-run this script.")
        sys.exit(1)
    return mafft_path


def load_all_sequences() -> list[tuple[str, str]]:
    """Load sequences from all split CSVs, deduplicating by name."""
    seen: set[str] = set()
    records: list[tuple[str, str]] = []
    for split in ("train", "val", "test"):
        csv_path = os.path.join(SPLITS_DIR, f"{split}.csv")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            name = str(row["name"])
            seq = str(row["seq"])
            if name not in seen and seq and seq.lower() != "nan":
                seen.add(name)
                records.append((name, seq))
    return records


def write_fasta(records: list[tuple[str, str]], path: str) -> None:
    """Write list of (name, seq) pairs to a FASTA file."""
    bio_records = [
        SeqRecord(Seq(seq), id=name, description="")
        for name, seq in records
    ]
    with open(path, "w") as f:
        SeqIO.write(bio_records, f, "fasta")


def run_mafft(mafft_path: str, input_fasta: str, output_fasta: str) -> None:
    """Run MAFFT alignment."""
    cmd = [mafft_path, "--auto", "--thread", "-1", input_fasta]
    print(f"Running: {' '.join(cmd)}")
    with open(output_fasta, "w") as out_f:
        result = subprocess.run(
            cmd,
            stdout=out_f,
            stderr=subprocess.PIPE,
            text=True,
        )
    if result.returncode != 0:
        print(f"MAFFT failed with return code {result.returncode}")
        print(result.stderr[-2000:] if result.stderr else "")
        sys.exit(1)
    print("MAFFT alignment complete.")


def analyze_alignment(aligned_fasta: str) -> None:
    """Parse aligned FASTA and print alignment width and mean gap fraction."""
    records = list(SeqIO.parse(aligned_fasta, "fasta"))
    if not records:
        print("ERROR: No records found in aligned FASTA.")
        sys.exit(1)

    alignment_width = len(records[0].seq)
    total_chars = 0
    total_gaps = 0
    for rec in records:
        seq_str = str(rec.seq)
        total_chars += len(seq_str)
        total_gaps += seq_str.count("-")

    mean_gap_fraction = total_gaps / total_chars if total_chars > 0 else 0.0

    print(f"\nAlignment statistics:")
    print(f"  Sequences aligned: {len(records)}")
    print(f"  Alignment width:   {alignment_width} columns")
    print(f"  Mean gap fraction: {mean_gap_fraction:.4f}")


def main() -> None:
    mafft_path = check_mafft()
    os.makedirs(ALIGNED_DIR, exist_ok=True)

    print("Loading sequences from train/val/test splits...")
    records = load_all_sequences()
    print(f"Loaded {len(records)} unique sequences.")

    print(f"Writing input FASTA to {INPUT_FASTA}")
    write_fasta(records, INPUT_FASTA)

    run_mafft(mafft_path, INPUT_FASTA, ALIGNED_FASTA)
    analyze_alignment(ALIGNED_FASTA)

    print(f"\nAligned FASTA written to {ALIGNED_FASTA}")


if __name__ == "__main__":
    main()
