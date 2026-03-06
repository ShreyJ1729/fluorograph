"""
Script 03: ESM-2 embeddings via Modal GPU.

Generates 1280-dim mean-pooled ESM-2 (esm2_t33_650M_UR50D) embeddings for all
898 fluorescent proteins. Uses Modal's A10G GPU for fast batch processing.

Usage:
    python scripts/03_esm2_embed.py          # Modal GPU (A10G, ~5-10 min, ~$0.50)
    python scripts/03_esm2_embed.py --local  # CPU fallback (~1-3 hrs, no Modal needed)

Output: features/esm2_embeddings.npz
  Keys:
    - names:      object array  (n_proteins,)        protein names
    - embeddings: float32 array (n_proteins, 1280)   ESM-2 mean-pooled layer-33 reps

Re-runs are incremental: already-cached proteins are skipped.
"""

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
SPLITS_DIR = os.path.join(_PROJECT_ROOT, "data", "splits")
FEATURES_DIR = os.path.join(_PROJECT_ROOT, "features")
OUTPUT_NPZ = os.path.join(FEATURES_DIR, "esm2_embeddings.npz")

EMBED_DIM = 1280
REPR_LAYER = 33
BATCH_SIZE_GPU = 32
BATCH_SIZE_CPU = 4


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def load_all_proteins() -> dict[str, str]:
    """Load all proteins from train/val/test splits. Returns {name: seq}."""
    proteins: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = os.path.join(SPLITS_DIR, f"{split}.csv")
        if not os.path.exists(path):
            print(f"ERROR: Split file not found: {path}")
            print("Run scripts/00_filter_split.py first.")
            sys.exit(1)
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            name = str(row["name"])
            seq = str(row["seq"])
            if name not in proteins:
                proteins[name] = seq
    return proteins


def load_cached() -> dict[str, np.ndarray]:
    """Load previously computed embeddings from npz cache. Returns {name: emb}."""
    if not os.path.exists(OUTPUT_NPZ):
        return {}
    data = np.load(OUTPUT_NPZ, allow_pickle=True)
    names: np.ndarray = data["names"]
    embeddings: np.ndarray = data["embeddings"]
    return {str(n): embeddings[i] for i, n in enumerate(names)}


def save_embeddings(cached: dict[str, np.ndarray]) -> None:
    """Persist all embeddings to npz (overwrites existing file)."""
    os.makedirs(FEATURES_DIR, exist_ok=True)
    names_arr = np.array(list(cached.keys()), dtype=object)
    embeddings_arr = np.array(list(cached.values()), dtype=np.float32)
    np.savez(OUTPUT_NPZ, names=names_arr, embeddings=embeddings_arr)


# ---------------------------------------------------------------------------
# Local CPU embedding (--local flag)
# ---------------------------------------------------------------------------


def embed_local(
    sequences: list[tuple[str, str]],
    batch_size: int = BATCH_SIZE_CPU,
) -> dict[str, np.ndarray]:
    """Compute ESM-2 embeddings locally on CPU (slow fallback)."""
    try:
        import esm  # type: ignore[import-untyped]
        import torch  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: fair-esm and torch are required for --local mode.")
        print("Install with: uv pip install fair-esm torch")
        sys.exit(1)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    results: dict[str, np.ndarray] = {}
    total = len(sequences)
    processed = 0

    for i in range(0, total, batch_size):
        batch = sequences[i : i + batch_size]
        _, _, tokens = batch_converter(batch)

        with torch.no_grad():
            out = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)

        representations: Any = out["representations"][REPR_LAYER]

        for j, (name, seq) in enumerate(batch):
            seq_len = len(seq)
            # Mean-pool over sequence positions (skip BOS token at index 0)
            emb: np.ndarray = representations[j, 1 : seq_len + 1].mean(0).numpy()
            results[name] = emb.astype(np.float32)

        processed += len(batch)
        print(f"  Progress: {processed}/{total} proteins processed")

    return results


# ---------------------------------------------------------------------------
# Modal GPU embedding (default)
# ---------------------------------------------------------------------------


def embed_modal(sequences: list[tuple[str, str]]) -> dict[str, np.ndarray]:
    """Compute ESM-2 embeddings on Modal A10G GPU."""
    try:
        import modal  # type: ignore[import-untyped]
    except ImportError:
        print("ERROR: modal not installed. Install with: uv pip install modal")
        sys.exit(1)

    # Build the Modal app and remote image
    app: Any = modal.App("fluorograph-esm2")
    image: Any = modal.Image.debian_slim(python_version="3.10").pip_install(
        "fair-esm", "torch", "numpy"
    )

    @app.function(gpu="A10G", image=image, timeout=600, serialized=True)  # type: ignore[misc]
    def _compute_on_gpu(seqs: list[tuple[str, str]]) -> dict[str, list[float]]:
        """Inner function executed on Modal's A10G GPU."""
        import esm  # noqa: PLC0415  # type: ignore[import-untyped]
        import torch  # noqa: PLC0415  # type: ignore[import-untyped]

        _model, _alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        _model = _model.cuda()
        _model.eval()
        _batch_converter = _alphabet.get_batch_converter()

        _batch_size = 32
        _repr_layer = 33
        _results: dict[str, list[float]] = {}
        _total = len(seqs)

        for _i in range(0, _total, _batch_size):
            _batch = seqs[_i : _i + _batch_size]
            _, _, _tokens = _batch_converter(_batch)
            _tokens = _tokens.cuda()

            with torch.no_grad():
                _out = _model(_tokens, repr_layers=[_repr_layer], return_contacts=False)

            _reps = _out["representations"][_repr_layer]

            for _j, (_name, _seq) in enumerate(_batch):
                _seq_len = len(_seq)
                _emb = _reps[_j, 1 : _seq_len + 1].mean(0).cpu().numpy()
                _results[_name] = _emb.tolist()

            _done = min(_i + _batch_size, _total)
            print(f"  Progress: {_done}/{_total} proteins processed")

        return _results

    with modal.enable_output():
        with app.run():
            raw: dict[str, list[float]] = _compute_on_gpu.remote(sequences)  # type: ignore[attr-defined]

    return {name: np.array(v, dtype=np.float32) for name, v in raw.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ESM-2 embeddings for fluorescent proteins"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=f"Run on CPU (slow ~1-3h fallback, batch_size={BATCH_SIZE_CPU}). "
        "Default: Modal A10G GPU (~5-10 min).",
    )
    args = parser.parse_args()

    # --- load proteins ---
    proteins = load_all_proteins()
    print(f"Total unique proteins: {len(proteins)}")

    # --- load cache ---
    cached = load_cached()
    print(f"Already cached:  {len(cached)} proteins")

    # --- determine what to compute ---
    todo: list[tuple[str, str]] = [
        (name, seq) for name, seq in proteins.items() if name not in cached
    ]
    print(f"To compute:      {len(todo)} proteins")

    if not todo:
        print("All proteins already embedded — nothing to do.")
        print(f"Output: {OUTPUT_NPZ}")
        return

    # --- compute ---
    if args.local:
        print(f"\nRunning locally on CPU (batch_size={BATCH_SIZE_CPU}) ...")
        print("NOTE: ~1-3 hours for 898 proteins on CPU.")
        new_embeddings = embed_local(todo)
    else:
        print(f"\nRunning on Modal GPU (A10G, batch_size={BATCH_SIZE_GPU}) ...")
        print("Estimated cost: ~$0.50 | Estimated time: ~5-10 min")
        new_embeddings = embed_modal(todo)

    # --- merge and persist ---
    cached.update(new_embeddings)
    save_embeddings(cached)

    print(f"\nEmbeddings saved:  {len(cached)} proteins")
    print(f"Embedding dim:     {EMBED_DIM}")
    print(f"Output:            {OUTPUT_NPZ}")
    print(
        f"Array shapes:      names ({len(cached)},)  "
        f"embeddings ({len(cached)}, {EMBED_DIM})"
    )


if __name__ == "__main__":
    main()
