# FluoroGraph Phase 1: Local Spectral Prediction Pipeline

**Version:** 1.0
**Author:** ShreyJ
**Date:** March 2026
**Status:** Active working document

---

## 1. Mission

Predict fluorescent protein excitation and emission maxima from amino acid sequence, running locally with Modal GPU for ESM-2 embedding computation. Phase 1 establishes a validated forward model with proper baselines, interpretability, and evaluation discipline. No agent loop. Sequential scripts.

---

## 2. Goals

| ID  | Goal                                                                                                   | Gate                      |
| --- | ------------------------------------------------------------------------------------------------------ | ------------------------- |
| G0  | Replicate FPredX (Tam & Zhang 2021): MAFFT alignment + one-hot + XGBoost. R² within 0.05 of published. | Gates all downstream work |
| G1  | Predict ex/em maxima from sequence, R² ≥ 0.85 (emission), MAE ≤ 15 nm                                  | G0 must pass first        |
| G2  | Interpretability: top predictive positions map to known chromophore-tuning residues (≥ 50% overlap)    | G0 must pass first        |
| G3  | Proper train/val/test splits, baselines, no data leakage, reproducible metrics                         | Runs alongside G1/G2      |

---

## 3. Non-Goals (Phase 1)

- Modal used ONLY for ESM-2 GPU embedding (script 03); all other scripts run local
- No ESMFold (skip — sequence-only, no structural contacts in Phase 1)
- No AlphaFold (Phase 2)
- No inverse design (Phase 2)
- No fine-tuning of ESM-2
- No wet-lab or QM/MM validation
- No real-time serving or API

---

## 4. Directory Structure

```
fluorograph/
├── data/
│   ├── fpbase_proteins.csv          # raw FPbase export (1110 rows)
│   ├── sequences.json               # FPbase sequences + state metadata (1040 entries)
│   ├── fpbase_merged.csv            # built by merge_dataset.py (899 complete rows)
│   ├── splits/
│   │   ├── train.csv                # ~629 proteins
│   │   ├── val.csv                  # ~135 proteins
│   │   └── test.csv                 # ~135 proteins
│   └── aligned/
│       └── all.fasta                # MAFFT output (all 898 filtered proteins)
├── features/
│   ├── onehot_train.npz             # keys: X, names, em_max, ex_max
│   ├── onehot_val.npz
│   ├── onehot_test.npz
│   ├── esm2_embeddings.npz          # {name: (1280,) array} for all 898
│   └── classical_features.csv       # AA composition, MW, pI, triplet, oligomerization, switch_type
├── models/
│   ├── xgboost_em.json
│   ├── xgboost_ex.json
│   └── xgboost_params.json          # best Optuna params per target
├── results/
│   ├── metrics.json                 # R², MAE, RMSE by split and model
│   ├── feature_importance.csv
│   └── figures/
│       ├── pred_vs_actual_em.png
│       ├── pred_vs_actual_ex.png
│       ├── feature_importance_top30.png
│       └── emission_bins_split.png
├── scripts/
│   ├── 00_filter_split.py           # filter merged.csv (seq 100-1200), stratified 70/15/15
│   ├── 01_mafft_align.py            # MAFFT on all 898, write aligned/all.fasta
│   ├── 02_onehot_encode.py          # one-hot from alignment, per split
│   ├── 03_esm2_embed.py             # ESM-2 650M on Modal GPU, cache npz locally
│   ├── 04_classical_features.py     # AA composition, pI, MW, triplet identity
│   ├── 05_train_xgboost.py          # Optuna 100 trials, 5-fold CV, eval on val/test
│   ├── 06_esm2_knn.py               # ChromaDB flat k-NN baseline (k=3,5,7,10)
│   ├── 07_interpretability.py       # feature importance → alignment positions
│   └── 08_report.py                 # compile metrics.json → phase1_report.md
├── merge_dataset.py                 # one-time: merges CSV + JSON → fpbase_merged.csv (DONE)
├── fluorograph_agent_prd.md         # original full PRD (all phases)
├── fluorograph_phase1_prd.md        # this document
└── requirements.txt
```

---

## 5. Data

### Source Files (present)

- `fpbase_proteins.csv` — 1,110 rows: Name, State, Ex max (nm), Em max (nm), spectral metadata
- `sequences.json` — 1,040 entries: uuid, name, seq, states[], doi
- `fpbase_merged.csv` — already built by `merge_dataset.py`

### Merged Dataset (already built)

- 1,110 rows total, 899 complete (seq + ex_max + em_max), 898 after seq-length filter (≥100 aa)
- Columns: name, uuid, slug, state, seq, seq_len, ex_max, em_max, stokes_shift, ext_coeff, qy, brightness, pka, oligomerization, maturation, lifetime, mw_kda, year, switch_type, aliases, doi, pdb, has_seq, has_ex_em, complete

### Splits (built by script 00)

- Filter: `complete == 1` AND `seq_len >= 100` → 898 proteins
- Strategy: stratified by emission bin, seed=42
- Sizes: train=629, val=135, test=135 (approximately 70/15/15 — rounding depends on bin sizes)

| Bin      | Range      | ~Count (of 898) |
| -------- | ---------- | --------------- |
| blue     | < 460 nm   | ~35             |
| cyan     | 460-499 nm | ~98             |
| green    | 500-529 nm | ~410            |
| yellow   | 530-569 nm | ~36             |
| orange   | 570-599 nm | ~98             |
| red      | 600-649 nm | ~99             |
| far-red  | 650-699 nm | ~38             |
| infrared | ≥ 700 nm   | ~24             |

**Rule:** test set is locked after creation. Never examine test metrics during model development. Val is the only feedback during tuning.

---

## 6. Pipeline Scripts (Execution Order)

### Script 00: Filter + Split

`scripts/00_filter_split.py`

- Load `fpbase_merged.csv`, filter: `complete == 1` AND `seq_len >= 100`
- Assign `em_bin` column (8 bins above)
- `StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)` → test
- `StratifiedShuffleSplit(n_splits=1, test_size=0.176, random_state=42)` → val from remaining (0.176 ≈ 15/85)
- Write: `data/splits/{train,val,test}.csv`
- Print bin distribution per split — verify each bin appears in train

### Script 01: MAFFT Alignment

`scripts/01_mafft_align.py`

- Combine all 898 sequences into `data/aligned/input.fasta`
- Run: `mafft --auto --thread -1 data/aligned/input.fasta > data/aligned/all.fasta`
- Parse aligned FASTA, print alignment width and gap fraction
- Requires MAFFT: `brew install mafft` or `conda install -c bioconda mafft`

### Script 02: One-Hot Encoding

`scripts/02_onehot_encode.py`

- Load `data/aligned/all.fasta`, look up split membership per protein
- Alphabet: 20 amino acids + `-` gap = 21 chars
- Per protein: flatten (alignment_width × 21) → 1D vector
- Write: `features/onehot_{train,val,test}.npz` with keys `X`, `names`, `em_max`, `ex_max`

### Script 03: ESM-2 Embeddings (Modal GPU)

`scripts/03_esm2_embed.py`

- Model: `esm2_t33_650M_UR50D` via `fair-esm`
- **Runs on Modal GPU** (A10G or T4) for speed — ~5-10 min vs 1-3 hrs on CPU
- Modal function takes list of (name, seq) pairs, returns dict of embeddings
- Process all 898 proteins, batch size 32 on GPU
- Extract mean-pooled embedding: (1280,) per protein
- Cache to `features/esm2_embeddings.npz` locally — skip proteins already present on re-run
- Fallback: if Modal unavailable, set `--local` flag to run on CPU (batch size 4-8, ~1-3 hrs)

### Script 04: Classical Features

`scripts/04_classical_features.py`

Sequence-derived features:

- 20-dim AA composition (fraction of each amino acid)
- Sequence length
- Isoelectric point (`Bio.SeqUtils.ProtParam`)
- Molecular weight
- Chromophore triplet identity (residues at alignment positions ~65-67, mapped back to raw index; one-hot encode the triplet)

FPbase metadata features:

- Oligomerization (one-hot: m / d / t / wd / other)
- Switch type (one-hot: b / ps / pa / pc / other)

Total: ~30 features. Write: `features/classical_features.csv`

### Script 05: Train XGBoost

`scripts/05_train_xgboost.py`

Two modes — run G0 first, verify it passes before running G1:

**G0 — FPredX replication (one-hot only):**

- Features: one-hot matrix (train set)
- Optuna: 100 trials, minimize 5-fold CV MAE on train
- Evaluate on val, then test
- Report: R², MAE, RMSE for em_max and ex_max
- Save: `models/xgboost_em.json`, `models/xgboost_ex.json`

**G1 — Enhanced (ablation over feature combinations):**
XGBoost handles heterogeneous concatenated features natively — no normalization needed between one-hot, ESM-2, and classical. Train separate models for each combination to identify what helps:

| Model | Features                                      | Dim (approx)                      |
| ----- | --------------------------------------------- | --------------------------------- |
| G1a   | One-hot only (= G0)                           | alignment_width × 21              |
| G1b   | ESM-2 only                                    | 1280                              |
| G1c   | Classical only                                | ~30                               |
| G1d   | One-hot + classical                           | alignment_width × 21 + ~30        |
| G1e   | ESM-2 + classical                             | 1280 + ~30                        |
| G1f   | **One-hot + ESM-2 + classical** (full concat) | alignment_width × 21 + 1280 + ~30 |

Same Optuna protocol per model. Compare val R² across all. Report best on test.

**Why concatenation works here:** XGBoost is a tree-based model — it selects features by information gain regardless of scale or type. Concatenating one-hot (sparse, positional) with ESM-2 (dense, global) with classical (low-dim, metadata) gives the model the richest signal to split on. No embedding alignment or projection needed.

Sanity check: `|test_R2 - val_R2| > 0.05` → flag and investigate leakage.

### Script 06: ESM-2 k-NN Baseline

`scripts/06_esm2_knn.py`

- ChromaDB in-memory collection (no server)
- Index train ESM-2 embeddings with {em_max, ex_max, name} metadata
- Query each val + test protein, k = 3, 5, 7, 10
- Prediction: distance-weighted average of neighbor labels
- Report: R², MAE, RMSE per k for em and ex
- Best k selected on val, reported on test

### Script 07: Interpretability

`scripts/07_interpretability.py`

- Load G0 XGBoost em_max model (one-hot features — directly position-interpretable)
- Feature importance by gain, map index → (alignment_position, amino_acid)
- Top 30 by total gain
- Cross-reference with known EGFP-like tuning positions: 65-67 (chromophore triad), Thr203, His148, Glu222, Arg96, Phe64, Asn146, Met163
- Compute: fraction of top 20 overlapping known positions
- Per-color: separate XGBoost on green-only and red-only subsets, compare top-10 positions
- Write: `results/feature_importance.csv`, `results/figures/feature_importance_top30.png`

### Script 08: Report

`scripts/08_report.py`

- Load `results/metrics.json`
- Write `results/phase1_report.md`: dataset summary, split stats, G0 vs FPredX table, G1 model comparison, k-NN baseline, interpretability, per-color R², figures referenced

---

## 7. Dependencies

```
# requirements.txt
fair-esm>=2.0.0
torch>=2.0.0
xgboost>=2.0.0
optuna>=3.0.0
chromadb>=0.4.0
scikit-learn>=1.3.0
biopython>=1.81
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
modal>=0.64.0            # for ESM-2 GPU inference (script 03)
```

External binary: `mafft` (check: `which mafft`).
Modal: account with GPU access. Only used for script 03 (ESM-2 embeddings).

---

## 8. Experiments

### Experiment 0: FPredX Replication (G0 — Gate)

**Hypothesis:** MAFFT + one-hot + XGBoost matches Tam & Zhang 2021 within ±0.05 R².

**Steps:** Run scripts 00 → 01 → 02 → 05 (G0 mode).

**Success:** R² within 0.05 of published for both em and ex.
**Failure:** STOP. Debug in order: (a) dataset size vs paper, (b) MAFFT flags (`--auto` vs `--globalpair`), (c) hyperparameter space, (d) data leakage check. Do not proceed until resolved.

---

### Experiment 1: Enhanced Prediction (G1)

**Ablation across 6 feature combinations (script 05) + k-NN baseline (script 06):**

- G1a: One-hot only (= G0, already done)
- G1b: ESM-2 only (XGBoost on 1280-dim)
- G1c: Classical only (~30-dim)
- G1d: One-hot + classical
- G1e: ESM-2 + classical
- G1f: **One-hot + ESM-2 + classical** (full concatenation)
- G1-kNN: ESM-2 k-NN (script 06, distance-weighted, best k from val)

All use same Optuna protocol. Concatenation is straightforward — XGBoost trees select features by gain regardless of scale/type.

**Success:** Best model R² ≥ 0.85 on test em_max.
**Failure:** If 0.80–0.84, report honestly — dataset size and green-class dominance cap performance. Proceed to G2/G3.

---

### Experiment 2: Flat k-NN Baseline

**Purpose:** Establish zero-learning retrieval ceiling. If k-NN > XGBoost, XGBoost is likely overfitting.

**Steps:** Script 06.

---

### Experiment 3: Interpretability (G2 + G3)

**Steps:** Script 07 on G0 model, per-color analysis.

**Success:** ≥ 50% of top 20 positions match known biology.

---

## 9. Success Metrics

| Goal | Metric                              | Target      | Hard Fail              |
| ---- | ----------------------------------- | ----------- | ---------------------- |
| G0   | em_max R² vs FPredX                 | Within 0.05 | > 0.05: stop           |
| G0   | ex_max R² vs FPredX                 | Within 0.05 | > 0.05: stop           |
| G1   | em_max R² (test)                    | ≥ 0.85      | < 0.75: investigate    |
| G1   | em_max MAE (test)                   | ≤ 15 nm     | > 25 nm: investigate   |
| G1   | ex_max R² (test)                    | ≥ 0.80      | —                      |
| G2   | Top-20 overlap with known positions | ≥ 50%       | < 25%: suspect overfit |
| G3   | Test never used for tuning          | Confirmed   | Any leakage: invalid   |
| G3   | Stratified splits, seed=42 logged   | Confirmed   | —                      |

---

## 10. Compute Budget

| Step                                                | Time           | Where     |
| --------------------------------------------------- | -------------- | --------- |
| Script 00 (filter, split)                           | < 1 min        | Local     |
| Script 01 (MAFFT alignment, 898 seqs)               | 5-15 min       | Local     |
| Script 02 (one-hot encoding)                        | < 1 min        | Local     |
| Script 03 (ESM-2 embeddings, 898 proteins)          | 5-10 min       | Modal GPU |
| Script 04 (classical features)                      | < 2 min        | Local     |
| Script 05 (XGBoost + Optuna, 100 trials × 6 models) | 30-90 min      | Local     |
| Script 06 (k-NN eval)                               | < 5 min        | Local     |
| Script 07 (interpretability)                        | < 5 min        | Local     |
| Script 08 (report)                                  | < 1 min        | Local     |
| **Total**                                           | **~1-2 hours** |           |

ESM-2 embeddings are cached locally — re-runs skip already-processed proteins.
Modal cost for script 03: ~$0.50 (A10G, 10 min).

---

## 11. Execution Protocol

```bash
# One-time: dataset already merged (merge_dataset.py already run)
python scripts/00_filter_split.py
python scripts/01_mafft_align.py        # needs mafft binary
python scripts/02_onehot_encode.py
python scripts/03_esm2_embed.py         # Modal GPU — fast, cached locally
python scripts/04_classical_features.py
python scripts/05_train_xgboost.py      # G0 first — verify before G1
python scripts/06_esm2_knn.py
python scripts/07_interpretability.py
python scripts/08_report.py
```

**After script 05 G0:** manually check G0 gate before running G1 ablation models.

---

## 12. Risks and Mitigations

| Risk                            | Mitigation                                                                                         |
| ------------------------------- | -------------------------------------------------------------------------------------------------- |
| MAFFT not installed             | `brew install mafft`. Script exits with install instructions.                                      |
| ESM-2 OOM on Modal              | Reduce batch to 16. If Modal unavailable, run local CPU with `--local` flag (batch 4-8, ~1-3 hrs). |
| G0 diverges from paper          | Debug: dataset size, MAFFT flags, whether paper uses CV vs held-out test.                          |
| Green-class dominance (411/898) | Stratified splits. Per-color breakdown reveals underperforming bins.                               |
| ESM-2 doesn't beat one-hot      | Legitimate finding. Report honestly. Interpretability still complete.                              |

---

## 13. Deliverables

1. `fpbase_merged.csv` — merged dataset (899 rows, 898 usable) ✅ **already built**
2. `data/splits/{train,val,test}.csv` — stratified splits, seed=42
3. `features/esm2_embeddings.npz` — ESM-2 (1280-dim) for all 898
4. `models/xgboost_em.json`, `models/xgboost_ex.json`
5. `results/metrics.json` — all R², MAE, RMSE across models and splits
6. `results/feature_importance.csv` — XGBoost position importance
7. `results/figures/` — prediction and interpretability plots
8. `results/phase1_report.md` — self-contained summary
