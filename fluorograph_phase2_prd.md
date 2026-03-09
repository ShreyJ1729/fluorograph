# FluoroGraph Phase 2: ESMFold Contacts + Inverse Design

**Version:** 1.0
**Author:** ShreyJ
**Date:** March 2026
**Status:** Planned — gate on Phase 1 completion

---

## 1. Gate (Phase 1 Required)

Do not start Phase 2 until ALL of the following exist and pass:

| Artifact                                                       | Check                                                       |
| -------------------------------------------------------------- | ----------------------------------------------------------- |
| `results/metrics.json`                                         | G0 R² within 0.05 of FPredX                                 |
| `results/metrics.json`                                         | G1 best test em_max R² ≥ 0.85 (or honest documented result) |
| `features/esm2_embeddings.npz`                                 | All 898 proteins embedded                                   |
| `results/feature_importance.csv`                               | Top-30 XGBoost positions extracted                          |
| ChromaDB index (in-memory, rebuilt from `esm2_embeddings.npz`) | k-NN queries functional                                     |
| `models/xgboost_em.json`                                       | Best Phase 1 emission model                                 |

---

## 2. Mission

Phase 2 has two tracks that run sequentially:

**Track A — ESMFold Contacts:** Add ESMFold 3D structural contacts to the forward model and test whether they improve Phase 1 emission prediction. Honest null result is acceptable.

**Track B — Inverse Design:** Use the best forward model (Phase 1 or A-improved) plus Phase 1 interpretability to propose mutation candidates toward four target emission wavelengths. Validate top candidates with AlphaFold.

---

## 3. Goals

| ID  | Goal                                                                                        | Gate                  |
| --- | ------------------------------------------------------------------------------------------- | --------------------- |
| G4a | Run ESMFold on all 898 proteins; cache 8Å contact maps                                      | None (starts Phase 2) |
| G4b | Determine if contact features improve Phase 1 best R² by ≥0.01. Honest null accepted.       | G4a                   |
| G5  | Propose ≥1 candidate sequence within ±15nm of target for ≥2 of 4 target wavelengths         | G4a, G4b              |
| G6  | AlphaFold validates ≥50% of submitted candidates (pLDDT > 70, barrel RMSD < 2Å vs scaffold) | G5                    |

---

## 4. Non-Goals (Phase 2)

- No RuVector — ChromaDB from Phase 1 is the retrieval backend
- No fine-tuning of ESM-2 or ESMFold
- No wet-lab validation — computational proposals only
- No multi-mutation combinatorics — single-position substitutions only
- No optimizing for brightness, photostability, or folding efficiency — emission wavelength only
- No generating new embeddings — reuse `esm2_embeddings.npz` from Phase 1

---

## 5. Directory Structure (Phase 2 additions)

```
fluorograph/
├── ... (Phase 1 structure preserved)
├── features/
│   ├── contact_maps/                   # ESMFold output per protein
│   │   └── {name}.npz                  # keys: contacts (N×N bool), plddt (N,)
│   └── contact_features.csv            # per-protein aggregated contact features
├── models/
│   ├── xgboost_em_contact.json         # contact-augmented model (if improvement found)
│   └── xgboost_ex_contact.json
├── results/
│   ├── inverse_design/
│   │   ├── scaffolds.csv               # 5 nearest per target, with em_max and distance
│   │   ├── candidates.csv              # all scored candidates (pre-AlphaFold filter)
│   │   └── alphafold/
│   │       ├── {candidate_name}_ranked_0.pdb
│   │       └── {candidate_name}_scores.json
│   ├── phase2_report.md
│   └── figures/
│       ├── contact_ablation.png         # R² with vs without contact features
│       └── candidate_predictions.png    # predicted vs target emission per candidate
└── scripts/
    ├── 09_esmfold_contacts.py
    ├── 10_contact_features.py
    ├── 11_scaffold_retrieval.py
    ├── 12_mutation_enumeration.py
    ├── 13_alphafold_validate.py
    └── 14_phase2_report.py
```

---

## 6. Data

Phase 2 does not add new proteins. All 898 proteins from Phase 1 are re-used.

**ESMFold outputs (new in Phase 2):**

- Contact map: binary N×N matrix, residue pairs with Cα–Cα distance ≤ 8Å
- pLDDT: per-residue confidence score (0–100)
- Stored as sparse npz: `contacts` (scipy sparse), `plddt` (array), `seq_len`
- Cache on first run; skip proteins already present on re-run

**Chromophore positions:** alignment positions 65–67 (triad), 148, 203, 222 from Phase 1 interpretability.

---

## 7. Inverse Design Targets

| Target | Wavelength | Color family | Rationale                                           |
| ------ | ---------- | ------------ | --------------------------------------------------- |
| T1     | 480 nm     | Blue         | Underrepresented, ~35 proteins in dataset           |
| T2     | 550 nm     | Green-yellow | Boundary between dominant green and yellow clusters |
| T3     | 620 nm     | Red          | Well-studied family (mCherry lineage)               |
| T4     | 700 nm     | Far-red      | Clinically relevant; sparse training data           |

---

## 8. Pipeline Scripts (Execution Order)

### Script 09: ESMFold Contacts (Modal GPU)

`scripts/09_esmfold_contacts.py`

- **Runs on:** Modal GPU (A10G)
- Load all 898 protein sequences from `fpbase_merged.csv`
- Skip proteins already in `features/contact_maps/`
- Batch size: 1 (ESMFold is memory-heavy; one at a time on A10G)
- For each protein:
  - Run ESMFold, extract predicted structure
  - Compute Cα–Cα distance matrix
  - Threshold at 8Å → binary contact matrix (scipy sparse)
  - Extract per-residue pLDDT
  - Save to `features/contact_maps/{name}.npz`
- Print: total processed, failed (log + skip), coverage
- **Time estimate:** ~1–2 hrs on A10G (~5–10 sec/protein × 898)

### Script 10: Contact Feature Extraction + Ablation

`scripts/10_contact_features.py`

- **Runs on:** local
- Load all contact maps from `features/contact_maps/`
- For each protein, compute:
  - `mean_plddt`: mean pLDDT across all residues
  - `plddt_chromophore`: mean pLDDT at chromophore triad positions (65–67)
  - `chromophore_contacts`: number of residues in contact with any triad position
  - `contact_density`: total edges / (N × (N-1) / 2)
  - `long_range_contacts`: contacts between residues ≥12 positions apart / total contacts
- Write: `features/contact_features.csv` (one row per protein, 5 features)
- **Ablation (retrain XGBoost):**
  - Reload Phase 1 best feature set (e.g., one-hot + ESM-2 + classical)
  - Append contact features (5 columns)
  - Same Optuna protocol (100 trials, 5-fold CV on train, eval on val)
  - Compare val R² vs Phase 1 best — report delta
  - If improvement ≥ 0.01: save as `models/xgboost_em_contact.json` — use for inverse design
  - If no improvement: log result honestly, proceed with Phase 1 model for inverse design
  - Plot: bar chart of R² across feature sets → `results/figures/contact_ablation.png`

### Script 11: Scaffold Retrieval

`scripts/11_scaffold_retrieval.py`

- **Runs on:** local
- Rebuild ChromaDB in-memory from `features/esm2_embeddings.npz` (train + val proteins only — do not include test set as scaffolds)
- For each of the 4 targets (480, 550, 620, 700 nm):
  - Filter ChromaDB metadata: proteins with `em_max` within ±20nm of target → candidate scaffolds
  - If fewer than 3 match, fall back to 5 nearest by ESM-2 cosine similarity with no em filter
  - Return top 5 by smallest `|em_max - target|`, breaking ties by ESM-2 distance to target centroid
- Write: `results/inverse_design/scaffolds.csv` with columns: `target_nm, scaffold_name, scaffold_em_max, delta_nm, esm2_distance`
- Print scaffold table to stdout for inspection

### Script 12: Mutation Enumeration + Scoring

`scripts/12_mutation_enumeration.py`

- **Runs on:** local
- Load:
  - `results/feature_importance.csv` (Phase 1 XGBoost G0 one-hot model, alignment positions ranked by gain)
  - `results/inverse_design/scaffolds.csv`
  - Best forward model: contact-augmented if G4b improved, otherwise Phase 1 best
  - `features/classical_features.csv`, `features/esm2_embeddings.npz`
- **Confidence intervals:** Train a quantile regression ensemble using XGBoost `reg:quantileerror`:
  - Three models per target: q=0.1, q=0.5, q=0.9
  - CI = (q10, q90) prediction interval
  - Train once on full train set, use for all candidate scoring

**Per scaffold:**

1. Identify top-15 emission-predictive alignment positions (from `feature_importance.csv`, filtered to positions within the scaffold's alignment span)
2. Map alignment positions → raw sequence positions using the MAFFT alignment
3. For each position × each of 20 amino acids (excluding wild-type):
   - Build mutant sequence string
   - Recompute one-hot (single position change from scaffold's one-hot vector)
   - Retrieve scaffold ESM-2 embedding (approximation: use scaffold embedding unchanged — full re-embedding would require Modal; flag in report)
   - Recompute classical features (AA composition changes by ±1/N; pI and MW via BioPython)
   - Concatenate features → predict with forward model (q50 = point estimate, q10/q90 = CI)
4. Filter candidates:
   - `|q50_predicted_em - target_nm| ≤ 20nm`
   - BLOSUM62 substitution score ≥ -2 (use `Bio.Align.substitution_matrices.load("BLOSUM62")`)
5. Rank by: `(|q50_predicted_em - target_nm|) + 0.1 × max(0, -BLOSUM62_score)` (lower = better)
6. Keep top 3 per scaffold (up to 5 scaffolds × 3 = 15 per target, 60 total max)

- Write: `results/inverse_design/candidates.csv` with columns:
  `target_nm, scaffold_name, scaffold_em_max, position_alignment, position_raw, wt_aa, mut_aa, blosum62, predicted_em_q10, predicted_em_q50, predicted_em_q90, delta_from_target, rank`
- Print top-3 candidates per target to stdout in human-readable format:
  ```
  Target: 620nm (red)
    Scaffold: mCherry (em=610nm)
      Rank 1: T203H → predicted 619nm (CI: 611–627nm), BLOSUM: 1
      Rank 2: I161S → predicted 624nm (CI: 614–634nm), BLOSUM: -1
      Rank 3: N146D → predicted 616nm (CI: 609–623nm), BLOSUM: 2
  ```

**Note on ESM-2 approximation:** Mutant ESM-2 embeddings are approximated as the scaffold embedding (single-residue changes have small but non-zero effect). Flag this explicitly in the report. Re-embedding each mutant would require 60 × Modal GPU calls; this is out of Phase 2 scope.

### Script 13: AlphaFold Validation (Modal GPU)

`scripts/13_alphafold_validate.py`

- **Runs on:** Modal GPU (A100, or A10G with reduced MSA)
- Load top 3 candidates per target from `results/inverse_design/candidates.csv`
  - Select the 3 with smallest `delta_from_target` AND `blosum62 ≥ 0`, across all scaffolds per target
  - Total: ≤ 12 candidates (3 per target × 4 targets)
- Use **ColabFold** on Modal (faster than full AF2 — uses MMseqs2 for MSA, not Jackhmmer)
- Per candidate:
  - Apply mutation to scaffold sequence (string substitution)
  - Run ColabFold → top-ranked PDB + confidence JSON
  - Extract: `mean_plddt`, `plddt_chromophore` (positions 65–67 in raw sequence)
  - Align predicted structure to scaffold structure (if scaffold PDB exists in FPbase; skip RMSD if no scaffold PDB) using BioPython MMCIF parser + Superimposer
  - Chromophore triad check: verify residues at 65–67 have backbone RMSD < 1Å vs scaffold
- Save: `results/inverse_design/alphafold/{candidate_name}_ranked_0.pdb` and `{candidate_name}_scores.json`
- Write validation table to `results/inverse_design/alphafold_validation.csv`:
  `candidate, target_nm, predicted_em_q50, mean_plddt, plddt_chromophore, backbone_rmsd_vs_scaffold, triad_rmsd, pass`
- Pass criteria: `mean_plddt > 70` AND (`backbone_rmsd_vs_scaffold < 2.0` OR no scaffold PDB)
- **Budget: hard cap at 20 AlphaFold predictions.** If ≤ 12 candidates pass step 1, run all. Print cost estimate before running (user confirmation prompt if > 15 predictions).

### Script 14: Phase 2 Report

`scripts/14_phase2_report.py`

- Load: Phase 1 metrics, contact ablation delta R², candidates table, AlphaFold validation table
- Write: `results/phase2_report.md`
  - **Track A:** ESMFold contact ablation — table of R² across feature sets, figure reference
  - **Track B:** Inverse design — scaffold selection rationale, candidate table, CI plot
  - **Track C:** AlphaFold validation — pass/fail per candidate, structural notes
  - **Honest assessment:** If contact features didn't help, say so. If fewer than 2 targets produced candidates, say so.

---

## 9. Success Metrics

| Goal | Metric                                       | Target                   | Honest Fail Handling                         |
| ---- | -------------------------------------------- | ------------------------ | -------------------------------------------- |
| G4a  | ESMFold runs complete                        | 898 proteins cached      | Log failures, report coverage                |
| G4b  | Contact feature R² delta (em_max, val)       | ≥ +0.01 OR null reported | null = valid result                          |
| G5   | Targets with ≥1 candidate within ±15nm       | ≥ 2 of 4                 | < 2: report proof-of-concept for best target |
| G5   | Candidates with BLOSUM62 ≥ -2                | ≥ 80% of submitted       | lower = relax to -4, note in report          |
| G6   | AlphaFold pass rate (pLDDT > 70)             | ≥ 50% of submitted       | < 50%: discuss scaffold selection issues     |
| G6   | Barrel RMSD < 2Å (where scaffold PDB exists) | ≥ 50%                    | < 50%: discuss triad disruption              |

**Hard stops:**

- ESMFold coverage < 800/898: investigate failures before proceeding to script 10
- No candidates pass script 12 filter for any target: widen delta threshold to ±25nm, re-run, log
- AlphaFold budget > 20 predictions: prompt user before running

---

## 10. Compute Budget

| Step                                             | Where      | Time         | Cost (est.) |
| ------------------------------------------------ | ---------- | ------------ | ----------- |
| Script 09 (ESMFold, 898 proteins)                | Modal A10G | 1–2 hrs      | $4–8        |
| Script 10 (feature extraction + XGBoost retrain) | Local      | 30–60 min    | —           |
| Script 11 (scaffold retrieval)                   | Local      | < 5 min      | —           |
| Script 12 (mutation enumeration + scoring)       | Local      | 5–15 min     | —           |
| Script 13 (AlphaFold, ≤12 candidates)            | Modal A100 | 1–2 hrs      | $4–8        |
| Script 14 (report)                               | Local      | < 5 min      | —           |
| **Total**                                        |            | **~3–5 hrs** | **~$10–20** |

ESMFold contact maps cached on disk — re-runs skip already-processed proteins.
AlphaFold structures cached — do not re-run if PDB already exists in `alphafold/`.

---

## 11. Dependencies

**New packages (add to `requirements.txt`):**

```
# Already present from Phase 1:
fair-esm>=2.0.0
biopython>=1.81
modal>=0.64.0
xgboost>=2.0.0

# New in Phase 2:
# ColabFold (install on Modal container):
# pip install colabfold[alphafold-minus-jax] @ https://github.com/sokrypton/ColabFold
scipy>=1.11.0                # sparse contact matrix storage
```

**External:**

- Modal A10G for ESMFold (script 09)
- Modal A100 for AlphaFold/ColabFold (script 13)
- BLOSUM62 loaded via `Bio.Align.substitution_matrices.load("BLOSUM62")` (BioPython, already in Phase 1)

---

## 12. Risks and Mitigations

| Risk                                          | Mitigation                                                                                            |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| ESMFold OOM on A10G                           | Single-protein batches, fall back to cpu offload flag                                                 |
| ColabFold not available on Modal              | Fall back to ESMFold pLDDT only (no RMSD), note limitation                                            |
| Contact features don't improve R²             | Expected and valid. Use Phase 1 model for inverse design. Report honestly.                            |
| No candidates within ±15nm for a target       | Widen threshold to ±25nm. Narrow to 1–2 targets. Report as proof-of-concept.                          |
| ESM-2 approximation inflates candidate scores | Flag prominently in report. Optional: re-embed top-3 candidates per target on Modal GPU (adds ~$1–2). |
| Scaffold PDBs missing from FPbase             | Skip RMSD; report pLDDT only for those candidates                                                     |
| AlphaFold run fails on a candidate            | Skip, log, proceed with remaining candidates                                                          |

---

## 13. Execution Protocol

```bash
# Phase 1 complete and all gate artifacts confirmed before starting

python scripts/09_esmfold_contacts.py      # Modal GPU (A10G) — ~1-2 hrs
python scripts/10_contact_features.py      # local — check contact ablation result
python scripts/11_scaffold_retrieval.py    # local — inspect scaffold table before continuing
python scripts/12_mutation_enumeration.py  # local — inspect candidate table before running AlphaFold
python scripts/13_alphafold_validate.py    # Modal GPU (A100) — confirm count ≤ 20 before running
python scripts/14_phase2_report.py         # local
```

**Manual checkpoints:**

- After script 09: verify `features/contact_maps/` has ≥ 800 files before proceeding
- After script 10: confirm contact ablation result logged, decide which forward model to use
- After script 11: inspect `scaffolds.csv` — if any target has no scaffold within ±20nm, widen to ±30nm
- After script 12: inspect top candidates — if BLOSUM scores are all < -2, relax threshold to -4
- Before script 13: confirm candidate count; print cost estimate; proceed only after visual check

---

## 14. Deliverables

1. `features/contact_maps/` — ESMFold contact maps for all 898 proteins
2. `features/contact_features.csv` — aggregated contact statistics
3. `models/xgboost_em_contact.json` — contact-augmented model (only if G4b passes)
4. `results/inverse_design/scaffolds.csv` — scaffold selection per target
5. `results/inverse_design/candidates.csv` — all scored mutation candidates
6. `results/inverse_design/alphafold/` — AlphaFold PDBs + score JSONs
7. `results/inverse_design/alphafold_validation.csv` — pass/fail table
8. `results/phase2_report.md` — self-contained summary of both tracks
9. `results/figures/contact_ablation.png` — R² comparison
10. `results/figures/candidate_predictions.png` — predicted emission vs target per candidate
