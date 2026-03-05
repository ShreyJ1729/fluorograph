<!-- Note: This file is full of bloat and needs to be refactored. -->

# FluoroGraph: Autonomous Research Agent for Fluorescent Protein Spectral Prediction

## PRD v2.0

**Author:** ShreyJ
**Target Audience:** FluoroGraph collaborators
**Date:** March 2026

---

## 1. Mission

Build an autonomous AI research agent that predicts fluorescent protein excitation/emission spectra from amino acid sequence, using graph-based structural representations (RuVector) to outperform the current state-of-the-art (FPredX). The agent runs overnight on Modal, self-directs its research via a Ralph loop, produces publication-quality reports, and optionally generates candidate sequences for desired target wavelengths.

---

## 2. Phased Goals

This project has three phases with explicit gates. Each phase is independently valuable and produces its own deliverables.

### Phase 1: Spectral Prediction Pipeline (MVP)

**Gate: None — this is the starting point.**

- **G0:** Replicate FPredX on our FPbase dataset. Run MAFFT alignment, one-hot encoding, and XGBoost exactly as described in Tam & Zhang 2021. Validates data pipeline. **Gates all downstream work.**
- **G1:** Predict ex/em maxima from sequence with R² ≥ 0.85 (emission), matching or exceeding FPredX.
- **G2:** Identify which sequence features/positions are most predictive of spectral color (interpretability report).
- **G3:** Produce evaluation metrics with proper train/val/test splits, baselines, and sanity checks.

### Phase 2: Inverse Design

**Gate: Phase 1 G0 + G1 achieved. Forward model must work before inverse design is meaningful.**

- **G4:** Generate novel candidate sequences for a desired target emission wavelength.
- **G5:** Validate structural plausibility of proposed mutations using AlphaFold on Modal GPU.

### Phase 3: Autonomous Discovery

**Gate: Phase 1 complete. RuVector full graph stack operational (Cypher, GNN, attention layers, SONA). This is a separate research effort and may not run in the same overnight session.**

- **G6:** Literature-guided autonomous discovery of structure-function relationships using RuVector graph queries, dynamic ontology construction, and SONA-reinforced query generation.

---

## 3. Non-Goals & Out of Scope

- **QM/MM spectral computation.** No first-principles spectra (TD-DFT). All predictions from ML on experimental FPbase data.
- **Wet-lab validation.** Computational proposals only.
- **Foundation model training.** ESM-2 is a frozen feature extractor. No fine-tuning.
- **Real-time serving.** Overnight batch job, not a user-facing API.
- **Multi-objective optimization.** We predict ex/em maxima only — not brightness, photostability, folding efficiency, etc.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   OUTER RALPH LOOP                       │
│              (Research Director Agent)                    │
│                                                          │
│  Reads: PRD + Research Log + Results                     │
│  Decides: Next research direction                        │
│  Writes: Updated PRD, dispatches to Inner Loop           │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              INNER RALPH LOOP                       │ │
│  │           (Experiment Runner Agent)                  │ │
│  │                                                      │ │
│  │  Compute Tools:                                      │ │
│  │    ├── data_pipeline    (MAFFT align, feature eng)   │ │
│  │    ├── esm2             (protein embeddings)         │ │
│  │    ├── esmfold          (structure prediction)       │ │
│  │    ├── chromadb         (flat vector baseline)       │ │
│  │    ├── ruvector         (graph embed, similarity)    │ │
│  │    ├── train_model      (XGBoost, eval metrics)      │ │
│  │    ├── alphafold        (structure validation, GPU)   │ │
│  │    └── analyze          (plots, statistics)           │ │
│  │                                                      │ │
│  │  Orchestration Tools:                                │ │
│  │    ├── literature       (web search, paper lookup)    │ │
│  │    ├── report           (generate markdown/figures)   │ │
│  │    ├── update_prd       (rewrite own PRD section)     │ │
│  │    └── log              (append to research log)      │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘

Infrastructure: Modal
  ├── CPU containers: RuVector (Rust), MAFFT, XGBoost, ChromaDB, Agent
  └── GPU containers: ESMFold (A10G), AlphaFold (A100/A10G, on-demand)
```

---

## 5. Dependencies & Prerequisites

### Infrastructure

- Modal account with GPU access (A10G minimum, A100 preferred for AlphaFold)
- Claude API access for agent loop
- Sufficient Modal CPU quota for ~12 hours

### Data

- `fpbase_merged.csv` — 1,110 proteins, 899 with complete (sequence, ex_max, em_max)
- FPbase mutation lineage data (for graph edges in Phase 3)

### Software

- MAFFT, ESM-2 weights (`esm2_t33_650M_UR50D`), ESMFold weights, XGBoost + Optuna, ChromaDB, AlphaFold2 weights (Phase 2 only)

### RuVector Capability Matrix

Required capabilities differ by phase. **Verify before starting each phase.**

**Phase 1 — Required:**

- Vector storage/indexing (per-sequence embeddings with metadata)
- Cosine similarity k-NN retrieval
- Graph construction (nodes + edges from adjacency/contacts)
- Graph embedding computation (fixed-dim from topology + node features)
- PyO3 Python bindings (`maturin build --release`)

**Phase 2 — Required (additive):**

- Nearest-neighbor retrieval with metadata filtering
- Per-residue embedding similarity analysis

**Phase 3 — Required (additive):**

- Cypher query interface over protein graph
- GNN re-ranking layer
- Hyperedge support (chromophore pockets)
- Hyperbolic embeddings (Poincaré space)
- EdgeFeaturedAttention and NeighborhoodAttention
- SONA: LoRA + EWC++ for query strategy adaptation
- SONA ReasoningBank

**If any Phase 3 capability is missing: STOP. Do not attempt Phase 3.**

---

## 6. Data

### Source

- FPbase merged dataset: 1,110 proteins, 899 with complete (sequence, ex_max, em_max)
- File: `fpbase_merged.csv`

### Splits

- Train: 70% (629), Validation: 15% (135), Test: 15% (135)
- Strategy: stratified by emission wavelength bin (blue/cyan/green/yellow/orange/red/far-red/infrared)
- Random seed: 42

### Data Augmentation (Agent-Discoverable)

- MAY incorporate deep mutational scanning data (Sarkisyan et al. 2016) as auxiliary signal
- MAY pull UniProt/GenBank homologs for embedding pretraining only
- All augmentation decisions must be logged with rationale

---

## 7. Feature Engineering Pipeline

### What RuVector Stores Per Protein

**1. Per-sequence embedding (ESM-2 mean-pooled, ~1280-dim)**
Global protein representation. Drives k-NN retrieval.

**2. Per-residue embeddings (ESM-2, seq_len × 1280)**
Node features in RuVector graphs. Enable fine-grained interpretability.

**3. Classical sequence features**
AA composition (20-dim), sequence length, isoelectric point, molecular weight, chromophore triplet identity (positions ~65-67), FPbase metadata (QY, EC, brightness, pKa, oligomerization, maturation time, lifetime), switch type.

**4. Graph structure**
Nodes = residues with ESM-2 embeddings. Edges = sequence adjacency (i, i±1) + ESMFold spatial contacts (8Å). Edge weights = inverse distance.

### Stage 1: FPredX Baseline Replication

MAFFT alignment → one-hot encoding (21 features/position) → XGBoost → ex_max, em_max. Feature importance top 20 by gain. This XGBoost model remains valuable for interpretability (G2) even after RuVector becomes primary predictor.

### Stage 2: ESM-2 Embeddings + k-NN (Primary Predictor)

ESM-2 encoding → index in ChromaDB (flat baseline) and RuVector → k-NN prediction via distance-weighted average → compare to Stage 1.

### Stage 3: Structural Contact Edges

ESMFold structures → contact maps at 8Å → graph construction with spatial edges → graph embeddings → k-NN → ablation (sequence-only vs. contact-enriched vs. contact-only).

### Stage 4: Interpretability

RuVector neighborhood analysis + XGBoost feature importance → cross-validate agreement → compare to known positions (65-67 triad, 148, 203, 222) → novel position discovery → per-color analysis.

---

## 8. Tools

### Compute Tools

#### `data_pipeline`

- **I/O:** raw CSV → feature matrices (.npz), alignment (.fasta), classical features CSV
- **Actions:** MAFFT alignment, one-hot encoding, classical feature computation
- **Runs on:** Modal CPU

#### `esm2`

- **I/O:** sequence(s) → per-sequence (1280-dim) + per-residue (seq_len × 1280) embeddings
- **Runs on:** Modal CPU (650M model fits in CPU memory)
- **Batch:** ~32 proteins/batch, cache all

#### `esmfold`

- **I/O:** sequence → contact map (8Å), pLDDT, optionally PDB
- **Runs on:** Modal GPU (A10G), seconds per protein
- **Batch:** all 899, ~1-2 hours

#### `chromadb`

- **I/O:** ESM-2 embeddings + metadata → k-NN predictions, neighbor lists
- **Runs on:** Modal CPU
- **Purpose:** flat vector retrieval baseline. Establishes what any off-the-shelf vector DB achieves.

#### `ruvector`

- **I/O:** embeddings + contact maps → graph embeddings, k-NN predictions, similarity analysis
- **Actions:** vector indexing, graph construction, graph embedding, k-NN retrieval, per-residue similarity
- **Runs on:** Modal CPU
- **Build:** `maturin build --release` → PyO3 wheel (`import ruvector`)

#### `train_model`

- **I/O:** features + targets → trained model, metrics (R², MAE, RMSE, per-color), feature importance
- **Actions:** XGBoost + Optuna, cross-validation, held-out eval
- **Runs on:** Modal CPU

#### `alphafold`

- **I/O:** sequence → PDB, pLDDT, contact map
- **Runs on:** Modal GPU (A100/A10G)
- **Phase 2 only. Budget: ~20-50 predictions.** Does NOT predict spectra.

#### `analyze`

- **I/O:** model results → plots (.png), analysis text
- **Runs on:** Modal CPU (matplotlib/seaborn)

### Orchestration Tools

#### `literature`

- Web search for papers → summary with citations. Runs via Claude API.

#### `report`

- Research log + analysis → structured markdown report. Runs via Claude API.

#### `update_prd`

- Rewrites PRD sections. CANNOT modify §1-3 or §13. Can modify §6-8 and §10.

#### `log`

- Timestamped entries (observation/decision/result/error) → research log.

---

## 9. Ralph Loop Protocol

### Outer Loop: Research Director

**Trigger:** after every inner loop completion or 2-hour timeout.

1. Read PRD, research log, latest results
2. Assess progress against current phase goals
3. Decide: **CONTINUE** | **PIVOT** (rewrite experiment plan) | **DEEPEN** (more compute on promising finding) | **ADVANCE** (phase gate met, next phase) | **REPORT** | **HALT**
4. Log decision with rationale

### Inner Loop: Experiment Runner

1. Read experiment plan from §10
2. Execute sequential tool calls, logging results
3. On failure: retry once with modified params, then log error and continue
4. Return to outer loop after completion or 2-hour mark

**Experiment format:**

```
EXPERIMENT: [name]
HYPOTHESIS: [what we expect]
STEPS: [numbered tool calls]
SUCCESS_CRITERIA: [thresholds]
FAILURE_ACTION: [what to do if not met]
```

---

## 10. Experiment Plan (Agent-Mutable)

### Phase 1 Experiments

#### Experiment 0: FPredX Replication (G0) — GATE

**Hypothesis:** Reproduce FPredX (Tam & Zhang 2021) with MAFFT → one-hot → XGBoost.
**Steps:**

1. `literature`: Pull FPredX paper — exact R², MAE, dataset size, hyperparameters
2. `data_pipeline`: MAFFT alignment on 899 sequences
3. `data_pipeline`: One-hot encoding (21 features/position)
4. `train_model`: XGBoost → em_max (Optuna 100 trials, 5-fold CV)
5. `train_model`: XGBoost → ex_max (same protocol)
6. `analyze`: Compare metrics to published. Top 20 feature importance.
7. `analyze`: If R² diverges >0.05, investigate dataset/alignment/hyperparameter differences
8. `log`: Record baseline

**Success:** R² within 0.05 of published for both em and ex.
**Failure:** DO NOT proceed. Debug alignment, data leakage, dataset version, their exact software versions, errata.

#### Experiment 1: ESM-2 Flat vs. Graph-Aware Retrieval (G1)

**Hypothesis:** RuVector graph-aware retrieval outperforms flat ChromaDB because spectral properties depend on local chromophore environment.

**1a — Flat baseline (ChromaDB):**

1. `esm2`: Encode all 899 proteins, cache
2. Store in ChromaDB with metadata
3. Leave-one-out k-NN (k=5), distance-weighted
4. `analyze`: R², MAE, RMSE, per-color breakdown
5. `analyze`: Tune k (3, 5, 7, 10)

**1b — Graph-aware (RuVector, sequence-only edges):**

1. `ruvector`: Index per-sequence embeddings (apples-to-apples vs 1a)
2. `ruvector`: Build graphs — residue nodes, sequence adjacency edges only
3. `ruvector`: Graph embeddings → k-NN (same k as 1a)
4. `analyze`: Compare to 1a. Where do they disagree?

**Success:** 1b shows ≥0.02 R² improvement over 1a.
**Failure:** If equal, proceed — contact edges in Exp 2 may help. If worse, check embedding dim and graph params.

#### Experiment 2: Structural Contact Edges (G1)

**Hypothesis:** ESMFold 3D contacts capture chromophore-environment interactions that flat retrieval and sequence-only graphs miss.

**Steps:**

1. `esmfold`: All 899 proteins → 8Å contact maps, cache
2. `ruvector`: Graphs with adjacency + spatial edges
3. `ruvector`: Contact-enriched graph embeddings → k-NN
4. `analyze`: Three-way comparison (1a flat vs 1b seq-only vs 2 contacts)
5. `analyze`: Ablation on edge types
6. `literature`: Known long-range FP contacts
7. `log`: Key result for RuVector thesis

**Success:** Contact-enriched R² > both flat and seq-only by ≥0.02.
**Failure:** Try cutoffs (6/10/12Å), inverse-distance weighting, chromophore-proximal only. If still nothing, report honestly that ESM-2 already encodes structure.

#### Experiment 3: Interpretability (G2, G3)

**Hypothesis:** Predictive features converge on known chromophore-tuning residues.

**Steps:**

1. `analyze`: XGBoost top 30 → alignment positions
2. `analyze`: RuVector neighborhood — which residues drive neighbor selection?
3. `analyze`: Cross-reference both methods
4. `analyze`: Compare to known positions (65-67, 148, 203, 222)
5. `analyze`: Per-color analysis
6. `literature`: Novel model-flagged positions
7. `report`: Draft interpretability section

**Success:** ≥50% of top 20 features match known positions.
**Failure:** Validate novel positions via literature. If biologically nonsensical, investigate overfitting.

---

### Phase 1 → Phase 2 Gate

- G0 achieved (FPredX replication within tolerance)
- G1 achieved (R² ≥ 0.85 emission, or best achievable with honest reporting)
- G2 and G3 deliverables produced
- Phase 1 report drafted

---

### Phase 2 Experiments

#### Experiment 4: Inverse Design (G4)

**Hypothesis:** RuVector retrieval + XGBoost importance can propose emission-shifting mutations.

**Steps:**

1. Targets: 480nm (blue), 550nm (green-yellow), 620nm (red), 700nm (far-red)
2. `ruvector`: 5 nearest FPs per target (scaffolds)
3. `analyze`: Top 10 emission-shifting positions per scaffold
4. `analyze`: Enumerate substitutions, filter by predicted shift + evolutionary plausibility
5. `train_model`: Forward model → predicted emission + CI
6. `literature`: Similar mutations in literature

**Output per target:**

```
Target: 620nm (red)
Scaffold: mCherry (em=610nm)
  Candidate 1: +T203H/S65G → predicted 618nm (±8nm)
  Candidate 2: +I161S → predicted 625nm (±12nm)
```

**Success:** ≥2 of 4 targets produce candidates within ±15nm.
**Failure:** Narrow to single color family, report as proof-of-concept.

#### Experiment 5: Structural Validation (G5)

**Hypothesis:** AlphaFold confirms proposed mutations preserve the barrel fold.

**Steps:**

1. `alphafold`: Top 3 candidates per target (~12-20 predictions)
2. Check: pLDDT >70, barrel RMSD <2Å, chromophore triad geometry preserved
3. `report`: Candidate table (scaffold, mutations, predicted em, CI, pLDDT, RMSD, plausibility)

**Success:** ≥50% pass structural validation.

---

### Phase 2 → Phase 3 Gate

- Phase 1 complete
- All Phase 3 RuVector capabilities verified operational (Cypher, GNN, hyperedges, hyperbolic embeddings, attention layers, SONA + ReasoningBank)
- **If any missing: STOP. Document gaps.**

---

### Phase 3 Experiments

#### Experiment 6: Literature-Guided Discovery (G6)

**This is a discovery task, not a prediction task.**

**Steps:**

1. `ruvector`: Full knowledge graph — 899 proteins, residue nodes, sequence/spatial/mutation edges, metadata
2. `literature`: Scan 10-15 FP review papers → extract claims → build initial ontology
3. `ruvector`: Create ontology types from literature
4. **Discovery loop (50-100 cycles):**
   a. Select literature claim or generate hypothesis
   b. Translate to Cypher query
   c. Evaluate: significance, novelty, plausibility
   d. SONA score → reinforce productive patterns
   e. Generate follow-ups (combinatorial, cross-lineage, unexplored positions)
   f. Update ontology
   g. Log findings
5. `analyze`: Compile high-reward findings
6. `report`: Discovery section ranked by SONA reward

**Dynamic Ontology:**
Initial (from data): `(:Protein)`, `(:Residue)`, `[:HAS_RESIDUE]`, `[:SEQUENCE_ADJACENT]`, `[:SPATIAL_CONTACT]`, `[:MUTANT_OF]`
Literature-derived (grows): `(:ChromophoreType)`, `(:ProtonationState)`, `[:ESPT_PATHWAY]`, `[:PI_STACKING]`, `[:HYDROGEN_BOND]`, `(:ColorFamily)`, `[:EVOLVED_FROM]`, `(:BarrelContact)`

**SONA Rewards:** Novel relationship +1.0 | Novel combinatorial +0.8 | Quantitative refinement +0.4 | Known confirmation +0.2 | No pattern 0.0 | Malformed query -0.3 | Implausible finding -0.5

**RuVector features leveraged:** Cypher queries, GNN re-ranking, hyperedges, hyperbolic embeddings, EdgeFeaturedAttention, NeighborhoodAttention, SONA LoRA+EWC++, ReasoningBank.

**Success:** ≥3 novel findings, ≥5 confirmed, ≥10 new ontology types, positive SONA trend.
**Failure:** Seed with diverse literature. If no novelty, report confirmations.

#### Experiment 7: Final Report

Compile: abstract, methods, results (baseline vs RuVector), interpretability, inverse design (if Phase 2), discovery (if Phase 3), discussion, appendix.

---

## 11. Risks & Mitigations

### High

**RuVector build failure.** PyO3 bindings don't compile → graph pipeline blocked.
→ Experiment 1a (ChromaDB) is a complete fallback. Phase 1 still produces results.

**FPredX replication fails.** Data pipeline or dataset is flawed.
→ Explicit gating. Debug before proceeding.

**ESM-2 doesn't beat one-hot.** Dataset too small for embeddings to help.
→ Legitimate finding. Report honestly. Interpretability (G2) still valuable.

### Medium

**ESMFold contacts don't improve predictions.** ESM-2 already encodes structure.
→ Try multiple cutoffs. Report null result.

**Phase 3 capabilities don't exist.** Gap between current RuVector and full Cypher+SONA stack.
→ Strict gate. No workarounds.

**Small dataset (899 proteins).** Limits model complexity.
→ Leave-one-out eval, stratified splits, augmentation options.

### Low

**Compute budget exceeded.** → 2x hard cap. ESMFold cheap, AlphaFold budgeted.

**Claude API issues during overnight run.** → 2-hour checkpoints, recoverable research log.

---

## 12. Compute Budget

| Resource                                      | Usage       | Cost        |
| --------------------------------------------- | ----------- | ----------- |
| Modal CPU (agent + data + XGBoost + ChromaDB) | ~4 hrs      | ~$3-6       |
| Modal CPU (ESM-2, 899 proteins)               | ~1-2 hrs    | ~$2-4       |
| Modal CPU (RuVector, graph build + k-NN)      | ~1-2 hrs    | ~$2-4       |
| Modal GPU A10G (ESMFold, 899 proteins)        | ~1-2 hrs    | ~$4-8       |
| Modal GPU A10G (AlphaFold, ~20-50, Phase 2)   | ~1-2 hrs    | ~$4-8       |
| Claude API (~200 calls × ~2K tokens)          | —           | ~$10-20     |
| **Total**                                     | **~12 hrs** | **~$25-50** |

---

## 13. Safety & Constraints (Immutable)

- Agent CANNOT modify this section, §1-3
- Agent CANNOT make external API calls beyond Modal, Claude API, web search
- Agent CANNOT delete data or overwrite original CSV
- Agent MUST log every decision and tool call
- Agent MUST halt if budget exceeds 2x estimate
- Agent MUST produce Phase 1 report minimum, even if later phases fail
- All random seeds logged for reproducibility
- No hallucinated metrics — all numbers from actual evaluation

---

## 14. Success Criteria

### Phase 1 (MVP)

| Goal | Metric                 | Threshold     |
| ---- | ---------------------- | ------------- |
| G0   | R² matches FPredX      | Within 0.05   |
| G1   | R² (emission)          | ≥ 0.85        |
| G1   | MAE (emission)         | ≤ 15 nm       |
| G2   | Features match biology | ≥ 50% overlap |
| G3   | Proper evaluation      | Complete      |

### Phase 2

| Goal | Metric                  | Threshold        |
| ---- | ----------------------- | ---------------- |
| G4   | Plausible candidates    | ≥ 2 of 4 targets |
| G5   | AlphaFold confirms fold | pLDDT > 70       |

### Phase 3

| Goal | Metric          | Threshold                |
| ---- | --------------- | ------------------------ |
| G6   | Novel findings  | ≥ 3 novel, ≥ 5 confirmed |
| G6   | SONA trend      | Positive reward          |
| G6   | Ontology growth | ≥ 10 new types           |

---

## 15. MVP Definition

**Minimum shippable output = Phase 1 complete:**

- FPredX baseline replicated (Exp 0)
- ESM-2 k-NN evaluated against flat and graph-aware retrieval (Exp 1-2)
- Interpretability with biological validation (Exp 3)
- Publication-quality report with honest assessment of graph-aware vs. flat retrieval

Everything beyond Phase 1 adds value but is not required for the overnight run to be successful.

---

## 16. Deliverables

### Phase 1 (MVP)

1. **Research Report** — methods, results, interpretability (markdown + figures)
2. **Trained Models** — XGBoost for ex/em prediction
3. **Vector Indices** — RuVector + ChromaDB with ESM-2 embeddings
4. **Research Log** — timestamped decisions, pivots, results

### Phase 2 (if completed)

5. **Candidate Sequences** — predictions, CIs, structural validation
6. **Updated Report** — inverse design section

### Phase 3 (if completed)

7. **Discovery Report** — novel findings, SONA trajectory, ontology
8. **Protein Knowledge Graph** — RuVector with literature-derived ontology

### Always

9. **This PRD** — final agent-modified version
10. **Code** — all Modal functions, tools, agent loop (reproducible)
