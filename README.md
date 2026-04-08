# HgO/Au(111) — Full MLFF Adsorption & Dynamics Study

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)
![CHGNet](https://img.shields.io/badge/CHGNet-v0.3.0-4B8BBE)
![MACE](https://img.shields.io/badge/MACE--MP--0-large-F44336)
![ASE](https://img.shields.io/badge/ASE-3.22%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20%7C%20Colab-yellow?logo=google-colab)

> **A publication-grade computational benchmark of HgO adsorption and dynamics on Au(111) using dual machine-learning force fields (CHGNet v0.3.0 + MACE-MP-0), with full uncertainty quantification, statistical thermodynamics, molecular dynamics, and AI-assisted manuscript preparation.**

---

## Table of Contents

- [Overview](#overview)
- [Scientific Background](#scientific-background)
- [Key Results](#key-results)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Figures Produced](#figures-produced)
- [AI-Assisted Paper Writing](#ai-assisted-paper-writing)
- [Known Issues & Limitations](#known-issues--limitations)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository contains a complete, self-contained computational pipeline for studying the adsorption, dynamics, and thermodynamics of mercury oxide (HgO) on the Au(111) surface. All calculations are performed using machine-learning force fields (MLFFs) — no DFT required at runtime — making the study feasible on GPU-equipped cloud platforms (Kaggle, Google Colab).

The pipeline covers the full lifecycle of a computational surface science study:

1. Slab construction and reference structure relaxation
2. Multi-site adsorption energy calculation (4 high-symmetry sites × 2 tilt angles × 2 MLFFs)
3. Epistemic uncertainty quantification via weight-perturbation ensembles
4. Statistical sampling and energy landscape topology analysis
5. Vibrational frequency analysis with experimental scale-factor calibration
6. Statistical thermodynamics (ΔG(T), ZPE corrections, desorption temperature)
7. Langevin NVT molecular dynamics at 300 K and 500 K
8. Global potential energy surface (PES) mapping
9. Bond activation and coordination number analysis
10. Multi-molecule coverage effects
11. Publication-quality figure generation (11 figures)
12. AI-assisted LaTeX manuscript preparation

> **No synthetic data is used anywhere in this pipeline. All numerical results come directly from MLFF energy evaluations.**

---

## Scientific Background

Mercury (Hg) is a globally regulated environmental pollutant, and gold-based sorbents are among the most effective capture materials due to the strong Hg–Au interaction. Understanding the adsorption mechanism of oxidized mercury species (HgO) on Au(111) at the atomic level is critical for designing next-generation capture materials.

This study uses two state-of-the-art universal MLFFs:

| Model | Architecture | Parameters | Training Set |
|---|---|---|---|
| **CHGNet v0.3.0** | Graph Neural Network | 412,525 | MPtrj (Materials Project trajectories) |
| **MACE-MP-0 (large)** | Equivariant message-passing | — | MPtrj |

The Au(111) slab is modeled as a 4×4×5 supercell (80 atoms) with 20 Å vacuum, using the experimental lattice constant a = 4.0780 Å (Wyckoff, 1963). Four high-symmetry adsorption sites are studied: **ontop**, **bridge**, **fcc hollow**, and **hcp hollow**.

---

## Key Results

### Adsorption Energies (CHGNet)

| Site | E_ads (eV) | d(Hg–O) (Å) | h(O) (Å) | Tilt (°) |
|------|-----------|-------------|---------|---------|
| **bridge** | **−2.0423** | 2.1803 | 2.100 | 61.6 |
| hcp | −2.0086 | 2.2920 | 2.097 | 72.2 |
| ontop | −1.9768 | 2.2824 | 2.070 | 72.6 |
| fcc | −1.9673 | 2.2886 | 1.945 | 67.6 |

**Stability order:** bridge > hcp > ontop > fcc

### Uncertainty Quantification

- σ_UQ ≈ **216 meV** for all CHGNet sites (dominated by slab energy fluctuation: σ_slab = 149 meV)
- All inter-site energy differences (34–75 meV) are **below the epistemic uncertainty threshold** — site discrimination at MLFF level is statistically unresolved without slab-energy cancellation
- MACE σ_UQ ≈ 146 meV but exhibits Hg–O bond dissociation artifacts at tilted geometries

### Molecular Dynamics

| Condition | D (cm²/s) | Δr_lateral (Å) | Bond breaking |
|---|---|---|---|
| Bridge, 300 K | 6.51 × 10⁻¹⁹ | 0.159 | No |
| Bridge, 500 K | 1.25 × 10⁻¹⁸ | 0.509 | Yes (t ≈ 1.12 ps) |
| hcp, 300 K | 8.62 × 10⁻¹⁹ | 0.402 | Yes (frame 121) |

### PES Diffusion Barrier

| Model | E_barrier (meV) |
|---|---|
| CHGNet | 138.0 |
| MACE-MP-0 | 76.6 |

### Thermodynamics

All sites remain thermodynamically stable (ΔG < 0) across the full 200–900 K range studied. No desorption crossover is observed — HgO capture on Au(111) is effectively irreversible under all tested conditions.

### Coverage Effects (bridge site)

| N molecules | E_ads/mol (eV) | E_lateral (meV) |
|---|---|---|
| 1 | −2.0423 | — |
| 2 | −2.3118 | −269.5 (attractive) |
| 4 | −2.2906 | −248.3 (attractive) |

Cooperative binding is observed at all studied coverages.

---

## Pipeline Architecture

```
notebook.ipynb
│
├── Cell 1  — Package installation
├── Cell 2  — Configuration & physical constants
├── Cell 3  — Validation framework & provenance logging
├── Cell 4  — Atoms store (pickle cache)
├── Cell 5  — Materials Project API + literature values
├── Cell 6  — MLFF initialization (CHGNet + MACE + ensembles)
├── Cell 7  — Reference structures (HgO molecule + Au slab)
├── Cell 8  — Adsorption calculations (4 sites × 2 tilts × 2 MLFFs)
├── Cell 9  — Uncertainty quantification (ensemble σ_UQ, SNR)
├── Cell 10 — Statistical sampling (N=20 perturbation runs, GMM)
├── Cell 11 — Vibrational analysis (ASE Vibrations, scale factors, ZPE)
├── Cell 12 — Thermodynamics (ΔG(T), Sackur-Tetrode, rigid rotor)
├── Cell 13 — Publication figures (Figs. 1–6)
├── Cell 14 — Module A: Molecular dynamics (Langevin NVT)
├── Cell 15 — Module B: Global PES mapping (12×12 grid)
├── Cell 16 — Module C: Bond activation & coordination analysis
├── Cell 17 — Module D: Coverage effects (N=1,2,4 molecules)
├── Cell 19 — Benchmarking table compilation
├── Cell 20 — Final provenance report & failure summary
├── Cell 21 — Advanced visualizations (inline display)
└── Cell 24 — Comprehensive mechanistic dashboard
```

---

## Installation

### Requirements

- Python ≥ 3.10
- CUDA-capable GPU strongly recommended (tested on NVIDIA with CUDA 12.x)
- ~8 GB GPU memory for full pipeline with MACE-MP-0 large

### Quick Install

```bash
pip install chgnet>=0.3.0 ase>=3.22 mp-api>=0.39 pymatgen>=2024.1 \
            scikit-learn>=1.3 plotly>=5.15 scipy>=1.10 \
            mace-torch>=0.3.0
```

### Full Install (with optional dashboard)

```bash
pip install chgnet>=0.3.0 ase>=3.22 mp-api>=0.39 pymatgen>=2024.1 \
            scikit-learn>=1.3 plotly>=5.15 dash>=2.14 \
            dash-bootstrap-components>=1.5 scipy>=1.10 mace-torch>=0.3.0
```

> **Note:** `mace-torch` installation may fail on some platforms. The notebook falls back to CHGNet-only mode automatically if MACE is unavailable.

### Running on Kaggle (Recommended)

1. Create a new Kaggle notebook with GPU accelerator enabled (P100 or T4)
2. Upload `notebook.ipynb` or paste cell contents
3. Set your Materials Project API key in Cell 2:
   ```python
   MP_API_KEY = "your_api_key_here"
   ```
4. Run all cells sequentially

### Running on Google Colab

```python
# Run this first in Colab
!pip install -q chgnet mace-torch ase mp-api pymatgen scikit-learn scipy

# Then mount Drive if you want to save outputs
from google.colab import drive
drive.mount('/content/drive')
```

Change `BASE_DIR` in Cell 2 from `/kaggle/working/hgo_benchmark` to `/content/drive/MyDrive/hgo_benchmark`.

---

## Usage

### Full Pipeline

Run all cells in order from Cell 1 to Cell 24. Total runtime on a single GPU:

| Section | Approximate Time |
|---|---|
| Installation (Cell 1) | 3–5 min |
| MLFF initialization (Cell 6) | 1–2 min |
| Reference structures (Cell 7) | 1–2 min |
| Adsorption calculations (Cell 8) | 8–12 min |
| UQ ensemble (Cell 9) | 2–3 min |
| Statistical sampling (Cell 10) | 4–6 min |
| Vibrational analysis (Cell 11) | 3–5 min |
| Thermodynamics (Cell 12) | < 1 min |
| Figures (Cell 13) | 1–2 min |
| Molecular dynamics (Cell 14) | 8–12 min |
| PES mapping (Cell 15) | 3–5 min |
| Bond & coverage (Cells 16–17) | 3–5 min |
| **Total** | **~40–60 min** |

### CHGNet-Only Mode

If MACE is unavailable, the pipeline continues automatically. All CHGNet modules run fully. MACE-dependent comparisons are skipped gracefully.

### Adjusting Computational Cost

Key parameters in Cell 2 to scale compute up or down:

```python
MD_STEPS       = 2000    # Increase to 10000+ for production MD
PES_GRID_N     = 12      # Increase to 20+ for higher-resolution PES
N_ENSEMBLE     = 5       # Increase to 10+ for tighter UQ bounds
N_RUNS         = 20      # Increase to 50+ for statistical sampling
COVERAGE_N_MOL = [1,2,4] # Add more coverage points as needed
```

---

## Output Structure

After a full run, outputs are organized as follows:

```
/kaggle/working/hgo_benchmark/
│
├── data/
│   ├── provenance.json          # Full calculation provenance (65 entries)
│   ├── mp_reference.json        # Materials Project Au bulk data
│   ├── literature.json          # Experimental reference values
│   ├── chgnet_ads.json          # All adsorption results (CHGNet)
│   ├── uq_results.json          # σ_UQ and SNR matrices
│   ├── sampling_results.json    # Statistical sampling (bridge, hcp)
│   ├── vib_results.json         # Vibrational frequencies and ZPE
│   ├── thermo_results.json      # ΔG(T) arrays for all sites
│   ├── md_results.json          # MD diffusion and bond statistics
│   ├── pes_summary.json         # PES barrier summary
│   ├── bond_activation_results.json
│   ├── coverage_results.json
│   └── benchmarking_table.json  # Consolidated comparison table
│
├── figures/
│   ├── Fig01_UQ_Diagnosis.png
│   ├── Fig02_Adsorption.png
│   ├── Fig03_Sampling.png
│   ├── Fig04_Thermodynamics.png
│   ├── Fig05_Vibrations.png
│   ├── Fig06_Summary.png
│   ├── FigMD1_MD_Dynamics.png
│   ├── FigMD2_Migration_Trajectory.png
│   ├── FigPES1_ContourMap.png
│   ├── FigBOND1_BondActivation.png
│   └── FigCOV1_Coverage.png
│
├── atoms_store/                 # Relaxed Atoms objects (.pkl)
│   ├── hgo_CHGNet.pkl
│   ├── au_slab_CHGNet.pkl
│   ├── ads_best_CHGNet_bridge.pkl
│   └── ...
│
└── trajectories/                # Langevin MD trajectory files (.traj)
    ├── md_CHGNet_bridge_300K.traj
    ├── md_CHGNet_bridge_500K.traj
    └── md_CHGNet_hcp_300K.traj
```

---

## Figures Produced

| Figure | Content | Section |
|---|---|---|
| Fig. 1 | Epistemic σ_UQ per site — CHGNet vs MACE | UQ |
| Fig. 2 | Adsorption energies ± σ_UQ + site resolution SNR | Adsorption |
| Fig. 3 | Statistical sampling violin plots (bridge, hcp) | Sampling |
| Fig. 4 | ΔG(T) curves with uncertainty bands | Thermodynamics |
| Fig. 5 | Vibrational frequencies, scale factors, ZPE | Vibrations |
| Fig. 6 | Summary dashboard | All |
| Fig. MD-1 | MSD and Hg–O bond dynamics (300K, 500K) | MD |
| Fig. MD-2 | Surface migration trajectory on Au(111) plane | MD |
| Fig. PES-1 | Global PES contour map (CHGNet + MACE) | PES |
| Fig. BOND-1 | Bond activation: Δd, CN, corrugation | Bonding |
| Fig. COV-1 | Coverage effects: E_ads/mol, lateral interactions | Coverage |

All figures are saved at 300 DPI to `/figures/` and are ready for manuscript inclusion.

---

## AI-Assisted Paper Writing

This project includes an experiment in AI-assisted scientific manuscript preparation. A structured prompt was developed to instruct a large language model (Claude, Anthropic) to write a complete, publication-ready LaTeX manuscript based exclusively on the numerical results produced by this pipeline.

The prompt enforces:

- **Zero fabrication** — only values from the computational output are used
- **Full LaTeX formatting** with siunitx, booktabs, chemformula
- **In-text figure placement** — figures appear beside the paragraphs that discuss them
- **Critical physical interpretation** — not just reporting numbers, but explaining them
- **Honest treatment of limitations** — σ_UQ exceeding inter-site differences, MACE bond-breaking artifacts, and the MP API anomaly are all explicitly discussed
- **Complete citation framework** with BibTeX entries

The prompt and methodology are documented in `paper_prompt.md` in this repository. The generated manuscript targets *The Journal of Physical Chemistry C* or *Physical Chemistry Chemical Physics* (both Q1).

> **Transparency note:** The computational results are 100% real MLFF calculations. The manuscript text is AI-generated from those results using the structured prompt. This is documented explicitly as part of the study's reproducibility record.

---

## Known Issues & Limitations

### MACE Bond Dissociation at Tilted Geometries
MACE-MP-0 produces unphysical Hg–O bond elongation (2.98–3.67 Å, expected 1.8–2.5 Å) for all 45° tilt configurations. These are caught by the validation framework and logged as failures. Only 0° tilt MACE results are used in cross-model comparisons. Physical interpretation: MACE's equivariant descriptor may overestimate metal-oxygen attraction at non-vertical orientations for heavy elements like Hg.

### Materials Project API Anomaly
The MP API returned a = 2.9495 Å for Au (mp-81), which is physically incorrect for fcc gold (~4.08 Å). This was rejected by the lattice validation check. The experimental value (Wyckoff, 1963) was used instead. This may indicate a BCC polymorph was returned by the API at the time of calculation.

### σ_UQ vs Inter-Site Energy Differences
All inter-site energy differences (34–75 meV) fall below the CHGNet epistemic uncertainty (~216 meV). This means site selectivity predictions from CHGNet alone should be interpreted with caution and ideally validated by DFT single-points on the MLFF-relaxed geometries.

### MD Trajectory Length
The default MD run (2000 steps × 1 fs = 2 ps) is sufficient for structural characterization but short for converged diffusion coefficients. For production diffusion studies, increase `MD_STEPS` to at least 10,000.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `chgnet` | ≥ 0.3.0 | Primary MLFF |
| `mace-torch` | ≥ 0.3.0 | Secondary MLFF (optional) |
| `ase` | ≥ 3.22 | Atomic simulation environment |
| `pymatgen` | ≥ 2024.1 | Structure handling |
| `mp-api` | ≥ 0.39 | Materials Project API client |
| `scikit-learn` | ≥ 1.3 | GMM basin analysis |
| `scipy` | ≥ 1.10 | Statistics, linear regression |
| `torch` | ≥ 2.0 | Backend for MLFFs |
| `numpy` | ≥ 1.24 | Numerical arrays |
| `matplotlib` | ≥ 3.7 | Figure generation |
| `plotly` | ≥ 5.15 | Interactive plots (optional) |

---

## Citation

If you use this pipeline or results in your work, please cite:

```bibtex
@misc{hgo_au111_mlff_2024,
  title   = {{HgO/Au(111) Full MLFF Adsorption and Dynamics Study}},
  author  = {Computational Surface Science Group},
  year    = {2024},
  url     = {https://github.com/YOUR_USERNAME/YOUR_REPO},
  note    = {CHGNet v0.3.0 + MACE-MP-0 dual-MLFF benchmark}
}
```

Also cite the underlying MLFF models:

```bibtex
@article{chgnet2023,
  title   = {A universal graph deep learning interatomic potential for the elements},
  author  = {Deng, Bowen and others},
  journal = {Nature Machine Intelligence},
  year    = {2023},
  doi     = {10.1038/s42256-023-00716-3}
}

@article{mace2023,
  title   = {{MACE-MP-0}: A Foundation Model for Atomistic Materials Science},
  author  = {Batatia, Ilyes and others},
  journal = {arXiv preprint arXiv:2401.00096},
  year    = {2023}
}

@article{callear1962,
  title   = {The photochemistry of mercury(II) oxide},
  author  = {Callear, A. B. and Norrish, R. G. W.},
  journal = {Proceedings of the Royal Society of London A},
  year    = {1962},
  doi     = {10.1098/rspa.1963.0022}
}
```

---

## License

MIT License — see `LICENSE` for details.

---

*All calculations in this repository use real MLFF energy evaluations. No synthetic data, placeholder values, or interpolated results are present anywhere in the pipeline. Provenance for all 65 logged results is available in `data/provenance.json`.*
