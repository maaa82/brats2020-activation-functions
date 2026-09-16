# Activation functions for necrotic-core segmentation in glioblastoma (BraTS 2020)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22803086.svg)](https://doi.org/10.5281/zenodo.22803086)

Code, per-patient results and figures for the paper

> Saleh, M.M.; Hussein, E.M.; Salih, M.E.; Ahmed, M.A.A.
> **Enhancing 3D MRI-Based Necrotic Core Segmentation in Glioblastoma Using
> Activation Functions in Deep Learning.** *Informatics* **2026**, *13*(7), 118.
> [doi:10.3390/informatics13070118](https://doi.org/10.3390/informatics13070118) (open access, CC BY)

The study is a controlled, single-variable ablation: **twelve activation
functions** are swapped into an otherwise fixed **residual 3D U-Net**
(`ImprovedUNet3D`, ≈ 22.93 M parameters) trained on the **BraTS 2020** training
cohort, and every model is scored per patient on the five regions **NCR, ED, ET,
TC, WT** with Dice, 95 % Hausdorff distance, sensitivity and precision. Three
follow-up experiments — a **three-seed** repeat, **five-fold cross-validation**
and a **same-GPU timing** run — test how robust the single-run findings are.

This repository contains the training, evaluation and analysis code, the
per-patient metric files behind every results table, the training logs, and
the paper figures. It does **not** contain imaging data or model weights
(see §7 and §8).

- **Companion repository:** the follow-up study on activation × normalisation
  strategy in nnU-Net (BraTS 2023) lives at
  [maaa82/brats-activation-normalisation](https://github.com/maaa82/brats-activation-normalisation).
  Nothing from that study is duplicated here.

---

## 1. Experimental protocol (what the archived results describe)

| Item | Value |
|---|---|
| Data | BraTS 2020 training set, 369 patients, modalities FLAIR / T1 / T1ce / T2 |
| Preprocessing | per-modality z-score normalisation over brain voxels (> 0); channels stacked as `[FLAIR, T1, T1ce, T2]`; centre crop `[56:184, 56:184, 13:141]` → 128 × 128 × 128 × 4; label 4 remapped to 3 |
| Split | volume-based 80 / 20. Patient folders are sorted, the first `int(369 × 0.2) = 73` become the validation set (`BraTS20_Training_001` – `_073`), the remaining 296 the training set |
| Model | `ImprovedUNet3D`: 4-level encoder–decoder of residual blocks (Conv3d–BN–act–Conv3d–BN, 1 × 1 × 1 skip), channels 32–64–128–256–512, transposed-conv upsampling |
| Loss | unweighted cross-entropy + soft Dice (mean over the four classes) |
| Optimiser | AdamW, lr 3 × 10⁻⁴, `CosineAnnealingLR(T_max = 100)`, batch 2, AMP |
| Epochs / seed | 100 epochs; `seed_everything(42)` with deterministic cuDNN; one run per activation |
| Checkpoint selection | epoch with the highest mean validation Dice over the three tumour classes |
| Augmentation | none (deliberate, to isolate the activation-function effect) |
| Metrics | Dice, HD95, sensitivity, precision per patient and region (§4) |
| Statistics | Wilcoxon signed-rank test vs ReLU, paired by patient; `*` p < 0.05, `**` p < 0.01; Holm and Benjamini–Hochberg adjustment in `analysis/` |

### The twelve activations

| # | Name in code | Function | Run folder |
|---|---|---|---|
| 1 | `relu` | ReLU (baseline) | `Exp1_ReLU` |
| 2 | `leaky_relu` | Leaky ReLU, slope 0.01 | `Exp2_Leaky` |
| 3 | `prelu` | PReLU | `Exp3_PReLU` |
| 4 | `elu` | ELU | `Exp4_ELU` |
| 5 | `gelu` | GELU | `Exp5_GELU` |
| 6 | `swish` | Swish / SiLU | `Exp6_Swish` |
| 7 | `mish` | Mish | `Exp7_Mish` |
| 8 | `elish` | ELiSH | `Exp8_ELiSH` |
| 9 | `hard_elish` | HardELiSH | `Exp9_HardELiSH` |
| 10 | `logish` | Logish | `Exp10_Logish` |
| 11 | `smish` | Smish | `Exp11_Smish` |
| 12 | `tanhexp` | TanhExp | `Exp12_TanhExp` |

The custom functions (Mish, ELiSH, HardELiSH, Logish, Smish, TanhExp) are
defined in `src/train.py` and repeated verbatim in `src/evaluate.py` and the
notebooks; `get_activation(name)` is the single lookup used everywhere.

---

## 2. Repository layout

```
README.md
LICENSE                     MIT
CITATION.cff                cite the paper (preferred) or this repository
environment.md              software and hardware used
requirements.txt
src/
  preprocess.py             NIfTI -> normalised, cropped 4-channel .npy volumes + 80/20 split
  train.py                  one training run:  python train.py --act swish --name Exp6_Swish
  run_all.sh                the twelve runs of the main study, in order
  evaluate.py               per-patient metrics for every trained model + Wilcoxon summary
colab/                      the two Google Colab notebook exports, unmodified (provenance)
notebooks/
  multiseed_reliability.ipynb     4 activations x seeds 42/1337/2025   (Table 11, Supp. S1)
  crossvalidation_5fold.ipynb     4 activations x 5 folds, 369 patients (Table 12, Supp. S3)
  same_hardware_timing.ipynb      12 activations timed on one A100     (Table 9, Fig. 13, Supp. S2)
analysis/
  ci_and_corrected_pvalues.py     95 % CIs + Holm/BH-adjusted p-values from raw_predictions.csv
results/
  main_study/               Tables 5-8: raw_predictions.csv and its summaries
  multiseed/                Table 11
  crossval/                 Table 12
  timing/                   Table 9 / Figure 13
  training_logs/            per-epoch log.csv of the archived runs
figures/                    paper figures as rendered from these results
```

`src/preprocess.py`, `src/train.py` and `src/run_all.sh` are extracted from the
Colab export in `colab/BraTS2020_Training_colab_export.py` (the training script
and the run sequencer are `%%writefile` cells there); `src/evaluate.py` collects
the model definitions, the metric engine and the statistics pass from
`colab/BraTS_evaluate_colab_export.py`, leaving the figure cells in the export.
The executable code is unchanged — only comments and console messages were
tidied — and the unmodified exports are kept in `colab/` for provenance. The
code is authoritative for what the archived results describe.

---

## 3. Reproducing the main study

1. Obtain BraTS 2020 (§7) and place the `BraTS20_Training_*` folders under one
   directory.
2. Edit the paths at the top of `src/preprocess.py` (`RAW_PATH`,
   `PROCESSED_PATH`) and run it once. It writes `train/{images,masks}` and
   `val/{images,masks}` as `.npy` files, one per patient, and skips patients
   already processed.
3. Edit `DATA_DIR` and the `SAVE_DIR` root in `src/train.py`, then either run
   `bash src/run_all.sh` (all twelve runs, resumable — a run that finds
   `latest.pth` in its folder continues from that epoch) or a single run:

   ```bash
   python src/train.py --act swish --name Exp6_Swish --epochs 100
   ```

   Each run writes `best_model.pth`, `latest.pth` and `log.csv` to
   `<SAVE_DIR>/<name>/`.
4. Edit `RESULTS_DIR` / `DATA_DIR` in `src/evaluate.py` and run
   `python src/evaluate.py`. It scores every `Exp*/best_model.pth` on the
   validation set, appends to `raw_predictions.csv` (resumable), and writes
   `final_summary_table_extended.csv`. `--stats-only` recomputes the summary
   from an existing `raw_predictions.csv` without a GPU.
5. `python analysis/ci_and_corrected_pvalues.py` adds the 95 % confidence
   intervals and the multiple-comparison-corrected p-values (§5).

The three notebooks are self-contained Colab notebooks that reuse the
preprocessed volumes from step 2; each records its own configuration at the top.

---

## 4. Metrics

For each patient and each region (`NCR` = label 1, `ED` = label 2, `ET` = label 3,
`TC` = labels 1 ∪ 3, `WT` = labels 1 ∪ 2 ∪ 3), `compute_metrics` in
`src/evaluate.py` returns:

- **Dice** = 2 |P ∩ G| / (|P| + |G|)
- **HD95** = 95th percentile of the pooled symmetric surface distances, computed
  with the Euclidean distance transform on the 1 mm isotropic BraTS grid. When
  exactly one of prediction and ground truth is empty the case is penalised at
  **373 mm** (the diagonal of the 240 × 240 × 155 volume); when both are empty
  the case scores Dice 1, HD95 0.
- **Sensitivity** = |P ∩ G| / |G|, **Precision** = |P ∩ G| / |P|

---

## 5. Result files

All CSVs use the activation names of §1 and the patient file names of the
preprocessed data (`BraTS20_Training_XXX.npy`). Column names follow the pattern
`<Metric>_<Region>` with `Metric ∈ {Dice, HD95, Sens, Prec}` and
`Region ∈ {ET, TC, WT, NCR, ED}`.

### `results/main_study/` — Tables 5–8

| File | Rows | Contents |
|---|---|---|
| `raw_predictions.csv` | 876 = 12 × 73 | per-patient metrics of every activation on the 73 validation patients (`Experiment`, `Activation`, `Patient`, 20 metric columns) |
| `final_summary_table_extended.csv` | 12 | mean, SD and Wilcoxon mark (`*`/`**`) vs ReLU for every metric × region — written by `src/evaluate.py` |
| `ci_tables.csv` | 240 | mean, SD, percentile-bootstrap 95 % CI (5 000 resamples, seed 42) and normal-approximation 95 % CI for every activation × metric × region — the CIs printed in Tables 5–8 |
| `corrected_pvalues.csv` | 220 | Wilcoxon p-value of every non-baseline activation vs ReLU with Holm and Benjamini–Hochberg adjustment, applied within each metric family across its 11 × 5 = 55 comparisons |

`analysis/ci_and_corrected_pvalues.py` regenerates the last two files from
`raw_predictions.csv` (the archived `ci_tables.csv` is reproduced byte for
byte; the p-values agree to machine precision). The headline comparison —
Swish vs ReLU on NCR Dice, 0.676 vs 0.663 — has raw p = 4.4 × 10⁻⁴,
Holm p = 0.024, BH p = 0.012.

### `results/multiseed/` — Table 11

Four activations (ReLU, Swish, PReLU, TanhExp) retrained with seeds 42, 1337
and 2025, everything else unchanged, scored on the same 73 validation patients.

| File | Rows | Contents |
|---|---|---|
| `raw_predictions_multiseed.csv` | 876 = 4 × 3 × 73 | per-patient metrics, keyed by `Activation`, `Seed`, `Patient` |
| `per_seed_summary.csv` | 12 | mean over patients for each (activation, seed) |
| `across_seed_summary.csv` | 4 | mean ± SD **across the three per-seed means**, with `n_seeds` |
| `seed_significance.csv` | 180 | within-seed Wilcoxon test of each activation vs ReLU, paired by patient |

The seed-42 runs in this folder are **fresh trainings**, not the main-study
checkpoints: the multi-seed notebook additionally seeds the DataLoader
generator and workers, so data order differs from the main study and the
seed-42 numbers here do not coincide with Tables 5–8. Read `n_seeds` and the
per-seed file before comparing cells; three seeds is a small sample.

### `results/crossval/` — Table 12

The same four activations trained with seed 42 on each of five patient-level
folds (`KFold(n_splits=5, shuffle=True, random_state=42)` over the sorted list
of all 369 volumes; fold sizes 74 / 74 / 74 / 74 / 73). Every patient is scored
once, by the model that did not see it.

| File | Rows | Contents |
|---|---|---|
| `raw_predictions_cv.csv` | 1 476 = 4 × 369 | out-of-fold per-patient metrics, keyed by `Activation`, `Fold`, `Patient` |
| `per_fold_summary.csv` | 20 | mean over the held-out patients of each (activation, fold) |
| `across_fold_summary.csv` | 4 | mean ± SD across the five per-fold means, with `n_folds` |

Fold-level HD95 means are much larger than in the main study because the full
369-patient cohort includes cases with no enhancing tumour and other
empty-region outcomes: 102 of the 1 476 ET rows carry the 373 mm
empty-vs-non-empty penalty, against none of the 876 rows of the main-study
validation set. Compare like with like, or filter the penalty rows explicitly.

### `results/timing/` — Table 9, Figure 13, Supplementary Table S2

`timing_summary.csv`: for each of the twelve activations, median seconds per
training epoch (5 epochs, first discarded), mean inference milliseconds per
128³ volume (11 volumes, first discarded) and peak allocated VRAM, all measured
in one session on a single **NVIDIA A100-SXM4-40GB** with the training
pipeline of §1. No cross-GPU normalisation is involved.

### `results/training_logs/`

Per-epoch `log.csv` files written by the training scripts. Main-study logs have
20 columns (`epoch`, train/validation loss, mean and per-class Dice, precision
and sensitivity, epoch time in seconds, peak memory in GB); multi-seed logs have
the reduced 10-column format written by the notebook (`epoch, train_loss,
val_loss, mean_dice, dice_bg, dice_ncr, dice_ed, dice_et, time, mem_gb`). The
archived set is the one held locally by the authors at the time of writing;
runs without a log here were trained identically and their scores are in the
result CSVs above.

---

## 6. Figures

| File | Paper figure | Made from |
|---|---|---|
| `fig06_validation_dice_convergence.png` | Figure 6 — validation Dice over 100 epochs, 12 activations | `results/training_logs/main_study/*/log.csv` |
| `fig07_hd95_by_region.png` | Figure 7 — HD95 by region | `raw_predictions.csv` |
| `fig08_hd95_necrotic_core_boxplot.png` | Figure 8 — NCR HD95 box plot, 12 activations | `raw_predictions.csv` |
| `fig09_dice_radar_four_activations.png` | Figure 9 — Dice radar, ReLU / Swish / PReLU / TanhExp | `raw_predictions.csv` |
| `fig11_qualitative_patient057_slice82.png` | Figure 11 — patient 057, slice 82, ReLU vs Swish overlay | validation volumes + checkpoints |
| `fig13_same_hardware_efficiency.png` | Figure 13 — single-GPU efficiency dashboard | `results/timing/timing_summary.csv` + `raw_predictions.csv` |
| `supp_dice_boxplots_all_regions.png` | extended box plots of per-patient Dice for all five regions | `raw_predictions.csv` |

The plotting code is in `colab/BraTS_evaluate_colab_export.py`.

---

## 7. Data

**BraTS data are not redistributed here.** No imaging volumes, label maps or
preprocessed `.npy` files are included. The BraTS 2020 training data must be
obtained from the challenge organisers under the BraTS data-use conditions.

If you use the data, cite:

- B. H. Menze *et al.*, "The Multimodal Brain Tumor Image Segmentation
  Benchmark (BRATS)," *IEEE Transactions on Medical Imaging*, vol. 34, no. 10,
  pp. 1993–2024, 2015.
- S. Bakas *et al.*, "Advancing The Cancer Genome Atlas glioma MRI collections
  with expert segmentation labels and radiomic features," *Scientific Data*,
  vol. 4, 170117, 2017.
- S. Bakas *et al.*, "Identifying the Best Machine Learning Algorithms for
  Brain Tumor Segmentation, Progression Assessment, and Overall Survival
  Prediction in the BRATS Challenge," *arXiv:1811.02629*, 2018.

---

## 8. Trained checkpoints

Model weights are not in this repository (each `best_model.pth` is ≈ 92 MB;
44 trained models across the main study, the seed repeat and the folds).
Checkpoints are **available on request from the authors**.

---

## 9. Citing

Please cite the paper (§ top). `CITATION.cff` carries the paper as the
preferred citation and this repository as the software record.

The code and results in this repository are archived on Zenodo:
**[10.5281/zenodo.22803086](https://doi.org/10.5281/zenodo.22803086)** — this
DOI always resolves to the latest version. To cite the exact snapshot released
as `v1.0`, use [10.5281/zenodo.22803087](https://doi.org/10.5281/zenodo.22803087).

## 10. Contact

- Mushtaq Mahyoob Saleh (corresponding author) — <mushtaqkassem20@gmail.com>
- Mohamed A. A. Ahmed (repository maintainer) — <maaa82@sustech.edu>
