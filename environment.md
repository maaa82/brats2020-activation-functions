# Environment

## Software

All training and evaluation ran in **Google Colab** notebooks (the exports are in
`colab/`). The code relies on:

| Component | Requirement |
|---|---|
| Python | 3.10 or later |
| PyTorch | ≥ 2.6 — the scripts use `torch.amp.GradScaler('cuda')` / `torch.amp.autocast('cuda')`, and resume checkpoints (which hold optimiser and scheduler state) are loaded with `weights_only=False`, as PyTorch ≥ 2.6 requires |
| numpy, pandas, scipy | metrics (`scipy.ndimage.distance_transform_edt`), statistics (`scipy.stats.wilcoxon`) |
| nibabel | reading the BraTS NIfTI volumes in `src/preprocess.py` |
| scikit-learn | `KFold` in the cross-validation notebook |
| matplotlib, seaborn, tqdm | figures and progress bars |

Exact package versions were those of the Colab runtime at the time of each
run (February 2026 for the main study, June 2026 for the three follow-up
experiments) and were not pinned. `requirements.txt` lists the packages
without version pins; the analysis script in `analysis/` has been checked
with pandas 3.0 / scipy 1.17 / numpy 2.x and reproduces the archived files.

## Hardware

- **Main study and follow-ups:** Google Colab GPU sessions (NVIDIA A100 40 GB
  and L4). A 128³ × 4-channel volume at batch 2 needs ≈ 12–13 GB of GPU
  memory in mixed precision (`max_mem_gb` in the training logs); one epoch
  of the main study took ≈ 255 s on the session that produced the archived
  logs.
- **Timing experiment (`results/timing/`):** all twelve activations were
  timed in a single session on one **NVIDIA A100-SXM4-40GB**, so the
  computational-cost comparison in the paper involves no cross-GPU
  normalisation.

## Install (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or the CUDA build for your driver
pip install -r requirements.txt
```

The scripts carry the Google Drive paths used in Colab at the top of each
file; change them to local paths before running.
