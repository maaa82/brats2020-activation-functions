"""Evaluation for the 12-activation study: per-patient metrics and paired statistics.

The model, activation and dataset definitions, the metric engine
(`compute_metrics`), the evaluation loop (`run_evaluation`) and the statistics
pass (`summarise_with_wilcoxon`) are taken from the Colab notebook export
archived in colab/BraTS_evaluate_colab_export.py, which also contains the
figure-generation cells.  The computation is unchanged; the Google Drive paths
are collected at the top of the file, console messages were tidied, and nothing
runs on import.

Outputs (main study):
  raw_predictions.csv               one row per (activation, patient): Dice, HD95,
                                    sensitivity and precision for ET, TC, WT, NCR, ED
  final_summary_table_extended.csv  per-activation mean, SD and Wilcoxon significance
                                    mark vs ReLU for every metric x region
"""
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ==========================================
# PATHS (Google Drive layout used for the reported runs)
# ==========================================
RESULTS_DIR = '/content/drive/MyDrive/BraTS_Results'
DATA_DIR = '/content/drive/MyDrive/BraTS2020_Preprocessed_128'
SAVE_RAW_PATH = os.path.join(RESULTS_DIR, 'raw_predictions.csv')
SAVE_SUMMARY_PATH = os.path.join(RESULTS_DIR, 'final_summary_table_extended.csv')
BASELINE = 'relu'  # The baseline to compare against

# ==========================================
# 1. REPLICATE MODEL ARCHITECTURE
# ==========================================
# (Must match train.py exactly to load weights)

class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(F.softplus(x))
class ELiSH(nn.Module):
    def forward(self, x): return F.elu(x) * torch.sigmoid(x)
class HardELiSH(nn.Module):
    def forward(self, x): return F.elu(x) * F.hardsigmoid(x)
class Logish(nn.Module):
    def forward(self, x): return x * torch.log(1 + torch.sigmoid(x))
class Smish(nn.Module):
    def forward(self, x): return x * torch.tanh(torch.log(1 + torch.sigmoid(x)))
class TanhExp(nn.Module):
    def forward(self, x): return x * torch.tanh(torch.exp(torch.clamp(x, max=20)))

def get_activation(name):
    name = name.lower()
    if name == 'relu': return nn.ReLU(inplace=True)
    if name == 'leaky_relu': return nn.LeakyReLU(0.01, inplace=True)
    if name == 'prelu': return nn.PReLU()
    if name == 'elu': return nn.ELU(inplace=True)
    if name == 'gelu': return nn.GELU()
    if name == 'swish': return nn.SiLU(inplace=True)
    if name == 'mish': return Mish()
    if name == 'elish': return ELiSH()
    if name == 'hard_elish': return HardELiSH()
    if name == 'logish': return Logish()
    if name == 'smish': return Smish()
    if name == 'tanhexp': return TanhExp()
    raise ValueError(f"Activation {name} not supported.")

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, act_name):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.act = get_activation(act_name)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.skip = nn.Conv3d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x):
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + self.skip(x))

class ImprovedUNet3D(nn.Module):
    def __init__(self, in_c, out_c, act_name):
        super().__init__()
        self.enc1 = ResidualBlock(in_c, 32, act_name); self.pool = nn.MaxPool3d(2)
        self.enc2 = ResidualBlock(32, 64, act_name)
        self.enc3 = ResidualBlock(64, 128, act_name)
        self.enc4 = ResidualBlock(128, 256, act_name)
        self.bottleneck = ResidualBlock(256, 512, act_name)
        self.up4 = nn.ConvTranspose3d(512, 256, 2, 2); self.dec4 = ResidualBlock(512, 256, act_name)
        self.up3 = nn.ConvTranspose3d(256, 128, 2, 2); self.dec3 = ResidualBlock(256, 128, act_name)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, 2); self.dec2 = ResidualBlock(128, 64, act_name)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, 2); self.dec1 = ResidualBlock(64, 32, act_name)
        self.out = nn.Conv3d(32, out_c, 1)
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4=self.dec4(torch.cat([self.up4(b), e4], 1)); d3=self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2=self.dec2(torch.cat([self.up2(d3), e2], 1)); d1=self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)

class BraTSDataset(Dataset):
    """Returns the file name as a third element so that results can be paired by patient."""
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir; self.mask_dir = mask_dir
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = np.load(os.path.join(self.img_dir, self.files[idx])).astype(np.float32)
        mask = np.load(os.path.join(self.mask_dir, self.files[idx])).astype(np.longlong)
        return (torch.from_numpy(img).permute(3, 2, 0, 1),
                torch.from_numpy(mask).permute(2, 0, 1),
                self.files[idx])

# ==========================================
# 2. METRIC COMPUTATION
# ==========================================
def compute_metrics(pred, gt):
    """Dice, HD95 (mm, 95th percentile of symmetric surface distances via the
    Euclidean distance transform; empty-vs-non-empty penalised at 373 mm),
    sensitivity and precision for one binary region."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0, 0.0, 1.0, 1.0

    intersection = (pred & gt).sum()
    dice = (2. * intersection) / (pred.sum() + gt.sum() + 1e-5)
    sensitivity = intersection / (gt.sum() + 1e-5)
    precision = intersection / (pred.sum() + 1e-5)

    try:
        if pred.sum() > 0 and gt.sum() > 0:
            dt_gt = distance(1-gt)
            dt_pred = distance(1-pred)
            hd95 = np.percentile(np.hstack([dt_gt[pred], dt_pred[gt]]), 95)
        else:
            hd95 = 373.0
    except Exception:
        hd95 = 373.0

    return dice, hd95, sensitivity, precision

# ==========================================
# 3. EVALUATION LOOP
# ==========================================
def run_evaluation():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    val_ds = BraTSDataset(f"{DATA_DIR}/val/images", f"{DATA_DIR}/val/masks")
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # Resume logic
    if os.path.exists(SAVE_RAW_PATH):
        print(f"Resuming from existing file: {SAVE_RAW_PATH}")
        try:
            df_existing = pd.read_csv(SAVE_RAW_PATH)
            processed_pairs = set(zip(df_existing['Activation'], df_existing['Patient']))
        except pd.errors.EmptyDataError:
            processed_pairs = set()
    else:
        print("Starting fresh evaluation.")
        processed_pairs = set()

    experiments = sorted(glob.glob(os.path.join(RESULTS_DIR, "Exp*")))
    experiments = [p for p in experiments if os.path.isdir(p)]
    print(f"Evaluating {len(experiments)} models...")

    for exp_path in experiments:
        exp_name = os.path.basename(exp_path)
        parts = exp_name.split('_', 1)
        if len(parts) < 2:
            continue
        raw_name = parts[1].lower()
        if "leaky" in raw_name: act_name = "leaky_relu"
        elif "hard" in raw_name: act_name = "hard_elish"
        elif "tanh" in raw_name: act_name = "tanhexp"
        else: act_name = raw_name

        model_file = os.path.join(exp_path, "best_model.pth")
        if not os.path.exists(model_file):
            continue

        print(f"Testing: {act_name}")
        model = ImprovedUNet3D(4, 4, act_name).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()

        new_results = []
        with torch.no_grad():
            for x, y, fname in tqdm(val_dl, leave=False):
                patient_id = fname[0]
                if (act_name, patient_id) in processed_pairs:
                    continue
                x = x.to(device)
                y_np = y.numpy()[0]
                with torch.amp.autocast('cuda'):
                    pred = model(x)
                    p_cls = pred.argmax(dim=1).cpu().numpy()[0]

                regions = {
                    'WT': (p_cls > 0, y_np > 0),
                    'TC': ((p_cls==1)|(p_cls==3), (y_np==1)|(y_np==3)),
                    'ET': (p_cls==3, y_np==3),
                    'NCR': (p_cls==1, y_np==1),
                    'ED': (p_cls==2, y_np==2)
                }
                row = {'Experiment': exp_name, 'Activation': act_name, 'Patient': patient_id}
                for region_name, (p_mask, t_mask) in regions.items():
                    d, h, s, p = compute_metrics(p_mask.astype(np.uint8), t_mask.astype(np.uint8))
                    row[f'Dice_{region_name}'] = d
                    row[f'HD95_{region_name}'] = h
                    row[f'Sens_{region_name}'] = s
                    row[f'Prec_{region_name}'] = p
                new_results.append(row)

                if len(new_results) >= 10:
                    pd.DataFrame(new_results).to_csv(SAVE_RAW_PATH, mode='a',
                                                     header=not os.path.exists(SAVE_RAW_PATH), index=False)
                    new_results = []
        if new_results:
            pd.DataFrame(new_results).to_csv(SAVE_RAW_PATH, mode='a',
                                             header=not os.path.exists(SAVE_RAW_PATH), index=False)

    print("Evaluation complete.")
    return pd.read_csv(SAVE_RAW_PATH) if os.path.exists(SAVE_RAW_PATH) else None

# ==========================================
# 4. STATISTICS: MEAN +/- SD AND WILCOXON SIGNED-RANK TEST VS RELU
# ==========================================
def summarise_with_wilcoxon(df_results, baseline=BASELINE):
    """Per-activation mean and SD for every metric x region, plus a significance
    mark from a Wilcoxon signed-rank test against the baseline, paired by patient
    ('*' p < 0.05, '**' p < 0.01).  No multiple-comparison correction is applied
    here; see analysis/ci_and_corrected_pvalues.py for Holm / BH adjustment."""
    activations = sorted(df_results['Activation'].unique())
    metrics_map = {m: ['ET', 'TC', 'WT', 'NCR', 'ED'] for m in ['Dice', 'Sens', 'Prec', 'HD95']}
    summary_rows = []
    for act in activations:
        subset = df_results[df_results['Activation'] == act]
        row_data = {'Activation': act}
        for metric_type, regions in metrics_map.items():
            for region in regions:
                col_name = f"{metric_type}_{region}"
                values = subset[col_name].dropna()
                row_data[f"{col_name}_mean"] = values.mean()
                row_data[f"{col_name}_std"] = values.std()
                sig_mark = ""
                if act != baseline:
                    base_subset = df_results[df_results['Activation'] == baseline][['Patient', col_name]]
                    merged = pd.merge(subset[['Patient', col_name]], base_subset,
                                      on='Patient', suffixes=('', '_base'))
                    curr_vals = merged[col_name]
                    base_vals = merged[f"{col_name}_base"]
                    if len(curr_vals) > 0 and not (curr_vals == base_vals).all():
                        try:
                            _, p = wilcoxon(curr_vals, base_vals)
                            if p < 0.01: sig_mark = "**"
                            elif p < 0.05: sig_mark = "*"
                        except Exception:
                            pass
                row_data[f"{col_name}_sig"] = sig_mark
        summary_rows.append(row_data)
    return pd.DataFrame(summary_rows)


def print_table(df, metric_prefix, title):
    print(f"\n{'='*102}\n {title}\n{'='*102}")
    header = f"{'Activation':<15} | {'ET':<18} | {'TC':<18} | {'WT':<18} | {'NCR':<18} | {'ED':<18}"
    print(header); print("-" * len(header))
    for _, row in df.iterrows():
        line = f"{row['Activation']:<15}"
        for r in ['ET', 'TC', 'WT', 'NCR', 'ED']:
            val_str = f"{row[f'{metric_prefix}_{r}_mean']:.3f} ± {row[f'{metric_prefix}_{r}_std']:.3f}{row[f'{metric_prefix}_{r}_sig']}"
            line += f" | {val_str:<18}"
        print(line)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate the 12 trained models, or only re-summarise an existing raw_predictions.csv.")
    ap.add_argument('--stats-only', action='store_true', help='skip inference; recompute the summary table from raw_predictions.csv')
    ap.add_argument('--raw', default=SAVE_RAW_PATH)
    ap.add_argument('--summary', default=SAVE_SUMMARY_PATH)
    args = ap.parse_args()

    df = pd.read_csv(args.raw) if args.stats_only else run_evaluation()
    if df is None or df.empty:
        raise SystemExit("No results found.")
    df_summary = summarise_with_wilcoxon(df)
    print_table(df_summary, "Dice", "DICE SIMILARITY COEFFICIENT (Higher is Better)")
    print_table(df_summary, "Sens", "SENSITIVITY / RECALL (Higher is Better)")
    print_table(df_summary, "Prec", "PRECISION / PPV (Higher is Better)")
    print_table(df_summary, "HD95", "95% HAUSDORFF DISTANCE (Lower is Better)")
    df_summary.to_csv(args.summary, index=False)
    print(f"\nExtended summary table saved to: {args.summary}")
