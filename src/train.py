"""Training script for the 12-activation study (ImprovedUNet3D, CE + soft Dice, AdamW, cosine schedule, 100 epochs, seed 42).

Extracted from the Colab notebook export archived in colab/
BraTS2020_Training_colab_export.py (code unchanged; comments tidied). Paths point
at the Google Drive layout used for the reported runs; set DATA_DIR and the
SAVE_DIR root for a local setup.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import random

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = "/content/drive/MyDrive/BraTS2020_Preprocessed_128"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 2. ACTIVATIONS (12 functions)
# ==========================================
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
    # Piecewise-linear / classical
    if name == 'relu': return nn.ReLU(inplace=True)
    if name == 'leaky_relu': return nn.LeakyReLU(0.01, inplace=True)
    if name == 'prelu': return nn.PReLU()
    if name == 'elu': return nn.ELU(inplace=True)
    # Smooth, self-gated
    if name == 'gelu': return nn.GELU()
    if name == 'swish': return nn.SiLU(inplace=True)
    if name == 'mish': return Mish()
    # Recent smooth variants
    if name == 'elish': return ELiSH()
    if name == 'hard_elish': return HardELiSH()
    if name == 'logish': return Logish()
    if name == 'smish': return Smish()
    if name == 'tanhexp': return TanhExp()
    raise ValueError(f"Activation {name} not supported.")

# ==========================================
# 3. DATASET
# ==========================================
class BraTSDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            img = np.load(os.path.join(self.img_dir, self.files[idx])).astype(np.float32)
            mask = np.load(os.path.join(self.mask_dir, self.files[idx])).astype(np.longlong)
            img = torch.from_numpy(img).permute(3, 2, 0, 1) # (C, D, H, W)
            mask = torch.from_numpy(mask).permute(2, 0, 1)  # (D, H, W)
            return img, mask
        except Exception:
            # Fallback for an unreadable file: return an all-zero volume
            return torch.zeros((4, 128, 128, 128)), torch.zeros((128, 128, 128), dtype=torch.long)

# ==========================================
# 4. MODEL (Improved 3D U-Net)
# ==========================================
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
        # Encoder
        self.enc1 = ResidualBlock(in_c, 32, act_name); self.pool = nn.MaxPool3d(2)
        self.enc2 = ResidualBlock(32, 64, act_name)
        self.enc3 = ResidualBlock(64, 128, act_name)
        self.enc4 = ResidualBlock(128, 256, act_name)
        self.bottleneck = ResidualBlock(256, 512, act_name)
        # Decoder
        self.up4 = nn.ConvTranspose3d(512, 256, 2, 2); self.dec4 = ResidualBlock(512, 256, act_name)
        self.up3 = nn.ConvTranspose3d(256, 128, 2, 2); self.dec3 = ResidualBlock(256, 128, act_name)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, 2); self.dec2 = ResidualBlock(128, 64, act_name)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, 2); self.dec1 = ResidualBlock(64, 32, act_name)
        self.out = nn.Conv3d(32, out_c, 1)

    def forward(self, x):
        e1=self.enc1(x)
        e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2))
        e4=self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4=self.dec4(torch.cat([self.up4(b), e4], 1))
        d3=self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2=self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1=self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)

# ==========================================
# 5. TRAINING ENGINE
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--act', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    seed_everything()
    SAVE_DIR = f"/content/drive/MyDrive/BraTS_Results/{args.name}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Started: {args.name} | activation: {args.act}")

    # Model, optimiser, loss, scheduler
    model = ImprovedUNet3D(4, 4, args.act).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler('cuda')
    ce_loss = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # --- Resume from latest.pth if present ---
    ckpt_path = os.path.join(SAVE_DIR, "latest.pth")
    start_epoch = 0; history = []

    if os.path.exists(ckpt_path):
        print("Resuming from checkpoint...")
        # weights_only=False: the checkpoint holds optimiser and scheduler state (PyTorch >= 2.6)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch']
        history = ckpt.get('history', [])

        # Restore the scheduler state so the learning rate continues where it left off
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        else:
            # Checkpoint without scheduler state: fast-forward the scheduler
            print("   (re-aligning scheduler)")
            for _ in range(start_epoch): scheduler.step()

    # Data loaders
    train_dl = DataLoader(BraTSDataset(f"{DATA_DIR}/train/images", f"{DATA_DIR}/train/masks"),
                          batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(BraTSDataset(f"{DATA_DIR}/val/images", f"{DATA_DIR}/val/masks"),
                        batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    # --- Main loop ---
    for epoch in range(start_epoch, args.epochs):
        torch.cuda.reset_peak_memory_stats()
        epoch_start = time.time()

        # 1. Train one epoch
        model.train()
        t_loss = 0
        for x, y in tqdm(train_dl, desc=f"Ep {epoch+1} Train", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model(x)
                # Combined loss: unweighted cross-entropy + soft Dice
                p_soft = F.softmax(pred, 1)
                y_oh = F.one_hot(y, 4).permute(0,4,1,2,3).float()
                inter = (p_soft * y_oh).sum((2,3,4))
                union = p_soft.sum((2,3,4)) + y_oh.sum((2,3,4))
                dice_loss = 1 - ((2*inter+1e-5)/(union+1e-5)).mean()
                loss = ce_loss(pred, y) + dice_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            t_loss += loss.item()

        train_loss_avg = t_loss / len(train_dl)

        # 2. Validate
        model.eval()
        v_loss = 0
        dices = np.zeros(4)
        precisions = np.zeros(4)
        sensitivities = np.zeros(4)

        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"Ep {epoch+1} Val", leave=False):
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda'):
                    pred = model(x)
                    # Validation loss
                    p_soft = F.softmax(pred, 1)
                    y_oh = F.one_hot(y, 4).permute(0,4,1,2,3).float()
                    inter = (p_soft * y_oh).sum((2,3,4))
                    union = p_soft.sum((2,3,4)) + y_oh.sum((2,3,4))
                    dice_loss = 1 - ((2*inter+1e-5)/(union+1e-5)).mean()
                    loss = ce_loss(pred, y) + dice_loss
                v_loss += loss.item()

                # Per-class validation metrics
                p_cls = pred.argmax(1)
                for c in range(4):
                    p = (p_cls==c).float(); t = (y==c).float()
                    tp = (p * t).sum()
                    fp = (p * (1-t)).sum()
                    fn = ((1-p) * t).sum()

                    dices[c] += ((2*tp + 1e-5)/(2*tp + fp + fn + 1e-5)).item()
                    precisions[c] += ((tp + 1e-5)/(tp + fp + 1e-5)).item()
                    sensitivities[c] += ((tp + 1e-5)/(tp + fn + 1e-5)).item()

        # 3. Epoch statistics
        val_loss_avg = v_loss / len(val_dl)
        dices /= len(val_dl)
        precisions /= len(val_dl)
        sensitivities /= len(val_dl)

        mean_dice = dices[1:].mean()
        mean_prec = precisions[1:].mean()
        mean_sens = sensitivities[1:].mean()

        duration = time.time() - epoch_start
        max_mem = torch.cuda.max_memory_allocated() / 1e9 # GB

        print(f"Ep {epoch+1}: T_Loss={train_loss_avg:.4f} | V_Loss={val_loss_avg:.4f} | Dice={mean_dice:.4f} | Mem={max_mem:.2f}GB")

        # 4. Save checkpoint, best model and log
        history.append([epoch+1, train_loss_avg, val_loss_avg, mean_dice, *dices, mean_prec, *precisions, mean_sens, *sensitivities, duration, max_mem])

        # Resume checkpoint (includes scheduler state)
        torch.save({
            'epoch': epoch+1,
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), # Saving Scheduler State
            'history': history
        }, ckpt_path)

        # Best model = highest mean Dice over the three tumour classes
        if mean_dice >= max([h[3] for h in history]):
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))

        # Per-epoch log
        cols = ['epoch','train_loss','val_loss','mean_dice',
                'dice_bg','dice_necro','dice_edema','dice_enh',
                'mean_prec','prec_bg','prec_necro','prec_edema','prec_enh',
                'mean_sens','sens_bg','sens_necro','sens_edema','sens_enh',
                'time','max_mem_gb']
        pd.DataFrame(history, columns=cols).to_csv(os.path.join(SAVE_DIR, "log.csv"), index=False)

        scheduler.step()

    print("Experiment complete.")
