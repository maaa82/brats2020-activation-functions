"""BraTS 2020 preprocessing: z-score normalisation, 4-channel stacking, 128^3 centre crop, label 4 -> 3, 80/20 volume split.

Extracted verbatim from the Colab notebook export archived in colab/
BraTS2020_Training_colab_export.py. Paths point at the Google Drive layout used
for the reported runs; override DATA_DIR / PROCESSED_PATH for a local setup.
"""


import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

# === CONFIGURATION ===
# Using the path verified in previous steps
RAW_PATH = "/content/drive/MyDrive/MICCAI_BraTS2020_TrainingData"
PROCESSED_PATH = "/content/drive/MyDrive/BraTS2020_Preprocessed_128"

# === HELPER: Find files robustly ===
def find_file(folder, suffix_list):
    """Finds a file ending with any of the suffixes in the folder."""
    try:
        files = os.listdir(folder)
        for f in files:
            for suffix in suffix_list:
                if f.endswith(suffix):
                    return os.path.join(folder, f)
    except FileNotFoundError:
        return None
    return None

# === 1. IDENTIFY PATIENTS ===
# Look for folders starting with BraTS20_Training
search_pattern = os.path.join(RAW_PATH, "BraTS20_Training_*")
patient_folders = sorted(glob.glob(search_pattern))

valid_patients = []
print(f"🔍 Scanning {len(patient_folders)} folders for complete data...")

for p in patient_folders:
    # We need all 4 modalities + seg
    has_seg = find_file(p, ["seg.nii", "seg.nii.gz"])
    if has_seg:
        valid_patients.append(p)

print(f"✅ Found {len(valid_patients)} valid patients.")

if len(valid_patients) == 0:
    raise ValueError("❌ No valid patient folders found. Check your RAW_PATH.")

# === 2. SETUP OUTPUT DIRS ===
for split in ['train', 'val']:
    os.makedirs(os.path.join(PROCESSED_PATH, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_PATH, split, 'masks'), exist_ok=True)

# Split 80% Train, 20% Val
val_count = int(len(valid_patients) * 0.2)
train_patients = valid_patients[val_count:]
val_patients = valid_patients[:val_count]

# === 3. PREPROCESSING FUNCTION ===
def z_score_normalize(image):
    mask = image > 0
    if mask.sum() == 0: return image
    mean = image[mask].mean()
    std = image[mask].std()
    return (image - mean) / (std + 1e-8)

def process_batch(patient_list, split):
    print(f"🚀 Processing {split} set ({len(patient_list)} patients)...")

    for folder in tqdm(patient_list):
        patient_id = os.path.basename(folder)

        # Output filenames
        out_img = os.path.join(PROCESSED_PATH, split, 'images', f"{patient_id}.npy")
        out_mask = os.path.join(PROCESSED_PATH, split, 'masks', f"{patient_id}.npy")

        # Skip if already exists (Resumable preprocessing!)
        if os.path.exists(out_img) and os.path.exists(out_mask):
            continue

        # Find files
        f_t1 = find_file(folder, ["t1.nii", "t1.nii.gz"])
        f_t1ce = find_file(folder, ["t1ce.nii", "t1ce.nii.gz"])
        f_t2 = find_file(folder, ["t2.nii", "t2.nii.gz"])
        f_flair = find_file(folder, ["flair.nii", "flair.nii.gz"])
        f_seg = find_file(folder, ["seg.nii", "seg.nii.gz"])

        if not all([f_t1, f_t1ce, f_t2, f_flair, f_seg]):
            continue # Skip incomplete

        try:
            # Load
            v_t1 = nib.load(f_t1).get_fdata().astype(np.float32)
            v_t1ce = nib.load(f_t1ce).get_fdata().astype(np.float32)
            v_t2 = nib.load(f_t2).get_fdata().astype(np.float32)
            v_flair = nib.load(f_flair).get_fdata().astype(np.float32)
            v_seg = nib.load(f_seg).get_fdata().astype(np.uint8)

            # Fix Labels (4 -> 3)
            v_seg[v_seg == 4] = 3

            # Normalize
            v_t1 = z_score_normalize(v_t1)
            v_t1ce = z_score_normalize(v_t1ce)
            v_t2 = z_score_normalize(v_t2)
            v_flair = z_score_normalize(v_flair)

            # Stack 4 Channels: [FLAIR, T1, T1ce, T2]
            combined = np.stack([v_flair, v_t1, v_t1ce, v_t2], axis=-1)

            # Crop Center 128 (BraTS is 240x240x155)
            # Center X: 120 -> 56:184
            # Center Y: 120 -> 56:184
            # Center Z: 77 -> 13:141
            combined = combined[56:184, 56:184, 13:141, :]
            v_seg = v_seg[56:184, 56:184, 13:141]

            # Save
            np.save(out_img, combined)
            np.save(out_mask, v_seg)

        except Exception as e:
            print(f"Error processing {patient_id}: {e}")

# Run It
if __name__ == "__main__":
    process_batch(train_patients, 'train')
    process_batch(val_patients, 'val')
    print("\n✅ Preprocessing Complete!")
