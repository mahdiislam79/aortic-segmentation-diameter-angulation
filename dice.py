import os
import numpy as np
import torch
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from monai.utils.enums import MetricReduction
import nibabel as nib
import pandas as pd
from tqdm import tqdm

# ----- SETTINGS -----
gt_dir = '/gpfs/data/fs71894/mahdi_i/Multimodal_V2/test_labels'
pred_dir = '/gpfs/data/fs71894/mahdi_i/Multimodal_V2/enrique_inference/post_complete/mednext'
output_csv = '/gpfs/data/fs71894/mahdi_i/Multimodal_V2/enrique_inference/post_complete/per_class_dice_mednext.csv'
class_labels = {1: 'Aorta', 2: 'Left Iliac', 3: 'Right Iliac'}

# ----- INIT METRIC -----
dice_metric = DiceMetric(include_background=False, reduction=MetricReduction.MEAN)

results = []
total_cases = 0
skipped_cases = 0
num_classes = 4  # background + 3 target classes

print("🔍 Starting Dice score evaluation...\n")

# ----- PROCESS FILES -----
for filename in tqdm(sorted(os.listdir(gt_dir))):
    if not filename.endswith('.nii.gz'):
        continue

    case_id = filename.replace('.nii.gz', '')
    gt_path = os.path.join(gt_dir, filename)
    # pred_path = os.path.join(pred_dir, filename)
    pred_path = os.path.join(pred_dir, f"postcomplete_{case_id}.nii.gz")

    if not os.path.exists(pred_path):
        print(f"⚠️  Missing prediction for {case_id}")
        skipped_cases += 1
        continue

    # Load and convert to tensors
    gt = torch.from_numpy(nib.load(gt_path).get_fdata()).long()
    pred = torch.from_numpy(nib.load(pred_path).get_fdata()).long()

    if gt.shape != pred.shape:
        print(f"⚠️  Shape mismatch in {case_id}")
        skipped_cases += 1
        continue

    # Add batch and channel dimensions
    gt = gt.unsqueeze(0).unsqueeze(0)
    pred = pred.unsqueeze(0).unsqueeze(0)

    # One-hot encode
    gt_onehot = one_hot(gt, num_classes=num_classes)
    pred_onehot = one_hot(pred, num_classes=num_classes)

    # Compute Dice per class (excluding background)
    dice = dice_metric(y_pred=pred_onehot, y=gt_onehot)
    dice = dice.cpu().numpy().squeeze()
    dice_metric.reset()

    total_cases += 1
    print(f"🧪 {case_id} - Dice scores:")
    if dice.ndim == 1:
        for i, score in enumerate(dice):
            label = class_labels[i + 1]  # skip background
            score_value = float(score) if not np.isnan(score) else None
            if score_value is not None:
                print(f"   ✅ {label}: {score_value:.4f}")
            else:
                print(f"   ❌ {label}: NaN (class missing in GT or prediction)")
            results.append({
                'Case': case_id,
                'Class': label,
                'Dice': score_value
            })
    else:
        print(f"⚠️  Unexpected Dice shape for {case_id}: {dice.shape}")
        skipped_cases += 1

# ----- SAVE TO CSV -----
df = pd.DataFrame(results)
df.dropna(subset=['Dice'], inplace=True)
df.to_csv(output_csv, index=False)

# ----- SUMMARY -----
avg_per_class = df.groupby("Class")["Dice"].mean()
overall_avg = df["Dice"].mean()

print("\n📊 Average Dice Score per Class:")
for cls, val in avg_per_class.items():
    print(f"   {cls}: {val:.4f}")

print(f"\n🎯 Overall Average Dice Score: {overall_avg:.4f}")
print(f"\n✅ Dice scores saved to: {output_csv}")
print(f"📁 Processed {total_cases} cases, skipped {skipped_cases} cases.\n")
