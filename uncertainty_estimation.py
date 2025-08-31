import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm

cases_with_no_iliac = [15, 20, 26, 30, 50, 60, 63, 74, 75, 78, 86, 101, 107, 111, 112, 123, 130, 139, 143, 146, 151]

# add the cases with no iliac to the list using the number of the case [15 -> case_015.npz]
cases_with_no_iliac = [f'case_{str(i).zfill(3)}' for i in cases_with_no_iliac]

# ----- SETTINGS -----
source_folder = '/gpfs/data/fs71894/mahdi_i/nnunet_accessroute/inference/inference_stage1/labelsTr'
filtered_folder = '/gpfs/data/fs71894/mahdi_i/nnunet_accessroute/inference/inference_stage1/proba'
output_path = '/gpfs/data/fs71894/mahdi_i/nnunet_accessroute/inference/inference_stage1/uncertainty_summary_filtered_cases.csv'

# File & model settings
npz_key = 'probabilities'
foreground_threshold = 0.01
uncertain_range = (0.4, 0.6)
confident_threshold = 0.8
very_confident_threshold = 0.9
class_names = ['Aorta', 'Left Iliac', 'Right Iliac']

# ----- PREPARE -----
os.makedirs(filtered_folder, exist_ok=True)
npz_files = [f for f in os.listdir(source_folder) if f.endswith('.npz')]
# Filter out cases with no iliac
npz_files = [f for f in npz_files if os.path.splitext(f)[0] not in cases_with_no_iliac]

print(f"Found {len(npz_files)} .npz files in source folder.\n")

# ----- COPY FILES TO NEW FOLDER -----
for file in npz_files:
    src_path = os.path.join(source_folder, file)
    dst_path = os.path.join(filtered_folder, file)
    shutil.copy2(src_path, dst_path)

print(f"✅ Copied all .npz files to: {filtered_folder}\n")

# ----- UNCERTAINTY CALCULATION -----
all_metrics = []

for file in tqdm(npz_files, desc="Processing copied files"):
    case_id = os.path.splitext(file)[0]

    try:
        data = np.load(os.path.join(filtered_folder, file))
        probs = data[npz_key]
        num_classes = probs.shape[0]

        if num_classes == 4:
            foreground_probs = probs[1:]
        elif num_classes == 3:
            foreground_probs = probs
        else:
            print(f"Skipping {file}: Unexpected number of classes = {num_classes}")
            continue

        for i in range(foreground_probs.shape[0]):
            probs_i = foreground_probs[i].flatten()
            probs_i = probs_i[probs_i > foreground_threshold]

            if len(probs_i) == 0:
                continue

            mean_prob = np.mean(probs_i)
            std_prob = np.std(probs_i)
            entropy = -(np.clip(probs_i, 1e-6, 1 - 1e-6) * np.log(np.clip(probs_i, 1e-6, 1 - 1e-6)) + 
                        (1 - probs_i) * np.log(np.clip(1 - probs_i, 1e-6, 1 - 1e-6)))
            mean_entropy = np.mean(entropy)
            frac_uncertain = np.sum((probs_i > uncertain_range[0]) & (probs_i < uncertain_range[1])) / len(probs_i)
            frac_confident = np.sum(probs_i > confident_threshold) / len(probs_i)
            frac_very_confident = np.sum(probs_i > very_confident_threshold) / len(probs_i)

            all_metrics.append({
                'Case': case_id,
                'Class': class_names[i],
                'Mean Probability': mean_prob,
                'Std Probability': std_prob,
                'Mean Entropy': mean_entropy,
                '% Uncertain [0.4-0.6]': frac_uncertain * 100,
                '% Confident > 0.8': frac_confident * 100,
                '% Very Confident > 0.9': frac_very_confident * 100
            })

    except Exception as e:
        print(f"⚠️ Error processing {file}: {e}")
        continue

# ----- SAVE RESULTS -----
df_all = pd.DataFrame(all_metrics)
df_all.to_csv(output_path, index=False)

print(f"\n✅ Saved uncertainty summary to: {output_path}")
