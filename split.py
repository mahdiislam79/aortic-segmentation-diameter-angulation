import os
import random

# --- Config ---
image_dir = "/gpfs/data/fs71894/mahdi_i/ActiveLearning/dataset/images"  
output_dir = "/gpfs/data/fs71894/mahdi_i/ActiveLearning/dataset"  # <<< Can be changed

# --- Get all image files ---
all_images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.nii', '.nii.gz'))])
assert len(all_images) == 171, f"Expected 171 images, found {len(all_images)}"

# --- Split into train and test ---
random.seed(43)
test_images = random.sample(all_images, 26)
train_images = [f for f in all_images if f not in test_images]
train_images = random.sample(train_images, 145)  

# --- Helper to write to file ---
def write_list_to_file(filename, items):
    with open(filename, "w") as f:
        for item in items:
            f.write(f"{item}\n")

# --- Write files ---
write_list_to_file(os.path.join(output_dir, "train_filenames.txt"), train_images)
write_list_to_file(os.path.join(output_dir, "test_filenames.txt"), test_images)

write_list_to_file(os.path.join(output_dir, "train_paths.txt"), [os.path.join(image_dir, f) for f in train_images])
write_list_to_file(os.path.join(output_dir, "test_paths.txt"), [os.path.join(image_dir, f) for f in test_images])

print("Train/test split completed. Files saved.")
