import os
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, binary_fill_holes
from skimage.morphology import ball
from tqdm import tqdm

# === Directories ===
input_dir = "/gpfs/data/fs71894/mahdi_i/ActiveLearning/dataset/nnunet_data/nnUNet_raw/Dataset012_iter2/labelsTr_copy"
output_dir = "/gpfs/data/fs71894/mahdi_i/ActiveLearning/dataset/nnunet_data/nnUNet_raw/Dataset012_iter2/labelsTr"
os.makedirs(output_dir, exist_ok=True)

# === Functions ===
def load_nifti(file_path):
    """Load a NIfTI file."""
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata(), nifti_img.affine, nifti_img.header

def clean_binary_mask(mask_bin, struct_size=2): # Increased default ball size
    """Apply morphological closing, opening, and fill holes on a binary mask."""
    struct = ball(struct_size) # Use a larger structuring element for better smoothing/filling
    mask_cleaned = binary_closing(mask_bin, structure=struct)
    mask_cleaned = binary_opening(mask_cleaned, structure=struct)
    mask_cleaned = binary_fill_holes(mask_cleaned)
    # Ensure the output type matches the expected label type for consistency in clean_multilabel_mask
    return mask_cleaned.astype(bool) # Keep as boolean for intermediate and then convert when assigning

def clean_multilabel_mask(mask, label_values):
    """Clean each label in a multi-label mask individually and recombine."""
    # Determine the original data type of the mask to preserve it for the output
    original_dtype = mask.dtype
    cleaned_mask = np.zeros_like(mask, dtype=original_dtype)

    for label in label_values:
        bin_mask = (mask == label)
        cleaned_bin_mask = clean_binary_mask(bin_mask)
        # Only assign the label if the cleaned binary mask is True
        cleaned_mask[cleaned_bin_mask] = label
    return cleaned_mask

def save_nifti(data, affine, header, output_path):
    nib.save(nib.Nifti1Image(data, affine, header), output_path)

# === Main Processing Loop ===
label_values = [1, 2, 3]  # exclude background (0)
mask_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".nii.gz")])

for fname in tqdm(mask_files, desc="Post-processing multi-label masks"):
    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname)

    mask_data, affine, header = load_nifti(input_path)
    cleaned_mask = clean_multilabel_mask(mask_data, label_values)
    save_nifti(cleaned_mask, affine, header, output_path)
