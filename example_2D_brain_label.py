import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def example_2D_brain_label():
    img_path = "Task01_BrainTumour/imagesTr/BRATS_001.nii.gz"
    seg_path = "Task01_BrainTumour/labelsTr/BRATS_001.nii.gz"

    img_nib = nib.load(img_path)
    seg_nib = nib.load(seg_path)

    img_data = img_nib.get_fdata() # shape (240, 240, 155, 4) 
    seg_data = seg_nib.get_fdata() # shape (240, 240, 155)

    flair_data = img_data[..., 0] # showing flair image

    slice_idx = flair_data.shape[2] // 2
    flair_slice = flair_data[:, :, slice_idx]
    seg_slice = seg_data[:, :, slice_idx]

    binary_mask = np.where(seg_slice > 0, 1, 0) # merging all the different tumor types

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(flair_slice, cmap='gray')
    plt.title("Image FLAIR - Slice {}".format(slice_idx))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Masque binaire (fusion de labels 1,2,3)")
    plt.axis('off')

    plt.show()
