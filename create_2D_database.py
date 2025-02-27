import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Creating the training dataset
def create_2D_database():

  train_folder = "Task01_BrainTumour/imagesTr"
  label_folder = "Task01_BrainTumour/labelsTr"

  output_flair_folder_tr = "flair_middle_train"
  output_mask_folder_tr = "mask_binary_middle"
  os.makedirs(output_flair_folder_tr, exist_ok=True)
  os.makedirs(output_mask_folder_tr, exist_ok=True)

  img_files_tr = sorted(glob.glob(os.path.join(train_folder, "*.nii.gz")))

  for img_path in img_files_tr:
      filename = os.path.basename(img_path)
      patient_id = filename.replace(".nii.gz", "")

      label_path = os.path.join(label_folder, filename)
      if not os.path.exists(label_path):
          continue

      img_nib = nib.load(img_path)
      img_data = img_nib.get_fdata()

      seg_nib = nib.load(label_path)
      seg_data = seg_nib.get_fdata()

      flair_data = img_data[..., 0]  # Extracting the FLAIR modality (channel 0) (There are 4 channels, the FLAIR one is the most common)

      # Taking the middle part of the brain
      slice_idx = flair_data.shape[2] // 2
      flair_slice = flair_data[:, :, slice_idx]
      seg_slice = seg_data[:, :, slice_idx]

      # Binary mask, 0 for background and 1 for tumor (merging the different tumor classes)
      binary_mask = np.where(seg_slice > 0, 1, 0)

      flair_out_path = os.path.join(output_flair_folder_tr, f"{patient_id}.png")
      mask_out_path = os.path.join(output_mask_folder_tr, f"{patient_id}_mask.png")

      plt.imsave(flair_out_path, flair_slice, cmap='gray')
      plt.imsave(mask_out_path, binary_mask, cmap='gray')

      print(f"[TRAIN] Patient {patient_id} traité, slice index {slice_idx} sauvegardée.")

  print("Extraction des coupes d'entraînement terminée.")


  # Creating the testing dataset

  test_folder = "Task01_BrainTumour/imagesTs"
  output_flair_folder_ts = "flair_middle_test"
  os.makedirs(output_flair_folder_ts, exist_ok=True)

  img_files_ts = sorted(glob.glob(os.path.join(test_folder, "*.nii.gz")))

  for img_path in img_files_ts:
      filename = os.path.basename(img_path)
      patient_id = filename.replace(".nii.gz", "")

      img_nib = nib.load(img_path)
      img_data = img_nib.get_fdata()

      flair_data = img_data[..., 0] # Same thing as before

      slice_idx = flair_data.shape[2] // 2
      flair_slice = flair_data[:, :, slice_idx]

      flair_out_path = os.path.join(output_flair_folder_ts, f"{patient_id}.png")

      plt.imsave(flair_out_path, flair_slice, cmap='gray')

      print(f"[TEST] Patient {patient_id} traité, slice index {slice_idx} sauvegardée.")

  print("Extraction des coupes de test terminée.")
