import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from siren_pytorch import SirenNet
from PIL import Image
import torchvision.transforms as transforms
from train_and_save_CNN import UNet
import os


def sinco_pipeline(random_file):
    #we compute over a random file the prediction of the model
    folder_path = "/content/drive/MyDrive/OVO_project/flair_middle_train"
    file_list = os.listdir(folder_path)
    file_names = [k.split('.')[0] for k in file_list]


    flair_img_path = f"/content/drive/MyDrive/OVO_project/flair_middle_train/{random_file}.png"
    mask_img_path  = f"/content/drive/MyDrive/OVO_project/mask_binary_middle/{random_file}_mask.png"

    flair_img = Image.open(flair_img_path).convert("L")
    flair_slice = np.array(flair_img, dtype=np.float32) / 255.0  # Normalization in [0,1]

    mask_img = Image.open(mask_img_path).convert("L")
    mask_np = np.array(mask_img, dtype=np.float32) / 255.0
    mask_np = (mask_np > 0.5).astype(np.float32)

    H, W = flair_slice.shape
    print(f"Dimensions de l'image: {H}x{W}")

    target_seg = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Preparation for SIREN
    coords = np.stack(np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H)), -1)
    coords = coords.reshape(-1, 2)
    targets = flair_slice.reshape(-1, 1)

    coords = torch.tensor(coords, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    model = SirenNet(
        dim_in=2,        # coords (x,y)
        dim_hidden=64,
        dim_out=1,       # intensity value of the pixel 
        num_layers=3,
        w0_initial=30.
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = model.to(device)
    coords = coords.to(device)
    targets = targets.to(device)
    target_seg = target_seg.to(device)


    # Creation of the segmentation model and loading of the weights
    seg_model = UNet(in_channels=1, out_channels=1).to(device)
    seg_model.load_state_dict(torch.load("unet_brats.pth", map_location=device))
    seg_model.eval()
    # Freeze the weights of the segmentation model
    for param in seg_model.parameters():
        param.requires_grad = False

    def soft_dice_loss(pred, target, epsilon=1e-6):
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(1)
        dice = (2 * intersection + epsilon) / (pred_flat.sum(1) + target_flat.sum(1) + epsilon)
        return 1 - dice.mean()


    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    lambda_reg = 1e-5  # The regularization parameter, which differs from the article

    num_epochs = 50000  # according to the article
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(coords)
        loss_reconstruction = criterion(outputs, targets)

        outputs_img = outputs.reshape(1, 1, H, W)
        seg_output = seg_model(outputs_img)
        loss_segmentation = soft_dice_loss(seg_output, target_seg) # The dice loss, as in the article

        loss = loss_reconstruction + lambda_reg * loss_segmentation # The total loss, as in the article
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Rec Loss: {loss_reconstruction.item():.6f}, "
                f"Seg Loss: {loss_segmentation.item():.6f}, Total Loss: {loss.item():.6f}")

    with torch.no_grad():
        outputs_final = model(coords).cpu().numpy()
    reconstructed = outputs_final.reshape(H, W)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(flair_slice, cmap='gray')
    plt.title("Image originale")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed, cmap='gray')
    plt.title("Image reconstruite par SIREN")
    plt.axis('off')
    plt.show()

    image_name = os.path.basename(flair_img_path).replace(".png", "")
    save_folder = f"reconstructed/{image_name}"
    os.makedirs(save_folder, exist_ok=True)
    save_filename = f"{image_name}_reconstructed_lambda_{lambda_reg}.png"
    save_path = f"{save_folder}/{save_filename}"
    plt.imsave(save_path, reconstructed, cmap='gray')
    print(f"Image reconstruite sauvegardée dans : {save_path}")