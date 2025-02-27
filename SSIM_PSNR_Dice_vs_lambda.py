import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from train_and_save_CNN import UNet
import torch


def compute_ssim(original, reconstructed):
    return ssim(original, reconstructed, data_range=1.0)

def compute_psnr(original, reconstructed):
    return psnr(original, reconstructed, data_range=1.0)

def compute_dice(mask_gt, mask_pred, epsilon=1e-6):
    mask_gt = mask_gt.flatten()
    mask_pred = mask_pred.flatten()
    intersection = np.sum(mask_gt * mask_pred)
    dice = (2 * intersection + epsilon) / (np.sum(mask_gt) + np.sum(mask_pred) + epsilon)
    return dice

def compute_metrics(file):

    original_path = rf"/content/drive/MyDrive/OVO_project/flair_middle_train/{file}.png"
    mask_path     = rf"/content/drive/MyDrive/OVO_project/mask_binary_middle/{file}_mask.png"
    recon_base    = rf"/content/drive/MyDrive/OVO_project/reconstructed/{file}"

    original_img = np.array(Image.open(original_path).convert("L"), dtype=np.float32) / 255.0
    mask_img     = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    mask_img     = (mask_img > 0.5).astype(np.float32)

    lambdas = [0, 1e-5]

    ssim_values = []
    psnr_values = []
    dice_values = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model = UNet(in_channels=1, out_channels=1).to(device)
    seg_model.load_state_dict(torch.load(r"/content/drive/MyDrive/OVO_project/unet_brats.pth", map_location=device))
    seg_model.eval()
    for param in seg_model.parameters():
        param.requires_grad = False

    for lam in lambdas:
        recon_filename = f"{file}_reconstructed_lambda_{lam}.png"
        recon_path = f"{recon_base}/{recon_filename}"
        annotated_path = f"{recon_base}/{recon_filename.replace('.png', '_annotated.png')}"

        recon_img = np.array(Image.open(recon_path).convert("L"), dtype=np.float32) / 255.0
        
        ssim_val = compute_ssim(original_img, recon_img)
        ssim_values.append(ssim_val)
        
        psnr_val = compute_psnr(original_img, recon_img)
        psnr_values.append(psnr_val)
        
        recon_tensor = torch.tensor(recon_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            seg_output = seg_model(recon_tensor)
        seg_output = seg_output.cpu().numpy().squeeze()
        seg_pred = (seg_output > 0.5).astype(np.float32)
        dice_val = compute_dice(mask_img, seg_pred)
        dice_values.append(dice_val)

        save_annotated_image(recon_path, annotated_path, psnr_val, ssim_val, dice_val)
        print(f"Lambda {lam}: PSNR = {psnr_val:.2f}, SSIM = {ssim_val:.3f}, Dice = {dice_val:.3f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(lambdas, psnr_values, marker='o', label='PSNR')
    plt.plot(lambdas, ssim_values, marker='o', label='SSIM')
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Métrique')
    plt.title('PSNR & SSIM vs Lambda')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(lambdas, dice_values, marker='o', color='green', label='Dice Score')
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Dice Score')
    plt.title('Dice Score vs Lambda')
    plt.legend()

    plt.tight_layout()
    plt.show()


from PIL import ImageDraw, ImageFont

def save_annotated_image(image_path, output_path, psnr, ssim, dice):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default()

    text = f"PSNR: {psnr:.2f}\nSSIM: {ssim:.3f}\nDice: {dice:.3f}"
    draw.text((10, 10), text, fill=(255, 0, 0), font=font)

    img.save(output_path)
    print(f"Image annotée sauvegardée : {output_path}")