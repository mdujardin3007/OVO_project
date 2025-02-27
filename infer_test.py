import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from train_and_save_CNN import BrainTumorDataset, UNet

def infer_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device is: {device}')
    batch_size = 8

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    flair_dir = "flair_middle_test"
    mask_dir = "mask_binary_middle"

    dataset = BrainTumorDataset(flair_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Creating the model and loading the weights
    model = UNet(in_channels=1, out_channels=1).to(device)
    model_path = "/content/drive/MyDrive/OVO_project/unet_brats.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Inference over a batch only
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            outputs = model(images)
            break  

    images_np = images.cpu().numpy()
    outputs_np = outputs.cpu().numpy()

    plt.figure(figsize=(2*batch_size, 9))
    for i in range(batch_size):
        plt.subplot(2, batch_size, i + 1)
        plt.imshow(images_np[i, 0, :, :], cmap='gray')
        plt.title("Input")
        plt.axis('off')

        plt.subplot(2, batch_size, batch_size + i + 1)
        plt.imshow(outputs_np[i, 0, :, :], cmap='gray')
        plt.title("Prediction")
        plt.axis('off')

    plt.tight_layout()
    plt.show()