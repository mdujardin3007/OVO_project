import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from siren_pytorch import SirenNet
from PIL import Image
import os

def siren_compute(random_file):

    flair_img = Image.open(rf"/content/drive/MyDrive/OVO_project/flair_middle_train/{random_file}.png").convert("L")
    flair_slice = np.array(flair_img, dtype=np.float32) / 255.0  # Normalization in [0,1]

    H, W = flair_slice.shape

    # Prepare the coords and the targets for the SIREN
    coords = np.stack(np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H)), -1)
    coords = coords.reshape(-1, 2)
    targets = flair_slice.reshape(-1, 1)

    coords = torch.tensor(coords, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    # Import of the SIREN model
    model = SirenNet(
        dim_in=2,        # coordinates
        dim_hidden=64,
        dim_out=1,       # pixel intensity
        num_layers=3,
        w0_initial=30.
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    device = next(model.parameters()).device
    coords = coords.to(device)
    targets = targets.to(device)

    # Same optimizer (Adam) and lr parameter of the paper
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    num_epochs = 50000  
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(coords)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

    # After training, we can reconstruct the image
    with torch.no_grad():
        outputs = model(coords).cpu().numpy()
    reconstructed = outputs.reshape(H, W)

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

    if not os.path.exists(rf"/content/drive/MyDrive/OVO_project/reconstructed\{random_file}"):
        os.mkdir(rf"/content/drive/MyDrive/OVO_project/reconstructed\{random_file}")
    plt.imsave(rf"/content/drive/MyDrive/OVO_project/reconstructed\{random_file}\{random_file}_reconstructed_lambda_0.png", reconstructed, cmap='gray')
