import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Definition of the dataset
class BrainTumorDataset(Dataset):
    def __init__(self, flair_dir, mask_dir, transform=None):
        self.flair_paths = sorted(glob.glob(os.path.join(flair_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.flair_paths)

    def __getitem__(self, idx):
        flair_path = self.flair_paths[idx]
        mask_path = self.mask_paths[idx]
        flair_img = Image.open(flair_path).convert("L")
        mask_img = Image.open(mask_path).convert("L")
        if self.transform:
            flair_img = self.transform(flair_img)
            mask_img = self.transform(mask_img)
        mask_img = (mask_img > 0.5).float()
        return flair_img, mask_img


transform = transforms.Compose([
    transforms.ToTensor(), 
])

# Def of the UNet model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        rev_features = features[::-1]
        for feature in rev_features:
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
        return torch.sigmoid(self.final_conv(x))



def train_and_save_CNN():
  flair_dir = "/content/drive/MyDrive/OVO_project/flair_middle_train"
  mask_dir = "/content/drive/MyDrive/OVO_project/mask_binary_middle"

  # Vérification que le Dataset charge correctement
  dataset = BrainTumorDataset(flair_dir, mask_dir, transform=transform)
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True,pin_memory=True) # Higher batch size for faster training


  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"deviced used: {device}")
  model = UNet(in_channels=1, out_channels=1).to(device)
  criterion = nn.BCELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.0001)
  num_epochs = 75

  model.train()
  for epoch in range(num_epochs):
      epoch_loss = 0.0
      for images, masks in dataloader:
          images = images.to(device)
          masks = masks.to(device)

          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, masks)
          loss.backward()
          optimizer.step()
          epoch_loss += loss.item() * images.size(0)
      epoch_loss /= len(dataset)
      print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

  # Saving the model
  torch.save(model.state_dict(), "/content/drive/MyDrive/OVO_project/unet_brats.pth")
