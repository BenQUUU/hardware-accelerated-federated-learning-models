import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import os
import time

IMG_SIZE = 256
LATENT_DIM = 32

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input: 3 x 256 x 256
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # -> 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # -> 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, 7)  # Kernel 7x7
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class MVTecDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, image

def get_files(data_path, split="train"):
    if split == "train":
        pattern = os.path.join(data_path, "train", "good", "*.png")
    else:
        pattern = os.path.join(data_path, "test", "*", "*.png")
    
    files = sorted(glob.glob(pattern))
    return files

def load_partitioned_data(cid, total_clients, data_path, batch_size=8):
    files = get_files(data_path, split="train")
    
    total_files = len(files)
    if total_files == 0:
        raise ValueError(f"Nie znaleziono plików w {data_path}. Sprawdź ścieżkę!")

    partition_size = total_files // total_clients
    start_idx = cid * partition_size
    end_idx = start_idx + partition_size
    
    if cid == total_clients - 1:
        end_idx = total_files

    my_files = files[start_idx:end_idx]
    print(f"[Client {cid}] Loading {len(my_files)} images (Index: {start_idx}-{end_idx})")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = MVTecDataset(my_files, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_test_data(data_path, batch_size=32):
    files = get_files(data_path, split="test")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = MVTecDataset(files, transform=transform)
    return DataLoader(dataset, batch_size=batch_size)

def train(net, trainloader, epochs, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, _ in trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

def test(net, testloader, device):
    criterion = nn.MSELoss()
    total_loss = 0.0
    steps = 0
    net.eval()
    with torch.no_grad():
        for images, _ in testloader:
            images = images.to(device)
            outputs = net(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()
            steps += 1
    return total_loss / steps


def train_by_time(net, trainloader, timeout, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()

    start_time = time.time()
    epochs_done = 0
    total_images = 0

    while (time.time() - start_time) < timeout:
        for images, _ in trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            total_images += len(images)

            if (time.time() - start_time) > timeout:
                break
        else:
            epochs_done += 1

    return epochs_done, total_images