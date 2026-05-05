import os
import glob
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

IMG_SIZE = 256

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

class MVTecTestDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def get_files(data_path, split="train"):
    if split == "train":
        pattern = os.path.join(data_path, "train", "good", "*.png")
    else:
        pattern = os.path.join(data_path, "test", "*", "*.png")
    return sorted(glob.glob(pattern))

def get_test_files_and_labels(data_path):
    files = get_files(data_path, split="test")
    labels = []
    for f in files:
        parent_dir = os.path.basename(os.path.dirname(f))
        if parent_dir == "good":
            labels.append(0)
        else:
            labels.append(1)
    return files, labels


def load_partitioned_data(cid, total_clients, data_path, batch_size=8):
    files = get_files(data_path, split="train")
    total_files = len(files)
    if total_files == 0:
        raise ValueError(f"Nie znaleziono plików w {data_path}. Sprawdź ścieżkę!")

    random.Random(42).shuffle(files)

    partition_size = total_files // total_clients
    start_idx = cid * partition_size

    if cid == total_clients - 1:
        end_idx = total_files
    else:
        end_idx = start_idx + partition_size

    my_files = files[start_idx:end_idx]
    print(f"[Client {cid}] Loading {len(my_files)} images (Index: {start_idx}-{end_idx})")

    # BAZOWE TRANSFORMACJE (wspólne dla wszystkich)
    base_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]

    # ZMIENNA BADAWCZA: Logika Covariate Shift
    if cid == 0:
        client_transform = transforms.Compose(base_transforms)
        print(f"[Client {cid}] Profil REFERENCYJNY: Czyste dane (Jetson 1)")
    elif cid == 1:
        # Ten Jetson udaje zepsutą kamerę
        client_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4, hue=0.1),
            *base_transforms
        ])
        print(f"[Client {cid}] Profil ANOMALII: Zdegradowany sensor ColorJitter (Jetson 2)")
    else:
        client_transform = transforms.Compose(base_transforms)
        print(f"[Client {cid}] Profil STANDARDOWY")

    dataset = MVTecDataset(my_files, transform=client_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_test_data(data_path, batch_size=32):
    files, labels = get_test_files_and_labels(data_path)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = MVTecTestDataset(files, labels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)