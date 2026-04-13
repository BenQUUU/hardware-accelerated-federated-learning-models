import os
import glob
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

    if cid == 0:
        start_idx = 0
        end_idx = int(total_files * 0.7)
    else:
        start_idx = int(total_files * 0.7)
        end_idx = total_files

    my_files = files[start_idx:end_idx]
    print(f"[Client {cid}] Loading {len(my_files)} images (Index: {start_idx}-{end_idx})")

    base_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(180), # Augmentacja obrotu
        transforms.RandomHorizontalFlip(), # Augmentacja odbicia
        transforms.ToTensor(),
    ]

    if cid == 0:
        client_transform = transforms.Compose(base_transforms)
        print(f"[Client {cid}] Non-IID Profile: Ideal Conditions")
    else:
        client_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
            *base_transforms
        ])
        print(f"[Client {cid}] Non-IID Profile: Degraded Sensor / Lighting")

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