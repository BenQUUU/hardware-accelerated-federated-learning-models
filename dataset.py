import os
import json
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMG_SIZE = 256

class IndustrialAnomalyDataset(Dataset):
    def __init__(self, dataset_name, root_path, class_name, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform
        self.image_paths = []
        self.labels = []  # 0: normal (OK), 1: anomaly (NG)

        if dataset_name == "mvtec":
            self._load_mvtec(root_path, class_name, is_train)
        elif dataset_name == "visa":
            self._load_visa(root_path, class_name, is_train)
        elif dataset_name == "realiad":
            self._load_realiad(root_path, class_name, is_train)
        else:
            raise ValueError(f"[ERROR] Nieobsługiwany zbiór danych: {dataset_name}")

    def _load_mvtec(self, root, cls, is_train):
        cls_dir = os.path.join(root, cls)
        split_dir = "train" if is_train else "test"
        target_dir = os.path.join(cls_dir, split_dir)

        for condition in os.listdir(target_dir):
            condition_dir = os.path.join(target_dir, condition)
            if not os.path.isdir(condition_dir):
                continue
                
            label = 0 if condition == "good" else 1
            
            for img_name in os.listdir(condition_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(condition_dir, img_name))
                    self.labels.append(label)

    def _load_visa(self, root, cls, is_train):
        csv_path = os.path.join(root, cls, "image_anno.csv")
        normal_paths = []
        anomaly_paths = []

        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_full_path = os.path.join(root, row['image'])
                if row['label'] == 'normal':
                    normal_paths.append(img_full_path)
                else:
                    anomaly_paths.append(img_full_path)

        # Sortowanie gwarantujące determinizm przed podziałem
        normal_paths.sort()
        split_idx = int(len(normal_paths) * 0.8)

        if is_train:
            self.image_paths = normal_paths[:split_idx]
            self.labels = [0] * len(self.image_paths)
        else:
            self.image_paths = normal_paths[split_idx:] + anomaly_paths
            self.labels = [0] * (len(normal_paths) - split_idx) + [1] * len(anomaly_paths)

    def _load_realiad(self, root, cls, is_train):
        json_path = os.path.join(root, cls, f"{cls}.json")
        split_key = "train" if is_train else "test"

        with open(json_path, mode='r', encoding='utf-8') as f:
            data = json.load(f)

        prefix = data["meta"]["prefix"]
        normal_class_name = data["meta"]["normal_class"]

        for item in data[split_key]:
            img_path = os.path.join(root, prefix, item["image_path"])
            label = 0 if item["anomaly_class"] == normal_class_name else 1
            
            self.image_paths.append(img_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Trening (Autoenkoder) potrzebuje (image, image), ewaluacja potrzebuje (image, label)
        if self.is_train:
            return image, image
        else:
            return image, label

def load_partitioned_data(cid, total_clients, data_path, dataset_name, class_name, apply_shift=False,
                          batch_size=8, num_workers=0, pin_memory=False, partition_mode="split"):
    # Czysty pipeline bazowy (bez losowych flipow) => spojny z sanity_check.py.
    # Jedyna celowa perturbacja Non-IID to ColorJitter ponizej (covariate shift).
    base_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
    client_transform = transforms.Compose(base_transforms)

    full_dataset = IndustrialAnomalyDataset(dataset_name, data_path, class_name, is_train=True, transform=client_transform)

    # Covariate Shift aktywowany tylko z flagą uruchomieniową
    if cid == 1 and apply_shift:
        shift_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
        ])
        full_dataset.transform = shift_transform
        print(f"[Client {cid}] Profil ANOMALII: Covariate Shift aktywny.")

    total_files = len(full_dataset)
    if total_files == 0:
        raise ValueError(f"Brak plików w {data_path} dla {dataset_name}/{class_name}.")

    if partition_mode == "whole":
        # Non-IID danych: kazdy klient uzywa CALEJ swojej klasy (rozne klasy per klient).
        # Podzialem jest samo przypisanie klasy, wiec nie tniemy zbioru na partycje.
        client_dataset = full_dataset
        print(f"[Client {cid}] Tryb WHOLE: pełna klasa '{class_name}' ({total_files} obrazów treningowych).")
    else:
        # Tryb 'split': jedna klasa dzielona rowno na total_clients (dane ~IID).
        partition_size = total_files // total_clients
        lengths = [partition_size] * total_clients
        lengths[-1] += total_files - sum(lengths)

        datasets = torch.utils.data.random_split(full_dataset, lengths, generator=torch.Generator().manual_seed(42))
        client_dataset = datasets[cid]
        print(f"[Client {cid}] Tryb SPLIT: załadowano {len(client_dataset)} obrazów treningowych.")

    # generator z ustalonym ziarnem => powtarzalna kolejność batchy między uruchomieniami
    loader_generator = torch.Generator().manual_seed(42)
    return DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        generator=loader_generator,
    )

def load_test_data(data_path, dataset_name, class_name, batch_size=32, num_workers=0, pin_memory=False):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = IndustrialAnomalyDataset(dataset_name, data_path, class_name, is_train=False, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )