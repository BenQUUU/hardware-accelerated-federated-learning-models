import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics import roc_auc_score

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_loss(original_features, reconstructed_features):
    # Stabilny błąd MSE obliczany na zdesaturowanej z pikseli przestrzeni
    return F.mse_loss(reconstructed_features, original_features)

def train(net, trainloader, epochs, device):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    
    for epoch in range(epochs):
        for images, _ in trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            orig_feat, rec_feat = net(images)
            loss = compute_loss(orig_feat, rec_feat)
            loss.backward()
            optimizer.step()

def train_by_time(net, data_iterator, trainloader, timeout, device):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    start_time = time.time()
    epochs_done = 0
    total_images = 0

    while (time.time() - start_time) < timeout:
        try:
            images, _ = next(data_iterator)
        except StopIteration:
            data_iterator = iter(trainloader)
            epochs_done += 1
            try:
                images, _ = next(data_iterator)
            except StopIteration:
                break

        images = images.to(device)
        optimizer.zero_grad()
        orig_feat, rec_feat = net(images)
        loss = compute_loss(orig_feat, rec_feat)
        loss.backward()
        optimizer.step()
        total_images += len(images)

    return epochs_done, total_images, data_iterator

def test(net, testloader, device):
    net.eval()
    all_losses = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            orig_feat, rec_feat = net(images)

            loss_map = F.mse_loss(rec_feat, orig_feat, reduction='none')

            loss_per_spatial_cell = loss_map.mean(dim=1)

            loss_flat = loss_per_spatial_cell.view(loss_per_spatial_cell.size(0), -1)

            loss_per_image = loss_flat.max(dim=1)[0]
            
            all_losses.extend(loss_per_image.cpu().numpy())
            all_labels.extend(labels.numpy())

    auroc = roc_auc_score(all_labels, all_losses)
    avg_loss = np.mean(all_losses)
    
    return auroc, avg_loss