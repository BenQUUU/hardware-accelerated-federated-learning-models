import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        # Ostatnia warstwa kodera zwraca tensor [Batch, 96, 16, 16]
        self.encoder = mobilenet.features[:14] 
        
        # Zamrożenie wag kodera
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Wąskie gardło kompresujące mapę cech przestrzennie
        self.bottleneck = nn.Sequential(
            nn.Conv2d(96, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=3, padding=1), # Zaledwie 4 kanały!
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        
        # Odtworzenie mapy cech (zwróć uwagę na brak Sigmoid, cechy nie są w przedziale 0-1)
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 96, kernel_size=3, padding=1)
        )

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        with torch.no_grad():
            original_features = self.encoder(x_norm)
            
        compressed = self.bottleneck(original_features)
        reconstructed_features = self.decoder(compressed)
        
        # Zwracamy oryginał z MobileNetu i próbę jego odtworzenia
        return original_features, reconstructed_features