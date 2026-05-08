import torch
import torch.nn as nn
from torchvision import models

class Autoencoder(nn.Module):
    def __init__(self, extractor_name="mobilenet"):
        super(Autoencoder, self).__init__()

        if extractor_name == "mobilenet":
            net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

            self.encoder = net.features[:14]
            out_channels = 96

        elif extractor_name == "squeezenet":
            net = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)

            self.encoder = net.features[:8]
            out_channels = 256

        elif extractor_name == "shufflenet":
            net = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)

            self.encoder = nn.Sequential(
                net.conv1, net.maxpool, net.stage2
            )
            out_channels = 116

        else:
            raise ValueError(f"Nieznany model: {extractor_name}")

        for param in self.encoder.parameters():
            param.requires_grad = False

        # 2. Twój autorski autoenkoder (Wąskie gardło kompresujące do 4 kanałów!)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        with torch.no_grad():
            original_features = self.encoder(x_norm)

        compressed = self.bottleneck(original_features)
        reconstructed_features = self.decoder(compressed)

        return original_features, reconstructed_features