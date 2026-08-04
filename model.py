import platform

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import quantization as tvq
from torch.ao.quantization import (
    QConfig, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver,
)

# Parametry FEDEROWANE = tylko trenowalna glowica (bottleneck + decoder). Zamrozony
# ekstraktor NIE jest przesylany ani agregowany -- kazdy wezel trzyma go lokalnie.
# Dzieki temu wezel bez akceleracji moze uzywac innej precyzji ekstraktora (FP16/INT8),
# a agregacja glowicy (zawsze FP32, identyczna strukturalnie) pozostaje spojna.
HEAD_PREFIXES = ("bottleneck", "decoder")


def is_head_param(name: str) -> bool:
    """Czy dany klucz state_dict nalezy do trenowalnej glowicy (podlega federacji)."""
    return name.startswith(HEAD_PREFIXES) and "num_batches_tracked" not in name


_OUT_CHANNELS = {"mobilenet": 96, "shufflenet": 116, "squeezenet": 256}
_MOBILENET_CUT = 14


def _build_base_extractor(name):
    """Zamrozony ekstraktor cech w FP32 (jak w oryginalnej architekturze)."""
    if name == "mobilenet":
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        return net.features[:_MOBILENET_CUT]
    if name == "shufflenet":
        net = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        return nn.Sequential(net.conv1, net.maxpool, net.stage2)
    if name == "squeezenet":
        net = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        return net.features[:8]
    raise ValueError(f"Unknown model: {name}")


class _CastEncoder(nn.Module):
    """Ekstraktor w obnizonej precyzji zmiennoprzecinkowej (FP16/BF16).

    Wejscie i wyjscie pozostaja FP32 -- glowica uczy sie dalej w pelnej precyzji.
    """

    def __init__(self, features, dtype):
        super().__init__()
        self.dtype = dtype
        self.features = features.to(dtype)

    def forward(self, x):
        return self.features(x.to(self.dtype)).float()


class _QuantizableEncoder(nn.Module):
    """Ucięty MobileNet V2 w stubach kwantyzacji (sciezka PTQ, INT8)."""

    def __init__(self, features):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.features = features
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.features(self.quant(x)))


def _ptq_qconfig():
    # Wagi per-channel (kluczowe dla konwolucji glebokosciowych MobileNetu);
    # qnnpack (ARM) wymaga reduce_range=False.
    activation = MovingAverageMinMaxObserver.with_args(
        qscheme=torch.per_tensor_affine, dtype=torch.quint8, reduce_range=False,
    )
    weight = MovingAveragePerChannelMinMaxObserver.with_args(
        qscheme=torch.per_channel_symmetric, dtype=torch.qint8,
    )
    return QConfig(activation=activation, weight=weight)


def _set_quantized_engine():
    machine = platform.machine().lower()
    is_arm = machine.startswith(("arm", "aarch64"))
    available = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    preferred = ["qnnpack"] if is_arm else ["x86", "fbgemm", "onednn"]
    for engine in preferred:
        if engine in available:
            torch.backends.quantized.engine = engine
            return engine
    if available:
        torch.backends.quantized.engine = available[0]
        return available[0]
    raise RuntimeError("Ta instalacja PyTorch nie wspiera kwantyzacji INT8.")


def _build_int8_encoder_prepared():
    """Ucięty ekstraktor przygotowany do PTQ (przed kalibracja/konwersja -- patrz calibrate())."""
    _set_quantized_engine()
    qnet = tvq.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT, quantize=False)
    qnet.eval()
    try:
        qnet.fuse_model(is_qat=False)
    except TypeError:
        qnet.fuse_model()
    enc = _QuantizableEncoder(qnet.features[:_MOBILENET_CUT])
    enc.eval()
    enc.qconfig = _ptq_qconfig()
    torch.ao.quantization.prepare(enc, inplace=True)
    return enc


class Autoencoder(nn.Module):
    def __init__(self, extractor_name="mobilenet", extractor_precision="fp32"):
        super().__init__()
        self.extractor_precision = extractor_precision
        self._int8_pending = False

        out_channels = _OUT_CHANNELS[extractor_name]

        if extractor_precision == "fp32":
            self.encoder = _build_base_extractor(extractor_name)
        elif extractor_precision == "fp16":
            self.encoder = _CastEncoder(_build_base_extractor(extractor_name), torch.float16)
        elif extractor_precision == "int8":
            if extractor_name != "mobilenet":
                raise ValueError("Kwantyzacja INT8 wspierana tylko dla ekstraktora 'mobilenet'.")
            self.encoder = _build_int8_encoder_prepared()
            self._int8_pending = True
        else:
            raise ValueError(f"Nieznana precyzja ekstraktora: {extractor_precision}")

        for param in self.encoder.parameters():
            param.requires_grad = False

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

    def calibrate(self, loader, num_batches=10):
        """Kalibracja i konwersja INT8 na lokalnych obrazach poprawnych.

        Bez znaczenia (no-op) dla FP32/FP16. Wymaga urzadzenia CPU (qnnpack).
        """
        if not self._int8_pending:
            return
        self.eval()
        seen = 0
        with torch.no_grad():
            for images, _ in loader:
                self.encoder((images - self.mean) / self.std)
                seen += 1
                if seen >= num_batches:
                    break
        torch.ao.quantization.convert(self.encoder, inplace=True)
        self._int8_pending = False

    def train(self, mode=True):
        # Enkoder zawsze w trybie eval: wagi zamrozone (requires_grad=False),
        # a to gwarantuje, ze statystyki BatchNorm (running_mean/var) NIE dryfuja
        # podczas treningu i nie sa zaburzane przez covariate shift w FedAvg.
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        with torch.no_grad():
            original_features = self.encoder(x_norm)

        compressed = self.bottleneck(original_features)
        reconstructed_features = self.decoder(compressed)

        return original_features, reconstructed_features
