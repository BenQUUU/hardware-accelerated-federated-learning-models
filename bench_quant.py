"""
Benchmark uciętego ekstraktora cech w trzech precyzjach: FP32, FP16, INT8.

Cel: sprawdzić, czy obniżenie precyzji daje realne przyspieszenie na urządzeniu bez
akceleracji sprzętowej (Raspberry Pi 5 / ARM), zanim scenariusz zostanie wpięty w federację.
Mierzy również odchylenie map cech od referencji FP32 -- F jest jednocześnie celem
rekonstrukcji autoenkodera, więc błąd precyzji trafia wprost w funkcję straty.

Uruchomienie (Raspberry Pi):
    python bench_quant.py --data_path ./data --dataset mvtec --class_name metal_nut \
        --out results_out/bench_quant_rpi.csv

Skrypt nie wymaga serwera ani klientów -- to pomiar czysto lokalny.
"""

import argparse
import csv
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import quantization as tvq
from torch.ao.quantization import (
    QConfig, MinMaxObserver, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver,
)

import dataset as ds

# Normalizacja jest częścią modelu (model.py), nie transformacji -- trzeba ją tu odtworzyć.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

CUT = 14  # model.py: net.features[:14] -> 96 kanałów

# torchvision.models.quantization akceptuje wyłącznie te backendy dla gotowych wag INT8.
QAT_BACKENDS = ("fbgemm", "qnnpack")


class CastEncoder(nn.Module):
    """Ekstraktor w obniżonej precyzji zmiennoprzecinkowej (FP16 lub BF16).

    Wejście i wyjście pozostają FP32, tak jak w docelowym modelu, gdzie wąskie gardło
    i dekoder uczą się dalej w pełnej precyzji.

    FP16: 5 bitów wykładnika, 10 bitów mantysy -- wysoka dokładność, wąski zakres.
    BF16: 8 bitów wykładnika, 7 bitów mantysy -- zakres jak FP32, niższa dokładność.
    """

    def __init__(self, features, dtype):
        super().__init__()
        self.dtype = dtype
        self.features = features.to(dtype)

    def forward(self, x):
        return self.features(x.to(self.dtype)).float()


class QuantizableEncoder(nn.Module):
    """Ucięty MobileNet V2 opakowany w stuby kwantyzacji (ścieżka PTQ)."""

    def __init__(self, features):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.features = features
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.features(self.quant(x)))


class PrequantizedEncoder(nn.Module):
    """Ucięty ekstraktor z oficjalnych wag INT8 (QAT) torchvision."""

    def __init__(self, qnet):
        super().__init__()
        self.quant = qnet.quant
        self.features = qnet.features[:CUT]
        self.dequant = torch.ao.nn.quantized.DeQuantize()

    def forward(self, x):
        return self.dequant(self.features(self.quant(x)))


def pick_backend():
    machine = platform.machine().lower()
    is_arm = machine.startswith(("arm", "aarch64"))
    available = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    preferred = ["qnnpack"] if is_arm else ["x86", "fbgemm", "onednn"]
    for engine in preferred:
        if engine in available:
            return engine, is_arm
    if available:
        return available[0], is_arm
    raise RuntimeError("Ta instalacja PyTorch nie wspiera kwantyzacji INT8.")


def _float_features():
    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    return net.features[:CUT]


def build_fp32():
    """Enkoder dokładnie taki, jak w model.py -- punkt odniesienia."""
    enc = _float_features()
    enc.eval()
    return enc


def build_cast(dtype):
    enc = CastEncoder(_float_features(), dtype)
    enc.eval()
    return enc


def build_int8_qat(backend):
    """Wagi INT8 wytrenowane metodą QAT przez torchvision -- wariant zalecany.

    PTQ na MobileNet V2 silnie degraduje mapy cech: konwolucje głębokościowe mają szeroki
    zakres dynamiczny per kanał, a kwantyzacja aktywacji działa per-tensor.
    """
    if backend not in QAT_BACKENDS:
        raise RuntimeError(
            f"backend '{backend}' nie obsluguje gotowych wag INT8 "
            f"(wymagany jeden z: {', '.join(QAT_BACKENDS)})"
        )
    qnet = tvq.mobilenet_v2(weights=tvq.MobileNet_V2_QuantizedWeights.DEFAULT, quantize=True)
    qnet.eval()
    enc = PrequantizedEncoder(qnet)
    enc.eval()
    return enc


def _ptq_qconfig(per_channel):
    """Konfiguracja kwantyzacji dla qnnpack (ARM wymaga reduce_range=False).

    per_channel=True: osobna skala na kanał wag -- kluczowe dla konwolucji głębokościowych
    MobileNetu, gdzie zakres dynamiczny per kanał jest szeroki. per_channel=False: jedna
    skala na cały tensor wag (wariant, który degradował cechy w poprzednim pomiarze).
    """
    activation = MovingAverageMinMaxObserver.with_args(
        qscheme=torch.per_tensor_affine, dtype=torch.quint8, reduce_range=False,
    )
    if per_channel:
        weight = MovingAveragePerChannelMinMaxObserver.with_args(
            qscheme=torch.per_channel_symmetric, dtype=torch.qint8,
        )
    else:
        weight = MinMaxObserver.with_args(
            qscheme=torch.per_tensor_symmetric, dtype=torch.qint8,
        )
    return QConfig(activation=activation, weight=weight)


def build_int8_ptq(calib_batches, per_channel):
    """Statyczna kwantyzacja potreningowa, kalibrowana na lokalnych obrazach poprawnych."""
    qnet = tvq.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT, quantize=False)
    qnet.eval()
    try:
        qnet.fuse_model(is_qat=False)
    except TypeError:
        qnet.fuse_model()

    enc = QuantizableEncoder(qnet.features[:CUT])
    enc.eval()
    enc.qconfig = _ptq_qconfig(per_channel)
    torch.ao.quantization.prepare(enc, inplace=True)
    with torch.no_grad():
        for batch in calib_batches:
            enc(batch)
    torch.ao.quantization.convert(enc, inplace=True)
    return enc


def normalize(images):
    return (images - MEAN) / STD


def collect_batches(loader, count):
    out = []
    for images, _ in loader:
        out.append(normalize(images))
        if len(out) >= count:
            break
    if not out:
        raise RuntimeError("Loader nie zwrócił żadnej partii danych.")
    return out


def probe(module, batch):
    """Czy ten wariant w ogóle wykonuje się na tej platformie."""
    try:
        with torch.no_grad():
            module(batch)
        return None
    except (RuntimeError, NotImplementedError) as exc:
        return str(exc).strip().splitlines()[0]


def time_forward(module, batches, iters, warmup):
    with torch.no_grad():
        for i in range(warmup):
            module(batches[i % len(batches)])
        times = []
        for i in range(iters):
            batch = batches[i % len(batches)]
            start = time.perf_counter()
            module(batch)
            times.append(time.perf_counter() - start)
    return statistics.median(times)


def feature_deviation(reference, module, batches):
    """Jak bardzo dany wariant przesuwa cel rekonstrukcji względem FP32."""
    abs_err, rel_err, cos = [], [], []
    with torch.no_grad():
        for batch in batches:
            ref = reference(batch)
            got = module(batch).float()
            diff = got - ref
            abs_err.append(diff.abs().mean().item())
            rel_err.append((diff.norm() / ref.norm()).item())
            cos.append(
                nn.functional.cosine_similarity(ref.flatten(1), got.flatten(1), dim=1).mean().item()
            )
    return statistics.mean(abs_err), statistics.mean(rel_err), statistics.mean(cos)


def module_size_mb(module):
    tmp = Path("_tmp_size.pt")
    torch.save(module.state_dict(), tmp)
    size = tmp.stat().st_size / 1024 ** 2
    tmp.unlink()
    return size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--dataset", type=str, choices=["mvtec", "visa", "realiad"], required=True)
    p.add_argument("--class_name", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8, help="Tak jak w treningu federacyjnym.")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--calib_batches", type=int, default=10)
    p.add_argument("--threads", type=int, default=0, help="0 = domyślne ustawienie PyTorch.")
    p.add_argument("--int8_mode", type=str, choices=["qat", "ptq", "both"], default="both")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    backend, is_arm = pick_backend()
    torch.backends.quantized.engine = backend

    print(f"Platforma      : {platform.machine()} ({'ARM' if is_arm else 'x86'})")
    print(f"Backend INT8   : {backend}")
    print(f"Watki CPU      : {torch.get_num_threads()}")
    print(f"Wsad           : {args.batch_size} x 3 x {ds.IMG_SIZE} x {ds.IMG_SIZE}")
    print(f"PyTorch        : {torch.__version__}\n")

    loader = ds.load_partitioned_data(
        cid=0, total_clients=1, data_path=args.data_path, dataset_name=args.dataset,
        class_name=args.class_name, batch_size=args.batch_size, partition_mode="whole",
    )
    batches = collect_batches(loader, max(args.calib_batches, args.warmup + 1, 5))

    print("Budowanie wariantow...")
    reference = build_fp32()
    variants = [("FP32", reference)]

    variants.append(("FP16", build_cast(torch.float16)))
    variants.append(("BF16", build_cast(torch.bfloat16)))

    if args.int8_mode in ("qat", "both"):
        try:
            variants.append(("INT8-QAT", build_int8_qat(backend)))
        except RuntimeError as exc:
            print(f"  [!] INT8-QAT niedostepny: {exc}")
    if args.int8_mode in ("ptq", "both"):
        variants.append(("INT8-PT", build_int8_ptq(batches[: args.calib_batches], per_channel=False)))
        variants.append(("INT8-PC", build_int8_ptq(batches[: args.calib_batches], per_channel=True)))

    with torch.no_grad():
        print(f"  ksztalt mapy cech: {tuple(reference(batches[0]).shape)}")

    rows, skipped = [], []
    for name, module in variants:
        err = probe(module, batches[0])
        if err:
            skipped.append((name, err))
            continue
        rows.append((name, module))

    for name, err in skipped:
        print(f"  [!] {name} nieobslugiwany na tej platformie: {err}")

    print(f"\nPomiar ({args.iters} iteracji na wariant)...")
    results = []
    base_time = None
    for name, module in rows:
        t = time_forward(module, batches, args.iters, args.warmup)
        if name == "FP32":
            base_time = t
            mae = rel = 0.0
            cos = 1.0
        else:
            mae, rel, cos = feature_deviation(reference, module, batches[:5])
        results.append({
            "nazwa": name, "czas_ms": t * 1000, "obrazy_s": args.batch_size / t,
            "rozmiar_mb": module_size_mb(module), "przysp": base_time / t if base_time else 1.0,
            "blad_abs": mae, "blad_wzgl": rel * 100, "cos": cos,
        })

    width = 12
    header = f"{'':20}" + "".join(f"{r['nazwa']:>{width}}" for r in results)
    print("\n" + header)
    print("-" * len(header))
    for label, key, fmt in [
        ("czas/wsad [ms]", "czas_ms", "{:.1f}"),
        ("obrazy/s", "obrazy_s", "{:.1f}"),
        ("rozmiar [MB]", "rozmiar_mb", "{:.2f}"),
        ("przyspieszenie", "przysp", "{:.2f}x"),
        ("blad wzgl. [%]", "blad_wzgl", "{:.2f}"),
        ("podob. cosinusowe", "cos", "{:.4f}"),
    ]:
        cells = "".join(f"{fmt.format(r[key]):>{width}}" for r in results)
        print(f"{label:20}{cells}")

    print()
    best = max((r for r in results if r["nazwa"] != "FP32"), key=lambda r: r["przysp"], default=None)
    if best is None or best["przysp"] < 1.05:
        print("  [!] Zaden wariant nie przyspieszyl zauwazalnie -- rozwaz opis jako wynik")
        print("      negatywny albo przejscie na symulowana zmiane precyzji.")
    else:
        print(f"  Najszybszy wariant: {best['nazwa']} ({best['przysp']:.2f}x, "
              f"blad wzgledny {best['blad_wzgl']:.1f}%)")
        if best["blad_wzgl"] > 20:
            print("  [!] Odchylenie map cech jest duze -- spodziewaj sie wplywu na AUROC.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["metryka"] + [r["nazwa"] for r in results])
            for label, key, fmt in [
                ("czas_wsad_ms", "czas_ms", "{:.2f}"), ("obrazy_s", "obrazy_s", "{:.2f}"),
                ("rozmiar_mb", "rozmiar_mb", "{:.3f}"), ("przyspieszenie", "przysp", "{:.3f}"),
                ("blad_bezwzgledny", "blad_abs", "{:.6f}"),
                ("blad_wzgledny_proc", "blad_wzgl", "{:.3f}"),
                ("podobienstwo_cos", "cos", "{:.6f}"),
            ]:
                w.writerow([label] + [fmt.format(r[key]) for r in results])
            w.writerow(["platforma", platform.machine()])
            w.writerow(["backend", backend])
            w.writerow(["watki", torch.get_num_threads()])
            w.writerow(["torch", torch.__version__])
        print(f"\nZapisano: {out}")


if __name__ == "__main__":
    main()
