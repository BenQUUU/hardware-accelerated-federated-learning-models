import argparse
import time
import csv
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import model
import dataset
import engine
from profiler import HardwareProfiler

def main():
    parser = argparse.ArgumentParser(description="Sanity Check for Autoencoder model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to main data folder")
    parser.add_argument("--dataset", type=str, choices=["mvtec", "visa", "realiad"], required=True, help="Dataset name")
    parser.add_argument("--class_name", type=str, required=True, help="Class name (e.g., metal_nut, pcb1)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--extractor", type=str, choices=["mobilenet", "shufflenet", "squeezenet"], default="mobilenet", help="Extractor model")
    parser.add_argument("--hw_profile", type=str, choices=["cpu", "cuda", "jetson", "rpi"], default="cuda", help="Hardware profiler backend for the power/energy comparison")
    args = parser.parse_args()

    engine.set_seed()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Sanity Check ===")
    print(f"Target device: {device}")
    print(f"Dataset: {args.dataset} | Class: {args.class_name} | Extractor: {args.extractor}")
    print(f"Number of epochs: {args.epochs}")

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    # Inicjalizacja nowego, uniwersalnego Datasetu
    train_dataset = dataset.IndustrialAnomalyDataset(
        dataset_name=args.dataset, 
        root_path=args.data_path, 
        class_name=args.class_name, 
        is_train=True, 
        transform=transform_train
    )
    
    if len(train_dataset) == 0:
        print(f"[ERROR] No training files found for {args.dataset}/{args.class_name} in {args.data_path}.")
        return

    print(f"Loaded {len(train_dataset)} training images.")
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # Ładowanie zbioru testowego przez zaktualizowaną funkcję pomocniczą
    testloader = dataset.load_test_data(args.data_path, args.dataset, args.class_name, batch_size=args.batch_size)
    print(f"Loaded test images (good and defective).")

    net = model.Autoencoder(extractor_name=args.extractor).to(device)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print("\nStarting model training...")
    hw_profiler = HardwareProfiler(device_type=args.hw_profile)
    hw_profiler.start()
    t0 = time.time()
    try:
        engine.train(net, trainloader, epochs=args.epochs, device=device)
    finally:
        train_time = time.time() - t0
        hw = hw_profiler.stop()
    print("Training finished.")

    print("\nEvaluating on test dataset...")
    auroc, avg_loss = engine.test(net, testloader, device=device)

    # Energia = srednia moc * czas treningu (W * s = J)
    energy_j = hw["avg_power_w"] * train_time
    time_per_epoch = train_time / args.epochs if args.epochs else 0.0

    print("\n=== SANITY CHECK RESULTS ===")
    print(f"Average evaluation loss (Top-K): {avg_loss:.5f}")
    print(f"AUROC score:                     {auroc:.4f}")
    print(f"Train time:                      {train_time:.1f}s ({time_per_epoch:.2f}s/epoch)")
    print(f"Avg GPU / Power / VRAM:          {hw['avg_gpu_percent']:.1f}% | {hw['avg_power_w']:.2f} W | {hw['avg_vram_percent']:.1f}%")
    print(f"Energy (train):                  {energy_j:.0f} J")
    print(f"Params (total / trainable):      {total_params:,} / {trainable_params:,}")
    print("===========================\n")

    # Wspolny CSV: dokladnosc + efektywnosc per (dataset, klasa, ekstraktor, urzadzenie)
    results_csv = "sanity_results.csv"
    write_header = not os.path.exists(results_csv)
    with open(results_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "Dataset", "Class", "Extractor", "HW_Profile", "Epochs", "Batch",
                "AUROC", "Train_Time_s", "Time_per_Epoch_s",
                "Avg_CPU_%", "Avg_GPU_%", "Avg_Power_W", "Avg_VRAM_%", "Energy_J",
                "Total_Params", "Trainable_Params"
            ])
        writer.writerow([
            args.dataset, args.class_name, args.extractor, args.hw_profile, args.epochs, args.batch_size,
            f"{auroc:.4f}", f"{train_time:.1f}", f"{time_per_epoch:.2f}",
            f"{hw['avg_cpu_percent']:.1f}", f"{hw['avg_gpu_percent']:.1f}", f"{hw['avg_power_w']:.2f}",
            f"{hw['avg_vram_percent']:.1f}", f"{energy_j:.0f}",
            total_params, trainable_params
        ])
    print(f"=> Appended to {results_csv}")

    if auroc < 0.60:
        print("[VERDICT] Model is basically guessing randomly. Architecture still failing.")
    elif auroc < 0.82:
        print("[VERDICT] Model notices something, but score is below expected threshold (0.82+).")
    else:
        print("[VERDICT] Acceptable result! Model can distinguish anomalies. You can proceed to FL.")

if __name__ == "__main__":
    main()