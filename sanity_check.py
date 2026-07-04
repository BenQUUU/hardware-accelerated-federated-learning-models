import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import model
import dataset
import engine

def main():
    parser = argparse.ArgumentParser(description="Sanity Check for Autoencoder model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to main data folder")
    parser.add_argument("--dataset", type=str, choices=["mvtec", "visa", "realiad"], required=True, help="Dataset name")
    parser.add_argument("--class_name", type=str, required=True, help="Class name (e.g., metal_nut, pcb1)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--extractor", type=str, choices=["mobilenet", "shufflenet", "squeezenet"], default="mobilenet", help="Extractor model")
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

    print("\nStarting model training...")
    engine.train(net, trainloader, epochs=args.epochs, device=device)
    print("Training finished.")

    print("\nEvaluating on test dataset...")
    auroc, avg_loss = engine.test(net, testloader, device=device)
    
    print("\n=== SANITY CHECK RESULTS ===")
    print(f"Average evaluation loss (Top-K): {avg_loss:.5f}")
    print(f"AUROC score:                     {auroc:.4f}")
    print("===========================\n")

    if auroc < 0.60:
        print("[VERDICT] Model is basically guessing randomly. Architecture still failing.")
    elif auroc < 0.82:
        print("[VERDICT] Model notices something, but score is below expected threshold (0.82+).")
    else:
        print("[VERDICT] Acceptable result! Model can distinguish anomalies. You can proceed to FL.")

if __name__ == "__main__":
    main()