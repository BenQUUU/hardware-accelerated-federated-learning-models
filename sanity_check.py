import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import model
import dataset
import engine

def main():
    parser = argparse.ArgumentParser(description="Sanity Check for Autoencoder model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to mvtec/metal_nut")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (recommended min. 50)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    engine.set_seed()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Sanity Check ===")
    print(f"Target device: {device}")
    print(f"Number of epochs: {args.epochs}")

    files = dataset.get_files(args.data_path, split="train")
    if not files:
        print("ERROR: No training files found. Check the --data_path.")
        return

    print(f"Loaded {len(files)} training images.")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    
    train_dataset = dataset.MVTecDataset(files, transform=transform)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    testloader = dataset.load_test_data(args.data_path, batch_size=args.batch_size)
    print(f"Loaded test images (good and defective).")

    net = model.Autoencoder(extractor_name="shufflenet").to(device)

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