import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import model
import dataset
import engine

def main():
    parser = argparse.ArgumentParser(description="Sanity Check dla modelu Autoenkodera")
    parser.add_argument("--data_path", type=str, required=True, help="Ścieżka do mvtec/metal_nut")
    parser.add_argument("--epochs", type=int, default=100, help="Liczba epok treningowych (zalecane min. 50)")
    parser.add_argument("--batch_size", type=int, default=16, help="Rozmiar paczki danych")
    args = parser.parse_args()

    engine.set_seed()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Rozpoczynam Sanity Check ===")
    print(f"Urządzenie docelowe: {device}")
    print(f"Liczba epok: {args.epochs}")

    files = dataset.get_files(args.data_path, split="train")
    if not files:
        print("BŁĄD: Nie znaleziono plików treningowych. Sprawdź ścieżkę --data_path.")
        return

    print(f"Załadowano {len(files)} obrazów treningowych.")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    
    train_dataset = dataset.MVTecDataset(files, transform=transform)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    testloader = dataset.load_test_data(args.data_path, batch_size=args.batch_size)
    print(f"Załadowano obrazy testowe (dobre i z defektami).")

    net = model.Autoencoder(extractor_name="shufflenet").to(device)

    print("\nRozpoczynam trening modelu...")
    engine.train(net, trainloader, epochs=args.epochs, device=device)
    print("Trening zakończony.")

    print("\nPrzeprowadzam ewaluację na zbiorze testowym...")

    auroc, avg_loss = engine.test(net, testloader, device=device)
    
    print("\n=== WYNIKI SANITY CHECK ===")
    print(f"Średni błąd ewaluacji (Top-K):   {avg_loss:.5f}")
    print(f"Wskaźnik AUROC:                  {auroc:.4f}")
    print("===========================\n")

    if auroc < 0.60:
        print("[WERDYKT] Model w zasadzie zgaduje losowo. Architektura nadal zawodzi.")
    elif auroc < 0.82:
        print("[WERDYKT] Model coś zauważa, ale wynik jest poniżej oczekiwanego progu (0.82+).")
    else:
        print("[WERDYKT] Wynik akceptowalny! Model potrafi odróżnić anomalie. Możesz przechodzić do FL.")

if __name__ == "__main__":
    main()