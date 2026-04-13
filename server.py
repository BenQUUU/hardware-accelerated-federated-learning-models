import argparse
import csv
from typing import List, Tuple
from collections import OrderedDict
import matplotlib.pyplot as plt

import flwr as fl
import torch
from flwr.common import Metrics

import model
import dataset
import engine

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to mvtec/metal_nut folder")
parser.add_argument("--min_clients", type=int, default=2, help="Minimum clients required to proceed")
parser.add_argument("--total_clients", type=int, default=2, help="Total expected clients")
parser.add_argument("--exp_name", type=str, default="baseline", help="Experiment name for CSV/Plots")
parser.add_argument("--timeout", type=float, default=20.0, help="Client training timeout in seconds")
parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
args = parser.parse_args()

engine.set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

history_auroc = []
history_times = []
current_fit_round = 1

csv_filename = f"metrics_{args.exp_name}.csv"
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "AUROC", "Round_Max_Time_s"])

csv_hw_filename = f"hw_metrics_{args.exp_name}.csv"
with open(csv_hw_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Client_ID", "Train_Time_s", "Epochs", "CPU_%", "RAM_%", "GPU_%", "VRAM_%"])

def get_on_fit_config_fn(timeout_seconds: float):
    def fit_config(server_round: int):
        return {"timeout": timeout_seconds}
    return fit_config

def get_evaluate_fn(data_path):
    testloader = dataset.load_test_data(data_path)

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        net = model.Autoencoder().to(DEVICE)
        
        # Modyfikacja wczytywania częściowych wag
        trainable_keys = [k for k in net.state_dict().keys() if "encoder" not in k]
        params_dict = zip(trainable_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=False)

        auroc, avg_mse = engine.test(net, testloader, device=DEVICE)
        print(f"ROUND {server_round} EVALUATION: AUROC = {auroc:.4f} | Avg MSE = {avg_mse:.5f}")

        history_auroc.append((server_round, auroc))
        current_max_time = history_times[-1] if history_times else 0.0

        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([server_round, f"{auroc:.4f}", f"{current_max_time:.2f}"])

        return avg_mse, {"auroc": auroc}
    return evaluate

def fit_metrics_aggregation_fn(fit_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global current_fit_round
    stats = []
    
    for _, m in fit_metrics:
        stats.append({
            "cid": m.get("cid", -1),
            "time": m.get("train_time", 0.0),
            "epochs": m.get("epochs_done", 0),
            "cpu": m.get("avg_cpu_percent", 0.0),
            "ram": m.get("avg_ram_percent", 0.0),
            "gpu": m.get("avg_gpu_percent", 0.0),
            "vram": m.get("avg_vram_percent", 0.0)
        })

    stats.sort(key=lambda x: x["cid"])
    max_time = 0.0
    
    print(f"\n--- ROUND {current_fit_round} FIT RESULTS ---")
    print(f"ID | Time(s) | Epochs | CPU% | RAM% | GPU% | VRAM%")
    
    with open(csv_hw_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        for s in stats:
            print(f" {s['cid']} |  {s['time']:.2f}  |   {s['epochs']}    | {s['cpu']:.1f} | {s['ram']:.1f} | {s['gpu']:.1f} | {s['vram']:.1f}")
            writer.writerow([
                current_fit_round, s['cid'], f"{s['time']:.2f}", s['epochs'], 
                f"{s['cpu']:.1f}", f"{s['ram']:.1f}", f"{s['gpu']:.1f}", f"{s['vram']:.1f}"
            ])
            if s['time'] > max_time:
                max_time = s['time']
                
    print(f"--------------------------------------------------")
    history_times.append(max_time)
    current_fit_round += 1
    
    return {"max_time": max_time}

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=get_evaluate_fn(args.data_path),
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        on_fit_config_fn=get_on_fit_config_fn(timeout_seconds=args.timeout),
        min_fit_clients=args.min_clients,
        min_available_clients=args.total_clients
    )

    print(f"Starting Server on {DEVICE}. Waiting for {args.total_clients} clients...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    if history_auroc:
        rounds = [x[0] for x in history_auroc]
        scores = [x[1] for x in history_auroc]

        plt.figure(figsize=(10, 5))
        plt.plot(rounds, scores, marker='o', linestyle='-', color='r', label='AUROC (Anomaly Detection)')
        plt.title(f'Krzywa ewaluacji - {args.exp_name}')
        plt.xlabel('Runda komunikacyjna (Server Round)')
        plt.ylabel('Wskaźnik AUROC')
        plt.grid(True)
        plt.legend()
        plot_filename = f'plot_auroc_{args.exp_name}.png'
        plt.savefig(plot_filename)
        print(f"\n=> Wygenerowano wykres: {plot_filename}")