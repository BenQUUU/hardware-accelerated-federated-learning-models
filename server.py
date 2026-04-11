import flwr as fl
import torch
from collections import OrderedDict
import centralized
import argparse
import csv
import matplotlib.pyplot as plt
from typing import List, Tuple
from flwr.common import Metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to mvtec/metal_nut folder")
parser.add_argument("--min_clients", type=int, default=2, help="Minimum clients required to proceed")
parser.add_argument("--total_clients", type=int, default=2, help="Total expected clients")
parser.add_argument("--exp_name", type=str, default="baseline", help="Experiment name for CSV/Plots")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

history_mse = []
history_times = []

csv_filename = f"metrics_{args.exp_name}.csv"
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "MSE_Loss", "Round_Max_Time_s"])


def get_evaluate_fn(data_path):
    testloader = centralized.load_test_data(data_path)

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        net = centralized.Autoencoder().to(DEVICE)

        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        loss = centralized.test(net, testloader, device=DEVICE)
        print(f"ROUND {server_round} EVALUATION: MSE Loss = {loss:.5f}")

        history_mse.append((server_round, loss))
        current_max_time = history_times[-1] if history_times else 0.0

        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([server_round, f"{loss:.5f}", f"{current_max_time:.2f}"])

        return loss, {"mse": loss}

    return evaluate


def fit_metrics_aggregation_fn(fit_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    stats = []
    for _, m in fit_metrics:
        stats.append((m["cid"], m["train_time"], m["epochs_done"]))

    stats.sort(key=lambda x: x[0])

    max_time = 0.0
    print(f"\n--- ROUND RESULTS ---")
    print(f"ID | Time (s) | Epochs Done")
    for cid, t, epochs in stats:
        print(f" {cid} |  {t:.2f}   |   {epochs}")
        if t > max_time:
            max_time = t
    print(f"---------------------")

    history_times.append(max_time)
    return {"max_time": max_time}


strategy = fl.server.strategy.FedAvg(
    evaluate_fn=get_evaluate_fn(args.data_path),
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    min_fit_clients=args.min_clients,
    min_available_clients=args.total_clients
)

print(f"Starting Server on {DEVICE}. Waiting for {args.total_clients} clients...")
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)

if history_mse:
    rounds = [x[0] for x in history_mse]
    losses = [x[1] for x in history_mse]

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, losses, marker='o', linestyle='-', color='b', label='MSE Loss (Autoencoder)')
    plt.title(f'Krzywa uczenia - {args.exp_name}')
    plt.xlabel('Runda komunikacyjna (Server Round)')
    plt.ylabel('Błąd średniokwadratowy (MSE)')
    plt.grid(True)
    plt.legend()
    plot_filename = f'plot_loss_{args.exp_name}.png'
    plt.savefig(plot_filename)
    print(f"\n=> Wygenerowano wykres: {plot_filename}")
    print(f"=> Zapisano surowe dane do: {csv_filename}")