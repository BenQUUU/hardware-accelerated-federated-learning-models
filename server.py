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


current_server_round = 0


class RobustFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round: int, results, failures):
        # Zapamiętujemy realny numer rundy serwera, aby fit_metrics_aggregation_fn
        # (którego sygnatura nie dostaje server_round) logował go spójnie z ewaluacją.
        global current_server_round
        current_server_round = server_round

        valid_results = [(client, fit_res) for client, fit_res in results if fit_res.num_examples > 0]

        if not valid_results:
            print(f"\n[SERVER] CRITICAL CLUSTER FAILURE IN ROUND {server_round}. All clients rejected the task!")
            print("[SERVER] Server safely skips global model update and waits for the next round.\n")

            return None, {}

        return super().aggregate_fit(server_round, valid_results, failures)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to main data folder")
parser.add_argument("--dataset", type=str, choices=["mvtec", "visa", "realiad"], required=True, help="Dataset name")
parser.add_argument("--class_name", type=str, required=True, help="Nazwa klasy, lub kilka po przecinku dla Non-IID danych (np. 'metal_nut,pcb1'). Model globalny jest ewaluowany na kazdej z nich.")
parser.add_argument("--min_clients", type=int, default=2, help="Minimum clients required to proceed")
parser.add_argument("--total_clients", type=int, default=2, help="Total expected clients")
parser.add_argument("--exp_name", type=str, default="baseline", help="Experiment name for CSV/Plots")
parser.add_argument("--timeout", type=float, default=20.0, help="Client training timeout in seconds")
parser.add_argument("--rounds", type=int, default=50, help="Number of federated rounds")
parser.add_argument("--extractor", type=str, choices=["mobilenet", "shufflenet", "squeezenet"], default="mobilenet", help="It must be the same as the customers")
args = parser.parse_args()

engine.set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lista klas ewaluacyjnych (jedna, lub kilka po przecinku dla Non-IID danych).
CLASS_NAMES = [c.strip() for c in args.class_name.split(",") if c.strip()]
MULTICLASS = len(CLASS_NAMES) > 1

history_auroc = []
history_times = []

csv_filename = f"metrics_{args.exp_name}.csv"
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    if MULTICLASS:
        writer.writerow(["Round"] + [f"AUROC_{c}" for c in CLASS_NAMES] + ["AUROC_mean", "Round_Max_Time_s"])
    else:
        writer.writerow(["Round", "AUROC", "Round_Max_Time_s"])

csv_hw_filename = f"hw_metrics_{args.exp_name}.csv"
with open(csv_hw_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Round", "Client_ID", "Train_Time_s", "Epochs", "CPU_%", "RAM_%", "GPU_%", "VRAM_%", "Power_W"])

def get_on_fit_config_fn(timeout_seconds: float):
    def fit_config(server_round: int):
        return {"timeout": timeout_seconds}
    return fit_config


def get_evaluate_fn(data_path, dataset_name, class_names):
    # Jeden testloader na klase; model globalny oceniany jest na kazdej z nich.
    testloaders = [
        (c, dataset.load_test_data(data_path, dataset_name, c, pin_memory=(DEVICE.type == "cuda")))
        for c in class_names
    ]

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        net = model.Autoencoder(extractor_name=args.extractor).to(DEVICE)

        valid_keys = [k for k in net.state_dict().keys() if "num_batches_tracked" not in k]

        if len(valid_keys) != len(parameters):
            print(
                f"[WARNING] Layer mismatch! Expected: {len(valid_keys)}, Got: {len(parameters)}")

        params_dict = zip(valid_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=False)

        per_class = []
        mses = []
        for c, testloader in testloaders:
            auroc, avg_mse = engine.test(net, testloader, device=DEVICE)
            per_class.append((c, auroc))
            mses.append(avg_mse)

        mean_auroc = sum(a for _, a in per_class) / len(per_class)
        mean_mse = sum(mses) / len(mses)

        if len(per_class) > 1:
            details = " | ".join(f"{c}={a:.4f}" for c, a in per_class)
            print(f"ROUND {server_round} EVAL: AUROC_mean = {mean_auroc:.4f}  [{details}] | Avg MSE = {mean_mse:.5f}")
        else:
            print(f"ROUND {server_round} EVALUATION: AUROC = {mean_auroc:.4f} | Avg MSE = {mean_mse:.5f}")

        history_auroc.append((server_round, mean_auroc))
        current_max_time = history_times[-1] if history_times else 0.0

        with open(csv_filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            if len(per_class) > 1:
                writer.writerow(
                    [server_round] + [f"{a:.4f}" for _, a in per_class] + [f"{mean_auroc:.4f}", f"{current_max_time:.2f}"]
                )
            else:
                writer.writerow([server_round, f"{mean_auroc:.4f}", f"{current_max_time:.2f}"])

        return mean_mse, {"auroc": mean_auroc}

    return evaluate


def fit_metrics_aggregation_fn(fit_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    stats = []

    for _, m in fit_metrics:
        stats.append({
            "cid": m.get("cid", -1),
            "time": m.get("train_time", 0.0),
            "epochs": m.get("epochs_done", 0),
            "cpu": m.get("avg_cpu_percent", 0.0),
            "ram": m.get("avg_ram_percent", 0.0),
            "gpu": m.get("avg_gpu_percent", 0.0),
            "vram": m.get("avg_vram_percent", 0.0),
            "power": m.get("avg_power_w", 0.0)
        })

    stats.sort(key=lambda x: x["cid"])
    max_time = 0.0

    print(f"\n--- ROUND {current_server_round} FIT RESULTS ---")
    print(f"ID | Time(s) | Epochs | CPU% | RAM% | GPU% | VRAM% | Power(W)")

    with open(csv_hw_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        for s in stats:
            print(
                f" {s['cid']} |  {s['time']:.2f}  |   {s['epochs']}    | {s['cpu']:.1f} | {s['ram']:.1f} | {s['gpu']:.1f} | {s['vram']:.1f} | {s['power']:.2f}")
            writer.writerow([
                current_server_round, s['cid'], f"{s['time']:.2f}", s['epochs'],
                f"{s['cpu']:.1f}", f"{s['ram']:.1f}", f"{s['gpu']:.1f}", f"{s['vram']:.1f}", f"{s['power']:.2f}"
            ])
            if s['time'] > max_time:
                max_time = s['time']

    print(f"--------------------------------------------------")
    history_times.append(max_time)

    return {"max_time": max_time}

if __name__ == "__main__":
    strategy = RobustFedAvg(
        evaluate_fn=get_evaluate_fn(args.data_path, args.dataset, CLASS_NAMES),
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        on_fit_config_fn=get_on_fit_config_fn(timeout_seconds=args.timeout),
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        fraction_evaluate=0.0,
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
        plt.title(f'Evaluation Curve - {args.exp_name}')
        plt.xlabel('Communication Round (Server Round)')
        plt.ylabel('AUROC Score')
        plt.grid(True)
        plt.legend()
        plot_filename = f'plot_auroc_{args.exp_name}.png'
        plt.savefig(plot_filename)
        print(f"\n=> Generated plot: {plot_filename}")
