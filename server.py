import flwr as fl
import torch
from collections import OrderedDict
import centralized
import argparse
from typing import List, Tuple
from flwr.common import Metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to mvtec/metal_nut folder for evaluation")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_evaluate_fn(data_path):
    testloader = centralized.load_test_data(data_path)

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        net = centralized.Autoencoder().to(DEVICE)
        
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        # Test
        loss = centralized.test(net, testloader, device=DEVICE)
        print(f"ROUND {server_round} EVALUATION: MSE Loss = {loss:.5f}")
        
        return loss, {"mse": loss}

    return evaluate

def fit_config(server_round: int):
    return {
        "timeout": 16.0,
        "current_round": server_round,
    }

def fit_metrics_aggregation_fn(fit_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    stats = []
    for _, m in fit_metrics:
        stats.append((m["cid"], m["train_time"], m["epochs_done"]))

    stats.sort(key=lambda x: x[0])

    print(f"\n--- ADAPTIVE TRAINING RESULTS ---")
    print(f"ID | Time (s) | Epochs Done")
    for cid, time, epochs in stats:
        print(f" {cid} |  {time:.2f}   |   {epochs}")
    print(f"---------------------------------")

    return {"dummy": 0}

strategy = fl.server.strategy.FedAvg(
    evaluate_fn=get_evaluate_fn(args.data_path),
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    min_fit_clients=2,
    min_available_clients=2,
)

print(f"Starting Server on {DEVICE}. Waiting for clients...")
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)