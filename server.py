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

def fit_metrics_aggregation_fn(fit_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    train_times = [metrics["train_time"] for _, metrics in fit_metrics]

    max_time = max(train_times)
    avg_time = sum(train_times) / len(train_times)
    
    print(f"\n--- CLIENT METRICS ---")
    print(f"Client times: {train_times}")
    print(f"Round Max Time: {max_time:.4f}s")
    print(f"----------------------\n")
    
    return {"max_train_time": max_time, "avg_train_time": avg_time}

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