import argparse
import time
import random
from collections import OrderedDict
import flwr as fl
import torch

import model
import dataset
import engine
from profiler import HardwareProfiler

parser = argparse.ArgumentParser(description="Flower Client")
parser.add_argument("--cid", type=int, required=True, help="Client ID (0-indexed)")
parser.add_argument("--total_clients", type=int, default=2, help="Total clients in swarm")
parser.add_argument("--server_ip", type=str, default="127.0.0.1", help="Server IP address")
parser.add_argument("--data_path", type=str, required=True, help="Path to mvtec/metal_nut folder")
parser.add_argument("--device", type=str, default="cuda", help="Device (cpu/cuda)")
parser.add_argument("--mode", type=str, choices=["epoch", "time", "robust"], default="time", help="Learning mode")
parser.add_argument("--hw_profile", type=str, choices=["cpu", "cuda", "jetson"], default="cpu", help="Hardware profiler type")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for the mode 'epoch'")
parser.add_argument("--extractor", type=str, choices=["mobilenet", "shufflenet", "squeezenet"], default="mobilenet", help="Selecting an extractor model")
args = parser.parse_args()

engine.set_seed()
DEVICE = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
print(f"Client {args.cid} starting on device: {DEVICE} in {args.mode.upper()} mode")

net = model.Autoencoder(extractor_name=args.extractor).to(DEVICE)
trainloader = dataset.load_partitioned_data(args.cid, args.total_clients, args.data_path)

def set_parameters(net, parameters):
    valid_keys = [k for k in net.state_dict().keys() if "num_batches_tracked" not in k]
    params_dict = zip(valid_keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=False)

def get_parameters(net):
    return [val.cpu().numpy() for name, val in net.state_dict().items() if "num_batches_tracked" not in name]

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.data_iterator = iter(trainloader)

    def get_parameters(self, config):
        return get_parameters(net)

    def fit(self, parameters, config):
        if args.mode == "robust":
            if args.cid == 1 and random.random() < 0.20:
                print(f"[Client {args.cid}] SIMULATION: Network Disconnect! Pomijam rundę.")
                time.sleep(2)
                return parameters, 0, {"train_time": 0.0, "cid": args.cid, "epochs_done": 0, "avg_cpu_percent": 0.0,
                                       "avg_ram_percent": 0.0, "avg_gpu_percent": 0.0, "avg_vram_percent": 0.0}
            elif args.cid == 2 and random.random() < 0.40:
                print(f"[Client {args.cid}] SIMULATION: Device BUSY. Pomijam rundę.")
                time.sleep(2)
                return parameters, 0, {"train_time": 0.0, "cid": args.cid, "epochs_done": 0, "avg_cpu_percent": 100.0,
                                       "avg_ram_percent": 0.0, "avg_gpu_percent": 0.0, "avg_vram_percent": 0.0}

        set_parameters(net, parameters)
        start_time = time.time()
        epochs_done = 0
        num_examples = len(trainloader.dataset)

        hw_profiler = HardwareProfiler(device_type=args.hw_profile)
        hw_profiler.start()

        try:
            if args.mode == "epoch":
                print(f"[Client {args.cid}] Starting EPOCH training ({args.epochs} epochs)...")
                engine.train(net, trainloader, epochs=args.epochs, device=DEVICE)
                epochs_done = args.epochs
            else:
                timeout = float(config.get("timeout", 15.0))
                print(f"[Client {args.cid}] Starting TIME training ({timeout}s)...")
                epochs_done, num_examples, self.data_iterator = engine.train_by_time(
                    net, self.data_iterator, trainloader, timeout=timeout, device=DEVICE
                )
        finally:
            hw_metrics = hw_profiler.stop()

        duration = time.time() - start_time
        print(f"[Client {args.cid}] Finished. Epochs: {epochs_done}. Time: {duration:.2f}s")
        print(f"[Client {args.cid}] HW Metrics: CPU: {hw_metrics['avg_cpu_percent']:.1f}% | GPU: {hw_metrics['avg_gpu_percent']:.1f}%")

        if args.mode == "time":
            actual_dataset_size = num_examples
        else:
            actual_dataset_size = len(trainloader.dataset)

        metrics_to_send = {
            "train_time": duration,
            "cid": args.cid,
            "epochs_done": epochs_done
        }
        metrics_to_send.update(hw_metrics)

        return self.get_parameters({}), actual_dataset_size, metrics_to_send

    def evaluate(self, parameters, config):
        return 0.0, len(trainloader.dataset), {"accuracy": 0.0}

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address=f"{args.server_ip}:8080",
        client=FlowerClient(),
    )