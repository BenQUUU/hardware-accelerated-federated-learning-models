import argparse
import time
import random
from collections import OrderedDict
import flwr as fl
import torch
import centralized

parser = argparse.ArgumentParser(description="Flower Client")
parser.add_argument("--cid", type=int, required=True, help="Client ID (0-indexed)")
parser.add_argument("--total_clients", type=int, default=2, help="Total clients in swarm")
parser.add_argument("--server_ip", type=str, default="127.0.0.1", help="Server IP address")
parser.add_argument("--data_path", type=str, required=True, help="Path to mvtec/metal_nut folder")
parser.add_argument("--device", type=str, default="cuda", help="Device (cpu/cuda)")
parser.add_argument("--mode", type=str, choices=["epoch", "time", "robust"], default="time", help="Learning mode")
parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for the mode 'epoch'")
args = parser.parse_args()

DEVICE = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
print(f"Client {args.cid} starting on device: {DEVICE} in {args.mode.upper()} mode")

net = centralized.Autoencoder().to(DEVICE)
trainloader = centralized.load_partitioned_data(args.cid, args.total_clients, args.data_path)

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        # SYMULACJA AWARII (Tylko w trybie 'robust')
        if args.mode == "robust":
            if args.cid == 1 and random.random() < 0.20: # 20% szans na błąd sieci
                print(f"[Client {args.cid}] SIMULATION: Network Disconnect!")
                raise Exception("Network Disconnect")
            elif args.cid == 2 and random.random() < 0.40: # 40% szans na zajętość Jetsona
                print(f"[Client {args.cid}] SIMULATION: Device BUSY (Production Priority).")
                time.sleep(2)
                raise Exception("Device Busy")

        set_parameters(net, parameters)
        start_time = time.time()
        epochs_done = 0
        num_examples = len(trainloader.dataset)

        if args.mode == "epoch":
            print(f"[Client {args.cid}] Starting EPOCH training ({args.epochs} epochs)...")
            centralized.train(net, trainloader, epochs=args.epochs, device=DEVICE)
            epochs_done = args.epochs
        else:
            timeout = float(config.get("timeout", 15.0))
            print(f"[Client {args.cid}] Starting TIME training ({timeout}s)...")
            epochs_done, num_examples = centralized.train_by_time(
                net, trainloader, timeout=timeout, device=DEVICE
            )

        duration = time.time() - start_time
        print(f"[Client {args.cid}] Finished. Epochs: {epochs_done}. Time: {duration:.2f}s")

        actual_dataset_size = len(trainloader.dataset)

        return self.get_parameters({}), actual_dataset_size, {
            "train_time": duration,
            "cid": args.cid,
            "epochs_done": epochs_done
        }

    def evaluate(self, parameters, config):
        return 0.0, len(trainloader.dataset), {"accuracy": 0.0}

# Start
fl.client.start_numpy_client(
    server_address=f"{args.server_ip}:8080",
    client=FlowerClient(),
)