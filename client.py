import argparse
import time
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
args = parser.parse_args()

DEVICE = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
print(f"Client {args.cid} starting on device: {DEVICE}")

net = centralized.Autoencoder().to(DEVICE)
trainloader = centralized.load_partitioned_data(args.cid, args.total_clients, args.data_path)

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        start_time = time.time()
        centralized.train(net, trainloader, epochs=5, device=DEVICE)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"Client {args.cid} training finished in {duration:.4f}s")

        return self.get_parameters({}), len(trainloader.dataset), {
            "train_time": duration, 
            "cid": args.cid
        }

    def evaluate(self, parameters, config):
        return 0.0, len(trainloader.dataset), {"accuracy": 0.0}

# Start
fl.client.start_numpy_client(
    server_address=f"{args.server_ip}:8080",
    client=FlowerClient(),
)