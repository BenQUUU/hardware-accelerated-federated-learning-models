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


def parse_args():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID (0-indexed)")
    parser.add_argument("--total_clients", type=int, default=2, help="Total clients in swarm")
    parser.add_argument("--server_ip", type=str, default="127.0.0.1", help="Server IP address")
    parser.add_argument("--data_path", type=str, required=True, help="Path to main data folder")
    parser.add_argument("--dataset", type=str, choices=["mvtec", "visa", "realiad"], required=True, help="Dataset name")
    parser.add_argument("--class_name", type=str, required=True, help="Class name (e.g., metal_nut, pcb1)")
    parser.add_argument("--apply_shift", action="store_true", help="Activate Covariate Shift for client 1")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu/cuda)")
    parser.add_argument("--mode", type=str, choices=["epoch", "time", "robust"], default="time", help="Learning mode")
    parser.add_argument("--hw_profile", type=str, choices=["cpu", "cuda", "jetson", "rpi"], default="cpu", help="Hardware profiler type")
    parser.add_argument("--epochs", type=int, default=2, help="Number of local epochs per round for the mode 'epoch'")
    parser.add_argument("--extractor", type=str, choices=["mobilenet", "shufflenet", "squeezenet"], default="mobilenet", help="Selecting an extractor model")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes. Dziala na Windows/Linux (setup jest pod main()).")
    parser.add_argument("--weighting", type=str, choices=["steps", "dataset"], default="steps",
                        help="FedAvg weighting for 'time' mode: 'steps' = number of processed samples (rewards faster hardware), 'dataset' = dataset size (classic FedAvg, decoupled from hardware speed).")
    parser.add_argument("--partition_mode", type=str, choices=["split", "whole"], default="split",
                        help="'split' = one class split across clients (data ~ IID); 'whole' = each client takes the entire class (different classes per client => non-IID data).")
    parser.add_argument("--extractor_precision", type=str, choices=["fp32", "fp16", "int8", "auto"], default="fp32",
                        help="Precision of the FROZEN feature extractor (the trainable head stays FP32). "
                             "'auto' = fp32 with acceleration (CUDA), fp16 without. "
                             "int8 requires CPU (qnnpack) and only the 'mobilenet' extractor.")
    return parser.parse_args()


def set_parameters(net, parameters):
    # Federujemy TYLKO trenowalna glowice (bottleneck + decoder). Zamrozony ekstraktor
    # zostaje lokalny -- dzieki temu wezel moze miec inna precyzje (FP16/INT8) bez rozjazdu.
    head_keys = [k for k in net.state_dict().keys() if model.is_head_param(k)]
    params_dict = zip(head_keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=False)


def get_parameters(net):
    return [val.cpu().numpy() for name, val in net.state_dict().items() if model.is_head_param(name)]


def select_precision(requested, device):
    """Auto-detekcja precyzji ekstraktora: przy braku akceleracji sprzetowej (CPU)
    obniza precyzje do FP16. Jawne fp32/fp16/int8 pomijaja detekcje."""
    if requested != "auto":
        return requested
    accelerated = device.type == "cuda"
    return "fp32" if accelerated else "fp16"


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, args, device):
        self.net = net
        self.trainloader = trainloader
        self.args = args
        self.device = device
        self.data_iterator = iter(trainloader)

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        args = self.args
        if args.mode == "robust":
            if args.cid == 0 and random.random() < 0.20:
                print(f"[Client {args.cid}] SIMULATION: Network Disconnect! Skipping round.")
                time.sleep(2)
                return parameters, 0, {"train_time": 0.0, "cid": args.cid, "epochs_done": 0, "avg_cpu_percent": 0.0,
                                       "avg_ram_percent": 0.0, "avg_gpu_percent": 0.0, "avg_vram_percent": 0.0, "avg_power_w": 0.0}
            elif args.cid == 1 and random.random() < 0.30:
                print(f"[Client {args.cid}] SIMULATION: Device BUSY. Skipping round.")
                time.sleep(2)
                return parameters, 0, {"train_time": 0.0, "cid": args.cid, "epochs_done": 0, "avg_cpu_percent": 100.0,
                                       "avg_ram_percent": 0.0, "avg_gpu_percent": 0.0, "avg_vram_percent": 0.0, "avg_power_w": 0.0}

        set_parameters(self.net, parameters)
        start_time = time.time()
        epochs_done = 0
        num_examples = len(self.trainloader.dataset)

        hw_profiler = HardwareProfiler(device_type=args.hw_profile)
        hw_profiler.start()

        try:
            # Tryb 'robust' trenuje po epokach (jak 'epoch'), aby scenariusz 5 byl
            # wprost porownywalny ze scenariuszem 3 -- jedyna roznica to dropouty.
            if args.mode in ("epoch", "robust"):
                print(f"[Client {args.cid}] Starting EPOCH training ({args.epochs} epochs)...")
                engine.train(self.net, self.trainloader, epochs=args.epochs, device=self.device)
                epochs_done = args.epochs
            else:
                timeout = float(config.get("timeout", 15.0))
                print(f"[Client {args.cid}] Starting TIME training ({timeout}s)...")
                epochs_done, num_examples, self.data_iterator = engine.train_by_time(
                    self.net, self.data_iterator, self.trainloader, timeout=timeout, device=self.device
                )
        finally:
            hw_metrics = hw_profiler.stop()

        duration = time.time() - start_time
        print(f"[Client {args.cid}] Finished. Epochs: {epochs_done}. Time: {duration:.2f}s")
        print(f"[Client {args.cid}] HW Metrics: CPU: {hw_metrics['avg_cpu_percent']:.1f}% | GPU: {hw_metrics['avg_gpu_percent']:.1f}%")

        # Waga w agregacji FedAvg. W trybie 'time' szybszy sprzet przetwarza wiecej
        # probek (num_examples z powtorzeniami) -- 'steps' nagradza go wieksza waga,
        # 'dataset' odkleja wage od predkosci HW (klasyczny FedAvg po rozmiarze zbioru).
        if args.mode == "time" and args.weighting == "steps":
            actual_dataset_size = num_examples
        else:
            actual_dataset_size = len(self.trainloader.dataset)

        metrics_to_send = {
            "train_time": duration,
            "cid": args.cid,
            "epochs_done": epochs_done
        }
        metrics_to_send.update(hw_metrics)

        return self.get_parameters({}), actual_dataset_size, metrics_to_send

    def evaluate(self, parameters, config):
        return 0.0, len(self.trainloader.dataset), {"accuracy": 0.0}


def main():
    args = parse_args()

    engine.set_seed()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    precision = select_precision(args.extractor_precision, device)
    if precision == "int8" and device.type != "cpu":
        print(f"[Client {args.cid}] INT8 requires CPU (qnnpack) -- switching device to CPU.")
        device = torch.device("cpu")
    if args.extractor_precision == "auto":
        print(f"[Client {args.cid}] Auto-detected precision: {precision} "
              f"({'accelerator present' if device.type == 'cuda' else 'no accelerator'}).")
    print(f"Client {args.cid} starting on device: {device} in {args.mode.upper()} mode | extractor: {precision}")

    net = model.Autoencoder(extractor_name=args.extractor, extractor_precision=precision).to(device)
    trainloader = dataset.load_partitioned_data(
        args.cid, args.total_clients, args.data_path, args.dataset, args.class_name, args.apply_shift,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"), partition_mode=args.partition_mode
    )
    if precision == "int8":
        net.calibrate(trainloader)  # kalibracja + konwersja INT8 na lokalnych danych

    fl.client.start_numpy_client(
        server_address=f"{args.server_ip}:8080",
        client=FlowerClient(net, trainloader, args, device),
    )


if __name__ == "__main__":
    main()
