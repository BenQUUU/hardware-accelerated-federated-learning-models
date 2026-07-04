# Edge-Optimized Federated Learning for Industrial Anomaly Detection

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/FL-Flower-F4D03F?style=flat)](https://flower.dev/)

## Overview
This repository contains the implementation of a highly adaptive Federated Learning (FL) system designed for unsupervised anomaly detection in industrial IoT environments (Industry 4.0). 

The project addresses the critical hardware and networking challenges of deploying deep learning models across deeply heterogeneous edge networks. By moving away from standard synchronous algorithms like `FedAvg`, this system implements a custom **Time-Window Aggregation** strategy to completely eliminate the **Straggler Effect** while maintaining high detection precision.

## Key Features
* **Early-Exit Architecture:** Utilizes a pre-trained backbone (`MobileNetV2`, `ShuffleNetV2`, or `SqueezeNet`) as a **fully frozen** feature extractor — weights *and* BatchNorm running statistics are locked, so only a lightweight bottleneck (4-channels) and decoder are trained. This reduces the trainable parameter count by over 90%, enabling training on memory-constrained devices.
* **Time-Bound Aggregation:** Replaces rigid epoch-based training rounds with strict time windows. This allows High-Performance GPUs to execute dozens of epochs while CPU-only devices contribute fractional updates, maximizing total cluster throughput. Aggregation weighting is configurable (`--weighting`) to either reward faster hardware (`steps`) or decouple weight from device speed (`dataset`).
* **Real Hardware Energy Profiling:** Per-device power is *measured*, not estimated — via NVML (NVIDIA GPUs), `jtop`/INA3221 (Jetson), and the on-board PMIC through `vcgencmd` (Raspberry Pi 5). Intel RAPL is used on x86/Linux; on Windows, CPU power is deliberately reported as unmeasured rather than fabricated.
* **Configurable Non-IID:** Supports data heterogeneity via covariate shift (sensor noise) and a `--partition_mode whole` mode that assigns a distinct product class to each client, with per-class + mean AUROC evaluation on the server.
* **Unsupervised Anomaly Detection:** Learns to compress and reconstruct nominal states. Anomalies (scratches, dents) are detected via Mean Squared Error (MSE) thresholding.
* **Fault Tolerance:** Robust server-side logic capable of handling asynchronous node dropouts and covariate shifts (sensor noise).

## Heterogeneous Hardware Topology
The system is designed and tested on a highly asymmetric star topology cluster. Each tier maps to a `--hw_profile` and a real power-measurement source:

| Tier | Device | `--hw_profile` | Power source |
|------|--------|----------------|--------------|
| Server | PC / Workstation, RTX 3060 12GB (also a client) | `cuda` | NVML |
| Tier 1 (High Performance) | Laptop RTX PRO 1000 (Blackwell), RTX 3060 | `cuda` | NVML |
| Tier 2 (Embedded Edge AI) | NVIDIA Jetson Orin Nano (ARM + GPU, Unified Memory) | `jetson` | `jtop` / INA3221 |
| Tier 3 (IoT / CPU-only) | Raspberry Pi 5 (ARM CPU) | `rpi` | `vcgencmd` PMIC |

> **Note (Blackwell):** the RTX PRO 1000 laptop (`sm_120`) requires a PyTorch build against CUDA 12.8+. Verify `torch.cuda.get_device_capability()` before benchmarking, otherwise runs may silently fall back to CPU.

## Dataset
Supports three industrial anomaly-detection benchmarks: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), VisA, and Real-IAD (`--dataset {mvtec,visa,realiad}`). Training uses only nominal (defect-free) images; the test split mixes nominal and defective samples for AUROC scoring.

Two partitioning strategies simulate Non-IID industrial production lines:
* `--partition_mode split` (default): one product class is split evenly across clients (data ~IID; heterogeneity comes from hardware + optional covariate shift).
* `--partition_mode whole`: each client trains on its **entire** class, so assigning a different class per client produces strong data-level Non-IID.

## Prerequisites
* Python 3.8+
* `torch` and `torchvision`
* `flwr` (Flower framework)
* `scikit-learn` (for AUROC evaluation)
* Profiling tools: `psutil` (all tiers), `pynvml` (NVIDIA GPUs), `jtop` (Jetson). Raspberry Pi 5 power uses the system `vcgencmd` binary (no extra package).

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/BenQUUU/hardware-accelerated-federated-learning-models.git
cd hardware-accelerated-federated-learning-models
pip install -r requirements.txt
```

## Usage

**1. Start the Central Server:**
Initialize the Flower server with the custom time-based strategy.
```bash
python server.py --data_path data/ --dataset mvtec --class_name metal_nut --extractor mobilenet --exp_name time_test --rounds 15 --min_clients 5 --total_clients 5 --timeout 15.0
```

**2. Start Edge Clients:**
On your respective edge devices, connect to the server. Set `--hw_profile` to match the device for correct power/metric logging:
```bash
python client.py --cid 0 --server_ip 127.0.0.1 --data_path data/ --dataset mvtec --class_name metal_nut --hw_profile cuda --extractor mobilenet --mode time --total_clients 5
```

**3. (Optional) Data-level Non-IID with per-client classes:**
The server evaluates the global model on every class (per-class + mean AUROC); each client trains on its own class:
```bash
# Server evaluates on both classes
python server.py --data_path data/ --dataset mvtec --class_name metal_nut,pcb1 --total_clients 2 --exp_name noniid_data ...
# Client 0 = whole metal_nut, Client 1 = whole pcb1
python client.py --cid 0 --class_name metal_nut --partition_mode whole --hw_profile rpi ...
python client.py --cid 1 --class_name pcb1      --partition_mode whole --hw_profile jetson ...
```

### Experiment Configuration Flags

| Flag | Where | Values | Purpose |
|------|-------|--------|---------|
| `--mode` | client | `time` \| `epoch` \| `robust` | Time-bounded, fixed-epoch, or fault-injection (dropout) training. |
| `--weighting` | client | `steps` \| `dataset` | FedAvg weight in `time` mode: processed-samples (rewards fast HW) vs. dataset size (speed-independent). |
| `--hw_profile` | client | `cuda` \| `jetson` \| `rpi` \| `cpu` | Selects the real power-measurement backend for the device. |
| `--partition_mode` | client | `split` \| `whole` | IID split of one class vs. whole-class-per-client (data Non-IID). |
| `--apply_shift` | client | flag | Applies covariate shift (ColorJitter) to client 1. |
| `--num_workers` | client | int (default `0`) | DataLoader workers. Use `>0` only on Linux/Jetson (fork); keep `0` on Windows (spawn). |
| `--extractor` | both | `mobilenet` \| `shufflenet` \| `squeezenet` | Frozen backbone; **must match** between server and clients. |
| `--class_name` | server | comma-separated list | One or more classes to evaluate the global model on. |
| `--timeout` | server | float (seconds) | Per-round training time window sent to clients. |

Both `metrics_<exp>.csv` (AUROC per round) and `hw_metrics_<exp>.csv` (per-client time, epochs, CPU/RAM/GPU/VRAM, power) are written for each experiment; round numbering is consistent across both even when a round is skipped by the fault-tolerant strategy.

## Experimental Results

Extensive hardware profiling demonstrates that the proposed time-aware strategy:
* Prevents idle time on GPU-accelerated nodes (100% utilization).
* Successfully mitigates the straggler effect caused by Raspberry Pi 5.
* Achieves competitive **AUROC** scores despite severe data imbalance and hardware asymmetry.


