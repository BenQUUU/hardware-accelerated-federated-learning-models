# Edge-Optimized Federated Learning for Industrial Anomaly Detection

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/FL-Flower-F4D03F?style=flat)](https://flower.dev/)

## Overview
This repository contains the implementation of a highly adaptive Federated Learning (FL) system designed for unsupervised anomaly detection in industrial IoT environments (Industry 4.0). 

The project addresses the critical hardware and networking challenges of deploying deep learning models across deeply heterogeneous edge networks. By moving away from standard synchronous algorithms like `FedAvg`, this system implements a custom **Time-Window Aggregation** strategy to completely eliminate the **Straggler Effect** while maintaining high detection precision.

## Key Features
* **Early-Exit Architecture:** Utilizes a pre-trained `MobileNet` as a frozen feature extractor. Only a lightweight bottleneck (4-channels) and decoder are trained in the federated network. This reduces memory footprint and trainable parameters by over 90%, enabling training on memory-constrained devices.
* **Time-Bound Aggregation:** Replaces rigid epoch-based training rounds with strict time windows. This allows High-Performance GPUs to execute dozens of epochs while CPU-only devices contribute fractional updates, maximizing total cluster throughput.
* **Unsupervised Anomaly Detection:** Learns to compress and reconstruct nominal states. Anomalies (scratches, dents) are detected via Mean Squared Error (MSE) thresholding.
* **Fault Tolerance:** Robust server-side logic capable of handling asynchronous node dropouts and covariate shifts (sensor noise).

## Heterogeneous Hardware Topology
The system is designed and tested on a highly asymmetric star topology cluster:
* **Central Server:** PC / Workstation (Weight aggregation via gRPC).
* **Tier 1 (High Performance):** NVIDIA RTX 1000 Ada, RTX 3060 12GB.
* **Tier 2 (Embedded Edge AI):** NVIDIA Jetson Orin Nano (ARM + GPU, Unified Memory).
* **Tier 3 (IoT / CPU-only):** Raspberry Pi 5 (ARM CPU).

## Dataset
This project uses the industrial standard [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad), specifically focusing on the `metal_nut` category. Data is partitioned locally on edge nodes to simulate Non-IID industrial production lines.

## Prerequisites
* Python 3.8+
* `torch` and `torchvision`
* `flwr` (Flower framework)
* `scikit-learn` (for AUROC evaluation)
* Profiling tools: `pynvml` (Tier 1), `jtop` (Tier 2), `psutil` (Tier 3)

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
python server.py --data_path data/metal_nut/ --extractor mobilenet --exp_name time_test --rounds 15 --min_clients 5 --total_clients 5 --timeout 15.0
```

**2. Start Edge Clients:**
On your respective edge devices, connect to the server. You can specify the hardware profile for proper metric logging:
```bash
python client.py --cid 0 --server_ip 127.0.0.1 --data_path data/metal_nut/ --hw_profile cuda --extractor mobilenet --mode time --total_clients 5 
```

## Experimental Results

Extensive hardware profiling demonstrates that the proposed time-aware strategy:
* Prevents idle time on GPU-accelerated nodes (100% utilization).
* Successfully mitigates the straggler effect caused by Raspberry Pi 5.
* Achieves competitive **AUROC** scores despite severe data imbalance and hardware asymmetry.


