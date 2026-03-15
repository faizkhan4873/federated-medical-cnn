import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import flwr as fl

from models.cnn_model import CNNModel
from server import build_strategy
from training.fl_client import FedProxClient
from utils.dataset_loader import load_federated_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SIM] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main(args):
    # Reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load federated splits — uses ImageFolder on dataset/chest_xray/train
    client_loaders, test_loader = load_federated_data(
        data_root=args.data_root,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        alpha=args.alpha,
    )

    def client_fn(cid: str) -> FedProxClient:
        i = int(cid)
        train_loader, val_loader = client_loaders[i]
        return FedProxClient(
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            client_id=f"client_{i}",
        )

    strategy = build_strategy(min_clients=args.num_clients)

    log.info(
        f"Starting simulation | "
        f"clients={args.num_clients} rounds={args.rounds} alpha={args.alpha}"
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.25 if device.type == "cuda" else 0.0,
        },
    )

    # Print convergence table
    log.info("\n=== Simulation complete ===")
    acc_hist = history.metrics_distributed.get("accuracy", [])
    auc_hist = history.metrics_distributed.get("auc", [])
    f1_hist  = history.metrics_distributed.get("f1", [])

    if acc_hist:
        log.info("Round | Accuracy | AUC     | F1")
        log.info("-" * 42)
        for i, (rnd, acc) in enumerate(acc_hist):
            auc = auc_hist[i][1] if i < len(auc_hist) else float("nan")
            f1  = f1_hist[i][1]  if i < len(f1_hist)  else float("nan")
            log.info(f"  {rnd:3d} | {acc:.4f}   | {auc:.4f} | {f1:.4f}")

    # Save final global model
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(CNNModel().state_dict(), save_path)
        log.info(f"Checkpoint saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedProx simulation")
    parser.add_argument("--data-root",   default="dataset/chest_xray",
                        help="Root folder containing train/ and test/ subfolders")
    parser.add_argument("--num-clients", type=int,   default=3)
    parser.add_argument("--rounds",      type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--alpha",       type=float, default=0.5,
                        help="Dirichlet alpha (lower = more heterogeneous)")
    parser.add_argument("--save-path",   default="checkpoints/global_model.pth")
    main(parser.parse_args())
