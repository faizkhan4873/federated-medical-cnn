import argparse
import logging
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import Metrics, Scalar
from flwr.server.strategy import FedAvg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVER] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    return {
        key: sum(n * float(m[key]) for n, m in metrics) / total
        for key in metrics[0][1]
    }


def fit_config(server_round: int) -> Dict[str, Scalar]:
    return {
        "local_epochs": 2,
        "lr":           1e-3 if server_round <= 5 else 5e-4,
        "mu":           0.01,
        "server_round": server_round,
    }


def build_strategy(min_clients: int = 2) -> FedAvg:
    return FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )


def main(server_address="0.0.0.0:8080", num_rounds=20, min_clients=2):
    log.info(f"Starting FedProx server | address={server_address} rounds={num_rounds}")

    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=build_strategy(min_clients),
    )

    if history.metrics_distributed:
        log.info("Final round metrics:")
        for key, values in history.metrics_distributed.items():
            if values:
                log.info(f"  {key}: {values[-1][1]:.4f}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address",     default="0.0.0.0:8080")
    parser.add_argument("--rounds",      type=int, default=20)
    parser.add_argument("--min-clients", type=int, default=2)
    args = parser.parse_args()
    main(args.address, args.rounds, args.min_clients)
