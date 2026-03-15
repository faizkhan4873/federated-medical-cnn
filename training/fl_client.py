from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl

from models.cnn_model import CNNModel
from utils.metrics import evaluate


class FedProxClient(fl.client.NumPyClient):
    """
    Flower client implementing FedProx.

    Each round the server sends global weights w_global.
    Local loss is augmented with a proximal term:

        L(w) = L_task(w) + (mu/2) * ||w - w_global||^2

    This bounds client drift under non-IID hospital data.

    mu = 0.0  ->  equivalent to FedAvg
    mu = 0.01 ->  recommended starting point
    mu = 0.1  ->  stronger control if clients diverge
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        client_id: str = "client",
    ):
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.client_id    = client_id
        self.model        = CNNModel().to(device)
        self.criterion    = nn.CrossEntropyLoss()

    # ── Flower interface ───────────────────────────────────────────────

    def get_parameters(self, config) -> List[np.ndarray]:
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)

        epochs = int(config.get("local_epochs", 2))
        lr     = float(config.get("lr", 1e-3))
        mu     = float(config.get("mu", 0.01))

        loss = self._train(epochs=epochs, lr=lr, mu=mu)

        print(
            f"[{self.client_id}] fit | "
            f"round={config.get('server_round', '?')} "
            f"epochs={epochs} lr={lr} mu={mu} loss={loss:.4f}"
        )
        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            {"train_loss": float(loss)},
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)

        val_loss = self._val_loss()
        metrics  = evaluate(self.model, self.val_loader, self.device)

        print(
            f"[{self.client_id}] eval | "
            f"loss={val_loss:.4f} "
            f"acc={metrics['accuracy']:.4f} "
            f"auc={metrics['auc']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )
        return (
            float(val_loss),
            len(self.val_loader.dataset),
            {
                "accuracy": float(metrics["accuracy"]),
                "f1":       float(metrics["f1"]),
                "auc":      float(metrics["auc"]),
            },
        )

    # ── Internal helpers ───────────────────────────────────────────────

    def _train(self, epochs: int, lr: float, mu: float) -> float:
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

        # Snapshot global weights — frozen for proximal term this round
        global_params = [p.detach().clone() for p in self.model.parameters()]

        self.model.train()
        total_loss  = 0.0
        num_batches = 0

        for _epoch in range(epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # Task loss
                task_loss = self.criterion(self.model(images), labels)

                # Proximal term: (mu/2) * ||w - w_global||^2
                prox = torch.tensor(0.0, device=self.device)
                for w_local, w_global in zip(
                    self.model.parameters(), global_params
                ):
                    prox = prox + (w_local - w_global).pow(2).sum()

                loss = task_loss + (mu / 2.0) * prox
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss  += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _val_loss(self) -> float:
        self.model.eval()
        total = 0.0
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            total += self.criterion(self.model(images), labels).item()
        return total / max(len(self.val_loader), 1)
