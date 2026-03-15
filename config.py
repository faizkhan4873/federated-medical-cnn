from dataclasses import dataclass


@dataclass
class FedProxConfig:

    # Dataset
    data_root:       str   = "dataset/chest_xray"
    batch_size:      int   = 32
    num_workers:     int   = 4

    # Federated learning
    num_rounds:      int   = 20
    num_clients:     int   = 3
    fraction_fit:    float = 1.0

    # FedProx proximal coefficient
    # 0.0  -> FedAvg (no proximal penalty)
    # 0.01 -> mild drift control (default)
    # 0.1  -> strong drift control
    mu:              float = 0.01

    # Local training
    local_epochs:    int   = 2
    lr:              float = 1e-3
    lr_decay_at:     int   = 5
    momentum:        float = 0.9
    weight_decay:    float = 1e-4
    grad_clip:       float = 1.0

    # Dirichlet heterogeneity
    # 0.1 -> very skewed, 0.5 -> moderate, 10.0 -> near-IID
    dirichlet_alpha: float = 0.5

    # Model
    num_classes:     int   = 2
    dropout:         float = 0.5

    # Paths
    checkpoint_dir:  str   = "checkpoints"
    log_dir:         str   = "logs"


CFG = FedProxConfig()
