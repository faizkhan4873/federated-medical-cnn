import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ── Transforms ─────────────────────────────────────────────────────────────────

def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


# ── Original load_data (unchanged interface) ───────────────────────────────────

def load_data(
    data_root: str = "dataset/chest_xray",
    batch_size: int = 32,
):
    """
    Drop-in replacement for the original load_data().
    Keeps the same return signature: (train_loader, test_loader).
    Train split now uses augmentation; test split uses eval transforms.
    """
    train_dataset = datasets.ImageFolder(
        f"{data_root}/train",
        transform=get_transforms(train=True),
    )
    test_dataset = datasets.ImageFolder(
        f"{data_root}/test",
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, test_loader


# ── Federated split ────────────────────────────────────────────────────────────

def load_federated_data(
    data_root: str = "dataset/chest_xray",
    num_clients: int = 3,
    batch_size: int = 32,
    alpha: float = 0.5,
    seed: int = 42,
):
    """
    Load dataset and partition training data across num_clients using a
    Dirichlet distribution (alpha) to simulate non-IID hospital data.

    Returns:
        client_loaders : list of (train_loader, val_loader) per client
        test_loader    : shared global test loader (unchanged)

    Dirichlet alpha:
        0.1  -> very heterogeneous (extreme non-IID)
        0.5  -> moderate           (realistic hospital setting)
        10.0 -> near-uniform       (close to IID)
    """
    rng = np.random.default_rng(seed)

    # Full training set — used only for index splitting here
    full_train = datasets.ImageFolder(
        f"{data_root}/train",
        transform=get_transforms(train=True),
    )

    # Group indices by class label
    label_to_indices = {}
    for idx, (_, label) in enumerate(full_train.samples):
        label_to_indices.setdefault(label, []).append(idx)

    for indices in label_to_indices.values():
        rng.shuffle(indices)

    # Dirichlet split per class
    client_indices = [[] for _ in range(num_clients)]
    for label, indices in label_to_indices.items():
        proportions = rng.dirichlet(np.ones(num_clients) * alpha)
        proportions = (proportions * len(indices)).astype(int)
        proportions[-1] = len(indices) - proportions[:-1].sum()  # fix rounding

        start = 0
        for cid, count in enumerate(proportions):
            client_indices[cid].extend(indices[start: start + count])
            start += count

    # Log split stats
    class_names = full_train.classes
    print(f"Federated split (alpha={alpha}, {num_clients} clients):")
    for i, idxs in enumerate(client_indices):
        labels    = [full_train.samples[j][1] for j in idxs]
        n_pneu    = sum(labels)
        pct       = 100 * n_pneu / len(labels) if labels else 0.0
        print(
            f"  Client {i}: {len(idxs):4d} samples | "
            f"{class_names[1]} {n_pneu}/{len(labels)} ({pct:.0f}%)"
        )

    # Build per-client DataLoaders
    # Train subset uses augmentation; val subset (20%) uses eval transforms
    client_loaders = []
    for i, idxs in enumerate(client_indices):
        rng.shuffle(idxs)
        n_val   = max(1, int(len(idxs) * 0.2))
        val_idx = list(idxs[:n_val])
        trn_idx = list(idxs[n_val:])

        # Separate dataset instances to apply different transforms
        train_ds = datasets.ImageFolder(
            f"{data_root}/train",
            transform=get_transforms(train=True),
        )
        val_ds = datasets.ImageFolder(
            f"{data_root}/train",
            transform=get_transforms(train=False),
        )

        train_loader = DataLoader(
            Subset(train_ds, trn_idx),
            batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        val_loader = DataLoader(
            Subset(val_ds, val_idx),
            batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )
        client_loaders.append((train_loader, val_loader))

    # Global test loader (shared, no augmentation)
    test_loader = DataLoader(
        datasets.ImageFolder(
            f"{data_root}/test",
            transform=get_transforms(train=False),
        ),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    return client_loaders, test_loader
