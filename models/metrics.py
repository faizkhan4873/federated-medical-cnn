import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Full evaluation pass. Returns dict with accuracy, f1, auc,
    confusion matrix, and classification report.
    """
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "auc":      float(roc_auc_score(y_true, y_prob)),
        "cm":       confusion_matrix(y_true, y_pred).tolist(),
        "report":   classification_report(
                        y_true, y_pred,
                        target_names=["NORMAL", "PNEUMONIA"],
                        zero_division=0,
                    ),
    }


def print_metrics(metrics: dict, prefix: str = "") -> None:
    tag = f"[{prefix}] " if prefix else ""
    print(f"{tag}Accuracy : {metrics['accuracy']:.4f}")
    print(f"{tag}F1       : {metrics['f1']:.4f}")
    print(f"{tag}AUC      : {metrics['auc']:.4f}")
    print(f"{tag}Confusion matrix:\n{np.array(metrics['cm'])}")
    print(f"{tag}Report:\n{metrics['report']}")
