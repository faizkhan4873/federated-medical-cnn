import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn_model import CNNModel
from utils.dataset_loader import load_data
from utils.metrics import evaluate, print_metrics

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Data ───────────────────────────────────────────────────────────────────────
train_loader, test_loader = load_data()

# ── Model ──────────────────────────────────────────────────────────────────────
model     = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ── Training loop ──────────────────────────────────────────────────────────────
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}  Loss: {running_loss:.4f}")

# ── Evaluation ─────────────────────────────────────────────────────────────────
metrics = evaluate(model, test_loader, device)
print_metrics(metrics, prefix="Test")

# ── Save model ─────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "cnn_medical_model.pth")
print("Model saved: cnn_medical_model.pth")
