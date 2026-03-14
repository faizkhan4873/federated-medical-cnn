import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_data():

    train_dataset = datasets.ImageFolder(
        "dataset/chest_xray/train",
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        "dataset/chest_xray/test",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    return train_loader, test_loader