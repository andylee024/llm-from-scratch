import numpy as np
import torch

def evaluate_batch_loss(x_batch, y_batch, model):
    """Calculates loss for a single batch"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    _, loss = model(x_batch, y_batch)
    return loss

def evaluate_dataset_loss(data_loader, model):
    """Iterates through data loader and calculates loss per batch"""
    model.eval()
    if len(data_loader) == 0:
        raise ValueError("dataloader has zero length")

    batch_losses = [evaluate_batch_loss(x_batch, y_batch, model).item() for (x_batch, y_batch) in data_loader]
    return np.mean(batch_losses)
