import numpy as np
import os
import torch


class TrainingState:
    """Class to track training state and metrics."""
    
    def __init__(self, output_dir="/runs"):
        
        self.metrics = {
            'train_loss': [],
            'validation_loss': [],
            'learning_rates': [],
            'epochs': []
        }
        
        # track best model
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def update_metrics(self, epoch, train_loss, val_loss, learning_rate):
        """Update metrics with latest values."""
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['learning_rates'].append(learning_rate)
        self.current_epoch = epoch
        
    def is_best_model(self, val_loss):
        """Check if current validation loss is best so far."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False
    
    def save_checkpoint(self, model, optimizer, is_best=False, use_wandb=False):
        """Save model checkpoint."""
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'metrics': self.metrics,
        }
        
        if is_best:
            filename = 'best_model.pt'
        else:
            filename = f'epoch_{self.current_epoch}.pt'
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Log to wandb if enabled
        if use_wandb and is_best:
            import wandb
            wandb.save(checkpoint_path)
        
        return checkpoint_path
    
    def log_metrics(self, train_loss, val_loss, learning_rate, use_wandb=False):
        """Log current metrics to console and optionally to wandb."""
        # Print to console
        print(f"Epoch {self.current_epoch}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"LR: {learning_rate:.6f}")
        
        # Log to wandb if enabled
        if use_wandb:
            import wandb
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "epoch": self.current_epoch
            })


def evaluate_batch_loss(x_batch, y_batch, model):
    """Calculates loss for a single batch"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    _, loss = model(x_batch, y_batch)
    return loss

def evaluate_dataset_loss(data_loader, model, device):
    """Evaluate model on dataset and return average loss."""
    model.eval()
    total_loss = 0
    total_batches = len(data_loader)

    if total_batches <= 0:
        raise ValueError("dataloader has length 0")
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            _, loss = model(x=x_batch, targets=y_batch)
            total_loss += loss.item()
    
    return total_loss / total_batches 
