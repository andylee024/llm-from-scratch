import numpy as np
import os
import torch
import wandb

from minigpt.utils.tokenization import text_to_token_ids, token_ids_to_text

from dotenv import load_dotenv
load_dotenv()


class TrainingState:
    """Class to track training state and metrics."""
    
    def __init__(self, output_dir="/runs", use_wandb=True):
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epochs': []
        }
        
        # track best model
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def update_metrics(self, epoch, train_loss, val_loss, learning_rate):
        """Update metrics with latest values."""
        self.metrics['epochs'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['learning_rates'].append(learning_rate)
        
    def is_best_model(self, val_loss):
        """Check if current validation loss is best so far."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False
    
    def save_checkpoint(self, epoch, model, optimizer, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
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
        if self.use_wandb and is_best:
            wandb.save(checkpoint_path)
        
        return checkpoint_path
    
    def log_metrics(self, epoch, train_loss, val_loss, learning_rate):
        print(f"Epoch {epoch}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"LR: {learning_rate:.6f}")
        
        if self.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "epoch": epoch
            })

    def init_wandb(self, project, entity, config, run_name):
        entity = os.getenv("WANDB_ENTITY")
        project = os.getenv("WANDB_PROJECT")

        if wandb.run is None:  
            self.use_wandb = True
            wandb.init(
                project=project,
                entity=entity,
                config=config,
                name=run_name or f"run_{os.path.basename(self.output_dir)}",
                dir=self.output_dir
            )
            print(f"Initialized wandb with project: {project}")
            return True

        else:
            print("wandb already initialized")
            return False
    
    def watch_model(self, model, log="gradients", log_freq=100):
        """Set up model monitoring."""
        if self.use_wandb:
            wandb.watch(model, log=log, log_freq=log_freq)


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


def generate_sample(epoch, model, prompt, tokenizer, device, max_new_tokens=50, block_size=25, print_result=True):
    """Generate text sample from a prompt and optionally print the result."""
    # Convert prompt to token IDs
    input_ids = text_to_token_ids(prompt, tokenizer).to(device)
    
    # Generate tokens
    x = input_ids
    for _ in range(max_new_tokens):
        x_conditioned = x[:, -block_size:] 
        
        with torch.no_grad():
            logits, _ = model(x_conditioned)
            
        # Decode next token prediction
        next_token_logits = logits[:, -1, :]
        next_token_probabilities = torch.softmax(next_token_logits, dim=-1)
        next_token_prediction = torch.argmax(next_token_probabilities, dim=-1, keepdim=True)
        # next_token_prediction = torch.multinomial(next_token_probabilities, dim=-1, keepdim=True)
        
        x = torch.cat((x_conditioned, next_token_prediction), dim=1)
    
    # Convert tokens back to text
    generated_text = token_ids_to_text(x[0], tokenizer)
    
    # Print the result if requested
    if print_result:
        print(f"\n{'='*40}")
        print(f"Epoch: {epoch}")
        print(f"Prompt: \"{prompt}\"")
        print(f"Model Output: \"{generated_text}\"")
        print(f"{'='*40}\n")
    
    return generated_text