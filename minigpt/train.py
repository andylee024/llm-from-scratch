"""An easy train.py for running model training"""
import os
import tiktoken
import torch
from tqdm import tqdm

from datetime import datetime
from dotenv import load_dotenv

from minigpt.data.datasets import BinaryDataset

from minigpt.utils.evaluation import TrainingState, evaluate_dataset_loss, generate_sample
from minigpt.model.gpt2 import create_gpt2_config, create_gpt2_model 

load_dotenv()

def run_training(model, optimizer, train_dataset, val_dataset, max_iters, eval_freq, device, training_state,
                batch_size=8, block_size=256, tokenizer=None):
    """Training function using BinaryDataset with iteration-based training.
    
    Args:
        model: The model to train
        optimizer: The optimizer to use
        train_dataset: BinaryDataset for training data
        val_dataset: BinaryDataset for validation data
        max_iters: Maximum number of iterations to train for
        eval_freq: Frequency of evaluation in iterations
        device: Device to train on
        training_state: TrainingState object to track metrics
        batch_size: Batch size for training
        block_size: Context length for the model
        tokenizer: Tokenizer for generating samples
    """
    
    model.train()
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    evaluation_prompt = "Hello, how's it going?"
    
    iter_num = 0
    best_val_loss = float('inf')
    
    # For tracking loss over iterations
    running_loss = 0.0
    running_iters = 0
    
    # Main training loop with progress bar
    pbar = tqdm(range(max_iters), desc="Training")
    for iter_num in pbar:
        # Get random batch from training dataset
        x_batch, y_batch = train_dataset.get_random_batch(
            batch_size=batch_size, 
            block_size=block_size, 
            device_type=device_type, 
            device=device
        )
        
        # Training step
        optimizer.zero_grad()
        _, loss = model(x=x_batch, targets=y_batch)
        loss.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
        running_iters += 1
        
        # Update progress bar with current loss
        pbar.set_postfix({"loss": loss.item()})
        
        # Evaluate periodically
        should_evaluate = (iter_num % eval_freq == 0) or (iter_num == max_iters - 1)
        
        if should_evaluate and running_iters > 0:
            # Calculate average training loss
            train_loss = running_loss / running_iters
            running_loss = 0.0
            running_iters = 0
            
            # Set model to eval mode for validation
            model.eval()
            
            # Evaluate on multiple batches for stability
            with torch.no_grad():
                val_losses = []
                for _ in range(10):  # Use 10 random batches for validation
                    val_x, val_y = val_dataset.get_random_batch(
                        batch_size=batch_size,
                        block_size=block_size,
                        device_type=device_type,
                        device=device
                    )
                    _, val_loss = model(x=val_x, targets=val_y)
                    val_losses.append(val_loss.item())
                
                val_loss = sum(val_losses) / len(val_losses)
            
            # Get current learning rate
            lr = optimizer.param_groups[0]['lr']
            
            # Log metrics
            training_state.update_metrics(iter=iter_num, train_loss=train_loss, val_loss=val_loss, learning_rate=lr)
            training_state.log_metrics(iter=iter_num, train_loss=train_loss, val_loss=val_loss, learning_rate=lr)
            
            # Save model if it's the best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                training_state.save_checkpoint(iter=iter_num, model=model, optimizer=optimizer, is_best=True)
            
            # Generate sample text
            if tokenizer is not None:
                generate_sample(
                    iter=iter_num,
                    model=model,
                    prompt=evaluation_prompt,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=50,
                    print_result=True
                )
            
            # Set model back to training mode
            model.train()
            
            # Print progress
            print(f"Iteration {iter_num}/{max_iters} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Save final model
    training_state.save_checkpoint(iter=iter_num, model=model, optimizer=optimizer, is_best=False)
    
    return training_state


def setup_binary_datasets(train_bin, val_bin):
    """Create BinaryDataset objects for train and validation data
    
    Args:
        train_bin: Path to training binary file
        val_bin: Path to validation binary file
        
    Returns:
        train_dataset, val_dataset: BinaryDataset objects
    """
    train_dataset = BinaryDataset(train_bin)
    val_dataset = BinaryDataset(val_bin)
    
    print(f"Training dataset has {len(train_dataset):,} tokens")
    print(f"Validation dataset has {len(val_dataset):,} tokens")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Hard-coded configuration values
    model_name = "gpt2-small"
    learning_rate = 0.0004
    weight_decay = 0.1
    
    # Training configuration
    max_iters = 2000       # Maximum number of training iterations
    eval_freq = 100        # Evaluate every N iterations
    batch_size = 8         # Batch size for training
    block_size = 256       # Context length for the model
    
    # Data paths
    train_bin = "data/shakespeare/train.bin"
    val_bin = "data/shakespeare/val.bin"
    
    # Weights & Biases config
    use_wandb = True       # Set to False to disable W&B logging
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"./runs/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Create config
    config = create_gpt2_config(model_name)
    config.update({
        'batch_size': batch_size,
        'block_size': block_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'max_iters': max_iters,
        'eval_freq': eval_freq
    })

    # Setup training state
    training_state = TrainingState(output_dir=output_dir, use_wandb=use_wandb)
    if use_wandb:
        training_state.init_wandb(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"),
            config=config,
            run_name=run_id
        )

    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load data
    train_dataset, val_dataset = setup_binary_datasets(train_bin, val_bin)

    # Create model
    model = create_gpt2_model(model_name)
    model.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Run training
    run_training(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_iters=max_iters,
        eval_freq=eval_freq,
        device=device,
        training_state=training_state,
        batch_size=batch_size,
        block_size=block_size,
        tokenizer=tokenizer
    )