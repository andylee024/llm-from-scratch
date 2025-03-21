"""An easy train.py for running model training"""
import os
import tiktoken
import torch

from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

from minigpt.data.datasets import BinaryDataset
from minigpt.model.gpt2 import create_gpt2_config, create_gpt2_model 
from minigpt.utils.evaluation import TrainingState, evaluate_dataset_loss, generate_sample

load_dotenv()

def evaluate_model(model, val_dataset, training_state, device, iter_num, optimizer, 
                  training_args, train_loss=None, tokenizer=None, max_iters=None):
    """Evaluate model on validation data and handle logging, checkpointing, and sample generation"""
    batch_size = training_args.get('batch_size', 8)
    block_size = training_args.get('block_size', 256)
    evaluation_prompt = training_args.get('evaluation_prompt', "Hello, how's it going?")
    eval_iters = training_args.get('eval_iters', 20)
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    lr = optimizer.param_groups[0]['lr']

    # get validation loss 
    model.eval()
    with torch.no_grad():
        val_losses = []
        for _ in tqdm(range(eval_iters), desc="eval iter progress"):  
            val_x, val_y = val_dataset.get_random_batch(
                batch_size=batch_size,
                block_size=block_size,
                device_type=device_type,
                device=device
            )
            _, val_loss = model(x=val_x, targets=val_y)
            val_losses.append(val_loss.item())
        val_loss = sum(val_losses) / len(val_losses)

    # log model metrics 
    training_state.update_metrics(iter=iter_num, train_loss=train_loss, val_loss=val_loss, learning_rate=lr)
    training_state.log_metrics(iter=iter_num, train_loss=train_loss, val_loss=val_loss, learning_rate=lr)

    # save model checkpoint if best 
    is_best = training_state.is_best_model(val_loss)
    if is_best:
        training_state.save_checkpoint(iter=iter_num, model=model, optimizer=optimizer, is_best=True)
    
    # generate sample text
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
    
    # Print progress
    print(f"Iteration {iter_num}/{max_iters} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val Loss: {training_state.best_val_loss:.4f}")
    return val_loss


def run_training(model, optimizer, train_dataset, val_dataset, device, training_state, 
                training_args, tokenizer=None):
    """Training function using BinaryDataset with iteration-based training."""
    # get training variables
    max_iters = training_args.get('max_iters', 2000)
    eval_freq = training_args.get('eval_freq', 100)
    batch_size = training_args.get('batch_size', 8)
    block_size = training_args.get('block_size', 256)
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    iter_num = 0
    running_loss = 0.0
    running_iters = 0
    
    # run main training loop
    model.train()
    for iter_num in tqdm(range(max_iters), desc="Training"):

        # batch data
        x_batch, y_batch = train_dataset.get_random_batch(
            batch_size=batch_size, 
            block_size=block_size, 
            device_type=device_type, 
            device=device
        )

        # model update
        optimizer.zero_grad()
        _, loss = model(x=x_batch, targets=y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_iters += 1
        
        # run evaluation
        should_evaluate = (iter_num % eval_freq == 0) or (iter_num == max_iters - 1)
        
        if should_evaluate and running_iters > 0:
            train_loss = running_loss / running_iters
            running_loss = 0.0
            running_iters = 0
            
            evaluate_model(
                model=model,
                val_dataset=val_dataset,
                training_state=training_state,
                device=device,
                iter_num=iter_num,
                optimizer=optimizer,
                training_args=training_args,
                train_loss=train_loss,
                tokenizer=tokenizer,
                max_iters=max_iters
            )
            
            model.train()
    
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
    
    # Set model config
    model_name = "gpt2-small"
    model_config = create_gpt2_config(model_name)

    # Set training args
    training_args = {
        'learning_rate': 0.0004,
        'weight_decay': 0.1,
        'max_iters': 2000,       # Maximum number of training iterations
        'eval_freq': 200,        # Evaluate every N iterations
        'eval_iters': 50,
        'batch_size': 34,         # Batch size for training
        'block_size': model_config['block_size'],       # Context length for the model
        'evaluation_prompt': "Hello, how's it going?",
        'use_wandb': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # Add device type here
    }
    
    # Set data binaries
    train_bin = "data/shakespeare/train.bin"
    val_bin = "data/shakespeare/val.bin"
    
    # Set device
    device = torch.device(training_args['device'])
    print(f"Using device: {device}")

    # Setup run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"./runs/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Setup training state
    training_state = TrainingState(output_dir=output_dir, use_wandb=training_args['use_wandb'])
    if training_args['use_wandb']:
        training_state.init_wandb(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT"),
            config=training_args,
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
        lr=training_args['learning_rate'],
        weight_decay=training_args['weight_decay']
    )

    # Run training
    run_training(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        training_state=training_state,
        training_args=training_args,
        tokenizer=tokenizer
    )