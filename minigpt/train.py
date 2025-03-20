"""An easy train.py for running model training"""
import os
import tiktoken
import torch

from datetime import datetime
from dotenv import load_dotenv

from minigpt.data.dataloaders import create_dataloader
from minigpt.utils.evaluation import TrainingState, evaluate_dataset_loss, generate_sample
from minigpt.model.gpt2 import create_gpt2_config, create_gpt2_model 

load_dotenv()

def run_training(model, optimizer, train_loader, val_loader, num_epochs, 
                 eval_freq, device, training_state):
    
    for epoch_idx in range(num_epochs):

        # train 
        model.train()
        epoch_total_loss = 0.
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            _, loss = model(x=x_batch, targets=y_batch)
            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
        
        
        # evaluate 
        should_evaluate = (epoch_idx % eval_freq == 0) or (epoch_idx == num_epochs - 1)
        evaluation_prompt = "Hello, how's it going?"
        if should_evaluate: 

            # gather metrics
            train_loss = epoch_total_loss / len(train_loader)
            val_loss = evaluate_dataset_loss(model=model, data_loader=val_loader, device=device)
            lr = optimizer.param_groups[0]['lr']

            # log metrics
            training_state.update_metrics(epoch=epoch_idx, train_loss=train_loss, val_loss=val_loss, learning_rate=lr)
            training_state.log_metrics(epoch=epoch_idx, train_loss=train_loss, val_loss=val_loss, learning_rate=lr)

            # log best model 
            if training_state.is_best_model(val_loss):
                training_state.save_checkpoint(epoch=epoch_idx, model=model, optimizer=optimizer, is_best=True)

            # see model output 
            generate_sample(
                epoch=epoch_idx,
                model=model,
                prompt=evaluation_prompt,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=50,
                print_result=True
            )

    # save final model
    training_state.save_checkpoint(epoch=epoch_idx, model=model, optimizer=optimizer, is_best=False)
    
    return training_state


def _setup_dataloaders(file_paths, tokenizer, batch_size=4, max_length=256, stride=256, train_ratio=0.85):
    """Create 2 dataloaders based on file paths for train and validation"""
    raw_text = ""

    # ingest data
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            raw_text += f.read() + "\n \n"
    
    # create train and validation splits
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]

    # instantiate loaders 
    train_loader = create_dataloader(
        txt=train_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = create_dataloader(
        txt=val_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":

    model_name = "gpt2-small"

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # set run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"./runs/{run_id}"

    # set config
    config = create_gpt2_config(model_name)

    # track model training
    use_wandb = True  
    training_state = TrainingState(output_dir=output_dir, use_wandb=use_wandb)
    if use_wandb:
        training_state.init_wandb(entity=os.getenv("WANDB_ENTITY"),
                                  project=os.getenv("WANDB_PROJECT"),
                                  config=config,
                                  run_name=run_id)

    # load data 
    data_path = "data/the-verdict.txt"
    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, validation_loader = _setup_dataloaders(file_paths=[data_path],
                                                         tokenizer=tokenizer,
                                                         batch_size=4, 
                                                         max_length=256, 
                                                         stride=256,
                                                         train_ratio=0.85)

    # load model
    model = create_gpt2_model(model_name)
    model.to(device)

    # load optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, 
        weight_decay=0.1
    )

    # train model
    run_training(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=validation_loader,
        num_epochs=20,
        eval_freq=5,
        device=device,
        training_state=training_state
    )