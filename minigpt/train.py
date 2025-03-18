"""An easy train.py for running model training"""

import tiktoken
import torch

from minigpt.model.gpt2 import GPTConfig, GPTModel, generate
from minigpt.utils.tokenization import text_to_token_ids, token_ids_to_text
from minigpt.data.dataloaders import create_dataloader


def setup_dataloaders(file_paths, tokenizer, batch_size=4, max_length=256, stride=256, train_ratio=0.85):
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


def run_training(model,
                 optimizer,
                 data_loader,
                 num_epochs):
    
    for epoch_idx in range(num_epochs):
        model.train()

        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()
            _, loss = model(x=x_batch, targets=y_batch)
            loss.backward()
            optimizer.step()
        
        # Print the loss for the epoch
        print(f"Epoch {epoch_idx+1}, Loss: {loss.item():.4f}")


def generate_sample():
    pass


if __name__ == "__main__":

    # data_path = "data/the-verdict.txt"
    data_path = "/Users/andylee/Projects/llm-from-scratch/data/the-verdict.txt"
    gpt_config = GPTConfig(block_size=256)

    # tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # data loaders
    train_loader, validation_loader = setup_dataloaders(file_paths=[data_path],
                                                        tokenizer=tokenizer,
                                                        batch_size=4, 
                                                        max_length=256,
                                                        stride=256,
                                                        train_ratio=0.85)
    
    # model 
    model = GPTModel(gpt_config)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, 
        weight_decay=0.1
    )

    run_training(model=model, optimizer=optimizer, data_loader=train_loader, num_epochs=4)

