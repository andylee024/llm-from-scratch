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


def run_training():
    pass

def train_epoch(data_loader, optimizer, model):

    for epoch in range(num_epochs):
        model.train()

        # iterate through training batches 
        for input_batch, target_batch in train_loader:

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            # print losses based on eval_freq
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # generate new token predictions after each epoch 
        generate_and_print_sample(model, tokenizer, device, start_context)
    
    return train_losses, val_losses, track_tokens_seen
    pass

def generate_sample():
    pass


if __name__ == "__main__":

    # data_path = "data/the-verdict.txt"
    data_path = "/Users/andylee/Projects/llm-from-scratch/data/the-verdict.txt"
    gpt_config = GPTConfig(block_size=256)

    # tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # data loaders
    train_dataloader, validation_dataloader = setup_dataloaders(file_paths=[data_path],
                                                                tokenizer=tokenizer,
                                                                batch_size=4, 
                                                                max_length=256,
                                                                stride=256,
                                                                train_ratio=0.85)
    
    for x_batch, y_batch in train_dataloader:
        print(x_batch.shape)
        print(y_batch.shape)
        break

    # model 
    model = GPTModel(gpt_config)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, 
        weight_decay=0.1
    )
