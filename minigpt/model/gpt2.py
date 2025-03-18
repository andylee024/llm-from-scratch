import tiktoken
import torch
import torch.nn as nn

from minigpt.utils.tokenization import text_to_token_ids, token_ids_to_text
from minigpt.model.components.transformer_block import LayerNorm, TransformerBlock

class GPTConfig:
    """Configuration class for GPT model parameters."""
    
    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50257,  # GPT-2 vocab_size of 50257
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.1,
        bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create a config from a dictionary of parameters."""
        return cls(**config_dict)
    
    def to_dict(self):
        """Convert config to a dictionary."""
        return {
            'block_size': self.block_size,
            'vocab_size': self.vocab_size,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'n_embd': self.n_embd,
            'dropout': self.dropout,
            'bias': self.bias,
        }


class GPTModel(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop_emb = nn.Dropout(config.dropout)
        self.trf_blocks = nn.Sequential(*[TransformerBlock({
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "emb_dim": config.n_embd,
            "n_heads": config.n_head,
            "n_layers": config.n_layer,
            "drop_rate": config.dropout,
            "qkv_bias": config.bias
        }) for _ in range(config.n_layer)])
        self.final_norm = LayerNorm(config.n_embd)
        self.out_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x, targets=None):

        # read input X
        _, seq_len = x.shape

        # map X to embedding space
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = tok_embeds + pos_embeds
        
        # process X through transformer blocks
        x = self.drop_emb(x)
        x = self.trf_blocks(x) 
        x = self.final_norm(x)

        # FF to generate logits
        logits = self.out_head(x)

        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        else:
            loss = None
        
        return logits, loss

    def from_pretrained(cls, model_type, override_args=None):
        """Load pretrained weights into the GPT class"""
        pass

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):

        # get context for prediction
        idx_conditional = idx[:, -context_size:]

        # generate predictions with model
        with torch.no_grad():
            logits = model(idx_conditional)

        # generate logits
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == eos_id:
                break

        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)

    return idx

if __name__ == "__main__":
    config = GPTConfig()
    model = GPTModel(config)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create test input (batch_size=1, seq_len=context_size)
    idx = torch.zeros((1, config.block_size), dtype=torch.long)
    
    # Test generation
    generated = generate(
        model,
        idx,
        max_new_tokens=10,
        context_size=config.block_size,
        temperature=0.0,
        top_k=None
    )

    tokenizer = tiktoken.get_encoding("gpt2")
    print("Test generation shape:", generated.shape)
    print("Sample generated tokens:", token_ids_to_text(generated[0, -10:], tokenizer))


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """Training loop of NN"""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        # iterate through training batches 
        for input_batch, target_batch in train_loader:
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
    